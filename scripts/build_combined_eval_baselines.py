"""
Step 1 — Combined Amateur/Professional Eval Baseline Builder
============================================================

Scores audio clips from two sources and writes a unified JSON.

  Sources
  -------
  professional : ccmusic-database/acapella  (HuggingFace, CC-BY-NC-ND 4.0)
                 132 clips, 22 singers, 9-dim expert scores already in the dataset
  amateur      : PopBuTFy amateur clips     (local, simulated — pro singer performing poorly)

  Models
  ------
  singmos      : SingMOS-Pro  (wav2vec2-large, ~95M, scalar MOS 1–5)
                 NOTE: shows inverted separation on PopBuTFy vs ccmusic — measures acoustic
                 recording quality, not singing skill. Run already completed; use --no-singmos
                 to skip and only run SongEvalGenerator.
  songevalgen  : SongEvalGenerator (MuQ-large + head, ~321M, single MOS scalar 1–5)
                 Uses generate_tag() — same interface as score_vocalverse2.py.

Output
------
  data/combined_eval_baselines.json
    {
      "ccmusic_professional":  { <clip_id>: { "expert_scores": {...}, "singmos": float,
                                              "songevalgen": float } },
      "vocalset_professional": { <clip_id>: { "singmos": float, "songevalgen": float } },
      "popbutfy_professional": { <clip_id>: { "singmos": float, "songevalgen": float } },
      "popbutfy_amateur":      { <clip_id>: { "singmos": float, "songevalgen": float } },
      "yt_professional":       { <clip_id>: { "singmos": float, "songevalgen": float } },  # optional
      "yt_amateur":            { <clip_id>: { "singmos": float, "songevalgen": float } },  # optional
      "_meta":                 { ... }
    }

  data/combined_eval_summary.json
    Per-level statistics (mean/median/p25/p75/std) for every scored dimension.

Usage
-----
  # SongEvalGenerator only — scores all 3 levels (recommended next run)
  # Also scores PopBuTFy professional which was missing from the SingMOS run
  python scripts/build_combined_eval_baselines.py \\
      --popbutfy data/popbutfy \\
      --no-singmos

  # Merge SongEvalGen scores into existing SingMOS baselines (preserves singmos keys)
  python scripts/build_combined_eval_baselines.py \\
      --popbutfy data/popbutfy \\
      --no-singmos \\
      --merge-existing data/combined_eval_baselines.json

  # Run both models (default)
  python scripts/build_combined_eval_baselines.py \\
      --popbutfy data/popbutfy

  # Quick sanity check on a few clips
  python scripts/build_combined_eval_baselines.py \\
      --popbutfy data/popbutfy \\
      --no-singmos \\
      --max-amateur 10

  # Score YouTube cover clips with SongEvalGenerator
  python scripts/build_combined_eval_baselines.py \\
      --yt-manifest data/yt_covers/manifest.json \\
      --no-singmos \\
      --merge-existing data/combined_eval_baselines.json

  # GTSinger technique characterisation mode (separate output, English singers only by default)
  # Scores Control_Group vs Technique_Group per technique, prints delta table.
  # Does NOT write to combined_eval_baselines.json — saves to data/gtsinger_technique_scores.json
  python scripts/build_combined_eval_baselines.py \\
      --gtsinger data/gtsinger_audio \\
      --gtsinger-lang English

  # Full run — baselines + GTSinger + YouTube covers
  python scripts/build_combined_eval_baselines.py \\
      --popbutfy data/popbutfy \\
      --gtsinger data/gtsinger_audio \\
      --yt-manifest data/yt_covers/manifest.json

Setup
-----
  pip install datasets librosa soundfile tqdm s3prl
  pip install muq peft          # for SongEvalGenerator
  export QWENFEAT_ROOT=/path/to/QwenFeat   # cloned HF repo with audioscore/
"""

import argparse
import json
import os
import pathlib
import sys
import warnings
import logging

import numpy as np
import librosa
from tqdm import tqdm

# Suppress noisy transformer/whisper warnings
logging.getLogger("transformers.models.whisper.feature_extraction_whisper").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*sampling_rate.*")


# ── helpers ───────────────────────────────────────────────────────────────────

def percentile_stats(values):
    arr = np.array([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if len(arr) == 0:
        return {"mean": None, "median": None, "p25": None, "p75": None, "std": None, "n": 0}
    return {
        "mean":   round(float(np.mean(arr)),   4),
        "median": round(float(np.median(arr)), 4),
        "p25":    round(float(np.percentile(arr, 25)), 4),
        "p75":    round(float(np.percentile(arr, 75)), 4),
        "std":    round(float(np.std(arr)),    4),
        "n":      int(len(arr)),
    }


# ── SingMOS-Pro loader ────────────────────────────────────────────────────────

_singmos_model  = None
_singmos_device = None
_SINGMOS_SR     = 16000

def load_singmos(device="cuda"):
    global _singmos_model, _singmos_device
    if _singmos_model is not None:
        return _singmos_model

    import torch
    import torchaudio as _ta
    import types as _types

    # Patch torchaudio 2.x incompatibilities in s3prl
    if not hasattr(_ta, "set_audio_backend"):
        _ta.set_audio_backend = lambda *a, **kw: None
    if not hasattr(_ta, "sox_effects"):
        _sox = _types.ModuleType("torchaudio.sox_effects")
        _sox.apply_effects_tensor = lambda w, sr, effects, **kw: (w, sr)
        _ta.sox_effects = _sox
        sys.modules["torchaudio.sox_effects"] = _sox

    _singmos_device = device
    print("[SingMOS] Loading SingMOS-Pro ...")
    model = torch.hub.load("South-Twilight/SingMOS:v1.1.2", "singmos_pro", trust_repo=True)
    model = model.to(device).eval()
    _singmos_model = model
    print("[SingMOS] Loaded.")
    return model


def singmos_score(y_16k: np.ndarray) -> "float | None":
    """Score a 16 kHz mono array. Returns MOS float or None on error."""
    if _singmos_model is None:
        return None
    try:
        import torch
        wave   = torch.tensor(y_16k, dtype=torch.float32).unsqueeze(0).to(_singmos_device)
        length = torch.tensor([wave.shape[1]], dtype=torch.long).to(_singmos_device)
        with torch.no_grad():
            s = _singmos_model(wave, length)
        return float(np.clip(s.item() if hasattr(s, "item") else s, 1.0, 5.0))
    except Exception as e:
        warnings.warn(f"[SingMOS] inference error: {e}")
        return None


# ── SongEvalGenerator loader ──────────────────────────────────────────────────

_seg_model  = None
_seg_device = None
# generate_tag() returns a single scalar MOS (1–5), not per-dimension scores

def load_songevalgen(device="cuda"):
    global _seg_model, _seg_device

    if _seg_model is not None:
        return _seg_model

    qwenfeat_root  = pathlib.Path(os.environ.get("QWENFEAT_ROOT", "/tmp/QwenFeat"))
    audioscore_src = qwenfeat_root / "audioscore" / "src"
    # matches score_vocalverse2.py CKPT_DIR — no double nesting
    ckpt_dir = (qwenfeat_root / "audioscore" / "ckpts" /
                "SongEvalGenerator" / "step_2_al_audio" /
                "best_model_step_132000")

    if not audioscore_src.exists():
        raise FileNotFoundError(
            f"audioscore source not found at {audioscore_src}.\n"
            "Set QWENFEAT_ROOT or clone huggingface.co/karl-wang/QwenFeat-Vocal-Score"
        )
    if not ckpt_dir.exists():
        raise FileNotFoundError(
            f"SongEvalGenerator checkpoint not found at {ckpt_dir}.\n"
            "Download LFS files: see score_vocalverse2.py setup instructions."
        )

    sys.path.insert(0, str(audioscore_src))
    from audioscore.model import SongEvalGenerator_audio_lora  # noqa

    print(f"[SongEvalGen] Loading SongEvalGenerator from {ckpt_dir} ...")
    model = SongEvalGenerator_audio_lora()
    model.load_model(str(ckpt_dir))
    model = model.to(device).eval()
    _seg_model  = model
    _seg_device = device
    print("[SongEvalGen] Loaded.")
    return model


def seg_score(audio_path: str) -> "float | None":
    """Score an audio file using generate_tag() — returns MOS float 1–5 or None."""
    if _seg_model is None:
        return None
    try:
        score = _seg_model.generate_tag(audio_path)
        if hasattr(score, "item"):
            score = score.item()
        return round(float(score), 3)
    except Exception as e:
        warnings.warn(f"[SongEvalGen] inference error: {e}")
        return None


# ── ccmusic/acapella source ───────────────────────────────────────────────────

def load_ccmusic_clips():
    """
    Download ccmusic-database/acapella from HuggingFace and return:
        list of { "clip_id", "singer_id", "audio_array", "sr", "expert_scores" }

    Dataset structure (verified):
      - 6 splits: song1..song6, 22 rows each = 132 clips total
      - audio: torchcodec AudioDecoder; decode via .get_all_samples()
               returns AudioSamples with .data (channels, T) tensor at 48 kHz stereo
      - expert scores: float columns on 10-pt scale (4 judges averaged)
    """
    print("[ccmusic] Loading ccmusic-database/acapella from HuggingFace ...")
    from datasets import load_dataset, concatenate_datasets  # noqa

    ds_dict = load_dataset("ccmusic-database/acapella")
    ds = concatenate_datasets([ds_dict[split] for split in sorted(ds_dict.keys())])
    print(f"[ccmusic] {len(ds)} clips across {len(ds_dict)} song splits.")

    score_keys = [
        "pitch", "rhythm", "vocal_range", "timbre", "pronunciation",
        "vibrato", "dynamic", "breath_control", "overall_performance",
    ]

    clips = []
    for i, row in enumerate(ds):
        # AudioDecoder.get_all_samples() → AudioSamples(.data, .sample_rate)
        samples = row["audio"].get_all_samples()
        arr = np.array(samples.data, dtype=np.float32)  # (C, T)
        if arr.ndim == 2:
            arr = arr.mean(axis=0)                       # stereo → mono
        sr = int(samples.sample_rate)

        expert = {k: round(float(row[k]), 3) for k in score_keys if row.get(k) is not None}

        clips.append({
            "clip_id":       f"ccmusic_{i:04d}",
            "singer_id":     int(row["singer_id"]),
            "audio_array":   arr,
            "sr":            sr,
            "expert_scores": expert,
        })

    return clips


# ── PopBuTFy sources ──────────────────────────────────────────────────────────

def _load_popbutfy_by_suffix(dataset_dir, suffix, clip_prefix, label, max_clips=None):
    root = pathlib.Path(dataset_dir)
    audio_exts = {".wav", ".flac", ".mp3"}
    clips = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        if not folder.name.lower().endswith(suffix):
            continue
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() in audio_exts:
                clips.append({"clip_id": f"{clip_prefix}_{folder.name}_{f.stem}", "path": f})
    if max_clips is not None:
        clips = clips[:max_clips]
    print(f"[PopBuTFy] {len(clips)} {label} clips found in {root}")
    return clips


def load_popbutfy_amateur(dataset_dir, max_clips=None):
    return _load_popbutfy_by_suffix(dataset_dir, "_amateur", "popbutfy_am", "amateur", max_clips)


def load_popbutfy_professional(dataset_dir, max_clips=None):
    return _load_popbutfy_by_suffix(dataset_dir, "_professional", "popbutfy_pro", "professional", max_clips)


# ── VocalSet source ───────────────────────────────────────────────────────────

# ── YouTube covers source ─────────────────────────────────────────────────────

def load_yt_covers_clips(manifest_path, vocals_only=True):
    """
    Load clips from a download_yt_covers.py manifest, split into pro and amateur.
    vocals_only=True (default): only include clips where demucs succeeded
                                (vocals_wav exists) — full mix would corrupt scores.
    Returns (pro_clips, am_clips) each as list of {"clip_id", "path"}.
    """
    with open(manifest_path) as fh:
        records = json.load(fh)

    pro_clips, am_clips = [], []
    skipped = 0

    for rec in records:
        if not rec.get("download_ok"):
            continue

        vocals_path = pathlib.Path(rec.get("vocals_wav", ""))
        full_path   = pathlib.Path(rec.get("full_wav", ""))

        if vocals_only:
            if not rec.get("demucs_ok") or not vocals_path.exists():
                skipped += 1
                continue
            use_path = vocals_path
        else:
            use_path = vocals_path if vocals_path.exists() else full_path
            if not use_path.exists():
                skipped += 1
                continue

        slug    = rec.get("slug", "unknown")
        vid_id  = rec.get("video_id") or use_path.stem
        clip_id = f"yt_{slug}_{rec['role']}_{vid_id}"
        entry   = {"clip_id": clip_id, "path": use_path}

        if rec["role"] == "pro":
            pro_clips.append(entry)
        elif rec["role"] == "amateur":
            am_clips.append(entry)

    print(f"[YT covers] {len(pro_clips)} pro + {len(am_clips)} amateur clips "
          f"from {manifest_path}  ({skipped} skipped — no vocals stem)")
    if skipped:
        print(f"  Tip: run download_yt_covers.py --skip-existing to separate remaining clips.")
    return pro_clips, am_clips


def load_vocalset_clips(dataset_dir, max_clips=None):
    """
    Load VocalSet excerpts/straight clips — one per singer per song.
    These are trained professional vocalists singing without technique embellishment,
    making them the cleanest professional reference in the dataset.

    Structure: data_by_singer/<singer>/excerpts/straight/<singer_song_straight.wav>
    20 singers × 3 songs = 60 clips total.
    """
    root = pathlib.Path(dataset_dir) / "data_by_singer"
    clips = []

    for singer_dir in sorted(root.iterdir()):
        if not singer_dir.is_dir():
            continue
        straight_dir = singer_dir / "excerpts" / "straight"
        if not straight_dir.exists():
            continue
        for f in sorted(straight_dir.glob("*.wav")):
            clips.append({
                "clip_id": f"vocalset_{singer_dir.name}_{f.stem}",
                "path":    f,
            })

    if max_clips is not None:
        clips = clips[:max_clips]

    print(f"[VocalSet] {len(clips)} excerpts/straight clips found in {root}")
    return clips


# ── scoring loops ─────────────────────────────────────────────────────────────

def score_popbutfy_clips(clips, use_singmos, use_seg, existing=None, label="clips"):
    """
    Score clips that are already on disk — pass path directly to generate_tag().
    existing: dict of already-scored results to merge into (avoids re-scoring).
    label: tqdm progress bar description.
    """
    results = {} if existing is None else dict(existing)
    sr_16k  = _SINGMOS_SR

    for clip in tqdm(clips, desc=label):
        clip_id = clip["clip_id"]

        # skip if already has all requested scores
        existing_rec = results.get(clip_id, {})
        needs_singmos = use_singmos and "singmos"     not in existing_rec
        needs_seg     = use_seg     and "songevalgen" not in existing_rec
        if not needs_singmos and not needs_seg:
            continue

        if needs_singmos:
            try:
                y, sr = librosa.load(str(clip["path"]), sr=None, mono=True)
                y16 = librosa.resample(y, orig_sr=sr, target_sr=sr_16k) if sr != sr_16k else y
                existing_rec["singmos"] = singmos_score(y16)
            except Exception as e:
                warnings.warn(f"[PopBuTFy] load error {clip['path'].name}: {e}")
                continue

        if needs_seg:
            existing_rec["songevalgen"] = seg_score(str(clip["path"]))

        results[clip_id] = existing_rec

    return results


def score_ccmusic_clips_merge(clips, use_singmos, use_seg, existing=None):
    """Like score_ccmusic_clips but merges into existing results."""
    import tempfile, soundfile as sf

    results = {} if existing is None else dict(existing)
    sr_16k  = _SINGMOS_SR

    for clip in tqdm(clips, desc="ccmusic professional"):
        clip_id = clip["clip_id"]

        existing_rec = results.get(clip_id, {"expert_scores": clip["expert_scores"]})
        needs_singmos = use_singmos and "singmos"     not in existing_rec
        needs_seg     = use_seg     and "songevalgen" not in existing_rec
        if not needs_singmos and not needs_seg:
            results[clip_id] = existing_rec
            continue

        y  = clip["audio_array"]
        sr = clip["sr"]

        if needs_singmos:
            y16 = librosa.resample(y, orig_sr=sr, target_sr=sr_16k) if sr != sr_16k else y
            existing_rec["singmos"] = singmos_score(y16)

        if needs_seg:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                sf.write(tmp_path, y, sr)
                existing_rec["songevalgen"] = seg_score(tmp_path)
            finally:
                pathlib.Path(tmp_path).unlink(missing_ok=True)

        results[clip_id] = existing_rec

    return results


# ── GTSinger technique characterisation ──────────────────────────────────────

GTSINGER_TECHNIQUES = ["Breathy", "Glissando", "Mixed_Voice_and_Falsetto", "Pharyngeal", "Vibrato"]

# Technique folder name → list of technique group subfolder names
# Mixed_Voice_and_Falsetto splits into two separate groups on disk
GTSINGER_TECH_GROUP_NAMES = {
    "Breathy":                  ["Breathy_Group"],
    "Glissando":                ["Glissando_Group"],
    "Mixed_Voice_and_Falsetto": ["Mixed_Voice_Group", "Falsetto_Group"],
    "Pharyngeal":               ["Pharyngeal_Group"],
    "Vibrato":                  ["Vibrato_Group"],
}


def load_gtsinger_pairs(dataset_dir, languages=None):
    """
    Scan GTSinger root and return Control/Technique clip pairs.

    Structure:
      <lang>/<Singer>/<Technique>/<Song>/Control_Group/*.wav
      <lang>/<Singer>/<Technique>/<Song>/<Technique>_Group/*.wav

    Returns dict keyed by technique:
      { technique: [ {"singer", "song", "lang",
                       "control_clips": [path, ...],
                       "technique_clips": [path, ...]} ] }
    """
    root = pathlib.Path(dataset_dir)
    available_langs = [d.name for d in root.iterdir() if d.is_dir()]

    if languages is None:
        languages = available_langs
    else:
        languages = [l for l in languages if l in available_langs]

    pairs = {t: [] for t in GTSINGER_TECHNIQUES}

    for lang in languages:
        for singer_dir in sorted((root / lang).iterdir()):
            if not singer_dir.is_dir():
                continue
            for tech in GTSINGER_TECHNIQUES:
                tech_dir = singer_dir / tech
                if not tech_dir.exists():
                    continue
                for song_dir in sorted(tech_dir.iterdir()):
                    if not song_dir.is_dir():
                        continue
                    ctrl  = sorted((song_dir / "Control_Group").glob("*.wav")) \
                            if (song_dir / "Control_Group").exists() else []
                    tgrp = []
                    for grp_name in GTSINGER_TECH_GROUP_NAMES.get(tech, [f"{tech}_Group"]):
                        tgrp += sorted((song_dir / grp_name).glob("*.wav")) \
                                if (song_dir / grp_name).exists() else []
                    if ctrl or tgrp:
                        pairs[tech].append({
                            "lang":            lang,
                            "singer":          singer_dir.name,
                            "song":            song_dir.name,
                            "control_clips":   ctrl,
                            "technique_clips": tgrp,
                        })

    total = sum(
        len(g["control_clips"]) + len(g["technique_clips"])
        for tech_list in pairs.values() for g in tech_list
    )
    langs_used = ", ".join(languages)
    print(f"[GTSinger] {total} clips across {len(languages)} language(s) ({langs_used})")
    for tech, groups in pairs.items():
        n_ctrl = sum(len(g["control_clips"])   for g in groups)
        n_tech = sum(len(g["technique_clips"]) for g in groups)
        print(f"  {tech:<28} control={n_ctrl:4d}  technique={n_tech:4d}")
    return pairs


def score_gtsinger(pairs, use_singmos, use_seg, output_path, device="cuda"):
    """
    Score GTSinger Control vs Technique clips and print a delta table.
    Saves per-clip results to output_path.
    """
    results = {}   # technique → { "control": [...scores], "technique": [...scores] }

    for tech, groups in pairs.items():
        ctrl_scores  = []
        tech_scores  = []
        clip_records = []

        all_clips = []
        for g in groups:
            for p in g["control_clips"]:
                all_clips.append(("control", g["singer"], g["song"], p))
            for p in g["technique_clips"]:
                all_clips.append(("group", g["singer"], g["song"], p))

        for group_type, singer, song, path in tqdm(all_clips, desc=f"GTSinger/{tech}"):
            rec = {"path": str(path), "singer": singer, "song": song, "group": group_type}

            if use_singmos:
                try:
                    y, sr = librosa.load(str(path), sr=None, mono=True)
                    y16   = librosa.resample(y, orig_sr=sr, target_sr=_SINGMOS_SR) \
                            if sr != _SINGMOS_SR else y
                    rec["singmos"] = singmos_score(y16)
                except Exception as e:
                    warnings.warn(f"[GTSinger/SingMOS] {path.name}: {e}")
                    rec["singmos"] = None

            if use_seg:
                rec["songevalgen"] = seg_score(str(path))

            clip_records.append(rec)
            score_key = "singmos" if use_singmos else "songevalgen"
            s = rec.get(score_key)
            if s is not None:
                (ctrl_scores if group_type == "control" else tech_scores).append(s)

        results[tech] = {"clips": clip_records,
                         "control_scores":   ctrl_scores,
                         "technique_scores": tech_scores}

    # Print delta table
    model_label = ("SingMOS+SongEvalGen" if use_singmos and use_seg
                   else "SingMOS" if use_singmos else "SongEvalGen")
    print(f"\n{'Technique':<28}  {'Control mean':>13}  {'Technique mean':>14}  {'Delta':>7}  {'n_ctrl':>6}  {'n_tech':>6}")
    print("─" * 80)

    for tech in GTSINGER_TECHNIQUES:
        r = results.get(tech, {})
        ctrl = r.get("control_scores",   [])
        tech_s = r.get("technique_scores", [])
        if not ctrl and not tech_s:
            continue
        cm = round(float(np.mean(ctrl)),   3) if ctrl   else None
        tm = round(float(np.mean(tech_s)), 3) if tech_s else None
        delta_str = f"{tm - cm:+.3f}" if cm is not None and tm is not None else "n/a"
        cm_str = f"{cm:.3f}" if cm is not None else "n/a"
        tm_str = f"{tm:.3f}" if tm is not None else "n/a"
        print(f"  {tech:<26}  {cm_str:>13}  {tm_str:>14}  {delta_str:>7}  {len(ctrl):>6}  {len(tech_s):>6}")

    print(f"\n  ({model_label} scores used for delta; positive = technique scores higher than neutral)")

    # Save
    save_data = {
        tech: {
            "control_stats":   percentile_stats(r["control_scores"]),
            "technique_stats": percentile_stats(r["technique_scores"]),
            "clips":           r["clips"],
        }
        for tech, r in results.items()
    }
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(save_data, fh, indent=2)
    print(f"\nGTSinger results saved → {output_path}")


# ── summary statistics ────────────────────────────────────────────────────────

def compute_summary(levels_results, use_singmos, use_seg):
    """
    levels_results: ordered dict of { level_name: results_dict }
    expert scores are read from ccmusic_professional only.
    """
    summary = {}

    def gather(results, key):
        return [rec[key] for rec in results.values() if rec.get(key) is not None]

    for level, results in levels_results.items():
        summary[level] = {}

        if use_singmos:
            summary[level]["singmos"] = percentile_stats(gather(results, "singmos"))

        if use_seg:
            summary[level]["songevalgen"] = percentile_stats(gather(results, "songevalgen"))

        if level == "ccmusic_professional":
            expert_keys = [
                "pitch", "rhythm", "vocal_range", "timbre", "pronunciation",
                "vibrato", "dynamic", "breath_control", "overall_performance",
            ]
            for k in expert_keys:
                vals = [
                    rec["expert_scores"].get(k)
                    for rec in results.values()
                    if rec.get("expert_scores") and rec["expert_scores"].get(k) is not None
                ]
                if vals:
                    summary[level][f"expert_{k}"] = percentile_stats(vals)

    return summary


def print_summary(summary, use_singmos, use_seg):
    levels = list(summary.keys())
    # collect all dim names from any level
    dims = []
    seen = set()
    for lvl in levels:
        for d in summary[lvl]:
            if d not in seen:
                dims.append(d)
                seen.add(d)

    col_w = 16
    print(f"\n{'Dimension':<35}", end="")
    for lvl in levels:
        print(f"  {lvl[:col_w-2]:>{col_w}} mean", end="")
    print()
    print("─" * (35 + (col_w + 7) * len(levels)))

    for dim in dims:
        print(f"  {dim:<33}", end="")
        for lvl in levels:
            m = summary[lvl].get(dim, {}).get("mean")
            print(f"  {str(round(m, 3)) if m is not None else 'n/a':>{col_w}}", end="")
        print()


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Build combined amateur/professional eval baselines (Step 1)"
    )
    p.add_argument("--popbutfy",       default="data/popbutfy",
                   help="Path to PopBuTFy root directory (default: data/popbutfy)")
    p.add_argument("--vocalset",        default="data/vocalset",
                   help="Path to VocalSet root directory (default: data/vocalset)")
    p.add_argument("--gtsinger",        default=None, metavar="DIR",
                   help="Path to GTSinger audio root to run technique characterisation mode. "
                        "If omitted, GTSinger scoring is skipped.")
    p.add_argument("--gtsinger-lang",   nargs="+", default=["English"], metavar="LANG",
                   help="Language(s) to include in GTSinger run (default: English). "
                        "Pass 'all' to include every language folder.")
    p.add_argument("--gtsinger-output", default="data/gtsinger_technique_scores.json",
                   help="Output path for GTSinger technique scores "
                        "(default: data/gtsinger_technique_scores.json)")
    p.add_argument("--output",         default="data/combined_eval_baselines.json",
                   help="Output JSON path (default: data/combined_eval_baselines.json)")
    p.add_argument("--summary-output", default="data/combined_eval_summary.json",
                   help="Summary stats JSON path (default: data/combined_eval_summary.json)")
    p.add_argument("--no-singmos",     action="store_true",
                   help="Skip SingMOS-Pro scoring")
    p.add_argument("--no-songevalgen", action="store_true",
                   help="Skip SongEvalGenerator scoring")
    p.add_argument("--merge-existing", metavar="JSON",
                   help="Load existing baselines JSON and add new scores without re-scoring "
                        "clips that already have the requested keys")
    p.add_argument("--yt-manifest",    default=None, metavar="JSON",
                   help="Path to manifest.json produced by download_yt_covers.py. "
                        "Adds yt_professional and yt_amateur levels to the output. "
                        "Only clips with demucs_ok vocals are included by default.")
    p.add_argument("--yt-full-mix",    action="store_true",
                   help="Include YT clips that only have the full mix (no vocals stem). "
                        "Not recommended — backing track inflates scores.")
    p.add_argument("--max-amateur",    type=int, default=None,
                   help="Limit number of amateur clips (for quick testing)")
    p.add_argument("--device",         default="cuda", choices=["cpu", "cuda"])
    return p.parse_args()


def main():
    args = parse_args()
    use_singmos = not args.no_singmos
    use_seg     = not args.no_songevalgen

    if not use_singmos and not use_seg:
        print("Both models disabled — nothing to do. Remove --no-singmos or --no-songevalgen.")
        sys.exit(1)

    # Load existing results if merging
    existing_ccmusic_pro, existing_pb_pro, existing_pb_am = {}, {}, {}
    existing_vocalset, existing_yt_pro, existing_yt_am    = {}, {}, {}
    if args.merge_existing:
        merge_path = pathlib.Path(args.merge_existing)
        if merge_path.exists():
            with open(merge_path) as fh:
                prev = json.load(fh)
            existing_ccmusic_pro = prev.get("ccmusic_professional", prev.get("professional", {}))
            existing_pb_pro      = prev.get("popbutfy_professional", {})
            existing_pb_am       = prev.get("popbutfy_amateur",      prev.get("amateur", {}))
            existing_vocalset    = prev.get("vocalset_professional", {})
            existing_yt_pro      = prev.get("yt_professional", {})
            existing_yt_am       = prev.get("yt_amateur", {})
            print(f"[merge] Loaded {len(existing_ccmusic_pro)} ccmusic_pro + "
                  f"{len(existing_pb_pro)} pb_pro + {len(existing_pb_am)} pb_am + "
                  f"{len(existing_vocalset)} vocalset + "
                  f"{len(existing_yt_pro)} yt_pro + {len(existing_yt_am)} yt_am "
                  f"from {merge_path}")
        else:
            print(f"[merge] {merge_path} not found — starting fresh.")

    # Load models
    if use_singmos:
        try:
            load_singmos(args.device)
        except Exception as e:
            print(f"[SingMOS] Failed to load: {e}\nContinuing without SingMOS.")
            use_singmos = False

    if use_seg:
        try:
            load_songevalgen(args.device)
        except Exception as e:
            print(f"[SongEvalGen] Failed to load: {e}\nContinuing without SongEvalGenerator.")
            use_seg = False

    if not use_singmos and not use_seg:
        print("Both models failed to load. Exiting.")
        sys.exit(1)

    # Load sources
    ccmusic_clips   = load_ccmusic_clips()
    pb_pro_clips    = load_popbutfy_professional(args.popbutfy, max_clips=args.max_amateur)
    pb_am_clips     = load_popbutfy_amateur(args.popbutfy,      max_clips=args.max_amateur)
    vocalset_clips  = load_vocalset_clips(args.vocalset)

    yt_pro_clips, yt_am_clips = [], []
    if args.yt_manifest:
        vocals_only = not args.yt_full_mix
        yt_pro_clips, yt_am_clips = load_yt_covers_clips(args.yt_manifest,
                                                          vocals_only=vocals_only)
        if args.max_amateur and yt_am_clips:
            yt_am_clips = yt_am_clips[: args.max_amateur]

    print(f"\nSources: {len(ccmusic_clips)} ccmusic_pro  |  "
          f"{len(pb_pro_clips)} pb_pro  |  {len(pb_am_clips)} pb_am  |  "
          f"{len(vocalset_clips)} vocalset_pro  |  "
          f"{len(yt_pro_clips)} yt_pro  |  {len(yt_am_clips)} yt_am")
    print(f"Models: singmos={use_singmos}  songevalgen={use_seg}")
    if args.merge_existing:
        print("Merge mode: existing scores preserved, only missing keys scored.")
    print()

    # Score (merge-aware)
    ccmusic_results  = score_ccmusic_clips_merge(
        ccmusic_clips,  use_singmos, use_seg, existing=existing_ccmusic_pro)
    pb_pro_results   = score_popbutfy_clips(
        pb_pro_clips,   use_singmos, use_seg, existing=existing_pb_pro,
        label="PopBuTFy professional")
    pb_am_results    = score_popbutfy_clips(
        pb_am_clips,    use_singmos, use_seg, existing=existing_pb_am,
        label="PopBuTFy amateur")
    vocalset_results = score_popbutfy_clips(
        vocalset_clips, use_singmos, use_seg, existing=existing_vocalset,
        label="VocalSet professional")

    yt_pro_results = score_popbutfy_clips(
        yt_pro_clips, use_singmos, use_seg, existing=existing_yt_pro,
        label="YT professional") if yt_pro_clips else existing_yt_pro
    yt_am_results  = score_popbutfy_clips(
        yt_am_clips,  use_singmos, use_seg, existing=existing_yt_am,
        label="YT amateur")  if yt_am_clips  else existing_yt_am

    levels_results = {
        "ccmusic_professional":  ccmusic_results,
        "vocalset_professional": vocalset_results,
        "popbutfy_professional": pb_pro_results,
        "popbutfy_amateur":      pb_am_results,
    }
    if yt_pro_results or yt_am_results:
        levels_results["yt_professional"] = yt_pro_results
        levels_results["yt_amateur"]      = yt_am_results

    # Summary stats
    summary = compute_summary(levels_results, use_singmos, use_seg)
    print_summary(summary, use_singmos, use_seg)

    # Controlled separation: same-singer PopBuTFy pro vs amateur
    print()
    for key, label in [("singmos", "SingMOS"), ("songevalgen", "SongEvalGen")]:
        pro_m = summary.get("popbutfy_professional", {}).get(key, {}).get("mean")
        am_m  = summary.get("popbutfy_amateur",      {}).get(key, {}).get("mean")
        if pro_m is not None and am_m is not None:
            sep = pro_m - am_m
            verdict = "good separation" if sep > 0.3 else "poor separation — ceiling/inversion likely"
            print(f"{label} PopBuTFy pro−am (controlled): {sep:+.3f}  →  {verdict}")

    # YT cover separation: pro original vs amateur cover (different singers — weaker test)
    if yt_pro_results or yt_am_results:
        print()
        for key, label in [("singmos", "SingMOS"), ("songevalgen", "SongEvalGen")]:
            yt_pro_m = summary.get("yt_professional", {}).get(key, {}).get("mean")
            yt_am_m  = summary.get("yt_amateur",      {}).get(key, {}).get("mean")
            if yt_pro_m is not None and yt_am_m is not None:
                sep = yt_pro_m - yt_am_m
                verdict = "good separation" if sep > 0.3 else "weak"
                print(f"{label} YT pro−amateur (different singers): {sep:+.3f}  →  {verdict}")

    # Save full results
    output = {
        "ccmusic_professional":  ccmusic_results,
        "vocalset_professional": vocalset_results,
        "popbutfy_professional": pb_pro_results,
        "popbutfy_amateur":      pb_am_results,
        "_meta": {
            "ccmusic_professional_source":  "ccmusic-database/acapella (HF)",
            "vocalset_professional_source": "VocalSet data_by_singer/*/excerpts/straight/",
            "popbutfy_professional_source": "PopBuTFy *_Professional folders",
            "popbutfy_amateur_source":      "PopBuTFy *_Amateur folders",
            "n_ccmusic_professional":       len(ccmusic_results),
            "n_vocalset_professional":      len(vocalset_results),
            "n_popbutfy_professional":      len(pb_pro_results),
            "n_popbutfy_amateur":           len(pb_am_results),
            "models_used":                  (["singmos"] if use_singmos else []) +
                                            (["songevalgen"] if use_seg else []),
            "note": (
                "PopBuTFy amateur/professional are the same singer performing poorly vs. well — "
                "controlled pair. ccmusic expert scores are on a 10-pt scale. "
                "VocalSet clips are trained professional singers (excerpts/straight only)."
            ),
        },
    }
    if yt_pro_results or yt_am_results:
        output["yt_professional"] = yt_pro_results
        output["yt_amateur"]      = yt_am_results
        output["_meta"]["yt_professional_source"] = "YouTube cover dataset — original artist vocals (demucs)"
        output["_meta"]["yt_amateur_source"]      = "YouTube cover dataset — amateur cover vocals (demucs)"
        output["_meta"]["n_yt_professional"]      = len(yt_pro_results)
        output["_meta"]["n_yt_amateur"]           = len(yt_am_results)

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2)
    print(f"\nFull results saved → {out_path}")

    sum_path = pathlib.Path(args.summary_output)
    with open(sum_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Summary stats  saved → {sum_path}")

    # ── GTSinger technique characterisation (optional, runs after baselines) ──
    if args.gtsinger:
        gt_root = pathlib.Path(args.gtsinger)
        if not gt_root.exists():
            print(f"\n[GTSinger] Directory not found: {gt_root} — skipping.")
        else:
            langs = None if (len(args.gtsinger_lang) == 1 and
                             args.gtsinger_lang[0].lower() == "all") \
                         else args.gtsinger_lang
            print(f"\n{'='*60}")
            print("GTSinger Technique Characterisation")
            print(f"{'='*60}")
            gt_pairs = load_gtsinger_pairs(gt_root, languages=langs)
            score_gtsinger(gt_pairs, use_singmos, use_seg,
                           output_path=args.gtsinger_output, device=args.device)


if __name__ == "__main__":
    main()
