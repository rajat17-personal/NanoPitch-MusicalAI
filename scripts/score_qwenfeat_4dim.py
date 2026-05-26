"""
QwenFeat 4-Dimension Vocal Scorer
==================================

Runs the Qwen2-Audio-7B + 4 LoRA checkpoint system from QwenFeat-Vocal-Score
to produce per-dimension scores (1–5) for a set of audio files:

  Dim 0 — Technique  (专业技巧): pitch accuracy, ornaments, vocal control
  Dim 1 — Emotion    (情感表达): style match, expression, dynamics
  Dim 2 — Timbre     (音色与音质): tone quality, resonance, clarity
  Dim 3 — Breath     (气息控制): breath placement, phrase continuity

The 7B base model is loaded ONCE and shared across all 4 LoRA adapters,
so GPU memory is ~14–15 GB fp16 total (tight on 16 GB — close other processes).

Expected runtime: ~15–25 s/clip × 4 passes = 60–100 s/clip.
For 200 clips: 3–5 hours. Designed to run overnight.

Setup (one-time)
----------------
  # 7B weights must be actually downloaded (not just LFS pointers)
  python -c "
  from huggingface_hub import snapshot_download
  snapshot_download('Qwen/Qwen2-Audio-7B-Instruct',
                    local_dir='$HOME/QwenFeat/ckpts/Qwen2-Audio-7B-Instruct')
  "
  pip install peft transformers accelerate

Usage
-----
  # Score individual files
  QWENFEAT_ROOT=~/QwenFeat python scripts/score_qwenfeat_4dim.py \\
      audio/clip1.wav audio/clip2.wav

  # Score all vocal stems from the YouTube cover dataset
  QWENFEAT_ROOT=~/QwenFeat python scripts/score_qwenfeat_4dim.py \\
      --from-manifest data/yt_covers/manifest.json \\
      --output        data/qwenfeat_4dim_scores.json

  # Score ccmusic clips for correlation analysis (uses vocals extracted earlier)
  QWENFEAT_ROOT=~/QwenFeat python scripts/score_qwenfeat_4dim.py \\
      --from-manifest data/yt_covers/manifest.json \\
      --output        data/qwenfeat_4dim_scores.json \\
      --skip-existing

  # Limit to N clips (smoke test before overnight run)
  QWENFEAT_ROOT=~/QwenFeat python scripts/score_qwenfeat_4dim.py \\
      --from-manifest data/yt_covers/manifest.json \\
      --max-clips 5

Output
------
  data/qwenfeat_4dim_scores.json
  {
    "<audio_path>": {
      "technique": 3,   "technique_text": "...",
      "emotion":   4,   "emotion_text":   "...",
      "timbre":    3,   "timbre_text":    "...",
      "breath":    4,   "breath_text":    "...",
      "role":      "pro" | "amateur" | null,
      "slug":      "<song_slug>" | null
    },
    ...
  }
"""

import argparse
import json
import os
import pathlib
import sys
import time
import warnings

import numpy as np

# ── locate QwenFeat repo ──────────────────────────────────────────────────────

QWENFEAT_ROOT = pathlib.Path(os.environ.get("QWENFEAT_ROOT", str(pathlib.Path.home() / "QwenFeat")))
QWENAUDIO_SRC = QWENFEAT_ROOT / "qwenaudio" / "src"
QWENAUDIO_CKPTS = QWENFEAT_ROOT / "ckpts"
BASE_MODEL_PATH = QWENAUDIO_CKPTS / "Qwen2-Audio-7B-Instruct"

# LoRA checkpoints — from infer.py in qwenaudio/scripts/
SCORE_CKPTS = [
    str(QWENAUDIO_CKPTS / "train_ds_4_score_al/denoise/0/score/best_model_epoch/8"),
    str(QWENAUDIO_CKPTS / "train_ds_4_al/denoise/1/score/best_model_epoch_39/lora_weights"),
    str(QWENAUDIO_CKPTS / "train_ds_4_feat_score_al/denoise/2/score/best_model_epoch/25"),
    str(QWENAUDIO_CKPTS / "train_ds_4_feat_score_al/denoise/3/score/best_model_epoch/5"),
]
TEXT_CKPT = str(QWENAUDIO_CKPTS / "generator-lora-32-16-textonly-simple-v2-int4/best_model_epoch_16/lora_weights")

# dim 1 (Emotion) uses its own text LoRA and top2_mode=True
TEXT_CKPTS = [
    TEXT_CKPT,
    str(QWENAUDIO_CKPTS / "train_ds_4_al/denoise/1/text/best_model_epoch_39/lora_weights"),
    TEXT_CKPT,
    TEXT_CKPT,
]
TOP2_MODES = [False, True, False, False]

DIM_NAMES = ["technique", "emotion", "timbre", "breath"]
DIM_ZH    = ["专业技巧", "情感表达", "音色与音质", "气息控制"]

SR = 16000
MAX_SECONDS = 30   # qwenaudio clips audio to first 30s


# ── model loading ─────────────────────────────────────────────────────────────

_processor_group = None


def _check_weights():
    """Verify 7B weights are actually downloaded (not LFS pointers)."""
    shard = BASE_MODEL_PATH / "model-00001-of-00005.safetensors"
    if not shard.exists():
        raise FileNotFoundError(
            f"7B model shard not found at {shard}\n"
            "Download with:\n"
            "  from huggingface_hub import snapshot_download\n"
            f"  snapshot_download('Qwen/Qwen2-Audio-7B-Instruct', local_dir='{BASE_MODEL_PATH}')"
        )
    size = shard.stat().st_size
    if size < 1_000_000:   # LFS pointer files are ~135 bytes
        raise RuntimeError(
            f"Model shard at {shard} is only {size} bytes — looks like an LFS pointer.\n"
            "Run snapshot_download to pull actual weights (see --help for command)."
        )


def load_processor_group():
    global _processor_group
    if _processor_group is not None:
        return _processor_group

    _check_weights()

    if not QWENAUDIO_SRC.exists():
        raise FileNotFoundError(
            f"qwenaudio source not found at {QWENAUDIO_SRC}.\n"
            f"Set QWENFEAT_ROOT correctly (currently: {QWENFEAT_ROOT})"
        )

    sys.path.insert(0, str(QWENAUDIO_SRC))
    # must cd to qwenaudio root so relative config/ paths resolve
    orig_dir = os.getcwd()
    os.chdir(str(QWENFEAT_ROOT / "qwenaudio"))

    try:
        import qwenaudio.processor as proc_mod

        print(f"[QwenFeat] Loading Qwen2-Audio-7B from {BASE_MODEL_PATH} ...")
        t0 = time.time()
        pg = proc_mod.ProcessorGroup(
            base_model_name=str(BASE_MODEL_PATH),
            processor_name=str(BASE_MODEL_PATH),
        )
        for i in range(4):
            print(f"[QwenFeat] Adding dim {i} ({DIM_NAMES[i]}) LoRA ...")
            pg.add(SCORE_CKPTS[i], TEXT_CKPTS[i])
            pg.models[i].top2_mode = TOP2_MODES[i]

        print(f"[QwenFeat] All 4 dims loaded in {time.time()-t0:.1f}s")
        _processor_group = pg
        return pg

    finally:
        os.chdir(orig_dir)


# ── scoring ───────────────────────────────────────────────────────────────────

def score_file(path, pg):
    """
    Score a single audio file across all 4 dimensions.
    Returns dict with technique/emotion/timbre/breath scores (1–5) + text.
    """
    import librosa

    path = str(path)
    y, sr = librosa.load(path, sr=SR, mono=True)
    y = y[: SR * MAX_SECONDS]   # clip to 30s — matches infer.py behaviour

    result = {}
    for i, dim in enumerate(DIM_NAMES):
        t0 = time.time()
        try:
            out = pg.models[i].generate(y, i, simple_model=True)
            result[dim]           = out.get("score", None)
            result[f"{dim}_text"] = out.get("text", "")
        except Exception as e:
            warnings.warn(f"[QwenFeat] dim {dim} failed on {pathlib.Path(path).name}: {e}")
            result[dim]           = None
            result[f"{dim}_text"] = f"error: {e}"
        print(f"  dim={dim:<10} score={result[dim]}  ({time.time()-t0:.1f}s)")

    return result


def score_files(paths, pg, existing=None, meta_map=None):
    """
    Score a list of paths. existing: dict of already-scored records (skip those).
    meta_map: {path: {role, slug}} for manifest-sourced runs.
    """
    results = {} if existing is None else dict(existing)
    meta_map = meta_map or {}

    for i, path in enumerate(paths):
        path_str = str(path)
        if path_str in results:
            print(f"[{i+1}/{len(paths)}] skipping (already scored): {pathlib.Path(path_str).name}")
            continue

        print(f"\n[{i+1}/{len(paths)}] {pathlib.Path(path_str).name}")
        t0 = time.time()
        rec = score_file(path_str, pg)
        rec.update(meta_map.get(path_str, {"role": None, "slug": None}))
        results[path_str] = rec
        print(f"  total: {time.time()-t0:.1f}s  scores: "
              + "  ".join(f"{d}={rec.get(d)}" for d in DIM_NAMES))

    return results


# ── manifest loading ──────────────────────────────────────────────────────────

def load_from_manifest(manifest_path):
    """
    Load paths and metadata from download_yt_covers.py manifest.
    Returns (paths, meta_map) where meta_map[path] = {role, slug}.
    Uses vocals_wav if it exists (post-demucs), falls back to full_wav.
    """
    with open(manifest_path) as fh:
        records = json.load(fh)

    paths = []
    meta_map = {}

    for rec in records:
        if rec.get("skipped") and not rec.get("download_ok"):
            continue

        # prefer separated vocals, fall back to full mix
        wav = rec.get("vocals_wav") or rec.get("full_wav")
        if not wav:
            continue
        p = pathlib.Path(wav)
        if not p.exists():
            # try full_wav as fallback
            p = pathlib.Path(rec.get("full_wav", ""))
            if not p.exists():
                warnings.warn(f"[manifest] file not found: {wav}")
                continue

        paths.append(p)
        meta_map[str(p)] = {
            "role": rec.get("role"),
            "slug": rec.get("slug"),
            "title": rec.get("title", ""),
            "video_id": rec.get("video_id", ""),
        }

    print(f"[manifest] {len(paths)} clips loaded from {manifest_path}")
    pro_n = sum(1 for m in meta_map.values() if m["role"] == "pro")
    am_n  = sum(1 for m in meta_map.values() if m["role"] == "amateur")
    print(f"           pro={pro_n}  amateur={am_n}")
    return paths, meta_map


# ── summary ───────────────────────────────────────────────────────────────────

def print_summary(results):
    from scipy import stats as _stats

    roles = {"pro": [], "amateur": [], "unknown": []}
    for rec in results.values():
        role = rec.get("role") or "unknown"
        if role not in roles:
            role = "unknown"
        roles[role].append(rec)

    print("\n\nSummary")
    print("─" * 70)
    header = f"  {'Role':<12}" + "".join(f"  {d[:8]:>10}" for d in DIM_NAMES)
    print(header)
    print("─" * 70)

    for role, recs in roles.items():
        if not recs:
            continue
        row = f"  {role:<12}"
        for dim in DIM_NAMES:
            vals = [r[dim] for r in recs if r.get(dim) is not None]
            mean = round(float(np.mean(vals)), 2) if vals else None
            row += f"  {str(mean) if mean else 'n/a':>10}"
        row += f"  (n={len(recs)})"
        print(row)

    # separation
    if roles["pro"] and roles["amateur"]:
        print("\n  Separation (pro − amateur):")
        for dim in DIM_NAMES:
            pro_vals = [r[dim] for r in roles["pro"]     if r.get(dim) is not None]
            am_vals  = [r[dim] for r in roles["amateur"] if r.get(dim) is not None]
            if pro_vals and am_vals:
                sep = np.mean(pro_vals) - np.mean(am_vals)
                verdict = "good" if sep > 0.3 else "weak"
                print(f"    {dim:<12}: {sep:+.2f}  ({verdict})")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="QwenFeat 4-dimension vocal scorer (overnight run)"
    )
    p.add_argument("files", nargs="*",
                   help="Audio files to score directly")
    p.add_argument("--from-manifest", metavar="JSON",
                   help="Load paths from download_yt_covers.py manifest "
                        "(uses vocals_wav if available, else full_wav)")
    p.add_argument("--output", default="data/qwenfeat_4dim_scores.json",
                   help="Output JSON path (default: data/qwenfeat_4dim_scores.json)")
    p.add_argument("--skip-existing", action="store_true",
                   help="Load --output if it exists and skip already-scored clips")
    p.add_argument("--max-clips", type=int, default=None,
                   help="Limit total clips scored (useful for smoke test)")
    p.add_argument("--summary-only", action="store_true",
                   help="Print summary of existing output file without scoring")
    return p.parse_args()


def main():
    args = parse_args()

    out_path = pathlib.Path(args.output)

    # summary-only mode
    if args.summary_only:
        if not out_path.exists():
            print(f"No output file found at {out_path}")
            sys.exit(1)
        with open(out_path) as fh:
            results = json.load(fh)
        print(f"Loaded {len(results)} records from {out_path}")
        print_summary(results)
        return

    # collect paths
    paths = []
    meta_map = {}

    if args.from_manifest:
        ps, mm = load_from_manifest(args.from_manifest)
        paths.extend(ps)
        meta_map.update(mm)

    if args.files:
        for f in args.files:
            p = pathlib.Path(f)
            if not p.exists():
                warnings.warn(f"File not found: {f}")
                continue
            paths.append(p)

    if not paths:
        print("No audio files specified. Use positional args or --from-manifest.")
        sys.exit(1)

    if args.max_clips:
        paths = paths[: args.max_clips]
        print(f"[limit] Capped at {args.max_clips} clips.")

    # load existing results for skip-existing
    existing = {}
    if args.skip_existing and out_path.exists():
        with open(out_path) as fh:
            existing = json.load(fh)
        print(f"[skip-existing] Loaded {len(existing)} already-scored records.")

    # estimate time
    n_todo = sum(1 for p in paths if str(p) not in existing)
    est_min = n_todo * 90 / 60   # ~90s per clip (4 dims × ~22s average)
    print(f"\nClips to score: {n_todo}  (estimated {est_min:.0f} min = {est_min/60:.1f} hr)")
    print("Starting...\n")

    # load model — expensive, do after path validation
    pg = load_processor_group()

    # score
    results = score_files(paths, pg, existing=existing, meta_map=meta_map)

    # save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print(f"\nResults saved → {out_path}  ({len(results)} total records)")

    print_summary(results)


if __name__ == "__main__":
    main()
