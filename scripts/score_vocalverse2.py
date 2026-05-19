"""
Option B — VocalVerse2 (MuQ / audioscore) Vocal Scorer
=======================================================

Runs the lightweight MuQ-based SongEvalGenerator_audio_lora checkpoint from
  https://huggingface.co/karl-wang/QwenFeat-Vocal-Score
on a list of audio files and prints scores.

This is the local-runnable option: the MuQ encoder + LoRA scoring head are
much smaller than the 7B Qwen model and fit alongside the VocalCoach model on
a 15 GB GPU. It produces a single aesthetic score on a 1–5 MOS scale (higher
= better; generate_tag returns 5 - raw_logit).

Setup (one-time)
----------------
    # 1. Clone the audioscore source into a sibling directory
    git clone https://huggingface.co/karl-wang/QwenFeat-Vocal-Score /tmp/QwenFeat

    # 2. Download the actual model weights (LFS files are not pulled by git clone)
    python -c "
from huggingface_hub import hf_hub_download
import os
root, repo = '/tmp/QwenFeat', 'karl-wang/QwenFeat-Vocal-Score'
for f in [
    'audioscore/ckpts/SongEvalGenerator/step_2_al_audio/best_model_step_132000/muq_lora.pt',
    'audioscore/ckpts/SongEvalGenerator/step_2_al_audio/best_model_step_132000/weights.pt',
    'audioscore/ckpts/SongEvalGenerator/step_2_al_audio/best_model_step_132000/type.txt',
    'audioscore/src/SongEval/ckpt/model.safetensors',
]:
    os.makedirs(os.path.join(root, os.path.dirname(f)), exist_ok=True)
    hf_hub_download(repo_id=repo, filename=f, local_dir=root)
    print('downloaded', f)
"

    # 3. Install audioscore dependencies into your current env
    pip install muq==0.1.0 pyworld pyloudnorm   # omit parselmouth — Python 3.14 incompatible

    # 4. Set QWENFEAT_ROOT to the cloned repo so this script finds the source
    export QWENFEAT_ROOT=/tmp/QwenFeat

Usage
-----
    # Single file
    PYTHONPATH=. python scripts/score_vocalverse2.py evalrecordings/house.wav

    # Multiple files
    PYTHONPATH=. python scripts/score_vocalverse2.py path/to/a.wav path/to/b.wav

    # Batch across a PopBuTFy song pair — prints amateur vs professional side by side
    PYTHONPATH=. python scripts/score_vocalverse2.py \\
        --popbutfy data/popbutfy \\
        --singer Female1 \\
        --song my_heart_will_go_on

    # Build population baselines across ALL PopBuTFy clips (like buildPopBuTFyBaselines.py)
    PYTHONPATH=. python scripts/score_vocalverse2.py \\
        --build-baselines data/popbutfy \\
        --output data/vocalverse2_baselines.json

Output
------
    Each scored file prints:
        path | vocalverse2_score (1.0–5.0 MOS) | grade

    --build-baselines prints per-level statistics and saves JSON with
    mean / median / p25 / p75 / std for amateur and professional.
"""

import argparse
import json
import os
import sys
import warnings
import pathlib

import numpy as np
import librosa
from tqdm import tqdm

# ── Locate the QwenFeat repo source ──────────────────────────────────────────
QWENFEAT_ROOT = pathlib.Path(
    os.environ.get("QWENFEAT_ROOT", "/tmp/QwenFeat")
)
AUDIOSCORE_SRC = QWENFEAT_ROOT / "audioscore" / "src"
CKPT_DIR = QWENFEAT_ROOT / "audioscore" / "ckpts" / "SongEvalGenerator" / "step_2_al_audio" / "best_model_step_132000"

SR = 16000   # MuQ / audioscore models expect 16 kHz

GRADE_THRESHOLDS = [
    (4.5, "excellent"),
    (4.0, "good"),
    (3.5, "acceptable"),
    (2.5, "developing"),
    (0.0, "needs_work"),
]


def grade(score):
    for threshold, label in GRADE_THRESHOLDS:
        if score >= threshold:
            return label
    return "needs_work"


def load_model(device="cuda"):
    """Load SongEvalGenerator_audio_lora from the audioscore checkpoint."""
    if not AUDIOSCORE_SRC.exists():
        raise FileNotFoundError(
            f"audioscore source not found at {AUDIOSCORE_SRC}.\n"
            "Set QWENFEAT_ROOT to the cloned QwenFeat-Vocal-Score repo."
        )
    if not CKPT_DIR.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {CKPT_DIR}.\n"
            "Download the HF repo: huggingface.co/karl-wang/QwenFeat-Vocal-Score"
        )

    sys.path.insert(0, str(AUDIOSCORE_SRC))
    from audioscore.model import SongEvalGenerator_audio_lora   # noqa: PLC0415

    model = SongEvalGenerator_audio_lora()
    model.load_model(str(CKPT_DIR))
    model = model.to(device)
    model.eval()
    print(f"[VocalVerse2] SongEvalGenerator_audio_lora loaded from {CKPT_DIR}")
    return model


def score_file(model, audio_path, device="cuda"):
    """Score a single audio file. Returns float on 1–5 MOS scale."""
    score = model.generate_tag(str(audio_path))
    if hasattr(score, "item"):
        score = score.item()
    return float(score)


def score_files(model, paths, device="cuda"):
    results = []
    for p in paths:
        try:
            s = score_file(model, p, device)
            results.append({"path": str(p), "score": round(s, 2), "grade": grade(s)})
        except Exception as e:
            warnings.warn(f"[VocalVerse2] Failed on {p}: {e}")
            results.append({"path": str(p), "score": None, "grade": "error"})
    return results


def popbutfy_comparison(model, dataset_dir, singer, song, device="cuda"):
    """Score all clips for an amateur/professional pair and compare."""
    import re
    root = pathlib.Path(dataset_dir)

    def find_folder(level):
        pat = re.compile(
            rf"^{re.escape(singer)}#singing#{re.escape(song)}_{level}$",
            re.IGNORECASE,
        )
        for d in sorted(root.iterdir()):
            if d.is_dir() and pat.match(d.name):
                return d
        return None

    def sorted_clips(folder):
        return sorted(
            folder.glob("*.mp3"),
            key=lambda p: int(re.search(r"_(\d+)\.mp3$", p.name).group(1)),
        )

    am_dir  = find_folder("Amateur")
    pro_dir = find_folder("Professional")
    if am_dir is None or pro_dir is None:
        print(f"Could not find folders for {singer} / {song}")
        return

    am_clips  = sorted_clips(am_dir)
    pro_clips = sorted_clips(pro_dir)
    n = min(len(am_clips), len(pro_clips))
    print(f"\nSong: {song}  |  Singer: {singer}  |  Pairs: {n}")
    print(f"{'Clip':<6} {'Amateur':>10} {'Grade':<14} {'Pro':>10} {'Grade':<14} {'Diff':>8}")
    print("─" * 66)

    am_scores, pro_scores = [], []
    for i in range(n):
        as_ = score_file(model, am_clips[i], device)
        ps_ = score_file(model, pro_clips[i], device)
        am_scores.append(as_)
        pro_scores.append(ps_)
        diff = ps_ - as_
        print(f"  {i:<4} {round(as_,1):>10} {grade(as_):<14} {round(ps_,1):>10} {grade(ps_):<14} {round(diff,1):>+8}")

    print("─" * 66)
    print(f"  {'mean':<4} {round(np.mean(am_scores),1):>10} {'':14} {round(np.mean(pro_scores),1):>10} {'':14} {round(np.mean(pro_scores)-np.mean(am_scores),1):>+8}")
    print(f"\n  Amateur mean : {round(np.mean(am_scores),2)}")
    print(f"  Pro mean     : {round(np.mean(pro_scores),2)}")
    separation = np.mean(pro_scores) - np.mean(am_scores)
    print(f"  Separation   : {round(separation,2)} MOS pts  ({'pro scores higher — model differentiates' if separation > 0.5 else 'small gap — model may not differentiate well'})")


def _collect_split_files(dataset_dir):
    """Scan PopBuTFy root and split by folder-name suffix (_Amateur / _Professional)."""
    audio_exts = {".wav", ".flac", ".mp3"}
    splits = {"amateur": [], "professional": []}
    root = pathlib.Path(dataset_dir)
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        name_lower = folder.name.lower()
        if name_lower.endswith("_amateur"):
            label = "amateur"
        elif name_lower.endswith("_professional"):
            label = "professional"
        else:
            continue
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() in audio_exts:
                splits[label].append(f)
    return splits


def _percentile_stats(values):
    arr = np.array([v for v in values if np.isfinite(v)], dtype=np.float64)
    if len(arr) == 0:
        return {"mean": None, "median": None, "p25": None, "p75": None, "std": None, "n": 0}
    return {
        "mean":   round(float(np.mean(arr)), 4),
        "median": round(float(np.median(arr)), 4),
        "p25":    round(float(np.percentile(arr, 25)), 4),
        "p75":    round(float(np.percentile(arr, 75)), 4),
        "std":    round(float(np.std(arr)), 4),
        "n":      int(len(arr)),
    }


def build_baselines(model, dataset_dir, output_path, device="cuda"):
    """Score every clip in PopBuTFy and save population-level MOS statistics."""
    splits = _collect_split_files(dataset_dir)
    print(f"\nPopBuTFy discovered:")
    for label, files in splits.items():
        print(f"  {label}: {len(files)} clips")

    scores = {"amateur": [], "professional": []}
    errors = {"amateur": 0, "professional": 0}

    for label, files in splits.items():
        for f in tqdm(files, desc=label):
            try:
                s = score_file(model, f, device)
                scores[label].append(s)
            except Exception as e:
                warnings.warn(f"[skip] {f.name}: {e}")
                errors[label] += 1

    print(f"\n  amateur  : {len(scores['amateur'])} scored, {errors['amateur']} errors")
    print(f"  professional: {len(scores['professional'])} scored, {errors['professional']} errors")

    stats = {label: _percentile_stats(scores[label]) for label in scores}

    print(f"\n{'Level':<16} {'Mean':>6} {'Median':>8} {'P25':>6} {'P75':>6} {'Std':>6} {'N':>6}")
    print("─" * 56)
    for label, s in stats.items():
        if s["n"] == 0:
            continue
        print(f"  {label:<14} {s['mean']:>6.3f} {s['median']:>8.3f} "
              f"{s['p25']:>6.3f} {s['p75']:>6.3f} {s['std']:>6.3f} {s['n']:>6}")

    sep = (stats["professional"].get("mean") or 0) - (stats["amateur"].get("mean") or 0)
    print(f"\n  Pro − Amateur mean separation: {sep:+.3f} MOS pts "
          f"({'differentiates' if sep > 0.5 else 'small gap'})")

    output = {
        "amateur":      stats["amateur"],
        "professional": stats["professional"],
        "separation_mean": round(sep, 4),
        "_meta": {
            "model": "VocalVerse2 / SongEvalGenerator_audio_lora",
            "checkpoint": str(CKPT_DIR),
            "dataset_dir": str(dataset_dir),
            "n_amateur": len(scores["amateur"]),
            "n_professional": len(scores["professional"]),
        },
    }

    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(output, fh, indent=2)
    print(f"\nBaselines saved → {output_path}")


def parse_args():
    p = argparse.ArgumentParser(description="VocalVerse2 (MuQ) vocal scorer")
    p.add_argument("files", nargs="*", help="Audio files to score")
    p.add_argument("--popbutfy", metavar="DIR",
                   help="PopBuTFy root dir for amateur/pro song-pair comparison mode")
    p.add_argument("--singer", default="Female1")
    p.add_argument("--song",   default="my_heart_will_go_on")
    p.add_argument("--build-baselines", metavar="DIR",
                   help="Score ALL clips in PopBuTFy dir and save population stats")
    p.add_argument("--output", default="data/vocalverse2_baselines.json",
                   help="Output JSON path for --build-baselines (default: data/vocalverse2_baselines.json)")
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    return p.parse_args()


def main():
    args = parse_args()

    warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
    warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
    warnings.filterwarnings("ignore", category=UserWarning, module="pyworld")

    try:
        model = load_model(args.device)
    except FileNotFoundError as e:
        print(f"Setup error: {e}")
        sys.exit(1)

    if args.build_baselines:
        build_baselines(model, args.build_baselines, args.output, args.device)
        return

    if args.popbutfy:
        popbutfy_comparison(model, args.popbutfy, args.singer, args.song, args.device)
        return

    if not args.files:
        print("No files specified. Use --help for usage.")
        sys.exit(1)

    results = score_files(model, args.files, args.device)
    print(f"\n{'File':<50} {'MOS (1–5)':>10} {'Grade'}")
    print("─" * 75)
    for r in results:
        name = pathlib.Path(r["path"]).name
        score_str = f"{r['score']:.1f}" if r["score"] is not None else "error"
        print(f"  {name:<48} {score_str:>7}  {r['grade']}")


if __name__ == "__main__":
    main()
