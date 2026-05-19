"""
D5 — PopBuTFy Population Baselines
====================================

Runs VocalCoach model inference + feature extraction across all amateur and
professional recordings in PopBuTFy, then computes population-level statistics
for every coaching metric. The output JSON is used to anchor the per-clip
coaching report ("above/below amateur median", "approaching pro average", etc.).

PopBuTFy actual structure (folder-name encodes level):
    <dataset-dir>/
        Singer#singing#SongName_Amateur/   *.wav
        Singer#singing#SongName_Professional/  *.wav
        ...

Level is detected from the suffix of each folder name:
    ends with _Amateur       → amateur
    ends with _Professional  → professional

Output: data/popbutfy_baselines.json
    {
      "amateur":      { metric: {mean, median, p25, p75, std} },
      "professional": { metric: {mean, median, p25, p75, std} },
      "thresholds": {
          metric: {"amateur_median": X, "pro_median": Y,
                   "amateur_p75": Z,  "pro_p25": W}
      }
    }

Usage
-----
    python scripts/buildPopBuTFyBaselines.py \\
        --dataset-dir /data/popbutfy \\
        --checkpoint  vocalcoach/runs/conformer_probe_balanced/checkpoints/best_loss.pth \\
        --output      data/popbutfy_baselines.json \\
        --device      cuda
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import librosa
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vocalcoach.model import VocalCoachConformer, VocalCoachTCN, bin_to_f0
from vocalcoach.features import extract_all, summarise, phrase_aggregate, SR, HOP_LENGTH
from vocalcoach.singmos import score_mos

TECHNIQUE_NAMES = ["vibrato", "breathy", "falsetto", "belt"]


def parse_args():
    p = argparse.ArgumentParser(description="Build PopBuTFy population baselines.")
    p.add_argument("--dataset-dir", required=True,
                   help="root dir containing Singer#singing#Song_Amateur/ folders")
    p.add_argument("--checkpoint", required=True,
                   help="path to VocalCoach best_loss.pth")
    p.add_argument("--output", default="data/popbutfy_baselines.json",
                   help="output JSON path (default: data/popbutfy_baselines.json)")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--min-duration", type=float, default=1.0,
                   help="skip clips shorter than this (seconds)")
    return p.parse_args()


def collect_split_files(dataset_dir):
    """Scan PopBuTFy root and split files by level detected from folder name suffix.

    Folder name pattern: Singer#singing#SongName_Amateur  or  ..._Professional
    Returns: {"amateur": [path, ...], "professional": [path, ...]}
    """
    audio_exts = {".wav", ".flac", ".mp3"}
    splits = {"amateur": [], "professional": []}

    for folder_name in sorted(os.listdir(dataset_dir)):
        folder_path = os.path.join(dataset_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        name_lower = folder_name.lower()
        if name_lower.endswith("_amateur"):
            label = "amateur"
        elif name_lower.endswith("_professional"):
            label = "professional"
        else:
            continue  # skip unrecognised folders

        for fname in sorted(os.listdir(folder_path)):
            if os.path.splitext(fname)[1].lower() in audio_exts:
                splits[label].append(os.path.join(folder_path, fname))

    return splits


def load_model(checkpoint_path, device):
    import torch
    ckpt = torch.load(checkpoint_path, map_location=device)
    sd = ckpt["state_dict"]
    saved_args = ckpt.get("args", {})
    if hasattr(saved_args, "__dict__"):
        saved_args = vars(saved_args)

    hidden = int(sd["input_proj.weight"].shape[0])
    n_layers = max(int(k.split(".")[1]) for k in sd if k.startswith("blocks.")) + 1
    arch = saved_args.get("arch", "conformer") if isinstance(saved_args, dict) else "conformer"
    deep = (saved_args.get("deep_technique_head", False)
            if isinstance(saved_args, dict) else False)

    if arch == "tcn":
        model = VocalCoachTCN(hidden=hidden, n_blocks=n_layers,
                              causal=False, deep_technique_head=deep)
    else:
        model = VocalCoachConformer(hidden=hidden, n_layers=n_layers,
                                    causal=False, deep_technique_head=deep)

    import torch as _torch
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model, _torch


def run_model(model, torch, y, device):
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=512, hop_length=HOP_LENGTH,
        n_mels=40, fmin=50, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_t = torch.tensor(mel_db.T[None], dtype=torch.float32).to(device)

    with torch.no_grad():
        out_vad, out_pitch, out_technique, _ = model(mel_t)

    vad = out_vad.squeeze(0).squeeze(-1).cpu().numpy()
    pitch_post = out_pitch.squeeze(0).cpu().numpy()

    pitch_confidence = pitch_post.max(axis=-1)
    voicing_thresh = max(0.05, pitch_confidence.max() * 0.30)
    voiced_mask = pitch_confidence > voicing_thresh
    pitch_bin = pitch_post.argmax(axis=-1).astype(float)
    f0_hz = np.where(voiced_mask, bin_to_f0(pitch_bin), 0.0).astype(np.float32)
    if vad.max() < 0.1:
        vad = voiced_mask.astype(np.float32)

    tech_probs = None
    if out_technique is not None:
        tech_probs = out_technique.squeeze(0).cpu().numpy()

    return f0_hz, vad, tech_probs


def collect_metrics(y, f0_hz, vad, tech_probs):
    """Run feature extraction and return a flat dict of per-clip scalars."""
    feats = extract_all(y, sr=SR, f0_hz=f0_hz, vad=vad)
    summary = summarise(feats, f0_hz=f0_hz)
    phrases = phrase_aggregate(f0_hz, vad,
                               technique_probs=tech_probs,
                               rms_db=feats["rms_db"])

    # Clip-level technique means over voiced frames
    voiced_mask = vad > 0.5
    tech_clip = {}
    if tech_probs is not None and voiced_mask.any():
        vt = tech_probs[voiced_mask[:len(tech_probs)]]
        for k, name in enumerate(TECHNIQUE_NAMES):
            if k < vt.shape[1]:
                tech_clip[f"technique_{name}"] = float(np.mean(vt[:, k]))

    # Phrase-level aggregates
    arcs = [p["rms_arc_db"] for p in phrases if p.get("rms_arc_db") is not None]
    vib_phrases = [p for p in phrases if p.get("vibrato", {}).get("has_vibrato")]

    voiced_phrases = [p for p in phrases
                      if p["f0_mean_hz"] == p["f0_mean_hz"] and p["f0_mean_hz"]]
    if voiced_phrases:
        total_dur = sum(p["duration_s"] for p in voiced_phrases)
        f0_mean = sum(p["f0_mean_hz"] * p["duration_s"] for p in voiced_phrases) / total_dur
        # Convert to cents stability
        f0_std_vals = [p["f0_std_hz"] for p in voiced_phrases
                       if p["f0_std_hz"] == p["f0_std_hz"]]
        f0_std_hz = float(np.mean(f0_std_vals)) if f0_std_vals else float("nan")
        cents_std = (1200 * abs(__import__("math").log2((f0_mean + f0_std_hz) / f0_mean))
                     if f0_mean > 0 and f0_std_hz == f0_std_hz else float("nan"))
    else:
        f0_mean = float("nan")
        cents_std = float("nan")

    row = {
        "f0_mean_hz":            f0_mean,
        "pitch_stability_cents": cents_std,
        "hnr_mean_db":           summary.get("hnr_mean_db", float("nan")),
        "jitter_mean_pct":       summary.get("jitter_mean_pct", float("nan")),
        "shimmer_mean_pct":      summary.get("shimmer_mean_pct", float("nan")),
        "h1_h2_mean_db":         summary.get("h1_h2_mean_db", float("nan")),
        "rms_mean_db":           summary.get("rms_mean_db", float("nan")),
        "vibrato_phrase_frac":   summary.get("vibrato_phrase_frac", 0.0),
        "vibrato_rate_hz":       summary.get("vibrato_rate_hz_mean", float("nan")),
        "vibrato_depth_cents":   summary.get("vibrato_depth_cents_mean", float("nan")),
        "rms_arc_mean_db":       float(np.mean(arcs)) if arcs else float("nan"),
        "n_vibrato_phrases":     len(vib_phrases),
        "n_phrases":             len(phrases),
        "onsets_per_phrase":     (summary.get("n_onsets", 0) / max(1, len(phrases))),
    }
    row.update(tech_clip)

    # MOS (may be None if SingMOS not installed)
    mos = score_mos(y, sr=SR)
    if mos is not None:
        row["mos"] = mos

    return row


def percentile_stats(values):
    """Return summary stats for a list of floats, ignoring nan."""
    arr = np.array([v for v in values if v == v and not np.isinf(v)], dtype=np.float64)
    if len(arr) == 0:
        return {"mean": None, "median": None, "p25": None, "p75": None,
                "std": None, "n": 0}
    return {
        "mean":   round(float(np.mean(arr)), 4),
        "median": round(float(np.median(arr)), 4),
        "p25":    round(float(np.percentile(arr, 25)), 4),
        "p75":    round(float(np.percentile(arr, 75)), 4),
        "std":    round(float(np.std(arr)), 4),
        "n":      int(len(arr)),
    }


def process_files(files, label, model, torch, device, min_frames, dataset_dir):
    """Run inference + feature extraction on a list of audio file paths."""
    if not files:
        warnings.warn(f"No {label} files found.")
        return []

    rows = []
    for path in tqdm(files, desc=label):
        try:
            y, _ = librosa.load(path, sr=SR, mono=True)
        except Exception as e:
            print(f"  [skip] {path}: {e}")
            continue
        if len(y) < min_frames * HOP_LENGTH:
            continue
        try:
            f0_hz, vad, tech_probs = run_model(model, torch, y, device)
            row = collect_metrics(y, f0_hz, vad, tech_probs)
            row["file"] = os.path.relpath(path, dataset_dir)
            rows.append(row)
        except Exception as e:
            print(f"  [error] {path}: {e}")
    return rows


def main():
    args = parse_args()

    import torch
    device = torch.device(args.device)
    model, torch_mod = load_model(args.checkpoint, device)
    min_frames = int(args.min_duration * SR / HOP_LENGTH)

    # Discover files by folder-name suffix (_Amateur / _Professional)
    file_splits = collect_split_files(args.dataset_dir)
    print(f"\nPopBuTFy discovered:")
    for label, files in file_splits.items():
        print(f"  {label}: {len(files)} clips across "
              f"{len(set(os.path.dirname(f) for f in files))} song folders")

    splits = {}
    for label, files in file_splits.items():
        print(f"\nProcessing {label} ...")
        splits[label] = process_files(
            files, label, model, torch_mod, device, min_frames, args.dataset_dir)
        print(f"  {len(splits[label])} clips processed")

    # Aggregate all metric keys
    all_keys = sorted({k for rows in splits.values() for row in rows for k in row
                       if k != "file" and isinstance(row[k], (int, float))})

    baselines = {}
    for label, rows in splits.items():
        baselines[label] = {}
        for key in all_keys:
            vals = [row[key] for row in rows if key in row]
            baselines[label][key] = percentile_stats(vals)

    # Thresholds: amateur_median, pro_median, amateur_p75, pro_p25
    # "approaching pro" = value between pro_p25 and pro_median
    thresholds = {}
    for key in all_keys:
        t = {}
        for label in ("amateur", "professional"):
            b = baselines.get(label, {}).get(key, {})
            if b.get("n", 0) > 0:
                t[f"{label[:3]}_median"] = b["median"]
                t[f"{label[:3]}_p25"]    = b["p25"]
                t[f"{label[:3]}_p75"]    = b["p75"]
        thresholds[key] = t

    output = {
        "amateur":      baselines.get("amateur", {}),
        "professional": baselines.get("professional", {}),
        "thresholds":   thresholds,
        "_meta": {
            "n_amateur":      len(splits.get("amateur", [])),
            "n_professional": len(splits.get("professional", [])),
            "checkpoint":     args.checkpoint,
            "dataset_dir":    args.dataset_dir,
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nBaselines saved → {args.output}")
    print(f"  Amateur clips  : {output['_meta']['n_amateur']}")
    print(f"  Pro clips      : {output['_meta']['n_professional']}")
    print(f"  Metrics tracked: {len(all_keys)}")

    # Print a quick comparison table
    print(f"\n{'Metric':<28} {'Amateur median':>15} {'Pro median':>12}")
    print("─" * 58)
    for key in all_keys:
        am = baselines.get("amateur", {}).get(key, {}).get("median")
        pr = baselines.get("professional", {}).get(key, {}).get("median")
        if am is not None or pr is not None:
            print(f"  {key:<26} {str(round(am,3)) if am is not None else 'n/a':>15} "
                  f"{str(round(pr,3)) if pr is not None else 'n/a':>12}")


if __name__ == "__main__":
    main()
