"""
OOD Pitch + VAD Evaluation
===========================
Evaluates a VocalCoach checkpoint on out-of-distribution datasets that provide
frame-level F0 annotations (Vocadito, and later iKala).

The script uses the same mel feature extraction as api.py / train.py:
  SR=16000, hop=160 (10 ms), n_fft=512, n_mels=40, fmin=50, fmax=8000,
  power_to_db(ref=np.max)

Reports: VAD Acc | VDR | VFA↓ | vF1 | RPA | RCA | Gross | Med.c
Same format as evaluate.py so results can be compared directly.

Usage
-----
  # Vocadito (all 40 clips)
  python scripts/evalOOD.py \\
      --checkpoint vocalcoach/runs/stage1_conformer_128_vadfix/checkpoints/best_metric.pth \\
      --dataset vocadito \\
      --data-dir data/vocadito

  # Save JSON for update_results.py
  python scripts/evalOOD.py \\
      --checkpoint ... --dataset vocadito --data-dir data/vocadito \\
      --json results/ood_vadfix.json
"""

import argparse
import csv
import json
import os
import sys
import warnings

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vocalcoach.model import build_model, viterbi_decode
from vocalcoach.evaluate import pitch_metrics, print_pitch_table

# ── Audio / feature constants — must match api.py / train.py ────────
SR         = 16000
HOP_LENGTH = 160    # 10 ms
N_FFT      = 512
N_MELS     = 40
FMIN       = 50
FMAX       = 8000


def extract_mel(y: np.ndarray) -> np.ndarray:
    """Return (T, 40) log-mel matching api.py pipeline."""
    import librosa
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)   # (40, T)
    return mel_db.T.astype(np.float32)               # (T, 40)


def load_audio(path: str) -> np.ndarray:
    """Load audio, resample to SR, mix to mono."""
    import soundfile as sf
    import librosa
    y, sr = sf.read(path, always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return y


def regrid_f0(times: np.ndarray, f0_hz: np.ndarray, n_frames: int) -> np.ndarray:
    """
    Resample an annotation (times, f0) to the model's 10 ms frame grid.
    For each output frame t, picks the nearest annotation sample.
    Frames with no voiced annotation remain 0 (unvoiced).

    Args:
        times:    (M,) annotation timestamps in seconds
        f0_hz:    (M,) F0 in Hz (0 = unvoiced)
        n_frames: target number of 10 ms frames

    Returns:
        (n_frames,) float32 — F0 in Hz on the model grid
    """
    frame_times = np.arange(n_frames) * (HOP_LENGTH / SR)   # (T,)
    # Nearest-neighbour interpolation
    idx = np.searchsorted(times, frame_times)
    idx = np.clip(idx, 0, len(times) - 1)
    # Also check idx-1 for closer neighbour
    idx_prev = np.maximum(idx - 1, 0)
    dist_next = np.abs(frame_times - times[idx])
    dist_prev = np.abs(frame_times - times[idx_prev])
    nearest = np.where(dist_prev < dist_next, idx_prev, idx)
    return f0_hz[nearest].astype(np.float32)


# ══════════════════════════════════════════════════════════════════════
# Dataset loaders — each returns a list of (name, wav_path, f0_path)
# ══════════════════════════════════════════════════════════════════════

def _vocadito_clips(data_dir: str):
    audio_dir = os.path.join(data_dir, "Audio")
    f0_dir    = os.path.join(data_dir, "Annotations", "F0")
    clips = []
    for fname in sorted(os.listdir(audio_dir)):
        if not fname.endswith(".wav") or "Zone" in fname:
            continue
        stem = fname.replace(".wav", "")            # e.g. vocadito_1
        f0_name = f"{stem}_f0.csv"
        f0_path = os.path.join(f0_dir, f0_name)
        if not os.path.exists(f0_path):
            print(f"  [warn] no F0 annotation for {fname}, skipping")
            continue
        clips.append((stem, os.path.join(audio_dir, fname), f0_path))
    return clips


def _load_f0_csv(path: str):
    """Read a CSV with (time, f0_hz) rows. Returns (times, f0) arrays."""
    times, f0s = [], []
    with open(path, newline='') as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            try:
                times.append(float(row[0]))
                f0s.append(float(row[1]))
            except ValueError:
                continue
    return np.array(times, dtype=np.float64), np.array(f0s, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════
# Core OOD eval loop
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_ood(model, clips, device, label="VocalCoach",
             voicing_threshold=0.3, onset_penalty=1.0):
    """
    Run model on each (name, wav_path, f0_csv_path) triple and collect
    per-clip pitch_metrics dicts.

    Returns dict matching print_pitch_table format: one key per clip name,
    plus '_macro_rpa' and '_macro_vf1'.
    """
    from tqdm import tqdm

    model.eval()
    clip_results = []

    for name, wav_path, f0_path in tqdm(clips, desc=f"  OOD eval {label}", leave=False):
        # Load and featurise audio
        y   = load_audio(wav_path)
        mel = extract_mel(y)          # (T, 40)
        T   = mel.shape[0]

        # Load and grid-align F0 annotation
        times, f0_ann = _load_f0_csv(f0_path)
        f0_ref = regrid_f0(times, f0_ann, T)    # (T,) Hz, 0=unvoiced

        # Model inference
        mel_t = torch.from_numpy(mel).unsqueeze(0).to(device)  # (1, T, 40)
        v, p, _, _, _, _ = model(mel_t)
        pv = v.squeeze().cpu().numpy()    # (T,)
        pp = p.squeeze(0).cpu().numpy()   # (T, 360)

        # Decode pitch
        f0_dec = viterbi_decode(pp, voicing_threshold=voicing_threshold,
                                onset_penalty=onset_penalty)   # (T,) Hz

        m = pitch_metrics(f0_dec, f0_ref, vad_pred=pv)
        m['name'] = name
        clip_results.append(m)

    if not clip_results:
        return {}

    def smean(lst, key):
        vals = [x[key] for x in lst
                if not np.isnan(x.get(key, float('nan')))]
        return float(np.mean(vals)) if vals else float('nan')

    # Aggregate: one "overall" bucket (Vocadito has no SNR dimension)
    keys = ['vad_acc', 'vdr', 'vf1', 'vfa', 'rpa', 'rca', 'gross', 'median_cents']
    results = {
        'overall': {k: smean(clip_results, k) for k in keys},
    }
    results['_macro_rpa'] = results['overall']['rpa']
    results['_macro_vf1'] = results['overall']['vf1']
    results['_clip_results'] = clip_results   # full per-clip data for --csv / --json
    return results


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="VocalCoach OOD pitch+VAD evaluation")
    p.add_argument("--checkpoint", required=True,
                   help="VocalCoach checkpoint (.pth)")
    p.add_argument("--dataset", required=True, choices=["vocadito"],
                   help="Which OOD dataset to evaluate (vocadito)")
    p.add_argument("--data-dir", required=True,
                   help="Root directory of the dataset")
    p.add_argument("--device", default="auto",
                   help="cpu / cuda / mps / auto")
    p.add_argument("--voicing-threshold", type=float, default=0.3)
    p.add_argument("--onset-penalty", type=float, default=1.0)
    p.add_argument("--json", default=None,
                   help="Save results JSON to this path")
    p.add_argument("--csv", default=None,
                   help="Save per-clip CSV to this path")
    return p.parse_args()


def main():
    args = parse_args()

    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Load checkpoint
    warnings.warn("Loading checkpoint via torch.load — trusted source only.",
                  RuntimeWarning)
    ckpt   = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    arch   = ckpt.get("arch", "tcn")
    causal = ckpt.get("causal", False)
    kwargs = dict(ckpt.get("model_kwargs", {}))
    kwargs.pop("causal", None)
    model  = build_model(arch, causal=causal, **kwargs).to(device)
    model.load_state_dict(ckpt["state_dict"])
    run_label = (f"VocalCoach{arch.upper()} "
                 f"(causal={causal}, epoch={ckpt.get('epoch', '?')})")
    print(f"Loaded: {run_label}")

    # Collect clips
    data_dir = os.path.abspath(args.data_dir)
    if args.dataset == "vocadito":
        clips = _vocadito_clips(data_dir)
        dataset_label = f"Vocadito ({len(clips)} clips)"
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print(f"Dataset: {dataset_label}")

    # Evaluate
    results = eval_ood(
        model, clips, device,
        label=run_label,
        voicing_threshold=args.voicing_threshold,
        onset_penalty=args.onset_penalty,
    )

    if not results:
        print("No results — check dataset path.")
        return

    # Display
    display = {k: v for k, v in results.items()
               if not k.startswith('_') and k != '_clip_results'}
    display['_macro_rpa'] = results['_macro_rpa']
    display['_macro_vf1'] = results['_macro_vf1']
    print_pitch_table(display, label=f"{run_label} | OOD: {dataset_label}")

    # Per-clip summary (top worst/best by VDR)
    clip_r = results.get('_clip_results', [])
    if clip_r:
        clip_r_sorted = sorted(
            [c for c in clip_r if not np.isnan(c.get('vdr', float('nan')))],
            key=lambda x: x['vdr'],
        )
        print(f"\n  Worst 5 clips by VDR:")
        for c in clip_r_sorted[:5]:
            print(f"    {c['name']:<20}  VDR={c['vdr']:.1%}  VFA={c.get('vfa', float('nan')):.1%}"
                  f"  RPA={c['rpa']:.1%}  vF1={c['vf1']:.1%}")
        print(f"\n  Best 5 clips by VDR:")
        for c in clip_r_sorted[-5:]:
            print(f"    {c['name']:<20}  VDR={c['vdr']:.1%}  VFA={c.get('vfa', float('nan')):.1%}"
                  f"  RPA={c['rpa']:.1%}  vF1={c['vf1']:.1%}")

    # Save JSON
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        out = {
            'dataset': args.dataset,
            'checkpoint': args.checkpoint,
            'onset_penalty': args.onset_penalty,
            'n_clips': len(clip_r),
            'overall': results.get('overall', {}),
            'per_clip': clip_r,
        }
        with open(args.json, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to {args.json}")

    # Save per-clip CSV
    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        import csv as csv_mod
        fields = ['name', 'vad_acc', 'vdr', 'vf1', 'vfa', 'rpa', 'rca', 'gross', 'median_cents']
        with open(args.csv, 'w', newline='') as f:
            writer = csv_mod.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            for c in clip_r:
                writer.writerow({k: (f"{c[k]:.4f}" if not np.isnan(c.get(k, float('nan'))) else '')
                                 for k in fields if k != 'name'} | {'name': c['name']})
        print(f"Per-clip CSV saved to {args.csv}")


if __name__ == "__main__":
    main()
