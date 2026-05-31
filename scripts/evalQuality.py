"""
Quality Head Evaluation
========================
Evaluates a VocalCoach checkpoint's quality head against three reference datasets:

  MSE / V3  — quality_mse.npz   : SingMOS-Pro pseudo-labels (scalar score per clip)
               Reports: Spearman ρ, Pearson r, MAE, score distribution
  ccmusic / V2 — quality_ccmusic.npz : 9-dim expert labels
               Reports: per-dim Spearman ρ + MAE, mean across dims
  pairs / V1  — quality_pairs.npz  : PopBuTFy pro/amateur pairs
               Reports: ranking accuracy (% pairs where score(pro) > score(am))

Aggregates per-clip quality by mean-pooling the per-frame head output over
voiced frames only (VAD > 0.5). If no voiced frames exist the clip is skipped.

Usage
-----
  # Evaluate all three tasks (skip any whose --*-npz flag is omitted)
  python scripts/evalQuality.py \\
      --checkpoint vocalcoach/runs/stage2_quality_v3_from_gttech/checkpoints/best_metric.pth \\
      --mse-npz    data/quality/quality_mse.npz \\
      --ccmusic-npz data/quality/quality_ccmusic.npz \\
      --pairs-npz  data/quality_50k/quality_pairs.npz \\
      --log results/quality_log.json

  # Scalar-only (V3 checkpoint)
  python scripts/evalQuality.py \\
      --checkpoint vocalcoach/runs/stage2_quality_v3_from_gttech/checkpoints/best_metric.pth \\
      --mse-npz data/quality/quality_mse.npz

  # 9-dim only (V2 checkpoint)
  python scripts/evalQuality.py \\
      --checkpoint vocalcoach/runs/stage2_quality_v2_from_gttech/checkpoints/best_metric.pth \\
      --ccmusic-npz data/quality/quality_ccmusic.npz
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime, timezone

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vocalcoach.model import build_model

CCMUSIC_DIMS = [
    "Pitch", "Rhythm", "Timbre", "Breath",
    "Vibrato", "Dynamic", "Pronunciation", "Vocal Range", "Overall",
]

# ── Model inference ───────────────────────────────────────────────────────────

def _score_clips(model, mel_flat, lengths, device, batch_size=32):
    """
    Run the quality head over a flat packed mel array, return per-clip scores.

    mel_flat : (total_frames, 40) float16
    lengths  : (n_clips,) int32
    Returns  : list of np.ndarray, one per clip — shape (n_dims,) or scalar
               Clips with zero voiced frames are returned as None.
    """
    model.eval()
    scores = []
    offset = 0
    with torch.no_grad():
        for length in lengths:
            length = int(length)
            mel_clip = mel_flat[offset: offset + length].astype(np.float32)
            offset += length

            # Chunk long clips to avoid OOM (max 1200 frames ~12s)
            chunk_size = 1200
            frame_scores = []
            frame_vad    = []
            for start in range(0, len(mel_clip), chunk_size):
                chunk = mel_clip[start: start + chunk_size]
                t = torch.from_numpy(chunk).unsqueeze(0).to(device)  # (1, T, 40)
                out = model(t)
                # out: (vad, pitch, technique, quality, note_onset, note_offset[, emb])
                vad_logit = out[0].squeeze(0).squeeze(-1)   # (T,)
                q_out     = out[3]                          # (1, T, dims) or None
                if q_out is None:
                    raise RuntimeError(
                        "Checkpoint has no quality head — was it trained with --quality-variant?")
                q_out = q_out.squeeze(0)                    # (T, dims)
                frame_scores.append(q_out.cpu().numpy())
                frame_vad.append(torch.sigmoid(vad_logit).cpu().numpy())

            all_scores = np.concatenate(frame_scores, axis=0)   # (T, dims)
            all_vad    = np.concatenate(frame_vad,    axis=0)   # (T,)

            voiced_mask = all_vad > 0.5
            if voiced_mask.sum() == 0:
                scores.append(None)
                continue

            clip_score = all_scores[voiced_mask].mean(axis=0)   # (dims,) or (1,)
            if clip_score.shape == (1,):
                clip_score = float(clip_score[0])
            scores.append(clip_score)

    return scores


# ── Evaluation tasks ──────────────────────────────────────────────────────────

def eval_mse(model, npz_path, device):
    """Spearman ρ / Pearson r / MAE against SingMOS-Pro scalar scores."""
    from scipy.stats import spearmanr, pearsonr

    d = np.load(npz_path, allow_pickle=True)
    mel_flat = d["mel"]
    lengths  = d["lengths"]
    gt       = d["scores"].astype(np.float32)   # (n_clips,)

    print(f"\n  quality_mse: {len(lengths):,} clips, {mel_flat.shape[0]:,} frames")
    raw = _score_clips(model, mel_flat, lengths, device)

    pred, ref = [], []
    for i, s in enumerate(raw):
        if s is not None:
            pred.append(float(s) if isinstance(s, (float, np.floating)) else float(np.mean(s)))
            ref.append(float(gt[i]))

    pred = np.array(pred)
    ref  = np.array(ref)
    skipped = len(lengths) - len(pred)

    rho, _  = spearmanr(pred, ref)
    r, _    = pearsonr(pred, ref)
    mae     = float(np.mean(np.abs(pred - ref)))

    print(f"  Clips scored: {len(pred):,}  (skipped {skipped} no-voiced)")
    print(f"  Spearman ρ : {rho:+.4f}")
    print(f"  Pearson  r : {r:+.4f}")
    print(f"  MAE        : {mae:.4f}")
    print(f"  Pred range : [{pred.min():.3f}, {pred.max():.3f}]  "
          f"GT range: [{ref.min():.3f}, {ref.max():.3f}]")

    return {"spearman_rho": float(rho), "pearson_r": float(r),
            "mae": mae, "n_clips": len(pred), "n_skipped": skipped}


def eval_ccmusic(model, npz_path, device):
    """Per-dim Spearman ρ and MAE against 9-dim ccmusic expert labels."""
    from scipy.stats import spearmanr

    d = np.load(npz_path, allow_pickle=True)
    mel_flat = d["mel"]
    lengths  = d["lengths"]
    gt       = d["scores"].astype(np.float32)   # (n_clips, 9)

    print(f"\n  quality_ccmusic: {len(lengths):,} clips, {mel_flat.shape[0]:,} frames")
    raw = _score_clips(model, mel_flat, lengths, device)

    pred_list, ref_list = [], []
    for i, s in enumerate(raw):
        if s is not None:
            arr = np.atleast_1d(s)
            if arr.shape[0] != 9:
                raise RuntimeError(
                    f"Expected 9-dim quality head output, got {arr.shape[0]}. "
                    "Is this a V2 checkpoint (--quality-variant 2)?")
            pred_list.append(arr)
            ref_list.append(gt[i])

    pred = np.stack(pred_list)   # (n, 9)
    ref  = np.stack(ref_list)    # (n, 9)
    skipped = len(lengths) - len(pred_list)

    print(f"  Clips scored: {len(pred_list):,}  (skipped {skipped} no-voiced)")
    print(f"\n  {'Dimension':<18} {'Spearman ρ':>12} {'MAE':>8} {'Pred μ':>8} {'GT μ':>8}")
    print(f"  {'-'*56}")

    dim_results = {}
    rhos = []
    for j, dim_name in enumerate(CCMUSIC_DIMS):
        rho, _ = spearmanr(pred[:, j], ref[:, j])
        mae    = float(np.mean(np.abs(pred[:, j] - ref[:, j])))
        rhos.append(rho)
        dim_results[dim_name] = {"spearman_rho": float(rho), "mae": mae}
        print(f"  {dim_name:<18} {rho:>+12.4f} {mae:>8.4f} "
              f"{pred[:,j].mean():>8.3f} {ref[:,j].mean():>8.3f}")

    mean_rho = float(np.mean(rhos))
    print(f"  {'─'*56}")
    print(f"  {'Mean':<18} {mean_rho:>+12.4f}")

    return {"dims": dim_results, "mean_spearman_rho": mean_rho,
            "n_clips": len(pred_list), "n_skipped": skipped}


def eval_pairs(model, npz_path, device):
    """Ranking accuracy: fraction of pairs where score(pro) > score(am)."""
    d = np.load(npz_path, allow_pickle=True)
    mel_pro   = d["mel_pro"]
    mel_am    = d["mel_am"]
    len_pro   = d["lengths_pro"]
    len_am    = d["lengths_am"]

    # Cap at 2000 pairs for speed (random subset, seed=0)
    n_pairs = min(2000, len(len_pro))
    rng = np.random.default_rng(0)
    idx = rng.choice(len(len_pro), size=n_pairs, replace=False)
    idx.sort()

    # Build contiguous subsets by picking the selected pairs
    pro_mels, pro_lens, am_mels, am_lens = [], [], [], []
    pro_offset, am_offset = 0, 0
    for i in range(len(len_pro)):
        lp, la = int(len_pro[i]), int(len_am[i])
        if i in set(idx):
            pro_mels.append(mel_pro[pro_offset: pro_offset + lp])
            pro_lens.append(lp)
            am_mels.append(mel_am[am_offset: am_offset + la])
            am_lens.append(la)
        pro_offset += lp
        am_offset  += la

    pro_flat = np.concatenate(pro_mels, axis=0)
    am_flat  = np.concatenate(am_mels,  axis=0)

    print(f"\n  quality_pairs: evaluating {n_pairs:,} pairs "
          f"(of {len(len_pro):,} total, seed=0 subset)")

    raw_pro = _score_clips(model, pro_flat, np.array(pro_lens, dtype=np.int32), device)
    raw_am  = _score_clips(model, am_flat,  np.array(am_lens,  dtype=np.int32), device)

    correct = skipped = 0
    margins = []
    for sp, sa in zip(raw_pro, raw_am):
        if sp is None or sa is None:
            skipped += 1
            continue
        sp_s = float(sp) if isinstance(sp, (float, np.floating)) else float(np.mean(sp))
        sa_s = float(sa) if isinstance(sa, (float, np.floating)) else float(np.mean(sa))
        margins.append(sp_s - sa_s)
        if sp_s > sa_s:
            correct += 1

    total_ranked = len(margins)
    acc = correct / total_ranked if total_ranked > 0 else 0.0
    mean_margin = float(np.mean(margins)) if margins else 0.0

    print(f"  Pairs ranked: {total_ranked:,}  (skipped {skipped} no-voiced)")
    print(f"  Ranking accuracy: {acc*100:.1f}%  (random chance = 50.0%)")
    print(f"  Mean margin (pro−am): {mean_margin:+.4f}")

    return {"ranking_accuracy": acc, "mean_margin": mean_margin,
            "n_pairs": total_ranked, "n_skipped": skipped}


# ── Log helpers ───────────────────────────────────────────────────────────────

def _load_log(path):
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"runs": []}


def _save_log(log, path):
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\n  Results appended → {path}")


def _infer_run_name(ckpt_path):
    parts = os.path.normpath(ckpt_path).split(os.sep)
    # .../runs/<run_name>/checkpoints/<file>
    try:
        runs_idx = parts.index("runs")
        return f"{parts[runs_idx+1]}/{os.path.splitext(parts[-1])[0]}"
    except (ValueError, IndexError):
        return os.path.splitext(os.path.basename(ckpt_path))[0]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Evaluate VocalCoach quality head (V1/V2/V3)")
    p.add_argument("--checkpoint", required=True,
                   help="Path to checkpoint .pth")
    p.add_argument("--mse-npz",     default=None,
                   help="quality_mse.npz — scalar SingMOS-Pro scores (V3/V2)")
    p.add_argument("--ccmusic-npz", default=None,
                   help="quality_ccmusic.npz — 9-dim expert labels (V2)")
    p.add_argument("--pairs-npz",   default=None,
                   help="quality_pairs.npz — pro/amateur ranking pairs (V1)")
    p.add_argument("--log", default=None,
                   help="JSON log to append results to (default: no log)")
    p.add_argument("--run-name", default=None,
                   help="Override run name in log (default: inferred from path)")
    p.add_argument("--device", default=None,
                   help="cuda / cpu (default: auto)")
    args = p.parse_args()

    if not any([args.mse_npz, args.ccmusic_npz, args.pairs_npz]):
        p.error("Provide at least one of --mse-npz, --ccmusic-npz, --pairs-npz")

    # ── Device ───────────────────────────────────────────────────────
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # ── Load model ───────────────────────────────────────────────────
    warnings.warn("Loading checkpoint via torch.load — trusted source only.",
                  RuntimeWarning)
    ckpt   = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    arch   = ckpt.get("arch", "tcn")
    causal = ckpt.get("causal", False)
    kwargs = dict(ckpt.get("model_kwargs", {}))
    kwargs.pop("causal", None)
    model  = build_model(arch, causal=causal, **kwargs).to(device)
    model.load_state_dict(ckpt["state_dict"])
    epoch  = ckpt.get("epoch", "?")
    run_name = args.run_name or _infer_run_name(args.checkpoint)

    q_dims = getattr(model, "quality_dims", 0)
    print(f"Loaded: VocalCoach{arch.upper()} epoch={epoch}  quality_dims={q_dims}")
    print(f"Run: {run_name}")
    if q_dims == 0:
        print("ERROR: checkpoint has no quality head (quality_dims=0). "
              "Use a checkpoint trained with --quality-variant.")
        sys.exit(1)

    # ── Banner ───────────────────────────────────────────────────────
    w = 82
    print()
    print("═" * w)
    print(f"  Quality Head Evaluation  |  {run_name}")
    print("═" * w)

    # ── Run evaluations ──────────────────────────────────────────────
    record = {
        "run_name":  run_name,
        "checkpoint": args.checkpoint,
        "epoch":     epoch,
        "quality_dims": q_dims,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results":   {},
    }

    if args.mse_npz:
        print(f"\n── V3/MSE  (SingMOS-Pro distillation) {'─'*40}")
        record["results"]["mse"] = eval_mse(model, args.mse_npz, device)

    if args.ccmusic_npz:
        print(f"\n── V2/ccmusic  (9-dim expert labels) {'─'*41}")
        record["results"]["ccmusic"] = eval_ccmusic(model, args.ccmusic_npz, device)

    if args.pairs_npz:
        print(f"\n── V1/pairs  (ranking accuracy) {'─'*46}")
        record["results"]["pairs"] = eval_pairs(model, args.pairs_npz, device)

    print()
    print("═" * w)

    # ── Log ──────────────────────────────────────────────────────────
    if args.log:
        log = _load_log(args.log)
        log.setdefault("quality_runs", []).append(record)
        _save_log(log, args.log)


if __name__ == "__main__":
    main()
