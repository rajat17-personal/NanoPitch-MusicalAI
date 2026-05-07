"""
VocalCoach Training Script
==========================

Trains VocalCoachTCN or VocalCoachConformer for multi-task singing analysis:
  1. VAD — is the singer vocalising at each frame?
  2. Pitch posteriorgram — what pitch is being sung? (decoded via Viterbi)
  3. Technique classification — vibrato / breathy / falsetto / belt / straight
     (multi-label sigmoid, not mutually exclusive)

Phase 1 focus: non-causal (causal=False) models for offline post-session analysis.
Phase 4 stretch: retrain with causal=True for ONNX Runtime Web streaming.

Data layout (NPZ files)
-----------------------
The trainer expects pre-extracted features in NPZ format.  Three sources are
supported and can be mixed via --data-dirs:

  clean.npz  (required for pitch/VAD supervision)
    mel:      (total_frames, 40)   — log-mel spectrogram
    f0:       (total_frames,)      — RMVPE f0 in Hz  (0 = unvoiced)
    vad:      (total_frames,)      — voice activity label  (0 or 1)
    lengths:  (n_clips,)           — frame count per clip

  technique.npz  (required for technique head, can be None)
    mel:       (total_clips, T, 40)  — padded or fixed-length mel
    technique: (total_clips, N_TECHNIQUES)  — binary clip-level labels
    f0:        (total_clips, T)     — per-frame f0 (0=unvoiced; optional)
    vad:       (total_clips, T)     — per-frame VAD (optional)

  test.npz  (optional — for evaluation)
    clips:   (N, T, 40)
    f0:      (N, T)
    vad:     (N, T)
    snr:     (N,)

Usage
-----
# Arch A — TCN, non-causal (Phase 1 Experiment A)
python vocalcoach/train.py \\
    --arch tcn --causal false \\
    --data-dir data \\
    --technique-dir data/vocalset \\
    --output-dir vocalcoach/runs/tcn_noncausal

# Arch B — Conformer, non-causal (Phase 1 Experiment B)
python vocalcoach/train.py \\
    --arch conformer --causal false \\
    --data-dir data \\
    --technique-dir data/vocalset \\
    --output-dir vocalcoach/runs/conformer_noncausal

# Resume from checkpoint
python vocalcoach/train.py \\
    --arch tcn --causal false \\
    --data-dir data \\
    --technique-dir data/vocalset \\
    --output-dir vocalcoach/runs/tcn_noncausal \\
    --resume vocalcoach/runs/tcn_noncausal/checkpoints/epoch_010.pth

Monitor with TensorBoard:
    tensorboard --logdir vocalcoach/runs
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, average_precision_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from vocalcoach.model import (
    build_model,
    f0_to_posteriorgram, viterbi_decode,
    PITCH_BINS, PITCH_FMIN, PITCH_CENTS_PER_BIN, N_MELS,
    N_TECHNIQUES, TECHNIQUE_NAMES,
)


# ═══════════════════════════════════════════════════════════════════════
# Arguments
# ═══════════════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser(description="VocalCoach multi-task trainer")

# Paths
parser.add_argument("--data-dir", type=str, default="../data",
                    help="directory with clean.npz (mel/f0/vad) and optional test.npz")
parser.add_argument("--technique-dir", type=str, default=None,
                    help="directory with technique.npz (mel/technique/f0/vad); "
                         "if None, technique head is trained on pseudo-labels (zeros)")
parser.add_argument("--output-dir", type=str, default="./runs/default",
                    help="where to save checkpoints and TensorBoard logs")
parser.add_argument("--resume", type=str, default=None,
                    help="path to checkpoint to resume training from")

# Architecture
parser.add_argument("--arch", type=str, default="tcn",
                    choices=["tcn", "conformer"],
                    help="model architecture. 'tcn' = VocalCoachTCN, "
                         "'conformer' = VocalCoachConformer")
parser.add_argument("--causal", type=str, default="false",
                    choices=["true", "false"],
                    help="causal=true: streaming-capable (left-pad only); "
                         "causal=false: non-causal (symmetric, higher accuracy)")
parser.add_argument("--hidden", type=int, default=None,
                    help="hidden dimension (default: 128 for TCN, 64 for Conformer)")
parser.add_argument("--n-blocks", type=int, default=None,
                    help="number of TCN blocks or Conformer layers "
                         "(default: 8 for TCN, 4 for Conformer)")

# Device
parser.add_argument("--device", type=str, default="auto",
                    help="cpu / cuda / mps / auto")

# Hyperparameters
parser.add_argument("--epochs", type=int, default=80)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--seq-len", type=int, default=300,
                    help="training clip length in frames (300 = 3 seconds)")
parser.add_argument("--num-workers", type=int, default=4)
parser.add_argument("--scheduler", type=str, default="cosine_warmup",
                    choices=["constant", "cosine_warmup"],
                    help="LR schedule")

# Loss weights
parser.add_argument("--w-vad", type=float, default=0.5,
                    help="weight for VAD loss (BCE)")
parser.add_argument("--w-pitch", type=float, default=1.0,
                    help="weight for pitch posteriorgram loss (VAD-masked BCE)")
parser.add_argument("--w-technique", type=float, default=2.0,
                    help="weight for technique classification loss (multi-label BCE). "
                         "Higher than pitch/VAD because technique labels are sparse.")

# VAD loss
parser.add_argument("--vad-pos-weight", type=float, default=2.3,
                    help="upweight voiced frames (≈ (1-p_voiced)/p_voiced)")

# Technique loss
parser.add_argument("--technique-clip-weight", type=float, default=1.0,
                    help="relative weight of clip-level technique loss vs frame-level; "
                         "used when --technique-dir provides clip-level labels only")

# Pitch supervision
parser.add_argument("--pitch-sigma", type=float, default=1.2,
                    help="Gaussian sigma (bins) for pitch posteriorgram target")

# Evaluation frequency
parser.add_argument("--eval-every", type=int, default=5,
                    help="run evaluation every N epochs")

# Early stopping
parser.add_argument("--patience", type=int, default=20,
                    help="stop if macro technique F1 does not improve for N epochs "
                         "(0 = disabled)")

# Gradient clipping
parser.add_argument("--grad-clip", type=float, default=5.0)


# ═══════════════════════════════════════════════════════════════════════
# Datasets
# ═══════════════════════════════════════════════════════════════════════

class PitchVADDataset(Dataset):
    """Loads clean.npz for pitch and VAD supervision.

    Samples random windows of length seq_len from each clip, returning
    (mel, f0, vad) tuples. Technique labels are returned as all-zeros
    here — technique supervision comes from TechniqueDataset.
    """

    def __init__(self, data_dir, seq_len=300):
        self.seq_len = seq_len
        clean = np.load(os.path.join(data_dir, "clean.npz"))
        self.mel = clean["mel"].astype(np.float32)      # (total_frames, 40)
        self.f0  = clean["f0"].astype(np.float32)       # (total_frames,)
        self.vad = clean["vad"].astype(np.float32)      # (total_frames,)
        lengths  = clean["lengths"]

        self.segments = []
        offset = 0
        for length in lengths:
            if length >= seq_len:
                self.segments.append((offset, offset + length))
            offset += length

        self.rng = np.random.default_rng()
        print(f"PitchVADDataset: {len(self.mel):,} frames, "
              f"{len(self.segments)} usable segments")

    def __len__(self):
        return min(len(self.segments) * 5, 20000)

    def __getitem__(self, _):
        seg_idx = self.rng.integers(len(self.segments))
        s, e = self.segments[seg_idx]
        t = self.rng.integers(0, e - s - self.seq_len + 1)
        t += s

        mel = self.mel[t:t + self.seq_len]              # (T, 40)
        f0  = self.f0 [t:t + self.seq_len]              # (T,)
        vad = self.vad[t:t + self.seq_len]              # (T,)
        technique = np.zeros((self.seq_len, N_TECHNIQUES), dtype=np.float32)
        has_technique = np.float32(0.0)
        return mel, f0, vad, technique, has_technique


class TechniqueDataset(Dataset):
    """Loads technique.npz for technique classification supervision.

    Supports two label formats:
    - Clip-level: technique.npz with key 'technique' shape (N, N_TECHNIQUES)
      — labels broadcast over all voiced frames in the clip.
    - Frame-level: technique.npz with key 'technique' shape (N, T, N_TECHNIQUES)
      — direct per-frame supervision.

    The returned has_technique flag (1.0) tells the collator to include these
    samples in the technique loss computation.
    """

    def __init__(self, technique_dir, seq_len=300):
        self.seq_len = seq_len
        data = np.load(os.path.join(technique_dir, "technique.npz"))

        self.mel = data["mel"].astype(np.float32)             # (N, T, 40)
        technique_raw = data["technique"].astype(np.float32)  # (N, N_TECH) or (N, T, N_TECH)

        self.f0  = data["f0"].astype(np.float32)  if "f0"  in data else None
        self.vad = data["vad"].astype(np.float32) if "vad" in data else None

        # Normalise to (N, T, N_TECHNIQUES)
        N, T_mel = self.mel.shape[0], self.mel.shape[1]
        if technique_raw.ndim == 2:
            # Clip-level: broadcast to (N, T, N_TECH)
            self.technique = np.broadcast_to(
                technique_raw[:, np.newaxis, :], (N, T_mel, N_TECHNIQUES)
            ).copy()
            self.clip_level = True
        else:
            self.technique = technique_raw   # (N, T, N_TECH)
            self.clip_level = False

        self.rng = np.random.default_rng()
        print(f"TechniqueDataset: {N} clips, seq_len={T_mel}, "
              f"clip_level={self.clip_level}")

    def __len__(self):
        return len(self.mel) * 3

    def __getitem__(self, _):
        idx = self.rng.integers(len(self.mel))
        mel = self.mel[idx]                    # (T, 40)
        T = mel.shape[0]

        # Crop or pad to seq_len
        if T >= self.seq_len:
            t0 = self.rng.integers(0, T - self.seq_len + 1)
            mel = mel[t0:t0 + self.seq_len]
            technique = self.technique[idx, t0:t0 + self.seq_len]
            f0  = self.f0 [idx, t0:t0 + self.seq_len] if self.f0  is not None \
                  else np.zeros(self.seq_len, dtype=np.float32)
            vad = self.vad[idx, t0:t0 + self.seq_len] if self.vad is not None \
                  else np.ones(self.seq_len, dtype=np.float32)
        else:
            pad = self.seq_len - T
            mel = np.pad(mel, ((0, pad), (0, 0)))
            technique = np.pad(self.technique[idx], ((0, pad), (0, 0)))
            f0  = np.pad(self.f0 [idx], (0, pad)) if self.f0  is not None \
                  else np.zeros(self.seq_len, dtype=np.float32)
            vad = np.pad(self.vad[idx], (0, pad)) if self.vad is not None \
                  else np.concatenate([np.ones(T, dtype=np.float32),
                                       np.zeros(pad, dtype=np.float32)])

        has_technique = np.float32(1.0)
        return (mel.astype(np.float32),
                f0.astype(np.float32),
                vad.astype(np.float32),
                technique.astype(np.float32),
                has_technique)


def make_joint_loader(pitch_vad_dir, technique_dir, seq_len, batch_size,
                      num_workers):
    """Interleave pitch/VAD and technique batches using ConcatDataset."""
    from torch.utils.data import ConcatDataset

    pv = PitchVADDataset(pitch_vad_dir, seq_len)
    if technique_dir is not None and os.path.exists(
            os.path.join(technique_dir, "technique.npz")):
        tech = TechniqueDataset(technique_dir, seq_len)
        dataset = ConcatDataset([pv, tech])
        print(f"Joint dataset: {len(pv)} pitch/VAD + {len(tech)} technique samples")
    else:
        dataset = pv
        print("No technique.npz found — technique head will train on pseudo-labels.")

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    return loader


# ═══════════════════════════════════════════════════════════════════════
# Loss helpers
# ═══════════════════════════════════════════════════════════════════════

def _build_pitch_target_gpu(f0_dev, sigma_bins, device):
    """Vectorised on-device pitch posteriorgram target. (B, T) f0 → (B, T, 360)."""
    voiced = (f0_dev > 0).float().unsqueeze(-1)                     # (B, T, 1)
    safe_f0 = torch.where(f0_dev > 0, f0_dev, torch.ones_like(f0_dev))
    bins = 1200.0 * torch.log2(safe_f0 / PITCH_FMIN) / PITCH_CENTS_PER_BIN  # (B, T)
    bin_grid = torch.arange(PITCH_BINS, device=device,
                            dtype=torch.float32).view(1, 1, -1)     # (1, 1, 360)
    dist = bin_grid - bins.unsqueeze(-1)                            # (B, T, 360)
    return torch.exp(-0.5 * (dist / sigma_bins) ** 2) * voiced      # (B, T, 360)


def compute_loss(pred_vad, pred_pitch, pred_technique,
                 vad_target, f0_target, technique_target, has_technique,
                 args, device):
    """Compute multi-task loss.

    Args:
        pred_vad:         (B, T, 1)
        pred_pitch:       (B, T, 360)
        pred_technique:   (B, T, N)
        vad_target:       (B, T)
        f0_target:        (B, T)
        technique_target: (B, T, N)
        has_technique:    (B,)  — 1.0 if sample has technique labels, else 0.0
    """
    # ── VAD loss (pos-weighted BCE) ──────────────────────────────────────
    bce_none = nn.BCELoss(reduction='none')
    vad_pred_sq = pred_vad.squeeze(-1)                             # (B, T)
    pw = torch.where(vad_target > 0.5,
                     torch.full_like(vad_target, args.vad_pos_weight),
                     torch.ones_like(vad_target))
    vad_loss = (pw * bce_none(vad_pred_sq, vad_target)).mean()

    # ── Pitch loss (VAD-masked BCE on posteriorgram) ─────────────────────
    pitch_target = _build_pitch_target_gpu(f0_target, args.pitch_sigma, device)
    voiced_mask = (f0_target > 0).float().unsqueeze(-1)            # (B, T, 1)
    pitch_loss = (voiced_mask * bce_none(pred_pitch, pitch_target)).mean()

    # ── Technique loss (multi-label BCE, only on samples with labels) ────
    mask = has_technique.view(-1, 1, 1)                            # (B, 1, 1)
    tech_per_elem = bce_none(pred_technique, technique_target)     # (B, T, N)
    if mask.sum() > 0:
        technique_loss = (mask * tech_per_elem).sum() / (mask.sum() * tech_per_elem.shape[1] * tech_per_elem.shape[2])
    else:
        technique_loss = torch.tensor(0.0, device=device)

    total = (args.w_vad * vad_loss
             + args.w_pitch * pitch_loss
             + args.w_technique * technique_loss)
    return total, vad_loss, pitch_loss, technique_loss


# ═══════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, scheduler, writer,
                    epoch, device, args, global_step_offset=0):
    model.train()
    running = {'total': 0.0, 'vad': 0.0, 'pitch': 0.0, 'technique': 0.0}
    n_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}", unit="batch")
    for batch_idx, (mel, f0, vad, technique, has_technique) in enumerate(pbar):
        mel           = mel.to(device)
        f0            = f0.to(device)
        vad           = vad.to(device)
        technique     = technique.to(device)
        has_technique = has_technique.to(device)

        pred_vad, pred_pitch, pred_technique = model(mel)

        total, vad_l, pitch_l, tech_l = compute_loss(
            pred_vad, pred_pitch, pred_technique,
            vad, f0, technique, has_technique,
            args, device,
        )

        optimizer.zero_grad()
        total.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if args._sched_step == "iter":
            scheduler.step()

        step = global_step_offset + batch_idx
        writer.add_scalar("train/grad_norm", grad_norm, step)

        running['total']     += total.item()
        running['vad']       += vad_l.item()
        running['pitch']     += pitch_l.item()
        running['technique'] += tech_l.item()
        n_batches += 1

        pbar.set_postfix(
            loss=f"{running['total']/n_batches:.4f}",
            vad=f"{running['vad']/n_batches:.4f}",
            pitch=f"{running['pitch']/n_batches:.4f}",
            tech=f"{running['technique']/n_batches:.4f}",
        )

    if n_batches == 0:
        warnings.warn("No batches processed this epoch.", RuntimeWarning)
        return float("nan")

    for k, v in running.items():
        writer.add_scalar(f"train/{k}", v / n_batches, epoch)
    writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], epoch)
    return running['total'] / n_batches


# ═══════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, data_dir, technique_dir, writer, epoch, device, args):
    """Evaluate pitch RPA (from test.npz) and technique F1 (from technique.npz)."""
    model.eval()
    results = {}

    # ── Pitch / VAD evaluation ───────────────────────────────────────────
    test_path = os.path.join(data_dir, "test.npz")
    if os.path.exists(test_path):
        test = np.load(test_path)
        clips   = test['clips']   # (N, T, 40)
        f0_all  = test['f0']      # (N, T)
        snrs    = test['snr']     # (N,)
        N = clips.shape[0]

        clip_results = []
        for i in range(N):
            mel = torch.from_numpy(clips[i].astype(np.float32)).unsqueeze(0).to(device)
            v, p, _ = model(mel)
            pv = v.squeeze().cpu().numpy()
            pp = p.squeeze(0).cpu().numpy()
            T  = pv.shape[0]

            f0r = f0_all[i, :T].astype(np.float32)
            f0d = viterbi_decode(pp)

            vg = f0r > 0
            vp = f0d > 0
            vdr = float(np.mean(vp[vg])) if vg.sum() > 0 else float('nan')
            both = vg & vp
            if both.sum() > 0:
                ce = np.abs(1200 * np.log2(
                    f0d[both] / (f0r[both] + 1e-10) + 1e-10))
                rpa = float(np.mean(ce < 50))
            else:
                rpa = float('nan')
            clip_results.append({'snr': float(snrs[i]), 'vdr': vdr, 'rpa': rpa})

        by_snr = {}
        for r in clip_results:
            by_snr.setdefault(r['snr'], []).append(r)

        def smean(vals):
            v = [x for x in vals if not np.isnan(x)]
            return float(np.mean(v)) if v else float('nan')

        print(f"\n  {'Condition':<10}  {'VDR':>8}  {'RPA':>8}")
        print(f"  {'─'*10}  {'─'*8}  {'─'*8}")
        rpa_values = []
        for snr in sorted(by_snr.keys(), key=lambda x: x if np.isfinite(x) else 1e6):
            c   = by_snr[snr]
            tag = "clean" if not np.isfinite(snr) else f"{snr:+.0f} dB"
            vd  = smean([x['vdr'] for x in c])
            rp  = smean([x['rpa'] for x in c])
            print(f"  {tag:<10}  {vd:8.1%}  {rp:8.1%}")
            if not np.isnan(rp):
                rpa_values.append(rp)
            stag = tag.replace(' ', '').replace('+', 'p').replace('-', 'n')
            writer.add_scalar(f"eval/vdr_{stag}", vd, epoch)
            writer.add_scalar(f"eval/rpa_{stag}", rp, epoch)

        macro_rpa = float(np.mean(rpa_values)) if rpa_values else float('nan')
        results['macro_rpa'] = macro_rpa
        writer.add_scalar("eval/macro_rpa", macro_rpa, epoch)
        print(f"  Macro RPA: {macro_rpa:.4f}")

    # ── Technique F1 evaluation ──────────────────────────────────────────
    tech_path = technique_dir and os.path.join(technique_dir, "technique.npz")
    if tech_path and os.path.exists(tech_path):
        data = np.load(tech_path)
        mel_all  = data["mel"].astype(np.float32)     # (N, T, 40)
        tech_all = data["technique"].astype(np.float32)  # (N, N_TECH) or (N, T, N_TECH)

        all_pred, all_true = [], []
        for i in range(len(mel_all)):
            mel_t = torch.from_numpy(mel_all[i]).unsqueeze(0).to(device)
            _, _, pred_tech = model(mel_t)
            # Average over time for clip-level prediction
            pred_clip = pred_tech.squeeze(0).mean(0).cpu().numpy()   # (N_TECH,)
            all_pred.append(pred_clip)

            if tech_all.ndim == 2:
                all_true.append(tech_all[i])                          # (N_TECH,)
            else:
                all_true.append(tech_all[i].max(0))                   # (T, N_TECH) → (N_TECH,)

        all_pred = np.stack(all_pred)   # (N, N_TECH)
        all_true = np.stack(all_true)   # (N, N_TECH)
        pred_bin = (all_pred > 0.5).astype(int)

        print(f"\n  {'Technique':<12}  {'F1':>6}  {'AP':>6}")
        print(f"  {'─'*12}  {'─'*6}  {'─'*6}")
        f1s, aps = [], []
        for k, name in enumerate(TECHNIQUE_NAMES):
            if all_true[:, k].sum() == 0:
                continue
            f1 = f1_score(all_true[:, k], pred_bin[:, k], zero_division=0)
            ap = average_precision_score(all_true[:, k], all_pred[:, k])
            print(f"  {name:<12}  {f1:6.3f}  {ap:6.3f}")
            f1s.append(f1)
            aps.append(ap)
            writer.add_scalar(f"eval/f1_{name}", f1, epoch)
            writer.add_scalar(f"eval/ap_{name}", ap, epoch)

        macro_f1 = float(np.mean(f1s)) if f1s else float('nan')
        macro_ap = float(np.mean(aps)) if aps else float('nan')
        results['macro_f1'] = macro_f1
        results['macro_ap'] = macro_ap
        writer.add_scalar("eval/macro_f1", macro_f1, epoch)
        writer.add_scalar("eval/macro_ap", macro_ap, epoch)
        print(f"  Macro F1: {macro_f1:.4f}  Macro AP: {macro_ap:.4f}")

    print()
    return results


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    args = parser.parse_args()
    args.causal = args.causal.lower() == "true"

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
    print(f"Device: {device}  |  Arch: {args.arch}  |  causal={args.causal}")

    data_dir     = os.path.abspath(args.data_dir)
    tech_dir     = os.path.abspath(args.technique_dir) if args.technique_dir else None
    output_dir   = os.path.abspath(args.output_dir)
    ckpt_dir     = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Build model
    model_kwargs = dict(causal=args.causal)
    if args.hidden   is not None: model_kwargs['hidden']   = args.hidden
    if args.n_blocks is not None: model_kwargs['n_blocks'] = args.n_blocks
    model = build_model(args.arch, **model_kwargs)

    start_epoch = 1
    resume_ckpt = None
    if args.resume:
        warnings.warn(
            "Loading checkpoint executes Python deserialization — only use "
            "checkpoints from trusted sources.", RuntimeWarning)
        resume_ckpt = torch.load(args.resume, map_location="cpu",
                                 weights_only=False)
        model.load_state_dict(resume_ckpt["state_dict"])
        start_epoch = resume_ckpt.get("epoch", 0) + 1
        print(f"Resumed from epoch {start_epoch - 1}")

    model.to(device)

    # Data
    loader = make_joint_loader(data_dir, tech_dir, args.seq_len,
                               args.batch_size, args.num_workers)

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.98), weight_decay=1e-4)

    from torch.optim.lr_scheduler import (LinearLR, CosineAnnealingLR,
                                           SequentialLR, LambdaLR)
    if args.scheduler == "constant":
        scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        args._sched_step = "epoch"
    else:  # cosine_warmup
        total_iters  = len(loader) * args.epochs
        warmup_iters = len(loader)
        warmup  = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_iters)
        cosine  = CosineAnnealingLR(optimizer,
                                    T_max=total_iters - warmup_iters,
                                    eta_min=args.lr * 0.01)
        scheduler = SequentialLR(optimizer, [warmup, cosine],
                                 milestones=[warmup_iters])
        args._sched_step = "iter"

    if resume_ckpt:
        if "optimizer" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer"])
        if "scheduler" in resume_ckpt:
            scheduler.load_state_dict(resume_ckpt["scheduler"])
        del resume_ckpt

    writer = SummaryWriter(log_dir=os.path.join(output_dir, "tb"))

    best_loss      = float("inf")
    best_metric    = 0.0    # macro F1 if technique data present, else macro RPA
    patience_count = 0
    global_step    = 0

    for epoch in range(start_epoch, start_epoch + args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, loader, optimizer, scheduler, writer,
            epoch, device, args, global_step_offset=global_step,
        )
        global_step += len(loader)

        if args._sched_step == "epoch":
            scheduler.step()

        dt = time.time() - t0
        print(f"  Epoch {epoch} — {dt:.1f}s — loss={train_loss:.5f}")

        # Checkpoint every epoch
        ckpt = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "arch": args.arch,
            "causal": args.causal,
            "model_kwargs": model_kwargs,
            "args": vars(args),
            "loss": train_loss,
        }
        torch.save(ckpt, os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pth"))

        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(ckpt, os.path.join(ckpt_dir, "best_loss.pth"))

        # Evaluation
        if epoch % args.eval_every == 0 or epoch == start_epoch:
            eval_res = evaluate(model, data_dir, tech_dir,
                                writer, epoch, device, args)
            if eval_res:
                # Prefer macro F1 as primary metric (technique is the goal);
                # fall back to macro RPA when no technique eval data.
                if 'macro_f1' in eval_res:
                    metric = eval_res['macro_f1']
                    metric_name = "macro_f1"
                elif 'macro_rpa' in eval_res:
                    metric = eval_res['macro_rpa']
                    metric_name = "macro_rpa"
                else:
                    metric = float('nan')
                    metric_name = "none"

                if not np.isnan(metric) and metric > best_metric:
                    best_metric = metric
                    patience_count = 0
                    torch.save(ckpt, os.path.join(ckpt_dir, "best_metric.pth"))
                    print(f"  New best {metric_name}: {best_metric:.4f} — checkpoint saved.")
                elif not np.isnan(metric):
                    patience_count += args.eval_every
                    if args.patience > 0 and patience_count >= args.patience:
                        print(f"  Early stopping after {epoch} epochs "
                              f"(best {metric_name}: {best_metric:.4f})")
                        break

    writer.close()
    print(f"\nTraining complete.")
    print(f"  Best loss checkpoint:   {ckpt_dir}/best_loss.pth")
    print(f"  Best metric checkpoint: {ckpt_dir}/best_metric.pth  "
          f"(value={best_metric:.4f})")

    import gc
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
