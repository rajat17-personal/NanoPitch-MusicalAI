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
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
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
parser.add_argument("--data-dir", type=str, default=None,
                    help="directory with clean.npz (mel/f0/vad) and optional test.npz. "
                         "Omit to train on technique clips only (all heads still train "
                         "via f0/vad labels in technique_train.npz).")
parser.add_argument("--technique-dirs", type=str, nargs="*", default=None,
                    help="one or more directories containing technique_train.npz "
                         "(e.g. data/vocalset data/gtsinger_technique). "
                         "Datasets are concatenated. If None, technique head "
                         "trains on pseudo-labels (zeros).")
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
parser.add_argument("--deep-technique-head", action="store_true",
                    help="replace the single Linear technique head with a 2-layer MLP "
                         "(Linear→GELU→Dropout→Linear). Recommended with --probe-mode "
                         "where the backbone is frozen and the head must do more work.")
parser.add_argument("--balance-datasets", action="store_true",
                    help="use WeightedRandomSampler so each technique source dataset "
                         "contributes equally to each batch. Fixes large/small dataset "
                         "imbalance (e.g. GTSinger 10k vs VocalSet 824 clips) that "
                         "causes minority-source classes to get F1=0.")

# Device
parser.add_argument("--device", type=str, default="cuda",
                    help="cpu / cuda / mps / auto")

# Hyperparameters
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--seq-len", type=int, default=300,
                    help="training clip length in frames (300 = 3 seconds)")
parser.add_argument("--num-workers", type=int, default=2)
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

# Per-class technique positive weights (Option 1 — class rebalancing)
parser.add_argument("--technique-pos-weights", type=float, nargs=5,
                    default=[2.9, 4.2, 1.0, 4.0, 1.9],
                    metavar=("VIB", "BRE", "FAL", "BELT", "STR"),
                    help="Per-class BCE positive weights for technique head "
                         "(n_neg/n_pos per class). Order: vibrato breathy falsetto belt straight. "
                         "Recompute if the technique dataset changes:\n"
                         "  VocalSet only (824 clips):          2.9  4.2  inf  4.0  1.9\n"
                         "  GTSinger-tech only (9601 clips):    1.9  3.3  1.0  inf  inf\n"
                         "  VocalSet + GTSinger (default):      2.9  4.2  1.0  4.0  1.9\n"
                         "  (inf = class absent; set to 1.0 to ignore that class)")

# Backbone freeze (Option B — stop technique gradients reaching backbone)
parser.add_argument("--freeze-backbone-epochs", type=int, default=0,
                    help="freeze backbone weights for this many epochs after resuming, "
                         "so only the technique head trains. Backbone unfreezes after N epochs "
                         "for joint fine-tuning. Use with --resume from a pitch-only checkpoint. "
                         "Rule of thumb: 20-30 epochs of frozen technique head, then unfreeze.")
parser.add_argument("--probe-mode", action="store_true",
                    help="MERT-style linear probing: freeze backbone AND pitch/VAD heads forever. "
                         "Only head_technique is trainable for the entire run. "
                         "Use with --resume from a pitch-only checkpoint (e.g. Run 4). "
                         "Pitch/VAD heads never see technique data — zero distribution drift. "
                         "Incompatible with --freeze-backbone-epochs.")

# Curriculum training (Option 3 — phase loss weights)
parser.add_argument("--curriculum", action="store_true",
                    help="Enable curriculum training: technique loss weight is zeroed "
                         "for the first --curriculum-warmup epochs, then linearly ramped "
                         "to --w-technique over --curriculum-ramp epochs. "
                         "Pitch/VAD heads converge before technique gradients compete. "
                         "The LR schedule runs unaffected — only w_technique changes.")
parser.add_argument("--curriculum-warmup", type=int, default=30,
                    help="epochs to train pitch+VAD only before enabling technique loss "
                         "(--curriculum only). Rule of thumb: set to ~30%% of total epochs.")
parser.add_argument("--curriculum-ramp", type=int, default=10,
                    help="epochs over which w_technique linearly ramps from 0 to its "
                         "target value (--curriculum only).")

# Pitch supervision
parser.add_argument("--pitch-sigma", type=float, default=1.2,
                    help="Gaussian sigma (bins) for pitch posteriorgram target")

# Evaluation frequency
parser.add_argument("--eval-every", type=int, default=5,
                    help="run evaluation every N epochs")

# Early stopping
parser.add_argument("--patience", type=int, default=0,
                    help="stop if macro technique F1 does not improve for N epochs "
                         "(0 = disabled)")

# Gradient clipping
parser.add_argument("--grad-clip", type=float, default=5.0)

# Noise augmentation
parser.add_argument("--augment", type=str, default="none",
                    choices=["none", "noise", "noise_specaug"],
                    help="'none' = clean-only (NanoPitch baseline comparison). "
                         "'noise' = log-mel noise mixing only (logaddexp). "
                         "'noise_specaug' = noise mixing + SpecAugment.")
parser.add_argument("--noise-dir", type=str, default=None,
                    help="directory containing noise.npz "
                         "(defaults to --data-dir if not set)")
parser.add_argument("--snr-range", type=float, nargs=2, default=[-10.0, 30.0],
                    help="min/max SNR in dB for noise mixing")
parser.add_argument("--p-clean", type=float, default=0.0,
                    help="fraction of batch rows passed through without noise "
                         "(0=always mix, 0.1=10%% rows stay clean)")
parser.add_argument("--snr-bias", type=float, default=1.0,
                    help="SNR draw exponent: <1 biases toward high SNR (cleaner), "
                         ">1 toward low SNR (noisier). 1.0 = uniform.")
parser.add_argument("--freq-mask-param", type=int, default=4,
                    help="SpecAugment: max mel-band width per frequency mask")
parser.add_argument("--n-freq-masks", type=int, default=2,
                    help="SpecAugment: number of frequency masks per sample")
parser.add_argument("--time-mask-param", type=int, default=10,
                    help="SpecAugment: max frame width per time mask")
parser.add_argument("--n-time-masks", type=int, default=2,
                    help="SpecAugment: number of time masks per sample")


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
        # Keep float16 in RAM — 2× smaller than float32. Cast on slice in __getitem__.
        self.mel = clean["mel"]                          # (total_frames, 40) float16
        self.f0  = clean["f0"]                          # (total_frames,)    float16
        self.vad = clean["vad"]                         # (total_frames,)    float16
        lengths  = clean["lengths"]

        self.segments = []
        offset = 0
        for length in lengths:
            if length >= seq_len:
                self.segments.append((offset, offset + length))
            offset += length

        self.rng = np.random.default_rng()
        mem_mb = (self.mel.nbytes + self.f0.nbytes + self.vad.nbytes) / 1024**2
        print(f"PitchVADDataset: {len(self.mel):,} frames, "
              f"{len(self.segments)} usable segments, RAM={mem_mb:.0f} MB")

    def __len__(self):
        return min(len(self.segments) * 5, 20000)

    def __getitem__(self, _):
        seg_idx = self.rng.integers(len(self.segments))
        s, e = self.segments[seg_idx]
        t = self.rng.integers(0, e - s - self.seq_len + 1)
        t += s

        mel = self.mel[t:t + self.seq_len].astype(np.float32)   # (T, 40)
        f0  = self.f0 [t:t + self.seq_len].astype(np.float32)   # (T,)
        vad = self.vad[t:t + self.seq_len].astype(np.float32)   # (T,)
        technique = np.zeros((self.seq_len, N_TECHNIQUES), dtype=np.float32)
        has_technique = np.float32(0.0)
        return mel, f0, vad, technique, has_technique


class TechniqueDataset(Dataset):
    """Loads technique_train.npz (flat format from extractVocalSet.py) for
    technique classification supervision.

    Flat format (produced by extractVocalSet.py):
      mel:       (total_frames, 40)        float16
      f0:        (total_frames,)           float16
      vad:       (total_frames,)           float16
      technique: (n_clips, N_TECHNIQUES)  float32  — clip-level binary labels
      lengths:   (n_clips,)               int32

    Each __getitem__ samples a random seq_len window from a random clip and
    broadcasts that clip's technique label over the window.
    has_technique=1.0 signals the loss function to include these samples in
    the technique head gradient.
    """

    def __init__(self, technique_dir, seq_len=300,
                 filename="technique_train.npz"):
        self.seq_len = seq_len
        path = os.path.join(technique_dir, filename)
        data = np.load(path, allow_pickle=False)

        # Keep float16 in RAM — cast to float32 only on slice in __getitem__.
        self.mel       = data["mel"]                            # (total_frames, 40) float16
        self.f0        = data["f0"]                            # (total_frames,)    float16
        self.vad       = data["vad"]                           # (total_frames,)    float16
        self.technique = data["technique"].astype(np.float32)  # (n_clips, N_TECH) — small
        lengths        = data["lengths"]

        # Build (flat_start, flat_end, clip_idx) for each clip long enough
        self.segments = []
        offset = 0
        for clip_idx, length in enumerate(lengths):
            length = int(length)
            if length >= seq_len:
                self.segments.append((offset, offset + length, clip_idx))
            offset += length

        self.rng = np.random.default_rng()
        voiced_pct = float(np.mean(self.vad > 0)) * 100
        mem_mb = (self.mel.nbytes + self.f0.nbytes + self.vad.nbytes) / 1024**2
        print(f"TechniqueDataset ({filename}): {len(lengths)} clips, "
              f"{len(self.mel):,} frames, voiced={voiced_pct:.1f}%, "
              f"{len(self.segments)} usable segments, RAM={mem_mb:.0f} MB")

    def __len__(self):
        return min(len(self.segments) * 5, 15000)

    def __getitem__(self, _):
        seg_idx = self.rng.integers(len(self.segments))
        s, e, clip_idx = self.segments[seg_idx]

        t0 = self.rng.integers(0, e - s - self.seq_len + 1) + s
        mel = self.mel[t0:t0 + self.seq_len].astype(np.float32)
        f0  = self.f0 [t0:t0 + self.seq_len].astype(np.float32)
        vad = self.vad[t0:t0 + self.seq_len].astype(np.float32)

        # Broadcast clip-level label over the sampled window
        clip_label = self.technique[clip_idx]                   # (N_TECH,)
        technique  = np.broadcast_to(
            clip_label[np.newaxis, :], (self.seq_len, len(clip_label))
        ).copy()                                                 # (T, N_TECH)

        has_technique = np.float32(1.0)
        return mel, f0, vad, technique, has_technique


def make_joint_loader(pitch_vad_dir, technique_dirs, seq_len, batch_size,
                      num_workers, balance_datasets=False):
    """Combine PitchVADDataset and one or more TechniqueDatasets via ConcatDataset.

    pitch_vad_dir: directory with clean.npz, or None to skip (technique-only mode).
    technique_dirs: list of directories each containing technique_train.npz.
    balance_datasets: when True and multiple technique sources are present, use
        WeightedRandomSampler so each source dataset contributes equally to each
        batch regardless of its clip count. Fixes the 13:1 GTSinger/VocalSet
        imbalance that causes belt/straight F1=0 in combined runs.

    Technique clips carry f0/vad labels, so all three heads (pitch, VAD, technique)
    train even when pitch_vad_dir is None. Omitting pitch_vad_dir avoids the overlap
    where VocalSet/GTSinger clips appear in both clean.npz and technique_train.npz.
    """
    from torch.utils.data import ConcatDataset, WeightedRandomSampler

    datasets = []
    dataset_sizes = []
    if pitch_vad_dir is not None:
        pv = PitchVADDataset(pitch_vad_dir, seq_len)
        datasets.append(pv)
        dataset_sizes.append(len(pv))
        n_pv = len(pv)
    else:
        n_pv = 0

    tech_datasets = []
    for tech_dir in (technique_dirs or []):
        # Accept both naming conventions: technique_train.npz (VocalSet)
        # and technique_gtsinger_train.npz (GTSinger WAV extraction).
        for fname in ("technique_train.npz", "technique_gtsinger_train.npz"):
            tech_path = os.path.join(tech_dir, fname)
            if os.path.exists(tech_path):
                tech_datasets.append(TechniqueDataset(tech_dir, seq_len, filename=fname))
                break
        else:
            print(f"  [warn] technique_train.npz not found in {tech_dir} — skipping")

    datasets.extend(tech_datasets)
    dataset_sizes.extend(len(d) for d in tech_datasets)

    if not datasets:
        raise RuntimeError("No training data found — pass --data-dir and/or --technique-dirs.")

    n_tech = sum(len(d) for d in tech_datasets)
    if n_pv and n_tech:
        print(f"Joint dataset: {n_pv} pitch/VAD + {n_tech} technique samples "
              f"({len(tech_datasets)} technique source(s))")
    elif n_tech:
        print(f"Technique-only dataset: {n_tech} samples "
              f"({len(tech_datasets)} source(s)) — pitch/VAD heads train via f0/vad in technique clips")
    else:
        print(f"Pitch/VAD-only dataset: {n_pv} samples — technique head trains on pseudo-labels")

    dataset = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

    # Dataset-balanced sampler: each source dataset contributes 1/N_sources
    # of each batch, regardless of its clip count.  This prevents a large
    # GTSinger set (10 625 clips) from drowning out VocalSet (824 clips).
    sampler = None
    if balance_datasets and len(datasets) > 1:
        weights = []
        for size in dataset_sizes:
            w = 1.0 / (len(datasets) * size)   # uniform within dataset, equal across
            weights.extend([w] * size)
        weights = torch.tensor(weights, dtype=torch.float64)
        # Draw as many samples per epoch as the largest dataset × n_sources
        # so no dataset is under-sampled relative to a shuffle baseline.
        n_samples = max(dataset_sizes) * len(datasets)
        sampler = WeightedRandomSampler(weights, num_samples=n_samples,
                                        replacement=True)
        print(f"  Dataset-balanced sampler: {len(datasets)} sources, "
              f"{n_samples} samples/epoch (largest×{len(datasets)})")
        print(f"  Per-source weights: " +
              ", ".join(f"{sz} clips → {1/len(datasets):.1%}/epoch"
                        for sz in dataset_sizes))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    return loader


# ═══════════════════════════════════════════════════════════════════════
# Noise augmentation
# ═══════════════════════════════════════════════════════════════════════

class NoisePool:
    """Wraps noise.npz for fast random-window draws during training.

    noise.npz has the same flat layout as clean.npz:
      mel:     (total_frames, 40)  float16
      lengths: (n_clips,)          int32

    Draws are independent of the main Dataset — no DataLoader overhead.
    """

    def __init__(self, noise_dir, seq_len):
        path = os.path.join(noise_dir, "noise.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"noise.npz not found at {path}. "
                "Download from huggingface.co/datasets/smulelabs/NanoPitch-PreExtract "
                "or set --augment none to skip noise mixing.")
        data = np.load(path)
        self.mel = data["mel"]                       # (total_frames, 40) float16 — keep small
        lengths  = data["lengths"]
        self.segments = []
        offset = 0
        for length in lengths:
            if length >= seq_len:
                self.segments.append((offset, offset + int(length)))
            offset += int(length)
        self.seq_len = seq_len
        self.rng = np.random.default_rng()
        total_h = len(self.mel) * 160 / 16000 / 3600
        mem_mb = self.mel.nbytes / 1024**2
        print(f"NoisePool: {len(self.mel):,} frames ({total_h:.1f}h), "
              f"{len(self.segments)} usable segments, RAM={mem_mb:.0f} MB")

    def draw_batch(self, batch_size):
        """Return (batch_size, seq_len, 40) float32 array of noise windows."""
        out = np.empty((batch_size, self.seq_len, 40), dtype=np.float32)
        for i in range(batch_size):
            idx = self.rng.integers(len(self.segments))
            s, e = self.segments[idx]
            t = self.rng.integers(0, e - s - self.seq_len + 1) + s
            out[i] = self.mel[t:t + self.seq_len]
        return out


def augment_mel_batch(mel_clean, mel_noise, args, device):
    """Mix clean and noise log-mel at a random SNR per sample.

    Uses logaddexp so mixing is equivalent to linear-domain addition:
      log(exp(mel_clean) + scale * exp(mel_noise))

    mel_clean, mel_noise: (B, T, 40) tensors on ``device``
    Returns augmented (B, T, 40) tensor.
    """
    B = mel_clean.size(0)
    u = torch.rand(B, 1, 1, device=device)
    if args.snr_bias != 1.0:
        u = u.pow(args.snr_bias)
    lo, hi = args.snr_range
    snr_db = u * (hi - lo) + lo
    gain_offset = -snr_db * (np.log(10.0) / 20.0)
    mixed = torch.logaddexp(mel_clean, mel_noise + gain_offset)
    if args.p_clean > 0.0:
        keep_clean = torch.rand(B, 1, 1, device=device) < args.p_clean
        mixed = torch.where(keep_clean, mel_clean, mixed)
    return mixed


def spec_augment(mel, args):
    """SpecAugment: independent random frequency and time masking per sample.

    mel: (B, T, 40) on any device. Returns masked tensor (in-place clone).
    """
    mel = mel.clone()
    B, T, F = mel.shape
    for b in range(B):
        for _ in range(args.n_freq_masks):
            f = torch.randint(1, args.freq_mask_param + 1, (1,)).item()
            f0 = torch.randint(0, max(F - f, 1), (1,)).item()
            mel[b, :, f0:f0 + f] = 0.0
        for _ in range(args.n_time_masks):
            t = torch.randint(1, args.time_mask_param + 1, (1,)).item()
            t0 = torch.randint(0, max(T - t, 1), (1,)).item()
            mel[b, t0:t0 + t, :] = 0.0
    return mel


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
                 args, device, w_technique_override=None):
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
    # Per-class positive weights (Option 1 — class rebalancing).
    # Upweights minority classes so the model can't win by predicting all-negative.
    # UPDATE these weights if the technique dataset changes:
    #   VocalSet only (824 clips):         2.9  4.2  inf  4.0  1.9
    #   GTSinger-tech only (9601 clips):   1.9  3.3  1.0  inf  inf
    #   VocalSet + GTSinger (default):     2.9  4.2  1.0  4.0  1.9
    # (inf = class absent in that dataset; use 1.0 to ignore)
    tech_pos_w = torch.tensor(args.technique_pos_weights,
                              dtype=torch.float32, device=device)  # (N,)
    class_w = torch.where(
        technique_target > 0.5,
        tech_pos_w.view(1, 1, -1).expand_as(technique_target),
        torch.ones_like(technique_target),
    )                                                               # (B, T, N)
    mask = has_technique.view(-1, 1, 1)                            # (B, 1, 1)
    tech_per_elem = bce_none(pred_technique, technique_target)     # (B, T, N)
    if mask.sum() > 0:
        technique_loss = (mask * class_w * tech_per_elem).sum() / \
                         (mask.sum() * tech_per_elem.shape[1] * tech_per_elem.shape[2])
    else:
        technique_loss = torch.tensor(0.0, device=device)

    w_tech = w_technique_override if w_technique_override is not None else args.w_technique
    total = (args.w_vad * vad_loss
             + args.w_pitch * pitch_loss
             + w_tech * technique_loss)
    return total, vad_loss, pitch_loss, technique_loss


# ═══════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════

def _get_w_technique(epoch, args):
    """Return effective technique loss weight for this epoch (curriculum Option 3).

    When --curriculum is not set, returns args.w_technique unchanged.
    When set: weight is 0 for the first --curriculum-warmup epochs, then ramps
    linearly to args.w_technique over --curriculum-ramp epochs.
    The LR schedule is unaffected — only this scalar changes per epoch.
    """
    if not args.curriculum:
        return args.w_technique
    if epoch <= args.curriculum_warmup:
        return 0.0
    ramp_progress = min(epoch - args.curriculum_warmup, args.curriculum_ramp)
    return args.w_technique * ramp_progress / args.curriculum_ramp


def _set_backbone_frozen(model, frozen: bool, probe_mode: bool = False):
    """Freeze or unfreeze backbone weights.

    probe_mode=True: freeze backbone + pitch/VAD heads; only head_technique trains.
      This is the MERT linear-probing setup — pitch/VAD heads never see technique
      data so there is zero distribution drift on the pitch evaluation set.

    probe_mode=False (default): freeze backbone only; pitch/VAD heads stay trainable.
      Used for staged training where pitch/VAD heads continue refining.
    """
    if probe_mode:
        trainable_tops = {"head_technique"}
    elif frozen:
        trainable_tops = {"head_vad", "head_pitch", "head_technique"}
    else:
        trainable_tops = None  # unfreeze everything

    for name, param in model.named_parameters():
        top = name.split(".")[0]
        if trainable_tops is None:
            param.requires_grad = True
        else:
            param.requires_grad = (top in trainable_tops)

    state = "FROZEN" if (frozen or probe_mode) else "unfrozen"
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Backbone {state} — {trainable:,} trainable parameters")


def train_one_epoch(model, loader, optimizer, scheduler, writer,
                    epoch, device, args, noise_pool=None,
                    global_step_offset=0):
    model.train()
    running = {'total': 0.0, 'vad': 0.0, 'pitch': 0.0, 'technique': 0.0}
    n_batches = 0

    do_noise    = args.augment in ("noise", "noise_specaug") and noise_pool is not None
    do_specaug  = args.augment == "noise_specaug"

    pbar = tqdm(loader, desc=f"Epoch {epoch}", unit="batch")
    for batch_idx, (mel, f0, vad, technique, has_technique) in enumerate(pbar):
        mel           = mel.to(device)
        f0            = f0.to(device)
        vad           = vad.to(device)
        technique     = technique.to(device)
        has_technique = has_technique.to(device)

        if do_noise:
            B = mel.size(0)
            noise_np = noise_pool.draw_batch(B)
            noise_t  = torch.from_numpy(noise_np).to(device)
            mel = augment_mel_batch(mel, noise_t, args, device)
        if do_specaug:
            mel = spec_augment(mel, args)

        pred_vad, pred_pitch, pred_technique, _ = model(mel)

        effective_w_technique = _get_w_technique(epoch, args)
        total, vad_l, pitch_l, tech_l = compute_loss(
            pred_vad, pred_pitch, pred_technique,
            vad, f0, technique, has_technique,
            args, device,
            w_technique_override=effective_w_technique,
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
        return float("nan"), {'vad': float('nan'), 'pitch': float('nan'), 'technique': float('nan')}

    avgs = {k: v / n_batches for k, v in running.items()}
    for k, v in avgs.items():
        writer.add_scalar(f"train/{k}", v, epoch)
    writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], epoch)
    return avgs['total'], avgs


# ═══════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, data_dir, technique_dirs, writer, epoch, device, args):
    """Evaluate pitch RPA (from test.npz) and technique F1 (from technique*.npz).

    technique_dirs: list of directories (or None/empty). All test files found
    across all dirs are merged so multi-dataset runs evaluate all classes.
    """
    if isinstance(technique_dirs, str):
        technique_dirs = [technique_dirs]  # back-compat
    model.eval()
    results = {}

    # ── Pitch / VAD evaluation ───────────────────────────────────────────
    test_path = os.path.join(data_dir, "test.npz") if data_dir else None
    if test_path and os.path.exists(test_path):
        test = np.load(test_path)
        clips   = test['clips']   # (N, T, 40)
        f0_all  = test['f0']      # (N, T)
        snrs    = test['snr']     # (N,)
        N = clips.shape[0]

        clip_results = []
        for i in range(N):
            mel = torch.from_numpy(clips[i].astype(np.float32)).unsqueeze(0).to(device)
            v, p, _, _ = model(mel)
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
    # Collect test files from all technique dirs and merge them so that
    # multi-dataset runs (VocalSet + GTSinger) evaluate all classes together.
    tech_files = []
    for tdir in (technique_dirs or []):
        for _fname in ("technique_test.npz", "technique_gtsinger_test.npz"):
            _p = os.path.join(tdir, _fname)
            if os.path.exists(_p):
                tech_files.append(_p)
                break

    if tech_files:
        mel_parts, tech_parts, len_parts = [], [], []
        for tf in tech_files:
            d = np.load(tf, allow_pickle=True)
            mel_parts.append(d["mel"].astype(np.float32))
            tech_parts.append(d["technique"].astype(np.float32))
            len_parts.append(d["lengths"].astype(np.int32))
        mel_flat = np.concatenate(mel_parts, axis=0)
        tech_all = np.concatenate(tech_parts, axis=0)
        lengths  = np.concatenate(len_parts,  axis=0)
        print(f"  Technique eval: {len(tech_files)} file(s), {len(lengths)} clips")

        # Split flat mel back into per-clip tensors
        all_pred, all_true = [], []
        offset = 0
        for clip_idx, clip_len in enumerate(lengths):
            clip_mel = mel_flat[offset:offset + clip_len]   # (T, 40)
            offset  += clip_len
            if clip_len == 0:
                continue
            mel_t = torch.from_numpy(clip_mel).unsqueeze(0).to(device)  # (1,T,40)
            with torch.no_grad():
                _, _, pred_tech, _ = model(mel_t)
            # Average over time for clip-level prediction
            pred_clip = pred_tech.squeeze(0).mean(0).cpu().numpy()   # (N_TECH,)
            all_pred.append(pred_clip)
            all_true.append(tech_all[clip_idx])                       # (N_TECH,)

        all_pred = np.stack(all_pred)   # (N, N_TECH)
        all_true = np.stack(all_true)   # (N, N_TECH)
        pred_bin = (all_pred > 0.5).astype(int)

        print(f"\n  {'Technique':<12}  {'Prec':>6}  {'Recall':>6}  {'F1':>6}  {'AP':>6}")
        print(f"  {'─'*12}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}")
        f1s, aps = [], []
        for k, name in enumerate(TECHNIQUE_NAMES):
            if all_true[:, k].sum() == 0:
                continue
            p, r, f1, _ = precision_recall_fscore_support(
                all_true[:, k], pred_bin[:, k], average='binary', zero_division=0)
            ap = average_precision_score(all_true[:, k], all_pred[:, k])
            print(f"  {name:<12}  {p:6.3f}  {r:6.3f}  {f1:6.3f}  {ap:6.3f}")
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

    data_dir     = os.path.abspath(args.data_dir) if args.data_dir else None
    tech_dirs    = [os.path.abspath(d) for d in args.technique_dirs] \
                   if args.technique_dirs else []
    output_dir   = os.path.abspath(args.output_dir)
    ckpt_dir     = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Build model
    model_kwargs = dict(causal=args.causal)
    if args.hidden   is not None: model_kwargs['hidden']   = args.hidden
    if args.n_blocks is not None: model_kwargs['n_blocks'] = args.n_blocks
    if args.deep_technique_head:  model_kwargs['deep_technique_head'] = True
    model = build_model(args.arch, **model_kwargs)

    start_epoch = 1
    resume_ckpt = None
    if args.resume:
        warnings.warn(
            "Loading checkpoint executes Python deserialization — only use "
            "checkpoints from trusted sources.", RuntimeWarning)
        resume_ckpt = torch.load(args.resume, map_location="cpu",
                                 weights_only=False)
        missing, unexpected = model.load_state_dict(
            resume_ckpt["state_dict"], strict=False)
        start_epoch = resume_ckpt.get("epoch", 0) + 1
        if missing:
            print(f"  Checkpoint missing keys (will init from scratch): {missing}")
        if unexpected:
            print(f"  Checkpoint unexpected keys (ignored): {unexpected}")
        print(f"Resumed from epoch {start_epoch - 1}")

    model.to(device)

    # Noise pool (loaded once; drawn per-batch in training loop)
    noise_pool = None
    if args.augment != "none":
        noise_dir = (os.path.abspath(args.noise_dir) if args.noise_dir
                     else data_dir)
        if noise_dir is None:
            raise RuntimeError(
                "--augment requires noise.npz: pass --noise-dir or --data-dir")
        noise_pool = NoisePool(noise_dir, args.seq_len)
    print(f"Augmentation: {args.augment}"
          + (f"  SNR=[{args.snr_range[0]},{args.snr_range[1]}] dB"
             f"  p_clean={args.p_clean}" if args.augment != "none" else ""))

    # Data
    loader = make_joint_loader(data_dir, tech_dirs, args.seq_len,
                               args.batch_size, args.num_workers,
                               balance_datasets=args.balance_datasets)

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
        if "optimizer" in resume_ckpt and not args.probe_mode:
            optimizer.load_state_dict(resume_ckpt["optimizer"])
        elif "optimizer" in resume_ckpt and args.probe_mode:
            print("  Probe mode: skipping optimizer state (param groups changed)")
        if "scheduler" in resume_ckpt and not args.probe_mode:
            scheduler.load_state_dict(resume_ckpt["scheduler"])
        del resume_ckpt

    writer = SummaryWriter(log_dir=os.path.join(output_dir, "tb"))

    best_loss      = float("inf")
    best_metric    = 0.0    # macro F1 if technique data present, else macro RPA
    patience_count = 0
    global_step    = 0

    if args.probe_mode:
        _set_backbone_frozen(model, frozen=True, probe_mode=True)
        print(f"  Probe mode: backbone + pitch/VAD heads frozen for all {args.epochs} epochs")
        # Rebuild optimizer over trainable params only so frozen params get no momentum state
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, betas=(0.9, 0.98), weight_decay=1e-4,
        )

    freeze_until = start_epoch + args.freeze_backbone_epochs
    if not args.probe_mode and args.freeze_backbone_epochs > 0:
        _set_backbone_frozen(model, frozen=True)
        print(f"  Backbone frozen for epochs {start_epoch}–{freeze_until - 1}, "
              f"unfreezes at epoch {freeze_until}")

    for epoch in range(start_epoch, start_epoch + args.epochs):
        if not args.probe_mode and args.freeze_backbone_epochs > 0 and epoch == freeze_until:
            _set_backbone_frozen(model, frozen=False)
            print(f"  Epoch {epoch}: backbone unfrozen — joint fine-tuning begins")

        t0 = time.time()
        train_loss, train_losses = train_one_epoch(
            model, loader, optimizer, scheduler, writer,
            epoch, device, args, noise_pool=noise_pool,
            global_step_offset=global_step,
        )
        global_step += len(loader)

        if args._sched_step == "epoch":
            scheduler.step()

        dt = time.time() - t0
        print(f"  Epoch {epoch} — {dt:.1f}s — loss={train_loss:.5f}"
              f"  (vad={train_losses['vad']:.4f}"
              f"  pitch={train_losses['pitch']:.4f}"
              f"  tech={train_losses['technique']:.4f})")

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
            eval_res = evaluate(model, data_dir, tech_dirs,
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
