"""
VocalCoach Training Script
==========================

Trains VocalCoachTCN or VocalCoachConformer for multi-task singing analysis:
  1. VAD — is the singer vocalising at each frame?
  2. Pitch posteriorgram — what pitch is being sung? (decoded via Viterbi)
  3. Technique classification — vibrato / breathy / falsetto / belt / straight
     (multi-label sigmoid, not mutually exclusive)
  4. Quality scoring head — singing quality score(s) via frozen backbone
     (Variants 1 / 2 / 3 — see VOCALCOACH_PROPOSAL.md)

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

Quality scoring variants (--quality-variant):
  1  Scalar contrastive head trained on PopBuTFy same-singer pairs.
     Requires --popbutfy-dir and --baselines-json.
     Uses margin ranking loss: score(pro) > score(amateur) by --ranking-margin.

  2  Multi-dim head (9 outputs) trained on ccmusic expert labels (MSE) +
     optional SingMOS-Pro pretraining (MSE on AudioScore pseudo-labels) +
     contrastive calibration on PopBuTFy pairs.
     Requires --ccmusic-wavs-dir and --baselines-json.
     Optional: --singmos-scores-json for Stage 1 pretraining.

  3  Scalar head with MSE distillation from AudioScore on SingMOS-Pro clips,
     followed by contrastive calibration on PopBuTFy pairs.
     Requires --singmos-scores-json and --popbutfy-dir and --baselines-json.

  All variants: --probe-mode is required (backbone frozen; only head_quality trains).
  Combine with --resume pointing to a trained pitch+technique checkpoint.

Usage
-----
# Train pitch + VAD (Arch B — Conformer, non-causal)
python vocalcoach/train.py \\
    --arch conformer --causal false \\
    --data-dir data \\
    --technique-dirs data/vocalset \\
    --output-dir vocalcoach/runs/conformer_noncausal

# Prepare quality NPZs first (run once):
#   python scripts/prepareQualityData.py --popbutfy-dir data/popbutfy \
#       --baselines-json data/combined_eval_baselines.json \
#       --output-dir data/quality
#   (add --ccmusic-wavs-dir / --singmos-scores-json for Variants 2/3)

# Train quality head — Variant 1 (contrastive, scalar)
python vocalcoach/train.py \\
    --arch conformer --causal false \\
    --quality-variant 1 \\
    --probe-mode \\
    --resume vocalcoach/runs/conformer_probe_technique/checkpoints/best_metric.pth \\
    --quality-pairs-npz data/quality/quality_pairs.npz \\
    --output-dir vocalcoach/runs/conformer_quality_v1

# Train quality head — Variant 2 (multi-dim, ccmusic + SingMOS-Pro + contrastive)
python vocalcoach/train.py \\
    --arch conformer --causal false \\
    --quality-variant 2 \\
    --probe-mode \\
    --resume vocalcoach/runs/conformer_probe_technique/checkpoints/best_metric.pth \\
    --quality-pairs-npz   data/quality/quality_pairs.npz \\
    --quality-ccmusic-npz data/quality/quality_ccmusic.npz \\
    --quality-mse-npz     data/quality/quality_mse.npz \\
    --output-dir vocalcoach/runs/conformer_quality_v2

# Train quality head — Variant 3 (MSE distil from AudioScore + contrastive)
python vocalcoach/train.py \\
    --arch conformer --causal false \\
    --quality-variant 3 \\
    --probe-mode \\
    --resume vocalcoach/runs/conformer_probe_technique/checkpoints/best_metric.pth \\
    --quality-pairs-npz data/quality/quality_pairs.npz \\
    --quality-mse-npz   data/quality/quality_mse.npz \\
    --output-dir vocalcoach/runs/conformer_quality_v3

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
                    help="number of TCN blocks (default: 8 for TCN); ignored for Conformer — use --n-layers instead")
parser.add_argument("--n-layers", type=int, default=None,
                    help="number of Conformer blocks (default: 4); Conformer only")
parser.add_argument("--n-heads", type=int, default=None,
                    help="number of attention heads for Conformer (default: 4); must divide --hidden evenly")
parser.add_argument("--deep-technique-head", action="store_true",
                    help="replace the single Linear technique head with a 2-layer MLP "
                         "(Linear→GELU→Dropout→Linear). Recommended with --probe-mode "
                         "where the backbone is frozen and the head must do more work.")
parser.add_argument("--balance-datasets", action="store_true",
                    help="use WeightedRandomSampler so each technique source dataset "
                         "contributes equally to each batch. Fixes large/small dataset "
                         "imbalance (e.g. GTSinger 10k vs VocalSet 824 clips) that "
                         "causes minority-source classes to get F1=0.")

# ── Note segmentation head (Variant 4) ───────────────────────────────────────
parser.add_argument("--note-head", action="store_true", default=False,
                    help="enable note segmentation head (Variant 4): adds head_note_onset "
                         "and head_note_offset — two binary frame-level classifiers. "
                         "Requires --note-dirs pointing to dirs with note_train.npz. "
                         "Default off — no change to existing runs.")
parser.add_argument("--note-dirs", type=str, nargs="*", default=None,
                    help="directories containing note_train.npz for note onset/offset "
                         "supervision. If None while --note-head is set, note head "
                         "trains on zero pseudo-labels.")
parser.add_argument("--w-note", type=float, default=1.0,
                    help="weight for note onset+offset loss (BCE). Only active with "
                         "--note-head. Default 1.0.")

# ── Warm-up all heads before technique/quality ────────────────────────────────
parser.add_argument("--warmup-heads-epochs", type=int, default=0,
                    help="train pitch+VAD (+ note if --note-head) for this many epochs "
                         "before enabling technique and quality losses. "
                         "0 = disabled (default, existing behaviour). "
                         "Use e.g. 30-50 to let pitch/VAD/note converge before "
                         "technique gradients compete. Supersedes --curriculum-warmup "
                         "when both are set.")

# ── Dataset curriculum: pretrain on one source, fine-tune on another ──────────
parser.add_argument("--pretrain-data-dir", type=str, default=None,
                    help="if set, train on this data source alone for "
                         "--pretrain-epochs epochs before switching to the main "
                         "--data-dir. Implements dataset curriculum: e.g. GTSinger-only "
                         "warm-up then fine-tune on VocalSet+GTSinger. "
                         "Default None = single dataset throughout (existing behaviour).")
parser.add_argument("--pretrain-epochs", type=int, default=30,
                    help="number of epochs to train exclusively on --pretrain-data-dir "
                         "before switching to --data-dir. Only active when "
                         "--pretrain-data-dir is set. Default 30.")
parser.add_argument("--pretrain-technique-dirs", type=str, nargs="*", default=None,
                    help="technique directories to use during --pretrain-epochs. "
                         "If None, uses --technique-dirs for all epochs.")

# Device
parser.add_argument("--device", type=str, default="cuda",
                    help="cpu / cuda / mps / auto")

# Hyperparameters
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--lr-backbone", type=float, default=None,
                    help="separate learning rate for backbone parameters. "
                         "When set, backbone uses this LR and all heads use --lr. "
                         "Default None = all parameters share --lr (single-group behavior). "
                         "Typical use: --lr 3e-4 --lr-backbone 3e-5 to let the "
                         "technique head learn faster while the backbone shifts slowly.")
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
parser.add_argument("--contrastive-technique", action="store_true", default=False,
                    help="add supervised contrastive (SupCon) loss on technique embeddings "
                         "alongside the standard BCE classification loss. Clips with the "
                         "same technique label are pulled together; different labels are "
                         "pushed apart. Requires --technique-dirs. "
                         "Controlled by --w-contrastive-technique and --contrastive-temp.")
parser.add_argument("--w-contrastive-technique", type=float, default=0.5,
                    help="weight for the technique SupCon loss term (added to --w-technique * BCE). "
                         "Default 0.5. Only active when --contrastive-technique is set.")
parser.add_argument("--contrastive-temp", type=float, default=0.07,
                    help="temperature for SupCon loss (shared by technique and future quality "
                         "contrastive variants). Lower = sharper distribution. "
                         "Default 0.07 (SimCLR / SupCon paper default).")

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

# Backbone freeze
parser.add_argument("--freeze-backbone-epochs", type=int, default=0,
                    help="freeze backbone weights for this many epochs after resuming, "
                         "so only the technique head trains. Backbone unfreezes after N epochs "
                         "for joint fine-tuning. Use with --resume from a pitch-only checkpoint. "
                         "Rule of thumb: 20-30 epochs of frozen technique head, then unfreeze.")
parser.add_argument("--probe-mode", action="store_true",
                    help="MERT-style linear probing: freeze backbone AND pitch/VAD heads forever. "
                         "When --quality-variant is set, also freezes head_technique — only "
                         "head_quality trains. When not set, only head_technique trains. "
                         "Use with --resume from a trained pitch+technique checkpoint. "
                         "Incompatible with --freeze-backbone-epochs.")

# Curriculum training
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

# ── Quality scoring head ──────────────────────────────────────────────────────
parser.add_argument("--quality-variant", type=int, default=0,
                    choices=[0, 1, 2, 3],
                    help="Quality head training variant (0 = disabled). "
                         "Pre-extract NPZs with scripts/prepareQualityData.py first. "
                         "1: scalar contrastive on PopBuTFy pairs (--quality-pairs-npz). "
                         "2: 9-dim MSE on ccmusic + SingMOS-Pro + contrastive "
                         "(all three --quality-*-npz flags). "
                         "3: scalar MSE distil + contrastive "
                         "(--quality-mse-npz + --quality-pairs-npz). "
                         "All variants require --probe-mode and --resume.")
parser.add_argument("--quality-pairs-npz", type=str, default=None,
                    help="path to quality_pairs.npz produced by prepareQualityData.py. "
                         "Required for all quality variants (contrastive stage).")
parser.add_argument("--quality-mse-npz", type=str, default=None,
                    help="path to quality_mse.npz (SingMOS-Pro AudioScore scalars). "
                         "Required for --quality-variant 2 and 3.")
parser.add_argument("--quality-ccmusic-npz", type=str, default=None,
                    help="path to quality_ccmusic.npz (ccmusic 9-dim expert scores). "
                         "Required for --quality-variant 2.")
parser.add_argument("--ranking-margin", type=float, default=0.5,
                    help="margin for pairwise ranking loss: score(pro) - score(amateur) > margin. "
                         "Larger values demand clearer separation.")
parser.add_argument("--w-ranking", type=float, default=1.0,
                    help="weight for contrastive ranking loss relative to other quality losses.")
parser.add_argument("--w-quality-mse", type=float, default=1.0,
                    help="weight for MSE quality regression loss (Variants 2 and 3).")
parser.add_argument("--quality-epochs-mse", type=int, default=30,
                    help="epochs to run MSE pretraining before switching to contrastive "
                         "(Variants 2 and 3). After this, both MSE and ranking losses are active.")

# Pitch supervision
parser.add_argument("--pitch-sigma", type=float, default=1.2,
                    help="Gaussian sigma (bins) for pitch posteriorgram target")

# Evaluation frequency
parser.add_argument("--eval-every", type=int, default=5,
                    help="run evaluation every N epochs")

# Early stopping
parser.add_argument("--patience", type=int, default=0,
                    help="stop if best metric does not improve for N epochs "
                         "(0 = disabled)")
parser.add_argument("--metric-vdr-weight", type=float, default=1.0,
                    help="relative weight of VDR_clean vs F1/RPA in the compound "
                         "checkpoint metric: score = primary + w * vdr_clean. "
                         "Default 1.0 = equal additive weight. "
                         "Increase (e.g. 2.0) to penalize VDR regression more heavily; "
                         "decrease (e.g. 0.5) to prioritize F1/RPA; "
                         "0.0 = track F1/RPA only.")

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


class NoteDataset(Dataset):
    """Loads note_train.npz for note onset/offset frame-level supervision (Variant 4).

    NPZ schema (flat layout, produced by extractNotes.py or similar):
      mel:     (total_frames, 40)   float16
      onset:   (total_frames,)      float32  — 1.0 at note onset frames
      offset:  (total_frames,)      float32  — 1.0 at note offset frames
      lengths: (n_clips,)           int32

    Returns (onset_target, offset_target, has_note) per sample where
    has_note=1.0 signals that this sample has real labels.
    """

    def __init__(self, note_dir, seq_len=300, filename="note_train.npz"):
        self.seq_len = seq_len
        path = os.path.join(note_dir, filename)
        data = np.load(path, allow_pickle=False)

        self.mel    = data["mel"]                          # (total_frames, 40) float16
        self.onset  = data["onset"].astype(np.float32)    # (total_frames,)
        self.offset = data["offset"].astype(np.float32)   # (total_frames,)
        lengths     = data["lengths"]

        self.segments = []
        offset_idx = 0
        for length in lengths:
            length = int(length)
            if length >= seq_len:
                self.segments.append((offset_idx, offset_idx + length))
            offset_idx += length

        self.rng = np.random.default_rng()
        print(f"NoteDataset ({filename}): {len(lengths)} clips, "
              f"{len(self.mel):,} frames, {len(self.segments)} usable segments")

    def __len__(self):
        return min(len(self.segments) * 5, 15000)

    def __getitem__(self, _):
        seg_idx = self.rng.integers(len(self.segments))
        s, e = self.segments[seg_idx]
        t0 = self.rng.integers(0, e - s - self.seq_len + 1) + s
        onset  = self.onset [t0:t0 + self.seq_len]        # (T,)
        offset = self.offset[t0:t0 + self.seq_len]        # (T,)
        return onset, offset, np.float32(1.0)


# ═══════════════════════════════════════════════════════════════════════
# Quality scoring datasets  (NPZ-backed — pre-extracted by prepareQualityData.py)
# ═══════════════════════════════════════════════════════════════════════

class MseQualityDataset(Dataset):
    """Scalar AudioScore targets from quality_mse.npz (Variants 2 and 3 Stage 1).

    NPZ schema (produced by scripts/prepareQualityData.py) — flat layout:
      mel:     (total_frames, 40)  float16  — concatenated full clips
      scores:  (n_clips,)          float32  — one scalar per clip
      lengths: (n_clips,)          int32

    __getitem__ samples a random seq_len window from a random clip, matching
    the PitchVADDataset pattern so seq_len changes in train.py take effect.
    """

    def __init__(self, npz_path, seq_len=300):
        self.seq_len = seq_len
        data = np.load(npz_path, allow_pickle=True)
        self.mel    = data['mel']                           # (total_frames, 40) float16
        self.scores = data['scores'].astype(np.float32)    # (n_clips,)
        lengths     = data['lengths']

        self.segments = []
        offset = 0
        for clip_idx, length in enumerate(lengths):
            length = int(length)
            if length >= seq_len:
                self.segments.append((offset, offset + length, clip_idx))
            offset += length

        self.rng = np.random.default_rng()
        print(f"MseQualityDataset: {len(lengths)} clips "
              f"({len(self.segments)} usable ≥{seq_len} frames) from {npz_path}")

    def __len__(self):
        return len(self.segments) * 5   # multiple passes per clip per epoch

    def __getitem__(self, _):
        s, e, clip_idx = self.segments[self.rng.integers(len(self.segments))]
        t0 = int(self.rng.integers(0, e - s - self.seq_len + 1)) + s
        mel = self.mel[t0:t0 + self.seq_len].astype(np.float32)
        return mel, self.scores[clip_idx]


class CcmusicQualityDataset(Dataset):
    """9-dim expert score targets from quality_ccmusic.npz (Variant 2 Stage 2).

    NPZ schema (produced by scripts/prepareQualityData.py) — flat layout:
      mel:     (total_frames, 40)  float16
      scores:  (n_clips, 9)        float32  — CCMUSIC_DIMS order
      lengths: (n_clips,)          int32

    Uses a higher repeat multiplier (×20) since ccmusic has only 132 clips —
    each epoch draws many random windows from each clip.
    """

    def __init__(self, npz_path, seq_len=300):
        self.seq_len = seq_len
        data = np.load(npz_path, allow_pickle=True)
        self.mel    = data['mel']                            # (total_frames, 40) float16
        self.scores = data['scores'].astype(np.float32)     # (n_clips, 9)
        lengths     = data['lengths']

        self.segments = []
        offset = 0
        for clip_idx, length in enumerate(lengths):
            length = int(length)
            if length >= seq_len:
                self.segments.append((offset, offset + length, clip_idx))
            offset += length

        self.rng = np.random.default_rng()
        print(f"CcmusicQualityDataset: {len(lengths)} clips "
              f"({len(self.segments)} usable), {self.scores.shape[1]}-dim targets "
              f"from {npz_path}")

    def __len__(self):
        return len(self.segments) * 20   # small dataset — many windows per clip

    def __getitem__(self, _):
        s, e, clip_idx = self.segments[self.rng.integers(len(self.segments))]
        t0 = int(self.rng.integers(0, e - s - self.seq_len + 1)) + s
        mel = self.mel[t0:t0 + self.seq_len].astype(np.float32)
        return mel, self.scores[clip_idx]


class PairedQualityDataset(Dataset):
    """PopBuTFy pro/amateur pairs for contrastive ranking from quality_pairs.npz.

    NPZ schema (produced by scripts/prepareQualityData.py) — flat layout:
      mel_pro:     (total_frames_pro, 40)  float16
      mel_am:      (total_frames_am,  40)  float16
      lengths_pro: (n_pairs,)  int32
      lengths_am:  (n_pairs,)  int32

    Each __getitem__ draws independent random windows from the pro and amateur
    clips of a randomly selected pair. Windows are drawn independently so the
    model sees different temporal contexts for each clip within a pair.
    """

    def __init__(self, npz_path, seq_len=300):
        self.seq_len = seq_len
        data = np.load(npz_path, allow_pickle=True)
        self.mel_pro = data['mel_pro']   # (total_frames_pro, 40) float16
        self.mel_am  = data['mel_am']    # (total_frames_am,  40) float16
        lengths_pro  = data['lengths_pro'].astype(np.int32)
        lengths_am   = data['lengths_am'].astype(np.int32)

        # Build segment index for each side independently
        self.segs_pro, self.segs_am = [], []
        off_pro = off_am = 0
        for i, (lp, la) in enumerate(zip(lengths_pro, lengths_am)):
            lp, la = int(lp), int(la)
            if lp >= seq_len and la >= seq_len:
                self.segs_pro.append((off_pro, off_pro + lp))
                self.segs_am.append((off_am,  off_am  + la))
            off_pro += lp
            off_am  += la

        self.rng = np.random.default_rng()
        print(f"PairedQualityDataset: {len(self.segs_pro)} usable pairs "
              f"(of {len(lengths_pro)} total) from {npz_path}")

    def __len__(self):
        return len(self.segs_pro)

    def __getitem__(self, _):
        idx = self.rng.integers(len(self.segs_pro))
        sp, ep = self.segs_pro[idx]
        sa, ea = self.segs_am[idx]
        t0p = int(self.rng.integers(0, ep - sp - self.seq_len + 1)) + sp
        t0a = int(self.rng.integers(0, ea - sa - self.seq_len + 1)) + sa
        mel_pro = self.mel_pro[t0p:t0p + self.seq_len].astype(np.float32)
        mel_am  = self.mel_am[t0a:t0a + self.seq_len].astype(np.float32)
        return mel_pro, mel_am


def make_quality_loaders(args, seq_len, batch_size, num_workers):
    """Build DataLoaders for the active quality variant stages.

    Returns a dict with keys from {'mse', 'ccmusic', 'pairs'} depending on variant.
    Empty dict if quality_variant == 0.
    """
    if args.quality_variant == 0:
        return {}

    loaders = {}

    # Stage 1: MSE pretraining on SingMOS-Pro AudioScore scalars (Variants 2 and 3)
    if args.quality_variant in (2, 3):
        if not args.quality_mse_npz:
            raise RuntimeError(
                "--quality-variant 2/3 requires --quality-mse-npz "
                "(run scripts/prepareQualityData.py --singmos-scores-json ... first)")
        ds_mse = MseQualityDataset(args.quality_mse_npz, seq_len)
        loaders['mse'] = DataLoader(ds_mse, batch_size=batch_size, shuffle=True,
                                    drop_last=True, num_workers=num_workers,
                                    pin_memory=True,
                                    persistent_workers=(num_workers > 0))

    # Stage 2: ccmusic 9-dim expert labels (Variant 2 only)
    if args.quality_variant == 2:
        if not args.quality_ccmusic_npz:
            raise RuntimeError(
                "--quality-variant 2 requires --quality-ccmusic-npz "
                "(run scripts/prepareQualityData.py --ccmusic-wavs-dir ... first)")
        ds_cc = CcmusicQualityDataset(args.quality_ccmusic_npz, seq_len)
        loaders['ccmusic'] = DataLoader(ds_cc, batch_size=min(16, batch_size),
                                        shuffle=True, drop_last=False,
                                        num_workers=num_workers, pin_memory=True,
                                        persistent_workers=(num_workers > 0))

    # Contrastive stage: PopBuTFy pairs (all variants)
    if not args.quality_pairs_npz:
        raise RuntimeError(
            "--quality-variant requires --quality-pairs-npz "
            "(run scripts/prepareQualityData.py --popbutfy-dir ... first)")
    ds_pairs = PairedQualityDataset(args.quality_pairs_npz, seq_len)
    loaders['pairs'] = DataLoader(ds_pairs, batch_size=batch_size, shuffle=True,
                                  drop_last=True, num_workers=num_workers,
                                  pin_memory=True,
                                  persistent_workers=(num_workers > 0))

    return loaders


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
        for fname in ("technique_train.npz", "technique_gtsinger_train.npz", "note_train.npz"):
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

def supcon_loss(embeddings, technique_labels, has_technique, temperature=0.07):
    """Supervised Contrastive loss on mean-pooled clip embeddings.

    embeddings:       (B, T, hidden)  — raw backbone features before heads
    technique_labels: (B, T, N)       — clip-level labels broadcast over T
    has_technique:    (B,)            — 1.0 for samples with technique labels
    temperature:      scalar

    Only samples with has_technique=1 participate. Clips sharing at least one
    technique class are treated as positives; clips sharing no class are negatives.
    Returns scalar loss (0.0 if fewer than 2 labelled samples in batch).
    """
    # Select only labelled samples
    mask = has_technique > 0.5                          # (B,)
    if mask.sum() < 2:
        return torch.tensor(0.0, device=embeddings.device)

    # Mean-pool over time → (B', hidden); L2-normalise
    z = embeddings[mask].mean(dim=1)                    # (B', hidden)
    z = torch.nn.functional.normalize(z, dim=-1)

    # Clip-level labels: take first time-step (all frames identical for technique)
    labels = technique_labels[mask, 0, :]               # (B', N)  float

    # Positive mask: pairs sharing at least one technique class
    # dot product of binary label vectors > 0  ↔  at least one shared class
    pos_mask = (labels @ labels.T) > 0                  # (B', B')
    # Remove self-pairs from positives
    pos_mask.fill_diagonal_(False)

    if pos_mask.sum() == 0:
        return torch.tensor(0.0, device=embeddings.device)

    # Similarity matrix
    sim = (z @ z.T) / temperature                       # (B', B')
    # Subtract max for numerical stability (log-sum-exp trick)
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()

    # Exclude self from denominator
    B = z.size(0)
    self_mask = ~torch.eye(B, dtype=torch.bool, device=z.device)
    exp_sim = torch.exp(sim) * self_mask                # (B', B')

    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

    # Mean over positive pairs
    n_pos = pos_mask.sum(dim=1).float().clamp(min=1)
    loss = -(log_prob * pos_mask).sum(dim=1) / n_pos
    return loss.mean()

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
                 args, device, w_technique_override=None,
                 note_onset=None, note_offset=None,
                 note_onset_target=None, note_offset_target=None,
                 has_note=None):
    """Compute multi-task loss.

    Args:
        pred_vad:           (B, T, 1)
        pred_pitch:         (B, T, 360)
        pred_technique:     (B, T, N)
        vad_target:         (B, T)
        f0_target:          (B, T)
        technique_target:   (B, T, N)
        has_technique:      (B,)  — 1.0 if sample has technique labels, else 0.0
        note_onset:         (B, T, 1) or None — predicted note onset probability
        note_offset:        (B, T, 1) or None — predicted note offset probability
        note_onset_target:  (B, T) or None — binary onset frame labels
        note_offset_target: (B, T) or None — binary offset frame labels
        has_note:           (B,) or None — 1.0 if sample has note labels
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

    # ── Note segmentation loss (Variant 4) — only when --note-head is active ──
    note_loss = torch.tensor(0.0, device=device)
    if (note_onset is not None and note_offset is not None
            and note_onset_target is not None and note_offset_target is not None
            and has_note is not None):
        note_mask = has_note.view(-1, 1, 1)                         # (B, 1, 1)
        if note_mask.sum() > 0:
            onset_loss  = bce_none(note_onset.squeeze(-1),  note_onset_target)   # (B, T)
            offset_loss = bce_none(note_offset.squeeze(-1), note_offset_target)  # (B, T)
            note_loss = ((note_mask.squeeze(-1) * onset_loss).sum()
                         + (note_mask.squeeze(-1) * offset_loss).sum()) / \
                        (note_mask.sum() * note_onset_target.shape[1] * 2)
        total = total + args.w_note * note_loss

    return total, vad_loss, pitch_loss, technique_loss, note_loss


def compute_quality_loss(model, quality_batch, device, args, epoch):
    """Compute quality head losses for the current epoch's active stages.

    quality_batch is a dict with a subset of keys:
      'mse':     (mel, score) from MseQualityDataset — scalar AudioScore targets
      'ccmusic': (mel, scores_9d) from CcmusicQualityDataset — 9-dim expert targets
      'pairs':   (mel_pro, mel_am) from PairedQualityDataset — contrastive pairs

    Returns (total_quality_loss, loss_dict) where loss_dict has per-component values.
    Active stages depend on variant and epoch:
      Variant 1:  pairs only (all epochs)
      Variant 2:  mse (epochs ≤ quality_epochs_mse), ccmusic (all epochs),
                  pairs (all epochs)
      Variant 3:  mse (epochs ≤ quality_epochs_mse), pairs (all epochs)
    """
    losses = {}
    use_mse     = args.quality_variant in (2, 3) and 'mse'     in quality_batch
    use_ccmusic = args.quality_variant == 2       and 'ccmusic' in quality_batch
    use_pairs   = 'pairs' in quality_batch

    mse_stage_active = epoch <= args.quality_epochs_mse

    # ── MSE on AudioScore scalars (SingMOS-Pro clips) ─────────────────────
    if use_mse and mse_stage_active:
        mel_mse, score_targets = quality_batch['mse']
        mel_mse       = mel_mse.to(device)
        score_targets = score_targets.to(device)           # (B,)
        _, _, _, q, _, _ = model(mel_mse)
        pred_scalar = q[:, 0]                              # (B,) — first dim
        mse_loss = F.mse_loss(pred_scalar, score_targets)
        losses['mse'] = mse_loss * args.w_quality_mse

    # ── MSE on ccmusic 9-dim expert labels ────────────────────────────────
    if use_ccmusic:
        mel_cc, expert_targets = quality_batch['ccmusic']
        mel_cc          = mel_cc.to(device)
        expert_targets  = expert_targets.to(device)        # (B, 9)
        _, _, _, q, _, _ = model(mel_cc)                   # q: (B, 9)
        cc_loss = F.mse_loss(q, expert_targets)
        losses['ccmusic'] = cc_loss * args.w_quality_mse

    # ── Contrastive ranking on PopBuTFy pairs ─────────────────────────────
    if use_pairs:
        mel_pro, mel_am = quality_batch['pairs']
        mel_pro = mel_pro.to(device)
        mel_am  = mel_am.to(device)
        _, _, _, q_pro, _, _ = model(mel_pro)    # (B, Q)
        _, _, _, q_am,  _, _ = model(mel_am)     # (B, Q)
        # Use dim 0 (overall quality scalar) for ranking regardless of Q
        score_pro = q_pro[:, 0]            # (B,)
        score_am  = q_am[:, 0]             # (B,)
        target = torch.ones_like(score_pro)
        ranking_loss = F.margin_ranking_loss(
            score_pro, score_am, target, margin=args.ranking_margin)
        losses['ranking'] = ranking_loss * args.w_ranking

    if not losses:
        return torch.tensor(0.0, device=device), {}

    total = sum(losses.values())
    return total, {k: v.item() for k, v in losses.items()}


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


def _set_backbone_frozen(model, frozen: bool, probe_mode: bool = False,
                         quality_probe: bool = False):
    """Freeze or unfreeze backbone weights.

    quality_probe=True: freeze everything except head_quality — used when
      training the scoring head on top of a fully trained pitch+technique checkpoint.

    probe_mode=True (no quality_probe): freeze backbone + pitch/VAD heads;
      only head_technique trains. The MERT linear-probing setup.

    probe_mode=False: freeze backbone only; pitch/VAD/technique heads stay trainable.
    """
    if quality_probe:
        trainable_tops = {"head_quality"}
    elif probe_mode:
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

    state = "FROZEN" if (frozen or probe_mode or quality_probe) else "unfrozen"
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Backbone {state} — {trainable:,} trainable parameters")


def train_one_epoch(model, loader, optimizer, scheduler, writer,
                    epoch, device, args, noise_pool=None,
                    global_step_offset=0, quality_loaders=None,
                    note_loader=None):
    """Train for one epoch.

    quality_loaders: dict returned by make_quality_loaders, or None/empty.
      When non-empty, each quality loader is iterated in lock-step with the main
      loader (cycling the shorter ones). The pitch/VAD/technique loss is zero when
      quality_variant > 0 and probe_mode is set — only head_quality trains.
    """
    model.train()
    running = {'total': 0.0, 'vad': 0.0, 'pitch': 0.0, 'technique': 0.0,
               'note': 0.0, 'contrastive': 0.0,
               'quality_mse': 0.0, 'quality_ccmusic': 0.0, 'quality_ranking': 0.0}
    n_batches = 0

    quality_loaders = quality_loaders or {}
    quality_iters   = {k: iter(v) for k, v in quality_loaders.items()}

    do_noise   = args.augment in ("noise", "noise_specaug") and noise_pool is not None
    do_specaug = args.augment == "noise_specaug"
    is_quality_only = args.quality_variant > 0  # backbone frozen, only head_quality trains

    # When training quality head, the main pitch/technique loader is skipped —
    # we iterate quality loaders directly. Use the 'pairs' loader to set epoch length.
    if is_quality_only:
        primary_iter = iter(quality_loaders.get('pairs', []))
        pbar = tqdm(quality_loaders.get('pairs', []),
                    desc=f"Epoch {epoch} [quality]", unit="batch")
    else:
        primary_iter = None
        pbar = tqdm(loader, desc=f"Epoch {epoch}", unit="batch")

    def _next_quality(key):
        """Draw next batch from a quality loader, cycling if exhausted."""
        try:
            return next(quality_iters[key])
        except StopIteration:
            quality_iters[key] = iter(quality_loaders[key])
            return next(quality_iters[key])

    if is_quality_only:
        # Quality-head-only training loop
        for batch_idx, (mel_pro, mel_am) in enumerate(pbar):
            quality_batch = {'pairs': (mel_pro, mel_am)}
            if 'mse' in quality_loaders:
                quality_batch['mse'] = _next_quality('mse')
            if 'ccmusic' in quality_loaders:
                quality_batch['ccmusic'] = _next_quality('ccmusic')

            total, q_losses = compute_quality_loss(model, quality_batch, device, args, epoch)

            optimizer.zero_grad()
            total.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            if args._sched_step == "iter":
                scheduler.step()

            step = global_step_offset + batch_idx
            writer.add_scalar("train/grad_norm", grad_norm, step)

            running['total'] += total.item()
            for k, v in q_losses.items():
                running[f'quality_{k}'] += v
            n_batches += 1

            pbar.set_postfix(
                loss=f"{running['total']/n_batches:.4f}",
                rank=f"{running['quality_ranking']/n_batches:.4f}",
                mse=f"{running['quality_mse']/n_batches:.4f}",
            )
    else:
        # Standard pitch/VAD/technique training loop
        use_contrastive = getattr(args, 'contrastive_technique', False) and bool(getattr(args, 'technique_dirs', None))
        use_note_head   = getattr(args, 'note_head', False)
        # Warmup: zero out technique (and quality) loss for first N epochs
        warmup_active = (getattr(args, 'warmup_heads_epochs', 0) > 0
                         and epoch <= args.warmup_heads_epochs)

        note_iter = iter(note_loader) if note_loader is not None else None

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

            if use_contrastive:
                pred_vad, pred_pitch, pred_technique, _, note_onset, note_offset, embeddings = model(
                    mel, return_embeddings=True)
            else:
                pred_vad, pred_pitch, pred_technique, _, note_onset, note_offset = model(mel)

            # Warmup: suppress technique loss until pitch/VAD/note have converged
            effective_w_technique = 0.0 if warmup_active else _get_w_technique(epoch, args)

            # Note targets — draw from note_loader if available, else zeros
            if use_note_head:
                if note_iter is not None:
                    try:
                        note_batch = next(note_iter)
                    except StopIteration:
                        note_iter = iter(note_loader)
                        note_batch = next(note_iter)
                    note_onset_tgt  = note_batch[0].to(device)  # (B, T)
                    note_offset_tgt = note_batch[1].to(device)  # (B, T)
                    has_note_batch  = note_batch[2].to(device)  # (B,)
                else:
                    B_n, T_n = mel.shape[0], mel.shape[1]
                    note_onset_tgt  = torch.zeros(B_n, T_n, device=device)
                    note_offset_tgt = torch.zeros(B_n, T_n, device=device)
                    has_note_batch  = torch.zeros(B_n, device=device)
            else:
                note_onset_tgt = note_offset_tgt = has_note_batch = None

            total, vad_l, pitch_l, tech_l, note_l = compute_loss(
                pred_vad, pred_pitch, pred_technique,
                vad, f0, technique, has_technique,
                args, device,
                w_technique_override=effective_w_technique,
                note_onset=note_onset, note_offset=note_offset,
                note_onset_target=note_onset_tgt,
                note_offset_target=note_offset_tgt,
                has_note=has_note_batch,
            )

            if use_contrastive and not warmup_active:
                con_l = supcon_loss(
                    embeddings, technique, has_technique,
                    temperature=args.contrastive_temp,
                )
                total = total + args.w_contrastive_technique * con_l
            else:
                con_l = torch.tensor(0.0)

            optimizer.zero_grad()
            total.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            if args._sched_step == "iter":
                scheduler.step()

            step = global_step_offset + batch_idx
            writer.add_scalar("train/grad_norm", grad_norm, step)

            running['total']       += total.item()
            running['vad']         += vad_l.item()
            running['pitch']       += pitch_l.item()
            running['technique']   += tech_l.item()
            running['note']        += note_l.item()
            running['contrastive'] += con_l.item()
            n_batches += 1

            pbar.set_postfix(
                loss=f"{running['total']/n_batches:.4f}",
                vad=f"{running['vad']/n_batches:.4f}",
                pitch=f"{running['pitch']/n_batches:.4f}",
                tech=f"{running['technique']/n_batches:.4f}",
                **({"note": f"{running['note']/n_batches:.4f}"} if use_note_head else {}),
                **({"con": f"{running['contrastive']/n_batches:.4f}"} if use_contrastive else {}),
            )

    if n_batches == 0:
        warnings.warn("No batches processed this epoch.", RuntimeWarning)
        return float("nan"), {k: float('nan') for k in running}

    avgs = {k: v / n_batches for k, v in running.items()}
    for k, v in avgs.items():
        if v > 0 or k == 'total':
            writer.add_scalar(f"train/{k}", v, epoch)
    writer.add_scalar("train/lr", optimizer.param_groups[-1]['lr'], epoch)
    if len(optimizer.param_groups) > 1:
        writer.add_scalar("train/lr_backbone", optimizer.param_groups[0]['lr'], epoch)
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
            v, p, _, _, _, _ = model(mel)
            pv = v.squeeze().cpu().numpy()
            pp = p.squeeze(0).cpu().numpy()
            T  = pv.shape[0]

            f0r = f0_all[i, :T].astype(np.float32)
            f0d = viterbi_decode(pp)

            vg = f0r > 0
            vp = f0d > 0
            vdr = float(np.mean(vp[vg])) if vg.sum() > 0 else float('nan')
            vad_acc = float(np.mean((vp > 0) == (vg > 0)))
            both = vg & vp
            if both.sum() > 0:
                ce = np.abs(1200 * np.log2(
                    f0d[both] / (f0r[both] + 1e-10) + 1e-10))
                rpa = float(np.mean(ce < 50))
            else:
                rpa = float('nan')
            clip_results.append({'snr': float(snrs[i]), 'vdr': vdr, 'rpa': rpa, 'vad_acc': vad_acc})

        by_snr = {}
        for r in clip_results:
            by_snr.setdefault(r['snr'], []).append(r)

        def smean(vals):
            v = [x for x in vals if not np.isnan(x)]
            return float(np.mean(v)) if v else float('nan')

        print(f"\n  {'Condition':<10}  {'VAD Acc':>8}  {'VDR':>8}  {'RPA':>8}")
        print(f"  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*8}")
        rpa_values = []
        for snr in sorted(by_snr.keys(), key=lambda x: x if np.isfinite(x) else 1e6):
            c    = by_snr[snr]
            tag  = "clean" if not np.isfinite(snr) else f"{snr:+.0f} dB"
            va   = smean([x['vad_acc'] for x in c])
            vd   = smean([x['vdr'] for x in c])
            rp   = smean([x['rpa'] for x in c])
            print(f"  {tag:<10}  {va:8.1%}  {vd:8.1%}  {rp:8.1%}")
            if not np.isnan(rp):
                rpa_values.append(rp)
            stag = tag.replace(' ', '').replace('+', 'p').replace('-', 'n')
            writer.add_scalar(f"eval/vad_acc_{stag}", va, epoch)
            writer.add_scalar(f"eval/vdr_{stag}", vd, epoch)
            writer.add_scalar(f"eval/rpa_{stag}", rp, epoch)
            if not np.isfinite(snr) and not np.isnan(vd):
                results['vdr_clean'] = vd
            if not np.isfinite(snr) and not np.isnan(va):
                results['vad_acc_clean'] = va

        macro_rpa = float(np.mean(rpa_values)) if rpa_values else float('nan')
        results['macro_rpa'] = macro_rpa
        writer.add_scalar("eval/macro_rpa", macro_rpa, epoch)
        print(f"  Macro RPA: {macro_rpa:.4f}")

    # ── Technique F1 evaluation ──────────────────────────────────────────
    # Collect test files from all technique dirs and merge them so that
    # multi-dataset runs (VocalSet + GTSinger) evaluate all classes together.
    tech_files = []
    for tdir in (technique_dirs or []):
        for _fname in ("technique_test.npz", "technique_gtsinger_test.npz", "note_test.npz"):
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
                _, _, pred_tech, _, _, _ = model(mel_t)
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

    # Validate quality variant dependencies
    if args.quality_variant > 0 and not args.probe_mode:
        raise RuntimeError("--quality-variant requires --probe-mode")
    if args.quality_variant > 0 and not args.resume:
        raise RuntimeError("--quality-variant requires --resume (trained pitch+technique checkpoint)")

    # Build model — quality_head dim: 0=off, 1=scalar, 9=multi-dim
    quality_head_dim = {0: 0, 1: 1, 2: 9, 3: 1}[args.quality_variant]
    model_kwargs = dict(causal=args.causal)
    if args.hidden   is not None: model_kwargs['hidden']   = args.hidden
    if args.n_blocks is not None: model_kwargs['n_blocks'] = args.n_blocks
    if args.n_layers is not None: model_kwargs['n_layers'] = args.n_layers
    if args.n_heads  is not None: model_kwargs['n_heads']  = args.n_heads
    if args.deep_technique_head:  model_kwargs['deep_technique_head'] = True
    if quality_head_dim:          model_kwargs['quality_head'] = quality_head_dim
    if args.note_head:            model_kwargs['note_head'] = True
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
    quality_loaders = {}
    if args.quality_variant > 0:
        quality_loaders = make_quality_loaders(args, args.seq_len, args.batch_size, args.num_workers)
        # Quality training doesn't need the main pitch/VAD/technique loader
        loader = None
        pretrain_loader = None
    else:
        # Dataset curriculum: build pretrain loader if --pretrain-data-dir is set
        pretrain_loader = None
        if args.pretrain_data_dir:
            pretrain_tech = ([os.path.abspath(d) for d in args.pretrain_technique_dirs]
                             if args.pretrain_technique_dirs else tech_dirs)
            pretrain_loader = make_joint_loader(
                os.path.abspath(args.pretrain_data_dir), pretrain_tech,
                args.seq_len, args.batch_size, args.num_workers,
                balance_datasets=args.balance_datasets)
            print(f"Dataset curriculum: pretrain on {args.pretrain_data_dir} "
                  f"for {args.pretrain_epochs} epochs, then switch to main dataset.")
        loader = make_joint_loader(data_dir, tech_dirs, args.seq_len,
                                   args.batch_size, args.num_workers,
                                   balance_datasets=args.balance_datasets)

    # Note loader — separate DataLoader for --note-dirs (Variant 4)
    note_loader = None
    if args.note_head and args.note_dirs:
        from torch.utils.data import ConcatDataset
        note_dirs_abs = [os.path.abspath(d) for d in args.note_dirs]
        note_datasets = []
        for nd in note_dirs_abs:
            note_path = os.path.join(nd, "note_train.npz")
            if os.path.exists(note_path):
                note_datasets.append(NoteDataset(nd, args.seq_len))
            else:
                print(f"  [warn] note_train.npz not found in {nd} — skipping")
        if note_datasets:
            note_ds = ConcatDataset(note_datasets) if len(note_datasets) > 1 else note_datasets[0]
            note_loader = DataLoader(note_ds, batch_size=args.batch_size, shuffle=True,
                                     drop_last=True, num_workers=args.num_workers,
                                     pin_memory=True,
                                     persistent_workers=(args.num_workers > 0))
            print(f"Note loader: {len(note_datasets)} source(s), "
                  f"{len(note_ds)} samples/epoch")

    # Optimizer + scheduler
    # Split into two param groups when --lr-backbone is set so the backbone
    # shifts slowly while heads learn at the full --lr rate.
    _HEAD_PREFIXES = ("head_vad", "head_pitch", "head_technique", "head_quality")
    if args.lr_backbone is not None and not args.probe_mode:
        backbone_params = [p for n, p in model.named_parameters()
                           if not any(n.startswith(h) for h in _HEAD_PREFIXES)]
        head_params     = [p for n, p in model.named_parameters()
                           if any(n.startswith(h) for h in _HEAD_PREFIXES)]
        optimizer = torch.optim.AdamW(
            [{"params": backbone_params, "lr": args.lr_backbone},
             {"params": head_params,     "lr": args.lr}],
            betas=(0.9, 0.98), weight_decay=1e-4)
        print(f"  Differential LR: backbone={args.lr_backbone:.2e}  heads={args.lr:.2e}  "
              f"({len(backbone_params)} backbone params, {len(head_params)} head params)")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      betas=(0.9, 0.98), weight_decay=1e-4)

    from torch.optim.lr_scheduler import (LinearLR, CosineAnnealingLR,
                                           SequentialLR, LambdaLR)
    if args.scheduler == "constant":
        scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        args._sched_step = "epoch"
    else:  # cosine_warmup
        ref_loader   = quality_loaders.get('pairs', loader)
        total_iters  = len(ref_loader) * args.epochs
        warmup_iters = len(ref_loader)
        warmup  = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_iters)
        cosine  = CosineAnnealingLR(optimizer,
                                    T_max=total_iters - warmup_iters,
                                    eta_min=args.lr * 0.01)
        scheduler = SequentialLR(optimizer, [warmup, cosine],
                                 milestones=[warmup_iters])
        args._sched_step = "iter"

    if resume_ckpt:
        _param_groups_changed = args.probe_mode or (args.lr_backbone is not None)
        if "optimizer" in resume_ckpt and not _param_groups_changed:
            optimizer.load_state_dict(resume_ckpt["optimizer"])
        elif "optimizer" in resume_ckpt and _param_groups_changed:
            reason = "probe mode" if args.probe_mode else "differential LR (param groups changed)"
            print(f"  Skipping optimizer state: {reason}")
        if "scheduler" in resume_ckpt and not _param_groups_changed:
            scheduler.load_state_dict(resume_ckpt["scheduler"])
        del resume_ckpt

    writer = SummaryWriter(log_dir=os.path.join(output_dir, "tb"))

    best_loss      = float("inf")
    best_metric    = 0.0    # macro F1 if technique data present, else macro RPA
    patience_count = 0
    global_step    = 0

    if args.quality_variant > 0:
        # Quality head probe: everything frozen except head_quality
        _set_backbone_frozen(model, frozen=True, quality_probe=True)
        print(f"  Quality probe (variant {args.quality_variant}): "
              f"only head_quality trains for all {args.epochs} epochs")
    elif args.probe_mode:
        _set_backbone_frozen(model, frozen=True, probe_mode=True)
        print(f"  Probe mode: backbone + pitch/VAD heads frozen for all {args.epochs} epochs")

    if args.probe_mode or args.quality_variant > 0:
        # Rebuild optimizer over trainable params only so frozen params get no momentum state
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, betas=(0.9, 0.98), weight_decay=1e-4,
        )

    freeze_until = start_epoch + args.freeze_backbone_epochs
    if not args.probe_mode and args.quality_variant == 0 and args.freeze_backbone_epochs > 0:
        _set_backbone_frozen(model, frozen=True)
        print(f"  Backbone frozen for epochs {start_epoch}–{freeze_until - 1}, "
              f"unfreezes at epoch {freeze_until}")

    ref_loader_len = len(quality_loaders.get('pairs', loader)) if quality_loaders else len(loader)

    # Warmup notification
    if getattr(args, 'warmup_heads_epochs', 0) > 0:
        print(f"  Head warmup: technique/quality losses suppressed for epochs "
              f"{start_epoch}–{start_epoch + args.warmup_heads_epochs - 1}")

    for epoch in range(start_epoch, start_epoch + args.epochs):
        if (args.quality_variant == 0 and not args.probe_mode
                and args.freeze_backbone_epochs > 0 and epoch == freeze_until):
            _set_backbone_frozen(model, frozen=False)
            print(f"  Epoch {epoch}: backbone unfrozen — joint fine-tuning begins")

        # Dataset curriculum: use pretrain_loader for first --pretrain-epochs epochs
        if (pretrain_loader is not None
                and (epoch - start_epoch) < args.pretrain_epochs):
            active_loader = pretrain_loader
            if (epoch - start_epoch) == 0:
                print(f"  Epoch {epoch}: curriculum pretrain phase "
                      f"({args.pretrain_epochs} epochs on {args.pretrain_data_dir})")
        else:
            active_loader = loader
            if (pretrain_loader is not None
                    and (epoch - start_epoch) == args.pretrain_epochs):
                print(f"  Epoch {epoch}: curriculum switches to main dataset")

        t0 = time.time()
        train_loss, train_losses = train_one_epoch(
            model, active_loader, optimizer, scheduler, writer,
            epoch, device, args, noise_pool=noise_pool,
            global_step_offset=global_step,
            quality_loaders=quality_loaders if args.quality_variant > 0 else None,
            note_loader=note_loader,
        )
        global_step += ref_loader_len

        if args._sched_step == "epoch":
            scheduler.step()

        dt = time.time() - t0
        if args.quality_variant > 0:
            print(f"  Epoch {epoch} — {dt:.1f}s — loss={train_loss:.5f}"
                  f"  (ranking={train_losses.get('quality_ranking', 0):.4f}"
                  f"  mse={train_losses.get('quality_mse', 0):.4f}"
                  f"  ccmusic={train_losses.get('quality_ccmusic', 0):.4f})")
        else:
            note_str = (f"  note={train_losses['note']:.4f}"
                        if getattr(args, 'note_head', False) else "")
            print(f"  Epoch {epoch} — {dt:.1f}s — loss={train_loss:.5f}"
                  f"  (vad={train_losses['vad']:.4f}"
                  f"  pitch={train_losses['pitch']:.4f}"
                  f"  tech={train_losses['technique']:.4f}"
                  f"{note_str})")

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

        # Evaluation — every eval_every epochs for both quality and standard paths
        if epoch % args.eval_every == 0 or epoch == start_epoch:
            if args.quality_variant > 0:
                # Quality has no held-out eval set; use neg ranking loss as the tracked metric
                metric = -train_losses.get('quality_ranking', train_loss)
                metric_name = "neg_ranking_loss"
                eval_res = {'neg_ranking_loss': metric}
            else:
                eval_res = evaluate(model, data_dir, tech_dirs,
                                    writer, epoch, device, args)
                if 'macro_f1' in eval_res:
                    vdr_clean = eval_res.get('vdr_clean', 1.0)
                    w = args.metric_vdr_weight
                    metric = eval_res['macro_f1'] + w * vdr_clean
                    metric_name = f"f1+{w}*vdr"
                elif 'macro_rpa' in eval_res:
                    # Use RPA + w*VDR_clean so the checkpoint reflects both pitch
                    # accuracy and voice detection — RPA alone barely varies and
                    # picks early epochs where VDR is still low.
                    vdr_clean = eval_res.get('vdr_clean', 1.0)
                    w = args.metric_vdr_weight
                    metric = eval_res['macro_rpa'] + w * vdr_clean
                    metric_name = f"rpa+{w}*vdr"
                else:
                    metric = float('nan')
                    metric_name = "none"
        else:
            eval_res = {}

        if eval_res and not np.isnan(metric):
            if metric > best_metric:
                best_metric = metric
                patience_count = 0
                torch.save(ckpt, os.path.join(ckpt_dir, "best_metric.pth"))
                print(f"  New best {metric_name}: {best_metric:.4f} — checkpoint saved.")
            else:
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
