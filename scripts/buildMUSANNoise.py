"""
Build noise.npz from MUSAN (or any flat audio directory)
=========================================================

Extracts log-mel spectrograms from noise/music/speech audio and saves them in
the noise.npz format expected by vocalcoach/train.py:

    mel:     (total_frames, 40)  float16 — log-mel, dB re 1.0
    lengths: (n_clips,)          int32   — frame count per clip

This is intentionally simpler than extractFeatures.py — no F0 or VAD is
needed because noise clips are used only for augmentation mixing.

All three MUSAN splits are useful and recommended:
  music/  (42.6h) — pitched harmonic interferer; most important for Vocadito-
                    style recordings where singers perform over backing music
  noise/  ( 6.2h) — environmental/room sounds (free-sound, sound-bible)
  speech/ (60.4h) — LibriVox + US-Gov speech; teaches VAD to fire on singing
                    not just any harmonic voiced source

Mel parameters match api.py / train.py exactly:
    SR=16000, hop=160 (10 ms), n_fft=512, win=400, n_mels=40, fmin=50, fmax=8000
    power_to_db(ref=1.0)  ← fixed ref keeps inter-clip dynamics intact

Usage
-----
    # All three MUSAN splits — recommended (109 h)
    python scripts/buildMUSANNoise.py \\
        --input-dir  data/musan \\
        --output     data/musan_noise.npz

    # Music only (42.6 h) — quickest subset to test
    python scripts/buildMUSANNoise.py \\
        --input-dir  data/musan/music \\
        --output     data/musan_music_noise.npz

    # Merge with existing FSDNoisy18k noise.npz
    python scripts/buildMUSANNoise.py \\
        --input-dir  data/musan/music \\
        --output     data/musan_noise.npz \\
        --merge      data/noise.npz \\
        --merge-out  data/noise_musan_merged.npz
"""

import argparse
import os

import librosa
import numpy as np
from tqdm import tqdm

# ── Constants — must match vocalcoach/api.py / train.py / features.py ──
SR         = 16000
HOP_LENGTH = 160     # 10 ms
WIN_LENGTH = 400     # 25 ms
N_FFT      = 512
N_MELS     = 40
FMIN       = 50.0    # matches api.py (not 31.7 — noise doesn't need full pitch range)
FMAX       = 8000.0

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def extract_mel(y: np.ndarray) -> np.ndarray:
    """(T, 40) log-mel, dB re 1.0 power unit — fixed reference preserves dynamics."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH, n_mels=N_MELS,
        fmin=FMIN, fmax=FMAX, window="hann", center=True,
    )
    return librosa.power_to_db(mel, ref=1.0).T.astype(np.float32)  # (T, 40)


def collect_files(input_dir: str) -> list[str]:
    files = []
    for root, _, fnames in os.walk(input_dir):
        for f in sorted(fnames):
            if os.path.splitext(f)[1].lower() in AUDIO_EXTS:
                files.append(os.path.join(root, f))
    return sorted(files)


def build_noise_npz(input_dir: str, min_dur_s: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Extract mel from all audio under input_dir. Returns (mel_all, lengths)."""
    files = collect_files(input_dir)
    if not files:
        raise SystemExit(f"No audio files found in {input_dir}")
    print(f"Found {len(files)} audio files in {input_dir}")

    min_frames = int(min_dur_s * SR / HOP_LENGTH)
    mel_chunks, lengths = [], []
    skipped = 0

    for path in tqdm(files, desc="Extracting mel"):
        try:
            y, _ = librosa.load(path, sr=SR, mono=True)
        except Exception as e:
            tqdm.write(f"  [skip] {os.path.basename(path)}: {e}")
            skipped += 1
            continue

        if len(y) < min_frames * HOP_LENGTH:
            skipped += 1
            continue

        mel = extract_mel(y)   # (T, 40)
        mel_chunks.append(mel)
        lengths.append(len(mel))

    if not mel_chunks:
        raise SystemExit("No valid clips extracted.")

    print(f"  Processed: {len(lengths)}, skipped: {skipped}")
    mel_all = np.concatenate(mel_chunks, axis=0)
    len_arr = np.array(lengths, dtype=np.int32)
    return mel_all, len_arr


def parse_args():
    p = argparse.ArgumentParser(description="Build noise.npz from MUSAN or any audio dir")
    p.add_argument("--input-dir", required=True,
                   help="directory of audio files to extract (e.g. data/musan/music)")
    p.add_argument("--output", required=True,
                   help="path for output noise NPZ (e.g. data/musan_noise.npz)")
    p.add_argument("--min-duration", type=float, default=0.5,
                   help="skip clips shorter than this many seconds (default: 0.5)")
    p.add_argument("--merge", default=None,
                   help="path to an existing noise.npz to concatenate with the new data")
    p.add_argument("--merge-out", default=None,
                   help="output path for merged NPZ (required if --merge is set)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.merge and not args.merge_out:
        raise SystemExit("--merge-out is required when --merge is set")

    mel_new, len_new = build_noise_npz(args.input_dir, args.min_duration)

    # ── Save standalone output ──────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    np.savez(args.output,
             mel=mel_new.astype(np.float16),
             lengths=len_new)

    total_h = mel_new.shape[0] * HOP_LENGTH / SR / 3600
    print(f"\nSaved {args.output}")
    print(f"  Clips  : {len(len_new)}")
    print(f"  Frames : {mel_new.shape[0]:,}  ({total_h:.2f} h)")
    print(f"  mel    : {mel_new.shape}  → float16")

    # ── Optional merge with existing noise.npz ─────────────────────────
    if args.merge:
        print(f"\nMerging with {args.merge} ...")
        existing = np.load(args.merge)
        mel_ex  = existing["mel"].astype(np.float32)
        len_ex  = existing["lengths"].astype(np.int32)

        mel_merged = np.concatenate([mel_ex, mel_new], axis=0)
        len_merged = np.concatenate([len_ex, len_new], axis=0)

        np.savez(args.merge_out,
                 mel=mel_merged.astype(np.float16),
                 lengths=len_merged)

        total_h_merged = mel_merged.shape[0] * HOP_LENGTH / SR / 3600
        print(f"Saved merged {args.merge_out}")
        print(f"  Clips  : {len(len_merged)} ({len(len_ex)} existing + {len(len_new)} new)")
        print(f"  Frames : {mel_merged.shape[0]:,}  ({total_h_merged:.2f} h)")


if __name__ == "__main__":
    main()
