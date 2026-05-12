"""
Annotated-VocalSet Feature Extraction
======================================

Extracts mel / F0 / VAD / technique labels AND per-note MIDI annotations from
the Annotated-VocalSet dataset.  Saves two NPZ files for training/evaluation:

  note_train.npz — training singers
  note_test.npz  — held-out test singers (default: m2, m4, f4, f8)

Annotation format
-----------------
Expects one annotation file per WAV, with tab- or comma-separated columns:

  onset_sec   offset_sec   midi_pitch   [lyric]

Annotation files are discovered by replacing the audio extension with
--ann-ext (default: .txt).  Any file with a header row containing "onset"
is auto-detected and the header skipped.

Example directory layout (two supported arrangements):

  Arrangement A — annotations alongside audio:
    VocalSet/data_by_singer/m1/vibrato/a_vibrato.wav
    VocalSet/data_by_singer/m1/vibrato/a_vibrato.txt

  Arrangement B — parallel annotation tree rooted at --ann-dir:
    VocalSet/data_by_singer/m1/vibrato/a_vibrato.wav
    annotations/m1/vibrato/a_vibrato.txt

Output NPZ schema
-----------------
  mel:          (total_frames, 40)      float16  — log-mel spectrogram
  f0:           (total_frames,)         float16  — Hz (0 = unvoiced)
  vad:          (total_frames,)         float16  — per-frame binary label
  technique:    (n_clips, N_TECH)       float32  — clip-level technique labels
  lengths:      (n_clips,)              int32    — frames per clip

  note_onsets:  (total_notes,)          int32    — onset frame index (absolute)
  note_offsets: (total_notes,)          int32    — offset frame index (absolute)
  note_midi:    (total_notes,)          uint8    — MIDI pitch 0-127
  note_clip:    (total_notes,)          int32    — which clip this note belongs to
  n_notes:      (n_clips,)              int32    — notes per clip (0 if unannotated)

Feature 8 — Note segmentation:
  Use note_onsets / note_offsets (already in frame units) to segment the
  predicted f0 track into notes.

Feature 9 — Per-note pitch accuracy:
  For each annotated note, compare the median predicted f0 within
  [onset, offset] frames against the MIDI pitch (converted to Hz).
  Accuracy = fraction of notes within 50 cents of ground truth.

Usage
-----
  python scripts/extractAnnotatedVocalSet.py \\
      --vocalset-dir /data/VocalSet \\
      --output-dir   data/annotated_vocalset

  # With parallel annotation tree
  python scripts/extractAnnotatedVocalSet.py \\
      --vocalset-dir /data/VocalSet \\
      --ann-dir      /data/AnnotatedVocalSet \\
      --output-dir   data/annotated_vocalset

  # Override test singers
  python scripts/extractAnnotatedVocalSet.py \\
      --vocalset-dir /data/VocalSet \\
      --output-dir   data/annotated_vocalset \\
      --test-singers m2 f4
"""

import argparse
import csv
import os
import sys

import librosa
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from vocalcoach.model import PITCH_BINS, PITCH_FMIN, PITCH_CENTS_PER_BIN, N_TECHNIQUES

try:
    import rmvpe as _rmvpe_mod
    HAS_RMVPE = True
except ImportError:
    HAS_RMVPE = False

# ── Constants ────────────────────────────────────────────────────────────────

SR          = 16_000
HOP_LENGTH  = 160        # 10 ms at 16 kHz
N_MELS      = 40
FMIN        = 50.0
FMAX        = 8_000.0
WIN_LENGTH  = 1024

DEFAULT_TEST_SINGERS = {"m2", "m4", "f4", "f8"}

# Technique folder → (TECHNIQUE_NAMES index, canonical name)
TECHNIQUE_MAP = {
    "vibrato":  0,
    "vibrado":  0,
    "breathy":  1,
    "belt":     3,
    "straight": 4,
    "strait":   4,
}

MIDI_A4 = 69
MIDI_HZ_A4 = 440.0


# ── Helpers ──────────────────────────────────────────────────────────────────

def midi_to_hz(midi):
    return MIDI_HZ_A4 * (2.0 ** ((midi - MIDI_A4) / 12.0))


def extract_mel(audio, sr):
    if sr != SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SR, n_fft=WIN_LENGTH, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=1.0)
    return librosa.power_to_db(mel, ref=1.0).T.astype(np.float32)  # (T, 40)


def extract_f0_vad(audio, sr, n_frames, rmvpe_model=None):
    if sr != SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    if rmvpe_model is not None:
        f0 = rmvpe_model.infer_from_audio(audio, thred=0.03)
        f0 = np.array(f0, dtype=np.float32)
        if len(f0) < n_frames:
            f0 = np.pad(f0, (0, n_frames - len(f0)))
        else:
            f0 = f0[:n_frames]
    else:
        f0_raw, _, _ = librosa.pyin(
            audio, fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=SR, hop_length=HOP_LENGTH, frame_length=WIN_LENGTH)
        f0_raw = np.nan_to_num(f0_raw, nan=0.0).astype(np.float32)
        if len(f0_raw) < n_frames:
            f0_raw = np.pad(f0_raw, (0, n_frames - len(f0_raw)))
        else:
            f0_raw = f0_raw[:n_frames]
        f0 = f0_raw

    vad = (f0 > 0).astype(np.float32)
    return f0, vad


def parse_annotations(ann_path):
    """Parse note annotation file → list of (onset_sec, offset_sec, midi_pitch).

    Accepts:
      - tab- or comma-separated
      - optional header row (auto-skipped if first field contains 'onset')
      - 3+ columns: onset  offset  midi_pitch  [lyric ...]
    """
    notes = []
    if not os.path.exists(ann_path):
        return notes
    with open(ann_path, newline='') as f:
        dialect = csv.Sniffer().sniff(f.read(2048), delimiters='\t,')
        f.seek(0)
        reader = csv.reader(f, dialect)
        for row in reader:
            if not row or len(row) < 3:
                continue
            if 'onset' in row[0].lower():  # header row
                continue
            try:
                onset  = float(row[0])
                offset = float(row[1])
                midi   = int(float(row[2]))
                if offset > onset and 0 <= midi <= 127:
                    notes.append((onset, offset, midi))
            except (ValueError, IndexError):
                continue
    return notes


def find_annotation(wav_path, ann_dir, ann_ext, vocalset_dir):
    """Find annotation file for a given WAV.

    Tries alongside the WAV first; falls back to ann_dir parallel tree.
    """
    base = os.path.splitext(wav_path)[0] + ann_ext
    if os.path.exists(base):
        return base
    if ann_dir:
        rel = os.path.relpath(wav_path, vocalset_dir)
        candidate = os.path.join(ann_dir, os.path.splitext(rel)[0] + ann_ext)
        if os.path.exists(candidate):
            return candidate
    return None


# ── Core extraction ──────────────────────────────────────────────────────────

def extract_split(wav_paths, technique_labels, output_path, rmvpe_model, min_frames):
    mel_list, f0_list, vad_list, tech_list, length_list = [], [], [], [], []
    note_onsets_list, note_offsets_list, note_midi_list, note_clip_list, n_notes_list = \
        [], [], [], [], []

    clip_idx = 0
    skipped = 0
    unannotated = 0

    for wav_path, tech_label, ann_path in tqdm(wav_paths, desc=f"  → {os.path.basename(output_path)}"):
        try:
            audio, sr = librosa.load(wav_path, sr=None, mono=True)
        except Exception as e:
            print(f"  [skip] {wav_path}: {e}")
            skipped += 1
            continue

        mel = extract_mel(audio, sr)
        n_frames = len(mel)
        if n_frames < min_frames:
            skipped += 1
            continue

        f0, vad = extract_f0_vad(audio, sr, n_frames, rmvpe_model)

        # ── Note annotations → frame indices ──────────────────────────────
        notes = parse_annotations(ann_path) if ann_path else []
        if not notes:
            unannotated += 1

        frame_offset = sum(length_list)
        for onset_sec, offset_sec, midi in notes:
            onset_fr  = int(round(onset_sec  * SR / HOP_LENGTH))
            offset_fr = int(round(offset_sec * SR / HOP_LENGTH))
            onset_fr  = max(0, min(onset_fr,  n_frames - 1))
            offset_fr = max(onset_fr + 1, min(offset_fr, n_frames))
            note_onsets_list.append(frame_offset + onset_fr)
            note_offsets_list.append(frame_offset + offset_fr)
            note_midi_list.append(midi)
            note_clip_list.append(clip_idx)

        mel_list.append(mel.astype(np.float16))
        f0_list.append(f0.astype(np.float16))
        vad_list.append(vad.astype(np.float16))
        tech_list.append(tech_label)
        length_list.append(n_frames)
        n_notes_list.append(len(notes))
        clip_idx += 1

    if not mel_list:
        print(f"  [warn] No clips extracted for {output_path}")
        return

    mel_flat = np.concatenate(mel_list, axis=0)
    f0_flat  = np.concatenate(f0_list,  axis=0)
    vad_flat = np.concatenate(vad_list,  axis=0)
    tech_arr = np.stack(tech_list).astype(np.float32)
    len_arr  = np.array(length_list, dtype=np.int32)

    note_on  = np.array(note_onsets_list,  dtype=np.int32)
    note_off = np.array(note_offsets_list, dtype=np.int32)
    note_mi  = np.array(note_midi_list,    dtype=np.uint8)
    note_cl  = np.array(note_clip_list,    dtype=np.int32)
    n_notes  = np.array(n_notes_list,      dtype=np.int32)

    np.savez_compressed(
        output_path,
        mel=mel_flat, f0=f0_flat, vad=vad_flat,
        technique=tech_arr, lengths=len_arr,
        note_onsets=note_on, note_offsets=note_off,
        note_midi=note_mi, note_clip=note_cl, n_notes=n_notes,
    )

    total_notes = len(note_on)
    annotated_clips = int(np.sum(n_notes > 0))
    print(f"  Saved {output_path}")
    print(f"    clips={clip_idx}, frames={len(mel_flat):,}, notes={total_notes:,} "
          f"({annotated_clips}/{clip_idx} clips annotated), skipped={skipped}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Extract Annotated-VocalSet features (mel, f0, vad, technique, notes)")
    p.add_argument("--vocalset-dir", required=True,
                   help="root of VocalSet dataset (contains data_by_singer/)")
    p.add_argument("--ann-dir",      default=None,
                   help="root of annotation tree (parallel to vocalset-dir). "
                        "If omitted, annotations are looked up alongside each WAV.")
    p.add_argument("--ann-ext",      default=".txt",
                   help="annotation file extension (default: .txt)")
    p.add_argument("--output-dir",   default="data/annotated_vocalset")
    p.add_argument("--test-singers", nargs="+", default=list(DEFAULT_TEST_SINGERS))
    p.add_argument("--min-dur",      type=float, default=0.5,
                   help="minimum clip duration in seconds (default: 0.5)")
    p.add_argument("--rmvpe",        default=None,
                   help="path to RMVPE checkpoint for F0 extraction "
                        "(falls back to librosa pyin if not provided)")
    p.add_argument("--device",       default="cuda")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    test_singers = set(args.test_singers)
    min_frames   = int(args.min_dur * SR / HOP_LENGTH)

    # ── Load RMVPE if available ───────────────────────────────────────────
    rmvpe_model = None
    if args.rmvpe and HAS_RMVPE:
        import torch
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "RMVPE"))
        from model import E2E0
        rmvpe_model = E2E0(4, 1, (2, 2))
        ckpt = torch.load(args.rmvpe, map_location="cpu")
        rmvpe_model.load_state_dict(ckpt)
        rmvpe_model.to(args.device).eval()
        print(f"Loaded RMVPE from {args.rmvpe}")
    else:
        print("RMVPE not available — using librosa pyin for F0 (slower)")

    # ── Walk VocalSet directory structure ─────────────────────────────────
    data_root = os.path.join(args.vocalset_dir, "data_by_singer")
    if not os.path.isdir(data_root):
        sys.exit(f"data_by_singer/ not found under {args.vocalset_dir}")

    train_items, test_items = [], []
    n_unannotated = 0

    for singer in sorted(os.listdir(data_root)):
        singer_dir = os.path.join(data_root, singer)
        if not os.path.isdir(singer_dir):
            continue
        split = test_items if singer in test_singers else train_items

        for technique_folder in sorted(os.listdir(singer_dir)):
            if technique_folder not in TECHNIQUE_MAP:
                continue
            tech_idx = TECHNIQUE_MAP[technique_folder]
            tech_label = np.zeros(N_TECHNIQUES, dtype=np.float32)
            tech_label[tech_idx] = 1.0

            tech_dir = os.path.join(singer_dir, technique_folder)
            for fname in sorted(os.listdir(tech_dir)):
                if not fname.lower().endswith(".wav"):
                    continue
                wav_path = os.path.join(tech_dir, fname)
                ann_path = find_annotation(wav_path, args.ann_dir,
                                           args.ann_ext, args.vocalset_dir)
                if ann_path is None:
                    n_unannotated += 1
                split.append((wav_path, tech_label, ann_path))

    print(f"Found {len(train_items)} train clips, {len(test_items)} test clips "
          f"({n_unannotated} without annotation files)")

    print("\nExtracting training split...")
    extract_split(
        train_items,
        [t for _, t, _ in train_items],
        os.path.join(args.output_dir, "note_train.npz"),
        rmvpe_model, min_frames,
    )

    print("\nExtracting test split...")
    extract_split(
        test_items,
        [t for _, t, _ in test_items],
        os.path.join(args.output_dir, "note_test.npz"),
        rmvpe_model, min_frames,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
