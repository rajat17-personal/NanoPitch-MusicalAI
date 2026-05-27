"""
Annotated-VocalSet Feature Extraction
======================================

Extracts mel / F0 / VAD / technique labels AND per-note onset/offset annotations
from the Annotated VocalSet dataset (Kim et al.).  Saves two NPZ files:

  note_train.npz — training singers
  note_test.npz  — held-out test singers (default: female4 female8 male2 male4)

Directory layout assumed
------------------------
  VocalSet audio:
    data/vocalset/data_by_singer/{female1..9,male1..11}/{exercise}/{technique}/*.wav

  Annotated VocalSet CSVs (the "raw" variant with per-frame F0 + onset markers):
    data/annotated_vocalset/raw 1/csv/{technique}/{stem}.csv

  CSV filename stem matches WAV stem exactly, e.g.:
    WAV:  data_by_singer/female1/arpeggios/belt/f1_arpeggios_belt_c_a.wav
    CSV:  raw 1/csv/belt/f1_arpeggios_belt_c_a.csv

Annotation CSV format (comma-separated, one header row)
---------------------------------------------------------
  Time (second), F0, Amplitude, Onset, Offset, Transition
  - Time:      frame timestamp in seconds (hop ≈ 11.61 ms, ~86 fps)
  - F0:        fundamental frequency in Hz (0 = unvoiced)
  - Onset:     "True" on note onset frames, blank otherwise
  - Offset:    "True" on note offset frames, blank otherwise

Note extraction: consecutive onset→offset pairs define notes.  F0 values
within the window are converted to MIDI pitch via median.

Output NPZ schema
-----------------
  mel:          (total_frames, 40)      float16  — log-mel spectrogram
  f0:           (total_frames,)         float16  — Hz (0 = unvoiced)
  vad:          (total_frames,)         float16  — per-frame binary (1 = voiced)
  technique:    (n_clips, N_TECH)       float32  — clip-level one-hot technique label
  lengths:      (n_clips,)              int32    — mel frames per clip

  note_onsets:  (total_notes,)          int32    — onset mel-frame (absolute)
  note_offsets: (total_notes,)          int32    — offset mel-frame (absolute)
  note_midi:    (total_notes,)          uint8    — median MIDI pitch of note
  note_clip:    (total_notes,)          int32    — clip index this note belongs to
  n_notes:      (n_clips,)              int32    — notes per clip (0 if unannotated)

Usage
-----
  python scripts/extractAnnotatedVocalSet.py \\
      --vocalset-dir data/vocalset \\
      --ann-dir      "data/annotated_vocalset/raw 1/csv" \\
      --output-dir   data/annotated_vocalset \\
      --rmvpe        rmvpe.pt \\
      --device       cuda
"""

import argparse
import csv
import os
import re
import sys

import librosa
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from vocalcoach.model import N_TECHNIQUES

try:
    from src.inference import RMVPE as _RMVPE_cls  # noqa: F401
    HAS_RMVPE = True
except ImportError:
    HAS_RMVPE = False

# ── Constants ────────────────────────────────────────────────────────────────

SR          = 16_000
HOP_LENGTH  = 160        # 10 ms at 16 kHz
N_MELS      = 40
FMIN        = 31.7       # Hz — must match extractFeatures.py and model.py
FMAX        = 8_000.0
WIN_LENGTH  = 400        # 25 ms — must match extractFeatures.py

# Annotated VocalSet annotation hop: ~11.61 ms (SR=44100, hop=512 → 11.61ms)
ANN_HOP_SEC = 512 / 44100

MIDI_A4    = 69
MIDI_HZ_A4 = 440.0

# Techniques extracted from VocalSet (index must match TECHNIQUE_NAMES in model.py)
TECHNIQUE_MAP = {
    "vibrato":  0,
    "breathy":  1,
    "belt":     3,
    "straight": 4,
}

# Singer folder name → short prefix used in filenames
# female1→f1, female2→f2, ..., male1→m1, male10→m10, male11→m11
def _singer_prefix(folder_name: str) -> str:
    m = re.fullmatch(r'(female|male)(\d+)', folder_name)
    if not m:
        return None
    letter = 'f' if m.group(1) == 'female' else 'm'
    return f"{letter}{m.group(2)}"

# Default held-out singers (folder names)
DEFAULT_TEST_SINGERS = {"female4", "female8", "male2", "male4"}


# ── Helpers ──────────────────────────────────────────────────────────────────

def hz_to_midi(hz: float) -> int:
    if hz <= 0:
        return 0
    return int(round(12 * np.log2(hz / MIDI_HZ_A4) + MIDI_A4))


def extract_mel(audio: np.ndarray, sr: int) -> np.ndarray:
    if sr != SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SR, n_fft=WIN_LENGTH, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=1.0)
    return librosa.power_to_db(mel, ref=1.0).T.astype(np.float32)  # (T, 40)


def extract_f0_vad(audio: np.ndarray, sr: int, n_frames: int,
                   rmvpe_model=None) -> tuple:
    if sr != SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    if rmvpe_model is not None:
        f0 = np.array(
            rmvpe_model.infer_from_audio(audio, sample_rate=SR, thred=0.03),
            dtype=np.float32)
    else:
        f0_raw, _, _ = librosa.pyin(
            audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
            sr=SR, hop_length=HOP_LENGTH, frame_length=WIN_LENGTH)
        f0 = np.nan_to_num(f0_raw, nan=0.0).astype(np.float32)

    if len(f0) < n_frames:
        f0 = np.pad(f0, (0, n_frames - len(f0)))
    else:
        f0 = f0[:n_frames]
    vad = (f0 > 0).astype(np.float32)
    return f0, vad


def parse_annotations(csv_path: str, n_mel_frames: int) -> list:
    """Parse Annotated VocalSet CSV → list of (onset_mel_frame, offset_mel_frame, midi).

    The CSV uses onset/offset markers on individual frames (~86fps).  We convert
    annotation frame timestamps to mel frame indices (10ms hop) and pair
    consecutive onset→offset rows to form notes.
    """
    if not csv_path or not os.path.exists(csv_path):
        return []

    ann_frames = []  # list of (time_sec, f0_hz, is_onset, is_offset)
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader, None)  # skip header row
        for row in reader:
            if len(row) < 4:
                continue
            try:
                t   = float(row[0])
                f0  = float(row[1])
                ons = row[3].strip().lower() == 'true'
                off = row[4].strip().lower() == 'true' if len(row) > 4 else False
                ann_frames.append((t, f0, ons, off))
            except (ValueError, IndexError):
                continue

    if not ann_frames:
        return []

    # Pair onset→offset markers to form note segments
    notes = []
    pending_onset = None   # (time_sec, f0_buffer)
    f0_buf = []

    for t, f0, is_onset, is_offset in ann_frames:
        if is_onset:
            if pending_onset is not None and f0_buf:
                # previous note closed by new onset (treat as implicit offset)
                onset_t, _ = pending_onset
                off_t = t
                voiced = [h for h in f0_buf if h > 0]
                midi = hz_to_midi(float(np.median(voiced))) if voiced else 0
                if midi > 0:
                    onset_fr  = int(round(onset_t / (HOP_LENGTH / SR)))
                    offset_fr = int(round(off_t   / (HOP_LENGTH / SR)))
                    onset_fr  = max(0, min(onset_fr, n_mel_frames - 1))
                    offset_fr = max(onset_fr + 1, min(offset_fr, n_mel_frames))
                    notes.append((onset_fr, offset_fr, midi))
            pending_onset = (t, [])
            f0_buf = [f0] if f0 > 0 else []

        elif pending_onset is not None:
            if f0 > 0:
                f0_buf.append(f0)

            if is_offset:
                onset_t, _ = pending_onset
                voiced = [h for h in f0_buf if h > 0]
                midi = hz_to_midi(float(np.median(voiced))) if voiced else 0
                if midi > 0:
                    onset_fr  = int(round(onset_t / (HOP_LENGTH / SR)))
                    offset_fr = int(round(t       / (HOP_LENGTH / SR)))
                    onset_fr  = max(0, min(onset_fr, n_mel_frames - 1))
                    offset_fr = max(onset_fr + 1, min(offset_fr, n_mel_frames))
                    notes.append((onset_fr, offset_fr, midi))
                pending_onset = None
                f0_buf = []

    return notes


# ── Core extraction ──────────────────────────────────────────────────────────

def extract_split(items, output_path, rmvpe_model, min_frames):
    """items: list of (wav_path, tech_label_vec, csv_path_or_None)"""
    mel_list, f0_list, vad_list, tech_list, length_list = [], [], [], [], []
    note_onsets_list, note_offsets_list = [], []
    note_midi_list, note_clip_list, n_notes_list = [], [], []

    clip_idx = 0
    skipped = 0
    unannotated = 0

    for wav_path, tech_label, csv_path in tqdm(items,
                                                desc=f"  → {os.path.basename(output_path)}"):
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

        frame_offset = sum(length_list)
        notes = parse_annotations(csv_path, n_frames)
        if not notes:
            unannotated += 1

        for onset_fr, offset_fr, midi in notes:
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
        print(f"  [warn] No clips extracted → {output_path}")
        return

    np.savez_compressed(
        output_path,
        mel=np.concatenate(mel_list),
        f0=np.concatenate(f0_list),
        vad=np.concatenate(vad_list),
        technique=np.stack(tech_list).astype(np.float32),
        lengths=np.array(length_list, dtype=np.int32),
        note_onsets=np.array(note_onsets_list,  dtype=np.int32),
        note_offsets=np.array(note_offsets_list, dtype=np.int32),
        note_midi=np.array(note_midi_list,       dtype=np.uint8),
        note_clip=np.array(note_clip_list,       dtype=np.int32),
        n_notes=np.array(n_notes_list,           dtype=np.int32),
    )
    annotated = int(np.sum(np.array(n_notes_list) > 0))
    print(f"  Saved {output_path}")
    print(f"    clips={clip_idx}, frames={sum(length_list):,}, "
          f"notes={len(note_onsets_list):,} ({annotated}/{clip_idx} clips annotated), "
          f"skipped={skipped}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Extract Annotated-VocalSet features (mel, f0, vad, technique, notes)")
    p.add_argument("--vocalset-dir", required=True,
                   help="VocalSet root (contains data_by_singer/)")
    p.add_argument("--ann-dir", required=True,
                   help="Path to the 'raw 1/csv' annotation directory, e.g. "
                        "\"data/annotated_vocalset/raw 1/csv\"")
    p.add_argument("--output-dir",   default="data/annotated_vocalset")
    p.add_argument("--test-singers", nargs="+",
                   default=sorted(DEFAULT_TEST_SINGERS),
                   help="Singer folder names to hold out (default: female4 female8 male2 male4)")
    p.add_argument("--min-dur",  type=float, default=0.5,
                   help="Minimum clip duration in seconds (default: 0.5)")
    p.add_argument("--rmvpe",    default=None,
                   help="Path to RMVPE checkpoint (.pt). Falls back to librosa pyin if omitted.")
    p.add_argument("--device",   default="cuda")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    test_singers = set(args.test_singers)
    min_frames   = int(args.min_dur * SR / HOP_LENGTH)

    # ── Load RMVPE if provided ────────────────────────────────────────────
    rmvpe_model = None
    if args.rmvpe:
        if not HAS_RMVPE:
            print("[warn] RMVPE not importable — falling back to librosa pyin")
        else:
            from src.inference import RMVPE
            rmvpe_model = RMVPE(args.rmvpe)
            rmvpe_model.model = rmvpe_model.model.to(args.device)
            print(f"Loaded RMVPE from {args.rmvpe} on {args.device}")
    else:
        print("No --rmvpe provided — using librosa pyin for F0 (slower, less accurate)")

    # ── Walk VocalSet data_by_singer ──────────────────────────────────────
    data_root = os.path.join(args.vocalset_dir, "data_by_singer")
    if not os.path.isdir(data_root):
        sys.exit(f"data_by_singer/ not found under {args.vocalset_dir}")
    if not os.path.isdir(args.ann_dir):
        sys.exit(f"Annotation dir not found: {args.ann_dir}")

    train_items, test_items = [], []
    missing_ann = 0

    for singer_folder in sorted(os.listdir(data_root)):
        singer_dir = os.path.join(data_root, singer_folder)
        if not os.path.isdir(singer_dir):
            continue
        split = test_items if singer_folder in test_singers else train_items

        for exercise in sorted(os.listdir(singer_dir)):
            exercise_dir = os.path.join(singer_dir, exercise)
            if not os.path.isdir(exercise_dir):
                continue

            for technique in sorted(os.listdir(exercise_dir)):
                if technique not in TECHNIQUE_MAP:
                    continue
                tech_idx = TECHNIQUE_MAP[technique]
                tech_label = np.zeros(N_TECHNIQUES, dtype=np.float32)
                tech_label[tech_idx] = 1.0

                tech_dir = os.path.join(exercise_dir, technique)
                for fname in sorted(os.listdir(tech_dir)):
                    if not fname.lower().endswith(".wav"):
                        continue
                    wav_path = os.path.join(tech_dir, fname)
                    stem     = os.path.splitext(fname)[0]
                    csv_path = os.path.join(args.ann_dir, technique, f"{stem}.csv")
                    if not os.path.exists(csv_path):
                        csv_path = None
                        missing_ann += 1
                    split.append((wav_path, tech_label, csv_path))

    print(f"Found {len(train_items)} train clips, {len(test_items)} test clips "
          f"({missing_ann} without annotation CSV)")

    print("\nExtracting training split...")
    extract_split(train_items, os.path.join(args.output_dir, "note_train.npz"),
                  rmvpe_model, min_frames)

    print("\nExtracting test split...")
    extract_split(test_items, os.path.join(args.output_dir, "note_test.npz"),
                  rmvpe_model, min_frames)

    print("\nDone.")


if __name__ == "__main__":
    main()
