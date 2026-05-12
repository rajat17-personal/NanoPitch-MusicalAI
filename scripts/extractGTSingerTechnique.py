"""
GTSinger Technique Label Extraction
=====================================

Downloads technique-labeled clips from the GTSinger HuggingFace dataset,
extracts mel / F0 / VAD, and saves them in the flat NPZ format used by
vocalcoach/train.py's TechniqueDataset.

Why a separate script from extractFeatures.py
----------------------------------------------
The pre-extracted clean.npz (from smulelabs/NanoPitch-PreExtract) has no
technique labels — it was created for pitch/VAD training only. This script
re-processes GTSinger clips that carry technique annotations and outputs a
SEPARATE technique_gtsinger_*.npz. The two NPZ files serve different Dataset
classes in train.py:

  clean.npz                → PitchVADDataset  (pitch + VAD supervision)
  technique_gtsinger_*.npz → TechniqueDataset (technique head supervision)

GTSinger technique → our label mapping
---------------------------------------
  vibrato    → 0  (vibrato)
  breathy    → 1  (breathy)
  falsetto   → 2  (falsetto)  ← unique to GTSinger; VocalSet has no falsetto
  glissando  → skip  (pitch slide ornament, not in our 5-class taxonomy)
  mixed_voice→ skip
  pharyngeal → skip

GTSinger dataset on HuggingFace (AaronZ345/GTSinger)
-----------------------------------------------------
The dataset has per-clip technique annotations.  Each example has:
  audio        — waveform dict with 'array' and 'sampling_rate'
  technique    — one of the strings above  (field may also be 'singing_method'
                 or 'style' depending on the dataset version; the script tries
                 common names in order)
  split        — 'train' / 'test' (used to produce two output files)

If the HuggingFace dataset schema changes, set --technique-field to the
correct field name and --train-split / --test-split as needed.

Output NPZ schema (same as extractVocalSet.py):
  mel:       (total_frames, 40)         float16
  f0:        (total_frames,)            float16
  vad:       (total_frames,)            float16
  technique: (n_clips, N_TECHNIQUES)   float32  — clip-level binary labels
  lengths:   (n_clips,)                int32
  ids:       (n_clips,)                object   — clip identifier for debugging

Usage
-----
  # Stream from HuggingFace (no download required, slow on large datasets):
  python scripts/extractGTSingerTechnique.py \\
      --output-dir data/gtsinger_technique \\
      --rmvpe-model rmvpe.pt \\
      --device cuda

  # From a local clone of the HuggingFace dataset:
  python scripts/extractGTSingerTechnique.py \\
      --local-dir /path/to/GTSinger \\
      --output-dir data/gtsinger_technique \\
      --rmvpe-model rmvpe.pt \\
      --device cuda

  # Limit clips per technique (useful for quick tests):
  python scripts/extractGTSingerTechnique.py \\
      --output-dir data/gtsinger_technique \\
      --max-per-technique 200
"""

import argparse
import os
import sys

import librosa
import numpy as np
from tqdm import tqdm

# ── Constants — must match vocalcoach/model.py ────────────────────────
SR          = 16000
N_MELS      = 40
HOP_LENGTH  = 160
WIN_LENGTH  = 400
N_FFT       = 512
FMIN        = 31.7
FMAX        = 8000.0
VAD_TOP_DB  = 30

TECHNIQUE_NAMES = ['vibrato', 'breathy', 'falsetto', 'belt', 'straight']

# AaronZ345/GTSinger provides per-note binary technique indicator fields:
#   vibrato_tech, breathy_tech, falsetto_tech (list<int64>, 1 = technique active)
# belt and straight are absent — those come from VocalSet.
# singing_method is the music genre (pop/classical/…), not a vocal technique.

HF_REPO_ID = "AaronZ345/GTSinger"


def _gtsinger_features():
    """Explicit Features schema for AaronZ345/GTSinger.

    Derived from the CastError traceback (2026-05-10). The dataset card schema
    does not match the actual JSON fields, so we must pass this explicitly.
    audio=decode=False keeps the raw {bytes, path} struct for manual decoding.
    """
    try:
        from datasets import Features, Audio, Value, Sequence
        return Features({
            "item_name":       Value("string"),
            "txt":             Sequence(Value("string")),
            "ph":              Sequence(Value("string")),
            "ph_durs":         Sequence(Value("float64")),
            "word_durs":       Sequence(Value("float64")),
            "ep_pitches":      Sequence(Value("int64")),
            "ep_notedurs":     Sequence(Value("float64")),
            "ep_types":        Sequence(Value("int64")),
            "ph2words":        Sequence(Value("int64")),
            "mix_tech":        Sequence(Value("int64")),
            "falsetto_tech":   Sequence(Value("int64")),
            "breathy_tech":    Sequence(Value("int64")),
            "pharyngeal_tech": Sequence(Value("int64")),
            "vibrato_tech":    Sequence(Value("int64")),
            "glissando_tech":  Sequence(Value("int64")),
            "tech":            Sequence(Value("string")),
            "wav_fn":          Value("string"),
            "language":        Value("string"),
            "singer":          Value("string"),
            "speech_fn":       Value("string"),
            "emotion":         Value("string"),
            "singing_method":  Value("string"),
            "pace":            Value("string"),
            "range":           Value("string"),
            "note_start":      Sequence(Value("float64")),
            "ph_start":        Sequence(Value("float64")),
            "ph_end":          Sequence(Value("float64")),
            "mix":             Sequence(Value("string")),
            "note_end":        Sequence(Value("float64")),
            "label":           Value("int64"),
            "word":            Value("string"),
            "glissando":       Sequence(Value("string")),
            "pharyngeal":      Sequence(Value("string")),
            "note":            Sequence(Value("int64")),
            "breathy":         Sequence(Value("string")),
            "note_dur":        Sequence(Value("float64")),
            "falsetto":        Sequence(Value("string")),
            "vibrato":         Sequence(Value("string")),
            "audio":           Audio(decode=False),
            "start_time":      Value("float64"),
            "end_time":        Value("float64"),
        })
    except ImportError:
        return None


def _patch_tech_features(repo_id, token):
    """Return the dataset's own declared Features with 'tech' patched to string.

    The dataset card wrongly declares tech as ClassLabel (int64) but the actual
    parquet/JSON data stores it as a comma-separated string (e.g. "0", "2,6").
    Keeping all other fields — especially Audio() — at their declared defaults
    preserves the proper streaming audio loading pipeline.  Falls back to the
    fully explicit schema if the builder info cannot be fetched.
    """
    try:
        from datasets import load_dataset_builder, Value, Features
        builder = load_dataset_builder(repo_id, token=token)
        if builder.info.features is not None:
            patched = Features({**builder.info.features, "tech": Value("string")})
            print("  Patched 'tech' field to string; all other features from dataset card.")
            return patched
    except Exception as e:
        print(f"  Warning: could not read dataset builder info ({e}); using explicit schema.")
    return _gtsinger_features()


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract GTSinger technique labels into flat NPZ format.")
    p.add_argument("--output-dir", default="data/gtsinger_technique",
                   help="directory for output NPZ files")
    p.add_argument("--local-dir", default=None,
                   help="path to a locally downloaded GTSinger dataset "
                        "(skips HuggingFace download)")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--rmvpe-model", default="rmvpe.pt")
    p.add_argument("--train-split", default="train",
                   help="HF dataset split name for training data (default: train)")
    p.add_argument("--test-split", default="test",
                   help="HF dataset split name for test data (default: test)")
    p.add_argument("--no-streaming", action="store_true",
                   help="load from HF cache instead of streaming (faster if already "
                        "downloaded; Arrow schema avoids CastErrors)")
    p.add_argument("--hf-name", default=None,
                   help="HuggingFace dataset config/name (e.g. 'meta' if the cache "
                        "path contains 'AaronZ345___gt_singer/meta/')")
    p.add_argument("--audio-dir", default=None,
                   help="local directory containing downloaded GTSinger WAV files "
                        "(e.g. data/gtsinger_audio); wav_fn paths are resolved "
                        "relative to this dir before falling back to hf_hub_download")
    p.add_argument("--max-per-technique", type=int, default=None,
                   help="max clips per technique per split (None = all)")
    p.add_argument("--min-duration", type=float, default=0.5)
    return p.parse_args()


# ── Audio helpers ─────────────────────────────────────────────────────

def resample_to_16k(audio_array, orig_sr):
    if orig_sr == SR:
        return audio_array.astype(np.float32)
    return librosa.resample(audio_array.astype(np.float32),
                            orig_sr=orig_sr, target_sr=SR)


def extract_mel(y):
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH, n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
        window="hann", center=True,
    )
    return librosa.power_to_db(mel, ref=1.0).T.astype(np.float32)  # (T, 40)


def extract_vad(y, n_frames):
    frame_vad = np.zeros(n_frames, dtype=np.float32)
    for s, e in librosa.effects.split(y, top_db=VAD_TOP_DB):
        sf = s // HOP_LENGTH
        ef = min(e // HOP_LENGTH, n_frames)
        if sf < ef:
            frame_vad[sf:ef] = 1.0
    return frame_vad


# ── Dataset loading ───────────────────────────────────────────────────

def load_gtsinger(local_dir, train_split, test_split, streaming, hf_name):
    """Load GTSinger from HuggingFace (cached or streaming) or a local directory.

    Returns (train_dataset, test_dataset) — iterable HF Dataset objects.
    """
    try:
        from datasets import load_dataset, load_from_disk
    except ImportError:
        raise SystemExit(
            "HuggingFace 'datasets' library not installed.\n"
            "  pip install datasets")

    token = os.environ.get("HF_TOKEN") or True

    if local_dir:
        print(f"Loading GTSinger from local path: {local_dir}")
        ds = load_from_disk(local_dir)
        train_ds = ds[train_split] if train_split in ds else ds
        test_ds  = ds[test_split]  if test_split  in ds else None
    elif streaming:
        print(f"Streaming GTSinger from HuggingFace ({HF_REPO_ID}) ...")
        print("  Tip: download first for faster repeated runs (--no-streaming).")
        features = _gtsinger_features()  # explicit schema to fix CastError
        kwargs = dict(split=train_split, streaming=True, token=token, features=features)
        if hf_name:
            kwargs["name"] = hf_name
        train_ds = load_dataset(HF_REPO_ID, **kwargs)
        try:
            kwargs["split"] = test_split
            test_ds = load_dataset(HF_REPO_ID, **kwargs)
        except Exception:
            test_ds = None
            print(f"  Note: split '{test_split}' not found — train only.")
    else:
        # Non-streaming: reads from HF cache (Arrow file already has correct schema).
        print(f"Loading GTSinger from HuggingFace cache ({HF_REPO_ID}) ...")
        kwargs = dict(token=token)
        if hf_name:
            kwargs["name"] = hf_name
            print(f"  Using config/name: {hf_name}")
        train_ds = load_dataset(HF_REPO_ID, split=train_split, **kwargs)
        try:
            test_ds = load_dataset(HF_REPO_ID, split=test_split, **kwargs)
        except Exception:
            test_ds = None
            print(f"  Note: split '{test_split}' not found — train only.")

    return train_ds, test_ds


# ── Per-split extraction ──────────────────────────────────────────────

def extract_split(dataset, rmvpe, device, min_frames, max_per_tech, split_name,
                  audio_dir=None):
    """Iterate through a GTSinger dataset split and extract features.

    Returns (mel_chunks, f0_chunks, vad_chunks, technique_labels,
             lengths, clip_ids, skipped_count)
    """
    mel_chunks, f0_chunks, vad_chunks = [], [], []
    technique_labels, lengths, clip_ids = [], [], []
    counts = {i: 0 for i in range(len(TECHNIQUE_NAMES))}
    skip_reasons = {"tech_unmapped": 0, "audio_bad": 0,
                    "audio_short": 0, "frame_short": 0}

    for idx, example in enumerate(tqdm(dataset, desc=f"  {split_name}")):
        # Derive clip-level label from per-note binary technique fields.
        # Each *_tech field is a list<int64> where 1 = technique active for that note.
        label = np.zeros(len(TECHNIQUE_NAMES), dtype=np.float32)
        if any(example.get('vibrato_tech') or []):   label[0] = 1.0  # vibrato
        if any(example.get('breathy_tech') or []):   label[1] = 1.0  # breathy
        if any(example.get('falsetto_tech') or []):  label[2] = 1.0  # falsetto
        # label[3]=belt, label[4]=straight absent in GTSinger — supplied by VocalSet

        if label.sum() == 0:
            skip_reasons["tech_unmapped"] += 1
            continue

        mapped = [i for i in range(len(TECHNIQUE_NAMES)) if label[i] > 0]
        primary = mapped[0]
        if max_per_tech and counts[primary] >= max_per_tech:
            continue

        # Audio: the JSON has audio=null; actual WAV files are in HF repo LFS.
        # Use wav_fn to download from HF Hub (cached after first run).
        audio_arr, orig_sr = None, SR
        audio_data = example.get("audio")
        if isinstance(audio_data, dict):
            raw_bytes = audio_data.get("bytes")
            raw_path  = audio_data.get("path")
            try:
                import io
                if raw_bytes:
                    audio_arr, orig_sr = librosa.load(io.BytesIO(raw_bytes), sr=None, mono=True)
                elif raw_path:
                    audio_arr, orig_sr = librosa.load(raw_path, sr=None, mono=True)
            except Exception:
                pass

        if audio_arr is None:
            wav_fn = example.get("wav_fn")
            if wav_fn:
                local_path = os.path.join(audio_dir, wav_fn) if audio_dir else None
                try:
                    if local_path and os.path.exists(local_path):
                        audio_arr, orig_sr = librosa.load(local_path, sr=None, mono=True)
                    else:
                        from huggingface_hub import hf_hub_download
                        hf_token = os.environ.get("HF_TOKEN") or True
                        cached = hf_hub_download(repo_id=HF_REPO_ID, filename=wav_fn,
                                                 repo_type="dataset", token=hf_token)
                        audio_arr, orig_sr = librosa.load(cached, sr=None, mono=True)
                except Exception:
                    pass

        if audio_arr is None or len(audio_arr) == 0:
            skip_reasons["audio_bad"] += 1
            continue

        y = resample_to_16k(np.asarray(audio_arr), orig_sr)
        if len(y) < min_frames * HOP_LENGTH:
            skip_reasons["audio_short"] += 1
            continue

        log_mel = extract_mel(y)
        f0_hz   = rmvpe.infer_from_audio(y, sample_rate=SR, device=device).astype(np.float32)
        T       = min(len(log_mel), len(f0_hz))
        if T < min_frames:
            skip_reasons["frame_short"] += 1
            continue

        log_mel = log_mel[:T]
        f0_hz   = f0_hz[:T]
        frame_vad = extract_vad(y, T)

        label = np.zeros(len(TECHNIQUE_NAMES), dtype=np.float32)
        for our_idx in mapped:
            label[our_idx] = 1.0

        mel_chunks.append(log_mel)
        f0_chunks.append(f0_hz)
        vad_chunks.append(frame_vad)
        technique_labels.append(label)
        lengths.append(T)
        clip_ids.append(str(idx))
        counts[primary] += 1

    total_skipped = sum(skip_reasons.values())
    print(f"\n  Extracted per technique:")
    for i, name in enumerate(TECHNIQUE_NAMES):
        print(f"    {name:<12}: {counts[i]:4d} clips")
    print(f"  Skipped total: {total_skipped}")
    for reason, n in skip_reasons.items():
        if n:
            print(f"    {reason:<16}: {n}")

    return (mel_chunks, f0_chunks, vad_chunks,
            technique_labels, lengths, clip_ids, total_skipped)


# ── Save ─────────────────────────────────────────────────────────────

def save_split(output_path, mel_chunks, f0_chunks, vad_chunks,
               technique_labels, lengths, clip_ids):
    if not mel_chunks:
        print(f"  [warn] No clips extracted — skipping {output_path}")
        return

    mel_all  = np.concatenate(mel_chunks).astype(np.float16)
    f0_all   = np.concatenate(f0_chunks).astype(np.float16)
    vad_all  = np.concatenate(vad_chunks).astype(np.float16)
    tech_all = np.stack(technique_labels)
    len_arr  = np.array(lengths, dtype=np.int32)
    ids_arr  = np.array(clip_ids, dtype=object)

    np.savez(output_path, mel=mel_all, f0=f0_all, vad=vad_all,
             technique=tech_all, lengths=len_arr, ids=ids_arr)

    total_frames = len(mel_all)
    voiced_pct   = float(np.mean(vad_all > 0)) * 100
    print(f"\n  Saved {output_path}")
    print(f"    clips={len(lengths)}, frames={total_frames:,} "
          f"({total_frames * HOP_LENGTH / SR / 3600:.2f} hrs), "
          f"voiced={voiced_pct:.1f}%")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    min_frames = int(args.min_duration * SR / HOP_LENGTH)

    try:
        from rmvpe import RMVPE
    except ImportError:
        raise SystemExit("RMVPE not installed. Run: pip install rmvpe")

    if not os.path.exists(args.rmvpe_model):
        raise SystemExit(
            f"RMVPE checkpoint not found: {args.rmvpe_model}\n"
            "Download from: https://huggingface.co/lj1995/VoiceConversionWebUI")

    print(f"Loading RMVPE from {args.rmvpe_model} ...")
    rmvpe = RMVPE(args.rmvpe_model, hop_length=HOP_LENGTH)

    train_ds, test_ds = load_gtsinger(
        args.local_dir, args.train_split, args.test_split,
        streaming=not args.no_streaming, hf_name=args.hf_name)

    os.makedirs(args.output_dir, exist_ok=True)

    mapped = ['vibrato', 'breathy', 'falsetto']
    print(f"\nExtracting training split (mapped via *_tech fields: {mapped}) ...")
    tr = extract_split(train_ds, rmvpe, args.device, min_frames,
                       args.max_per_technique, "train", audio_dir=args.audio_dir)
    save_split(os.path.join(args.output_dir, "technique_gtsinger_train.npz"),
               *tr[:-1])

    if test_ds is not None:
        print("\nExtracting test split ...")
        te = extract_split(test_ds, rmvpe, args.device, min_frames,
                           args.max_per_technique, "test", audio_dir=args.audio_dir)
        save_split(os.path.join(args.output_dir, "technique_gtsinger_test.npz"),
                   *te[:-1])

    print(f"\nDone. Technique fields used from AaronZ345/GTSinger:")
    print(f"  vibrato_tech  → label[0] vibrato")
    print(f"  breathy_tech  → label[1] breathy")
    print(f"  falsetto_tech → label[2] falsetto")
    print(f"  (mix_tech / pharyngeal_tech / glissando_tech → skipped)")
    print(f"\nTo use in training, pass both VocalSet and GTSinger dirs:")
    print(f"  python vocalcoach/train.py \\")
    print(f"      --technique-dir data/vocalset \\  # for vibrato/breathy/belt/straight")
    print(f"      # Re-run with --technique-dir data/gtsinger_technique to add falsetto")
    print(f"  (ConcatDataset merging across both dirs is a future enhancement)")


if __name__ == "__main__":
    main()
