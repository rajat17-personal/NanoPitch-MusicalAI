"""
Quality Scoring Data Preparation
=================================

Pre-extracts log-mel features from raw audio and writes NPZ files consumed
by vocalcoach/train.py quality head training. Run this ONCE before training;
training reads the NPZs directly (no librosa on the hot path).

NPZ schemas — flat layout (same pattern as clean.npz / technique_train.npz)
---------------------------------------------------------------------------

quality_mse.npz   — SingMOS-Pro clips with scalar MOS targets
  mel:     (total_frames, 40)  float16  — concatenated log-mel, all clips
  scores:  (n_clips,)          float32  — MOS scalar per clip (1–5 scale)
  lengths: (n_clips,)          int32    — frame count per clip
  ids:     (n_clips,)          object   — clip_id strings

quality_ccmusic.npz   — ccmusic clips with 9-dim expert score targets
  mel:     (total_frames, 40)  float16
  scores:  (n_clips, 9)        float32  — expert dims (CCMUSIC_DIMS order)
  lengths: (n_clips,)          int32
  ids:     (n_clips,)          object

quality_pairs.npz   — PopBuTFy amateur/professional contrastive pairs
  mel_pro:  (total_frames_pro, 40)  float16  — concatenated pro clips
  mel_am:   (total_frames_am,  40)  float16  — concatenated amateur clips
  lengths_pro: (n_pairs,)  int32    — frame count per pro clip
  lengths_am:  (n_pairs,)  int32    — frame count per amateur clip
  ids_pro:  (n_pairs,)     object
  ids_am:   (n_pairs,)     object

Full clips are stored; seq_len windowing happens in train.py __getitem__,
exactly like PitchVADDataset / TechniqueDataset. This means:
  - --seq-len in train.py takes effect on quality data too
  - Each epoch draws fresh random windows (true augmentation)
  - NPZ does not need to be regenerated when seq-len changes

Mel extraction parameters (must match vocalcoach/model.py and extractFeatures.py):
  SR=16000, hop=160 (10ms), win=400 (25ms), n_fft=512, n_mels=40,
  fmin=31.7, fmax=8000, ref=1.0 (fixed — NOT per-file ref=np.max)

SingMOS-Pro
-----------
The dataset is loaded from the HuggingFace cache (downloaded via
`load_dataset("TangRain/SingMOS-Pro")`).  Pass the snapshot root directory
with --singmos-snapshot.  The snapshot must contain:
  wav/                 — 7981 WAV files named sys{NNNN}-utt{NNNN}.wav
  info/score.json      — utterance-level MOS scores

To find your snapshot path:
  python3 -c "
  from huggingface_hub import snapshot_download
  print(snapshot_download('TangRain/SingMOS-Pro'))
  "
Or use the cached path directly:
  ~/.cache/huggingface/hub/datasets--TangRain--SingMOS-Pro/snapshots/<hash>

Usage
-----
# Variant 1 only (PopBuTFy contrastive pairs)
python scripts/prepareQualityData.py \\
    --popbutfy-dir     data/popbutfy \\
    --baselines-json   data/combined_eval_baselines.json \\
    --output-dir       data/quality

# Variant 2/3 — add SingMOS-Pro MSE source
python scripts/prepareQualityData.py \\
    --popbutfy-dir       data/popbutfy \\
    --baselines-json     data/combined_eval_baselines.json \\
    --singmos-snapshot   ~/.cache/huggingface/hub/datasets--TangRain--SingMOS-Pro/snapshots/<hash> \\
    --output-dir         data/quality

# Variant 2 — add ccmusic 9-dim expert labels
python scripts/prepareQualityData.py \\
    --popbutfy-dir       data/popbutfy \\
    --baselines-json     data/combined_eval_baselines.json \\
    --ccmusic-wavs-dir   data/ccmusic_wavs \\
    --singmos-snapshot   ~/.cache/huggingface/hub/datasets--TangRain--SingMOS-Pro/snapshots/<hash> \\
    --output-dir         data/quality
"""

import argparse
import json
import os
import sys
from multiprocessing.pool import ThreadPool

import librosa
import numpy as np
from tqdm import tqdm

# ── Mel extraction constants — must match extractFeatures.py and model.py ──
SR         = 16000
HOP_LENGTH = 160      # 10 ms
WIN_LENGTH = 400      # 25 ms
N_FFT      = 512
N_MELS     = 40
FMIN       = 31.7     # Hz — matches PITCH_FMIN in model.py
FMAX       = 8000.0

# ccmusic expert score dimensions — order must match CcmusicQualityDataset in train.py
CCMUSIC_DIMS = [
    'pitch', 'rhythm', 'vocal_range', 'timbre', 'pronunciation',
    'vibrato', 'dynamic', 'breath_control', 'overall_performance',
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Pre-extract quality training NPZs from raw audio (full clips).")

    p.add_argument("--output-dir", default="data/quality",
                   help="directory to write NPZ files (default: data/quality)")
    p.add_argument("--max-clips", type=int, default=None,
                   help="cap total clips per source (useful for quick sanity runs)")
    p.add_argument("--num-workers", type=int, default=4,
                   help="parallel workers for mel extraction (default: 4, use 0 for single-process)")

    # PopBuTFy contrastive pairs (Variant 1/2/3)
    p.add_argument("--popbutfy-dir", type=str, default=None,
                   help="PopBuTFy root dir (Singer#singing#Song_Amateur/ folders). "
                        "Required to produce quality_pairs.npz.")
    p.add_argument("--baselines-json", type=str, default=None,
                   help="path to combined_eval_baselines.json. Required when "
                        "--popbutfy-dir or --ccmusic-wavs-dir is set.")

    # ccmusic 9-dim expert labels (Variant 2)
    p.add_argument("--ccmusic-wavs-dir", type=str, default=None,
                   help="directory of ccmusic_NNNN.wav files. "
                        "Produces quality_ccmusic.npz.")

    # SingMOS-Pro MOS scalar targets (Variants 2 and 3)
    p.add_argument("--singmos-snapshot", type=str, default=None,
                   help="Path to SingMOS-Pro HuggingFace snapshot directory "
                        "(must contain wav/ and info/score.json). "
                        "Produces quality_mse.npz.")

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════
# Mel extraction
# ═══════════════════════════════════════════════════════════════════════

def load_mel(path: str) -> np.ndarray:
    """Load a WAV/FLAC file and return log-mel spectrogram (T, N_MELS) float32.

    Uses fixed reference (ref=1.0) so inter-clip dynamics are preserved.
    This matches extractFeatures.py — do NOT use ref=np.max here.
    """
    y, _ = librosa.load(path, sr=SR, mono=True)
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH, n_mels=N_MELS,
        fmin=FMIN, fmax=FMAX, window="hann", center=True,
    )
    return librosa.power_to_db(mel, ref=1.0).T.astype(np.float32)  # (T, N_MELS)


def _load_mel_worker(path: str):
    try:
        return path, load_mel(path)
    except Exception:
        return path, None


def parallel_load_mels(paths: list, num_workers: int, desc: str) -> dict:
    """Load mels for a list of paths. Uses ThreadPool — no pickling overhead."""
    results = {}
    if num_workers <= 1:
        for path in tqdm(paths, desc=desc):
            _, mel = _load_mel_worker(path)
            results[path] = mel
    else:
        with ThreadPool(num_workers) as pool:
            for path, mel in tqdm(pool.imap(_load_mel_worker, paths),
                                  total=len(paths), desc=desc):
                results[path] = mel
    return results


# ═══════════════════════════════════════════════════════════════════════
# PopBuTFy pair indexing
# ═══════════════════════════════════════════════════════════════════════

def _index_popbutfy(root: str, suffix: str, prefix: str) -> dict:
    """Scan PopBuTFy root for folders ending in suffix, return clip_id → path."""
    audio_exts = {'.wav', '.flac', '.mp3'}
    mapping = {}
    for folder in sorted(os.scandir(root), key=lambda e: e.name):
        if not folder.is_dir() or not folder.name.lower().endswith(suffix):
            continue
        for f in sorted(os.scandir(folder.path), key=lambda e: e.name):
            if os.path.splitext(f.name)[1].lower() in audio_exts:
                stem = os.path.splitext(f.name)[0]
                cid  = f"{prefix}_{folder.name}_{stem}"
                mapping[cid] = f.path
    return mapping


def _singer_song_key(folder_name: str) -> str:
    """Strip trailing _Amateur or _Professional to get singer×song identity."""
    for suffix in ('_Professional', '_Amateur', '_professional', '_amateur'):
        if folder_name.endswith(suffix):
            return folder_name[:-len(suffix)]
    return folder_name


def build_pairs(popbutfy_dir: str, baselines: dict) -> list:
    """Return list of (pro_path, am_path, pro_id, am_id) matched by singer×song."""
    pro_paths = _index_popbutfy(popbutfy_dir, '_professional', 'popbutfy_pro')
    am_paths  = _index_popbutfy(popbutfy_dir, '_amateur',      'popbutfy_am')

    pro_scored = set(baselines.get('popbutfy_professional', {}).keys())
    am_scored  = set(baselines.get('popbutfy_amateur',      {}).keys())

    pro_by_ss: dict = {}
    for cid, path in pro_paths.items():
        if cid in pro_scored:
            folder = os.path.basename(os.path.dirname(path))
            pro_by_ss.setdefault(_singer_song_key(folder), []).append((cid, path))

    am_by_ss: dict = {}
    for cid, path in am_paths.items():
        if cid in am_scored:
            folder = os.path.basename(os.path.dirname(path))
            am_by_ss.setdefault(_singer_song_key(folder), []).append((cid, path))

    pairs = []
    for ss in sorted(pro_by_ss.keys()):
        if ss not in am_by_ss:
            continue
        for pro_id, pro_p in pro_by_ss[ss]:
            for am_id, am_p in am_by_ss[ss]:
                pairs.append((pro_p, am_p, pro_id, am_id))

    n_matched = sum(1 for ss in pro_by_ss if ss in am_by_ss)
    print(f"  PopBuTFy: {len(pairs)} pairs from {n_matched} matched singer×song keys")
    return pairs


# ═══════════════════════════════════════════════════════════════════════
# Writers — flat layout, full clips, variable lengths
# ═══════════════════════════════════════════════════════════════════════

def write_mse_singmos(snapshot_dir: str, output_path: str, max_clips, num_workers: int):
    """Build quality_mse.npz from SingMOS-Pro HuggingFace snapshot."""
    scores_json = os.path.join(snapshot_dir, "info", "score.json")
    wav_dir     = os.path.join(snapshot_dir, "wav")

    if not os.path.exists(scores_json):
        sys.exit(f"  score.json not found at {scores_json}")
    if not os.path.isdir(wav_dir):
        sys.exit(f"  wav/ directory not found at {wav_dir}")

    with open(scores_json) as f:
        score_data = json.load(f)
    utterances = score_data.get("utterance", {})

    items = []
    for clip_id, entry in utterances.items():
        mos = entry.get("score", {}).get("mos")
        if mos is None:
            continue
        wav = os.path.join(wav_dir, f"{clip_id}.wav")
        if os.path.exists(wav):
            items.append((clip_id, wav, float(mos)))

    if max_clips:
        items = items[:max_clips]

    print(f"  SingMOS-Pro: {len(items)} clips with MOS targets "
          f"(range {min(x[2] for x in items):.2f}–{max(x[2] for x in items):.2f})")
    if not items:
        print("  [warn] No WAV files found — skipping quality_mse.npz")
        return

    mels = parallel_load_mels([w for _, w, _ in items], num_workers, "quality_mse")

    mel_parts, sc, lengths, ids = [], [], [], []
    for clip_id, wav, mos in items:
        m = mels.get(wav)
        if m is None:
            print(f"  [warn] {clip_id}: load failed")
            continue
        mel_parts.append(m.astype(np.float16))
        sc.append(mos)
        lengths.append(m.shape[0])
        ids.append(clip_id)

    np.savez(
        output_path,
        mel=np.concatenate(mel_parts, axis=0),
        scores=np.array(sc, dtype=np.float32),
        lengths=np.array(lengths, dtype=np.int32),
        ids=np.array(ids, dtype=object),
    )
    print(f"  Saved {len(ids)} clips, {sum(lengths):,} frames → {output_path}")


def write_ccmusic(baselines_json: str, ccmusic_wavs_dir: str, output_path: str,
                  max_clips, num_workers: int):
    """Build quality_ccmusic.npz — full clips, flat layout."""
    with open(baselines_json) as f:
        data = json.load(f)
    cc = data.get('ccmusic_professional', {})

    items = []
    wavs_dir = os.path.abspath(ccmusic_wavs_dir)
    for clip_id, entry in cc.items():
        wav = os.path.join(wavs_dir, f"{clip_id}.wav")
        if not os.path.exists(wav):
            continue
        expert = entry.get('expert_scores', {})
        scores_9d = np.array(
            [expert.get(d, 0.0) for d in CCMUSIC_DIMS], dtype=np.float32)
        items.append((clip_id, wav, scores_9d))
    if max_clips:
        items = items[:max_clips]

    print(f"  ccmusic: {len(items)} clips with 9-dim expert targets")
    if not items:
        print("  [warn] No matching ccmusic WAVs found — skipping quality_ccmusic.npz")
        return

    mels = parallel_load_mels([w for _, w, _ in items], num_workers, "quality_ccmusic")

    mel_parts, sc, lengths, ids = [], [], [], []
    for clip_id, wav, scores_9d in items:
        m = mels.get(wav)
        if m is None:
            print(f"  [warn] {clip_id}: load failed")
            continue
        mel_parts.append(m.astype(np.float16))
        sc.append(scores_9d)
        lengths.append(m.shape[0])
        ids.append(clip_id)

    np.savez(
        output_path,
        mel=np.concatenate(mel_parts, axis=0),
        scores=np.stack(sc).astype(np.float32),
        lengths=np.array(lengths, dtype=np.int32),
        ids=np.array(ids, dtype=object),
    )
    print(f"  Saved {len(ids)} clips, {sum(lengths):,} frames → {output_path}")


def write_pairs(popbutfy_dir: str, baselines_json: str, output_path: str,
                max_clips, num_workers: int):
    """Build quality_pairs.npz — full clips, flat layout, separate pro/am buffers.

    Uses memmap to write mel data directly to disk — avoids loading all pairs
    into RAM at once (full dataset is ~39 GB expanded, ~1 GB unique files).
    """
    with open(baselines_json) as f:
        baselines = json.load(f)

    pairs = build_pairs(popbutfy_dir, baselines)
    if max_clips:
        # Shuffle before capping so the subset spans all singer×song keys.
        # build_pairs() emits a sorted cartesian product, so pairs[:N] without
        # shuffling would draw only from the first few singers. Fixed seed keeps
        # the subset reproducible across regenerations.
        rng = np.random.default_rng(0)
        rng.shuffle(pairs)
        pairs = pairs[:max_clips]
    if not pairs:
        print("  [warn] No matched pairs found — skipping quality_pairs.npz")
        return

    # Load unique audio files in parallel — fits in RAM (~1.1 GB for full set)
    all_paths = list({p for pro_p, am_p, _, _ in pairs for p in (pro_p, am_p)})
    mels = parallel_load_mels(all_paths, num_workers, "quality_pairs (loading)")

    # Filter to valid pairs and collect lengths — O(n) pass, no mel copies
    valid = []
    for pro_p, am_p, pro_id, am_id in pairs:
        mp_ = mels.get(pro_p)
        ma_ = mels.get(am_p)
        if mp_ is not None and ma_ is not None:
            valid.append((pro_p, am_p, pro_id, am_id, mp_.shape[0], ma_.shape[0]))

    total_pro = sum(v[4] for v in valid)
    total_am  = sum(v[5] for v in valid)
    n_pairs   = len(valid)
    print(f"  Writing {n_pairs:,} pairs ({total_pro:,} pro frames, {total_am:,} am frames) ...")

    # Pre-allocate memmaps on disk — never holds more than one clip in RAM at a time
    tmp_pro = output_path + ".mel_pro.tmp"
    tmp_am  = output_path + ".mel_am.tmp"
    mm_pro  = np.memmap(tmp_pro, dtype=np.float16, mode='w+', shape=(total_pro, N_MELS))
    mm_am   = np.memmap(tmp_am,  dtype=np.float16, mode='w+', shape=(total_am,  N_MELS))

    lengths_pro, lengths_am = [], []
    ids_pro, ids_am = [], []
    off_pro = off_am = 0

    for pro_p, am_p, pro_id, am_id, lp, la in tqdm(valid, desc="quality_pairs (writing)"):
        mm_pro[off_pro:off_pro + lp] = mels[pro_p].astype(np.float16)
        mm_am [off_am :off_am  + la] = mels[am_p ].astype(np.float16)
        off_pro += lp;  off_am += la
        lengths_pro.append(lp);  lengths_am.append(la)
        ids_pro.append(pro_id);  ids_am.append(am_id)

    # Flush memmaps before packing into npz
    del mm_pro, mm_am

    mel_pro_arr = np.memmap(tmp_pro, dtype=np.float16, mode='r', shape=(total_pro, N_MELS))
    mel_am_arr  = np.memmap(tmp_am,  dtype=np.float16, mode='r', shape=(total_am,  N_MELS))

    np.savez(
        output_path,
        mel_pro=mel_pro_arr,
        mel_am=mel_am_arr,
        lengths_pro=np.array(lengths_pro, dtype=np.int32),
        lengths_am=np.array(lengths_am,   dtype=np.int32),
        ids_pro=np.array(ids_pro, dtype=object),
        ids_am=np.array(ids_am,  dtype=object),
    )
    del mel_pro_arr, mel_am_arr
    os.remove(tmp_pro);  os.remove(tmp_am)

    print(f"  Saved {n_pairs:,} pairs, "
          f"{total_pro:,} pro frames, {total_am:,} am frames → {output_path}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    produced = []

    if args.singmos_snapshot:
        out = os.path.join(args.output_dir, "quality_mse.npz")
        print(f"\n[1/3] Building quality_mse.npz from SingMOS-Pro snapshot ...")
        write_mse_singmos(args.singmos_snapshot, out, args.max_clips, args.num_workers)
        produced.append(out)

    if args.ccmusic_wavs_dir:
        if not args.baselines_json:
            sys.exit("--ccmusic-wavs-dir requires --baselines-json")
        out = os.path.join(args.output_dir, "quality_ccmusic.npz")
        print(f"\n[2/3] Building quality_ccmusic.npz ...")
        write_ccmusic(args.baselines_json, args.ccmusic_wavs_dir, out,
                      args.max_clips, args.num_workers)
        produced.append(out)

    if args.popbutfy_dir:
        if not args.baselines_json:
            sys.exit("--popbutfy-dir requires --baselines-json")
        out = os.path.join(args.output_dir, "quality_pairs.npz")
        print(f"\n[3/3] Building quality_pairs.npz ...")
        write_pairs(args.popbutfy_dir, args.baselines_json, out,
                    args.max_clips, args.num_workers)
        produced.append(out)

    if not produced:
        sys.exit("Nothing to do — pass at least one of: "
                 "--popbutfy-dir, --ccmusic-wavs-dir, --singmos-snapshot")

    print(f"\nDone. Files written:")
    for p in produced:
        if os.path.exists(p):
            size_mb = os.path.getsize(p) / 1024**2
            print(f"  {p}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
