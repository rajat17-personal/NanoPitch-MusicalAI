"""
YouTube Cover Dataset Builder
==============================

Downloads original artist recordings (professional reference) and individual
cover versions (amateur proxy) for a set of songs, then separates vocals from
the backing track using demucs.

Directory layout produced
-------------------------
  data/yt_covers/
    <song_slug>/
      pro/
        original.wav          ← original artist, vocals only (demucs)
        original_full.wav     ← original artist, full mix (pre-demucs)
      amateur/
        <video_id>.wav        ← cover, vocals only (demucs)
        <video_id>_full.wav   ← cover, full mix (pre-demucs)
    manifest.json             ← per-clip metadata (url, title, role, duration)

Input format  (songs.json)
--------------------------
A JSON file listing songs to download:

  [
    {
      "slug": "rolling_in_the_deep",
      "pro_url": "https://www.youtube.com/watch?v=rYEDA3JcQqw",
      "cover_urls": [
        "https://www.youtube.com/watch?v=COVER1",
        "https://www.youtube.com/watch?v=COVER2"
      ]
    },
    ...
  ]

Usage
-----
  # 1. Install dependencies
  pip install yt-dlp demucs

  # 2. Create songs.json with your song list (see above)

  # 3. Download and separate vocals
  python scripts/download_yt_covers.py --songs data/yt_covers/songs.json

  # 4. Dry-run to check what would be downloaded (no actual download)
  python scripts/download_yt_covers.py --songs data/yt_covers/songs.json --dry-run

  # 5. Skip vocal separation (download full mix only)
  python scripts/download_yt_covers.py --songs data/yt_covers/songs.json --no-demucs

  # 6. Skip already-downloaded songs (safe to re-run)
  python scripts/download_yt_covers.py --songs data/yt_covers/songs.json --skip-existing

Notes
-----
- Each cover URL should be an individual video URL, not a playlist.
- yt-dlp downloads the best available audio stream and converts to wav.
- demucs `htdemucs` two-stem model separates vocals from accompaniment.
  The vocals stem is kept; the no_vocals stem is discarded.
- Clips shorter than MIN_DURATION_S (default 30s) are skipped — too short
  for reliable MOS scoring.
- If demucs fails on a clip, the full mix wav is kept and flagged in manifest.
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import warnings

MIN_DURATION_S = 30   # discard clips shorter than this after download


# ── dependency checks ─────────────────────────────────────────────────────────

def check_deps():
    missing = []
    try:
        import yt_dlp  # noqa
    except ImportError:
        missing.append("yt-dlp")
    try:
        import demucs  # noqa
    except ImportError:
        missing.append("demucs")
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        sys.exit(1)


# ── download ──────────────────────────────────────────────────────────────────

def download_audio(url, out_path, dry_run=False):
    """
    Download best audio from url, convert to 16kHz mono wav at out_path.
    Returns (success: bool, title: str, duration_s: float).
    """
    import yt_dlp

    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "format":           "bestaudio/best",
        "outtmpl":          str(out_path.with_suffix("")),   # yt-dlp adds ext
        "postprocessors": [{
            "key":            "FFmpegExtractAudio",
            "preferredcodec": "wav",
        }],
        "postprocessor_args": ["-ar", "16000", "-ac", "1"],  # 16kHz mono
        "quiet":            True,
        "no_warnings":      True,
        "noplaylist":       True,   # single video only — no accidental playlist
    }

    if dry_run:
        with yt_dlp.YoutubeDL({"quiet": True, "noplaylist": True}) as ydl:
            try:
                info = ydl.extract_info(url, download=False)
                title    = info.get("title", "unknown")
                duration = info.get("duration", 0)
                print(f"  [dry-run] {title[:60]}  ({duration}s)")
                return True, title, float(duration)
            except Exception as e:
                print(f"  [dry-run] ERROR extracting info: {e}")
                return False, "", 0.0

    # actual download
    actual_out = out_path.with_suffix(".wav")
    if actual_out.exists():
        return True, actual_out.stem, _wav_duration(actual_out)

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info  = ydl.extract_info(url, download=True)
            title    = info.get("title", "unknown")
            duration = info.get("duration", 0)
            return True, title, float(duration)
        except Exception as e:
            warnings.warn(f"[yt-dlp] Failed {url}: {e}")
            return False, "", 0.0


def _wav_duration(path):
    try:
        import librosa
        y, sr = librosa.load(str(path), sr=None, mono=True)
        return len(y) / sr
    except Exception:
        return 0.0


# ── vocal separation ──────────────────────────────────────────────────────────

def separate_vocals(full_mix_path, vocals_out_path, dry_run=False):
    """
    Run demucs htdemucs two-stem separation on full_mix_path.
    Moves the vocals stem to vocals_out_path.
    Returns True on success.
    """
    if dry_run:
        print(f"  [dry-run] demucs {pathlib.Path(full_mix_path).name}")
        return True

    full_mix_path  = pathlib.Path(full_mix_path)
    vocals_out_path = pathlib.Path(vocals_out_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            sys.executable, "-m", "demucs",
            "--two-stems", "vocals",
            "--out",       tmpdir,
            "--name",      "htdemucs",
            str(full_mix_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            warnings.warn(f"[demucs] Failed on {full_mix_path.name}:\n{result.stderr[-500:]}")
            return False

        # demucs writes: <tmpdir>/htdemucs/<stem_name>/vocals.wav
        stem_name   = full_mix_path.stem
        vocals_stem = pathlib.Path(tmpdir) / "htdemucs" / stem_name / "vocals.wav"

        if not vocals_stem.exists():
            warnings.warn(f"[demucs] vocals.wav not found at expected path {vocals_stem}")
            return False

        vocals_out_path.parent.mkdir(parents=True, exist_ok=True)
        vocals_stem.rename(vocals_out_path)
        return True


# ── per-song processing ───────────────────────────────────────────────────────

def process_song(song, output_root, skip_existing, no_demucs, dry_run, existing_by_url=None):
    slug     = song["slug"]
    pro_url  = song["pro_url"]
    covers   = [u for u in song.get("cover_urls", []) if "FILL_ME_IN" not in u]
    skipped_placeholders = len(song.get("cover_urls", [])) - len(covers)

    song_dir = pathlib.Path(output_root) / slug
    records  = []

    print(f"\n{'='*60}")
    print(f"Song: {slug}  ({len(covers)} covers)", end="")
    if skipped_placeholders:
        print(f"  [{skipped_placeholders} FILL_ME_IN skipped]", end="")
    print(f"\n{'='*60}")

    existing_by_url = existing_by_url or {}

    def _should_skip(url, file_path):
        """Skip only if the output file actually exists on disk."""
        return skip_existing and file_path.exists()

    # ── professional (original artist) ───────────────────────────────────────
    print(f"\n[pro] {pro_url}")
    pro_full   = song_dir / "pro" / "original_full.wav"
    pro_vocals = song_dir / "pro" / "original.wav"
    target_pro = pro_vocals if not no_demucs else pro_full

    if _should_skip(pro_url, target_pro):
        print("  skipped (already exists)")
        prev = existing_by_url.get(pro_url, {})
        rec = {"slug": slug, "role": "pro", "url": pro_url,
               "full_wav": str(pro_full), "vocals_wav": str(pro_vocals),
               "title": prev.get("title", ""), "duration_s": prev.get("duration_s", 0),
               "download_ok": True, "demucs_ok": prev.get("demucs_ok", None),
               "skipped": True}
    else:
        ok, title, dur = download_audio(pro_url, pro_full, dry_run=dry_run)
        rec = {"slug": slug, "role": "pro", "url": pro_url,
               "title": title, "duration_s": dur,
               "full_wav": str(pro_full), "vocals_wav": str(pro_vocals),
               "download_ok": ok, "demucs_ok": False, "skipped": False}

        if ok and dur < MIN_DURATION_S and not dry_run:
            print(f"  WARNING: only {dur:.1f}s — below {MIN_DURATION_S}s minimum, keeping anyway")

        if ok and not no_demucs and not dry_run:
            print(f"  separating vocals ...")
            rec["demucs_ok"] = separate_vocals(pro_full, pro_vocals, dry_run=dry_run)
        elif ok:
            rec["demucs_ok"] = None   # skipped by --no-demucs

    records.append(rec)

    # ── amateur covers ────────────────────────────────────────────────────────
    for i, url in enumerate(covers):
        # use video ID extracted from url as filename, fallback to index
        vid_id = _extract_video_id(url) or f"cover_{i:03d}"
        print(f"\n[amateur {i+1}/{len(covers)}] {url}")

        full_path   = song_dir / "amateur" / f"{vid_id}_full.wav"
        vocals_path = song_dir / "amateur" / f"{vid_id}.wav"
        target_am   = vocals_path if not no_demucs else full_path

        if _should_skip(url, target_am):
            print("  skipped (already exists)")
            prev = existing_by_url.get(url, {})
            rec = {"slug": slug, "role": "amateur", "url": url,
                   "video_id": vid_id,
                   "full_wav": str(full_path), "vocals_wav": str(vocals_path),
                   "title": prev.get("title", ""), "duration_s": prev.get("duration_s", 0),
                   "download_ok": True, "demucs_ok": prev.get("demucs_ok", None),
                   "skipped": True}
        else:
            ok, title, dur = download_audio(url, full_path, dry_run=dry_run)
            rec = {"slug": slug, "role": "amateur", "url": url,
                   "video_id": vid_id, "title": title, "duration_s": dur,
                   "full_wav": str(full_path), "vocals_wav": str(vocals_path),
                   "download_ok": ok, "demucs_ok": False, "skipped": False}

            if ok and dur < MIN_DURATION_S and not dry_run:
                print(f"  WARNING: only {dur:.1f}s — below {MIN_DURATION_S}s minimum")

            if ok and not no_demucs and not dry_run:
                print(f"  separating vocals ...")
                rec["demucs_ok"] = separate_vocals(full_path, vocals_path, dry_run=dry_run)
            elif ok:
                rec["demucs_ok"] = None

        records.append(rec)

    return records


def _extract_video_id(url):
    """Extract YouTube video ID from a watch URL."""
    import urllib.parse
    parsed = urllib.parse.urlparse(url)
    if "youtu.be" in parsed.netloc:
        return parsed.path.lstrip("/").split("/")[0]
    qs = urllib.parse.parse_qs(parsed.query)
    ids = qs.get("v", [])
    return ids[0] if ids else None


# ── manifest ──────────────────────────────────────────────────────────────────

def load_manifest(path):
    path = pathlib.Path(path)
    if path.exists():
        with open(path) as fh:
            return json.load(fh)
    return []


def save_manifest(records, path):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(records, fh, indent=2)

    pro_ok  = sum(1 for r in records if r.get("role") == "pro"     and r.get("download_ok"))
    am_ok   = sum(1 for r in records if r.get("role") == "amateur" and r.get("download_ok"))
    dem_ok  = sum(1 for r in records if r.get("demucs_ok") is True)
    dem_fail = sum(1 for r in records if r.get("demucs_ok") is False and not r.get("skipped"))

    print(f"\n{'='*60}")
    print(f"Manifest saved → {path}")
    print(f"  pro downloads:     {pro_ok}")
    print(f"  amateur downloads: {am_ok}")
    print(f"  demucs success:    {dem_ok}")
    print(f"  demucs failures:   {dem_fail}  (full mix kept as fallback)")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Download YouTube cover dataset for singing eval")
    p.add_argument("--songs",          required=True,
                   help="Path to songs.json file")
    p.add_argument("--output",         default="data/yt_covers",
                   help="Root output directory (default: data/yt_covers)")
    p.add_argument("--manifest",       default="data/yt_covers/manifest.json",
                   help="Manifest JSON output path (default: data/yt_covers/manifest.json)")
    p.add_argument("--no-demucs",      action="store_true",
                   help="Skip vocal separation — keep full mix only")
    p.add_argument("--skip-existing",  action="store_true",
                   help="Skip clips whose output wav already exists")
    p.add_argument("--dry-run",        action="store_true",
                   help="Print what would be downloaded without downloading")
    p.add_argument("--songs-filter",   nargs="+", metavar="SLUG",
                   help="Only process these song slugs (useful for partial re-runs)")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.dry_run:
        check_deps()

    songs_path = pathlib.Path(args.songs)
    if not songs_path.exists():
        # write a starter template if the file doesn't exist yet
        template = [
            {
                "slug": "rolling_in_the_deep",
                "pro_url": "https://www.youtube.com/watch?v=rYEDA3JcQqw",
                "cover_urls": [
                    "https://www.youtube.com/watch?v=COVER_URL_1",
                    "https://www.youtube.com/watch?v=COVER_URL_2"
                ]
            },
            {
                "slug": "hallelujah",
                "pro_url": "https://www.youtube.com/watch?v=YrLk4vdY28Q",
                "cover_urls": [
                    "https://www.youtube.com/watch?v=COVER_URL_1"
                ]
            }
        ]
        songs_path.parent.mkdir(parents=True, exist_ok=True)
        with open(songs_path, "w") as fh:
            json.dump(template, fh, indent=2)
        print(f"songs.json not found — created template at {songs_path}")
        print("Edit it with real YouTube URLs then re-run.")
        sys.exit(0)

    with open(songs_path) as fh:
        songs = json.load(fh)

    if args.songs_filter:
        songs = [s for s in songs if s["slug"] in args.songs_filter]
        print(f"Filtered to {len(songs)} songs: {[s['slug'] for s in songs]}")

    all_records = load_manifest(args.manifest) if args.skip_existing else []
    # Index existing records by (slug, video_id/role) so per-song processing can
    # skip individual clips that already succeeded, without skipping the whole song.
    existing_by_url = {r["url"]: r for r in all_records} if args.skip_existing else {}

    for song in songs:
        records = process_song(
            song,
            output_root=args.output,
            skip_existing=args.skip_existing,
            no_demucs=args.no_demucs,
            dry_run=args.dry_run,
            existing_by_url=existing_by_url,
        )
        # Replace stale records for this song with fresh ones
        slug = song["slug"]
        all_records = [r for r in all_records if r.get("slug") != slug]
        all_records.extend(records)

    if not args.dry_run:
        save_manifest(all_records, args.manifest)
    else:
        n_pro  = sum(1 for s in songs for _ in [s["pro_url"]])
        n_am   = sum(len(s.get("cover_urls", [])) for s in songs)
        print(f"\n[dry-run] Would download: {n_pro} pro + {n_am} amateur clips")
        print(f"          Vocal separation: {'disabled' if args.no_demucs else 'enabled (htdemucs)'}")


if __name__ == "__main__":
    main()
