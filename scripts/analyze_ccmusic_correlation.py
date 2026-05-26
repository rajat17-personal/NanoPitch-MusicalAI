"""
Correlation analysis: SingMOS-Pro + SongEvalGenerator vs expert/MOS scores
===========================================================================

Two correlation modes:

1. ccmusic mode (default)
   Computes Pearson and Spearman correlations between each model score and each
   of the 9 expert dimensions in ccmusic-database/acapella.
   Flags which dimensions either model tracks meaningfully (|r| > 0.2).

2. SingMOS-Pro mode  (--singmos-pro)
   Scores all 7,981 clips in TangRain/SingMOS-Pro (already cached locally)
   using SongEvalGenerator, then correlates against human MOS annotations.
   Can be filtered by system type: gt / svs / svc / svr / all (default: all).
   Scores are cached to --singmos-pro-cache so subsequent runs are instant.

Usage
-----
    # ccmusic correlation only
    python scripts/analyze_ccmusic_correlation.py

    # SingMOS-Pro correlation (scores all 7981 clips — takes ~2 hrs first run)
    python scripts/analyze_ccmusic_correlation.py --singmos-pro

    # SingMOS-Pro: GT clips only (578 clips, ~15 min)
    python scripts/analyze_ccmusic_correlation.py --singmos-pro --singmos-type gt

    # SingMOS-Pro: use cached scores (instant after first run)
    python scripts/analyze_ccmusic_correlation.py --singmos-pro \
        --singmos-pro-cache data/singmos_pro_seg_scores.json

    # Run both in one go
    python scripts/analyze_ccmusic_correlation.py --singmos-pro --singmos-type all \
        --baselines data/combined_eval_baselines.json
"""

import argparse
import json
import pathlib
import sys
import warnings

import numpy as np
from scipy import stats

# ── ccmusic correlation ───────────────────────────────────────────────────────

EXPERT_KEYS = [
    "pitch", "rhythm", "vocal_range", "timbre", "pronunciation",
    "vibrato", "dynamic", "breath_control", "overall_performance",
]

MODELS = ["singmos", "songevalgen"]

MODEL_LABELS = {
    "singmos":     "SingMOS-Pro",
    "songevalgen": "SongEvalGen",
}


def load_ccmusic_vectors(baselines_path):
    with open(baselines_path) as fh:
        data = json.load(fh)

    pro = data.get("ccmusic_professional", {})

    vectors = {m: [] for m in MODELS}
    expert  = {k: [] for k in EXPERT_KEYS}

    for rec in pro.values():
        exp = rec.get("expert_scores", {})
        if not all(exp.get(k) is not None for k in EXPERT_KEYS):
            continue
        if not all(rec.get(m) is not None for m in MODELS):
            continue
        for m in MODELS:
            vectors[m].append(rec[m])
        for k in EXPERT_KEYS:
            expert[k].append(exp[k])

    n = len(vectors["singmos"])
    print(f"Loaded {n} complete ccmusic clips (both models + all expert scores)")
    return {m: np.array(vectors[m]) for m in MODELS}, \
           {k: np.array(expert[k]) for k in EXPERT_KEYS}


def correlation_table(model_vecs, expert_vecs, models=None):
    if models is None:
        models = MODELS
    col_w = 14
    header_dim = 28

    print(f"\n{'':>{header_dim}}", end="")
    for m in models:
        print(f"  {'Pearson r':>{col_w}}  {'Spearman r':>{col_w}}", end="")
    print()

    print(f"  {'Expert dimension':<{header_dim-2}}", end="")
    for m in models:
        print(f"  {MODEL_LABELS.get(m, m)[:col_w]:>{col_w}}  {'':>{col_w}}", end="")
    print()
    print("─" * (header_dim + (col_w * 2 + 6) * len(models)))

    results = {}
    for k in EXPERT_KEYS:
        ev = expert_vecs[k]
        print(f"  {k:<{header_dim-2}}", end="")
        results[k] = {}
        for m in models:
            mv = model_vecs[m]
            pr, pp = stats.pearsonr(mv, ev)
            sr, sp = stats.spearmanr(mv, ev)
            sig_p = "*" if pp < 0.05 else " "
            sig_s = "*" if sp < 0.05 else " "
            results[k][m] = {"pearson": pr, "pearson_p": pp,
                             "spearman": sr, "spearman_p": sp}
            print(f"  {pr:>+{col_w-1}.3f}{sig_p}  {sr:>+{col_w-1}.3f}{sig_s}", end="")
        print()

    return results


def model_summary(model_vecs, expert_vecs, results, models=None):
    if models is None:
        models = MODELS
    print("\n\nModel Summary")
    print("─" * 60)
    for m in models:
        pearsons  = [results[k][m]["pearson"]  for k in EXPERT_KEYS]
        spearmans = [results[k][m]["spearman"] for k in EXPERT_KEYS]
        mv = model_vecs[m]
        print(f"\n{MODEL_LABELS.get(m, m)}")
        print(f"  Score range   : {mv.min():.3f} – {mv.max():.3f}  "
              f"(mean {mv.mean():.3f}, std {mv.std():.3f})")
        print(f"  Best Pearson  : {max(pearsons, key=abs):+.3f}  "
              f"({EXPERT_KEYS[np.argmax(np.abs(pearsons))]})")
        print(f"  Best Spearman : {max(spearmans, key=abs):+.3f}  "
              f"({EXPERT_KEYS[np.argmax(np.abs(spearmans))]})")
        print(f"  Mean |Pearson|: {np.mean(np.abs(pearsons)):.3f}")

        meaningful = [k for k in EXPERT_KEYS if abs(results[k][m]["pearson"]) > 0.2]
        print(f"  Dims |r|>0.2  : {meaningful if meaningful else 'none'}")

        overall_r = results["overall_performance"][m]["pearson"]
        if overall_r > 0.2:
            direction = "correct — higher model score = higher expert rating"
        elif overall_r < -0.2:
            direction = "INVERTED — higher model score = LOWER expert rating"
        else:
            direction = "no meaningful correlation with overall_performance"
        print(f"  Direction     : {direction}")


def gtsinger_model_check(gt_path):
    try:
        with open(gt_path) as fh:
            d = json.load(fh)
    except FileNotFoundError:
        return

    print("\n\nGTSinger Technique Scores — both models")
    print("─" * 80)
    print(f"  {'Technique':<28}  {'n_ctrl':>6}  {'SingMOS ctrl':>12}  {'SingMOS tech':>12}  "
          f"{'SEG ctrl':>10}  {'SEG tech':>10}  {'SEG delta':>10}")
    print("─" * 80)

    for tech, v in d.items():
        clips = v.get("clips", [])
        ctrl_clips = [c for c in clips if c["group"] == "control"]
        tech_clips = [c for c in clips if c["group"] != "control"]

        def mean_key(clip_list, key):
            vals = [c[key] for c in clip_list if c.get(key) is not None]
            return round(float(np.mean(vals)), 3) if vals else None

        sm_c  = mean_key(ctrl_clips, "singmos")
        sm_t  = mean_key(tech_clips, "singmos")
        seg_c = mean_key(ctrl_clips, "songevalgen")
        seg_t = mean_key(tech_clips, "songevalgen")
        seg_d = round(seg_t - seg_c, 3) if seg_c and seg_t else None

        def fmt(v): return f"{v:.3f}" if v is not None else "n/a"
        seg_d_str = f"{seg_d:+.3f}" if seg_d is not None else "n/a"
        print(f"  {tech:<28}  {len(ctrl_clips):>6}  {fmt(sm_c):>12}  {fmt(sm_t):>12}  "
              f"{fmt(seg_c):>10}  {fmt(seg_t):>10}  {seg_d_str:>10}")

    print("\n  * positive SEG delta = technique scores HIGHER than neutral (SongEvalGen rewards it)")
    print("  * negative SEG delta = technique penalised relative to neutral")


# ── SingMOS-Pro correlation ───────────────────────────────────────────────────

SINGMOS_PRO_SNAPSHOT = (
    pathlib.Path.home()
    / ".cache/huggingface/hub"
    / "datasets--TangRain--SingMOS-Pro"
    / "snapshots"
)

SINGMOS_TYPES = ("gt", "svs", "svc", "svr")


def _find_singmos_pro_snapshot():
    if not SINGMOS_PRO_SNAPSHOT.exists():
        return None
    snapshots = sorted(SINGMOS_PRO_SNAPSHOT.iterdir())
    return snapshots[-1] if snapshots else None


def load_singmos_pro_meta(type_filter=None):
    """
    Returns list of dicts: {utt_id, wav_path, mos, sys_type, dataset}.
    type_filter: list of types to include, e.g. ['gt'], or None for all.
    """
    snap = _find_singmos_pro_snapshot()
    if snap is None:
        raise FileNotFoundError(
            "SingMOS-Pro not found in HF cache. Run:\n"
            "  from datasets import load_dataset\n"
            "  load_dataset('TangRain/SingMOS-Pro', split='train')"
        )

    with open(snap / "info" / "score.json") as fh:
        scores = json.load(fh)
    with open(snap / "info" / "sys_info.json") as fh:
        sys_info = json.load(fh)

    utt_scores = scores["utterance"]
    wav_dir    = snap / "wav"

    records = []
    for utt_id, entry in utt_scores.items():
        sys_id   = entry["sys_id"]
        sys_type = sys_info[sys_id]["type"]
        if type_filter and sys_type not in type_filter:
            continue
        wav_path = wav_dir / pathlib.Path(entry["wav"]).name
        if not wav_path.exists():
            warnings.warn(f"[SingMOS-Pro] wav not found: {wav_path}")
            continue
        records.append({
            "utt_id":   utt_id,
            "wav_path": str(wav_path),
            "mos":      entry["score"]["mos"],
            "sys_type": sys_type,
            "dataset":  sys_info[sys_id]["dataset"],
        })

    type_counts = {}
    for r in records:
        type_counts[r["sys_type"]] = type_counts.get(r["sys_type"], 0) + 1
    print(f"[SingMOS-Pro] {len(records)} clips loaded  |  " +
          "  ".join(f"{t}={n}" for t, n in sorted(type_counts.items())))
    return records


def score_singmos_pro(records, cache_path=None):
    """
    Run SongEvalGenerator on all records. Returns {utt_id: seg_score}.
    Loads from cache_path if it exists; saves to it after scoring.
    """
    # load cache
    cached = {}
    if cache_path and pathlib.Path(cache_path).exists():
        with open(cache_path) as fh:
            cached = json.load(fh)
        print(f"[SingMOS-Pro] Loaded {len(cached)} cached scores from {cache_path}")

    todo = [r for r in records if r["utt_id"] not in cached]
    if not todo:
        print("[SingMOS-Pro] All clips already cached — skipping inference.")
        return cached

    # lazy-load SongEvalGenerator only if we need to score
    print(f"[SingMOS-Pro] Scoring {len(todo)} clips with SongEvalGenerator ...")
    try:
        import os, torch
        qwenfeat_root  = pathlib.Path(os.environ.get("QWENFEAT_ROOT",
                                                      str(pathlib.Path.home() / "QwenFeat")))
        audioscore_src = qwenfeat_root / "audioscore" / "src"
        ckpt_dir       = (qwenfeat_root / "audioscore" / "ckpts" /
                          "SongEvalGenerator" / "step_2_al_audio" / "best_model_step_132000")

        if not audioscore_src.exists():
            raise FileNotFoundError(f"audioscore source not found at {audioscore_src}")
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"SongEvalGenerator checkpoint not found at {ckpt_dir}")

        sys.path.insert(0, str(audioscore_src))
        from audioscore.model import SongEvalGenerator_audio_lora

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SongEvalGenerator_audio_lora()
        model.load_model(str(ckpt_dir))
        model = model.to(device).eval()
        print(f"[SingMOS-Pro] SongEvalGenerator loaded on {device}")
    except Exception as e:
        print(f"[SingMOS-Pro] Failed to load SongEvalGenerator: {e}")
        sys.exit(1)

    results = dict(cached)
    from tqdm import tqdm
    for rec in tqdm(todo, desc="SingMOS-Pro scoring"):
        try:
            score = model.generate_tag(rec["wav_path"])
            results[rec["utt_id"]] = float(score)
        except Exception as e:
            warnings.warn(f"[SingMOS-Pro] Failed {rec['utt_id']}: {e}")
            results[rec["utt_id"]] = None

        # save incrementally every 100 clips
        if cache_path and len(results) % 100 == 0:
            pathlib.Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w") as fh:
                json.dump(results, fh)

    if cache_path:
        pathlib.Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as fh:
            json.dump(results, fh)
        print(f"[SingMOS-Pro] Scores saved → {cache_path}")

    return results


def analyze_singmos_pro(records, seg_scores, type_filter=None):
    """
    Correlate SongEvalGen scores against SingMOS-Pro human MOS.
    Prints per-type breakdown and overall Pearson/Spearman.
    """
    types_to_analyze = type_filter if type_filter else list(SINGMOS_TYPES) + ["all"]

    # build arrays filtered by type
    def _arrays(type_sel):
        seg, mos = [], []
        for r in records:
            if type_sel != "all" and r["sys_type"] != type_sel:
                continue
            s = seg_scores.get(r["utt_id"])
            if s is None:
                continue
            seg.append(s)
            mos.append(r["mos"])
        return np.array(seg), np.array(mos)

    print("\n" + "=" * 70)
    print("SingMOS-Pro: SongEvalGen vs Human MOS")
    print("  * = p < 0.05")
    print("=" * 70)
    print(f"  {'Type':<10}  {'n':>5}  {'SEG mean':>9}  {'MOS mean':>9}  "
          f"{'Pearson r':>10}  {'Spearman r':>11}")
    print("─" * 70)

    for type_sel in (list(SINGMOS_TYPES) + ["all"]):
        seg_arr, mos_arr = _arrays(type_sel)
        if len(seg_arr) < 5:
            continue
        pr, pp = stats.pearsonr(seg_arr, mos_arr)
        sr, sp = stats.spearmanr(seg_arr, mos_arr)
        sig_p = "*" if pp < 0.05 else " "
        sig_s = "*" if sp < 0.05 else " "
        print(f"  {type_sel:<10}  {len(seg_arr):>5}  {seg_arr.mean():>9.3f}  "
              f"{mos_arr.mean():>9.3f}  {pr:>+9.3f}{sig_p}  {sr:>+10.3f}{sig_s}")

    # direction check on gt clips
    seg_gt, mos_gt = _arrays("gt")
    if len(seg_gt) >= 5:
        pr_gt, _ = stats.pearsonr(seg_gt, mos_gt)
        print(f"\n  GT clips: Pearson r = {pr_gt:+.3f}  "
              f"({'correct direction ✓' if pr_gt > 0.1 else 'weak / inverted ✗'})")

    # MOS range SongEvalGen hits in gt vs svs
    seg_svs, mos_svs = _arrays("svs")
    if len(seg_gt) and len(seg_svs):
        print(f"\n  SEG score range  —  gt: {seg_gt.min():.2f}–{seg_gt.max():.2f} "
              f"(mean {seg_gt.mean():.2f})  |  "
              f"svs: {seg_svs.min():.2f}–{seg_svs.max():.2f} "
              f"(mean {seg_svs.mean():.2f})")
        gt_vs_svs = seg_gt.mean() - seg_svs.mean()
        verdict = "good — model prefers real humans over synthesis" if gt_vs_svs > 0.1 \
                  else "weak — model does not clearly prefer GT over SVS"
        print(f"  GT − SVS mean   : {gt_vs_svs:+.3f}  →  {verdict}")


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--baselines",           default="data/combined_eval_baselines.json")
    p.add_argument("--gtsinger",            default="data/gtsinger_technique_scores.json")
    p.add_argument("--singmos-pro",         action="store_true",
                   help="Run SingMOS-Pro MOS correlation analysis")
    p.add_argument("--singmos-type",        nargs="+", default=None,
                   metavar="TYPE",
                   help="System types to include: gt svs svc svr  (default: all)")
    p.add_argument("--singmos-pro-cache",   default="data/singmos_pro_seg_scores.json",
                   metavar="JSON",
                   help="Cache file for SongEvalGen scores on SingMOS-Pro clips "
                        "(default: data/singmos_pro_seg_scores.json)")
    p.add_argument("--skip-ccmusic",        action="store_true",
                   help="Skip the ccmusic correlation section")
    return p.parse_args()


def main():
    args = parse_args()

    # ── ccmusic correlation ───────────────────────────────────────────────────
    if not args.skip_ccmusic:
        model_vecs, expert_vecs = load_ccmusic_vectors(args.baselines)

        print("\n" + "=" * 70)
        print("Correlation: Model scores vs ccmusic expert dimensions")
        print("  * = p < 0.05  (statistically significant)")
        print("=" * 70)
        results = correlation_table(model_vecs, expert_vecs)
        model_summary(model_vecs, expert_vecs, results)
        gtsinger_model_check(args.gtsinger)

    # ── SingMOS-Pro correlation ───────────────────────────────────────────────
    if args.singmos_pro:
        type_filter = args.singmos_type  # None → all types included
        records = load_singmos_pro_meta(type_filter=type_filter)
        seg_scores = score_singmos_pro(records, cache_path=args.singmos_pro_cache)
        analyze_singmos_pro(records, seg_scores, type_filter=type_filter)


if __name__ == "__main__":
    main()
