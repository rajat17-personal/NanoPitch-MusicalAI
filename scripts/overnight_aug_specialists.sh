#!/usr/bin/env bash
# overnight_aug_specialists.sh
# ─────────────────────────────────────────────────────────────────────────────
# Overnight batch (~9h on RTX 4080 Super). Run from repo root:
#   bash scripts/overnight_aug_specialists.sh 2>&1 | tee overnight_$(date +%Y%m%d).log
#
# Goal of this batch (decided this session):
#   1. Train the aligned note head properly (the mel/label-decoupling bug is fixed).
#   2. Train TWO domain-SPECIALIST technique heads instead of one combined head:
#        - VocalSet specialist (vibrato/breathy/belt/straight) on vocalset_aug
#        - GTSinger specialist (vibrato/breathy/falsetto)      on gtsinger_aug
#      Combined VocalSet+GTSinger training trades off (datasets disagree on shared
#      classes + near-disjoint label spaces) — specialists sidestep that conflict.
#   3. Keep a clean attribution: each specialist also gets an UN-augmented baseline
#      run so we can tell whether augmentation actually helped (one-variable test).
#
# KEY LESSON baked in (from the stopped fine-tune run):
#   Fine-tuning the gainaug backbone on technique works (macro AP 0.70 frozen ->
#   0.93 unfrozen at ep~375) BUT erodes the loudness-robust VAD (macro vF1 88% ->
#   74% by ep390, clean VDR 87% -> 63%). The sweet spot is a SHORT unfreeze window.
#   So fine-tune runs here use: differential LR (gentle backbone), short patience,
#   gain-aug kept ON during fine-tune, and an OOD gate after — to catch VAD erosion.
#   Selection is on macro AP (threshold-free, stable) — already wired in train.py.
#
# Run order (sequential):
#   N1   note-head aligned, gainaug backbone          (~75 min)  ← the fix validated
#   OOD N1
#   V1   VocalSet specialist (aug, fine-tune)         (~85 min)  ← belt/0.79 push
#   OOD V1
#   V0   VocalSet baseline (UN-aug, fine-tune)        (~70 min)  ← attribution control
#   V2a  VocalSet warm-joint (short gentle adapt)     (~31 min)  ┐ joint-first-then-
#   V2b  VocalSet probe from V2a (backbone re-frozen) (~33 min)  ┘ freeze: bounded VAD erosion
#   OOD V2b
#   G1   GTSinger specialist (aug, fine-tune)         (~120 min) ← falsetto specialist
#   OOD G1
#   G0   GTSinger baseline (UN-aug, fine-tune)        (~90 min)  ← attribution control
#   Q1   quality V3 scalar (probe from N1)            (~35 min)  ┐
#   Q2   quality V2 9-dim ccmusic (probe from N1)     (~35 min)  ┤ OOD-safe probes
#   Q3   quality V1 scalar pairs-only (probe from N1) (~25 min)  ┘ (frozen backbone)
#   update_results + note-eval + OOD summary
#   Total ~8h core + patience buffers — fits 9h.
# ─────────────────────────────────────────────────────────────────────────────

set -uo pipefail   # NOTE: no -e — one failing run must not kill the whole night.

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

# Gainaug backbone (OOD VDR 67%) — the substrate for every run here.
GAIN="vocalcoach/runs/stage1_attn4_notehead_gainaug_merged_pitchvad/checkpoints/best_metric.pth"

VOCADITO="data/vocadito"
OOD_LOG="results/ood_log.json"
NOTE_LOG="results/note_log.json"

ARCH="--arch tcn --hidden 256 --n-blocks 8 --n-attn-layers 4 --n-heads 4"

log() { echo; echo "════════════════════════════════════════════════════════"; echo "  [$(date '+%H:%M:%S')]  $*"; echo "════════════════════════════════════════════════════════"; }

ood() {
    local run_name="$1" ckpt="$2"
    log "OOD eval: $run_name"
    if [[ -f "$ckpt" ]]; then
        python scripts/evalOOD.py \
            --checkpoint "$ckpt" \
            --dataset vocadito --data-dir "$VOCADITO" \
            --log "$OOD_LOG" \
            --regression-threshold 0.10 || echo "  [OOD] rejected/errored — continuing"
    else
        echo "  [OOD] checkpoint missing ($ckpt) — run likely failed; continuing"
    fi
}

START_TS=$(date +%s)

# ═════════════════════════════════════════════════════════════════════════════
# N1 — Note head, ALIGNED (the decoupled mel/label bug is fixed; note forward now
#      runs on the note loader's own mel at --note-batch-size). Gain-aug kept ON so
#      the backbone retains loudness robustness. Resumes from the gainaug backbone.
# ═════════════════════════════════════════════════════════════════════════════
log "N1 — note-head aligned (gainaug backbone)"
python vocalcoach/train.py \
    --data-dir data/merged_pitchvad --noise-dir data --eval-dir data \
    --note-head --deep-note-head --note-dirs data/annotated_vocalset \
    --w-note 3.0 --note-pos-weight 35 --note-batch-size 16 \
    $ARCH \
    --seq-len 600 --batch-size 64 --num-workers 0 \
    --w-vad 0.2 --w-pitch 2 --vad-pos-weight 2.0 --pitch-sigma 0.8 \
    --augment noise_specaug --gain-aug-db 40 --p-clean 0.3 \
    --freeze-n-blocks 6 --save-best-only \
    --epochs 60 --patience 20 --lr 3e-4 \
    --resume "$GAIN" \
    --output-dir vocalcoach/runs/stage1_gainaug_notehead_aligned

python scripts/evalNoteHead.py \
    --checkpoint vocalcoach/runs/stage1_gainaug_notehead_aligned/checkpoints/best_metric.pth \
    --log "$NOTE_LOG" || echo "  [note-eval] errored — continuing"
ood "stage1_gainaug_notehead_aligned" \
    "vocalcoach/runs/stage1_gainaug_notehead_aligned/checkpoints/best_metric.pth"

# ═════════════════════════════════════════════════════════════════════════════
# V1 — VocalSet SPECIALIST on AUGMENTED data (vocalset_aug, 7416 clips).
#      Fine-tune: frozen head warmup (25 ep) then gentle backbone fine-tune via
#      differential LR (backbone 1e-5, heads 3e-4) so VAD erodes slowly. Patience 12
#      so it stops near the macro-AP peak instead of running long and eroding VAD.
#      Gain-aug ON so the backbone keeps loudness robustness while adapting.
# ═════════════════════════════════════════════════════════════════════════════
log "V1 — VocalSet specialist (augmented, fine-tune)"
python vocalcoach/train.py \
    --data-dir data --technique-dirs data/vocalset_aug \
    $ARCH \
    --seq-len 600 --batch-size 64 --num-workers 0 \
    --w-vad 0.05 --w-pitch 2 --w-technique 1.5 \
    --technique-pos-weights 2.9 4.2 1.0 4.0 1.9 \
    --pitch-sigma 0.8 --augment noise_specaug --gain-aug-db 40 --p-clean 0.3 \
    --deep-technique-head \
    --freeze-backbone-epochs 25 \
    --lr 3e-4 --lr-backbone 1e-5 \
    --contrastive-technique --w-contrastive-technique 0.7 --contrastive-temp 0.07 \
    --save-best-only --epochs 70 --patience 12 \
    --resume "$GAIN" \
    --output-dir vocalcoach/runs/stage2_vocalset_aug_specialist
ood "stage2_vocalset_aug_specialist" \
    "vocalcoach/runs/stage2_vocalset_aug_specialist/checkpoints/best_metric.pth"

# ═════════════════════════════════════════════════════════════════════════════
# V0 — VocalSet baseline on UN-AUGMENTED data (data/vocalset). Identical recipe to
#      V1 except the data. This is the attribution control: V1 - V0 = the effect of
#      pitch/time augmentation on the VocalSet specialist. Without this we cannot
#      tell whether any V1 gain came from augmentation or just from fine-tuning.
# ═════════════════════════════════════════════════════════════════════════════
log "V0 — VocalSet baseline (UN-augmented, fine-tune) — attribution control"
python vocalcoach/train.py \
    --data-dir data --technique-dirs data/vocalset data/annotated_vocalset \
    $ARCH \
    --seq-len 600 --batch-size 64 --num-workers 0 \
    --w-vad 0.05 --w-pitch 2 --w-technique 1.5 \
    --technique-pos-weights 2.9 4.2 1.0 4.0 1.9 \
    --pitch-sigma 0.8 --augment noise_specaug --gain-aug-db 40 --p-clean 0.3 \
    --deep-technique-head \
    --freeze-backbone-epochs 25 \
    --lr 3e-4 --lr-backbone 1e-5 \
    --contrastive-technique --w-contrastive-technique 0.7 --contrastive-temp 0.07 \
    --save-best-only --epochs 70 --patience 12 \
    --resume "$GAIN" \
    --output-dir vocalcoach/runs/stage2_vocalset_baseline_finetune

# ═════════════════════════════════════════════════════════════════════════════
# V2 — VocalSet specialist, "JOINT-FIRST-THEN-FREEZE" variant (bounded VAD erosion).
#      Two chained runs (the proven warmjoint→probe recipe):
#        V2a: a SHORT, gentle joint phase (backbone unfrozen, low LR) — adapts the
#             backbone toward technique for just 15 epochs. VAD erodes only here.
#        V2b: resume V2a with --probe-mode (backbone RE-FROZEN) — VAD is now LOCKED
#             at the V2a level and can't erode further; the head polishes on the
#             adapted-but-fixed features.
#      Contrast with V1 (frozen→unfreeze→erode unboundedly): V2 caps the erosion.
#      Tests whether a brief backbone touch + lock beats V1's open-ended fine-tune.
#      Augmented data (vocalset_aug) so it's comparable to V1.
# ═════════════════════════════════════════════════════════════════════════════
log "V2a — VocalSet warm-joint (short gentle backbone adapt, augmented)"
python vocalcoach/train.py \
    --data-dir data --technique-dirs data/vocalset_aug \
    $ARCH \
    --seq-len 600 --batch-size 64 --num-workers 0 \
    --w-vad 0.05 --w-pitch 2 --w-technique 0.5 \
    --technique-pos-weights 2.9 4.2 1.0 4.0 1.9 \
    --pitch-sigma 0.8 --augment noise_specaug --gain-aug-db 40 --p-clean 0.3 \
    --deep-technique-head \
    --lr 1e-4 \
    --save-best-only --epochs 15 --patience 15 \
    --resume "$GAIN" \
    --output-dir vocalcoach/runs/stage2_vocalset_aug_warmjoint
ood "stage2_vocalset_aug_warmjoint" \
    "vocalcoach/runs/stage2_vocalset_aug_warmjoint/checkpoints/best_metric.pth"

log "V2b — VocalSet probe from warm-joint (backbone RE-FROZEN, head polish)"
python vocalcoach/train.py \
    --data-dir data --technique-dirs data/vocalset_aug \
    $ARCH \
    --seq-len 600 --batch-size 64 --num-workers 0 \
    --w-vad 0.05 --w-pitch 2 --w-technique 1.5 \
    --technique-pos-weights 2.9 4.2 1.0 4.0 1.9 \
    --pitch-sigma 0.8 --augment noise_specaug --gain-aug-db 40 --p-clean 0.3 \
    --probe-mode --deep-technique-head \
    --contrastive-technique --w-contrastive-technique 0.7 --contrastive-temp 0.07 \
    --save-best-only --epochs 60 --patience 20 \
    --resume vocalcoach/runs/stage2_vocalset_aug_warmjoint/checkpoints/best_metric.pth \
    --output-dir vocalcoach/runs/stage2_vocalset_aug_warmjoint_probe
ood "stage2_vocalset_aug_warmjoint_probe" \
    "vocalcoach/runs/stage2_vocalset_aug_warmjoint_probe/checkpoints/best_metric.pth"

# ═════════════════════════════════════════════════════════════════════════════
# G1 — GTSinger SPECIALIST on AUGMENTED data (gtsinger_aug, 13887 clips, falsetto
#      9288). Classes present: vibrato/breathy/falsetto (belt/straight absent).
#      Pos-weights flipped for GTSinger's distribution: falsetto is the MAJORITY,
#      so it gets weight 1.0 and vibrato/breathy are upweighted (~3.7). Same gentle
#      fine-tune recipe as V1.
# ═════════════════════════════════════════════════════════════════════════════
log "G1 — GTSinger specialist (augmented, fine-tune)"
python vocalcoach/train.py \
    --data-dir data --technique-dirs data/gtsinger_aug \
    $ARCH \
    --seq-len 600 --batch-size 64 --num-workers 0 \
    --w-vad 0.05 --w-pitch 2 --w-technique 1.5 \
    --technique-pos-weights 3.7 3.7 1.0 1.0 1.0 \
    --pitch-sigma 0.8 --augment noise_specaug --gain-aug-db 40 --p-clean 0.3 \
    --deep-technique-head \
    --freeze-backbone-epochs 25 \
    --lr 3e-4 --lr-backbone 1e-5 \
    --contrastive-technique --w-contrastive-technique 0.7 --contrastive-temp 0.07 \
    --save-best-only --epochs 60 --patience 12 \
    --resume "$GAIN" \
    --output-dir vocalcoach/runs/stage2_gtsinger_aug_specialist
ood "stage2_gtsinger_aug_specialist" \
    "vocalcoach/runs/stage2_gtsinger_aug_specialist/checkpoints/best_metric.pth"

# ═════════════════════════════════════════════════════════════════════════════
# G0 — GTSinger baseline on UN-AUGMENTED data (data/gtsinger_technique). Attribution
#      control for the GTSinger specialist (G1 - G0 = augmentation effect).
# ═════════════════════════════════════════════════════════════════════════════
log "G0 — GTSinger baseline (UN-augmented, fine-tune) — attribution control"
python vocalcoach/train.py \
    --data-dir data --technique-dirs data/gtsinger_technique \
    $ARCH \
    --seq-len 600 --batch-size 64 --num-workers 0 \
    --w-vad 0.05 --w-pitch 2 --w-technique 1.5 \
    --technique-pos-weights 3.7 3.7 1.0 1.0 1.0 \
    --pitch-sigma 0.8 --augment noise_specaug --gain-aug-db 40 --p-clean 0.3 \
    --deep-technique-head \
    --freeze-backbone-epochs 25 \
    --lr 3e-4 --lr-backbone 1e-5 \
    --contrastive-technique --w-contrastive-technique 0.7 --contrastive-temp 0.07 \
    --save-best-only --epochs 60 --patience 12 \
    --resume "$GAIN" \
    --output-dir vocalcoach/runs/stage2_gtsinger_baseline_finetune

# ═════════════════════════════════════════════════════════════════════════════
# QUALITY HEADS — all PROBES (frozen backbone, only head_quality trains), so they
# are OOD-SAFE (cannot touch VAD/pitch) and short. They resume from the N1
# note-head checkpoint: the best self-contained substrate this batch produces
# (good VAD/pitch + aligned note head). Quality data npzs are pre-extracted:
#   quality_mse.npz (SingMOS-Pro pseudo-labels), quality_ccmusic.npz (9-dim expert),
#   quality_pairs.npz (PopBuTFy pro/amateur contrastive pairs).
# ═════════════════════════════════════════════════════════════════════════════
QBASE="vocalcoach/runs/stage1_gainaug_notehead_aligned/checkpoints/best_metric.pth"
# Fallback to the gainaug backbone if N1 failed to produce a checkpoint.
[[ -f "$QBASE" ]] || QBASE="$GAIN"

# ─── Q1: Quality V3 — scalar score (SingMOS-Pro MSE distill → PopBuTFy ranking) ──
# A single "how good is this singing" score. MSE warmup on SingMOS pseudo-labels
# (30 ep) then contrastive ranking on pro/amateur pairs.
log "Q1 — Quality V3 scalar (distill + ranking), probe from N1"
python vocalcoach/train.py \
    $ARCH \
    --seq-len 600 --batch-size 64 --num-workers 0 \
    --epochs 60 --patience 20 \
    --probe-mode \
    --quality-variant 3 \
    --quality-mse-npz data/quality/quality_mse.npz \
    --quality-pairs-npz data/quality_50k/quality_pairs.npz \
    --quality-epochs-mse 30 \
    --w-quality-mse 1.0 --w-ranking 1.0 --ranking-margin 0.5 \
    --save-best-only \
    --resume "$QBASE" \
    --output-dir vocalcoach/runs/stage2_quality_v3_from_notehead

# ─── Q2: Quality V2 — 9-dim expert head (ccmusic MSE + SingMOS warmup + ranking) ─
# Pitch/Rhythm/Timbre/Breath/Vibrato/Dynamic/Pronunciation/Range/Overall →
# the radar-chart dimensions for the demo UI. Separate head shape from V3.
log "Q2 — Quality V2 9-dim ccmusic, probe from N1"
python vocalcoach/train.py \
    $ARCH \
    --seq-len 600 --batch-size 64 --num-workers 0 \
    --epochs 60 --patience 20 \
    --probe-mode \
    --quality-variant 2 \
    --quality-mse-npz data/quality/quality_mse.npz \
    --quality-ccmusic-npz data/quality/quality_ccmusic.npz \
    --quality-pairs-npz data/quality_50k/quality_pairs.npz \
    --quality-epochs-mse 30 \
    --w-quality-mse 1.0 --w-ranking 1.0 --ranking-margin 0.5 \
    --save-best-only \
    --resume "$QBASE" \
    --output-dir vocalcoach/runs/stage2_quality_v2_from_notehead

# ─── Q3: Quality V1 — scalar, pairs-only ranking (no MSE warmup) ─────────────────
# Pure contrastive ranking on PopBuTFy pairs — the simplest scalar quality head.
# Ablation vs V3: does the SingMOS MSE warmup actually help, or is ranking enough?
log "Q3 — Quality V1 scalar (pairs-only ranking), probe from N1"
python vocalcoach/train.py \
    $ARCH \
    --seq-len 600 --batch-size 64 --num-workers 0 \
    --epochs 50 --patience 20 \
    --probe-mode \
    --quality-variant 1 \
    --quality-pairs-npz data/quality_50k/quality_pairs.npz \
    --w-ranking 1.0 --ranking-margin 0.5 \
    --save-best-only \
    --resume "$QBASE" \
    --output-dir vocalcoach/runs/stage2_quality_v1_from_notehead

# ═════════════════════════════════════════════════════════════════════════════
# Results
# ═════════════════════════════════════════════════════════════════════════════
log "All runs complete — updating results table"
for spec in \
    "stage2_vocalset_aug_specialist|data/vocalset_aug" \
    "stage2_vocalset_baseline_finetune|data/vocalset data/annotated_vocalset" \
    "stage2_vocalset_aug_warmjoint_probe|data/vocalset_aug" \
    "stage2_gtsinger_aug_specialist|data/gtsinger_aug" \
    "stage2_gtsinger_baseline_finetune|data/gtsinger_technique"
do
    run="${spec%%|*}"; tdirs="${spec##*|}"
    echo "  Updating results for $run (tech=$tdirs) ..."
    python vocalcoach/update_results.py \
        --run-dir "vocalcoach/runs/$run" \
        --data-dir data \
        --technique-dirs $tdirs \
        --name "$run" 2>/dev/null || echo "  [skip] update_results failed for $run"
done

ELAPSED=$(( ($(date +%s) - START_TS) / 60 ))
log "Done in ${ELAPSED} min. Check VOCALCOACH_RESULTS.md, results/ood_log.json, results/note_log.json"

echo ""
echo "Attribution summary (did augmentation help? compare aug vs baseline macro AP):"
echo "  VocalSet: stage2_vocalset_aug_specialist  vs  stage2_vocalset_baseline_finetune"
echo "  GTSinger: stage2_gtsinger_aug_specialist  vs  stage2_gtsinger_baseline_finetune"
echo ""
echo "OOD VDR (watch for VAD erosion from fine-tuning — gainaug backbone was 67%):"
python3 -c "
import json
try:
    d=json.load(open('results/ood_log.json'))
    names=['stage1_gainaug_notehead_aligned','stage2_vocalset_aug_specialist','stage2_vocalset_aug_warmjoint_probe','stage2_gtsinger_aug_specialist']
    print(f'  {\"Run\":<42} {\"VDR\":>6} {\"vF1\":>6} {\"RPA\":>6} Status')
    print('  '+'-'*70)
    for r in d['runs']:
        if any(n in r['run_name'] for n in names):
            o=r['overall']
            print(f'  {r[\"run_name\"]:<42} {o[\"vdr\"]*100:>5.1f}% {o[\"vf1\"]*100:>5.1f}% {o[\"rpa\"]*100:>5.1f}% {r.get(\"status\",\"?\")}')
except Exception as e:
    print('  could not read ood_log:', e)
"
