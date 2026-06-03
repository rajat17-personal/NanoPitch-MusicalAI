#!/usr/bin/env bash
# validate_session_changes.sh
# ─────────────────────────────────────────────────────────────────────────────
# Targeted validation batch (~3.5h) for the changes made this session. Run from
# repo root:  bash scripts/validate_session_changes.sh 2>&1 | tee validate_$(date +%Y%m%d).log
#
# Each run tests one specific change:
#   R1  quality V3 BN-safe re-run + OOD   → confirm the BatchNorm-eval fix holds
#                                            VAD (prior run collapsed 67%→19%)
#   R2  note-head + --w-note-metric        → does selecting-FOR-note lift F1>0.765?
#   R3  spectral-tilt backbone + OOD       → does mic-coloration aug beat gainaug 67%?
#   R4  union-fusion accuracy eval         → the deployable 5-class system's accuracy
#                                            (eval-only, no training)
# ─────────────────────────────────────────────────────────────────────────────

set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"

GAIN="vocalcoach/runs/stage1_attn4_notehead_gainaug_merged_pitchvad/checkpoints/best_metric.pth"
NOTE_CKPT="vocalcoach/runs/stage1_gainaug_notehead_aligned/checkpoints/best_metric.pth"
VS_SPEC="vocalcoach/runs/stage2_vocalset_aug_warmjoint_probe/checkpoints/best_metric.pth"
GT_SPEC="vocalcoach/runs/stage2_gtsinger_aug_specialist/checkpoints/best_metric.pth"
VOCADITO="data/vocadito"; OOD_LOG="results/ood_log.json"; NOTE_LOG="results/note_log.json"
ARCH="--arch tcn --hidden 256 --n-blocks 8 --n-attn-layers 4 --n-heads 4"

log() { echo; echo "════════════════════════════════════════════"; echo "  [$(date '+%H:%M:%S')]  $*"; echo "════════════════════════════════════════════"; }
ood() { local n="$1" c="$2"; log "OOD: $n"; [[ -f "$c" ]] && python scripts/evalOOD.py --checkpoint "$c" --dataset vocadito --data-dir "$VOCADITO" --log "$OOD_LOG" --regression-threshold 0.10 || echo "  [OOD] missing/rejected — continuing"; }

# ─── R1: Quality V3 BN-safe — the decisive BatchNorm-fix test ────────────────
# Same recipe as the overnight quality run, but train.py now pins frozen BN to
# eval (no running-stat drift). OOD VDR should now hold ~67% instead of 19%.
# Resume from the GAINAUG BACKBONE (67% OOD VDR), NOT the note-head run — the
# note-head run eroded VAD to 13% (it left the VAD head + top blocks trainable on
# clean merged data). A quality probe inherits its base's VAD unchanged, so to test
# whether the quality probe preserves VAD we must start from the 67% backbone.
log "R1 — Quality V3 (BN-safe), probe from gainaug backbone"
QBASE="$GAIN"
python vocalcoach/train.py \
    $ARCH --seq-len 600 --batch-size 64 --num-workers 0 \
    --epochs 60 --patience 20 --probe-mode --quality-variant 3 \
    --quality-mse-npz data/quality/quality_mse.npz \
    --quality-pairs-npz data/quality_50k/quality_pairs.npz \
    --quality-epochs-mse 30 --w-quality-mse 1.0 --w-ranking 1.0 --ranking-margin 0.5 \
    --save-best-only --resume "$QBASE" \
    --output-dir vocalcoach/runs/stage2_quality_v3_bnsafe
# Quality saves best_loss.pth (no held-out metric). OOD-eval it to check VAD.
ood "stage2_quality_v3_bnsafe" \
    "vocalcoach/runs/stage2_quality_v3_bnsafe/checkpoints/best_loss.pth"

# ─── R2: Note head with note-F1 selection ────────────────────────────────────
# BN-safe + --w-note-metric so note F1 drives checkpoint selection (previously
# the note head was never selected for; converged at 0.765 by coincidence).
log "R2 — Note head, note-F1 selection (--w-note-metric 1.0)"
python vocalcoach/train.py \
    --data-dir data/merged_pitchvad --noise-dir data --eval-dir data \
    --note-head --deep-note-head --note-dirs data/annotated_vocalset \
    --w-note 3.0 --note-pos-weight 35 --note-batch-size 16 --w-note-metric 1.0 \
    $ARCH --seq-len 600 --batch-size 64 --num-workers 0 \
    --w-vad 0.2 --w-pitch 2 --vad-pos-weight 2.0 --pitch-sigma 0.8 \
    --augment noise_specaug --gain-aug-db 40 --p-clean 0.3 \
    --freeze-n-blocks 6 --save-best-only --epochs 60 --patience 20 --lr 3e-4 \
    --resume "$GAIN" \
    --output-dir vocalcoach/runs/stage1_notehead_noteselect
python scripts/evalNoteHead.py \
    --checkpoint vocalcoach/runs/stage1_notehead_noteselect/checkpoints/best_metric.pth \
    --log "$NOTE_LOG" || echo "  [note-eval] errored"
ood "stage1_notehead_noteselect" \
    "vocalcoach/runs/stage1_notehead_noteselect/checkpoints/best_metric.pth"

# ─── R3: Spectral-tilt backbone (new exploration) ────────────────────────────
# gainaug + spectral tilt. One new variable vs the gainaug backbone: does mic-
# coloration augmentation push OOD VDR past 67%? Same Stage-1 recipe as gainaug.
log "R3 — gainaug + spectral-tilt backbone"
python vocalcoach/train.py \
    --data-dir data/merged_pitchvad --noise-dir data --eval-dir data \
    --note-head --deep-note-head --note-dirs data/annotated_vocalset \
    --w-note 3.0 --note-pos-weight 35 --note-batch-size 16 --w-note-metric 1.0 \
    $ARCH --seq-len 600 --batch-size 64 --num-workers 0 \
    --w-vad 0.2 --w-pitch 2 --vad-pos-weight 2.0 --pitch-sigma 0.8 \
    --augment noise_specaug --gain-aug-db 40 --spec-tilt-db 8 --p-clean 0.3 \
    --freeze-n-blocks 6 --save-best-only --epochs 60 --patience 20 --lr 3e-4 \
    --resume "$GAIN" \
    --output-dir vocalcoach/runs/stage1_gainaug_spectilt
ood "stage1_gainaug_spectilt" \
    "vocalcoach/runs/stage1_gainaug_spectilt/checkpoints/best_metric.pth"

# ─── R4: Union-fusion accuracy (eval-only, no training) ──────────────────────
log "R4 — Union-fusion 5-class accuracy"
python scripts/evalFusion.py \
    --vocalset-ckpt "$VS_SPEC" --gtsinger-ckpt "$GT_SPEC" \
    --vocalset-test data/vocalset_aug/technique_test.npz \
    --gtsinger-test data/gtsinger_aug/technique_gtsinger_test.npz \
    || echo "  [fusion-eval] errored"

log "Validation batch complete. Compare:"
echo "  R1 quality VDR: should be ~67% (was 19% with BN bug) → confirms BN fix"
echo "  R2 note F1:     should be >=0.765 → confirms note-F1 selection helps"
echo "  R3 spectilt VDR: vs gainaug 67% → does spectral tilt help OOD VAD?"
echo "  R4 fusion macro AP/F1*: the deployable 5-class system baseline"
