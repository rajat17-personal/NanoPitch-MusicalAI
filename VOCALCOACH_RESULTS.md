# VocalCoach Experiment Tracker

Checkpoint reported: `best_loss.pth` — lowest training loss across all heads
(pitch×2 + vad×0.05 + technique×2). Pass `--checkpoint best_metric.pth` to
`update_results.py` to use the best eval-metric checkpoint instead (best macro F1
when technique data present, best macro RPA otherwise — can be premature when
technique F1 becomes non-zero before pitch head converges).

**Decoder note**: VocalCoach uses **offline Viterbi** (globally optimal DP over
the full clip). NanoPitch RESULTS.MD tracks **realtime (greedy) Viterbi**. Deltas
vs NanoPitch are optimistic by ~1–2% RPA due to this decoder difference, independent
of architecture. A fair comparison requires NanoPitch offline scores (not stored in RESULTS.MD).

## NanoPitch Reference Lines

Scores from `RESULTS.MD` — realtime (greedy) Viterbi. Checkpoint files no longer on disk.

| Tier | NanoPitch run | Arch | VAD Acc | rtRPA | rtVDR | rtMed¢ | Augmentation |
|---|---|---|---|---|---|---|---|
| Arch baseline | `wpitch2.0` (Run 2) | GRU-64 | 80.2 | 91.6 | 69.3 | 31.0 | none |
| Best balanced | `seq600_btch16_wVad0.05wPitch2_cosine_specaug_n10t30SNR_pitchsigma0.8` (Run 26) | GRU-64 | 81.6 | 96.1 | 65.9 | 15.8 | noise+SpecAug |

Per-condition rtRPA for reference:

| Run | -5 dB | +0 dB | +5 dB | +10 dB | +20 dB | clean | Macro |
|---|---|---|---|---|---|---|---|
| NP Run 2  (arch) | 89.0 | 89.5 | 88.5 | 92.3 | 93.9 | 96.0 | 91.6 |
| NP Run 26 (aug)  | 95.1 | 94.3 | 94.4 | 96.9 | 97.9 | 98.1 | 96.1 |

To add a completed run:

```bash
cd /path/to/NanoPitch-MusicalAI

# Pitch-only run (no technique labels)
python vocalcoach/update_results.py \
    --run-dir vocalcoach/runs/<run_name> \
    --data-dir data \
    --name <row_label> --note "what you expected"

# With technique evaluation
python vocalcoach/update_results.py \
    --run-dir vocalcoach/runs/<run_name> \
    --data-dir data \
    --technique-dir data/vocalset \
    --name <row_label> --note "what you expected"

# Delete a row (removes from all tables)
python vocalcoach/update_results.py --delete <row_label>
```

## Runs

| # | Run | Arch | Key args | Note | VAD Acc | offRPA | offVDR | offMed¢ | mF1 | mAP |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `tcn_gtsinger_noncausal_aug` | tcn/noncausal | `--seq-len`=600, `--epochs`=100, `--batch-size`=16, `--lr`=0.001, `--w-vad`=0.05, `--w-pitch`=2, `--vad-pos-weight`=1, `--pitch-sigma`=0.8, `--augment`=noise_specaug, `--patience`=0 | tcn best ckpt comparision | 81.6 | 98.5 | 72.3 | 10.2 | — | — |
| 2 | `conformer_gtsinger_noncausal_aug` | conformer/noncausal | `--arch`=conformer, `--seq-len`=600, `--epochs`=100, `--batch-size`=16, `--lr`=0.001, `--w-vad`=0.05, `--w-pitch`=2, `--vad-pos-weight`=1, `--pitch-sigma`=0.8, `--augment`=noise_specaug, `--patience`=0 | conformer best ckpt comparision | 82.2 | 99.3 | 80.4 | 2.2 | — | — |
| 3 | `Output dir ignored` | tcn/noncausal | data=GTSinger, tech=VocalSet+  | Output dir ignored | 78.2 | 92.4 | 26.7 | 76.9 | 0.810 | 0.833 |
| 4 | `conformer_vocalset_rescaled` | conformer/noncausal | data=GTSinger, tech=VocalSet, `--arch`=conformer, `--batch-size`=16, `--w-pitch`=4, `--w-technique`=0.5, `--augment`=noise_specaug | Conformer rescaled | 82.4 | 97.3 | 57.3 | 11.7 | 0.655 | 0.685 |
| 5 | `conformer_probe_technique` | conformer/noncausal | tech=vocalset, `--arch`=conformer, `--seq-len`=600, `--epochs`=50, `--batch-size`=64, `--w-vad`=0.05, `--w-pitch`=2, `--augment`=noise_specaug, `--resume`=conformer_gtsinger_noncausal_aug | conformer probe technique | 82.6 | 99.5 | 61.5 | 1.2 | 0.395 | 0.521 |
| 6 | `tcn_probe_technique_v2` | tcn/noncausal | tech=vocalset, `--seq-len`=600, `--epochs`=50, `--batch-size`=64, `--w-vad`=0.05, `--w-pitch`=2, `--augment`=noise_specaug, `--resume`=tcn_gtsinger_noncausal_aug_clean | tcn probe technique v2 | 81.4 | 97.8 | 76.9 | 16.8 | 0.374 | 0.498 |
| 7 | `conformer_probe_deep_vocalsetonly` | conformer/noncausal | tech=vocalset, `--arch`=conformer, `--seq-len`=600, `--batch-size`=64, `--w-vad`=0.05, `--w-pitch`=2, `--augment`=noise_specaug, `--resume`=conformer_gtsinger_noncausal_aug | conformer_probe_deep_vocalsetonly | 82.5 | 99.4 | 60.5 | 1.2 | 0.353 | 0.537 |
| 8 | `tcn_causal_live` | tcn/causal | data=GTSinger, `--causal`=on, `--seq-len`=600, `--epochs`=50, `--batch-size`=128, `--w-vad`=0.05, `--w-pitch`=2, `--vad-pos-weight`=1, `--technique-pos-weights`=2.9 4.2 1 4 1.9, `--pitch-sigma`=0.8, `--augment`=noise_specaug, `--resume`=tcn_pitch_base_v2 | tcn causal live | 80.7 | 97.0 | 56.0 | 21.8 | 0.412 | 0.245 |
| 9 | `stage1_pitchonly_A` | conformer/noncausal | data=GTSinger, `--arch`=conformer, `--seq-len`=600, `--epochs`=120, `--lr`=0.001, `--w-vad`=0.05, `--w-pitch`=2, `--technique-pos-weights`=2.9 4.2 1 4 1.9, `--pitch-sigma`=0.8, `--augment`=noise_specaug, `--patience`=30, `--resume`=stage1_pitchonly_A | conformer pitch+VAD only, GTSinger, augmented — Stage 1A backbone | 82.1 | 99.0 | 85.8 | 2.5 | — | — |
| 10 | `stage1_pitchonly_B` | conformer/noncausal | data=GTSinger, tech=VocalSet+annotated_vocalset, `--arch`=conformer, `--seq-len`=600, `--epochs`=120, `--lr`=0.001, `--w-vad`=0.05, `--w-pitch`=2, `--w-technique`=0, `--technique-pos-weights`=2.9 4.2 1 4 1.9, `--pitch-sigma`=0.8, `--augment`=noise_specaug, `--patience`=20, `--resume`=stage1_pitchonly_B | conformer pitch+VAD, GTSinger+VocalSet+AnnotatedVocalSet, augmented — Stage 1B backbone | 80.8 | 99.2 | 76.4 | 2.4 | — | — |
| 11 | `stage2_probe_A` | conformer/noncausal | data=GTSinger, tech=VocalSet+annotated_vocalset, `--arch`=conformer, `--seq-len`=600, `--epochs`=80, `--batch-size`=64, `--probe-mode`=on, `--w-vad`=0.05, `--w-pitch`=2, `--technique-pos-weights`=2.9 4.2 1 4 1.9, `--pitch-sigma`=0.8, `--augment`=noise_specaug, `--patience`=20, `--resume`=stage1_pitchonly_A, `--onset-penalty`=1.0 | MERT probe off 1A — VocalSet+AnnotatedVocalSet technique head | 81.8 | 99.1 | 86.7 | 1.5 | 0.547 | 0.518 |
| 12 | `stage2_warmjoint` | conformer/noncausal | data=GTSinger, tech=VocalSet+annotated_vocalset, `--arch`=conformer, `--seq-len`=600, `--epochs`=15, `--batch-size`=64, `--lr`=0.0001, `--w-vad`=0.05, `--w-pitch`=2, `--w-technique`=0.5, `--technique-pos-weights`=2.9 4.2 1 4 1.9, `--pitch-sigma`=0.8, `--augment`=noise_specaug, `--resume`=stage1_pitchonly_A, `--onset-penalty`=1.0 | warm joint 15ep off 1A — VocalSet+AnnotatedVocalSet | 82.3 | 99.1 | 82.4 | 3.4 | 0.690 | 0.700 |
| 13 | `stage2_probe_B` | conformer/noncausal | data=GTSinger, tech=VocalSet+annotated_vocalset, `--arch`=conformer, `--seq-len`=600, `--epochs`=80, `--batch-size`=64, `--probe-mode`=on, `--w-vad`=0.05, `--w-pitch`=2, `--technique-pos-weights`=2.9 4.2 1 4 1.9, `--pitch-sigma`=0.8, `--augment`=noise_specaug, `--patience`=20, `--resume`=stage2_warmjoint, `--onset-penalty`=1.0 | warm joint 15ep then probe off 1A — VocalSet+AnnotatedVocalSet | 81.5 | 99.0 | 79.9 | 3.2 | 0.676 | 0.671 |
| 14 | `stage2_joint_lowlr` | conformer/noncausal | data=GTSinger, tech=VocalSet+annotated_vocalset, `--arch`=conformer, `--seq-len`=600, `--epochs`=60, `--batch-size`=64, `--lr`=3e-05, `--w-vad`=0.05, `--w-pitch`=2, `--w-technique`=1, `--technique-pos-weights`=2.9 4.2 1 4 1.9, `--pitch-sigma`=0.8, `--augment`=noise_specaug, `--patience`=20, `--resume`=stage1_pitchonly_A, `--onset-penalty`=1.0 | Stage 2 Joint low lr off stage1_pitchonly_A | 82.3 | 99.0 | 79.4 | 3.4 | 0.670 | 0.604 |
| 15 | `stage2_joint_from_warmjoint` | conformer/noncausal | data=GTSinger, tech=VocalSet+annotated_vocalset, `--arch`=conformer, `--seq-len`=600, `--epochs`=60, `--batch-size`=64, `--lr`=3e-05, `--w-vad`=0.05, `--w-pitch`=2, `--w-technique`=1, `--technique-pos-weights`=2.9 4.2 1 4 1.9, `--pitch-sigma`=0.8, `--augment`=noise_specaug, `--patience`=20, `--resume`=stage2_warmjoint, `--onset-penalty`=1.0 | Stage 2 Joint off stage2_warmjoint | 82.3 | 99.0 | 80.2 | 3.6 | 0.681 | 0.705 |
| 16 | `stage2_joint_lowlr_v2` | conformer/noncausal | data=GTSinger, tech=VocalSet+annotated_vocalset, `--arch`=conformer, `--seq-len`=600, `--epochs`=80, `--batch-size`=64, `--lr`=3e-05, `--w-vad`=0.05, `--w-pitch`=2, `--w-technique`=1, `--technique-pos-weights`=2.9 4.2 1 4 1.9, `--pitch-sigma`=0.8, `--augment`=noise_specaug, `--patience`=20, `--resume`=stage2_joint_lowlr, `--onset-penalty`=1.0 | Stage 2 Joint low lr v2 off stage2_joint_lowlr | 82.5 | 99.2 | 79.3 | 3.1 | 0.678 | 0.691 |
| 17 | `stage2_joint_1e5` | conformer/noncausal | data=GTSinger, tech=VocalSet+annotated_vocalset, `--arch`=conformer, `--seq-len`=600, `--epochs`=60, `--batch-size`=64, `--lr`=1e-05, `--w-vad`=0.05, `--w-pitch`=2, `--w-technique`=1, `--technique-pos-weights`=2.9 4.2 1 4 1.9, `--pitch-sigma`=0.8, `--augment`=noise_specaug, `--patience`=20, `--resume`=stage2_joint_lowlr, `--onset-penalty`=1.0 | Stage 2 Joint 1e5 off stage2_joint_lowlr | 82.4 | 99.2 | 77.1 | 3.7 | 0.662 | 0.613 |
| 18 | `stage2_joint_difflr` | conformer/noncausal | data=GTSinger, tech=VocalSet+annotated_vocalset, `--arch`=conformer, `--seq-len`=600, `--epochs`=60, `--batch-size`=64, `--w-vad`=0.05, `--w-pitch`=2, `--w-technique`=1, `--technique-pos-weights`=2.9 4.2 1 4 1.9, `--pitch-sigma`=0.8, `--augment`=noise_specaug, `--patience`=20, `--resume`=stage2_joint_lowlr, `--onset-penalty`=1.0 | Stage 2 Joint difflr off stage2_joint_lowlr | 82.6 | 99.2 | 78.9 | 2.7 | 0.656 | 0.588 |
| 19 | `stage2_joint_difflr_v2` | conformer/noncausal | data=GTSinger, tech=VocalSet+annotated_vocalset, `--arch`=conformer, `--seq-len`=600, `--epochs`=80, `--batch-size`=64, `--w-vad`=0.05, `--w-pitch`=2, `--w-technique`=1, `--technique-pos-weights`=2.9 4.2 1 4 1.9, `--pitch-sigma`=0.8, `--augment`=noise_specaug, `--patience`=25, `--resume`=stage2_joint_difflr, `--onset-penalty`=1.0 | Stage 2 Joint difflr v2 off stage2_joint_difflr | 81.5 | 99.2 | 77.6 | 2.5 | 0.725 | 0.681 |
| 20 | `stage2_joint_difflr_v3` | conformer/noncausal | data=GTSinger, tech=VocalSet+annotated_vocalset, `--arch`=conformer, `--seq-len`=600, `--epochs`=60, `--batch-size`=64, `--w-vad`=0.05, `--w-pitch`=2, `--w-technique`=1, `--technique-pos-weights`=2.9 4.2 1 4 1.9, `--pitch-sigma`=0.8, `--augment`=noise_specaug, `--patience`=25, `--resume`=stage2_joint_difflr_v2, `--onset-penalty`=1.0 | Stage 2 Joint difflr v3 off stage2_joint_difflr_v2 | 81.9 | 99.2 | 77.6 | 2.3 | 0.726 | 0.689 |
| 21 | `stage2_joint_difflr_v4` | conformer/noncausal | data=GTSinger, tech=VocalSet+annotated_vocalset, `--arch`=conformer, `--seq-len`=600, `--epochs`=60, `--batch-size`=64, `--w-vad`=0.05, `--w-pitch`=2, `--w-technique`=1, `--technique-pos-weights`=2.9 4.2 1 4 1.9, `--pitch-sigma`=0.8, `--augment`=noise_specaug, `--patience`=25, `--resume`=stage2_joint_difflr_v3, `--onset-penalty`=1.0 | Stage 2 Joint difflr v4 off stage2_joint_difflr_v3 | 81.9 | 99.2 | 77.4 | 2.3 | 0.715 | 0.678 |
| 22 | `stage2_gtsinger_supcon` | conformer/noncausal | data=GTSinger, tech=VocalSet+annotated_vocalset+GTSinger-tech, `--arch`=conformer, `--seq-len`=600, `--epochs`=50, `--batch-size`=64, `--lr`=0.0001, `--w-vad`=0.05, `--w-pitch`=2, `--w-technique`=0.5, `--technique-pos-weights`=2.9 4.2 1 4 1.9, `--pitch-sigma`=0.8, `--augment`=noise_specaug, `--patience`=20, `--resume`=stage2_joint_difflr_v3, `--onset-penalty`=1.0 | Stage 2 gtsinger contrastive off stage2_joint_difflr_v3 | 81.7 | 98.8 | 71.8 | 3.6 | 0.643 | 0.537 |
| 23 | `stage1_conformer_96` | conformer/noncausal | data=GTSinger, `--arch`=conformer, `--hidden`=96, `--seq-len`=600, `--epochs`=150, `--batch-size`=64, `--lr`=0.001, `--w-vad`=0.05, `--w-pitch`=2, `--technique-pos-weights`=2.9 4.2 1 4 1.9, `--pitch-sigma`=0.8, `--augment`=noise_specaug, `--patience`=30, `--onset-penalty`=1.0 | stage1_conformer_96 | 81.7 | 99.1 | 88.3 | 2.3 | 0.397 | 0.204 |
| 24 | `stage1_conformer_128` | conformer/noncausal | data=GTSinger, `--arch`=conformer, `--hidden`=128, `--seq-len`=600, `--epochs`=150, `--batch-size`=64, `--lr`=0.001, `--w-vad`=0.05, `--w-pitch`=2, `--technique-pos-weights`=2.9 4.2 1 4 1.9, `--pitch-sigma`=0.8, `--augment`=noise_specaug, `--patience`=30, `--onset-penalty`=1.0 | stage1_conformer_128 | 81.9 | 99.4 | 89.3 | 1.3 | 0.412 | 0.279 |

## Leaderboard

_(Re-sorted on every update: by macro F1 when technique data present, else macro RPA.
Primary sort column is the last numeric column.)_

| # | Run | Arch | VAD Acc | offRPA | offVDR | offMed¢ | mF1 | mAP | Primary |
|---|---|---|---|---|---|---|---|---|---|
| 2 | `conformer_gtsinger_noncausal_aug` | conformer/noncausal | 82.2 | 99.3 | 80.4 | 2.2 | — | — | 99.3000 |
| 10 | `stage1_pitchonly_B` | conformer/noncausal | 80.8 | 99.2 | 76.4 | 2.4 | — | — | 99.2000 |
| 9 | `stage1_pitchonly_A` | conformer/noncausal | 82.1 | 99.0 | 85.8 | 2.5 | — | — | 99.0000 |
| 1 | `tcn_gtsinger_noncausal_aug` | tcn/noncausal | 81.6 | 98.5 | 72.3 | 10.2 | — | — | 98.5000 |
| 3 | `Output dir ignored` | tcn/noncausal | 78.2 | 92.4 | 26.7 | 76.9 | 0.810 | 0.833 | 0.8100 |
| 20 | `stage2_joint_difflr_v3` | conformer/noncausal | 81.9 | 99.2 | 77.6 | 2.3 | 0.726 | 0.689 | 0.7260 |
| 19 | `stage2_joint_difflr_v2` | conformer/noncausal | 81.5 | 99.2 | 77.6 | 2.5 | 0.725 | 0.681 | 0.7250 |
| 21 | `stage2_joint_difflr_v4` | conformer/noncausal | 81.9 | 99.2 | 77.4 | 2.3 | 0.715 | 0.678 | 0.7150 |
| 12 | `stage2_warmjoint` | conformer/noncausal | 82.3 | 99.1 | 82.4 | 3.4 | 0.690 | 0.700 | 0.6900 |
| 15 | `stage2_joint_from_warmjoint` | conformer/noncausal | 82.3 | 99.0 | 80.2 | 3.6 | 0.681 | 0.705 | 0.6810 |
| 16 | `stage2_joint_lowlr_v2` | conformer/noncausal | 82.5 | 99.2 | 79.3 | 3.1 | 0.678 | 0.691 | 0.6780 |
| 13 | `stage2_probe_B` | conformer/noncausal | 81.5 | 99.0 | 79.9 | 3.2 | 0.676 | 0.671 | 0.6760 |
| 14 | `stage2_joint_lowlr` | conformer/noncausal | 82.3 | 99.0 | 79.4 | 3.4 | 0.670 | 0.604 | 0.6700 |
| 17 | `stage2_joint_1e5` | conformer/noncausal | 82.4 | 99.2 | 77.1 | 3.7 | 0.662 | 0.613 | 0.6620 |
| 18 | `stage2_joint_difflr` | conformer/noncausal | 82.6 | 99.2 | 78.9 | 2.7 | 0.656 | 0.588 | 0.6560 |
| 4 | `conformer_vocalset_rescaled` | conformer/noncausal | 82.4 | 97.3 | 57.3 | 11.7 | 0.655 | 0.685 | 0.6550 |
| 22 | `stage2_gtsinger_supcon` | conformer/noncausal | 81.7 | 98.8 | 71.8 | 3.6 | 0.643 | 0.537 | 0.6430 |
| 11 | `stage2_probe_A` | conformer/noncausal | 81.8 | 99.1 | 86.7 | 1.5 | 0.547 | 0.518 | 0.5470 |
| 8 | `tcn_causal_live` | tcn/causal | 80.7 | 97.0 | 56.0 | 21.8 | 0.412 | 0.245 | 0.4120 |
| 24 | `stage1_conformer_128` | conformer/noncausal | 81.9 | 99.4 | 89.3 | 1.3 | 0.412 | 0.279 | 0.4120 |
| 23 | `stage1_conformer_96` | conformer/noncausal | 81.7 | 99.1 | 88.3 | 2.3 | 0.397 | 0.204 | 0.3970 |
| 5 | `conformer_probe_technique` | conformer/noncausal | 82.6 | 99.5 | 61.5 | 1.2 | 0.395 | 0.521 | 0.3950 |
| 6 | `tcn_probe_technique_v2` | tcn/noncausal | 81.4 | 97.8 | 76.9 | 16.8 | 0.374 | 0.498 | 0.3740 |
| 7 | `conformer_probe_deep_vocalsetonly` | conformer/noncausal | 82.5 | 99.4 | 60.5 | 1.2 | 0.353 | 0.537 | 0.3530 |

## Per-condition offRPA

_(Sorted by Macro RPA descending — offline Viterbi. NanoPitch reference rows use realtime Viterbi; see NanoPitch Reference Lines above.)_

| # | Run | -5 dB | +0 dB | +5 dB | +10 dB | +20 dB | clean | Macro |
|---|---|---|---|---|---|---|---|---|
| 5 | `conformer_probe_technique` | 99.0 | 99.0 | 99.6 | 99.7 | 99.6 | 99.8 | 99.5 |
| 7 | `conformer_probe_deep_vocalsetonly` | 99.0 | 99.0 | 99.6 | 99.8 | 99.5 | 99.8 | 99.4 |
| 24 | `stage1_conformer_128` | 99.2 | 99.2 | 99.3 | 99.5 | 99.6 | 99.6 | 99.4 |
| 2 | `conformer_gtsinger_noncausal_aug` | 99.2 | 99.2 | 99.2 | 99.5 | 99.4 | 99.5 | 99.3 |
| 10 | `stage1_pitchonly_B` | 99.1 | 99.0 | 99.4 | 99.3 | 99.2 | 99.2 | 99.2 |
| 16 | `stage2_joint_lowlr_v2` | 99.3 | 99.4 | 99.1 | 99.2 | 99.0 | 99.0 | 99.2 |
| 17 | `stage2_joint_1e5` | 99.4 | 99.1 | 98.9 | 99.4 | 99.2 | 99.0 | 99.2 |
| 18 | `stage2_joint_difflr` | 99.3 | 99.2 | 99.0 | 99.4 | 99.2 | 99.0 | 99.2 |
| 19 | `stage2_joint_difflr_v2` | 99.3 | 99.1 | 99.3 | 99.3 | 99.0 | 99.3 | 99.2 |
| 20 | `stage2_joint_difflr_v3` | 99.4 | 99.2 | 99.3 | 99.2 | 99.0 | 99.4 | 99.2 |
| 21 | `stage2_joint_difflr_v4` | 99.5 | 99.2 | 99.4 | 99.1 | 98.8 | 99.3 | 99.2 |
| 11 | `stage2_probe_A` | 99.0 | 98.9 | 98.8 | 99.2 | 99.2 | 99.5 | 99.1 |
| 12 | `stage2_warmjoint` | 99.2 | 99.2 | 99.0 | 99.2 | 99.0 | 99.2 | 99.1 |
| 23 | `stage1_conformer_96` | 99.1 | 98.0 | 99.3 | 99.3 | 99.2 | 99.6 | 99.1 |
| 9 | `stage1_pitchonly_A` | 99.0 | 98.1 | 99.1 | 99.2 | 99.2 | 99.4 | 99.0 |
| 13 | `stage2_probe_B` | 99.1 | 98.9 | 99.4 | 98.9 | 98.8 | 98.9 | 99.0 |
| 14 | `stage2_joint_lowlr` | 99.3 | 99.1 | 98.8 | 99.2 | 98.7 | 98.9 | 99.0 |
| 15 | `stage2_joint_from_warmjoint` | 99.1 | 99.1 | 99.1 | 99.0 | 98.7 | 99.1 | 99.0 |
| 22 | `stage2_gtsinger_supcon` | 98.6 | 98.2 | 99.1 | 98.8 | 99.1 | 98.9 | 98.8 |
| 1 | `tcn_gtsinger_noncausal_aug` | 97.4 | 97.0 | 99.2 | 99.1 | 99.2 | 99.1 | 98.5 |
| 6 | `tcn_probe_technique_v2` | 97.5 | 95.6 | 97.7 | 99.0 | 98.1 | 98.9 | 97.8 |
| 4 | `conformer_vocalset_rescaled` | 97.6 | 96.2 | 96.8 | 97.5 | 97.8 | 98.1 | 97.3 |
| 8 | `tcn_causal_live` | 94.5 | 96.3 | 97.1 | 98.7 | 96.6 | 98.9 | 97.0 |
| 3 | `Output dir ignored` | 84.5 | 90.5 | 92.2 | 91.7 | 97.3 | 98.1 | 92.4 |

## Per-technique F1

_(Sorted by macro F1 descending. Evaluated on VocalSet held-out set. `—` = no test clips for that class. Clip Acc = argmax accuracy matching MuQ/AST SOTA metric.)_

| # | Run | vibrato | breathy | falsetto | belt | straight | macro F1 | clip acc |
|---|---|---|---|---|---|---|---|---|
| 3 | `Output dir ignored` | 0.821 | 0.919 | — | 0.723 | 0.779 | 0.810 | 80.3% |
| 20 | `stage2_joint_difflr_v3` | 0.932 | 0.615 | — | 0.658 | 0.700 | 0.726 | 62.6% |
| 19 | `stage2_joint_difflr_v2` | 0.932 | 0.606 | — | 0.658 | 0.706 | 0.725 | 64.5% |
| 21 | `stage2_joint_difflr_v4` | 0.932 | 0.609 | — | 0.615 | 0.703 | 0.715 | 61.6% |
| 12 | `stage2_warmjoint` | 0.865 | 0.648 | — | 0.566 | 0.680 | 0.690 | 60.6% |
| 15 | `stage2_joint_from_warmjoint` | 0.918 | 0.639 | — | 0.506 | 0.660 | 0.681 | 61.1% |
| 16 | `stage2_joint_lowlr_v2` | 0.839 | 0.714 | — | 0.494 | 0.664 | 0.678 | 66.5% |
| 13 | `stage2_probe_B` | 0.855 | 0.613 | — | 0.538 | 0.697 | 0.676 | 62.1% |
| 14 | `stage2_joint_lowlr` | 0.885 | 0.655 | — | 0.455 | 0.683 | 0.670 | 57.1% |
| 17 | `stage2_joint_1e5` | 0.857 | 0.615 | — | 0.504 | 0.673 | 0.662 | 59.1% |
| 18 | `stage2_joint_difflr` | 0.885 | 0.627 | — | 0.442 | 0.670 | 0.656 | 55.2% |
| 4 | `conformer_vocalset_rescaled` | 0.944 | 0.506 | — | 0.515 | 0.656 | 0.655 | 63.5% |
| 22 | `stage2_gtsinger_supcon` | 0.836 | 0.673 | — | 0.415 | 0.649 | 0.643 | 50.2% |
| 11 | `stage2_probe_A` | 0.545 | 0.701 | — | 0.386 | 0.556 | 0.547 | 43.8% |
| 8 | `tcn_causal_live` | 0.411 | 0.377 | — | 0.329 | 0.529 | 0.412 | 19.7% |
| 24 | `stage1_conformer_128` | 0.450 | 0.338 | — | 0.336 | 0.524 | 0.412 | 27.6% |
| 23 | `stage1_conformer_96` | 0.405 | 0.331 | — | 0.329 | 0.524 | 0.397 | 19.7% |
| 5 | `conformer_probe_technique` | 0.438 | 0.620 | — | 0.000 | 0.524 | 0.395 | 40.9% |
| 6 | `tcn_probe_technique_v2` | 0.436 | 0.487 | — | 0.049 | 0.524 | 0.374 | 36.5% |
| 7 | `conformer_probe_deep_vocalsetonly` | 0.444 | 0.442 | — | 0.000 | 0.526 | 0.353 | 45.8% |

## Per-technique F1 (GTSinger held-out)

_(Sorted by macro F1 descending. Only runs evaluated against GTSinger technique test set. Classes: vibrato/breathy/falsetto only — belt/straight absent from GTSinger.)_

| # | Run | vibrato | breathy | falsetto | macro F1 | clip acc |
|---|---|---|---|---|---|---|
| 22 | `stage2_gtsinger_supcon` | 0.304 | 0.336 | 0.820 | 0.486 | 59.3% |
| 8 | `tcn_causal_live` | 0.248 | 0.311 | 0.820 | 0.460 | 0.0% |
| 24 | `stage1_conformer_128` | 0.220 | 0.328 | 0.820 | 0.456 | 1.6% |
| 23 | `stage1_conformer_96` | 0.233 | 0.311 | 0.820 | 0.455 | 0.0% |
| 17 | `stage2_joint_1e5` | 0.285 | 0.316 | 0.000 | 0.200 | 3.3% |
| 13 | `stage2_probe_B` | 0.243 | 0.325 | 0.000 | 0.189 | 2.0% |
| 14 | `stage2_joint_lowlr` | 0.245 | 0.315 | 0.000 | 0.187 | 5.0% |
| 16 | `stage2_joint_lowlr_v2` | 0.233 | 0.327 | 0.000 | 0.187 | 1.5% |
| 18 | `stage2_joint_difflr` | 0.247 | 0.313 | 0.000 | 0.187 | 2.4% |
| 11 | `stage2_probe_A` | 0.218 | 0.331 | 0.000 | 0.183 | 6.8% |
| 15 | `stage2_joint_from_warmjoint` | 0.222 | 0.323 | 0.000 | 0.182 | 0.4% |
| 12 | `stage2_warmjoint` | 0.227 | 0.315 | 0.000 | 0.180 | 2.3% |
| 20 | `stage2_joint_difflr_v3` | 0.223 | 0.317 | 0.000 | 0.180 | 5.7% |
| 21 | `stage2_joint_difflr_v4` | 0.223 | 0.314 | 0.000 | 0.179 | 5.2% |
| 19 | `stage2_joint_difflr_v2` | 0.216 | 0.316 | 0.000 | 0.177 | 3.5% |
| 7 | `conformer_probe_deep_vocalsetonly` | 0.193 | 0.237 | 0.000 | 0.143 | 5.5% |

## Change log

_(fill in as runs complete)_
