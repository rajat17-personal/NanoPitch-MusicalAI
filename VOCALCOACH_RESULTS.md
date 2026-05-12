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
| 1 | `tcn_gtsinger_noncausal` | tcn/noncausal | `--seq-len`=600, `--epochs`=100, `--batch-size`=16, `--lr`=0.001, `--w-vad`=0.05, `--w-pitch`=2, `--vad-pos-weight`=1, `--pitch-sigma`=0.8, `--patience`=0 | TCN arch comparision | 78.5 | 96.3 | 75.4 | 17.6 | — | — |
| 2 | `conformer_gtsinger_noncausal` | conformer/noncausal | `--arch`=conformer, `--seq-len`=600, `--epochs`=100, `--batch-size`=16, `--lr`=0.001, `--w-vad`=0.05, `--w-pitch`=2, `--vad-pos-weight`=1, `--pitch-sigma`=0.8, `--patience`=0 | conformer arch comparision | 78.9 | 97.1 | 77.7 | 11.7 | — | — |
| 3 | `tcn_gtsinger_noncausal_aug` | tcn/noncausal | `--seq-len`=600, `--epochs`=100, `--batch-size`=16, `--lr`=0.001, `--w-vad`=0.05, `--w-pitch`=2, `--vad-pos-weight`=1, `--pitch-sigma`=0.8, `--augment`=noise_specaug, `--patience`=0 | tcn best ckpt comparision | 81.6 | 98.5 | 72.3 | 10.2 | — | — |
| 4 | `conformer_gtsinger_noncausal_aug` | conformer/noncausal | `--arch`=conformer, `--seq-len`=600, `--epochs`=100, `--batch-size`=16, `--lr`=0.001, `--w-vad`=0.05, `--w-pitch`=2, `--vad-pos-weight`=1, `--pitch-sigma`=0.8, `--augment`=noise_specaug, `--patience`=0 | conformer best ckpt comparision | 82.2 | 99.3 | 80.4 | 2.2 | — | — |
| 5 | `tcn_vocalset_gtsinger_noncausal_aug_noTechnique` | tcn/noncausal | `--seq-len`=600, `--epochs`=100, `--batch-size`=16, `--lr`=0.001, `--w-vad`=0.05, `--w-pitch`=2, `--vad-pos-weight`=1, `--pitch-sigma`=0.8, `--augment`=noise_specaug, `--patience`=0 | _tbd_ | 81.3 | 99.1 | 59.9 | 3.4 | — | — |
| 6 | `conformer_vocalset_gtsinger_noncausal_aug_noTechnique` | conformer/noncausal | `--arch`=conformer, `--seq-len`=600, `--epochs`=100, `--batch-size`=16, `--lr`=0.001, `--w-vad`=0.05, `--w-pitch`=2, `--vad-pos-weight`=1, `--pitch-sigma`=0.8, `--augment`=noise_specaug, `--patience`=0 | _tbd_ | 81.5 | 99.3 | 64.2 | 2.5 | — | — |
| 7 | `tcn_vocalset_gtsinger_noncausal_aug` | tcn/noncausal | `--seq-len`=600, `--epochs`=100, `--batch-size`=16, `--lr`=0.001, `--w-vad`=0.05, `--w-pitch`=2, `--vad-pos-weight`=1, `--pitch-sigma`=0.8, `--augment`=noise_specaug, `--patience`=0 | _tbd_ | 66.4 | nan | 0.0 | nan | 0.326 | 0.653 |
| 8 | `tcn_vocalset_gtsinger_noncausal_aug_r8` | tcn/noncausal | data=GTSinger, tech=VocalSet+GTSinger-tech, `--batch-size`=16, `--lr`=0.001, `--w-vad`=0.05, `--w-pitch`=2, `--vad-pos-weight`=1, `--pitch-sigma`=0.8, `--augment`=noise_specaug | _tbd_ | 81.1 | 99.0 | 9.3 | 12.2 | 0.223 | 0.729 |
| 9 | `conformer_vocalset_gtsinger_noncausal_aug_r10` | conformer/noncausal | data=GTSinger, tech=VocalSet+GTSinger-tech, `--arch`=conformer, `--epochs`=50, `--batch-size`=16, `--lr`=0.001, `--w-vad`=0.05, `--w-pitch`=2, `--vad-pos-weight`=1, `--pitch-sigma`=0.8, `--augment`=noise_specaug | _tbd_ | 87.1 | nan | 0.0 | nan | 0.649 | 0.728 |
| 10 | `Output dir ignored` | tcn/noncausal | data=GTSinger, tech=VocalSet+  | Output dir ignored | 78.2 | 92.4 | 26.7 | 76.9 | 0.810 | 0.833 |
| 11 | `tcn_vocalset_only_curriculum` | tcn/noncausal | data=GTSinger, tech=VocalSet, `--batch-size`=16, `--lr`=0.001, `--w-vad`=0.05, `--w-pitch`=2, `--vad-pos-weight`=1, `--pitch-sigma`=0.8, `--augment`=noise_specaug, `--voicing-threshold`=0.15 | curriculum+pos_weights, threshold=0.15 | 81.5 | 99.3 | 20.4 | 10.3 | 0.626 | 0.763 |
| 12 | `conformer_vocalset_rescaled` | conformer/noncausal | data=GTSinger, tech=VocalSet, `--arch`=conformer, `--batch-size`=16, `--w-pitch`=4, `--w-technique`=0.5, `--augment`=noise_specaug | Conformer rescaled | 82.4 | 97.3 | 57.3 | 11.7 | 0.655 | 0.685 |
| 13 | `tcn_stage1_pitchonly` | tcn/noncausal | data=GTSinger, `--seq-len`=600, `--epochs`=50, `--batch-size`=128, `--w-vad`=0.05, `--w-pitch`=2, `--augment`=noise_specaug | tcn two-stage stage1, freeze-backbone-epochs=20 | 82.8 | 95.6 | 53.4 | 31.5 | 0.265 | 0.296 |
| 14 | `tcn_stage2_technique` | tcn/noncausal | data=GTSinger, tech=vocalset, `--seq-len`=600, `--batch-size`=128, `--w-vad`=0.05, `--w-pitch`=2, `--augment`=noise_specaug, `--resume`=tcn_stage1_pitchonly | tcn two-stage stage2, freeze-backbone-epochs=20 | 81.6 | 97.2 | 35.4 | 18.8 | 0.576 | 0.675 |
| 15 | `conformer_stage1_pitchonly` | conformer/noncausal | data=GTSinger, `--arch`=conformer, `--seq-len`=600, `--epochs`=50, `--batch-size`=128, `--w-vad`=0.05, `--w-pitch`=2, `--augment`=noise_specaug | conformer two-stage stage1, freeze-backbone-epochs=20 | 81.8 | 97.0 | 67.5 | 17.0 | 0.183 | 0.203 |
| 16 | `conformer_stage2_technique` | conformer/noncausal | data=GTSinger, tech=vocalset, `--arch`=conformer, `--seq-len`=600, `--batch-size`=128, `--w-vad`=0.05, `--w-pitch`=2, `--augment`=noise_specaug, `--resume`=conformer_stage1_pitchonly | conformer two-stage stage2, freeze-backbone-epochs=20 | 83.6 | 96.2 | 32.7 | 17.6 | 0.632 | 0.683 |
| 17 | `tcn_technique` | tcn/noncausal | tech=vocalset, `--seq-len`=600, `--batch-size`=64, `--w-vad`=0.05, `--w-pitch`=2, `--augment`=noise_specaug | tcn vocal set technique only | 65.8 | 0.0 | 0.1 | 418.4 | 0.444 | 0.575 |
| 18 | `tcn_technique_gtsinger` | tcn/noncausal | tech=vocalCoach, `--seq-len`=600, `--batch-size`=128, `--w-vad`=0.05, `--w-pitch`=2, `--augment`=noise_specaug | tcn gtsinger technique only | 67.2 | nan | 0.0 | nan | 0.226 | 0.411 |
| 19 | `conformer_probe_technique` | conformer/noncausal | tech=vocalset, `--arch`=conformer, `--seq-len`=600, `--epochs`=50, `--batch-size`=64, `--w-vad`=0.05, `--w-pitch`=2, `--augment`=noise_specaug, `--resume`=conformer_gtsinger_noncausal_aug | conformer probe technique | 82.6 | 99.5 | 61.5 | 1.2 | 0.395 | 0.521 |
| 20 | `tcn_probe_technique` | tcn/noncausal | tech=vocalset, `--seq-len`=600, `--epochs`=50, `--batch-size`=64, `--w-vad`=0.05, `--w-pitch`=2, `--augment`=noise_specaug, `--resume`=tcn_gtsinger_noncausal_aug | tcn probe technique | 66.4 | nan | 0.0 | nan | 0.583 | 0.710 |
| 21 | `tcn_probe_technique_v2` | tcn/noncausal | tech=vocalset, `--seq-len`=600, `--epochs`=50, `--batch-size`=64, `--w-vad`=0.05, `--w-pitch`=2, `--augment`=noise_specaug, `--resume`=tcn_gtsinger_noncausal_aug_clean | tcn probe technique v2 | 81.4 | 97.8 | 76.9 | 16.8 | 0.374 | 0.498 |

## Leaderboard

_(Re-sorted on every update: by macro F1 when technique data present, else macro RPA.
Primary sort column is the last numeric column.)_

| # | Run | Arch | VAD Acc | offRPA | offVDR | offMed¢ | mF1 | mAP | Primary |
|---|---|---|---|---|---|---|---|---|---|
| 4 | `conformer_gtsinger_noncausal_aug` | conformer/noncausal | 82.2 | 99.3 | 80.4 | 2.2 | — | — | 99.3000 |
| 6 | `conformer_vocalset_gtsinger_noncausal_aug_noTechnique` | conformer/noncausal | 81.5 | 99.3 | 64.2 | 2.5 | — | — | 99.3000 |
| 5 | `tcn_vocalset_gtsinger_noncausal_aug_noTechnique` | tcn/noncausal | 81.3 | 99.1 | 59.9 | 3.4 | — | — | 99.1000 |
| 3 | `tcn_gtsinger_noncausal_aug` | tcn/noncausal | 81.6 | 98.5 | 72.3 | 10.2 | — | — | 98.5000 |
| 2 | `conformer_gtsinger_noncausal` | conformer/noncausal | 78.9 | 97.1 | 77.7 | 11.7 | — | — | 97.1000 |
| 1 | `tcn_gtsinger_noncausal` | tcn/noncausal | 78.5 | 96.3 | 75.4 | 17.6 | — | — | 96.3000 |
| 10 | `Output dir ignored` | tcn/noncausal | 78.2 | 92.4 | 26.7 | 76.9 | 0.810 | 0.833 | 0.8100 |
| 12 | `conformer_vocalset_rescaled` | conformer/noncausal | 82.4 | 97.3 | 57.3 | 11.7 | 0.655 | 0.685 | 0.6550 |
| 9 | `conformer_vocalset_gtsinger_noncausal_aug_r10` | conformer/noncausal | 87.1 | — | 0.0 | — | 0.649 | 0.728 | 0.6490 |
| 16 | `conformer_stage2_technique` | conformer/noncausal | 83.6 | 96.2 | 32.7 | 17.6 | 0.632 | 0.683 | 0.6320 |
| 11 | `tcn_vocalset_only_curriculum` | tcn/noncausal | 81.5 | 99.3 | 20.4 | 10.3 | 0.626 | 0.763 | 0.6260 |
| 20 | `tcn_probe_technique` | tcn/noncausal | 66.4 | — | 0.0 | — | 0.583 | 0.710 | 0.5830 |
| 14 | `tcn_stage2_technique` | tcn/noncausal | 81.6 | 97.2 | 35.4 | 18.8 | 0.576 | 0.675 | 0.5760 |
| 17 | `tcn_technique` | tcn/noncausal | 65.8 | 0.0 | 0.1 | 418.4 | 0.444 | 0.575 | 0.4440 |
| 19 | `conformer_probe_technique` | conformer/noncausal | 82.6 | 99.5 | 61.5 | 1.2 | 0.395 | 0.521 | 0.3950 |
| 21 | `tcn_probe_technique_v2` | tcn/noncausal | 81.4 | 97.8 | 76.9 | 16.8 | 0.374 | 0.498 | 0.3740 |
| 7 | `tcn_vocalset_gtsinger_noncausal_aug` | tcn/noncausal | 66.4 | — | 0.0 | — | 0.326 | 0.653 | 0.3260 |
| 13 | `tcn_stage1_pitchonly` | tcn/noncausal | 82.8 | 95.6 | 53.4 | 31.5 | 0.265 | 0.296 | 0.2650 |
| 18 | `tcn_technique_gtsinger` | tcn/noncausal | 67.2 | — | 0.0 | — | 0.226 | 0.411 | 0.2260 |
| 8 | `tcn_vocalset_gtsinger_noncausal_aug_r8` | tcn/noncausal | 81.1 | 99.0 | 9.3 | 12.2 | 0.223 | 0.729 | 0.2230 |
| 15 | `conformer_stage1_pitchonly` | conformer/noncausal | 81.8 | 97.0 | 67.5 | 17.0 | 0.183 | 0.203 | 0.1830 |

## Per-condition offRPA

_(Sorted by Macro RPA descending — offline Viterbi. NanoPitch reference rows use realtime Viterbi; see NanoPitch Reference Lines above.)_

| # | Run | -5 dB | +0 dB | +5 dB | +10 dB | +20 dB | clean | Macro |
|---|---|---|---|---|---|---|---|---|
| 19 | `conformer_probe_technique` | 99.0 | 99.0 | 99.6 | 99.7 | 99.6 | 99.8 | 99.5 |
| 4 | `conformer_gtsinger_noncausal_aug` | 99.2 | 99.2 | 99.2 | 99.5 | 99.4 | 99.5 | 99.3 |
| 6 | `conformer_vocalset_gtsinger_noncausal_aug_noTechnique` | 99.2 | 99.3 | 99.6 | 99.4 | 99.1 | 99.3 | 99.3 |
| 11 | `tcn_vocalset_only_curriculum` | 97.8 | 99.4 | 99.7 | 99.8 | 99.5 | 99.4 | 99.3 |
| 5 | `tcn_vocalset_gtsinger_noncausal_aug_noTechnique` | 98.2 | 99.1 | 98.8 | 99.4 | 99.3 | 99.5 | 99.1 |
| 8 | `tcn_vocalset_gtsinger_noncausal_aug_r8` | 99.3 | 99.4 | 96.7 | 99.5 | 99.7 | 99.5 | 99.0 |
| 3 | `tcn_gtsinger_noncausal_aug` | 97.4 | 97.0 | 99.2 | 99.1 | 99.2 | 99.1 | 98.5 |
| 21 | `tcn_probe_technique_v2` | 97.5 | 95.6 | 97.7 | 99.0 | 98.1 | 98.9 | 97.8 |
| 12 | `conformer_vocalset_rescaled` | 97.6 | 96.2 | 96.8 | 97.5 | 97.8 | 98.1 | 97.3 |
| 14 | `tcn_stage2_technique` | 95.5 | 98.0 | 94.9 | 98.5 | 98.1 | 98.2 | 97.2 |
| 2 | `conformer_gtsinger_noncausal` | 94.3 | 96.7 | 97.0 | 97.6 | 97.5 | 99.2 | 97.1 |
| 15 | `conformer_stage1_pitchonly` | 97.2 | 94.3 | 97.5 | 97.9 | 96.8 | 98.4 | 97.0 |
| 1 | `tcn_gtsinger_noncausal` | 92.0 | 97.0 | 95.3 | 97.3 | 97.0 | 99.3 | 96.3 |
| 16 | `conformer_stage2_technique` | 95.8 | 95.3 | 96.1 | 96.9 | 96.0 | 97.1 | 96.2 |
| 13 | `tcn_stage1_pitchonly` | 92.4 | 93.9 | 94.3 | 98.1 | 96.2 | 98.4 | 95.6 |
| 10 | `Output dir ignored` | 84.5 | 90.5 | 92.2 | 91.7 | 97.3 | 98.1 | 92.4 |
| 7 | `tcn_vocalset_gtsinger_noncausal_aug` | nan | nan | nan | nan | nan | nan | nan |
| 9 | `conformer_vocalset_gtsinger_noncausal_aug_r10` | nan | nan | nan | nan | nan | nan | nan |
| 17 | `tcn_technique` | nan | 0.0 | 0.0 | nan | 0.0 | 0.0 | — |
| 18 | `tcn_technique_gtsinger` | nan | nan | nan | nan | nan | nan | nan |
| 20 | `tcn_probe_technique` | nan | nan | nan | nan | nan | nan | nan |

## Per-technique F1

_(Sorted by macro F1 descending. `—` = no test clips for that class. Clip Acc = argmax accuracy matching MuQ/AST SOTA metric. GT mF1 = macro F1 on GTSinger held-out set.)_

| # | Run | vibrato | breathy | falsetto | belt | straight | macro F1 | clip acc | GT mF1 |
|---|---|---|---|---|---|---|---|---|---|
| 10 | `Output dir ignored` | 0.821 | 0.919 | — | 0.723 | 0.779 | 0.810 | 80.3% | — |
| 12 | `conformer_vocalset_rescaled` | 0.944 | 0.506 | — | 0.515 | 0.656 | 0.655 | 63.5% | — |
| 9 | `conformer_vocalset_gtsinger_noncausal_aug_r10` | 0.791 | 0.758 | — | 0.491 | 0.554 | 0.649 | 53.7% | — |
| 16 | `conformer_stage2_technique` | 0.885 | 0.468 | — | 0.551 | 0.624 | 0.632 | 63.5% | — |
| 11 | `tcn_vocalset_only_curriculum` | 0.845 | 0.747 | — | 0.217 | 0.692 | 0.626 | 70.0% | — |
| 20 | `tcn_probe_technique` | 0.872 | 0.514 | — | 0.360 | 0.585 | 0.583 | 63.1% | — |
| 14 | `tcn_stage2_technique` | 0.703 | 0.640 | — | 0.346 | 0.617 | 0.576 | 56.2% | — |
| 17 | `tcn_technique` | 0.458 | 0.655 | — | 0.140 | 0.524 | 0.444 | 40.4% | — |
| 19 | `conformer_probe_technique` | 0.438 | 0.620 | — | 0.000 | 0.524 | 0.395 | 40.9% | — |
| 21 | `tcn_probe_technique_v2` | 0.436 | 0.487 | — | 0.049 | 0.524 | 0.374 | 36.5% | — |
| 7 | `tcn_vocalset_gtsinger_noncausal_aug` | 0.854 | 0.000 | — | 0.049 | 0.400 | 0.326 | — | — |
| 13 | `tcn_stage1_pitchonly` | 0.402 | 0.329 | — | 0.329 | 0.000 | 0.265 | 21.7% | — |
| 18 | `tcn_technique_gtsinger` | 0.411 | 0.495 | — | 0.000 | 0.000 | 0.226 | 16.7% | — |
| 8 | `tcn_vocalset_gtsinger_noncausal_aug_r8` | 0.623 | 0.000 | — | 0.140 | 0.130 | 0.223 | 59.6% | — |
| 15 | `conformer_stage1_pitchonly` | 0.402 | 0.329 | — | 0.000 | 0.000 | 0.183 | 9.9% | — |
## Change log

_(fill in as runs complete)_
