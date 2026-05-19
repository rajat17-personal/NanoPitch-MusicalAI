# VocalCoach
**Musical AI Final Project · Rajat Sharma**

---

## What VocalCoach Does

Multi-task singing analysis model: one audio clip in → pitch track + voice activity + vocal technique labels out, with natural language coaching feedback. Extends NanoPitch (pitch-only GRU baseline, 96.1% RPA) with technique classification and coaching outputs.

**Four outputs per clip:** F0 posteriorgram · VAD · technique label (vibrato / breathy / falsetto / belt / straight) · signal-processing features (HNR, jitter, shimmer, spectral tilt)

---

## Architecture: TCN vs Conformer

Two architectures trained for comparison, both ~450K parameters, `causal=False` (offline):

| | TCN | Conformer |
|---|---|---|
| Backbone | Dilated Conv1d ×8 (dilation 1→128, ~5s receptive field) | FF → Self-Attention → Conv → FF ×4 (full-clip global attention) |
| Pitch baseline (no technique) | RPA 98.5%, VDR 72.3% | RPA 99.3%, VDR 80.4% ← best |
| Live mode | Retrain `causal=True` → ONNX → browser | Not streaming-compatible; fallback to NanoPitch GRU |

**Conformer wins on pitch accuracy.** TCN is the live-mode candidate (causal retrain).

---

## Datasets

| Dataset | Role | Notes |
|---|---|---|
| **GTSinger** (AaronZ345/GTSinger) | Pitch + VAD training | ~hours of polyphonic singing phrases; mel + RMVPE F0 + VAD labels → `clean.npz` |
| **VocalSet** (Wilkins & Seetharaman, ISMIR 2018) | Technique head training | 10h, 20 singers, 17 techniques → 5-class taxonomy (vibrato / breathy / falsetto / belt / straight); clip-level binary labels → `technique_train.npz` |
| **GTSinger technique** | Technique training (dropped) | Per-clip labels via `any(phoneme_has_technique)` → noisy; confirmed harmful (mF1 dropped from 0.810→0.223 vs VocalSet-only). Not used in probe-mode runs. |
| **FSDNoisy18k** (42h) | Noise augmentation | `noise.npz` — mixed into training batches via log-mel `logaddexp` at random SNR [−10, +30 dB] |
| **NanoPitch test set** | Pitch + VAD evaluation | 100 GTSinger held-out clips at 6 SNR conditions (−5, 0, +5, +10, +20 dB, clean) |
| **VocalSet held-out split** | Technique evaluation | Singers m2, m4, f4, f8 withheld — same split as VocalSet paper |
| **PopBuTFy** | Population baseline evaluation (D5) | 28,508 clips across amateur/professional singers. Used to build population-level medians for coaching context — not used in model training. |

**Known gap:** VocalSet has no falsetto clips; GTSinger technique data is 50% falsetto but label quality is poor (noisy clip-level annotation). Falsetto is currently unlearnable without a better-labelled source.

---

## The Joint Training Problem

Adding a technique head to either architecture collapses voice detection rate (VDR) — the fraction of voiced frames correctly decoded.

**Root cause:** Technique loss gradients are ~10× larger than pitch gradients at epoch 1. The shared backbone is captured by technique discrimination before pitch representations are established. Every mitigation tried degraded either VDR or mF1:

| Approach | Best VDR | Best mF1 | Outcome |
|---|---|---|---|
| Joint training, default weights | 26.7% | 0.810 | mF1 fine, VDR unusable |
| Rescaled weights (w_pitch=4, w_tech=0.5) | 57.3% | 0.655 | Best joint tradeoff — still −23 VDR pts |
| Two-stage freeze (backbone frozen 20 ep) | 35.4% | 0.576 | Pitch/VAD heads drifted on VocalSet mel |
| Technique-only dataset (VocalSet / GTSinger) | 0.1% | 0.444 | Domain mismatch: train≠eval distribution |

**Key finding:** The two-stage freeze protected the backbone but left `head_pitch` and `head_vad` free to adapt to VocalSet's mel distribution (isolated vocal exercises ≠ GTSinger polyphonic phrases). VDR degraded even during the frozen phase.

---

## Solution: MERT-Style Linear Probing

Inspired by MERT (Li et al., ICLR 2024): pre-train backbone on primary task, freeze **everything**, train only a new task-specific head.

- Load best pitch checkpoint (Run 4, Conformer, VDR=80.4%)
- Freeze: `input_proj` + all backbone blocks + `head_pitch` + `head_vad`
- Train only: `head_technique` on VocalSet

---

## Best Results (28 runs)

### Pitch + VAD — Best Checkpoints

| Metric | Best run | Value | Architecture |
|---|---|---|---|
| **offRPA** | Run 19 `conformer_probe_technique` | **99.5%** | Conformer / noncausal |
| **offVDR** | Run 4 `conformer_gtsinger_noncausal_aug` | **80.4%** | Conformer / noncausal |
| **offMed¢** | Run 19 `conformer_probe_technique` | **1.2 ¢** | Conformer / noncausal |
| VAD Acc | Run 9 `conformer_vocalset_gtsinger_noncausal_aug_r10` | 87.1% | Conformer / noncausal |

**Probe-mode Conformer (Run 19) is the deployed checkpoint**: highest RPA (99.5%) and best median pitch error (1.2¢). VDR (61.5%) trails the pitch-only baseline (80.4%) due to domain mismatch on VocalSet mel at eval time.

**TCN probe (Run 21)** achieves VDR 76.9% (highest of any run with technique), making it the best live-mode candidate once causal retraining is complete.

### Per-SNR offRPA — Conformer Probe (Run 19)

| Condition | −5 dB | 0 dB | +5 dB | +10 dB | +20 dB | clean | Macro     |
| --------- | ----- | ---- | ----- | ------ | ------ | ----- | --------- |
| offRPA    | 99.0  | 99.0 | 99.6  | 99.7   | 99.6   | 99.8  | **99.5%** |

### Technique Classification — Best Checkpoints

| Metric | Best run | Value | Note |
|---|---|---|---|
| **mF1 (VocalSet)** | Run 10 (joint, best loss) | **0.810** | VDR=26.7% — pitch unusable |
| **mF1 (probe-mode)** | Run 19 `conformer_probe_technique` | **0.395** | VDR=61.5%, RPA=99.5% — deployed |
| **mAP** | Run 10 | **0.833** | Same caveat as mF1 |
| **Clip Acc** | Run 12 `conformer_vocalset_rescaled` | **63.5%** | vs MuQ SOTA 81.5% / AST 82.0% |
| **Vibrato F1** | Run 12 `conformer_vocalset_rescaled` | **0.944** | Best single-class |
| **Breathy F1** | Run 10 | **0.919** | Best single-class |

**Trade-off summary:** Best mF1 (Run 10, 0.810) comes at the cost of VDR=26.7% — the model skips ~73% of voiced frames and is unusable for pitch/VAD coaching. The probe-mode checkpoint (Run 19) sacrifices ~41 mF1 points to preserve pitch tracking. This is the correct trade-off for a coaching app where pitch accuracy is primary.

### Per-technique F1 — Deployed Checkpoint (Run 19, Conformer Probe)

| vibrato | breathy | falsetto | belt | straight | mF1 | Clip Acc |
|---|---|---|---|---|---|---|
| 0.438 | 0.620 | — | 0.000 | 0.524 | 0.395 | 40.9% |

Belt F1=0.000 and falsetto unlearnable (no VocalSet falsetto clips). These are known gaps.

---

## Evaluation Strategy

### Signal-Processing Features (Phase 2 — Implemented)

Derived from the VocalCoach model's F0 + VAD outputs. Computed post-inference, no additional training required:

| Feature | Method | Notes |
|---|---|---|
| Vibrato rate/depth | Autocorrelation of detrended F0 residual, 4–8 Hz window | Requires ≥300 ms voiced segment. Conflicts with classifier-based `technique.vibrato` — both measure vibrato but via independent paths |
| HNR, jitter, shimmer | Cycle-to-cycle F0 statistics | Values unreliable in absolute terms due to noisy probe-mode F0; meaningful only relative to PopBuTFy population medians |
| DTW pitch deviation | Pure-numpy DTW in cents space, mean-normalised | Key-invariant; measures pitch contour shape match vs a reference recording |
| Phrase segmentation | VAD gaps ≥150 ms split phrases; <100 ms segments dropped | Uses pitch-confidence proxy for voicing (30% of clip max) in probe checkpoint |

### Population Baselines (D5 — Implemented)

`scripts/buildPopBuTFyBaselines.py` ran VocalCoach inference across all 28,508 PopBuTFy clips (14,525 amateur + 13,983 professional) to build population-level medians. Output: `data/popbutfy_baselines.json`.

Each coaching report includes a `population_context` block comparing the user's metrics to amateur/professional medians with band labels: `pro_range` / `approaching_pro` / `amateur_range` / `below_amateur`.

**Known limitation:** Most acoustic metrics (HNR, jitter, shimmer) show small amateur/professional separation because both groups are studio recordings and the noisy probe-mode F0 inflates all cycle-to-cycle variance estimates equally.

### Perceptual Quality Scoring (D6 — Partially Implemented, Under Review)

**SingMOS-Pro** (wav2vec2-large, trained on SVS/SVC/SVR data): implemented in `vocalcoach/singmos.py`. Scores audio quality on a 1–5 MOS scale.

**Problem identified:** SingMOS-Pro does not meaningfully differentiate amateur from professional singing in PopBuTFy — both groups score ~4.8 (ceiling effect). Both groups are studio recordings; MOS measures perceived acoustic quality / naturalness, not singing skill. End users of a vocal coaching app are not expected to have studio microphones, so MOS is not a useful signal for the core coaching task.

**Plan:** Replace SingMOS-Pro with a singing-specific perceptual scorer. Two candidates evaluated:

| Model | Size | Output | Amateur/Pro differentiation | Status |
|---|---|---|---|---|
| **VocalVerse2 / MuQ** (audioscore) | Small (MuQ encoder + scoring head) | Single aesthetic score 50–99 | Unknown — to be verified on PopBuTFy | Local eval script written: `scripts/score_vocalverse2.py` |
| **VocalVerse1 / Qwen2-Audio-7B** (qwenaudio) | ~15 GB fp16 + 4 LoRA adapters | 4 scores: Timbre / Breath / Emotion / Technique | Unknown — to be verified | Colab notebook written: `notebooks/VocalVerse1_Colab.ipynb` |

Both models are from Wang et al., *"Singing Timbre Popularity Assessment Based on Multimodal Large Foundation Model"*, ACM MM 2025 ([doi:10.1145/3746027.3758148](https://doi.org/10.1145/3746027.3758148)). Trained on expert-annotated VocalVerse dataset (165 amateur raters + 4 professional vocal experts, 4 dimensions).

**Next step:** Run both on a PopBuTFy song pair (amateur vs professional) and check whether pro scores consistently exceed amateur scores. If separation is meaningful, integrate the best model as the MOS replacement in the API and population context chart.

---

## Coaching Pipeline (Phase 2 — Implemented)

Full inference pipeline implemented as a FastAPI server (`vocalcoach/api.py`):

```
Audio clip → VocalCoach model (F0 + VAD + technique)
           → signal-processing features (HNR, jitter, vibrato, DTW)
           → score_report() [rule-based coaching, 0–100 score]
           → compare_to_baselines() [PopBuTFy population context]
           → [optional] generate_critique() [Claude API LLM critique]
           → JSON coaching report
```

Key components:
- `vocalcoach/coach.py` — rule-based coaching with per-axis observations and weighted overall score
- `vocalcoach/features.py` — phrase segmentation, vibrato detection, DTW
- `vocalcoach/singmos.py` — SingMOS-Pro wrapper (graceful degradation if unavailable)
- `notebooks/VocalCoach_Inference.ipynb` — full demo notebook with population context chart, DTW timeline, phrase analysis, technique radar

---

## Metric Conflicts and Known Issues

Several reported metrics use independent measurement paths and can contradict each other:

| Conflict | Root cause |
|---|---|
| `technique.vibrato` (classifier) ≠ `vibrato.n_vibrato_phrases` (signal-proc) | Classifier fires on mel-spectrogram pattern; vibrato detector requires clean autocorrelation of F0. Noisy probe-mode F0 fails the 0.3 regularity threshold even when the classifier correctly detects vibrato. |
| High `jitter_mean_pct` / negative `hnr_mean_db` | Probe-mode F0 is noisy; cycle-to-cycle statistics are inflated. Values are internally consistent (same systematic bias) so population comparisons are valid, but absolute values are meaningless. |
| `pitch.f0_stability_std_hz` flags vibrato singers as "unstable" | Correct vibrato oscillation raises std. This metric does not distinguish intentional vibrato from pitch drift. |
| `n_phrases` in summary vs phrase count in list | `summarise()` returns `n_voiced_segments` (raw VAD onset count); `phrase_aggregate()` returns merged/filtered phrase count. Different denominators — expected. |

**Root cause for most issues:** Probe-mode checkpoint F0 is used for both phrase segmentation and acoustic feature extraction. A RMVPE-based F0 extractor (independent of the VocalCoach model) would fix HNR/jitter/vibrato detection quality without retraining.

---

## Open Questions / Pending

1. **Perceptual scorer:** Run VocalVerse1 (Colab) and VocalVerse2 (local) on PopBuTFy pairs; confirm amateur/pro separation before integrating into API.
2. **mF1 gap (probe-mode 0.395 vs joint-best 0.810):** Deeper probe head (`Linear(128→64)→GELU→Linear(64→5)`) + 100 epochs expected to gain 10–15 mF1 points.
3. **Belt F1=0.000 in probe runs:** Belt is the rarest VocalSet class; pos_weight tuning needed.
4. **Falsetto coverage:** VocalSet has no falsetto. Requires a new data source or using GTSinger technique clips with cleaned labels.
5. **RMVPE F0 for acoustic features:** Would fix HNR/jitter/shimmer accuracy and improve vibrato detection — independent of VocalCoach model training.
6. **Live causal TCN retrain:** TCN probe (Run 21, VDR=76.9%) is the confirmed live candidate; causal retrain pending.

---

## Evaluation Metrics Reference

**Pitch + VAD** (`evaluate.py` → offline Viterbi decoder):

| Metric | What it measures |
|---|---|
| offRPA | Raw Pitch Accuracy — fraction of voiced frames within 50 cents of ground truth |
| offVDR | Voice Detection Rate — fraction of truly-voiced frames decoded as voiced (key indicator of joint-training damage) |
| offMed¢ | Median pitch error in cents on correctly voiced frames |
| VAD Acc | Binary voiced/unvoiced frame accuracy |

**Technique** (clip-level, matching SOTA comparison protocol):

| Metric | What it measures |
|---|---|
| per-class F1 | Vibrato / breathy / falsetto / belt / straight independently |
| macro F1 (mF1) | Unweighted mean F1 across present classes |
| mAP | Mean Average Precision — threshold-free technique ranking quality |
| Clip Acc | `mean(frame_probs) > 0.5` argmax accuracy — matches MuQ / AST SOTA metric (81.5% / 82.0%) |

**Key evaluation insight:** VDR was the canary for joint-training health across all 28 runs. RPA measures accuracy *on frames the model chose to decode* — a model can achieve high RPA by silently skipping most voiced frames. VDR catches this; joint training consistently collapsed VDR to <30% while RPA appeared healthy.
