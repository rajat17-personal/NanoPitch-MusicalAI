# VocalCoach
**Musical AI Final Project · Rajat Sharma**

---

## What VocalCoach Does

Multi-task singing analysis model: one audio clip in → pitch track + voice activity + vocal technique labels out, with natural language coaching feedback. Extends NanoPitch (pitch-only GRU baseline, 96.1% RPA) with technique classification and coaching outputs.

**Four outputs per clip:** F0 posteriorgram · VAD · technique label (vibrato / breathy / falsetto / belt / straight) · signal-processing features (HNR, jitter, shimmer, spectral tilt)

---

## Architecture: TCN vs Conformer

Two architectures trained for comparision, both ~450K parameters, `causal=False` (offline):

| | TCN | Conformer |
|---|---|---|
| Backbone | Dilated Conv1d ×8 (dilation 1→128, ~5s receptive field) | FF → Self-Attention → Conv → FF ×4 (full-clip global attention) |
| Pitch baseline (no technique) | RPA 98.5%, VDR 72.3% | RPA 99.3%, VDR 80.4% ← best |
| Live mode | Retrain `causal=True` → ONNX → browser | Not streaming-compatible; fallback to NanoPitch GRU |

**Conformer wins on pitch accuracy.** TCN is the live-mode candidate (causal retrain, Phase 2).

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
- Train only: `head_technique` (single `Linear(128→5)`) on VocalSet

**Results:**

| Run | Approach | VDR | RPA | mF1 |
|---|---|---|---|---|
| 4 | Conformer pitch-only baseline | 80.4% | 99.3% | — |
| 12 | Best joint model (rescaled) | 57.3% | 97.3% | 0.655 |
| **19** | **Conformer probe-mode** | **61.5%** | **99.5% ↑** | **0.395** |
| **21** | **TCN probe-mode** | **76.9% ↑** | **97.8%** | **0.374** |

Run 19 is the **only run in 21 experiments where RPA improved** over the pitch baseline (+0.2%). Run 21 VDR (76.9%) exceeds the TCN pitch-only baseline (72.3%) — first time adding technique has not hurt pitch tracking.

---

## Open Questions for Final Project

1. **mF1 gap**: Probe-mode mF1 (0.374–0.395) is lower than best joint run (0.810, Run 10). The probe head is a single linear layer with 50 epochs on pitch-optimised features. **Plan:** deeper probe head (`Linear(128→64) → GELU → Linear(64→5)`) + 100 epochs + GTSinger technique data added to probe training. Expected gain: 10–15 mF1 points without touching backbone.

2. **Remaining VDR gap (~19 points, Conformer)**: Frozen `head_vad` was trained on GTSinger mel; VocalSet mel at eval time is a different distribution. The gap is a domain mismatch floor, not a gradient conflict artifact. **Question:** does this matter for the vocal coaching? VDR=61.5% on a held-out GTSinger test set is likely higher on real user audio (closer to GTSinger distribution than VocalSet exercises).

3. **Falsetto coverage**: VocalSet has no falsetto clips; GTSinger technique dataset is 50% falsetto. Probe head trained on VocalSet cannot learn falsetto. **Plan:** add GTSinger technique clips to probe training data.

4. **Phase 2 architecture decision**: Use **Conformer** as the offline analysis backbone (best RPA/VDR), **TCN causal retrain** for live streaming pitch. Probe-mode is the confirmed training strategy for any new task heads added in Phase 2.

---

## Evaluation

Two evaluation scripts track all runs in `VOCALCOACH_RESULTS.md`:

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

**Key evaluation insight:** VDR was the canary for joint-training health across all 21 runs. RPA measures accuracy *on frames the model chose to decode* — a model can achieve high RPA by silently skipping most voiced frames. VDR catches this; joint training consistently collapsed VDR to <30% while RPA appeared healthy.

---

## What's Working / What's Next

**Done:** Two architectures trained and compared · multi-task training pipeline (noise augmentation, SpecAugment, curriculum, pos_weights, staged freeze) · probe-mode implementation confirmed · 21 experiment runs tracked

**Phase 2 targets:** Deeper probe head · DTW reference pitch comparison (primary coaching metric) · post-processing feature pipeline (vibrato rate/depth, HNR, jitter) ·
