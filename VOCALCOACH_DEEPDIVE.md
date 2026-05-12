# VocalCoach+ Deep Dive Review

**Date:** 2026-05-10 (updated 2026-05-11)
**Author:** Rajat Sharma

---

## Overview

VocalCoach is a multi-task singing analysis model that extends the NanoPitch GRU pitch tracker into a full coaching system. NanoPitch (~333K params, 96.1% rtRPA) serves as the pitch accuracy baseline only. VocalCoach adds technique classification, voice quality scoring, and a rich post-processing feature pipeline.

---

## 1. Current State Assessment

### What's Solid

| Component | Status | Comment |
|---|---|---|
| TCN + Conformer architectures | Good | Correct inductive biases; 360-bin Gaussian posterior + offline Viterbi matches RMVPE-class quality |
| Multi-label technique output | Correct | Co-occurrence (vibrato + breathy) handled properly via sigmoid (not softmax) |
| Multi-source data loading | Good | VocalSet + GTSinger concatenation working via `--technique-dirs` |
| features.py (Features 14–22) | Implemented | HNR, jitter, shimmer, breath detection, vocal onset steepness all present |
| Noise augmentation | Working | `noise_specaug` is the right call for robustness; matches NanoPitch run 26 setup |
| Evaluation harness | Complete | precision/recall/F1/AP + per-SNR pitch table in evaluate.py |
| Annotated-VocalSet script | Written | `scripts/extractAnnotatedVocalSet.py` — per-note MIDI onset/offset extraction ready |

### Active Blockers

| Issue | Root Cause | Fix |
|---|---|---|
| Run 7: VDR=0, RPA=nan | Technique-only model never sees test.npz pitch distribution → sigmoid pitch posteriors stay below `voicing_threshold=0.3` → Viterbi decodes all frames as silence | Run 8: add `--data-dir data` to technique training |
| No technique run with valid pitch metrics yet | Same distribution mismatch issue | Runs 8 and 10 are the first meaningful technique+pitch runs |
| Experiment C (MuQ backbone) not started | **MERT replaced by MuQ** (Tencent AILab, Jan 2025 — 81.5% VocalSet technique vs MERT lower). Requires model.py integration; no `train_mert.py` needed — MuQ plugs into same train.py via `--arch muq` once integrated | Missing implementation |

### Augmentation vs VDR Analysis (Verified from Results Table)

| Comparison | RPA effect | VDR effect |
|---|---|---|
| TCN no-aug → aug (Run 1→3) | +2.2% | **−3.1%** (hurts) |
| Conformer no-aug → aug (Run 2→4) | +2.2% | **+2.7%** (improves) |
| Merged data effect (Run 3→5, 4→6) | — | −12% to −16% (the real culprit) |

Large VDR drops come from merged data overlap, **not** augmentation. Augmentation is net-positive for Conformer and only mildly negative for TCN.

---

## 2. Plan Gaps — Things Not Yet Thought Of

### Gap A: Pitch-Conditioned Technique Head (High Impact, Low Effort)

**Current architecture**: backbone features → [VAD | Pitch | Technique] heads — all three see the same representation.

Vibrato detection is fundamentally an F0 oscillation pattern. The mel spectrogram is an indirect proxy. Research (self-supervised contrastive learning for singing voices, IEEE 2022) shows that augmenting technique classification with explicit pitch information gives a measurable gain.

**Recommendation**: After the pitch head, detach and feed a summary of the pitch posteriorgram (rolling variance of the peak bin) as an auxiliary feature concatenated into the technique head input. Detach ensures no gradient flows back through pitch → technique path — no training dependency issue.

```python
# In forward(), after pitch = torch.sigmoid(self.head_pitch(x)):
pitch_peak = pitch.detach().max(dim=-1, keepdim=True).values  # (B, T, 1)
tech_input = torch.cat([x, pitch_peak], dim=-1)               # (B, T, hidden+1)
technique = torch.sigmoid(self.head_technique_fc(tech_input))
```

---

### Gap B: Register / Passaggio Detection (New Post-Processing Feature)

Chest-to-head register transition is one of the most common coaching pain points — especially for untrained singers. It is acoustically detectable from harmonic tuning shifts (H2/H4 dominance in chest register → H3/H4 in head register) and is **not** captured by the current technique taxonomy (vibrato / breathy / falsetto / belt / straight).

Acoustic analysis literature shows this is detectable without labeled data: it correlates with a spectral tilt discontinuity around the passaggio pitch range (E4–G4 for males, A4–C5 for females).

**Recommendation**: Add as **Feature 28** in `features.py` (post-processing only, no new model head needed). Compute from H1-H2 tilt + predicted pitch track. Output: per-frame register label (chest/mixed/head) and transition timestamps. No model retraining required.

---

### Gap C: Phrase-Level Output Aggregation (UX Critical)

All model outputs are frame-level (10ms). The coaching report needs phrase-level summaries. This is mentioned vaguely in Phase 3 but has no concrete implementation design.

**Recommendation**: Add a `phrase_aggregate()` function to `features.py` that:
1. Uses VAD output to segment phrases (voiced regions separated by ≥ 150ms gaps)
2. Within each phrase, computes:
   - Mean technique probability per class
   - Vibrato rate/depth via autocorrelation of F0
   - Intonation stability (F0 std dev within phrase)
   - Energy arc (RMS polynomial fit — crescendo/decrescendo shape)
3. Returns a list of dicts, one per phrase

This is the most direct path from raw model output to a coaching report UI. Pure post-processing, no model changes required.

---

### Gap D: Vibrato Onset Latency (Missing from Feature Taxonomy)

Features 5–7 cover vibrato rate, depth, and regularity. But professional vocology research identifies **vibrato onset latency** (how many ms into a sustained note before vibrato begins) as a key quality indicator. Classical training targets < 200ms onset; untrained singers often have > 500ms or no vibrato at all.

**Recommendation**: Add **Feature 27** — vibrato onset latency. Computable from the F0 track per note:
- Find note onset (from Feature 8, using Annotated-VocalSet data)
- Detect first oscillation in F0 via autocorrelation within a 300ms window post-onset
- Output in milliseconds per note

---

### Gap E: Singer's Formant / Resonance Placement (Missing Feature)

`features.py` has H1-H2 spectral tilt (Feature 17) as a breathiness indicator. But classical singing coaching is heavily focused on *resonance placement* — specifically whether the singer is producing a "singer's formant cluster" (~2.5–3.5 kHz), which is what allows a trained voice to project over orchestral accompaniment.

This is not in the current feature taxonomy.

**Recommendation**: Add to `features.py`:
- Spectral energy ratio in [2000–4000 Hz] band vs [500–2000 Hz] band — singer's formant indicator
- Peak frequency and amplitude in the 2–4 kHz range
- Optionally: formant tracking (F1/F2) using LPC analysis (available via `scipy.signal.lfilter`)

**Effort**: Low. Pure signal processing, no model changes.

---

### Gap F: Multi-Scale Technique Architecture (Medium Effort, Phase 2)

Different techniques operate at different timescales:
- **Breath technique**: constant over an entire phrase (~3–10 s)
- **Belt/straight**: consistent over a sustained note (~500ms–2s)
- **Vibrato**: oscillation at 5–8 Hz (~125–200ms per cycle)
- **Vocal onset (breathy attack)**: attack shape over first 100ms

The current Conformer architecture with global attention partially addresses this, but a multi-scale head — applying technique classification at multiple temporal resolutions (frame / note / phrase) with learned pooling — could work better.

**Recommendation**: Defer to Phase 2. First validate that runs 9/10 give sensible technique F1 at the frame level, then consider multi-scale pooling if vibrato F1 in particular is low.

---

### Gap G: Contrastive Pretraining on Singing Data (Medium Effort, Phase 2/3)

Research (arXiv:2302.07077, self-supervised contrastive for singing voices) shows that using pitch-shift augmentation as a contrastive invariance signal significantly improves technique embeddings. Key insight: a pitch-shifted version of a vibrato clip should still be classified as vibrato, even though the mel spectrogram looks completely different. Standard noise augmentation doesn't enforce this invariance.

**Recommendation**: Not for Phase 1, but Phase 2/3: pretrain the backbone (TCN or Conformer) on unlabeled singing audio using NT-Xent loss with pitch-shift augmentations as positive pairs, before fine-tuning on technique labels. GTSinger and VocalSet (even unannotated portions) can serve as the pretraining corpus. Self-supervised Contrastive Learning for Singing Voices (IEEE 2022) reported a 9.12% improvement on VocalSet technique classification over non-augmented baselines.

---

## 3. Missing Datasets Worth Integrating

Ranked by usefulness for VocalCoach specifically:

| Dataset | Size | What it adds | Priority | Access |
|---|---|---|---|---|
| **SINGSTYLE111** | Multilingual, style transfer pairs | Ground truth for amateur→professional style gap; directly complements PopBuTFy | **High** (Phase 2) | Public (CMU) |
| **M3 Mistake Dataset** | arXiv:2602.06917 (Feb 2026) | Labeled singing mistakes (pitch/amplitude/spectral errors); enables M3 head | **High if available** | Contact authors |
| **NUS-48E** | 48 songs, phonetically annotated | Vowel articulation and formant analysis; articulation coaching | Medium (Phase 2) | Zenodo |
| **CSD (Children's Song Dataset)** | Children's choir recordings | Beginner/learner domain; register breaks common; complements SingPAD | Medium (Phase 3) | Open |
| **NHSS** | Parallel speech+singing, 10 singers | Enables speech-vs-singing quality comparison; voice quality pathology | Medium (Phase 3) | Open |
| **DALI** | 5,358 songs with lyrics alignment | Real-world accompaniment robustness; lyric alignment (Feature 25) | Low-Medium (Phase 3) | Open (registration) |

### Key Research Finding: M3 Dataset (February 2026)

Paper: *"Automatic Detection and Analysis of Singing Mistakes for Music Pedagogy"* (arXiv:2602.06917, Feb 2026). Uses CNN/CRNN to detect pitch errors, amplitude errors, and spectral deviations — exactly what the M3 head in the plan needs. This paper and dataset are new enough that they postdate the original plan. Worth checking if the dataset is publicly released and if so, integrating it would directly enable the M3 (mistake detection) head listed as a Stretch goal.

---

## 4. Architecture & Training Improvements

### 4a. Voicing Threshold Auto-Calibration

The VDR=0 issue is partly a calibration problem. `voicing_threshold=0.3` in `viterbi_decode()` was tuned on GTSinger distribution. When technique models train on VocalSet-only, their pitch posteriors are calibrated to a different distribution.

**Fix beyond Run 8**: Add a temperature scaling step in `evaluate.py` — after collecting all pitch posteriors on the development set, find the optimal voicing threshold via grid search (0.1 to 0.5, step 0.05). Standard post-hoc calibration — no model changes needed.

### 4b. Per-Head Loss Logging

Current: `pitch×2 + vad×0.05 + technique×2`. When technique data is scarce relative to pitch/VAD clips, effective technique loss per batch may be very small and not visible.

**Recommendation**: Log per-head loss values at each epoch in `train.py`:
```
Epoch 10 | total=0.832 | pitch=0.641 | vad=0.003 | technique=0.188
```
This makes it immediately visible if technique head is being swamped.

### 4c. Sigmoid vs Softmax for Pitch Head (Phase 2 consideration)

The pitch head uses 360 independent sigmoid outputs. This is correct for multi-pitch, but means the model can predict "no pitch" by producing all near-zero values — which is what triggers VDR=0. RMVPE uses 360+1 bins (softmax over 360 pitch bins + 1 "unvoiced" bin), which is more calibrated.

**Not recommended mid-experiment** — requires retraining from scratch. Worth considering for Phase 2 if VDR remains problematic after Run 8/10.

### 4d. GTSinger Technique Evaluation in update_results.py

Currently technique evaluation uses VocalSet held-out test singers. The plan mentions `technique_gtsinger_test.npz` as a secondary benchmark. Adding GTSinger technique test evaluation would give cross-dataset generalization metrics — important for the academic contribution section demonstrating the model works across technique taxonomies.

---

## 5. Published Research Landscape (2020–2026)

### Relevant Papers

| Paper | Year | Relevance |
|---|---|---|
| MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training (arXiv:2306.00107) | 2023 | Music SSL backbone — **fallback** for Experiment C if MuQ integration blocked |
| **MuQ: Self-Supervised Music Representation (Tencent AILab)** | 2025 | **Primary Experiment C backbone** — 81.5% VocalSet technique acc; Mel-RVQ tokens; replaces MERT |
| Self-Supervised Contrastive Learning for Singing Voices (IEEE) | 2022 | +9.12% VocalSet technique F1 using pitch-shift contrastive pairs |
| SingMOS: Open-Source Singing Voice Dataset for MOS Prediction (arXiv:2406.10911) | 2024 | Quality benchmark — already in plan |
| Automatic Detection and Analysis of Singing Mistakes for Music Pedagogy (arXiv:2602.06917) | 2026 | M3 head training data |
| CVSM: Contrastive Vocal Similarity Modeling (arXiv:2510.03025) | 2024 | Vocal style embeddings via artist-identity contrastive learning |
| Hybrid dual-path: Conformer+Transformer for singing separation (ScienceDirect) | 2024 | TCN+Conformer hybrid architecture for local+global context |
| ChordSync: Conformer-Based Alignment of Music Audio (arXiv:2408.00674) | 2024 | Conformer effectiveness for fine-grained temporal music tasks |

### Commercial Apps Scoring Signals

| App | Scoring signals | What's missing vs VocalCoach |
|---|---|---|
| Smule | Pitch + tone + rhythm composite; instant visual feedback; social sharing | No technique classification |
| Yousician | Star ratings (1–5), XP/progression bars, combo meters; pitch deviation in cents | No technique or quality head |
| Generic karaoke | Pitch deviation, timing offset, volume consistency, vibrato presence | No coaching feedback generation |

**Applicable UX lessons**:
- **Phrase-level scoring** over frame-level for better UX (natural feedback granularity)
- **Comparative scoring**: user tone delta vs. professional reference (PopBuTFy-style)
- **Star rating output** from quality head (map MOS [1–5] to stars) for coaching gamification
- **Streak mechanics**: consecutive correct technique outputs ("3 clean vibratos in a row")

---

## 6. Priority Experiment Queue

> **Status as of 2026-05-11**: Runs 1–7 completed. Run 7 (`tcn_vocalset_gtsinger_noncausal_aug`) is the first technique run — shows vibrato F1=0.854, belt F1=0.049, straight F1=0.400, **breathy F1=0.000** (class imbalance collapse). VDR=0/RPA=nan in Run 7 confirmed the blocker described above. Runs 8–10 below are the next to execute.

| Priority | Run | Key flags | Why | Status |
|---|---|---|---|---|
| **1** | Run 8: TCN + technique + `--data-dir` | `--arch tcn --data-dir data --technique-dirs data/vocalset data/gtsinger_technique --augment noise_specaug` | Fixes VDR=0 from Run 7 (pitch posterior calibration); also address breathy collapse via class weighting | **TODO** |
| **2** | Run 9: Conformer technique-only | `--arch conformer --technique-dirs data/vocalset data/gtsinger_technique --augment noise_specaug` | Conformer baseline; VDR=0 expected — compare to Run 10 | **TODO** |
| **3** | Run 10: Conformer + technique + `--data-dir` | `--arch conformer --data-dir data --technique-dirs data/vocalset data/gtsinger_technique --augment noise_specaug` | Main result for the paper | **TODO** |
| **4** | Experiment C: **MuQ** backbone | Integrate MuQ into model.py, then `--arch muq --data-dir data --technique-dirs data/vocalset data/gtsinger_technique` | **MERT replaced by MuQ** (higher VocalSet technique acc, Mel-RVQ tokens). Needs model.py integration first | **NOT STARTED** |

---

## 7. Recommended Plan Updates

### Add to Phase 0/1 (low effort, implement now)

- [ ] **Feature 27** — Vibrato onset latency in `features.py`
- [ ] **Feature 28** — Register/passaggio detection in `features.py` (H1-H2 + pitch track)
- [ ] **Feature 29** — Singer's formant energy ratio [2–4 kHz] in `features.py`
- [ ] `phrase_aggregate()` function in `features.py` — phrase-level coaching output (UX critical)
- [ ] Per-head loss logging in `train.py`
- [ ] Pitch-conditioned technique head (concatenate detached pitch peak into technique head input)

### Add to Phase 2 (medium effort)

- [ ] Voicing threshold auto-calibration via grid search in `evaluate.py`
- [ ] GTSinger technique test evaluation added to `update_results.py`
- [ ] SINGSTYLE111 dataset integration
- [ ] Formant tracking (F1/F2) via LPC in `features.py`
- [ ] M3 mistake detection head — contingent on dataset availability
- [ ] NUS-48E integration for articulation analysis

### Keep as planned (still correct)

- Phase 2: SingMOS-Pro quality head + PopBuTFy data
- Phase 3: Whisper lyric ASR → word-level pitch errors + LLM critique (Claude API)
- Phase 4: ONNX Runtime Web browser deployment

### Elevate from Stretch to Phase 2/3

- **M3 head (mistake detection)**: Directly supported by arXiv:2602.06917 dataset/methodology. Contact authors for dataset access — if available, this is the highest-value Stretch item to elevate.
- **Passaggio detection**: Should move from "not planned" to Phase 2 — critical for voice training coaching.

---

## 8. Academic Contribution Summary

> "We design VocalCoach — a multi-task singing analysis model trained and compared across two architectures (non-causal TCN, Conformer), evaluating pitch accuracy on MIR-1K, technique classification F1 on VocalSet, and perceptual quality correlation on SingMOS-Pro against human expert ratings. We show that [winning architecture] provides competitive pitch accuracy (vs. NanoPitch GRU baseline, 96.1% rtRPA) while additionally enabling multi-label technique classification and natural language coaching feedback via Whisper lyric alignment and LLM critique — demonstrating that a unified offline coaching model trained on publicly available singing datasets is practical without requiring separate per-task models. A causal variant supporting live browser deployment via ONNX Runtime Web is also provided."

---

## Sources

- [MERT: Acoustic Music Understanding Model](https://arxiv.org/abs/2306.00107)
- [Automatic Detection and Analysis of Singing Mistakes for Music Pedagogy](https://arxiv.org/abs/2602.06917)
- [CVSM: Contrastive Vocal Similarity Modeling](https://arxiv.org/abs/2510.03025)
- [SingMOS: Open-Source Singing Voice MOS Dataset](https://arxiv.org/abs/2406.10911)
- [Self-Supervised Contrastive Learning for Singing Voices (IEEE 2022)](https://www.researchgate.net/publication/360437001_Self-Supervised_Contrastive_Learning_for_Singing_Voices)
- [Hybrid dual-path: Conformer+Transformer for Singing Voice Separation](https://www.sciencedirect.com/science/article/abs/pii/S0167639324001420)
- [ChordSync: Conformer-Based Chord Alignment](https://arxiv.org/html/2408.00674v1)
- [Acoustic Analysis for Chest-to-Head Register Transition](https://www.researchgate.net/publication/336615571_Acoustic_Analysis_for_Chest-to-Head_Register_Transition_in_Singing_Voice)
- [NHSS: Speech and Singing Parallel Database](https://ar5iv.labs.arxiv.org/html/2012.00337)
- [SINGSTYLE111: Multilingual Singing Dataset with Style Transfer](https://www.cs.cmu.edu/~music/shuqid/SingStyle111__A_Multilingual_Singing_Dataset_With_Style_Transfer.pdf)
- [GTSinger: GitHub Repository](https://github.com/AaronZ345/GTSinger)
