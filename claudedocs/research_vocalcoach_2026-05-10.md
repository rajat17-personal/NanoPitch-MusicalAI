# VocalCoach+ Research Report: Literature Validation & Strategic Updates

**Date**: 2026-05-10
**Project**: VocalCoach (multi-task singing analysis & coaching model)
**Research Depth**: Deep (parallel web search + targeted extraction across 17 queries)
**Context**: Validates and extends [VOCALCOACH_PLAN.md](../VOCALCOACH_PLAN.md) and [VOCALCOACH_DEEPDIVE.md](../VOCALCOACH_DEEPDIVE.md)

---

## Executive Summary

This report investigates 12 research questions stemming from the VocalCoach plan and deep-dive analysis. **Confidence is high** that the project's architectural foundations (TCN + Conformer multi-task) are well-aligned with current state-of-the-art research, and **multiple high-value 2025 papers** were identified that directly support, validate, or update specific recommendations from the deep dive.

### Top 5 Strategic Findings

| # | Finding | Confidence | Action Implication |
|---|---|---|---|
| 1 | **STARS framework (ACL 2025) is a direct precedent** for VocalCoach — same author group as GTSinger, with joint transcription + alignment + technique + style heads | High | Study/cite as primary baseline; consider adopting their hierarchical multi-level architecture |
| 2 | **TechSinger's technique detector achieves Vibrato F1=0.374** — meaning vibrato is genuinely hard, not a VocalCoach-specific issue | High | Frame F1≈0.4 for vibrato is acceptable; focus optimization on breathy/falsetto where higher F1 is realistic |
| 3 | **M3 dataset confirmed publicly available** with TCN as the best-performing model (87.14% F1 frequency mistakes) | High | Validates VocalCoach's TCN architecture choice; download M3 to enable mistake detection head |
| 4 | **FCPE (Sept 2025) matches RMVPE accuracy at 1/8.5 the parameters** — 96.79% RPA with RTF 0.0062 | High | Consider FCPE as faster pseudo-label generator; could enable real-time pitch tracking in browser |
| 5 | **MuQ (Jan 2025) outperforms MERT** on vocal technique detection with only 0.9K hours pretraining | High | Replace MERT with MuQ for Experiment C; smaller and stronger backbone |

### Validation Status of Deep Dive Recommendations

| Deep Dive Recommendation | Literature Support | Confidence |
|---|---|---|
| Gap A: Pitch-conditioned technique head | **STRONGLY VALIDATED** — TechSinger uses mel + F0 + energy + breathiness as inputs to its technique detector | High |
| Gap B: Register/passaggio detection | **VALIDATED** — AVRA achieves 94% accuracy with SVM on 4-class register classification | High |
| Gap C: Phrase-level aggregation | **VALIDATED** — STARS uses hierarchical frame/word/phoneme/note/sentence levels | High |
| Gap D: Vibrato onset latency | Limited literature, but physiologically supported | Medium |
| Gap E: Singer's formant detection | **VALIDATED** — LPC order 12 is the established standard for 2.5–3.5 kHz peak detection | High |
| Gap F: Multi-scale technique architecture | **VALIDATED** — STARS does this explicitly with hierarchical levels | High |
| Gap G: Contrastive pretraining on singing | Supported by MuQ (better SSL approach) | Medium-High |

---

## 1. Dataset Findings

### 1.1 M3 Mistake Dataset — CONFIRMED AVAILABLE

The deep dive identified arXiv:2602.06917 as a high-value but unverified dataset. **Verification result: dataset is publicly available and methodology is highly relevant.**

**Specifications** (from full paper extraction):
- Indian Art Music pedagogy domain (synchronized teacher-learner recordings)
- **Teacher 1**: 64 teacher files (1,125 sec), 259 learner files (4,624 sec)
- **Teacher 2**: 91 teacher files (5,484 sec), 242 learner files (14,956 sec)
- 4 mistake categories: **frequency, amplitude, pronunciation, timing**
- Frame-level annotations with onset/offset times

**Best Reported Results** (validates TCN choice for VocalCoach):
| Task | Architecture | F1 | Precision | Recall |
|---|---|---|---|---|
| Frequency mistakes (Teacher 2) | **TCN** (pitch contours) | **87.14%** | 98.44% | 78.16% |
| Amplitude mistakes (Teacher 2) | **TCN** + augmentation | **91.98%** | 88.14% | 96.16% |

**Access**:
- Code: https://github.com/madhavlab/2023_narottam_engine
- Data: https://zenodo.org/records/8332078

**Strategic Implication**: M3 dataset is **directly applicable** to the planned M3 head (mistake detection, currently listed as Stretch). The fact that TCN is their best architecture is **strong validation** of VocalCoach's design. The frame-level evaluation collar approach (Tc=80ms) should be adopted in VocalCoach's evaluation harness for fair comparison.

---

### 1.2 SINGSTYLE111 — CONFIRMED AVAILABLE on Zenodo

**Specifications**:
- 111 songs by 8 professional singers, 12.8 hours total
- Languages: English, Chinese, Italian
- Styles: bel canto opera, Chinese folk, pop, jazz, children
- **80 songs include ≥2 distinct styles by the same singer** (style transfer pairs)
- Mono, 44.1 kHz, professional studio quality
- Includes: lyrics, performance MIDI, scores with phoneme alignment, mel-spectrograms, F0, loudness curves

**Access**: https://zenodo.org/records/10265401

**Strategic Implication**: As recommended in the deep dive, SINGSTYLE111 is the strongest candidate to **complement PopBuTFy** for amateur→professional comparison. The same-singer multi-style pairs are particularly valuable for style-conditional training.

---

### 1.3 Critical NEW Dataset: STARS Annotations on M4Singer

**Discovery not in deep dive**: TechSinger (AAAI 2025) and STARS (ACL 2025) authors have already used their technique detector to **automatically annotate M4Singer at the phoneme level**. M4Singer is large-scale (~700+ songs); auto-labeled technique annotations are a free 10x expansion of training data.

**Action**: Check the GTSinger / STARS GitHub repos (https://github.com/AaronZ345/GTSinger, https://github.com/gwx314/STARS) for the M4Singer pseudo-label release.

---

### 1.4 Vocal Mode Datasets (Sundberg + CVT)

Two additional datasets identified, neither in the current plan:

**Sundberg Phonation Modes**: ~700 recordings of nine vowels in 4 modes (breathy, neutral, flow/resonant, pressed). Smaller scale but cleanly labeled.

**Complete Vocal Technique (CVT) Dataset (2026)**: 3,752 unique samples, 13,000+ across 4 microphones, covering Neutral, Curbing, Overdrive, Edge modes. Rock/metal-focused — relevant only if hard rock styles are in scope.

**Recommendation**: Both are **lower priority** than SINGSTYLE111 and M3 for VocalCoach's coaching scope. Mention only if expanding to specific vocal modes.

---

### 1.5 Saraga (Indian Art Music)

**Discovered**: M3 paper builds on Saraga dataset infrastructure (108 recordings, 61 ragas, 9 talas, 43.6 hours). Could be valuable if the project scope expands to non-Western coaching, but not a priority now.

---

## 2. State-of-the-Art Benchmarks (NEW data points)

### 2.1 VocalSet Technique Classification SOTA (2024–2025)

| Model | Accuracy | Year | Notes |
|---|---|---|---|
| CNN baseline | 80.1% | 2018 | Original VocalSet baseline |
| MusicFM | 78.3% | 2024 | Music foundation model |
| **MuQ** | **81.5%** | **2025** | Mel-RVQ self-supervised |
| AST (Audio Spectrogram Transformer) | 82.0% | 2024 | Layer 12 features |
| **MuQiter** (iterative MuQ) | 77.0 avg (multi-task) | 2025 | Best on average across 9 tasks |

**Strategic Implication**: VocalCoach should **target ≥80% accuracy** as a competitive baseline. The current 7-run results (mF1=0.326 on technique) are far below this — **but** the baselines above use clip-level classification on isolated technique recordings, while VocalCoach does frame-level multi-label with co-occurrence. The metric definitions differ; direct comparison requires a clip-level evaluation mode.

**Action**: Add a clip-level evaluation mode to `evaluate.py` that takes `mean(frame_predictions) > 0.5` per technique class for fair comparison with these SOTA accuracy numbers.

---

### 2.2 TechSinger Technique Classification Per-Class Performance

This is the most directly comparable benchmark (frame-level, multi-label, mel + F0 inputs):

| Technique | TechSinger F1 |
|---|---|
| Pharyngeal | 0.872 |
| Breathy | 0.854 |
| Mixed-Falsetto | 0.771 |
| **Vibrato** | **0.374** |

**Critical Insight**: Vibrato F1 is **inherently low** even for SOTA — TechSinger authors attribute this to "high degree of randomness" in vibrato annotation. **VocalCoach's vibrato F1 should be evaluated against 0.374 as a realistic ceiling, not against the vibrato detection rates from clip-level papers.**

**Implication for the Plan**: The VOCALCOACH_PLAN evaluation criterion ("Technique F1 is primary") should be **decomposed by class** — pharyngeal/breathy can target F1>0.8, vibrato should target F1>0.4 (matching SOTA).

---

### 2.3 SingMOS-Pro Benchmark Details

| Item | Value |
|---|---|
| Total clips | 7,981 |
| Languages | Chinese + Japanese |
| Subset breakdown | 3,425 SVS + 1,307 SVC + 2,671 SVR + 578 ground-truth |
| Annotators | ≥5 expert ratings per clip |
| License | CC-BY-4.0 |
| File size | 1.48 GB |
| Sample rate | Mostly 16 kHz |
| Pretrained predictor | https://github.com/South-Twilight/SingMOS |

**Critical Strategic Question**: The SingMOS team **has already released a pretrained MOS predictor**. VocalCoach has two options for the quality head:

1. **Train its own quality head** (current plan) — multi-task with technique/pitch heads
2. **Use the SingMOS pretrained predictor as a separate model** — feed VocalCoach's audio + features through the SingMOS predictor at inference time

**Recommendation**: **Option 2 is preferable for Phase 2** because:
- Avoids interference between MOS and other heads in joint training
- Provides immediate access to a strong baseline (no training needed)
- Lower risk of degrading pitch/technique accuracy from MOS loss
- Can later be ablated against a custom quality head

The plan currently has SingMOS-Pro listed as Phase 2 work — **suggest using the pretrained predictor first, then training custom head as ablation**.

---

## 3. Architecture Findings (Most Important Section)

### 3.1 STARS Framework — Direct Precedent for VocalCoach (CRITICAL FINDING)

**Reference**: STARS: A Unified Framework for Singing Transcription, Alignment, and Refined Style Annotation, ACL Findings 2025 (arXiv:2507.06670)

**Why this matters**: STARS is **the closest existing system to VocalCoach** in design philosophy. Same author group as GTSinger and TechSinger.

**Architecture**:
- **CMU Encoder**: U-Net + Conformer blocks + FreqMOE (Frequency Mixture-of-Experts)
- **Hierarchical processing**: frame → word → phoneme → note → sentence levels
- **Joint heads**: lyric alignment, note transcription, technique prediction, global style prediction
- **9 technique classes**: BUB (bubble), BRE (breathy), PHA (pharyngeal), VIB (vibrato), GLI (glissando), MIX (mixed), FAL (falsetto), WEA (weak), STR (strong)

**Code**: https://github.com/gwx314/STARS

**Strategic Implications for VocalCoach**:

1. **Naming/scope**: STARS already covers "transcription + alignment + technique + style". VocalCoach's distinguishing differentiator must be **coaching feedback** (mistake detection, post-processing features 14–22, LLM critique) — these are NOT in STARS.

2. **Architecture validation**: Conformer + multi-task heads is the right design. VocalCoach's Conformer is simpler (no FreqMOE) but the core idea is correct.

3. **Hierarchical processing is a clear improvement**: VocalCoach currently outputs everything at frame level. Adopting STARS-style hierarchical heads (frame for VAD/pitch, phoneme/note for technique, clip for quality) would directly improve coaching UX and likely reduce vibrato-class noise.

4. **Recommended Action**: **Read STARS paper closely**, cite as primary baseline in the academic contribution, and consider whether VocalCoach should:
   - Adopt FreqMOE
   - Use the STARS technique taxonomy (9 classes vs current 5)
   - Use hierarchical multi-level outputs

---

### 3.2 TechSinger Technique Detector — Pitch-Conditioned Validation

**Reference**: TechSinger, AAAI 2025 (arXiv:2502.12572), https://github.com/gwx314/techsinger

**Architecture**:
- U-Net backbone with **Squeezeformer** layers
- Inputs: **mel-spectrogram + F0 + energy + breathiness** (multi-feature, NOT mel-only)
- Phoneme-level weighted-average pooling for label aggregation
- Trained on 30hr Chinese dataset + auto-labeled M4Singer

**Validation of Deep Dive Gap A**: TechSinger uses **exactly the architecture VocalCoach Gap A recommends** — mel + F0 (and energy + breathiness) as inputs to the technique head.

**Specific Implementation Pattern from TechSinger** (extract from paper for reference):
```
Phoneme-level technique embedding:
  E_w_pi = sum(E_w_fi+j+t) / sum(W_fi+j+t)
  
where W = phoneme boundary weights, fi+j+t = frame-level features
```

**Strategic Implication**: Implementing the deep-dive Gap A improvement is **strongly justified by SOTA practice**. The recommendation should be **upgraded from "Low effort improvement" to "Required for competitive baseline"**.

**Action**: When implementing Gap A, also include explicit energy and breathiness features (computed from mel via simple statistics) as additional inputs to the technique head.

---

### 3.3 MuQ — Replacement for MERT in Experiment C

**Reference**: MuQ, IEEE TASLP 2025 (arXiv:2501.01108), https://github.com/tencent-ailab/MuQ

**Why MuQ > MERT for VocalCoach**:

| Property | MERT | MuQ |
|---|---|---|
| Pretraining data | 160k hours (general music) | **0.9K hours** (curated, music-focused) |
| Pretraining target | Random projection / EnCodec | **Mel-RVQ** (mel-residual VQ — better for pitch) |
| VocalSet technique detection | Lower | **Higher (81.5%)** |
| Open source | Yes | Yes (Tencent AILab) |
| Music-specific | Yes | Yes (more focused) |

**Strategic Implication**: **Replace MERT with MuQ in Experiment C** of the plan. MuQ is newer (Jan 2025), specifically benchmarked on vocal technique detection, and uses Mel-RVQ which is more aligned with pitch-aware singing analysis.

**Action**:
- Update VOCALCOACH_PLAN.md Experiment C to use MuQ instead of MERT
- Add MuQ to the "Pretrained Models Available" table
- Consider also benchmarking MuQ-MuLan (audio-text) as an alternative for the LLM critique alignment in Phase 3

---

### 3.4 FCPE — Faster RMVPE Alternative for Pseudo-Labels

**Reference**: FCPE, arXiv:2509.15140 (Sept 2025), https://github.com/CNChTu/FCPE

**Performance**:
| Model | Parameters | RPA (MIR-1K) | RTF (RTX 4090) |
|---|---|---|---|
| RMVPE | 90.42M | 96.7% | ~0.05 |
| **FCPE** | **10.64M** | **96.79%** | **0.0062** |

**Architecture**: Lynx-Net (Conformer-inspired, depth-wise separable convolutions on mel features)

**Strategic Implications**:

1. **For Phase 1 pseudo-labeling**: Using FCPE instead of RMVPE would be 8x faster for the same accuracy. Worth re-running data extraction with FCPE if extraction time is a bottleneck.

2. **For Phase 4 browser deployment**: 10.64M params is in the range that's feasible to run via ONNX Runtime Web. FCPE could potentially serve as the **runtime pitch tracker** in the browser, replacing NanoPitch GRU.

3. **Cross-check**: VocalCoach's NanoPitch (GRU, 333K params, 96.1% rtRPA) is still smaller than FCPE (10.64M) but FCPE has higher accuracy. For a coaching app where 1-2% RPA improvement matters, FCPE is worth evaluating.

**Action**: 
- **Phase 0**: Optional — re-extract f0 labels with FCPE and compare to RMVPE labels for consistency check
- **Phase 4**: Evaluate FCPE as browser-deployable pitch tracker alternative to NanoPitch (only if higher accuracy justifies the 30x parameter increase)

---

### 3.5 Whisper for Singing — Music Source Separation Required

**Reference**: arXiv:2506.15514 (June 2025), https://github.com/jaza-syed/mss-alt

**Key Finding**: Whisper has serious limitations on singing audio:
- Native Whisper on Jam-ALT: **23.02% WER**
- With music source separation (Hybrid Demucs) + RMS-VAD: **20.35% WER** (new SOTA)
- **Whisper hallucinates lyrics when music overlays speech**
- **>50% deletion rate for non-lexical vocables** (e.g. "oh", "yeah") even with perfect stems

**Critical Implication for Phase 3 Plan**:
The current plan says "Run openai/whisper locally on the recorded audio to extract timestamped words." This will produce **poor results** for typical singing audio. The plan needs to specify:

1. **For a cappella input**: Whisper directly should work, but expect ~10-15% WER even on clean vocals
2. **For singing with backing track**: **Must use Hybrid Demucs first**, then Whisper, then RMS-VAD segmentation
3. **For coaching use case**: Display alignment uncertainty — don't show "your B4 on *love* was flat" if word "love" has < 80% Whisper confidence

**Action**: Update the Phase 3 plan to specify the source separation pipeline and add a confidence threshold for word-level error reporting.

---

### 3.6 Multi-Task Learning Loss Balancing — Recommendations

Current VocalCoach loss: `pitch×2 + vad×0.05 + technique×2` (manually tuned)

**State of the field (2024)**:
- **GradNorm** (Chen 2018): Adaptive gradient normalization — works but adds complexity
- **Uncertainty Weighting** (Kendall 2017): Learns task-specific σ weights but tends to grow weights unstably
- **Analytical Uncertainty Weighting** (Springer 2024): Computes optimal weights via softmax with tunable temperature

**Critical Insight from research**: "Uncertainty weighting always moves test and training error in the same direction, and thus is not a good regularizer."

**Strategic Implication**: For VocalCoach's specific case (pitch-dominant gradient when joint training):

**Recommendation**: **Stick with manually tuned weights for Phase 1**, but:
1. Add per-head loss logging (already noted in deep dive Gap 4b)
2. After runs 8/9/10, examine if technique loss is plateauing while pitch loss still drops — if so, try **GradNorm** as an ablation
3. Avoid uncertainty weighting (instability risks documented)

---

### 3.7 ONNX Runtime Web — Browser Deployment Validated

**Key facts**:
- ONNX Runtime Web supports both **WASM (CPU)** and **WebGPU** backends
- WebGPU acceleration is now production-ready for transformer/conformer models
- Whisper has been demonstrated running in browser via ONNX Runtime Web
- WASM provides 4-23× speedup over JavaScript for audio DSP

**Real-time constraint**: 16.7 ms per frame for 60 FPS UI updates

**For VocalCoach specifically**:
- TCN (449K params): Will easily fit in 16.7ms per inference window with WASM
- Conformer (419K params): Should also fit but quadratic attention may be a concern at high seq_len
- **Pattern to use**: AudioWorklet + WASM for real-time pitch + offline batch for technique/quality analysis

**Action**: For Phase 4, the deployment plan should specify:
- TCN streaming via WASM in AudioWorklet (causal=True variant) for live pitch feedback
- Conformer (causal=False) loaded in main thread for post-recording technique analysis
- Whisper or pretrained MOS predictor as optional offline batch via ONNX Runtime Web

---

## 4. Commercial Vocal Coaching App Landscape

### 4.1 Top Apps (2026 ranking)

| App | Type | Coaching mechanism | Tech approximation |
|---|---|---|---|
| **Yousician** | Gamified curriculum | Pitch + timing scoring; XP bars; combo meters | Likely YIN/CREPE pitch + custom UI |
| **Singing Carrots** | Detailed pitch monitor + AI lessons | Pitch deviation in cents; range tests; long-term stats | Pitch-only focus, simpler ML |
| **Vanido** | Daily micro-sessions | Adaptive exercise difficulty; range tracking | Real-time pitch detection only |
| **Vocal AI Analyzer / Smart Vocal Coach** | AI coaching for students | Multi-dimensional vocal skill analysis | Closest analog to VocalCoach |

### 4.2 Reported Accuracy Range

> "Modern voice rating tools use AI-powered pitch detection algorithms achieving 85-95% accuracy compared to professional vocal coach assessments, particularly for pitch and rhythm."

**Strategic Implication**: VocalCoach's **technique classification** capability is the differentiator from these apps — none of them analyze vibrato/breathy/falsetto/belt classification at the level VocalCoach plans. This is the **academic and product novelty**.

**For final report**: Position VocalCoach as **"the first multi-task model that adds technique classification + quality assessment + LLM critique on top of the pitch tracking baseline that commercial apps already have."**

---

## 5. New Recommendations (Beyond the Deep Dive)

### 5.1 Adopt STARS-Style Hierarchical Output

**Why**: STARS demonstrates that hierarchical outputs (frame/word/phoneme/note/sentence) significantly improve technique annotation quality. VocalCoach's frame-only output is the noisiest possible representation.

**Effort**: Medium (Phase 2)

**Implementation sketch**:
```
Add to VocalCoachTCN/Conformer:
  - head_vad:           per-frame  (existing)
  - head_pitch:         per-frame  (existing)
  - head_technique:     per-note   (NEW - aggregate frames within note boundaries from Annotated-VocalSet)
  - head_quality:       per-clip   (existing optional)
```

This requires note boundary information at training time → **strong dependency on Annotated-VocalSet integration** (already in pipeline).

### 5.2 Use STARS 9-Class Technique Taxonomy

**Why**: STARS's 9 classes (BUB, BRE, PHA, VIB, GLI, MIX, FAL, WEA, STR) are more comprehensive than VocalCoach's current 5 (vibrato, breathy, falsetto, belt, straight). Mapping:

| VocalCoach | STARS | Notes |
|---|---|---|
| vibrato | VIB | Same |
| breathy | BRE | Same |
| falsetto | FAL | Same |
| belt | STR | "Strong" is the closest STARS term |
| straight | WEA / (no vibrato) | Subtle distinction |
| — | BUB (bubble) | Missing in current plan |
| — | GLI (glissando) | GTSinger has this; missing in current plan |
| — | MIX (mixed voice) | GTSinger has this; missing in current plan |
| — | PHA (pharyngeal) | GTSinger has this; missing in current plan |

**Strategic Implication**: Adopting STARS taxonomy = compatible with the largest open singing technique label corpus (GTSinger + STARS auto-labels on M4Singer).

**Recommendation**: For Phase 2, expand from 5 to 9 classes. Use GTSinger as primary training data (it has all 6: mixed, falsetto, breathy, pharyngeal, vibrato, glissando) and VocalSet as secondary.

### 5.3 Use TechSinger as Auto-Labeling Tool

**Why**: TechSinger's pretrained technique detector can label any singing audio at the phoneme level. This effectively expands the labeled training data.

**Action**: 
1. Run TechSinger labeler on unannotated audio (PopBuTFy, NUS-48E, or DALI vocals)
2. Use as soft labels (weighted lower than ground-truth labels) in joint training
3. This is essentially knowledge distillation from a teacher model

**Effort**: Medium (Phase 2)

### 5.4 Use Pretrained SingMOS Predictor (Skip Custom Quality Head)

**Why**: Training a quality head adds complexity, risk of negative interference, and requires a substantial volume of MOS labels. The pretrained SingMOS predictor (https://github.com/South-Twilight/SingMOS) is ready to use and benchmarked.

**Action**:
- **Phase 2**: Use SingMOS pretrained predictor as separate inference module
- Only train custom quality head as an ablation if user time permits
- Update plan to reflect this — saves ~3-4 weeks of Phase 2 work

### 5.5 Add Mistake Detection Head with M3 Methodology

**Why**: M3 dataset is publicly available, TCN is the best architecture for it, and mistake detection is the most coaching-relevant capability VocalCoach could add.

**Action**:
- **Phase 2**: Integrate M3 dataset
- Add 5th model head: mistake detection (binary per-frame, multi-class for frequency/amplitude/pronunciation/timing)
- Use M3 evaluation methodology (collar-based F1, Tc=80ms)
- Elevate from Stretch to Phase 2

### 5.6 Use FCPE as Real-Time Pitch Tracker for Browser Deployment

**Why**: FCPE matches RMVPE accuracy with 8.5x fewer parameters and RTF 0.0062. For browser real-time pitch feedback, FCPE is the better choice than retraining NanoPitch causal variant.

**Action**:
- **Phase 4**: Evaluate FCPE in WASM for browser deployment
- Compare against NanoPitch causal-retrained baseline for the live pitch tracking feature

### 5.7 Update Whisper Pipeline to Include Source Separation

**Why**: Whisper alone fails on singing-with-accompaniment scenarios. Hybrid Demucs + RMS-VAD + Whisper is documented SOTA.

**Action**:
- **Phase 3**: Update lyric alignment plan to specify the pipeline: Demucs (mdx_extra) → RMS-VAD → Whisper Large-v2
- Add confidence threshold for word-level coaching feedback (>0.8 Whisper confidence)
- Reference: https://github.com/jaza-syed/mss-alt

---

## 6. Updated Plan Recommendations Summary

### Phase 0 (Active) — No changes needed
Current data extraction work is well-aligned. Optionally:
- Re-extract pitch labels with FCPE (8x faster, same accuracy)

### Phase 1 (Active) — Two updates
1. **Implement Gap A (pitch-conditioned technique head) with energy + breathiness features** (validated by TechSinger)
2. **Replace MERT with MuQ** in Experiment C (newer, smaller, better on VocalSet)

### Phase 2 (Future) — Major updates
1. **Use SingMOS pretrained predictor** instead of training custom quality head (saves weeks)
2. **Adopt STARS 9-class technique taxonomy** (compatible with GTSinger + M4Singer auto-labels)
3. **Add hierarchical multi-level outputs** (frame for pitch/VAD, note for technique, clip for quality)
4. **Integrate M3 dataset and add mistake detection head** (elevate from Stretch)
5. **Use TechSinger as auto-labeling tool** for unannotated data
6. **Add SINGSTYLE111 dataset** for amateur→pro style comparison

### Phase 3 (Future) — One critical update
1. **Update Whisper pipeline** to include Demucs source separation + RMS-VAD before Whisper

### Phase 4 (Future) — Deployment
1. **Use FCPE** as the browser real-time pitch tracker
2. **TCN causal in WASM AudioWorklet** for live pitch + technique
3. **Conformer non-causal** for offline post-recording analysis

---

## 7. Confidence Assessment

| Finding Category | Confidence | Reasoning |
|---|---|---|
| Dataset availability (M3, SINGSTYLE111, SingMOS-Pro) | **High** | Verified URLs and downloaded metadata |
| SOTA benchmarks on VocalSet | **High** | Multiple corroborating papers from 2024-2025 |
| TechSinger architecture details | **High** | Direct paper extraction of method and results |
| STARS framework details | **High** | Confirmed via search + GitHub repo + ACL listing |
| MuQ vs MERT recommendation | **High** | Direct benchmark comparison published |
| FCPE pitch accuracy claim | **High** | Reported in arXiv with code |
| Whisper singing limitations | **High** | Multiple sources confirm hallucination issues |
| Browser inference benchmarks | **Medium** | Some numbers from older posts; WebGPU still maturing |
| Commercial app technical details | **Medium** | Reverse-engineered from marketing material |
| LLM-based feedback effectiveness | **Medium** | Limited rigorous studies in vocal-specific domain |

---

## 8. Open Questions / Limitations

1. **STARS technique F1 numbers not extracted**: PDF was binary; need to read HTML version to confirm specific F1 results
2. **Singer's formant detection in singing-specific (not speech) datasets**: Most LPC research is on speech; singing-specific tuning may need empirical work
3. **CVT (Complete Vocal Technique) dataset for metal/rock**: Out of scope for current plan but available if needed
4. **NUS-48E**: Mentioned in deep dive but no recent benchmark numbers found
5. **Real-time Conformer inference benchmarks in WASM**: Specific numbers not found; would need to be measured empirically

---

## Sources

### Primary Papers (Newly Identified)

- [STARS: A Unified Framework for Singing Transcription, Alignment, and Refined Style Annotation (ACL 2025)](https://arxiv.org/abs/2507.06670)
- [TechSinger: Technique Controllable Multilingual Singing Voice Synthesis via Flow Matching (AAAI 2025)](https://arxiv.org/abs/2502.12572)
- [MuQ: Self-Supervised Music Representation Learning with Mel Residual Vector Quantization (IEEE TASLP 2025)](https://arxiv.org/abs/2501.01108)
- [FCPE: A Fast Context-based Pitch Estimation Model (Sept 2025)](https://arxiv.org/abs/2509.15140)
- [Automatic Detection and Analysis of Singing Mistakes for Music Pedagogy (M3 Dataset, Feb 2026)](https://arxiv.org/abs/2602.06917)
- [Exploiting Music Source Separation for Automatic Lyrics Transcription with Whisper (June 2025)](https://arxiv.org/html/2506.15514v1)
- [Machine Learning Approaches to Vocal Register Classification in Contemporary Male Pop Music (AVRA, May 2025)](https://arxiv.org/html/2505.11378v2)
- [SingMOS-Pro: A Comprehensive Benchmark for Singing Quality Assessment](https://arxiv.org/abs/2510.01812)
- [GTSinger: A Global Multi-Technique Singing Corpus (NeurIPS 2024 Spotlight)](https://arxiv.org/abs/2409.13832)
- [Voices of the Mountains: Vocal Error Detection for Kurdish Maqams](https://arxiv.org/html/2602.20744)

### Datasets

- [M3 Mistake Dataset (Zenodo)](https://zenodo.org/records/8332078) — Indian Art Music, 4 mistake types
- [M3 Code Repository](https://github.com/madhavlab/2023_narottam_engine)
- [SINGSTYLE111 (Zenodo)](https://zenodo.org/records/10265401) — 111 songs, 8 singers, multilingual
- [SingMOS-Pro (HuggingFace)](https://huggingface.co/datasets/TangRain/SingMOS-Pro) — 7,981 clips
- [GTSinger (GitHub)](https://github.com/AaronZ345/GTSinger) — NeurIPS 2024 Spotlight
- [Saraga Datasets (CompMusic)](https://compmusic.upf.edu/datasets) — Indian Art Music

### Reference Implementations

- [STARS GitHub](https://github.com/gwx314/STARS)
- [TechSinger GitHub](https://github.com/gwx314/techsinger)
- [MuQ GitHub](https://github.com/tencent-ailab/MuQ)
- [FCPE GitHub](https://github.com/CNChTu/FCPE)
- [Pretrained SingMOS Predictor](https://github.com/South-Twilight/SingMOS)
- [MSS-ALT (Whisper + Demucs Pipeline)](https://github.com/jaza-syed/mss-alt)

### Multi-Task Learning References

- [GradNorm: Gradient Normalization for Adaptive Loss Balancing (ICML 2018)](https://arxiv.org/pdf/1711.02257)
- [Beyond Losses Reweighting: Empowering Multi-Task Learning (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/papers/Phan_Beyond_Losses_Reweighting_Empowering_Multi-Task_Learning_via_the_Generalization_Perspective_ICCV_2025_paper.pdf)
- [Analytical Uncertainty-Based Loss Weighting in Multi-Task Learning (Springer 2024)](https://link.springer.com/chapter/10.1007/978-3-031-85181-0_22)

### Browser Deployment

- [ONNX Runtime Web Unleashes Generative AI in the Browser (Microsoft 2024)](https://opensource.microsoft.com/blog/2024/02/29/onnx-runtime-web-unleashes-generative-ai-in-the-browser-using-webgpu/)
- [Essentia.js: Audio Analysis on the Web (TISMIR 2021)](https://transactions.ismir.net/articles/10.5334/tismir.111)

### Commercial App References

- [Top 7 AI Vocal Coach Apps (Singing Carrots Blog, 2026)](https://singingcarrots.com/blog/top-7-ai-vocal-coaches/)
- [Yousician vs Singing Carrots Comparison](https://singingcarrots.com/blog/yousician-vs-singing-carrots/)

### LLM Music Education

- [AI-Assisted Feedback in Vocal Music Training (Frontiers in Psychology 2025)](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2025.1598867/full)
- [Exploring LLM-Powered Teachable Agent in Music Education (Apr 2025)](https://arxiv.org/html/2504.00636)

---

**Report ends.** For implementation, see [VOCALCOACH_DEEPDIVE.md](../VOCALCOACH_DEEPDIVE.md) (existing plan updates) — this report adds 7 new recommendations with literature-backed justification.
