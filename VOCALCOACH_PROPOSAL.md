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

| Metric              | Best run                              | Value     | Note                                  |
| ------------------- | ------------------------------------- | --------- | ------------------------------------- |
| **mF1 (VocalSet)**  | Run 10 (joint, best loss)             | **0.810** | VDR=26.7% — pitch unusable            |
| **mF1 (probe-mode)**| Run 19 `conformer_probe_technique`    | **0.395** | VDR=61.5%, RPA=99.5% — deployed       |
| **mAP**             | Run 10                                | **0.833** | Same caveat as mF1                    |
| **Clip Acc**        | Run 12 `conformer_vocalset_rescaled`  | **63.5%** | vs MuQ SOTA 81.5% / AST 82.0%         |
| **Vibrato F1**      | Run 12 `conformer_vocalset_rescaled`  | **0.944** | Best single-class                     |
| **Breathy F1**      | Run 10                                | **0.919** | Best single-class                     |

**Trade-off summary:** Best mF1 (Run 10, 0.810) comes at the cost of VDR=26.7% — the model skips ~73% of voiced frames and is unusable for pitch/VAD coaching. The probe-mode checkpoint (Run 19) sacrifices ~41 mF1 points to preserve pitch tracking. This is the correct trade-off for a coaching app where pitch accuracy is primary.

---

## Evaluation Strategy

### Signal-Processing Features (Phase 2 — Implemented)

Derived from the VocalCoach model's F0 + VAD outputs. Computed post-inference, no additional training required:

| Feature              | Method                                                    | Notes                                                                                                                                  |
| -------------------- | --------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| Vibrato rate/depth   | Autocorrelation of detrended F0 residual, 4–8 Hz window  | Requires ≥300 ms voiced segment. Conflicts with classifier-based `technique.vibrato` — both measure vibrato but via independent paths   |
| HNR, jitter, shimmer | Cycle-to-cycle F0 statistics                              | Values unreliable in absolute terms due to noisy probe-mode F0; meaningful only relative to PopBuTFy population medians                |
| DTW pitch deviation  | Pure-numpy DTW in cents space, mean-normalised            | Key-invariant; measures pitch contour shape match vs a reference recording                                                             |
| Phrase segmentation  | VAD gaps ≥150 ms split phrases; <100 ms segments dropped | Uses pitch-confidence proxy for voicing (30% of clip max) in probe checkpoint                                                           |

### Population Baselines (D5 — Implemented)

`scripts/buildPopBuTFyBaselines.py` ran VocalCoach inference across all 28,508 PopBuTFy clips (14,525 amateur + 13,983 professional) to build population-level medians. Output: `data/popbutfy_baselines.json`.

Each coaching report includes a `population_context` block comparing the user's metrics to amateur/professional medians with band labels: `pro_range` / `approaching_pro` / `amateur_range` / `below_amateur`.

**Known limitation:** Most acoustic metrics (HNR, jitter, shimmer) show small amateur/professional separation because both groups are studio recordings and the noisy probe-mode F0 inflates all cycle-to-cycle variance estimates equally.

### Perceptual Quality Scoring (D6 — Partially Implemented, Under Review)

**SingMOS-Pro** (wav2vec2-large, trained on SVS/SVC/SVR data): implemented in `vocalcoach/singmos.py`. Scores audio quality on a 1–5 MOS scale.

**Problem identified:** SingMOS-Pro does not meaningfully differentiate amateur from professional singing in PopBuTFy — both groups score ~4.8 (ceiling effect). Both groups are studio recordings; MOS measures perceived acoustic quality / naturalness, not singing skill. End users of a vocal coaching app are not expected to have studio microphones, so MOS is not a useful signal for the core coaching task.

**Plan:** Replace SingMOS-Pro with a singing-specific perceptual scorer. Candidate evaluated:

| Model | Size | Output | Amateur/Pro differentiation | Status |
| --- | --- | --- | --- | --- |
| **SongEvalGenerator / MuQ** (audioscore) | ~321M (MuQ-large ~300M + head ~21M) | 5 scores: Coherence / Musicality / Memorability / Clarity / Naturalness [1–5] | Poor separation observed on PopBuTFy — same ceiling issue as SingMOS-Pro | Colab cells 7a–7c in `notebooks/VocalVerse1_Colab.ipynb`; local eval in `scripts/score_vocalverse2.py` |

VocalVerse1 (Qwen2-Audio-7B + 4 LoRA adapters, ~15 GB) removed from consideration — model size makes it impractical for any offline or online pipeline. Both SingMOS-Pro and SongEvalGenerator score similarly on PopBuTFy (ceiling effect ~4.8/5 for both amateur and professional), indicating the problem is not the model choice but the lack of skill-discriminative training signal. See *Perceptual Quality Scoring — Extended Options* section for the path forward.

Both SongEvalGenerator and the original VocalVerse2 are from Wang et al., *"Singing Timbre Popularity Assessment Based on Multimodal Large Foundation Model"*, ACM MM 2025 ([doi:10.1145/3746027.3758148](https://doi.org/10.1145/3746027.3758148)).

---

## Coaching Pipeline 

Full inference pipeline implemented (`vocalcoach/api.py`):

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

1. **Perceptual scorer:** SingMOS-Pro and SongEvalGenerator both show ceiling effects (~4.8/5) on PopBuTFy — no meaningful amateur/pro separation. VocalVerse1 (7B) removed from consideration. Path forward: contrastive calibration + scoring head on VocalCoach backbone (see *Perceptual Quality Scoring — Extended Options*).
2. **mF1 gap (probe-mode 0.395 vs joint-best 0.810):** Deeper probe head (`Linear(128→64)→GELU→Linear(64→5)`) + 100 epochs expected to gain 10–15 mF1 points.
3. **Belt F1=0.000 in probe runs:** Belt is the rarest VocalSet class; pos_weight tuning needed.
4. **Falsetto coverage:** VocalSet has no falsetto. Requires a new data source or using GTSinger technique clips with cleaned labels.
5. **RMVPE F0 for acoustic features:** Would fix HNR/jitter/shimmer accuracy and improve vibrato detection — independent of VocalCoach model training.
6. **Live causal TCN retrain:** TCN probe (Run 21, VDR=76.9%) is the confirmed live candidate; causal retrain pending.

---

## Perceptual Quality Scoring — Extended Options (Research Phase)

Three approaches evaluated for replacing SingMOS-Pro, which showed ceiling effects (~4.8/5 for both amateur and professional PopBuTFy clips).

---

### Option A — Better Benchmark Datasets

The goal is a benchmark that separates real amateur from professional singers across multiple quality dimensions.

**Dataset landscape (as of May 2026):**

| Dataset | Audio | Quality labels | Amateur singers | License | Notes |
| --- | --- | --- | --- | --- | --- |
| **ccmusic-database/acapella** | ✅ HuggingFace | 9-dim expert scores (Pitch, Rhythm, Timbre, Breath, Vibrato, Dynamic, Pronunciation, Vocal Range, Overall) — 4 judges, China Conservatory of Music | Wide ability range (score 1.25–10) | CC-BY-NC-ND 4.0 | 132 clips, 22 singers, Mandarin pop; best available multi-dim expert labels |
| **SingEval** | ⚠️ Requires DAMP access | Crowdsourced holistic quality scores (public on GitHub) | ✅ Real Smule karaoke amateurs | Research only | 400 clips, 4 songs × 100 performers; audio not freely available |
| **SingMOS-Pro** | ✅ HuggingFace | 3-dim MOS (Lyrics, Melody, Overall) — 78 annotators, 7,981 clips | ❌ Synthesized voice (SVS/SVC) | Open-source | Best for pre-training a scoring head; domain gap to real singing |
| **SongEval** | ✅ HuggingFace | 5-dim expert MOS (Coherence, Memorability, Breathing/Phrasing, Structure, Musicality) — Shanghai Conservatory | ❌ Professional/produced recordings | CC-BY-NC-SA 4.0 | 2,399 songs, 140h; full songs with accompaniment |
| **VocalSet** | ✅ Zenodo (CC-BY 4.0) | Technique labels only (no quality/MOS scores) | ❌ All professional singers | CC-BY 4.0 | 10h, 20 singers, 17 techniques; best for technique-aware encoding |
| **DAMP / SingEval audio** | ❌ No longer available | — | ✅ Real Smule karaoke amateurs | Smule proprietary | Access programme discontinued; removed from consideration |

**Key gap:** No public dataset combines real untrained amateur audio + multi-dimensional expert quality labels at meaningful scale. The closest pairing is:

- **Professional reference:** ccmusic-database/acapella (9-dim expert scores, but pro-dominant)
- **Amateur proxy:** PopBuTFy amateur side (simulated — professional singers deliberately performing poorly, not real novices)

**For MOS labels on ccmusic/acapella:** Run SongEvalGenerator (Option C below, ~321M params) or SingMOS-Pro (~95M) — no need for the 7B QwenFeat model.

**Other publicly available MOS/quality models:**

| Model | Params | Output | Weights | Notes |
|---|---|---|---|---|
| **SongEvalGenerator** (audioscore, this repo) | ~321M | 5-dim: Coherence / Musicality / Memorability / Clarity / Naturalness [1–5] | ✅ In HF repo | MuQ backbone; no Qwen dependency; fastest to try |
| **SingMOS-Pro predictor** | ~95M | 3-dim: Lyrics / Melody / Overall MOS [1–5] | ✅ HuggingFace (TangRain/SingMOS-Pro) | Trained on synthesized voice; domain gap to real singing |
| **SingMOS predictor** | ~95M | Scalar MOS [1–5] | ✅ PyTorch Hub (South-Twilight/SingMOS) | Single dim; trained on SVS/SVC outputs |
| **TG-Critic** | Small | Scalar quality score | Partial (GitHub YuejieGao/TG-CRITIC) | Reference-free; CQT-based; no confirmed weights |
| **SCOREQ** | Unknown | Scalar quality | ✅ MIT (`pip install scoreq`) | Speech-domain; domain gap to singing |
| UTMOS | ~95M | Scalar MOS | ✅ Apache 2.0 | Poor on singing; do not use without re-fine-tuning |

---

### Option B — Lightweight Scoring Head on VocalCoach Backbone

Attach quality regression heads directly to the existing VocalCoach Conformer backbone. The backbone already runs for every coaching inference; adding a scoring head costs negligible additional compute.

**How it works:**

1. Take the mean-pooled hidden states from the frozen VocalCoach Conformer backbone
2. Add parallel linear heads — one per quality dimension (e.g. Pitch Accuracy, Breath Control, Timbre, Overall)
3. Train heads only; backbone stays frozen (same MERT-style linear probe approach used for `head_technique`)

**Inference overhead:** A linear head maps `hidden_dim → n_dims` (e.g. 256 → 4). This adds ~1K parameters and sub-millisecond latency — effectively zero relative to the backbone forward pass.

**Training pipeline (recommended):**

1. Pre-train head on **SingMOS-Pro** (7,981 clips, 3-dim MOS) — establishes quality-discriminative representations in the head weights
2. Fine-tune probe on **ccmusic-database/acapella** (132 clips, 9-dim expert scores) — maps to domain-specific dimensions (Breath Control, Timbre, Vibrato, etc.)
3. Apply **contrastive calibration** on **PopBuTFy pairs** — anchors the output scale to real amateur vs. professional separation

**What is contrastive calibration?**
Rather than training with absolute MOS scores ("this clip is 3.2/5"), contrastive calibration trains with *relative pairs*: "clip A (professional) should score higher than clip B (amateur)." PopBuTFy provides this supervision for free — every `(pro, amateur)` pair is a labelled ranking without any human MOS annotation. The model learns to push professional scores above amateur scores, calibrating where each skill tier lands on the output scale. This uses a pairwise ranking loss (e.g. margin ranking loss: `max(0, margin − (score_pro − score_amateur))`). It does not change what the model measures — it anchors the scale to real skill separation observed in the data.

**Data constraint:** 132 clips (ccmusic) is at the edge of the linear probe regime. Pre-training on SingMOS-Pro first is essential. The VoiceMOS 2024 semi-supervised track demonstrated SSL-MOS heads can be trained with "very small amounts of labeled data" — 132 expert-scored clips is feasible for a linear probe on a frozen backbone.

**Key prior art:**

- PS-SQA (arXiv:2411.11123, Nov 2024) — SSL + regression head for singing, low-resource bias correction, VoiceMOS 2024 Track 2 winner
- SAMOS (arXiv:2411.11232, Nov 2024) — multi-task SSL heads for singing MOS
- SingMOS-Pro (arXiv:2510.01812, Oct 2025) — multi-dim singing MOS predictor with released weights

---

### Option C — SongEvalGenerator (audioscore, Lightweight Standalone)

The `audioscore/` sub-directory of the QwenFeat-Vocal-Score repository contains a fully standalone scoring system requiring no Qwen dependency.

**Architecture:**

```text
MuQ-large-msd-iter  (~300M params, music SSL encoder, 24 kHz input)
         ↓  hidden_states[6]  (layer 6, shape [B, T, 1024])
SongEvalGenerator   (~21M params)
  ├─ FFD: Linear(1024→4096) → ReLU → Linear(4096→1024)
  ├─ 4× MultiheadAttention(d=1024, h=8, dropout=0.2)
  ├─ Linear(2048→5)  [mean-pool + max-pool concatenated]
  └─ Tanh() * 2.0 + 3   →  scores in range [1, 5]
         ↓
5 dimensions: Coherence · Musicality · Memorability · Clarity · Naturalness
```

**Key facts:**

- Total: ~321M params vs ~7B for QwenFeat LoRA system (~22× smaller)
- Checkpoint: `ckpts/SongEvalGenerator/step_2_al_audio/best_model_step_132000/` — 151 MB total (`muq_lora.pt` 50 MB + `weights.pt` 101 MB)
- Production `generate_tag()` returns a single scalar (index 1 / Musicality, inverted); the standalone `SongEval/eval.py` exposes all 5 dimensions
- No Qwen dependency; `audioscore/requirements.txt` lists only `muq`, `peft`, `transformers`, `torchaudio`, `librosa`, `soundfile`, `safetensors`

**Approximate dimension mapping to VocalCoach dimensions:**

| SongEvalGenerator dim | VocalCoach / QwenFeat approximate equivalent |
|---|---|
| Musicality | Overall quality proxy (most useful for coaching) |
| Clarity | Vocal Technique / diction / articulation |
| Naturalness | Breath Control / phrasing |
| Coherence | Emotional Expression / musical continuity |
| Memorability | Timbre / voice distinctiveness |

**Implementation status:** Colab cells 7a → 7b → 7c added to `notebooks/VocalVerse1_Colab.ipynb`:

- **7a**: Downloads audioscore source + SongEvalGenerator checkpoint (~200 MB; skips the 7B model entirely)
- **7b**: Loads the 321M model, defines `lite_score(path)` helper
- **7c**: Side-by-side comparison table — SongEvalGenerator vs QwenFeat 4-LoRA on the same PopBuTFy clips, with per-clip timing

**Inference overhead vs VocalCoach (offline eval):**

| Model | Params | Relative size | Approx time/clip | Use case |
| --- | --- | --- | --- | --- |
| VocalCoach Conformer/TCN | ~450K | 1× | ~5–10 ms | Real-time + offline |
| SingMOS-Pro (wav2vec2-base head) | ~95M | ~211× | ~1–2 s | Offline only |
| SongEvalGenerator (MuQ + head) | ~321M | ~713× | ~3–8 s | Offline only |
| Option B scoring head on VocalCoach | ~450K + ~1K | ~1× | sub-ms extra | Real-time + offline |

SongEvalGenerator is ~713× larger than VocalCoach. For one-time baseline-building across the full PopBuTFy corpus it is acceptable (run once, save JSON). For per-clip coaching feedback it is too heavy — Option B (head on the existing backbone) is the right path there.

**Status:** Both SingMOS-Pro and SongEvalGenerator show the same ceiling effect on PopBuTFy — neither is a useful drop-in without addressing the fundamental training signal problem. The path forward is Option B + contrastive calibration (see ordered plan below).

---

## Scoring Head Variants — Implementation Plan

All three variants share the same frozen VocalCoach Conformer backbone (pitch + VAD + technique heads frozen, probe-mode checkpoint Run 19). Only `head_quality` differs across variants. Implement in order: Variant 1 is the baseline, Variant 2 is the multi-dimension extension, Variant 3 is an alternative initialization strategy that can be combined with either of the first two.

Metric Conflicts (noisy probe-mode F0, vibrato classifier disagreement) are in the acoustic feature path and are **independent** of this scoring head track — they do not need to be fixed first.

---

### Variant 1 — Contrastive scoring head on PopBuTFy pairs (baseline, implement first)

**Training objective:** Pairwise margin ranking loss using PopBuTFy same-singer amateur/professional pairs. No absolute MOS labels required — every `(pro, amateur)` pair from the same singer on the same song provides free supervision. The head must learn to rank `score(pro) > score(amateur)` by at least `margin`.

```python
# For each (pro_clip, amateur_clip) pair from PopBuTFy:
ranking_loss = F.margin_ranking_loss(
    score_pro, score_amateur,
    target=torch.ones_like(score_pro),  # pro should score higher
    margin=0.5
)
```

Backbone is frozen (probe mode). Only `head_quality` (a small linear layer: `hidden_dim → 1`) is trained. Output scale is arbitrary — normalize to 0–100 at display time.

**Why PopBuTFy for contrastive:** Same singer, same song, different skill level — recording quality, singer identity, and song difficulty are all controlled out. This is the cleanest possible pairwise signal. AudioScore's validated +0.36 pro−amateur delta on these pairs (vs SingMOS-Pro's inverted −0.05) confirms the signal is real.

**Training data:** ~28K PopBuTFy clips → thousands of valid `(pro, amateur)` pairs per song.

**Deliverable:** `head_quality` added to `vocalcoach/model.py`; training script with ranking loss; validation: check pro−amateur delta on held-out PopBuTFy clips and ccmusic low vs high scorers.

---

### Variant 2 — Multi-dimension scoring head supervised by ccmusic expert labels (implement second)

**What changes vs Variant 1:** Instead of a single scalar output trained contrastively, `head_quality` outputs **multiple dimensions** corresponding to ccmusic's 9 expert-rated categories: Pitch, Rhythm, Timbre, Breath Control, Vibrato, Dynamic, Pronunciation, Vocal Range, Overall.

**Why this is meaningful:** AudioScore's per-dimension Pearson r values (pitch +0.659, rhythm +0.627, vocal_range +0.623, timbre +0.599) show that the VocalCoach backbone already encodes enough information to predict each dimension independently — the head just needs to be trained to separate them. A multi-dimension output maps directly to the radar-chart display in the coaching report.

**Important clarification on what the r values mean and do not mean:** AudioScore outputs **one single scalar** per clip. The per-dimension r values are computed by taking AudioScore's scalar output across all 132 ccmusic clips and correlating it against each of ccmusic's 9 expert columns separately (e.g., `pearsonr(audioscore_scores, ccmusic_pitch_labels)` → r=+0.659). These r values are *validation evidence* that the backbone representations co-vary with each expert dimension — they do not mean AudioScore can decompose its output into pitch/rhythm/timbre numbers. For Variant 2, ccmusic's 9 expert columns become 9 separate MSE regression targets for `head_quality` directly. AudioScore is not involved in Variant 2's training signal.

**Training pipeline:**

1. Pre-train `head_quality` (9 outputs) on **SingMOS-Pro** (7,981 clips, 3-dim MOS: Lyrics/Melody/Overall) — establishes MOS-scale calibration with large N; the 3 MOS dims supervise a subset of the 9 output neurons (Overall, Breath/Melody proxy, Timbre/Naturalness proxy).
2. Fine-tune on **ccmusic-database/acapella** (132 clips, 9-dim expert scores, CC-BY-NC-ND 4.0) — maps all 9 output dimensions directly to expert labels via MSE. 132 clips is at the edge of the linear probe regime; SingMOS-Pro pre-training is essential here.
3. Contrastive calibration on **PopBuTFy pairs** (same as Variant 1) — anchors the Overall dimension's absolute scale to real skill separation.

**Data constraint note:** 132 clips for 9 regression targets is tight. Use L2 regularization on head weights and early stopping on a ccmusic held-out split (10–15 clips). Pre-training on SingMOS-Pro is not optional.

**Deliverable:** `head_quality` with 9 outputs; per-dimension scores in coaching report JSON; radar chart updated in `VocalCoach_Inference.ipynb`.

---

### Variant 4 — Note Segmentation Head (note-level pitch accuracy from learned head)

**What this adds:** A new frame-level head `head_note` that predicts note onset and offset probabilities at every time step, trained jointly with the pitch and VAD heads (not with technique). At inference the model produces note boundaries directly from audio — no post-hoc signal processing needed.

**Training data:** `data/annotated_vocalset/note_train.npz` — 824 VocalSet clips, 18,412 annotated notes, 755/824 clips with note boundaries. Derived from the Annotated VocalSet dataset (Kim et al.) using per-frame onset/offset markers.

**Why not GTSinger for this:** GTSinger's technique NPZ (`technique_train.npz`) has no note-level timing annotations — only clip-level technique labels and RMVPE-derived F0. The 9,601 GTSinger clips provide pitch/VAD supervision but no note segmentation labels. Only the Annotated VocalSet subset (824 train clips) provides onset/offset supervision.

**Is 824 clips enough to converge?** Probably yes for onset detection, with caveats:

- 18,412 notes across ~150 min of audio is a reasonable scale for a thin binary classification head on top of frozen conformer features
- VocalSet is controlled (isolated vocal exercises, clean studio audio) — onsets are clean and consistent; the head won't fight background noise during training
- The frozen backbone already encodes temporal dynamics (conformer global attention) — the head only needs to learn to read onset signatures from existing representations, not learn them from scratch
- The real risk is generalisation: VocalSet exercises have regular, evenly-spaced notes; real singing has irregular phrasing. Plan to validate on a held-out VocalSet split (female4/female8/male2/male4 test set, ~200 clips) before deploying

**Architecture:** Two thin output heads sharing backbone features with pitch/VAD:

```python
head_note_onset  = nn.Linear(hidden_dim, 1)   # P(onset at frame t)
head_note_offset = nn.Linear(hidden_dim, 1)   # P(offset at frame t)
```

Training loss: binary cross-entropy on onset/offset frames (same supervision source as VAD). Note frames are sparse (~1–2% of frames are onsets) — use `pos_weight` tuning as with technique head.

**Training:** Joint with pitch/VAD in Stage 1. The backbone is **not** frozen for this head — note segmentation benefits from backbone co-adaptation with pitch, since onset correlates with F0 onset transients. Add `--w-note` weight arg (recommend starting at 0.5 to avoid dominating pitch loss).

**What this enables at inference — note-level coaching metrics without any extra compute:**

| Metric | How computed | What it catches that frame-level RPA misses |
| --- | --- | --- |
| **Note-level pitch accuracy** | For each predicted note segment, take median F0; compare to nearest MIDI pitch within 100 cents | Singer who overshoots onset then corrects — frame RPA looks fine, note accuracy is low |
| **Intonation drift (cents)** | Median(predicted F0) − nearest_MIDI per note, averaged | Systematic flat/sharp tendency per key or phrase |
| **Pitch stability within note** | Std dev of F0 within predicted note window | Distinguishes intentional vibrato from unsteady pitch; quantifies wobble |
| **Attack speed** | Frames from predicted onset until F0 reaches ±50¢ of target | Measures how cleanly the singer "finds" the note; coaches note preparation |

**Comparison to post-hoc signal processing:** Frame-level RPA already reports pitch accuracy across voiced frames. The key difference is segmentation: post-hoc note finding (peak-picking on VAD gaps, autocorrelation-based methods) is noisy on real singing with vibrato and glides. A learned onset detector trained on annotated note boundaries should give cleaner note segments, making per-note statistics more reliable — especially `attack speed` and `intonation drift`, which require accurate note start times.

**Training command (Stage 1 with note head, run after extractAnnotatedVocalSet):**

```bash
python vocalcoach/train.py \
    --data-dir        data \
    --technique-dirs  data/vocalset/technique_train.npz \
                      data/gtsinger_technique/technique_train.npz \
                      data/annotated_vocalset/note_train.npz \
    --arch            conformer --causal false \
    --seq-len         600 --lr 1e-3 \
    --w-vad 0.05 --w-pitch 2.0 --pitch-sigma 0.8 --w-note 0.5 \
    --augment noise specaugment \
    --epochs 120 --eval-every 5 --patience 30 \
    --ckpt-dir checkpoints/stage1_pitch_note \
    --run-name stage1_conformer_note_head
```

**Deliverable:** `head_note_onset` + `head_note_offset` in `model.py`; note-level metrics in `evaluate.py`; per-note accuracy, drift, stability, and attack speed in coaching report JSON.

---

### Variant 3 — Scalar MSE distillation from AudioScore as initialization (alternative to contrastive pre-training)

**What this is:** Instead of starting `head_quality` from random weights, pre-train it via MSE regression against AudioScore's scalar output on a large scored corpus, then fine-tune with contrastive ranking (as in Variant 1 or 2). This is an **initialization strategy**, not a standalone variant — it replaces the SingMOS-Pro pre-training step.

**Why not PopBuTFy for scalar distillation:** AudioScore's absolute scale on PopBuTFy clips is uncalibrated — the pro−amateur gap is only +0.36 on a 1–5 scale, which is weak regression signal. MSE on poorly-separated values produces a head that learns to output ~3.5 for everything.

**Better datasets for scalar distillation:**

| Dataset | N clips | AudioScore scores available | Human MOS available | Notes |
| --- | --- | --- | --- | --- |
| **SingMOS-Pro** (TangRain) | 7,981 | ✅ `data/singmos_pro_seg_scores.json` | ✅ `score.json` in HF snapshot | Best option — large N, human MOS ground truth, AudioScore already correlated at r=+0.301 vs GT clips |
| **ccmusic-database/acapella** | 132 | ✅ `data/combined_eval_baselines.json` | ✅ 9-dim expert scores | Small but highest-quality labels; use for fine-tune not pre-train |
| **Combined** | ~8K | ✅ both cached | ✅ both available | Pre-train on SingMOS-Pro, fine-tune on ccmusic |

**Why SingMOS-Pro works for this:** AudioScore r=+0.301 vs human MOS on GT (real singer) clips means the model's scalar output is meaningfully correlated with perceived singing quality at population scale. Pre-training `head_quality` to match AudioScore's output on these 7,981 clips gives the head a calibrated starting point before contrastive fine-tuning on PopBuTFy.

**Recommended combined pipeline (Variant 3 + Variant 1 or 2):**

```text
1. MSE pre-train head_quality on SingMOS-Pro
   → target: AudioScore scalar scores (data/singmos_pro_seg_scores.json)
   → loss: MSE(head_output, audioscore_score)

2. Fine-tune on ccmusic (Variant 2 only)
   → target: 9-dim expert scores
   → loss: MSE per dimension

3. Contrastive calibration on PopBuTFy pairs
   → loss: margin_ranking_loss(score_pro, score_amateur, margin=0.5)
```

**Deliverable:** Modified training script with 3-stage pipeline; ablation comparing random-init contrastive (Variant 1) vs distillation-init contrastive (Variant 3 + 1) on ccmusic Pearson r and PopBuTFy pro−amateur delta.

---

### Replacing SingMOS-Pro in the API

Once any variant validates meaningful amateur/professional separation (target: ≥0.3 point gap on ccmusic low vs high scorers, correct direction on PopBuTFy pairs):

- Remove external AudioScore/SingMOS-Pro model call from `vocalcoach/singmos.py`
- Route quality scores through `head_quality` outputs instead
- Update population context chart in `VocalCoach_Inference.ipynb` with the new dimensions

**Deliverable:** `vocalcoach/singmos.py` replaced or wrapped; updated coaching report JSON schema.

---

### Not blocking the above (fix later, independently)

- **Metric Conflicts (noisy F0):** RMVPE-based F0 extractor would fix HNR/jitter/shimmer and vibrato detection. Independent of perceptual scoring. Low priority until coaching report quality becomes the bottleneck.

---

## Future Scope — YouTube Cover Dataset

A scalable source of real amateur singing data with implicit professional references:

**Approach:** Collect audio from YouTube covers of 10–15 popular pop songs (e.g. Rolling in the Deep, Hallelujah, Let It Go, Shallow). Label original artist official uploads as **professional** reference; all cover versions as **amateur** proxy. This gives a large, naturalistic amateur corpus without manual labelling.

**Implementation sketch:**

```bash
pip install yt-dlp demucs

# Download original (professional reference)
yt-dlp -x --audio-format wav -o "data/yt_covers/pro/%(title)s.%(ext)s" "<official video URL>"

# Download covers playlist (amateur)
yt-dlp -x --audio-format wav -o "data/yt_covers/amateur/%(title)s_%(id)s.%(ext)s" "<playlist URL>"

# Separate vocals from backing track before scoring
python -m demucs --two-stems=vocals data/yt_covers/amateur/*.wav
```

**Key design decisions before starting:**

- Vocal isolation via `demucs` (two-stem: vocals + accompaniment) is required before running SingMOS or SongEvalGenerator — both models expect dry vocal audio
- Song selection: pick songs with many covers (100+ results on YouTube) and clear original artist recordings
- Quality filtering: discard clips shorter than 30s or with very low audio bitrate

**Why this is future scope and not current priority:** The existing PopBuTFy + ccmusic + VocalSet combination covers the controlled skill range well enough for Steps 1–3. YouTube scraping adds breadth but not controlled pairs — the per-song original/cover pairing is weaker supervision than PopBuTFy's same-singer amateur/professional pairs. Revisit once the scoring head (Step 3) needs more training data to generalise.

- **mF1 gap (probe 0.395 vs joint 0.810):** Deeper probe head + more epochs. Independent task.
- **Belt F1=0.000:** pos_weight tuning. Independent.
- **Falsetto coverage:** Needs new data source. Independent.
- **Live causal TCN retrain:** Independent of quality scoring track.

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
