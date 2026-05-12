# VocalCoach — Project Plan

**Course:** Musical AI (Final Project)
**Author:** Rajat Sharma
**Last updated:** 2026-05-11

---

## Scope Decisions (2026-05-10)

Following the brainstorm and research review (see [VOCALCOACH_DEEPDIVE.md](VOCALCOACH_DEEPDIVE.md) and [claudedocs/research_vocalcoach_2026-05-10.md](claudedocs/research_vocalcoach_2026-05-10.md)):

| Decision | Choice | Implication |
| --- | --- | --- |
| Project type | Academic final project + demo | Multi-arch comparison and dataset breadth justified; not a shipping product |
| Target genre | Genre-agnostic | No style classifier; technique outputs are descriptive, not evaluative-by-genre |
| Target user | Intermediate/expert primary + beginner UI toggle | Two-tier LLM critique prompts; same model |
| Interaction mode | Offline-primary + light live | VocalCoachTCN (causal retrain) for live pitch — primary target. NanoPitch GRU as fallback if time runs out or Conformer wins Phase 1. |

**Out of scope** (consciously dropped — see "Out of Scope" section near end of doc):
custom MOS quality head (use pretrained SingMOS), 9-class STARS technique expansion, causal Conformer retrain, M3 mistake detection head (genre transfer risk), Sing-MD few-shot critiques, SINGSTYLE111 amateur/pro pairs, style-aware feedback.

---

## Project Overview

VocalCoach is a multi-task singing analysis model that goes beyond pitch
tracking to provide actionable coaching feedback. It is a **new project**, not
an extension of NanoPitch. NanoPitch (GRU baseline, ~333K params, 96.1% rtRPA)
serves only as a comparison baseline for pitch accuracy experiments **and** as
the live-mode pitch tracker in the demo (offline analysis runs on VocalCoach).

The system takes a singing audio clip and produces:

| Output | Type | Where used |
| --- |--- | --- |
| VAD (voiced/unvoiced) | Per-frame probability | Breath detection, note segmentation |
| F0 pitch track | Per-frame 360-bin posteriorgram | All pitch-related coaching metrics |
| Technique label | Multi-label per frame | Vibrato / breathy / falsetto / belt-or-strong (5→4 classes — see Feature Taxonomy) |
| Quality score (MOS) | Per-clip scalar (via pretrained SingMOS predictor) | Sanity-check signal; **not a custom head** |
| DTW reference distance | Per-phrase scalar | "How far are you from the target?" — primary coaching metric |
| Acoustic features | Post-processing | HNR, spectral tilt, dynamics, jitter, shimmer, singer's formant |
| Natural language critique | LLM API call (Claude) | Two-tier output: expert (numerical) and beginner (encouragement) modes |

---

## Architecture Decision: TCN vs Conformer

**Phase 1 trains both architectures with `causal=False` (non-causal, offline).**
Offline is the primary interaction mode (see Scope Decisions above), so the
non-causal variants give maximum accuracy without compromising the demo path.

**Live mode plan (priority order):**
1. **Primary target — VocalCoachTCN causal retrain**: TCN with `causal=True` exports cleanly to ONNX and runs in browser via `onnxruntime-web` (Conv1d + Linear fully supported, no WASM hacks). If TCN wins Phase 1, retrain with `causal=True` for streaming live pitch + technique.
2. **If Conformer wins Phase 1** — live mode falls back to NanoPitch GRU. `MultiheadAttention` is ONNX-exportable but Conformer's quadratic attention over the full clip is not streaming-compatible without chunking; the engineering cost is high relative to the demo benefit.
3. **Safe fallback — NanoPitch GRU** (333K params, 96.1% rtRPA, already deployed under `deployment/`): use as live pitch indicator if time runs out or architecture decision makes live VocalCoach impractical. No new work required.

| Property | VocalCoachTCN | VocalCoachConformer | NanoPitch GRU (fallback) |
| --- |--- | --- | --- |
| Backbone | Dilated Conv1d stack | Conv + Multi-head Self-Attention (Macaron) | GRU |
| Offline mode | `causal=False` | `causal=False` | N/A (pitch+VAD only) |
| Live mode | `causal=True` retrain (target) | Not practical for streaming | Already deployed |
| Tasks | VAD + Pitch + Technique | VAD + Pitch + Technique | Pitch + VAD only |
| Context | ~5 s geometric (dilation doubles per block) | Full clip (global attention) | Recurrent |
| Default params | ~449K (hidden=128, 8 blocks) | ~419K (hidden=64, 4 layers) | ~333K |
| ONNX export | Clean (Conv1d + Linear) | Exportable but not streaming | Already deployed |

**Experiment C** (Phase 1) also tests a pretrained **MuQ** backbone (Tencent
AILab, Jan 2025) with lightweight classification heads — establishing whether
self-supervised music pretraining significantly reduces the labelled singing
data needed. **MuQ replaces MERT** as the primary pretrained backbone because
it reports 81.5% on VocalSet technique detection vs MERT's lower numbers, and
uses Mel-RVQ tokens which are more pitch-aware than MERT's targets. MERT
remains as a fallback if MuQ checkpoints prove unstable.

Architecture selection criterion: technique classification F1 (primary,
per-class — see Phase 1 evaluation), with pitch accuracy as a secondary
constraint (must remain within 5% of NanoPitch's 96.1% rtRPA on the same
test split).

### Deployment Path (Phase 4)

ONNX Runtime Web is the recommended browser deployment option for both
architectures — no C code or Emscripten required, just a single export call:

```python
torch.onnx.export(model, dummy_input, "vocalcoach.onnx",
    input_names=["mel"], output_names=["vad", "pitch", "technique"],
    dynamic_axes={"mel": {1: "time"}})
```

Then run in browser via `onnxruntime-web` (JS library). TCN exports cleanly
(Conv1d + Linear ops fully supported). Conformer exports too — PyTorch's
`MultiheadAttention` is ONNX-compatible.

**Live mode does NOT require Conformer retrain** — the existing NanoPitch
GRU runs in the browser via the deployment artifacts under `deployment/`
and provides per-frame pitch + VAD for the live indicator. Offline analysis
loads VocalCoach (TCN or Conformer winner) separately on the recorded clip.

---

## Pretrained Models Available

| Model | Size | Features | License | Use in Project |
| --- |--- | --- |--- | --- |
| RMVPE | 90M | Frame-level F0 (360-bin) | Public | Pitch pseudo-labels (already used) |
| FCPE | 10.6M | Frame-level F0, RTF 0.0062, 96.79% RPA | Public | Optional faster pseudo-label generator (Phase 0) |
| CREPE (tiny–full) | Configurable | Frame-level F0 | MIT | Alternative/validation pitch labels |
| **MuQ** (Tencent AILab) | ~95M | Self-supervised music tokens (Mel-RVQ) | Open | **Exp C backbone — preferred over MERT** (81.5% VocalSet technique acc) |
| MERT-v1-95M | 95M | Frame-level music features | Academic | Fallback Exp C backbone if MuQ blocked |
| **SingMOS predictor** | ~10M | Clip-level MOS prediction | Open | **Replaces custom quality head** (used at inference, not trained) |
| TechSinger detector | U-Net+Squeezeformer | Phoneme-level technique labels | Open | Optional auto-labeling tool for unlabeled audio (Phase 2) |
| Hybrid Demucs (mdx_extra) | ~80M | Music source separation | MIT | Required preprocessing for Whisper on backed singing (Phase 3) |
| Whisper (large-v2) | 1.5B | Lyric ASR | MIT | Lyric alignment after Demucs (Phase 3) |
| HuBERT (base) | 94M | Frame-level, 50 Hz | MIT | Speech transfer learning baseline (optional) |

**Key change**: SingMOS pretrained predictor is used at inference time as a sanity-check signal — **no custom quality head is trained**. This avoids multi-task interference and saves ~3-4 weeks of Phase 2 work.

---

## Full Feature Taxonomy

### From Model Heads (Learned)

| # | Feature | Head type | Notes |
| --- |--- | --- |--- |
| 1 | F0 pitch track | 360-bin sigmoid | 20-cent resolution, B0–B6 |
| 2 | VAD | Binary sigmoid | Voiced/unvoiced per frame |
| 3 | Technique class | Multi-label sigmoid | **4 classes (collapsed)**: vibrato / breathy / falsetto / strong (belt+straight merged — both indicate sustained loud production without vibrato) |
| 4 | Pitch-conditioned technique input | Concat layer | **NEW**: detached pitch posterior peak + variance fed into technique head input (validated by TechSinger AAAI 2025) |

### Post-Processing from F0 Track

| # | Feature | Method | Coaching value |
| --- |--- | --- |--- |
| 5 | Vibrato rate (Hz) | Short-time autocorrelation of F0 oscillation | Target: 5–8 Hz for trained singers |
| 6 | Vibrato depth (cents) | Amplitude of F0 oscillation | Target: 50–200 cents |
| 7 | Vibrato regularity | Coefficient of variation of rate | Consistency metric |
| 8 | Note onset / offset | F0 jumps + energy transients | Note boundary detection (covers legato) |
| 9 | Per-note pitch accuracy | Mean F0 vs nearest semitone, per note | "Your B4 was 30 cents flat" |
| 10 | Intonation stability | Within-note F0 std dev | Separates wobbly from stable notes |
| 11 | Pitch drift | Slope of F0 within a sustained note | Sharp/flat drift within a note |
| 12 | Phrase contour shape | Polynomial fit over phrase F0 | Is the melodic arch correct? |
| 13 | Legato score | Note duration + inter-note gap patterns | Phrasing and connectivity |

### From Mel / Waveform (Signal Processing, No Model)

| # | Feature | Method | Coaching value |
| --- |--- | --- |--- |
| 14 | Dynamics / RMS | Per-frame RMS energy | Volume control, crescendo/decrescendo |
| 15 | HNR (Harmonics-to-Noise Ratio) | Autocorrelation of waveform | Vocal clarity, voice health |
| 16 | Spectral centroid | Weighted mel band mean | Brightness, resonance placement |
| 17 | Spectral tilt / H1-H2 | First two harmonic amplitudes | Breathiness indicator |
| 18 | Jitter | Cycle-to-cycle F0 variation | Micro-instability, voice health |
| 19 | Shimmer | Cycle-to-cycle amplitude variation | Amplitude stability |
| 20 | MFCCs (13 coefficients) | Mel cepstrum | Timbre characterisation |
| 21 | Breath detection | High-freq noise bursts + VAD gaps | Breath placement and control |
| 22 | Vocal onset steepness | Attack envelope shape | Note attack quality (hard vs soft) |

### From Reference / Paired Data (Phase 2)

| # | Feature | Method | Coaching value |
| --- |--- | --- |--- |
| 23 | **DTW pitch distance** | DTW alignment vs reference F0 | **Primary coaching metric across genres**: "How far are you from the target?" |
| 24 | Relative quality rank | Amateur vs professional distance (PopBuTFy) | Quantified gap to professional level |

### Lyric / Score Input (Phase 3)

| # | Feature | Method | Coaching value |
| --- |--- | --- |--- |
| 25 | Phoneme-level accuracy | Lyric alignment via Demucs + Whisper pipeline | Pronunciation, vowel quality |
| 26 | Rhythm deviation | DTW to MIDI score | Timing precision |
| 27 | Vibrato onset latency | F0 autocorrelation in 300ms window post-note-onset | Classical training targets <200ms onset |
| 28 | Register / passaggio detection | H1-H2 tilt + pitch track discontinuity | Chest/mix/head register transitions |
| 29 | Singer's formant energy | Spectral energy ratio [2-4kHz] vs [0.5-2kHz] | Resonance placement |
| 30 | Per-axis output scores | 3-5 scalars per phrase (pitch / vibrato / dynamics / breath / overall) | Per-dimension feedback |
| 31 | Audio quality pre-flight | RMS, SNR, reverb estimate on input | Detects bad mic / room |
| 32 | Output confidence display | Posteriorgram entropy + technique probability margins | Surfaces model uncertainty |
| 33 | Range tracking (session) | Min/max pitch over session(s) | "Your range expanded by 2 semitones" |

---

## Data NPZ Format

All extraction scripts produce compatible output:

| Key | Shape | Dtype | Used by |
| --- | --- | --- | --- |
| `mel` | `(total_frames, 40)` | float16 | PitchVADDataset + TechniqueDataset |
| `f0` | `(total_frames,)` | float16 | PitchVADDataset + TechniqueDataset |
| `vad` | `(total_frames,)` | float16 | PitchVADDataset + TechniqueDataset |
| `lengths` | `(n_clips,)` | int32 | Both — segment boundary index |
| `technique` | `(n_clips, 5)` | float32 | TechniqueDataset only — clip-level |

---

## Phase 0 — Data Pipeline & Feature Extraction

> Legend: ✅ Done · 🔲 TODO (started) · ⬜ Not started · ❌ Dropped

| ID | Category | Task | Effort | Status |
| --- | --- | --- | --- | --- |
| D2 | Data | VocalSet extraction — `scripts/extractVocalSet.py`: mel + RMVPE F0 + VAD + clip-level technique labels (vibrato, breathy, belt, straight; vibrado merged). Train/test split: m2, m4, f4, f8 held out. | Medium | ✅ Done |
| D3 | Data | GTSinger technique labels — `scripts/extractGTSingerTechnique.py`: maps vibrato/breathy/falsetto → 5-class taxonomy, skips glissando/mixed/pharyngeal. **Known issue**: uses `any(phoneme_has_technique)` so a clip is labelled vibrato=1 if *any* phoneme has it — windows from non-vibrato portions receive noisy labels. Confirmed harmful: dropping GTSinger from technique training improved mF1 0.223→0.810. | Low | ✅ Done (noisy) |
| F3–F9 | Features | Signal processing feature module — `vocalcoach/features.py`: RMS, HNR, spectral centroid, H1-H2 tilt, jitter, shimmer, MFCCs, breath detection, vocal onset steepness (Features 14–22). Called at inference, not during training. | Low | ✅ Done |
| M-eval-h | Training | Per-head loss logging in `train.py` — print vad/pitch/technique losses separately each epoch. | Very Low | ✅ Done |
| M-eval-c | Evaluation | Clip-level accuracy in `evaluate.py` — `mean(frame_preds) > 0.5` per class; needed for fair SOTA comparison (MuQ 81.5%, AST 82.0%). | Low | ✅ Done |
| M-loader | Training | Multi-source technique loading — `TechniqueDataset` flat NPZ with `lengths`; `make_joint_loader` accepts `--technique-dirs` list. | Low | ✅ Done |
| M-aug | Training | Noise augmentation — `NoisePool` + `augment_mel_batch` + `spec_augment`. `logaddexp` log-mel mixing from `noise.npz` (FSDNoisy18k 42h). CLI: `--augment none\|noise\|noise_specaug`. | Low | ✅ Done |
| M-posw | Training | Per-class technique positive weights (`--technique-pos-weights`) — upweights minority classes in BCE to prevent all-negative collapse. Default: VocalSet+GTSinger combined ratios. | Low | ✅ Done |
| M-curr | Training | Curriculum training (`--curriculum`) — zeroes w_technique for `--curriculum-warmup` epochs, then linearly ramps to target over `--curriculum-ramp` epochs. LR schedule unaffected. | Low | ✅ Done |
| D7 | Data | Annotated-VocalSet — per-note onset/offset timestamps + MIDI pitch. Script `scripts/extractAnnotatedVocalSet.py` written; needs final integration into `evaluate.py`. Blocks: note segmentation (F8), per-note accuracy (F9), hierarchical technique head. | Medium | 🔲 Script done, integration pending |

---

## Phase 1 — Architecture Experiments

> All experiments use `causal=False` (non-causal, offline). Live mode delegated to NanoPitch GRU.

### Architecture Diagrams

```
TCN:       mel (40) → Conv1d(40→128) → TCNBlock×8 (dilations 1,2,4,…,128, symmetric pad) → LayerNorm
                    → VAD head (128→1)
                    → Pitch head (128→360)
                    → Technique head: concat(backbone, pitch_peak.detach(), pitch_var.detach()) → Linear → 5 classes

Conformer: mel (40) → Linear(40→64) → ConformerBlock×4 → LayerNorm
                    → VAD head (64→1)
                    → Pitch head (64→360)
                    → Technique head with pitch conditioning (same pattern) → 5 classes

MuQ:       raw audio → MuQ (frozen, ~95M) → frame features → Linear(MuQ_dim→128)
                     → VAD head + Pitch head + Technique head
```

Each ConformerBlock: FF(0.5) → Self-Attention (full clip) → Conv(kernel=31, symmetric) → FF(0.5) → LN

| Property | TCN | Conformer | MuQ |
| --- | --- | --- | --- |
| Params | ~449K (hidden=128, 8 blocks) | ~419K (hidden=64, 4 layers) | ~95M frozen + ~50K heads |
| Context | ~5.1s geometric | Full clip (global attention) | Full clip |
| Status | ✅ Runs complete | ✅ Runs complete | ⬜ Not started |

### Experiment Tasks

| ID | Category | Task | Effort | Status |
| --- | --- | --- | --- | --- |
| M-ExpA | Training | TCN pitch/VAD runs — 4 runs completed (offRPA up to 99.1%); see VOCALCOACH_RESULTS.md | — | ✅ Done |
| M-ExpA-t | Training | TCN first technique run (Run 7) — vibrato F1=0.854, straight 0.400, belt 0.049, breathy 0.000; VDR=0/RPA=nan (missing `--data-dir`). | — | ✅ Done |
| M-ExpA-r8 | Training | Run 8: TCN + technique + `--data-dir data` — fixed VDR=0; RPA=99.0%, mF1=0.223. VDR still low (9.3%) — pitch posteriors diffuse due to GTSinger technique noise. | — | ✅ Done |
| M-ExpA-r11 | Training | Run 11: TCN + VocalSet-only technique + curriculum + pos_weights — mF1=0.314; VDR tanked at epoch 30 (curriculum ramp onset). | — | ✅ Done |
| M-ExpA-def | Training | Run 10 ("output-dir ignored"): TCN + default hyperparams (`--w-pitch 1 --w-vad 0.5`) — mF1=0.810, clip acc=80.3% (≈SOTA), but RPA=92.4%. Default weights leave more backbone capacity for technique vs NanoPitch-tuned values. | — | ✅ Done |
| M-ExpB | Training | Conformer pitch/VAD runs — 3 runs, offRPA up to 99.3% | — | ✅ Done |
| M-ExpB-t | Training | Run 9: Conformer + technique (50 epochs only) — mF1=0.649, VDR=0/RPA=nan (undertrained, loss still high at epoch 50). | — | ✅ Done |
| M-GapA | Model | Pitch-conditioned technique head — detached pitch peak + variance into technique head input (validated by TechSinger AAAI 2025). | Low | ✅ Done |
| M-ExpB-def | Training | **Run 12 (next)**: Conformer + VocalSet-only technique + default weights — direct parallel to Run 10 (TCN mF1=0.810). Command: `--arch conformer --data-dir data --technique-dirs data/vocalset --epochs 100 --augment noise_specaug --technique-pos-weights 2.9 4.2 1.0 4.0 1.9`. Main paper result for Conformer technique comparison. | Low | ⬜ Not started |
| M-weights | Training | **Run 13 (next, parallel to Run 12)**: TCN + VocalSet-only + balanced loss weights — tests middle ground between Run 10 (default, mF1=0.810/RPA=92.4%) and NanoPitch-tuned (mF1=0.223/RPA=99%). Command adds `--w-vad 0.1 --w-pitch 1.5 --w-technique 2`. Target: RPA >95% AND mF1 >0.6. Resolves the open weight tension for the paper. | Low | ⬜ Not started |
| M-live | Deployment | Causal retrain for live mode — retrain Phase 1 winning TCN with `--causal true` + `--output-dir vocalcoach/runs/tcn_causal_live`. Only do after Run 12/13 confirm TCN as winner. Conformer fallback = NanoPitch GRU (no work needed). | Low | ⬜ Not started — blocked on arch selection |
| M-ExpC | Model+Training | Exp C: MuQ backbone integration in `model.py` + training run. MuQ reports 81.5% VocalSet technique acc. Academic-only (95M params — can't deploy to browser). Not a Phase 1 blocker. | Medium | ⬜ Not started |
| M-vdr | Training | VDR recovery check — VDR drops at curriculum ramp onset (epoch 30) as technique gradients compete with pitch. Check Run 10 TensorBoard: does VDR recover after epoch 30, or stay at ~26%? If transient, the ramp is fine; if structural, try `--curriculum-ramp 20`. NanoPitch-tuned `--w-pitch 2 --w-vad 0.05` confirmed to hurt technique vs defaults — middle ground (`--w-pitch 1.5 --w-vad 0.1`) tested in Run 13. | Low | 🔲 Observed, Run 13 is the fix |
| M-sgd | Training | Scheduler experiments (optional) — `CosineAnnealingWarmRestarts` to help escape local minima in sparse technique head. **(a) With curriculum**: set `T_0 ≥ curriculum_warmup + curriculum_ramp` (≥40 epochs). **(b) Without curriculum** (joint training): set `T_0` to ~30–40% of total epochs; all heads benefit together from early restarts. Add `--scheduler cosine_restarts --restart-t0 INT --restart-tmult FLOAT` flags. Only needed if Run 12/13 still show poor technique on some classes. | Low | ⬜ Not started |
| M-gteval | Evaluation | GTSinger technique test evaluation — filename mismatch: `update_results.py` looks for `technique_test.npz` but GTSinger extraction saves `technique_gtsinger_test.npz`. Low priority as no GTSinger test split exists yet (deferred to D3-fix in Phase 2). | Low | 🔲 Filename mismatch, low priority |

### Evaluation Targets (Phase 1)

| Metric | Task | Benchmark | Goal |
| --- |--- | --- |--- |
| RPA, RCA, Gross Error | Pitch | NanoPitch test set (6 SNR conditions) | Within 5% of NanoPitch 96.1% rtRPA |
| VAD Accuracy, VDR | VAD | NanoPitch test set | VDR >50% on clean condition |
| **Per-class F1** | Technique | VocalSet held-out (m2, m4, f4, f8) | Vibrato ≥0.4 (TechSinger SOTA=0.374); breathy/falsetto >0.8 |
| **Clip-level accuracy** | Technique | VocalSet held-out | SOTA: MuQ 81.5%, AST 82.0% |
| Training time / epoch | Efficiency | RTX 4080 Super | TCN vs Conformer comparison |

---

## Phase 2 — Full Multi-Task Coaching Model

> Uses winning architecture from Phase 1. Focus: richer post-processing + DTW reference + coaching UX scaffolding.

| ID | Category | Task | Effort | Status |
| --- | --- | --- | --- | --- |
| D3-fix | Data | GTSinger label quality fix — **(1) Coverage filter**: use `ph_start`/`ph_end` timestamps to compute fraction of clip frames where technique is active; skip clips below 50% coverage. **(2) Test split**: re-run `extractGTSingerTechnique.py` with held-out singer split to produce `technique_test.npz`. Note: increasing `--seq-len` reduces noise variance but does not remove label bias — fix must be in extraction. | Medium | ⬜ Not started |
| D5 | Data | PopBuTFy paired data — `scripts/extractFeatures.py` with `--dataset-dir` for amateur and professional folders. | Medium | ⬜ Not started |
| D6 | Data | SingMOS pretrained predictor — run at inference for clip-level MOS sanity check. No custom head trained. Repository: https://github.com/South-Twilight/SingMOS | Low | ⬜ Not started |
| D8 | Data | M4Singer pseudo-labels — check STARS/GTSinger GitHub for released labels; free 10× technique training data expansion. | Low | ⬜ Not started |
| F-ph | Features | `phrase_aggregate()` in `features.py` — segment phrases via VAD (≥150ms gaps), compute per-phrase: mean technique prob, vibrato rate/depth via autocorrelation, F0 std dev, RMS energy arc. **UX critical.** | Low | 🔲 Not implemented |
| F1 | Features | Vibrato rate, depth, regularity (Features 5–7) — short-time autocorrelation of F0 track. | Low | 🔲 Not implemented |
| F2 | Features | Note segmentation + per-note accuracy (Features 8–9) — blocked on D7 (Annotated-VocalSet integration). | Medium | 🔲 Blocked on D7 |
| F5 | Features | DTW reference pitch comparison — primary coaching metric. Per-frame deviation in cents, per-phrase aggregate, alignment offset. | Medium | ⬜ Not started |
| F10 | Features | Phrase pitch contour shape (Feature 12) — polynomial fit over phrase F0. | Low | 🔲 Not implemented |
| F11 | Features | Vibrato onset latency (Feature 27) — 300ms autocorrelation window post note-onset; classical target <200ms. | Low | 🔲 Not implemented |
| F12 | Features | Register / passaggio detection (Feature 28) — H1-H2 tilt + pitch track discontinuity; AVRA achieves 94% with SVM. | Medium | 🔲 Not implemented |
| F13 | Features | Singer's formant energy (Feature 29) — spectral energy ratio [2–4 kHz] vs [0.5–2 kHz]; LPC order 12. | Low | 🔲 Not implemented |
| F14 | Features | Per-axis output scoring (Feature 30) — 3-5 scalars per phrase (pitch / vibrato / dynamics / breath / overall). | Low | 🔲 Not implemented |
| F15 | Features | Audio quality pre-flight (Feature 31) — RMS / SNR / reverb estimate on input. | Very Low | 🔲 Not implemented |
| F16 | Features | Output confidence (Feature 32) — posteriorgram entropy + technique probability margins. | Low | 🔲 Not implemented |
| F17 | Features | Range tracking (Feature 33) — min/max pitch per session in lightweight JSON. | Low | 🔲 Not implemented |
| M-cal | Evaluation | Voicing threshold calibration — `--voicing-threshold` flag added to `evaluate.py` and `update_results.py`. Grid search confirmed: multi-task models need 0.15–0.20 (pitch-only default 0.30 is too high after backbone adapts to technique features; VAD sigmoid still separates voiced/silence correctly at 0.80 vs 0.29, but Viterbi uses pitch posterior peaks which shift down). Re-evaluate Run 11 with `--voicing-threshold 0.15` to get true VDR ~72%. | Low | ✅ Flag implemented — re-eval pending |
| M-hier | Model | Hierarchical technique head (note-level aggregation) — STARS-validated; requires D7 (Annotated-VocalSet) for note boundaries. | Medium | ⬜ Not started |
| U1 | UX | Post-session coaching report (structured JSON) feeding Phase 3 LLM critique. | Low | 🔲 Not implemented |
| U2 | UX | Color-coded pitch timeline. | Very Low | 🔲 Not implemented |

---

## Phase 3 — Coaching Report & UX

| ID | Category | Task | Effort | Status |
| --- | --- | --- | --- | --- |
| Whisper | Pipeline | Lyric ASR: Hybrid Demucs (mdx_extra) → RMS-VAD → Whisper large-v2 → DTW-align word timestamps to F0 track. Native Whisper on singing = 23% WER (hallucinates over music). Add confidence threshold: only show word-level feedback when Whisper word confidence >0.8. | Medium | ⬜ Not started |
| LLM | UX | Two-tier LLM critique — structured JSON → Claude API. Expert mode: numerical metrics + technical language. Beginner mode: encouragement + plain language. User toggles in UI. | Low | ⬜ Not started |
| U3 | UX | Reference comparison UI — upload reference audio → DTW-align F0 → deviation overlay. | Medium | ⬜ Not started |
| U4 | UX | Beginner/Expert mode toggle (LLM prompt branch). | Low | ⬜ Not started |
| U5 | UX | Audio quality pre-flight UI — warn if mic SNR too low, reverb too high, or signal clipped (uses F15). | Low | ⬜ Not started |
| U6 | UX | Confidence display in UI — grey out technique badges with margin <0.2 (uses F16). | Low | ⬜ Not started |
| U7 | UX | Range tracker chart — min/max comfortable note over session history (uses F17). | Low | ⬜ Not started |
| E2 | Explainability | Integrated Gradients on mel input (Captum) — which frequency bands drive technique prediction? | Low | ⬜ Not started |
| D4 | Data | SingPAD learner evaluation — test model on beginner recordings (request-gated via professor). | Medium | ⬜ Not started |

---

## Phase 4 — Deployment & MIR Enrichment (Stretch)

```
Browser
├── Live mode:    VocalCoachTCN (causal=True, ONNX) → live pitch + technique  [primary target]
│                 NanoPitch GRU (deployment/) → live pitch only               [fallback]
└── Offline mode: VocalCoach (TCN or Conformer winner, causal=False, ONNX) → full report
```

| ID | Category | Task | Effort | Status |
| --- | --- | --- | --- | --- |
| Live | Deployment | NanoPitch GRU live pitch — already deployed under `deployment/`. No new work. | None | ✅ Done |
| FastAPI | Deployment | FastAPI demo (recommended path) — VocalCoach runs server-side on 4080 Super; browser renders UI. Sufficient for academic demo. | Low | ⬜ Not started |
| ONNX | Deployment | ONNX Runtime Web browser deployment — `torch.onnx.export()` → `onnxruntime-web`. Only if fully-offline browser demo required. | Medium | ⬜ Not started |
| BeatNet | MIR | Beat/tempo for rhythm coaching (when backing track uploaded). | Low | ⬜ Optional |
| KeyDet | MIR | Key detection for pitch-in-key accuracy (Essentia). | Low | ⬜ Optional |
| E3 | Explainability | Temporal self-attention weights visualisation (Conformer). | Medium | ⬜ Not started |

---

## Benchmarks

### Primary (required in final report)

| Benchmark | Task | Metric | Compare against |
| --- |--- | --- |--- |
| **VocalSet test split** | Technique classification (held-out singers m2, m4, f4, f8) | Per-class F1, macro F1, clip acc | GTSinger paper baselines, MuQ 81.5%, AST 82.0% |
| **NanoPitch test set** | Pitch + VAD at 6 SNR levels | VAD Acc, offRPA, offVDR, Med¢ | NanoPitch GRU (96.1% rtRPA), RMVPE, CREPE |
| **SingMOS-Pro** | Quality MOS (pretrained predictor sanity check) | Pearson LCC, Spearman SRCC | PS-SQA (VoiceMOS 2024 winner) |

### Secondary

| Benchmark | Task | Metric | Notes |
| --- |--- | --- |--- |
| **GTSinger technique split** | Technique (vibrato, falsetto, breathy) | F1 | Multilingual, professional singers; test split pending D3-fix |
| **iKala** | Pitch in polyphonic context | RPA, RCA | Tests pitch robustness with accompaniment |

---

## Academic Contribution

> "We design VocalCoach — a multi-task singing analysis model trained and
> compared across three architectures (non-causal TCN, Conformer, MuQ-pretrained),
> with a **pitch-conditioned technique head** that fuses detached pitch posterior
> features into the technique classifier (validated by TechSinger AAAI 2025).
> We evaluate pitch accuracy on the NanoPitch test set, technique classification
> F1 (per-class) on VocalSet and GTSinger held-out splits, and perceptual quality
> via the pretrained SingMOS predictor as a sanity check.
> We show that [winning architecture] provides competitive pitch accuracy while
> additionally enabling multi-label technique classification, **DTW reference
> comparison** as the primary coaching metric, and natural language coaching
> feedback via a Demucs → Whisper → Claude pipeline with two-tier (expert /
> beginner) critique generation — demonstrating that a unified offline coaching
> model trained on publicly available singing datasets is practical without
> requiring separate per-task models. The system is paired with the existing
> NanoPitch GRU for live in-browser pitch tracking, making the deployment
> two-tier (live pitch + offline multi-task analysis) without requiring a
> causal retrain of the multi-task backbone."

---

## Out of Scope

Listed so the rationale is preserved and future-you doesn't reconsider without remembering why they were cut.

### Model / Training

| Item | Why dropped |
| --- | --- |
| Custom MOS quality head | Pretrained SingMOS predictor exists. Multi-task interference risk + ~3-4 weeks avoided. Clip-level MOS has low coaching utility vs per-axis scores (Feature 30). |
| 9-class STARS technique expansion | BUB/GLI/PHA/MIX classes don't improve coaching feedback. 5 classes sufficient. STARS taxonomy available via GTSinger if scope expands in Phase 2. |
| Mistake detection head (M3 dataset) | M3 is Indian Art Music — genre transfer risk. Cite methodology only (TCN F1=87.14% on frequency mistakes validates VocalCoach design). |
| Causal Conformer retrain | Live mode handled by NanoPitch GRU. `causal=True` flag in `build_model()` remains functional but not on roadmap. |
| Belt vs Straight as separate classes | Near-synonyms in informal usage; collapsed to "strong". |
| Sing-MD few-shot critiques | Claude API generates coaching critiques without few-shot examples. |
| Style / genre classifier | Genre-agnostic scope. |
| Hierarchical multi-level heads for Phase 1 | Deferred to Phase 2 — validate frame-level F1 first. |
| TCN + Conformer hybrid backbone | Both architectures cover local+global tradeoff; hybrid adds complexity without clear gain. |
| Uncertainty-based loss weighting (Kendall 2017) | Documented instability; manual weights work at current scale. |
| Option 4: Freeze pitch/VAD heads after convergence | Fragile — freezing backbone prevents encoder adapting to technique features; freezing only heads requires surgical gradient detachment. Curriculum (`--curriculum`) achieves same ordering without freezing. Revisit only if curriculum + pos_weights both fail. |

### Datasets Not Pursued

| Item | Why dropped |
| --- | --- |
| SINGSTYLE111 amateur/pro pairs | PopBuTFy provides the same signal; genre-agnostic scope removes need for style-pair data. |
| NUS-48E (phonetic annotation) | Whisper handles lyric transcription in Phase 3. |
| CSD (Children's Song Dataset) | SingPAD covers beginner domain; CSD adds no unique signal. |
| NHSS (speech+singing parallel) | Pathology detection out of scope. |
| DALI (5,358 songs with lyrics) | Real-world accompaniment robustness deferred to Phase 4 stretch. |
| Sundberg phonation modes / CVT | Don't align with VocalSet/GTSinger labels; genre-specific. |

### UX / Product

| Item | Why dropped |
| --- | --- |
| User accounts / authentication | Academic demo; session-level state only. |
| Social sharing / community features | Out of scope for academic project. |
| Exercise prescription engine | Product feature, not a research contribution. |
| Long-term progress analytics dashboard | Range tracker (F17) is the minimum viable progress signal. |
| Style-aware feedback | Requires style classifier (dropped). |
| Posture / breath visualization | Audio-only input by design. |

---

## Key File Locations

| File | Purpose |
| --- |--- |
| `vocalcoach/model.py` | TCN + Conformer architecture definitions |
| `vocalcoach/train.py` | Training loop — multi-task loss, curriculum, pos_weights |
| `vocalcoach/evaluate.py` | Evaluation: pitch table + technique F1 + clip accuracy |
| `vocalcoach/features.py` | Signal processing feature extraction (inference only) |
| `vocalcoach/update_results.py` | Append runs to VOCALCOACH_RESULTS.md |
| `scripts/extractVocalSet.py` | VocalSet → technique_train/test.npz |
| `scripts/extractGTSingerTechnique.py` | GTSinger → technique_gtsinger_train/test.npz |
| `scripts/extractAnnotatedVocalSet.py` | Annotated-VocalSet → per-note MIDI + timestamps |
| `training/model.py` | NanoPitch GRU baseline (reference only) |
| `VOCALCOACH_RESULTS.md` | Experiment tracker — all run metrics |
| `RESULTS.MD` | NanoPitch experiment tracker (baseline comparison numbers) |
