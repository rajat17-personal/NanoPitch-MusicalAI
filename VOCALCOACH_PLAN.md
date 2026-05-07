# VocalCoach — Project Plan

**Course:** Musical AI (Final Project)
**Author:** Rajat Sharma
**Last updated:** 2026-05-06

---

## Project Overview

VocalCoach is a multi-task singing analysis model that goes beyond pitch
tracking to provide actionable coaching feedback. It is a **new project**, not
an extension of NanoPitch. NanoPitch (GRU baseline, ~333K params, 96.1% rtRPA)
serves only as a comparison baseline for pitch accuracy experiments.

The system takes a singing audio clip and produces:

| Output | Type | Where used |
| --- |--- | --- |
| VAD (voiced/unvoiced) | Per-frame probability | Breath detection, note segmentation |
| F0 pitch track | Per-frame 360-bin posteriorgram | All pitch-related coaching metrics |
| Technique label | Multi-label per frame | Vibrato / breathy / falsetto / belt / straight |
| Quality score (MOS) | Per-session scalar | Overall coaching grade |
| Acoustic features | Post-processing | HNR, spectral tilt, dynamics, jitter, shimmer |
| Natural language critique | LLM API call | Human-readable coaching report |

---

## Architecture Decision: TCN vs Conformer

**Phase 1 trains both architectures with `causal=False` (non-causal, offline).**
This gives maximum accuracy for the coaching app with no streaming constraints.
The causal variants are not needed for Phase 1 — the `causal` flag is a
one-parameter change in `build_model()` and can be revisited in Phase 4 if
live tracking becomes a priority.

| Property | VocalCoachTCN | VocalCoachConformer |
| --- |--- | --- |
| Backbone | Dilated Conv1d stack | Conv + Multi-head Self-Attention (Macaron) |
| Phase 1 mode | `causal=False` — symmetric padding | `causal=False` — full clip attention |
| Context | ~5 s geometric (dilation doubles per block) | Full clip (global attention) |
| Training speed | Fully parallel | Parallel |
| Multi-task fit | Good | Best |
| Default params | ~449K (hidden=128, 8 blocks) | ~419K (hidden=64, 4 layers) |
| Live tracking later | `causal=True` + longer `seq_len` — one parameter change | `causal=True` — one parameter change |
| Browser deployment | ONNX Runtime Web (medium effort) | ONNX Runtime Web (medium effort) |

**Experiment C** (Phase 1) also tests a pretrained **MERT-v1-95M** backbone
with lightweight classification heads — establishing whether pretraining on
160k hours of music significantly reduces the labelled singing data needed.

Architecture selection criterion: technique classification F1 (primary) +
pitch accuracy (must remain competitive with NanoPitch baseline).

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

If live tracking is later required: retrain with `causal=True` and longer
`seq_len` (same training script, one argument change), then re-export to ONNX.

---

## Pretrained Models Available

| Model | Size | Features | License | Use in Project |
| --- |--- | --- |--- | --- |
| RMVPE | ~M | Frame-level F0 (360-bin) | Public | Pitch pseudo-labels (already used) |
| CREPE (tiny–full) | Configurable | Frame-level F0 | MIT | Alternative/validation pitch labels |
| MERT-v1-95M | 95M | Frame-level music features | Academic | Exp C backbone — fine-tune heads only |
| Music2Vec-v1 | 95M | Frame + clip-level | CC-BY-NC | Lightweight backbone alternative |
| HuBERT (base) | 94M | Frame-level, 50 Hz | MIT | Speech transfer learning baseline |
| CLAP | Medium | Clip-level audio-text | Open | LLM text feedback alignment (Phase 3) |

No pretrained singing technique classifier or quality predictor exists publicly —
all technique/quality models are trained from scratch.

---

## Full Feature Taxonomy

### From Model Heads (Learned)

| # | Feature | Head type | Notes |
| --- |--- | --- |--- |
| 1 | F0 pitch track | 360-bin sigmoid | 20-cent resolution, B0–B6 |
| 2 | VAD | Binary sigmoid | Voiced/unvoiced per frame |
| 3 | Technique class | Multi-label sigmoid | Vibrato / breathy / falsetto / belt / straight |
| 4 | Quality score (MOS) | Regression | Requires SingMOS-Pro training data (Phase 2) |

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

### From Reference / Paired Data (Optional)

| # | Feature | Method | Coaching value |
| --- |--- | --- |--- |
| 23 | DTW pitch distance | DTW alignment vs reference F0 | "How far are you from the target?" |
| 24 | Relative quality rank | Amateur vs professional distance (PopBuTFy) | Quantified gap to professional level |

### Optional (with Lyrics / Score Input)

| # | Feature | Method | Coaching value |
| --- |--- | --- |--- |
| 25 | Phoneme-level accuracy | Lyric alignment (Annotated-VocalSet / DALI) | Pronunciation, vowel quality |
| 26 | Rhythm deviation | DTW to MIDI score | Timing precision |

---

## Dataset Plan

| Dataset | Phase | What it provides | Access |
| --- |--- | --- |--- |
| GTSinger | 0 | Mel + RMVPE pitch labels + **6 technique labels** (currently unused) | Open (HuggingFace) |
| VocalSet | 0 | 10.1 hrs, 20 singers, 17 vocal technique labels per clip | Open (Zenodo) |
| Annotated-VocalSet | 0 | Per-note technique labels + onset/offset timestamps | Open (Zenodo) |
| PopBuTFy | 0 | Paired amateur/professional recordings of same songs | Open (NeuralSVB GitHub) |
| SingMOS-Pro | 2 | Human MOS ratings (melody + lyric + overall) — 7,981 clips | Open (HuggingFace) |
| Sing-MD (VocalVerse) | 3 | Expert 4-dim scores + **text critiques** — 1,000 clips | Open (HuggingFace) |
| SingPAD | 3 | Learner sight-singing dataset (beginner domain) | Request (via professor) |

---

## Phase 0 — Data Pipeline & Feature Extraction (Parallel, User)

> User is building this while Phase 1 model experiments run.

- [ ] Extend GTSinger loader to emit technique labels (currently only pitch used)
- [ ] Integrate VocalSet: mel extraction + RMVPE pitch labels + technique labels → same `.npz` format as NanoPitch
- [ ] Integrate Annotated-VocalSet: note onset/offset timestamps
- [ ] Integrate PopBuTFy: paired amateur/professional audio → mel + RMVPE labels
- [ ] Implement signal processing feature module (features 14–22): HNR, spectral centroid, tilt, jitter, shimmer, RMS, breath detection
- [ ] Build evaluation harness: per-task metrics (RPA/RCA/gross error, VAD acc, per-class F1)

**Preprocessing reference:** GTSinger pipeline documented in NanoPitch README at
`https://github.com/smulelabs/NanoPitch`. Same mel + RMVPE extraction applies
to VocalSet (confirmed by course staff).

---

## Phase 1 — Architecture Experiments ✅ (In Progress)

> `vocalcoach/model.py` — both architectures implemented. Training scripts TBD.
> All Phase 1 experiments use `causal=False` (non-causal, offline, best accuracy).

### Experiment A: VocalCoachTCN (non-causal)

```
mel (40) → Conv1d(40→128) → TCNBlock×8 (dilations 1,2,4,…,128, symmetric pad) → LayerNorm
         → VAD head (128→1) + Pitch head (128→360) + Technique head (128→5)
```

- `causal=False` — symmetric dilation, sees past + future frames
- Receptive field: ~5.1 s (8 blocks, doubling dilation, symmetric)
- ~449K parameters (hidden=128)
- Train pitch-only first, validate vs NanoPitch RPA, then add technique head

### Experiment B: VocalCoachConformer (non-causal)

```
mel (40) → Linear(40→64) → ConformerBlock×4 → LayerNorm
         → VAD head (64→1) + Pitch head (64→360) + Technique head (64→5)
```

Each ConformerBlock: FF(0.5) → Self-Attention (full clip) → Conv(kernel=31, symmetric) → FF(0.5) → LN

- `causal=False` — full bidirectional attention over entire clip
- ~419K parameters (hidden=64)
- Expected: better technique F1 than TCN due to global phrase context

### Experiment C: MERT-95M Backbone (pretrained, optional)

```
raw audio → MERT-v1-95M (frozen) → frame features (768-dim)
          → Linear(768→128) → VAD head + Pitch head + Technique head
```

- Frozen backbone: only train lightweight heads (~50K new params)
- Tests whether 160k hours of music pretraining closes the technique
  classification gap vs training from scratch on VocalSet alone
- Requires: `pip install transformers`

### If Live Tracking Needed Later (Phase 4)

No code changes required. Retrain the winning architecture with:

```bash
# causal=True retrain — one argument change
build_model('tcn',       causal=True, hidden=128, n_blocks=8, seq_len=600)
build_model('conformer', causal=True, hidden=64,  n_layers=4, seq_len=600)
```

Longer `seq_len` compensates for the loss of future context. Export to ONNX
for browser deployment via `onnxruntime-web`.

### Evaluation (all experiments)

| Metric | Task | Benchmark | Notes |
| --- |--- | --- |--- |
| RPA, RCA, Gross Error | Pitch | MIR-1K + NanoPitch test set | Compare vs NanoPitch (96.1% rtRPA), CREPE, RMVPE |
| VAD Accuracy | VAD | NanoPitch test set | |
| Per-class F1 + macro F1 | Technique | VocalSet held-out test split | Primary evaluation criterion |
| Training time / epoch | Efficiency | — | On RTX 4080 Super |
| Params | Model size | — | |

**Selection criterion:** technique F1 is primary. Pitch accuracy must remain
reasonable — degradation vs NanoPitch is acceptable since this is a coaching
model, not a dedicated pitch tracker.

---

## Phase 2 — Full Multi-Task Coaching Model

> Uses winning architecture from Phase 1. Adds quality score and PopBuTFy data.

- [ ] Add SingMOS-Pro quality head (regression, trained on MOS labels)
      — either as 4th head on winning model or small separate network fed by
        its embeddings (lower interference risk)
- [ ] Integrate PopBuTFy paired data: amateur audio exposed to model during
      training so it generalises to beginner-domain singing
- [ ] Run ablation: joint training (all heads) vs sequential (pitch→technique→quality)
- [ ] Full feature extraction pipeline: post-processing features 5–22 implemented
      as `features.py` module consuming model F0 + VAD outputs
- [ ] Basic coaching report: per-note accuracy, technique label, vibrato metrics,
      HNR, dynamics, breath pattern — structured JSON + simple web display

---

## Phase 3 — Coaching Report & UX (Final Demo)

- [ ] **Lyric ASR (Whisper):** Run `openai/whisper` locally on the recorded
      audio to extract timestamped words. DTW-align word timestamps to the
      model's F0 track so pitch errors are reported as "your E4 on *love*
      was 25 cents flat" rather than "frame 340 was flat". Whisper runs
      locally — no API dependency. (MOSS lyric ASR workflow uses Qwen3-Omni
      via API; Whisper is the local equivalent.)
- [ ] LLM text feedback: structured coaching report → Claude API → natural
      language critique. Use Sing-MD text critiques as few-shot examples.
- [ ] Reference comparison: upload reference audio, DTW-align F0 tracks,
      show deviation overlay
- [ ] Explainability: Integrated Gradients on mel input (Captum) — which
      frequency bands drive technique prediction?
- [ ] SingPAD evaluation (if obtained): test model on beginner recordings
- [ ] Full evaluation against human labels: MOS correlation on SingMOS-Pro,
      technique F1 on VocalSet held-out set

---

## Phase 4 — Deployment & MIR Enrichment (Stretch Goal)

### Deployment Options

| Option | Effort | Notes |
| --- |--- | --- |
| **Python API (FastAPI) + web frontend** | Low | Recommended first step — works for any architecture, GPU-accelerated on 4080 Super, ~2-day implementation |
| **ONNX Runtime Web** | Medium | Export winning model with `torch.onnx.export()`, run in browser via `onnxruntime-web` JS library — no C code, no Emscripten, works for both TCN and Conformer |
| Custom WASM (NanoPitch style) | Very High | Hand-written C inference — not recommended; ONNX Runtime Web supersedes this |

**Recommended path:** Python API first (always achievable), then ONNX Runtime
Web if browser-native deployment is required.

### Causal Retrain for Live Tracking (if needed)

If live pitch tracking during singing is added (via the winning VocalCoach
architecture rather than NanoPitch), retrain with:

```python
model = build_model('tcn',       causal=True, hidden=128, n_blocks=8)
# or
model = build_model('conformer', causal=True, hidden=64,  n_layers=4)
```

Also increase `seq_len` during training (e.g. 600 frames) to give the causal
model more past context since it can no longer look forward. No other code
changes required — the padding logic and attention masking are already
implemented in `vocalcoach/model.py`.

### MIR Feature Enrichment (from MOSS pipeline tools)

Selectively use MIR extraction to enrich the coaching report — not for core
model training, but for additional post-processing features:

| Tool | Feature | Use in VocalCoach | Effort |
| --- |--- | --- |--- |
| **BeatNet** (standalone) | Beat positions, tempo (BPM) | Rhythm coaching — are notes on the beat? Maps to feature F26 (rhythm deviation) | Low |
| **Essentia** key detection | Musical key + tuning frequency | Pitch accuracy relative to key, not just nearest semitone; "consistently sharp in D major" | Low |
| Chordino | Chord sequences | Expected pitch range per phrase; weakly useful | Medium |
| MOSS full pipeline | All of the above + captions | Data preprocessing for large non-singing corpora; not needed for VocalSet/GTSinger workflow | High |

Note: these tools extract features from the full audio mix, not the isolated
voice. Most useful when the user uploads a backing track alongside their
recording. For a cappella recordings they still provide key and tempo context.

---

## Complete Improvements Table

| ID | Feature | Category | Phase | Effort | New Model? |
| --- |--- | --- |--- | --- |--- |
| D3 | GTSinger technique labels | Data | 0 | Low | No |
| D2 | VocalSet + Annotated-VocalSet | Data | 0 | Medium | No |
| D5 | PopBuTFy (amateur/professional pairs) | Data | 0–2 | Medium | No |
| D1 | SingMOS-Pro evaluation benchmark | Data | 2 | Low | No |
| D4 | SingPAD (learner data) | Data | 3 | Medium | No |
| Sing-MD | Expert scores + text critiques | Data | 3 | Medium | No |
| F1 | Vibrato rate, depth, regularity | Feature | 2 | Low | No |
| F2 | Note segmentation + per-note accuracy | Feature | 2 | Medium | No |
| F3 | Dynamics / RMS tracking | Feature | 2 | Very Low | No |
| F4 | Breath / pause detection | Feature | 2 | Low | No |
| F5 | Reference pitch comparison (DTW) | Feature | 3 | Medium | No |
| F6 | HNR, spectral centroid, tilt | Feature | 2 | Low | No |
| F7 | Jitter + shimmer | Feature | 2 | Medium | No |
| F8 | MFCCs (timbre) | Feature | 2 | Very Low | No |
| F9 | Vocal onset steepness | Feature | 2 | Low | No |
| F10 | Phrase pitch contour shape | Feature | 2 | Low | No |
| M-ExpA | VocalCoachTCN (non-causal) | Model | 1 | Medium | Yes — new model |
| M-ExpB | VocalCoachConformer (non-causal) | Model | 1 | Medium | Yes — new model |
| M-ExpC | MERT backbone + heads | Model | 1 | Medium | Yes (pretrained) |
| M2 | Quality score (MOS) head | Model | 2 | High | Extends winning model |
| M3 | Mistake detection & classification | Model | Stretch | High | Yes — new head/model |
| M4 | Score-informed evaluation (MIDI) | Model | Stretch | High | No (DTW) |
| U1 | Post-session coaching report | UX | 2–3 | Low | No |
| U2 | Color-coded pitch timeline | UX | 2 | Very Low | No |
| U3 | Reference track overlay | UX | 3 | Medium | No |
| LLM | Natural language critique (Claude API) | UX | 3 | Low | No (API) |
| E1 | Posteriorgram entropy (confidence) | Explainability | 2 | Very Low | No |
| E2 | Integrated Gradients (mel saliency) | Explainability | 3 | Low | No |
| E3 | Temporal self-attention weights | Explainability | 3 | Medium | Extends model |
| Genre | Style / genre conditioning | Stretch | Stretch | High | Extends model |
| Whisper | Lyric ASR → word-level pitch error labels | UX | 3 | Low | No (local model) |
| ONNX | ONNX Runtime Web browser deployment | Deployment | 4 | Medium | No |
| Causal | Causal retrain for live tracking | Deployment | 4 | Low | Re-uses model |
| BeatNet | Beat/tempo for rhythm coaching | Feature | 4 | Low | No |
| KeyDet | Key detection for pitch-in-key accuracy | Feature | 4 | Low | No |

---

## Benchmarks

### Primary (required in final report)

| Benchmark | Task evaluated | Metric | Prior work to compare |
| --- |--- | --- |--- |
| **VocalSet test split** | Technique classification (17 classes, held-out singers) | Per-class F1, macro F1 | GTSinger baseline classifiers, SVQTD classifiers |
| **SingMOS-Pro** | Quality score (MOS) correlation — 7,981 clips, 5 expert raters | Pearson LCC, Spearman SRCC, KTAU | PS-SQA (VoiceMOS 2024 winner), SingMOS-Pro baseline systems |
| **MIR-1K** | Pitch accuracy — 1,000 karaoke clips, annotated F0 | RPA, RCA, Voicing F1 | NanoPitch (96.1% RPA), RMVPE, CREPE, SwiftF0 |

### Secondary (strengthen evaluation section)

| Benchmark | Task | Metric | Notes |
| --- |--- | --- |--- |
| **GTSinger technique split** | Technique (6 classes: vibrato, falsetto, breathy, glissando, mixed, pharyngeal) | F1 | Complements VocalSet — multilingual, professional singers |
| **SingMOS** | Quality MOS prediction — 3,421 clips | LCC, SRCC | Older/smaller than SingMOS-Pro but well-established in prior work |
| **iKala** | Pitch in polyphonic context — 252 clips with accompaniment | RPA, RCA | Tests pitch robustness when backing track is present |
| **NanoPitch test set** | Pitch + VAD at 6 SNR levels | VAD Acc, rtRPA, rtVDR, Med¢ | Direct comparison with NanoPitch GRU baseline |

### MIREX Tracks (optional, for academic completeness)

| MIREX Track | Relevance | Metric |
| --- |--- | --- |
| Singing Voice Separation (iKala) | Pitch robustness with accompaniment | SDR, SIR, SAR |
| Query by Singing/Humming | Melody representation quality | Top-10 hit rate |
| Singing Transcription (RWC-MDB-P) | Pitch + note accuracy on polyphonic music | MIDI-based evaluation |

### Comparison Baselines Summary

| Your model head | Compare against |
| --- |--- |
| Pitch (F0 track) | NanoPitch GRU, RMVPE, CREPE (full), SwiftF0 |
| Technique classification | GTSinger paper baselines, SVQTD published classifiers |
| Quality score (MOS) | PS-SQA (VoiceMOS 2024 winner), SingMOS-Pro reported baselines |

---

## Evaluation Summary

| Phase | Metric | Benchmark | Goal |
| --- |--- | --- |--- |
| 1 | RPA, RCA, Gross Error | MIR-1K + NanoPitch test set | Competitive with NanoPitch 96.1% rtRPA |
| 1 | Technique F1 (per-class + macro) | VocalSet held-out test split | Primary criterion — demonstrate multi-task benefit |
| 1 | Training time / epoch | — | Quantify TCN vs Conformer efficiency on RTX 4080 Super |
| 2 | MOS Pearson LCC / Spearman SRCC | SingMOS-Pro | r > 0.7 target |
| 2 | Technique F1 (GTSinger classes) | GTSinger technique split | Generalisation across technique taxonomies |
| 3 | Lyric-aligned pitch error rate | Whisper + F0 alignment | Per-word coaching precision |
| 3 | Beginner generalisation | SingPAD (if available) | Technique F1 does not collapse on learner data |

---

## Academic Contribution

> "We design VocalCoach — a multi-task singing analysis model trained and
> compared across three architectures (non-causal TCN, Conformer, MERT-pretrained),
> evaluating pitch accuracy on MIR-1K, technique classification F1 on VocalSet,
> and perceptual quality correlation on SingMOS-Pro against human expert ratings.
> We show that [winning architecture] provides competitive pitch accuracy while
> additionally enabling technique classification and natural language coaching
> feedback via Whisper lyric alignment and LLM critique — demonstrating that a
> unified offline coaching model trained on publicly available singing datasets
> is practical without requiring separate per-task models. A causal variant
> supporting live browser deployment via ONNX Runtime Web is also provided."

---

## Key File Locations

| File | Purpose |
| --- |--- |
| `vocalcoach/model.py` | TCN + Conformer architecture definitions (Phase 1) |
| `vocalcoach/train.py` | Training loop — multi-task loss (Phase 1, TBD) |
| `vocalcoach/evaluate.py` | Evaluation metrics (Phase 1, TBD) |
| `vocalcoach/features.py` | Signal processing feature extraction (Phase 2, TBD) |
| `training/model.py` | NanoPitch GRU baseline (reference only) |
| `RESULTS.MD` | NanoPitch experiment tracker (baseline comparison numbers) |
| `datasets.csv` | All 32 dataset options with access and license info |
