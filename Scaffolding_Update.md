# VocalCoach Update

## 1. VocalCoach Inference Stack

The offline inference pipeline is a FastAPI server plus demo notebook. Key capabilities:

- `/analyse` endpoint accepts an audio upload and optional reference file, returns a
  full coaching report: pitch timeline, phrase segmentation, voice quality metrics
  (HNR, jitter, shimmer, RMS arc), technique classifier probabilities, DTW reference
  comparison, and population context bands
- Perceptual quality scoring uses **AudioScore** (MuQ-large-msd-iter + LoRA scoring head)
  rather than SingMOS-Pro — AudioScore validated at Pearson r=+0.66 vs expert pitch scores
  and correctly separates professional from amateur singing (+0.36 delta on PopBuTFy
  controlled pairs). SingMOS-Pro was rejected as it does,
  not reward singing techniques (inverted direction vs all expert dimensions).
- Population context shows where the user lands relative to PopBuTFy amateur/professional
  medians across 10 metrics
- `scripts/buildPopBuTFyBaselines.py` pre-computes those population statistics across
  all ~28K PopBuTFy clips

---

## 2. MuQ / AudioScore as MOS Scoring Signal

**What was decided:** SingMOS-Pro rates all technique clips lower than their control variants and shows no meaningful separation between PopBuTFy amateur and professional clips. AudioScore (MuQ backbone) scores professional clips higher and rewards expressive techniques — the correct signal for singing quality assessment.

**Correlation results vs ccmusic 9-dim expert scores (132 clips):**

| Expert dimension | AudioScore Pearson r | Direction |
|---|---|---|
| pitch | +0.659* | correct |
| rhythm | +0.627* | correct |
| vocal_range | +0.623* | correct |
| timbre | +0.599* | correct |
| overall_performance | +0.592* | correct |
| breath_control | +0.492* | correct |

Mean `|r|` = 0.578 across all 9 dimensions. SingMOS mean `|r|` = 0.271, inverted direction.

**SingMOS-Pro dataset validation (7,981 clips, TangRain/SingMOS-Pro):**

AudioScore scored all clips.

| Type | n | Pearson vs MOS | AudioScore mean |
|---|---|---|---|
| gt (real singers) | 578 | +0.301* | 2.072 |
| svc | 1307 | +0.205* | 1.923 |
| svr | 2671 | +0.275* | 1.652 |
| svs | 3425 | +0.159* | 1.622 |

GT mean (2.072) is +0.45 above SVS mean (1.622) — model correctly scores real human
recordings higher than synthesised output without being told the type.

**PopBuTFy controlled separation (same singer, eliminates recording quality confound):**

| Model | Pro − Amateur delta | Verdict |
|---|---|---|
| SingMOS-Pro | −0.05 | inverted |
| AudioScore | +0.36 | correct |

---

## 3. GTSinger Technique

AudioScore run on GTSinger Control\_Group vs Technique\_Group pairs (English,
5 techniques).

| Technique | AudioScore delta (tech − control) | Interpretation |
|---|---|---|
| Vibrato | +0.809 | strongly rewarded |
| Glissando | +0.639 | strongly rewarded |
| Breathy | +0.098 | neutral |
| Pharyngeal | −0.256 | penalised |
| Mixed Voice / Falsetto | n/a | folder name fix needed |

Vibrato and Glissando responses are musically correct — expressive techniques score
higher. Pharyngeal being penalised.

---

## 4. YouTube Cover Dataset Pipeline

**Motivation:** PopBuTFy is same-singer controlled. YouTube covers extend to
different-singer pro-vs-amateur comparison across 10 canonical songs.

- `yt-dlp` downloads best audio → 16kHz mono WAV
- `demucs htdemucs` two-stem vocal separation per clip
- Manifest JSON tracks download/demucs status per clip for resumable runs

Songs configured: rolling\_in\_the\_deep, hallelujah, shallow, someone\_like\_you,
all\_of\_me, let\_it\_go, the\_scientist, my\_heart\_will\_go\_on, fix\_you, stay\_with\_me

---

## 6. Scoring Head on VocalCoach Model (Planned)

The proposal documents a Phase 2 path: attach a lightweight MOS regression head
on top of the frozen VocalCoach backbone (pitch + VAD + technique embeddings) and
supervise it using AudioScore pseudo-labels from PopBuTFy.

Rationale: AudioScore validates at +0.66 Pearson vs expert pitch scores and +0.36
pro − amateur separation on controlled pairs. Using it as soft supervision means the
VocalCoach model gains a holistic quality signal without requiring new human
annotations, while the pitch/technique heads remain interpretable.

**Training approach:** Rather than standard regression against AudioScore scalar values,
the scoring head will be trained with a **contrastive/pairwise supervised objective** —
using PopBuTFy's same-singer amateur/professional pairs directly. Each pair provides a
clean supervision signal: the model must learn to rank the professional recording above
the amateur recording by the same singer on the same song. This eliminates recording
quality and singer identity as confounds, forcing the head to learn what "better singing"
actually looks like independent of production quality or vocal timbre.

---

### Pearson r correlation

Pearson r measures the linear correlation between two continuous variables. Here it is
computed between the **inference model's output scores** (AudioScore or SingMOS-Pro scalar
values, one per clip) and the **ground-truth labels from the dataset** (e.g., the 9
expert-rated dimension scores from ccmusic, or human MOS ratings from SingMOS-Pro).

A high positive r (e.g., +0.66) means the model's scores move in the same direction as
the human expert labels — clips that experts rated highly also get high AudioScore values.
A negative r means the opposite (the model is systematically wrong in direction, as
SingMOS-Pro was).
