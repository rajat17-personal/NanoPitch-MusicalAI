"""
VocalCoach FastAPI Demo Server — Phase 4
=========================================

Runs server-side on the 4080 Super; browser uploads audio and receives a full
coaching report JSON + optional LLM critique.

Endpoints
---------
  POST /analyse          — full offline analysis (feature extraction + model inference)
  POST /analyse/critique — same + LLM natural-language critique appended
  GET  /health           — liveness check

Usage
-----
    # Start server (from repo root):
    uvicorn vocalcoach.api:app --host 0.0.0.0 --port 8000 --reload

    # Client:
    curl -X POST http://localhost:8000/analyse \
         -F "audio=@recording.wav" \
         -F "reference=@reference.wav"   # optional

Dependencies
------------
    pip install fastapi uvicorn python-multipart soundfile

Model checkpoint is loaded once at startup from VOCALCOACH_CHECKPOINT env var
(or --checkpoint CLI arg when launching directly).
"""

import io
import os
import tempfile
import traceback
from typing import Optional

import numpy as np

try:
    import soundfile as sf
    _HAS_SF = True
except ImportError:
    _HAS_SF = False

try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False
    # Dummy so module can be imported for type-checking / testing without FastAPI
    class FastAPI:  # type: ignore
        def __init__(self, **kw): pass
        def post(self, *a, **kw): return lambda f: f
        def get(self, *a, **kw): return lambda f: f
        def add_middleware(self, *a, **kw): pass

import torch

from vocalcoach.model import bin_to_f0
from vocalcoach.features import (
    extract_all, summarise, phrase_aggregate, compute_dtw_distance,
    SR, HOP_LENGTH
)
from vocalcoach.coach import build_report, generate_critique, score_report, compare_to_baselines
from vocalcoach.singmos import score_mos, mos_grade

# ── App init ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="VocalCoach Demo API",
    description="Offline singing analysis: pitch, VAD, technique, vibrato, DTW",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model singleton ─────────────────────────────────────────────────────────

_model = None
_device = None
_args = None  # argparse namespace used during training (for arch/seq_len)

TECHNIQUE_NAMES = ["vibrato", "breathy", "falsetto", "belt"]


def _load_model():
    global _model, _device, _args

    if _model is not None:
        return  # already loaded

    checkpoint_path = os.environ.get("VOCALCOACH_CHECKPOINT", "")
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise RuntimeError(
            "Set VOCALCOACH_CHECKPOINT env var to the path of best_loss.pth"
        )

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from vocalcoach.model import VocalCoachTCN, VocalCoachConformer
    ckpt = torch.load(checkpoint_path, map_location=_device)
    saved_args = ckpt.get("args", {})

    import argparse
    _args = argparse.Namespace(**saved_args) if isinstance(saved_args, dict) else saved_args

    # hidden / n_layers may be None in probe-mode checkpoints (inherited from
    # a resumed run). Derive them directly from the state dict instead.
    sd = ckpt["state_dict"]
    hidden_from_sd = int(sd["input_proj.weight"].shape[0])
    n_layers_from_sd = max(
        int(k.split(".")[1]) for k in sd if k.startswith("blocks.")
    ) + 1

    arch = getattr(_args, "arch", "conformer")
    if arch == "tcn":
        model = VocalCoachTCN(
            hidden=hidden_from_sd,
            n_blocks=n_layers_from_sd,
            causal=getattr(_args, "causal", False),
            deep_technique_head=getattr(_args, "deep_technique_head", False),
        )
    else:
        model = VocalCoachConformer(
            hidden=hidden_from_sd,
            n_layers=n_layers_from_sd,
            causal=getattr(_args, "causal", False),
            deep_technique_head=getattr(_args, "deep_technique_head", False),
        )
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.to(_device).eval()
    _model = model


@app.on_event("startup")
async def startup():
    try:
        _load_model()
        print(f"[VocalCoach API] Model loaded on {_device}")
    except Exception as e:
        print(f"[VocalCoach API] WARNING: model not loaded at startup: {e}")


# ── Audio helpers ───────────────────────────────────────────────────────────

def _load_audio(upload: bytes, target_sr: int = SR) -> np.ndarray:
    """Load uploaded audio bytes, resample to target_sr, return mono float32."""
    if not _HAS_SF:
        raise RuntimeError("pip install soundfile to handle audio uploads")

    with io.BytesIO(upload) as buf:
        y, sr = sf.read(buf, dtype="float32", always_2d=False)

    if y.ndim > 1:
        y = y.mean(axis=1)

    if sr != target_sr:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

    return y.astype(np.float32)


def _run_model(y: np.ndarray) -> dict:
    """Run VocalCoach model on a waveform, return per-frame numpy outputs."""
    _load_model()

    import librosa
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=512, hop_length=HOP_LENGTH,
        n_mels=40, fmin=50, fmax=8000,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # Shape: (1, T, n_mels) — batch of 1
    mel_t = torch.tensor(mel_db.T[None], dtype=torch.float32).to(_device)

    with torch.no_grad():
        # Returns (vad, pitch, technique, quality) — all already sigmoid'd
        out_vad, out_pitch, out_technique, _ = _model(mel_t)

    # (B, T, 1) → (T,)
    vad = out_vad.squeeze(0).squeeze(-1).cpu().numpy()
    # (B, T, 360) → (T, 360)
    pitch_post = out_pitch.squeeze(0).cpu().numpy()

    # Voicing: use pitch-peak confidence as proxy (VAD sigmoid may be
    # near-zero in probe-mode checkpoints whose VAD head wasn't retrained).
    # A frame is voiced when its peak bin probability exceeds 30% of the
    # clip maximum — equivalent to the per-class threshold sweep logic.
    pitch_confidence = pitch_post.max(axis=-1)           # (T,)
    voicing_thresh = max(0.05, pitch_confidence.max() * 0.30)
    voiced_mask = pitch_confidence > voicing_thresh

    # F0 from posteriorgram argmax → Hz via model's own bin_to_f0()
    pitch_bin = np.argmax(pitch_post, axis=-1).astype(float)  # (T,)
    f0_hz = np.where(voiced_mask, bin_to_f0(pitch_bin), 0.0).astype(np.float32)

    # Also expose a calibrated VAD: prefer model VAD if it fires, else fall
    # back to pitch confidence voicing so downstream features work correctly.
    if vad.max() < 0.1:
        vad = voiced_mask.astype(np.float32)

    result = {"f0_hz": f0_hz, "vad": vad}

    if out_technique is not None:
        # (B, T, N) → (T, N) — already sigmoid'd
        result["technique_probs"] = out_technique.squeeze(0).cpu().numpy()

    return result


# ── Core analysis pipeline ───────────────────────────────────────────────────

def _analyse(y: np.ndarray, y_ref: Optional[np.ndarray] = None) -> dict:
    """Full pipeline: model → features → phrase aggregation → report."""
    model_out = _run_model(y)
    f0_hz = model_out["f0_hz"]
    vad   = model_out["vad"]
    tech_probs = model_out.get("technique_probs")

    feats = extract_all(y, sr=SR, f0_hz=f0_hz, vad=vad)
    summary = summarise(feats, f0_hz=f0_hz)

    rms_db = feats["rms_db"]
    phrases = phrase_aggregate(
        f0_hz, vad,
        technique_probs=tech_probs,
        rms_db=rms_db,
        sr=SR, hop_length=HOP_LENGTH,
    )

    # Clip-level technique means (voiced frames only)
    technique_clip = None
    if tech_probs is not None:
        voiced_mask = vad > 0.5
        if voiced_mask.any():
            voiced_tech = tech_probs[voiced_mask]
            technique_clip = {
                name: float(np.mean(voiced_tech[:, k]))
                for k, name in enumerate(TECHNIQUE_NAMES)
                if k < voiced_tech.shape[1]
            }

    # DTW vs reference
    dtw_result = None
    if y_ref is not None:
        ref_out = _run_model(y_ref)
        dtw_result = compute_dtw_distance(f0_hz, ref_out["f0_hz"])

    clip_dur = len(y) / SR
    report = build_report(phrases, summary, technique_clip, dtw_result, clip_dur)
    report["_feats_summary"] = summary

    # D6 — SingMOS perceptual quality (graceful: None if package not installed)
    mos = score_mos(y, sr=SR)
    report["mos"] = {
        "score": round(mos, 2) if mos is not None else None,
        "grade": mos_grade(mos),
        "scale": "1-5 (ITU-T P.800)",
    }

    # D5 — PopBuTFy population context (graceful: {} if baselines not yet built)
    baselines_path = os.path.join(os.path.dirname(__file__), "..", "data", "popbutfy_baselines.json")
    population_context = compare_to_baselines(report, baselines_path=baselines_path)
    if population_context:
        report["population_context"] = population_context

    return report


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    model_loaded = _model is not None
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "device": str(_device) if _device else None,
        "cuda_available": torch.cuda.is_available(),
    }


def _sanitize(obj):
    """Recursively replace nan/inf with None so JSONResponse doesn't choke."""
    if isinstance(obj, float):
        if obj != obj or obj == float("inf") or obj == float("-inf"):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


@app.post("/analyse")
async def analyse(
    audio: UploadFile = File(..., description="Singing audio file (WAV/MP3/FLAC)"),
    reference: Optional[UploadFile] = File(None, description="Reference audio for DTW"),
):
    """Analyse a singing clip and return structured coaching metrics."""
    try:
        y = _load_audio(await audio.read())
        y_ref = None
        if reference is not None:
            y_ref = _load_audio(await reference.read())

        report = _analyse(y, y_ref)
        report["coaching"] = score_report(report)
        return JSONResponse(content=_sanitize(report))

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


@app.post("/analyse/critique")
async def analyse_with_critique(
    audio: UploadFile = File(..., description="Singing audio file"),
    reference: Optional[UploadFile] = File(None, description="Reference audio for DTW"),
    mode: str = Form("expert", description="Feedback mode: 'expert' or 'beginner'"),
):
    """Analyse + generate LLM natural-language coaching critique."""
    if mode not in ("expert", "beginner"):
        raise HTTPException(status_code=422, detail="mode must be 'expert' or 'beginner'")

    try:
        y = _load_audio(await audio.read())
        y_ref = None
        if reference is not None:
            y_ref = _load_audio(await reference.read())

        report = _analyse(y, y_ref)
        report["coaching"] = score_report(report)
        critique = generate_critique(report, mode=mode)
        report["critique"] = {"mode": mode, "text": critique}
        return JSONResponse(content=_sanitize(report))

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


# ── Direct launch ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="", help="Path to best_loss.pth")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    cli = parser.parse_args()

    if cli.checkpoint:
        os.environ["VOCALCOACH_CHECKPOINT"] = cli.checkpoint

    uvicorn.run("vocalcoach.api:app", host=cli.host, port=cli.port, reload=cli.reload)
