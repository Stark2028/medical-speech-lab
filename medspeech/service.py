"""Small, bounded inference service for the ASR research demonstration.

The service does not persist uploads or transcripts. It is not a clinical tool.
"""

from __future__ import annotations

import asyncio
import io
import math
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

MODEL_ID = os.getenv("MODEL_ID", "openai/whisper-tiny.en")
MODEL_REVISION = os.getenv(
    "MODEL_REVISION", "87c7102498dcde7456f24cfd30239ca606ed9063"
)
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_DURATION_SECONDS = 30.0
WEB_DIR = Path(__file__).resolve().parents[1] / "web"


def decode_wav(payload: bytes) -> tuple[np.ndarray, float]:
    """Validate a short WAV and return finite mono 16 kHz float32 audio."""
    import soundfile as sf
    from scipy.signal import resample_poly

    if len(payload) < 44 or not payload.startswith(b"RIFF") or payload[8:12] != b"WAVE":
        raise ValueError("Upload a WAV audio file.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise ValueError("Audio exceeds the 8 MB limit.")
    try:
        with sf.SoundFile(io.BytesIO(payload)) as source:
            if source.channels not in (1, 2):
                raise ValueError("Audio must have one or two channels.")
            if not 8000 <= source.samplerate <= 48000:
                raise ValueError("Audio sample rate must be 8–48 kHz.")
            duration = len(source) / source.samplerate
            if duration <= 0 or duration > MAX_DURATION_SECONDS:
                raise ValueError("Audio must be between 0 and 30 seconds.")
            samples = source.read(dtype="float32", always_2d=True)
            sample_rate = source.samplerate
    except (OSError, RuntimeError) as error:
        raise ValueError("The WAV file could not be decoded.") from error
    mono = samples.mean(axis=1)
    if not np.isfinite(mono).all():
        raise ValueError("Audio contains invalid samples.")
    if float(np.sqrt(np.mean(mono.astype(np.float64) ** 2))) < 1e-5:
        raise ValueError("Audio is silent or too quiet to transcribe.")
    if sample_rate != 16000:
        common = math.gcd(sample_rate, 16000)
        mono = resample_poly(mono, 16000 // common, sample_rate // common)
    return np.asarray(mono, dtype=np.float32), duration


class WhisperEngine:
    """One process owns one model; requests are serialized by the API."""

    def __init__(self, model_id: str = MODEL_ID, revision: str | None = MODEL_REVISION):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        local = Path(model_id).exists()
        revision = None if local else revision
        self.model_id = model_id
        self.revision = revision
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = WhisperProcessor.from_pretrained(model_id, revision=revision)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_id, revision=revision
        ).to(self.device)
        self.model.eval()

    def transcribe(self, samples: np.ndarray) -> str:
        import torch

        inputs = self.processor(
            samples, sampling_rate=16000, return_tensors="pt", return_attention_mask=True
        )
        with torch.inference_mode():
            tokens = self.model.generate(
                input_features=inputs.input_features.to(self.device),
                attention_mask=inputs.attention_mask.to(self.device),
                max_new_tokens=128,
            )
        return self.processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()


def create_app(engine: WhisperEngine | None = None):
    """Create an app with an injectable engine for offline API tests."""
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = engine if engine is not None else WhisperEngine()
        app.state.inference_slot = asyncio.Semaphore(1)
        yield
        app.state.engine = None

    app = FastAPI(
        title="Medical Speech Lab API",
        description="Research demo for short English symptom utterances. Not for clinical use.",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/health/live", include_in_schema=False)
    def live():
        return {"status": "ok"}

    @app.get("/health/ready", include_in_schema=False)
    def ready(request: Request):
        if getattr(request.app.state, "engine", None) is None:
            raise HTTPException(status_code=503, detail="Model is loading.")
        return {"status": "ready"}

    @app.get("/api/v1/model")
    def model_info(request: Request):
        current = getattr(request.app.state, "engine", None)
        if current is None:
            raise HTTPException(status_code=503, detail="Model is loading.")
        return {
            "model": current.model_id,
            "revision": current.revision,
            "purpose": "research demonstration",
            "clinical_use": False,
            "audio_retained": False,
            "max_duration_seconds": MAX_DURATION_SECONDS,
        }

    @app.post("/api/v1/transcribe")
    async def transcribe(request: Request):
        if request.headers.get("content-type", "").split(";")[0].lower() not in (
            "audio/wav", "audio/wave", "audio/x-wav"
        ):
            raise HTTPException(status_code=415, detail="Send a WAV file as audio/wav.")
        claimed_size = request.headers.get("content-length")
        if claimed_size and claimed_size.isdigit() and int(claimed_size) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Audio exceeds the 8 MB limit.")
        chunks, total = [], 0
        async for chunk in request.stream():
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="Audio exceeds the 8 MB limit.")
            chunks.append(chunk)
        try:
            samples, duration = decode_wav(b"".join(chunks))
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        current = getattr(request.app.state, "engine", None)
        if current is None:
            raise HTTPException(status_code=503, detail="Model is loading.")
        try:
            await asyncio.wait_for(request.app.state.inference_slot.acquire(), timeout=0.25)
        except asyncio.TimeoutError as error:
            raise HTTPException(status_code=429, detail="Model is busy. Try again shortly.") from error
        started = time.perf_counter()
        try:
            text = await run_in_threadpool(current.transcribe, samples)
        finally:
            request.app.state.inference_slot.release()
        return {
            "text": text,
            "duration_seconds": round(duration, 2),
            "processing_seconds": round(time.perf_counter() - started, 2),
            "model": current.model_id,
            "research_only": True,
        }

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/assets/{asset_name}", include_in_schema=False)
    def asset(asset_name: str):
        if asset_name not in {"styles.css", "app.js", "favicon.svg"}:
            raise HTTPException(status_code=404)
        return FileResponse(WEB_DIR / asset_name)

    return app


app = create_app()
