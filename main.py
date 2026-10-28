import io
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import numpy as np
from faster_whisper import WhisperModel, decode_audio
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 10 * 1024 * 1024
MAX_DURATION_SECONDS = 60
SAMPLE_RATE = 16000
ALLOWED_EXTENSIONS = {".webm", ".ogg", ".wav", ".m4a", ".mp3", ".mp4", ".aac", ".flac"}


class TranscriptionResponse(BaseModel):
    text: str
    language: str
    duration: float


def warm_up(model: WhisperModel) -> None:
    # CUDA library errors (cuBLAS/cuDNN) often surface only on first inference.
    segments, _ = model.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), language="en")
    list(segments)


def load_model() -> WhisperModel:
    model_name = os.getenv("WHISPER_MODEL", "large-v3")
    try:
        model = WhisperModel(model_name, device="cuda", compute_type="int8_float16")
        warm_up(model)
        logger.info("Loaded %s on CUDA (int8_float16)", model_name)
        return model
    except Exception:
        logger.warning("CUDA model initialization failed; falling back to CPU", exc_info=True)
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    logger.info("Loaded %s on CPU (int8)", model_name)
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.whisper_model = load_model()
    yield


app = FastAPI(lifespan=lifespan)


def load_audio(content: bytes) -> np.ndarray:
    # Decode instead of reading container metadata: browser MediaRecorder webm often has no duration.
    try:
        audio = decode_audio(io.BytesIO(content), sampling_rate=SAMPLE_RATE)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Không đọc được file audio") from exc

    if audio.size == 0:
        raise HTTPException(status_code=400, detail="Audio contains no samples")
    if audio.size / SAMPLE_RATE > MAX_DURATION_SECONDS:
        raise HTTPException(status_code=413, detail="Audio must be 60 seconds or shorter")
    return audio


@app.post("/transcribe", response_model=TranscriptionResponse)
def transcribe(
    file: UploadFile = File(...),
    language: Literal["vi", "en", "ja"] = Form(...),
    prompt: str | None = Form(default=None),
) -> TranscriptionResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail="Audio file must be webm, ogg, wav, m4a, mp3, mp4, aac, or flac",
        )

    content = file.file.read(MAX_FILE_SIZE + 1)
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Audio file must be 10 MB or smaller")
    if not content:
        raise HTTPException(status_code=400, detail="Audio file is empty")

    audio = load_audio(content)

    model: WhisperModel = app.state.whisper_model
    segments, info = model.transcribe(
        audio,
        language=language,
        vad_filter=True,
        beam_size=5,
        initial_prompt=prompt or None,
    )
    text = "".join(segment.text for segment in segments).strip()

    return TranscriptionResponse(
        text=text,
        language=info.language,
        duration=round(info.duration, 3),
    )


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        app,
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
    )
