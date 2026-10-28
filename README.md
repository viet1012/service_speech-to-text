# Speech-to-text service

FastAPI service using faster-whisper. The model loads once when the app starts; it tries CUDA (`int8_float16`) first and falls back to CPU (`int8`).

## Install

Use Python 3.10 or newer, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Install **ffmpeg** and make sure it is on `PATH` (e.g. `winget install Gyan.FFmpeg`). Audio is decoded via PyAV (bundled with faster-whisper), so webm/ogg recorded by a browser `MediaRecorder` works even without duration metadata.

## Run

```powershell
python main.py
```

This starts the service with Uvicorn. Configure with environment variables: `WHISPER_MODEL` (default `large-v3`), `HOST` (default `127.0.0.1`), and `PORT` (default `8000`). For example, set `WHISPER_MODEL=small` to use a smaller model. To launch with the Uvicorn CLI and default bind settings, use `uvicorn main:app --host 127.0.0.1 --port 8000`.

`POST /transcribe` accepts multipart fields `file` (webm/ogg/wav/m4a/mp3/mp4/aac/flac), `language` (`vi`, `en`, or `ja`), and optional `prompt` (glossary/context). Files are limited to 10 MB and 60 seconds. The JSON response contains `text`, detected `language`, and audio `duration` in seconds.

## GPU

faster-whisper runs on CTranslate2 (no PyTorch needed). GPU mode requires the CUDA 12 cuBLAS and cuDNN 9 libraries on `PATH`, e.g. `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12` or a full CUDA Toolkit install.

**RTX 50xx (Blackwell):** needs CUDA **12.8+** and a matching NVIDIA driver (R570+); older CUDA builds have no kernels for these GPUs. Use a recent `ctranslate2` release built against CUDA 12.8+ if the default wheel fails.

At startup the service runs a short warm-up on CUDA; if anything fails (no GPU, missing DLLs, unsupported architecture) it logs a warning and falls back to CPU `int8`. Check the startup log to see which device is in use.
"# service_speech-to-text" 
