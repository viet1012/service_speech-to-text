@echo off
cd /d D:\Viet\service_speech-to-text
set HOST=0.0.0.0
set PORT=7011
set HF_HUB_OFFLINE=1
call .venv\Scripts\activate.bat
python main.py