@echo off
setlocal
cd /d "%~dp0"
set "FFMPEG_BIN=ffmpeg"
set "FFPROBE_BIN=ffprobe"
if exist "%~dp0ffmpeg\bin\ffmpeg.exe" (
  set "FFMPEG_BIN=%~dp0ffmpeg\bin\ffmpeg.exe"
  set "FFPROBE_BIN=%~dp0ffmpeg\bin\ffprobe.exe"
)
if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Repository runtime is missing. Run setup.cmd first.
  exit /b 1
)
".venv\Scripts\python.exe" -m audio_sound.delivery_verifier --ffmpeg-bin "%FFMPEG_BIN%" --ffprobe-bin "%FFPROBE_BIN%" %*
