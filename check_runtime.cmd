@echo off
setlocal
set "PATH=%~dp0ffmpeg\bin;%PATH%"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" scripts\audio_cleanup.py doctor --python-executable ".venv\Scripts\python.exe" --ffmpeg-bin "%~dp0ffmpeg\bin\ffmpeg.exe" --ffprobe-bin "%~dp0ffmpeg\bin\ffprobe.exe" %*
) else (
  python scripts\audio_cleanup.py doctor --ffmpeg-bin "%~dp0ffmpeg\bin\ffmpeg.exe" --ffprobe-bin "%~dp0ffmpeg\bin\ffprobe.exe" %*
)
