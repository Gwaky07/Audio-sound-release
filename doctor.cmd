@echo off
setlocal
set "FFMPEG_BIN=ffmpeg"
set "FFPROBE_BIN=ffprobe"
if exist "%~dp0ffmpeg\bin\ffmpeg.exe" (
  set "PATH=%~dp0ffmpeg\bin;%PATH%"
  set "FFMPEG_BIN=%~dp0ffmpeg\bin\ffmpeg.exe"
  set "FFPROBE_BIN=%~dp0ffmpeg\bin\ffprobe.exe"
)
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" scripts\audio_cleanup.py doctor --python-executable ".venv\Scripts\python.exe" --ffmpeg-bin "%FFMPEG_BIN%" --ffprobe-bin "%FFPROBE_BIN%" %*
) else (
  python scripts\audio_cleanup.py doctor --ffmpeg-bin "%FFMPEG_BIN%" --ffprobe-bin "%FFPROBE_BIN%" %*
)
