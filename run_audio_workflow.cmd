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
  ".venv\Scripts\python.exe" scripts\audio_skill_workflow.py run --ffmpeg-bin "%FFMPEG_BIN%" --ffprobe-bin "%FFPROBE_BIN%" --python-executable ".venv\Scripts\python.exe" %*
) else (
  python scripts\audio_skill_workflow.py run --ffmpeg-bin "%FFMPEG_BIN%" --ffprobe-bin "%FFPROBE_BIN%" %*
)
