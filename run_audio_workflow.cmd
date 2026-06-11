@echo off
setlocal
set "PATH=%~dp0ffmpeg\bin;%PATH%"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" scripts\audio_skill_workflow.py run --ffmpeg-bin "%~dp0ffmpeg\bin\ffmpeg.exe" --ffprobe-bin "%~dp0ffmpeg\bin\ffprobe.exe" --python-executable ".venv\Scripts\python.exe" %*
) else (
  python scripts\audio_skill_workflow.py run --ffmpeg-bin "%~dp0ffmpeg\bin\ffmpeg.exe" --ffprobe-bin "%~dp0ffmpeg\bin\ffprobe.exe" %*
)
