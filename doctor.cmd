@echo off
setlocal
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" scripts\audio_cleanup.py doctor --python-executable ".venv\Scripts\python.exe" %*
) else (
  python scripts\audio_cleanup.py doctor %*
)
