@echo off
setlocal
if not exist ".venv\Scripts\python.exe" (
  python -m venv .venv
)
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" scripts\audio_cleanup.py setup --python-executable ".venv\Scripts\python.exe" %*
