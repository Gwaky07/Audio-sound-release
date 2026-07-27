@echo off
setlocal EnableDelayedExpansion
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"
set "PATH=%~dp0ffmpeg\bin;%PATH%"
if not exist ".venv\Scripts\python.exe" (
  set "PYTHON_CMD="
  py -3.11 -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)" >nul 2>&1 && set "PYTHON_CMD=py -3.11"
  if not defined PYTHON_CMD py -3.10 -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 10) else 1)" >nul 2>&1 && set "PYTHON_CMD=py -3.10"
  if not defined PYTHON_CMD python -c "import sys; raise SystemExit(0 if sys.version_info[:2] in ((3, 10), (3, 11)) else 1)" >nul 2>&1 && set "PYTHON_CMD=python"
  if not defined PYTHON_CMD (
    echo [ERROR] Python 3.11 or 3.10 is required. Python 3.12+ is unsupported by the bundled models.
    exit /b 1
  )
  !PYTHON_CMD! -m venv .venv || exit /b 1
)
".venv\Scripts\python.exe" -c "import sys; raise SystemExit(0 if sys.version_info[:2] in ((3, 10), (3, 11)) else 1)" || (
  echo [ERROR] Existing .venv uses an unsupported Python. Remove .venv and run setup.cmd again.
  exit /b 1
)
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install --editable .
if errorlevel 1 (
  echo [ERROR] Failed to install repository entry points into .venv.
  exit /b 1
)
".venv\Scripts\python.exe" scripts\audio_cleanup.py setup --python-executable ".venv\Scripts\python.exe" %*
