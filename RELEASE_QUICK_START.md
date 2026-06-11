# Audio-sound Release Quick Start

This package includes the repository source, Codex skill, ffmpeg, Respiro-en, and respiro-en.pt.
It does not include a Python virtual environment.

On a receiving Windows machine:

1. Install Python 3.10 or 3.11.
2. Open PowerShell or Command Prompt in this folder.
3. Run `setup.cmd`.
4. Run `check_runtime.cmd` or `doctor.cmd`.
5. Process audio with:

```bat
run_audio_workflow.cmd "D:\audio\raw-voice.wav"
```

After processing, use the final audio from:

```text
output\修音成品\
```

Final names are `修音版_<原音频文件名>.wav` and `修音版_<原音频文件名>.mp3`. If the same source is processed again, the next files are `_01`, `_02`, and so on. Internal stage audio is cleaned by default so users do not need to choose from many intermediate WAV files.

For troubleshooting only, keep internal stage audio with:

```bat
run_audio_workflow.cmd "D:\audio\raw-voice.wav" --keep-intermediate-audio
```

The helper scripts prepend this package's `ffmpeg\bin` to `PATH`, so the receiving machine does not need a system ffmpeg install.

The package `.env` points the processing workflow to:

- `tools\Respiro-en`
- `tools\respiro-en.pt`

If `doctor` reports DeepFilterNet or SpectraMini runtime packages as unavailable, run `setup.cmd` again and check the installed Python version. Respiro-en repository and weights are bundled; the processing report should show whether Respiro-en was actually attempted and succeeded.
