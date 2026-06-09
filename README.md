# Audio-sound

`Audio-sound` is a standalone repository for spoken-word audio processing. It is designed to be triggered from Codex with natural language, but the runtime capability lives in this repository itself: setup, doctor, inspect, clean/process, preset routing, reporting, and output layout are all owned locally.

The Feishu document and the `audio-preprocess` repository were treated as references while shaping the design. They are not runtime dependencies.

## Scope

This repository currently focuses on stable automation for:

- voiceovers
- narration
- lecture and course audio
- interview stems with moderate noise
- batch cleanup with report output

The processing path is:

1. extract source audio to WAV
2. run primary denoise via DeepFilterNet
3. apply FFmpeg mastering filters
4. export clean WAV
5. export transcript-ready MP3
6. write JSON and Markdown reports

## Repository layout

- `.codex/skills/audio-cleanup/`
  - project skill for natural-language routing
- `audio_sound/`
  - config, bootstrap, pipeline, CLI
- `presets/`
  - standalone JSON presets
- `scripts/audio_cleanup.py`
  - local CLI entrypoint
- `docs/`
  - architecture, tuning, reference notes
- `tests/`
  - unit tests for command building and repository behavior

## Prerequisites

- Python 3.10+
- `ffmpeg`
- `ffprobe`

Optional but recommended for the full processing path:

- `torch==2.3.1`
- `torchaudio==2.3.1`
- `deepfilternet`

## Quick start

List presets:

```bash
python scripts/audio_cleanup.py list-presets
```

Check local runtime:

```bash
python scripts/audio_cleanup.py doctor
```

Install recommended local runtime packages:

```bash
python scripts/audio_cleanup.py setup
```

Remove generated outputs and local machine state from the repository:

```bash
python scripts/audio_cleanup.py clean-repo
```

Preview what would be removed first:

```bash
python scripts/audio_cleanup.py clean-repo --dry-run
```

Or on Windows, use the repo helper which creates `.venv` first:

```bash
setup.cmd
```

To verify a clean checkout on Windows:

```bash
doctor.cmd
```

Inspect one file:

```bash
python scripts/audio_cleanup.py inspect "D:/audio/raw-voice.wav"
```

Process one file:

```bash
python scripts/audio_cleanup.py clean "D:/audio/raw-voice.wav"
```

`process` is an alias:

```bash
python scripts/audio_cleanup.py process "D:/audio/raw-voice.wav"
```

Batch process a folder recursively:

```bash
python scripts/audio_cleanup.py clean "D:/audio/batch" --preset review --recursive
```

Dry-run one file:

```bash
python scripts/audio_cleanup.py clean "D:/audio/raw-voice.wav" --dry-run
```

## Presets

- `fast`
  - fast first pass, lighter dynamic processing
- `safe`
  - default spoken-word preset
- `review`
  - cleanup plus heuristic silence candidate reporting
- `voice-isolate`
  - AU-style cleanup using captured noise-only windows to suppress non-voice residue more selectively

Runtime overrides:

- `--target-lufs`
- `--denoise-strength light|medium|aggressive`
- `--disable-gate`
- `--enable-silence-report`
- `--noise-window start:end`
  - repeat this flag to capture multiple clean noise-only spans from the same file
  - example: `--noise-window 143.089208:144.093687 --noise-window 161.583729:169.578792`

Example voice-isolate run:

```bash
python scripts/audio_cleanup.py clean "D:/audio/raw-voice.wav" --preset voice-isolate --noise-window 143.089208:144.093687 --noise-window 161.583729:169.578792
```

## Output layout

Default runs write to `output/run-<timestamp>/`.

Each input file gets an ASCII-only job folder:

```text
output/run-20260527-120000/
  20260527-120000_voice_take_01/
    audio_preprocess/
      audio_raw.wav
      audio_df.wav
      audio_clean.wav
      audio_process_report.json
      audio_process_report.md
      deepfilternet_out/
    transcript_ready/
      audio.mp3
```

Batch summary files are written at the run root:

- `batch-summary.json`
- `batch-summary.md`

## Handoff

For the cleanest shareable repository:

1. Run `python scripts/audio_cleanup.py clean-repo --dry-run`
2. Run `python scripts/audio_cleanup.py clean-repo`
3. Share the repository without `.venv/`, `output/`, `scratch/`, `.omx/`, `.worktrees/`, or cache folders

For the receiving machine:

1. Install Python 3.10+, `ffmpeg`, and `ffprobe`
2. Run `setup.cmd`
3. Run `doctor.cmd`
4. Keep source media outside the repository when possible, or use `scratch/` for temporary local work

The repository is meant to stay source-only. Generated audio, reports, and temporary review files should remain under ignored working directories such as `output/` or `scratch/`.

## Codex usage

This repository contains a project skill at [`.codex/skills/audio-cleanup/SKILL.md`](</S:/Agent/Auto jianji/Audio-sound/.codex/skills/audio-cleanup/SKILL.md>).

Typical prompts:

- “清理这个口播，保守一点”
- “批量处理这个文件夹，先跑快一点”
- “做 review 模式，把可疑停顿标出来”
- “Clean this voiceover and give me transcript-ready audio”

The skill maps intent to local presets and CLI flags, then calls this repository’s own pipeline.

## Verification

Run tests:

```bash
python -m unittest discover tests
```

Run runtime doctor:

```bash
python -m audio_sound.cli doctor
```

## Current non-goals

- exact Adobe Audition parity
- semantic mouth-click classification
- GUI editing
- diarization or ASR workflows

Those can be added later, but the current repository is intentionally centered on stable, automatable audio cleanup.
