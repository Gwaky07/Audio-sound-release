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
2. detect breath regions via Respiro-en
3. apply SpectraMini-style breath control and mouth de-click cleanup
4. run primary denoise via DeepFilterNet
5. apply FFmpeg mastering filters
6. export clean WAV
7. export transcript-ready MP3
8. write JSON and Markdown reports

Legacy FFmpeg breath ducking and breath-onset cleanup are still available as compatibility filters, but they are no longer part of the default path.

## Repository layout

- `.codex/skills/audio-sound/`
  - single project skill for the approved repo-local audio workflow
- `audio_sound/`
  - config, bootstrap, pipeline, CLI
- `audio_sound/skill_workflow.py`
  - stable repository workflow with spectrogram and delivery artifacts
- `presets/`
  - standalone JSON presets
- `scripts/audio_cleanup.py`
  - local CLI entrypoint
- `scripts/audio_skill_workflow.py`
  - stable workflow entrypoint
- `docs/`
  - architecture, tuning, reference notes
- `tests/`
  - unit tests for command building and repository behavior

## Prerequisites

- Python 3.10+
- `ffmpeg`
- `ffprobe`

Optional but recommended for the full processing path:

- `torch==2.2.2`
- `torchaudio==2.2.2`
- `librosa==0.10.0`
- `intervaltree==3.1.0`
- `deepfilternet`

Optional external assets for the breath-first path:

- local `Respiro-en` repository checkout
- local `respiro-en.pt` model weights

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

Install real Respiro-en assets locally and write `.env` automatically:

```bash
python scripts/audio_cleanup.py setup-respiro
```

Or on Windows, use the repo helper which creates `.venv` first:

```bash
setup.cmd
```

Inspect one file:

```bash
python scripts/audio_cleanup.py inspect "D:/audio/raw-voice.wav"
```

Process one file with the lower-level preset CLI:

```bash
python scripts/audio_cleanup.py clean "D:/audio/raw-voice.wav"
```

Run the final-delivery stable repository workflow:

```bash
python scripts/audio_skill_workflow.py run "D:/audio/raw-voice.wav"
```

Installed entrypoint for the same final-delivery workflow:

```bash
audio-skill-workflow run "D:/audio/raw-voice.wav"
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

Run with the repo-local `.venv` and real Respiro-en assets from `.env`:

```bash
.venv/Scripts/python.exe scripts/audio_cleanup.py clean "D:/audio/raw-voice.wav" --python-executable ".venv/Scripts/python.exe"
```

If `AUDIO_SOUND_RESPIRO_REPO` or `AUDIO_SOUND_RESPIRO_WEIGHTS` is missing, the pipeline falls back to the local heuristic breath detector and still completes the full cleanup flow.

## Presets

- `fast`
  - fast first pass, lighter dynamic processing
- `safe`
  - default spoken-word preset with breath-first ordering
- `review`
  - cleanup plus heuristic silence candidate reporting
- `voice-isolate`
  - AU-style cleanup using captured noise-only windows to suppress non-voice residue more selectively

Runtime overrides:

- `--target-lufs`
- `--denoise-strength light|medium|aggressive`
- `--disable-gate`
- `--enable-silence-report`
- `--attenuation-db`
- `--respiro-threshold`
- `--respiro-min-length-ms`
- `--respiro-repo`
- `--respiro-weights`
- `--enable-legacy-breath-filters`
  - opt back into the older `breath_ducking` and `breath_onset_cleanup` compatibility filters
- `--skip-spectramini`
- `--skip-deepfilternet`
- `--noise-window start:end`
  - repeat this flag to capture multiple clean noise-only spans from the same file
  - example: `--noise-window 143.089208:144.093687 --noise-window 161.583729:169.578792`

Example voice-isolate run:

```bash
python scripts/audio_cleanup.py clean "D:/audio/raw-voice.wav" --preset voice-isolate --noise-window 143.089208:144.093687 --noise-window 161.583729:169.578792
```

Example real Respiro-en run with explicit paths:

```bash
python scripts/audio_cleanup.py clean "D:/audio/raw-voice.wav" --respiro-repo "D:/audio-tools/Respiro-en" --respiro-weights "D:/audio-tools/respiro-en.pt"
```

Example compatibility run with the older legacy breath filters restored:

```bash
python scripts/audio_cleanup.py clean "D:/audio/raw-voice.wav" --preset voice-isolate --enable-legacy-breath-filters
```

Describe stable workflow modes:

```bash
python scripts/audio_skill_workflow.py describe-modes
```

Stable workflow with focused spectrogram windows:

```bash
python scripts/audio_skill_workflow.py run "D:/audio/raw-voice.wav" --focus-window pause_a,12,12 --focus-window pause_b,29,3
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

## Codex usage

This repository contains a single project skill at [`.codex/skills/audio-sound/SKILL.md`](</C:/Users/Guanghe/.codex/worktrees/7fdc/Audio-sound/.codex/skills/audio-sound/SKILL.md>).

Default rule:

- In this repository, if you ask Codex to process or clean audio, it should treat that as a request for the final usable version by default.
- The default target is the approved strict spoken-word finish: remove breaths, gasps, saliva noise, pause residue, and room noise as cleanly as possible, keep loudness and peak in the approved range, preserve Chinese filenames, and deliver directly usable WAV/MP3 outputs.
- Unless you explicitly ask for another mode, Codex should prefer `reference-legacy` and continue with node inspection plus exact repair until the result is clean enough to deliver.
- For final delivery, prefer `audio-skill-workflow run ...` or `python scripts/audio_skill_workflow.py run ...`; keep `audio_cleanup.py clean` for lower-level preset control, inspection, setup, and compatibility paths.

Recommended Chinese prompts:

- “用 `audio-sound` 按最终成品标准处理这个音频，直接给我可用版本。”
- “用 `audio-sound` 清理这个中文口播，去掉气口、吸气、呼吸音、口水音，顺便处理降噪、响度和音量，按最终版交付。”
- “用 `audio-sound` 批量处理这个文件夹，默认都按最终可用成品做，不要先出轻处理版。”
- “用 `audio-sound` 处理完后继续巡检频谱，把残留呼吸音和细丝节点补干净。”
- “用 `audio-sound` 按仓库认可标准处理，并保留中文原文件名加后缀输出。”

The skill maps this intent to the repository workflow, validates the runtime, runs the approved cleanup chain, checks spectrograms and loudness, and applies exact timestamp repair when needed.

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
