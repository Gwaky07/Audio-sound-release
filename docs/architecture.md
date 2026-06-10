# Architecture

## Goal

Make `Audio-sound` a standalone audio processing repository that can be called directly from Codex or from the shell, with no runtime dependence on another local repository.

## Layers

### 1. Skill layer

Location:

- `.codex/skills/audio-sound/`

Responsibility:

- map natural-language intent to the approved workflow mode and the minimum required CLI overrides
- keep Codex invocation stable even as the processing internals evolve
- cover both first-pass delivery and exact timestamp repair in one repository-local entry

### 2. Bootstrap layer

Location:

- `audio_sound/bootstrap.py`

Responsibility:

- inspect Python, FFmpeg, FFprobe, Respiro-en runtime prerequisites, and DeepFilterNet availability
- provide repo-local setup/install commands, including Respiro-en asset bootstrapping
- emit machine-readable runtime reports

### 3. Config layer

Location:

- `audio_sound/config.py`
- `presets/*.json`

Responsibility:

- load and validate presets
- apply runtime overrides without mutating source presets
- resolve environment and local Python defaults

### 4. Pipeline layer

Location:

- `audio_sound/pipeline.py`

Responsibility:

- discover media files
- inspect media with FFprobe
- build deterministic extraction, Respiro-en-first cleanup, denoise, and finalize commands
- maintain ASCII-stable output layout
- generate per-file and batch reports

Runtime note:

- if local Respiro-en repo and weights are configured, the pipeline runs the real detector
- otherwise it falls back to the local heuristic breath detector so the full cleanup chain remains runnable

### 5. CLI layer

Location:

- `audio_sound/cli.py`
- `scripts/audio_cleanup.py`

Responsibility:

- expose `list-presets`, `describe-preset`, `inspect`, `doctor`, `setup`, `clean`, and `process`

## Processing contract

For each input file:

1. extract mono WAV with FFmpeg
2. detect breath regions via Respiro-en
3. apply SpectraMini-style breath control and mouth de-click cleanup
4. run DeepFilterNet on the extracted WAV
5. run FFmpeg mastering filters on the denoised WAV
6. export transcript-ready MP3
7. write JSON and Markdown reports

Default path note:

- the primary path is `Respiro-en -> SpectraMini-style cleanup -> DeepFilterNet -> FFmpeg mastering`
- older `breath_ducking` and `breath_onset_cleanup` FFmpeg filters remain in the repo only as explicit compatibility mode
- compatibility is opt-in through CLI/runtime overrides and is not enabled by default presets

## Why this split

This keeps the natural-language surface stable while making the actual processing chain local, inspectable, and testable. It also keeps future upgrades contained: ASR, click-detection heuristics, or additional mastering stages can be added in the pipeline/config layers without changing the user-facing skill contract.

## Reference inputs, not runtime dependencies

The earlier Feishu notes and the `audio-preprocess` repository informed:

- output layout ideas
- bootstrap expectations
- a sensible DeepFilterNet + FFmpeg ordering

But `Audio-sound` owns its own runtime contract. It should continue to work even if the reference repository disappears or changes.
