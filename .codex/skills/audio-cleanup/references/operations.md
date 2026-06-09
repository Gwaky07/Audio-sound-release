# Operations

## Prerequisites

- Python 3.10+
- `ffmpeg`
- `ffprobe`

Optional but recommended:

- `torch==2.3.1`
- `torchaudio==2.3.1`
- `deepfilternet`

## Useful commands

List presets:

```bash
python ../../../scripts/audio_cleanup.py list-presets
```

Describe one preset:

```bash
python ../../../scripts/audio_cleanup.py describe-preset safe
```

Check runtime:

```bash
python ../../../scripts/audio_cleanup.py doctor
```

Install recommended runtime packages:

```bash
python ../../../scripts/audio_cleanup.py setup
```

Inspect one media file:

```bash
python ../../../scripts/audio_cleanup.py inspect "<input-file>"
```

Dry-run one batch:

```bash
python ../../../scripts/audio_cleanup.py clean "<input-path>" --preset safe --dry-run
```

## Output layout

Each real run writes under `output/run-<timestamp>/`.

Each file gets:

- `audio_preprocess/audio_raw.wav`
- `audio_preprocess/audio_df.wav`
- `audio_preprocess/audio_clean.wav`
- `audio_preprocess/audio_process_report.json`
- `audio_preprocess/audio_process_report.md`
- `transcript_ready/audio.mp3`

## Operational notes

- Use `doctor` before claiming the full pipeline is ready.
- Use `--dry-run` when the user first wants to inspect the generated command plan.
- Tune presets or CLI overrides instead of writing custom FFmpeg one-offs.
- If runtime dependencies are missing, stop and say so plainly.
