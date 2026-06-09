---
name: audio-cleanup
description: "Use this skill whenever the user wants audio cleanup, spoken-word post-processing, denoise, normalization, batch voice cleanup, lecture cleanup, or natural-language audio handling in this repository."
---

# Audio Cleanup Skill

This skill routes natural-language audio processing requests to the standalone local pipeline in `../../../scripts/audio_cleanup.py`.

Read `references/preset-routing.md` when the user intent is ambiguous. Read `references/operations.md` for command patterns and runtime checks.

## What this skill is for

- spoken-word cleanup
- voiceover cleanup
- lecture or course audio cleanup
- batch cleanup of many files
- review-oriented runs that should also emit heuristic candidate regions

## Default behavior

1. Resolve the target input path.
2. Choose a preset:
   - `safe` by default
   - `fast` for speed or rough first pass
   - `review` when the user wants markers, QA, or candidate regions
3. Add only the minimum overrides implied by the request.
4. Run the local CLI.
5. Report output root and generated artifacts.

## Command patterns

Single file, default safe preset:

```bash
python ../../../scripts/audio_cleanup.py clean "<input-file>"
```

Folder, fast preset:

```bash
python ../../../scripts/audio_cleanup.py clean "<input-folder>" --preset fast --recursive
```

Review mode:

```bash
python ../../../scripts/audio_cleanup.py clean "<input-file-or-folder>" --preset review
```

Aggressive denoise:

```bash
python ../../../scripts/audio_cleanup.py clean "<input-file>" --preset safe --denoise-strength aggressive
```

Dry-run:

```bash
python ../../../scripts/audio_cleanup.py clean "<input-file>" --dry-run
```

## Rules

- Use `safe` unless the wording clearly calls for `fast` or `review`.
- Prefer repository CLI flags over ad hoc shell pipelines.
- Do not frame this repository as a wrapper around another local repo.
- Do not claim exact Adobe Audition parity.
- If `ffmpeg`, `ffprobe`, or DeepFilterNet is missing, say that clearly based on `doctor`.
- When the user asks for “fully automatic”, keep the distinction clear between deterministic cleanup and heuristic review reporting.
