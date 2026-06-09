# Audio-sound Standalone Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `Audio-sound` into a standalone audio processing repository that owns setup, doctor, cleanup, reporting, and Codex-triggered audio automation without runtime coupling to `audio-preprocess`.

**Architecture:** Keep the Codex skill as the natural-language control surface, but move all runtime capability into this repository. Split responsibilities into config/preset parsing, bootstrap/runtime checks, deterministic pipeline construction, and a CLI that exposes stable commands for setup, doctor, inspect, and clean/process.

**Tech Stack:** Python 3.10+, FFmpeg/FFprobe, optional DeepFilterNet via `df.enhance`, JSON presets, `unittest`

---

### Task 1: Define standalone behavior with tests

**Files:**
- Create: `tests/test_config.py`
- Create: `tests/test_bootstrap.py`
- Create: `tests/test_cli.py`
- Modify: `tests/test_pipeline.py`

- [ ] Add tests that describe standalone preset parsing, runtime reporting, output layout generation, CLI routing, and dry-run processing.
- [ ] Run the targeted test suite and confirm the new tests fail against the current backend-coupled implementation.

### Task 2: Replace backend-coupled config and runtime code

**Files:**
- Create: `audio_sound/config.py`
- Create: `audio_sound/bootstrap.py`
- Modify: `audio_sound/__init__.py`
- Modify: `pyproject.toml`

- [ ] Add typed preset/config dataclasses and JSON preset loading helpers.
- [ ] Add bootstrap/install/runtime inspection helpers for Python, FFmpeg, FFprobe, and DeepFilterNet.
- [ ] Update package metadata to reflect standalone audio processing scope.

### Task 3: Rewrite the processing pipeline around local capability

**Files:**
- Modify: `audio_sound/pipeline.py`

- [ ] Remove backend selection and any `audio-preprocess` runtime resolution.
- [ ] Implement media discovery, FFprobe inspection, ASCII-stable output layout, command builders, dry-run reporting, real execution, silence detection, and Markdown summary generation.
- [ ] Keep command construction deterministic and testable without real media assets.

### Task 4: Rewrite the CLI and repo entrypoints

**Files:**
- Modify: `audio_sound/cli.py`
- Modify: `scripts/audio_cleanup.py`
- Create: `.env.example`
- Create: `requirements.txt`
- Create: `setup.cmd`
- Create: `doctor.cmd`
- Create: `.cargo-home/config.toml`

- [ ] Expose `list-presets`, `describe-preset`, `inspect`, `doctor`, `setup`, `clean`, and `process`.
- [ ] Make `process` an alias of `clean`.
- [ ] Add repo-local helper files for setup and doctor workflows.

### Task 5: Rewrite presets and docs around the standalone contract

**Files:**
- Modify: `presets/fast.json`
- Modify: `presets/safe.json`
- Modify: `presets/review.json`
- Modify: `README.md`
- Modify: `docs/architecture.md`
- Modify: `.codex/skills/audio-cleanup/SKILL.md`
- Modify: `.codex/skills/audio-cleanup/references/operations.md`
- Modify: `.codex/skills/audio-cleanup/references/preset-routing.md`

- [ ] Expand presets to include extraction, DeepFilterNet, mastering, analysis, and transcript-export sections.
- [ ] Remove all descriptions that frame this repository as a backend selector or wrapper.
- [ ] Keep the skill focused on natural-language routing into the local standalone CLI.

### Task 6: Verify the repository end to end

**Files:**
- Verify only

- [ ] Run `python -m unittest discover tests`.
- [ ] Run `python -m audio_sound.cli doctor`.
- [ ] Run `python scripts/audio_cleanup.py list-presets`.
- [ ] Run `python scripts/audio_cleanup.py describe-preset safe`.
- [ ] If FFmpeg is available, generate a small sample and run a dry-run clean command to confirm the standalone report shape.
