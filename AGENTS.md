# Audio-sound Agent Instructions

This repository is a Windows-oriented audio cleanup workflow for Chinese spoken-word audio. When a user asks to process, clean, repair, or prepare audio in this repository, treat it as a request for a final usable delivery by default, not a lightweight preview.

## Default Workflow

Use the repo workflow entrypoint for final delivery:

```powershell
python scripts/audio_skill_workflow.py run "<input-audio>"
```

The default workflow mode is `reference-legacy`. Use another mode only when the user clearly asks for it:

- `reference-style` for a more natural and conservative result
- `final` for balanced final delivery
- `voice-isolate` when noise windows from the same file are provided
- `review` when the user wants markers and review artifacts

Do not use `scripts/audio_cleanup.py clean` as the default final-delivery path. Keep it for setup, inspection, lower-level preset control, compatibility, and troubleshooting.

## Runtime Setup

Before processing audio on a fresh machine, check the runtime:

```powershell
python scripts/audio_cleanup.py doctor
```

If dependencies are missing, run:

```powershell
setup.cmd
```

For release packages that include `tools/Respiro-en`, `tools/respiro-en.pt`, and `ffmpeg/`, use the package-local `.env`. If Respiro assets are not present, run:

```powershell
python scripts/audio_cleanup.py setup-respiro
```

Do not claim Respiro-en, DeepFilterNet, or SpectraMini were used unless `doctor` or the workflow report shows they are actually available. If the workflow falls back to heuristic processing, say so.

## Delivery Rules

- Preserve the original Chinese filename and append the cleanup suffix plus the run identifier.
- Produce final WAV and MP3 outputs.
- Generate and review spectrogram evidence for strict cleanup work.
- If a specific timestamp still has breath, saliva noise, pause residue, or thin artifacts, inspect that local region first and then use exact narrow-window repair.
- Avoid broad edits that can swallow speech onsets or restore isolated noise in silence.
- DeepFilterNet dropout repair may only restore short gaps inside continuous speech; it must not restore isolated noise in leading silence or pause regions.
- After exact repair, re-export MP3 and re-check loudness.

## Keep Out Of Git

Do not commit local runtime assets or generated outputs:

- `.venv/`
- `.env`
- `tools/`
- `output/`
- `scratch/`
- `.omx/`
- `__pycache__/`

Before sharing source, run:

```powershell
python scripts/audio_cleanup.py clean-repo
```

## Skill Reference

The detailed Codex skill lives at:

```text
.codex/skills/audio-sound/SKILL.md
```

Read its `references/workflow.md`, `references/commands.md`, and `references/acceptance.md` when deeper process detail is needed.
