# Reference SOP Notes

The original AU notes were treated as a reference, not as a strict contract.

## What informed the implementation

- denoise should happen early
- dynamic cleanup matters for breath handling
- output loudness should be normalized
- manual review still matters for subtle speech artifacts

## How the repository generalizes that idea

Instead of hard-wiring one editor-specific workflow, the repository expresses the same broad intent as:

- reusable presets
- deterministic filter chains
- optional review markers
- report artifacts that support tuning

## What is intentionally broader than the AU notes

- batch folder processing
- natural-language routing
- output reports
- preset-driven overrides
- repo structure designed for future non-FFmpeg stages

## What is still weaker than manual AU work

- speech-aware mouth-click removal
- semantic artifact recognition
- editor-grade visual spectral cleanup

That gap is expected. The first version focuses on reliable automation and a stable interface, not perfect human-equivalent restoration.

