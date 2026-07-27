# Tuning Guide

## Target problem

This repository is optimized first for spoken-word cleanup:

- voiceovers
- course audio
- narration
- interview stems with moderate room noise

## Parameters worth tuning first

### Denoise

Current control surface:

- `--denoise-strength light`
- `--denoise-strength medium`
- `--denoise-strength aggressive`

If speech starts sounding brittle, step down before touching other filters.

### Gate

The gate is intentionally conservative in `safe` and `review`.

If word tails are being clipped:

- add `--disable-gate`
- or lower the gate threshold in the preset JSON

### Loudness

Override integrated loudness with:

```powershell
.\.venv\Scripts\python.exe scripts\audio_cleanup.py clean "<input>" --target-lufs -14
```

### Review markers

If review mode produces too many candidate regions, relax:

- `analysis.silence_threshold_db`
- `analysis.silence_min_duration`

Those values live in `presets/review.json`.

## Suggested calibration workflow

1. Collect 5 to 10 representative source files.
2. Run `safe` and `review`.
3. Compare reports and listening results.
4. Tune preset JSON, not ad hoc shell commands.
5. Keep one preset change per iteration so regressions stay obvious.
