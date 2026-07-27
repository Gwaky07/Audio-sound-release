from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audio_sound.narrow_onset_cleanup import (  # noqa: E402
    NarrowConfig,
    _build_detection_windows,
    _load_samples,
    _narrow_windows,
    _pre_silence_windows,
    _write_report,
    main,
    parse_args,
)

__all__ = [
    "NarrowConfig",
    "_build_detection_windows",
    "_load_samples",
    "_narrow_windows",
    "_pre_silence_windows",
    "_write_report",
    "main",
    "parse_args",
]


if __name__ == "__main__":
    raise SystemExit(main())
