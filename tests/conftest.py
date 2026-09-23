from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.audio import AudioSignal, condition  # noqa: E402
from tests.synth import SR  # noqa: E402


@pytest.fixture
def make_signal():
    """Wrap raw samples into a conditioned AudioSignal without denoising."""

    def _make(samples) -> AudioSignal:  # type: ignore[no-untyped-def]
        return condition(samples, SR, denoise="never")

    return _make
