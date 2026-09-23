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


def make_eval(samples, spans: dict[int, tuple[float, float]], parsed, haraka_ms: float = 200.0,  # type: ignore[no-untyped-def]
              mode: str = "studio", pauses=None):
    """EvalContext over synthetic audio with a known alignment (unit index -> (start_s, end_s))."""
    from app.acoustic.features import AcousticContext
    from app.acoustic.tempo import TempoEstimate
    from app.models import AlignedUnit, Alignment
    from app.tajweed_rules.base import EvalContext

    audio = condition(samples, SR, denoise="never")
    align = Alignment(units={i: AlignedUnit(i, a, b) for i, (a, b) in spans.items()}, method="test")
    return EvalContext(parsed=parsed, alignment=align, ctx=AcousticContext(audio),
                       tempo=TempoEstimate(haraka_ms, 10, "test"), mode=mode, pauses=pauses or [])


@pytest.fixture
def eval_ctx():
    return make_eval
