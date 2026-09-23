"""Formant-based Tafkheem / Tarqeeq judgement."""

from __future__ import annotations

import pytest

import app.acoustic.features as features
from app.acoustic.features import AcousticContext
from app.acoustic.tafkheem import judge_weight
from app.models import Status, Vowel
from tests.synth import concat, vowel


@pytest.mark.parametrize("use_praat", [True, False])
def test_formant_tracker_recovers_f2(make_signal, monkeypatch, use_praat: bool) -> None:  # type: ignore[no-untyped-def]
    if use_praat and not features.HAVE_PARSELMOUTH:
        pytest.skip("praat-parselmouth not installed")
    monkeypatch.setattr(features, "HAVE_PARSELMOUTH", use_praat and features.HAVE_PARSELMOUTH)
    ctx = AcousticContext(make_signal(concat(vowel(0.3, (700, 1100, 2500)), vowel(0.3, (700, 1800, 2600)))))
    heavy, light = ctx.formants(0.05, 0.28), ctx.formants(0.35, 0.58)
    assert heavy.f2 == pytest.approx(1100, rel=0.06)
    assert light.f2 == pytest.approx(1800, rel=0.06)
    assert heavy.f1 == pytest.approx(700, rel=0.08)


def _fm(f1: float, f2: float):  # type: ignore[no-untyped-def]
    return features.FormantEstimate(f1, f2, 2600, 80, 100, 10)


def test_relative_heaviness_index() -> None:
    ref = 1800 - 700
    status, _, h = judge_weight(True, _fm(700, 1100), ref, Vowel.FATHA)
    assert status is Status.PASS and h == pytest.approx(1 - 400 / 1100)
    assert judge_weight(True, _fm(700, 1750), ref, Vowel.FATHA)[0] is Status.FAIL
    assert judge_weight(False, _fm(700, 1780), ref, Vowel.FATHA)[0] is Status.PASS
    assert judge_weight(False, _fm(700, 1150), ref, Vowel.FATHA)[0] is Status.FAIL


def test_absolute_fallback_only_for_fathah() -> None:
    assert judge_weight(True, _fm(700, 1200), None, Vowel.FATHA)[0] is Status.PASS
    assert judge_weight(False, _fm(700, 1900), None, Vowel.FATHA)[0] is Status.PASS
    assert judge_weight(True, _fm(350, 800), None, Vowel.DAMMA)[0] is Status.SKIPPED
