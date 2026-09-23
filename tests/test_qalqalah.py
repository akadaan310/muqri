"""Qalqalah release-burst detection on synthetic plosives."""

from __future__ import annotations

from app.acoustic.features import AcousticContext
from app.acoustic.qalqalah import analyze_qalqalah, detect_release_burst
from app.models import AlignedUnit, Alignment, RuleInstance, RuleType, Status
from tests.synth import SR, burst, concat, silence, vowel


def plosive(with_burst: bool, burst_amp: float = 0.4):  # type: ignore[no-untyped-def]
    """/a/ vowel + 50 ms occlusion + (optional) release burst + short echo vowel."""
    parts = [vowel(0.15), silence(0.05)]
    if with_burst:
        parts += [burst(amp=burst_amp), vowel(0.05, (500, 1500, 2500), amp=0.2)]
    else:
        parts += [silence(0.06)]
    return concat(*parts, silence(0.1))


def test_burst_detected_after_occlusion() -> None:
    res = detect_release_burst(plosive(True), SR, letter_start_s=0.15, letter_end_s=0.2)
    assert res.closure_found and res.present
    assert 40 <= res.closure_ms <= 70
    assert res.flux_ratio >= 3 and res.rise_db >= 12
    assert abs(res.release_s - 0.2) < 0.015


def test_swallowed_qalqalah_has_no_burst() -> None:
    res = detect_release_burst(plosive(False), SR, letter_start_s=0.15, letter_end_s=0.2)
    assert not res.present


def test_unstopped_letter_has_no_occlusion() -> None:
    # Continuous voicing: the letter was voweled instead of stopped.
    res = detect_release_burst(concat(vowel(0.15), vowel(0.15, (500, 1500, 2500))), SR, letter_start_s=0.15)
    assert not res.closure_found and not res.present


def _diag(samples, make_signal, detail: str = "kubra"):  # type: ignore[no-untyped-def]
    ctx = AcousticContext(make_signal(samples))
    rule = RuleInstance(RuleType.QALQALAH, 0, "ٱلْفَلَقِ", [7], letter="ق", detail=detail, at_waqf=True)
    align = Alignment(units={7: AlignedUnit(7, 0.15, 0.21)}, method="test")
    return analyze_qalqalah(rule, align, ctx)


def test_analyze_qalqalah_statuses(make_signal) -> None:  # type: ignore[no-untyped-def]
    good = _diag(plosive(True), make_signal)
    assert good.status is Status.PASS
    assert good.feedback == "Clear acoustic release burst detected on letter Qaf (ق)."
    bad = _diag(plosive(False), make_signal)
    assert bad.status is Status.FAIL
