"""Madd duration counting and dynamic tempo estimation."""

from __future__ import annotations

import pytest

from app.acoustic.madd import analyze_madd, classify_counts
from app.acoustic.tempo import estimate_tempo, short_syllable_units
from app.models import AlignedUnit, Alignment, RuleType, Status
from app.tajweed_rules import parse_text


def uniform_alignment(parsed, haraka_s: float, overrides: dict[int, float] | None = None) -> Alignment:
    """Every pronounced unit lasts one harakah unless overridden (duration in seconds)."""
    overrides = overrides or {}
    t = 0.0
    units = {}
    for u in parsed.units:
        if not u.pronounced:
            continue
        d = overrides.get(u.index, haraka_s)
        units[u.index] = AlignedUnit(u.index, t, t + d)
        t += d
    return Alignment(units=units, method="test")


def test_classify_counts_bands() -> None:
    assert classify_counts(2.0, 2, 2) == (Status.PASS, 1.0)
    assert classify_counts(1.75, 2, 2)[0] is Status.PASS  # within ±15 %
    assert classify_counts(1.5, 2, 2)[0] is Status.WARNING
    assert classify_counts(1.0, 2, 2)[0] is Status.FAIL
    assert classify_counts(5.0, 4, 5)[0] is Status.PASS
    assert classify_counts(3.1, 4, 5)[0] is Status.WARNING
    assert classify_counts(0.0, 2, 2) == (Status.FAIL, 0.0)


def test_tempo_is_median_of_plain_short_syllables() -> None:
    parsed = parse_text("إِذَا جَآءَ نَصْرُ ٱللَّهِ وَٱلْفَتْحُ")
    candidates = short_syllable_units(parsed)
    assert candidates, "expected plain CV syllables"
    # Perturb one syllable heavily: the median (with IQR rejection) must ignore it.
    align = uniform_alignment(parsed, 0.2, {candidates[0]: 0.9})
    tempo = estimate_tempo(parsed, align)
    assert tempo.method == "median_short_syllable"
    assert tempo.haraka_ms == pytest.approx(200.0, abs=1e-6)
    assert tempo.harakat_per_minute == pytest.approx(300.0)


def test_madd_muttasil_measured_counts() -> None:
    parsed = parse_text("إِذَا جَآءَ نَصْرُ ٱللَّهِ وَٱلْفَتْحُ")
    rule = next(r for r in parsed.rules if r.rule_type is RuleType.MADD_MUTTASIL)
    carrier, madd = rule.unit_indices
    # Carrier CV = 1 harakah (0.21 s), the madd letter held for 4 more => 5 counts in total.
    align = uniform_alignment(parsed, 0.21, {madd: 0.84})
    diag = analyze_madd(rule, align, haraka_ms=210.0)
    assert diag.measured_harakat == pytest.approx(5.0)
    assert diag.status is Status.PASS
    assert diag.feedback.startswith("Excellent prolongation. Measured 1050ms (5.0 counts)")

    short = uniform_alignment(parsed, 0.21, {madd: 0.441})  # 3.1 counts
    diag = analyze_madd(rule, short, haraka_ms=210.0)
    assert diag.measured_harakat == pytest.approx(3.1)
    assert diag.status is Status.WARNING
    assert "target was 4–5 counts" in diag.feedback


def test_madd_tabii_too_long_and_phrase_final_leniency() -> None:
    parsed = parse_text("وَلَا ٱلضَّآلِّينَ")
    lazim = next(r for r in parsed.rules if r.rule_type is RuleType.MADD_LAZIM)
    align = uniform_alignment(parsed, 0.2, {lazim.unit_indices[1]: 2.8})  # 15 counts
    assert analyze_madd(lazim, align, 200.0).status is Status.FAIL
    lenient = analyze_madd(lazim, align, 200.0, phrase_final=True)
    assert lenient.status is Status.WARNING and "before a stop" in lenient.feedback


def test_missing_alignment_is_skipped() -> None:
    parsed = parse_text("إِذَا جَآءَ")
    rule = parsed.rules[0]
    diag = analyze_madd(rule, Alignment(units={}, method="test"), 200.0)
    assert diag.status is Status.SKIPPED
