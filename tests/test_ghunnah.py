"""Nasal energy ratio and Ghunnah scoring."""

from __future__ import annotations

import pytest

from app.acoustic.features import AcousticContext
from app.acoustic.ghunnah import analyze_ghunnah, nasal_energy_ratio, oral_reference_ner
from app.models import AlignedUnit, Alignment, RuleInstance, RuleType, Status
from tests.synth import SR, concat, nasal_murmur, vowel


def test_nasal_murmur_has_higher_ner_than_open_vowel() -> None:
    assert nasal_energy_ratio(nasal_murmur(0.2), SR) > nasal_energy_ratio(vowel(0.2), SR) + 15


def _setup(make_signal, nasal_s: float, nasal: bool = True):  # type: ignore[no-untyped-def]
    mid = nasal_murmur(nasal_s) if nasal else vowel(nasal_s, seed=3)
    samples = concat(vowel(0.2), vowel(0.2, seed=1), mid, vowel(0.2, seed=2))
    ctx = AcousticContext(make_signal(samples))
    align = Alignment(units={
        0: AlignedUnit(0, 0.0, 0.2), 1: AlignedUnit(1, 0.2, 0.4),
        2: AlignedUnit(2, 0.4, 0.4 + nasal_s), 3: AlignedUnit(3, 0.4 + nasal_s, 0.6 + nasal_s),
    }, method="test")
    rule = RuleInstance(RuleType.GHUNNAH, 0, "إِنَّ", [2], expected_harakat=(2, 2), letter="ن", detail="mushaddad")
    oral = oral_reference_ner([(0.03, 0.17), (0.23, 0.37)], ctx)
    return rule, align, ctx, oral


def test_full_ghunnah_passes(make_signal) -> None:  # type: ignore[no-untyped-def]
    rule, align, ctx, oral = _setup(make_signal, 0.4)
    diag = analyze_ghunnah(rule, align, ctx, haraka_ms=200.0, oral_reference_db=oral)
    assert diag.measured_harakat == pytest.approx(2.0)
    assert diag.status is Status.PASS
    assert diag.metrics["nasal_contrast_db"] > 8


def test_rushed_ghunnah_warns(make_signal) -> None:  # type: ignore[no-untyped-def]
    rule, align, ctx, oral = _setup(make_signal, 0.32)
    diag = analyze_ghunnah(rule, align, ctx, haraka_ms=200.0, oral_reference_db=oral)
    assert diag.measured_harakat == pytest.approx(1.6)
    assert diag.status is Status.WARNING
    assert "slightly rushed (1.6 counts). Hold nasal resonance longer." in diag.feedback


def test_oral_instead_of_nasal_is_flagged(make_signal) -> None:  # type: ignore[no-untyped-def]
    rule, align, ctx, oral = _setup(make_signal, 0.4, nasal=False)
    diag = analyze_ghunnah(rule, align, ctx, haraka_ms=200.0, oral_reference_db=oral)
    assert diag.status is not Status.PASS
    assert "Weak nasalization" in diag.feedback
