"""Mudood engine: all Madd types measured in harakat of the reciter's own (local) tempo.

A Madd's length is the whole long syllable (carrier + madd letter) divided by the harakah, so a
natural madd measures 2 counts. Targets follow the tareeq (Shatibiyyah 4–5 / Tayyibah 2 for
Munfasil and Silah Kubra) and tolerances are absolute, in harakat, per madd type.

Waqf handling:

* Over-extension of the last Madd before a stop is capped at WARNING: phrase-final lengthening
  is universal, and master reciters routinely exceed the canonical count there.
* In ``taraweeh_adapted`` mode, a Madd cut short right before a breath is reported as
  ``VALID_NECESSARY_PAUSE`` (Waqf al-Dharoori) rather than a timing error.
"""

from __future__ import annotations

from app.models import RuleDiagnostic, RuleInstance, RuleType, Status
from app.tajweed_rules.base import EvalContext, band_status, ms, name_of, skipped

# Tolerance windows in harakat (spec §4C). ``None`` = flexible (esthetic) madd.
TOLERANCE: dict[RuleType, float | None] = {
    RuleType.MADD_TABII: 0.25,
    RuleType.MADD_MUTTASIL: 0.40,
    RuleType.MADD_MUNFASIL: 0.40,
    RuleType.MADD_LAZIM: 0.30,
    RuleType.MADD_BADAL: 0.25,
    RuleType.MADD_IWAD: 0.25,
    RuleType.MADD_SILAH_SUGHRA: 0.35,
    RuleType.MADD_SILAH_KUBRA: 0.35,
    RuleType.MADD_ARID: None,
    RuleType.MADD_LEEN: None,
}
FLEXIBLE_TOL = 0.25


def classify_counts(measured: float, lo: float, hi: float, tol: float) -> tuple[Status, float]:
    """Status and score for a measured count against a target band with absolute tolerance."""
    if measured <= 0:
        return Status.FAIL, 0.0
    return band_status(measured, lo, hi, tol)


def _fmt(lo: float, hi: float) -> str:
    return f"{lo:g}" if lo == hi else f"{lo:g}–{hi:g}"


def validate_madd(rule: RuleInstance, ev: EvalContext) -> RuleDiagnostic:
    if rule.expected_harakat is None:
        raise ValueError(f"{rule.rule_type} has no expected harakat")
    lo, hi = rule.expected_harakat
    name = name_of(rule)
    span = ev.span(rule.unit_indices)
    if span is None:
        return skipped(rule, f"{name} on '{rule.word}' could not be located in the audio.")
    start, end = span
    haraka = ev.haraka_ms((start + end) / 2)
    if haraka <= 0:
        return skipped(rule, "Tempo unavailable", span)
    dur_ms = (end - start) * 1000.0
    measured = dur_ms / haraka
    tol = TOLERANCE.get(rule.rule_type)
    flexible = tol is None
    status, score = classify_counts(measured, lo, hi, FLEXIBLE_TOL if tol is None else tol)
    if flexible and measured > hi:
        # 'Aaridh/Leen are esthetic: any length up to ~6 counts is valid, more is a stylistic excess.
        status, score = band_status(measured, lo, hi + 0.5, 1.0)

    phrase_final = rule.word_index in ev.parsed.phrase_final_words()
    target = _fmt(lo, hi)
    metrics = {"duration_ms": dur_ms, "haraka_ms": haraka}
    if status is Status.PASS:
        feedback = f"Excellent prolongation. Measured {dur_ms:.0f}ms ({measured:.1f} counts); target {target}."
    elif measured < lo:
        feedback = (f"{name} on '{rule.word}' was shortened: measured {measured:.1f} counts ({dur_ms:.0f}ms); "
                    f"target was {target} counts. Hold the vowel longer.")
        last = rule.unit_indices[-1]
        pause = ev.pause_after(last)
        if ev.mode == "taraweeh_adapted" and pause is not None and pause.breath:
            status, score = Status.VALID_NECESSARY_PAUSE, 1.0
            feedback = (f"{name} on '{rule.word}' was cut to {measured:.1f} counts right before a breath: treated "
                        "as a necessary pause (Waqf al-Dharoori), not a Tajweed error.")
    else:
        feedback = (f"{name} on '{rule.word}' was over-extended: measured {measured:.1f} counts ({dur_ms:.0f}ms); "
                    f"target was {target} counts.")
        if phrase_final and status is Status.FAIL:
            status, score = Status.WARNING, max(score, 0.6)
            feedback += " Some lengthening before a stop is natural; keep it proportionate."
    if rule.rule_type in (RuleType.MADD_ARID, RuleType.MADD_LEEN) and status is Status.PASS:
        nearest = min((2, 4, 6), key=lambda c: abs(c - measured))
        feedback += f" Realised as the {nearest}-count option."
    return RuleDiagnostic(
        rule_type=rule.rule_type, word=rule.word, start_ms=ms(start), end_ms=ms(end), status=status,
        feedback=feedback, expected_harakat=lo, expected_range=(lo, hi), measured_harakat=measured, score=score,
        metrics=metrics,
    )
