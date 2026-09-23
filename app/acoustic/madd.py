"""Madd (vowel prolongation) duration analyzer.

The measured length of a Madd is the whole long syllable — the carrier consonant with its vowel
plus the madd letter — divided by the reciter's own harakah. This matches how teachers count:
a short open syllable is 1 harakah, a natural madd syllable is 2.
"""

from __future__ import annotations

from app.models import Alignment, RuleDiagnostic, RuleInstance, RuleType, Status

DEFAULT_TOLERANCE = 0.15

RULE_NAMES: dict[RuleType, str] = {
    RuleType.MADD_TABII: "Madd Tabi'i",
    RuleType.MADD_MUTTASIL: "Madd Muttasil",
    RuleType.MADD_MUNFASIL: "Madd Munfasil",
    RuleType.MADD_LAZIM: "Madd Lazim",
    RuleType.MADD_ARID: "Madd 'Arid lil-Sukun",
}


def _fmt_range(lo: float, hi: float) -> str:
    return f"{lo:g}" if lo == hi else f"{lo:g}–{hi:g}"


def classify_counts(measured: float, lo: float, hi: float,
                    tolerance: float = DEFAULT_TOLERANCE) -> tuple[Status, float]:
    """Return (status, score in [0,1]) for a measured harakat count against [lo, hi]."""
    if measured <= 0:
        return Status.FAIL, 0.0
    lo_ok, hi_ok = lo * (1 - tolerance), hi * (1 + tolerance)
    if lo_ok <= measured <= hi_ok:
        return Status.PASS, 1.0
    if measured < lo_ok:
        rel = (lo_ok - measured) / lo
    else:
        rel = (measured - hi_ok) / hi
    score = max(0.0, 1.0 - rel / (2 * tolerance + 0.2))
    return (Status.WARNING if rel <= tolerance else Status.FAIL), score


def analyze_madd(rule: RuleInstance, alignment: Alignment, haraka_ms: float,
                 *, tolerance: float = DEFAULT_TOLERANCE, phrase_final: bool = False) -> RuleDiagnostic:
    """Score a Madd rule.

    ``phrase_final`` marks a Madd in the last word before a stop. Reciters (including master
    reciters) lengthen phrase-final syllables well beyond their running tempo, so
    over-extension there is reported as a WARNING rather than a FAIL.
    """
    if rule.expected_harakat is None:
        raise ValueError(f"Rule {rule.rule_type} has no expected harakat")
    lo, hi = rule.expected_harakat
    name = RULE_NAMES.get(rule.rule_type, str(rule.rule_type))
    expected_value = lo
    span = alignment.span(rule.unit_indices)
    if span is None or haraka_ms <= 0:
        return RuleDiagnostic(
            rule_type=rule.rule_type, word=rule.word, start_ms=0, end_ms=0, status=Status.SKIPPED,
            feedback=f"{name} on '{rule.word}' could not be located in the audio.",
            expected_harakat=expected_value, expected_range=(lo, hi),
        )
    start, end = span
    dur_ms = (end - start) * 1000.0
    measured = dur_ms / haraka_ms
    status, score = classify_counts(measured, lo, hi, tolerance)
    target = _fmt_range(lo, hi)
    if status is Status.PASS:
        feedback = f"Excellent prolongation. Measured {dur_ms:.0f}ms ({measured:.1f} counts); target {target}."
    elif measured < lo:
        feedback = (
            f"{name} on '{rule.word}' was shortened: measured {measured:.1f} counts ({dur_ms:.0f}ms); "
            f"target was {target} counts. Hold the vowel longer."
        )
    else:
        feedback = (
            f"{name} on '{rule.word}' was over-extended: measured {measured:.1f} counts ({dur_ms:.0f}ms); "
            f"target was {target} counts."
        )
        if phrase_final and status is Status.FAIL:
            status, score = Status.WARNING, max(score, 0.6)
            feedback += " Some lengthening before a stop is natural; keep it proportionate."
    return RuleDiagnostic(
        rule_type=rule.rule_type, word=rule.word, start_ms=int(round(start * 1000)),
        end_ms=int(round(end * 1000)), status=status, feedback=feedback,
        expected_harakat=expected_value, expected_range=(lo, hi), measured_harakat=measured,
        score=score, metrics={"duration_ms": dur_ms},
    )
