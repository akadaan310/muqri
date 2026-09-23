"""Tafkheem / Tarqeeq (heavy vs. light articulation) analyzer.

Velarisation/pharyngealisation of heavy letters retracts the tongue root, which lowers the
second formant of the adjacent vowel (F2 ≈ 1000–1400 Hz after heavy letters versus > 1700 Hz
after light ones for an open /a/) and raises F1, collapsing the **F2 − F1 distance**.

Absolute formants depend on the speaker's vocal tract, so the analyzer first builds a per-vowel
**light reference** from ordinary coronal/dorsal letters of the same recitation (excluding
gutturals, labials, raa, lam and any letter adjacent to a heavy one) and scores each rule by the
relative collapse of F2 − F1:

    H = 1 − (F2 − F1) / (F2_ref − F1_ref)

Absolute F2 thresholds are used only for fathah when no reference is available.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.acoustic.features import AcousticContext, FormantEstimate
from app.models import Alignment, RuleDiagnostic, RuleInstance, RuleType, Status, Vowel
from app.tajweed_rules import HEAVY_LETTERS, ParsedText

HEAVY_F2_MAX_HZ = 1400.0
LIGHT_F2_MIN_HZ = 1700.0
HEAVY_INDEX_PASS = 0.15
HEAVY_INDEX_WARN = 0.07
LIGHT_INDEX_PASS = 0.10
LIGHT_INDEX_FAIL = 0.20
VOWEL_START, VOWEL_END = 0.15, 0.75  # fraction of a CV unit used as the vowel core
MIN_REFERENCE_TOKENS = 2

# Gutturals, labials, raa, lam and hamza have their own F2 effects and make poor references.
_NON_REFERENCE = HEAVY_LETTERS | frozenset("رلعحهءأإؤئبمفو")
_F2_LOWERING = HEAVY_LETTERS | frozenset("رعح")


@dataclass(slots=True)
class WeightReference:
    f1_by_vowel: dict[Vowel, float] = field(default_factory=dict)
    f2_by_vowel: dict[Vowel, float] = field(default_factory=dict)
    counts: dict[Vowel, int] = field(default_factory=dict)

    def distance(self, vowel: Vowel | None) -> float | None:
        if vowel is None or vowel not in self.f2_by_vowel:
            return None
        return self.f2_by_vowel[vowel] - self.f1_by_vowel[vowel]


def vowel_window(unit_indices: list[int], alignment: Alignment) -> tuple[float, float] | None:
    """Core of the vowel carried by the first unit (and any madd letter after it).

    Aligned CV units start at the consonant release, so the vowel fills most of the unit; its
    last quarter is coarticulated with the following letter and is excluded.
    """
    present = [alignment.units[i] for i in unit_indices if i in alignment.units]
    if not present or present[0].unit_index != unit_indices[0]:
        return None
    first, last = present[0], present[-1]
    start = first.start_s + VOWEL_START * first.duration_s
    end = last.end_s - (1.0 - VOWEL_END) * last.duration_s
    return (start, end) if end > start else None


def _near_heavy(parsed: ParsedText, index: int) -> bool:
    """True if a heavy, pharyngeal or raa letter is adjacent to ``index`` (their F2 lowering spreads)."""
    for j in (index - 1, index + 1, index + 2):
        if 0 <= j < len(parsed.units):
            other = parsed.units[j]
            if other.pronounced and other.char in _F2_LOWERING:
                return True
    return False


def build_reference(parsed: ParsedText, alignment: Alignment, ctx: AcousticContext) -> WeightReference:
    """Median F1/F2 per short vowel over light, non-coarticulated consonants."""
    samples: dict[Vowel, list[tuple[float, float]]] = {}
    for u in parsed.units:
        if not u.pronounced or u.vowel is None or u.madd_letter or u.char in _NON_REFERENCE:
            continue
        if _near_heavy(parsed, u.index):
            continue
        win = vowel_window([u.index], alignment)
        if win is None:
            continue
        fm = ctx.formants(*win)
        if fm.valid:
            samples.setdefault(u.vowel, []).append((fm.f1, fm.f2))
    ref = WeightReference()
    for vowel, vals in samples.items():
        if len(vals) >= MIN_REFERENCE_TOKENS:
            arr = np.asarray(vals)
            ref.f1_by_vowel[vowel] = float(np.median(arr[:, 0]))
            ref.f2_by_vowel[vowel] = float(np.median(arr[:, 1]))
            ref.counts[vowel] = len(vals)
    return ref


def _unit_vowel(parsed: ParsedText, rule: RuleInstance) -> Vowel | None:
    unit = parsed.units[rule.unit_indices[0]]
    if unit.vowel is not None:
        return unit.vowel
    if len(rule.unit_indices) > 1:
        madd = parsed.units[rule.unit_indices[1]]
        return {"ا": Vowel.FATHA, "و": Vowel.DAMMA, "ي": Vowel.KASRA}.get(madd.char)
    return None


def judge_weight(expect_heavy: bool, fm: FormantEstimate, ref_distance: float | None,
                 vowel: Vowel | None) -> tuple[Status, float, float | None]:
    """Return (status, score, heaviness_index)."""
    if ref_distance is not None and ref_distance > 100.0:
        h = 1.0 - (fm.f2 - fm.f1) / ref_distance
        if expect_heavy:
            if h >= HEAVY_INDEX_PASS:
                return Status.PASS, 1.0, h
            if h >= HEAVY_INDEX_WARN:
                return Status.WARNING, 0.6, h
            return Status.FAIL, 0.2, h
        if h < LIGHT_INDEX_PASS:
            return Status.PASS, 1.0, h
        if h < LIGHT_INDEX_FAIL:
            return Status.WARNING, 0.6, h
        return Status.FAIL, 0.2, h
    if vowel == Vowel.FATHA:
        if expect_heavy:
            if fm.f2 <= HEAVY_F2_MAX_HZ:
                return Status.PASS, 1.0, None
            if fm.f2 <= LIGHT_F2_MIN_HZ:
                return Status.WARNING, 0.6, None
            return Status.FAIL, 0.2, None
        if fm.f2 >= LIGHT_F2_MIN_HZ:
            return Status.PASS, 1.0, None
        if fm.f2 > HEAVY_F2_MAX_HZ:
            return Status.WARNING, 0.6, None
        return Status.FAIL, 0.2, None
    return Status.SKIPPED, 0.0, None


def analyze_weight(rule: RuleInstance, parsed: ParsedText, alignment: Alignment, ctx: AcousticContext,
                   reference: WeightReference) -> RuleDiagnostic:
    expect_heavy = rule.rule_type == RuleType.TAFKHEEM
    letter = rule.letter or ""
    label = "Tafkheem (heavy)" if expect_heavy else "Tarqeeq (light)"
    win = vowel_window(rule.unit_indices, alignment)
    span = alignment.span(rule.unit_indices)
    if win is None or span is None:
        return RuleDiagnostic(
            rule_type=rule.rule_type, word=rule.word, start_ms=0, end_ms=0, status=Status.SKIPPED,
            feedback=f"{label} of {letter} in '{rule.word}' could not be located in the audio.", letter=letter,
        )
    start, end = span
    fm = ctx.formants(*win)
    if not fm.valid:
        return RuleDiagnostic(
            rule_type=rule.rule_type, word=rule.word, start_ms=int(start * 1000), end_ms=int(end * 1000),
            status=Status.SKIPPED, letter=letter,
            feedback=f"Formants could not be tracked on {letter} in '{rule.word}' (unvoiced or too short).",
        )
    vowel = _unit_vowel(parsed, rule)
    ref_d = reference.distance(vowel)
    status, score, h = judge_weight(expect_heavy, fm, ref_d, vowel)

    metrics: dict[str, float] = {"f1_hz": fm.f1, "f2_hz": fm.f2, "f2_minus_f1_hz": fm.f2 - fm.f1}
    if ref_d is not None:
        metrics["reference_f2_minus_f1_hz"] = ref_d
    if h is not None:
        metrics["heaviness_index"] = h
    ref_txt = f" vs. {ref_d:.0f} Hz on light letters" if ref_d is not None else ""
    if status is Status.SKIPPED:
        feedback = f"No light reference for this vowel; {label} of {letter} in '{rule.word}' not scored."
    elif status is Status.PASS:
        feedback = (
            f"{letter} in '{rule.word}' was correctly {'heavy' if expect_heavy else 'light'} "
            f"(F2−F1 {fm.f2 - fm.f1:.0f} Hz{ref_txt})."
        )
    elif expect_heavy:
        feedback = (
            f"{letter} in '{rule.word}' sounded too light (F2−F1 {fm.f2 - fm.f1:.0f} Hz{ref_txt}). "
            "Raise the back of the tongue to give it Tafkheem."
        )
    else:
        feedback = (
            f"{letter} in '{rule.word}' sounded too heavy (F2−F1 {fm.f2 - fm.f1:.0f} Hz{ref_txt}). "
            "Keep it light (Tarqeeq)."
        )
    return RuleDiagnostic(
        rule_type=rule.rule_type, word=rule.word, start_ms=int(round(start * 1000)),
        end_ms=int(round(end * 1000)), status=status, feedback=feedback,
        score=score if status is not Status.SKIPPED else None, metrics=metrics, letter=letter,
    )
