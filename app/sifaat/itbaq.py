"""Itbaq vs. Infitah for ص ض ط ظ: the tongue body clamps against the palate.

Acoustically F2 collapses towards F1 (heaviness), F3 − F2 widens relative to light letters, and
energy above 3.5 kHz is attenuated. The heaviness verdict decides PASS/FAIL; the two secondary
cues can only downgrade a PASS to WARNING when both are clearly absent.
"""

from __future__ import annotations

import numpy as np

from app.models import RuleDiagnostic, RuleInstance, RuleType, Status
from app.sifaat.formants import judge_weight, measure_itbaq, unit_vowel, vowel_window
from app.tajweed_rules.base import EvalContext, ms, skipped


def validate_itbaq(rule: RuleInstance, ev: EvalContext) -> RuleDiagnostic:
    win = vowel_window(rule.unit_indices, ev.alignment)
    if win is None:
        return skipped(rule, f"{rule.letter} in '{rule.word}' could not be located in the audio.")
    fm = ev.ctx.formants(*win)
    if not fm.valid:
        return skipped(rule, f"Formants could not be tracked on {rule.letter} in '{rule.word}'.", win)
    vowel = unit_vowel(ev.parsed, rule.unit_indices)
    ref = ev.weight_reference
    status, score, h = judge_weight(True, fm, ref.distance(vowel), vowel)
    if status is Status.SKIPPED:
        return skipped(rule, f"No light reference for Itbaq of {rule.letter} in '{rule.word}'.", win)
    itbaq = measure_itbaq(fm, ev.ctx.audio.segment(*win), ev.ctx.sr, ref, vowel)
    metrics = {"f2_hz": fm.f2, "f3_minus_f2_hz": itbaq.f3_minus_f2}
    if h is not None:
        metrics["heaviness_index"] = h
    conv = itbaq.convergence
    if conv is not None and np.isfinite(conv):
        metrics["f3_f2_widening"] = conv
    if itbaq.hf_attenuation_db is not None and np.isfinite(itbaq.hf_attenuation_db):
        metrics["hf_attenuation_db"] = itbaq.hf_attenuation_db
    secondary = [conv is not None and np.isfinite(conv) and conv >= 0.1,
                 itbaq.hf_attenuation_db is not None and itbaq.hf_attenuation_db >= 2.0]
    known = [conv is not None and np.isfinite(conv), itbaq.hf_attenuation_db is not None]
    if status is Status.PASS and all(known) and not any(secondary):
        status, score = Status.WARNING, 0.6
    feedback = {
        Status.PASS: f"{rule.letter} clamped (Itbaq): F2 {fm.f2:.0f} Hz.",
        Status.WARNING: f"{rule.letter} in '{rule.word}' was only partly clamped; press the tongue body to the palate.",
        Status.FAIL: f"{rule.letter} in '{rule.word}' was open (Infitah) instead of clamped: F2 {fm.f2:.0f} Hz.",
    }[status]
    return RuleDiagnostic(
        rule_type=rule.rule_type, word=rule.word, start_ms=ms(win[0]), end_ms=ms(win[1]), status=status,
        feedback=feedback, score=score, metrics=metrics, letter=rule.letter,
    )


VALIDATORS = {RuleType.ITBAQ: validate_itbaq}
