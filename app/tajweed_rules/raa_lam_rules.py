"""Conditional Tafkheem & Tarqeeq: isti'la letters, Raa, and the Lam of the Divine Name.

Which way each Raa/Lam must be read is decided by the parser (``TajweedParser._raa_verdict``):

* Tafkheem: raa with fathah/dammah; raa sakinah after fathah/dammah, after a temporary
  ('aaridha) kasrah, or after an original kasrah but before an isti'la letter (مِرْصَادًا).
* Tarqeeq: raa with kasrah; raa sakinah after an original kasrah; raa after yaa leen at a stop.
* Jawaz al-Wajhayn: both are valid (فِرْقٍ, مِصْر, ٱلْقِطْر); scored as PASS either way, and the
  realised variant is reported.
* Lam of Allah: heavy after fathah/dammah, light after kasrah.

The acoustic judgement is the F2 − F1 collapse against the reciter's own light reference (see
``app.sifaat.formants``).
"""

from __future__ import annotations

from app.models import RuleDiagnostic, RuleInstance, RuleType, Status
from app.sifaat.formants import judge_weight, unit_vowel, vowel_window
from app.tajweed_rules.base import EvalContext, ms, skipped


def validate_weight(rule: RuleInstance, ev: EvalContext) -> RuleDiagnostic:
    letter = rule.letter or ""
    either = rule.rule_type is RuleType.JAWAZ_WAJHAYN
    expect_heavy = rule.rule_type is RuleType.TAFKHEEM
    label = "Jawaz al-Wajhayn" if either else ("Tafkheem (heavy)" if expect_heavy else "Tarqeeq (light)")
    win = vowel_window(rule.unit_indices, ev.alignment)
    span = ev.span(rule.unit_indices)
    if win is None or span is None:
        return skipped(rule, f"{label} of {letter} in '{rule.word}' could not be located in the audio.")
    start, end = span
    fm = ev.ctx.formants(*win)
    if not fm.valid:
        return skipped(rule, f"Formants could not be tracked on {letter} in '{rule.word}' (unvoiced or too short).",
                       span)
    vowel = unit_vowel(ev.parsed, rule.unit_indices)
    ref_d = ev.weight_reference.distance(vowel)
    metrics: dict[str, float] = {"f1_hz": fm.f1, "f2_hz": fm.f2, "f2_minus_f1_hz": fm.f2 - fm.f1}
    if ref_d is not None:
        metrics["reference_f2_minus_f1_hz"] = ref_d
    ref_txt = f" vs. {ref_d:.0f} Hz on light letters" if ref_d is not None else ""

    if either:
        heavy_status, _, h = judge_weight(True, fm, ref_d, vowel)
        if h is not None:
            metrics["heaviness_index"] = h
        realised = "heavy" if heavy_status is Status.PASS else "light"
        return RuleDiagnostic(
            rule_type=rule.rule_type, word=rule.word, start_ms=ms(start), end_ms=ms(end), status=Status.PASS,
            feedback=f"{letter} in '{rule.word}' may be read heavy or light; it was read {realised} "
                     f"(F2−F1 {fm.f2 - fm.f1:.0f} Hz{ref_txt}).",
            score=1.0, metrics=metrics, letter=letter,
        )

    status, score, h = judge_weight(expect_heavy, fm, ref_d, vowel)
    if h is not None:
        metrics["heaviness_index"] = h
    if status is Status.SKIPPED:
        feedback = f"No light reference for this vowel; {label} of {letter} in '{rule.word}' not scored."
    elif status is Status.PASS:
        feedback = (f"{letter} in '{rule.word}' was correctly {'heavy' if expect_heavy else 'light'} "
                    f"(F2−F1 {fm.f2 - fm.f1:.0f} Hz{ref_txt}).")
    elif expect_heavy:
        feedback = (f"{letter} in '{rule.word}' sounded too light (F2−F1 {fm.f2 - fm.f1:.0f} Hz{ref_txt}). "
                    "Raise the back of the tongue to give it Tafkheem.")
    else:
        feedback = (f"{letter} in '{rule.word}' sounded too heavy (F2−F1 {fm.f2 - fm.f1:.0f} Hz{ref_txt}). "
                    "Keep it light (Tarqeeq).")
    if rule.detail and status is not Status.SKIPPED:
        feedback += f" [{rule.detail}]"
    return RuleDiagnostic(
        rule_type=rule.rule_type, word=rule.word, start_ms=ms(start), end_ms=ms(end), status=status,
        feedback=feedback, score=score if status is not Status.SKIPPED else None, metrics=metrics, letter=letter,
    )


VALIDATORS = {
    RuleType.TAFKHEEM: validate_weight,
    RuleType.TARQEEQ: validate_weight,
    RuleType.JAWAZ_WAJHAYN: validate_weight,
}
