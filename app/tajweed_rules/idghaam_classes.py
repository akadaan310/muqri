"""Idghaam classes of letters other than noon/meem: Mithlayn, Mutajanisayn, Mutaqaribayn.

* **Kamil** (complete): the first letter disappears into the second, which is read mushaddad. For
  stop consonants this means one closure and a single release (ٱضْرِب بِّعَصَاكَ, قَد تَّبَيَّنَ,
  نَخْلُقكُّم); a second release means the first letter was pronounced separately.
* **Naqis** (incomplete): ط into ت (أَحَطتُ, بَسَطتَ, فَرَّطتُمْ). The ط keeps its itbaq/isti'la, so
  the preceding vowel shows F2 lowering, but it is not released (no qalqalah) before the ت.
"""

from __future__ import annotations

from app.models import RuleDiagnostic, RuleInstance, RuleType, Status
from app.sifaat.formants import judge_weight, unit_vowel
from app.tajweed_rules.base import EvalContext, ms, name_of, skipped, worst
from app.tajweed_rules.qalqalah_engine import count_releases

STOP_LETTERS = frozenset("بتدطقكج")
MIN_GEMINATE_HARAKAT = 0.9


def validate_kamil(rule: RuleInstance, ev: EvalContext) -> RuleDiagnostic:
    name = name_of(rule)
    target = rule.unit_indices[-1]
    span = ev.unit_span(target)
    if span is None:
        return skipped(rule, f"{name} on '{rule.word}' could not be located in the audio.")
    start, end = span
    haraka = ev.haraka_ms((start + end) / 2)
    counts = (end - start) * 1000.0 / haraka
    tgt_char = ev.parsed.units[target].char
    metrics: dict[str, float] = {"merged_duration_ms": (end - start) * 1000.0, "haraka_ms": haraka}
    status, score = Status.PASS, 1.0
    feedback = f"{rule.letter} merged completely into the mushaddad {tgt_char} ({counts:.1f} harakat)."
    if tgt_char in STOP_LETTERS:
        seg = ev.ctx.audio.segment(max(0.0, start - 0.1), end)
        releases = count_releases(seg, ev.ctx.sr)
        metrics["releases"] = float(releases)
        if releases >= 2:
            status, score = Status.FAIL, 0.2
            feedback = (f"{rule.letter} was released separately before {tgt_char} ({releases} releases): "
                        "complete the idghaam into a single mushaddad letter.")
    if counts < MIN_GEMINATE_HARAKAT and status is Status.PASS:
        status, score = Status.WARNING, 0.6
        feedback = (f"The merged {tgt_char} was short ({counts:.1f} harakat); the idghaam should produce a full "
                    "shaddah.")
    return RuleDiagnostic(
        rule_type=rule.rule_type, word=rule.word, start_ms=ms(start), end_ms=ms(end), status=status,
        feedback=feedback, measured_harakat=counts, score=score, metrics=metrics, letter=rule.letter,
    )


def validate_naqis(rule: RuleInstance, ev: EvalContext) -> RuleDiagnostic:
    taa, t = rule.unit_indices[0], rule.unit_indices[-1]
    span = ev.span([taa, t])
    if span is None:
        return skipped(rule, f"Idghaam Naqis on '{rule.word}' could not be located in the audio.")
    start, end = span
    metrics: dict[str, float] = {}
    status, score = Status.PASS, 1.0
    parts: list[str] = []
    # 1) No separate release of the ط.
    releases = count_releases(ev.ctx.audio.segment(start, end), ev.ctx.sr)
    metrics["releases"] = float(releases)
    if releases >= 2:
        status, score = Status.FAIL, 0.2
        parts.append("The ط was released on its own (qalqalah) before ت; hold it into the ت.")
    # 2) Itbaq retained: the vowel before ط carries the heavy F2 lowering.
    prev = next((u for u in reversed(ev.parsed.units[:taa]) if u.pronounced), None)
    prev_span = ev.unit_span(prev.index) if prev is not None else None
    if prev is not None and prev_span is not None:
        dur = prev_span[1] - prev_span[0]
        fm = ev.ctx.formants(prev_span[0] + 0.5 * dur, prev_span[1])
        if fm.valid:
            vowel = unit_vowel(ev.parsed, [prev.index])
            verdict, _, h = judge_weight(True, fm, ev.weight_reference.distance(vowel), vowel)
            metrics["f2_before_taa_hz"] = fm.f2
            if h is not None:
                metrics["itbaq_retention_index"] = h
            if verdict is Status.FAIL:
                status = worst(status, Status.WARNING)
                score = min(score, 0.6)
                parts.append(f"The itbaq of ط was lost (F2 {fm.f2:.0f} Hz before it): keep the tongue raised.")
    if not parts:
        parts.append("ط assimilated into ت while keeping its itbaq (idghaam naqis).")
    return RuleDiagnostic(
        rule_type=rule.rule_type, word=rule.word, start_ms=ms(start), end_ms=ms(end), status=status,
        feedback=" ".join(parts), score=score, metrics=metrics, letter=rule.letter,
    )


def validate_idgham_class(rule: RuleInstance, ev: EvalContext) -> RuleDiagnostic:
    return validate_naqis(rule, ev) if rule.detail == "naqis" else validate_kamil(rule, ev)


VALIDATORS = {
    RuleType.IDGHAM_MITHLAYN: validate_idgham_class,
    RuleType.IDGHAM_MUTAJANISAYN: validate_idgham_class,
    RuleType.IDGHAM_MUTAQARIBAYN: validate_idgham_class,
}
