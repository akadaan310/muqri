"""Shiddah / Tawassut / Rakhawah: how completely a sakin letter blocks the sound.

* Shiddah (أجد قط بكت): the sound is locked — a full occlusion (silent stretch) inside the letter.
* Rakhawah (all others except the tawassut letters): the sound runs — no occlusion at all.
* Tawassut (لن عمر): in between — the sound continues but is damped.

The physical test is the occlusion inside the sakin letter's core. The spec's duration ratios
(1.0 : 1.5 : 2.2 harakat) are reported as metrics and only ever raise a WARNING: they are a model
of the canonical sound rather than an established acoustic norm.
"""

from __future__ import annotations

from app.models import RuleDiagnostic, RuleInstance, RuleType, Status
from app.sifaat.hams_jahr import consonant_window
from app.tajweed_rules.base import EvalContext, ms, skipped, worst

RATIO_TOL = 0.6  # ±60 % of the expected ratio
MIN_OCCLUSION_MS = 15.0


def validate_sukoon_class(rule: RuleInstance, ev: EvalContext) -> RuleDiagnostic:
    idx = rule.unit_indices[0]
    span = ev.unit_span(idx)
    win = consonant_window(ev, idx)
    if span is None or win is None:
        return skipped(rule, f"{rule.letter} in '{rule.word}' could not be located in the audio.")
    haraka = ev.haraka_ms((span[0] + span[1]) / 2)
    ratio = (span[1] - span[0]) * 1000.0 / haraka
    expected = rule.expected_harakat[0] if rule.expected_harakat else 1.0
    runs = ev.silence_runs(*win, below_peak_db=25.0, min_ms=MIN_OCCLUSION_MS)
    occlusion = max(((b - a) * 1000 for a, b in runs), default=0.0)
    metrics = {"duration_ratio": ratio, "expected_ratio": expected, "occlusion_ms": occlusion}
    label = {RuleType.SHIDDAH: "Shiddah", RuleType.TAWASSUT: "Tawassut", RuleType.RAKHAWAH: "Rakhawah"}[rule.rule_type]
    if rule.rule_type is RuleType.SHIDDAH:
        ok = occlusion >= MIN_OCCLUSION_MS
        status = Status.PASS if ok else Status.WARNING
        feedback = (f"{rule.letter} locked the sound ({occlusion:.0f}ms occlusion)." if ok else
                    f"{rule.letter} in '{rule.word}' did not fully block the sound (Shiddah needs a complete closure).")
    else:
        ok = occlusion < 30.0
        status = Status.PASS if ok else (Status.FAIL if rule.rule_type is RuleType.RAKHAWAH else Status.WARNING)
        feedback = (f"{rule.letter} let the sound run ({label})." if ok else
                    f"{rule.letter} in '{rule.word}' was cut off ({occlusion:.0f}ms silence); {label} letters must "
                    "not stop the sound.")
    if abs(ratio - expected) > RATIO_TOL * expected:
        status = worst(status, Status.WARNING)
        feedback += f" Duration {ratio:.1f}× harakah (model {expected:g}×)."
    score = {Status.PASS: 1.0, Status.WARNING: 0.6, Status.FAIL: 0.2}[status]
    return RuleDiagnostic(
        rule_type=rule.rule_type, word=rule.word, start_ms=ms(span[0]), end_ms=ms(span[1]), status=status,
        feedback=feedback, score=score, metrics=metrics, letter=rule.letter,
    )


VALIDATORS = {RuleType.SHIDDAH: validate_sukoon_class, RuleType.TAWASSUT: validate_sukoon_class,
              RuleType.RAKHAWAH: validate_sukoon_class}
