"""Hamzat al-Wasl dropping in continuous reading, and the saktāt of Hafs.

* **Hamzat al-Wasl** is silent in wasl: the previous word must glide straight into the next letter.
  An inserted glottal stop shows up as a short silent gap (15–250 ms) followed by an abrupt onset.
  When the following letter is itself a stop consonant its closure is naturally silent, so only a
  real pause is flagged there.
* **Sakt** (عِوَجَا ۜ قَيِّمًا، مِن مَّرْقَدِنَا ۜ هَٰذَا، مَنْ ۜ رَاقٍ، بَلْ ۜ رَانَ): a silence of
  200–400 ms *without taking a breath*. At مَالِيَهْ ۜ هَلَكَ the sakt is optional (idghaam is also
  valid).
"""

from __future__ import annotations

import numpy as np

from app.models import RuleDiagnostic, RuleInstance, RuleType, Status
from app.sifaat.hams_jahr import detect_breath
from app.tajweed_rules.base import EvalContext, ms, skipped

SAKT_MS = (200.0, 400.0)
SAKT_WARN_MS = (100.0, 800.0)
GLOTTAL_GAP_MS = (15.0, 250.0)
ONSET_RISE_DB = 15.0
STOP_LETTERS = frozenset("بتدطقكجء")


def validate_hamzat_wasl(rule: RuleInstance, ev: EvalContext) -> RuleDiagnostic:
    nxt_idx = rule.unit_indices[-1]
    nxt = ev.unit_span(nxt_idx)
    if nxt is None:
        return skipped(rule, f"Hamzat al-wasl in '{rule.word}' could not be located in the audio.")
    boundary = nxt[0]
    window = (boundary - 0.12, boundary + 0.06)
    runs = ev.silence_runs(*window, below_peak_db=30.0, min_ms=15.0)
    gap = max(((b - a) * 1000 for a, b in runs), default=0.0)
    next_is_stop = ev.parsed.units[nxt_idx].char in STOP_LETTERS
    metrics = {"gap_ms": gap}
    if gap > GLOTTAL_GAP_MS[1]:
        status, score = Status.WARNING, 0.5
        feedback = (f"A {gap:.0f}ms pause before '{rule.word}': if you stop here, restart with the hamza; in wasl "
                    "the hamzat al-wasl must be dropped.")
    elif gap >= GLOTTAL_GAP_MS[0] and not next_is_stop:
        db, _ = ev.rms_db
        end_run = max(runs, key=lambda r: r[1] - r[0])[1]
        k = int(end_run * 1000)
        rise = float(np.max(db[k: k + 20]) - np.min(db[max(0, k - 10): k])) if k + 20 <= len(db) else 0.0
        metrics["onset_rise_db"] = rise
        if rise >= ONSET_RISE_DB:
            status, score = Status.FAIL, 0.2
            feedback = (f"A glottal stop was inserted before '{rule.word}' ({gap:.0f}ms gap, abrupt onset): drop the "
                        "hamzat al-wasl in continuous reading.")
        else:
            status, score = Status.PASS, 1.0
            feedback = "Hamzat al-wasl dropped; words joined smoothly."
    else:
        status, score = Status.PASS, 1.0
        feedback = "Hamzat al-wasl dropped; words joined smoothly."
    return RuleDiagnostic(
        rule_type=rule.rule_type, word=rule.word, start_ms=ms(window[0]), end_ms=ms(window[1]), status=status,
        feedback=feedback, score=score, metrics=metrics,
    )


def validate_sakt(rule: RuleInstance, ev: EvalContext) -> RuleDiagnostic:
    last, nxt = (ev.unit_span(i) for i in rule.unit_indices[:2])
    if last is None or nxt is None:
        return skipped(rule, f"Sakt after '{rule.word}' could not be located in the audio.")
    runs = ev.silence_runs(last[0], nxt[0] + 0.1, below_peak_db=30.0, min_ms=30.0)
    optional = "optional" in rule.detail
    if not runs:
        if optional:
            return RuleDiagnostic(
                rule_type=rule.rule_type, word=rule.word, start_ms=ms(last[0]), end_ms=ms(nxt[0]),
                status=Status.PASS, score=1.0, feedback="Read with idghaam instead of sakt (both are valid here).",
            )
        return RuleDiagnostic(
            rule_type=rule.rule_type, word=rule.word, start_ms=ms(last[0]), end_ms=ms(nxt[0]), status=Status.FAIL,
            score=0.1, feedback=f"No sakt after '{rule.word}': Hafs requires a brief breathless pause here.",
            metrics={"sakt_silence_duration_ms": 0.0},
        )
    a, b = max(runs, key=lambda r: r[1] - r[0])
    dur = (b - a) * 1000
    breath = detect_breath(ev.ctx, a, b)
    metrics = {"sakt_silence_duration_ms": dur, "breath_ms": breath.breath_ms}
    if breath.breath:
        status, score = Status.FAIL, 0.2
        feedback = f"A breath was taken during the sakt after '{rule.word}' ({dur:.0f}ms); sakt is without breath."
    elif SAKT_MS[0] <= dur <= SAKT_MS[1]:
        status, score = Status.PASS, 1.0
        feedback = f"Sakt of {dur:.0f}ms without breath after '{rule.word}'."
    elif SAKT_WARN_MS[0] <= dur <= SAKT_WARN_MS[1]:
        status, score = Status.WARNING, 0.6
        feedback = (f"Sakt after '{rule.word}' lasted {dur:.0f}ms; aim for {SAKT_MS[0]:.0f}–{SAKT_MS[1]:.0f}ms "
                    "without breath.")
    elif dur < SAKT_WARN_MS[0]:
        status, score = Status.FAIL, 0.2
        feedback = f"The sakt after '{rule.word}' was too short ({dur:.0f}ms) to be heard."
    else:
        status, score = Status.FAIL, 0.3
        feedback = f"A full stop ({dur:.0f}ms) was made instead of a brief sakt after '{rule.word}'."
    return RuleDiagnostic(
        rule_type=rule.rule_type, word=rule.word, start_ms=ms(a), end_ms=ms(b), status=status, feedback=feedback,
        score=score, metrics=metrics,
    )


VALIDATORS = {RuleType.HAMZAT_WASL: validate_hamzat_wasl, RuleType.SAKT: validate_sakt}
