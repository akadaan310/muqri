"""Hams vs. Jahr (breathy vs. voiced) via the Harmonics-to-Noise Ratio, and breath detection.

* Hams letters (فحثه شخص سكت) let the breath run: aperiodic noise, HNR < 3 dB.
* Jahr letters hold the breath back and are voiced: HNR > 12 dB.

The same aperiodicity measure identifies an **inhalation** in a pause: unvoiced, spectrally flat
noise well above the room noise floor. It is used by the Sakt validator (a sakt is taken without
breath) and by the fatigue detector (breath-driven Waqf al-Dharoori).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.acoustic.features import AcousticContext, short_time_rms, spectral_flatness
from app.models import RuleDiagnostic, RuleInstance, RuleType, Status
from app.tajweed_rules.base import EvalContext, ms, skipped

HAMS_HNR_MAX = 3.0
JAHR_HNR_MIN = 12.0
BREATH_MIN_MS = 120.0
BREATH_ABOVE_FLOOR_DB = 8.0
BREATH_BELOW_SPEECH_DB = 40.0
BREATH_FLATNESS_MIN = 0.15


@dataclass(slots=True)
class BreathEvidence:
    breath: bool
    breath_ms: float
    level_above_floor_db: float
    flatness: float


def detect_breath(ctx: AcousticContext, start_s: float, end_s: float) -> BreathEvidence:
    """Is there an inhalation in ``[start_s, end_s]``?

    Frames count as breath when they are unvoiced, audible (8 dB above the noise floor or within
    40 dB of the voice), and spectrally flat in 500–5000 Hz (turbulent airflow, not a tone or hum).
    """
    seg = ctx.audio.segment(start_s, end_s)
    if len(seg) < int(0.05 * ctx.sr):
        return BreathEvidence(False, 0.0, 0.0, float("nan"))
    db_all = 20 * np.log10(short_time_rms(ctx.x, ctx.sr, frame_ms=20.0, hop_ms=10.0) + 1e-12)
    floor, peak = float(np.percentile(db_all, 5)), float(np.percentile(db_all, 95))
    rms = short_time_rms(seg, ctx.sr, frame_ms=20.0, hop_ms=10.0)
    db = 20 * np.log10(rms + 1e-12)
    # Audible airflow: clearly above the room noise, or within 40 dB of the voice when a recording
    # has no true silence to estimate the floor from.
    loud = (db > floor + BREATH_ABOVE_FLOOR_DB) | ((db > peak - BREATH_BELOW_SPEECH_DB) & (db < peak - 8.0))
    f0 = ctx.f0_in(start_s, end_s)
    voiced_fraction = f0.size / max(1, int((end_s - start_s) / 0.01))
    flat = spectral_flatness(seg, ctx.sr, 500.0, 5000.0)
    breath_ms = float(loud.sum() * 10.0)
    is_breath = breath_ms >= BREATH_MIN_MS and voiced_fraction < 0.3 and flat >= BREATH_FLATNESS_MIN
    level = float(np.median(db[loud]) - floor) if loud.any() else 0.0
    return BreathEvidence(bool(is_breath), breath_ms, level, flat)


def consonant_window(ev: EvalContext, index: int) -> tuple[float, float] | None:
    """The consonantal part of a sakin letter's unit (its core, away from neighbouring vowels)."""
    span = ev.unit_span(index)
    if span is None:
        return None
    dur = span[1] - span[0]
    return span[0] + 0.15 * dur, span[1] - 0.15 * dur


FRICATIVES = frozenset("فحثهشخصسزذظغض")
VOICED_FRACTION_JAHR = 0.5
VOICED_FRACTION_HAMS = 0.3


def validate_hams_jahr(rule: RuleInstance, ev: EvalContext) -> RuleDiagnostic:
    """Jahr = the vocal folds vibrate; Hams = the breath runs through unvoiced.

    Sonorants are judged by HNR (Hams < 3 dB, Jahr > 12 dB). Fricatives are noisy by nature even
    when voiced (ز ذ ظ غ ض), so for them the voicing fraction of the frication is decisive.
    """
    letter = rule.letter or ""
    idx = rule.unit_indices[0]
    fricative = letter in FRICATIVES
    win = ev.frication_window(idx) if fricative else None
    win = win or consonant_window(ev, idx)
    if win is None or win[1] - win[0] < 0.03:
        return skipped(rule, f"{letter} in '{rule.word}' too short to measure.")
    hams = rule.rule_type is RuleType.HAMS
    hnr = ev.ctx.hnr(*win)
    voiced = ev.voicing_fraction(*win)
    metrics = {"voicing_fraction": voiced}
    if np.isfinite(hnr):
        metrics["hnr_db"] = hnr
    if fricative:
        if not np.isfinite(voiced):
            return skipped(rule, f"Voicing unavailable for {letter} in '{rule.word}'.", win)
        ok = voiced < VOICED_FRACTION_HAMS if hams else voiced >= VOICED_FRACTION_JAHR
        near = voiced < VOICED_FRACTION_JAHR if hams else voiced >= VOICED_FRACTION_HAMS
        value = f"voiced {voiced * 100:.0f}% of the frication"
    else:
        if not np.isfinite(hnr):
            return skipped(rule, f"HNR unavailable for {letter} in '{rule.word}'.", win)
        ok = hnr < HAMS_HNR_MAX if hams else hnr > JAHR_HNR_MIN
        near = hnr < HAMS_HNR_MAX + 6 if hams else hnr > JAHR_HNR_MIN - 6
        value = f"HNR {hnr:.1f} dB"
    status, score = (Status.PASS, 1.0) if ok else ((Status.WARNING, 0.6) if near else (Status.FAIL, 0.2))
    if hams:
        feedback = (f"{letter} breathy as required ({value})." if ok else
                    f"{letter} in '{rule.word}' was voiced ({value}); Hams needs the breath to flow.")
    else:
        feedback = (f"{letter} voiced with the breath held ({value})." if ok else
                    f"{letter} in '{rule.word}' lost its voicing ({value}); Jahr holds the breath back.")
    return RuleDiagnostic(
        rule_type=rule.rule_type, word=rule.word, start_ms=ms(win[0]), end_ms=ms(win[1]), status=status,
        feedback=feedback, score=score, metrics=metrics, letter=letter,
    )


VALIDATORS = {RuleType.HAMS: validate_hams_jahr, RuleType.JAHR: validate_hams_jahr}
