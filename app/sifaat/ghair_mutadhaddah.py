"""Sifaat without opposites (Ghair Mutadhaddah): Safir, Tafashhi, Istitaalah, Takreer.

* **Safir** (ص س ز): a whistle — a spike of energy above 5 kHz relative to the reciter's average.
* **Tafashhi** (ش): the air spreads — flat, broad noise across 2.5–6 kHz.
* **Istitaalah** (ض sakinah): the sound extends along the tongue's edge; it must not bounce like
  a Qalqalah release.
* **Takreer** (ر): a single tap. Rolling it (two or more taps), especially on raa mushaddadah,
  is the error Tajweed warns about.
"""

from __future__ import annotations

import numpy as np

from app.acoustic.features import band_power_db, short_time_rms, spectral_flatness
from app.models import RuleDiagnostic, RuleInstance, RuleType, Status
from app.tajweed_rules.base import EvalContext, ms, skipped
from app.tajweed_rules.qalqalah_engine import count_releases

SAFIR_SPIKE_DB = 6.0
TAFASHHI_FLATNESS = 0.3
TAP_DIP_DB = 6.0
TAP_MS = (6.0, 45.0)


def _hf_tilt(x: np.ndarray, sr: int) -> float:
    """Level above 5 kHz relative to 1–4 kHz (dB)."""
    return band_power_db(x, sr, 5000.0, sr / 2) - band_power_db(x, sr, 1000.0, 4000.0)


def _utterance_hf_tilt(ev: EvalContext) -> float:
    if "utterance_hf_tilt" not in ev.cache:
        ev.cache["utterance_hf_tilt"] = _hf_tilt(ev.ctx.audio.samples, ev.ctx.sr)
    return float(ev.cache["utterance_hf_tilt"])  # type: ignore[arg-type]


def validate_safir(rule: RuleInstance, ev: EvalContext) -> RuleDiagnostic:
    span = ev.unit_span(rule.unit_indices[0])
    if span is None:
        return skipped(rule, f"{rule.letter} in '{rule.word}' could not be located in the audio.")
    utt = _utterance_hf_tilt(ev)
    if not np.isfinite(utt) or utt < -45.0:
        return skipped(rule, "The recording has no energy above 5 kHz (low bitrate); Safir cannot be measured.", span)
    win = ev.frication_window(rule.unit_indices[0])
    if win is None:
        return skipped(rule, f"No frication found for {rule.letter} in '{rule.word}'.", span)
    span = win
    seg = ev.ctx.audio.segment(*win)
    spike = _hf_tilt(seg, ev.ctx.sr) - utt
    if spike >= SAFIR_SPIKE_DB:
        status, score, fb = Status.PASS, 1.0, f"{rule.letter} whistled clearly (+{spike:.1f} dB above 5 kHz)."
    elif spike >= 0.0:
        status, score, fb = Status.WARNING, 0.6, f"{rule.letter} in '{rule.word}' had a weak whistle (+{spike:.1f} dB)."
    else:
        status, score, fb = Status.FAIL, 0.2, f"{rule.letter} in '{rule.word}' lost its Safir whistle ({spike:.1f} dB)."
    return RuleDiagnostic(rule_type=rule.rule_type, word=rule.word, start_ms=ms(span[0]), end_ms=ms(span[1]),
                          status=status, feedback=fb, score=score, metrics={"safir_high_band_spike_db": spike},
                          letter=rule.letter)


def validate_tafashhi(rule: RuleInstance, ev: EvalContext) -> RuleDiagnostic:
    span = ev.frication_window(rule.unit_indices[0])
    if span is None:
        return skipped(rule, "No ش frication found near its aligned position.")
    seg = ev.ctx.audio.segment(*span)
    flat = spectral_flatness(seg, ev.ctx.sr, 2500.0, 6000.0)
    if not np.isfinite(flat):
        return skipped(rule, "ش too short to measure.", span)
    if flat >= TAFASHHI_FLATNESS:
        status, score, fb = Status.PASS, 1.0, f"ش spread across the palate (flatness {flat:.2f})."
    elif flat >= TAFASHHI_FLATNESS / 2:
        status, score, fb = Status.WARNING, 0.6, f"ش in '{rule.word}' was narrow (flatness {flat:.2f}); let it spread."
    else:
        status, score, fb = Status.FAIL, 0.2, f"ش in '{rule.word}' did not spread (flatness {flat:.2f})."
    return RuleDiagnostic(rule_type=rule.rule_type, word=rule.word, start_ms=ms(span[0]), end_ms=ms(span[1]),
                          status=status, feedback=fb, score=score, metrics={"tafashhi_spectral_flatness": flat},
                          letter=rule.letter)


def validate_istitaalah(rule: RuleInstance, ev: EvalContext) -> RuleDiagnostic:
    span = ev.unit_span(rule.unit_indices[0])
    if span is None:
        return skipped(rule, "ض could not be located in the audio.")
    haraka = ev.haraka_ms((span[0] + span[1]) / 2)
    ratio = (span[1] - span[0]) * 1000.0 / haraka
    releases = count_releases(ev.ctx.audio.segment(span[0], span[1] + 0.06), ev.ctx.sr)
    metrics = {"istitaalah_duration_ratio": ratio, "releases": float(releases)}
    if releases >= 1:
        status, score, fb = Status.WARNING, 0.5, (f"ض in '{rule.word}' bounced like a Qalqalah letter; extend it "
                                                  "along the tongue's edge instead.")
    elif ratio < 1.0:
        status, score, fb = Status.WARNING, 0.6, f"ض in '{rule.word}' was clipped ({ratio:.1f} harakat); let it extend."
    else:
        status, score, fb = Status.PASS, 1.0, f"ض extended without bouncing ({ratio:.1f} harakat)."
    return RuleDiagnostic(rule_type=rule.rule_type, word=rule.word, start_ms=ms(span[0]), end_ms=ms(span[1]),
                          status=status, feedback=fb, score=score, metrics=metrics, letter=rule.letter)


def count_taps(x: np.ndarray, sr: int) -> int:
    """Number of brief amplitude dips (tongue taps) in a raa segment."""
    rms = short_time_rms(x, sr, frame_ms=4.0, hop_ms=1.0)
    if rms.size < 10:
        return 0
    db = 20 * np.log10(rms + 1e-12)
    # Local reference: running maximum over ±20 ms.
    k = 20
    ref = np.array([db[max(0, i - k): i + k].max() for i in range(len(db))])
    dip = db < ref - TAP_DIP_DB
    taps, run = 0, 0
    for d in np.append(dip, False):
        if d:
            run += 1
        else:
            if TAP_MS[0] <= run <= TAP_MS[1]:
                taps += 1
            run = 0
    return taps


def validate_takreer(rule: RuleInstance, ev: EvalContext) -> RuleDiagnostic:
    span = ev.unit_span(rule.unit_indices[0])
    if span is None:
        return skipped(rule, "ر could not be located in the audio.")
    dur = span[1] - span[0]
    seg = ev.ctx.audio.segment(span[0], span[0] + (0.7 if "mushaddad" in rule.detail else 1.0) * dur)
    taps = count_taps(seg, ev.ctx.sr)
    metrics = {"taps": float(taps), "takreer_single_strike_flag": 1.0 if taps <= 1 else 0.0}
    if taps <= 1:
        status, score, fb = Status.PASS, 1.0, "ر struck once (no rolling)."
    elif "mushaddad" in rule.detail:
        status, score, fb = Status.FAIL, 0.2, f"ر in '{rule.word}' was rolled ({taps} taps); strike it once only."
    else:
        status, score, fb = Status.WARNING, 0.6, f"ر in '{rule.word}' was slightly rolled ({taps} taps)."
    return RuleDiagnostic(rule_type=rule.rule_type, word=rule.word, start_ms=ms(span[0]), end_ms=ms(span[1]),
                          status=status, feedback=fb, score=score, metrics=metrics, letter=rule.letter)


VALIDATORS = {
    RuleType.SAFIR: validate_safir,
    RuleType.TAFASHHI: validate_tafashhi,
    RuleType.ISTITAALAH: validate_istitaalah,
    RuleType.TAKREER: validate_takreer,
}
