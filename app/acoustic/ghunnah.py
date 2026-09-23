"""Ghunnah (nasalization) analyzer.

A nasal murmur is characterised by a strong low nasal formant (Fn ≈ 250 Hz) and an
anti-formant (spectral zero) that suppresses energy around 750–1100 Hz. The **Nasal Energy
Ratio** (NER) compares those two bands:

    NER_dB = 10·log10( E[150–400 Hz] / E[750–1100 Hz] )

Open oral vowels, whose F1 sits in 500–800 Hz, give NER near 0 dB. Because close vowels and
microphones shift the absolute value, the score uses the *nasal contrast*: NER of the Ghunnah
window minus the median NER of the reciter's own open (fathah) vowels, when those are
available. The rule is scored on both duration (≥ 2 harakat) and nasal contrast.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.acoustic.features import AcousticContext, band_energy_ratio_db
from app.acoustic.madd import classify_counts
from app.models import Alignment, RuleDiagnostic, RuleInstance, RuleType, Status

NASAL_BAND = (150.0, 400.0)
ZERO_BAND = (750.0, 1100.0)
NER_STRONG_DB = 8.0
NER_WEAK_DB = 3.0

RULE_NAMES: dict[RuleType, str] = {
    RuleType.GHUNNAH: "Ghunnah",
    RuleType.IKHFA: "Ikhfa",
    RuleType.IDGHAM_GHUNNAH: "Idgham with Ghunnah",
    RuleType.IQLAB: "Iqlab",
    RuleType.IKHFA_SHAFAWI: "Ikhfa Shafawi",
    RuleType.IDGHAM_SHAFAWI: "Idgham Shafawi",
}


@dataclass(slots=True)
class NasalMeasurement:
    ner_db: float
    low_freq_ratio: float
    f1_hz: float
    b1_hz: float


def nasal_energy_ratio(x: np.ndarray, sr: int) -> float:
    return band_energy_ratio_db(x, sr, NASAL_BAND, ZERO_BAND)


def measure_nasality(ctx: AcousticContext, start_s: float, end_s: float) -> NasalMeasurement:
    seg = ctx.audio.segment(start_s, end_s)
    ner = nasal_energy_ratio(seg, ctx.sr) if len(seg) >= 64 else float("nan")
    low = band_energy_ratio_db(seg, ctx.sr, (80.0, 500.0), (500.0, 4000.0)) if len(seg) >= 64 else float("nan")
    fm = ctx.formants(start_s, end_s)
    return NasalMeasurement(ner_db=ner, low_freq_ratio=low, f1_hz=fm.f1, b1_hz=fm.b1)


def ghunnah_window(rule: RuleInstance, alignment: Alignment) -> tuple[float, float] | None:
    """The time window in which the nasal resonance must be held."""
    first = alignment.units.get(rule.unit_indices[0])
    if rule.detail == "tanween" and first is not None:
        # The nasal of tanween is the tail of its carrier syllable.
        start = first.start_s + 0.5 * first.duration_s
        end = first.end_s
        for idx in rule.unit_indices[1:]:
            if idx in alignment.units:
                end = max(end, alignment.units[idx].end_s)
        return start, end
    return alignment.span(rule.unit_indices)


def oral_reference_ner(windows: list[tuple[float, float]], ctx: AcousticContext) -> float | None:
    """Median NER over oral open-vowel windows of the same recitation (speaker baseline)."""
    vals = []
    for start, end in windows:
        seg = ctx.audio.segment(start, end)
        if len(seg) >= 64:
            vals.append(nasal_energy_ratio(seg, ctx.sr))
    return float(np.median(vals)) if len(vals) >= 2 else None


def analyze_ghunnah(rule: RuleInstance, alignment: Alignment, ctx: AcousticContext, haraka_ms: float,
                    *, tolerance: float = 0.15, oral_reference_db: float | None = None) -> RuleDiagnostic:
    name = RULE_NAMES.get(rule.rule_type, str(rule.rule_type))
    lo, hi = rule.expected_harakat or (2.0, 2.0)
    window = ghunnah_window(rule, alignment)
    if window is None or haraka_ms <= 0:
        return RuleDiagnostic(
            rule_type=rule.rule_type, word=rule.word, start_ms=0, end_ms=0, status=Status.SKIPPED,
            feedback=f"{name} on '{rule.word}' could not be located in the audio.",
            expected_harakat=lo, letter=rule.letter,
        )
    start, end = window
    dur_ms = (end - start) * 1000.0
    counts = dur_ms / haraka_ms
    # A Ghunnah may be held somewhat longer than 2 counts without error; only penalise rushing
    # or gross over-extension.
    dur_status, dur_score = classify_counts(counts, lo, hi + 1.0, tolerance)
    nasal = measure_nasality(ctx, start, end)

    contrast = nasal.ner_db - oral_reference_db if oral_reference_db is not None else nasal.ner_db
    if not np.isfinite(contrast):
        nasal_score, nasal_ok = 0.5, None
    elif contrast >= NER_STRONG_DB:
        nasal_score, nasal_ok = 1.0, True
    elif contrast >= NER_WEAK_DB:
        nasal_score, nasal_ok = 0.6, None
    else:
        nasal_score, nasal_ok = 0.2, False

    score = 0.6 * dur_score + 0.4 * nasal_score
    if dur_status is Status.FAIL:
        status = Status.FAIL
    elif nasal_ok is False:
        status = Status.FAIL if dur_status is Status.WARNING else Status.WARNING
    else:
        status = dur_status

    parts: list[str] = []
    if dur_status is Status.PASS:
        parts.append(f"Held for {counts:.1f} counts ({dur_ms:.0f}ms).")
    elif counts < lo:
        parts.append(
            f"{name} was {'slightly ' if dur_status is Status.WARNING else ''}rushed ({counts:.1f} counts). "
            "Hold nasal resonance longer."
        )
    else:
        parts.append(f"{name} was over-extended ({counts:.1f} counts).")
    if nasal_ok is True:
        parts.append(f"Clear nasal resonance (NER {nasal.ner_db:+.1f} dB).")
    elif nasal_ok is False:
        parts.append(
            f"Weak nasalization (NER {nasal.ner_db:+.1f} dB): the sound was oral; let it resonate through the nose."
        )
    elif np.isfinite(nasal.ner_db):
        parts.append(f"Moderate nasal resonance (NER {nasal.ner_db:+.1f} dB).")

    metrics = {"duration_ms": dur_ms, "nasal_energy_ratio_db": nasal.ner_db,
               "low_frequency_ratio_db": nasal.low_freq_ratio}
    if oral_reference_db is not None:
        metrics["nasal_contrast_db"] = contrast
    if np.isfinite(nasal.f1_hz):
        metrics["f1_hz"] = nasal.f1_hz
    if np.isfinite(nasal.b1_hz):
        metrics["f1_bandwidth_hz"] = nasal.b1_hz
    return RuleDiagnostic(
        rule_type=rule.rule_type, word=rule.word, start_ms=int(round(start * 1000)),
        end_ms=int(round(end * 1000)), status=status, feedback=" ".join(parts),
        expected_harakat=lo, measured_harakat=counts, score=score,
        metrics={k: v for k, v in metrics.items() if np.isfinite(v)}, letter=rule.letter,
    )
