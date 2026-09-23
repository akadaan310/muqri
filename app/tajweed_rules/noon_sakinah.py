"""Noon Sakinah & Tanween (Izhaar Halqi, Idghaam with/without Ghunnah, Iqlab, Ikhfa' Haqiqi) and
Ghunnah Mushaddadah.

All timings are fractions of the local harakah. Nasality is the Nasal Energy Ratio contrast
against the reciter's own open vowels (see ``app.sifaat.formants``).
"""

from __future__ import annotations

import numpy as np

from app.models import RuleDiagnostic, RuleInstance, RuleType, Status
from app.sifaat.formants import measure_nasality, nasal_verdict
from app.tajweed_rules.base import EvalContext, band_status, ms, name_of, skipped, worst

DURATION_TOL = 0.25  # harakat
IZHAR_MAX = 1.3  # harakat: a clear (non-nasalised) noon/meem
IZHAR_WARN = 1.6
HEAVY_IKHFA_F2_MAX = 1300.0
LIGHT_IKHFA_F2_MIN = 1700.0
HEAVY_IKHFA_TARGETS = frozenset("صضطظق")
NO_GHUNNAH_MAX_CONTRAST_DB = 3.0
NO_GHUNNAH_FAIL_CONTRAST_DB = 8.0


def nasal_window(rule: RuleInstance, ev: EvalContext, *, include_target: bool) -> tuple[float, float] | None:
    """Window in which the nasal sound must be held.

    For tanween the nasal is the tail of the carrier syllable. For Idghaam the (assimilated) noon's
    nasal is carried by the target letter, so the target span is included.
    """
    first = ev.alignment.units.get(rule.unit_indices[0])
    targets = rule.unit_indices[1:] if include_target else []
    if "tanween" in rule.detail and first is not None:
        start, end = first.start_s + 0.5 * first.duration_s, first.end_s
    elif first is not None:
        start, end = first.start_s, first.end_s
    else:
        span = ev.span(targets) if targets else None
        if span is None:
            return None
        start, end = span
    for idx in targets:
        u = ev.alignment.units.get(idx)
        if u is not None:
            end = max(end, u.end_s)
    return start, end


def nasal_hold(rule: RuleInstance, ev: EvalContext, window: tuple[float, float], lo: float, hi: float,
               extra: dict[str, float] | None = None) -> RuleDiagnostic:
    """Score a held Ghunnah: duration in [lo, hi] harakat and audible nasal resonance."""
    name = name_of(rule)
    start, end = ev.voiced_extent(*window)
    haraka = ev.haraka_ms((start + end) / 2)
    dur_ms = (end - start) * 1000.0
    counts = dur_ms / haraka
    dur_status, dur_score = band_status(counts, lo, hi, DURATION_TOL)
    nasal = measure_nasality(ev.ctx, start, end)
    ref = ev.oral_ner_reference
    contrast = nasal.ner_db - ref if ref is not None else nasal.ner_db
    nasal_ok, nasal_score = nasal_verdict(contrast)

    if dur_status is Status.FAIL:
        status = Status.FAIL
    elif nasal_ok is False:
        status = Status.FAIL if dur_status is Status.WARNING else Status.WARNING
    else:
        status = dur_status
    parts: list[str] = []
    if dur_status is Status.PASS:
        parts.append(f"{name} held for {counts:.1f} counts ({dur_ms:.0f}ms).")
    elif counts < lo:
        parts.append(f"{name} was {'slightly ' if dur_status is Status.WARNING else ''}rushed ({counts:.1f} counts). "
                     "Hold nasal resonance longer.")
    else:
        parts.append(f"{name} was over-extended ({counts:.1f} counts).")
    if nasal_ok is True:
        parts.append(f"Clear nasal resonance (NER {nasal.ner_db:+.1f} dB).")
    elif nasal_ok is False:
        parts.append(f"Weak nasalization (NER {nasal.ner_db:+.1f} dB): the sound was oral; let it resonate "
                     "through the nose.")
    elif np.isfinite(nasal.ner_db):
        parts.append(f"Moderate nasal resonance (NER {nasal.ner_db:+.1f} dB).")
    metrics = {"duration_ms": dur_ms, "haraka_ms": haraka, "nasal_energy_ratio_db": nasal.ner_db,
               "low_frequency_ratio_db": nasal.low_freq_ratio, **(extra or {})}
    if ref is not None:
        metrics["nasal_contrast_db"] = contrast
    for key, val in (("f1_hz", nasal.f1_hz), ("f2_hz", nasal.f2_hz), ("f1_bandwidth_hz", nasal.b1_hz)):
        if np.isfinite(val):
            metrics[key] = val
    return RuleDiagnostic(
        rule_type=rule.rule_type, word=rule.word, start_ms=ms(start), end_ms=ms(end), status=status,
        feedback=" ".join(parts), expected_harakat=lo, expected_range=(lo, hi), measured_harakat=counts,
        score=0.6 * dur_score + 0.4 * nasal_score, metrics={k: float(v) for k, v in metrics.items() if np.isfinite(v)},
        letter=rule.letter,
    )


def labial_closure_ms(ev: EvalContext, window: tuple[float, float]) -> float:
    """Longest fully silent stretch inside a nasal window (lips clenched instead of humming)."""
    runs = ev.silence_runs(window[0], window[1], below_peak_db=35.0, min_ms=20.0)
    return max((b - a) * 1000 for a, b in runs) if runs else 0.0


# --------------------------------------------------------------------------- validators
def validate_ghunnah_mushaddadah(rule: RuleInstance, ev: EvalContext) -> RuleDiagnostic:
    window = nasal_window(rule, ev, include_target=False)
    if window is None:
        return skipped(rule, f"Ghunnah on '{rule.word}' could not be located in the audio.")
    return nasal_hold(rule, ev, window, 2.0, 2.5)


def validate_idgham_ghunnah(rule: RuleInstance, ev: EvalContext) -> RuleDiagnostic:
    window = nasal_window(rule, ev, include_target=True)
    if window is None:
        return skipped(rule, f"Idghaam on '{rule.word}' could not be located in the audio.")
    diag = nasal_hold(rule, ev, window, 2.0, 2.5)
    if "naqis" in rule.detail and diag.status is Status.PASS:
        diag.feedback += " Merged into و/ي with the ghunnah retained (idghaam naqis)."
    return diag


def validate_iqlab_or_ikhfa_shafawi(rule: RuleInstance, ev: EvalContext) -> RuleDiagnostic:
    window = nasal_window(rule, ev, include_target=False)
    if window is None:
        return skipped(rule, f"{name_of(rule)} on '{rule.word}' could not be located in the audio.")
    closure = labial_closure_ms(ev, window)
    diag = nasal_hold(rule, ev, window, 2.0, 2.0, {"labial_closure_ms": closure})
    if closure > 40.0 and diag.status is Status.PASS:
        diag.status = Status.WARNING
        diag.score = min(diag.score or 1.0, 0.7)
        diag.feedback += (f" The lips closed fully for {closure:.0f}ms before ب; keep a light labial contact "
                          "with the nasal hum.")
    elif diag.status is Status.PASS and rule.rule_type is RuleType.IQLAB:
        diag.feedback += " Noon converted to a light meem before ب."
    return diag


def validate_ikhfa(rule: RuleInstance, ev: EvalContext) -> RuleDiagnostic:
    window = nasal_window(rule, ev, include_target=False)
    if window is None:
        return skipped(rule, f"Ikhfa' on '{rule.word}' could not be located in the audio.")
    heavy = "heavy" in rule.detail or (rule.letter or "") in HEAVY_IKHFA_TARGETS
    # Tongue anticipation is strongest in the second half of the nasal hold.
    mid = (window[0] + window[1]) / 2
    fm = ev.ctx.formants(mid, window[1])
    extra: dict[str, float] = {}
    anticipation: Status = Status.SKIPPED
    if fm.valid:
        extra["anticipation_f2_hz"] = fm.f2
        if heavy:
            anticipation = Status.PASS if fm.f2 < HEAVY_IKHFA_F2_MAX else Status.WARNING
        else:
            anticipation = Status.PASS if fm.f2 > LIGHT_IKHFA_F2_MIN else Status.WARNING
        extra["ikhfa_formant_anticipation_score"] = 1.0 if anticipation is Status.PASS else 0.0
    diag = nasal_hold(rule, ev, window, 2.0, 2.0, extra)
    if anticipation is Status.WARNING:
        diag.status = worst(diag.status, Status.WARNING)
        diag.score = (diag.score or 0.0) * 0.8
        target = f"toward the heavy {rule.letter} (F2 < {HEAVY_IKHFA_F2_MAX:.0f} Hz)" if heavy else \
            f"toward the light {rule.letter} (F2 > {LIGHT_IKHFA_F2_MIN:.0f} Hz)"
        diag.feedback += f" The tongue did not anticipate the makhraj {target}: F2 was {fm.f2:.0f} Hz."
    elif anticipation is Status.PASS:
        diag.feedback += f" Tongue anticipated the {'heavy' if heavy else 'light'} {rule.letter} (F2 {fm.f2:.0f} Hz)."
    return diag


def clear_letter(rule: RuleInstance, ev: EvalContext, what: str) -> RuleDiagnostic:
    """Izhaar: the noon/meem is short (≤ 1.3 harakat) and released without an inserted pause."""
    window = nasal_window(rule, ev, include_target=False)
    if window is None:
        return skipped(rule, f"{name_of(rule)} on '{rule.word}' could not be located in the audio.")
    start, end = ev.voiced_extent(*window)
    haraka = ev.haraka_ms((start + end) / 2)
    counts = (end - start) * 1000.0 / haraka
    if counts <= IZHAR_MAX:
        status, score = Status.PASS, 1.0
        feedback = f"{what} pronounced clearly ({counts:.1f} counts, no ghunnah prolongation)."
    elif counts <= IZHAR_WARN and rule.rule_type is RuleType.IZHAR_HALQI:
        status, score = Status.WARNING, 0.6
        feedback = f"{what} was slightly prolonged ({counts:.1f} counts); Izhaar needs a crisp, unheld sound."
    else:
        status, score = Status.FAIL, 0.2
        feedback = (f"{what} was held for {counts:.1f} counts (> {IZHAR_MAX:g}): this is Ikhfa'/Ghunnah where Izhaar "
                    "is required.")
    metrics = {"duration_ms": (end - start) * 1000.0, "haraka_ms": haraka}
    # No inserted silence (sakt) between the clear letter and the next one.
    nxt = ev.unit_span(rule.unit_indices[1]) if len(rule.unit_indices) > 1 else None
    if nxt is not None:
        gap_runs = ev.silence_runs(end - 0.05, nxt[0] + 0.05, below_peak_db=30.0, min_ms=20.0)
        gap = max(((b - a) * 1000 for a, b in gap_runs), default=0.0)
        metrics["inserted_silence_ms"] = gap
        limit = max(150.0, 0.8 * haraka)
        if gap > limit:
            status = worst(status, Status.FAIL)
            score = min(score, 0.3)
            feedback += f" A {gap:.0f}ms pause was inserted before the next letter; Izhaar must not become a sakt."
    return RuleDiagnostic(
        rule_type=rule.rule_type, word=rule.word, start_ms=ms(start), end_ms=ms(end), status=status,
        feedback=feedback, expected_harakat=1.0, expected_range=(0.0, IZHAR_MAX), measured_harakat=counts,
        score=score, metrics=metrics, letter=rule.letter,
    )


def validate_izhar_halqi(rule: RuleInstance, ev: EvalContext) -> RuleDiagnostic:
    return clear_letter(rule, ev, "Noon" if "noon" in rule.detail else "Tanween")


def validate_idgham_no_ghunnah(rule: RuleInstance, ev: EvalContext) -> RuleDiagnostic:
    span = ev.span(rule.unit_indices[1:])
    if span is None:
        return skipped(rule, f"Idghaam on '{rule.word}' could not be located in the audio.")
    nasal = measure_nasality(ev.ctx, *span)
    ref = ev.oral_ner_reference
    contrast = nasal.ner_db - ref if ref is not None else nasal.ner_db
    if not np.isfinite(contrast):
        return skipped(rule, "Nasality could not be measured", span)
    if contrast < NO_GHUNNAH_MAX_CONTRAST_DB:
        status, score = Status.PASS, 1.0
        feedback = "Noon fully assimilated into the ل/ر with no residual ghunnah."
    elif contrast < NO_GHUNNAH_FAIL_CONTRAST_DB:
        status, score = Status.WARNING, 0.6
        feedback = f"Some nasal resonance remained (contrast {contrast:+.1f} dB); assimilate the noon completely."
    else:
        status, score = Status.FAIL, 0.2
        feedback = f"Ghunnah was retained (contrast {contrast:+.1f} dB): Idghaam into ل/ر has no ghunnah."
    return RuleDiagnostic(
        rule_type=rule.rule_type, word=rule.word, start_ms=ms(span[0]), end_ms=ms(span[1]), status=status,
        feedback=feedback, score=score, metrics={"nasal_energy_ratio_db": nasal.ner_db, "nasal_contrast_db": contrast},
        letter=rule.letter,
    )


VALIDATORS = {
    RuleType.GHUNNAH: validate_ghunnah_mushaddadah,
    RuleType.IDGHAM_GHUNNAH: validate_idgham_ghunnah,
    RuleType.IQLAB: validate_iqlab_or_ikhfa_shafawi,
    RuleType.IKHFA: validate_ikhfa,
    RuleType.IZHAR_HALQI: validate_izhar_halqi,
    RuleType.IDGHAM_NO_GHUNNAH: validate_idgham_no_ghunnah,
}
