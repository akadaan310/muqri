"""Dynamic harakah (beat) estimation.

In Tajweed a *harakah* is the time needed to articulate one short vowel; every Madd and Ghunnah
is specified as a multiple of it. Because reciters range from slow Tahqeeq to quick Hadr, the
unit must be measured from the recitation itself: we take the median duration of all
open short syllables (consonant + fathah/dammah/kasrah) that are not affected by any
lengthening rule, with IQR outlier rejection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.models import MADD_RULES, NASAL_RULES, Alignment, RuleType
from app.tajweed_rules import ParsedText

MIN_HARAKA_MS = 80.0
MAX_HARAKA_MS = 900.0
MIN_SAMPLES = 3


class TempoError(RuntimeError):
    pass


@dataclass(slots=True)
class TempoEstimate:
    haraka_ms: float
    n_samples: int
    method: str
    iqr_ms: float = 0.0

    @property
    def harakat_per_minute(self) -> float:
        return 60_000.0 / self.haraka_ms

    def to_dict(self) -> dict[str, object]:
        return {
            "base_haraka_duration_ms": round(self.haraka_ms, 1),
            "tempo_bpm_harakat": round(self.harakat_per_minute, 1),
            "samples": self.n_samples,
            "method": self.method,
            "spread_iqr_ms": round(self.iqr_ms, 1),
        }


def short_syllable_units(parsed: ParsedText) -> list[int]:
    """Indices of units that form a plain short open syllable (1 harakah)."""
    excluded: set[int] = set()
    for rule in parsed.rules:
        if rule.rule_type in MADD_RULES:
            excluded.update(rule.unit_indices)
        elif rule.rule_type in NASAL_RULES:
            # Only the nasal carrier is lengthened; an Ikhfa/Iqlab target letter is a normal syllable.
            excluded.add(rule.unit_indices[0])
            if rule.rule_type in (RuleType.IDGHAM_GHUNNAH, RuleType.IDGHAM_SHAFAWI):
                excluded.update(rule.unit_indices)
    pron = [u for u in parsed.units if u.pronounced]
    last = pron[-1].index if pron else -1
    out: list[int] = []
    for pos, u in enumerate(pron):
        if u.vowel is None or u.tanween or u.shadda or u.madd_letter or u.index in excluded:
            continue
        if u.index == last or pos == 0:  # boundary syllables are lengthened / clipped
            continue
        nxt = pron[pos + 1] if pos + 1 < len(pron) else None
        if nxt is not None and nxt.madd_letter:
            continue
        out.append(u.index)
    return out


def expected_total_harakat(parsed: ParsedText) -> float:
    """Canonical length of the text in harakat (used when alignment gives too few samples)."""
    total = 0.0
    for u in parsed.units:
        if not u.pronounced:
            continue
        if u.vowel is not None and not u.madd_letter:
            total += 1.0
        elif u.sukun:
            total += 0.5
    for rule in parsed.rules:
        if rule.expected_harakat is not None:
            lo, hi = rule.expected_harakat
            total += (lo + hi) / 2 - 1.0  # the carrier syllable is already counted once
    return max(total, 1.0)


def estimate_tempo(parsed: ParsedText, alignment: Alignment, *,
                   voiced_duration_s: float | None = None) -> TempoEstimate:
    durations = np.array(
        [alignment.units[i].duration_s * 1000.0 for i in short_syllable_units(parsed) if i in alignment.units],
        dtype=np.float64,
    )
    durations = durations[(durations >= MIN_HARAKA_MS * 0.5) & (durations <= MAX_HARAKA_MS)]
    if durations.size >= MIN_SAMPLES:
        q1, q3 = np.percentile(durations, [25, 75])
        iqr = q3 - q1
        kept = durations[(durations >= q1 - 1.5 * iqr) & (durations <= q3 + 1.5 * iqr)]
        if kept.size >= MIN_SAMPLES:
            haraka = float(np.median(kept))
            return TempoEstimate(
                haraka_ms=float(np.clip(haraka, MIN_HARAKA_MS, MAX_HARAKA_MS)),
                n_samples=int(kept.size), method="median_short_syllable", iqr_ms=float(iqr),
            )

    # Fallback: total aligned speech divided by the canonical number of harakat.
    if voiced_duration_s is None:
        if not alignment.units:
            raise TempoError("Cannot estimate tempo: alignment is empty")
        starts = [u.start_s for u in alignment.units.values()]
        ends = [u.end_s for u in alignment.units.values()]
        voiced_duration_s = max(ends) - min(starts)
    haraka = voiced_duration_s * 1000.0 / expected_total_harakat(parsed)
    return TempoEstimate(
        haraka_ms=float(np.clip(haraka, MIN_HARAKA_MS, MAX_HARAKA_MS)),
        n_samples=int(durations.size), method="global_rate_fallback",
    )
