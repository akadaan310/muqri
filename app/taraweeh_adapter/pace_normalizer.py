"""Hadr / Tadweer / Tahqeeq pace normalization.

Live Taraweeh is often recited in Hadr (harakah ≈ 100–140 ms) where studio Tahqeeq runs at
≈ 220–280 ms. Every timing rule is therefore judged as a multiple of the harakah, never in
milliseconds, so a 4-count Madd at 400 ms (Hadr) scores exactly like one at 960 ms (Tahqeeq).

Live imams also change pace within a passage (slower at the start of a rak'ah, faster later,
slowing again before ruku'). The **local** harakah around each rule — the median of the plain
short syllables within ±4 s — replaces the global one, so a Madd is judged against the tempo the
reciter was actually using at that moment.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from app.acoustic.tempo import MAX_HARAKA_MS, MIN_HARAKA_MS, short_syllable_units
from app.models import Alignment
from app.tajweed_rules.parser import ParsedText

TAHQEEQ_REFERENCE_MS = 250.0
LOCAL_WINDOW_S = 4.0
MIN_LOCAL_SAMPLES = 5


def classify_pace(haraka_ms: float) -> str:
    if haraka_ms < 150.0:
        return "hadr"
    if haraka_ms <= 220.0:
        return "tadweer"
    return "tahqeeq"


def hadr_to_tahqeeq_ratio(haraka_ms: float) -> float:
    """How much faster than a reference Tahqeeq (250 ms harakah) the recitation runs."""
    return TAHQEEQ_REFERENCE_MS / haraka_ms if haraka_ms > 0 else float("nan")


@dataclass(slots=True)
class LocalTempo:
    times_s: np.ndarray
    durations_ms: np.ndarray
    global_ms: float
    window_s: float = LOCAL_WINDOW_S

    def __call__(self, t: float) -> float:
        if self.times_s.size == 0:
            return self.global_ms
        sel = np.abs(self.times_s - t) <= self.window_s
        if sel.sum() < MIN_LOCAL_SAMPLES:
            nearest = np.argsort(np.abs(self.times_s - t))[:MIN_LOCAL_SAMPLES]
            sel = np.zeros_like(sel)
            sel[nearest] = True
        local = float(np.median(self.durations_ms[sel]))
        return float(np.clip(local, MIN_HARAKA_MS, MAX_HARAKA_MS))

    @property
    def variability(self) -> float:
        """Coefficient of variation of the local tempo across the passage."""
        if self.times_s.size < MIN_LOCAL_SAMPLES:
            return 0.0
        curve = np.array([self(t) for t in self.times_s])
        return float(np.std(curve) / np.mean(curve))


def build_local_tempo(parsed: ParsedText, alignment: Alignment, global_ms: float) -> LocalTempo:
    times, durs = [], []
    for i in short_syllable_units(parsed):
        u = alignment.units.get(i)
        if u is None:
            continue
        d = u.duration_s * 1000.0
        if MIN_HARAKA_MS * 0.5 <= d <= MAX_HARAKA_MS:
            times.append((u.start_s + u.end_s) / 2)
            durs.append(d)
    t, d = np.asarray(times), np.asarray(durs)
    if d.size >= 4:  # reject outliers with the global IQR
        q1, q3 = np.percentile(d, [25, 75])
        keep = (d >= q1 - 1.5 * (q3 - q1)) & (d <= q3 + 1.5 * (q3 - q1))
        t, d = t[keep], d[keep]
    return LocalTempo(times_s=t, durations_ms=d, global_ms=global_ms)


def local_haraka_function(parsed: ParsedText, alignment: Alignment, global_ms: float) -> Callable[[float], float]:
    return build_local_tempo(parsed, alignment, global_ms)
