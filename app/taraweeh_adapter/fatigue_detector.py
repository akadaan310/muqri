"""Breath, fatigue and pitch isolation for long live recitations.

* **Pauses and breaths.** Every silence of 150 ms or more is located and checked for an
  inhalation (``app.sifaat.hams_jahr.detect_breath``).
* **Waqf al-Dharoori.** A pause after a word where the text continues means the imam stopped (for
  breath). The text is then re-parsed with a stop there, so waqf rules apply (the vowel is dropped,
  hamzat al-wasl is pronounced on resuming). That stop is not a Tajweed error, and a Madd cut
  short right before such a breath is reported as ``VALID_NECESSARY_PAUSE``.
* **Fatigue envelope.** Over a long session, breaths come more often and the voice source
  weakens (steeper spectral tilt). The report tracks both as oxygen-depletion markers.
* **Pitch isolation.** F0 movement (maqam, emotional drops at verse ends) is reported as style
  only. Timing rules use durations and formant rules use Praat's pitch-adaptive tracker, so the
  melody never enters a Tajweed score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.acoustic.features import AcousticContext
from app.audio import frame_rms_db
from app.models import Alignment
from app.sifaat.hams_jahr import detect_breath
from app.tajweed_rules.base import Pause
from app.tajweed_rules.parser import ParsedText
from app.taraweeh_adapter.dereverb import spectral_tilt_db_per_octave

MIN_PAUSE_S = 0.15
DYNAMIC_STOP_S = 0.30


def _flatness_per_frame(x: np.ndarray, sr: int, frame: int, hop: int) -> np.ndarray:
    n = max(0, 1 + (len(x) - frame) // hop)
    if n == 0:
        return np.zeros(0)
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    spec = np.abs(np.fft.rfft(x[idx] * np.hanning(frame)[None, :], axis=1)) ** 2 + 1e-20
    freqs = np.fft.rfftfreq(frame, 1 / sr)
    band = spec[:, (freqs >= 500) & (freqs < 5000)]
    return np.exp(np.mean(np.log(band), axis=1)) / np.mean(band, axis=1)


def detect_pauses(ctx: AcousticContext, alignment: Alignment | None = None, *,
                  min_pause_s: float = MIN_PAUSE_S) -> list[Pause]:
    """Stretches ≥ ``min_pause_s`` without phonation, with a breath flag and the preceding unit.

    A frame is "not phonation" when it is quiet, or when it is unvoiced, spectrally flat noise
    well below the speech level — an inhalation is audible but is not recitation. Unvoiced
    fricatives are louder and much shorter than 150 ms, so they do not form pauses.
    """
    hop, frame = int(0.01 * ctx.sr), int(0.02 * ctx.sr)
    db = frame_rms_db(ctx.x, ctx.sr, 20.0, 10.0).astype(np.float64)
    if db.size < 5:
        return []
    floor, peak = float(np.percentile(db, 5)), float(np.percentile(db, 95))
    quiet = db < max(floor + 6.0, peak - 30.0)
    flat = _flatness_per_frame(ctx.x, ctx.sr, frame, hop)
    t, f0 = ctx.f0_track
    frame_t = np.arange(len(db)) * 0.01 + 0.01
    voiced = np.interp(frame_t, t, np.isfinite(f0).astype(float)) > 0.5 if t.size else np.zeros(len(db), bool)
    n = min(len(db), len(flat))
    noise = np.zeros(len(db), bool)
    noise[:n] = (~voiced[:n]) & (flat[:n] > 0.25) & (db[:n] < peak - 12.0)
    nonspeech = np.append(quiet | noise, False)
    pauses: list[Pause] = []
    start = None
    for k, q in enumerate(nonspeech):
        if q and start is None:
            start = k
        elif not q and start is not None:
            a, b = start * 0.01, k * 0.01
            if b - a >= min_pause_s and a > 0.05:  # ignore leading silence
                ev = detect_breath(ctx, a, b)
                pauses.append(Pause(a, b, ev.breath, _unit_before(alignment, a) if alignment else None))
            start = None
    # Trailing silence is not a pause inside the recitation.
    if pauses and pauses[-1].end_s >= len(ctx.x) / ctx.sr - 0.03:
        pauses.pop()
    return pauses


def _unit_before(alignment: Alignment, t: float) -> int | None:
    best, best_start = None, -1.0
    for idx, u in alignment.units.items():
        if u.start_s <= t and u.start_s > best_start:
            best, best_start = idx, u.start_s
    return best


def dynamic_stops(parsed: ParsedText, alignment: Alignment, pauses: list[Pause],
                  min_pause_s: float = DYNAMIC_STOP_S) -> set[int]:
    """Word indices after which the reciter paused although the text has no stop there."""
    stops: set[int] = set()
    for p in pauses:
        if p.duration_s < min_pause_s or p.after_unit is None:
            continue
        word = parsed.words[parsed.units[p.after_unit].word_index]
        if word.stop_after or word.index == len(parsed.words) - 1:
            continue
        last = max((i for i in word.unit_indices if i in alignment.units), default=None)
        if last is None:
            continue
        # The pause must fall at the end of the word (its last unit), not in the middle of it.
        if p.after_unit == last or alignment.units[last].end_s <= p.start_s + 0.15:
            stops.add(word.index)
    return stops


@dataclass(slots=True)
class FatigueReport:
    duration_min: float
    breaths: int
    breaths_per_minute: float
    tilt_drift_db: float  # spectral tilt of the last third minus the first third (dB/octave)
    f0_drift_semitones: float
    oxygen_depletion: bool

    def to_dict(self) -> dict[str, float | int | bool | None]:
        def r(v: float, n: int = 2) -> float | None:
            return round(v, n) if np.isfinite(v) else None

        return {
            "duration_min": r(self.duration_min), "breaths": self.breaths,
            "breaths_per_minute": r(self.breaths_per_minute), "tilt_drift_db": r(self.tilt_drift_db),
            "f0_drift_semitones": r(self.f0_drift_semitones), "oxygen_depletion": self.oxygen_depletion,
        }


def fatigue_report(ctx: AcousticContext, pauses: list[Pause]) -> FatigueReport:
    dur = len(ctx.x) / ctx.sr
    breaths = sum(1 for p in pauses if p.breath)
    rate = breaths / (dur / 60.0) if dur > 0 else 0.0
    thirds = np.array_split(ctx.x, 3) if dur >= 9.0 else []
    tilt_drift = float("nan")
    f0_drift = float("nan")
    if len(thirds) == 3:
        tilt_drift = spectral_tilt_db_per_octave(thirds[2], ctx.sr) - spectral_tilt_db_per_octave(thirds[0], ctx.sr)
        a = ctx.f0_in(0.0, dur / 3)
        b = ctx.f0_in(2 * dur / 3, dur)
        if a.size > 10 and b.size > 10:
            f0_drift = float(12 * np.log2(np.median(b) / np.median(a)))
    depleted = bool(rate > 6.0 or (np.isfinite(tilt_drift) and tilt_drift < -2.0))
    return FatigueReport(dur / 60.0, breaths, rate, tilt_drift, f0_drift, depleted)


def pitch_profile(ctx: AcousticContext, span: tuple[float, float] | None = None) -> dict[str, float | None]:
    """Stylistic melody statistics, reported and never scored."""
    start, end = span if span is not None else (0.0, len(ctx.x) / ctx.sr)
    f0 = ctx.f0_in(start, end)
    if f0.size < 10:
        return {"f0_median_hz": None, "f0_range_semitones": None, "f0_std_semitones": None}
    st = 12 * np.log2(f0 / np.median(f0))
    return {
        "f0_median_hz": round(float(np.median(f0)), 1),
        "f0_range_semitones": round(float(np.percentile(st, 95) - np.percentile(st, 5)), 2),
        "f0_std_semitones": round(float(np.std(st)), 2),
    }
