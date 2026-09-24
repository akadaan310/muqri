"""Waqf, sakt and endurance — where the reciter actually stopped, and whether it held up.

Waqf has two halves and only one of them is measurable from what we have:

* **Permissibility** — whether a given stopping place is allowed, preferred, forbidden, or either.
  That lives in the waqf signs (ۖ ۗ ۚ ۛ ۘ, U+06D6–U+06DB). **They are not in any text source we
  have**: zero occurrences across all 6,236 ayahs in the store, and `quran_transcript`'s `Aya`
  exposes none of them in `uthmani`, `imlaey` or `rasm`. Judging permissibility needs a marked
  muṣḥaf text, which is a *data* gap, not a modelling one, and is recorded as such rather than
  quietly approximated.
* **Execution** — where the reciter *did* stop, and whether that stop was legitimate in the plainest
  sense. That is measurable now, and it is what this module does.

Three things fall out of the same pause detection:

    waqf       a stop at a word boundary or an ayah end — fine; a stop INSIDE a word is an error
               under every reading, no waqf table needed to say so
    sakt       a brief cut without breath: short, and shorter than the reciter's own stops
    endurance  how breath is managed across a passage — pause count and whether stops lengthen as
               the reciter tires, which is the fatigue the treatises warn about

Pause detection needs the WAVEFORM. The posteriors cannot supply it: CTC is peaky, blank dominates
between every symbol, and thresholding blank marked 8.6 s of a 14.1 s clip of continuous recitation
as "pause". So `find_stops` takes the audio and returns nothing without it, rather than guessing.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

FRAME_S = 0.04
# A sakt is "a brief cut without breath"; a waqf is a full stop. The boundary is taken from the
# reciter's own distribution rather than a fixed millisecond value, because a Tahqiq reciter's brief
# cut is longer than a Hadr reciter's full stop.
MIN_PAUSE_S = 0.12
SAKT_MAX_S = 0.45


@dataclass(slots=True)
class Stop:
    """One silence in the recitation, and what it lands on."""

    after_unit: int
    start_s: float
    duration_s: float
    kind: str                 # waqf | sakt
    position: str             # ayah_end | word_boundary | mid_word
    word_index: int | None
    legitimate: bool

    def to_dict(self) -> dict:  # type: ignore[type-arg]
        return {"after_unit": self.after_unit, "start_s": round(self.start_s, 3),
                "duration_s": round(self.duration_s, 3), "kind": self.kind,
                "position": self.position, "word_index": self.word_index,
                "legitimate": self.legitimate}


def _word_of(char_index: int, word_ph: list[list[int]]) -> int | None:
    for w, span in enumerate(word_ph):
        if span and span[0] >= 0 and span[0] <= char_index < span[1]:
            return w
    return None


def silent_frames(wave, sr: int = 16000, frame_s: float = FRAME_S, drop_db: float = 32.0):  # type: ignore[no-untyped-def]
    """Per-frame silence mask from the WAVEFORM, not from the posteriors.

    CTC posteriors cannot answer this. The model is peaky: blank dominates between every symbol, so
    thresholding blank marks most of a clip as silent — on a 14.1 s clip of continuous recitation,
    `blank > 0.999` produced 8.6 s of "pause". Silence is an acoustic question and needs the audio.

    A frame is silent when its RMS sits `drop_db` below the clip's speech level, taken at a high
    percentile so one loud onset does not set the bar.
    """
    import numpy as np
    x = np.asarray(wave, dtype="float32")
    n = max(1, int(frame_s * sr))
    if x.size < n:
        return np.zeros(0, dtype=bool)
    frames = x[: x.size // n * n].reshape(-1, n)
    db = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1) + 1e-12))
    return db < (float(np.percentile(db, 90)) - drop_db)


def find_stops(units, word_ph: list[list[int]], wave=None, *,  # type: ignore[no-untyped-def]
               min_pause_s: float = MIN_PAUSE_S) -> list[Stop]:
    """Silences between units, classified by where they fall.

    Requires `wave`: silence is acoustic and the posteriors cannot supply it (see `silent_frames`).
    Without audio this returns nothing rather than guessing — the dump-based research path has no
    waveform, so waqf is simply not judged there.
    """
    if wave is None or not units:
        return []
    sil = silent_frames(wave)
    if sil.size == 0:
        return []
    stops: list[Stop] = []
    for i in range(len(units) - 1):
        lo, hi = units[i].frames[1], units[i + 1].frames[0] - 1
        if hi < lo:
            continue
        seg = sil[max(0, lo):min(len(sil), hi + 1)]
        dur = int(seg.sum()) * FRAME_S if seg.size else 0.0
        if dur < min_pause_s:
            continue
        end_char = units[i].char_span[1]
        w_here = _word_of(end_char, word_ph)
        w_next = _word_of(units[i + 1].char_span[0], word_ph)
        position = "mid_word" if (w_here is not None and w_here == w_next) else "word_boundary"
        stops.append(Stop(after_unit=i, start_s=lo * FRAME_S, duration_s=dur,
                          kind="waqf", position=position, word_index=w_here,
                          legitimate=position != "mid_word"))

    # the last unit always ends the clip; that stop is the ayah end, which is always legitimate
    if units:
        stops.append(Stop(after_unit=len(units) - 1,
                          start_s=units[-1].frames[1] * FRAME_S, duration_s=0.0,
                          kind="waqf", position="ayah_end",
                          word_index=_word_of(units[-1].char_span[1], word_ph), legitimate=True))

    # sakt vs waqf, on this reciter's own scale: the short tail of their own pauses
    real = [s for s in stops if s.duration_s > 0]
    if len(real) >= 4:
        cut = min(SAKT_MAX_S, statistics.median(s.duration_s for s in real))
        for s in real:
            if s.duration_s <= cut:
                s.kind = "sakt"
    return stops


def endurance(stops_by_ayah: list[list[Stop]]) -> dict:  # type: ignore[type-arg]
    """Breath management across a passage.

    The treatises' fatigue failure is stops getting longer and more frequent as the reciter tires, so
    a trend is reported rather than an average — the same reasoning as madd drift.
    """
    per: list[tuple[int, float, int]] = []
    for i, stops in enumerate(stops_by_ayah):
        real = [s for s in stops if s.duration_s > 0]
        per.append((i, sum(s.duration_s for s in real), len(real)))
    if len(per) < 4:
        return {"ayahs": len(per), "enough_data": False}

    xs = [p[0] for p in per]
    pause_s = [p[1] for p in per]
    counts = [p[2] for p in per]

    def slope(ys: list[float]) -> float:
        mx, my = statistics.mean(xs), statistics.mean(ys)
        den = sum((x - mx) ** 2 for x in xs)
        return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0

    half = max(xs) / 2
    first = [y for x, y in zip(xs, pause_s) if x < half] or pause_s
    second = [y for x, y in zip(xs, pause_s) if x >= half] or pause_s
    s_pause = slope(pause_s)
    return {
        "ayahs": len(per), "enough_data": True,
        "total_pause_s": round(sum(pause_s), 2),
        "pauses_per_ayah": round(statistics.mean(counts), 2),
        "pause_slope_s_per_ayah": round(s_pause, 4),
        "start_median_pause_s": round(statistics.median(first), 3),
        "end_median_pause_s": round(statistics.median(second), 3),
        # as with madd drift, a slope alone is not fatigue: the middle has to move too
        "tiring": bool(s_pause > 0.05 and len(per) >= 6
                       and statistics.median(second) - statistics.median(first) > 0.1),
    }


def roll_up(stops_by_ayah: list[list[Stop]]) -> dict:  # type: ignore[type-arg]
    """Report-level summary of stopping."""
    flat = [s for stops in stops_by_ayah for s in stops]
    real = [s for s in flat if s.duration_s > 0]
    mid = [s for s in flat if s.position == "mid_word"]
    return {
        "stops": len(real), "sakt": sum(1 for s in real if s.kind == "sakt"),
        "waqf": sum(1 for s in real if s.kind == "waqf"),
        "mid_word_stops": len(mid),
        "median_pause_s": round(statistics.median([s.duration_s for s in real]), 3) if real else None,
        "illegitimate": [s.to_dict() for s in mid[:20]],
        "permissibility": "not judged — the waqf signs (U+06D6–U+06DB) are absent from every "
                          "available text source, so which stops are preferred or forbidden cannot "
                          "be determined; only whether a stop was made at a legitimate place",
    }
