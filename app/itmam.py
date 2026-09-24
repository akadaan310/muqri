"""Itmām al-Ḥarakāt — every vowel fully formed, and the sequences where they collapse.

Treatise III's universal law: a ḥarakah is not a short sound but a complete muscular configuration,
and every one must be fully formed and equal to every other. The two ways it fails are already
measured per instance — **ikhtilās** (the vowel cut below one count) and **ishbāʿ** (stretched past
it into an unwritten madd). What was missing is the law itself: not "is this vowel right" but
"are they all right, and do they stay right".

The treatise names three sequences where they stop staying right, because each makes a different
demand on the articulators:

    consecutive ḍammah      كُتُبُهُ      the lips must re-round for every vowel; the usual failure
                                        is the later ones flattening toward /o/
    alternating vowels      يَعِدُكُمُ    the tongue and lips reconfigure completely each time, and
                                        the usual failure is one of them being swallowed
    ḍammah → sukūn          كُنتُمْ      the vowel must complete BEFORE the consonant closes, and
                                        the usual failure is the vowel being clipped by the closure

Each is measured as the run's own trend: whether vowel duration decays across the sequence. A
declining run is the failure the treatise describes; scatter is not.

**These rates are NOT yet trustworthy as verdicts, and the report says so.** Measured on six ayahs
each, Husary Muallim (an anchor) scores *worse* than Shuraym (a fast imam): ikhtilas 0.239 vs 0.145,
alternating-run collapse 0.533 vs 0.367. An anchor ranking below a fast imam means the metric is
measuring something other than quality. The most likely confound is pauses — Husary Muallim is a
teaching reciter with 37 % of frames silent, and onset-to-onset duration charges a vowel for the
silence that follows it, which inflates some vowels, raises the median, and pushes the rest below the
truncation threshold. Excluding vowels adjacent to a detected stop is the obvious next test. Until
that is settled these are descriptive, not scored.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

FATHA, DAMMA, KASRA = "َ", "ُ", "ِ"
SHORT_V = frozenset("َُِ")
MADD = frozenset("اۥۦ")
# The treatise puts ikhtilas near two-thirds of a count and ishba' beyond one and a half — but those
# are fractions of a FULL haraka, and a vowel unit measures about 0.75 of one (the rest is the
# consonant's contact). Applying 0.67 directly read 34 % of an anchor's vowels as truncated, which is
# not credible. The thresholds are therefore relative to the reciter's own median vowel.
IKHTILAS_RATIO = 0.67
ISHBA_RATIO = 1.5


@dataclass(slots=True)
class Sequence:
    """One run of vowels of a named kind, and whether it held up across the run."""

    kind: str                  # consecutive_damma | alternating | damma_to_sukun
    start_unit: int
    length: int
    counts: list[float]
    decay: float               # per-position slope of the vowel's own counts; negative = collapsing
    collapsing: bool

    def to_dict(self) -> dict:  # type: ignore[type-arg]
        return {"kind": self.kind, "start_unit": self.start_unit, "length": self.length,
                "counts": [round(c, 2) for c in self.counts], "decay": round(self.decay, 4),
                "collapsing": self.collapsing}


def _vowel_runs(units) -> list[tuple[str, int, list[int]]]:  # type: ignore[no-untyped-def]
    """Locate the three named sequences as lists of vowel-unit indices."""
    vi = [i for i, u in enumerate(units) if u.symbol in SHORT_V]
    runs: list[tuple[str, int, list[int]]] = []
    i = 0
    while i < len(vi):
        j = i
        while (j + 1 < len(vi) and vi[j + 1] - vi[j] <= 2):   # consonant between vowels
            j += 1
        idxs = vi[i:j + 1]
        if len(idxs) >= 3:
            syms = [units[k].symbol for k in idxs]
            if all(s == DAMMA for s in syms):
                runs.append(("consecutive_damma", idxs[0], idxs))
            elif len(set(syms)) >= 2:
                runs.append(("alternating", idxs[0], idxs))
        i = j + 1

    # ḍammah immediately before a sākin consonant: the vowel must finish before the closure
    for k, u in enumerate(units):
        if u.symbol != DAMMA or k + 2 >= len(units):
            continue
        nxt, after = units[k + 1], units[k + 2]
        if nxt.symbol not in SHORT_V and nxt.symbol not in MADD and after.symbol not in SHORT_V:
            runs.append(("damma_to_sukun", k, [k]))
    return runs


def sequences(units) -> list[Sequence]:  # type: ignore[no-untyped-def]
    """Vowel sequences in one clip, each with the trend of its vowels' durations."""
    out: list[Sequence] = []
    for kind, start, idxs in _vowel_runs(units):
        counts = [units[k].duration_counts for k in idxs]
        counts = [c for c in counts if c is not None]
        if len(counts) < 3:
            # a single-vowel case (ḍammah→sukūn) carries no trend; it is reported by rate instead
            if counts:
                out.append(Sequence(kind=kind, start_unit=start, length=len(counts), counts=counts,
                                    decay=0.0, collapsing=False))
            continue
        xs = list(range(len(counts)))
        mx, my = statistics.mean(xs), statistics.mean(counts)
        den = sum((x - mx) ** 2 for x in xs)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, counts)) / den if den else 0.0
        out.append(Sequence(kind=kind, start_unit=start, length=len(counts), counts=counts,
                            decay=slope,
                            # the treatise's failure is a decaying run, not scatter
                            # a run must lose a real fraction of its opening vowel, not merely
                            # trend down: -0.08 flagged 63 % of an anchor's runs
                            collapsing=bool(slope < -0.08 and counts[0] > 0
                                            and counts[-1] < 0.6 * counts[0])))
    return out


def itmam(units_by_ayah) -> dict:  # type: ignore[no-untyped-def]
    """The universal law across the whole submission: are all the vowels complete and equal?"""
    per_vowel: dict[str, list[float]] = {FATHA: [], DAMMA: [], KASRA: []}
    for units in units_by_ayah:
        for u in units:
            if u.symbol in per_vowel and u.duration_counts is not None:
                per_vowel[u.symbol].append(u.duration_counts)
    allv = [c for v in per_vowel.values() for c in v]
    if len(allv) < 10:
        return {"enough_data": False, "n": len(allv)}
    base = statistics.median(allv)
    lo, hi = IKHTILAS_RATIO * base, ISHBA_RATIO * base

    meds = {k: statistics.median(v) for k, v in per_vowel.items() if len(v) >= 4}
    spread = (max(meds.values()) - min(meds.values())) if len(meds) >= 2 else None
    return {
        "enough_data": True, "n": len(allv),
        "per_vowel": {{FATHA: "fatha", DAMMA: "damma", KASRA: "kasra"}[k]:
                      {"n": len(per_vowel[k]), "median_counts": round(m, 3)}
                      for k, m in meds.items()},
        # isochrony: the three must be equal. Reported, but see the note -- the spread is measurable
        # only because sub-frame timing broke the 40 ms quantisation.
        "isochrony_spread": round(spread, 4) if spread is not None else None,
        "median_vowel_counts": round(base, 3),
        "ikhtilas_rate": round(sum(c < lo for c in allv) / len(allv), 4),
        "ishba_rate": round(sum(c > hi for c in allv) / len(allv), 4),
        "confidence": "unvalidated",
        "note": "Descriptive only. An anchor currently scores worse than a fast imam on ikhtilas "
                "(0.239 vs 0.145), so this is not yet a quality signal; pause-inflated durations are "
                "the suspected confound. The isochrony spread additionally reads 0.0 here because "
                "these durations still come from the quantised Viterbi — sub-frame onsets "
                "(research_agency_lab/experiments/subframe) fix that and are not yet ported in.",
    }


def roll_up(seqs_by_ayah: list[list[Sequence]]) -> dict:  # type: ignore[type-arg]
    flat = [s for ss in seqs_by_ayah for s in ss]
    per: dict[str, dict] = {}
    for s in flat:
        d = per.setdefault(s.kind, {"n": 0, "collapsing": 0, "decays": []})
        d["n"] += 1
        d["collapsing"] += int(s.collapsing)
        d["decays"].append(s.decay)
    for d in per.values():
        ds = d.pop("decays")
        d["median_decay"] = round(statistics.median(ds), 4) if ds else None
        d["collapse_rate"] = round(d["collapsing"] / d["n"], 4) if d["n"] else None
    return per
