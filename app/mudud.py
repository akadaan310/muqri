"""Aqwā al-Mudūd — which madd governs when two causes meet, and drift across a passage.

Two mudūd rules can apply to the same madd letter at once. The classical resolution is a strict
strength order, given in Samannūdī's line أَقْوَى المُدُودِ لاَزِمٌ فَمَا اتَّصَلْ فَعارِضٌ فَذُو انْفِصَالٍ فَبَدَلْ:

    lāzim  >  muttaṣil  >  'āriḍ li-s-sukūn  >  munfaṣil  >  badal

The stronger cause governs and the weaker is overridden. This matters before any judging happens: if
both madd badal (2 counts) and madd lāzim (6) are located on one letter, grading against badal would
call a correct six-count hold a gross over-lengthening. The engine has to decide what is *required*
before it can say what was *given*.

`drift` is the other thing a single verse cannot show. Taswiyah says every instance of a category is
held identically; the classical failure is not random scatter but a **trend** — a reciter opens a
passage giving munfaṣil four counts and, as fatigue sets in, drifts to three and then two and a half.
That is a slope over position, not a variance, so the CV cannot see it.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

# strongest first; anything absent is weaker than everything listed
STRENGTH_ORDER: tuple[str, ...] = (
    "madd_lazim",
    "madd_muttasil",
    "madd_arid_lissukun",
    "madd_munfasil",
    "madd_badal",
)
_RANK = {r: i for i, r in enumerate(STRENGTH_ORDER)}
MUDUD = frozenset(STRENGTH_ORDER) | {
    "madd_tabii", "madd_leen", "madd_iwad", "madd_silah_sughra", "madd_silah_kubra",
}


def rank(rule: str) -> int:
    """Lower is stronger. Unlisted mudūd rank below every named cause; non-mudūd rank last."""
    return _RANK.get(rule, len(STRENGTH_ORDER) if rule in MUDUD else len(STRENGTH_ORDER) + 1)


@dataclass(slots=True)
class Resolution:
    """What actually governs a madd letter, and what it overrode."""

    governing: str
    overridden: list[str]
    expected_counts: tuple[float, float] | None


def resolve(bounds: list) -> tuple[list, list[Resolution]]:  # type: ignore[type-arg]
    """Drop mudūd rules that a stronger cause on the same units overrides.

    Returns the surviving bound rules and a record of each resolution, so a report can say *why* a
    six-count hold was required where the text also shows a badal.
    """
    by_units: dict[tuple[int, ...], list] = {}
    others = []
    for b in bounds:
        if b.rule_type in MUDUD and b.unit_indices:
            by_units.setdefault(tuple(b.unit_indices), []).append(b)
        else:
            others.append(b)

    kept, notes = [], []
    for _units, group in by_units.items():
        if len(group) == 1:
            kept.append(group[0])
            continue
        group.sort(key=lambda b: rank(b.rule_type))
        winner, losers = group[0], group[1:]
        kept.append(winner)
        notes.append(Resolution(governing=winner.rule_type,
                                overridden=[b.rule_type for b in losers],
                                expected_counts=winner.expected_counts))
    return others + kept, notes


def drift(verdicts_by_ayah: list[list]) -> dict[str, dict]:  # type: ignore[type-arg]
    """Per madd category, the trend of the given count across the passage.

    A negative slope is the classical fatigue drift — the category shrinking as the reciter tires.
    Reported per category with the count of instances, since a slope on three points means nothing.
    """
    series: dict[str, list[tuple[int, float]]] = {}
    for i, vs in enumerate(verdicts_by_ayah):
        for v in vs:
            if v.rule not in MUDUD:
                continue
            got = (v.evidence or {}).get("given_counts")
            if got is None:
                continue
            series.setdefault(v.rule, []).append((i, float(got)))

    out: dict[str, dict] = {}
    for rule, pts in series.items():
        if len(pts) < 6:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        mx, my = statistics.mean(xs), statistics.mean(ys)
        den = sum((x - mx) ** 2 for x in xs)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0
        first_half = [y for x, y in pts if x < max(xs) / 2] or ys
        second_half = [y for x, y in pts if x >= max(xs) / 2] or ys
        out[rule] = {
            "n": len(pts),
            "slope_counts_per_ayah": round(slope, 4),
            "start_median": round(statistics.median(first_half), 2),
            "end_median": round(statistics.median(second_half), 2),
            # A slope alone is not drift: on the anchor, madd tabii showed slope -0.10 while the
            # start and end medians were both 1.87, i.e. a few outliers tilting the line. Real drift
            # has to move the middle of the distribution as well.
            "drifting": bool(abs(slope) > 0.05 and len(pts) >= 10 and max(xs) >= 4
                             and abs(statistics.median(second_half) - statistics.median(first_half))
                             > 0.25),
        }
    return out
