"""Aqwa al-Mudud resolution and passage drift."""

from __future__ import annotations

from dataclasses import dataclass

from app.mudud import MUDUD, STRENGTH_ORDER, drift, rank, resolve


@dataclass
class FakeBound:
    rule_type: str
    unit_indices: list
    expected_counts: tuple | None


@dataclass
class FakeVerdict:
    rule: str
    evidence: dict


def test_strength_order_is_the_classical_one() -> None:
    """lazim > muttasil > 'arid > munfasil > badal (Samannudi's line)."""
    assert STRENGTH_ORDER == ("madd_lazim", "madd_muttasil", "madd_arid_lissukun",
                              "madd_munfasil", "madd_badal")
    assert rank("madd_lazim") < rank("madd_muttasil") < rank("madd_arid_lissukun") \
        < rank("madd_munfasil") < rank("madd_badal")
    assert rank("madd_tabii") > rank("madd_badal"), "an unnamed madd yields to every named cause"
    assert rank("qalqalah") > rank("madd_tabii"), "a non-madd rule never competes"


def test_stronger_cause_governs_the_same_letter() -> None:
    """Grading a six-count lazim against a two-count badal would call it a gross over-lengthening."""
    kept, notes = resolve([
        FakeBound("madd_badal", [7], (2, 2)),
        FakeBound("madd_lazim", [7], (6, 6)),
        FakeBound("madd_tabii", [3], (2, 2)),
    ])
    rules = {b.rule_type for b in kept}
    assert "madd_lazim" in rules and "madd_badal" not in rules
    assert "madd_tabii" in rules, "a madd on a different letter is untouched"
    assert notes[0].governing == "madd_lazim"
    assert notes[0].overridden == ["madd_badal"]
    assert notes[0].expected_counts == (6, 6)


def test_non_madd_rules_pass_through_untouched() -> None:
    kept, notes = resolve([FakeBound("qalqalah", [1], None), FakeBound("jahr", [1], None)])
    assert len(kept) == 2 and not notes


def test_drift_needs_a_real_shift_not_just_a_slope() -> None:
    """A few outliers can tilt a line while the middle of the distribution stays put.

    Measured on the anchor: madd tabii had slope -0.10 with start and end medians both 1.87. That is
    not fatigue drift and must not be reported as such.
    """
    steady = [[FakeVerdict("madd_tabii", {"given_counts": 2.0}) for _ in range(4)] for _ in range(6)]
    steady[0].append(FakeVerdict("madd_tabii", {"given_counts": 6.0}))   # one outlier
    out = drift(steady)
    assert out["madd_tabii"]["drifting"] is False

    falling = [[FakeVerdict("madd_munfasil", {"given_counts": 4.0 - 0.4 * i}) for _ in range(4)]
               for i in range(6)]
    out = drift(falling)
    assert out["madd_munfasil"]["drifting"] is True
    assert out["madd_munfasil"]["slope_counts_per_ayah"] < 0
    assert out["madd_munfasil"]["end_median"] < out["madd_munfasil"]["start_median"]


def test_drift_ignores_categories_with_too_few_instances() -> None:
    thin = [[FakeVerdict("madd_lazim", {"given_counts": 6.0})] for _ in range(3)]
    assert "madd_lazim" not in drift(thin)
