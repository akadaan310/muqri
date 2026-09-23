"""Binding the parser's located rules to the phoneme units that realise them.

This join is what turns "a madd error here" into "madd munfaṣil in وَمَا أَنزَلَ — you gave 2.1
counts, it requires 4". Every case below is a structure that broke an earlier version, so they are
regression guards rather than illustrations.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research_agency_lab/experiments/learner_eval"))

quran_transcript = pytest.importorskip("quran_transcript")
Aya, quran_phonetizer = quran_transcript.Aya, quran_transcript.quran_phonetizer

from app.rule_bind import bind, ph_units  # noqa: E402
from app.tajweed_rules.parser import TajweedParser  # noqa: E402

try:
    import muaalem_dump as md
    from muaalem_eval import MOSHAF
except Exception:  # pragma: no cover - needs the transformers shim
    md = MOSHAF = None

pytestmark = pytest.mark.skipif(MOSHAF is None, reason="muaalem_eval unavailable")


def bound_for(surah: int, ayah: int):  # type: ignore[no-untyped-def]
    u = Aya(surah, ayah).get().uthmani
    parsed = TajweedParser().parse(u)
    r = quran_phonetizer(u, MOSHAF, remove_spaces=True)
    rules = bind(parsed, r.phonemes, md.word_spans(u, r.mappings))
    return rules, ph_units(r.phonemes), len(parsed.rules)


def symbols(units, b) -> str:  # type: ignore[no-untyped-def]
    return "".join(units[i][0] * (units[i][2] - units[i][1] + 1) for i in b.unit_indices)


def test_ph_units_matches_julia_semantics() -> None:
    assert ph_units("ااب") == [("ا", 0, 1), ("ب", 2, 2)]
    assert ph_units("") == []


def test_madd_binds_to_the_madd_run_not_the_consonant() -> None:
    rules, units, _ = bound_for(2, 2)
    madd = [b for b in rules if b.rule_type == "madd_tabii" and b.word_index == 0]
    assert madd, "no madd tabii bound in the first word of 2:2"
    assert symbols(units, madd[0]) == "اا"
    assert madd[0].expected_counts == (2, 2)


def test_two_instances_in_one_word_bind_to_different_runs() -> None:
    """وَيُقِيمُونَ carries two madd tabii; if both bind to one run, a real error becomes invisible."""
    rules, units, _ = bound_for(2, 3)
    madd = [b for b in rules if b.rule_type == "madd_tabii" and b.word_index == 3]
    assert len(madd) == 2
    assert madd[0].unit_indices != madd[1].unit_indices
    assert {symbols(units, m) for m in madd} == {"ۦۦ", "ۥۥ"}


def test_idgham_binds_across_the_word_boundary() -> None:
    """The assimilated letter lands on the NEXT word's first letter, doubled.

    2:5 is هُدًۭى مِّن رَّبِّهِمْ: the tanwin merges into the mim with ghunnah, and the nun of مِّن
    merges into the raa without it.
    """
    rules, units, _ = bound_for(2, 5)
    with_gh = [b for b in rules if b.rule_type == "idgham_ghunnah"]
    no_gh = [b for b in rules if b.rule_type == "idgham_no_ghunnah"]
    assert with_gh and symbols(units, with_gh[0]) == "مممم"
    assert with_gh[0].shadda == "nasal"       # held for two counts of ghunnah
    assert no_gh and symbols(units, no_gh[0]) == "رر"
    assert no_gh[0].shadda == "plain"         # a collision, no ghunnah


def test_shadda_is_two_letters_sakin_plus_voweled() -> None:
    """A shadda letter is a sakin half and a voweled half collided, so it occupies a repeated run.

    Nasals are held four (two counts of ghunnah); everything else is a run of two.
    """
    rules, units, _ = bound_for(2, 5)
    plain = [b for b in rules if b.shadda == "plain"]
    nasal = [b for b in rules if b.shadda == "nasal"]
    assert plain and all(len(symbols(units, b)) == 2 for b in plain)
    assert nasal and all(len(symbols(units, b)) >= 4 for b in nasal)


def test_hamzat_wasl_binds_although_the_hamza_is_dropped() -> None:
    """ٱلْكِتَٰبُ is recited "l-kitaabu" — there is no hamza to point at, so it binds to the first unit."""
    rules, units, _ = bound_for(2, 2)
    hw = [b for b in rules if b.rule_type == "hamzat_wasl"]
    assert hw, "hamzat wasl must still bind when the hamza is dropped"
    assert all(b.unit_indices for b in hw)


def test_bind_rate_is_high_on_a_real_ayah() -> None:
    rules, _units, located = bound_for(2, 5)
    assert len(rules) / located >= 0.9, f"only {len(rules)}/{located} rules bound"


@pytest.mark.parametrize(("surah", "ayah"), [(1, 1), (2, 2), (2, 5), (36, 1), (112, 1)])
def test_binding_never_raises_and_units_are_in_range(surah: int, ayah: int) -> None:
    rules, units, _ = bound_for(surah, ayah)
    for b in rules:
        assert b.unit_indices, f"{b.rule_type} bound with no units"
        assert all(0 <= i < len(units) for i in b.unit_indices)
        assert b.mechanism in {"durational", "attribute", "segmental"}
