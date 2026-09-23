"""Hafs orthography regressions (deep-research 00 Tier 0): the engine must not expect a wrong reading."""

from __future__ import annotations

import pytest

from app.models import RuleType, Vowel
from app.tajweed_rules.parser import TajweedParser, parse_text


def sakt_words(parsed) -> list[str]:  # type: ignore[no-untyped-def]
    return [r.word for r in parsed.rules if r.rule_type is RuleType.SAKT]


# 2:245 (abridged): the small seen over ص means "read seen", not a sakt after يَقْبِضُ
BASTU = "وَٱللَّهُ يَقْبِضُ وَيَبْصُۜطُ وَإِلَيْهِ تُرْجَعُونَ"
# 18:1 end + 18:2 start, with the stand-alone sakt sign
IWAJA = ["وَلَمْ يَجْعَل لَّهُۥ عِوَجَا ۜ", "قَيِّمًا لِّيُنذِرَ"]


def test_seen_mark_inside_a_word_is_not_a_sakt() -> None:
    p = parse_text(BASTU, include_sifaat=False)
    assert sakt_words(p) == []
    assert any("س" in w.text and "ص" not in w.text for w in p.words if "بْ" in w.text)


def test_standalone_sakt_sign_is_kept_in_wasl_and_dropped_at_waqf() -> None:
    joined = TajweedParser(include_sifaat=False).parse(IWAJA, continue_after={3})
    assert sakt_words(joined) == ["عِوَجَا"]
    stopped = TajweedParser(include_sifaat=False).parse(IWAJA[:1])
    assert sakt_words(stopped) == []  # a full waqf at 18:1 is valid; no "sakt instead of stop" FAIL


def test_rectangular_zero_alif_is_read_only_at_waqf() -> None:
    at_waqf = parse_text("أَنَا۠", include_sifaat=False)
    assert [u.pronounced for u in at_waqf.units] == [True, True, True]
    assert [r.rule_type for r in at_waqf.rules] == [RuleType.MADD_TABII]
    in_wasl = parse_text("أَنَا۠ رَبُّكُمْ", include_sifaat=False)
    assert not in_wasl.units[2].pronounced


@pytest.mark.parametrize("word,vowel", [
    ("ٱمْشُوا۟", Vowel.KASRA), ("ٱقْضُوٓا۟", Vowel.KASRA), ("ٱبْنُوا۟", Vowel.KASRA), ("ٱمْرُؤٌا", Vowel.KASRA),
    ("ٱسْمُهُۥ", Vowel.KASRA), ("ٱنظُرُوٓا۟", Vowel.DAMMA), ("ٱلْحَمْدُ", Vowel.FATHA), ("ٱهْدِنَا", Vowel.KASRA),
])
def test_hamzat_wasl_ibtida_vowel(word: str, vowel: Vowel) -> None:
    assert parse_text(word, include_sifaat=False).units[0].vowel is vowel
