from __future__ import annotations

import pytest

from app.models import RuleType
from app.quran_text import QuranTextError, get_ayah_text, strip_basmala
from app.tajweed_rules import TajweedParseError, parse_text


def rules_of(text: str, **kw) -> list[tuple[str, str]]:
    return [(r.rule_type.value, r.word) for r in parse_text(text, **kw).rules]


def test_madd_classification_fatiha_7() -> None:
    rules = parse_text(get_ayah_text(1, 7)).rules
    by_type = {(r.rule_type, r.word) for r in rules}
    assert (RuleType.MADD_LAZIM, "ٱلضَّآلِّينَ") in by_type
    assert (RuleType.MADD_ARID, "ٱلضَّآلِّينَ") in by_type
    assert (RuleType.MADD_TABII, "ٱلَّذِينَ") in by_type
    lazim = next(r for r in rules if r.rule_type is RuleType.MADD_LAZIM)
    assert lazim.expected_harakat == (6, 6)


def test_madd_munfasil_and_muttasil() -> None:
    kawthar = parse_text(get_ayah_text(108, 1)).rules
    munfasil = [r for r in kawthar if r.rule_type is RuleType.MADD_MUNFASIL]
    assert [r.word for r in munfasil] == ["إِنَّآ"]
    assert munfasil[0].expected_harakat == (4, 5)
    muttasil = [r for r in parse_text("إِذَا جَآءَ نَصْرُ ٱللَّهِ").rules if r.rule_type is RuleType.MADD_MUTTASIL]
    assert [r.word for r in muttasil] == ["جَآءَ"]


def test_madd_is_elided_before_sakin_of_next_word() -> None:
    parsed = parse_text(get_ayah_text(113, 4))
    fi = parsed.words[3]
    assert fi.text == "فِى"
    assert all(not parsed.units[i].madd_letter for i in fi.unit_indices)
    assert not any(r.word == "فِى" and r.rule_type.value.startswith("madd") for r in parsed.rules)


def test_noon_sakinah_rules() -> None:
    assert ("ikhfa", "مِن") in rules_of(get_ayah_text(113, 2))
    assert ("idgham_ghunnah", "وَمَن") in rules_of("وَمَن يَعْمَلْ")
    assert ("iqlab", "مِن") in rules_of("مِنۢ بَعْدِ")  # display text drops the iqlab mark
    # Idgham without ghunnah (noon into lam) is not a nasal rule; the noon is assimilated.
    parsed = parse_text(get_ayah_text(112, 4))
    assert not any(r.rule_type in {RuleType.IKHFA, RuleType.IDGHAM_GHUNNAH} for r in parsed.rules)
    noon = parsed.units[parsed.words[1].unit_indices[-1]]
    assert noon.char == "ن" and noon.assimilated and noon.silent


def test_meem_sakinah_and_ghunnah_mushaddadah() -> None:
    assert ("ikhfa_shafawi", "تَرْمِيهِم") in rules_of("تَرْمِيهِم بِحِجَارَةٍ")
    assert ("ghunnah", "إِنَّ") in rules_of(get_ayah_text(103, 2))
    # The idgham target meem is not double-counted as a separate ghunnah.
    types = [t for t, _ in rules_of("لَهُم مَّا")]
    assert types.count("idgham_shafawi") == 1 and "ghunnah" not in types


def test_qalqalah_kubra_and_sughra() -> None:
    falaq = [r for r in parse_text(get_ayah_text(113, 1)).rules if r.rule_type is RuleType.QALQALAH]
    assert [(r.letter, r.detail) for r in falaq] == [("ق", "kubra")]
    asr = [r for r in parse_text(get_ayah_text(103, 3)).rules if r.rule_type is RuleType.QALQALAH]
    assert ("ب", "sughra") in [(r.letter, r.detail) for r in asr]
    # Continuing (no stop) removes the kubra.
    assert not any(r.rule_type is RuleType.QALQALAH for r in parse_text(get_ayah_text(113, 1), stop_at_end=False).rules)


def test_tafkheem_and_tarqeeq() -> None:
    rules = parse_text(get_ayah_text(1, 7)).rules
    weight_types = {RuleType.TAFKHEEM, RuleType.TARQEEQ}
    weights = {(r.rule_type.value, r.letter, r.word) for r in rules if r.rule_type in weight_types}
    assert ("tafkheem", "ط", "صِرَٰطَ") in weights
    assert ("tarqeeq", "ر", "غَيْرِ") in weights
    # Lam of the Divine Name: heavy after fathah/dammah, light after kasrah.
    assert ("tafkheem", "ٱللَّهُ") in rules_of("قُلْ هُوَ ٱللَّهُ أَحَدٌ")
    assert ("tarqeeq", "لِلَّهِ") in rules_of(get_ayah_text(1, 2))
    # Raa sakinah at a stop after a sakin letter looks back to the fathah: heavy.
    assert ("tafkheem", "بِٱلصَّبْرِ") in rules_of(get_ayah_text(103, 3))


def test_orthography_resolution() -> None:
    parsed = parse_text(get_ayah_text(1, 1))
    silent = [u.char for u in parsed.units if u.silent]
    assert silent.count("ٱ") == 3  # all hamzat al-wasl mid-stream are silent
    assert any(u.synthetic and u.madd_letter for u in parsed.units)  # dagger alif in ٱلرَّحْمَٰنِ
    # Initial hamzat al-wasl is pronounced with fathah before the article.
    first = parse_text(get_ayah_text(1, 2)).units[0]
    assert first.char == "ء" and not first.silent and first.vowel == "a"


def test_basmala_is_stripped_and_bad_references_rejected() -> None:
    assert get_ayah_text(113, 1).startswith("قُلْ")
    assert strip_basmala("بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ قُلْ", 112, 1) == "قُلْ"
    assert strip_basmala("بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ", 1, 1).startswith("بِسْمِ")
    with pytest.raises(QuranTextError):
        get_ayah_text(113, 9)
    with pytest.raises(TajweedParseError):
        parse_text("hello")
