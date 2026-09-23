"""The conditional weight of ر and ل — the strictest contextual rules in tajweed.

Every other sifah we judge is a fixed property of the letter, so the *expected* value is trivially
right. Tafkhīm/tarqīq of ر and of the lām of the Divine Name is not: it depends on the vowel on the
letter, on the vowel **before** it, on whether that kasrah is original or incidental, and on whether
a ḥarf isti'lā' follows in the same word. The engine judges realisation against the phonetizer's
expected label, so if that label is wrong the verdict is wrong no matter how good the acoustic model
is. These tests pin the classical conditions.

**Scope matters.** A rule whose condition lies inside the word may be checked on the word alone; the
lām of الله depends on the vowel of the *preceding word*, so it must be checked on the full ayah.
Phonetising `قَالَ ٱللَّهُ` as a bare fragment yields a light lām — an artefact of the missing
context, not a bug in the phonetizer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research_agency_lab/experiments/learner_eval"))

quran_transcript = pytest.importorskip("quran_transcript")
Aya, quran_phonetizer = quran_transcript.Aya, quran_transcript.quran_phonetizer

try:
    from muaalem_eval import MOSHAF
except Exception:  # pragma: no cover - the eval module pulls the transformers shim
    MOSHAF = None

pytestmark = pytest.mark.skipif(MOSHAF is None, reason="muaalem_eval unavailable")

HEAVY, LIGHT = "mofakham", "moraqaq"


def weights(text: str, letter: str) -> list[str]:
    r = quran_phonetizer(text, MOSHAF, remove_spaces=True)
    return [e.tafkheem_or_taqeeq for e in r.sifat for ch in e.phonemes if ch == letter]


def ayah_weights(surah: int, ayah: int, letter: str) -> list[str]:
    return weights(Aya(surah, ayah).get().uthmani, letter)


# ---- ر: conditions that live inside the word, so the word alone is the right scope ----------------
@pytest.mark.parametrize(
    ("word", "expected", "why"),
    [
        ("رَبِّ", HEAVY, "ra with fathah"),
        ("رُسُلٌ", HEAVY, "ra with dammah"),
        ("رِجَالٌ", LIGHT, "ra with kasrah"),
        ("ٱلْأَرْضِ", HEAVY, "ra sakinah after fathah"),
        ("قُرْءَانٌ", HEAVY, "ra sakinah after dammah"),
        ("فِرْعَوْنَ", LIGHT, "ra sakinah after an ORIGINAL kasrah, no isti'la following"),
        ("مِرْصَادًا", HEAVY, "ra sakinah after kasrah but ص (isti'la) follows in the same word"),
        ("قِرْطَاسٍ", HEAVY, "ra sakinah after kasrah but ط (isti'la) follows in the same word"),
        ("خَيْرٍ", LIGHT, "ra preceded by a sakin yaa (leen)"),
    ],
)
def test_raa_weight_word_scope(word: str, expected: str, why: str) -> None:
    got = weights(word, "ر")
    assert got, f"no ra found in {word}"
    assert got[0] == expected, f"{why}: {word} -> {got[0]}, expected {expected}"


def test_raa_isti3la_exception_is_not_blanket() -> None:
    """The isti'la exception must apply only when isti'la actually follows.

    فِرْعَوْن and مِرْصَاد differ *only* in the letter after the sakin ra, so a phonetizer that
    ignored the condition would give them the same weight. This is the sharpest single check.
    """
    assert weights("فِرْعَوْنَ", "ر")[0] == LIGHT
    assert weights("مِرْصَادًا", "ر")[0] == HEAVY


# ---- ل of the Divine Name: the condition is the PRECEDING word's vowel, so full-ayah scope --------
def test_lam_jalalah_heavy_after_fathah() -> None:
    # 5:115 opens قَالَ ٱللَّهُ — the lam of قَالَ stays light, the jalalah lams go heavy
    got = ayah_weights(5, 115, "ل")
    assert HEAVY in got, "jalalah after a fathah must be heavy"


def test_lam_jalalah_heavy_when_ayah_initial() -> None:
    # 2:255 opens ٱللَّهُ لَآ إِلَٰهَ
    assert ayah_weights(2, 255, "ل")[0] == HEAVY


def test_lam_jalalah_light_after_kasrah() -> None:
    # 1:1 بِسْمِ ٱللَّهِ — every lam is light
    got = ayah_weights(1, 1, "ل")
    assert got and all(w == LIGHT for w in got), f"jalalah after a kasrah must be light, got {got}"


def test_ordinary_lam_is_always_light() -> None:
    assert all(w == LIGHT for w in weights("لَهُمْ", "ل"))


def test_isolated_fragment_loses_jalalah_context() -> None:
    """Documents the scope trap so nobody 'fixes' the phonetizer against a bad test.

    Phonetised on its own, قَالَ ٱللَّهُ has no preceding-word vowel to condition on and the lam
    comes back light; in the real ayah it is heavy. Weight tests for the Divine Name must use ayahs.
    """
    assert HEAVY not in weights("قَالَ ٱللَّهُ", "ل")
    assert HEAVY in ayah_weights(5, 115, "ل")
