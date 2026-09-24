"""A transcript in, the exact span of the Quran out -- the way a consumer app learns what was recited."""

from __future__ import annotations

import pytest

from app.verse_locate import locate


@pytest.mark.parametrize(("text", "verses"), [
    ("قل هو الله احد الله الصمد", [(112, 1), (112, 2)]),                        # across ayahs
    ("قالوا لا علم لنا انك انت علام الغيوب", [(5, 109, 7, 14)]),              # the end of an ayah
    ("لا تأخذه سنة ولا نوم", [(2, 255, 7, 11)]),                               # a phrase inside one
    ("الحمد لله رب العالمين الرحمن الرحيم مالك يوم الدين", [(1, 2), (1, 3), (1, 4)]),
    ("ذلك الكتاب لا ريب فيه هدى للمتقين", [(2, 2)]),                           # dagger alif both ways
])
def test_the_span_is_located_to_the_word(text, verses) -> None:  # type: ignore[no-untyped-def]
    loc = locate(text)
    assert loc is not None and loc.verses == verses
    assert loc.score >= 0.95


def test_text_that_is_not_quran_matches_poorly() -> None:
    for t in ("الله اكبر الله اكبر اشهد ان لا اله الا الله", "التحيات لله والصلوات والطيبات"):
        assert locate(t).score < 0.8
