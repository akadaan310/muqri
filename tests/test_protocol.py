"""The recording protocol: every scripted mistake must point at a real word, and scoring must
separate caught, near, missed and false alarms exactly."""

from __future__ import annotations

import pytest

from app.protocol import TESTS, Test, scorecard


def test_every_scripted_word_is_in_its_ayah() -> None:
    qt = pytest.importorskip("quran_transcript")
    from app.protocol import resolve
    res = resolve(lambda s, a: qt.Aya(s, a).get().uthmani.split())
    assert sum(len(v) for v in res.values()) == sum(len(t.mistakes) for t in TESTS)
    for t in TESTS:
        # one mistake per word, so every catch is attributable
        keys = [(m["ayah"], m["index"]) for m in res[t.id]]
        assert len(keys) == len(set(keys)), t.id


def _report(faulted: dict[tuple[int, int], list[dict]]) -> dict:  # type: ignore[type-arg]
    ayahs = {}
    for (a, w), fs in faulted.items():
        ayahs.setdefault(a, {"ayah": a, "words": [{"index": i, "faults": []} for i in range(8)]})
        ayahs[a]["words"][w]["faults"] = fs
    return {"ayahs": list(ayahs.values()), "summary": {"accuracy": 0.5}}


def test_scorecard_separates_caught_near_missed_and_false_alarms() -> None:
    t = Test("x", "x", 1, (1, 1), "", "", ())
    script = [{"ayah": 1, "index": 1, "word": "a", "catch": ["ghonna"]},
              {"ayah": 1, "index": 4, "word": "b", "catch": ["letter"]},
              {"ayah": 1, "index": 6, "word": "c", "catch": ["madd_tabii"]}]
    rep = _report({(1, 1): [{"type": "sifah", "sifah": "ghonna"}],       # caught, right kind
                   (1, 5): [{"type": "letter", "letter": "ص"}],          # next to #2: near
                   (1, 7): [], (1, 0): [{"type": "rule", "rule": "tafkheem"}]})  # false alarm
    object.__setattr__(t, "b_is_correct", False)
    c = scorecard(t, "mistakes", rep, script)
    assert (c["caught"], c["near"], c["missed"], c["right_kind"]) == (1, 1, 1, 1)
    assert [a["word_index"] for a in c["false_alarms"]] == [0]


def test_on_a_correct_take_every_fault_is_a_false_alarm() -> None:
    t = TESTS[0]
    c = scorecard(t, "correct", _report({(1, 2): [{"type": "rule", "rule": "madd_tabii"}]}), [])
    assert c["scripted"] == 0 and c["false_alarm_words"] == 1
