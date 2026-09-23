"""End-to-end: a submission at any scale in, a full mastery report out.

Runs against the real T300 posteriors when they are present. Skips on a clean checkout, where the
1.8 GB dump is gitignored.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[1]
DUMP = ROOT / "research_agency_lab/experiments/qaari_keys/modal_T300"

pytestmark = pytest.mark.skipif(not (DUMP / "index.jsonl").is_file(),
                                reason="T300 dump not present (gitignored, 1.8 GB)")


@pytest.fixture(scope="module")
def passage():  # type: ignore[no-untyped-def]
    """Consecutive ayahs from one reciter, concatenated into a single recording."""
    lay = json.loads((DUMP / "layout.json").read_text())
    recs = [json.loads(l) for l in (DUMP / "index.jsonl").open()]
    # skip the muqatta'at (2:1 is alif-laam-meem): the phonetizer spells the letter NAMES as one
    # word while the parser counts three, which is its own case and covered by test_rule_bind
    hus = [r for r in sorted((r for r in recs
                              if r.get("speaker") == "Husary_Muallim_128kbps" and "file" in r),
                             key=lambda r: (r["sura"], r["aya"]))
           if len(r.get("uthmani", "")) > 12][:6]
    mats = [np.fromfile(DUMP / r["file"], dtype="<f4").reshape(r["frames"], lay["columns"])
            for r in hus]
    return lay, hus, np.concatenate(mats, axis=0)


@pytest.fixture(scope="module")
def engine(passage):  # type: ignore[no-untyped-def]
    from app.engine import Engine
    lay, _hus, _big = passage
    return Engine(layout=lay)


def test_single_ayah_report_is_complete(passage, engine) -> None:  # type: ignore[no-untyped-def]
    lay, hus, _big = passage
    r = hus[0]
    lp = np.fromfile(DUMP / r["file"], dtype="<f4").reshape(r["frames"], lay["columns"])
    rep = engine.analyze(None, [(r["sura"], r["aya"])], posteriors=lp)

    assert rep["schema_version"]
    s = rep["summary"]
    assert s["ayahs"] == 1
    assert s["letters"] > 20, "a verse should yield tens of letters"
    # every letter carries identity + timing, every consonant ten sifat: hundreds of judgments
    assert s["judgments"] > 5 * s["letters"]
    assert s["rules_located"] > 5, "the parser locates many rules in any verse"
    assert 0.0 <= s["accuracy"] <= 1.0

    letter = rep["ayahs"][0]["letters"][0]
    assert {"symbol", "kind", "onset_s", "duration_s", "identity", "sifat"} <= set(letter)
    assert {"gop", "heard_instead", "confirmed"} <= set(letter["identity"])


def test_rules_are_named_with_what_they_require(passage, engine) -> None:  # type: ignore[no-untyped-def]
    """The point of the join: not "a madd error" but which madd, and what it requires."""
    lay, hus, _big = passage
    r = hus[0]
    lp = np.fromfile(DUMP / r["file"], dtype="<f4").reshape(r["frames"], lay["columns"])
    rep = engine.analyze(None, [(r["sura"], r["aya"])], posteriors=lp)
    rules = rep["ayahs"][0]["rules"]
    assert rules
    for v in rules:
        assert v["rule"] and v["word"], "every verdict names its rule and the word it sits in"
        assert v["mechanism"] in {"durational", "attribute", "segmental"}
        assert v["status"] in {"pass", "short", "long", "wrong", "unconfirmed", "no_evidence"}
    durational = [v for v in rules if v["mechanism"] == "durational" and v["expected_counts"]]
    assert durational, "a verse should contain at least one length-governed rule"
    for v in durational:
        # the absolute count scale is not calibrated, so these must be flagged, never silently shipped
        assert v.get("confidence") == "unvalidated"


def test_passage_segmentation_does_not_drift(passage, engine) -> None:  # type: ignore[no-untyped-def]
    """Each ayah is anchored to its global proportional position, so error cannot accumulate.

    Sequential walking drifted monotonically — 2,530 frames out by ayah 6, 150 phantom errors, report
    accuracy 0.55. The proportional anchor removes the accumulation: measured boundary errors on a
    six-ayah passage are [61, 868, 618, 332, 1096, 88], i.e. they wander but the walk recovers, and
    report accuracy rises to 0.93. Individual boundaries can still be several hundred frames out, so
    edge-of-ayah findings are the least reliable part of a long submission.
    """
    lay, hus, big = passage
    rep = engine.analyze(None, [(r["sura"], r["aya"]) for r in hus], posteriors=big)
    assert rep["summary"]["ayahs"] == len(hus)

    spans = [a["frames"] for a in rep["ayahs"]]
    assert all(b > a for a, b in spans), "every ayah must claim a non-empty span"
    assert spans == sorted(spans), "spans must advance through the recording"
    assert spans[-1][1] <= big.shape[0]

    true_ends = list(itertools.accumulate(r["frames"] for r in hus))
    errs = [abs(s[1] - t) for s, t in zip(spans, true_ends)]
    # the signature of drift is error growing toward the end; that is what must not happen
    assert errs[-1] <= max(errs), f"error grows to the end: {errs}"
    assert errs[-1] < 0.5 * max(errs) + 200, f"boundary error accumulating: {errs}"


def test_cross_ayah_mastery_statistics(passage, engine) -> None:  # type: ignore[no-untyped-def]
    """Taswiyah and tempo are properties of the whole submission, not of one clip."""
    lay, hus, big = passage
    rep = engine.analyze(None, [(r["sura"], r["aya"]) for r in hus], posteriors=big)
    m = rep["mastery"]
    assert m["tempo_haraka_s"] and 0.05 < m["tempo_haraka_s"] < 0.8
    assert m["tempo_mode"] in {"tahqiq", "tadwir", "hadr"}
    assert m["taswiyah"], "a passage must yield held-length classes for the consistency statistic"
    for cls, v in m["taswiyah"].items():
        assert v["n"] >= 4 and v["cv"] is not None, f"{cls} reported without enough instances"


def test_rule_practice_filter(passage, engine) -> None:  # type: ignore[no-untyped-def]
    """Practising one rule returns a report restricted to it."""
    lay, hus, big = passage
    rep = engine.analyze(None, [(r["sura"], r["aya"]) for r in hus], posteriors=big,
                         rule_filter="madd")
    assert rep["by_rule"], "no madd rules found to practise"
    assert all(k.startswith("madd") for k in rep["by_rule"])
    assert all(v["rule"].startswith("madd") for a in rep["ayahs"] for v in a["rules"])
