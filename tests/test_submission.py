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
        # the scale is fitted, so a length verdict is stated in tajweed counts against what the rule
        # requires -- "expected 2, given 1.71" -- and says how far it can be trusted
        assert v["evidence"]["given_counts"] is not None
        assert v["evidence"]["expected"] == v["expected_counts"]
        assert v.get("confidence") in {"validated", "unvalidated"}
    # the six-count level is the only one the fit does not recover, so it alone stays flagged
    six = [v for v in durational if sum(v["expected_counts"]) / 2 == 6.0]
    assert all(v["confidence"] == "unvalidated" for v in six)
    common = [v for v in durational if sum(v["expected_counts"]) / 2 in (1.5, 2.0, 4.0, 4.5)]
    assert common and all(v["confidence"] == "validated" for v in common)


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


def test_ghunnah_is_graded_by_the_four_maratib(passage, engine) -> None:  # type: ignore[no-untyped-def]
    """Maratib al-Ghunnah: akmal > kamilah > naqisah > anqas, graded by structural context.

    Measured on the anchor: akmal 2.20 counts, kamilah 2.43, naqisah 1.41. Akmal and kamilah are both
    the full two counts — the treatise separates them by oral articulation, not by length — so only
    naqisah is asserted to be shorter.
    """
    lay, hus, big = passage
    rep = engine.analyze(None, [(r["sura"], r["aya"]) for r in hus], posteriors=big)
    grades = rep["ghunnah_grades"]
    assert grades, "a passage must contain nasalisation to grade"
    assert set(grades) <= {"akmal", "kamilah", "naqisah", "anqas"}
    for g, v in grades.items():
        assert v["n"] > 0 and 0.0 <= v["accuracy"] <= 1.0
        assert v["median_counts"] is not None, f"{g} graded without a measured hold"
    if {"akmal", "naqisah"} <= set(grades):
        assert grades["naqisah"]["median_counts"] < grades["akmal"]["median_counts"], \
            "a clear sakin nun is held for less than a doubled one"


def test_letter_strength_reports_quwwa(passage, engine) -> None:  # type: ignore[no-untyped-def]
    """A letter's strength is the sum of the strong sifat it carries, not one attribute."""
    lay, hus, big = passage
    rep = engine.analyze(None, [(r["sura"], r["aya"]) for r in hus], posteriors=big)
    st = rep["letter_strength"]
    assert st["strong_sifat_expected"] > 0
    assert 0.0 <= st["ratio"] <= 1.0
    assert st["realised"] <= st["strong_sifat_expected"]
    per_letter = [l["strength"] for a in rep["ayahs"] for l in a["letters"]]
    assert any(s["expected"] for s in per_letter), "some letters must carry strong sifat"
    for s in per_letter:
        assert s["realised"] <= s["expected"]


def test_part_of_an_ayah_is_graded_with_the_ayahs_word_indices(passage, engine) -> None:  # type: ignore[no-untyped-def]
    """A learner may recite any stretch of words. Cut a master's recording where a word starts and
    grade the rest as a slice: the words keep their places in the ayah, and the slice grades as
    cleanly as the whole."""
    lay, hus, _big = passage
    r = max(hus, key=lambda x: len(x["uthmani"].split()))
    n = len(r["uthmani"].split())
    lp = np.fromfile(DUMP / r["file"], dtype="<f4").reshape(r["frames"], lay["columns"])
    whole = engine.analyze(None, [(r["sura"], r["aya"])], posteriors=lp)
    same = engine.analyze(None, [(r["sura"], r["aya"], 0, n - 1)], posteriors=lp)
    assert same["measurements"]["words"] == whole["measurements"]["words"], "the full range is the ayah"

    w0 = n // 2
    onset = min(l["onset_s"] for l in whole["measurements"]["letters"] if l["word"] == w0)
    cut = lp[max(0, int(onset / 0.04) - 3):]
    part = engine.analyze(None, [(r["sura"], r["aya"], w0, n - 1)], posteriors=cut)
    m = part["measurements"]
    assert [w["ref"] for w in m["words"]] == [f"{r['sura']}:{r['aya']}:{i}" for i in range(w0, n)]
    assert all(l["word"] >= w0 for l in m["letters"] if l["word"] is not None)
    assert part["ayahs"][0]["word_offset"] == w0
    ok_whole = whole["measurements"]["summary"]["words_all_correct"] / len(whole["measurements"]["words"])
    ok_part = m["summary"]["words_all_correct"] / len(m["words"])
    assert ok_part >= ok_whole - 0.25, (ok_part, ok_whole)

    with pytest.raises(ValueError):
        engine.analyze(None, [(r["sura"], r["aya"], 0, n)], posteriors=lp)


def test_measurements_follow_the_published_schema(passage, engine) -> None:  # type: ignore[no-untyped-def]
    """measurements/1 is the consumer apps' API: every report must validate against the schema file
    they are given, for one ayah, a passage, and part of an ayah."""
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((ROOT / "app/schemas/measurements-1.schema.json").read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    v = jsonschema.Draft202012Validator(schema)
    lay, hus, big = passage
    r = hus[0]
    lp = np.fromfile(DUMP / r["file"], dtype="<f4").reshape(r["frames"], lay["columns"])
    n = len(r["uthmani"].split())
    for verses, post in (([(r["sura"], r["aya"])], lp), ([(x["sura"], x["aya"]) for x in hus], big),
                         ([(r["sura"], r["aya"], 1, n - 1)], lp)):
        m = engine.analyze(None, verses, posteriors=post)["measurements"]
        errs = sorted(v.iter_errors(m), key=lambda e: list(e.path))
        assert not errs, [f"{list(e.path)}: {e.message}" for e in errs[:5]]
        # a realised characteristic was heard as the expected class
        for l in m["letters"]:
            for c in l["characteristics"].values():
                assert (c["observed"] == c["expected"]) == c["realised"] or c["expected"] == c["competitor"]


def test_a_declared_wajh_is_graded_as_declared(engine) -> None:  # type: ignore[no-untyped-def]
    """An app declares the munfasil length its learner follows. Inferred from the recording alone, a
    munfasil read short on purpose is taken for the qasr wajh and passes; declared, it is graded."""
    lay = json.loads((DUMP / "layout.json").read_text())
    for line in (DUMP / "index.jsonl").open():
        r = json.loads(line)
        if r.get("speaker") != "Husary_128kbps" or "file" not in r:
            continue
        lp = np.fromfile(DUMP / r["file"], dtype="<f4").reshape(r["frames"], lay["columns"])
        rep = engine.analyze(None, [(r["sura"], r["aya"])], posteriors=lp)
        mun = [x for x in rep["measurements"]["rules"] if x["rule"] == "madd_munfasil"
               and x["observed_counts"] and x["observed_counts"] >= 3.5]
        if mun:
            break
    else:
        pytest.skip("no tawassut munfasil in the Husary clips present")
    assert rep["wajh"]["source"] == "inferred" and rep["wajh"]["choice"] == "tawassut"
    as_qasr = engine.analyze(None, [(r["sura"], r["aya"])], posteriors=lp, wajh="qasr")
    got = {x["id"]: x for x in as_qasr["measurements"]["rules"]}
    assert as_qasr["wajh"]["source"] == "declared"
    assert all(got[m["id"]]["status"] == "long" and got[m["id"]]["deviation_counts"] > 0 for m in mun)
    as_taw = engine.analyze(None, [(r["sura"], r["aya"])], posteriors=lp, wajh="tawassut")
    assert [x["status"] for x in as_taw["measurements"]["rules"]] == [x["status"] for x in rep["measurements"]["rules"]]
    with pytest.raises(ValueError):
        engine.analyze(None, [(r["sura"], r["aya"])], posteriors=lp, wajh="madd")
