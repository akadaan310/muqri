"""The listening review: what is proposed, in what order, and how verdicts become precision."""

from __future__ import annotations

import json

import app.review as review


def _report() -> dict:  # type: ignore[type-arg]
    letters = [{"i": i, "symbol": s, "kind": "consonant", "word": w, "onset_s": 0.1 * i, "duration_s": 0.1,
                "identity": {"confirmed": True, "llr": -5, "heard_instead": ""}, "sifat": {}}
               for i, (s, w) in enumerate([("ا", 0), ("ن", 1), ("ر", 2)])]
    letters[1]["sifat"] = {"ghonna": {"expected": "[مغن]", "model_best": "[لا غنة]", "llr": -9.0, "realised": False},
                           "tikraar": {"expected": "x", "model_best": "y", "llr": -9.0, "realised": False}}
    letters[2]["sifat"] = {"tafkheem_or_taqeeq": {"expected": "a", "model_best": "b", "llr": -0.5, "realised": False}}
    rules = [
        {"rule": "madd_tabii", "mechanism": "durational", "status": "long", "word_index": 0, "word": "w0",
         "units": [0], "expected_counts": [2, 2], "evidence": {"given_counts": 5.5}},      # proposed
        {"rule": "madd_tabii", "mechanism": "durational", "status": "long", "word_index": 2, "word": "w2",
         "units": [2], "expected_counts": [2, 2], "evidence": {"given_counts": 3.0}},      # calibration
    ]
    return {"ayahs": [{"surah": 1, "ayah": 1, "frames": [10, 90], "haraka_s": 0.2, "letters": letters,
                       "rules": rules, "words": [{"index": i, "word": f"w{i}"} for i in range(3)]}]}


def test_only_real_departures_are_proposed() -> None:
    cands = review.mine(_report(), "r")
    kinds = sorted((c["kind"], c["detector"]) for c in cands)
    # the 5.5-count madd, the lost ghunnah, and the 5.5 vs 3.0 inconsistency; NOT the 3.0-count madd
    # (a count off is calibration), NOT takrir (must be concealed), NOT the weak tafkhim evidence
    assert kinds == [("characteristic", "sifah:ghonna"), ("consistency", "consistency:madd_tabii"),
                     ("length", "length:madd_tabii")]
    length = next(c for c in cands if c["kind"] == "length")
    assert length["start_s"] == 0.4 and length["magnitude"] == 2.75   # 10 frames in; 5.5 - (2 + 0.75)


def test_queue_skips_labelled_and_verdicts_become_precision(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(review, "CANDIDATES", tmp_path / "c.jsonl")
    monkeypatch.setattr(review, "LABELS", tmp_path / "l.jsonl")
    monkeypatch.setattr(review, "REVIEW_DIR", tmp_path)
    cands = review.mine(_report(), "r")
    review.CANDIDATES.write_text("".join(json.dumps(c) + "\n" for c in cands))
    first = review.queue(10)
    assert len(first) == 3 and len({c["detector"] for c in first}) == 3
    review.add_label(first[0]["id"], "yes", "clear")
    review.add_label(first[1]["id"], "no")
    assert len(review.queue(10)) == 1
    s = review.stats()
    assert s["labelled"] == 2
    assert s["detectors"][first[0]["detector"]]["precision"] == round(2 / 3, 3)   # Beta(1,1) posterior
