"""Calibration sessions: each take is scored against what it should produce."""

from __future__ import annotations

from app.sessions import exercises, score


def _m(rules, letters=(), failing_words=()):  # type: ignore[no-untyped-def]
    words = [{"ref": f"83:32:{i}", "text": f"w{i}", "all_correct": i not in dict(failing_words),
              "failing": dict(failing_words).get(i, [])} for i in range(6)]
    return {"recording": {"seconds_per_count": 0.24, "tempo_class": "tadwir", "wajh": {"choice": "tawassut"}},
            "rules": list(rules), "letters": list(letters), "words": words}


def _r(j, word, rule, status="pass", counts=None, dev=0.0):  # type: ignore[no-untyped-def]
    return {"id": f"83:32:R{j}", "word": word, "rule": rule, "status": status, "observed_counts": counts,
            "deviation_counts": dev, "z_masters": None}


def test_take_a_meets_its_spec_and_names_what_does_not() -> None:
    ex = exercises()["r1e1"]
    m = _m([_r(0, 0, "madd_tabii", counts=2.0), _r(1, 2, "madd_munfasil", counts=4.2),
            _r(2, 3, "ghunnah", "long", 3.4, 0.4), _r(3, 4, "madd_muttasil", counts=6.2),
            _r(4, 4, "madd_muttasil", counts=4.4), _r(5, 5, "madd_lazim", counts=6.5)],
           failing_words=[(3, ["83:32:R2"])])
    c = score(ex, "A", m)
    v = {(e["rule"], e["measured"]): e["verdict"] for e in c["expectations"]}
    assert v[("madd_munfasil", 4.2)] == "ok" and v[("madd_lazim", 6.5)] == "ok"
    assert v[("ghunnah", 3.4)] == "long" and v[("madd_muttasil", 6.2)] == "out of range"
    assert c["tempo"]["ok"] and [f["word"] for f in c["false_alarms"]] == [3]


def test_take_b_caught_only_with_the_right_signature() -> None:
    ex = exercises()["r1e1"]
    m = _m([_r(1, 2, "madd_munfasil", "short", 2.1, -1.9), _r(3, 4, "madd_muttasil", "pass", 4.3),
            _r(5, 5, "madd_lazim", "long", 8.0, 2.0)],
           failing_words=[(2, ["83:32:R1"]), (5, ["83:32:R5"]), (1, ["83:32:L3:identity"])])
    c = score(ex, "B", m)
    got = {r["word"]: r["verdict"] for r in c["mistakes"]}
    assert got == {2: "caught", 3: "missed", 4: "missed", 5: "flagged, other reason"}
    assert [(f["word"], f["control"]) for f in c["false_alarms"]] == [(1, True)]


def test_the_pages_scripts_parse() -> None:
    """A stray quote once blanked the whole sessions page ('Loading...' forever): every page's script
    must at least parse."""
    import shutil
    import subprocess
    import tempfile

    import pytest
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    from app.protocol_page import PROTOCOL_PAGE
    from app.sessions_page import SESSIONS_PAGE
    for page in (SESSIONS_PAGE, PROTOCOL_PAGE):
        js = page[page.index("<script>") + 8:page.rindex("</script>")]
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
            fh.write(js)
        run = subprocess.run([node, "--check", fh.name], capture_output=True, text=True)
        assert run.returncode == 0, run.stderr[-500:]


def test_consecutive_ayahs_never_share_audio_and_meet_at_the_pause() -> None:
    """al-Fatihah 1:7 (صِرَٰطَ ...) was aligned onto 1:6's ٱلصِّرَٰطَ, overlapping it by 226 frames."""
    from app.submission import settle_boundaries
    spans = [(14, 243), (221, 481), (255, 849)]                  # the certified reciter's take A
    pauses = [(0, 14), (123, 129), (202, 221), (249, 256), (300, 307), (373, 399), (854, 873)]
    assert settle_boundaries(spans, pauses) == [(14, 202), (221, 373), (399, 849)]
    assert settle_boundaries([(0, 100), (80, 200)]) == [(0, 90), (90, 200)]   # no audio: split the overlap
    assert settle_boundaries([(0, 100), (120, 200)]) == [(0, 100), (120, 200)]  # already apart: untouched


def test_every_exercise_points_at_real_words() -> None:
    """Each expectation, mistake and control names a word that exists in its ayah."""
    from app.engine import Engine
    e = Engine()
    for ex in exercises().values():
        n = {a: len(e.reference(ex.surah, a).uthmani.split()) for a in range(ex.ayahs[0], ex.ayahs[1] + 1)}
        for a, w in [(x.ayah, x.word) for x in ex.expect] + [(m.ayah, m.word) for m in ex.mistakes] + list(ex.controls):
            assert a in n and 0 <= w < n[a], (ex.id, a, w)


def test_listening_answers_are_saved_as_expert_labels(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from app import review
    from app.sessions import LISTEN, listen_json
    monkeypatch.setattr(review, "LABELS", tmp_path / "labels.jsonl")
    monkeypatch.setattr(review, "REVIEW_DIR", tmp_path)
    q = listen_json(3, {})
    assert {x["ref"] for x in q} >= {"54:2", "54:3", "54:19", "54:38"}
    assert sum(len(x["clips"]) for x in q) == sum(len(l.reciters) for l in LISTEN[3])
    cid = q[0]["clips"][0]["id"]
    review.add_label(cid, "no", "round 3 listening")
    again = listen_json(3, review.labels())
    assert again[0]["clips"][0]["answer"] == "no" and again[0]["clips"][1]["answer"] is None
