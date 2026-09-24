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
