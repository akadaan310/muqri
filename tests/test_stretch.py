"""The timing calculus' numpy path reproduces the Julia fit (substrate_library/julia/stretch.jl)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.stretch import judge, model, unit_of

FIX = Path(__file__).resolve().parents[1] / "research_agency_lab/experiments/timing/stretch_parity.json"


@pytest.mark.skipif(not FIX.is_file(), reason="run stretch.jl to write the parity fixture")
def test_unit_and_stretch_match_julia() -> None:
    fx = json.loads(FIX.read_text())
    assert len(fx) > 20
    for f in fx:
        u = unit_of(f["plain"])
        assert u == pytest.approx(f["unit"], rel=1e-9)
        assert f["seconds"] / u == pytest.approx(f["stretch"], rel=1e-9)


@pytest.mark.skipif(not model(), reason="no stretch model")
def test_a_stretch_is_judged_against_the_tempo_it_was_recited_at() -> None:
    m = model()
    t = next(x for x in m["tempo_norms"] if x["rule"] == "madd_lazim")
    u = m["U0"]
    typical = judge("madd_lazim", u * __import__("math").exp(t["a"]), u)
    assert typical["verdict"] == "pass" and abs(typical["z"]) < 1e-6 and typical["equivalent_counts"] == 6.0
    assert judge("madd_lazim", u * 2.0, u)["verdict"] == "short"            # a six-count madd held for 2 units
    # 'arid at a stop: all three levels legitimate, and graded short only
    assert judge("madd_arid_lissukun", u * 30, u)["verdict"] == "pass"
    assert judge("madd_arid_lissukun", u * 0.4, u)["verdict"] == "short"
