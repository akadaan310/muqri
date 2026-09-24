"""The blind-spot predictor's numpy path reproduces the Julia fit (substrate_library/julia/blindspots.jl)."""

from __future__ import annotations

import json

import pytest

from app.blindspots import MODEL, model, probability


@pytest.mark.skipif(not MODEL.is_file(), reason="run blindspots.jl")
def test_predictions_match_julia() -> None:
    d = json.loads(MODEL.read_text())
    assert len(d["parity"]) > 50
    for row in d["parity"]:
        assert probability(row["check"], row["features"]) == pytest.approx(row["p"], rel=1e-9, abs=1e-12)
    assert 0 < model()["tau"] < 1
