"""HTTP API smoke test (skipped unless fastapi is installed)."""

from __future__ import annotations

import pytest

from app.pipeline import AnalysisOptions
from tests.test_pipeline import synthesize

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")


def test_analyze_endpoint(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from fastapi.testclient import TestClient

    from app.api import create_app

    wav, align = synthesize(tmp_path)
    client = TestClient(create_app(AnalysisOptions(alignment_json=align, index_dir=None, timbre_backend="mfcc",
                                                   denoise="never", calibration=None)))
    assert client.get("/health").json() == {"status": "ok"}
    with wav.open("rb") as fh:
        resp = client.post("/analyze", files={"audio": ("r.wav", fh, "audio/wav")}, data={"surah": 113, "ayah": 2})
    assert resp.status_code == 200, resp.text
    assert resp.json()["recitation_summary"]["reference"] == "113:2"
    bad = client.post("/analyze", files={"audio": ("r.wav", b"not audio", "audio/wav")}, data={"surah": 113, "ayah": 2})
    assert bad.status_code == 422
