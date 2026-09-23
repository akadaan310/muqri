"""Benchmark tooling (strategic verses, aggregation, index build) and the v2 acceptance criteria.

The acceptance tests read ``benchmarks/results/summary.json`` produced by
``benchmarks/run_benchmark.py`` + ``benchmarks/summarize.py`` on real recordings. They are opt-in
(``QAARI_ACCEPTANCE=1``) because they need the benchmark run, and they report the measured numbers
rather than being tuned to pass.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from app.fingerprint import TAJWEED_DIM, TIMBRE_DIM, ReciterIndex
from benchmarks import summarize

ROOT = Path(__file__).resolve().parent.parent
SUMMARY = ROOT / "benchmarks" / "results" / "summary.json"
STRATEGIC = ROOT / "app" / "data" / "strategic_verses.json"


def test_strategic_verses_cover_every_rule_key() -> None:
    data = json.loads(STRATEGIC.read_text(encoding="utf-8"))
    per_key = data["per_key"]
    short = {k: v for k, v in data["coverage"].items() if v["selected"] < min(per_key, v["in_quran"])}
    assert not short
    keys = set(data["coverage"])
    for must in ("sakt", "qalqalah:akbar", "madd_lazim:harfi", "idgham_mutajanisayn:naqis", "jawaz_wajhayn",
                 "izhar_halqi", "iqlab", "madd_silah_kubra", "takreer", "istitaalah"):
        assert any(k.startswith(must) for k in keys), must
    assert len(data["verses"]) <= 80


def _row(reciter: str, category: str, mode: str, statuses: list[str], seed: int) -> dict:  # type: ignore[type-arg]
    rng = np.random.default_rng(seed)
    timbre = rng.standard_normal(TIMBRE_DIM)
    return {
        "reciter": reciter, "name": reciter.title(), "category": category, "mode": mode, "surah": 1, "ayah": 1,
        "haraka_ms": 200.0,
        "diagnostics": [{"rule_type": "madd_tabii", "status": s, "score": {"PASS": 1.0, "WARNING": 0.6}.get(s, 0.2)}
                        for s in statuses] + [{"rule_type": "hams", "status": "PASS", "score": 1.0}],
        "fingerprint": None if mode != "studio" else {
            "timbre": (timbre / np.linalg.norm(timbre)).tolist(), "tajweed": [0.5] * (TAJWEED_DIM - 1) + [None],
            "environment": [0.0] * 8, "backend": "test-backend"},
    }


def test_aggregate_counts_timing_fails_and_excludes_pauses() -> None:
    agg = summarize.aggregate(_row("x", "taraweeh", "studio", ["PASS", "FAIL", "VALID_NECESSARY_PAUSE"], 0)
                              ["diagnostics"])
    assert agg["timing_rules"] == 2 and agg["timing_fails"] == 1
    assert agg["sifaat"] == pytest.approx(100.0)
    assert agg["perfection"] == pytest.approx(60.0)


def test_summarize_builds_both_indices(tmp_path: Path) -> None:
    rows = [_row("husary", "studio", "studio", ["PASS"], 1), _row("dusari", "taraweeh", "studio", ["FAIL"], 2),
            _row("dusari", "taraweeh", "taraweeh_adapted", ["PASS"], 2)]
    runs = tmp_path / "runs.jsonl"
    runs.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    assert summarize.main(["--runs", str(runs), "--out-dir", str(tmp_path), "--index-dir", str(tmp_path)]) == 0
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["aggregate"]["taraweeh_timing_fp_reduction_pct"] == 100.0
    assert (tmp_path / "masterclass_reciters.faiss").exists() and (tmp_path / "taraweeh_reciters.faiss").exists()
    assert ReciterIndex.load(tmp_path, "taraweeh_reciters").find("Dossary") is not None


# -- acceptance (opt-in) -------------------------------------------------------------------------
acceptance = pytest.mark.skipif(
    os.environ.get("QAARI_ACCEPTANCE") != "1" or not SUMMARY.exists(),
    reason="set QAARI_ACCEPTANCE=1 after running the benchmark (benchmarks/README in the main README)",
)


def _reciter(summary: dict, name: str) -> dict:  # type: ignore[type-arg]
    key = name.lower()
    return next(r for r in summary["reciters"] if key in r["reciter"].lower() or key in r["name"].lower())


@acceptance
def test_hussary_perfection_at_least_98() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    score = _reciter(summary, "Husary_128")["modes"]["studio"]["perfection"]
    assert score >= 98.0, f"Al-Hussary perfection {score:.1f}"


@acceptance
def test_dosari_adapted_at_least_95() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    score = _reciter(summary, "Dussary")["modes"]["taraweeh_adapted"]["perfection"]
    assert score >= 95.0, f"Ad-Dosari adapted perfection {score:.1f}"


@acceptance
def test_adapter_reduces_timing_false_positives_by_90pct() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    red = summary["aggregate"]["taraweeh_timing_fp_reduction_pct"]
    assert red is not None and red >= 90.0, f"FP reduction {red}%"
