"""End-to-end pipeline and CLI on synthetic audio with a known alignment."""

from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

import main
from app.pipeline import AnalysisOptions, QaariEvaluator
from app.profiling import MfccStatsEmbedder, ReciterIndex, ReciterProfile, StyleVector
from app.tajweed_rules import parse_text
from tests.synth import SR, burst, concat, nasal_murmur, silence, vowel

TEXT = "مِن شَرِّ مَا خَلَقَ"  # Al-Falaq 113:2
HARAKA = 0.2


def synthesize(tmp_path):  # type: ignore[no-untyped-def]
    """Render one synthetic segment per pronounced unit and the matching alignment JSON."""
    parsed = parse_text(TEXT)
    pron = [u for u in parsed.units if u.pronounced]
    parts, rows, t = [silence(0.2)], [], 0.2
    for k, u in enumerate(pron):
        if u.char == "ن":  # ikhfa: 2-count nasal murmur
            seg = nasal_murmur(2 * HARAKA)
        elif u.madd_letter:  # madd tabi'i: carrier (1) + madd letter (1) = 2 counts
            seg = vowel(HARAKA, seed=k)
        elif u.char == "ق" and k == len(pron) - 1:  # qalqalah kubra: closure + burst + echo
            seg = concat(silence(0.06), burst(), vowel(0.06, (500, 1500, 2500), amp=0.25), silence(0.07))
        else:
            seg = vowel(HARAKA, (700, 1150, 2500) if u.char == "خ" else (700, 1750, 2600), seed=k)
        rows.append({"unit_index": u.index, "start_ms": t * 1000, "end_ms": (t + len(seg) / SR) * 1000})
        parts.append(seg)
        t += len(seg) / SR
    parts.append(silence(0.2))
    wav = tmp_path / "recitation.wav"
    sf.write(wav, concat(*parts), SR)
    align = tmp_path / "alignment.json"
    align.write_text(json.dumps({"units": rows}))
    return wav, align


@pytest.fixture
def index_dir(tmp_path):  # type: ignore[no-untyped-def]
    pytest.importorskip("faiss")
    rng = np.random.default_rng(0)
    idx = ReciterIndex(MfccStatsEmbedder.backend)
    for i, (name, style) in enumerate({
        "Mahmoud Khalil Al-Hussary": StyleVector(300, 1.0, 1.0, 20.0),
        "Siddiq Al-Minshawi": StyleVector(240, 1.3, 3.0, 12.0),
        "Mishary Alafasy": StyleVector(330, 0.9, 4.5, 8.0),
    }.items()):
        timbre = rng.standard_normal(128).astype(np.float32)
        idx.add(ReciterProfile(f"r{i}", name, timbre / np.linalg.norm(timbre), style, MfccStatsEmbedder.backend))
    out = tmp_path / "index"
    idx.save(out)
    return out


def test_pipeline_report_matches_spec(tmp_path, index_dir) -> None:  # type: ignore[no-untyped-def]
    wav, align = synthesize(tmp_path)
    ev = QaariEvaluator(AnalysisOptions(alignment_json=align, index_dir=index_dir, timbre_backend="mfcc",
                                        denoise="never"))
    result = ev.analyze_file(wav, surah=113, ayah=2)
    report = result.report

    summary = report["recitation_summary"]
    assert summary["reference"] == "113:2"
    assert summary["base_haraka_duration_ms"] == pytest.approx(200, abs=1)
    assert summary["tempo_bpm_harakat"] == pytest.approx(300, abs=2)
    assert summary["total_rules_evaluated"] >= 3
    assert summary["overall_tajweed_score"] is not None

    diags = {d["rule_type"]: d for d in report["detailed_rule_diagnostics"]}
    assert diags["madd_tabii"]["measured_harakat"] == pytest.approx(2.0, abs=0.05)
    assert diags["madd_tabii"]["status"] == "PASS"
    assert diags["ikhfa"]["measured_harakat"] == pytest.approx(2.0, abs=0.05)
    assert diags["ikhfa"]["status"] == "PASS"
    assert diags["qalqalah"]["status"] == "PASS"
    for d in report["detailed_rule_diagnostics"]:
        assert set(d) >= {"rule_type", "word", "location", "status", "feedback"}
        assert d["location"]["end_ms"] >= d["location"]["start_ms"]

    matches = report["reciter_similarity_match"]["top_matches"]
    assert len(matches) == 3
    assert matches[0]["reciter_name"] == "Mahmoud Khalil Al-Hussary"  # closest style (tempo 300, bias 1.0)
    assert {"reciter_name", "style_similarity_pct", "timbre_similarity_pct", "matched_traits"} <= set(matches[0])
    json.dumps(report, ensure_ascii=False)  # must be serialisable


def test_cli_analyze_writes_report(tmp_path, index_dir, capsys) -> None:  # type: ignore[no-untyped-def]
    wav, align = synthesize(tmp_path)
    out = tmp_path / "analysis_report.json"
    code = main.main([
        "analyze", "--audio", str(wav), "--surah", "113", "--ayah", "2", "--alignment-json", str(align),
        "--index-dir", str(index_dir), "--timbre-backend", "mfcc", "--denoise", "never", "--output", str(out),
    ])
    assert code == 0
    printed = capsys.readouterr().out
    assert "Overall Tajweed score" in printed and "madd_tabii" in printed
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["recitation_summary"]["reference"] == "113:2"


def test_cli_reports_errors(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    assert main.main(["analyze", "--audio", str(tmp_path / "nope.wav"), "--surah", "113", "--ayah", "2"]) == 1
    assert "not found" in capsys.readouterr().err
    assert main.main(["analyze", "--audio", "x.wav"]) == 2
    assert main.main(["rules", "--surah", "113", "--ayah", "1"]) == 0
    assert "qalqalah" in capsys.readouterr().out
