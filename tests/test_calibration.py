"""Reference-reciter calibration: robust bands, verdicts, reliability filtering, offline re-scoring."""

from __future__ import annotations

import pytest

from app.calibration import REVIEW_SCORE, Calibration, MetricBand, rule_key, verdict
from app.models import RuleDiagnostic, RuleType, Status
from app.scoring import summarize_diagnostics
from benchmarks import summarize

CAL = Calibration.from_dict({
    "count_scale": 0.8,
    "reliability": {"align_conf_min": 0.3, "align_min_ms": 40.0},
    "rules": {
        "madd_tabii": [{"metric": "core_counts", "lo": 1.8, "hi": 2.1, "scale": 0.2, "side": "both"}],
        "izhar_shafawi": [{"metric": "counts", "lo": 1.0, "hi": 1.0, "scale": 0.2, "side": "upper"}],
    },
    "category_weights": {"madd": 2.0},
})


def diag(rule: RuleType, status: Status = Status.FAIL, **metrics: float) -> RuleDiagnostic:
    return RuleDiagnostic(rule_type=rule, word="w", start_ms=0, end_ms=500, status=status, feedback="raw",
                          score=0.1, measured_harakat=metrics.get("counts"), metrics=dict(metrics))


def test_rule_keys() -> None:
    assert rule_key("ikhfa", "noon sakinah; light") == "ikhfa:light"
    assert rule_key("tafkheem", "raa: fathah/dammah", "ر") == "tafkheem:raa"
    assert rule_key("madd_lazim", "harfi: two-letter name") == "madd_lazim:harfi"
    assert rule_key("madd_tabii", "natural madd") == "madd_tabii"


def test_band_sidedness() -> None:
    both = MetricBand("m", 1.0, 2.0, 0.5)
    upper = MetricBand("m", 1.0, 2.0, 0.5, "upper")
    lower = MetricBand("m", 1.0, 2.0, 0.5, "lower")
    assert both.z(1.5) == 0 and both.z(0.0) == pytest.approx(2.0) and both.z(3.0) == pytest.approx(2.0)
    assert upper.z(0.0) == 0 and upper.z(3.0) == pytest.approx(2.0)
    assert lower.z(3.0) == 0 and lower.z(0.0) == pytest.approx(2.0)


def test_verdict_thresholds() -> None:
    assert verdict(2.0) == (Status.PASS, 1.0)
    assert verdict(2.5)[0] is Status.WARNING and verdict(2.5)[1] == pytest.approx(0.7)
    assert verdict(3.01)[0] is Status.FAIL
    assert verdict(9.0)[1] == 0.0


def test_calibrated_verdict_overrides_textbook() -> None:
    d = CAL.apply(diag(RuleType.MADD_TABII, counts=3.0, core_counts=2.0, core_ms=400, align_conf=0.9, align_min_ms=200))
    assert d.status is Status.PASS and d.score == 1.0
    assert d.measured_harakat == pytest.approx(2.4)  # count scale applied for display
    d = CAL.apply(diag(RuleType.MADD_TABII, Status.PASS, core_counts=3.0, core_ms=600, align_conf=0.9,
                       align_min_ms=200))
    assert d.status is Status.FAIL and d.metrics["calibrated_z"] == pytest.approx(4.5)


def test_one_sided_izhar_never_fails_for_being_crisp() -> None:
    d = CAL.apply(diag(RuleType.IZHAR_SHAFAWI, counts=0.3, core_ms=60, align_conf=0.9, align_min_ms=100))
    assert d.status is Status.PASS


def test_low_posterior_goes_to_review_not_dropped() -> None:
    d = CAL.apply(diag(RuleType.MADD_TABII, align_conf=0.1, align_min_ms=200, core_ms=300, core_counts=0.2))
    assert d.status is Status.REVIEW and d.score == REVIEW_SCORE and "posterior" in d.feedback


@pytest.mark.parametrize("metrics,reason", [
    ({"align_conf": 0.9, "align_min_ms": 20, "core_ms": 300, "core_counts": 0.2}, "collapsed"),
    ({"align_conf": 0.9, "align_min_ms": 200, "core_ms": 0.0, "core_counts": 0.0}, "voiced core"),
])
def test_aligner_artefacts_are_skipped(metrics: dict[str, float], reason: str) -> None:
    """Experts show these too (CTC collapse on sustained sounds), so they are not evidence of an error."""
    d = CAL.apply(diag(RuleType.MADD_TABII, **metrics))
    assert d.status is Status.SKIPPED and d.score is None and reason in d.feedback and "artefact" in d.feedback


def test_letter_outside_model_vocabulary_is_skipped() -> None:
    d = CAL.apply(diag(RuleType.MADD_TABII, align_conf=0.0, align_min_ms=0.0, core_ms=300, core_counts=2.0))
    assert d.status is Status.SKIPPED and d.score is None


def test_unreliable_spans_lower_the_index_instead_of_raising_it() -> None:
    """MNAR guard: every extra low-confidence span (the trace a wrong letter leaves) must not raise the score."""
    good = diag(RuleType.MADD_TABII, core_ms=240, core_counts=2.0, align_conf=0.9, align_min_ms=200)
    lost = {"core_ms": 240, "core_counts": 2.0, "align_conf": 0.1, "align_min_ms": 200}
    base = [CAL.apply(good), CAL.apply(good)]
    scores = [summarize_diagnostics(base + [CAL.apply(diag(RuleType.MADD_TABII, **lost)) for _ in range(k)]).overall
              for k in range(4)]
    assert all(b < a for a, b in zip(scores, scores[1:])), scores
    assert summarize_diagnostics(base + [CAL.apply(diag(RuleType.MADD_TABII, **lost))]).coverage == 1.0


def test_uncalibrated_rule_keeps_textbook_verdict() -> None:
    d = CAL.apply(diag(RuleType.QALQALAH, Status.WARNING, energy_rise_db=3.0))
    assert d.status is Status.WARNING and d.score == 0.1


def test_offline_recalibration_matches_live_path() -> None:
    metrics = {"core_counts": 2.9, "core_ms": 600.0, "align_conf": 0.9, "align_min_ms": 200.0}
    live = CAL.apply(diag(RuleType.MADD_TABII, **metrics))
    [offline] = summarize.recalibrate([{"rule_type": "madd_tabii", "status": "FAIL", "score": 0.1,
                                        "metrics": metrics}], CAL)
    assert offline["status"] == live.status.value and offline["score"] == pytest.approx(live.score)


def test_scorer_uses_calibrated_weights() -> None:
    from app.scoring import TajweedScorer

    assert TajweedScorer(CAL).weights["madd"] == 2.0
    assert TajweedScorer().weights["madd"] == 1.0
