"""The learner model and the knowledge report: the properties the product relies on."""

from __future__ import annotations

import numpy as np

from app.learner import CohortModel, project


def _cohort() -> CohortModel:
    # two skills that move together across reciters, a third independent
    rng = np.random.default_rng(3)
    per = {}
    for r in range(40):
        a = rng.normal(2.5, 0.8)
        per[f"r{r}"] = {"ghunnah": [int(40 / (1 + np.exp(-a))), 40],
                        "ikhfa": [int(40 / (1 + np.exp(-(a + rng.normal(0, 0.3))))), 40],
                        "qalqalah": [int(40 / (1 + np.exp(-rng.normal(3, 0.5)))), 40]}
    return CohortModel.fit(per)


def test_no_evidence_is_the_cohort_and_says_so() -> None:
    m = _cohort()
    post = m.posterior({})
    for s, d in post.items():
        assert d["basis"] == "prior"
        assert abs(d["estimate"] - 1 / (1 + np.exp(-m.mu[m.skills.index(s)]))) < 1e-4   # reported to 4 places


def test_evidence_moves_the_skill_and_its_correlate_but_not_the_independent_one() -> None:
    m = _cohort()
    base = m.posterior({})
    weak = m.posterior({"ghunnah": [2, 10]})
    assert weak["ghunnah"]["basis"] == "measured"
    assert weak["ghunnah"]["estimate"] < base["ghunnah"]["estimate"] - 0.1
    assert weak["ikhfa"]["estimate"] < base["ikhfa"]["estimate"] - 0.05     # transfer along the covariance
    assert weak["ikhfa"]["basis"] == "inferred"
    assert abs(weak["qalqalah"]["estimate"] - base["qalqalah"]["estimate"]) < 0.02


def test_all_pass_on_hundreds_of_instances_stays_finite_and_high() -> None:
    m = _cohort()
    d = m.posterior({"ghunnah": [500, 500]})["ghunnah"]
    assert 0.97 < d["estimate"] < 1.0 and d["interval90"][0] <= d["estimate"] <= d["interval90"][1]


def test_projection_covers_every_instance_with_a_skill() -> None:
    m = _cohort()
    inv = [{"surah": 1, "ayah": 1, "rules": [["ghunnah", 0, None], ["unknown_rule", 1, None]]},
           {"surah": 1, "ayah": 2, "rules": [["qalqalah", 0, None]]}]
    p = project(m.posterior({"ghunnah": [1, 10]}), inv)
    assert p["coverage"] == {"rule_instances": 3, "with_a_skill_estimate": 2}
    assert p["hardest_ayahs"][0]["ayah"] == 1          # the weak ghunnah makes 1:1 the risk
