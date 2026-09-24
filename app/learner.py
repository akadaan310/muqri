"""The learner model: from what one recitation shows, what the whole Quran would show.

A single recitation contains a handful of rules. The engine should still be able to say, with honest
uncertainty, how this learner would fare on every rule in the Quran -- and sharpen with every
recitation. That needs two things the grader alone does not have:

* a PRIOR over skills: how the pass rates of the 35 rule types vary, and CO-vary, across reciters. It
  is fitted on the full-dataset engine pass (41 reciters, 11,996 verses) as a multivariate normal on
  the log-odds of each rule's pass rate, the covariance shrunk toward its diagonal because 41 voices
  cannot pin down every pairwise correlation;
* an UPDATE: the learner's evidence on a rule is k passes in n instances, observed on the log-odds
  scale with the binomial's own variance, and the posterior is Gaussian conditioning. Observed skills
  pull the unobserved ones along the cohort covariance -- one recitation that shows the ghunnah
  informs ikhfa -- and the uncertainty shrinks only as far as the evidence justifies.

Every skill is reported with its estimate, a 90 % interval, and its BASIS: "measured" (the learner's
own instances), "inferred" (not seen, but its uncertainty cut by related evidence), or "prior" (nothing
about this learner yet). The client must never present an inferred or prior skill as a measurement.

`project` maps the posterior onto the Quran-wide inventory (every rule instance in all 6,236 ayahs):
expected rule accuracy per ayah and per surah, and where the learner's risk concentrates.
"""

from __future__ import annotations

import gzip
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "research_agency_lab/experiments/quran/cohort_model.json"
INVENTORY = ROOT / "research_agency_lab/experiments/quran/inventory.jsonl.gz"
FAIL = {"short", "long", "wrong"}
GRADED = FAIL | {"pass"}
SHRINK = 0.3              # covariance shrinkage toward the diagonal
Z90 = 1.645


def _logit(k: float, n: float) -> float:
    p = (k + 0.5) / (n + 1.0)
    return math.log(p / (1 - p))


def _sig(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def verse_tally(record: dict[str, Any]) -> dict[str, list[int]]:
    """All skills evidenced by one per-verse record: rules, each characteristic head, letter identity."""
    t = tally([(ru, st) for ru, st in record.get("rules", [])])
    for h, (n, k) in (record.get("heads") or {}).items():
        if h in ("tikraar",) or n <= 0:            # takrir is descriptive (concealed), never a skill
            continue
        t[f"sifah:{h}"] = [k, n]
    idn = record.get("identity")
    if idn and idn[0] > 0:
        t["letter_identity"] = [idn[1], idn[0]]
    return t


def tally(rule_statuses: list[tuple[str, str]]) -> dict[str, list[int]]:
    """rule -> [passes, graded instances] from (rule, status) pairs."""
    t: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for rule, st in rule_statuses:
        if st in GRADED:
            t[rule][0] += st == "pass"
            t[rule][1] += 1
    return dict(t)


@dataclass
class CohortModel:
    skills: list[str]
    mu: np.ndarray            # prior mean, log-odds
    sigma: np.ndarray         # prior covariance, log-odds
    phi: float = 1.0          # overdispersion: instances in one verse fail together, so n counts less
    shrink: float = SHRINK    # covariance shrinkage toward its diagonal (Ledoit-Wolf, from the data)

    @staticmethod
    def overdispersion(verse_tallies: list[dict[str, list[int]]]) -> float:
        """Pearson dispersion of per-verse pass counts around each (reciter-pooled) rule rate.

        `verse_tallies` holds one {rule: [passes, n]} per verse of ONE reciter (called per reciter and
        pooled by the caller). phi = sum (k - n p)^2 / (n p (1 - p)) / (#verses - #rules)."""
        rate: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for t in verse_tallies:
            for s, (k, n) in t.items():
                rate[s][0] += k
                rate[s][1] += n
        chi, dof = 0.0, 0
        for t in verse_tallies:
            for s, (k, n) in t.items():
                p = min(max(rate[s][0] / rate[s][1], 0.01), 0.99)
                chi += (k - n * p) ** 2 / (n * p * (1 - p))
                dof += 1
        dof -= len(rate)
        return max(1.0, chi / max(1, dof))

    @classmethod
    def fit(cls, per_reciter: dict[str, dict[str, list[int]]], min_n: int = 5, phi: float = 1.0) -> CohortModel:
        skills = sorted({s for t in per_reciter.values() for s, (_k, n) in t.items() if n >= min_n})
        R = len(per_reciter)
        Y = np.full((R, len(skills)), np.nan)
        for i, t in enumerate(per_reciter.values()):
            for j, s in enumerate(skills):
                if s in t and t[s][1] >= min_n:
                    Y[i, j] = _logit(*t[s])
        mu = np.nanmean(Y, axis=0)
        Yc = np.where(np.isnan(Y), 0.0, Y - mu)                 # missing -> the mean (no information)
        cov = Yc.T @ Yc / max(1, R - 1)
        lam = ledoit_wolf_intensity(Yc, cov)
        sigma = (1 - lam) * cov + lam * np.diag(np.diag(cov))
        sigma += np.eye(len(skills)) * 1e-3
        return cls(skills, mu, sigma, phi, lam)

    def to_json(self) -> dict[str, Any]:
        return {"skills": self.skills, "mu": self.mu.round(6).tolist(), "sigma": self.sigma.round(6).tolist(),
                "shrink": round(self.shrink, 4), "phi": round(self.phi, 4)}

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> CohortModel:
        d = json.loads(path.read_text())
        return cls(d["skills"], np.asarray(d["mu"]), np.asarray(d["sigma"]), d.get("phi", 1.0),
                   d.get("shrink", SHRINK))

    def posterior(self, evidence: dict[str, list[int]], floor_sd: float | None = None) -> dict[str, dict[str, Any]]:
        """Skill estimates for one learner given their evidence {skill: [passes, instances]}.

        `floor_sd` widens the prior for a population the cohort does not represent: the cohort is 41
        PROFESSIONAL reciters, nearly all near 97 % on most skills, so their spread is far narrower than
        learners'. With the floor a learner's own misses can move their estimate; without it one miss in
        tafkhim left the estimate at 0.973. It is an assumption, to be calibrated on learner recordings.
        """
        idx = {s: i for i, s in enumerate(self.skills)}
        O = [idx[s] for s, (_k, n) in evidence.items() if s in idx and n > 0]
        sigma = self.sigma
        if floor_sd:
            sd = np.sqrt(np.diag(sigma))
            scale = np.maximum(1.0, floor_sd / np.maximum(sd, 1e-9))
            sigma = sigma * np.outer(scale, scale)              # widen, keeping the correlations
        m, C = self.mu.copy(), sigma.copy()
        if O:
            # Laplace approximation of the binomial likelihood around the PRIOR mean (one IRLS step):
            # the working response y = mu + (k/n - p0) / (p0 (1 - p0)), variance phi / (n p0 (1 - p0)).
            # Smoothing k/n toward 1/2 first -- the obvious shortcut -- read a perfect 1/1 as 0.75,
            # below the cohort's ~0.95, and marked a flawless learner weak on every related rule.
            # The posterior mode over the observed skills, by Newton's method with backtracking on
            #   f(x) = -sum[k log s(x) + (n-k) log(1 - s(x))] / phi + (x - mu_O)' Sigma_OO^-1 (x - mu_O) / 2.
            # The prior keeps the mode finite even when k = n (every letter realised: the likelihood's
            # own maximum is at infinity, and a working-response iteration oscillated there); the line
            # search keeps every step an improvement. The unobserved skills follow by conditioning on
            # the mode, and the uncertainty is the Laplace curvature there.
            k = np.array([evidence[self.skills[i]][0] for i in O], dtype=float)
            n = np.array([evidence[self.skills[i]][1] for i in O], dtype=float)
            muO = self.mu[O]
            P = np.linalg.inv(sigma[np.ix_(O, O)])

            def f(x: np.ndarray) -> float:
                ls = -np.logaddexp(0, -x)                       # log s(x), stable
                l1 = -np.logaddexp(0, x)                        # log (1 - s(x))
                d = x - muO
                return float(-(k * ls + (n - k) * l1).sum() / self.phi + 0.5 * d @ P @ d)

            x = muO.copy()
            for _ in range(50):
                sx = _sig(x)
                g = -(k - n * sx) / self.phi + P @ (x - muO)
                H = np.diag(n * sx * (1 - sx) / self.phi) + P
                step = np.linalg.solve(H, g)
                t, fx = 1.0, f(x)
                while f(x - t * step) > fx - 1e-4 * t * (g @ step) and t > 1e-6:
                    t *= 0.5
                x = x - t * step
                if np.max(np.abs(t * step)) < 1e-7:
                    break
            sx = _sig(x)
            W = np.maximum(n * sx * (1 - sx) / self.phi, 1e-9)
            K = sigma[:, O] @ np.linalg.solve(sigma[np.ix_(O, O)] + np.diag(1.0 / W), np.eye(len(O)))
            m = self.mu + sigma[:, O] @ P @ (x - muO)            # conditional mean given the mode
            m[O] = x
            C = sigma - K @ sigma[O, :]
        out = {}
        for s, i in idx.items():
            v = max(C[i, i], 1e-9)
            prior_v = sigma[i, i]
            basis = "measured" if s in evidence and evidence[s][1] > 0 else (
                "inferred" if v < 0.8 * prior_v else "prior")
            # the posterior MEDIAN in probability: the right point estimate for an absolute-error
            # loss. The mean E[sigmoid] drags toward 1/2 under uncertainty and made a learner with no
            # evidence look worse than the cohort the prior came from.
            est = float(_sig(m[i]))
            out[s] = {"estimate": round(est, 4),
                      "interval90": [round(float(_sig(m[i] - Z90 * math.sqrt(v))), 4),
                                     round(float(_sig(m[i] + Z90 * math.sqrt(v))), 4)],
                      "basis": basis,
                      "evidence": evidence.get(s, [0, 0]),
                      "certainty": round(1 - v / prior_v, 3)}   # share of the prior uncertainty removed
        return out


def ledoit_wolf_intensity(Xc: np.ndarray, S: np.ndarray) -> float:
    """Optimal shrinkage of the sample covariance toward its diagonal (Ledoit & Wolf 2004), from the
    data: the estimated variance of the off-diagonal sample covariances over their squared size."""
    n = Xc.shape[0]
    if n < 3:
        return 1.0
    off = S - np.diag(np.diag(S))
    d2 = float((off ** 2).sum())
    if d2 <= 0:
        return 1.0
    b2 = 0.0
    for x in Xc:
        o = np.outer(x, x) - S
        b2 += float(((o - np.diag(np.diag(o))) ** 2).sum())
    b2 /= n * n
    return float(min(1.0, b2 / d2))


def load_inventory(path: Path = INVENTORY) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def project(skills: dict[str, dict[str, Any]], inventory: list[dict[str, Any]]) -> dict[str, Any]:
    """Expected rule accuracy on every ayah of the Quran, and where the learner's risk concentrates."""
    per_surah: dict[int, list[float]] = defaultdict(list)
    risk: Counter[str] = Counter()
    ayahs = []
    covered = total = 0
    for a in inventory:
        ps, rs = [], []
        for rule, _w, _c in a.get("rules", []):
            total += 1
            s = skills.get(rule)
            if s is None:
                continue
            covered += 1
            ps.append(s["estimate"])
            rs.append((rule, 1 - s["estimate"]))
            risk[rule] += 1 - s["estimate"]
        if not ps:
            continue
        exp = float(np.mean(ps))
        per_surah[a["surah"]].append(exp)
        ayahs.append((a["surah"], a["ayah"], exp, sorted(rs, key=lambda x: -x[1])[:2]))
    ayahs.sort(key=lambda x: x[2])
    return {
        "coverage": {"rule_instances": total, "with_a_skill_estimate": covered},
        "expected_rule_accuracy": round(float(np.mean([x[2] for x in ayahs])), 4) if ayahs else None,
        "by_surah": {str(s): round(float(np.mean(v)), 4) for s, v in sorted(per_surah.items())},
        "hardest_ayahs": [{"surah": s, "ayah": a, "expected": round(e, 4), "because": [r for r, _ in rs]}
                          for s, a, e, rs in ayahs[:20]],
        "where_the_risk_is": [{"rule": r, "expected_misses_across_the_quran": round(v, 1)}
                              for r, v in risk.most_common(10)],
    }
