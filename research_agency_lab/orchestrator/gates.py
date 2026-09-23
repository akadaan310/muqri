"""Local-first stage gates and budgeted planning for cloud runs.

Pipeline (a feed-forward control loop): every local stage writes a metrics file; a gate is a
predicate on those metrics; a cloud stage is *eligible* only when all its prerequisite gates pass.
Among eligible stages the planner picks by value per upper-bound dollar (greedy knapsack under the
remaining budget, see ``budget.Ledger``). After a cloud run, ``Ledger.observe`` feeds the measured
cost back into the rate estimate, closing the loop.

Gates (thresholds are explicit so they can be argued with):

* ``tests``        — pytest and the Julia frontier tests are green (recorded by ``check``).
* ``crosscheck``   — Julia and Octave agree on real data (worst rel. err < 1e-6).
* ``calibration``  — the near-gold anchor (Husary Muallim, held out of the calibration) scores
                     ≥ 97 and the median anchor FAIL rate over rule keys is ≤ 3 %. Other masters
                     are *not* gated: fast reciters and imams make real mistakes, and the engine
                     must flag them, not be tuned until they pass. Their scores are reported only.
* ``discrimination`` — masters vs non-masters (QuranMB learners): AUROC of the index ≥ 0.80.
* ``lahn``         — wrong-letter detector: text-swap recall ≥ 0.30 at 1 % expert false alarm and
                     clip-level AUROC on QuranMB planted errors ≥ 0.70.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METRICS = Path(__file__).resolve().parent / "metrics"
EXP = ROOT / "research_agency_lab" / "experiments"


def _load(path: Path) -> dict | None:  # type: ignore[type-arg]
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


@dataclass
class Gate:
    name: str
    source: Path
    check: Callable[[dict], tuple[bool, str]]  # type: ignore[type-arg]

    def evaluate(self) -> tuple[bool | None, str]:
        data = _load(self.source)
        if data is None:
            return None, f"no metrics yet ({self.source.relative_to(ROOT)})"
        return self.check(data)


def _tests(d: dict) -> tuple[bool, str]:  # type: ignore[type-arg]
    return bool(d.get("pytest_ok") and d.get("julia_ok")), f"pytest {d.get('pytest_ok')}, julia {d.get('julia_ok')}"


def _cross(d: dict) -> tuple[bool, str]:  # type: ignore[type-arg]
    w = float(d.get("worst_rel_err", 1.0))
    return w < 1e-6, f"worst rel.err {w:.2e}"


def _calib(d: dict) -> tuple[bool, str]:  # type: ignore[type-arg]
    anchor = d.get("anchor_min_held_out", 0.0)
    target = d.get("anchor_target", 97.0)
    fail = d.get("anchor_median_fail_rate", 1.0)
    return anchor >= target and fail <= 0.03, (f"anchor (held out) {anchor:.1f} (≥{target:.0f}), anchor median FAIL "
                                               f"{100 * fail:.1f}% (≤3%); peers LOO min {d.get('loo_min_peer', 0.0):.1f} (info)")


def _disc(d: dict) -> tuple[bool, str]:  # type: ignore[type-arg]
    a = d.get("auroc_master_vs_learner", 0.0)
    return a >= 0.80, f"AUROC masters vs learners {a:.3f} (≥0.80)"


def _lahn(d: dict) -> tuple[bool, str]:  # type: ignore[type-arg]
    r = d.get("textswap_consonant_recall_1pct", 0.0)
    a = d.get("quranmb_clip_auroc", 0.0)
    return r >= 0.30 and a >= 0.70, f"text-swap recall {r:.2f} (≥0.30), QuranMB clip AUROC {a:.3f} (≥0.70)"


GATES: dict[str, Gate] = {
    "tests": Gate("tests", METRICS / "tests.json", _tests),
    "crosscheck": Gate("crosscheck", METRICS / "crosscheck.json", _cross),
    "calibration": Gate("calibration", METRICS / "calibration.json", _calib),
    "discrimination": Gate("discrimination", METRICS / "learners.json", _disc),
    "lahn": Gate("lahn", METRICS / "lahn.json", _lahn),
}


@dataclass
class CloudStage:
    tag: str
    workload: str
    rows: int
    requires: tuple[str, ...]
    value: float  # prior value 1–5: expected decision impact of the result
    why: str
    cpu: float = 2.0
    mem_gib: float = 6.0
    gpu: str | None = None


# Candidate cloud work. Rows are ayah-rows (the unit the rate is measured in).
STAGES: list[CloudStage] = [
    CloudStage("imams-pilot", "engine_cpu2", 5 * 150, ("tests", "calibration"), 4.0,
               "5 taraweeh imams × 150 strategic ayahs: does calibration separate masters in live acoustics?"),
    CloudStage("imams-strategic", "engine_cpu2", 5 * 1200, ("tests", "calibration", "discrimination"), 3.0,
               "full strategic set for the imams, only after the pilot and the learner gate agree"),
    CloudStage("lahn-textswap-fullquran", "engine_cpu2", 6 * 6236, ("tests", "lahn"), 2.5,
               "per-letter GOP thresholds from 6 studio reciters × full Quran"),
]


def status() -> dict[str, tuple[bool | None, str]]:
    return {name: g.evaluate() for name, g in GATES.items()}


def plan(ledger) -> list[dict[str, object]]:  # type: ignore[no-untyped-def]
    """Eligible stages ranked by value per upper-bound dollar, greedily packed into the budget."""
    gates = status()
    rows: list[dict[str, object]] = []
    for s in STAGES:
        est = ledger.estimate(s.workload, s.rows, cpu=s.cpu, mem_gib=s.mem_gib, gpu=s.gpu)
        blocked = [g for g in s.requires if gates[g][0] is not True]
        rows.append({"tag": s.tag, "why": s.why, "blocked_by": blocked, **est,
                     "value_per_usd": round(s.value / max(est["upper_usd"], 0.01), 3)})
    remaining = ledger.remaining
    for r in sorted((r for r in rows if not r["blocked_by"]), key=lambda r: -r["value_per_usd"]):  # type: ignore[operator]
        r["fits_budget"] = r["upper_usd"] <= remaining  # type: ignore[operator]
        if r["fits_budget"]:
            remaining -= r["upper_usd"]  # type: ignore[operator]
    return rows
