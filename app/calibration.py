"""Reference-reciter calibration: robust z-scores against Al-Hussary and his ijaazah peers.

Textbook targets ("a natural madd is 2 counts ± 0.25") assume a perfect ruler. The engine's
measurements are not one: CTC spans absorb closures, the harakah unit is estimated, and every
metric has a reciter-independent bias. Instead of hand-set limits, ``app/data/calibration.json``
(built by ``research_agency_lab/substrate_library/julia/calibrate.jl`` from full-Qur'an runs)
records, per rule key and metric, how the reference reciters actually measure:

    reference hull  [lo, hi] = [min(m_H, m_C), max(m_H, m_C)]
                    m_H = Al-Hussary's median, m_C = the peers' consensus (median of their medians)
    robust scale    s = max(1.4826 · MAD pooled over the reference reciters, floor)
    z               = distance of x outside [lo, hi] / s   (one-sided for "upper"/"lower" metrics)

PASS |z| ≤ 2, WARNING 2 < |z| ≤ 3, FAIL beyond. A span whose alignment is unreliable (low CTC
posterior, collapsed to a few frames, or no voiced core for a duration rule) is SKIPPED rather than
failed. Rules without calibration keep the textbook validator verdict.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.models import RuleDiagnostic, RuleInstance, Status

CALIBRATION_PATH = Path(__file__).resolve().parent / "data" / "calibration.json"
Z_PASS = 2.0
Z_WARN = 3.0


def rule_key(rule_type: str, detail: str = "", letter: str | None = None) -> str:
    """Rule type plus the sub-type that changes what is measured (qalqalah level, kamil/naqis, …)."""
    rt, detail = str(rule_type), detail or ""
    if rt == "qalqalah":
        return f"qalqalah:{detail}"
    if rt.startswith("idgham_mu") or rt == "idgham_mithlayn":
        return f"{rt}:{detail}"
    if rt == "madd_lazim":
        return f"madd_lazim:{detail.split(':')[0]}"
    if rt == "ikhfa":
        return f"ikhfa:{detail.split('; ')[-1]}"
    if rt == "idgham_ghunnah":
        return f"idgham_ghunnah:{detail.split('; ')[-1]}"
    if rt in ("tafkheem", "tarqeeq") and letter in ("ر", "ل"):
        return f"{rt}:{'raa' if letter == 'ر' else 'lam_allah'}"
    if rt == "izhar_shafawi" and detail:
        return "izhar_shafawi:before_waw_faa"
    return rt


def key_of(rule: RuleInstance) -> str:
    return rule_key(rule.rule_type.value, rule.detail, rule.letter)


@dataclass(slots=True)
class MetricBand:
    metric: str
    lo: float
    hi: float
    scale: float
    side: str = "both"  # "both" | "upper" (only too-large fails) | "lower"

    def z(self, x: float) -> float:
        if x < self.lo:
            d = 0.0 if self.side == "upper" else (self.lo - x)
        elif x > self.hi:
            d = 0.0 if self.side == "lower" else (x - self.hi)
        else:
            d = 0.0
        return d / self.scale


@dataclass
class Calibration:
    count_scale: float = 1.0  # multiply engine counts so the anchor's natural madd median is 2.0
    rules: dict[str, list[MetricBand]] = field(default_factory=dict)
    reliability: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Calibration:
        rules = {
            key: [MetricBand(b["metric"], b["lo"], b["hi"], b["scale"], b.get("side", "both")) for b in bands]
            for key, bands in data.get("rules", {}).items()
        }
        return cls(float(data.get("count_scale", 1.0)), rules, dict(data.get("reliability", {})),
                   {k: v for k, v in data.items() if k not in ("rules", "reliability")})

    @classmethod
    def load(cls, path: str | Path | None = None) -> Calibration | None:
        p = Path(path) if path else CALIBRATION_PATH
        if not p.exists():
            return None
        return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))

    # -- alignment reliability -----------------------------------------------------------------
    def unreliable(self, metrics: dict[str, float], duration_rule: bool) -> str | None:
        conf_min = self.reliability.get("align_conf_min", 0.0)
        span_min = self.reliability.get("align_min_ms", 0.0)
        if "align_conf" in metrics and metrics["align_conf"] < conf_min:
            return f"alignment posterior {metrics['align_conf']:.2f} < {conf_min:.2f}"
        if "align_min_ms" in metrics and metrics["align_min_ms"] < span_min:
            return f"a unit collapsed to {metrics['align_min_ms']:.0f} ms"
        if duration_rule and metrics.get("core_ms", 1.0) <= 0.0:
            return "no voiced core found in the span"
        return None

    # -- verdict ---------------------------------------------------------------------------------
    def judge(self, rule_type: str, detail: str, letter: str | None, status: str,
              metrics: dict[str, float]) -> tuple[Status, float | None, float | None, str] | None:
        """(status, score, z, note) under the calibration, or None when the rule is not calibrated.

        Shared by the live scorer and the offline re-scoring of benchmark rows (benchmarks/summarize.py).
        """
        if status in (Status.SKIPPED.value, Status.VALID_NECESSARY_PAUSE.value):
            return None
        bands = self.rules.get(rule_key(rule_type, detail, letter)) or self.rules.get(rule_type)
        if not bands:
            return None
        duration_rule = any(b.metric.endswith("counts") for b in bands)
        reason = self.unreliable(metrics, duration_rule)
        if reason:
            return Status.SKIPPED, None, None, f"Not judged: {reason} (unreliable alignment)."
        zs = [b.z(x) for b in bands if (x := metrics.get(b.metric)) is not None and math.isfinite(x)]
        if not zs:
            return None
        z = max(zs)
        st, score = verdict(z)
        ref = "; ".join(f"{b.metric} ref {b.lo:.3g}–{b.hi:.3g} (s={b.scale:.2g})" for b in bands)
        return st, score, z, f"calibrated against the reference reciters: z={z:.1f}; {ref}"

    def apply(self, diag: RuleDiagnostic) -> RuleDiagnostic:
        if diag.measured_harakat is not None:
            diag.measured_harakat *= self.count_scale
        res = self.judge(diag.rule_type.value, diag.detail, diag.letter, diag.status.value, diag.metrics)
        if res is None:
            return diag
        status, score, z, note = res
        if status is Status.SKIPPED:
            diag.status, diag.score, diag.feedback = status, None, note
            return diag
        diag.metrics["calibrated_z"] = float(z or 0.0)
        if status is not diag.status:
            diag.feedback = f"{diag.feedback} [{note}]"
        diag.status, diag.score = status, score
        return diag


def verdict(z: float) -> tuple[Status, float]:
    """|z| ≤ 2 PASS (1.0); ≤ 3 WARNING (1.0 → 0.4); beyond FAIL (0.4 → 0 at z = 6)."""
    if z <= Z_PASS:
        return Status.PASS, 1.0
    if z <= Z_WARN:
        return Status.WARNING, 1.0 - 0.6 * (z - Z_PASS) / (Z_WARN - Z_PASS)
    return Status.FAIL, max(0.0, 0.4 * (1 - (z - Z_WARN) / 3.0))


@lru_cache(maxsize=4)
def default_calibration(path: str | None = None) -> Calibration | None:
    return Calibration.load(path)
