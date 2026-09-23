"""Validator dispatch, score aggregation and feedback summaries.

Two scores are reported:

* **Tajweed perfection index**: weighted mean over the Ahkaam (Mudood, Noon/Meem Sakinah,
  Ghunnah, Qalqalah, Idghaam classes, Raa/Lam, Wasl/Sakt). This is the "pure Tajweed rules" score.
* **Sifaat score**: articulation qualities (Hams/Jahr, Shiddah/Rakhawah, Itbaq, Safir, …), kept
  separate because these measurements are less mature than the rule checks.

``SKIPPED`` and ``VALID_NECESSARY_PAUSE`` diagnostics are excluded from both scores.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass

import numpy as np

from app.models import MADD_RULES, RuleDiagnostic, RuleType, Status, rule_category
from app.sifaat import ghair_mutadhaddah, hams_jahr, itbaq, sukoon_spectrum
from app.tajweed_rules import (
    idghaam_classes,
    meem_sakinah,
    mudood_engine,
    noon_sakinah,
    qalqalah_engine,
    raa_lam_rules,
    sakt_wasl,
)
from app.tajweed_rules.base import EvalContext, Validator, skipped

logger = logging.getLogger(__name__)

VALIDATORS: dict[RuleType, Validator] = {
    **{rt: mudood_engine.validate_madd for rt in MADD_RULES},
    **noon_sakinah.VALIDATORS,
    **meem_sakinah.VALIDATORS,
    **qalqalah_engine.VALIDATORS,
    **idghaam_classes.VALIDATORS,
    **raa_lam_rules.VALIDATORS,
    **sakt_wasl.VALIDATORS,
    **hams_jahr.VALIDATORS,
    **sukoon_spectrum.VALIDATORS,
    **itbaq.VALIDATORS,
    **ghair_mutadhaddah.VALIDATORS,
}

# Relative weight of each rule family in the perfection index. Timing and nasal rules are the most
# reliably measured; formant-based weight judgements and boundary checks the least.
CATEGORY_WEIGHTS: dict[str, float] = {
    "madd": 1.0, "noon": 1.0, "meem": 0.8, "ghunnah": 1.0, "qalqalah": 0.8, "idgham": 0.8, "weight": 0.6,
    "wasl": 0.5, "sifaat": 1.0,
}
_EXCLUDED = (Status.SKIPPED, Status.VALID_NECESSARY_PAUSE)


@dataclass(slots=True)
class ScoreSummary:
    overall: float | None  # Tajweed perfection index (Ahkaam only)
    sifaat: float | None
    by_category: dict[str, float]
    status_counts: dict[str, int]
    evaluated: int


class TajweedScorer:
    def evaluate(self, ev: EvalContext) -> list[RuleDiagnostic]:
        diagnostics: list[RuleDiagnostic] = []
        for rule in ev.parsed.rules:
            validator = VALIDATORS.get(rule.rule_type)
            if validator is None:
                diagnostics.append(skipped(rule, f"No validator for {rule.rule_type}"))
                continue
            try:
                diag = validator(rule, ev)
            except Exception as exc:  # noqa: BLE001 - one bad window must not sink the report
                logger.exception("Validator failed for %s on %s", rule.rule_type, rule.word)
                diag = skipped(rule, f"Analysis failed: {exc}")
            diag.ayah = rule.ayah
            diag.detail = rule.detail
            diagnostics.append(diag)
        diagnostics.sort(key=lambda d: (d.start_ms, d.end_ms))
        return diagnostics

    @staticmethod
    def summarize(diagnostics: list[RuleDiagnostic]) -> ScoreSummary:
        per_cat: dict[str, list[float]] = {}
        for d in diagnostics:
            if d.status in _EXCLUDED or d.score is None:
                continue
            per_cat.setdefault(rule_category(d.rule_type), []).append(d.score)

        def weighted(cats: list[str]) -> float | None:
            total_w = sum(CATEGORY_WEIGHTS[c] * len(per_cat[c]) for c in cats if c in per_cat)
            if not total_w:
                return None
            return sum(CATEGORY_WEIGHTS[c] * sum(per_cat[c]) for c in cats if c in per_cat) / total_w * 100

        ahkaam = [c for c in CATEGORY_WEIGHTS if c != "sifaat"]
        counts = Counter(str(d.status) for d in diagnostics)
        return ScoreSummary(
            overall=weighted(ahkaam), sifaat=weighted(["sifaat"]),
            by_category={c: round(float(np.mean(v)) * 100, 1) for c, v in per_cat.items()},
            status_counts={s.value: counts.get(s.value, 0) for s in Status},
            evaluated=sum(len(v) for c, v in per_cat.items() if c != "sifaat"),
        )


def consistency_notes(diagnostics: list[RuleDiagnostic]) -> list[str]:
    """Hafs requires each Madd type to be held with a consistent length throughout a recitation."""
    notes: list[str] = []
    for rule_type in (RuleType.MADD_MUTTASIL, RuleType.MADD_MUNFASIL, RuleType.MADD_ARID):
        vals = [d.measured_harakat for d in diagnostics if d.rule_type == rule_type and d.measured_harakat]
        if len(vals) >= 2 and (max(vals) - min(vals)) > 1.5:
            notes.append(
                f"{rule_type.value} lengths vary from {min(vals):.1f} to {max(vals):.1f} counts; "
                "keep the same length for every occurrence."
            )
    return notes
