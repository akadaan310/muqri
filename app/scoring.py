"""Rule diagnostic aggregation and feedback generation."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass

import numpy as np

from app.acoustic.features import AcousticContext
from app.acoustic.ghunnah import analyze_ghunnah, oral_reference_ner
from app.acoustic.madd import analyze_madd
from app.acoustic.qalqalah import analyze_qalqalah
from app.acoustic.tafkheem import analyze_weight, build_reference, vowel_window
from app.acoustic.tempo import TempoEstimate, short_syllable_units
from app.models import (
    MADD_RULES,
    NASAL_RULES,
    WEIGHT_RULES,
    Alignment,
    RuleDiagnostic,
    RuleType,
    Status,
    Vowel,
)
from app.tajweed_rules import ParsedText

logger = logging.getLogger(__name__)

# Relative weight of each rule family in the overall score. Duration-based rules are the most
# reliably measured; formant-based weight judgements the least.
CATEGORY_WEIGHTS: dict[str, float] = {"madd": 1.0, "nasal": 1.0, "qalqalah": 0.8, "weight": 0.6}


def category(rule_type: RuleType) -> str:
    if rule_type in MADD_RULES:
        return "madd"
    if rule_type in NASAL_RULES:
        return "nasal"
    if rule_type in WEIGHT_RULES:
        return "weight"
    return "qalqalah"


@dataclass(slots=True)
class ScoreSummary:
    overall: float | None
    by_category: dict[str, float]
    status_counts: dict[str, int]
    evaluated: int


class TajweedScorer:
    def __init__(self, *, madd_tolerance: float = 0.15) -> None:
        self.madd_tolerance = madd_tolerance

    def evaluate(self, parsed: ParsedText, alignment: Alignment, ctx: AcousticContext,
                 tempo: TempoEstimate) -> list[RuleDiagnostic]:
        haraka = tempo.haraka_ms
        weight_ref = None
        oral_ref: float | None = None
        if any(r.rule_type in NASAL_RULES for r in parsed.rules):
            windows = [
                w for i in short_syllable_units(parsed)
                if parsed.units[i].vowel == Vowel.FATHA and (w := vowel_window([i], alignment)) is not None
            ]
            oral_ref = oral_reference_ner(windows, ctx)

        diagnostics: list[RuleDiagnostic] = []
        for rule in parsed.rules:
            try:
                if rule.rule_type in MADD_RULES:
                    phrase_final = parsed.stop_at_end and rule.word_index == parsed.words[-1].index
                    diag = analyze_madd(rule, alignment, haraka, tolerance=self.madd_tolerance,
                                        phrase_final=phrase_final)
                elif rule.rule_type in NASAL_RULES:
                    diag = analyze_ghunnah(rule, alignment, ctx, haraka, oral_reference_db=oral_ref)
                elif rule.rule_type == RuleType.QALQALAH:
                    diag = analyze_qalqalah(rule, alignment, ctx)
                else:
                    if weight_ref is None:
                        weight_ref = build_reference(parsed, alignment, ctx)
                    diag = analyze_weight(rule, parsed, alignment, ctx, weight_ref)
            except Exception as exc:  # noqa: BLE001 - one bad window must not sink the report
                logger.exception("Analyzer failed for %s on %s", rule.rule_type, rule.word)
                diag = RuleDiagnostic(
                    rule_type=rule.rule_type, word=rule.word, start_ms=0, end_ms=0, status=Status.SKIPPED,
                    feedback=f"Analysis failed: {exc}", letter=rule.letter,
                )
            diagnostics.append(diag)
        diagnostics.sort(key=lambda d: (d.start_ms, d.end_ms))
        return diagnostics

    @staticmethod
    def summarize(diagnostics: list[RuleDiagnostic]) -> ScoreSummary:
        per_cat: dict[str, list[float]] = {}
        for d in diagnostics:
            if d.status is Status.SKIPPED or d.score is None:
                continue
            per_cat.setdefault(category(d.rule_type), []).append(d.score)
        by_category = {c: float(np.mean(v)) * 100 for c, v in per_cat.items()}
        total_w = sum(CATEGORY_WEIGHTS[c] * len(v) for c, v in per_cat.items())
        overall = (
            sum(CATEGORY_WEIGHTS[c] * sum(v) for c, v in per_cat.items()) / total_w * 100 if total_w else None
        )
        counts = Counter(str(d.status) for d in diagnostics)
        return ScoreSummary(
            overall=overall, by_category={k: round(v, 1) for k, v in by_category.items()},
            status_counts={s.value: counts.get(s.value, 0) for s in Status},
            evaluated=sum(len(v) for v in per_cat.values()),
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
