"""Lahn-jali diagnostics: one LAHN_LETTER / LAHN_HARAKAH verdict per pronounced unit.

Thresholds come from ``app/data/lahn_reference.json``, built by
``research_agency_lab/experiments/lahn_gop/build_reference.py`` from correct expert recitations:
per kind, ``fail`` / ``warn`` are the α = 1 % / 5 % quantiles, on correct expert text, of the
statistic named by ``stat``: the raw LLR (lower = worse), or the per-symbol robust z of
``app.lahn.gop.gop_z`` (higher = worse; the model's letter-specific bias removed). So an expert is
flagged at about those rates. The recall they buy is measured by the text-swap test
(``research_agency_lab/experiments/lahn_gop/``) and recorded in the same file. The reference is
tied to the CTC model it was built with; a different model gets no lahn verdicts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.lahn.gop import GopBand, UnitGOP, gop_z, letter_gop
from app.models import Alignment, RuleDiagnostic, RuleType, Status
from app.tajweed_rules import ParsedText

REFERENCE_PATH = Path(__file__).resolve().parents[1] / "data" / "lahn_reference.json"
WARN_SCORE = 0.6


@dataclass(slots=True)
class LahnReference:
    model: str
    thresholds: dict[str, dict[str, Any]]  # kind -> {"stat": "llr" | "z", "fail": τ, "warn": τ}
    bands: dict[str, GopBand] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path | None = None) -> LahnReference | None:
        p = Path(path) if path else REFERENCE_PATH
        if not p.exists():
            return None
        data = json.loads(p.read_text())
        bands = {k: GopBand(**v) for k, v in data.get("bands", {}).items()}
        meta = {k: v for k, v in data.items() if k not in ("model", "thresholds", "bands")}
        return cls(data["model"], data["thresholds"], bands, meta)

    def verdict(self, g: UnitGOP) -> tuple[Status, float] | None:
        th = self.thresholds.get(g.kind)
        if th is None:
            return None
        if th.get("stat", "llr") == "z":
            z = gop_z(self.bands, g.kind, g.target, g.llr)
            if z is None:
                return None
            bad = -z  # compare on the same "lower = worse" axis as the LLR
            fail, warn = -th["fail"], -th["warn"]
        else:
            bad, fail, warn = g.llr, th["fail"], th["warn"]
        if bad < fail:
            return Status.FAIL, 0.0
        if bad < warn:
            return Status.WARNING, WARN_SCORE
        return Status.PASS, 1.0


@lru_cache(maxsize=2)
def default_reference(path: str | None = None) -> LahnReference | None:
    return LahnReference.load(path)


def lahn_diagnostics(lp: Any, frame_s: float, vocab: dict[str, int], blank: int, parsed: ParsedText,
                     alignment: Alignment, reference: LahnReference) -> list[RuleDiagnostic]:
    units = {u.index: u for u in parsed.units}
    out: list[RuleDiagnostic] = []
    for g in letter_gop(lp, frame_s, vocab, blank, parsed, alignment):
        res = reference.verdict(g)
        if res is None:
            continue
        status, score = res
        u = units[g.unit_index]
        span = alignment.units[g.unit_index]
        word = parsed.words[u.word_index].text
        target, alt = g.label(g.target), g.label(g.best_alt)
        if g.kind == "consonant":
            rule, what = RuleType.LAHN_LETTER, f"{target} in '{word}'"
            wrong = f"{what} sounded closer to {alt}. Articulate {target} from its own makhraj."
        else:
            rule, what = RuleType.LAHN_HARAKAH, f"the {target} on {u.char} in '{word}'"
            wrong = f"{what} sounded closer to a {alt}. Re-read the word with its written vowel."
        feedback = f"{what} read as written." if status is Status.PASS else wrong
        out.append(RuleDiagnostic(
            rule_type=rule, word=word, start_ms=int(round(span.start_s * 1000)), end_ms=int(round(span.end_s * 1000)),
            status=status, feedback=feedback, score=score, letter=u.char,
            metrics={"gop_llr": round(g.llr, 3), "gop_frames": float(g.frames)},
            ayah=parsed.words[u.word_index].ayah, detail=f"{g.target}>{g.best_alt}",
        ))
    return out
