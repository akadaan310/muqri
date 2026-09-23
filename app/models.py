"""Shared, typed data structures used across the qaari-eval pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class RuleType(StrEnum):
    """Canonical Hafs 'an 'Asim Tajweed rules that the engine evaluates."""

    MADD_TABII = "madd_tabii"
    MADD_MUTTASIL = "madd_muttasil"
    MADD_MUNFASIL = "madd_munfasil"
    MADD_LAZIM = "madd_lazim"
    MADD_ARID = "madd_arid_lissukun"
    GHUNNAH = "ghunnah"
    IKHFA = "ikhfa"
    IDGHAM_GHUNNAH = "idgham_ghunnah"
    IQLAB = "iqlab"
    IKHFA_SHAFAWI = "ikhfa_shafawi"
    IDGHAM_SHAFAWI = "idgham_shafawi"
    QALQALAH = "qalqalah"
    TAFKHEEM = "tafkheem"
    TARQEEQ = "tarqeeq"


MADD_RULES: frozenset[RuleType] = frozenset(
    {
        RuleType.MADD_TABII,
        RuleType.MADD_MUTTASIL,
        RuleType.MADD_MUNFASIL,
        RuleType.MADD_LAZIM,
        RuleType.MADD_ARID,
    }
)

NASAL_RULES: frozenset[RuleType] = frozenset(
    {
        RuleType.GHUNNAH,
        RuleType.IKHFA,
        RuleType.IDGHAM_GHUNNAH,
        RuleType.IQLAB,
        RuleType.IKHFA_SHAFAWI,
        RuleType.IDGHAM_SHAFAWI,
    }
)

WEIGHT_RULES: frozenset[RuleType] = frozenset({RuleType.TAFKHEEM, RuleType.TARQEEQ})


class Status(StrEnum):
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"
    SKIPPED = "SKIPPED"


class Vowel(StrEnum):
    FATHA = "a"
    DAMMA = "u"
    KASRA = "i"


@dataclass(slots=True)
class LetterUnit:
    """One pronounceable (or silent) letter of the target text with its marks resolved."""

    index: int
    word_index: int
    char: str
    text: str
    vowel: Vowel | None = None
    tanween: bool = False
    shadda: bool = False
    sukun: bool = False
    explicit_sukun: bool = False
    silent: bool = False
    madd_letter: bool = False
    maddah_sign: bool = False
    synthetic: bool = False
    assimilated: bool = False
    elided: bool = False

    @property
    def pronounced(self) -> bool:
        return not self.silent


@dataclass(slots=True)
class Word:
    index: int
    text: str
    unit_indices: list[int] = field(default_factory=list)


@dataclass(slots=True)
class RuleInstance:
    """A Tajweed rule that the canonical text requires at a given location."""

    rule_type: RuleType
    word_index: int
    word: str
    unit_indices: list[int]
    expected_harakat: tuple[float, float] | None = None
    letter: str | None = None
    detail: str = ""
    at_waqf: bool = False


@dataclass(slots=True)
class AlignedUnit:
    unit_index: int
    start_s: float
    end_s: float
    confidence: float = 1.0

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


@dataclass(slots=True)
class Alignment:
    units: dict[int, AlignedUnit]
    method: str
    mean_confidence: float = 1.0

    def span(self, unit_indices: list[int]) -> tuple[float, float] | None:
        present = [self.units[i] for i in unit_indices if i in self.units]
        if not present:
            return None
        return min(u.start_s for u in present), max(u.end_s for u in present)


@dataclass(slots=True)
class RuleDiagnostic:
    rule_type: RuleType
    word: str
    start_ms: int
    end_ms: int
    status: Status
    feedback: str
    expected_harakat: float | None = None
    expected_range: tuple[float, float] | None = None
    measured_harakat: float | None = None
    score: float | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    letter: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "rule_type": str(self.rule_type),
            "word": self.word,
            "location": {"start_ms": self.start_ms, "end_ms": self.end_ms},
        }
        if self.letter:
            out["letter"] = self.letter
        if self.expected_harakat is not None:
            out["expected_harakat"] = self.expected_harakat
        if self.expected_range is not None:
            out["expected_harakat_range"] = list(self.expected_range)
        if self.measured_harakat is not None:
            out["measured_harakat"] = round(self.measured_harakat, 2)
        out["status"] = str(self.status)
        if self.score is not None:
            out["score"] = round(self.score, 3)
        if self.metrics:
            out["metrics"] = {k: round(float(v), 3) for k, v in self.metrics.items()}
        out["feedback"] = self.feedback
        return out


def dataclass_to_dict(obj: Any) -> dict[str, Any]:
    return asdict(obj)
