"""Shared, typed data structures used across the qaari-eval pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class RuleType(StrEnum):
    """Canonical Hafs 'an 'Asim Tajweed rules (Ahkaam) and articulation qualities (Sifaat)."""

    # Mudood
    MADD_TABII = "madd_tabii"
    MADD_MUTTASIL = "madd_muttasil"
    MADD_MUNFASIL = "madd_munfasil"
    MADD_LAZIM = "madd_lazim"
    MADD_ARID = "madd_arid_lissukun"
    MADD_LEEN = "madd_leen"
    MADD_BADAL = "madd_badal"
    MADD_IWAD = "madd_iwad"
    MADD_SILAH_SUGHRA = "madd_silah_sughra"
    MADD_SILAH_KUBRA = "madd_silah_kubra"
    # Noon sakinah / tanween
    IZHAR_HALQI = "izhar_halqi"
    IKHFA = "ikhfa"
    IDGHAM_GHUNNAH = "idgham_ghunnah"
    IDGHAM_NO_GHUNNAH = "idgham_no_ghunnah"
    IQLAB = "iqlab"
    # Meem sakinah
    IKHFA_SHAFAWI = "ikhfa_shafawi"
    IDGHAM_SHAFAWI = "idgham_shafawi"
    IZHAR_SHAFAWI = "izhar_shafawi"
    # Ghunnah mushaddadah
    GHUNNAH = "ghunnah"
    # Qalqalah (detail: sughra / kubra / akbar)
    QALQALAH = "qalqalah"
    # Idghaam classes of other letters (detail: kamil / naqis)
    IDGHAM_MITHLAYN = "idgham_mithlayn"
    IDGHAM_MUTAJANISAYN = "idgham_mutajanisayn"
    IDGHAM_MUTAQARIBAYN = "idgham_mutaqaribayn"
    # Raa / Lam
    TAFKHEEM = "tafkheem"
    TARQEEQ = "tarqeeq"
    JAWAZ_WAJHAYN = "jawaz_wajhayn"
    # Wasl / Waqf / Sakt
    HAMZAT_WASL = "hamzat_wasl"
    SAKT = "sakt"
    # Sifaat
    HAMS = "hams"
    JAHR = "jahr"
    SHIDDAH = "shiddah"
    TAWASSUT = "tawassut"
    RAKHAWAH = "rakhawah"
    ITBAQ = "itbaq"
    SAFIR = "safir"
    TAFASHHI = "tafashhi"
    ISTITAALAH = "istitaalah"
    TAKREER = "takreer"
    # Lahn jali: the letter / short vowel itself (substitution-aware GOP, app/lahn)
    LAHN_LETTER = "lahn_letter"
    LAHN_HARAKAH = "lahn_harakah"


class Tareeq(StrEnum):
    SHATIBIYYAH = "shatibiyyah"
    TAYYIBAH = "tayyibah"


MADD_RULES: frozenset[RuleType] = frozenset(
    {
        RuleType.MADD_TABII,
        RuleType.MADD_MUTTASIL,
        RuleType.MADD_MUNFASIL,
        RuleType.MADD_LAZIM,
        RuleType.MADD_ARID,
        RuleType.MADD_LEEN,
        RuleType.MADD_BADAL,
        RuleType.MADD_IWAD,
        RuleType.MADD_SILAH_SUGHRA,
        RuleType.MADD_SILAH_KUBRA,
    }
)

NOON_RULES: frozenset[RuleType] = frozenset(
    {RuleType.IZHAR_HALQI, RuleType.IKHFA, RuleType.IDGHAM_GHUNNAH, RuleType.IDGHAM_NO_GHUNNAH, RuleType.IQLAB}
)
MEEM_RULES: frozenset[RuleType] = frozenset(
    {RuleType.IKHFA_SHAFAWI, RuleType.IDGHAM_SHAFAWI, RuleType.IZHAR_SHAFAWI}
)
# Rules whose window must carry a held nasal resonance.
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
IDGHAM_CLASS_RULES: frozenset[RuleType] = frozenset(
    {RuleType.IDGHAM_MITHLAYN, RuleType.IDGHAM_MUTAJANISAYN, RuleType.IDGHAM_MUTAQARIBAYN}
)
WEIGHT_RULES: frozenset[RuleType] = frozenset({RuleType.TAFKHEEM, RuleType.TARQEEQ, RuleType.JAWAZ_WAJHAYN})
WASL_RULES: frozenset[RuleType] = frozenset({RuleType.HAMZAT_WASL, RuleType.SAKT})
LAHN_RULES: frozenset[RuleType] = frozenset({RuleType.LAHN_LETTER, RuleType.LAHN_HARAKAH})
SIFAAT_RULES: frozenset[RuleType] = frozenset(
    {
        RuleType.HAMS, RuleType.JAHR, RuleType.SHIDDAH, RuleType.TAWASSUT, RuleType.RAKHAWAH, RuleType.ITBAQ,
        RuleType.SAFIR, RuleType.TAFASHHI, RuleType.ISTITAALAH, RuleType.TAKREER,
    }
)


def rule_category(rule_type: RuleType) -> str:
    """Scoring family of a rule: madd, noon, meem, ghunnah, qalqalah, idgham, weight, wasl, lahn, sifaat."""
    if rule_type in LAHN_RULES:
        return "lahn"
    if rule_type in MADD_RULES:
        return "madd"
    if rule_type in NOON_RULES:
        return "noon"
    if rule_type in MEEM_RULES:
        return "meem"
    if rule_type is RuleType.GHUNNAH:
        return "ghunnah"
    if rule_type is RuleType.QALQALAH:
        return "qalqalah"
    if rule_type in IDGHAM_CLASS_RULES:
        return "idgham"
    if rule_type in WEIGHT_RULES:
        return "weight"
    if rule_type in WASL_RULES:
        return "wasl"
    return "sifaat"


class Status(StrEnum):
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"
    SKIPPED = "SKIPPED"
    # The alignment model could not follow the audio against the text. That is itself evidence of a
    # possible error (a substituted letter drives the CTC posterior down), so it is counted, not
    # dropped: dropping it made wrong recitations score *higher* (MNAR bias, deep-research 06 D2).
    REVIEW = "REVIEW"
    # A short or breath-driven stop in a live recitation (Waqf al-Dharoori), not a Tajweed error.
    VALID_NECESSARY_PAUSE = "VALID_NECESSARY_PAUSE"


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
    waqf_only: bool = False  # rectangular zero (U+06E0): an alif pronounced only at a stop (أَنَا۠)
    madd_letter: bool = False
    maddah_sign: bool = False
    synthetic: bool = False
    assimilated: bool = False
    elided: bool = False
    orig_vowel: Vowel | None = None  # vowel before waqf removed it
    wasla: bool = False  # a hamzat al-wasl (pronounced only at the start of a phrase)

    @property
    def pronounced(self) -> bool:
        return not self.silent


@dataclass(slots=True)
class Word:
    index: int
    text: str
    unit_indices: list[int] = field(default_factory=list)
    ayah: int = 0  # position of the ayah within the analysed passage
    stop_after: bool = False  # the reciter stops (waqf) after this word
    sakt_after: bool = False  # Hafs sakt (brief breathless pause) after this word


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
    ayah: int = 0


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
    ayah: int = 0
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "rule_type": str(self.rule_type),
            "word": self.word,
            "location": {"start_ms": self.start_ms, "end_ms": self.end_ms},
        }
        if self.letter:
            out["letter"] = self.letter
        if self.detail:
            out["detail"] = self.detail
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
