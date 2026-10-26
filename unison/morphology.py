"""Versioned morphological feature-grid carriers, not a universal 48-form claim."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class MorphologicalCell:
    features: tuple[tuple[str, str], ...]
    realized: str | None
    evidence: str | None
    status: str

    @property
    def feature_id(self) -> str:
        return "|".join(f"{key}={value}" for key, value in self.features)


@dataclass(frozen=True, slots=True)
class MorphologicalParadigm:
    language: str
    era: str
    dialect: str | None
    lemma: str
    grammar_id: str
    source: str
    cells: tuple[MorphologicalCell, ...]

    def __post_init__(self) -> None:
        if not all((self.language, self.era, self.lemma, self.grammar_id, self.source)):
            raise ValueError("Paradigm requires sourced language, era, lemma, and grammar ID")
        if len({cell.feature_id for cell in self.cells}) != len(self.cells):
            raise ValueError("Duplicate feature bundles")
        if len(self.cells) > 4096:
            raise ValueError("RESOURCE_LIMIT: paradigm feature cells")

    @property
    def form_count(self) -> int:
        return len(self.cells)

    @property
    def attested_forms(self) -> int:
        return sum(c.realized is not None and bool(c.evidence) for c in self.cells)

    @property
    def form_48_profile(self) -> bool:
        """Exactly 48 DECLARED slots, never an inherited Arabic language axiom."""
        return len(self.cells) == 48


def declare_paradigm(
    *, language: str, era: str, lemma: str, grammar_id: str, source: str,
    axes: Mapping[str, Sequence[str]],
    realizations: Mapping[str, tuple[str, str]] | None = None,
    guard: Callable[[tuple[tuple[str, str], ...]], bool] | None = None,
    dialect: str | None = None,
) -> MorphologicalParadigm:
    if not 1 <= len(axes) <= 8:
        raise ValueError("Require 1..8 explicit grammatical feature axes")
    if any(not choices or len(set(choices)) != len(choices) for choices in axes.values()):
        raise ValueError("Nonempty feature values must be unique")
    names = tuple(axes)
    combos = 1
    for values in axes.values():
        combos *= len(values)
    if combos > 4096:
        raise ValueError("RESOURCE_LIMIT: proposed paradigm too large")
    attested = realizations or {}
    rows = []
    for values in product(*(axes[name] for name in names)):
        features = tuple(zip(names, values))
        if guard is not None and not guard(features):
            continue
        key = "|".join(f"{k}={v}" for k, v in features)
        found = attested.get(key)
        rows.append(MorphologicalCell(
            features,
            found[0] if found else None,
            found[1] if found else None,
            "ATTESTED" if found else "UNKNOWN",
        ))
    return MorphologicalParadigm(language, era, dialect, lemma, grammar_id, source, tuple(rows))
