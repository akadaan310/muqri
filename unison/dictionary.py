"""Mu'jam: source-attributed indexed computational dictionaries.

An alphabetically sorted book is an ordered access path; an explicit key index
adds direct lookup, root/value retrieval, filtering and multilingual linking.
QITṬ sums are auxiliary research features, NEVER lexical meaning or proof of
translation. Heat is an observed count, not a fabricated social/semantic score.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from typing import Iterable
import unicodedata

from .registry import QittRegistry, RegistryError


def normalize_key(text: str) -> str:
    """Do not silently collapse distinct Arabic hamza/alif spellings."""
    return unicodedata.normalize("NFC", text).casefold()


@dataclass(frozen=True, slots=True)
class DictionaryEntry:
    entry_id: str
    language: str
    lemma: str
    part_of_speech: str
    gloss: str
    source: str
    root: str | None = None
    era: str | None = None
    dialect: str | None = None
    sense_id: str | None = None
    phonemes: tuple[str, ...] | None = None
    morphology: tuple[str, ...] = ()
    observed_frequency: int | None = None

    def __post_init__(self) -> None:
        if not all((self.entry_id, self.language, self.lemma, self.part_of_speech, self.source)):
            raise ValueError("Entry requires ID, language, lemma, POS and provenance")
        if self.observed_frequency is not None and self.observed_frequency < 0:
            raise ValueError("Observed frequency cannot be negative")


@dataclass(frozen=True, slots=True)
class Lookup:
    query: str
    address_type: str
    rows: tuple[DictionaryEntry, ...]
    complexity: str
    evidence: str


class MuajamIndex:
    """Explicit in-memory materialized index; updates build each index eagerly."""

    def __init__(self) -> None:
        self._ids: dict[str, DictionaryEntry] = {}
        self._keys: dict[tuple[str, str], set[str]] = {}
        self._roots: dict[tuple[str, str], set[str]] = {}
        self._values: dict[int, set[str]] = {}
        self._languages: dict[str, set[str]] = {}
        self._senses: dict[str, set[str]] = {}
        self._value_order: list[tuple[int, str]] = []
        self.registry = QittRegistry()

    def insert(self, entry: DictionaryEntry) -> None:
        if entry.entry_id in self._ids:
            raise ValueError("Duplicate entry ID requires versioned update/reindex")
        self._ids[entry.entry_id] = entry
        self._keys.setdefault((entry.language, normalize_key(entry.lemma)), set()).add(entry.entry_id)
        self._languages.setdefault(entry.language, set()).add(entry.entry_id)
        if entry.root:
            self._roots.setdefault((entry.language, normalize_key(entry.root)), set()).add(entry.entry_id)
        if entry.sense_id:
            self._senses.setdefault(entry.sense_id, set()).add(entry.entry_id)
        if entry.language == "ar":
            try:
                value = self.registry.word_value(entry.lemma)
            except RegistryError:
                value = None
            if value is not None:
                self._values.setdefault(value, set()).add(entry.entry_id)
                from bisect import insort
                insort(self._value_order, (value, entry.entry_id))

    def _collect(self, ids: Iterable[str]) -> tuple[DictionaryEntry, ...]:
        return tuple(self._ids[i] for i in sorted(ids))

    def lookup(self, language: str, lemma: str) -> Lookup:
        key = (language, normalize_key(lemma))
        return Lookup(lemma, "language+normalized_lemma", self._collect(self._keys.get(key, ())),
                      "expected O(1) hash-key access + O(k) emit",
                      "exact indexed lookup; no claim that normalization supplies semantics")

    def by_root(self, language: str, root: str) -> Lookup:
        return Lookup(root, "language+root", self._collect(self._roots.get((language, normalize_key(root)), ())),
                      "expected O(1) lookup + O(k) emit", "root must be supplied by a source or analysis")

    def by_native_value(self, value: int) -> Lookup:
        if type(value) is not int or value < 0:
            raise ValueError("Expected nonnegative native numeric value")
        return Lookup(str(value), "QITT_native_value",
                      self._collect(self._values.get(value, ())),
                      "expected O(1) key + O(k) emit",
                      "numeric collision groups are not word-meaning equivalences")

    def interval(self, low: int, high: int) -> Lookup:
        if low > high:
            raise ValueError("Interval boundaries out of order")
        lo = bisect_left(self._value_order, (low, ""))
        hi = bisect_right(self._value_order, (high, "\U0010ffff"))
        return Lookup(f"{low}..{high}", "QITT_interval",
                      self._collect(e for _, e in self._value_order[lo:hi]),
                      "O(log n + k) ordered index", "bounded interval of native values, not birth years")

    def polyglot(self, sense_id: str) -> Lookup:
        """Cross-language links require curated SAME-SENSE identifiers."""
        return Lookup(sense_id, "attested_cross_language_sense",
                      self._collect(self._senses.get(sense_id, ())),
                      "expected O(1) key + O(k) emit", "sense equivalence is source-asserted, never numeric")

    def heat_map(self, language: str) -> tuple[dict, ...]:
        """Observed usage only; absent corpus statistics remain NULL/UNKNOWN."""
        groups: dict[int | None, dict] = {}
        for entry_id in self._languages.get(language, ()):
            entry = self._ids[entry_id]
            group = None
            if language == "ar":
                try:
                    group = self.registry.word_value(entry.lemma) // 100
                except RegistryError:
                    pass
            row = groups.setdefault(group, {"band": group, "entries": 0, "observed_frequency": 0,
                                           "frequency_complete": True})
            row["entries"] += 1
            if entry.observed_frequency is None:
                row["frequency_complete"] = False
            else:
                row["observed_frequency"] += entry.observed_frequency
        return tuple(groups[x] for x in sorted(groups, key=lambda x: (x is None, x if x is not None else 0)))


def terminal_meem_letter_names() -> tuple[str, ...]:
    """Programmatically challenge a claimed exclusivity of جيم and ميم."""
    return tuple(x for x in ("جيم", "ميم", "لام") if x.endswith("م"))


def inspect_mim_prefix(derived_word: str, stem: str) -> dict:
    """Pattern hypothesis only; cannot establish actual Arabic derivation."""
    expected = "م" + stem
    return {
        "stem": stem, "derived": derived_word, "prefix_candidate": "م",
        "surface_match": normalize_key(derived_word) == normalize_key(expected),
        "status": "formal string hypothesis, not a morphological or etymological proof",
    }
