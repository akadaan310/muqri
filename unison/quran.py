"""Qur'anic disconnected-letter frames and lossless written/spoken separation.

Canonical recitation remains the existing MUQRI TajweedParser: _LETTER_NAMES,
MUQATTAAT, and verse-aware qira'a/wajh rules. This module provides an optional
STRUCTURAL AST and QITṬ numeric projection for research and rendering. It never
replaces the actual text of a verse or the recognized spoken letter names.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .registry import QittRegistry
from .text import parse_text, basic_rasm_projection

# Mirrors the exact MUQATTAAT inventory read from MUQRI's Tajweed parser.
# These are sequences, NOT an exhaustive license to infer a surah from glyphs.
MUQATTAAT = frozenset({
    "الم", "المص", "الر", "المر", "كهيعص", "طه", "طسم",
    "طس", "يس", "ص", "حم", "عسق", "ق", "ن",
})
INITIAL_LETTERS = frozenset("".join(MUQATTAAT))
if len(INITIAL_LETTERS) != 14:
    raise AssertionError("The source Muqatta'at inventory must have 14 distinct letters")

# A user-facing transcription of LETTER NAMES, not context-sensitive
# pronunciation targets. Pronunciation and Tajwid realization require MUQRI.
LETTER_NAMES = {
    "ا": "أَلِف", "ل": "لَام", "م": "مِيم", "ص": "صَاد",
    "ر": "رَا", "ك": "كَاف", "ه": "هَا", "ي": "يَا",
    "ع": "عَيْن", "ط": "طَا", "س": "سِين", "ح": "حَا",
    "ق": "قَاف", "ن": "نُون",
}
assert set(LETTER_NAMES) == INITIAL_LETTERS

@dataclass(frozen=True, slots=True)
class MuqattaatFrame:
    source: str
    recognized_sequence: bool
    role: Literal["verse_initial_candidate", "ordinary_word", "unknown"]
    reference: str | None
    letter_nodes: tuple[dict, ...]
    qitt_native_value: int | None
    spoken_policy: str
    epistemic_status: str

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "recognized_sequence": self.recognized_sequence,
            "role": self.role,
            "reference": self.reference,
            "letter_nodes": list(self.letter_nodes),
            "qitt_native_value": self.qitt_native_value,
            "spoken_policy": self.spoken_policy,
            "epistemic_status": self.epistemic_status,
        }


def classify_initials(
    text: str, *,
    role: Literal["verse_initial_candidate", "ordinary_word", "unknown"] = "unknown",
    reference: str | None = None,
) -> MuqattaatFrame:
    """Never interpret a matching ordinary Arabic word as initials by spelling alone."""
    if role not in {"verse_initial_candidate", "ordinary_word", "unknown"}:
        raise ValueError("Unknown Qur'anic textual role")
    ast = parse_text(text, language="ar")
    recognized = text in MUQATTAAT and role == "verse_initial_candidate"
    reg = QittRegistry()
    names = tuple({
        "letter": x,
        "name_written": LETTER_NAMES[x],
        "position": i,
        "native_weight_research": reg.native[x],
        "pronunciation": "DELEGATE_TO_MUQRI_TAJWID",
    } for i, x in enumerate(text)) if recognized else ()
    return MuqattaatFrame(
        source=ast.original,
        recognized_sequence=recognized,
        role=role,
        reference=reference,
        letter_nodes=names,
        qitt_native_value=sum(x["native_weight_research"] for x in names) if recognized else None,
        spoken_policy="MUQRI_TAJWID_LETTER_NAMES" if recognized else "CONTEXT_REQUIRED",
        epistemic_status="written_initials_recognized; numerical reading is proposed research" if recognized else "no initials inference",
    )


def quran_scene_source(surah: int, ayah: int, raw_text: str, *, edition: str) -> dict:
    """A source-stable, read-only verse scene record; never strips basmala by guess."""
    if not 1 <= surah <= 114 or ayah <= 0:
        raise ValueError("Invalid surah/ayah")
    if not edition:
        raise ValueError("Edition and reading context required")
    ast = parse_text(raw_text, language="ar")
    return {
        "kind": "quran_verse_source",
        "surah": surah, "ayah": ayah, "edition": edition,
        "raw": ast.original, "source_sha256": ast.source_sha256,
        "word_spans": [
            {"id": w.node_id, "raw": w.raw, "start_byte": w.start_byte, "end_byte": w.end_byte}
            for w in ast.words
        ],
        "rasm_candidate": basic_rasm_projection(raw_text),
        "qitt_status": "optional numerical overlay; not primary religious meaning",
        "spoken_targets": "delegate to existing MUQRI Tajweed parser with tareeq/waqf context",
        "basmala": "source- and verse-role-specific; never silently inserted/deleted here",
    }
