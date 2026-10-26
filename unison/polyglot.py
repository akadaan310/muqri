"""Parallel dictionary-learning lanes; UNKNOWN is an explicit outcome.

A polyglot board can show up to 70 languages, but it cannot create historical
senses, native pronunciations or dialect morphology without sourced resources.
"""
from __future__ import annotations
from dataclasses import dataclass
from .dictionary import MuajamIndex


@dataclass(frozen=True, slots=True)
class PolyglotLane:
    language: str
    entries: tuple[dict, ...]
    status: str
    evidence: str


def concept_board(index: MuajamIndex, sense_id: str, languages: tuple[str, ...]) -> tuple[PolyglotLane, ...]:
    if not languages or len(languages) > 70 or len(set(languages)) != len(languages):
        raise ValueError("UNISON accepts 1..70 distinct explicit language lanes")
    attestations = index.polyglot(sense_id).rows
    result = []
    for language in languages:
        options = [x for x in attestations if x.language == language]
        result.append(PolyglotLane(
            language,
            tuple({
                "entry_id": x.entry_id, "lemma": x.lemma, "era": x.era, "dialect": x.dialect,
                "gloss": x.gloss, "part_of_speech": x.part_of_speech,
                "phonemes": x.phonemes, "source": x.source,
            } for x in options),
            "ATTESTED_LEXICAL_SENSE" if options else "UNKNOWN_LANGUAGE_SENSE",
            "Curated sense identity is required; numeric register equality is NOT a translation",
        ))
    return tuple(result)
