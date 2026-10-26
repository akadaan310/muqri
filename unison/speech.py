"""Language-agnostic speech plans, with explicit language packs and UNKNOWN gaps.

Writing is never enough to infer correct spoken output. An Arabic letter name,
a word pronunciation, a discourse speaker switch and acoustic samples are
distinct typed operations. The current module produces an explainable plan,
not synthetic human-quality speech.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .text import parse_text


@dataclass(frozen=True, slots=True)
class LanguagePack:
    language: str
    locale: str
    era: str
    provenance: str
    lexicon: Mapping[str, tuple[str, ...]]
    dialect: str | None = None

    def __post_init__(self) -> None:
        if not all((self.language, self.locale, self.era, self.provenance)):
            raise ValueError("Language pack needs explicit locale, era and provenance")


@dataclass(frozen=True, slots=True)
class SpeechUnit:
    raw: str
    start_byte: int
    end_byte: int
    phonemes: tuple[str, ...] | None
    status: str


@dataclass(frozen=True, slots=True)
class SpeechPlan:
    source: str
    language: str
    locale: str
    era: str
    units: tuple[SpeechUnit, ...]
    complete: bool
    executable_audio: bool = False

    def as_dict(self) -> dict:
        return {
            "source": self.source, "language": self.language,
            "locale": self.locale, "era": self.era,
            "units": [
                {"raw": x.raw, "start_byte": x.start_byte, "end_byte": x.end_byte,
                 "phonemes": x.phonemes, "status": x.status} for x in self.units
            ],
            "complete": self.complete, "executable_audio": self.executable_audio,
            "evidence": "Explicit lexical phonemes, NOT an inferred universal G2P",
        }


def plan_speech(source: str, pack: LanguagePack) -> SpeechPlan:
    ast = parse_text(source, language=pack.language)
    output = []
    for item in ast.units:
        if item.kind != "word":
            output.append(SpeechUnit(item.raw, item.start_byte, item.end_byte, None,
                                     "STRUCTURAL_PUNCTUATION_OR_SPACE"))
        elif item.raw in pack.lexicon:
            phones = tuple(pack.lexicon[item.raw])
            output.append(SpeechUnit(item.raw, item.start_byte, item.end_byte, phones,
                                     "ATTESTED_LANGUAGE_PACK"))
        else:
            output.append(SpeechUnit(item.raw, item.start_byte, item.end_byte, None,
                                     "UNKNOWN_PRONUNCIATION"))
    return SpeechPlan(ast.original, pack.language, pack.locale, pack.era, tuple(output),
                      complete=all(u.status != "UNKNOWN_PRONUNCIATION" for u in output))


@dataclass(frozen=True, slots=True)
class QawlEvent:
    speaker: str
    utterance: str
    scope: str
    operation: str
    evidence: str | None = None


def qawl_transition(previous: QawlEvent | None, next_event: QawlEvent) -> dict:
    """Speaker retention is an explicit operation, not a universal Quranic law."""
    if next_event.operation not in {"SPEAK", "CONTINUE", "SWITCH", "QUOTED_SPEECH"}:
        raise ValueError("Unknown speech-act operation")
    if next_event.operation == "CONTINUE":
        if previous is None or previous.speaker != next_event.speaker:
            raise ValueError("CONTINUE requires a previously established matching speaker")
    if next_event.operation == "SWITCH":
        if previous is None or previous.speaker == next_event.speaker:
            raise ValueError("SWITCH requires two distinguishable identified speakers")
    return {
        "before": previous.speaker if previous else None,
        "after": next_event.speaker,
        "operation": next_event.operation,
        "scope": next_event.scope,
        "utterance": next_event.utterance,
        "source": next_event.evidence,
        "unknown": [] if next_event.evidence else ["source-of-speaker-identification"],
    }
