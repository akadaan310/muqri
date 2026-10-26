"""Loss-aware text syntax trees with exact Unicode source spans.

The AST is a parsed structural view; it is not an invented spoken reading.
RASM herein means an EXPLICITLY LABELLED simplification, not the canonical
'Uthmani orthographic science or an authorized replacement for a mushaf.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import unicodedata


@dataclass(frozen=True, slots=True)
class TextNode:
    node_id: str
    kind: str
    raw: str
    start_cp: int
    end_cp: int
    start_byte: int
    end_byte: int


@dataclass(frozen=True, slots=True)
class TextAST:
    original: str
    language: str
    source_sha256: str
    units: tuple[TextNode, ...]
    words: tuple[TextNode, ...]
    punctuation: tuple[TextNode, ...]
    spaces: tuple[TextNode, ...]

    def node_index(self) -> dict[str, TextNode]:
        """Build an indexed map once for EXPECTED O(1) keyed lookup thereafter."""
        return {x.node_id: x for x in self.units}


def _kind(c: str) -> str:
    if c.isspace():
        return "space"
    if unicodedata.category(c).startswith(("L", "M", "N")):
        return "word"
    return "punctuation"


def parse_text(source: str, *, language: str) -> TextAST:
    if not isinstance(source, str) or not isinstance(language, str) or not language.strip():
        raise ValueError("Source and explicit language label required")
    if len(source) > 1_000_000:
        raise ValueError("RESOURCE_LIMIT: source code points")
    # Preserve source exactly. NFC is available only as a derived view.
    offsets = [0]
    for c in source:
        offsets.append(offsets[-1] + len(c.encode("utf-8")))
    kind = None
    start = 0
    nodes = []
    for i, c in enumerate(source + "\0"):
        next_kind = _kind(c) if i < len(source) else None
        if next_kind != kind:
            if kind is not None:
                raw = source[start:i]
                node_id = hashlib.sha256(f"{start}:{i}:{raw}".encode()).hexdigest()[:20]
                nodes.append(TextNode(node_id, kind, raw, start, i, offsets[start], offsets[i]))
            kind = next_kind
            start = i
    result = tuple(nodes)
    assert "".join(x.raw for x in result) == source
    return TextAST(
        original=source, language=language,
        source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        units=result, words=tuple(x for x in result if x.kind == "word"),
        punctuation=tuple(x for x in result if x.kind == "punctuation"),
        spaces=tuple(x for x in result if x.kind == "space"),
    )


def basic_rasm_projection(source: str) -> dict:
    """Only a *simplified letter skeleton*, NEVER canonical Uthmani rasm."""
    ast = parse_text(source, language="ar")
    output = "".join(
        c for c in source
        if not unicodedata.category(c).startswith("M") and c != "\u0640"
    )
    return {
        "kind": "simplified_noncanonical_letter_skeleton",
        "original": ast.original,
        "skeleton": output,
        "lossy": output != source,
        "source_sha256": ast.source_sha256,
        "limitations": [
            "Not an authoritative Uthmani rasm transformation",
            "Hamza-bearing letters and orthographic Alif distinctions retained",
            "Quranic annotation marks may require separate script-aware parsing",
        ],
    }
