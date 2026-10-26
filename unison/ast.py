"""Indexed hierarchical, exact-span UNISON abstract syntax tree.

The grammar spine stores a Document > token (Word/Space/Punctuation) >
base-grapheme-cluster graph. Source bytes and semantic annotations are never
overwritten. O(1) means EXPECTED node lookup in an already built hash index,
not O(1) tokenization, parsing, transformations or semantic understanding.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import unicodedata

from .text import parse_text


@dataclass(frozen=True, slots=True)
class ASTNode:
    node_id: str
    kind: str
    parent_id: str | None
    source_span_bytes: tuple[int, int]
    codepoint_span: tuple[int, int]
    children: tuple[str, ...]
    original: str
    annotations: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class IndexedAST:
    root_id: str
    source: str
    source_sha256: str
    language: str
    nodes: dict[str, ASTNode]

    def get(self, node_id: str) -> ASTNode:
        """Expected O(1) hash lookup once the AST has been materialized."""
        return self.nodes[node_id]

    def children_of(self, node_id: str) -> tuple[ASTNode, ...]:
        return tuple(self.nodes[i] for i in self.get(node_id).children)

    def reconstruct(self) -> str:
        """Canonical document reconstruction using the root's direct children."""
        return "".join(n.original for n in self.children_of(self.root_id))

    def verify_spans(self) -> None:
        data = self.source.encode("utf-8")
        for n in self.nodes.values():
            start, end = n.source_span_bytes
            if data[start:end] != n.original.encode("utf-8"):
                raise ValueError(f"Source span mismatch: {n.node_id}")
        if self.reconstruct() != self.source:
            raise ValueError("Source not reversible")


def _clusters(word: str) -> list[tuple[int, int, str]]:
    """One letter plus trailing combining marks; source spans are exact.

    This is a deliberately *basic* Unicode cluster policy; extended grapheme
    segmentation and full Quranic typography require a dedicated adapter.
    """
    result: list[tuple[int, int, str]] = []
    start = 0
    for i, ch in enumerate(word):
        if i and not unicodedata.combining(ch):
            result.append((start, i, word[start:i]))
            start = i
    if word:
        result.append((start, len(word), word[start:]))
    return result


def build_indexed_ast(source: str, *, language: str) -> IndexedAST:
    ast = parse_text(source, language=language)
    root_id = sha256(("document:" + ast.source_sha256).encode()).hexdigest()[:24]
    nodes: dict[str, ASTNode] = {}
    direct: list[str] = []
    for t in ast.units:
        direct.append(t.node_id)
        cids: list[str] = []
        if t.kind == "word":
            for start, end, text in _clusters(t.raw):
                base_cp = t.start_cp + start
                finish_cp = t.start_cp + end
                byte_start = t.start_byte + len(t.raw[:start].encode("utf-8"))
                byte_end = t.start_byte + len(t.raw[:end].encode("utf-8"))
                cid = sha256(f"{t.node_id}:{byte_start}:{byte_end}:{text}".encode()).hexdigest()[:24]
                cids.append(cid)
                nodes[cid] = ASTNode(
                    cid, "grapheme_cluster", t.node_id, (byte_start, byte_end),
                    (base_cp, finish_cp), (), text,
                    ("combining_mark_present",) if any(unicodedata.combining(c) for c in text) else (),
                )
        nodes[t.node_id] = ASTNode(
            t.node_id, t.kind, root_id, (t.start_byte, t.end_byte),
            (t.start_cp, t.end_cp), tuple(cids), t.raw,
        )
    nodes[root_id] = ASTNode(
        root_id, "document", None, (0, len(source.encode("utf-8"))),
        (0, len(source)), tuple(direct), source,
        ("raw_exact", "semantic_interpretation_separate"),
    )
    result = IndexedAST(root_id, source, ast.source_sha256, language, nodes)
    result.verify_spans()
    return result
