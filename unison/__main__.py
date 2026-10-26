"""Offline UNISON/0.1 CLI: bounded numeric, textual and Qur'an-scene facilities.

No network, model loading, OS-permission bypass or MUQRI audio pipeline mutation.
Examples:
  python -m unison address 1000000
  python -m unison word "الرقيم"
  python -m unison quran 2 1 "الم" --edition source-review --initials
  python -m unison sort-plan '[[3,1],[4,2]]' --axis 1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .address_ops import CollectionStore
from .ast import build_indexed_ast
from .quran import classify_initials
from .registry import QittRegistry
from .surface_scene import quran_surface_scene


def run() -> int:
    parser = argparse.ArgumentParser(prog="unison", description="UNISON/0.1 source-safe computational language console")
    sub = parser.add_subparsers(dest="action", required=True)
    a = sub.add_parser("address", help="Typed QITT integer→letter-register address")
    a.add_argument("number", type=int)
    w = sub.add_parser("word", help="Research numeric projection, not a dictionary definition")
    w.add_argument("text")
    w.add_argument("--profile", default="bare_graphemes",
                   choices=("bare_graphemes", "diacritics_preserved", "shadda_expanded_opt_in"))
    q = sub.add_parser("quran", help="Emit read-only Qur'an scene; full text must come from a source")
    q.add_argument("surah", type=int)
    q.add_argument("ayah", type=int)
    q.add_argument("text")
    q.add_argument("--edition", required=True)
    q.add_argument("--initials", action="store_true", help="Caller confirms verse-initial muqatta'at context")
    q.add_argument("--output", type=Path, help="Output JSON scene path (default stdout)")
    ast = sub.add_parser("ast", help="Indexed source-preserving hierarchical AST")
    ast.add_argument("text")
    ast.add_argument("--language", required=True)
    s = sub.add_parser("sort-plan", help="Return admissible sorting strategies for a scoped collection")
    s.add_argument("json_array")
    s.add_argument("--axis", type=int, required=True)
    s.add_argument("--commit", action="store_true")
    s.add_argument("--policy", choices=("numeric_scalar", "aggregate_sum"), default="numeric_scalar")
    args = parser.parse_args()
    reg = QittRegistry()
    if args.action == "address":
        cells = reg.encode(args.number)
        result = {"decimal": args.number, "cells": [x.as_dict() for x in cells],
                  "roundtrip": reg.decode(cells), "native_glyph_projection_noninjective": reg.bare_projection(cells)}
    elif args.action == "word":
        val = reg.word_value(args.text, profile=args.profile)
        result = {"source": args.text, "profile": args.profile, "native_sum": val,
                  "address": [x.as_dict() for x in reg.encode(val)],
                  "disclaimer": "No natural-language meaning inferred from arithmetic"}
    elif args.action == "quran":
        result = quran_surface_scene(
            surah=args.surah, ayah=args.ayah, raw_text=args.text, edition=args.edition,
            role="verse_initial_candidate" if args.initials else "ordinary_word",
        )
    elif args.action == "ast":
        parsed = build_indexed_ast(args.text, language=args.language)
        result = {"root": parsed.root_id, "source": parsed.source,
                  "source_sha256": parsed.source_sha256,
                  "nodes": [
                      {"id": n.node_id, "kind": n.kind, "parent": n.parent_id,
                       "span": n.source_span_bytes, "original": n.original,
                       "children": n.children, "annotations": n.annotations}
                      for n in parsed.nodes.values()
                  ]}
    else:
        array = json.loads(args.json_array)
        store = CollectionStore()
        address = store.register(array)
        opts = store.sort_options(address.handle, axis=args.axis)
        result = {"handle": address.handle, "shape": address.shape, "options": opts}
        if args.commit:
            new = store.sorted_copy(address.handle, axis=args.axis, policy=args.policy)
            result["sorted"] = store.read(new.handle)
            result["reversible_parent"] = address.handle
            result["new_handle"] = new.handle
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.action == "quran" and args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
