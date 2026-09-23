#!/usr/bin/env python3
"""Select a compact set of ayahs that exercises every Tajweed rule and Sifah.

Every ayah of the Qur'an is parsed and its rule instances are bucketed by *key* (rule type plus
the sub-type that matters for evaluation: Qalqalah level, Idghaam kamil/naqis, Madd Lazim
harfi/kalimi, heavy/light Ikhfa, …). A weighted greedy set cover then picks ayahs until every key
has at least ``--per-key`` instances (or all of them, for rare keys such as the four saktāt,
Qalqalah Akbar or Idghaam Naqis), preferring short ayahs so a whole reciter can be benchmarked
quickly. The result is written to ``app/data/strategic_verses.json``.

    python datasets/strategic_verses.py --per-key 6
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.models import RuleInstance  # noqa: E402
from app.quran_text import get_full_quran  # noqa: E402
from app.tajweed_rules import TajweedParser  # noqa: E402

OUTPUT = ROOT / "app" / "data" / "strategic_verses.json"
# Always included: the ayahs every student knows (and the only Qalqalah Akbar at an ayah end).
ANCHORS = [(1, a) for a in range(1, 8)] + [(111, 1), (112, 1), (112, 4), (113, 1), (114, 1)]


def rule_key(rule: RuleInstance) -> str:
    rt = rule.rule_type.value
    if rt == "qalqalah":
        return f"qalqalah:{rule.detail}"
    if rt.startswith("idgham_mu") or rt == "idgham_mithlayn":
        return f"{rt}:{rule.detail}"
    if rt == "madd_lazim":
        return f"madd_lazim:{rule.detail.split(':')[0]}"
    if rt == "ikhfa":
        return f"ikhfa:{rule.detail.split('; ')[-1]}"
    if rt == "idgham_ghunnah":
        return f"idgham_ghunnah:{rule.detail.split('; ')[-1]}"
    if rt in ("tafkheem", "tarqeeq") and rule.letter in ("ر", "ل"):
        return f"{rt}:{'raa' if rule.letter == 'ر' else 'lam_allah'}"
    if rt == "izhar_shafawi" and rule.detail:
        return "izhar_shafawi:before_waw_faa"
    return rt


def select(per_key: int, max_words: int) -> dict[str, Any]:
    quran = get_full_quran()
    parser = TajweedParser(include_sifaat=True)
    counts: dict[tuple[int, int], collections.Counter[str]] = {}
    words: dict[tuple[int, int], int] = {}
    totals: collections.Counter[str] = collections.Counter()
    for ref, text in quran.items():
        parsed = parser.parse(text)
        c = collections.Counter(rule_key(r) for r in parsed.rules)
        counts[ref] = c
        words[ref] = len(parsed.words)
        totals.update(c)
    need = {k: min(per_key, v) for k, v in totals.items()}
    chosen: list[tuple[int, int]] = []
    got: collections.Counter[str] = collections.Counter()

    def take(ref: tuple[int, int]) -> None:
        chosen.append(ref)
        got.update(counts[ref])

    for ref in ANCHORS:
        take(ref)
    while True:
        missing = {k: n - got[k] for k, n in need.items() if got[k] < n}
        if not missing:
            break
        best, best_gain = None, 0.0
        for ref, c in counts.items():
            if ref in chosen:
                continue
            covered = sum(min(missing.get(k, 0), v) for k, v in c.items())
            if not covered:
                continue
            # Rare keys must be taken wherever they occur, even in long ayahs.
            rare = any(totals[k] <= per_key for k in c if k in missing)
            if words[ref] > max_words and not rare:
                continue
            gain = covered / (1.0 + words[ref] / 10.0)
            if gain > best_gain:
                best, best_gain = ref, gain
        if best is None:
            break
        take(best)
    chosen.sort()
    return {
        "per_key": per_key,
        "verses": [{"surah": s, "ayah": a, "words": words[(s, a)],
                    "covers": sorted(k for k in counts[(s, a)])} for s, a in chosen],
        "coverage": {k: {"selected": got[k], "in_quran": totals[k]} for k in sorted(totals)},
        "total_words": sum(words[r] for r in chosen),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-key", type=int, default=6)
    ap.add_argument("--max-words", type=int, default=30)
    ap.add_argument("--output", default=str(OUTPUT))
    args = ap.parse_args(argv)
    result = select(args.per_key, args.max_words)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Selected {len(result['verses'])} ayahs ({result['total_words']} words) -> {args.output}")
    for k, v in result["coverage"].items():
        print(f"  {k:34s} {v['selected']:4d} / {v['in_quran']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
