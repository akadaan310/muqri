#!/usr/bin/env python3
"""Which verses the letter corpus reads: every letter in every form and every rule in every form,
`--per` instances each, from as little audio as possible.

A cell is a letter x form or a rule x form:

  letter forms  fatha / kasra / damma (short), fatha_long / kasra_long / damma_long (the vowel carries
                a madd), sakin (a consonant follows), shadda (the letter doubled), stop (the ayah's last
                letter, read at the waqf)
  rule forms    the parser's rule type and detail (qalqalah sughra / kubra, ikhfa light / heavy,
                idgham_ghunnah naqis / kamil, the madds ...)

The whole Qur'an is read as text (no audio), each ayah as it is recited alone, stopping at its end.
Ayahs are then chosen greedily, most still-needed instances per phoneme first, until every cell that
exists anywhere has `--per` instances or has run out. Every reciter reads the same ayahs, so each cell
compares the same text positions across reciters.

    .venv/bin/python research_agency_lab/experiments/letter_corpus/plan.py --per 5
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

OUT = Path(__file__).with_name("plan.json")
SHORT = {"َ": "fatha", "ِ": "kasra", "ُ": "damma"}
MADD = set("اۥۦ")


def letter_cells(phonemes: str) -> list[tuple[str, str, int]]:
    """(letter, form, unit index) for every consonant unit of one ayah's phonemes."""
    from app.letters import CONSONANTS
    from app.rule_bind import ph_units
    units = ph_units(phonemes)
    out = []
    for i, (sym, a, b) in enumerate(units):
        if sym not in CONSONANTS:
            continue
        nxt = units[i + 1][0] if i + 1 < len(units) else None
        if b > a:
            form = "shadda"
        elif nxt is None:
            form = "stop"
        elif nxt in SHORT:
            long_ = i + 2 < len(units) and units[i + 2][0] in MADD
            form = SHORT[nxt] + ("_long" if long_ else "")
        else:
            form = "sakin"
        out.append((sym, form, i))
    return out


def ayah_cells(eng, parser, s: int, a: int) -> dict:  # type: ignore[no-untyped-def,type-arg]
    ref = eng.reference(s, a)
    cells = [{"cell": f"letter:{c}:{f}", "unit": i} for c, f, i in letter_cells(ref.phonemes)]
    for r in parser.parse(ref.uthmani).rules:
        detail = (r.detail or "").split(";")[-1].strip() if r.rule_type.value.startswith(("ikhfa", "idgham_ghunnah")) \
            else (r.detail or "")
        cells.append({"cell": f"rule:{r.rule_type.value}:{detail}", "word": r.word_index})
    return {"surah": s, "ayah": a, "phonemes": len(ref.phonemes), "cells": cells}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per", type=int, default=5)
    ap.add_argument("--max-phonemes", type=int, default=90, help="skip long ayahs (audio cost)")
    args = ap.parse_args()
    from quran_transcript import Aya
    from app.engine import Engine
    from app.tajweed_rules.parser import TajweedParser
    eng, parser = Engine(), TajweedParser()
    ayahs = []
    for s in range(1, 115):
        a = 1
        while True:
            try:
                Aya(s, a).get()
            except Exception:  # noqa: BLE001 - past the surah's last ayah
                break
            try:
                ayahs.append(ayah_cells(eng, parser, s, a))
            except Exception as e:  # noqa: BLE001 - one unparseable ayah must not stop the plan
                print(f"{s}:{a} skipped: {e!r}"[:160], file=sys.stderr)
            a += 1
    everywhere: dict[str, int] = defaultdict(int)
    for x in ayahs:
        for c in x["cells"]:
            everywhere[c["cell"]] += 1
    need = {c: min(args.per, n) for c, n in everywhere.items()}
    chosen, pool = [], [x for x in ayahs if x["phonemes"] <= args.max_phonemes]
    while any(v > 0 for v in need.values()) and pool:
        def gain(x):  # type: ignore[no-untyped-def]
            g, left = 0, dict(need)
            for c in x["cells"]:
                if left.get(c["cell"], 0) > 0:
                    g += 1
                    left[c["cell"]] -= 1
            return g / x["phonemes"]
        best = max(pool, key=gain)
        if gain(best) == 0:
            break
        pool.remove(best)
        chosen.append(best)
        for c in best["cells"]:
            if need.get(c["cell"], 0) > 0:
                need[c["cell"]] -= 1
    short = {c: args.per - need[c] for c in need if need[c] > 0}
    OUT.write_text(json.dumps({"per": args.per, "ayahs_read": len(ayahs), "cells": len(everywhere),
                               "cells_in_quran": dict(sorted(everywhere.items())),
                               "unfilled": short,
                               "chosen": [{"surah": x["surah"], "ayah": x["ayah"], "phonemes": x["phonemes"]}
                                          for x in chosen]}, ensure_ascii=False, indent=1))
    print(f"{len(ayahs)} ayahs read, {len(everywhere)} cells; chose {len(chosen)} ayahs "
          f"({sum(x['phonemes'] for x in chosen)} phonemes); {len(short)} cells short of {args.per} "
          f"(only rarer long ayahs hold them)")


if __name__ == "__main__":
    main()
