#!/usr/bin/env python3
"""The rule layer of the recitation graph, from the full-dataset engine pass (11,996 T300 verses).

Per reciter: which rules they keep (pass rate), their own lengths per madd type, which they break.
Across reciters: the rules every reciter keeps, the rules every master keeps that others break, rule
failures that come together in the same verse (pointwise mutual information), and the hardest rules
and verses over all 41 voices. Written as JSON for the graph builder and printed as findings.

    .venv/bin/python -m datastore.rule_layer
"""

from __future__ import annotations

import collections
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datastore.ingest import ladder  # noqa: E402

CLIPS = ROOT / "research_agency_lab/experiments/profiles/clips.jsonl"
OUT = ROOT / "research_agency_lab/experiments/calculus/rule_layer_T300.json"
FAIL = {"short", "long", "wrong"}
MASTERS = {"anchor", "studio"}


def main() -> None:
    rows = [r for r in map(json.loads, open(CLIPS)) if "error" not in r]
    tier = {r["speaker"]: ladder(r["speaker"]) for r in rows}
    keep = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))   # spk -> rule -> [pass, n]
    lengths = collections.defaultdict(lambda: collections.defaultdict(list))
    verse_fail: list[set[str]] = []
    verse_acc = collections.defaultdict(list)
    for r in rows:
        fails = set()
        for rule, st in r["rules"]:
            if st in FAIL | {"pass"}:
                keep[r["speaker"]][rule][1] += 1
                keep[r["speaker"]][rule][0] += st == "pass"
                if st in FAIL:
                    fails.add(rule)
        for rule, counts, _st in r["madd"]:
            lengths[r["speaker"]][rule].append(counts)
        verse_fail.append(fails)
        if r["word_accuracy"] is not None:
            verse_acc[(r["surah"], r["ayah"])].append((r["speaker"], r["word_accuracy"]))

    rules = sorted({rule for s in keep.values() for rule in s})
    rate = {s: {rule: (v[0] / v[1], v[1]) for rule, v in d.items() if v[1] >= 5} for s, d in keep.items()}
    spk = sorted(rate)
    universal = [ru for ru in rules if all(ru in rate[s] and rate[s][ru][0] >= 0.95 for s in spk if ru in rate[s])
                 and sum(ru in rate[s] for s in spk) >= 30]
    m_keep = [ru for ru in rules
              if all(rate[s].get(ru, (1, 0))[0] >= 0.9 for s in spk if tier[s] in MASTERS and ru in rate[s])
              and sum(1 for s in spk if tier[s] not in MASTERS and ru in rate[s] and rate[s][ru][0] < 0.8) >= 3]
    hardest = sorted(((ru, statistics.mean(rate[s][ru][0] for s in spk if ru in rate[s]),
                       sum(rate[s][ru][1] for s in spk if ru in rate[s])) for ru in rules
                      if sum(ru in rate[s] for s in spk) >= 20), key=lambda x: x[1])

    # rule failures that travel together within a verse
    V = len(verse_fail)
    single = collections.Counter(ru for f in verse_fail for ru in f)
    pair = collections.Counter((a, b) for f in verse_fail for a in f for b in f if a < b)
    together = sorted(((a, b, c, math.log2((c / V) / ((single[a] / V) * (single[b] / V))))
                       for (a, b), c in pair.items() if c >= 20), key=lambda x: -x[3])

    # hardest verses: shared by >= 30 reciters, lowest mean word accuracy
    verses = sorted(((sa, statistics.mean(a for _s, a in v), len(v)) for sa, v in verse_acc.items() if len(v) >= 30),
                    key=lambda x: x[1])

    print(f"{len(rows)} verses, {len(spk)} reciters, {len(rules)} rules")
    print(f"\nRULES EVERY RECITER KEEPS (>= 95 % on each of >= 30 reciters): {', '.join(universal) or '-'}")
    print(f"RULES EVERY MASTER KEEPS (>= 90 %) THAT >= 3 OTHERS BREAK (< 80 %): {', '.join(m_keep) or '-'}")
    print("\nHARDEST RULES (mean pass rate over reciters)")
    for ru, m, n in hardest[:10]:
        print(f"  {ru:24s} {m:.3f}   n={n}")
    print("\nRULE FAILURES THAT TRAVEL TOGETHER (same verse; PMI bits, >= 20 verses)")
    for a, b, c, p in together[:10]:
        print(f"  {a:22s} + {b:22s} PMI {p:5.2f}  n={c}")
    print("\nHARDEST VERSES (recited by >= 30 reciters; mean share of words fully correct)")
    for (s, a), m, n in verses[:10]:
        print(f"  {s}:{a:<4d} {m:.3f}  over {n} reciters")
    print("\nTHEIR OWN LENGTHS: median madd tabii / arid / munfasil in counts, masters vs hadr")
    for s in spk:
        if tier[s] in ("anchor",) or s.startswith(("Maher", "Saood", "Salah_Al")):
            L = {ru: round(statistics.median(v), 2) for ru, v in lengths[s].items() if len(v) >= 10}
            print(f"  {s:34s} {tier[s]:7s} tabii {L.get('madd_tabii', '-')}  arid {L.get('madd_arid_lissukun', '-')}"
                  f"  munfasil {L.get('madd_munfasil', '-')}  lazim {L.get('madd_lazim', '-')}")

    OUT.write_text(json.dumps({
        "reciters": {s: {"tier": tier[s], "rules": {ru: {"pass_rate": round(v[0], 4), "n": v[1]} for ru, v in rate[s].items()},
                         "lengths": {ru: {"median": round(statistics.median(v), 3), "n": len(v)}
                                     for ru, v in lengths[s].items() if len(v) >= 5}} for s in spk},
        "universal": universal, "masters_keep_others_break": m_keep,
        "hardest_rules": [{"rule": ru, "mean_pass": round(m, 4), "n": n} for ru, m, n in hardest],
        "fail_together": [{"a": a, "b": b, "verses": c, "pmi": round(p, 3)} for a, b, c, p in together],
        "hardest_verses": [{"surah": s, "ayah": a, "mean_word_accuracy": round(m, 4), "reciters": n}
                           for (s, a), m, n in verses],
    }, ensure_ascii=False, indent=1))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
