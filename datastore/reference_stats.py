#!/usr/bin/env python3
"""Reference distributions that give every measurement in a report its meaning.

A consumer app sees a learner's number -- a madd held 3.1 counts, a ghunnah whose evidence margin is
-4.2 -- and must know where that sits. This builds, from the full-dataset pass, the distribution of
the same quantity among the MASTERS (the anchors) and the whole COHORT (41 reciters):

  rules            per rule type: pass rate; for durational rules the quantiles of the held length
                   in counts (the same "given_counts" a report carries)
  characteristics  per letter x head, and per head overall: the quantiles of the evidence margin
                   (log p(expected class) - log p(best other class)) -- the same "llr" a report
                   carries on every letter -- and the share realised (margin > 0)

    .venv/bin/python -m datastore.reference_stats
"""

from __future__ import annotations

import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datastore.ingest import ladder  # noqa: E402

CALC = ROOT / "research_agency_lab/experiments/calculus/data"
CLIPS = ROOT / "research_agency_lab/experiments/profiles/clips.jsonl.gz"
OUT = ROOT / "research_agency_lab/experiments/quran/reference_stats.json"
Q = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]


def quantiles(x: np.ndarray) -> list[float] | None:
    return [round(float(v), 4) for v in np.quantile(x, Q)] if x.size >= 20 else None


def main() -> None:
    out: dict = {"quantile_levels": Q, "masters": "anchor tier (the three Husary recordings)",
                 "cohort": "all 41 T300 reciters", "rules": {}, "characteristics": {}}

    # rules: pass rates and held lengths
    FAIL = {"short", "long", "wrong"}
    rate = defaultdict(lambda: {"masters": [0, 0], "cohort": [0, 0]})
    lens = defaultdict(lambda: {"masters": [], "cohort": []})
    with gzip.open(CLIPS, "rt", encoding="utf-8") as f:
        for r in map(json.loads, f):
            if "error" in r:
                continue
            m = ladder(r["speaker"]) == "anchor"
            for rule, st in r["rules"]:
                if st in FAIL | {"pass"}:
                    for g in (["masters", "cohort"] if m else ["cohort"]):
                        rate[rule][g][0] += st == "pass"
                        rate[rule][g][1] += 1
            for rule, counts, _st in r["madd"]:
                for g in (["masters", "cohort"] if m else ["cohort"]):
                    lens[rule][g].append(counts)
    for rule in sorted(set(rate) | set(lens)):
        d = {}
        for g in ("masters", "cohort"):
            k, n = rate[rule][g]
            d[g] = {"pass_rate": round(k / n, 4) if n else None, "instances": n,
                    "counts_quantiles": quantiles(np.asarray(lens[rule][g]))}
        out["rules"][rule] = d

    # characteristics: evidence margins per letter x head
    h = json.loads((CALC / "header.json").read_text())
    n = h["n"]
    meta = np.fromfile(CALC / "meta.i32", dtype="<i4").reshape(n, 6)
    P = np.fromfile(CALC / "probs.f32", dtype="<f4").reshape(n, -1)
    E = np.fromfile(CALC / "expect.i32", dtype="<i4").reshape(n, len(h["heads"]))
    spk = np.fromfile(CALC / "speaker.i32", dtype="<i4")
    master = np.array([t == "anchor" for t in h["tiers"]])[spk]
    col = 0
    for k, (head, classes) in enumerate(zip(h["heads"], h["head_classes"])):
        p = np.log(P[:, col:col + len(classes)] + 1e-12)
        col += len(classes)
        e = E[:, k]
        ok = e >= 0
        pe = p[np.arange(n), np.clip(e, 0, None)]
        other = p.copy()
        other[np.arange(n), np.clip(e, 0, None)] = -np.inf
        margin = pe - other.max(axis=1)                  # the report's llr for this head
        by = {}
        for li, letter in enumerate(h["letters"]):
            sel = ok & (meta[:, 0] == li)
            if sel.sum() < 20:
                continue
            by[letter] = {g: {"margin_quantiles": quantiles(margin[sel & mk]),
                              "realised_rate": round(float((margin[sel & mk] > 0).mean()), 4) if (sel & mk).sum() else None,
                              "instances": int((sel & mk).sum())}
                          for g, mk in (("masters", master), ("cohort", np.ones(n, bool)))}
        out["characteristics"][head] = {
            "all_letters": {g: {"margin_quantiles": quantiles(margin[ok & mk]),
                                "realised_rate": round(float((margin[ok & mk] > 0).mean()), 4),
                                "instances": int((ok & mk).sum())}
                            for g, mk in (("masters", master), ("cohort", np.ones(n, bool)))},
            "by_letter": by}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False))
    print(f"{len(out['rules'])} rules, {len(out['characteristics'])} characteristic heads -> {OUT} "
          f"({OUT.stat().st_size // 1024} KB)")
    g = out["characteristics"]["ghonna"]["by_letter"].get("م", {})
    print("e.g. ghunnah on م, masters' margin quantiles:", g.get("masters", {}).get("margin_quantiles"))
    print("e.g. madd_tabii held counts, masters' quantiles:", out["rules"]["madd_tabii"]["masters"]["counts_quantiles"])


if __name__ == "__main__":
    main()
