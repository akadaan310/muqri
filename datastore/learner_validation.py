#!/usr/bin/env python3
"""Does one recitation really tell us about the whole Quran? Leave-one-reciter-out, measured.

For each of the 41 reciters in turn: fit the cohort model on the other 40, give the learner model only
k of the held-out reciter's verses (k = 1, 3, 10, 30; five random draws each), and predict that
reciter's pass rate on every rule. Score the predictions against the rates the reciter actually
shows on ALL their other verses (rules with >= 5 held-out instances), weighted by instances.

Three predictors, so the model has to earn its place:
  cohort      the other reciters' mean -- knows nothing about this reciter
  observed    the reciter's own rate on rules seen in the k verses, the cohort mean elsewhere
  model       the Gaussian-conditioning posterior (app.learner): the evidence also moves the unseen rules

Reported separately for rules SEEN in the k verses and rules NOT seen -- the second is the "whole
Quran from one recitation" claim, and it is only true if the model beats the cohort there.

    .venv/bin/python -m datastore.learner_validation
"""

from __future__ import annotations

import gzip
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.learner import CohortModel, MODEL_PATH, verse_tally  # noqa: E402

CLIPS = ROOT / "research_agency_lab/experiments/profiles/clips.jsonl.gz"
OUT = ROOT / "research_agency_lab/experiments/quran/learner_validation.json"


def main() -> None:
    with gzip.open(CLIPS, "rt", encoding="utf-8") as f:
        rows = [r for r in map(json.loads, f) if "error" not in r]
    verses = defaultdict(list)                  # speaker -> per-verse skill tallies (rules, heads, identity)
    for r in rows:
        verses[r["speaker"]].append(verse_tally(r))
    spk = sorted(verses)

    def pool(ts):  # type: ignore[no-untyped-def]
        out = defaultdict(lambda: [0, 0])
        for t in ts:
            for s, (k, n) in t.items():
                out[s][0] += k
                out[s][1] += n
        return dict(out)
    whole = {s: pool(verses[s]) for s in spk}

    # overdispersion, estimated from the training reciters' own verse-to-verse variation
    def phi_of(names):  # type: ignore[no-untyped-def]
        vals = [CohortModel.overdispersion(verses[s]) for s in names]
        return float(np.median(vals))
    res = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))     # (k, kind) -> predictor -> [err*w, w]
    for held in spk:
        train = [s for s in spk if s != held]
        cohort = CohortModel.fit({s: whole[s] for s in train}, phi=phi_of(train))
        idx = {s: i for i, s in enumerate(cohort.skills)}
        base = {s: float(1 / (1 + np.exp(-cohort.mu[i]))) for s, i in idx.items()}
        for k in (1, 3, 10, 30):
            for seed in range(5):
                rng = random.Random(seed * 97 + k)
                pick = set(rng.sample(range(len(verses[held])), min(k, len(verses[held]))))
                ev = pool([v for i, v in enumerate(verses[held]) if i in pick])
                rest = pool([v for i, v in enumerate(verses[held]) if i not in pick])
                post = cohort.posterior(ev)
                for s, (kk, n) in rest.items():
                    if s not in idx or n < 5:
                        continue
                    actual = kk / n
                    seen = s in ev and ev[s][1] > 0
                    pred = {"cohort": base[s],
                            "observed": (ev[s][0] + 0.5) / (ev[s][1] + 1) if seen else base[s],
                            "model": post[s]["estimate"]}
                    kind = "seen" if seen else "not seen"
                    for name, p in pred.items():
                        acc = res[(k, kind)][name]
                        acc[0] += abs(p - actual) * n
                        acc[1] += n
    table = {}
    print(f"{'verses given':>12s} {'rules':>9s} {'cohort MAE':>11s} {'observed MAE':>13s} {'model MAE':>10s} {'model vs cohort':>16s}")
    for (k, kind), d in sorted(res.items(), key=lambda x: (x[0][0], x[0][1])):
        mae = {n: v[0] / v[1] for n, v in d.items()}
        gain = 1 - mae["model"] / mae["cohort"]
        table[f"{k}|{kind}"] = {**{n: round(v, 4) for n, v in mae.items()}, "gain_vs_cohort": round(gain, 4)}
        print(f"{k:12d} {kind:>9s} {mae['cohort']:11.4f} {mae['observed']:13.4f} {mae['model']:10.4f} {100 * gain:15.1f}%")
    full = CohortModel.fit(whole, phi=phi_of(spk))
    print(f"overdispersion phi = {full.phi:.2f}, Ledoit-Wolf shrinkage = {full.shrink:.3f}")
    # the masters' reference for every skill: the anchors' pooled rate
    from datastore.ingest import ladder
    anchors = pool([t for s in spk if ladder(s) == "anchor" for t in verses[s]])
    masters = {sk: round(k / n, 4) for sk, (k, n) in anchors.items() if n >= 20}
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_text(json.dumps({**full.to_json(), "masters": masters}))
    OUT.write_text(json.dumps(table, indent=1))
    print(f"\ncohort model on all 41 -> {MODEL_PATH}\n-> {OUT}")


if __name__ == "__main__":
    main()
