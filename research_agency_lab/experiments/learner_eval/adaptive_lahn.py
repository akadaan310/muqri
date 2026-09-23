"""Offline test of lahn decision rules on QuranMB rows (no audio re-run): expert thresholds vs
speaker-adaptive (within-clip) standardisation.

Rules, each tuned to the same per-unit flag budget so they are compared at equal false-alarm cost:
  raw       flag if LLR < τ                          (τ from expert text, as deployed)
  clip      flag if (med_clip − LLR) / MAD_clip > κ  (the speaker's own clip is the reference)
Scored by substitution recall (annotated substitution with a matching flagged alternative) and
clip-level AUROC (flag rate, clips with vs without substitutions).

    .venv/bin/python research_agency_lab/experiments/learner_eval/adaptive_lahn.py QMB.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_learners import OUR_VOWEL, QMB_TO_OURS, VOWELS, auroc  # noqa: E402

BUDGETS = (0.01, 0.02, 0.05)


def kind(x: dict) -> str:  # type: ignore[type-arg]
    return "vowel" if x["rule"] == "lahn_harakah" else "consonant"


def scores(rows: list[dict], rule: str) -> list[list[float]]:  # type: ignore[type-arg]
    """Per clip, per lahn unit: a suspicion score (higher = more likely an error)."""
    out = []
    for r in rows:
        s = []
        llr = {k: np.array([x["llr"] for x in r["lahn"] if kind(x) == k]) for k in ("consonant", "vowel")}
        for x in r["lahn"]:
            v = llr[kind(x)]
            if rule == "raw":
                s.append(-x["llr"])
            else:
                med = float(np.median(v))
                mad = max(1.0, 1.4826 * float(np.median(np.abs(v - med))))
                s.append((med - x["llr"]) / mad)
        out.append(s)
    return out


def evaluate(rows: list[dict], sc: list[list[float]], budget: float) -> dict[str, float]:  # type: ignore[type-arg]
    res = {}
    for k in ("consonant", "vowel"):
        flat = np.array([s for r, ss in zip(rows, sc) for x, s in zip(r["lahn"], ss) if kind(x) == k])
        cut = float(np.quantile(flat, 1 - budget))
        hit = n = 0
        for r, ss in zip(rows, sc):
            flagged = {x["detail"].split(">")[1] for x, s in zip(r["lahn"], ss) if kind(x) == k and s > cut}
            for e in r["edits"]:
                if e["op"] != "replace" or not e["ann"]:
                    continue
                is_vowel = set(e["ref"]) <= VOWELS
                if (k == "vowel") != is_vowel:
                    continue
                want = OUR_VOWEL.get(e["ann"][0]) if is_vowel else QMB_TO_OURS.get(e["ann"][0])
                n += 1
                hit += int(want in flagged)
        res[f"{k}_recall"] = round(hit / n, 3) if n else float("nan")
    rate = np.array([np.mean([s > np.quantile(np.concatenate([np.array(q) for q in sc]), 1 - budget) for s in ss])
                     if ss else 0.0 for ss in sc])
    has = np.array([any(e["op"] == "replace" for e in r["edits"]) for r in rows])
    res["clip_auroc"] = round(auroc(rate[has], rate[~has]), 3)
    return res


def main() -> int:
    rows = [json.loads(line) for line in open(sys.argv[1])]
    rows = [r for r in rows if "error" not in r and r.get("lahn")]
    print(f"{len(rows)} clips")
    for rule in ("raw", "clip"):
        sc = scores(rows, rule)
        for b in BUDGETS:
            print(f"  {rule:5s} flag budget {b:.0%}: {evaluate(rows, sc, b)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
