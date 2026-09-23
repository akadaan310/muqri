"""Masters vs non-masters, and planted-error detection on QuranMB.v2.

1. Within QuranMB (same verses, same readers — no text confound):
   - clip-level: does the lahn flag count rise with the number of annotated errors (Spearman ρ), and
     separate clips with ≥ 1 letter/vowel error from clean clips (AUROC)?
   - error-level recall: for each annotated substitution, is there a non-PASS lahn verdict of the
     same kind (consonant vs vowel) in the clip whose alternative matches the produced phone?
2. Masters (studio-all, 6 ijazah reciters, full Quran) vs learners: AUROC of the perfection index.
   Confounded by text (different verses, clip lengths) — reported, but the within-QuranMB numbers are
   the clean test.

    .venv/bin/python research_agency_lab/experiments/learner_eval/analyze_learners.py QMB.jsonl [--json OUT] [--gate]
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
VOWELS = {"a", "i", "u", "aa", "ii", "uu", "A", "I", "U"}
# QuranMB phone -> our romanised consonant (for matching the produced phone to the flagged alternative)
QMB_TO_OURS = {"<": "'", "E": "ʿ", "x": "kh", "$": "sh", "g": "gh", "H": "ḥ", "S": "ṣ", "D": "ḍ", "T": "ṭ", "Z": "ẓ",
               "*": "dh", "^": "th", "v": "th", "q": "q", "k": "k", "s": "s", "t": "t", "d": "d", "z": "z", "h": "h"}
OUR_VOWEL = {"a": "a", "i": "i", "u": "u", "aa": "ā", "ii": "ī", "uu": "ū"}


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score_pos > score_neg) with ties counted half."""
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(allv) + 1)
    for v in np.unique(allv):  # average ranks for ties
        m = allv == v
        ranks[m] = ranks[m].mean()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = x.argsort().argsort(), y.argsort().argsort()
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rows")
    ap.add_argument("--json")
    ap.add_argument("--gate", action="store_true", help="also write orchestrator gate metrics")
    args = ap.parse_args()
    rows = [json.loads(line) for line in open(args.rows)]
    ok = [r for r in rows if "error" not in r and r.get("perfection") is not None]
    out: dict[str, object] = {"clips": len(rows), "analysed": len(ok), "failed": len(rows) - len(ok)}

    n_err, n_flag, n_units, has_err = [], [], [], []
    hits = {"consonant": [0, 0], "vowel": [0, 0]}
    for r in ok:
        subs = [e for e in r["edits"] if e["op"] == "replace"]
        cons = [e for e in subs if not set(e["ref"]) & VOWELS]
        vows = [e for e in subs if set(e["ref"]) <= VOWELS]
        flags = [x for x in r["lahn"] if x["status"] != "PASS"]
        n_err.append(len(r["edits"]))
        n_flag.append(len(flags))
        n_units.append(max(1, len(r["lahn"])))
        has_err.append(bool(subs))
        alts = {("vowel" if x["rule"] == "lahn_harakah" else "consonant", x["detail"].split(">")[1]) for x in flags}
        for e in cons:
            want = QMB_TO_OURS.get(e["ann"][0]) if e["ann"] else None
            hits["consonant"][1] += 1
            hits["consonant"][0] += int(want is not None and ("consonant", want) in alts)
        for e in vows:
            want = OUR_VOWEL.get(e["ann"][0]) if e["ann"] else None
            hits["vowel"][1] += 1
            hits["vowel"][0] += int(want is not None and ("vowel", want) in alts)
    e_arr, f_arr, u_arr, h_arr = map(np.array, (n_err, n_flag, n_units, has_err))
    rate = f_arr / u_arr
    out["qmb"] = {
        "clips_with_substitution": int(h_arr.sum()), "clean_clips": int((~h_arr).sum()),
        "spearman_errors_vs_flags": round(spearman(e_arr, f_arr), 3),
        "clip_auroc_flag_rate": round(auroc(rate[h_arr], rate[~h_arr]), 3),
        "flag_rate_per_unit": {"with_errors": round(float(rate[h_arr].mean()), 4),
                               "clean": round(float(rate[~h_arr].mean()), 4)},
        "substitution_recall": {k: {"hit": v[0], "n": v[1], "recall": round(v[0] / v[1], 3) if v[1] else None}
                                for k, v in hits.items()},
        "perfection_median": round(float(np.median([r["perfection"] for r in ok])), 1),
    }

    masters = []
    for f in glob.glob(str(ROOT / "benchmarks/results/modal/studio-all/*.jsonl")):
        for line in open(f):
            p = json.loads(line).get("perfection")
            if p is not None:
                masters.append(p)
    m, lrn = np.array(masters), np.array([r["perfection"] for r in ok])
    out["master_vs_learner"] = {"n_master_ayahs": int(m.size), "n_learner_clips": int(lrn.size),
                                "median_master": round(float(np.median(m)), 1),
                                "median_learner": round(float(np.median(lrn)), 1),
                                "auroc_master_vs_learner": round(auroc(m, lrn), 3),
                                "caveat": "different texts; engine run without calibration on both sides"}
    text = json.dumps(out, ensure_ascii=False, indent=1)
    if args.json:
        Path(args.json).write_text(text + "\n")
    if args.gate:
        gd = ROOT / "research_agency_lab/orchestrator/metrics"
        (gd / "learners.json").write_text(json.dumps(
            {"auroc_master_vs_learner": out["master_vs_learner"]["auroc_master_vs_learner"],  # type: ignore[index]
             "source": args.rows}) + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
