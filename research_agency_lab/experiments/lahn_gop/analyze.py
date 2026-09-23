"""Recall / false alarm of the lahn-jali GOP detector from text-swap rows, with reciter-LOO thresholds.

Threshold τ_α(kind) = α-quantile of the LLR on the untouched text of the *other* reciters (split
conformal on negatives), so the held-out reciter's false-alarm rate is ≈ α by construction and
its recall on swaps is an honest estimate.

    .venv/bin/python research_agency_lab/experiments/lahn_gop/analyze.py ROWS.jsonl [--json OUT]
"""

from __future__ import annotations

import argparse
import collections
import json

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from app.lahn.gop import build_reference, gop_z  # noqa: E402

ALPHAS = (0.01, 0.05)


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(LLR_pos < LLR_neg): lower LLR means 'error'."""
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1.0
    r_pos = ranks[: len(pos)].sum()
    return float(1 - (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rows")
    ap.add_argument("--json")
    args = ap.parse_args()
    rows = [json.loads(line) for line in open(args.rows)]
    reciters = sorted({r["reciter"] for r in rows})
    out: dict[str, object] = {"n_rows": len(rows), "reciters": reciters, "kinds": {}}
    for kind in ("consonant", "vowel"):
        neg = {rc: np.array([r["llr"] for r in rows if r["reciter"] == rc and r["kind"] == kind and not r["positive"]])
               for rc in reciters}
        pos = {rc: np.array([r["llr"] for r in rows if r["reciter"] == rc and r["kind"] == kind and r["positive"]])
               for rc in reciters}
        all_neg, all_pos = np.concatenate(list(neg.values())), np.concatenate(list(pos.values()))
        k: dict[str, object] = {"n_neg": int(all_neg.size), "n_pos": int(all_pos.size),
                                "auroc": round(auroc(all_pos, all_neg), 4),
                                "tau_all": {str(a): round(float(np.quantile(all_neg, a)), 3) for a in ALPHAS}}
        loo: dict[str, dict[str, object]] = {}
        for a in ALPHAS:
            rec_l, fa_l = [], []
            for rc in reciters:
                train = np.concatenate([neg[o] for o in reciters if o != rc])
                tau = float(np.quantile(train, a))
                if pos[rc].size:
                    rec_l.append(float(np.mean(pos[rc] < tau)))
                if neg[rc].size:
                    fa_l.append(float(np.mean(neg[rc] < tau)))
                loo.setdefault(rc, {})[f"recall@{a}"] = round(rec_l[-1], 3) if pos[rc].size else None
                loo[rc][f"false_alarm@{a}"] = round(fa_l[-1], 3) if neg[rc].size else None
            k[f"loo_recall@{a}"] = round(float(np.mean(rec_l)), 3)
            k[f"loo_false_alarm@{a}"] = round(float(np.mean(fa_l)), 3)
        k["per_reciter"] = loo
        tau1 = float(np.quantile(all_neg, 0.01))
        pairs: dict[str, list[float]] = collections.defaultdict(list)
        for r in rows:
            if r["positive"] and r["kind"] == kind:
                pairs[r["swap"]].append(r["llr"])
        k["per_swap"] = {s: {"n": len(v), "recall@0.01": round(float(np.mean(np.array(v) < tau1)), 3)}
                         for s, v in sorted(pairs.items(), key=lambda kv: -len(kv[1]))}
        # per-symbol standardised score (the runtime path), reference bands built reciter-LOO
        zpos: list[float] = []
        zneg: list[float] = []
        zrec: dict[str, tuple[list[float], list[float]]] = {}
        for rc in reciters:
            ref = build_reference([(r["kind"], r["target"], r["llr"]) for r in rows
                                   if r["reciter"] != rc and not r["positive"]])
            zp = [gop_z(ref, kind, r["target"], r["llr"]) for r in rows
                  if r["reciter"] == rc and r["kind"] == kind and r["positive"]]
            zn = [gop_z(ref, kind, r["target"], r["llr"]) for r in rows
                  if r["reciter"] == rc and r["kind"] == kind and not r["positive"]]
            zrec[rc] = ([z for z in zp if z is not None], [z for z in zn if z is not None])
            zpos += zrec[rc][0]
            zneg += zrec[rc][1]
        # higher z = more suspicious; auroc() expects lower = error, so negate
        k["z_auroc"] = round(auroc(-np.array(zpos), -np.array(zneg)), 4)
        for a in ALPHAS:
            rec_l, fa_l = [], []
            for rc in reciters:
                train = np.concatenate([np.array(zrec[o][1]) for o in reciters if o != rc])
                cut = float(np.quantile(train, 1 - a))
                if zrec[rc][0]:
                    rec_l.append(float(np.mean(np.array(zrec[rc][0]) > cut)))
                if zrec[rc][1]:
                    fa_l.append(float(np.mean(np.array(zrec[rc][1]) > cut)))
            k[f"z_loo_recall@{a}"] = round(float(np.mean(rec_l)), 3)
            k[f"z_loo_false_alarm@{a}"] = round(float(np.mean(fa_l)), 3)
            k[f"z_cut@{a}"] = round(float(np.quantile(np.array(zneg), 1 - a)), 3)
        out["kinds"][kind] = k  # type: ignore[index]
    text = json.dumps(out, ensure_ascii=False, indent=1)
    if args.json:
        open(args.json, "w").write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
