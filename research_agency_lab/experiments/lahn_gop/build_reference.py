"""Write ``app/data/lahn_reference.json`` from text-swap rows (see ``text_swap.py``).

Thresholds per kind: ``fail`` = 1 % and ``warn`` = 5 % quantile of the LLR on the correct text of
the expert reciters, i.e. the expert false-alarm rates by construction. The reciter-LOO recall /
false alarm from ``analyze.py`` are stored alongside as the measured operating point.

    .venv/bin/python research_agency_lab/experiments/lahn_gop/build_reference.py ROWS.jsonl ANALYSIS.json
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from app.lahn.gop import build_reference, gop_z  # noqa: E402

# statistic per kind, chosen on reciter-LOO recall at 1 % expert false alarm (analysis_6reciters.json)
STAT = {"consonant": "z", "vowel": "llr"}

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "app" / "data" / "lahn_reference.json"
MODEL = "TBOGamer22/wav2vec2-quran-phonetics"


def main() -> int:
    rows = [json.loads(line) for line in open(sys.argv[1])]
    analysis = json.loads(open(sys.argv[2]).read())
    negs = [r for r in rows if not r["positive"]]
    bands = build_reference([(r["kind"], r["target"], r["llr"]) for r in negs])
    thresholds, measured = {}, {}
    for kind in ("consonant", "vowel"):
        k = analysis["kinds"][kind]
        if STAT[kind] == "z":
            z = np.array([gop_z(bands, kind, r["target"], r["llr"]) for r in negs if r["kind"] == kind], dtype=float)
            thresholds[kind] = {"stat": "z", "fail": round(float(np.quantile(z, 0.99)), 3),
                                "warn": round(float(np.quantile(z, 0.95)), 3)}
            names = ("n_neg", "n_pos", "z_auroc", "z_loo_recall@0.01", "z_loo_false_alarm@0.01",
                     "z_loo_recall@0.05", "z_loo_false_alarm@0.05")
        else:
            neg = np.array([r["llr"] for r in negs if r["kind"] == kind])
            thresholds[kind] = {"stat": "llr", "fail": round(float(np.quantile(neg, 0.01)), 3),
                                "warn": round(float(np.quantile(neg, 0.05)), 3)}
            names = ("n_neg", "n_pos", "auroc", "loo_recall@0.01", "loo_false_alarm@0.01",
                     "loo_recall@0.05", "loo_false_alarm@0.05")
        measured[kind] = {m.removeprefix("z_"): k[m] for m in names}
    out = {
        "model": MODEL,
        "thresholds": thresholds,
        "built": datetime.date.today().isoformat(),
        "reciters": sorted({r["reciter"] for r in rows}),
        "method": "LLR quantiles on correct expert text; recall from counterfactual text swaps, reciter-LOO",
        "measured": measured,
        "bands": {key: {"median": round(b.median, 4), "scale": round(b.scale, 4), "n": b.n}
                  for key, b in sorted(bands.items())},
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
