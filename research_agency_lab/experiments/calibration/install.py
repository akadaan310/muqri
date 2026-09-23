#!/usr/bin/env python3
"""Install a calibrate.jl output as the engine calibration and record the calibration-gate metrics.

1. Copy CALIB.json to ``app/data/calibration.json`` (the previous file is kept as
   ``calibration_installed_v<old>.json`` next to CALIB). The tuned ``category_weights`` are moved to
   ``rejected_category_weights`` unless ``--keep-weights``: the weight search has a degenerate optimum
   (wasl and weight dominate) that buys peer scores without measuring anything.
2. Re-judge every diagnostic of the benchmark rows with ``Calibration.judge`` and write, to
   ``research_agency_lab/orchestrator/metrics/calibration.json``: the near-gold anchors' perfection
   (held out where the anchor is a peer) and per-rule FAIL rate, which the calibration gate judges
   (target ≈ 97, a few percent flagged); and, for information only, every peer's leave-one-out
   perfection and the per-rule FAIL rate over all peers — fast reciters make real mistakes, so no
   minimum applies to them.

    .venv/bin/python research_agency_lab/experiments/calibration/install.py CALIB.json ROWS.jsonl... [--metrics-only]
"""

from __future__ import annotations

import argparse
import collections
import json
import shutil
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from app.calibration import Calibration, rule_key  # noqa: E402

INSTALLED = ROOT / "app" / "data" / "calibration.json"
METRICS = ROOT / "research_agency_lab" / "orchestrator" / "metrics" / "calibration.json"


# Near-gold anchors: slow, studio, teaching-level recitations. They should score ~97 and have only a few
# percent of instances flagged; faster reciters and imams are expected to show real mistakes.
ANCHORS = ("Husary_Muallim_128kbps", "Husary_128kbps", "Husary_128kbps_Mujawwad")
ANCHOR_TARGET = 97.0


def expert_fail_rates(cal: Calibration, peers: set[str], rows: list[Path]) -> dict[str, dict[str, int]]:
    tally: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for path in rows:
        with path.open() as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("reciter") not in peers:
                    continue
                for d in r.get("diagnostics", []):
                    res = cal.judge(d["rule_type"], d.get("detail", ""), d.get("letter"), d["status"],
                                    d.get("metrics", {}))
                    if res is None:
                        continue
                    key = rule_key(d["rule_type"], d.get("detail", ""), d.get("letter"))
                    tally[key][res[0].value] += 1
    return {k: dict(v) for k, v in tally.items()}


def _rel(p: Path) -> str:
    p = p.resolve()
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("calib")
    ap.add_argument("rows", nargs="+")
    ap.add_argument("--keep-weights", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the metrics only; write nothing")
    ap.add_argument("--metrics-only", action="store_true", help="write the gate metrics; leave app/data alone")
    args = ap.parse_args()
    data = json.loads(Path(args.calib).read_text())
    if not args.keep_weights and "category_weights" in data:
        data["rejected_category_weights"] = data.pop("category_weights")
    if not (args.dry_run or args.metrics_only):
        if INSTALLED.exists():
            old = json.loads(INSTALLED.read_text()).get("version", "old")
            shutil.copy(INSTALLED, Path(args.calib).with_name(f"calibration_installed_v{old}.json"))
        INSTALLED.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    cal = Calibration.from_dict(data)
    peers = set(data.get("peers", []))
    rows = [Path(p) for p in args.rows]
    tally = expert_fail_rates(cal, peers, rows)
    judged = {k: sum(v.values()) for k, v in tally.items()}
    rates = {k: round(tally[k].get("FAIL", 0) / n, 4) for k, n in judged.items() if n}
    val = data.get("validation", {})
    loo = {k: v["perfection"] for k, v in val.get("leave_one_peer_out", {}).items() if v.get("perfection") is not None}
    in_sample = {k: v["perfection"] for k, v in val.get("calibrated_in_sample", {}).items()
                 if v.get("perfection") is not None}
    # Held-out score where the anchor is a peer; the calibration's own reference only has in-sample.
    anchors = {a: {"perfection": loo.get(a, in_sample.get(a)), "held_out": a in loo}
               for a in ANCHORS if a in loo or a in in_sample}
    a_tally = expert_fail_rates(cal, set(anchors), rows)
    a_rates = {k: v.get("FAIL", 0) / sum(v.values()) for k, v in a_tally.items() if sum(v.values())}
    held = [v["perfection"] for v in anchors.values() if v["held_out"]]
    out = {
        "calibration_version": data.get("version"),
        "source": _rel(Path(args.calib)),
        "anchor_target": ANCHOR_TARGET,
        "anchors": anchors,
        "anchor_min_held_out": min(held) if held else 0.0,
        "anchor_median_fail_rate": round(statistics.median(a_rates.values()), 4) if a_rates else 1.0,
        "anchor_fail_rate_by_rule": {k: round(v, 4) for k, v in sorted(a_rates.items(), key=lambda kv: -kv[1])},
        # Informational: other reciters are expected to show real mistakes, so no minimum applies.
        "loo_min_peer": min(loo.values()) if loo else 0.0,
        "loo": loo,
        "median_expert_fail_rate": statistics.median(rates.values()) if rates else 1.0,
        "fail_rate_by_rule": dict(sorted(rates.items(), key=lambda kv: -kv[1])),
        "judged_by_rule": judged,
    }
    if not args.dry_run:
        METRICS.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    worst = list(out["fail_rate_by_rule"].items())[:6]
    print(f"anchors {anchors}; anchor median FAIL {100 * out['anchor_median_fail_rate']:.1f}%; "
          f"LOO min peer {out['loo_min_peer']:.1f}; median expert FAIL {100 * out['median_expert_fail_rate']:.1f}%; "
          f"worst {worst}" + ("" if args.dry_run else f" -> {_rel(METRICS)}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
