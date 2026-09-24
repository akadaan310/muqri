#!/usr/bin/env python3
"""Aggregate benchmark runs into per-reciter scores and build the two reciter indices.

Outputs ``benchmarks/results/summary.json`` and ``summary.md``, plus
``index/masterclass_reciters.faiss`` and ``index/taraweeh_reciters.faiss`` from the studio-mode
fingerprints.

Timing false positives: on master reciters every duration-rule FAIL (Mudood, Ghunnah, Noon/Meem
holds, Izhaar) is treated as a measurement false positive, since these reciters are the reference.
The adapter's effect is the relative drop in those FAILs from ``studio`` to ``taraweeh_adapted`` mode.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.calibration import Calibration  # noqa: E402
from app.fingerprint import Fingerprint, ReciterIndex, ReciterProfile, merge_fingerprints  # noqa: E402
from app.models import MADD_RULES, MEEM_RULES, NOON_RULES, RuleType, rule_category  # noqa: E402
from app.scoring import CATEGORY_WEIGHTS  # noqa: E402

TIMING_RULES = {r.value for r in (*MADD_RULES, *NOON_RULES, *MEEM_RULES, RuleType.GHUNNAH)} - {
    RuleType.IDGHAM_NO_GHUNNAH.value}
EXCLUDED = {"SKIPPED", "VALID_NECESSARY_PAUSE"}


def load_rows(paths: list[Path]) -> list[dict]:  # type: ignore[type-arg]
    rows = []
    for p in paths:
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def aggregate(diags: list[dict], weights: dict[str, float] | None = None) -> dict[str, Any]:  # type: ignore[type-arg]
    weights = weights or CATEGORY_WEIGHTS
    per_cat: dict[str, list[float]] = collections.defaultdict(list)
    status: collections.Counter[str] = collections.Counter()
    timing_fail = timing_n = 0
    for d in diags:
        status[d["status"]] += 1
        if d["status"] in EXCLUDED or d.get("score") is None:
            continue
        cat = rule_category(RuleType(d["rule_type"]))
        per_cat[cat].append(d["score"])
        if d["rule_type"] in TIMING_RULES:
            timing_n += 1
            timing_fail += d["status"] == "FAIL"

    def weighted(cats: list[str]) -> float | None:
        w = sum(weights[c] * len(per_cat[c]) for c in cats if per_cat.get(c))
        return None if not w else sum(weights[c] * sum(per_cat[c]) for c in cats if per_cat.get(c)) / w * 100

    return {
        "perfection": weighted([c for c in weights if c != "sifaat"]),
        "sifaat": weighted(["sifaat"]),
        "categories": {c: round(float(np.mean(v)) * 100, 1) for c, v in sorted(per_cat.items())},
        "status_counts": dict(status),
        "timing_rules": timing_n,
        "timing_fails": timing_fail,
    }


def recalibrate(diags: list[dict], cal: Calibration) -> list[dict]:  # type: ignore[type-arg]
    """Re-judge raw benchmark diagnostics with the calibration (same code path as the live scorer)."""
    out = []
    for d in diags:
        res = cal.judge(d["rule_type"], d.get("detail", ""), d.get("letter"), d["status"], d.get("metrics", {}))
        if res is not None:
            status, score, z, _ = res
            d = {**d, "status": status.value, "score": score, "z": z}
        out.append(d)
    return out


def rule_gaps(rows: list[dict], cal: Calibration) -> dict[str, dict[str, dict[str, float]]]:  # type: ignore[type-arg]
    """Per reciter and rule key: signed distance from the reference hull in robust-σ units, pass rate."""
    acc: dict[str, dict[str, list[tuple[float, bool]]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        for d in r["diagnostics"]:
            key = d.get("key") or d["rule_type"]
            bands = cal.rules.get(key)
            if not bands:
                continue
            b = bands[0]
            x = d.get("metrics", {}).get(b.metric)
            if x is None or cal.unreliable(d.get("metrics", {}), b.metric.endswith("counts")):
                continue
            signed = (x - b.hi) / b.scale if x > b.hi else ((x - b.lo) / b.scale if x < b.lo else 0.0)
            acc[r["reciter"]][key].append((signed, b.z(x) <= 2.0))
    return {rec: {k: {"n": len(v), "median_signed_z": round(float(np.median([a for a, _ in v])), 2),
                      "pass_rate": round(100 * float(np.mean([p for _, p in v])), 1)}
                  for k, v in keys.items()} for rec, keys in acc.items()}


def rule_table(rows: list[dict], mode: str) -> dict[str, dict[str, float]]:  # type: ignore[type-arg]
    """Pass rate per rule type per reciter (for the breakdown table)."""
    out: dict[str, dict[str, list[float]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r["mode"] != mode:
            continue
        for d in r["diagnostics"]:
            if d["status"] in EXCLUDED:
                continue
            out[r["name"]][d["rule_type"]].append(1.0 if d["status"] == "PASS" else 0.0)
    return {n: {k: round(float(np.mean(v)) * 100, 1) for k, v in rules.items()} for n, rules in out.items()}


def build_indices(rows: list[dict], index_dir: Path) -> dict[str, int]:  # type: ignore[type-arg]
    parts: dict[str, list[Fingerprint]] = collections.defaultdict(list)
    meta: dict[str, tuple[str, str]] = {}
    for r in rows:
        fp = r.get("fingerprint")
        if r["mode"] != "studio" or not fp:
            continue

        def arr(xs: list) -> np.ndarray:  # type: ignore[type-arg]
            return np.array([np.nan if x is None else x for x in xs], dtype=np.float64)

        parts[r["reciter"]].append(Fingerprint(np.array(fp["timbre"], dtype=np.float32), arr(fp["tajweed"]),
                                               arr(fp["environment"]), fp["backend"]))
        meta[r["reciter"]] = (r["name"], r["category"])
    indices: dict[str, ReciterIndex] = {}
    for rid, fps in parts.items():
        name, category = meta[rid]
        merged = merge_fingerprints(fps)
        idx = indices.setdefault(category, ReciterIndex(merged.backend, category))
        idx.add(ReciterProfile(rid, name, merged, category, len(fps), "https://everyayah.com/data/" + rid))
    for idx in indices.values():
        idx.save(index_dir)
    return {c: len(i) for c, i in indices.items()}


def markdown(summary: dict) -> str:  # type: ignore[type-arg]
    lines = ["# qaari-eval v2 benchmark", "", f"Up to {summary['verses']} ayahs per reciter "
             "(the most complete reciter's count; see each row).", "",
             "| Reciter | Set | Raw textbook | Perfection (studio) | Perfection (adapted) | Sifaat |"
             " Timing FAILs studio → adapted | FP reduction |", "|---|---|---|---|---|---|---|---|"]
    for r in summary["reciters"]:
        s, a = r["modes"].get("studio", {}), r["modes"].get("taraweeh_adapted", {})

        def f(v: object) -> str:
            return "–" if v is None else f"{v:.1f}"

        red = r.get("timing_fp_reduction_pct")
        lines.append(f"| {r['name']} | {r['category']} | {f(s.get('raw_perfection'))} | {f(s.get('perfection'))} | "
                     f"{f(a.get('perfection'))} | "
                     f"{f(s.get('sifaat'))} | {s.get('timing_fails', '–')} → {a.get('timing_fails', '–')} "
                     f"(of {s.get('timing_rules', '–')}) | {f(red)}{'%' if red is not None else ''} |")
    agg = summary["aggregate"]
    lines += ["", "## Aggregates", ""]
    for k, v in agg.items():
        lines.append(f"- **{k}**: {v}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--out-dir", default=str(ROOT / "benchmarks" / "results"))
    ap.add_argument("--index-dir", default=str(ROOT / "index"))
    ap.add_argument("--no-index", action="store_true")
    ap.add_argument("--calibration", default=str(ROOT / "app" / "data" / "calibration.json"),
                    help="re-judge the raw rows with this calibration ('none' for textbook verdicts only)")
    args = ap.parse_args(argv)
    cal = Calibration.load(args.calibration) if args.calibration != "none" else None
    weights = {c: float((cal.meta.get("category_weights") or {}).get(c, w)) if cal else w
               for c, w in CATEGORY_WEIGHTS.items()}
    paths = [Path(p) for p in args.runs] if args.runs else sorted((ROOT / "benchmarks" / "results").glob("runs*.jsonl"))
    rows = load_rows(paths)
    by: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)  # type: ignore[type-arg]
    names: dict[str, tuple[str, str]] = {}
    for r in rows:
        by[(r["reciter"], r["mode"])].append(r)
        names[r["reciter"]] = (r["name"], r["category"])
    reciters: list[dict[str, Any]] = []
    for rid, (name, category) in names.items():
        modes: dict[str, dict[str, Any]] = {}
        for mode in ("studio", "taraweeh_adapted"):
            rs = by.get((rid, mode), [])
            if rs:
                raw_diags = [d for r in rs for d in r["diagnostics"]]
                agg = aggregate(recalibrate(raw_diags, cal), weights) if cal else aggregate(raw_diags)
                raw = aggregate(raw_diags)
                agg["raw_perfection"], agg["raw_timing_fails"] = raw["perfection"], raw["timing_fails"]
                agg["ayahs"] = len(rs)
                agg["haraka_ms_median"] = float(np.median([r["haraka_ms"] for r in rs]))
                modes[mode] = agg
        entry: dict[str, Any] = {"reciter": rid, "name": name, "category": category, "modes": modes}
        s, a = modes.get("studio"), modes.get("taraweeh_adapted")
        if s and a and s["timing_fails"]:
            fs, fa = s["timing_fails"], a["timing_fails"]
            entry["timing_fp_reduction_pct"] = round(100 * (fs - fa) / fs, 1)
        reciters.append(entry)
    reciters.sort(key=lambda r: (r["category"], -(r["modes"].get("studio", {}).get("perfection") or 0)))

    def mean_of(cat: str, mode: str, key: str) -> float | None:
        vals = [r["modes"][mode][key] for r in reciters if r["category"] == cat and mode in r["modes"]
                and r["modes"][mode][key] is not None]
        return round(float(np.mean(vals)), 1) if vals else None

    tw = [r for r in reciters if r["category"] == "taraweeh"]
    fails_s = sum(r["modes"].get("studio", {}).get("timing_fails", 0) for r in tw)
    fails_a = sum(r["modes"].get("taraweeh_adapted", {}).get("timing_fails", 0) for r in tw)
    summary = {
        "verses": max((r["modes"].get("studio", {}).get("ayahs", 0) for r in reciters), default=0),
        "reciters": reciters,
        "aggregate": {
            "studio_set_mean_perfection": mean_of("studio", "studio", "perfection"),
            "taraweeh_set_mean_perfection_unadapted": mean_of("taraweeh", "studio", "perfection"),
            "taraweeh_set_mean_perfection_adapted": mean_of("taraweeh", "taraweeh_adapted", "perfection"),
            "taraweeh_timing_fails_unadapted": fails_s,
            "taraweeh_timing_fails_adapted": fails_a,
            "taraweeh_timing_fp_reduction_pct": round(100 * (fails_s - fails_a) / fails_s, 1) if fails_s else None,
        },
        "pass_rate_by_rule_studio_mode": rule_table(rows, "studio"),
        "calibration": None if cal is None else {"path": args.calibration, "count_scale": cal.count_scale,
                                                 "category_weights": weights, "rule_keys": len(cal.rules)},
    }
    if cal is not None:
        summary["rule_gaps_vs_reference"] = rule_gaps([r for r in rows if r["mode"] == "studio"], cal)
    if not args.no_index:
        summary["indices"] = build_indices(rows, Path(args.index_dir))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "summary.md").write_text(markdown(summary), encoding="utf-8")
    print(markdown(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
