#!/usr/bin/env python3
"""Continuous cross-validation loop: keep Octave (DSP) and Julia (maths) checking the pipeline.

Run this after every collection wave. It (1) samples the freshly collected rows and re-measures
their diagnostic spans with the *independent* GNU Octave DSP engine, reporting how well the Python
pipeline's ``core_ms`` and formants agree with Octave (Pearson r + median absolute error); and
(2) re-runs the Julia calibration and model discovery on all rows so the reference bands, the
Husary/peer scores and the tempo/duration laws are refreshed and their drift is visible each wave.

    python research_agency_lab/compute_bridge/validate_loop.py \
        --runs 'benchmarks/results/modal/*/runs_*.jsonl' --sample 300

A timestamped JSON report is written under research_agency_lab/experiments/ and a summary printed.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import random
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.roster import ROSTER  # noqa: E402
from benchmarks.run_benchmark import download  # noqa: E402

JULIA = Path.home() / "julia-1.11.5" / "bin" / "julia"
JULIA_PROJ = ROOT / "research_agency_lab" / "substrate_library" / "julia"
OCTAVE_BRIDGE = ROOT / "research_agency_lab" / "compute_bridge" / "octave_bridge.py"
EXPERIMENTS = ROOT / "research_agency_lab" / "experiments"


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3:
        return float("nan")
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys, strict=True))
    dx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    dy = math.sqrt(sum((b - my) ** 2 for b in ys))
    return num / (dx * dy) if dx and dy else float("nan")


def _agreement(pairs: list[tuple[float, float]]) -> dict:
    if not pairs:
        return {"n": 0}
    py, oc = [p for p, _ in pairs], [o for _, o in pairs]
    diffs = [o - p for p, o in pairs]
    return {"n": len(pairs), "pearson_r": round(_pearson(py, oc), 4),
            "median_abs_err": round(statistics.median(abs(d) for d in diffs), 2),
            "mean_err": round(statistics.mean(diffs), 2),
            "p90_abs_err": round(sorted(abs(d) for d in diffs)[int(0.9 * (len(diffs) - 1))], 2)}


def octave_crosscheck(rows: list[dict], sample: int, python: str, cache: Path) -> dict:
    """Re-measure a random sample of spans with Octave and compare to the Python metrics."""
    studio = [r for r in rows if r.get("mode") == "studio"]
    if not studio:
        return {"skipped": "no studio rows"}
    random.seed(0)
    sub = random.sample(studio, min(sample, len(studio)))
    # Ensure the sampled audio is on disk (Octave needs the waveform).
    have = 0
    for r in sub:
        folder = r["reciter"]
        if folder in ROSTER and download(folder, r["surah"], r["ayah"], cache) is not None:
            have += 1
    with tempfile.TemporaryDirectory() as tmp:
        sub_path = Path(tmp) / "sample.jsonl"
        sub_path.write_text("\n".join(json.dumps(r) for r in sub), encoding="utf-8")
        oct_out = Path(tmp) / "sample_octave.jsonl"
        proc = subprocess.run(
            [python, str(OCTAVE_BRIDGE), str(sub_path), "--out", str(oct_out),
             "--modes", "studio", "--cache-dir", str(cache), "--jobs", "4"],
            capture_output=True, text=True)
        if proc.returncode != 0 or not oct_out.exists():
            return {"error": proc.stderr[-400:], "audio_available": have, "sampled": len(sub)}
        measured = [json.loads(line) for line in oct_out.read_text().splitlines() if line]
    core, f1, f2 = [], [], []
    for r in measured:
        for d in r["diagnostics"]:
            o, m = d.get("octave"), d.get("metrics") or {}
            if not o:
                continue
            if "core_ms" in o and "core_ms" in m and m["core_ms"] > 0:
                core.append((m["core_ms"], o["core_ms"]))
            if o.get("f1_hz") and m.get("f1_hz"):
                f1.append((m["f1_hz"], o["f1_hz"]))
            if o.get("f2_hz") and m.get("f2_hz"):
                f2.append((m["f2_hz"], o["f2_hz"]))
    return {"sampled_rows": len(sub), "audio_available": have,
            "core_ms": _agreement(core), "f1_hz": _agreement(f1), "f2_hz": _agreement(f2)}


def _julia(script: str, out: Path, runs: list[str]) -> dict:
    if not JULIA.exists():
        return {"error": f"julia not found at {JULIA}"}
    proc = subprocess.run([str(JULIA), f"--project={JULIA_PROJ}", str(JULIA_PROJ / script), str(out), *runs],
                          capture_output=True, text=True, cwd=str(ROOT))
    res = {"rc": proc.returncode, "stdout_tail": proc.stdout.splitlines()[-12:]}
    if out.exists():
        try:
            res["data"] = json.loads(out.read_text())
        except json.JSONDecodeError:
            pass
    if proc.returncode != 0:
        res["stderr_tail"] = proc.stderr.splitlines()[-8:]
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="glob(s) of run jsonl files")
    ap.add_argument("--sample", type=int, default=300, help="rows to Octave-cross-check")
    ap.add_argument("--python", default=str(ROOT / ".venv" / "bin" / "python"))
    ap.add_argument("--cache-dir", default=str(Path.home() / ".cache" / "qaari-eval" / "everyayah"))
    ap.add_argument("--calibration-out", default=str(ROOT / "app" / "data" / "calibration.json"))
    ap.add_argument("--skip-octave", action="store_true")
    ap.add_argument("--skip-julia", action="store_true")
    args = ap.parse_args(argv)

    paths = sorted({p for g in args.runs for p in glob.glob(g)})
    if not paths:
        print("no run files matched", file=sys.stderr)
        return 2
    rows = [json.loads(line) for p in paths for line in Path(p).read_text().splitlines() if line.strip()]
    reciters = sorted({r["reciter"] for r in rows})
    print(f"loaded {len(rows)} rows from {len(paths)} files; {len(reciters)} reciters")

    report: dict = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "n_rows": len(rows),
                    "n_files": len(paths), "reciters": reciters}

    if not args.skip_octave:
        print("== Octave DSP cross-check ==")
        report["octave"] = octave_crosscheck(rows, args.sample, args.python, Path(args.cache_dir))
        c = report["octave"].get("core_ms", {})
        print(f"   core_ms: n={c.get('n')} r={c.get('pearson_r')} "
              f"medAE={c.get('median_abs_err')}ms p90AE={c.get('p90_abs_err')}ms")

    if not args.skip_julia:
        print("== Julia calibration ==")
        cal = _julia("calibrate.jl", Path(args.calibration_out), paths)
        report["calibration"] = {k: cal[k] for k in ("rc", "stdout_tail", "stderr_tail") if k in cal}
        for line in cal.get("stdout_tail", []):
            print("   " + line)
        print("== Julia discovery ==")
        disc_out = EXPERIMENTS / "discovery_latest.json"
        disc = _julia("discover.jl", disc_out, paths)
        report["discovery"] = {k: disc[k] for k in ("rc", "stdout_tail", "stderr_tail") if k in disc}
        for line in disc.get("stdout_tail", []):
            print("   " + line)

    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    out = EXPERIMENTS / f"validation_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nvalidation report -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
