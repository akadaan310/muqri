#!/usr/bin/env python3
"""Run the GNU Octave DSP engine on every diagnostic span of benchmark rows.

    python research_agency_lab/compute_bridge/octave_bridge.py runs.jsonl --out runs_octave.jsonl \
        [--local-root /kaggle/input] [--rules madd_tabii,ghunnah,...] [--jobs 4]

Each input row's audio is decoded once (Quran-MD WAV or the EveryAyah MP3 cache), resampled to
16 kHz and written to a scratch WAV. The spans go to ``qaari_features.m`` in batches, one Octave
process per worker, and the measurements come back under ``diag["octave"]``.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.audio import load_audio  # noqa: E402
from benchmarks.run_benchmark import local_dirs  # noqa: E402

OCTAVE_DIR = ROOT / "research_agency_lab" / "substrate_library" / "octave"
FIELDS = ["core_ms", "voiced_frac", "f0_hz", "hnr_db", "f1_hz", "f2_hz", "f3_hz", "nasal_db", "burst_db", "hf_ratio"]
DEFAULT_RULES = {
    "madd_tabii", "madd_muttasil", "madd_munfasil", "madd_lazim", "madd_arid_lissukun", "madd_leen", "madd_badal",
    "madd_iwad", "madd_silah_sughra", "madd_silah_kubra", "ghunnah", "ikhfa", "idgham_ghunnah", "iqlab",
    "ikhfa_shafawi", "idgham_shafawi", "izhar_shafawi", "izhar_halqi", "qalqalah", "hams", "jahr", "safir",
    "tafashhi", "tafkheem", "tarqeeq",
}


def audio_path(row: dict, local: dict[str, Path], cache: Path) -> Path | None:  # type: ignore[type-arg]
    s, a = row["surah"], row["ayah"]
    if row["reciter"] in local:
        p = local[row["reciter"]] / f"{s:03d}_{a:03d}.wav"
        return p if p.exists() else None
    p = cache / row["reciter"] / f"{s:03d}{a:03d}.mp3"
    return p if p.exists() else None


def run_batch(batch: list[tuple[str, str, float, float]]) -> dict[str, dict[str, float]]:
    with tempfile.TemporaryDirectory() as tmp:
        jobs, out = Path(tmp) / "jobs.csv", Path(tmp) / "features.csv"
        with jobs.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["span_id", "wav_path", "start_s", "end_s"])
            w.writerows(batch)
        cmd = ["octave-cli", "--no-gui", "--quiet", "--eval",
               f"addpath('{OCTAVE_DIR}'); warning('off','all'); qaari_features('{jobs}','{out}')"]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if proc.returncode != 0 or not out.exists():
            err = [ln for ln in proc.stderr.splitlines() if ln.startswith("error")]
            raise RuntimeError(f"Octave failed: {' | '.join(err) or proc.stderr[-500:]}")
        res: dict[str, dict[str, float]] = {}
        with out.open() as fh:
            for r in csv.DictReader(fh):
                res[r["span_id"]] = {k: float(r[k]) for k in FIELDS if r.get(k) not in (None, "", "NaN", "nan")}
        return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--local-root", default=None)
    ap.add_argument("--cache-dir", default=str(Path.home() / ".cache" / "qaari-eval" / "everyayah"))
    ap.add_argument("--rules", default=",".join(sorted(DEFAULT_RULES)))
    ap.add_argument("--modes", default="studio")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--batch-rows", type=int, default=40)
    args = ap.parse_args(argv)
    rules, modes = set(args.rules.split(",")), set(args.modes.split(","))
    local = local_dirs(Path(args.local_root)) if args.local_root else {}
    rows = [json.loads(line) for p in args.runs for line in Path(p).read_text(encoding="utf-8").splitlines() if line]
    rows = [r for r in rows if r["mode"] in modes]
    wav_dir = Path(tempfile.mkdtemp(prefix="qaari_wav_"))

    batches: list[list[tuple[str, str, float, float]]] = [[]]
    for ri, row in enumerate(rows):
        src = audio_path(row, local, Path(args.cache_dir))
        if src is None:
            continue
        if src.suffix.lower() == ".wav":
            wav = src  # Octave reads (and resamples) WAV directly
        else:
            wav = wav_dir / f"{ri}.wav"
            sig = load_audio(src, denoise="never")
            sf.write(wav, np.asarray(sig.samples), sig.sr, subtype="PCM_16")
        for di, d in enumerate(row["diagnostics"]):
            loc = d.get("location")
            if d["rule_type"] in rules and loc and loc["end_ms"] > loc["start_ms"]:
                batches[-1].append((f"{ri}:{di}", str(wav), loc["start_ms"] / 1000, loc["end_ms"] / 1000))
        if (ri + 1) % args.batch_rows == 0:
            batches.append([])
    batches = [b for b in batches if b]
    merged: dict[str, dict[str, float]] = {}
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for k, res in enumerate(pool.map(run_batch, batches)):
            merged.update(res)
            print(f"octave batch {k + 1}/{len(batches)}: {len(res)} spans", flush=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for ri, row in enumerate(rows):
            for di, d in enumerate(row["diagnostics"]):
                feats = merged.get(f"{ri}:{di}")
                if feats:
                    d["octave"] = feats
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(merged)} spans measured by Octave -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
