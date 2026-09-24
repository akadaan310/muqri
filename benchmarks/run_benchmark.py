#!/usr/bin/env python3
"""Score reciters on the strategic verse set, in studio and Taraweeh-adapted modes.

For every (reciter, ayah, mode) one JSON line is appended to ``--output`` (default
``benchmarks/results/runs.jsonl``) with the compact diagnostics, the summary scores and — in studio
mode — the 232-d fingerprint. Runs are resumable: finished rows are skipped. Shard the work with
``--shard i/n`` to run several processes (or Kaggle/Colab sessions) in parallel.

    python benchmarks/run_benchmark.py                      # all reciters, strategic verses
    python benchmarks/run_benchmark.py --reciters Husary_128kbps --modes studio
    python benchmarks/run_benchmark.py --verses all --shard 0/8   # the entire Qur'an (GPU box)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.audio import AudioError, load_audio  # noqa: E402
from app.calibration import rule_key  # noqa: E402
from app.pipeline import AnalysisOptions, QaariEvaluator  # noqa: E402
from app.quran_text import AYAH_COUNTS, get_ayah_text  # noqa: E402
from benchmarks.roster import EVERYAYAH_URL, QURAN_MD, ROSTER  # noqa: E402

logger = logging.getLogger("benchmark")
STRATEGIC = ROOT / "app" / "data" / "strategic_verses.json"
DEFAULT_OUT = ROOT / "benchmarks" / "results" / "runs.jsonl"


def verse_list(spec: str) -> list[tuple[int, int]]:
    if spec == "strategic":
        data = json.loads(STRATEGIC.read_text(encoding="utf-8"))
        return [(v["surah"], v["ayah"]) for v in data["verses"]]
    if spec == "all":
        return [(s, a) for s in range(1, 115) for a in range(1, AYAH_COUNTS[s - 1] + 1)]
    out = []
    for part in spec.split(","):
        s, rng = part.split(":")
        a0, _, a1 = rng.partition("-")
        out += [(int(s), a) for a in range(int(a0), int(a1 or a0) + 1)]
    return out


def local_dirs(root: Path) -> dict[str, Path]:
    """Map EveryAyah folder -> directory holding ``SSS_AAA.wav`` under a Quran-MD root (any depth)."""
    out: dict[str, Path] = {}
    # Fixed depths only: a recursive glob over ~190k mounted WAV files takes minutes per reciter.
    prefixes = ["", "*/", "*/*/", "*/*/*/", "*/*/*/*/"]
    for folder, qid in QURAN_MD.items():
        for pre in prefixes:
            hits = [h for pat in (f"{pre}{qid}/001_001.wav", f"{pre}{qid}/{qid}/001_001.wav")
                    for h in root.glob(pat)]
            if hits:
                out[folder] = hits[0].parent
                break
    return out


def download(folder: str, surah: int, ayah: int, cache: Path) -> Path | None:
    dest = cache / folder / f"{surah:03d}{ayah:03d}.mp3"
    if dest.exists() and dest.stat().st_size > 1024:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = EVERYAYAH_URL.format(folder=folder, surah=surah, ayah=ayah)
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "qaari-eval"}),
                                        timeout=60) as resp:  # noqa: S310
                data = resp.read()
            tmp = dest.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return dest
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(2 ** (attempt + 1))
    return None


def compact(diag: dict) -> dict:  # type: ignore[type-arg]
    keep = ("rule_type", "detail", "word", "letter", "status", "score", "measured_harakat", "expected_harakat_range",
            "metrics", "location")
    out = {k: diag[k] for k in keep if k in diag}
    out["key"] = rule_key(diag["rule_type"], diag.get("detail", ""), diag.get("letter"))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reciters", nargs="*", help="EveryAyah folders (default: the whole roster)")
    ap.add_argument("--verses", default="strategic", help="'strategic', 'all', or e.g. '1:1-7,113:1-5'")
    ap.add_argument("--modes", nargs="*", default=["studio", "taraweeh_adapted"])
    ap.add_argument("--tareeq", default="shatibiyyah")
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    ap.add_argument("--cache-dir", default=str(Path.home() / ".cache" / "qaari-eval" / "everyayah"))
    ap.add_argument("--shard", default="0/1", help="i/n: process every n-th (reciter, ayah) pair")
    ap.add_argument("--timbre-backend", default="ecapa")
    ap.add_argument("--local-root", default=None,
                    help="Quran-MD WAV root (e.g. /kaggle/input); reciters found there are read locally")
    ap.add_argument("--threads", type=int, default=0, help="torch intra-op threads (0 = library default)")
    ap.add_argument("--skip-done", default=None,
                    help="JSON list of [reciter, surah, ayah, mode] already computed elsewhere (e.g. an earlier run)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(message)s")

    reciters = args.reciters or list(ROSTER)
    verses = verse_list(args.verses)
    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    done: set[tuple[str, int, int, str]] = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                done.add((r["reciter"], r["surah"], r["ayah"], r["mode"]))
            except (json.JSONDecodeError, KeyError):
                continue
    if args.skip_done and Path(args.skip_done).exists():
        done |= {tuple(k) for k in json.loads(Path(args.skip_done).read_text(encoding="utf-8"))}  # type: ignore[misc]
    cache = Path(args.cache_dir)
    texts = {v: get_ayah_text(*v) for v in verses}
    evaluators = {m: QaariEvaluator(AnalysisOptions(aligner="ctc", index_dir=None, mode=m, tareeq=args.tareeq,
                                                    denoise="never", timbre_backend=args.timbre_backend,
                                                    compute_fingerprint=(m == "studio"), calibration=None))
                  for m in args.modes}
    # Share one aligner and embedder across modes.
    first = next(iter(evaluators.values()))
    for e in evaluators.values():
        e._aligner = first.aligner  # noqa: SLF001
        if "studio" in evaluators:
            e._embedder = evaluators["studio"].embedder  # noqa: SLF001

    jobs = [(f, v) for f in reciters for v in verses]
    jobs = [j for k, j in enumerate(jobs) if k % shard_n == shard_i]
    # Don't even decode audio for jobs whose every mode is already done.
    jobs = [(f, v) for f, v in jobs if any((f, v[0], v[1], m) not in done for m in args.modes)]
    if args.threads:
        import torch

        torch.set_num_threads(args.threads)
    local = local_dirs(Path(args.local_root)) if args.local_root else {}
    if args.local_root:
        logger.info("Local Quran-MD sources: %s", {k: str(v) for k, v in local.items()})

    def fetch(job: tuple[str, tuple[int, int]]) -> Path | None:
        folder, (surah, ayah) = job
        if folder in local:
            p = local[folder] / f"{surah:03d}_{ayah:03d}.wav"
            return p if p.exists() else None
        return download(folder, surah, ayah, cache)

    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = dict(zip(jobs, pool.map(fetch, jobs), strict=True))
    n = 0
    with out.open("a", encoding="utf-8") as fh:
        for (folder, (surah, ayah)) in jobs:
            name, category = ROSTER.get(folder, (folder, "unknown"))
            path = paths[(folder, (surah, ayah))]
            if path is None:
                logger.warning("%s %d:%d unavailable", folder, surah, ayah)
                continue
            try:
                audio = load_audio(path, denoise="never")
            except AudioError as exc:
                logger.warning("%s %d:%d unreadable: %s", folder, surah, ayah, exc)
                continue
            for mode, ev in evaluators.items():
                if (folder, surah, ayah, mode) in done:
                    continue
                t0 = time.time()
                try:
                    res = ev.analyze_signal(audio, texts[(surah, ayah)], reference=f"{surah}:{ayah}")
                except Exception as exc:  # noqa: BLE001 - keep the batch going
                    logger.warning("%s %d:%d %s failed: %s", folder, surah, ayah, mode, exc)
                    continue
                s = res.report["recitation_summary"]
                row = {
                    "reciter": folder, "name": name, "category": category, "surah": surah, "ayah": ayah, "mode": mode,
                    "tareeq": args.tareeq, "perfection": s["tajweed_perfection_index"], "sifaat": s["sifaat_score"],
                    "haraka_ms": s["base_haraka_duration_ms"], "environment": s["acoustic_environment"],
                    "diagnostics": [compact(d) for d in res.report["detailed_rule_diagnostics"]],
                    "seconds": round(time.time() - t0, 2), "duration_s": round(audio.duration_s, 3),
                }
                if res.fingerprint is not None:
                    row["fingerprint"] = {"timbre": [round(float(x), 6) for x in res.fingerprint.timbre],
                                          "tajweed": [None if x != x else round(float(x), 5)
                                                      for x in res.fingerprint.tajweed],
                                          "environment": [None if x != x else round(float(x), 5)
                                                          for x in res.fingerprint.environment],
                                          "backend": res.fingerprint.backend}
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                n += 1
            logger.info("%s %d:%d done", name, surah, ayah)
    logger.info("Wrote %d rows to %s", n, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
