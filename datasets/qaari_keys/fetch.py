#!/usr/bin/env python3
"""QaariKeys stage D: fetch one tier's verses for every catalogued reciter, with open word timings.

* Audio: EveryAyah per-ayah MP3 for every reciter in ``build/reciters.json`` into the shared cache
  ``~/.cache/qaari-eval/everyayah/<folder>/SSSAAA.mp3`` (already-present files are kept). The folders
  quran-align timed (some are 64 kbps encodings) are fetched too, so its timings refer to the exact
  file they were made on.
* Word timings, two independent open sources, written per tier to ``build/timings_<tier>.jsonl``
  (one line per reciter-verse: ``{"source", "reciter", "surah", "ayah", "words": [[w0, w1, ms0, ms1], …]}``,
  words 1-based, w1 exclusive, times relative to the start of the clip):
    - quran-align (C. Fair, CC BY 4.0; ``~/.cache/qaari-eval/quran_align``) on EveryAyah files;
    - quran.com / QDC segments on its own surah recordings: the verse is cut out of the surah file
      into ``~/.cache/qaari-eval/qdc/<reciter_id>/SSSAAA.wav`` (16 kHz mono) and its segments shifted
      to the clip start.

    .venv/bin/python datasets/qaari_keys/fetch.py T10 [--workers 8] [--no-qdc]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"
CACHE = Path.home() / ".cache/qaari-eval"
EVERYAYAH_URL = "https://everyayah.com/data/{folder}/{s:03d}{a:03d}.mp3"
QDC_FILES = "https://api.qurancdn.com/api/qdc/audio/reciters/{rid}/audio_files?chapter={s}&segments=true"
QDC_RECITERS = "https://api.qurancdn.com/api/qdc/audio/reciters"
QDC_SKIP = {168}  # "Kids repeat": every ayah repeated, not a recitation
UA = {"User-Agent": "qaari-eval-dataset/1.0"}


def _get(url: str, timeout: float = 120) -> bytes:
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:  # noqa: S310
                return r.read()
        except Exception:  # noqa: BLE001
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))
    raise AssertionError


def tier_verses(tier: str) -> list[tuple[int, int]]:
    t = next(x for x in json.loads((BUILD / "tiers.json").read_text())["tiers"] if x["name"] == tier)
    return [(v["surah"], v["ayah"]) for v in t["verses"]]


def fetch_everyayah(folder: str, s: int, a: int) -> str:
    f = CACHE / "everyayah" / folder / f"{s:03d}{a:03d}.mp3"
    if f.exists() and f.stat().st_size > 0:
        return "cached"
    f.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = _get(EVERYAYAH_URL.format(folder=folder, s=s, a=a))
    except Exception as exc:  # noqa: BLE001
        return f"missing ({type(exc).__name__})"
    tmp = f.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.rename(f)
    return "downloaded"


def quran_align(verses: set[tuple[int, int]]) -> list[dict[str, object]]:
    out = []
    for p in sorted((CACHE / "quran_align").glob("*.json")):
        try:
            recs = json.loads(p.read_text())
        except json.JSONDecodeError:  # the 2016 release ships a crash log for As-Sudais
            print(f"  quran_align {p.name}: not JSON, skipped", file=sys.stderr)
            continue
        for rec in recs:
            if (rec["surah"], rec["ayah"]) in verses and rec.get("segments"):
                out.append({"source": "quran_align", "reciter": p.stem, "surah": rec["surah"], "ayah": rec["ayah"],
                            "words": [[w0 + 1, w1 + 1, ms0, ms1] for w0, w1, ms0, ms1 in rec["segments"]],
                            "align_stats": rec.get("stats")})
    return out


def qdc(verses: set[tuple[int, int]]) -> list[dict[str, object]]:
    import librosa
    import soundfile as sf
    reciters = [r for r in json.loads(_get(QDC_RECITERS))["reciters"] if r["id"] not in QDC_SKIP]
    out = []
    for r in reciters:
        rid = r["id"]
        for s in sorted({v[0] for v in verses}):
            want = sorted(a for (ss, a) in verses if ss == s)
            clips = [CACHE / "qdc" / str(rid) / f"{s:03d}{a:03d}.wav" for a in want]
            meta = json.loads(_get(QDC_FILES.format(rid=rid, s=s)))["audio_files"][0]
            timings = {t["verse_key"]: t for t in meta["verse_timings"]}
            surah_mp3 = CACHE / "qdc" / str(rid) / f"{s:03d}.mp3"
            if not all(c.exists() for c in clips) and not surah_mp3.exists():
                surah_mp3.parent.mkdir(parents=True, exist_ok=True)
                surah_mp3.write_bytes(_get(meta["audio_url"], timeout=600))
            for a, clip in zip(want, clips):
                t = timings.get(f"{s}:{a}")
                if t is None:
                    continue
                t0, t1 = float(t["timestamp_from"]), float(t["timestamp_to"])
                if not clip.exists():
                    y, _ = librosa.load(str(surah_mp3), sr=16000, mono=True, offset=t0 / 1000, duration=(t1 - t0) / 1000)
                    sf.write(clip, y.astype(np.float32), 16000)
                words = [[int(w), int(w) + 1, round(float(ms0) - t0), round(float(ms1) - t0)]
                         for w, ms0, ms1, *_ in t["segments"]]
                out.append({"source": "qdc", "reciter": f"qdc_{rid}", "name": r["name"],
                            "style": (r.get("style") or {}).get("name") if isinstance(r.get("style"), dict) else r.get("style"),
                            "surah": s, "ayah": a, "clip": str(clip), "words": words})
        print(f"  qdc {rid} {r['name']}: done", file=sys.stderr, flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tier")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-qdc", action="store_true")
    args = ap.parse_args()
    verses = tier_verses(args.tier)
    cat = json.loads((BUILD / "reciters.json").read_text())
    folders = sorted({str(r["folder"]) for r in cat["everyayah"]} |
                     {p.stem for p in (CACHE / "quran_align").glob("*.json")})
    jobs = [(f, s, a) for f in folders for s, a in verses]
    stats: dict[str, int] = {}
    with cf.ThreadPoolExecutor(args.workers) as ex:
        for res in ex.map(lambda j: fetch_everyayah(*j), jobs):
            k = res.split(" ")[0]
            stats[k] = stats.get(k, 0) + 1
    print(f"{args.tier}: {len(verses)} verses x {len(folders)} EveryAyah folders -> {stats}")
    vs = set(verses)
    rows = quran_align(vs) + ([] if args.no_qdc else qdc(vs))
    path = BUILD / f"timings_{args.tier}.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    by = {}
    for r in rows:
        by[r["source"]] = by.get(r["source"], 0) + 1
    print(f"word timings {by} -> {path.relative_to(HERE.parents[1])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
