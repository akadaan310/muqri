#!/usr/bin/env python3
"""Run the engine over the planned ayahs for each reciter and keep everything it measured.

Per (reciter, ayah): the EveryAyah mp3 (downloaded once into the shared cache), decoded to 16 kHz, run
through Engine.analyze with the makhraj test on, and the full measurements written to
data/measure/<reciter>/<sss><aaa>.json. Resumable: done clips are skipped. The learner's own
recordings come in through `--sessions`: the latest take A of every session exercise, measured the
same way (a drill's lines are text, not verses).

    .venv/bin/python research_agency_lab/experiments/letter_corpus/measure.py --workers 3 \\
        --reciters Husary_128kbps Minshawy_Murattal_128kbps Abdul_Basit_Murattal_192kbps \\
                   Alafasy_128kbps Abdurrahmaan_As-Sudais_192kbps
    .venv/bin/python research_agency_lab/experiments/letter_corpus/measure.py --sessions
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

HERE = Path(__file__).parent
DATA = HERE / "data"
PLAN = HERE / "plan.json"
LEARNER = "learner"
_ENG = None


def _engine():  # type: ignore[no-untyped-def]
    global _ENG
    if _ENG is None:
        from app.engine import Engine
        _ENG = Engine()
    return _ENG


def one(job: tuple[str, int, int]) -> str:
    rec, s, a = job
    out = DATA / "measure" / rec / f"{s:03d}{a:03d}.json"
    if out.is_file():
        return f"{rec} {s}:{a} done before"
    from app.webapp import decode_upload
    from datastore.review_queue import audio_path
    p = audio_path(rec, s, a)
    if not p.is_file() or p.stat().st_size == 0:
        return f"{rec} {s}:{a} NO AUDIO"
    t = time.time()
    try:
        wave = decode_upload(p.read_bytes())
        rep = _engine().analyze(wave, [(s, a)], makhraj=True)
    except Exception as e:  # noqa: BLE001 - one bad clip must not stop the run
        return f"{rec} {s}:{a} ERROR {e!r}"[:200]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"reciter": rec, "surah": s, "ayah": a, "audio": str(p),
                               "measurements": rep["measurements"]}, ensure_ascii=False))
    return f"{rec} {s}:{a} {time.time() - t:.1f} s"


def sessions() -> None:
    """The learner's latest take A of every session exercise, measured with the makhraj test on."""
    from app.sessions import exercises, parts
    from app.webapp import SESSIONS_DIR, decode_upload
    audio_ext = {".wav", ".mp3", ".m4a", ".ogg", ".webm", ".flac", ".aac", ".bin", ".mp4"}
    for ex in exercises().values():
        d = SESSIONS_DIR / ex.id / "A"
        takes = sorted(p for p in d.iterdir() if p.suffix.lower() in audio_ext) if d.is_dir() else []
        if not takes:
            continue
        p = takes[-1]
        out = DATA / "measure" / LEARNER / f"{ex.id}_{p.stem}.json"
        if out.is_file():
            continue
        wave = decode_upload(p.read_bytes())
        rep = _engine().analyze(wave, parts(ex), wajh=ex.wajh, makhraj=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"reciter": LEARNER, "exercise": ex.id, "surah": ex.surah, "lines": list(ex.lines),
                                   "audio": str(p), "measurements": rep["measurements"]}, ensure_ascii=False))
        print(f"{LEARNER} {ex.id} {p.name}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reciters", nargs="*", default=[])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--sessions", action="store_true", help="measure the learner's session takes")
    args = ap.parse_args()
    if args.sessions:
        sessions()
    plan = json.loads(PLAN.read_text())
    jobs = [(r, x["surah"], x["ayah"]) for r in args.reciters for x in plan["chosen"]]
    if not jobs:
        return
    with ProcessPoolExecutor(args.workers) as ex:
        for i, msg in enumerate(ex.map(one, jobs, chunksize=4), 1):
            print(f"[{i}/{len(jobs)}] {msg}", flush=True)


if __name__ == "__main__":
    main()
