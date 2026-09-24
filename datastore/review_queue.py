#!/usr/bin/env python3
"""Mine the professional recitations for mistakes an expert should confirm by ear.

Runs the serving engine on T300 clips -- posteriors from the dump, audio from EveryAyah so stops are
detected from the waveform and the reviewer can listen -- and appends every proposable candidate
(`app.review.mine`) to research_agency_lab/experiments/review/candidates.jsonl.

Verses recited by ALL reciters come first: the same verse across 41 voices is what later answers
"which verses, words and rules are hardest", and lets the reviewer hear a master and an imam on the
identical words.

    .venv/bin/python -m datastore.review_queue mine [verses_per_reciter] [procs]
    .venv/bin/python -m datastore.review_queue stats
"""

from __future__ import annotations

import collections
import json
import os
import subprocess
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.review import CANDIDATES, REVIEW_DIR, stats  # noqa: E402

DUMP = ROOT / "research_agency_lab/experiments/qaari_keys/modal_T300"
AUDIO = Path(os.path.expanduser("~/.cache/qaari-eval/everyayah"))
SR, HOP = 16000, 640


def audio_path(speaker: str, surah: int, ayah: int) -> Path:
    """The verse's EveryAyah mp3, downloaded once into the shared cache."""
    p = AUDIO / speaker / f"{surah:03d}{ayah:03d}.mp3"
    if not p.is_file() or p.stat().st_size == 0:
        p.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["curl", "-sfL", "--max-time", "60", "-o", str(p),
                        f"https://everyayah.com/data/{speaker}/{surah:03d}{ayah:03d}.mp3"], check=False)
    return p


_eng = None
_lay = None


def _init() -> None:
    global _eng, _lay
    warnings.filterwarnings("ignore")
    from app.engine import Engine  # noqa: PLC0415
    _lay = json.loads((DUMP / "layout.json").read_text())
    _eng = Engine(layout=_lay)


def _one(r: dict) -> list[dict]:  # type: ignore[type-arg]
    import librosa  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    from app.review import mine  # noqa: PLC0415
    try:
        lp = np.fromfile(DUMP / r["file"], dtype="<f4").reshape(r["frames"], _lay["columns"])  # type: ignore[index]
        p = audio_path(r["speaker"], r["sura"], r["aya"])
        wave = None
        if p.is_file() and p.stat().st_size:
            w = librosa.load(str(p), sr=SR, mono=True)[0][: r["frames"] * HOP]
            wave = np.pad(w, (0, r["frames"] * HOP - w.size))
        rep = _eng.analyze(wave, [(r["sura"], r["aya"])], posteriors=lp)  # type: ignore[union-attr]
        return mine(rep, r["speaker"])
    except Exception as exc:  # noqa: BLE001
        print(f"skip {r['id']}: {exc!r}", flush=True)
        return []


def mine_all(per: int = 4, procs: int = 3) -> None:
    by = collections.defaultdict(dict)
    for line in open(DUMP / "index.jsonl"):
        r = json.loads(line)
        if "file" in r:
            by[r["speaker"]][(r["sura"], r["aya"])] = r
    shared = sorted(set.intersection(*(set(v) for v in by.values())))
    # spread the shared verses over the mushaf rather than taking the first few of one surah
    step = max(1, len(shared) // per)
    chosen = shared[::step][:per]
    seen = {json.loads(line)["id"] for line in open(CANDIDATES)} if CANDIDATES.is_file() else set()
    jobs = [by[s][v] for s in sorted(by) for v in chosen]
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    new = 0
    with Pool(procs, initializer=_init) as p, open(CANDIDATES, "a") as f:
        for cands in p.imap_unordered(_one, jobs, chunksize=2):
            for c in cands:
                if c["id"] not in seen:
                    seen.add(c["id"])
                    f.write(json.dumps(c, ensure_ascii=False) + "\n")
                    new += 1
    print(f"{len(jobs)} verses ({', '.join(f'{s}:{a}' for s, a in chosen)}) x {len(by)} reciters "
          f"-> {new} new candidates")
    kinds = collections.Counter()
    for line in open(CANDIDATES):
        c = json.loads(line)
        kinds[(c["kind"], c["severity"])] += 1
    for k, n in sorted(kinds.items()):
        print(f"  {n:5d}  {k[0]:15s} {k[1]}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["mine"]:
        mine_all(*(int(a) for a in sys.argv[2:]))
    else:
        print(json.dumps(stats(), indent=1, ensure_ascii=False))
