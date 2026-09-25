#!/usr/bin/env python3
"""The held length of a clear nūn / mīm sākinah (iẓhār: ghunnah nāqiṣah) on the masters and the cohort.

The iẓhār rules have no counted length, so a nūn hidden with a full ghunnah before a throat letter
passed: round 5 (r5e3 B, مِّنْ أَلْفِ) held it 1.12 s, 3.62 counts, where the same reader's take A held
1.32 counts and the ghunnah-grade report (textbook band 0.6-1.8) already said "long" -- a verdict
nothing scored. This runs the engine over the T300 posteriors (41 reciters, 12,600 clips; no audio,
so no stop detection -- stops change madd, not the nūn) and writes every iẓhār instance's held
length in counts, from which a ceiling is read as for the other nasal bands (app/submission.py).

    .venv/bin/python research_agency_lab/experiments/izhar/naqisah.py [--workers 3]
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

DUMP = ROOT / "research_agency_lab/experiments/qaari_keys/modal_T300"
OUT = Path(__file__).with_name("naqisah.jsonl")      # committed gzipped: naqisah.jsonl.gz
_ENGINE = None


def _one(rec: dict) -> list[dict]:  # type: ignore[type-arg]
    global _ENGINE
    from app.engine import Engine
    lay = json.loads((DUMP / "layout.json").read_text())
    if _ENGINE is None:
        _ENGINE = Engine(layout=lay)
    lp = np.fromfile(DUMP / rec["file"], dtype="<f4").reshape(rec["frames"], lay["columns"])
    try:
        rep = _ENGINE.analyze(None, [(rec["sura"], rec["aya"])], posteriors=lp)
    except Exception as e:  # noqa: BLE001 - one bad clip must not stop the pass
        return [{"speaker": rec["speaker"], "surah": rec["sura"], "ayah": rec["aya"], "error": repr(e)[:200]}]
    a = rep["ayahs"][0]
    return [{"speaker": rec["speaker"], "surah": rec["sura"], "ayah": rec["aya"], "rule": g["rule"],
             "word": g["word"], "counts": g["given_counts"], "haraka_s": a.get("haraka_s"),
             # the clip's last word: its final letter absorbs the stop, so its length is not a hold
             "ayah_final": g["word"] == rec["uthmani"].split()[-1]}
            for g in a.get("ghunnah", []) if g["grade"] == "naqisah"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    recs = [json.loads(l) for l in (DUMP / "index.jsonl").open()]
    recs = [r for r in recs if "file" in r and (DUMP / r["file"]).is_file()]
    n = 0
    with OUT.open("w") as f, ProcessPoolExecutor(args.workers) as ex:
        for rows in ex.map(_one, recs, chunksize=20):
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
            if n % 500 == 0:
                print(f"{n}/{len(recs)} clips", flush=True)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
