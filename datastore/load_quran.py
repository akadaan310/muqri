#!/usr/bin/env python3
"""Load the whole Quran into the store: Uthmani text and the Hafs phonetic script of every ayah.

The phonetic script is muaalem's (``quran_transcript.quran_phonetizer``), so model output, the verse
identifier and the GOP all speak the same alphabet. ``collapsed`` folds every run of one symbol into
one (ااااا → ا), which makes retrieval independent of the reciter's madd lengths. Resumable: ayahs
already stored for the moshaf setting are skipped.

    .venv/bin/python -m datastore.load_quran [--procs 2]
"""

from __future__ import annotations

import argparse
import itertools
import multiprocessing as mp
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MOSHAF_ID = "hafs_m4"


def _moshaf():  # type: ignore[no-untyped-def]
    from quran_transcript import MoshafAttributes
    return MoshafAttributes(rewaya="hafs", madd_monfasel_len=4, madd_mottasel_len=4, madd_mottasel_waqf=4,
                            madd_aared_len=4)


def collapse(ph: str) -> str:
    return "".join(k for k, _ in itertools.groupby(ph))


def _one(ref: tuple[int, int]) -> tuple[int, int, str, str, str, list[list[int]]]:
    sys.path.insert(0, str(ROOT / "research_agency_lab/experiments/learner_eval"))
    from muaalem_dump import word_spans  # noqa: PLC0415
    from quran_transcript import Aya, quran_phonetizer
    s, a = ref
    u = Aya(s, a).get().uthmani
    r = quran_phonetizer(u, _moshaf(), remove_spaces=True)
    return s, a, u, r.phonemes, collapse(r.phonemes), word_spans(u, r.mappings)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=2)
    args = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    from datastore.store import connect, run

    con = connect()
    ayahs = [tuple(map(int, line.split("\t")[:2])) for line in
             (ROOT / "datasets/qaari_keys/build/ayahs.tsv").read_text().splitlines()]
    have = set(con.execute("SELECT surah, ayah FROM ayah_phonetic WHERE moshaf = ?", [MOSHAF_ID]).fetchall())
    todo = [r for r in ayahs if r not in have]
    print(f"{len(ayahs)} ayahs, {len(have)} stored, {len(todo)} to phonetise")
    t0 = time.time()
    with run(con, f"quran_phonetic:{MOSHAF_ID}", "loader", {"moshaf": MOSHAF_ID, "phonetizer": "quran-transcript 0.6.1"}):
        with mp.Pool(args.procs) as pool:
            batch = []
            for i, row in enumerate(pool.imap(_one, todo, chunksize=16), 1):
                batch.append(row)
                if len(batch) >= 200 or i == len(todo):
                    con.executemany("INSERT OR REPLACE INTO ayah_phonetic VALUES (?, ?, ?, ?, ?, ?)",
                                    [(MOSHAF_ID, s, a, ph, col, ws) for s, a, _, ph, col, ws in batch])
                    con.executemany("INSERT OR IGNORE INTO ayah (surah, ayah, uthmani) VALUES (?, ?, ?)",
                                    [(s, a, u) for s, a, u, *_ in batch])
                    batch = []
                    print(f"  {i}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
        meta = {tuple(map(int, line.split("\t")[:2])): tuple(map(int, line.split("\t")[2:4])) for line in
                (ROOT / "datasets/qaari_keys/build/ayahs.tsv").read_text().splitlines()}
        con.executemany("UPDATE ayah SET n_words = ?, n_letters = ? WHERE surah = ? AND ayah = ?",
                        [(w, n, s, a) for (s, a), (w, n) in meta.items()])
    n = con.execute("SELECT count(*), sum(length(phonemes)) FROM ayah_phonetic WHERE moshaf = ?", [MOSHAF_ID]).fetchone()
    print(f"ayah_phonetic[{MOSHAF_ID}]: {n[0]} ayahs, {n[1]} phoneme symbols ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
