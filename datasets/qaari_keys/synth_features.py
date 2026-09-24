"""Inputs for the synthesis pilot selection (synth_select.jl): Husary's seconds per ayah and the
QPS diphones of every ayah, beside the QaariKeys coverage keys (features.py).

A synthesizer must hear every transition between sounds, not only every rule, so diphones -- pairs of
consecutive QPS symbols after collapsing runs (a madd of 2 or 6 counts is one symbol here; its length
is covered by the rule keys) -- become a seventh key family.

Seconds: EveryAyah's Husary_128kbps files are constant 128 kbit/s, so seconds = bytes * 8 / 128000
minus 0.264 s of MP3 framing (measured: identical on all 296 T300 clips with known duration).

    .venv/bin/python datasets/qaari_keys/synth_features.py
"""

from __future__ import annotations

import itertools
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "datasets/qaari_keys/build"
OUT = BUILD / "synth"
FOLDER = "Husary_128kbps"
FRAMING_S = 0.264


def seconds(s: int, a: int) -> float:
    url = f"https://everyayah.com/data/{FOLDER}/{s:03d}{a:03d}.mp3"
    for _ in range(3):
        try:
            r = urllib.request.urlopen(urllib.request.Request(url, method="HEAD", headers={"User-Agent": "qaari"}),
                                       timeout=30)  # noqa: S310
            return round(int(r.headers["Content-Length"]) * 8 / 128000 - FRAMING_S, 3)
        except Exception:  # noqa: BLE001
            continue
    raise RuntimeError(f"no size for {url}")


def main() -> None:
    sys.path.insert(0, str(ROOT))
    from app.engine import Engine
    OUT.mkdir(parents=True, exist_ok=True)
    ay = [tuple(map(int, l.split("\t")[:2])) for l in (BUILD / "ayahs.tsv").open()]
    with ThreadPoolExecutor(24) as p:
        secs = list(p.map(lambda sa: seconds(*sa), ay))
    eng = Engine(layout={"columns": 0, "blank": 0, "levels": {}})
    di: list[dict[str, int]] = []
    for s, a in ay:
        ph = "".join(k for k, _ in itertools.groupby(eng.reference(s, a).phonemes))
        c: dict[str, int] = {}
        for x, y in zip(ph, ph[1:]):
            c[x + y] = c.get(x + y, 0) + 1
        di.append(c)
    keys = sorted({k for c in di for k in c})
    (OUT / "seconds.tsv").write_text("".join(f"{s}\t{a}\t{x}\n" for (s, a), x in zip(ay, secs)))
    (OUT / "diphone_keys.txt").write_text("".join(f"diphone:{k}\n" for k in keys))
    kid = {k: i + 1 for i, k in enumerate(keys)}
    with (OUT / "ayah_diphones.tsv").open("w") as fh:
        for i, c in enumerate(di, 1):
            for k, v in sorted(c.items()):
                fh.write(f"{i}\t{kid[k]}\t{v}\n")
    print(f"{len(ay)} ayahs, {sum(secs) / 3600:.2f} h, {len(keys)} diphones -> {OUT}")


if __name__ == "__main__":
    main()
