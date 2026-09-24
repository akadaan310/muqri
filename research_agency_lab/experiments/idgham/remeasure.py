#!/usr/bin/env python3
"""Re-measure idghām bi-ghunnah on the masters after the binder fix (app/rule_bind.py).

Before the fix, idghām nāqiṣ into ي/و bound to the vowel AFTER the merged letter (99:7 فَمَن يَعْمَلْ timed
the fatḥah of يَ, not the held ييي), so reference_stats' idgham_ghunnah quantiles describe the wrong
sound. This runs the engine on juz' 30 for the anchor and studio tiers and writes every
idgham_ghunnah instance with its subtype, the symbol it bound to, and the held length in counts.

    .venv/bin/python research_agency_lab/experiments/idgham/remeasure.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import librosa  # noqa: E402

from app.engine import Engine  # noqa: E402
from app.tajweed_rules.parser import TajweedParser  # noqa: E402
from datastore.ingest import ANCHOR  # noqa: E402
from datastore.review_queue import audio_path  # noqa: E402

OUT = Path(__file__).with_name("remeasure.jsonl")
SPEAKERS = sorted(ANCHOR) + ["Minshawy_Murattal_128kbps", "Abdul_Basit_Murattal_192kbps", "Alafasy_128kbps"]


def verses() -> list[tuple[int, int]]:
    from quran_transcript import Aya
    p, out = TajweedParser(), []
    for s in range(78, 115):
        y = 1
        while True:
            try:
                u = Aya(s, y).get().uthmani
            except Exception:  # noqa: BLE001 - past the surah's last ayah
                break
            if any(r.rule_type.value == "idgham_ghunnah" for r in p.parse(u).rules):
                out.append((s, y))
            y += 1
    return out


def main() -> None:
    eng, done = Engine(), set()
    if OUT.is_file():
        done = {(r["speaker"], r["surah"], r["ayah"]) for r in map(json.loads, OUT.open())}
    vs = verses()
    print(len(vs), "ayahs with idgham_ghunnah in juz' 30", flush=True)
    with OUT.open("a") as f:
        for sp in SPEAKERS:
            for s, y in vs:
                if (sp, s, y) in done:
                    continue
                src = audio_path(sp, s, y)
                if not src.is_file() or src.stat().st_size == 0:
                    continue
                try:
                    wave, _ = librosa.load(str(src), sr=16000, mono=True)
                    rep = eng.analyze(wave.astype("float32"), [(s, y)])
                except Exception as exc:  # noqa: BLE001 - one bad file must not stop the pass
                    f.write(json.dumps({"speaker": sp, "surah": s, "ayah": y, "error": str(exc)[:200]}) + "\n")
                    continue
                rows = []
                for a in rep["ayahs"]:
                    for r in a["rules"]:
                        if r["rule"] == "idgham_ghunnah":
                            ev = r.get("evidence") or {}
                            rows.append({"word": r["word"], "detail": r["detail"], "status": r["status"],
                                         "units": r.get("units"), "counts": ev.get("given_counts"),
                                         "sym": "".join(a["letters"][i]["symbol"] for i in (r.get("units") or [])
                                                        if i < len(a["letters"]))})
                f.write(json.dumps({"speaker": sp, "surah": s, "ayah": y, "rules": rows}, ensure_ascii=False) + "\n")
                f.flush()
            print(sp, "done", flush=True)


if __name__ == "__main__":
    main()
