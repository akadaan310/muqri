#!/usr/bin/env python3
"""TTS as raw material: fully vowelled verses spoken by neural Arabic voices, measured by the engine as
if they were recitations -- which letters, vowels, characteristics and rule lengths come out right before
any synthesis, and what the rules need done to them.

Voices: Microsoft's free neural voices through edge-tts (32 Arabic voices, 16 dialects).

    .venv/bin/python research_agency_lab/experiments/synthesis/tts_probe.py --verses 112:1-4 \\
        --voices ar-SA-HamedNeural ar-SA-ZariyahNeural ar-EG-ShakirNeural ar-AE-HamdanNeural
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import warnings
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
DATA = Path(__file__).with_name("data") / "tts"


async def speak(text: str, voice: str, out: Path, rate: str) -> None:
    import edge_tts
    await edge_tts.Communicate(text, voice, rate=rate).save(str(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verses", default="112:1-4")
    ap.add_argument("--voices", nargs="+", default=["ar-SA-HamedNeural", "ar-SA-ZariyahNeural"])
    ap.add_argument("--rate", default="-25%", help="slower than conversational, toward tartil")
    args = ap.parse_args()
    from quran_transcript import Aya
    from app.engine import Engine
    from app.webapp import decode_upload
    s, rng = args.verses.split(":")
    lo, _, hi = rng.partition("-")
    ayahs = [(int(s), a) for a in range(int(lo), int(hi or lo) + 1)]
    eng = Engine()
    DATA.mkdir(parents=True, exist_ok=True)
    summary = []
    for v in args.voices:
        for (su, a) in ayahs:
            text = Aya(su, a).get().uthmani
            p = DATA / f"{v}_{su:03d}{a:03d}.mp3"
            if not p.is_file():
                asyncio.run(speak(text, v, p, args.rate))
            m = eng.analyze(decode_upload(p.read_bytes()), [(su, a)], makhraj=True)["measurements"]
            L = [l for l in m["letters"] if l["identity"]["scored"]]
            cons = [l for l in L if l["kind"] == "consonant"]
            vow = [l for l in L if l["kind"] == "harakah"]
            heads = Counter(h for l in cons for h, c in l["characteristics"].items() if c["scored"] and not c["realised"])
            rules = [(r["rule"], r["status"], r["observed_counts"], r["expected_counts"]) for r in m["rules"]
                     if r["observed_counts"] is not None or r["status"] in ("wrong", "short", "long")]
            row = {"voice": v, "ref": f"{su}:{a}", "seconds": m["recording"]["audio_seconds"],
                   "count_unit_s": m["recording"]["seconds_per_count"],
                   "consonants_confirmed": f"{sum(l['identity']['confirmed'] for l in cons)}/{len(cons)}",
                   "vowels_confirmed": f"{sum(l['identity']['confirmed'] for l in vow)}/{len(vow)}",
                   "heads_not_realised": dict(heads), "rules": rules,
                   "words_all_correct": f"{m['summary']['words_all_correct']}/{m['summary']['words']}"}
            summary.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    (DATA / "probe.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
