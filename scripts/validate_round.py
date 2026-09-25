#!/usr/bin/env python3
"""Validate a round's exercises on master recitations before they go live.

    python scripts/validate_round.py 5                       # Husary and Minshawy
    python scripts/validate_round.py 5 --speakers Husary_128kbps

Each exercise's ayahs are fetched from EveryAyah (datastore.review_queue.audio_path), joined with
0.8 s of silence (a stop between ayahs), and scored as take A through the same path as
POST /sessions/submit. A master's correct reading must meet every expectation and raise no false
alarm; anything else is either an engine fault or an expectation to drop, with the reason in the
commit. Results go to research_agency_lab/experiments/sessions_baseline_round<n>.json.

Exercises that cut an ayah at declared stops (`Exercise.stops`) are skipped: EveryAyah files hold
whole ayahs recited without those stops.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.sessions import ROUNDS, parts, score  # noqa: E402
from app.webapp import decode_upload  # noqa: E402
from datastore.review_queue import audio_path  # noqa: E402

GAP_S = 0.8


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("round", type=int)
    ap.add_argument("--speakers", nargs="+", default=["Husary_128kbps", "Minshawy_Murattal_128kbps"])
    args = ap.parse_args(argv)
    from app.engine import Engine
    eng = Engine()
    out: dict[str, dict] = {}  # type: ignore[type-arg]
    clean = True
    for ex in ROUNDS[args.round]:
        if ex.stops:
            print(f"{ex.id}: skipped (declared stops inside an ayah)")
            continue
        for spk in args.speakers:
            waves = [decode_upload(audio_path(spk, ex.surah, a).read_bytes())
                     for a in range(ex.ayahs[0], ex.ayahs[1] + 1)]
            gap = np.zeros(int(GAP_S * 16000), dtype="float32")
            wave = np.concatenate([x for w in waves for x in (w, gap)][:-1])
            report = eng.analyze(wave, parts(ex), wajh=ex.wajh)
            card = score(ex, "A", report["measurements"])
            out.setdefault(ex.id, {})[spk] = card
            unmet = [e for e in card["expectations"] if e["verdict"] != "ok"]
            fas = card.get("false_alarms", [])
            clean &= not unmet and not fas
            met = len(card["expectations"]) - len(unmet)
            print(f"{ex.id} {spk}: expectations {met}/{len(card['expectations'])}, false alarms {len(fas)}")
            for e in unmet:
                print(f"    {e['verdict']}: {e['rule']} {e['text']} measured {e['measured']} "
                      f"band {e['expected_counts']}")
            for fa in fas:
                print(f"    false alarm: {fa['text']} {fa['failing'][:4]}")
    dest = ROOT / "research_agency_lab" / "experiments" / f"sessions_baseline_round{args.round}.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"wrote {dest}")
    return 0 if clean else 1


if __name__ == "__main__":
    sys.exit(main())
