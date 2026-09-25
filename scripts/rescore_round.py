#!/usr/bin/env python3
"""Re-score a round's recorded takes through the current engine.

    python scripts/rescore_round.py 4 --note "after the idgham binder fix"
    python scripts/rescore_round.py 4 --dry-run          # print the cards, write nothing

Each exercise's latest recording per take (research_agency_lab/experiments/session_recordings/
<exercise>/<take>/<stamp>.<ext>) goes through the same path as POST /sessions/submit:
Engine.analyze on the exercise's parts with its declared wajh, then sessions.score. The card is
written beside the recording and appended to sessions_results.jsonl with note "rescore of <stamp>
after ...", so the log keeps every scoring of every take. Run it on the machine that holds the
recordings; they never enter git.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.sessions import ROUNDS, parts, score, tempo_pair  # noqa: E402
from app.webapp import SESSIONS_DIR, SESSIONS_LOG, decode_upload  # noqa: E402

AUDIO_EXT = {".wav", ".mp3", ".m4a", ".ogg", ".webm", ".flac", ".aac", ".bin", ".mp4"}


def latest_take(ex_id: str, take: str) -> Path | None:
    d = SESSIONS_DIR / ex_id / take
    if not d.is_dir():
        return None
    audio = [p for p in d.iterdir() if p.suffix.lower() in AUDIO_EXT]
    return max(audio, key=lambda p: p.name) if audio else None


def summary(card: dict) -> str:  # type: ignore[type-arg]
    parts_ = []
    if card.get("expectations"):
        met = sum(e["verdict"] == "ok" for e in card["expectations"])
        parts_.append(f"expectations {met}/{len(card['expectations'])}")
    if card.get("mistakes"):
        caught = sum(m["verdict"] == "caught" for m in card["mistakes"])
        parts_.append(f"caught {caught}/{len(card['mistakes'])}")
    parts_.append(f"false alarms {len(card.get('false_alarms', []))}")
    return ", ".join(parts_)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("round", type=int)
    ap.add_argument("--note", default="the current engine", help="what changed since the take was scored")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    from app.engine import Engine
    eng = Engine()
    for ex in ROUNDS[args.round]:
        reports: dict[str, dict] = {}  # type: ignore[type-arg]
        for take in ("A", "B"):
            src = latest_take(ex.id, take)
            if src is None:
                print(f"{ex.id} {take}: no recording under {SESSIONS_DIR / ex.id / take}")
                continue
            t0 = time.time()
            wave = decode_upload(src.read_bytes())
            report = eng.analyze(wave, parts(ex), wajh=ex.wajh)
            report["audio_seconds"] = round(float(wave.size) / 16000, 2)
            card = score(ex, take, report["measurements"])
            reports[take] = report
            if ex.b_is_correct and take == "B" and "A" in reports:
                card["tempo_pair"] = tempo_pair(reports["A"].get("stretch") or {}, report.get("stretch") or {})
            print(f"{ex.id} {take} ({src.name}, {time.time() - t0:.0f} s): {summary(card)}")
            for e in card.get("expectations", []):
                if e["verdict"] != "ok":
                    print(f"    expectation {e['verdict']}: {e['rule']} {e['text']} measured {e['measured']}")
            for m in card.get("mistakes", []):
                if m["verdict"] != "caught":
                    print(f"    mistake {m['verdict']}: {m['do']}")
            for fa in card.get("false_alarms", []):
                print(f"    false alarm: {fa['text']} {fa['failing'][:4]}")
            if args.dry_run:
                continue
            stamp = time.strftime("%Y%m%d-%H%M%S")
            (src.parent / f"{stamp}.report.json").write_text(json.dumps(report, ensure_ascii=False))
            (src.parent / f"{stamp}.score.json").write_text(json.dumps(card, ensure_ascii=False))
            with SESSIONS_LOG.open("a") as fh:
                fh.write(json.dumps({"stamp": stamp, "note": f"rescore of {src.stem} after {args.note}", **card},
                                    ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
