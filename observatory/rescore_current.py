#!/usr/bin/env python3
"""Score the latest take A and take B of every Round 1-5 exercise with the CURRENT engine -- read-only.

Nothing under research_agency_lab/ is written: no report or score file beside the recording, no line
in sessions_results.jsonl. The cards (and a compact per-letter / per-rule / per-characteristic digest
of the measurements) go to observatory/data/current_engine.json, stamped with the git revision, so a
reviewer can set the as-recorded history (sessions_results.jsonl) beside the current engine's reading
of the same audio.

    .venv/bin/python observatory/rescore_current.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
OUT = ROOT / "observatory/data/current_engine.json"
ROUNDS_IN_SCOPE = (1, 2, 3, 4, 5)


def digest(m: dict) -> dict:  # type: ignore[type-arg]
    """Per-rule, per-letter and per-characteristic counts from one take's measurements."""
    rules: dict[str, Counter] = defaultdict(Counter)  # type: ignore[type-arg]
    for r in m["rules"]:
        rules[r["rule"]][r["status"]] += 1
    letters: dict[str, Counter] = defaultdict(Counter)  # type: ignore[type-arg]
    heads: dict[str, Counter] = defaultdict(Counter)  # type: ignore[type-arg]
    for l in m["letters"]:
        i = l["identity"]
        if i["scored"] and i["margin"] is not None:
            letters[l["symbol"]]["confirmed" if i["confirmed"] else "not_confirmed"] += 1
        for h, c in l["characteristics"].items():
            if c["scored"]:
                heads[h]["realised" if c["realised"] else "not_realised"] += 1
    return {"rules": {k: dict(v) for k, v in rules.items()}, "letters": {k: dict(v) for k, v in letters.items()},
            "characteristics": {k: dict(v) for k, v in heads.items()}, "summary": m.get("summary"),
            "recording": {k: m["recording"].get(k) for k in ("audio_seconds", "seconds_per_count", "tempo_class")}}


def main() -> int:
    from app.engine import Engine
    from app.sessions import ROUNDS, parts, score, tempo_pair
    from app.webapp import SESSIONS_DIR, decode_upload
    from scripts.rescore_round import latest_take
    rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    eng = Engine()
    out: dict = {"git_revision": rev, "scored_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),  # type: ignore[type-arg]
                 "note": "latest take per exercise, current engine, dry run: nothing historical was written",
                 "takes": []}
    for n in ROUNDS_IN_SCOPE:
        for ex in ROUNDS[n]:
            reports: dict = {}  # type: ignore[type-arg]
            for take in ("A", "B"):
                src = latest_take(ex.id, take)
                if src is None:
                    out["takes"].append({"round": n, "exercise": ex.id, "take": take, "recording": None})
                    continue
                wave = decode_upload(src.read_bytes())
                rep = eng.analyze(wave, parts(ex), wajh=ex.wajh)
                rep["audio_seconds"] = round(float(wave.size) / 16000, 2)
                card = score(ex, take, rep["measurements"])
                reports[take] = rep
                if ex.b_is_correct and take == "B" and "A" in reports:
                    card["tempo_pair"] = tempo_pair(reports["A"].get("stretch") or {}, rep.get("stretch") or {})
                card.pop("letter_matrix", None)
                out["takes"].append({"round": n, "exercise": ex.id, "take": take,
                                     "recording": str(src.relative_to(SESSIONS_DIR)), "card": card,
                                     "digest": digest(rep["measurements"])})
                print(f"{ex.id} {take} {src.name}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
