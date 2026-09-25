#!/usr/bin/env python3
"""Everything the Observatory shows that can be read from the repository and runtime, in one JSON file.

Read-only over the project: it reads the session definitions (app.sessions), the as-recorded session
log (research_agency_lab/experiments/sessions_results.jsonl), the recordings' own report files, the
git history, the benchmark rows, the T300 index and the letter-corpus index; it writes only
observatory/data/snapshot.json. The Neo4j snapshot (neo4j_snapshot.json), the current-engine rescore
(current_engine.json) and the Modal bill (modal_billing.json) are separate files beside it.

    .venv/bin/python observatory/build_snapshot.py
"""

from __future__ import annotations

import gzip
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT / "observatory/data"
ROUNDS_IN_SCOPE = (1, 2, 3, 4, 5)
AUDIO_EXT = {".wav", ".mp3", ".m4a", ".ogg", ".webm", ".flac", ".aac", ".bin", ".mp4"}


def git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def exercises() -> list[dict[str, Any]]:
    from app.sessions import ROUNDS, to_json
    out = []
    for n in ROUNDS_IN_SCOPE:
        for ex in ROUNDS[n]:
            d = to_json(ex)
            d["round"] = n
            d["signatures"] = [{"ayah": m.ayah, "word": m.word,
                                "catch": [{"kind": s.kind, "name": s.name, "status": list(s.status), "sign": s.sign,
                                           "letter": s.letter, "heard": list(s.heard), "word_offset": s.word_offset}
                                          for s in m.catch]} for m in ex.mistakes]
            d["controls"] = [list(c) for c in ex.controls]
            out.append(d)
    return out


def history() -> list[dict[str, Any]]:
    """Every scoring event in the session log, rounds 1-5: the first per recording is 'as recorded'."""
    rows = [json.loads(l) for l in (ROOT / "research_agency_lab/experiments/sessions_results.jsonl").open()]
    out = []
    for r in rows:
        n = int(r["exercise"][1])
        if n not in ROUNDS_IN_SCOPE:
            continue
        note = r.get("note") or ""
        src = note.split()[2] if note.startswith("rescore of ") else r["stamp"]
        ev = {"stamp": r["stamp"], "round": n, "exercise": r["exercise"], "take": r["take"], "recording_stamp": src,
              "kind": "rescore" if note.startswith("rescore") else "as recorded", "note": note,
              "tempo": r.get("tempo"), "false_alarms": [{"ayah": f["ayah"], "word": f["word"], "text": f["text"],
                                                         "failing": f["failing"]} for f in r.get("false_alarms", [])]}
        if "expectations" in r:
            ev["expectations"] = [{k: e.get(k) for k in ("ayah", "word", "text", "rule", "expected_counts", "measured",
                                                         "status", "z_masters", "verdict")} for e in r["expectations"]]
            ev["met"] = r.get("met")
        if "mistakes" in r:
            ev["mistakes"] = [{k: m.get(k) for k in ("ayah", "word", "text", "do", "verdict", "evidence", "engine_failing")}
                              for m in r["mistakes"]]
            ev["caught"] = r.get("caught")
        out.append(ev)
    return out


def recordings() -> list[dict[str, Any]]:
    from app.webapp import SESSIONS_DIR
    out = []
    for n in ROUNDS_IN_SCOPE:
        for d in sorted(SESSIONS_DIR.glob(f"r{n}e*/*")):
            for p in sorted(x for x in d.iterdir() if x.suffix.lower() in AUDIO_EXT):
                rep = p.with_suffix(".report.json")
                secs = None
                if rep.is_file():
                    try:
                        secs = json.loads(rep.read_text()).get("audio_seconds")
                    except json.JSONDecodeError:
                        pass
                out.append({"round": n, "exercise": d.parent.name, "take": d.name, "stamp": p.stem,
                            "format": p.suffix.lstrip("."), "bytes": p.stat().st_size, "audio_seconds": secs,
                            "url": f"/observatory/audio/{d.parent.name}/{d.name}/{p.name}"})
    return out


def engine_commits() -> list[dict[str, str]]:
    """Commits that touched the engine or the session scoring, 2026-09-24 onward (the rounds' span)."""
    log = git("log", "--since=2026-09-24T00:00:00", "--format=%H|%cI|%s", "--",
              "app/engine.py", "app/analysis.py", "app/submission.py", "app/measurements.py", "app/sessions.py",
              "app/waqf.py", "app/stretch.py", "app/rule_bind.py", "app/ghunnah.py", "app/mudud.py",
              "app/tajweed_rules", "app/blindspots.py")
    return [dict(zip(("hash", "time", "subject"), l.split("|", 2))) for l in log.splitlines() if l]


def corpus() -> dict[str, Any]:
    def rows(pat: str):  # type: ignore[no-untyped-def]
        for f in sorted(ROOT.glob(pat)):
            with gzip.open(f, "rt") as fh:
                for line in fh:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
    kag: dict[tuple[str, str], set] = defaultdict(set)  # type: ignore[type-arg]
    for pat in ("benchmarks/results/kaggle/full/*.jsonl.gz", "benchmarks/results/kaggle/fill/*.jsonl.gz"):
        for r in rows(pat):
            kag[(r.get("reciter"), r.get("mode"))].add((r.get("surah"), r.get("ayah")))
    studio: dict[str, set] = defaultdict(set)  # type: ignore[type-arg]
    for r in rows("benchmarks/results/studio-all/*.jsonl.gz"):
        studio[r["reciter"]].add((r["surah"], r["ayah"]))
    t300 = Counter(json.loads(l)["speaker"] for l in (ROOT / "research_agency_lab/experiments/qaari_keys/modal_T300/index.jsonl").open())
    from datastore.ingest import ANCHOR, IMAMS, STUDIO, ladder
    lc = ROOT / "research_agency_lab/experiments/letter_corpus/data/corpus.json"
    letter = None
    if lc.is_file():
        c = json.loads(lc.read_text())
        per = Counter(rec for cell in c["cells"] for rec, xs in cell["reciters"].items() for _ in xs)
        letter = {"cells": len(c["cells"]), "letter_cells": sum(x["kind"] == "letter" for x in c["cells"]),
                  "rule_cells": sum(x["kind"] == "rule" for x in c["cells"]), "instances_per_source": dict(per)}
    return {
        "quran_md_kaggle": {f"{r}|{m}": len(v) for (r, m), v in sorted(kag.items())},
        "studio_all_everyayah": {r: len(v) for r, v in sorted(studio.items())},
        "t300_posteriors": {"speakers": len(t300), "clips_per_speaker": sorted(set(t300.values())),
                            "tiers": {s: ladder(s) for s in sorted(t300)}},
        "tier_sets": {"anchor": sorted(ANCHOR), "studio": sorted(STUDIO), "imams": sorted(IMAMS)},
        "letter_corpus": letter,
        "full_quran_reciters": sorted({r for (r, _m), v in kag.items() if len(v) >= 6236}
                                      | {r for r, v in studio.items() if len(v) >= 6233}),
    }


def main() -> int:
    tests = subprocess.run([str(ROOT / ".venv/bin/python"), "-m", "pytest", "--collect-only", "-q", "-p", "no:warnings"],
                           cwd=ROOT, capture_output=True, text=True).stdout
    # -q prints "path: N" per test file (pyproject addopts), or "path::name" per test
    n_tests = sum(int(l.rsplit(": ", 1)[1]) if l.rsplit(": ", 1)[-1].isdigit() else 1
                  for l in tests.splitlines() if l.startswith("tests/"))
    snap = {
        "git": {"revision": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                "last_commit": git("log", "-1", "--format=%cI %s")},
        "tests_collected": n_tests,
        "exercises": exercises(),
        "history": history(),
        "recordings": recordings(),
        "engine_commits": engine_commits(),
        "corpus": corpus(),
    }
    (DATA / "snapshot.json").write_text(json.dumps(snap, ensure_ascii=False, indent=1))
    print(f"snapshot: {len(snap['exercises'])} exercises, {len(snap['history'])} scoring events, "
          f"{len(snap['recordings'])} recordings, {len(snap['engine_commits'])} engine commits, {n_tests} tests")
    return 0


if __name__ == "__main__":
    sys.exit(main())
