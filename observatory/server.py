#!/usr/bin/env python3
"""The Muqri Observatory: a read-only inspection layer in front of the running engine.

It does not import or modify the engine. It serves:

  /sessions, /sessions/rounds, /sessions/round/{1-5}
        the existing Sessions UI and its JSON, proxied from the running app (127.0.0.1:8088), limited to
        rounds 1-5, recording / upload controls hidden (every write is refused here anyway)
  /observatory/audio/{exercise}/{take}/{file}
        the stored round 1-5 recordings, read-only, confined to the recordings directory
  /muqri-observatory                  the technical page
  /muqri-observatory/report           the human-readable report (HTML); /muqri-observatory/report.md
  /api/observatory/manifest           the machine-readable manifest

Only GET and HEAD are accepted. Access needs the token in ~/.config/qaari/observatory.token, given once
as ?token=... (it is then kept in an HttpOnly cookie). Nothing here reads a credential, the database or
the shell; the Neo4j figures are a snapshot file taken beforehand (observatory/data/neo4j_snapshot.json).

    .venv/bin/python -m observatory.server            # 127.0.0.1:8095
"""

from __future__ import annotations

import html
import json
import os
import secrets
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "observatory/data"
BACKEND = os.environ.get("OBSERVATORY_BACKEND", "http://127.0.0.1:8088")
PORT = int(os.environ.get("OBSERVATORY_PORT", "8095"))
TOKEN_FILE = Path(os.path.expanduser("~/.config/qaari/observatory.token"))
RECORDINGS = ROOT / "research_agency_lab/experiments/session_recordings"
ROUNDS = (1, 2, 3, 4, 5)
AUDIO_TYPES = {".m4a": "audio/mp4", ".mp4": "audio/mp4", ".webm": "audio/webm", ".ogg": "audio/ogg",
               ".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac", ".aac": "audio/aac"}


def token() -> str:
    if not TOKEN_FILE.is_file():
        TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        TOKEN_FILE.write_text(secrets.token_urlsafe(24))
        TOKEN_FILE.chmod(0o600)
    return TOKEN_FILE.read_text().strip()


def load(name: str) -> Any:
    p = DATA / name
    return json.loads(p.read_text()) if p.is_file() else None


E = html.escape


# ---------------------------------------------------------------- session analysis --------------------------

def _latest_as_recorded(history: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    """Per (exercise, take): the as-recorded scoring of the latest recording."""
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for e in history:
        if e["kind"] == "as recorded":
            k = (e["exercise"], e["take"])
            if k not in out or e["stamp"] > out[k]["stamp"]:
                out[k] = e
    return out


def _card_counts(card: dict[str, Any], take: str) -> dict[str, int]:
    c: dict[str, int] = Counter()  # type: ignore[assignment]
    for e in card.get("expectations") or []:
        c["expectations"] += 1
        c["met"] += e.get("verdict") == "ok"
    for m in card.get("mistakes") or []:
        c["mistakes"] += 1
        c[{"caught": "caught", "missed": "missed"}.get(m.get("verdict"), "other_reason")] += 1
    c["false_alarm_words"] += len(card.get("false_alarms") or [])
    c[f"fa_{take}"] += len(card.get("false_alarms") or [])
    return c


def round_metrics() -> dict[str, Any]:
    snap = load("snapshot.json") or {}
    cur = load("current_engine.json") or {"takes": []}
    asrec = _latest_as_recorded(snap.get("history", []))
    rounds = {}
    for n in ROUNDS:
        a: Counter = Counter()
        c: Counter = Counter()
        for (ex, take), e in asrec.items():
            if int(ex[1]) == n:
                a.update(_card_counts(e, take))
        for t in cur["takes"]:
            if t["round"] == n and t.get("card"):
                c.update(_card_counts(t["card"], t["take"]))
        rounds[n] = {"as_recorded": dict(a), "current_engine": dict(c)}
    return rounds


def rescore_trails() -> dict[str, list[dict[str, Any]]]:
    """Every scoring of every recording, in order: how the same audio's verdicts moved as the code changed."""
    snap = load("snapshot.json") or {}
    trail: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in sorted(snap.get("history", []), key=lambda x: x["stamp"]):
        c = _card_counts(e, e["take"])
        trail[f"{e['exercise']} {e['take']} {e['recording_stamp']}"].append(
            {"stamp": e["stamp"], "kind": e["kind"], "note": e["note"], "met": c.get("met"), "expectations": c.get("expectations"),
             "caught": c.get("caught"), "mistakes": c.get("mistakes"), "false_alarms": c.get("false_alarm_words")})
    return trail


def perturbation_matrix() -> list[dict[str, Any]]:
    """Part 8: every scripted mistake of every take B, as recorded and on the current engine."""
    snap = load("snapshot.json") or {}
    cur = {(t["exercise"], t["take"]): t for t in (load("current_engine.json") or {"takes": []})["takes"]}
    exs = {x["id"]: x for x in snap.get("exercises", [])}
    asrec = _latest_as_recorded(snap.get("history", []))
    recs = {(r["exercise"], r["take"], r["stamp"]): r for r in snap.get("recordings", [])}
    rows = []
    for (ex_id, take), e in sorted(asrec.items()):
        ex = exs.get(ex_id)
        if not ex or take != "B" or ex.get("b_is_correct") or "mistakes" not in e:
            continue
        now = cur.get((ex_id, "B"), {}).get("card") or {}
        now_m = {(m["ayah"], m["word"]): m for m in now.get("mistakes", [])}
        rec = recs.get((ex_id, "B", e["stamp"]))
        for i, m in enumerate(e["mistakes"]):
            sig = ex["signatures"][i]["catch"] if i < len(ex["signatures"]) else []
            nm = now_m.get((m["ayah"], m["word"]), {})
            rows.append({"round": int(ex_id[1]), "exercise": ex_id, "surah": ex["surah"], "ayah": m["ayah"],
                         "word": m["word"], "text": m.get("text"), "perturbation": m["do"],
                         "target": "; ".join(f"{s['kind']}:{s['name'] or ''}{('/' + s['letter']) if s['letter'] else ''}"
                                             for s in sig),
                         "as_recorded": m["verdict"], "as_recorded_evidence": "; ".join(m.get("evidence") or m.get("engine_failing") or []),
                         "current": nm.get("verdict"), "current_evidence": "; ".join(nm.get("evidence") or nm.get("engine_failing") or []),
                         "false_alarms_elsewhere_as_recorded": len(e.get("false_alarms") or []),
                         "false_alarms_elsewhere_current": len(now.get("false_alarms") or []) if now else None,
                         "audio": rec["url"] if rec else None, "take_a_audio": _audio_for(snap, ex_id, "A")})
    return rows


def _audio_for(snap: dict[str, Any], ex_id: str, take: str) -> str | None:
    rs = [r for r in snap.get("recordings", []) if r["exercise"] == ex_id and r["take"] == take]
    return max(rs, key=lambda r: r["stamp"])["url"] if rs else None


def digest_by_round() -> dict[int, dict[str, Any]]:
    """Per round, take A on the current engine: rule statuses, letters confirmed, characteristics realised."""
    cur = load("current_engine.json") or {"takes": []}
    out: dict[int, dict[str, Any]] = {}
    for n in ROUNDS:
        rules: dict[str, Counter] = defaultdict(Counter)
        letters: dict[str, Counter] = defaultdict(Counter)
        heads: dict[str, Counter] = defaultdict(Counter)
        for t in cur["takes"]:
            if t["round"] != n or t["take"] != "A" or not t.get("digest"):
                continue
            for k, v in t["digest"]["rules"].items():
                rules[k].update(v)
            for k, v in t["digest"]["letters"].items():
                letters[k].update(v)
            for k, v in t["digest"]["characteristics"].items():
                heads[k].update(v)
        out[n] = {"rules": {k: dict(v) for k, v in sorted(rules.items())},
                  "letters": {k: dict(v) for k, v in sorted(letters.items())},
                  "characteristics": {k: dict(v) for k, v in sorted(heads.items())}}
    return out


# ---------------------------------------------------------------- manifest ---------------------------------

def _running_revision() -> str:
    import subprocess
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def manifest() -> dict[str, Any]:
    from observatory import content as C
    snap = load("snapshot.json") or {}
    neo = load("neo4j_snapshot.json") or {}
    bill = load("modal_billing.json") or {}
    cur = load("current_engine.json") or {}
    return {
        "system": {"name": "Muqri", "engines": C.ENGINES, "status_vocabulary": ["IMPLEMENTED", "TESTED",
                   "PARTIALLY IMPLEMENTED", "EXPERIMENTAL", "PLANNED", "CONCEPTUAL", "NOT VERIFIED"]},
        "git_revision": {"snapshot_built_at": snap.get("git", {}), "running": _running_revision()},
        "tests_collected": snap.get("tests_collected"),
        "deployment": {"backend": "FastAPI app/webapp.py, uvicorn, VM port 8088 (plain HTTP)",
                       "observatory": "observatory/server.py, 127.0.0.1:8095, read-only, token",
                       "public_https": "Cloudflare quick tunnel (trycloudflare.com) to the observatory only"},
        "components": C.PIPELINE, "measures": C.MEASURES, "makharij": C.makharij(),
        "sifat": [dict(zip(("name", "arabic", "letters", "acoustic", "detector", "score", "status", "evidence"), s))
                  for s in C.SIFAT],
        "models": C.MODELS, "reasoning_layers": C.REASONING,
        "corpus": snap.get("corpus"),
        "sessions": {"exercises": [{k: x[k] for k in ("id", "round", "title", "surah", "ayahs", "wajh")}
                                   for x in snap.get("exercises", [])],
                     "recordings": len(snap.get("recordings", [])), "scoring_events": len(snap.get("history", [])),
                     "current_engine_rescore": {"git_revision": cur.get("git_revision"), "scored_at": cur.get("scored_at")}},
        "metrics": {"by_round": round_metrics(), "perturbations": perturbation_matrix()},
        "engine_commits_during_rounds": snap.get("engine_commits"),
        "knowledge_graph": {k: neo.get(k) for k in ("version", "total_nodes", "total_relationships", "labels",
                                                    "relationships", "patterns", "indexes", "constraints")},
        "compute": {"julia_octave": C.JULIA_OCTAVE, "modal_apps": C.MODAL, "modal_billing": bill},
        "synthesis": C.SYNTHESIS, "qiraat": C.QIRAAT, "data_factory": C.FACTORY, "limitations": C.LIMITATIONS,
    }


# ---------------------------------------------------------------- public machine-review JSON ---------------
PUBLIC_PATH = "/api/observatory/public"
_LEAK_PATTERNS = {
    "ip address": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
    "absolute path": r"(?<![\w.])/(home|root|etc|opt|var|tmp|usr|mnt|srv)/",
    "home dir": r"~/",
    "credential word": r"(?i)password|passwd|secret|api[_-]?key|bearer |private[_ ]key|NEO4J_|MODAL_TOKEN|ANTHROPIC|OPENAI",
    "token query": r"(?i)token=",
    "localhost": r"(?i)localhost",
}


def sanitize_check(text: str, tok: str) -> list[str]:
    import re
    hits = [name for name, pat in _LEAK_PATTERNS.items() if re.search(pat, text)]
    if tok and tok in text:
        hits.append("observatory token")
    return hits


def _engine_status() -> dict[str, Any]:
    import urllib.request
    try:
        with urllib.request.urlopen(BACKEND + "/health", timeout=5) as r:
            h = json.loads(r.read())
        return {"reachable": True, "model_warm": h.get("model_warm"), "status": h.get("status")}
    except Exception:  # noqa: BLE001 - the report must serve even if the engine is down
        return {"reachable": False}


def public_review() -> dict[str, Any]:
    """The Observatory's technical content for automated review: counts, statuses, results -- no locations,
    addresses, credentials or tokens (checked by sanitize_check before serving)."""
    from observatory import content as C
    snap = load("snapshot.json") or {}
    neo = load("neo4j_snapshot.json") or {}
    bill = load("modal_billing.json") or {}
    cur = load("current_engine.json") or {}
    matrix = [{k: v for k, v in r.items() if k not in ("audio", "take_a_audio")} for r in perturbation_matrix()]
    takes = [{"round": t["round"], "exercise": t["exercise"], "take": t["take"],
              "recorded": t.get("recording") is not None,
              "card": {k: t["card"].get(k) for k in ("expectations", "met", "mistakes", "caught", "false_alarms", "tempo")}
              if t.get("card") else None} for t in cur.get("takes", [])]
    as_rec = [{k: e.get(k) for k in ("round", "exercise", "take", "stamp", "kind", "note", "met", "caught",
                                     "expectations", "mistakes", "false_alarms", "tempo")}
              for e in snap.get("history", [])]
    return {
        "about": "Muqri Observatory, machine-review view: a sanitized, read-only JSON of the technical Observatory. "
                 "Every statement describes the current implementation; status words: IMPLEMENTED (in the served "
                 "Sessions path), TESTED (tests or recorded measurements), PARTIALLY IMPLEMENTED, EXPERIMENTAL "
                 "(research code, measured), PLANNED, CONCEPTUAL, NOT VERIFIED.",
        "revisions": {"observatory_running": _running_revision(), "engine_snapshot": snap.get("git", {}).get("revision"),
                      "current_engine_rescore": cur.get("git_revision"), "rescored_at": cur.get("scored_at"),
                      "repository": "https://github.com/akadaan310/muqri (branch claude/qaari-eval-engine-btwkjt)"},
        "engine_status": _engine_status(),
        "human_pages": {"observatory": "/muqri-observatory", "report": "/muqri-observatory/report",
                        "sessions": "/sessions (rounds 1-5)", "access": "token link from the project owner"},
        "engines": C.ENGINES, "components": C.PIPELINE, "reasoning_layers": C.REASONING,
        "measures": C.MEASURES, "makharij": C.makharij(),
        "sifat": [dict(zip(("name", "arabic", "letters", "acoustic", "detector", "score", "status", "evidence"), s))
                  for s in C.SIFAT],
        "sessions": {
            "scope": "rounds 1-5 (round 6 is a letter-drill experiment, excluded)",
            "exercises": [{k: x.get(k) for k in ("id", "round", "title", "surah", "ayahs", "wajh", "goal", "spec",
                                                  "expect", "mistakes", "b_is_correct", "controls")}
                          for x in snap.get("exercises", [])],
            "recordings": [{k: r[k] for k in ("round", "exercise", "take", "stamp", "format", "audio_seconds")}
                           for r in snap.get("recordings", [])],
            "metrics_by_round": round_metrics(),
            "as_recorded_history": as_rec,
            "current_engine_takes": takes,
            "scripted_mistake_matrix": matrix,
            "per_rule_letter_characteristic_take_a_current": digest_by_round(),
            "engine_commits_during_rounds": snap.get("engine_commits"),
        },
        "models": C.MODELS,
        "corpus": snap.get("corpus"),
        "knowledge_graph": {"engine": "Neo4j Community (local only; not read by the engine at runtime)",
                            "version": neo.get("version"), "total_nodes": neo.get("total_nodes"),
                            "total_relationships": neo.get("total_relationships"), "labels": neo.get("labels"),
                            "relationship_counts": neo.get("relationships"), "patterns": neo.get("patterns")},
        "modal": {"apps": C.MODAL, "billing": {k: bill.get(k) for k in ("by_app_usd", "by_day_usd", "total_usd", "note")}},
        "julia_octave": C.JULIA_OCTAVE, "synthesis": C.SYNTHESIS, "qiraat": C.QIRAAT, "data_factory": C.FACTORY,
        "limitations": C.LIMITATIONS,
    }


# ---------------------------------------------------------------- HTML -------------------------------------
CSS = """
:root{--bg:#fbfaf7;--fg:#1b1b1b;--mut:#666;--line:#e2dfd6;--card:#fff;--ok:#1a7f4b;--bad:#b3261e;--warn:#8a6100;--code:#f3f1ea}
@media(prefers-color-scheme:dark){:root{--bg:#14140f;--fg:#ebebe6;--mut:#9a9a94;--line:#2e2e28;--card:#1c1c17;--ok:#4ade80;--bad:#f87171;--warn:#fbbf24;--code:#23231d}}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:24px 16px 80px} h1{font-size:1.5rem;margin:0 0 4px} h2{font-size:1.15rem;margin:34px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}
h3{font-size:1rem;margin:18px 0 6px} .sub,.mut{color:var(--mut);font-size:.85rem} a{color:inherit}
nav{display:flex;flex-wrap:wrap;gap:6px 12px;font-size:.84rem;margin:10px 0 4px} nav a{color:var(--mut)}
table{width:100%;border-collapse:collapse;font-size:.8rem;margin:6px 0 10px;display:block;overflow-x:auto}
td,th{padding:5px 7px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top} th{color:var(--mut);font-weight:600;white-space:nowrap}
.st{font:600 .68rem ui-monospace,monospace;padding:1px 6px;border-radius:4px;white-space:nowrap;border:1px solid currentColor}
.IMPLEMENTED,.TESTED,.EXISTS{color:var(--ok)} .PARTIAL,.EXPERIMENTAL{color:var(--warn)} .PLANNED,.CONCEPTUAL,.NOT{color:var(--mut)}
pre{background:var(--code);padding:10px 12px;border-radius:8px;overflow-x:auto;font-size:.8rem;line-height:1.35}
code{background:var(--code);padding:0 4px;border-radius:3px;font-size:.85em} .ar{direction:rtl;font-size:1.05rem}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin:10px 0}
.ok{color:var(--ok)} .bad{color:var(--bad)} .warn{color:var(--warn)} audio{height:28px;max-width:220px}
"""


def st(s: str) -> str:
    key = s.split()[0].split("(")[0].rstrip(",") if s else ""
    return f'<span class="st {E(key)}">{E(s)}</span>'


def table(rows: list[dict[str, Any]], cols: list[tuple[str, str]], status_col: str | None = None) -> str:
    h = "<table><tr>" + "".join(f"<th>{E(t)}</th>" for _k, t in cols) + "</tr>"
    for r in rows:
        h += "<tr>" + "".join(f"<td>{st(str(r.get(k, ''))) if k == status_col else _cell(r.get(k))}</td>" for k, _t in cols) + "</tr>"
    return h + "</table>"


def _cell(v: Any) -> str:
    if v is None:
        return '<span class="mut">—</span>'
    if isinstance(v, str) and v.startswith("/observatory/audio/"):
        return f'<audio controls preload="none" src="{E(v)}"></audio>'
    if isinstance(v, (dict, list)):
        return f"<code>{E(json.dumps(v, ensure_ascii=False))}</code>"
    return E(str(v))


PIPE_DIAGRAM = """Audio input (browser / phone upload; ffmpeg -> 16 kHz)
   │
   ▼
muaalem v3.2 acoustic model (CPU)  ── log-posteriors / 40 ms: 43 phonemes + 10 sifat heads
   │                                        ▲
   ▼                                        │ reference text: quran-transcript phonetiser (Ḥafs),
Alignment: joint CTC Viterbi over the       │ expected sifat per phoneme, rule parser
passage; each ayah padded into its pauses ◄─┘
   │
   ▼
Letter analysis (analyse_clip): identity vs classical confusions / deletion · sifat head margins ·
durations & count unit · [drills: makhraj-neighbour margins] · blind-spot gating (blindspots.json)
   │
   ▼
Rule analysis: rules located from text (parser) → bound to units → graded
   (lengths in counts vs bands · nasal bands from masters · wajh for munfaṣil · stops from waveform)
   │
   ▼
Reference: percentile / robust z vs masters (reference_stats.json) · stretch calculus (stretch_model.json)
   │
   ▼
Measurements JSON ──► Sessions scoring (expectations / scripted mistakes / false alarms) ──► session log
   │
   ▼
Evaluation: rescore_round.py / validate_round.py → a person changes code or thresholds → rescore
(no training loop; Neo4j is an offline research layer, not read at runtime)"""

GRAPH_DIAGRAM = """Surah ─HAS→ Ayah ─HAS→ QWord ─NEXT→ QWord          (text: 114 / 6,236 / 77,433)
QWord ─CARRIES{n}→ Rule ;  Ayah ─SAME_TAJWID / TRAINS→ Ayah
Letter ─HAS_SIFAH→ Sifah ;  Letter ─ARTICULATED_AT→ Makhraj ─IN→ Region ;  Sound ─PRECEDES{pmi}→ Sound
Reciter ─RECITED→ Performance ─OF→ Ayah ;  Performance ─HOLDS{stretch}→ Rule ;  Reciter ─IN_COMMUNITY→ Community
Context ─LETTER/BEFORE/AFTER→ Sound ;  Context ─FAILS{rate}→ Check      (reliability / blind spots)
Exercise ─SCRIPTS→ ScriptedMistake ─AT→ Word ;  Exercise ─RECORDED→ Take ─JUDGED{verdict}→ ScriptedMistake
Reciter ─SAID_RA{margin}→ RaContext ;  Skill ─COVARIES→ Skill ;  Rule ─PREREQUISITE_OF / FAILS_WITH→ Rule"""

FACTORY_DIAGRAM = """MASTER AUDIO ─► LETTER/RULE CORPUS ─► CONTROLLED PERTURBATION ─► SYNTHETIC AUDIO ─► PHYSICS VERIFICATION
     ▲                                                                                     │
     │                                                                                     ▼
NEW MASTER DISTRIBUTIONS ◄─ RETEST ◄─ TRAIN / IMPROVE ◄─ COMPARE EXPECTED vs DETECTED ◄─ MUQRI ENGINE"""


def page() -> str:
    from observatory import content as C
    snap = load("snapshot.json") or {}
    neo = load("neo4j_snapshot.json") or {}
    bill = load("modal_billing.json") or {}
    cur = load("current_engine.json") or {}
    rm = round_metrics()
    git = snap.get("git", {})
    parts: list[str] = []
    add = parts.append
    add(f"<h1>Muqri Observatory</h1><p class='sub'>Read-only technical view of the current implementation · git "
        f"<code>{E(git.get('revision', '')[:12])}</code> on <code>{E(git.get('branch', ''))}</code> · "
        f"{snap.get('tests_collected')} tests collected · <a href='/muqri-observatory/report'>full report</a> · "
        f"<a href='/api/observatory/manifest'>manifest JSON</a> · <a href='/sessions'>Sessions (rounds 1–5)</a></p>")
    add("<nav>" + " ".join(f"<a href='#{a}'>{t}</a>" for a, t in [
        ("arch", "Architecture"), ("engine", "Engine measures"), ("makharij", "17 makhārij"), ("sifat", "17 ṣifāt"),
        ("sessions", "Session lab"), ("perturb", "Perturbation matrix"), ("corpus", "Corpus"), ("anchors", "Master anchors"),
        ("graph", "Neo4j"), ("julia", "Julia + Octave"), ("modal", "Modal"), ("models", "Models"),
        ("synthesis", "Synthesis"), ("qiraat", "Qirāʾāt"), ("factory", "Data factory"), ("limits", "Limitations")]) + "</nav>")
    add("<p class='mut'>Status words: IMPLEMENTED (in the served Sessions path) · TESTED (tests or recorded measurements) · "
        "PARTIALLY IMPLEMENTED · EXPERIMENTAL (research code, measured) · PLANNED · CONCEPTUAL · NOT VERIFIED.</p>")

    add("<h2 id='arch'>1. Architecture</h2><h3>Two analysis engines</h3>")
    add(table(C.ENGINES, [("name", "engine"), ("status", "status"), ("module", "code"), ("used_by", "used by"),
                          ("model", "model"), ("reference_data", "reference data"), ("note", "note")], "status"))
    add(f"<h3>The Sessions engine path</h3><pre>{E(PIPE_DIAGRAM)}</pre>")
    add(table(C.PIPELINE, [("stage", "component"), ("status", "status"), ("module", "code"), ("language", "lang"),
                           ("library", "libraries"), ("model", "model / algorithm"), ("input", "input"),
                           ("output", "output"), ("compute", "compute"), ("evidence", "evidence"), ("note", "note")], "status"))
    add("<h3>Where the system relies on what</h3>" + table([{"layer": k, "where": v} for k, v in C.REASONING.items()],
                                                           [("layer", "layer"), ("where", "where it is used")]))

    add("<h2 id='engine'>2. What the engine measures</h2>")
    add(table(C.MEASURES, [("item", "item"), ("status", "status"), ("module", "code"), ("algorithm", "algorithm"),
                           ("features", "features"), ("uncertainty", "confidence / uncertainty"), ("output", "output schema"),
                           ("evidence", "validation evidence")], "status"))

    add("<h2 id='makharij'>3. The 17 makhārij</h2><p class='mut'>" + E(C.makharij()[0]["note"]) + "</p>")
    add(table(C.makharij(), [("n", "#"), ("makhraj", "makhraj"), ("letters", "letters"), ("representation", "acoustic representation"),
                             ("detection", "detection"), ("evidence", "test evidence"), ("score", "score"), ("status", "status")], "status"))

    add("<h2 id='sifat'>4. The 17 ṣifāt (and ghunnah)</h2>")
    add(table([dict(zip(("name", "arabic", "letters", "acoustic", "detector", "score", "status", "evidence"), s)) for s in C.SIFAT],
              [("name", "ṣifah"), ("arabic", ""), ("letters", "letters"), ("acoustic", "acoustic correlate"),
               ("detector", "current detector"), ("score", "score"), ("status", "status"), ("evidence", "evidence")], "status"))

    add("<h2 id='sessions'>5. Session lab, rounds 1–5</h2>")
    add("<p class='mut'>Each exercise was recorded by one certified reader, twice: take A to a written spec, take B with "
        "scripted mistakes (r2e4's take B is a second correct reading at another speed). <b>As recorded</b> = the engine "
        "that scored the take when it was submitted (first entry in sessions_results.jsonl). <b>Current engine</b> = the "
        f"same latest recording rescored by git <code>{E(str(cur.get('git_revision', ''))[:12])}</code> "
        f"({E(str(cur.get('scored_at')))}), dry run, written only to observatory/data/current_engine.json. "
        "Recording device: not stored by the system (uploads from a phone browser; formats below).</p>")
    rows = []
    for n, v in rm.items():
        a, c = v["as_recorded"], v["current_engine"]
        rows.append({"round": n, "mistakes caught (as recorded)": f"{a.get('caught', 0)}/{a.get('mistakes', 0)}",
                     "caught (current)": f"{c.get('caught', 0)}/{c.get('mistakes', 0)}",
                     "flagged other reason (as rec. → cur.)": f"{a.get('other_reason', 0)} → {c.get('other_reason', 0)}",
                     "expectations met (as recorded)": f"{a.get('met', 0)}/{a.get('expectations', 0)}",
                     "met (current)": f"{c.get('met', 0)}/{c.get('expectations', 0)}",
                     "false-alarm words A+B (as rec. → cur.)": f"{a.get('false_alarm_words', 0)} → {c.get('false_alarm_words', 0)}"})
    add(table(rows, [(k, k) for k in rows[0]]))
    add("<p class='mut'>What these numbers can and cannot establish: rounds use different passages, rules and mistake types, "
        "so round-to-round rates are not a controlled learning curve. The 'current engine' column re-reads OLD rounds with "
        "code changed after seeing them; a rise there is partly fit to these recordings. The only held-out evidence per "
        "change is the next round's first scoring. See the rescore trails and the engine commits below.</p>")
    for ex in snap.get("exercises", []):
        add(f"<div class='card'><b>{E(ex['id'])}</b> — {E(ex['title'])}<br><span class='mut'>surah {ex['surah']} ayahs "
            f"{ex['ayahs'][0]}–{ex['ayahs'][1]} · wajh {E(ex['wajh'])}</span><p>{E(ex['goal'])}</p>"
            f"<p class='mut'>Spec: {E(ex['spec'])}</p>"
            + "".join(f"<audio controls preload='none' src='{E(r['url'])}'></audio> <span class='mut'>{E(r['take'])} "
                      f"{E(r['stamp'])} · {E(r['format'])} · {r['audio_seconds'] or '?'} s</span><br>"
                      for r in snap.get("recordings", []) if r["exercise"] == ex["id"]) + "</div>")
    add("<h3>Engine commits during the rounds</h3>")
    add(table(snap.get("engine_commits", []), [("time", "time"), ("hash", "commit"), ("subject", "subject")]))
    add("<h3>Rescore trails (every scoring of the same recording, in order)</h3>")
    trows = []
    for k, evs in rescore_trails().items():
        trows.append({"recording": k, "trail": " → ".join(
            (f"{e['met']}/{e['expectations']}" if e["expectations"] else f"{e['caught']}/{e['mistakes']}") + f" ({e['false_alarms']} FA)"
            for e in evs), "notes": " | ".join(e["note"][len("rescore of 00000000-000000 "):] if e["kind"] == "rescore" else "as recorded"
                                             for e in evs)})
    add(table(trows, [("recording", "recording"), ("trail", "met or caught (false alarms), in order"), ("notes", "why rescored")]))
    dg = digest_by_round()
    add("<h3>Per-rule, per-letter, per-characteristic (take A, current engine)</h3>")
    for n in ROUNDS:
        d = dg[n]
        add(f"<details><summary>Round {n}</summary>"
            + table([{"rule": k, **v} for k, v in d["rules"].items()], [("rule", "rule")] + [(s, s) for s in
                    ("pass", "short", "long", "wrong", "unconfirmed", "no_evidence")])
            + table([{"letter": k, **v} for k, v in d["letters"].items()], [("letter", "letter"), ("confirmed", "confirmed"),
                                                                             ("not_confirmed", "not confirmed")])
            + table([{"head": k, **v} for k, v in d["characteristics"].items()], [("head", "characteristic head"),
                    ("realised", "realised"), ("not_realised", "not realised")]) + "</details>")

    add("<h2 id='perturb'>6. Deliberately perturbed recitations (take B matrix)</h2>")
    add("<p class='mut'>Every scripted mistake of every take B, rounds 1–5. 'target' is the engine signature that counts as "
        "catching it. Evidence strings carry the measured magnitude (counts, deviation) or margin (nats). "
        "'flagged, other reason' = the word was flagged, but not by the targeted check.</p>")
    add(table(perturbation_matrix(), [("exercise", "ex"), ("ayah", "ayah"), ("word", "w"), ("text", "word"),
                                      ("perturbation", "intended perturbation"), ("target", "intended check(s)"),
                                      ("as_recorded", "as recorded"), ("as_recorded_evidence", "evidence (as recorded)"),
                                      ("current", "current engine"), ("current_evidence", "evidence (current)"),
                                      ("false_alarms_elsewhere_as_recorded", "FA words, take (as rec.)"),
                                      ("false_alarms_elsewhere_current", "FA words (current)"),
                                      ("audio", "take B"), ("take_a_audio", "take A")]))

    co = snap.get("corpus", {})
    add("<h2 id='corpus'>7. Reciter corpus</h2>")
    add(f"<p>Reciters with a full-Qur'an measurement (≥ 6,233 of 6,236 ayahs in some run): <b>{len(co.get('full_quran_reciters', []))}</b>: "
        f"{E(', '.join(co.get('full_quran_reciters', [])))}. The figure of ~45 complete reciters is not supported by the data "
        f"here: the T300 set has {co.get('t300_posteriors', {}).get('speakers')} speakers × "
        f"{co.get('t300_posteriors', {}).get('clips_per_speaker')} ayahs each (a 300-ayah tier, not the full Qur'an).</p>")
    add("<h3>Quran-MD (Kaggle) rows, benchmark pipeline</h3>" + table(
        [{"reciter | mode": k, "ayahs": v} for k, v in co.get("quran_md_kaggle", {}).items()], [("reciter | mode", "reciter | mode"), ("ayahs", "ayahs")]))
    add("<h3>studio-all (Modal, EveryAyah), benchmark pipeline</h3>" + table(
        [{"reciter": k, "ayahs": v} for k, v in co.get("studio_all_everyayah", {}).items()], [("reciter", "reciter"), ("ayahs", "ayahs")]))
    add("<p class='mut'>Representation: ayah-level recordings (EveryAyah mp3 64–192 kbps; Quran-MD 64 kbps WAV). Verse "
        "segmentation comes from the source (one file per ayah). Word and letter segmentation are the engine's alignment "
        "(40 ms frames). Rights / licence metadata: none stored per recording. Studio vs mosque: a hand-assigned tier per "
        "reciter folder (datastore/ingest.py: anchor / studio / imam / fast); the benchmark pipeline also has a "
        "'taraweeh_adapted' mode (dereverberation + pace normalisation), not a separate recording set.</p>")
    lc = co.get("letter_corpus") or {}
    add("<h3>Individual letter audio corpus (pilot, EXPERIMENTAL)</h3>")
    add(f"<p>{lc.get('cells')} cells ({lc.get('letter_cells')} letter × form, {lc.get('rule_cells')} rule × form), up to 5 "
        f"instances per source: {E(json.dumps(lc.get('instances_per_source', {})))}. Chain: plan.py (363 short ayahs chosen "
        "so every cell has 5 instances) → measure (Engine.analyze with the makhraj test; Modal) → assemble.py (cut: the "
        "letter + its vowel/madd or qalqalah release, ±50 ms; boundaries = the engine's 40 ms Viterbi frames, NO "
        "refinement yet) → clean.py (DC removal, 60 Hz zero-phase high-pass, spectral gating from the ayah's quietest "
        "10 % of frames, zero-crossing cuts, 5 ms fades, −20 dBFS RMS / −1 dBFS peak; raw cut kept beside it). The "
        "cleaning's effect on the engine's measurements has not been measured. Row: reciter → surah → ayah → word → "
        "letter → form/context → clip span → identity, heads, vowel, makhraj margins → rule label. Served on the "
        "engine app at /letters (not through this observatory).</p>")

    add("<h2 id='anchors'>8. Master / reference anchors</h2>")
    tiers = co.get("tier_sets", {})
    add(f"<p>Masters are <b>assigned by hand</b> from recording-folder names (datastore/ingest.py::ladder): anchor = "
        f"{E(', '.join(tiers.get('anchor', [])))}; studio (peers) = {len(tiers.get('studio', []))} folders; imams = "
        f"{len(tiers.get('imams', []))}; everything else 'fast'. The Sessions engine's percentile_masters / z_masters are "
        "computed against the anchor tier's T300 instances (reference_stats.json). Learner recordings are whatever "
        "is uploaded; the system is told nothing else.</p><p><b>Inferring master-likeness without labels:</b> no "
        "such scorer is in the served path. Evidence against the benchmark pipeline's perfection index doing it: "
        "masters vs learners AUROC 0.111 (learners scored higher; different texts, no calibration) — "
        "research_agency_lab/experiments/learner_eval/results/learners_uncalibrated.json. The knowledge-graph study "
        "found reciter communities from 50 style features without labels (NMI 0.279 vs hand tiers, p = 0.009; "
        "GRAPH.md) — a clustering, not a master detector.</p>")

    add("<h2 id='graph'>9. Neo4j knowledge graph (snapshot)</h2>")
    add(f"<p>Neo4j {E(json.dumps(neo.get('version')))} on 127.0.0.1 only; not exposed; not read by the engine at runtime. "
        f"<b>{neo.get('total_nodes', 0):,}</b> nodes, <b>{neo.get('total_relationships', 0):,}</b> relationships "
        "(counted from a read-only session, snapshot file). Note: 16 Makhraj nodes (not 17) and a sessions layer of "
        "7 exercises / 37 takes (rounds 1–2 only).</p>")
    add(f"<pre>{E(GRAPH_DIAGRAM)}</pre>")
    add(table([{"label": k, "count": v, "properties": ", ".join((neo.get("properties") or {}).get(k, []))}
               for k, v in sorted((neo.get("labels") or {}).items(), key=lambda x: -x[1])],
              [("label", "node label"), ("count", "nodes"), ("properties", "properties (sampled)")]))
    add(table(neo.get("patterns", []), [("from", "from"), ("rel", "relationship"), ("to", "to"), ("count", "count")]))
    add(table(neo.get("indexes", []) + neo.get("constraints", []), [("name", "index / constraint"), ("type", "type"),
                                                                    ("labelsOrTypes", "on"), ("properties", "properties")]))

    J = C.JULIA_OCTAVE
    add(f"<h2 id='julia'>10. Julia + GNU Octave</h2><p>{E(J['summary'])}</p><pre>{E(J['flow'])}</pre>")
    add(table([{"file": f, "what": w} for f, w in J["julia"]], [("file", "Julia (research_agency_lab/substrate_library/julia/)"), ("what", "what it computes")]))
    add(table([{"file": f, "what": w} for f, w in J["octave"]], [("file", "Octave (substrate_library/octave/)"), ("what", "what it computes")]))

    add("<h2 id='modal'>11. Modal</h2>" + table(C.MODAL, [("app", "app"), ("file", "code"), ("compute", "compute"), ("what", "what ran")]))
    add(f"<p>Billing, from <code>{E(str(bill.get('source')))}</code>: total <b>${bill.get('total_usd')}</b>; by app "
        f"{E(json.dumps(bill.get('by_app_usd')))}; by day {E(json.dumps(bill.get('by_day_usd')))}. {E(str(bill.get('note', '')))}</p>")

    add("<h2 id='models'>12. Models</h2>" + table(C.MODELS, [("name", "model"), ("version", "version"), ("purpose", "purpose"),
        ("input", "input"), ("output", "output"), ("where", "where / compute"), ("training", "trained?"), ("interaction", "role")]))

    S = C.SYNTHESIS
    add("<h2 id='synthesis'>13. Synthesis</h2>")
    for k, t in [("already_implemented", "1. Already implemented"), ("experimentally_tested", "2. Experimentally tested"),
                 ("prototype", "3. Prototype"), ("planned", "4. Planned"), ("not_implemented", "5. Not yet implemented")]:
        add(f"<h3>{t}</h3><ul>" + "".join(f"<li>{E(x)}</li>" for x in S[k]) + "</ul>")
    add("<p class='mut'>Proposed architecture: docs/synthesis-engine.md and docs/epics/EPIC-1.md (plans, not implementation).</p>")

    Q = C.QIRAAT
    add("<h2 id='qiraat'>14. Qirāʾāt</h2>" + table([{"aspect": k.replace("_", " "), "current state": v} for k, v in Q.items()],
                                                   [("aspect", "aspect"), ("current state", "current state")]))

    add(f"<h2 id='factory'>15. Data generation factory</h2><pre>{E(FACTORY_DIAGRAM)}</pre>")
    add(table([{"stage": a, "status": b, "what": c} for a, b, c in C.FACTORY], [("stage", "stage"), ("status", "status"), ("what", "what exists")], "status"))

    add("<h2 id='limits'>16. Limitations</h2><ul>" + "".join(f"<li>{E(x)}</li>" for x in C.LIMITATIONS) + "</ul>")
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,"
            f"initial-scale=1'><title>Muqri Observatory</title><style>{CSS}</style></head><body><div class='wrap'>"
            + "".join(parts) + "</div></body></html>")


# ---------------------------------------------------------------- the app ----------------------------------

def create_app():  # type: ignore[no-untyped-def]
    import httpx
    from fastapi import FastAPI, Request
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response

    api = FastAPI(title="Muqri Observatory", docs_url=None, redoc_url=None, openapi_url=None)
    tok = token()

    @api.middleware("http")
    async def guard(request: Request, call_next):  # type: ignore[no-untyped-def]
        if request.method not in ("GET", "HEAD"):
            return JSONResponse({"error": "read-only inspection view"}, status_code=405)
        if request.url.path == PUBLIC_PATH:          # sanitized machine-review JSON: no token needed
            resp = await call_next(request)
            resp.headers["X-Robots-Tag"] = "noindex, nofollow"
            resp.headers["Cache-Control"] = "no-store"
            return resp
        q = request.query_params.get("token")
        if q is not None and secrets.compare_digest(q, tok):
            r = RedirectResponse(str(request.url.remove_query_params("token")), status_code=303)
            r.set_cookie("obs", tok, httponly=True, secure=True, samesite="lax", max_age=14 * 86400)
            return r
        if not secrets.compare_digest(request.cookies.get("obs", ""), tok):
            return HTMLResponse("<h3>Muqri Observatory</h3><p>Access needs the reviewer link (with its token).</p>",
                                status_code=401)
        resp = await call_next(request)
        resp.headers["X-Robots-Tag"] = "noindex, nofollow"
        resp.headers["Referrer-Policy"] = "no-referrer"
        return resp

    async def backend(path: str) -> httpx.Response:
        async with httpx.AsyncClient(timeout=30) as c:
            return await c.get(BACKEND + path)

    @api.get("/")
    def root():  # type: ignore[no-untyped-def]
        return RedirectResponse("/muqri-observatory")

    @api.get("/sessions", response_class=HTMLResponse)
    async def sessions_page():  # type: ignore[no-untyped-def]
        r = await backend("/sessions")
        banner = ("<div style='background:#8a6100;color:#fff;padding:6px 12px;font:600 13px system-ui;position:sticky;top:0;"
                  "z-index:60'>Read-only inspection view · rounds 1–5 · recording and uploading are disabled here · "
                  "<a style='color:#fff' href='/muqri-observatory'>Observatory</a></div>")
        hide = "<style>.rec,.phone,input[type=file],.lq button{display:none!important}</style>"
        return HTMLResponse(r.text.replace("<body>", "<body>" + banner, 1).replace("</head>", hide + "</head>", 1),
                            status_code=r.status_code)

    @api.get("/sessions/rounds")
    async def rounds():  # type: ignore[no-untyped-def]
        d = (await backend("/sessions/rounds")).json()
        d["rounds"] = [n for n in d.get("rounds", []) if n in ROUNDS]
        d["history"] = [h for h in d.get("history", []) if h.get("round") in ROUNDS]
        return JSONResponse(d)

    @api.get("/sessions/round/{n}")
    async def round_(n: int):  # type: ignore[no-untyped-def]
        if n not in ROUNDS:
            return JSONResponse({"error": "rounds 1-5 only in this view"}, status_code=404)
        r = await backend(f"/sessions/round/{n}")
        return Response(r.content, status_code=r.status_code, media_type="application/json")

    @api.get("/observatory/audio/{exercise}/{take}/{name}")
    def audio(exercise: str, take: str, name: str):  # type: ignore[no-untyped-def]
        root = RECORDINGS.resolve()
        p = (RECORDINGS / exercise / take / name).resolve()
        ok = (len(exercise) >= 2 and exercise[0] == "r" and exercise[1].isdigit() and int(exercise[1]) in ROUNDS
              and take in ("A", "B") and root in p.parents and p.suffix.lower() in AUDIO_TYPES and p.is_file())
        if not ok:
            return JSONResponse({"error": "no such recording"}, status_code=404)
        return FileResponse(p, media_type=AUDIO_TYPES[p.suffix.lower()])

    @api.get("/muqri-observatory", response_class=HTMLResponse)
    def observatory():  # type: ignore[no-untyped-def]
        return page()

    @api.get("/muqri-observatory/report", response_class=HTMLResponse)
    def report_html():  # type: ignore[no-untyped-def]
        from observatory.report import render_html
        return render_html()

    @api.get("/muqri-observatory/report.md", response_class=PlainTextResponse)
    def report_md():  # type: ignore[no-untyped-def]
        from observatory.report import render_markdown
        return render_markdown()

    @api.api_route(PUBLIC_PATH, methods=["GET", "HEAD"])
    async def public_json():  # type: ignore[no-untyped-def]
        body = public_review()
        text = json.dumps(body, ensure_ascii=False, indent=1)
        leaks = sanitize_check(text, tok)
        if leaks:                                    # fail closed: never serve something that looks sensitive
            return JSONResponse({"error": "withheld: sanitization check failed", "patterns": leaks}, status_code=500)
        return Response(text, media_type="application/json")

    @api.get("/api/observatory/manifest")
    def manifest_json():  # type: ignore[no-untyped-def]
        return JSONResponse(manifest())

    return api


def main() -> int:
    import uvicorn
    uvicorn.run(create_app(), host="127.0.0.1", port=PORT, log_level="warning", proxy_headers=True,
                forwarded_allow_ips="127.0.0.1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
