#!/usr/bin/env python3
"""Assemble the letter corpus: every cell's instances per reciter, as clean clips with their scores.

Reads data/measure/<reciter>/*.json (measure.py) and writes

  data/clips/<reciter>/<cell-slug>/<n>.wav      clean (clean.py), and <n>.raw.wav, the plain cut
  data/corpus.json                              the index the /letters page reads

A letter instance is the letter with what completes its form: the vowel after it (and the madd, for a
long vowel), the qalqalah echo after a sakin, nothing more at a stop. A rule instance spans the rule's
letters. Both get 50 ms of context either side: a letter has no clean edge, it is co-articulated.

Scores, all from the engine's own measurements:

  letter  every check the engine scores on it -- identity, the vowel's identity, each characteristic
          head -- as a margin in nats (log p(expected) - log p(best other)); `weakest` is the lowest
          (the check it came closest to failing), `clean` whether every check held; the makhraj
          test (against the letters at neighbouring points) is reported beside it, not scored
  rule    status (pass / short / long / wrong), counts against the expected band, z against the masters

Ranking per cell: letters by the median `weakest` margin over the reciter's instances (higher =
clearer), rules by the share that passed, then the median |z| against the masters (smaller = nearer
the masters' centre).

The masters read the same ayahs (plan.json), so their instances are the same text positions. The
learner's come from their session takes, whatever cells those hold.

    .venv/bin/python research_agency_lab/experiments/letter_corpus/assemble.py
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

HERE = Path(__file__).parent
DATA = HERE / "data"
PER = 5
PAD_S = 0.05
MAX_S = 3.0
SHORT = {"َ": "fatha", "ِ": "kasra", "ُ": "damma"}
MADD = set("اۥۦ")
FORMS = ["fatha", "kasra", "damma", "fatha_long", "kasra_long", "damma_long", "sakin", "shadda", "stop"]
FAST = {"Abdurrahmaan_As-Sudais_192kbps"}


def letter_cells(L: list[dict[str, Any]]) -> list[tuple[str, str, int]]:
    """(letter, form, index) per consonant of one clip's measured letters -- plan.letter_cells' forms."""
    from app.letters import CONSONANTS
    out = []
    for i, l in enumerate(L):
        if l["symbol"] not in CONSONANTS:
            continue
        nxt = L[i + 1]["symbol"] if i + 1 < len(L) and L[i + 1]["id"].rsplit(":", 1)[0] == l["id"].rsplit(":", 1)[0] else None
        if l["run_length"] >= 2:
            form = "shadda"
        elif nxt is None:
            form = "stop"
        elif nxt in SHORT:
            long_ = i + 2 < len(L) and L[i + 2]["symbol"] in MADD
            form = SHORT[nxt] + ("_long" if long_ else "")
        else:
            form = "sakin"
        out.append((l["symbol"], form, i))
    return out


def rule_cells(m: dict[str, Any], texts: dict[str, str]) -> list[tuple[str, str, dict[str, Any]]]:
    """(rule, detail, measured rule) per rule instance; details come from re-parsing the text."""
    parser = _parser()
    out = []
    by_line = defaultdict(list)
    for r in m["rules"]:
        by_line[r["id"].rsplit(":", 1)[0]].append(r)
    for key, rs in by_line.items():
        parsed = parser.parse(texts[key]).rules if key in texts else []
        pool = [(p.rule_type.value, p.word_index, p.detail or "") for p in parsed]
        for r in rs:
            hit = next((x for x in pool if x[0] == r["rule"] and x[1] == r["word"]), None)
            if hit:
                pool.remove(hit)
            d = hit[2] if hit else ""
            d = d.split(";")[-1].strip() if r["rule"].startswith(("ikhfa", "idgham_ghunnah")) else d
            out.append((r["rule"], d, r))
    return out


_P = None


def _parser():  # type: ignore[no-untyped-def]
    global _P
    if _P is None:
        from app.tajweed_rules.parser import TajweedParser
        _P = TajweedParser()
    return _P


def letter_span(L: list[dict[str, Any]], i: int, form: str) -> tuple[float, float, list[int]]:
    idx = [i]
    if form not in ("sakin", "stop"):
        j = i + 1
        idx.append(j)
        if j + 1 < len(L) and L[j + 1]["symbol"] in MADD:
            idx.append(j + 1)
    elif i + 1 < len(L) and L[i + 1]["symbol"] == "ڇ":
        idx.append(i + 1)
    t0 = L[idx[0]]["onset_s"]
    t1 = L[idx[-1]]["onset_s"] + L[idx[-1]]["duration_s"]
    return t0, t1, idx


def letter_score(L: list[dict[str, Any]], idx: list[int]) -> dict[str, Any]:
    l = L[idx[0]]
    checks: dict[str, float] = {}
    held = True
    if l["identity"]["scored"] and l["identity"]["margin"] is not None:
        checks["identity"] = l["identity"]["margin"]
        held &= l["identity"]["confirmed"]
    for k in idx[1:]:
        v = L[k]
        if v["symbol"] in SHORT and v["identity"]["margin"] is not None:
            checks["vowel"] = v["identity"]["margin"]
            held &= v["identity"]["confirmed"]
    for h, c in l["characteristics"].items():
        if c["scored"] and c["margin"] is not None:
            checks[h] = c["margin"]
            held &= c["realised"]
    mk = l.get("makhraj") or {}
    near = mk.get("neighbours") or {}
    return {"weakest": round(min(checks.values()), 2) if checks else None,
            "weakest_check": min(checks, key=checks.get) if checks else None,
            "clean": bool(held), "checks": {k: round(v, 2) for k, v in checks.items()},
            "heard": {h: c["observed"] for h, c in l["characteristics"].items() if c["scored"] and not c["realised"]},
            "identity_heard": None if l["identity"]["confirmed"] else l["identity"]["competitor"],
            "makhraj_point": mk.get("point"), "makhraj": {q: round(v, 2) for q, v in near.items()},
            "makhraj_held": mk.get("held"), "duration_s": round(sum(L[k]["duration_s"] for k in idx), 3)}


def rule_score(r: dict[str, Any]) -> dict[str, Any]:
    return {"status": r["status"], "counts": r["observed_counts"], "expected": r["expected_counts"],
            "deviation": r["deviation_counts"], "z_masters": r["z_masters"], "margin": r["evidence_margin"],
            "seconds": r["observed_seconds"]}


def slug(cell: str) -> str:
    return re.sub(r"[^\w؀-ۿ]+", "_", cell).strip("_")[:80]


def main() -> None:
    import soundfile as sf
    from app.webapp import decode_upload
    from datastore.review_queue import audio_path
    from research_agency_lab.experiments.letter_corpus.clean import clean_ayah, cut
    plan = json.loads((HERE / "plan.json").read_text())
    order = {(x["surah"], x["ayah"]): k for k, x in enumerate(plan["chosen"])}
    cells: dict[str, dict[str, Any]] = {}
    for rdir in sorted((DATA / "measure").iterdir()):
        rec = rdir.name
        files = sorted(rdir.glob("*.json"))
        docs = []
        for f in files:
            try:
                docs.append(json.loads(f.read_text()))
            except json.JSONDecodeError:        # being written by measure.py right now
                continue
        if rec != "learner":
            docs.sort(key=lambda d: order.get((d["surah"], d["ayah"]), 1e9))
        taken: dict[str, int] = defaultdict(int)
        for d in docs:
            m = d["measurements"]
            L = m["letters"]
            words = {w["ref"]: w["text"] for w in m["words"]}
            if d.get("lines"):
                texts = {f"0:{k}": t for k, t in enumerate(d["lines"], 1)}
            else:                           # every ayah the clip covers, whole (rule words are absolute)
                from quran_transcript import Aya
                keys = {x["id"].rsplit(":", 1)[0] for x in m["rules"]}
                texts = {k: Aya(*map(int, k.split(":"))).get().uthmani for k in keys}
            need = []
            for sym, form, i in letter_cells(L):
                need.append(("letter", f"letter:{sym}:{form}", i, None))
            for rule, detail, r in rule_cells(m, texts):
                need.append(("rule", f"rule:{rule}:{detail}", None, r))
            need = [n for n in need if taken[n[1]] < PER]
            if not need:
                continue
            src = Path(d["audio"]) if d.get("audio") and rec == "learner" else audio_path(rec, d["surah"], d["ayah"])
            raw = decode_upload(src.read_bytes())
            clean = clean_ayah(raw)
            dur = len(raw) / 16000
            for kind, cell, i, r in need:
                if taken[cell] >= PER:
                    continue
                if kind == "letter":
                    t0, t1, idx = letter_span(L, i, cell.split(":")[2])
                    score = letter_score(L, idx)
                    lid = L[i]["id"]
                    ref = f"{lid.rsplit(':', 1)[0]}:{L[i]['word']}"
                else:
                    ls = {x["id"]: x for x in L}
                    us = [ls[x] for x in r["letters"] if x in ls]
                    if not us:
                        continue
                    t0, t1 = min(u["onset_s"] for u in us), max(u["onset_s"] + u["duration_s"] for u in us)
                    score = rule_score(r)
                    ref = f"{r['id'].rsplit(':', 1)[0]}:{r['word']}"
                t0, t1 = max(0.0, t0 - PAD_S), min(dur, t1 + PAD_S, t0 + MAX_S)
                n = taken[cell]
                taken[cell] += 1
                out = DATA / "clips" / rec / slug(cell)
                out.mkdir(parents=True, exist_ok=True)
                sf.write(out / f"{n}.wav", cut(clean, t0, t1), 16000, subtype="PCM_16")
                sf.write(out / f"{n}.raw.wav", cut(raw, t0, t1, level=False), 16000, subtype="PCM_16")
                c = cells.setdefault(cell, {"cell": cell, "kind": kind, "reciters": {}})
                c["reciters"].setdefault(rec, []).append({
                    "clip": f"{rec}/{slug(cell)}/{n}.wav", "ref": ref, "word": words.get(ref, ""),
                    "source": src.name, "span_s": [round(t0, 3), round(t1, 3)], **score})
        print(f"{rec}: {sum(taken.values())} instances in {len(taken)} cells", flush=True)
    for c in cells.values():
        rank = []
        for rec, xs in c["reciters"].items():
            if c["kind"] == "letter":
                w = [x["weakest"] for x in xs if x["weakest"] is not None]
                rank.append({"reciter": rec, "n": len(xs), "median_weakest": round(statistics.median(w), 2) if w else None,
                             "clean": round(sum(x["clean"] for x in xs) / len(xs), 2)})
            else:
                zs = [abs(x["z_masters"]) for x in xs if x["z_masters"] is not None]
                rank.append({"reciter": rec, "n": len(xs), "passed": round(sum(x["status"] == "pass" for x in xs) / len(xs), 2),
                             "median_abs_z": round(statistics.median(zs), 2) if zs else None})
        if c["kind"] == "letter":
            rank.sort(key=lambda x: -(x["median_weakest"] if x["median_weakest"] is not None else -1e9))
        else:
            rank.sort(key=lambda x: (-x["passed"], x["median_abs_z"] if x["median_abs_z"] is not None else 1e9))
        c["ranking"] = rank
        parts = c["cell"].split(":", 2)
        c["what"], c["form"] = parts[1], parts[2]
    reciters = sorted({r for c in cells.values() for r in c["reciters"]})
    (DATA / "corpus.json").write_text(json.dumps({
        "reciters": reciters, "fast": sorted(FAST & set(reciters)), "per": PER, "forms": FORMS,
        "cells": sorted(cells.values(), key=lambda c: (c["kind"], c["what"], FORMS.index(c["form"]) if c["form"] in FORMS else 99, c["form"]))},
        ensure_ascii=False))
    print(f"{len(cells)} cells, {len(reciters)} reciters -> {DATA / 'corpus.json'}")


if __name__ == "__main__":
    main()
