#!/usr/bin/env python3
"""A performance profile of every reciter -- what they do, not a grade.

A master is not someone who never slips; it is someone whose recitation is under constant, conscious
control: their lengths are their own and consistent, their characteristics survive their tempo, and a
slip in one verse is corrected in the next. A fast taraweeh imam and a teaching reciter are different
instruments, not better and worse ones -- but a student should learn timing and characteristics from
the second, never the first. So each profile records:

* **tempo** -- the reciter's count unit in seconds, its class (tahqiq / tadwir / hadr), and whether the
  recitation can serve as a model for learners (not at hadr: no teacher lets a student recite for
  certification at prayer speed);
* **their own lengths** -- the median and spread of every madd type and of ghunnah, in counts;
* **characteristics** -- the realisation rate of each of the ten heads and of letter identity, and the
  reciter's strengths and weaknesses relative to the whole cohort (robust z-scores, never an absolute
  pass mark);
* **per surah and per verse** -- the same figures wherever the reciter recited, so two reciters can be
  compared on the identical verse (87 verses are shared by all 41 in T300);
* **self-correction across verses** -- when a rule or characteristic slips in one ayah, how often the
  very next ayah that needs it gets it right. Thin on T300 (about 25 consecutive pairs per reciter)
  and reported with its n; continuous passages from new recordings fill it in.

Everything is measured by the same engine the app serves (`Engine.analyze` on the T300 posteriors), so
a profile and a learner's report speak the same language.

    .venv/bin/python -m datastore.reciter_profile clips [per_reciter] [procs]   # engine over the dump
    .venv/bin/python -m datastore.reciter_profile profile                        # aggregate + DuckDB
    .venv/bin/python -m datastore.reciter_profile add REPORT.json SPEAKER [SOURCE]  # a saved report
"""

from __future__ import annotations

import collections
import json
import statistics
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DUMP = ROOT / "research_agency_lab/experiments/qaari_keys/modal_T300"
OUT = ROOT / "research_agency_lab/experiments/profiles"
CLIPS = OUT / "clips.jsonl"
PROFILES = OUT / "reciter_profiles.json"

# the engine's own tempo classes (app/submission.py) and what each is fit for
TEMPO_CLASSES = (("tahqiq", 0.30), ("tadwir", 0.20), ("hadr", 0.0))
LEARNING_MODEL = {"tahqiq": True, "tadwir": True, "hadr": False}
MIN_N = 30                        # a rate from fewer instances is reported, never ranked


def tempo_class(h: float | None) -> str | None:
    if not h:
        return None
    return next(name for name, lo in TEMPO_CLASSES if h >= lo)


def record(report: dict, speaker: str, source: str, surah: int, ayah: int) -> dict:  # type: ignore[type-arg]
    """The compact per-verse record a profile is built from, taken from one engine report."""
    a = report["ayahs"][0] if report.get("ayahs") else {}
    madd, rules = [], []
    for v in a.get("rules", []):
        rules.append((v["rule"], v["status"]))
        g = (v.get("evidence") or {}).get("given_counts")
        if v["mechanism"] == "durational" and g is not None and v["status"] in {"pass", "short", "long"}:
            madd.append((v["rule"], g, v["status"]))
    heads: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    ident = [0, 0]
    for l in a.get("letters", []):
        ident[0] += 1
        ident[1] += bool(l["identity"]["confirmed"])
        for h, sv in (l.get("sifat") or {}).items():
            heads[h][0] += 1
            heads[h][1] += bool(sv.get("realised"))
    s = report.get("summary", {})
    return {"source": source, "speaker": speaker, "surah": surah, "ayah": ayah,
            "haraka_s": a.get("haraka_s"), "haraka_source": a.get("haraka_source"),
            "word_accuracy": s.get("accuracy"), "rule_accuracy": s.get("rule_accuracy"),
            "letter_accuracy": s.get("letter_accuracy"),
            "words": len(a.get("words", [])),
            "faulted_words": sum(bool(w["faults"]) for w in a.get("words", [])),
            "madd": madd, "rules": rules, "heads": dict(heads), "identity": ident,
            "fault_families": sorted({f.get("rule") or f.get("sifah") or "letter"
                                      for w in a.get("words", []) for f in w["faults"]})}


# ------------------------------------------------------------------ stage 1: the engine over the dump
_eng = None
_lay = None


def _init() -> None:
    global _eng, _lay
    warnings.filterwarnings("ignore")
    from app.engine import Engine  # noqa: PLC0415
    _lay = json.loads((DUMP / "layout.json").read_text())
    _eng = Engine(layout=_lay)


def _clip(r: dict) -> dict | None:  # type: ignore[type-arg]
    import numpy as np  # noqa: PLC0415
    try:
        lp = np.fromfile(DUMP / r["file"], dtype="<f4").reshape(r["frames"], _lay["columns"])  # type: ignore[index]
        rep = _eng.analyze(None, [(r["sura"], r["aya"])], posteriors=lp)  # type: ignore[union-attr]
        return record(rep, r["speaker"], "everyayah", r["sura"], r["aya"])
    except Exception as exc:  # noqa: BLE001 - one bad clip must not stop 12,000
        return {"source": "everyayah", "speaker": r["speaker"], "surah": r["sura"], "ayah": r["aya"],
                "error": repr(exc)}


def clips(per: int = 10_000, procs: int = 3) -> None:
    recs, seen = [], collections.Counter()
    for line in open(DUMP / "index.jsonl"):
        r = json.loads(line)
        if "file" in r and seen[r["speaker"]] < per:
            seen[r["speaker"]] += 1
            recs.append(r)
    OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    with Pool(procs, initializer=_init) as p, open(CLIPS, "w") as f:
        for x in p.imap_unordered(_clip, recs, chunksize=4):
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
            n += 1
            if n % 500 == 0:
                print(f"{n}/{len(recs)} clips", flush=True)
    print(f"{n} clip records -> {CLIPS}")


def add(report_path: str, speaker: str, source: str = "learner") -> None:
    """Append a saved engine report (a protocol take, a learner's upload) to the clip records."""
    rep = json.loads(Path(report_path).read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    with open(CLIPS, "a") as f:
        for a in rep.get("ayahs", []):
            one = {**rep, "ayahs": [a]}
            f.write(json.dumps(record(one, speaker, source, a["surah"], a["ayah"]), ensure_ascii=False) + "\n")


# ------------------------------------------------------------------ stage 2: aggregate
def _q(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))]


def _robust_z(x: float, pop: list[float]) -> float | None:
    med = statistics.median(pop)
    mad = statistics.median(abs(p - med) for p in pop) or 1e-9
    return round((x - med) / (1.4826 * mad), 2)


def _self_correction(rows: list[dict]) -> dict:  # type: ignore[type-arg]
    """For consecutive ayahs: a family that slipped in ayah n -- is it right the next time it is needed?"""
    by = {(r["surah"], r["ayah"]): r for r in rows}
    slipped = corrected = 0
    for (s, a), r in by.items():
        nxt = by.get((s, a + 1))
        if not nxt:
            continue
        needed_next = {x for x, _st in nxt["rules"]} | set(nxt["heads"])
        for fam in r["fault_families"]:
            if fam in needed_next:
                slipped += 1
                corrected += fam not in nxt["fault_families"]
    return {"slips_followed_by_a_chance": slipped, "corrected_next_ayah": corrected,
            "rate": round(corrected / slipped, 4) if slipped else None,
            "note": "descriptive; T300 has ~25 consecutive pairs per reciter"}


def profile() -> None:
    rows = [json.loads(line) for line in open(CLIPS)]
    ok = [r for r in rows if "error" not in r]
    by = collections.defaultdict(list)
    for r in ok:
        by[(r["source"], r["speaker"])].append(r)

    # per-reciter head realisation rates, for the cohort comparison
    rate = {}
    for key, rs in by.items():
        agg = collections.defaultdict(lambda: [0, 0])
        for r in rs:
            for h, (n, k) in r["heads"].items():
                agg[h][0] += n
                agg[h][1] += k
        ids = [sum(r["identity"][0] for r in rs), sum(r["identity"][1] for r in rs)]
        agg["identity"] = ids
        rate[key] = {h: (k / n, n) for h, (n, k) in agg.items() if n}
    cohort = collections.defaultdict(list)
    for key, hs in rate.items():
        if key[0] == "everyayah":                   # the cohort is the professional reciters
            for h, (v, n) in hs.items():
                if n >= MIN_N:
                    cohort[h].append(v)

    profiles = []
    for (source, spk), rs in sorted(by.items()):
        hs = [r["haraka_s"] for r in rs if r["haraka_s"] and r.get("haraka_source") == "own"]
        h = statistics.median(hs) if hs else None
        cls = tempo_class(h)
        madd = collections.defaultdict(list)
        for r in rs:
            for rule, g, _st in r["madd"]:
                madd[rule].append(g)
        z = {hd: _robust_z(v, cohort[hd]) for hd, (v, n) in rate[(source, spk)].items()
             if n >= MIN_N and len(cohort[hd]) >= 5}
        ranked = sorted((v, k) for k, v in z.items() if v is not None)
        surahs = collections.defaultdict(list)
        for r in rs:
            surahs[r["surah"]].append(r)
        profiles.append({
            "source": source, "speaker": spk, "verses": len(rs),
            "tempo": {"haraka_s": round(h, 3) if h else None, "class": cls,
                      "learning_model": LEARNING_MODEL.get(cls) if cls else None,
                      "spread_s": round(_q(hs, 0.75) - _q(hs, 0.25), 3) if len(hs) >= 4 else None},
            "word_accuracy": round(statistics.mean(r["word_accuracy"] for r in rs if r["word_accuracy"] is not None), 4),
            "letter_accuracy": round(statistics.mean(r["letter_accuracy"] for r in rs if r["letter_accuracy"] is not None), 4),
            "lengths": {rule: {"n": len(v), "median_counts": round(statistics.median(v), 2),
                               "iqr": [round(_q(v, 0.25), 2), round(_q(v, 0.75), 2)]}
                        for rule, v in sorted(madd.items()) if len(v) >= 3},
            "characteristics": {hd: {"realised": round(v, 4), "n": n, "z": z.get(hd)}
                                for hd, (v, n) in sorted(rate[(source, spk)].items())},
            "strengths": [k for v, k in ranked[::-1][:3] if v > 0],
            "weaknesses": [k for v, k in ranked[:3] if v < 0],
            "self_correction": _self_correction(rs),
            "by_surah": {str(s): {"verses": len(v),
                                  "word_accuracy": round(statistics.mean(x["word_accuracy"] for x in v
                                                                         if x["word_accuracy"] is not None), 4),
                                  "haraka_s": round(statistics.median(x["haraka_s"] for x in v if x["haraka_s"]), 3)
                                  if any(x["haraka_s"] for x in v) else None}
                         for s, v in sorted(surahs.items())},
        })
    PROFILES.write_text(json.dumps({"reciters": profiles, "clips": len(ok),
                                    "errors": len(rows) - len(ok)}, ensure_ascii=False, indent=1))
    _to_store(ok, profiles)
    print(f"{len(profiles)} profiles from {len(ok)} verse records ({len(rows) - len(ok)} errors) -> {PROFILES}")
    print(f"{'reciter':42s} {'tempo':>6s} {'class':7s} {'learn':5s} {'words':>6s} {'letters':>7s}  strengths / weaknesses")
    for p in sorted(profiles, key=lambda p: -(p["tempo"]["haraka_s"] or 0)):
        t = p["tempo"]
        print(f"{p['speaker'][:42]:42s} {t['haraka_s'] or 0:6.3f} {t['class'] or '-':7s} "
              f"{'yes' if t['learning_model'] else 'no':5s} {p['word_accuracy']:6.3f} {p['letter_accuracy']:7.4f}"
              f"  +{','.join(p['strengths'])} / -{','.join(p['weaknesses'])}")


def _to_store(rows: list[dict], profiles: list[dict]) -> None:  # type: ignore[type-arg]
    """Two tables: every verse record, and every reciter's profile (the detail stays as JSON)."""
    from datastore.store import connect, run  # noqa: PLC0415
    con = connect()
    with run(con, "profile:reciters", "analysis"):
        con.execute("""CREATE TABLE IF NOT EXISTS recitation_verse (
            source VARCHAR, speaker VARCHAR, surah INTEGER, ayah INTEGER, haraka_s DOUBLE,
            word_accuracy DOUBLE, rule_accuracy DOUBLE, letter_accuracy DOUBLE,
            words INTEGER, faulted_words INTEGER, detail JSON)""")
        con.execute("""CREATE TABLE IF NOT EXISTS reciter_profile (
            source VARCHAR, speaker VARCHAR, verses INTEGER, haraka_s DOUBLE, tempo_class VARCHAR,
            learning_model BOOLEAN, word_accuracy DOUBLE, letter_accuracy DOUBLE,
            self_correction_rate DOUBLE, detail JSON)""")
        con.execute("DELETE FROM recitation_verse")
        con.execute("DELETE FROM reciter_profile")
        con.executemany("INSERT INTO recitation_verse VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        [(r["source"], r["speaker"], r["surah"], r["ayah"], r["haraka_s"],
                          r["word_accuracy"], r["rule_accuracy"], r["letter_accuracy"], r["words"],
                          r["faulted_words"], json.dumps({k: r[k] for k in ("madd", "heads", "identity",
                                                                            "fault_families")},
                                                         ensure_ascii=False)) for r in rows])
        con.executemany("INSERT INTO reciter_profile VALUES (?,?,?,?,?,?,?,?,?,?)",
                        [(p["source"], p["speaker"], p["verses"], p["tempo"]["haraka_s"],
                          p["tempo"]["class"], p["tempo"]["learning_model"], p["word_accuracy"],
                          p["letter_accuracy"], p["self_correction"]["rate"],
                          json.dumps(p, ensure_ascii=False)) for p in profiles])
    con.close()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "profile"
    if cmd == "clips":
        clips(*(int(a) for a in sys.argv[2:]))
    elif cmd == "add":
        add(*sys.argv[2:])
    else:
        profile()
