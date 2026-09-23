#!/usr/bin/env python3
"""Load the artifacts that already exist on disk into the persistent store (idempotent).

    .venv/bin/python -m datastore.ingest [all | keys | reciters | timings | graph | calibration | learners | engine | dump DIR TAG | ctc FILE TAG]

* keys        datasets/qaari_keys/build: verse_key, ayah_key, tier_verse
* reciters    build/reciters.json (EveryAyah, quran.com) + QuranMB speakers, with the grading ladder
* timings     build/timings_*.jsonl → word_timing
* graph       research_agency_lab/knowledge_map/graph.json → kg_node, kg_edge
* calibration experiments/calibration/*.json (+ the installed one) → calibration_version
* learners    learner_eval/results (QuranMB engine + muaalem rows, head-to-head) and gate metrics → test_result
* rules       qaari_keys/{textswap,ruleswap,sifat}_*.json counterfactual measurements -> test_result
* engine      benchmarks/results/**/runs_*.jsonl → engine_ayah, engine_diag (read by DuckDB directly)
* dump        a muaalem posterior dump directory → posterior_file (files stay where they are)
* ctc         julia ctc_run.jl output → unit_gop, deviation
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datastore.store import connect, run  # noqa: E402

KEYS = ROOT / "datasets/qaari_keys/build"
MUAALEM = "obadx/muaalem-model-v3_2"
# grading ladder (a tier for reporting, not a verdict): near-gold anchors, studio masters, fast / live
ANCHOR = {"Husary_128kbps", "Husary_Muallim_128kbps", "Husary_128kbps_Mujawwad"}
STUDIO = {"Minshawy_Murattal_128kbps", "Minshawy_Mujawwad_192kbps", "Abdul_Basit_Murattal_192kbps",
          "Abdul_Basit_Mujawwad_128kbps", "Hudhaify_128kbps", "Alafasy_128kbps", "Mohammad_al_Tablaway_128kbps",
          "Muhammad_Ayyoub_128kbps", "Mustafa_Ismail_48kbps", "mahmoud_ali_al_banna_32kbps"}
IMAMS = {"Yasser_Ad-Dussary_128kbps", "Nasser_Alqatami_128kbps", "Saood_ash-Shuraym_128kbps",
         "Abdurrahmaan_As-Sudais_192kbps", "Abdullaah_3awwaad_Al-Juhaynee_128kbps", "Salah_Al_Budair_128kbps",
         "Abdullah_Matroud_128kbps", "MaherAlMuaiqly128kbps"}


def ladder(folder: str) -> str:
    return "anchor" if folder in ANCHOR else "studio" if folder in STUDIO else "imam" if folder in IMAMS else "fast"


def keys(con) -> None:  # type: ignore[no-untyped-def]
    with run(con, "ingest:qaari_keys", "loader"):
        d = json.loads((KEYS / "ayah_keys.json").read_text())
        con.executemany("INSERT OR REPLACE INTO verse_key VALUES (?, ?)", [(k, k.split(":")[0]) for k in d["keys"]])
        rows = [(s, a, d["keys"][int(k)], n) for (s, a, *_), c in zip(d["ayahs"], d["counts"]) for k, n in c.items()]
        con.execute("DELETE FROM ayah_key")
        con.executemany("INSERT INTO ayah_key VALUES (?, ?, ?, ?)", rows)
        t = json.loads((KEYS / "tiers.json").read_text())
        con.execute("DELETE FROM tier_verse")
        con.executemany("INSERT INTO tier_verse VALUES (?, ?, ?, ?)",
                        [(x["name"], i, v["surah"], v["ayah"]) for x in t["tiers"] for i, v in enumerate(x["verses"])])
    print(f"keys: {len(d['keys'])} keys, {len(rows)} ayah-key rows, tiers {[x['name'] for x in t['tiers']]}")


def reciters(con) -> None:  # type: ignore[no-untyped-def]
    with run(con, "ingest:reciters", "loader"):
        c = json.loads((KEYS / "reciters.json").read_text())
        rows = [(f"everyayah:{r['folder']}", "everyayah", r["name"], r["style"], r["kbps"], ladder(str(r["folder"])),
                 json.dumps({"qurancom_id": r.get("qurancom_id")})) for r in c["everyayah"]]
        rows += [(f"qdc:{r['id']}", "qdc", r["name"], r.get("style"), None,
                  "anchor" if r["id"] in (6, 12) else "studio", json.dumps({"everyayah": r.get("everyayah")}))
                 for r in c["qurancom"]]
        rows += [(f"quranicaudio:{r['id']}", "quranicaudio", r["name"], None, None, "unrated",
                  json.dumps({"path": r.get("path"), "section": r.get("section")})) for r in c["quranicaudio"]]
        learners = {json.loads(line)["speaker"] for line in
                    open(ROOT / "research_agency_lab/experiments/learner_eval/results/quranmb_v2.jsonl")}
        rows += [(f"quranmb:{s}", "quranmb", s, None, None, "learner", "{}") for s in sorted(learners)]
        con.executemany("INSERT OR REPLACE INTO reciter VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
    print(f"reciters: {len(rows)}")


def timings(con) -> None:  # type: ignore[no-untyped-def]
    n = 0
    with run(con, "ingest:word_timing", "loader"):
        con.execute("DELETE FROM word_timing")
        for p in sorted(KEYS.glob("timings_*.jsonl")):
            rows = []
            for line in p.open():
                r = json.loads(line)
                rid = f"everyayah:{r['reciter']}" if r["source"] == "quran_align" else f"qdc:{r['reciter'][4:]}"
                rows += [(r["source"], rid, r["surah"], r["ayah"], w0, w1, ms0, ms1) for w0, w1, ms0, ms1 in r["words"]]
            con.executemany("INSERT INTO word_timing VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
            n += len(rows)
    print(f"word_timing: {n} rows")


def graph(con) -> None:  # type: ignore[no-untyped-def]
    g = json.loads((ROOT / "research_agency_lab/knowledge_map/graph.json").read_text())
    with run(con, "ingest:knowledge_graph", "loader", {"nodes": len(g["nodes"]), "edges": len(g["edges"])}):
        con.execute("DELETE FROM kg_node")
        con.execute("DELETE FROM kg_edge")
        con.executemany("INSERT OR REPLACE INTO kg_node VALUES (?, ?, ?, ?, ?)",
                        [(n["id"], n.get("type"), n.get("label"), n.get("status"), json.dumps(n, ensure_ascii=False))
                         for n in g["nodes"]])
        con.executemany("INSERT INTO kg_edge VALUES (?, ?, ?)",
                        [(e.get("from") or e.get("src"), e.get("to") or e.get("dst"), e.get("rel")) for e in g["edges"]])
    print(f"graph: {len(g['nodes'])} nodes, {len(g['edges'])} edges")


def calibration(con) -> None:  # type: ignore[no-untyped-def]
    installed = json.loads((ROOT / "app/data/calibration.json").read_text())
    metrics = json.loads((ROOT / "research_agency_lab/orchestrator/metrics/calibration.json").read_text())
    with run(con, "ingest:calibration", "loader"):
        for p in sorted((ROOT / "research_agency_lab/experiments/calibration").glob("calibration_v*.json")):
            b = json.loads(p.read_text())
            v = int(b.get("version") or p.stem.split("_v")[-1][0])
            is_inst = b.get("version") == installed.get("version") and "installed" not in p.stem
            con.execute("INSERT OR REPLACE INTO calibration_version VALUES (?, ?, ?, ?, ?)",
                        [v, str(p.relative_to(ROOT)), is_inst, json.dumps(b),
                         json.dumps(metrics) if metrics.get("calibration_version") == v else None])
    print("calibration versions:", con.execute("SELECT version, installed FROM calibration_version ORDER BY 1").fetchall())


def _results(con, run_id: str, test: str, scope: str, d: dict, prefix: str = "") -> int:  # type: ignore[no-untyped-def,type-arg]
    rows = []
    for k, v in d.items():
        if isinstance(v, dict):
            rows += [(run_id, test, scope, f"{prefix}{k}.{k2}", float(v2), None)
                     for k2, v2 in v.items() if isinstance(v2, (int, float)) and not isinstance(v2, bool)]
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            rows.append((run_id, test, scope, f"{prefix}{k}", float(v), None))
    con.execute("DELETE FROM test_result WHERE run_id = ? AND test = ? AND scope = ?", [run_id, test, scope])
    if rows:  # a metrics file with no numeric leaves (e.g. an empty gate) is not an error
        con.executemany("INSERT INTO test_result VALUES (?, ?, ?, ?, ?, ?)", rows)
    return len(rows)


def learners(con) -> None:  # type: ignore[no-untyped-def]
    R = ROOT / "research_agency_lab/experiments/learner_eval/results"
    n = 0
    with run(con, "test:quranmb_v2", "test", {"clips": 1642}):
        n += _results(con, "test:quranmb_v2", "learners_uncalibrated", "quranmb", json.loads((R / "learners_uncalibrated.json").read_text()))
        n += _results(con, "test:quranmb_v2", "head_to_head", "quranmb", json.loads((R / "head_to_head.json").read_text()))
    with run(con, "gates:current", "test"):
        for p in sorted((ROOT / "research_agency_lab/orchestrator/metrics").glob("*.json")):
            n += _results(con, "gates:current", f"gate:{p.stem}", "all", json.loads(p.read_text()))
    print(f"test_result: {n} rows")


def rules(con) -> None:  # type: ignore[no-untyped-def]
    """Counterfactual rule/letter/sifat measurements from the QaariKeys experiments."""
    K = ROOT / "research_agency_lab/experiments/qaari_keys"
    n = 0
    with run(con, "test:counterfactual", "test"):
        for stem, scope in (("textswap", "lahn"), ("ruleswap", "rules"), ("sifat", "sifat"),
                            ("tasawi", "tasawi"), ("sukoon", "sukoon"), ("distance", "distance"),
                            ("harakat", "harakat"), ("letter_coverage", "coverage")):
            for p in sorted(K.glob(f"{stem}_*.json")):
                tier = p.stem.split("_")[-1]
                d = json.loads(p.read_text())
                if stem == "textswap":
                    flat = {k: {m: v[m] for m in ("recall", "named_recall", "threshold", "swaps")}
                            for k, v in d.items() if isinstance(v, dict) and "recall" in v}
                elif stem == "ruleswap":
                    flat = {k: {m: v[m] for m in ("anchor_win_rate", "anchor_flag_rate",
                                                  "anchor_median_margin", "n_edits")
                                if v.get(m) is not None}
                            for k, v in d.get("families", {}).items()}
                elif stem == "sifat":
                    flat = {f"{lvl}.{cls}": {m: cv[m] for m in ("anchor_flag_rate", "threshold", "anchor_n")}
                            for lvl, classes in d.get("levels", {}).items() for cls, cv in classes.items()}
                elif stem == "tasawi":   # consistency of every held length, per class
                    flat = {cls: {m: cv[m] for m in ("anchor_median_counts", "anchor_cv", "cv_threshold",
                                                     "other_median_cv") if cv.get(m) is not None}
                            for cls, cv in d.get("classes", {}).items()}
                elif stem == "sukoon":   # rikhw/between/shadeed timing separation per reciter
                    flat = {r["spk"].split(":", 1)[-1]: {k: r[k] for k in
                            ("separation", "rikhw", "between", "shadeed", "haraka_s")}
                            for r in d.get("reciters", [])}
                    flat["_overall"] = {"anchor_separation": d.get("anchor_separation"),
                                        "other_separation": d.get("other_separation")}
                elif stem == "distance":  # rank of each reciter from the anchor centroid
                    flat = {r["reciter_id"].split(":", 1)[-1]: {"rank": r["rank"], "distance": r["distance"]}
                            for r in d.get("reciters", [])}
                elif stem == "harakat":   # vowel isochrony, ikhtilas and ishba per reciter
                    flat = {r["spk"].split(":", 1)[-1]: {k: r[k] for k in
                            ("isochrony", "ikhtilas_rate", "ishba_rate", "fatha", "damma", "kasra")
                            if r.get(k) is not None} for r in d.get("reciters", [])}
                else:                     # letter_coverage: is every expected unit actually recorded
                    flat = {r["spk"].split(":", 1)[-1]: {k: r[k] for k in
                            ("id_confirmed", "id_harakah", "id_madd", "id_consonant",
                             "attr_realised", "chars_per_unit", "haraka_s") if r.get(k) is not None}
                            for r in d.get("reciters", [])}
                n += _results(con, "test:counterfactual", f"{stem}:{tier}", scope, flat)
    print(f"counterfactual test_result: {n} rows")


def engine(con) -> None:  # type: ignore[no-untyped-def]
    files = sorted(glob.glob(str(ROOT / "benchmarks/results/**/runs_*.jsonl"), recursive=True)) + \
        sorted(glob.glob(str(ROOT / "benchmarks/results/local/*.jsonl")))
    files = sorted(set(files))
    with run(con, "engine:benchmark_rows", "loader", {"files": len(files)}):
        con.execute("DELETE FROM engine_ayah WHERE run_id = 'engine:benchmark_rows'")
        con.execute("DELETE FROM engine_diag WHERE run_id = 'engine:benchmark_rows'")
        src = f"read_json_auto({files!r}, format='newline_delimited', union_by_name=true, maximum_object_size=67108864)"
        con.execute(f"""INSERT INTO engine_ayah SELECT 'engine:benchmark_rows', 'everyayah:' || reciter, surah, ayah,
                        perfection, sifaat, haraka_ms, to_json(environment) FROM {src}""")
        con.execute(f"""INSERT INTO engine_diag
            SELECT 'engine:benchmark_rows', 'everyayah:' || r.reciter, r.surah, r.ayah, d.rule_type, d.key, d.detail,
                   d.word, d.letter, d.status, d.score, d.location.start_ms, d.location.end_ms, to_json(d.metrics)
            FROM (SELECT reciter, surah, ayah, unnest(diagnostics) AS d FROM {src}) r""")
    print("engine:", con.execute("SELECT count(*) FROM engine_ayah").fetchone()[0], "ayah rows,",
          con.execute("SELECT count(*) FROM engine_diag").fetchone()[0], "diagnostics")


def dump(con, directory: str, tag: str) -> None:  # type: ignore[no-untyped-def]
    d = Path(directory).resolve()
    rows = []
    for line in (d / "index.jsonl").open():
        r = json.loads(line)
        if "file" not in r:
            continue
        rows.append((r["id"], MUAALEM, str(d), r["file"], r["frames"], r.get("duration_s"), r.get("ref_ph"),
                     r.get("word_ph"), tag))
    with run(con, tag, "compute", {"dump_dir": str(d), "clips": len(rows)}):
        con.executemany("INSERT OR REPLACE INTO posterior_file VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    print(f"posterior_file: {len(rows)} clips from {d}")


def ctc(con, path: str, tag: str) -> None:  # type: ignore[no-untyped-def]
    units, devs = [], []
    for line in open(path):
        r = json.loads(line)
        units += [(r["id"], MUAALEM, i, u[0], u[1], u[2], u[3], u[4], u[5], tag) for i, u in enumerate(r["units"])]
        devs += [(r["id"], MUAALEM, x["kind"], x["ref_from"], x["ref_to"], x["at"], tag) for x in r.get("deviations", [])]
    with run(con, tag, "compute", {"file": path}):
        con.execute("DELETE FROM unit_gop WHERE run_id = ?", [tag])
        con.execute("DELETE FROM deviation WHERE run_id = ?", [tag])
        con.executemany("INSERT INTO unit_gop VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", units)
        con.executemany("INSERT INTO deviation VALUES (?, ?, ?, ?, ?, ?, ?)", devs)
    print(f"unit_gop: {len(units)}, deviation: {len(devs)} ({tag})")


def main(argv: list[str]) -> int:
    con = connect()
    what = argv[0] if argv else "all"
    if what == "dump":
        dump(con, argv[1], argv[2])
    elif what == "ctc":
        ctc(con, argv[1], argv[2])
    else:
        steps = [keys, reciters, timings, graph, calibration, learners, engine, rules]
        for f in steps if what == "all" else [f for f in steps if f.__name__ == what]:
            f(con)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
