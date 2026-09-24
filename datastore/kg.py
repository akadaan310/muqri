"""The Recitation Knowledge Graph (Neo4j + GDS): every dataset we hold, as connected layers.

    .venv/bin/python -m datastore.kg build [layer ...]     # all layers, or the named ones
    .venv/bin/python -m datastore.kg stats

Layers (the design and the questions each answers are in GRAPH.md):
  mushaf        (:Surah)-[:HAS]->(:Ayah)-[:HAS]->(:Word) with (:Word)-[:NEXT]->(:Word) through the
                whole text, (:Word)-[:CARRIES {n}]->(:Rule) for every located rule instance
  phonology     (:Sound) (the 42 symbols of the phonetic script), (:Letter)-[:HAS_SIFAH]->(:Sifah)
                (the 17 classical characteristics), (:Letter)-[:ARTICULATED_AT]->(:Makhraj)-[:IN]->(:Region),
                (:Sound)-[:PRECEDES {count, pmi}]->(:Sound) over all 6,236 ayahs
  calculus      the Sprint 3 graph: (:Reciter), (:Letter), (:Characteristic), (:Feature), (:Community),
                (:Rule), (:Measure), (:Concept) and REALISES / TRAVELS_WITH / NEAR / KEEPS /
                PREREQUISITE_OF / READY_FOR / FAILS_WITH / DEPENDS / PC_LINK / CAUSES
  performance   (:Reciter)-[:RECITED]->(:Performance {accuracies, unit})-[:OF]->(:Ayah) for 11,995 verses;
                (:Reciter)-[:HOLDS {level, stretch, n}]->(:Rule) -- each master's own stretch
  timing        (:RuleLevel {rule, level, a, b, sd, sd_low})-[:OF]->(:Rule),
                (:RuleLevel)-[:MULTIPLE_OF {ratio, q25, q75}]->(:RuleLevel {madd_tabii 2})
  skills        (:Rule)-[:COVARIES {partial_r}]->(:Rule) from the cohort model's precision matrix
  sessions      (:Exercise)-[:ON]->(:Ayah); (:Take {take, stamp, caught, false_alarms})-[:OF]->(:Exercise);
                (:ScriptedMistake {do})-[:AT]->(:Word), (:Take)-[:CAUGHT|MISSED]->(:ScriptedMistake)
  ra            (:RaContext {name, rule, reference_heavy, heard_heavy, margin_median, finding}) from ra.jl;
                (:Reciter)-[:SAID_RA {ayah, word, expected, heard, margin}]->(:RaContext) per instance
  (reliability  loaded by datastore.graph load-blindspots: (:Context), (:Check), FAILS)
"""

from __future__ import annotations

import csv
import glob
import gzip
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datastore.graph import _batches, driver

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "research_agency_lab/experiments"
CALC = EXP / "calculus/graph/neo4j"


def _write(q: str, rows: list[dict[str, Any]], n: int = 5000) -> None:
    with driver() as d, d.session() as s:
        for b in _batches(rows, n):
            s.run(q, rows=b)


def _exec(*qs: str) -> None:
    with driver() as d, d.session() as s:
        for q in qs:
            s.run(q)


# ---------------------------------------------------------------------------------------- mushaf
def mushaf() -> str:
    from quran_transcript import Aya
    from app.verse_detect import normalise
    _exec("CREATE CONSTRAINT surah_n IF NOT EXISTS FOR (s:Surah) REQUIRE s.n IS UNIQUE",
          "CREATE CONSTRAINT ayah_ref IF NOT EXISTS FOR (a:Ayah) REQUIRE a.ref IS UNIQUE",
          "CREATE CONSTRAINT qword_ref IF NOT EXISTS FOR (w:QWord) REQUIRE w.ref IS UNIQUE",
          "CREATE CONSTRAINT rule_name IF NOT EXISTS FOR (r:Rule) REQUIRE r.name IS UNIQUE")
    ayahs, words, x = [], [], Aya(1, 1)
    for _ in range(6236):
        g = x.get()
        ws = g.uthmani.split()
        ayahs.append({"ref": f"{g.sura_idx}:{g.aya_idx}", "surah": g.sura_idx, "ayah": g.aya_idx, "text": g.uthmani,
                      "words": len(ws)})
        for i, w in enumerate(ws):
            words.append({"ref": f"{g.sura_idx}:{g.aya_idx}:{i}", "ayah": f"{g.sura_idx}:{g.aya_idx}", "i": i,
                          "text": w, "bare": normalise(w)})
        x = x.step(1)
    _write("""UNWIND $rows AS r MERGE (s:Surah {n: r.surah})
              MERGE (a:Ayah {ref: r.ref}) SET a.surah = r.surah, a.ayah = r.ayah, a.text = r.text, a.words = r.words
              MERGE (s)-[:HAS]->(a)""", ayahs)
    _write("""UNWIND $rows AS r MATCH (a:Ayah {ref: r.ayah})
              MERGE (w:QWord {ref: r.ref}) SET w.i = r.i, w.text = r.text, w.bare = r.bare
              MERGE (a)-[:HAS]->(w)""", words)
    nxt = [{"a": words[i]["ref"], "b": words[i + 1]["ref"]} for i in range(len(words) - 1)]
    _write("UNWIND $rows AS r MATCH (a:QWord {ref: r.a}), (b:QWord {ref: r.b}) MERGE (a)-[:NEXT]->(b)", nxt)
    carries: Counter[tuple[str, str]] = Counter()
    with gzip.open(EXP / "quran/inventory.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            a = json.loads(line)
            for rule, w, _c in a.get("rules", []):
                carries[(f"{a['surah']}:{a['ayah']}:{w}", rule)] += 1
    _write("""UNWIND $rows AS r MATCH (w:QWord {ref: r.w}) MERGE (k:Rule {name: r.rule})
              MERGE (w)-[c:CARRIES]->(k) SET c.n = r.n""", [{"w": w, "rule": r, "n": n} for (w, r), n in carries.items()])
    return f"{len(ayahs)} ayahs, {len(words)} words, {len(carries)} word-rule links"


# ------------------------------------------------------------------------------------- phonology
MAKHRAJ = {  # letter -> (point, region), after the classical 17 makharij
    "ا": ("the chest cavity", "al-jawf"), "و": ("the lips", "ash-shafatan"), "ي": ("tongue middle", "al-lisan"),
    "ء": ("throat, deepest", "al-halq"), "ه": ("throat, deepest", "al-halq"), "ع": ("throat, middle", "al-halq"),
    "ح": ("throat, middle", "al-halq"), "غ": ("throat, nearest", "al-halq"), "خ": ("throat, nearest", "al-halq"),
    "ق": ("tongue root, soft palate", "al-lisan"), "ك": ("tongue back, hard palate", "al-lisan"),
    "ج": ("tongue middle", "al-lisan"), "ش": ("tongue middle", "al-lisan"), "ض": ("tongue side, molars", "al-lisan"),
    "ل": ("tongue edge, front", "al-lisan"), "ن": ("tongue tip, gum", "al-lisan"), "ر": ("tongue tip and back", "al-lisan"),
    "ط": ("tongue tip, incisor roots", "al-lisan"), "د": ("tongue tip, incisor roots", "al-lisan"),
    "ت": ("tongue tip, incisor roots", "al-lisan"), "ص": ("tongue tip, lower incisors", "al-lisan"),
    "ز": ("tongue tip, lower incisors", "al-lisan"), "س": ("tongue tip, lower incisors", "al-lisan"),
    "ظ": ("tongue tip, incisor edges", "al-lisan"), "ذ": ("tongue tip, incisor edges", "al-lisan"),
    "ث": ("tongue tip, incisor edges", "al-lisan"), "ف": ("lower lip, upper incisors", "ash-shafatan"),
    "ب": ("the lips", "ash-shafatan"), "م": ("the lips", "ash-shafatan"),
}


def phonology() -> str:
    from app.letter_matrix import IDHLAQ, ISTILA, ITBAQ, HAMS, SHIDDAH, SINGLES, TAWASSUT
    _exec("CREATE CONSTRAINT sifah_name IF NOT EXISTS FOR (s:Sifah) REQUIRE s.name IS UNIQUE",
          "CREATE CONSTRAINT makhraj_name IF NOT EXISTS FOR (m:Makhraj) REQUIRE m.name IS UNIQUE",
          "CREATE CONSTRAINT letter_ch IF NOT EXISTS FOR (l:Letter) REQUIRE l.ch IS UNIQUE")
    has = []
    for c in MAKHRAJ:
        if c == "ا":
            continue
        own = ["hams" if c in HAMS else "jahr",
               "shiddah" if c in SHIDDAH else ("tawassut" if c in TAWASSUT else "rakhawah"),
               "isti'la" if c in ISTILA else "istifal", "itbaq" if c in ITBAQ else "infitah",
               "idhlaq" if c in IDHLAQ else "ismat"] + [n for n, m in SINGLES.items() if c in m]
        has += [{"c": c, "s": s} for s in own]
    _write("""UNWIND $rows AS r MERGE (l:Letter {ch: r.c}) MERGE (s:Sifah {name: r.s}) MERGE (l)-[:HAS_SIFAH]->(s)""", has)
    _write("""UNWIND $rows AS r MERGE (l:Letter {ch: r.c}) MERGE (m:Makhraj {name: r.p}) MERGE (g:Region {name: r.g})
              MERGE (l)-[:ARTICULATED_AT]->(m) MERGE (m)-[:IN]->(g)""",
           [{"c": c, "p": p, "g": g} for c, (p, g) in MAKHRAJ.items()])
    # phonotactics: which sound follows which, over the whole Quran (collapsed runs), with PMI
    keys = [l.strip().split(":", 1)[1] for l in open(ROOT / "datasets/qaari_keys/build/synth/diphone_keys.txt")]
    pair: Counter[str] = Counter()
    for l in open(ROOT / "datasets/qaari_keys/build/synth/ayah_diphones.tsv"):
        _a, k, v = l.split("\t")
        pair[keys[int(k) - 1]] += int(v)
    left: Counter[str] = Counter()
    right: Counter[str] = Counter()
    for p, n in pair.items():
        left[p[0]] += n
        right[p[1]] += n
    tot = sum(pair.values())
    rows = [{"a": p[0], "b": p[1], "n": n, "pmi": round(math.log2(n * tot / (left[p[0]] * right[p[1]])), 4)}
            for p, n in pair.items() if len(p) == 2]
    _write("""UNWIND $rows AS r MERGE (a:Sound {symbol: r.a}) MERGE (b:Sound {symbol: r.b})
              MERGE (a)-[p:PRECEDES]->(b) SET p.count = r.n, p.pmi = r.pmi""", rows)
    _write("UNWIND $rows AS r MATCH (s:Sound {symbol: r.c}), (l:Letter {ch: r.c}) MERGE (s)-[:IS_LETTER]->(l)",
           [{"c": c} for c in MAKHRAJ])
    return f"{len(has)} letter-sifah, {len(MAKHRAJ)} makharij, {len(rows)} sound transitions"


# --------------------------------------------------------------------------------------- calculus
def _csv(name: str) -> list[list[str]]:
    return list(csv.reader(open(CALC / f"{name}.csv", encoding="utf-8")))


def calculus() -> str:
    _exec("CREATE CONSTRAINT reciter_name IF NOT EXISTS FOR (r:Reciter) REQUIRE r.name IS UNIQUE",
          "CREATE CONSTRAINT feature_id IF NOT EXISTS FOR (f:Feature) REQUIRE f.id IS UNIQUE",
          "CREATE CONSTRAINT charac_id IF NOT EXISTS FOR (c:Characteristic) REQUIRE c.id IS UNIQUE",
          "CREATE CONSTRAINT measure_name IF NOT EXISTS FOR (m:Measure) REQUIRE m.name IS UNIQUE",
          "CREATE CONSTRAINT concept_name IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE")
    W = _write
    W("UNWIND $rows AS r MERGE (x:Reciter {name: r[0]}) SET x.tier = r[1]", _csv("Reciter"))
    W("UNWIND $rows AS r MERGE (x:Letter {ch: r[0]})", _csv("Letter"))
    W("UNWIND $rows AS r MERGE (x:Characteristic {id: r[0]}) SET x.head = r[1], x.cls = r[2]", _csv("Characteristic"))
    W("UNWIND $rows AS r MERGE (x:Feature {id: r[0]}) SET x.letter = r[1], x.context = r[2], x.head = r[3], "
      "x.overruled = r[4] = 'True'", _csv("Feature"))
    W("UNWIND $rows AS r MERGE (x:Community {id: toInteger(r[0])})", _csv("Community"))
    W("UNWIND $rows AS r MERGE (x:Rule {name: r[0]}) SET x.stage = toInteger(r[1]), x.mean_pass = toFloat(r[2])", _csv("Rule"))
    W("UNWIND $rows AS r MERGE (x:Measure {name: r[0]})", _csv("Measure"))
    W("UNWIND $rows AS r MERGE (x:Concept {name: r[0]})", _csv("Concept"))
    W("UNWIND $rows AS r MATCH (a:Letter {ch: r[0]}), (b:Characteristic {id: r[1]}) MERGE (a)-[:HAS]->(b)", _csv("HAS"))
    W("UNWIND $rows AS r MATCH (a:Feature {id: r[0]}), (b:Letter {ch: r[1]}) MERGE (a)-[:OF_LETTER]->(b)", _csv("OF_LETTER"))
    W("UNWIND $rows AS r MATCH (a:Reciter {name: r[0]}), (b:Feature {id: r[1]}) "
      "MERGE (a)-[x:REALISES]->(b) SET x.rate = toFloat(r[2]), x.graded = toFloat(r[3])", _csv("REALISES"), 20000)
    W("UNWIND $rows AS r MATCH (a:Feature {id: r[0]}), (b:Feature {id: r[1]}) "
      "MERGE (a)-[x:TRAVELS_WITH]->(b) SET x.pmi = toFloat(r[2]), x.verses = toInteger(r[3])", _csv("TRAVELS_WITH"))
    W("UNWIND $rows AS r MATCH (a:Reciter {name: r[0]}), (b:Reciter {name: r[1]}) "
      "MERGE (a)-[x:NEAR]->(b) SET x.similarity = toFloat(r[2])", _csv("NEAR"))
    W("UNWIND $rows AS r MATCH (a:Reciter {name: r[0]}), (b:Community {id: toInteger(r[1])}) MERGE (a)-[:IN_COMMUNITY]->(b)",
      _csv("IN_COMMUNITY"))
    W("UNWIND $rows AS r MATCH (a:Reciter {name: r[0]}), (b:Rule {name: r[1]}) "
      "MERGE (a)-[x:KEEPS]->(b) SET x.pass_rate = toFloat(r[2]), x.n = toInteger(r[3])", _csv("KEEPS"))
    W("UNWIND $rows AS r MATCH (a:Rule {name: r[0]}), (b:Rule {name: r[1]}) "
      "MERGE (a)-[x:PREREQUISITE_OF]->(b) SET x.violation = toFloat(r[2]), x.reverse = toFloat(r[3]), x.n = toInteger(r[4])",
      _csv("PREREQUISITE_OF"))
    W("UNWIND $rows AS r MATCH (a:Reciter {name: r[0]}), (b:Rule {name: r[1]}) MERGE (a)-[:READY_FOR]->(b)", _csv("READY_FOR"))
    W("UNWIND $rows AS r MATCH (a:Rule {name: r[0]}), (b:Rule {name: r[1]}) "
      "MERGE (a)-[x:FAILS_WITH]->(b) SET x.pmi = toFloat(r[2]), x.verses = toInteger(r[3])", _csv("FAILS_WITH"))
    W("UNWIND $rows AS r MATCH (a:Measure {name: r[0]}), (b:Measure {name: r[1]}) "
      "MERGE (a)-[x:DEPENDS]->(b) SET x.mi_bits = toFloat(r[2])", _csv("DEPENDS"))
    W("UNWIND $rows AS r MATCH (a:Measure {name: r[0]}), (b:Measure {name: r[1]}) "
      "MERGE (a)-[x:PC_LINK]->(b) SET x.dir = r[2], x.r = toFloat(r[3])", _csv("PC_LINK"))
    W("UNWIND $rows AS r MATCH (a:Concept {name: r[0]}), (b:Concept {name: r[1]}) "
      "MERGE (a)-[x:CAUSES]->(b) SET x.claim = r[2], x.mechanism = r[3], x.provenance = r[4], x.verdict = r[5], "
      "x.estimate = toFloat(r[6]), x.ci_low = toFloat(r[7]), x.ci_high = toFloat(r[8]), x.strata = toInteger(r[9])", _csv("CAUSES"))
    return "the Sprint 3 calculus graph"


# ------------------------------------------------------------------------------------ performance
def performance() -> str:
    _exec("CREATE CONSTRAINT perf_id IF NOT EXISTS FOR (p:Performance) REQUIRE p.id IS UNIQUE")
    rows = []
    with gzip.open(EXP / "profiles/clips.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if "error" in r:
                continue
            rows.append({"id": f"{r['speaker']}|{r['surah']}:{r['ayah']}", "speaker": r["speaker"], "ayah": f"{r['surah']}:{r['ayah']}",
                         "word_acc": r.get("word_accuracy"), "rule_acc": r.get("rule_accuracy"),
                         "letter_acc": r.get("letter_accuracy"), "haraka_s": r.get("haraka_s"),
                         "faulted_words": r.get("faulted_words"), "words": r.get("words")})
    _write("""UNWIND $rows AS r MERGE (x:Reciter {name: r.speaker}) MATCH (a:Ayah {ref: r.ayah})
              MERGE (p:Performance {id: r.id}) SET p.word_accuracy = r.word_acc, p.rule_accuracy = r.rule_acc,
                  p.letter_accuracy = r.letter_acc, p.haraka_s = r.haraka_s, p.faulted_words = r.faulted_words, p.words = r.words
              MERGE (x)-[:RECITED]->(p) MERGE (p)-[:OF]->(a)""", rows)
    # each master's own stretch per rule (the timing calculus' raw material, per reciter)
    st: dict[tuple[str, str], list[float]] = defaultdict(list)
    with gzip.open(EXP / "timing/stretch_T300.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if "error" in r or len(r["plain"]) < 3:
                continue
            u = math.exp(sum(math.log(d) for _t, d in r["plain"]) / len(r["plain"]))
            for s in r["stretch"]:
                st[(r["speaker"], s["rule"])].append(s["seconds"] / u)
    holds = []
    for (spk, rule), xs in st.items():
        xs.sort()
        holds.append({"spk": spk, "rule": rule, "n": len(xs), "median": round(xs[len(xs) // 2], 3),
                      "q25": round(xs[len(xs) // 4], 3), "q75": round(xs[3 * len(xs) // 4], 3)})
    _write("""UNWIND $rows AS r MATCH (x:Reciter {name: r.spk}) MERGE (k:Rule {name: r.rule})
              MERGE (x)-[h:HOLDS]->(k) SET h.stretch_median = r.median, h.stretch_q25 = r.q25, h.stretch_q75 = r.q75, h.n = r.n""",
           holds)
    return f"{len(rows)} performances, {len(holds)} reciter-rule stretches"


# ------------------------------------------------------------------------------------------ timing
def timing() -> str:
    m = json.loads((EXP / "timing/stretch_model.json").read_text())
    rows = [{"key": f"{t['rule']}|{t['level']}", "rule": t["rule"], "level": t["level"], "a": t["a"], "b": t["b"],
             "sd": t["sd"], "sd_low": t.get("sd_low"), "n": t["n"], "at_u0": round(math.exp(t["a"]), 3)} for t in m["tempo_norms"]]
    _exec("CREATE CONSTRAINT rulelevel IF NOT EXISTS FOR (x:RuleLevel) REQUIRE x.key IS UNIQUE")
    _write("""UNWIND $rows AS r MERGE (k:Rule {name: r.rule}) MERGE (x:RuleLevel {key: r.key})
              SET x.rule = r.rule, x.level = r.level, x.a = r.a, x.b = r.b, x.sd = r.sd, x.sd_low = r.sd_low, x.n = r.n,
                  x.stretch_at_u0 = r.at_u0, x.u0 = $u0
              MERGE (x)-[:OF]->(k)""".replace("$u0", str(m["U0"])), rows)
    tab = next(t for t in m["tempo_norms"] if t["rule"] == "madd_tabii")
    mult = [{"key": r["key"], "ratio": round(math.exp(r["a"] - tab["a"]), 3)} for r in rows if r["rule"] != "madd_tabii"]
    _write("""UNWIND $rows AS r MATCH (x:RuleLevel {key: r.key}), (t:RuleLevel {key: 'madd_tabii|2.0'})
              MERGE (x)-[m:MULTIPLE_OF]->(t) SET m.ratio_at_u0 = r.ratio""", mult)
    return f"{len(rows)} rule levels"


# ------------------------------------------------------------------------------------------ skills
def skills() -> str:
    import numpy as np
    m = json.loads((EXP / "quran/cohort_model.json").read_text())
    S = np.asarray(m["sigma"])
    P = np.linalg.inv(S)
    d = np.sqrt(np.diag(P))
    pr = -P / np.outer(d, d)
    names = m["skills"]
    rows = [{"a": names[i], "b": names[j], "r": round(float(pr[i, j]), 4)}
            for i in range(len(names)) for j in range(i + 1, len(names)) if abs(pr[i, j]) >= 0.1]
    _write("""UNWIND $rows AS r MERGE (a:Skill {name: r.a}) MERGE (b:Skill {name: r.b})
              MERGE (a)-[c:COVARIES]-(b) SET c.partial_r = r.r""", rows)
    _write("UNWIND $rows AS r MATCH (s:Skill {name: r.n}), (k:Rule {name: r.n}) MERGE (s)-[:IS_RULE]->(k)",
           [{"n": n} for n in names])
    return f"{len(names)} skills, {len(rows)} partial-correlation links (|r| >= 0.1)"


# ---------------------------------------------------------------------------------------- sessions
def sessions() -> str:
    from app.sessions import exercises
    ex = exercises()
    _exec("CREATE CONSTRAINT exercise_id IF NOT EXISTS FOR (e:Exercise) REQUIRE e.id IS UNIQUE",
          "CREATE CONSTRAINT take_id IF NOT EXISTS FOR (t:Take) REQUIRE t.id IS UNIQUE",
          "CREATE CONSTRAINT scripted_id IF NOT EXISTS FOR (m:ScriptedMistake) REQUIRE m.id IS UNIQUE")
    erows, mrows = [], []
    for e in ex.values():
        erows.append({"id": e.id, "title": e.title, "goal": e.goal, "wajh": e.wajh,
                      "ayahs": [f"{e.surah}:{a}" for a in range(e.ayahs[0], e.ayahs[1] + 1)]})
        for i, mk in enumerate(e.mistakes):
            mrows.append({"id": f"{e.id}:m{i + 1}", "ex": e.id, "do": mk.do, "word": f"{e.surah}:{mk.ayah}:{mk.word}"})
    _write("""UNWIND $rows AS r MERGE (e:Exercise {id: r.id}) SET e.title = r.title, e.goal = r.goal, e.wajh = r.wajh
              WITH e, r UNWIND r.ayahs AS ay MATCH (a:Ayah {ref: ay}) MERGE (e)-[:ON]->(a)""", erows)
    _write("""UNWIND $rows AS r MATCH (e:Exercise {id: r.ex}) MERGE (m:ScriptedMistake {id: r.id}) SET m.do = r.do
              MERGE (e)-[:SCRIPTS]->(m) WITH m, r MATCH (w:QWord {ref: r.word}) MERGE (m)-[:AT]->(w)""", mrows)
    trows, crows = [], []
    for d in sorted(glob.glob(str(EXP / "session_recordings/r*"))):
        eid = Path(d).name
        for take in ("A", "B"):
            for f in sorted(glob.glob(f"{d}/{take}/*.score.json")):
                c = json.loads(Path(f).read_text())
                tid = f"{eid}:{take}:{Path(f).name[:15]}"
                trows.append({"id": tid, "ex": eid, "take": take, "stamp": Path(f).name[:15],
                              "caught": c.get("caught"), "met": c.get("met"), "false_alarms": len(c["false_alarms"]),
                              "unit_s": (c.get("tempo") or {}).get("seconds_per_count")})
                for i, m in enumerate(c.get("mistakes") or []):
                    crows.append({"take": tid, "m": f"{eid}:m{i + 1}", "verdict": m["verdict"],
                                  "evidence": "; ".join(m["evidence"] or m["engine_failing"])[:500]})
    _write("""UNWIND $rows AS r MATCH (e:Exercise {id: r.ex}) MERGE (t:Take {id: r.id})
              SET t.take = r.take, t.stamp = r.stamp, t.caught = r.caught, t.met = r.met, t.false_alarms = r.false_alarms,
                  t.unit_s = r.unit_s
              MERGE (t)-[:OF]->(e) MERGE (x:Reciter {name: 'certified_session_reciter'}) SET x.tier = 'certified'
              MERGE (x)-[:RECORDED]->(t)""", trows)
    _write("""UNWIND $rows AS r MATCH (t:Take {id: r.take}), (m:ScriptedMistake {id: r.m})
              MERGE (t)-[v:JUDGED]->(m) SET v.verdict = r.verdict, v.evidence = r.evidence""", crows)
    return f"{len(erows)} exercises, {len(mrows)} scripted mistakes, {len(trows)} takes"


# ---------------------------------------------------------------------------------------------- ra
def ra() -> str:
    """Every ra' the masters said, by the classical context it falls in (ra.jl names the context; the
    instances are re-classified here by the same calculus's output order)."""
    calc = json.loads((EXP / "quran/ra_calculus.json").read_text())
    _exec("CREATE CONSTRAINT racontext IF NOT EXISTS FOR (x:RaContext) REQUIRE x.name IS UNIQUE",
          "MATCH (x:RaContext) DETACH DELETE x")
    _write("""UNWIND $rows AS r MERGE (x:RaContext {name: r.context})
              SET x.rule = r.rule, x.n = r.n, x.reference_heavy = r.reference_heavy, x.heard_heavy = r.heard_heavy,
                  x.margin_median = r.margin_median, x.finding = r.finding, x.examples = r.examples""", calc["contexts"])
    inst = [json.loads(l) for l in (EXP / "quran/ra_instances.jsonl").read_text().splitlines()]
    ctx = [json.loads(l) for l in (EXP / "quran/ra_contexts.jsonl").read_text().splitlines()]
    rows = [{"speaker": r["speaker"], "ctx": c, "ayah": f"{r['surah']}:{r['ayah']}", "word": r["word"],
             "expected": r["expected"], "heard": r["observed"], "margin": r["margin"], "final": r["ayah_final"]}
            for r, c in zip((r for r in inst if "none" not in r), ctx)]
    _write("""UNWIND $rows AS r MERGE (s:Reciter {name: r.speaker}) WITH s, r MATCH (x:RaContext {name: r.ctx})
              CREATE (s)-[:SAID_RA {ayah: r.ayah, word: r.word, expected: r.expected, heard: r.heard, margin: r.margin,
                                     ayah_final: r.final}]->(x)""", rows)
    return f"{len(calc['contexts'])} ra' contexts, {len(rows)} ra' instances"


LAYERS = {"mushaf": mushaf, "phonology": phonology, "calculus": calculus, "performance": performance,
          "timing": timing, "skills": skills, "sessions": sessions, "ra": ra}


def stats() -> list[dict[str, Any]]:
    with driver() as d, d.session() as s:
        nodes = s.run("MATCH (n) UNWIND labels(n) AS l RETURN l AS label, count(*) AS n ORDER BY n DESC").data()
        rels = s.run("MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS n ORDER BY n DESC").data()
    return nodes + rels


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        for name in (sys.argv[2:] or list(LAYERS)):
            print(name, "->", LAYERS[name](), flush=True)
    elif cmd == "stats":
        for r in stats():
            print(r)
