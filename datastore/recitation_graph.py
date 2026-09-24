#!/usr/bin/env python3
"""The recitation graph: reciters, letters, characteristics and what connects them, queryable in Cypher.

Built from the calculus of characteristics (substrate_library/julia/calculus.jl) and the reciter sets
(reciter_sets.jl). Nodes and relationships:

    (:Reciter {name, tier})              41 T300 reciters; tier = anchor / studio / imam / fast
    (:Letter {char})                     the 28 letters
    (:Characteristic {head, class})      the 22 head classes (ghunnah [مغن], qalqalah [مقلقل], ...)
    (:Feature {id, letter, context, head, overruled})
                                         one characteristic of one letter in one context
    (:Community {id})                    reciter neighbourhoods (label propagation on kNN)

    (Letter)-[:HAS]->(Characteristic)                the canonical formal context (the books)
    (Feature)-[:OF_LETTER]->(Letter)
    (Reciter)-[:REALISES {rate, graded}]->(Feature)  hit rate and graded (soft) realisation
    (Feature)-[:TRAVELS_WITH {pmi, verses}]->(Feature)
                                                     departures co-occurring in the same verse
    (Reciter)-[:NEAR {similarity}]->(Reciter)        k nearest neighbours
    (Reciter)-[:IN_COMMUNITY]->(Community)
    (Rule {name, stage, mean_pass}) and the rule layer (datastore/rule_layer.py, structures.jl):
    (Reciter)-[:KEEPS {pass_rate, n}]->(Rule)        each reciter's pass rate per rule
    (Rule)-[:PREREQUISITE_OF {violation, reverse}]->(Rule)
                                                     the knowledge space's Hasse diagram
    (Reciter)-[:READY_FOR]->(Rule)                   the reciter's outer fringe: what to learn next
    (Rule)-[:FAILS_WITH {pmi, verses}]->(Rule)       rule failures that come together in a verse
    (Measure)-[:DEPENDS {mi_bits}]->(Measure)        the Chow-Liu backbone over all measurements
    (Measure)-[:PC_LINK {dir, r}]->(Measure)         the PC skeleton, oriented at v-structures
    (Concept)-[:CAUSES {claim, mechanism, provenance, verdict, estimate, ci_low, ci_high, strata}]->(Concept)
                                                     the causal layer (substrate_library/julia/causal.jl):
                                                     mechanisms and contexts -> the acts they produce,
                                                     each edge tested within (letter, reciter) strata

It is written twice: into an embedded Kùzu database (Cypher, queried below and from the app), and as
Neo4j bulk-import CSVs plus a LOAD CSV script (`neo4j/`), so the same graph loads into Neo4j unchanged.

    .venv/bin/python -m datastore.recitation_graph build
    .venv/bin/python -m datastore.recitation_graph query "MATCH ... RETURN ..."
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "research_agency_lab/experiments/calculus/data"
GRAPH_DIR = ROOT / "research_agency_lab/experiments/calculus/graph"
KUZU = GRAPH_DIR / "recitation.kuzu"
NEO = GRAPH_DIR / "neo4j"
CAUSAL = ROOT / "research_agency_lab/experiments/calculus/causal_edges.json"

SCHEMA = [
    "CREATE NODE TABLE Reciter(name STRING, tier STRING, PRIMARY KEY(name))",
    "CREATE NODE TABLE Letter(ch STRING, PRIMARY KEY(ch))",
    "CREATE NODE TABLE Characteristic(id STRING, head STRING, cls STRING, PRIMARY KEY(id))",
    "CREATE NODE TABLE Feature(id STRING, letter STRING, context STRING, head STRING, overruled BOOLEAN, PRIMARY KEY(id))",
    "CREATE NODE TABLE Community(id INT64, PRIMARY KEY(id))",
    "CREATE NODE TABLE Concept(name STRING, PRIMARY KEY(name))",
    "CREATE NODE TABLE Rule(name STRING, stage INT64, mean_pass DOUBLE, PRIMARY KEY(name))",
    "CREATE NODE TABLE Measure(name STRING, PRIMARY KEY(name))",
    "CREATE REL TABLE HAS(FROM Letter TO Characteristic)",
    "CREATE REL TABLE OF_LETTER(FROM Feature TO Letter)",
    "CREATE REL TABLE REALISES(FROM Reciter TO Feature, rate DOUBLE, graded DOUBLE)",
    "CREATE REL TABLE TRAVELS_WITH(FROM Feature TO Feature, pmi DOUBLE, verses INT64)",
    "CREATE REL TABLE NEAR(FROM Reciter TO Reciter, similarity DOUBLE)",
    "CREATE REL TABLE IN_COMMUNITY(FROM Reciter TO Community)",
    "CREATE REL TABLE CAUSES(FROM Concept TO Concept, claim STRING, mechanism STRING, provenance STRING, "
    "verdict STRING, estimate DOUBLE, ci_low DOUBLE, ci_high DOUBLE, strata INT64)",
    "CREATE REL TABLE KEEPS(FROM Reciter TO Rule, pass_rate DOUBLE, n INT64)",
    "CREATE REL TABLE PREREQUISITE_OF(FROM Rule TO Rule, violation DOUBLE, reverse DOUBLE, n INT64)",
    "CREATE REL TABLE READY_FOR(FROM Reciter TO Rule)",
    "CREATE REL TABLE FAILS_WITH(FROM Rule TO Rule, pmi DOUBLE, verses INT64)",
    "CREATE REL TABLE DEPENDS(FROM Measure TO Measure, mi_bits DOUBLE)",
    "CREATE REL TABLE PC_LINK(FROM Measure TO Measure, dir STRING, r DOUBLE)",
]


def _tables() -> dict[str, list[list]]:  # type: ignore[type-arg]
    calc = json.loads((DATA / "calculus_T300.json").read_text())
    sets = json.loads((DATA / "reciter_sets_T300.json").read_text())
    overruled = {x["feature"] for x in sets["overruled_expectations"]}
    t: dict[str, list[list]] = {k: [] for k in ("Reciter", "Letter", "Characteristic", "Feature", "Community",  # type: ignore[type-arg]
                                                  "HAS", "OF_LETTER", "REALISES", "TRAVELS_WITH", "NEAR",
                                                  "IN_COMMUNITY", "Concept", "CAUSES", "Rule", "Measure",
                                                  "KEEPS", "PREREQUISITE_OF", "READY_FOR", "FAILS_WITH",
                                                  "DEPENDS", "PC_LINK")}
    t["Reciter"] = [[r, tier] for r, tier in zip(sets["reciters"], sets["tiers"])]
    t["Letter"] = [[c] for c in calc["letters"]]
    t["Characteristic"] = [[a, a.split("=")[0], a.split("=")[1]] for a in calc["attributes"]]
    for l, attrs in zip(calc["letters"], calc["formal_context"]):
        t["HAS"] += [[l, calc["attributes"][j - 1]] for j in attrs]
    for f in sets["features"]:
        letter, ctx, head = f.split(" ")
        t["Feature"].append([f, letter, ctx, head, f in overruled])
        t["OF_LETTER"].append([f, letter])
    for r, rates, graded in zip(sets["reciters"], sets["rates"], sets["graded"]):
        t["REALISES"] += [[r, f, x, g] for f, x, g in zip(sets["features"], rates, graded) if x is not None]
    t["TRAVELS_WITH"] = [[c["a"], c["b"], round(c["pmi"], 4), c["together"]] for c in sets["companions"]]
    S = len(sets["reciters"])
    knn = sets["knn"]                     # Julia writes a matrix flat, column-major: W[i, j] = knn[j*S + i]
    for i in range(S):
        for j in range(S):
            w = knn[j * S + i] if not isinstance(knn[0], list) else knn[i][j]
            if w and i != j:
                t["NEAR"].append([sets["reciters"][i], sets["reciters"][j], round(w, 4)])
    causal = CAUSAL.parent / "causal_edges.json"
    if causal.is_file():
        es = json.loads(causal.read_text())["edges"]
        t["Concept"] = [[n] for n in sorted({e["cause"] for e in es} | {e["effect"] for e in es})]
        t["CAUSES"] = [[e["cause"], e["effect"], e["claim"], e["mechanism"], e["provenance"], e["verdict"],
                        e["estimate"], *(e["ci95"] or [None, None]), e["strata"]] for e in es]
    rl_path = ROOT / "research_agency_lab/experiments/calculus/rule_layer_T300.json"
    st_path = ROOT / "research_agency_lab/experiments/calculus/structures_T300.json"
    if rl_path.is_file() and st_path.is_file():
        rl = json.loads(rl_path.read_text())
        st = json.loads(st_path.read_text())
        mean_pass = {x["rule"]: x["mean_pass"] for x in rl["hardest_rules"]}
        names = sorted(set(st["rules"]) | set(mean_pass))
        t["Rule"] = [[n, st["stages"].get(n, -1), mean_pass.get(n)] for n in names]
        for spk, d in rl["reciters"].items():
            t["KEEPS"] += [[spk, ru, v["pass_rate"], v["n"]] for ru, v in d["rules"].items() if ru in names]
        t["PREREQUISITE_OF"] = [[x["a"], x["b"], x["violation"], x["reverse"], x["n"]] for x in st["prerequisites"]]
        for spk, d in st["reciter_states"].items():
            t["READY_FOR"] += [[spk, ru] for ru in d["fringe"]]
        t["FAILS_WITH"] = [[x["a"], x["b"], x["pmi"], x["verses"]] for x in rl["fail_together"]
                           if x["a"] in names and x["b"] in names]
        t["Measure"] = [[n] for n in st["variables"]]
        t["DEPENDS"] = [[e["a"], e["b"], e["mi_bits"]] for e in st["chow_liu"]]
        t["PC_LINK"] = [[e["a"], e["b"], e["dir"], e["r"]] for e in st["pc"]]
    for k, c in enumerate(sets["communities"]):
        t["Community"].append([k])
        t["IN_COMMUNITY"] += [[m, k] for m in c["members"]]
    return t


def build() -> None:
    import kuzu
    t = _tables()
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    NEO.mkdir(parents=True, exist_ok=True)
    for name, rows in t.items():                    # one CSV per table: Kùzu COPY and Neo4j LOAD CSV
        with open(NEO / f"{name}.csv", "w", newline="") as f:
            csv.writer(f).writerows(rows)
    if KUZU.exists():
        shutil.rmtree(KUZU) if KUZU.is_dir() else KUZU.unlink()
    db = kuzu.Database(str(KUZU))
    con = kuzu.Connection(db)
    for q in SCHEMA:
        con.execute(q)
    for name in t:
        con.execute(f"COPY {name} FROM '{NEO / (name + '.csv')}' (header=false)")
    (NEO / "import.cypher").write_text(NEO4J_IMPORT)
    print("graph:", {k: len(v) for k, v in t.items()}, "->", KUZU)


def query(q: str) -> list[list]:  # type: ignore[type-arg]
    import kuzu
    con = kuzu.Connection(kuzu.Database(str(KUZU), read_only=True))
    res = con.execute(q)
    rows = []
    while res.has_next():
        rows.append(res.get_next())
    return rows


NEO4J_IMPORT = """// Load the recitation graph into Neo4j: copy neo4j/*.csv into the server's import directory.
LOAD CSV FROM 'file:///Reciter.csv' AS r CREATE (:Reciter {name: r[0], tier: r[1]});
LOAD CSV FROM 'file:///Letter.csv' AS r CREATE (:Letter {ch: r[0]});
LOAD CSV FROM 'file:///Characteristic.csv' AS r CREATE (:Characteristic {id: r[0], head: r[1], cls: r[2]});
LOAD CSV FROM 'file:///Feature.csv' AS r CREATE (:Feature {id: r[0], letter: r[1], context: r[2], head: r[3], overruled: r[4] = 'True'});
LOAD CSV FROM 'file:///Community.csv' AS r CREATE (:Community {id: toInteger(r[0])});
CREATE INDEX FOR (n:Reciter) ON (n.name); CREATE INDEX FOR (n:Letter) ON (n.ch);
CREATE INDEX FOR (n:Characteristic) ON (n.id); CREATE INDEX FOR (n:Feature) ON (n.id);
LOAD CSV FROM 'file:///HAS.csv' AS r MATCH (a:Letter {ch: r[0]}), (b:Characteristic {id: r[1]}) CREATE (a)-[:HAS]->(b);
LOAD CSV FROM 'file:///OF_LETTER.csv' AS r MATCH (a:Feature {id: r[0]}), (b:Letter {ch: r[1]}) CREATE (a)-[:OF_LETTER]->(b);
LOAD CSV FROM 'file:///REALISES.csv' AS r MATCH (a:Reciter {name: r[0]}), (b:Feature {id: r[1]})
  CREATE (a)-[:REALISES {rate: toFloat(r[2]), graded: toFloat(r[3])}]->(b);
LOAD CSV FROM 'file:///TRAVELS_WITH.csv' AS r MATCH (a:Feature {id: r[0]}), (b:Feature {id: r[1]})
  CREATE (a)-[:TRAVELS_WITH {pmi: toFloat(r[2]), verses: toInteger(r[3])}]->(b);
LOAD CSV FROM 'file:///NEAR.csv' AS r MATCH (a:Reciter {name: r[0]}), (b:Reciter {name: r[1]})
  CREATE (a)-[:NEAR {similarity: toFloat(r[2])}]->(b);
LOAD CSV FROM 'file:///IN_COMMUNITY.csv' AS r MATCH (a:Reciter {name: r[0]}), (b:Community {id: toInteger(r[1])})
  CREATE (a)-[:IN_COMMUNITY]->(b);
LOAD CSV FROM 'file:///Concept.csv' AS r CREATE (:Concept {name: r[0]});
LOAD CSV FROM 'file:///CAUSES.csv' AS r MATCH (a:Concept {name: r[0]}), (b:Concept {name: r[1]})
  CREATE (a)-[:CAUSES {claim: r[2], mechanism: r[3], provenance: r[4], verdict: r[5], estimate: toFloat(r[6]),
                       ci_low: toFloat(r[7]), ci_high: toFloat(r[8]), strata: toInteger(r[9])}]->(b);
"""


if __name__ == "__main__":
    if sys.argv[1:2] == ["build"]:
        build()
    elif sys.argv[1:2] == ["query"]:
        for row in query(sys.argv[2]):
            print(row)
