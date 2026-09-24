"""The knowledge graph in Neo4j (with Graph Data Science): where the acoustic model can be trusted,
what the masters do, and how contexts resemble one another.

The server is a local Neo4j Community + GDS install (~/opt/neo4j, scripts/neo4j.sh), listening on
localhost only; credentials come from ~/.config/qaari/neo4j.env (NEO4J_URI / NEO4J_USER /
NEO4J_PASSWORD), never from the repo.

    python -m datastore.graph load-blindspots     # checks_T300 + blindspots.json -> the graph
    python -m datastore.graph similar-to "ط" tafkheem_or_taqeeq "ا"   # contexts like one, by GDS

Schema
  (:Sound {symbol})                       a phoneme of the Quran phonetic script
  (:Check {name})                         a characteristic head or 'identity'
  (:Context {key, letter, prev2, prev, next, position, word_position, ayah_position})
      -[:LETTER]->(:Sound)  -[:AFTER]->(:Sound) (the sound before)  -[:BEFORE]->(:Sound) (the sound after)
      -[:FAILS {n, failures, rate}]->(:Check)     what 41 professionals do on that check there
  (:Word {text})-[:POSITION {index, letter}]->(:WordCheck)... kept flat: (:Word)-[:FAILS {position,
      letter, check, n, failures, rate}]->(:Check)
"""

from __future__ import annotations

import gzip
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ENV = Path.home() / ".config/qaari/neo4j.env"
CHECKS = ROOT / "research_agency_lab/experiments/quran/checks_T300.json.gz"


def _env() -> dict[str, str]:
    env = dict(os.environ)
    if ENV.is_file():
        for line in ENV.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                env.setdefault(k.strip(), v.strip())
    return env


def driver():  # type: ignore[no-untyped-def]
    from neo4j import GraphDatabase
    e = _env()
    return GraphDatabase.driver(e.get("NEO4J_URI", "bolt://127.0.0.1:7687"),
                                auth=(e.get("NEO4J_USER", "neo4j"), e["NEO4J_PASSWORD"]))


def run(query: str, **params: Any) -> list[dict[str, Any]]:
    with driver() as d, d.session() as s:
        return s.run(query, **params).data()


SCHEMA = [
    "CREATE CONSTRAINT sound_symbol IF NOT EXISTS FOR (s:Sound) REQUIRE s.symbol IS UNIQUE",
    "CREATE CONSTRAINT check_name IF NOT EXISTS FOR (c:Check) REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT context_key IF NOT EXISTS FOR (c:Context) REQUIRE c.key IS UNIQUE",
    "CREATE CONSTRAINT word_text IF NOT EXISTS FOR (w:Word) REQUIRE w.text IS UNIQUE",
]


def _batches(rows: list[dict[str, Any]], n: int = 5000):  # type: ignore[no-untyped-def]
    for i in range(0, len(rows), n):
        yield rows[i:i + n]


def load_blindspots() -> dict[str, int]:
    """Contexts, sounds, checks and words with the professionals' failure counts (both reciter halves
    summed), from modal_profile.py::checks."""
    raw = json.load(gzip.open(CHECKS, "rt", encoding="utf-8"))
    ctx: dict[tuple[str, ...], dict[str, list[int]]] = {}
    for key, (n, k, _mg) in raw["ctx"].items():
        _fold, sym, head, prev2, prev, nxt, c, wpos, stop = key.split("|")
        a = ctx.setdefault((sym, prev2, prev, nxt, c, wpos, stop), {}).setdefault(head, [0, 0])
        a[0] += n
        a[1] += k
    words: dict[tuple[str, str, str, str], list[int]] = {}
    for key, (n, k, _mg) in raw["word"].items():
        _fold, word, pos, sym, head = key.split("|")
        a = words.setdefault((word, pos, sym, head), [0, 0])
        a[0] += n
        a[1] += k
    crow, frow = [], []
    for (sym, prev2, prev, nxt, c, wpos, stop), checks in ctx.items():
        key = "|".join((sym, prev2, prev, nxt, c, wpos, stop))
        crow.append({"key": key, "letter": sym, "prev2": prev2, "prev": prev, "next": nxt, "position": c,
                     "word_position": wpos, "ayah_position": stop})
        for head, (n, k) in checks.items():
            frow.append({"key": key, "check": head, "n": n, "failures": k, "rate": round(k / n, 4) if n else 0.0})
    wrow = [{"text": w, "position": int(p), "letter": s, "check": h, "n": n, "failures": k,
             "rate": round(k / n, 4) if n else 0.0} for (w, p, s, h), (n, k) in words.items() if n >= 5]
    with driver() as d, d.session() as s:
        for q in SCHEMA:
            s.run(q)
        s.run("MATCH (n) WHERE n:Context OR n:Word DETACH DELETE n")
        for b in _batches(crow):
            s.run("""UNWIND $rows AS r
                     MERGE (c:Context {key: r.key})
                     SET c.letter = r.letter, c.prev2 = r.prev2, c.prev = r.prev, c.next = r.next,
                         c.position = r.position, c.word_position = r.word_position, c.ayah_position = r.ayah_position
                     MERGE (l:Sound {symbol: r.letter}) MERGE (c)-[:LETTER]->(l)
                     MERGE (p:Sound {symbol: r.prev}) MERGE (c)-[:AFTER]->(p)
                     MERGE (x:Sound {symbol: r.next}) MERGE (c)-[:BEFORE]->(x)""", rows=b)
        for b in _batches(frow):
            s.run("""UNWIND $rows AS r
                     MATCH (c:Context {key: r.key}) MERGE (k:Check {name: r.check})
                     MERGE (c)-[f:FAILS]->(k) SET f.n = r.n, f.failures = r.failures, f.rate = r.rate""", rows=b)
        for b in _batches(wrow):
            s.run("""UNWIND $rows AS r
                     MERGE (w:Word {text: r.text}) MERGE (k:Check {name: r.check})
                     CREATE (w)-[:FAILS {position: r.position, letter: r.letter, n: r.n, failures: r.failures,
                                         rate: r.rate}]->(k)""", rows=b)
        counts = s.run("""MATCH (c:Context) WITH count(c) AS contexts
                          MATCH (w:Word) WITH contexts, count(w) AS words
                          MATCH ()-[f:FAILS]->() RETURN contexts, words, count(f) AS fails""").single()
    return dict(counts)


def similar_to(letter: str, check: str, prev: str, top: int = 15) -> list[dict[str, Any]]:
    """Contexts that resemble a given one (same neighbouring sounds, by GDS node similarity on the
    Context-Sound graph), with how often professionals fail the check there."""
    from graphdatascience import GraphDataScience
    e = _env()
    gds = GraphDataScience(e.get("NEO4J_URI", "bolt://127.0.0.1:7687"), auth=(e.get("NEO4J_USER", "neo4j"), e["NEO4J_PASSWORD"]))
    name = "context_sounds"
    if gds.graph.exists(name):
        gds.graph.drop(name)
    G, _ = gds.graph.project(name, ["Context", "Sound"], {"LETTER": {"orientation": "UNDIRECTED"},
                                                          "AFTER": {"orientation": "UNDIRECTED"},
                                                          "BEFORE": {"orientation": "UNDIRECTED"}})
    try:
        seeds = gds.run_cypher("""MATCH (c:Context {letter: $l, prev: $p})-[f:FAILS]->(:Check {name: $k})
                                  RETURN id(c) AS id, c.key AS key, f.rate AS rate, f.n AS n ORDER BY f.n DESC LIMIT 1""",
                               {"l": letter, "p": prev, "k": check})
        if seeds.empty:
            return []
        sim = gds.nodeSimilarity.filtered.stream(G, sourceNodeFilter=[int(seeds.id[0])], targetNodeFilter="Context",
                                                 topK=200, similarityCutoff=0.1)
        score = {int(a): float(b) for a, b in zip(sim.node2, sim.similarity)}
        rows = gds.run_cypher("""UNWIND $ids AS i MATCH (c:Context) WHERE id(c) = i
                                 MATCH (c)-[f:FAILS]->(:Check {name: $k})
                                 RETURN i AS id, c.key AS context, f.n AS n, f.rate AS rate""",
                              {"ids": list(score), "k": check})
        rows["similarity"] = [score[int(i)] for i in rows.id]
        rows = rows[rows.n >= 20].sort_values(["rate"], ascending=False).head(top).drop(columns=["id"])
        return [{"seed": seeds.key[0], "seed_rate": float(seeds.rate[0]), **r} for r in rows.to_dict("records")]
    finally:
        G.drop()


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "load-blindspots":
        print(load_blindspots())
    elif cmd == "similar-to":
        for r in similar_to(*sys.argv[2:5]):
            print(r)
