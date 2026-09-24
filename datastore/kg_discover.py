"""Discovery on the Recitation Knowledge Graph (GDS): questions asked of the data, not answers put in.

    .venv/bin/python -m datastore.kg_discover        # all analyses -> research_agency_lab/experiments/graph/

  schools      each master's style vector (own stretch per rule, pass rate per rule) -> kNN -> Louvain;
               agreement with the given tiers (normalised mutual information)
  phonotactics the Quran's own sound-to-sound transitions (PMI) -> Louvain, PageRank; does the text's
               sound structure follow the articulation regions or the characteristics? (NMI)
  blindspots   FastRP embeddings of the contexts where professionals fail a check -> kNN -> Louvain:
               families of blind spots and what their members share
  rules        PageRank on the prerequisite graph (the foundational rules); Louvain on the skill
               partial correlations (skill families)
  ayahs        node similarity on the ayah-rule graph: ayahs with the same tajwid fingerprint
  sessions     the certified reciter's ground truth: catch rate per exercise and per scripted mistake
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from datastore.graph import _env

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research_agency_lab/experiments/graph"


def gds():  # type: ignore[no-untyped-def]
    from graphdatascience import GraphDataScience
    e = _env()
    return GraphDataScience(e.get("NEO4J_URI", "bolt://127.0.0.1:7687"), auth=(e.get("NEO4J_USER", "neo4j"), e["NEO4J_PASSWORD"]))


def nmi(a: list[Any], b: list[Any]) -> float:
    """Normalised mutual information between two labelings (arithmetic mean normalisation)."""
    n = len(a)
    ca, cb, cab = Counter(a), Counter(b), Counter(zip(a, b))
    mi = sum(v / n * math.log((v / n) / (ca[x] / n * cb[y] / n)) for (x, y), v in cab.items())
    ha = -sum(v / n * math.log(v / n) for v in ca.values())
    hb = -sum(v / n * math.log(v / n) for v in cb.values())
    return round(2 * mi / (ha + hb), 4) if ha + hb > 0 else 0.0


def nmi_test(a: list[Any], b: list[Any], perms: int = 2000) -> dict[str, float]:
    """NMI against its permutation null (labels of b shuffled): observed, null p95, one-sided p."""
    rng = np.random.default_rng(7)
    obs = nmi(a, b)
    null = [nmi(a, list(rng.permutation(np.array(b, dtype=object)))) for _ in range(perms)]
    return {"nmi": obs, "null_p95": round(float(np.quantile(null, 0.95)), 4),
            "p": round((1 + sum(x >= obs for x in null)) / (perms + 1), 4)}


def _fresh(g, name: str) -> None:  # type: ignore[no-untyped-def]
    g.run_cypher("CALL gds.graph.drop($n, false) YIELD graphName RETURN graphName", {"n": name})


def _call(g, proc: str, graph: str, config: dict[str, Any], yields: str):  # type: ignore[no-untyped-def]
    """A GDS procedure by name, through Cypher (stable across client versions)."""
    return g.run_cypher(f"CALL gds.{proc}($g, $c) YIELD {yields}", {"g": graph, "c": config})


# ----------------------------------------------------------------------------------------- schools
def schools(g) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    rows = g.run_cypher("""MATCH (x:Reciter) WHERE x.tier IN ['anchor','studio','imam','fast']
                           OPTIONAL MATCH (x)-[h:HOLDS]->(k:Rule) WITH x, collect([k.name, h.stretch_median]) AS holds
                           OPTIONAL MATCH (x)-[kp:KEEPS]->(r:Rule) RETURN x.name AS name, x.tier AS tier, holds,
                           collect([r.name, kp.pass_rate]) AS keeps""")
    feats = sorted({f"h:{k}" for hs in rows.holds for k, v in hs if k and v is not None} |
                   {f"k:{k}" for ks in rows.keeps for k, v in ks if k and v is not None})
    X = np.full((len(rows), len(feats)), np.nan)
    idx = {f: i for i, f in enumerate(feats)}
    for r, (hs, ks) in enumerate(zip(rows.holds, rows.keeps)):
        for k, v in hs:
            if k and v is not None:
                X[r, idx[f"h:{k}"]] = math.log(v)
        for k, v in ks:
            if k and v is not None:
                X[r, idx[f"k:{k}"]] = v
    keep = np.isnan(X).mean(axis=0) < 0.2
    X, feats = X[:, keep], [f for f, k in zip(feats, keep) if k]
    X = np.where(np.isnan(X), np.nanmean(X, axis=0), X)
    Z = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1)
    g.run_cypher("UNWIND $rows AS r MATCH (x:Reciter {name: r.n}) SET x.style = r.v",
                 {"rows": [{"n": n, "v": [float(v) for v in z]} for n, z in zip(rows.name, Z)]})
    _fresh(g, "styles")
    g.run_cypher("""MATCH (x:Reciter) WHERE x.style IS NOT NULL
                    RETURN gds.graph.project('styles', x, null, {sourceNodeProperties: x {.style}, targetNodeProperties: null}) AS p""")
    _call(g, "knn.mutate", "styles", {"nodeProperties": {"style": "EUCLIDEAN"}, "topK": 5, "randomSeed": 7,
                                      "concurrency": 1, "mutateRelationshipType": "STYLE_NEAR", "mutateProperty": "score"},
          "relationshipsWritten")
    comm = _call(g, "louvain.stream", "styles", {"relationshipTypes": ["STYLE_NEAR"], "relationshipWeightProperty": "score",
                                                 "concurrency": 1}, "nodeId, communityId RETURN gds.util.asNode(nodeId).name AS name, communityId")
    _fresh(g, "styles")
    c_of = {n: int(c) for n, c in zip(comm.name, comm.communityId)}
    tier = dict(zip(rows.name, rows.tier))
    names = [n for n in rows.name if n in c_of]
    # what defines each community: the features whose mean departs most from the rest
    groups: dict[int, list[int]] = defaultdict(list)
    for i, n in enumerate(rows.name):
        if n in c_of:
            groups[c_of[n]].append(i)
    desc = []
    for c, members in sorted(groups.items(), key=lambda x: -len(x[1])):
        diff = Z[members].mean(0)
        top = np.argsort(-np.abs(diff))[:5]
        desc.append({"community": c, "size": len(members), "reciters": [rows.name[i] for i in members],
                     "tiers": dict(Counter(tier[rows.name[i]] for i in members)),
                     "defined_by": [{"feature": feats[j], "z": round(float(diff[j]), 2)} for j in top]})
    g.run_cypher("UNWIND $rows AS r MATCH (x:Reciter {name: r.n}) SET x.school = r.c",
                 {"rows": [{"n": n, "c": c} for n, c in c_of.items()]})
    return {"features": len(feats), "communities": desc,
            "nmi_with_tier": nmi_test([c_of[n] for n in names], [tier[n] for n in names])}


# ------------------------------------------------------------------------------------ phonotactics
def phonotactics(g) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    _fresh(g, "sounds")
    g.run_cypher("""MATCH (a:Sound)-[p:PRECEDES]->(b:Sound) WHERE p.pmi > 0 AND p.count >= 20
                    RETURN gds.graph.project('sounds', a, b, {relationshipProperties: {w: p.pmi}},
                                             {undirectedRelationshipTypes: ['*']}) AS p""")
    comm = _call(g, "louvain.stream", "sounds", {"relationshipWeightProperty": "w", "concurrency": 1}, "nodeId, communityId RETURN *")
    pr = _call(g, "pageRank.stream", "sounds", {"relationshipWeightProperty": "w"}, "nodeId, score RETURN *")
    _fresh(g, "sounds")
    ids = g.run_cypher("UNWIND $ids AS i MATCH (s:Sound) WHERE id(s) = i OPTIONAL MATCH (s)-[:IS_LETTER]->(l:Letter)"
                       " OPTIONAL MATCH (l)-[:ARTICULATED_AT]->(:Makhraj)-[:IN]->(rg:Region)"
                       " OPTIONAL MATCH (l)-[:HAS_SIFAH]->(sf:Sifah) WHERE sf.name IN ['hams','jahr']"
                       " OPTIONAL MATCH (l)-[:HAS_SIFAH]->(sp:Sifah) WHERE sp.name IN $height"
                       " RETURN i, s.symbol AS sym, rg.name AS region, sf.name AS voice, sp.name AS height",
                       {"ids": [int(i) for i in comm.nodeId], "height": ["isti'la", "istifal"]})
    info = {int(i): (s, r, v, h) for i, s, r, v, h in zip(ids.i, ids.sym, ids.region, ids.voice, ids.height)}
    c_of = {info[int(i)][0]: int(c) for i, c in zip(comm.nodeId, comm.communityId)}
    rank = sorted(((info[int(i)][0], round(float(s), 3)) for i, s in zip(pr.nodeId, pr.score)), key=lambda x: -x[1])
    cons = [v for v in info.values() if v[1]]
    clusters: dict[int, list[str]] = defaultdict(list)
    for s, c in c_of.items():
        clusters[c].append(s)
    return {"clusters": sorted(clusters.values(), key=len, reverse=True),
            "pagerank_top": rank[:10],
            "nmi_consonant_clusters_vs_region": nmi_test([c_of[v[0]] for v in cons], [v[1] for v in cons]),
            "nmi_vs_hams_jahr": nmi_test([c_of[v[0]] for v in cons], [v[2] for v in cons]),
            "nmi_vs_istila": nmi_test([c_of[v[0]] for v in cons], [v[3] for v in cons])}


# ------------------------------------------------------------------------------------- blind spots
def blindspots(g) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    _fresh(g, "ctx")
    g.run_cypher("""CALL gds.graph.project('ctx', ['Context', 'Sound'], {LETTER: {orientation: 'UNDIRECTED'},
                    AFTER: {orientation: 'UNDIRECTED'}, BEFORE: {orientation: 'UNDIRECTED'}}) YIELD graphName RETURN graphName""")
    _call(g, "fastRP.write", "ctx", {"embeddingDimension": 64, "writeProperty": "emb", "randomSeed": 7, "concurrency": 1,
                                     "iterationWeights": [0.0, 1.0, 1.0]}, "nodePropertiesWritten RETURN *")
    _fresh(g, "ctx")
    # families among the contexts professionals fail (>= 25 %, >= 20 checks)
    hot = g.run_cypher("""MATCH (c:Context)-[f:FAILS]->(k:Check) WHERE f.rate >= 0.25 AND f.n >= 20
                          WITH c, collect(k.name) AS checks, max(f.rate) AS rate
                          RETURN c.key AS key, checks, rate, c.emb AS emb""")
    if hot.empty:
        return {"families": []}
    E = np.vstack(hot.emb.values)
    E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)
    S = E @ E.T
    # connected components of the cosine > 0.9 graph: families
    parent = list(range(len(hot)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(hot)):
        for j in range(i + 1, len(hot)):
            if S[i, j] > 0.9:
                parent[find(i)] = find(j)
    fam: dict[int, list[int]] = defaultdict(list)
    for i in range(len(hot)):
        fam[find(i)].append(i)
    out = []
    for members in sorted(fam.values(), key=len, reverse=True)[:12]:
        keys = [hot.key[i].split("|") for i in members]
        shared = {name: Counter(k[j] for k in keys).most_common(1)[0] for j, name in
                  enumerate(["letter", "prev2", "prev", "next", "position", "word_position", "ayah_position"])}
        out.append({"size": len(members), "checks": dict(Counter(k for i in members for k in hot.checks[i])),
                    "shared": {k: v[0] for k, v in shared.items() if v[1] >= 0.8 * len(members)},
                    "mean_rate": round(float(np.mean([hot.rate[i] for i in members])), 3),
                    "examples": [hot.key[i] for i in members[:4]]})
    return {"hot_contexts": len(hot), "families": out}


# ------------------------------------------------------------------------------------------- rules
def rules(g) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    _fresh(g, "prereq")
    g.run_cypher("CALL gds.graph.project('prereq', 'Rule', 'PREREQUISITE_OF') YIELD graphName RETURN graphName")
    pr = _call(g, "pageRank.stream", "prereq", {}, "nodeId, score RETURN gds.util.asNode(nodeId).name AS n, score")
    _fresh(g, "prereq")
    found = sorted(((n, round(float(s), 3)) for n, s in zip(pr.n, pr.score)), key=lambda x: -x[1])
    _fresh(g, "skillcov")
    g.run_cypher("""MATCH (a:Skill)-[c:COVARIES]-(b:Skill) WHERE c.partial_r > 0
                    RETURN gds.graph.project('skillcov', a, b, {relationshipProperties: {w: c.partial_r}}) AS p""")
    comm = _call(g, "louvain.stream", "skillcov", {"relationshipWeightProperty": "w", "concurrency": 1},
                 "nodeId, communityId RETURN gds.util.asNode(nodeId).name AS n, communityId")
    _fresh(g, "skillcov")
    fam: dict[int, list[str]] = defaultdict(list)
    for n, c in zip(comm.n, comm.communityId):
        fam[int(c)].append(n)
    return {"prerequisite_pagerank": found[:12], "skill_families": sorted(fam.values(), key=len, reverse=True)}


# ------------------------------------------------------------------------------------------- ayahs
def ayahs(g) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    g.run_cypher("""MATCH (a:Ayah)-[:HAS]->(:QWord)-[c:CARRIES]->(k:Rule)
                    WHERE NOT k.name IN ['jahr','hams','rakhawah','safir','tafkheem','tarqeeq','takreer','tawassut','itbaq',
                                         'tafashhi','shiddah','istitaalah','hamzat_wasl','madd_tabii']
                    WITH a, k, sum(c.n) AS n MERGE (a)-[t:TRAINS]->(k) SET t.n = n""")
    _fresh(g, "ayahrule")
    g.run_cypher("MATCH ()-[s:SAME_TAJWID]->() DELETE s")
    g.run_cypher("CALL gds.graph.project('ayahrule', ['Ayah', 'Rule'], 'TRAINS') YIELD graphName RETURN graphName")
    _call(g, "nodeSimilarity.write", "ayahrule", {"topK": 5, "similarityCutoff": 0.5, "writeRelationshipType": "SAME_TAJWID",
                                                  "writeProperty": "jaccard", "concurrency": 1}, "relationshipsWritten RETURN *")
    _fresh(g, "ayahrule")
    dense = g.run_cypher("""MATCH (a:Ayah)-[t:TRAINS]->(k) WITH a, count(k) AS kinds, sum(t.n) AS inst
                            RETURN a.ref AS ayah, a.words AS words, kinds, inst ORDER BY kinds DESC, inst DESC LIMIT 10""")
    ex = g.run_cypher("""MATCH (a:Ayah {ref: '55:39'})-[s:SAME_TAJWID]->(b:Ayah) RETURN b.ref AS ayah, s.jaccard AS j
                         ORDER BY j DESC LIMIT 5""")
    return {"most_rule_kinds": dense.to_dict("records"), "like_55_39": ex.to_dict("records")}


# ---------------------------------------------------------------------------------------- sessions
def sessions(g) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    rows = g.run_cypher("""MATCH (t:Take)-[v:JUDGED]->(m:ScriptedMistake)<-[:SCRIPTS]-(e:Exercise)
                           WITH m, e, t, v ORDER BY t.stamp DESC
                           WITH m, e, collect(v.verdict)[0] AS latest
                           RETURN e.id AS exercise, m.do AS mistake, latest ORDER BY exercise""")
    return {"per_mistake": rows.to_dict("records"),
            "caught": int((rows.latest == "caught").sum()), "scripted": int(len(rows))}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    g = gds()
    res = {}
    for name, fn in (("schools", schools), ("phonotactics", phonotactics), ("blindspots", blindspots),
                     ("rules", rules), ("ayahs", ayahs), ("sessions", sessions)):
        try:
            res[name] = fn(g)
            print(f"== {name}\n{json.dumps(res[name], ensure_ascii=False, indent=1)[:2500]}\n", flush=True)
        except Exception as exc:  # noqa: BLE001 - one analysis must not sink the others
            res[name] = {"error": f"{type(exc).__name__}: {exc}"[:500]}
            print(f"== {name} FAILED: {res[name]['error']}", flush=True)
    (OUT / "discoveries.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
