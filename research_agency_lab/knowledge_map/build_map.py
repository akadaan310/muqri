"""Merge knowledge-map fragments into one graph and render an index for humans and agents.

    python research_agency_lab/knowledge_map/build_map.py            # rebuild graph.json + INDEX.md
    python research_agency_lab/knowledge_map/build_map.py q <id|text>  # neighbourhood of a node
    python research_agency_lab/knowledge_map/build_map.py gaps       # phenomena with no implemented model

Fragments live in ``fragments/*.json`` as ``{"nodes": [...], "edges": [...]}`` (schema in README.md).
Nodes with the same id are merged: later fragments fill missing fields, ``status`` keeps the most
advanced value, ``refs`` are unioned. Edges are de-duplicated; edges to unknown ids create stub
nodes (flagged ``stub: true``) so nothing is silently dropped.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
STATUS_RANK = {"implemented": 6, "partial": 5, "proposed": 4, "speculative": 3, "missing": 2,
               "not_observable": 1}
TYPE_BY_PREFIX = {"phen": "phenomenon", "math": "math", "algo": "algorithm", "pkg": "package",
                  "src": "source", "code": "code", "model": "model", "data": "dataset"}


def load() -> tuple[dict[str, dict[str, Any]], list[dict[str, str]], list[str]]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[tuple[str, str, str], dict[str, str]] = {}
    problems: list[str] = []
    for f in sorted((HERE / "fragments").glob("*.json")):
        try:
            frag = json.loads(f.read_text())
        except json.JSONDecodeError as exc:
            problems.append(f"{f.name}: invalid JSON ({exc})")
            continue
        for n in frag.get("nodes", []):
            nid = n.get("id")
            if not nid:
                problems.append(f"{f.name}: node without id")
                continue
            cur = nodes.setdefault(nid, {"id": nid, "sources": []})
            cur["sources"].append(f.stem)
            for k, v in n.items():
                if v in (None, "", []):
                    continue
                if k == "status" and k in cur:
                    if STATUS_RANK.get(v, 0) > STATUS_RANK.get(cur[k], 0):
                        cur[k] = v
                elif k == "refs":
                    cur["refs"] = sorted(set(cur.get("refs", [])) | set(v if isinstance(v, list) else [v]))
                elif k == "priority" and k in cur:
                    cur[k] = min(cur[k], v)
                else:
                    cur.setdefault(k, v)
        for e in frag.get("edges", []):
            a, b, r = e.get("from"), e.get("to"), e.get("rel")
            if a and b and r:
                edges[(a, b, r)] = {"from": a, "to": b, "rel": r, "source": f.stem}
    for a, b, _ in list(edges):
        for nid in (a, b):
            if nid not in nodes:
                nodes[nid] = {"id": nid, "type": TYPE_BY_PREFIX.get(nid.split(":")[0], "unknown"),
                              "label": nid.split(":", 1)[-1], "stub": True, "sources": []}
    for n in nodes.values():
        n.setdefault("type", TYPE_BY_PREFIX.get(n["id"].split(":")[0], "unknown"))
    return nodes, list(edges.values()), problems


def build() -> None:
    nodes, edges, problems = load()
    graph = {"nodes": sorted(nodes.values(), key=lambda n: n["id"]), "edges": edges, "problems": problems}
    (HERE / "graph.json").write_text(json.dumps(graph, ensure_ascii=False, indent=1))
    (HERE / "INDEX.md").write_text(render(nodes, edges, problems))
    by_type = defaultdict(int)
    for n in nodes.values():
        by_type[n["type"]] += 1
    print(f"{len(nodes)} nodes, {len(edges)} edges; " + ", ".join(f"{k} {v}" for k, v in sorted(by_type.items())))
    for p in problems:
        print("PROBLEM:", p)


def _models_of(nodes: dict[str, dict[str, Any]], edges: list[dict[str, str]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for e in edges:
        if e["rel"] in ("models", "measures") and nodes[e["to"]]["type"] == "phenomenon":
            out[e["to"]].append(e["from"])
        elif e["rel"] == "measured_by" and nodes[e["from"]]["type"] == "phenomenon":
            out[e["from"]].append(e["to"])
    return out


def render(nodes: dict[str, dict[str, Any]], edges: list[dict[str, str]], problems: list[str]) -> str:
    models = _models_of(nodes, edges)
    impl = lambda i: nodes[i].get("status") in ("implemented", "partial")  # noqa: E731
    lines = ["# Knowledge map index (generated — edit fragments/, then run build_map.py)", "",
             f"{len(nodes)} nodes · {len(edges)} edges · fragments: "
             + ", ".join(sorted({s for n in nodes.values() for s in n.get('sources', [])})), ""]
    lines += ["## Tajweed phenomena → models", "",
              "| phenomenon | level | engine status | implemented models / code | proposed models |",
              "|---|---|---|---|---|"]
    phen = sorted((n for n in nodes.values() if n["type"] == "phenomenon"),
                  key=lambda n: (["101", "intermediate", "advanced", "ijazah"].index(n["level"])
                                 if n.get("level") in ("101", "intermediate", "advanced", "ijazah") else 9, n["id"]))
    for n in phen:
        ms = models.get(n["id"], [])
        done = ", ".join(f"`{m}`" for m in ms if impl(m))
        todo = ", ".join(f"`{m}`" for m in ms if not impl(m))
        lines.append(f"| `{n['id']}` {n.get('label', '')} | {n.get('level', '')} | {n.get('status', '')} | {done} | {todo} |")
    for typ, title in (("math", "Mathematical constructs"), ("algorithm", "Algorithms"), ("package", "Packages"),
                       ("model", "Pretrained models"), ("dataset", "Datasets"), ("code", "Code")):
        sel = sorted((n for n in nodes.values() if n["type"] == typ),
                     key=lambda n: (-STATUS_RANK.get(n.get("status", ""), 0), n.get("priority", 9), n["id"]))
        if not sel:
            continue
        lines += ["", f"## {title}", "", "| id | status | prio | label | where / refs |", "|---|---|---|---|---|"]
        for n in sel:
            where = n.get("file") or ", ".join(n.get("refs", [])[:3])
            lines.append(f"| `{n['id']}` | {n.get('status', '')} | {n.get('priority', '')} | {n.get('label', '')} | {where} |")
    if problems:
        lines += ["", "## Build problems", ""] + [f"- {p}" for p in problems]
    return "\n".join(lines) + "\n"


def query(term: str) -> None:
    nodes, edges, _ = load()
    hits = [i for i in nodes if i == term] or [i for i, n in nodes.items()
                                               if term.lower() in (i + " " + str(n.get("label", ""))).lower()]
    for h in hits[:10]:
        n = nodes[h]
        print(f"\n{h}  [{n['type']}, {n.get('status', '?')}]  {n.get('label', '')}")
        for k in ("level", "priority", "file", "note", "refs"):
            if n.get(k):
                print(f"  {k}: {n[k]}")
        for e in edges:
            if e["from"] == h:
                print(f"  --{e['rel']}--> {e['to']}")
            elif e["to"] == h:
                print(f"  <--{e['rel']}-- {e['from']}")


def gaps() -> None:
    nodes, edges, _ = load()
    models = _models_of(nodes, edges)
    for n in sorted(nodes.values(), key=lambda n: n["id"]):
        if n["type"] != "phenomenon":
            continue
        ms = models.get(n["id"], [])
        if not any(nodes[m].get("status") in ("implemented", "partial") for m in ms):
            prop = [m for m in ms if nodes[m].get("status") in ("proposed", "speculative")]
            print(f"{n['id']:45s} {n.get('status', ''):12s} proposed: {', '.join(prop) or '-'}")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "q":
        query(" ".join(sys.argv[2:]))
    elif len(sys.argv) > 1 and sys.argv[1] == "gaps":
        gaps()
    else:
        build()
