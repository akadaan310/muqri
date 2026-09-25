"""The human-readable technical report: Markdown generated from observatory/content.py and the snapshot
files, and a small Markdown -> HTML renderer (no third-party dependency)."""

from __future__ import annotations

import html
import json
import re
from typing import Any


def _t(rows: list[dict[str, Any]], cols: list[tuple[str, str]]) -> str:
    def cell(v: Any) -> str:
        s = "—" if v is None else (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v))
        return s.replace("|", "\\|").replace("\n", " ")
    out = ["| " + " | ".join(t for _k, t in cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(cell(r.get(k)) for k, _t in cols) + " |" for r in rows]
    return "\n".join(out)


def render_markdown() -> str:
    from observatory import content as C
    from observatory.server import (FACTORY_DIAGRAM, GRAPH_DIAGRAM, PIPE_DIAGRAM, digest_by_round, load,
                                    perturbation_matrix, rescore_trails, round_metrics)
    snap = load("snapshot.json") or {}
    neo = load("neo4j_snapshot.json") or {}
    bill = load("modal_billing.json") or {}
    cur = load("current_engine.json") or {}
    git = snap.get("git", {})
    co = snap.get("corpus", {})
    L: list[str] = []
    a = L.append
    a("# Muqri: technical report on the current implementation\n")
    a(f"Git `{git.get('revision')}` on `{git.get('branch')}` ({git.get('last_commit')}). {snap.get('tests_collected')} "
      "tests collected. Every statement describes code and data present in this repository and runtime; anything "
      "else is marked PLANNED, CONCEPTUAL or NOT VERIFIED.\n")
    a("## 0. How to read this\n")
    a("- The live, clickable version is `/muqri-observatory`; the machine-readable one is `/api/observatory/manifest`.\n"
      "- Status words: IMPLEMENTED (in the served Sessions path), TESTED (tests or recorded measurements), PARTIALLY "
      "IMPLEMENTED, EXPERIMENTAL (research code, measured), PLANNED, CONCEPTUAL, NOT VERIFIED.\n"
      "- There are **two analysis engines**; results from one do not describe the other (section 1).\n")

    a("## 1. Architecture\n")
    a(_t(C.ENGINES, [("name", "engine"), ("status", "status"), ("module", "code"), ("used_by", "used by"), ("model", "model"),
                     ("reference_data", "reference data"), ("note", "note")]) + "\n")
    a("### The Sessions engine path\n\n```\n" + PIPE_DIAGRAM + "\n```\n")
    a(_t(C.PIPELINE, [("stage", "component"), ("status", "status"), ("module", "code"), ("language", "language"),
                      ("library", "libraries"), ("model", "model / algorithm"), ("input", "input"), ("output", "output"),
                      ("compute", "compute"), ("evidence", "evidence"), ("note", "note")]) + "\n")
    a("### Deterministic, statistical and learned parts\n")
    a(_t([{"layer": k, "where": v} for k, v in C.REASONING.items()], [("layer", "layer"), ("where", "where")]) + "\n")

    a("## 2. What the engine measures\n")
    a(_t(C.MEASURES, [("item", "item"), ("status", "status"), ("module", "code"), ("algorithm", "algorithm"),
                      ("features", "features"), ("uncertainty", "confidence / uncertainty"), ("output", "output"),
                      ("evidence", "evidence")]) + "\n")
    a("## 3. The 17 makhārij\n")
    a(C.makharij()[0]["note"] + "\n")
    a(_t(C.makharij(), [("n", "#"), ("makhraj", "makhraj"), ("letters", "letters"), ("representation", "representation"),
                        ("detection", "detection"), ("evidence", "evidence"), ("status", "status")]) + "\n")
    a("## 4. The 17 ṣifāt (and ghunnah)\n")
    a(_t([dict(zip(("name", "arabic", "letters", "acoustic", "detector", "score", "status", "evidence"), s)) for s in C.SIFAT],
         [("name", "ṣifah"), ("arabic", ""), ("letters", "letters"), ("acoustic", "acoustic correlate"),
          ("detector", "detector"), ("score", "score"), ("status", "status"), ("evidence", "evidence")]) + "\n")
    a("Idhlāq/iṣmāt are a classification of letters (fluency of articulation), not a property of a sound; the "
      "system declares them per letter (app/letters.py) and never measures them. Līn is timed as a rule (madd līn), "
      "inḥirāf has no detector, takrīr is reported but never scored.\n")

    a("## 5. Session lab, rounds 1–5\n")
    a("One certified reader recorded each exercise twice from a phone browser: take A to a written specification, "
      "take B with scripted mistakes, one per word, the rest intended correct (r2e4 B is a second correct take at "
      "another speed; r3e4 was never recorded). The recording device is not stored. **As recorded** = the engine "
      "that scored the take on submission. **Current engine** = the same latest recording rescored by "
      f"`{str(cur.get('git_revision'))[:12]}` at {cur.get('scored_at')} (dry run; historical files untouched).\n")
    rows = []
    for n, v in round_metrics().items():
        x, y = v["as_recorded"], v["current_engine"]
        rows.append({"round": n, "caught as recorded": f"{x.get('caught', 0)}/{x.get('mistakes', 0)}",
                     "caught current": f"{y.get('caught', 0)}/{y.get('mistakes', 0)}",
                     "other-reason flags": f"{x.get('other_reason', 0)} → {y.get('other_reason', 0)}",
                     "expectations as recorded": f"{x.get('met', 0)}/{x.get('expectations', 0)}",
                     "expectations current": f"{y.get('met', 0)}/{y.get('expectations', 0)}",
                     "false-alarm words": f"{x.get('false_alarm_words', 0)} → {y.get('false_alarm_words', 0)}"})
    a(_t(rows, [(k, k) for k in rows[0]]) + "\n")
    a("**What this can and cannot establish.** Each round uses a different passage, different rules and different "
      "mistake types, and the expectations and signatures were written by the people tuning the engine. Rates across "
      "rounds are therefore not a controlled learning curve. The current-engine column re-reads older rounds with "
      "code changed after seeing them, so a rise there is partly fit to these recordings. The closest thing to "
      "held-out evidence is each round's first scoring against the changes made before it (commit list below). "
      "Separating sensitivity from false alarms is possible per round from the two columns; separating genuine "
      "generalisation from fit is not possible with this data alone.\n")
    a("### Exercises\n")
    for ex in snap.get("exercises", []):
        recs = [r for r in snap.get("recordings", []) if r["exercise"] == ex["id"]]
        a(f"**{ex['id']} — {ex['title']}** (surah {ex['surah']}, ayahs {ex['ayahs'][0]}–{ex['ayahs'][1]}, wajh "
          f"{ex['wajh']}). {ex['goal']}\n\nSpec: {ex['spec']}\n\nRecordings: "
          + (", ".join(f"{r['take']} {r['stamp']} ({r['format']}, {r['audio_seconds'] or '?'} s)" for r in recs) or "none")
          + "\n")
    a("### Engine commits during the rounds\n")
    a(_t(snap.get("engine_commits", []), [("time", "time"), ("hash", "commit"), ("subject", "subject")]) + "\n")
    a("### Rescore trails\n")
    a(_t([{"recording": k, "trail": " → ".join((f"{e['met']}/{e['expectations']}" if e["expectations"] else
                                                f"{e['caught']}/{e['mistakes']}") + f" ({e['false_alarms']} FA)" for e in evs)}
          for k, evs in rescore_trails().items()], [("recording", "recording"), ("trail", "met or caught (false-alarm words), in order")]) + "\n")
    a("### Per rule / letter / characteristic, take A, current engine\n")
    for n, d in digest_by_round().items():
        a(f"**Round {n}.** Rules: " + "; ".join(f"{k} {v}" for k, v in d["rules"].items()) + ". Letters not confirmed: "
          + (", ".join(f"{k} {v.get('not_confirmed')}" for k, v in d["letters"].items() if v.get("not_confirmed")) or "none")
          + ". Characteristics not realised: " + (", ".join(f"{k} {v.get('not_realised')}" for k, v in
                                                            d["characteristics"].items() if v.get("not_realised")) or "none") + ".\n")

    a("## 6. Deliberately perturbed recitations: the take-B matrix\n")
    a(_t(perturbation_matrix(), [("exercise", "ex"), ("ayah", "ayah"), ("word", "w"), ("text", "word"),
                                 ("perturbation", "intended perturbation"), ("target", "intended check"),
                                 ("as_recorded", "as recorded"), ("as_recorded_evidence", "evidence"),
                                 ("current", "current"), ("current_evidence", "evidence (current)"),
                                 ("false_alarms_elsewhere_as_recorded", "FA words in take (as rec.)"),
                                 ("false_alarms_elsewhere_current", "FA words (current)"), ("audio", "audio")]) + "\n")

    a("## 7. Reciter corpus\n")
    a(f"Full-Qur'an measurements exist for **{len(co.get('full_quran_reciters', []))}** reciters: "
      f"{', '.join(co.get('full_quran_reciters', []))}. The T300 posterior set has "
      f"{co.get('t300_posteriors', {}).get('speakers')} speakers × {co.get('t300_posteriors', {}).get('clips_per_speaker')} "
      "ayahs. A count of ~45 complete reciters is not supported by the data in this repository.\n")
    a(_t([{"reciter | mode": k, "ayahs": v} for k, v in co.get("quran_md_kaggle", {}).items()], [("reciter | mode", "Quran-MD (Kaggle) reciter | mode"), ("ayahs", "ayahs")]) + "\n")
    a(_t([{"reciter": k, "ayahs": v} for k, v in co.get("studio_all_everyayah", {}).items()], [("reciter", "studio-all (Modal) reciter"), ("ayahs", "ayahs")]) + "\n")
    lc = co.get("letter_corpus") or {}
    a(f"**Letter corpus (EXPERIMENTAL):** {lc.get('cells')} cells; instances per source {json.dumps(lc.get('instances_per_source'), ensure_ascii=False)}. "
      "Boundaries are the engine's 40 ms Viterbi frames (no refinement implemented). Cleaning: DC removal, 60 Hz "
      "zero-phase high-pass, spectral gating from the ayah's quietest 10 % of frames, zero-crossing cuts, 5 ms fades, "
      "−20 dBFS; raw cut kept; effect on the engine not measured. Rights metadata: none stored.\n")
    a("## 8. Master anchors\n")
    tiers = co.get("tier_sets", {})
    a(f"Hand-assigned by folder name (datastore/ingest.py): anchor {', '.join(tiers.get('anchor', []))}; "
      f"{len(tiers.get('studio', []))} studio peers; {len(tiers.get('imams', []))} imams. No unsupervised master-likeness "
      "scorer is in the served path; the benchmark pipeline's perfection index ranked learners above masters "
      "(AUROC 0.111, caveats in learners_uncalibrated.json). External learner evidence (QuranMB v2, 1,642 clips): "
      "muaalem-based letter-substitution recall 0.562 at 1.33 reports per clean clip (head_to_head.json).\n")

    a("## 9. Neo4j knowledge graph (snapshot)\n")
    a(f"Neo4j {json.dumps(neo.get('version'))}, 127.0.0.1 only, not read by the engine at runtime. "
      f"{neo.get('total_nodes', 0):,} nodes, {neo.get('total_relationships', 0):,} relationships. 16 Makhraj nodes; "
      "sessions layer 7 exercises / 37 takes (rounds 1–2).\n\n```\n" + GRAPH_DIAGRAM + "\n```\n")
    a(_t([{"label": k, "count": v} for k, v in sorted((neo.get("labels") or {}).items(), key=lambda x: -x[1])],
         [("label", "label"), ("count", "nodes")]) + "\n")
    a(_t(neo.get("patterns", []), [("from", "from"), ("rel", "relationship"), ("to", "to"), ("count", "count")]) + "\n")

    J = C.JULIA_OCTAVE
    a(f"## 10. Julia and GNU Octave\n\n{J['summary']}\n\nData flow: {J['flow']}\n")
    a(_t([{"f": f, "w": w} for f, w in J["julia"]], [("f", "Julia"), ("w", "what")]) + "\n")
    a(_t([{"f": f, "w": w} for f, w in J["octave"]], [("f", "Octave"), ("w", "what")]) + "\n")
    a("## 11. Modal\n")
    a(_t(C.MODAL, [("app", "app"), ("file", "code"), ("compute", "compute"), ("what", "what ran")]) + "\n")
    a(f"Billing ({bill.get('source')}): total ${bill.get('total_usd')}; by app {json.dumps(bill.get('by_app_usd'))}. "
      f"{bill.get('note', '')}\n")
    a("## 12. Models\n")
    a(_t(C.MODELS, [("name", "model"), ("version", "version"), ("purpose", "purpose"), ("input", "input"), ("output", "output"),
                    ("where", "where"), ("training", "trained?"), ("interaction", "role")]) + "\n")
    S = C.SYNTHESIS
    a("## 13. Synthesis\n")
    for k, t in [("already_implemented", "Already implemented"), ("experimentally_tested", "Experimentally tested"),
                 ("prototype", "Prototype"), ("planned", "Planned"), ("not_implemented", "Not yet implemented")]:
        a(f"**{t}**\n\n" + "\n".join(f"- {x}" for x in S[k]) + "\n")
    a("The proposed architecture is in docs/synthesis-engine.md and docs/epics/EPIC-1.md (plans).\n")
    a("## 14. Qirāʾāt\n")
    a(_t([{"k": k.replace("_", " "), "v": v} for k, v in C.QIRAAT.items()], [("k", "aspect"), ("v", "current state")]) + "\n")
    a("## 15. Data generation factory\n\n```\n" + FACTORY_DIAGRAM + "\n```\n")
    a(_t([{"s": s, "st": st, "w": w} for s, st, w in C.FACTORY], [("s", "stage"), ("st", "status"), ("w", "what exists")]) + "\n")
    a("## 16. Limitations\n\n" + "\n".join(f"- {x}" for x in C.LIMITATIONS) + "\n")
    return "\n".join(L)


def render_html() -> str:
    from observatory.server import CSS
    md = render_markdown()
    out, lines, i = [], md.splitlines(), 0
    esc = html.escape

    def inline(s: str) -> str:
        s = esc(s)
        s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
        return re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("```"):
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            out.append("<pre>" + esc("\n".join(lines[i + 1:j])) + "</pre>")
            i = j + 1
            continue
        if ln.startswith("|") and i + 1 < len(lines) and lines[i + 1].startswith("|---"):
            head = [c.strip() for c in ln.strip("|").split(" | ")]
            rows = []
            j = i + 2
            while j < len(lines) and lines[j].startswith("|"):
                rows.append([c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\) \| ", lines[j].strip()[2:-2])])
                j += 1
            out.append("<table><tr>" + "".join(f"<th>{inline(h)}</th>" for h in head) + "</tr>"
                       + "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in rows) + "</table>")
            i = j
            continue
        m = re.match(r"(#{1,3}) (.*)", ln)
        if m:
            out.append(f"<h{len(m.group(1))}>{inline(m.group(2))}</h{len(m.group(1))}>")
        elif ln.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(f"<li>{inline(lines[i][2:])}</li>")
                i += 1
            out.append("<ul>" + "".join(items) + "</ul>")
            continue
        elif ln.strip():
            out.append(f"<p>{inline(ln)}</p>")
        i += 1
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,"
            f"initial-scale=1'><title>Muqri technical report</title><style>{CSS}</style></head><body><div class='wrap'>"
            "<p class='mut'><a href='/muqri-observatory'>← Observatory</a> · <a href='/muqri-observatory/report.md'>Markdown</a></p>"
            + "".join(out) + "</div></body></html>")
