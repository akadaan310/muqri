"""The knowledge report: everything one recitation (and a learner's history) says about their Quran.

The grader measures a recitation. This turns the measurements into knowledge a client application can
build on -- not a list of hints, but the learner's whole state, stated with its basis:

  skills        every one of the 35 rule types: estimate, 90 % interval, basis (measured / inferred /
                prior), evidence, and how the learner stands against the masters and the cohort
                (app/learner.py; leave-one-reciter-out: beats the cohort mean on rules the learner
                has NOT yet recited by 4.7 % from one verse, 16.6 % from three, 27.7 % from ten)
  strengths     skills measured above the masters' own rate -- the mastery the learner did not know
                they have
  lengths       the learner's own count for each madd type, how consistently they hold it, and the
                masters' count beside it -- calibration is the reciter's, consistency is the standard
  projection    every rule instance in all 6,236 ayahs scored with the learner's skills: expected
                accuracy per surah, the ayahs where they are most at risk, and which rules carry it
  path          the knowledge space's prerequisite structure: what the learner holds, what they are
                ready to learn next (every prerequisite held), ordered by stage and by how much of
                the Quran it would improve
  measured      the recitation's own words, letters, rules and magnitudes, as the grader reports them

Nothing inferred is presented as measured: every skill carries its basis, and the projection is
labelled a projection.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.learner import CohortModel, load_inventory, project, tally

ROOT = Path(__file__).resolve().parents[1]
STRUCTURES = ROOT / "research_agency_lab/experiments/calculus/structures_T300.json"
RULE_LAYER = ROOT / "research_agency_lab/experiments/calculus/rule_layer_T300.json"
HELD = 0.9
LEARNER_FLOOR_SD = 1.0      # log-odds; the learner-population prior (see CohortModel.posterior)
# a gap must be real AND worth working on: statistically below the masters (the 90 % interval's upper
# end under their rate) and at least two points below them -- a careful reading otherwise listed
# gaps of 0.003, which are the masters' own calibration, not work
MIN_GAP = 0.02


@lru_cache(maxsize=1)
def _model() -> CohortModel:
    return CohortModel.load()


@lru_cache(maxsize=1)
def _inventory() -> list[dict[str, Any]]:
    return load_inventory()


@lru_cache(maxsize=1)
def _structures() -> dict[str, Any]:
    return json.loads(STRUCTURES.read_text()) if STRUCTURES.is_file() else {}


@lru_cache(maxsize=1)
def _masters() -> dict[str, Any]:
    """The masters' (anchors') rate for every skill -- rules, characteristics, identity -- and their
    median length per madd type."""
    from app.learner import MODEL_PATH
    rates = json.loads(MODEL_PATH.read_text()).get("masters", {})
    lens: dict[str, list[float]] = defaultdict(list)
    if RULE_LAYER.is_file():
        for d in json.loads(RULE_LAYER.read_text())["reciters"].values():
            if d["tier"] == "anchor":
                for ru, v in d["lengths"].items():
                    lens[ru].append(v["median"])
    return {"rates": rates, "lengths": {k: statistics.median(v) for k, v in lens.items()}}


def evidence_of(report: dict[str, Any]) -> tuple[dict[str, list[int]], dict[str, list[float]]]:
    """Skill tallies (rules, each characteristic head, letter identity) and madd lengths from one report.

    Characteristic evidence uses exactly the scorer's own exclusions (app.submission.judged): takrir
    concealed, a shaddah's hold, istitalah off the ض."""
    from app.submission import judged
    statuses, lengths = [], defaultdict(list)
    heads: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    ident = [0, 0]
    for a in report.get("ayahs", []):
        for v in a.get("rules", []):
            statuses.append((v["rule"], v["status"]))
            g = (v.get("evidence") or {}).get("given_counts")
            if v["mechanism"] == "durational" and g is not None and v["status"] in {"pass", "short", "long"}:
                lengths[v["rule"]].append(g)
        for l in a.get("letters", []):
            ident[0] += bool(l["identity"]["confirmed"])
            ident[1] += 1
            for h, sv in (l.get("sifat") or {}).items():
                if judged(h, l.get("run_length", 1), l["symbol"]):
                    heads[f"sifah:{h}"][0] += bool(sv.get("realised"))
                    heads[f"sifah:{h}"][1] += 1
    ev = tally(statuses)
    ev.update({h: v for h, v in heads.items() if v[1]})
    if ident[1]:
        ev["letter_identity"] = ident
    return ev, dict(lengths)


def merge(history: dict[str, Any] | None, ev: dict[str, list[int]], lens: dict[str, list[float]]) -> dict[str, Any]:
    """Add one recitation's evidence to a learner's history."""
    h = history or {"rules": {}, "lengths": {}, "recitations": 0}
    for ru, (k, n) in ev.items():
        a = h["rules"].setdefault(ru, [0, 0])
        a[0] += k
        a[1] += n
    for ru, xs in lens.items():
        h["lengths"].setdefault(ru, []).extend(xs)
    h["recitations"] += 1
    return h


def build(report: dict[str, Any], history: dict[str, Any] | None = None) -> dict[str, Any]:
    ev, lens = evidence_of(report)
    h = merge(json.loads(json.dumps(history)) if history else None, ev, lens)
    model = _model()
    # the cohort is 41 professionals; a learner is not -- the prior is widened to a floor (an assumption
    # to calibrate on learner recordings), so a learner's own misses can move their estimates
    skills = model.posterior(h["rules"], floor_sd=LEARNER_FLOOR_SD)
    masters = _masters()
    idx = {s: i for i, s in enumerate(model.skills)}
    for s, d in skills.items():
        d["cohort"] = round(float(1 / (1 + 2.718281828 ** -model.mu[idx[s]])), 4)
        d["masters"] = round(masters["rates"][s], 4) if s in masters["rates"] else None

    strengths = sorted((s for s, d in skills.items() if d["basis"] == "measured" and d["masters"] is not None
                        and d["interval90"][0] >= d["masters"] - 0.02 and d["evidence"][1] >= 2),
                       key=lambda s: -skills[s]["evidence"][1])
    # to work on: measured AND confidently below the masters -- the 90 % interval's upper end under the
    # masters' own rate. The gap is the magnitude of the work, not a pass/fail.
    to_work_on = []
    for s, d in skills.items():
        ref = d["masters"] if d["masters"] is not None else d["cohort"]
        if d["basis"] == "measured" and d["interval90"][1] < ref and ref - d["estimate"] >= MIN_GAP:
            d["gap_to_masters"] = round(ref - d["estimate"], 4)
            to_work_on.append(s)
    to_work_on.sort(key=lambda s: -skills[s]["gap_to_masters"])

    lengths = {}
    for ru, xs in h["lengths"].items():
        if len(xs) >= 1:
            med = statistics.median(xs)
            lengths[ru] = {"median_counts": round(med, 2), "instances": len(xs),
                           "spread_counts": round(statistics.pstdev(xs), 2) if len(xs) >= 2 else None,
                           "masters_median_counts": round(masters["lengths"][ru], 2) if ru in masters["lengths"] else None}

    proj = project(skills, _inventory())

    st = _structures()
    held = {s for s, d in skills.items() if d["estimate"] >= HELD and d["basis"] != "prior"}
    known_weak = {s for s, d in skills.items() if d["basis"] != "prior" and s not in held}
    prereq: dict[str, set[str]] = defaultdict(set)
    for e in st.get("prerequisites", []):
        prereq[e["b"]].add(e["a"])
    stage = st.get("stages", {})
    impact = {x["rule"]: x["expected_misses_across_the_quran"] for x in proj["where_the_risk_is"]}
    # ready: not yet held, and no prerequisite KNOWN to be weak (an unmeasured one is unknown, not failed)
    ready = sorted((s for s in known_weak if not (prereq[s] & known_weak)),
                   key=lambda s: (stage.get(s, 99), -impact.get(s, 0)))
    later = sorted((s for s in known_weak if s not in ready), key=lambda s: stage.get(s, 99))

    return {
        "schema": "knowledge/1",
        "recitations_so_far": h["recitations"],
        "summary": {
            "skills_measured": sum(d["basis"] == "measured" for d in skills.values()),
            "skills_inferred": sum(d["basis"] == "inferred" for d in skills.values()),
            "skills_prior_only": sum(d["basis"] == "prior" for d in skills.values()),
            "projected_rule_accuracy_whole_quran": proj["expected_rule_accuracy"],
            "strengths": strengths, "to_work_on": to_work_on,
        },
        "skills": skills,
        "lengths": lengths,
        "wajh": report.get("wajh") or {},
        "tempo": (report.get("mastery") or {}).get("tempo_mode"),
        "projection": proj,
        "path": {"held": sorted(held), "ready_to_learn": ready,
                 "later": [{"rule": s, "needs_first": sorted(prereq[s] & known_weak)} for s in later]},
        "history": h,
        "honesty": "skills marked 'inferred' or 'prior' are estimated, not measured; the projection is a "
                   "projection from the learner's skills onto every rule instance of the Quran; the prior "
                   "is fitted on 41 professional reciters and widened for learners (an assumption until "
                   "calibrated on learner recordings)",
    }
