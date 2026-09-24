"""The measurement contract: every number a consumer app needs, for every recording.

The engine measures; the consumer decides what to tell the learner, in what order, with what
severity. So this is numbers only -- no advice, no labels -- flat, ID-linked, with units, and with
every measurement placed against the masters and the cohort so its magnitude is comparable across
rules, letters and characteristics:

  recording   tempo (seconds per count, class), wajh, basmala, stops
  words       reference, text, whether every judgement in it passed, the ids of those that did not
  rules       per located instance: expected range, observed counts and seconds, the SIGNED deviation
              outside the band (0 inside; negative = short), status, whether it is scored, and the
              percentile of the observed length among the masters' and the cohort's instances of the
              same rule
  letters     per letter: timing; identity (competitor, margin); per characteristic the expected and
              observed class (the class heard) and the competitor (the best class other than the
              expected one), the MARGIN (log p(expected) - log p(best other), nats; below 0 = not
              realised), whether it is scored, and its percentile among the masters' and the cohort's
              instances of the same characteristic on the same letter

Every compared value also carries z_masters, a robust z against the masters' distribution
((x - median) / (IQR / 1.349)): percentiles saturate at the 1st / 99th level, the z does not, so a
departure far beyond every master stays distinguishable from one just past the edge.

The contract is published as a JSON Schema, app/schemas/measurements-1.schema.json, with every field
documented; tests validate real reports against it. Adding an optional field keeps measurements/1;
removing, renaming or changing the meaning of one makes measurements/2.

Reference distributions: research_agency_lab/experiments/quran/reference_stats.json
(datastore/reference_stats.py), masters = the anchor tier, cohort = all 41 T300 reciters.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from app.submission import judged

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "research_agency_lab/experiments/quran/reference_stats.json"
SCHEMA = "measurements/1"


@lru_cache(maxsize=1)
def _ref() -> dict[str, Any]:
    return json.loads(REFERENCE.read_text()) if REFERENCE.is_file() else {}


def percentile(x: float | None, quantiles: list[float] | None) -> float | None:
    """Where x sits in a distribution given by its quantiles (0..1, linear between quantiles,
    clamped to the outermost levels -- a value beyond the 1st / 99th is reported at that level)."""
    if x is None or not quantiles:
        return None
    levels = _ref().get("quantile_levels")
    return round(float(np.interp(x, quantiles, levels)), 4) if levels else None


def robust_z(x: float | None, quantiles: list[float] | None) -> float | None:
    """(x - median) / (IQR / 1.349): unbounded, so a value far beyond every master is not clipped the
    way a percentile is at the 1st / 99th level."""
    if x is None or not quantiles:
        return None
    levels = _ref().get("quantile_levels", [])
    q25, q50, q75 = (quantiles[levels.index(v)] for v in (0.25, 0.5, 0.75))
    spread = (q75 - q25) / 1.349
    return round((x - q50) / spread, 3) if spread > 1e-9 else None


def _stats(x: float | None, d: dict[str, Any], key: str) -> dict[str, float | None]:
    return {"percentile_masters": percentile(x, (d.get("masters") or {}).get(key)),
            "percentile_cohort": percentile(x, (d.get("cohort") or {}).get(key)),
            "z_masters": robust_z(x, (d.get("masters") or {}).get(key))}


def _rule_pct(rule: str, counts: float | None) -> dict[str, float | None]:
    return _stats(counts, _ref().get("rules", {}).get(rule, {}), "counts_quantiles")


def _char_pct(head: str, letter: str, margin: float | None) -> dict[str, float | None]:
    c = _ref().get("characteristics", {}).get(head, {})
    return _stats(margin, c.get("by_letter", {}).get(letter) or c.get("all_letters") or {}, "margin_quantiles")


def build(report: dict[str, Any]) -> dict[str, Any]:
    words, rules, letters = [], [], []
    for a in report.get("ayahs", []):
        s, y = a["surah"], a["ayah"]
        off = a.get("word_offset", 0)      # a partial-ayah submission: word indices of the whole ayah
        wabs = lambda w: None if w is None else w + off  # noqa: E731
        t0 = a["frames"][0] * 0.04
        failing: dict[int, list[str]] = {w["index"]: [] for w in a.get("words", [])}
        lid = lambda i: f"{s}:{y}:L{i}"  # noqa: E731
        for i, l in enumerate(a.get("letters", [])):
            chars = {}
            for head, sv in (l.get("sifat") or {}).items():
                scored = judged(head, l.get("run_length", 1), l["symbol"])
                margin = sv.get("llr")
                pct = _char_pct(head, l["symbol"], margin)
                # model_best is the best class OTHER than the expected one: it is what was heard
                # only when the expected class lost
                chars[head] = {"expected": sv.get("expected"),
                               "observed": sv.get("expected") if sv.get("realised") else sv.get("model_best"),
                               "competitor": sv.get("model_best"), "margin": margin, "realised": bool(sv.get("realised")), "scored": scored, **pct}
                if scored and not sv.get("realised") and l.get("word") in failing:
                    failing[l["word"]].append(f"{lid(i)}:{head}")
            idn = l["identity"]
            if not idn["confirmed"] and l.get("word") in failing:
                failing[l["word"]].append(f"{lid(i)}:identity")
            letters.append({
                "id": lid(i), "word": wabs(l.get("word")), "symbol": l["symbol"], "kind": l["kind"],
                "uthmani_chars": l.get("uth", []), "run_length": l.get("run_length", 1),
                "onset_s": round(t0 + l["onset_s"], 3), "duration_s": l["duration_s"],
                "duration_counts": l.get("duration_counts"),
                "edge": bool(l.get("edge")),
                "identity": {"confirmed": idn["confirmed"], "competitor": idn["heard_instead"] or None,
                             "margin": None if idn.get("llr") is None else -idn["llr"]},
                "characteristics": chars})
        for j, v in enumerate(a.get("rules", [])):
            ev = v.get("evidence") or {}
            counts = ev.get("given_counts")
            exp = v.get("expected_counts")
            dev = None
            if counts is not None and exp:
                lo, hi = exp
                dev = round(counts - hi, 3) if counts > hi else (round(counts - lo, 3) if counts < lo else 0.0)
            secs = (ev.get("tempo_allowance") or {}).get("seconds")
            if secs is None and v.get("units"):
                idx = [u for u in v["units"] if 0 <= u < len(a.get("letters", []))]
                secs = round(sum(a["letters"][u]["duration_s"] for u in idx), 3) if idx else None
            scored = v["status"] in {"pass", "short", "long", "wrong"}
            pct = _rule_pct(v["rule"], counts) if counts is not None else \
                {"percentile_masters": None, "percentile_cohort": None, "z_masters": None}
            rid = f"{s}:{y}:R{j}"
            rules.append({
                "id": rid, "rule": v["rule"], "word": wabs(v["word_index"]), "mechanism": v["mechanism"],
                "status": v["status"], "scored": scored,
                "expected_counts": exp, "observed_counts": counts, "observed_seconds": secs,
                "deviation_counts": dev, **pct,
                "wajh": ev.get("wajh"), "tempo_allowance": "tempo_allowance" in ev,
                "letters": [lid(u) for u in v.get("units", [])],
                "evidence_margin": ev.get("llr")})
            if v["status"] in {"short", "long", "wrong"} and v["word_index"] in failing:
                failing[v["word_index"]].append(rid)
        for w in a.get("words", []):
            fails = failing.get(w["index"], [])
            words.append({"ref": f"{s}:{y}:{w['index'] + off}", "text": w["word"], "all_correct": not fails,
                          "failing": fails})
    m = report.get("mastery") or {}
    scored_chars = [c for l in letters for c in l["characteristics"].values() if c["scored"]]
    return {
        "schema": SCHEMA,
        "units": {"counts": "the reciter's own count unit (one vowelled letter)", "margin": "nats",
                  "percentile": "0..1 among the reference instances of the same rule / letter x characteristic",
                  "z_masters": "robust z against the masters: (x - median) / (IQR / 1.349)"},
        "recording": {"audio_seconds": report.get("audio_seconds"), "seconds_per_count": m.get("tempo_haraka_s"),
                      "tempo_class": m.get("tempo_mode"), "wajh": report.get("wajh") or {},
                      "basmala": report.get("basmala") or {},
                      "stops": [st for a in report.get("ayahs", []) for st in a.get("stops", [])],
                      # seconds of recording before the first letter and after the last: a
                      # recording cut inside a letter leaves that letter unmeasurable (edge)
                      "lead_room_s": round(letters[0]["onset_s"], 3) if letters else None,
                      "tail_room_s": round(report["audio_seconds"] - max(l["onset_s"] + l["duration_s"]
                                                                          for l in letters), 3)
                      if letters and report.get("audio_seconds") else None,
                      "edge_letters": sum(l["edge"] for l in letters)},
        "summary": {
            "words": len(words), "words_all_correct": sum(w["all_correct"] for w in words),
            "letters": len(letters), "identity_not_confirmed": sum(not l["identity"]["confirmed"] for l in letters),
            "characteristics_scored": len(scored_chars),
            "characteristics_not_realised": sum(not c["realised"] for c in scored_chars),
            "rules_scored": sum(r["scored"] for r in rules),
            "rules_failed": sum(r["status"] in {"short", "long", "wrong"} for r in rules)},
        "words": words, "rules": rules, "letters": letters,
    }
