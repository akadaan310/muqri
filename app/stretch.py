"""Stretchings measured against the recitation's own count: the timing calculus, serving side.

Every recitation sets its own clock. Its plain voweled letters -- a consonant carrying a short vowel,
followed by another letter, with nothing held (no madd, no shaddah, no nasal hold) -- each take one
count; their durations give the reciter's count unit here, locally (it follows a reciter who speeds
up or slows down within a passage). Every stretching -- a madd, a ghunnah, an ikhfa -- is then a
multiple of that unit: its STRETCH. What a correct stretch is for each rule is learned from the
masters' own recitations measured the same way (substrate_library/julia/stretch.jl), never fixed in
seconds, never binned by speed.

This module extracts the raw material from an engine report (`timing_record`) -- the plain spans and
the stretching instances with their times -- and applies a fitted stretch model (`apply`).
"""

from __future__ import annotations

from typing import Any

SHORT_VOWEL = "harakah"
HELD_KINDS = {"madd", "ikhfa_noon", "iqlab_meem"}


def plain_spans(letters: list[dict[str, Any]]) -> list[tuple[float, float]]:
    """(time at the middle, seconds) of every plain voweled letter: consonant, short vowel, then a
    letter that does not lengthen it. Nothing that touches the last unit (it absorbs the silence)."""
    out = []
    n = len(letters)
    for i in range(n - 2):
        c, v, nxt = letters[i], letters[i + 1], letters[i + 2]
        if c["kind"] != "consonant" or c["run_length"] != 1 or v["kind"] != SHORT_VOWEL:
            continue
        if nxt["kind"] in HELD_KINDS or i + 2 >= n - 1:
            continue
        t0, t1 = c["onset_s"], nxt["onset_s"]
        if t1 > t0:
            out.append((round((t0 + t1) / 2, 3), round(t1 - t0, 3)))
    return out


def instances(ayah: dict[str, Any]) -> list[dict[str, Any]]:
    """Every durational rule instance: rule, time, seconds (its units, as the grader measures them),
    nominal counts, and the current verdict. Instances on the clip's last unit are left out."""
    letters = ayah["letters"]
    last = len(letters) - 1
    out = []
    for r in ayah["rules"]:
        if r.get("mechanism") != "durational" or not r.get("units"):
            continue
        us = [u for u in r["units"] if 0 <= u < len(letters)]
        if not us or last in us:
            continue
        secs = sum(letters[u]["duration_s"] for u in us)
        t = letters[us[0]]["onset_s"] + secs / 2
        ev = r.get("evidence") or {}
        out.append({"rule": r["rule"], "word": r["word_index"], "t": round(t, 3), "seconds": round(secs, 3),
                    "nominal": r.get("expected_counts"), "status": r["status"],
                    "counts_now": ev.get("given_counts"), "wajh": ev.get("wajh")})
    return out


def timing_record(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Per ayah of an engine report: its plain spans and its stretching instances."""
    return [{"surah": a["surah"], "ayah": a["ayah"], "plain": plain_spans(a["letters"]),
             "stretch": instances(a)} for a in report.get("ayahs", [])]


# --------------------------------------------------------------------------------------------- apply
import json as _json
import math as _math
from functools import lru_cache as _lru_cache
from pathlib import Path as _Path

MODEL = _Path(__file__).resolve().parents[1] / "research_agency_lab/experiments/timing/stretch_model.json"
WAJH_RULES = ("madd_munfasil", "madd_silah_kubra")


@_lru_cache(maxsize=1)
def model() -> dict[str, Any]:
    return _json.loads(MODEL.read_text()) if MODEL.is_file() else {}


def unit_of(spans: list[float]) -> float | None:
    """The reciter's count unit here: geometric mean of the plain voweled letters' durations (the
    estimator that kept the masters' tabii most even; stretch.jl, step 1)."""
    xs = [d for d in spans if d > 0]
    return _math.exp(sum(_math.log(d) for d in xs) / len(xs)) if len(xs) >= 3 else None


def judge(rule: str, seconds: float, unit: float, wajh: str | None = None) -> dict[str, Any] | None:
    """One stretching against the masters' norm at this reciter's tempo: stretch in units, the
    expected stretch, the ratio, equivalent counts and z. None when the model has no norm for it."""
    m = model()
    if not m or unit <= 0 or seconds <= 0:
        return None
    norms = [t for t in m["tempo_norms"] if t["rule"] == rule]
    if rule in WAJH_RULES:
        lv = 2.0 if wajh and "qasr" in wajh else 4.5
        norms = [t for t in norms if t["level"] == lv]
    elif rule in m.get("fixed_levels", {}):
        norms = [t for t in norms if t["level"] == m["fixed_levels"][rule]]
    if not norms:
        return None
    rho = seconds / unit
    x = _math.log(unit / m["U0"])
    multi = rule in m.get("multilevel_rules", [])
    best = None
    for t in norms:
        d = _math.log(rho) - t["a"] - t["b"] * x
        # short on the lower half's spread, long on the full SD (the upper tail holds pauses and very
        # long holds); a multi-level rule's own spread is floored (levels assigned by thresholds)
        sd = t.get("sd_low", t["sd"]) if d < 0 else t["sd"]
        if multi:
            sd = max(sd, m["sd_floor_multilevel"])
        z = d / sd
        if best is None or abs(z) < abs(best[1]):
            best = (t, z)
    t, z = best
    expected = _math.exp(t["a"] + t["b"] * x)
    zt = m["z_threshold"]
    one_sided = rule in m.get("one_sided_rules", [])
    verdict = "short" if z < -zt else ("long" if z > zt and not one_sided else "pass")
    return {"stretch": round(rho, 3), "expected_stretch": round(expected, 3), "ratio": round(rho / expected, 3),
            "level": t["level"], "equivalent_counts": round(t["level"] * rho / expected, 2), "z": round(z, 2),
            "verdict": verdict}


# rules whose SHORT verdict the calculus decides: it flagged fewer professionals than the windows
# (leave-one-reciter-out: munfasil 0.051 vs 0.218, muttasil 0.022 vs 0.176, lazim 0.038 vs 0.077,
# silah kubra 0.090 vs 0.166, leen 0.038 vs 0.006 on long but it is graded short only here) and
# caught the certified reciter's cut munfasil / muttasil / lazim. "Long" stays with the windows: the
# upper side absorbs pauses. Nasals keep the calibrated band (it caught the cut ghunnah and ikhfa the
# calculus missed); the short madds keep their windows (fewer professional flags: tabii 0.020 vs 0.046).
DECIDES_SHORT = ("madd_munfasil", "madd_muttasil", "madd_lazim", "madd_silah_kubra", "madd_leen")
# ... and both ways for the lazim: it sits mid-word (no pause to absorb), the windows' six-count
# ceiling flagged 7.7 % of professionals -- Husary's 9.5-count lazim of ad-dallin among them -- and
# the calculus 3.8 %
DECIDES_BOTH = ("madd_lazim",)


def apply(report: dict[str, Any]) -> dict[str, Any]:
    """The whole recording: its unit per ayah (the recording's own when an ayah has too few plain
    letters), and every measurable stretching judged."""
    recs = timing_record(report)
    pooled = unit_of([d for r in recs for _t, d in r["plain"]])
    out = []
    for r in recs:
        u = unit_of([d for _t, d in r["plain"]]) or pooled
        for s in r["stretch"]:
            j = judge(s["rule"], s["seconds"], u, s.get("wajh")) if u else None
            out.append({"surah": r["surah"], "ayah": r["ayah"], "word": s["word"], "rule": s["rule"],
                        "seconds": s["seconds"], "unit_s": round(u, 3) if u else None, **(j or {"verdict": "no norm"})})
    return {"unit_s": round(pooled, 3) if pooled else None,
            "unit_by_ayah": [round(unit_of([d for _t, d in r["plain"]]) or 0, 3) for r in recs], "stretchings": out}
