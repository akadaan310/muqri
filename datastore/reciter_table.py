#!/usr/bin/env python3
"""The ranked reciter table the app ships: every reference reciter scored on the same verses.

Built from the counterfactual measurements rather than the legacy rule engine, because those are the
parts we have actually validated:

* **lahn**  — the reciter's *clean-text* flag rate from `textswap_gop.jl` (how often the engine thinks
  a letter or harakah was not what the text says), split consonant / vowel.
* **sifat** — mean flag rate across the ten muaalem attribute heads (`sifat_run.jl`).
* **rules** — flag rate on the rule-layer counterfactuals (`ruleswap_gop.jl`): how often an *edited*
  realisation explains the audio better than the correct one, i.e. the reciter did something else.
* **structure** — repetitions and skips found by `read_as_recited`.
* **quality** — free-decode vs reference edit distance, the data-integrity gate.

Every component is a *deviation rate against the anchor-calibrated thresholds*, so lower is closer to
the ijazah reference. The headline `score` is 100 × (1 − weighted deviation), which puts the Husary
anchors at the top by construction of the thresholds, not by fiat — and the ladder tier is reported
separately so a fast taraweeh imam is not mistaken for a poor reciter.

    .venv/bin/python -m datastore.reciter_table [--tier T300]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datastore.ingest import ladder  # noqa: E402  (the anchor/studio/imam/fast ladder)
from datastore.store import connect, run  # noqa: E402

K = ROOT / "research_agency_lab/experiments/qaari_keys"

SCHEMA = """
CREATE TABLE IF NOT EXISTS reciter_score (
    tier             TEXT,
    reciter_id       TEXT,
    ladder           TEXT,
    n_clips          INTEGER,
    lahn_consonant   DOUBLE,   -- clean-text flag rate, consonants
    lahn_vowel       DOUBLE,   -- clean-text flag rate, harakat
    sifat_flag       DOUBLE,   -- mean flag rate over the ten sifat heads
    rule_flag        DOUBLE,   -- mean flag rate over the rule-layer counterfactuals
    deviations       INTEGER,  -- repetitions + skips
    qc_edit          DOUBLE,   -- free decode vs reference, normalised edit distance
    tasawi_cv        DOUBLE,   -- mean robust CV of held lengths (consistency; lower better)
    separation       DOUBLE,   -- rikhw - shadeed sukoon duration, in own counts (higher = master)
    haraka_s         DOUBLE,   -- the reciter's own count unit, seconds
    score            DOUBLE,   -- 100 x (1 - weighted deviation)
    rank             INTEGER,
    PRIMARY KEY (tier, reciter_id)
);
"""

# weights: letters and harakat are lahn jali (the gravest), then the rule layer, then the sifat
# lahn jali is the gravest, then the rule layer, then the sifat, then tasawi (consistency).
# Weights are renormalised over whichever components a tier actually has.
W = {"lahn_consonant": 0.25, "lahn_vowel": 0.25, "rule_flag": 0.22, "sifat_flag": 0.13,
     "tasawi_cv": 0.15}


def _mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def collect(tier: str) -> list[dict]:  # type: ignore[type-arg]
    def load(stem):  # type: ignore[no-untyped-def]
        f = K / f"{stem}_{tier}.json"
        return json.loads(f.read_text()) if f.is_file() else None

    sf, rs, qc_f = load("sifat"), load("ruleswap"), load("qc")
    ts, tw, sk = load("textswap"), load("tasawi"), load("sukoon")
    qc = (qc_f or {}).get("reciters", {})

    rec: dict[str, dict] = {}
    if ts:  # lahn jali — the gravest errors, when the (slow) text-swap pass has landed
        for kind, key in (("consonant", "lahn_consonant"), ("vowel", "lahn_vowel")):
            for spk, v in ts[kind]["per_speaker"].items():
                rec.setdefault(spk, {})[key] = v["clean_flag_rate"]
        for spk, v in ts.get("deviations", {}).items():
            rec.setdefault(spk, {})["deviations"] = v
    if sf:
        for _lvl, classes in sf["levels"].items():
            for _cls, cv in classes.items():
                for spk, v in cv["per_speaker"].items():
                    rec.setdefault(spk, {}).setdefault("_sifat", []).append(v["flag_rate"])
    if rs:
        for _fam, fv in rs["families"].items():
            for spk, v in fv["per_speaker"].items():
                rec.setdefault(spk, {}).setdefault("_rule", []).append(v["flag_rate"])
    if tw:  # tasawi: consistency of every held length, in the speaker's own counts
        for _cls, cv in tw["classes"].items():
            for spk, v in cv["per_speaker"].items():
                if v["cv"] == v["cv"]:  # not NaN
                    rec.setdefault(spk, {}).setdefault("_cv", []).append(v["cv"])
    if sk:  # sukoon class separation: a mastery signal, higher is better
        for r in sk["reciters"]:
            rec.setdefault(r["spk"], {}).update(separation=r["separation"], haraka_s=r["haraka_s"])

    rows = []
    for spk, d in rec.items():
        d["sifat_flag"] = _mean(d.pop("_sifat", []))
        d["rule_flag"] = _mean(d.pop("_rule", []))
        d["tasawi_cv"] = _mean(d.pop("_cv", []))
        d["qc_edit"] = qc.get(spk, {}).get("median_edit")
        d["n_clips"] = qc.get(spk, {}).get("n", 0)
        present = {k: w for k, w in W.items() if d.get(k) is not None}
        tot = sum(present.values()) or 1.0
        dev = sum(w * d[k] for k, w in present.items()) / tot     # renormalise over what we have
        d.update(reciter_id=spk, ladder=ladder(spk.split(":", 1)[-1]),
                 score=round(100 * (1 - dev), 3), deviations=d.get("deviations", 0),
                 components=",".join(sorted(present)))
        rows.append(d)
    rows.sort(key=lambda r: -r["score"])
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="T300")
    a = ap.parse_args(argv)
    rows = collect(a.tier)
    con = connect()
    con.execute(SCHEMA)
    with run(con, f"table:reciter_score:{a.tier}", "loader", {"reciters": len(rows)}):
        con.execute("DELETE FROM reciter_score WHERE tier = ?", [a.tier])
        con.executemany("INSERT INTO reciter_score VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        [(a.tier, r["reciter_id"], r["ladder"], r.get("n_clips", 0),
                          r.get("lahn_consonant"), r.get("lahn_vowel"), r.get("sifat_flag"),
                          r.get("rule_flag"), r.get("deviations", 0), r.get("qc_edit"),
                          r.get("tasawi_cv"), r.get("separation"), r.get("haraka_s"),
                          r["score"], r["rank"]) for r in rows])
    print(f"{'#':>3} {'score':>7}  {'ladder':8} {'sifat':>7} {'rule':>7} {'tasawi':>7} "
          f"{'sep':>6} {'hrk':>5}  reciter")
    for r in rows:
        f = lambda k: "   -  " if r.get(k) is None else f"{r[k]:.4f}"  # noqa: E731
        g = lambda k, n=3: "  -  " if r.get(k) is None else f"{r[k]:.{n}f}"  # noqa: E731
        print(f"{r['rank']:>3} {r['score']:>7.3f}  {r['ladder']:8} {f('sifat_flag')} {f('rule_flag')} "
              f"{g('tasawi_cv')}   {g('separation')}  {g('haraka_s', 2)}  {r['reciter_id'].split(':', 1)[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
