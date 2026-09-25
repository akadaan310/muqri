"""Read-only extraction of the causal smoke run's stored evidence into review tables (markdown on stdout).
Sources: causal/data/records.json (local), causal/results/causal_table.json, causal/results/parity.json."""
import json
import math

import pathlib
C = str(pathlib.Path(__file__).resolve().parents[1])
R = json.load(open(f"{C}/data/records.json"))
T = json.load(open(f"{C}/results/causal_table.json"))
P = json.load(open(f"{C}/results/parity.json"))
by = {r["experiment_id"]: r for r in R}
TRANS = ("voicing", "duration", "formant", "f0", "nasal")
TARGET_PHYS = {"voicing": "voiced_fraction", "duration": "span_duration_s", "formant": "f2_hz", "f0": "f0_hz",
               "nasal": "nasal_ratio_db"}


def nat(uid):
    s = [r for r in R if r["experiment_id"].startswith(uid + "/same-letter|")]
    return {k: abs(v) for k, v in s[0]["delta_muqri"].items() if v is not None} if s else None


def fmt(v, n=3):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:+.{n}f}" if abs(v) < 1e4 else f"{v:+.1f}"
    return str(v)


def selectivity(r):
    uid, key = r["experiment_id"].split("/", 1)
    h = r["hypothesis"]
    coll = r["collateral"]["metrics_beyond_noise"]
    nv = nat(uid)
    beyond_nat = None if nv is None else [k for k, v in coll.items() if abs(v["delta"]) > nv.get(k, math.inf)]
    tg = h["muqri_targets"]
    if not tg:
        if len(coll) == 0:
            s = "null holds (0 beyond floor)"
        elif beyond_nat == []:
            s = f"null: {len(coll)} beyond floor, 0 beyond same-letter"
        else:
            s = f"null violated ({len(coll)} beyond floor" + (f", {len(beyond_nat)} beyond same-letter)" if beyond_nat is not None else ")")
        return s, beyond_nat
    tcls = [r["classification"].get(k) for k in tg]
    if any(c != "expected change" for c in tcls):
        return "no expected target response", beyond_nat
    if beyond_nat is None:
        return ("selective vs floor" if not coll else "non-selective vs floor (no natural-variation reference)"), beyond_nat
    return ("selective" if not beyond_nat else "non-selective"), beyond_nat


print("## 2. All 40 records (from data/records.json; same values as results/causal_table.json)\n")
print("| # | experiment | unit | transform | params | physical var | requested (dir / predicted) | measured Δ | phys status | Muqri target | before | after | Δ | target class | coll>floor | coll>same-letter | selectivity | verification_status |")
print("|" + "---|" * 18)
for i, r in enumerate(R, 1):
    u, h = r["unit"], r["hypothesis"]
    tg = list(h["muqri_targets"].items()) or [(None, 0)]
    for mq, exp in tg:
        s, bn = selectivity(r)
        pm = h["physical_metric"]
        print(f"| {i} | `{r['experiment_id']}` | {u['symbol']} {u['form']} {u['letter_id']} [{u['start_s']},{u['end_s']}] | "
              f"{r['transform']['name']} | {json.dumps(r['transform']['parameters'], ensure_ascii=False)} | {pm} | "
              f"{h['physical_direction']:+d} / {fmt(h['predicted_physical_delta'])} | {fmt(r['delta_physics'].get(pm)) if pm != '—' else '—'} | "
              f"{r['physics_verification'].get('status')} | {mq or '(none)'} | {fmt(r['muqri_baseline'].get(mq)) if mq else '—'} | "
              f"{fmt(r['muqri_transformed'].get(mq)) if mq else '—'} | {fmt(r['delta_muqri'].get(mq)) if mq else '—'} | "
              f"{r['classification'].get(mq, 'n/a') if mq else 'n/a'} | {len(r['collateral']['metrics_beyond_noise'])} | "
              f"{'n/a' if bn is None else len(bn)} | {s} | {r['verification_status']} |")

print("\n## 3. Boundary-shift controls\n")
mid = {"zay_fatha": ('voicing|{"level": 0.3}', "head:hams_or_jahr:margin"), "sad_fatha": ('formant|{"a": 0.8}', "head:tafkheem_or_taqeeq:margin"),
       "ghunnah": ('nasal|{"g_db": -12}', "head:ghonna:margin")}
for uid, (tk, tgt) in mid.items():
    base = by[f"{uid}/{tk}"]
    lo = [r for r in R if r["experiment_id"].startswith(f"{uid}/boundary-0.02")][0]
    hi = [r for r in R if r["experiment_id"].startswith(f"{uid}/boundary+0.02")][0]
    nv = nat(uid)
    nf = base["collateral"]["noise_floor_muqri"]
    print(f"### {uid}: `{tk}` applied at the Viterbi span [{base['unit']['start_s']}, {base['unit']['end_s']}] and shifted ±20 ms\n")
    print("| Muqri metric | untouched baseline | edit at span | edit at span −20 ms | edit at span +20 ms | |(−20)−(span)| | |(+20)−(span)| | noise floor | same-letter |Δ| | exceeds floor | exceeds same-letter |")
    print("|" + "---|" * 11)
    keys = [k for k in base["muqri_baseline"] if isinstance(base["muqri_baseline"][k], (int, float)) and not isinstance(base["muqri_baseline"][k], bool)]
    keys = sorted(keys, key=lambda k: (k != tgt, not k.startswith("identity"), k))
    for k in keys:
        b, e, m, p = base["muqri_baseline"].get(k), base["muqri_transformed"].get(k), lo["muqri_transformed"].get(k), hi["muqri_transformed"].get(k)
        if not all(isinstance(x, (int, float)) for x in (b, e, m, p)):
            continue
        dm, dp = abs(m - e), abs(p - e)
        fl = nf.get(k, 1.0)
        sl = None if nv is None else nv.get(k)
        ex_f = ",".join(s for s, d in (("−20", dm), ("+20", dp)) if d > fl) or "—"
        ex_n = "n/a" if sl is None else (",".join(s for s, d in (("−20", dm), ("+20", dp)) if d > sl) or "—")
        print(f"| {k}{' (target)' if k == tgt else ''} | {b:.3f} | {e:.3f} | {m:.3f} | {p:.3f} | {dm:.3f} | {dp:.3f} | {fl:.3f} | {fmt(sl)} | {ex_f} | {ex_n} |")
    print(f"\nPhysics on the edited span: span {base['physics_transformed'].get('span_duration_s')} s; the ±20 ms records measure physics on the SHIFTED span against the baseline's UNSHIFTED span, so their physics deltas mix the edit with 20 ms of different audio.\n")

print("\n## 5. Selected records\n")
tr = [r for r in R if r["transform"]["name"] in TRANS and not r["experiment_id"].split("/", 1)[1].startswith("boundary")]
succ = max((r for r in tr for k in r["hypothesis"]["muqri_targets"] if r["classification"].get(k) == "expected change"),
           key=lambda r: max(abs(r["delta_muqri"][k]) / r["collateral"]["noise_floor_muqri"].get(k, 1) for k in r["hypothesis"]["muqri_targets"]))
print(f"- strongest successful target response (largest |Δ|/floor among expected changes): `{succ['experiment_id']}` "
      + ", ".join(f"{k} Δ {succ['delta_muqri'][k]} / floor {succ['collateral']['noise_floor_muqri'].get(k)}" for k in succ["hypothesis"]["muqri_targets"]))
nulls = [r for r in tr if not r["hypothesis"]["muqri_targets"]]
cl = min(nulls, key=lambda r: (len(r["collateral"]["metrics_beyond_noise"]), max(abs(v) for v in r["delta_muqri"].values() if v is not None)))
print(f"- cleanest invariant/null response: `{cl['experiment_id']}` — {len(cl['collateral']['metrics_beyond_noise'])} metrics beyond floor; "
      f"largest |Δ| any Muqri metric {max(abs(v) for v in cl['delta_muqri'].values() if v is not None):.3f}; letters elsewhere flipped {cl['collateral']['letters_elsewhere_flipped']}")
cont = max(tr, key=lambda r: (len(selectivity(r)[1] or []), len(r["collateral"]["metrics_beyond_noise"])))
print(f"- strongest collateral contamination: `{cont['experiment_id']}` — {len(cont['collateral']['metrics_beyond_noise'])} beyond floor, "
      f"{len(selectivity(cont)[1] or [])} beyond same-letter variation")
cont2 = max(tr, key=lambda r: len(r["collateral"]["metrics_beyond_noise"]))
print(f"  (most metrics beyond the floor, any unit: `{cont2['experiment_id']}` — {len(cont2['collateral']['metrics_beyond_noise'])})")
bs = []
for uid, (tk, _t) in mid.items():
    base = by[f"{uid}/{tk}"]
    for r in R:
        if r["experiment_id"].startswith(f"{uid}/boundary"):
            for k, v in r["muqri_transformed"].items():
                e = base["muqri_transformed"].get(k)
                if isinstance(v, (int, float)) and isinstance(e, (int, float)) and not isinstance(v, bool):
                    bs.append((abs(v - e), r["experiment_id"], k, e, v))
b = max(bs)
print(f"- strongest boundary sensitivity: `{b[1]}` — {b[2]}: {b[3]:.3f} (edit at span) vs {b[4]:.3f} (shifted), |Δ| {b[0]:.3f}")
wr = [(abs(r["delta_muqri"][k]), r, k) for r in R for k in r["hypothesis"]["muqri_targets"] if r["classification"].get(k) == "unexpected change"]
if wr:
    w = max(wr, key=lambda x: x[0])
    print(f"- strongest wrong-direction target response: `{w[1]['experiment_id']}` — {w[2]} Δ {w[1]['delta_muqri'][w[2]]} (expected sign {w[1]['hypothesis']['muqri_targets'][w[2]]:+d}, floor {w[1]['collateral']['noise_floor_muqri'].get(w[2])})")
print("  all target 'unexpected change' records: " + "; ".join(f"`{x[1]['experiment_id']}` {x[2]} {x[1]['delta_muqri'][x[2]]}" for x in wr))
dd = max(P["spans"]["disagreements"], key=lambda x: max(abs((x["delta"][a] or 0) - (x["delta"][b] or 0)) for a in x["delta"] for b in x["delta"]))
print(f"- strongest cross-tool disagreement (largest spread of deltas among the 18): `{dd['experiment']}` {dd['metric']} Δ python {dd['delta']['python']}, julia {dd['delta']['julia']}, octave {dd['delta']['octave']}")
big = max(P["spans"]["rows"], key=lambda x: max(abs((x["delta"][a] or 0) - (x["delta"][b] or 0)) for a in x["delta"] for b in x["delta"]) if sum(v is not None for v in x["delta"].values()) >= 2 else 0)
print(f"  (largest spread of deltas over ALL parity rows, direction agreeing: `{big['experiment']}` {big['metric']} Δ {big['delta']})")

print("\n## 6. The 18 direction disagreements (results/parity.json → spans.disagreements)\n")
print("| # | experiment | metric | Δ python/Praat | Δ julia | Δ octave | directions | max |Δ| spread | is this the transform's target metric? |")
print("|" + "---|" * 9)
for i, x in enumerate(P["spans"]["disagreements"], 1):
    tfam = x["experiment"].split("/", 1)[1].split("|")[0]
    is_t = TARGET_PHYS.get(tfam) == x["metric"] and not x["experiment"].split("/", 1)[1].startswith("boundary")
    dirs = " ".join(f"{k[:2]}{'+' if (v or 0) > 0 else '−' if (v or 0) < 0 else '0'}" for k, v in x["delta"].items() if v is not None)
    spread = max(abs((x["delta"][a] or 0) - (x["delta"][b] or 0)) for a in x["delta"] for b in x["delta"] if x["delta"][a] is not None and x["delta"][b] is not None)
    print(f"| {i} | `{x['experiment']}` | {x['metric']} | {x['delta']['python']} | {x['delta']['julia']} | {x['delta']['octave']} | {dirs} | {spread:.2f} | {'YES' if is_t else 'no'} |")
print("\nZero-vs-nonzero conflicts on TARGET metrics (not counted in the 18, which only compare signs of non-zero deltas):\n")
for x in P["spans"]["rows"]:
    tfam = x["experiment"].split("/", 1)[1].split("|")[0]
    if TARGET_PHYS.get(tfam) == x["metric"] and not x["experiment"].split("/", 1)[1].startswith("boundary"):
        vals = {k: v for k, v in x["delta"].items() if v is not None}
        if len(vals) >= 2 and any(v == 0 for v in vals.values()) and any(v != 0 for v in vals.values()):
            print(f"- `{x['experiment']}` {x['metric']}: {vals}")

print("\n## 7. Positive controls\n")
for uid in ("sad_fatha", "zay_fatha"):
    r = [x for x in R if x["experiment_id"].startswith(f"{uid}/neighbour|")][0]
    print(f"### `{r['experiment_id']}` (span [{r['unit']['start_s']}, {r['unit']['end_s']}], unit type {r['unit']['unit_type']}, donor span {r['transform']['span']})\n")
    keys = ["identity:margin", "identity:confirmed", "identity:competitor", "vowel:identity:margin"] + sorted(k for k in r["muqri_baseline"] if k.startswith("makhraj:")) + \
        sorted(k for k in r["muqri_baseline"] if k.startswith("head:") and k.endswith(":margin"))
    print("| metric | before | after | Δ | class |")
    print("|---|---|---|---|---|")
    for k in keys:
        print(f"| {k} | {r['muqri_baseline'].get(k)} | {r['muqri_transformed'].get(k)} | {fmt(r['delta_muqri'].get(k))} | {r['classification'].get(k, '—')} |")
    print(f"\nverdict flips in unit: {r['collateral']['verdict_flips_in_unit']}; letters elsewhere flipped: {r['collateral']['letters_elsewhere_flipped']}\n")

print("\n## 4. Per-family data\n")
for fam in TRANS:
    rs = [r for r in tr if r["transform"]["name"] == fam]
    for r in rs:
        s, bn = selectivity(r)
        print(f"- `{r['experiment_id']}`: target " + (", ".join(f"{k} {r['classification'].get(k)} (Δ {r['delta_muqri'].get(k)})" for k in r["hypothesis"]["muqri_targets"]) or "none (invariance)")
              + f"; physics {r['physics_verification'].get('status')}; collateral>floor {len(r['collateral']['metrics_beyond_noise'])}; >same-letter {'n/a' if bn is None else len(bn)}; {s}")
