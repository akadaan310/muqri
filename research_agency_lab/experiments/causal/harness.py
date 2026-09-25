#!/usr/bin/env python3
"""The causal-experiment harness: one authentic unit, one single-variable transform at several levels,
controls, independent physics, the unchanged Muqri engine, and a classified record of every delta.

    .venv/bin/python research_agency_lab/experiments/causal/harness.py smoke      # the smoke-test matrix

Per unit it runs, all on the same ayah audio (EveryAyah, the letter corpus's measured instances):

  baseline        the untouched ayah through Engine.analyze (makhraj test on)
  determinism     the same audio again: the engine must return the same numbers
  noop_splice     the original samples spliced back: the harness must change nothing
  noop_world      WORLD resynthesis, no edit        } the processing footprint of each transform family,
  noop_pv         phase vocoder at k = 1            } used as its noise floor
  boundary +/-    the middle level of the first transform applied to the span shifted by 20 ms
  same-letter     the unit replaced by the same reciter's same letter + vowel from another ayah
  neighbour       ... by a neighbouring letter (a positive control: identity must move)
  levels          the transform at each parameter level

Physics (Praat) is measured on the span in the untouched and the altered audio; Octave and Julia
measure the same spans independently; Julia also re-derives every delta from the stored raw values.
Nothing is written outside research_agency_lab/experiments/causal/{data,results}.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from research_agency_lab.experiments.causal import physics as PH  # noqa: E402
from research_agency_lab.experiments.causal import schema as S  # noqa: E402
from research_agency_lab.experiments.causal import transforms as T  # noqa: E402

HERE = Path(__file__).parent
DATA = HERE / "data"
RESULTS = HERE / "results"
SR = 16000
MEASURE = ROOT / "research_agency_lab/experiments/letter_corpus/data/measure"

# transform -> (physics metric, direction as a function of the parameter, predicted delta, muqri targets)
HYPOTHESES: dict[str, Any] = {
    "voicing": lambda p, base, u: S.Hypothesis(
        "voiced_fraction", -1 if p["level"] < 1 else 0, None,
        {"head:hams_or_jahr:margin": -1},
        f"remove {100 * (1 - p['level']):.0f} % of the voicing of {u.symbol}: voiced fraction and HNR fall; "
        "Muqri's hams/jahr margin (toward jahr) falls; nothing else moves"),
    "duration": lambda p, base, u: S.Hypothesis(
        "span_duration_s", int(np.sign(p["k"] - 1)), round((p["k"] - 1) * (u.end_s - u.start_s), 4),
        ({f"rule:{u.form.split(':', 1)[1]}:counts": int(np.sign(p["k"] - 1))} if u.unit_type == "rule"
         else {"letter:duration_s": int(np.sign(p["k"] - 1))}),
        f"stretch the span x{p['k']}: its duration changes by (k-1) x {u.end_s - u.start_s:.3f} s; Muqri's length "
        "follows; identity and characteristics stay"),
    "formant": lambda p, base, u: S.Hypothesis(
        "f2_hz", 1 if p["a"] < 1 else -1,
        round(base["f2_hz"] * (1 / p["a"] - 1), 1) if base.get("f2_hz") else None,
        {"head:tafkheem_or_taqeeq:margin": -1 if u.symbol in "خصضغطقظ" else 1},
        f"warp the envelope by a = {p['a']}: formants move by 1/a; for a heavy letter the tafkhim margin falls"),
    "f0": lambda p, base, u: S.Hypothesis(
        "f0_hz", int(np.sign(p["semitones"])),
        round(base["f0_hz"] * (2 ** (p["semitones"] / 12) - 1), 1) if base.get("f0_hz") else None, {},
        f"shift the pitch {p['semitones']:+} semitones: F0 moves; pitch is no tajwid property, so every Muqri "
        "measurement should stay"),
    "nasal": lambda p, base, u: S.Hypothesis(
        "nasal_ratio_db", int(np.sign(p["g_db"])), None, {"head:ghonna:margin": int(np.sign(p["g_db"]))},
        f"nasal pole/anti-resonance {p['g_db']:+} dB: the 150-400 / 750-1100 Hz ratio moves the same way; "
        "Muqri's ghonna margin follows"),
}
NOOP_FOR = {"voicing": "noop_world", "formant": "noop_world", "f0": "noop_world", "nasal": "noop_world",
            "duration": "noop_pv"}


# ---------------------------------------------------------------- Muqri view ----------------------------------

def muqri_view(m: dict[str, Any], lid: str, idx: list[str], word: int) -> dict[str, Any]:
    """Numeric Muqri metrics of the unit: letter and vowel identity, every head, makhraj, durations, rules."""
    L = {x["id"]: x for x in m["letters"]}
    out: dict[str, Any] = {}
    l = L.get(lid)
    if l is None:
        return {"error": f"{lid} not in report"}
    out["identity:margin"] = l["identity"]["margin"]
    out["identity:confirmed"] = l["identity"]["confirmed"]
    out["identity:competitor"] = l["identity"]["competitor"]
    out["identity:blind_spot_p"] = l["identity"]["blind_spot_p"]
    for h, c in l["characteristics"].items():
        out[f"head:{h}:margin"] = c["margin"]
        out[f"head:{h}:realised"] = c["realised"]
        out[f"head:{h}:scored"] = c["scored"]
    for q, v in ((l.get("makhraj") or {}).get("neighbours") or {}).items():
        out[f"makhraj:{q}"] = v
    out["letter:duration_s"] = sum(L[i]["duration_s"] for i in idx if i in L)
    for i in idx[1:]:
        v = L.get(i)
        if v and v["kind"] == "harakah":
            out["vowel:identity:margin"] = v["identity"]["margin"]
            out["vowel:duration_s"] = v["duration_s"]
    prefix = lid.rsplit(":", 1)[0] + ":"
    for r in m["rules"]:
        if r["id"].startswith(prefix) and r["word"] == word:
            out[f"rule:{r['rule']}:status"] = r["status"]
            if r["observed_counts"] is not None:
                out[f"rule:{r['rule']}:counts"] = r["observed_counts"]
    return out


def elsewhere(m0: dict[str, Any], m1: dict[str, Any], exclude: set[str]) -> dict[str, Any]:
    """Letters OUTSIDE the unit whose identity or any characteristic verdict flipped."""
    a = {x["id"]: x for x in m0["letters"]}
    flips = []
    for x in m1["letters"]:
        y = a.get(x["id"])
        if y is None or x["id"] in exclude:
            continue
        if x["identity"]["confirmed"] != y["identity"]["confirmed"]:
            flips.append(f"{x['id']} identity")
        for h, c in x["characteristics"].items():
            if h in y["characteristics"] and c["realised"] != y["characteristics"][h]["realised"]:
                flips.append(f"{x['id']} {h}")
    return {"letters_elsewhere_flipped": len(flips), "flips": flips[:20]}


def numeric(d: dict[str, Any]) -> dict[str, float]:
    return {k: v for k, v in d.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}


# ---------------------------------------------------------------- the runner ----------------------------------

class Harness:
    def __init__(self, reciter: str) -> None:
        from app.engine import Engine
        from app.webapp import decode_upload
        from datastore.review_queue import audio_path
        self.eng, self.dec, self.path, self.rec = Engine(), decode_upload, audio_path, reciter
        self.docs = [json.loads(p.read_text()) for p in sorted((MEASURE / reciter).glob("*.json"))]
        self.rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        self.dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "app", "research_agency_lab/experiments/causal",
                                          "research_agency_lab/experiments/synthesis/world_lab.py",
                                          "research_agency_lab/experiments/letter_corpus/perturb.py"],
                                         cwd=ROOT, capture_output=True, text=True).stdout.strip())

    def wave(self, s: int, a: int) -> np.ndarray:  # type: ignore[type-arg]
        return self.dec(self.path(self.rec, s, a).read_bytes())

    def analyze(self, wave: np.ndarray, s: int, a: int) -> dict[str, Any]:  # type: ignore[type-arg]
        return self.eng.analyze(wave, [(s, a)], makhraj=True)["measurements"]

    # -- unit selection: the clearest clean instance -------------------------------------------------
    def find_letter(self, sym: str, form: str = "fatha", skip: int = 0) -> tuple[dict, int] | None:  # type: ignore[type-arg]
        from research_agency_lab.experiments.letter_corpus.assemble import letter_cells, letter_score
        c = []
        for d in self.docs:
            L = d["measurements"]["letters"]
            for sy, f, i in letter_cells(L):
                if sy == sym and f == form and not L[i]["edge"] and i + 1 < len(L) and not L[i + 1]["edge"]:
                    sc = letter_score(L, [i, i + 1])
                    if sc["clean"] and sc["weakest"] is not None:
                        c.append((sc["weakest"], d, i))
        c.sort(key=lambda x: -x[0])
        return (c[skip][1], c[skip][2]) if len(c) > skip else None

    def find_rule(self, rule: str) -> tuple[dict, dict] | None:  # type: ignore[type-arg]
        best = None
        for d in self.docs:
            L = {x["id"]: x for x in d["measurements"]["letters"]}
            for r in d["measurements"]["rules"]:
                if r["rule"] == rule and r["status"] == "pass" and r["observed_counts"] and \
                        all(x in L and not L[x]["edge"] for x in r["letters"]):
                    z = abs(r["z_masters"] or 9)
                    if best is None or z < best[0]:
                        best = (z, d, r)
        return (best[1], best[2]) if best else None

    def letter_unit(self, d: dict, i: int, span: str) -> tuple[S.Unit, list[str]]:  # type: ignore[type-arg]
        L = d["measurements"]["letters"]
        idx = [L[i]["id"], L[i + 1]["id"]]
        t0 = L[i]["onset_s"]
        t1 = (L[i]["onset_s"] + L[i]["duration_s"]) if span == "consonant" else L[i + 1]["onset_s"] + L[i + 1]["duration_s"]
        return S.Unit(self.rec, d["surah"], d["ayah"], L[i]["id"], L[i]["symbol"], "fatha", L[i]["word"],
                      "letter" if span == "consonant" else "letter+vowel", round(t0, 3), round(t1, 3),
                      "Muqri Viterbi alignment, 40 ms frames (no refinement)",
                      f"everyayah:{self.rec}/{d['surah']:03d}{d['ayah']:03d}.mp3"), idx

    def rule_unit(self, d: dict, r: dict) -> tuple[S.Unit, list[str]]:  # type: ignore[type-arg]
        L = {x["id"]: x for x in d["measurements"]["letters"]}
        us = [L[x] for x in r["letters"]]
        t0, t1 = min(u["onset_s"] for u in us), max(u["onset_s"] + u["duration_s"] for u in us)
        return S.Unit(self.rec, d["surah"], d["ayah"], us[0]["id"], us[0]["symbol"], f"rule:{r['rule']}", r["word"],
                      "rule", round(t0, 3), round(t1, 3), "Muqri Viterbi alignment, 40 ms frames (no refinement)",
                      f"everyayah:{self.rec}/{d['surah']:03d}{d['ayah']:03d}.mp3"), [u["id"] for u in us]

    # -- one unit, one or more transforms --------------------------------------------------------------
    def run_unit(self, uid: str, unit: S.Unit, idx: list[str], plan: list[tuple[str, list[dict[str, Any]]]],
                 donors: dict[str, tuple[np.ndarray, str]]) -> list[dict[str, Any]]:  # type: ignore[type-arg]
        s, a, t0, t1 = unit.surah, unit.ayah, unit.start_s, unit.end_s
        wave = self.wave(s, a)
        sha = hashlib.sha256(wave.tobytes()).hexdigest()[:16]
        m0 = self.analyze(wave, s, a)
        mb = muqri_view(m0, unit.letter_id, idx, unit.word)
        pb = PH.measure(wave, t0, t1)
        excl = set(idx)
        runs: dict[str, dict[str, Any]] = {}
        spans: list[tuple[str, np.ndarray, float, float]] = [(f"{uid}|baseline", wave, t0, t1)]  # type: ignore[type-arg]

        def run(key: str, name: str, p: dict[str, Any], w0: float = t0, w1: float = t1,
                donor: np.ndarray | None = None) -> None:  # type: ignore[type-arg]
            new, e1, impl = T.apply(wave, w0, w1, name, p, donor)
            m1 = self.analyze(new, s, a)
            runs[key] = {"name": name, "params": p, "impl": impl, "span": [w0, e1], "wave": new, "m": m1,
                         "view": muqri_view(m1, unit.letter_id, idx, unit.word),
                         "physics": PH.measure(new, w0, e1), "elsewhere": elsewhere(m0, m1, excl)}
            spans.append((f"{uid}|{key}", new, w0, e1))

        # controls
        m0b = self.analyze(wave, s, a)
        runs["determinism"] = {"name": "determinism", "params": {}, "impl": "Engine.analyze twice", "span": [t0, t1],
                               "wave": wave, "m": m0b, "view": muqri_view(m0b, unit.letter_id, idx, unit.word),
                               "physics": pb, "elsewhere": elsewhere(m0, m0b, excl)}
        for c in ("noop_splice", "noop_world", "noop_pv"):
            run(c, c, {})
        first, levels = plan[0]
        mid = levels[len(levels) // 2]
        for sh in (-0.02, 0.02):
            run(f"boundary{sh:+.2f}|{first}", first, mid, t0 + sh, t1 + sh)
        for kind, (dw, ref) in donors.items():
            run(f"{kind}|{ref}", "swap", {"donor": ref}, donor=dw)
        # transforms
        for name, lv in plan:
            for p in lv:
                run(f"{name}|{json.dumps(p, sort_keys=True)}", name, p)

        # independent cross-checks on every span (baseline and altered)
        oc, jl = PH.octave(spans), PH.julia(spans)
        dnoise_m = [S.deltas(numeric(mb), numeric(runs[k]["view"])) for k in ("determinism", "noop_splice")]
        dnoise_p = [S.deltas(numeric(pb), numeric(runs["noop_splice"]["physics"]))]
        records = []
        for key, r in runs.items():
            fam = r["name"]
            noop = NOOP_FOR.get(fam)
            nm = S.noise_floors(dnoise_m + ([S.deltas(numeric(mb), numeric(runs[noop]["view"]))] if noop else []))
            npf = S.noise_floors(dnoise_p + ([S.deltas(numeric(pb), numeric(runs[noop]["physics"]))] if noop else []),
                                 layer="physics")
            dm = S.deltas(numeric(mb), numeric(r["view"]))
            dp = S.deltas(numeric(pb), numeric(r["physics"]))
            if fam in HYPOTHESES and not key.startswith("boundary"):
                hyp = HYPOTHESES[fam](r["params"], pb, unit)
            elif fam == "swap" and key.startswith("neighbour"):
                hyp = S.Hypothesis("—", 0, None, {"identity:margin": -1}, "a neighbouring letter in the unit's place: "
                                   "identity must fall (positive control)")
            else:
                hyp = S.Hypothesis("—", 0, None, {}, f"control '{key}': nothing should move beyond the noise floor")
            cls = S.classify(dm, nm, hyp.muqri_targets)
            pv = S.verify_physics(hyp.physical_metric, hyp.physical_direction, hyp.predicted_physical_delta,
                                  dp.get(hyp.physical_metric), npf.get(hyp.physical_metric, 0.0)) \
                if hyp.physical_metric != "—" else {"status": "n/a (control)"}
            tgt = [cls.get(k) for k in hyp.muqri_targets]
            muqri_status = ("target: " + ", ".join(f"{k} {cls.get(k)}" for k in hyp.muqri_targets)) if tgt else \
                ("all stable" if all(v in (S.STABLE, S.INSUFFICIENT) for v in cls.values()) else "changes where none expected")
            collateral = {k: {"delta": dm[k], "noise": nm.get(k)} for k, v in cls.items()
                          if v == S.UNEXPECTED and k not in hyp.muqri_targets}
            flips = {k: [mb.get(k), r["view"].get(k)] for k in r["view"]
                     if (k.endswith(":realised") or k.endswith(":confirmed") or k.endswith(":status")
                         or k == "identity:competitor") and mb.get(k) != r["view"].get(k)}
            audio_rel = f"data/audio/{uid}/{key.replace('|', '__').replace(' ', '')[:120]}.wav"
            rec = S.Record(
                experiment_id=f"{uid}/{key}", schema=S.SCHEMA_VERSION, unit=unit,
                transform=S.Transform(fam, r["params"], r["impl"], f"[{r['span'][0]:.3f}, {r['span'][1]:.3f}] s"),
                hypothesis=hyp,
                requested_delta={"parameters": r["params"], "physical_direction": hyp.physical_direction,
                                 "predicted_physical_delta": hyp.predicted_physical_delta},
                physics_baseline=pb, physics_transformed=r["physics"], delta_physics=dp, physics_verification=pv,
                muqri_baseline=mb, muqri_transformed=r["view"], delta_muqri=dm, classification=cls,
                collateral={"metrics_beyond_noise": collateral, "verdict_flips_in_unit": flips, **r["elsewhere"],
                            "noise_floor_muqri": nm, "noise_floor_physics": npf},
                crosscheck={"octave": {"baseline": oc.get(f"{uid}|baseline"), "altered": oc.get(f"{uid}|{key}")},
                            "julia": {"baseline": jl.get(f"{uid}|baseline"), "altered": jl.get(f"{uid}|{key}")}},
                controls=[k for k in runs if not any(k.startswith(f) for f in HYPOTHESES)],
                reproducibility={"code_revision": self.rev, "code_dirty": self.dirty, "audio_sha256_16": sha,
                                 "source": unit.source_audio, "unit": f"{unit.letter_id} {unit.form} [{t0}, {t1}]",
                                 "transform": fam, "parameters": r["params"], "engine": "app.engine.Engine (makhraj=True)",
                                 "model": "obadx/muaalem-model-v3_2", "python": platform.python_version(),
                                 "versions": _versions(), "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                verification_status=f"physics {pv['status']}; muqri {muqri_status}",
                transformed_audio=audio_rel)
            _save_audio(HERE / audio_rel, r["wave"], r["span"])
            records.append(rec.to_dict())
            print(f"  {uid} {key}: {rec.verification_status}", flush=True)
        return records


def _versions() -> dict[str, str]:
    import importlib.metadata as md
    out = {}
    for p in ("numpy", "librosa", "pyworld", "praat-parselmouth", "torch", "quran-muaalem"):
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = "?"
    return out


def _save_audio(p: Path, wave: np.ndarray, span: list[float]) -> None:  # type: ignore[type-arg]
    import soundfile as sf
    p.parent.mkdir(parents=True, exist_ok=True)
    lo, hi = max(0.0, span[0] - 0.4), min(len(wave) / SR, span[1] + 0.4)
    sf.write(p, wave[int(lo * SR):int(hi * SR)], SR, subtype="PCM_16")


# ---------------------------------------------------------------- artifacts ------------------------------------

def natural_variation(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Per unit: |delta| each Muqri metric showed when the unit was replaced by the SAME letter from another
    ayah of the same reciter -- the size of ordinary instance-to-instance variation, a second reference
    scale for collateral changes (none for units without a same-letter donor)."""
    out: dict[str, dict[str, float]] = {}
    for r in records:
        if "/same-letter|" in r["experiment_id"]:
            out[r["experiment_id"].split("/", 1)[0]] = {k: abs(v) for k, v in r["delta_muqri"].items() if v is not None}
    return out


def causal_table(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    nat = natural_variation(records)
    rows = []
    for r in records:
        h, pv = r["hypothesis"], r["physics_verification"]
        coll = r["collateral"]["metrics_beyond_noise"]
        targets = h["muqri_targets"] or {"(none: all must stay)": 0}
        for mq, exp in targets.items():
            d = r["delta_muqri"].get(mq)
            rows.append({"experiment": r["experiment_id"], "transform": r["transform"]["name"],
                         "parameters": r["transform"]["parameters"], "physical_metric": h["physical_metric"],
                         "physical_delta": r["delta_physics"].get(h["physical_metric"]), "predicted_physical_delta":
                         h["predicted_physical_delta"], "physics_status": pv.get("status"),
                         "muqri_metric": mq, "muqri_delta": d, "muqri_noise_floor":
                         r["collateral"]["noise_floor_muqri"].get(mq), "expected_direction": exp,
                         "observed_direction": None if d is None else int(np.sign(d)),
                         "muqri_classification": r["classification"].get(mq, "n/a"),
                         "collateral_count": len(coll), "collateral": {k: v["delta"] for k, v in coll.items()},
                         "collateral_beyond_same_letter_variation": (
                             sorted(k for k, v in coll.items() if abs(v["delta"]) > nat[uid].get(k, float("inf")))
                             if (uid := r["experiment_id"].split("/", 1)[0]) in nat else "no same-letter control"),
                         "letters_elsewhere_flipped": r["collateral"]["letters_elsewhere_flipped"],
                         "verification_status": r["verification_status"]})
    return rows


def parity(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Python (Praat/numpy) vs Julia vs Octave on the same spans, and Julia's re-derived deltas vs Python's."""
    rows = []
    pairs = [("voiced_fraction", "voiced_fraction", "voiced_frac"), ("nasal_ratio_db", "nasal_ratio_db", None),
             ("rms_db", "rms_db", None), ("span_duration_s", "duration_s", None), ("f0_hz", None, "f0_hz"),
             ("hnr_db", None, "hnr_db"), ("f1_hz", None, "f1_hz"), ("f2_hz", None, "f2_hz")]
    for r in records:
        for py, jl, oc in pairs:
            b = {"python": r["physics_baseline"].get(py), "julia": (r["crosscheck"]["julia"]["baseline"] or {}).get(jl) if jl else None,
                 "octave": (r["crosscheck"]["octave"]["baseline"] or {}).get(oc) if oc else None}
            t = {"python": r["physics_transformed"].get(py), "julia": (r["crosscheck"]["julia"]["altered"] or {}).get(jl) if jl else None,
                 "octave": (r["crosscheck"]["octave"]["altered"] or {}).get(oc) if oc else None}
            d = {k: (round(t[k] - b[k], 4) if isinstance(t[k], (int, float)) and isinstance(b[k], (int, float)) else None)
                 for k in b}
            dirs = {k: int(np.sign(v)) for k, v in d.items() if v is not None and abs(v) > 1e-9}
            rows.append({"experiment": r["experiment_id"], "metric": py, "baseline": b, "altered": t, "delta": d,
                         "direction_agreement": (len(set(dirs.values())) <= 1) if len(dirs) >= 2 else None})
    return {"rows": rows, "disagreements": [x for x in rows if x["direction_agreement"] is False]}


def julia_table_parity(records: list[dict[str, Any]]) -> dict[str, Any]:
    import tempfile
    slim = [{"experiment_id": r["experiment_id"], "physics_baseline": numeric(r["physics_baseline"]),
             "physics_transformed": numeric(r["physics_transformed"]), "muqri_baseline": numeric(r["muqri_baseline"]),
             "muqri_transformed": numeric(r["muqri_transformed"])} for r in records]
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "r.json").write_text(json.dumps(slim))
        p = subprocess.run([str(PH.JULIA), f"--project={PH.JULIA_PROJECT}", str(PH.XCHECK), "table",
                            str(Path(d) / "r.json"), str(Path(d) / "o.json")], capture_output=True, text=True, timeout=900)
        if p.returncode != 0:
            return {"error": p.stderr[-400:]}
        jt = json.loads((Path(d) / "o.json").read_text())
    by = {r["experiment_id"]: r for r in records}
    mism = []
    for j in jt:
        r = by[j["experiment_id"]]
        py = (r["delta_physics"] if j["layer"] == "physics" else r["delta_muqri"]).get(j["metric"])
        if (py is None) != (j["delta"] is None) or (py is not None and abs(py - j["delta"]) > 1e-3):
            mism.append({"experiment": j["experiment_id"], "layer": j["layer"], "metric": j["metric"],
                         "python_delta": py, "julia_delta": j["delta"]})
    return {"deltas_compared": len(jt), "mismatches": mism}


def stability_report(records: list[dict[str, Any]]) -> str:
    out = ["# Causal smoke test: stability report", ""]
    for r in records:
        h = r["hypothesis"]
        cls = r["classification"]
        out.append(f"## {r['experiment_id']}")
        out.append(f"- TRANSFORM: {r['transform']['name']} {json.dumps(r['transform']['parameters'], ensure_ascii=False)} "
                   f"({r['transform']['implementation']})")
        out.append(f"- PHYSICS: {h['physical_metric']} requested direction {h['physical_direction']:+d}, predicted "
                   f"{h['predicted_physical_delta']}, measured {r['delta_physics'].get(h['physical_metric'])} -> "
                   f"{r['physics_verification'].get('status')}")
        for k, e in h["muqri_targets"].items():
            out.append(f"- TARGET {k}: EXPECTED {'increase' if e > 0 else 'decrease'}; OBSERVED delta "
                       f"{r['delta_muqri'].get(k)} (noise {r['collateral']['noise_floor_muqri'].get(k)}) -> {cls.get(k)}")
        unrel = [f"{k} = {v}" for k, v in sorted(cls.items()) if k not in h["muqri_targets"]
                 and (k.startswith(("identity:margin", "head:", "letter:", "vowel:", "rule:")))]
        out.append("- UNRELATED: " + "; ".join(unrel))
        nat = natural_variation(records).get(r["experiment_id"].split("/", 1)[0])
        if nat is not None:
            coll = r["collateral"]["metrics_beyond_noise"]
            out.append("- COLLATERAL beyond the no-op floor but within same-letter variation: "
                       + (", ".join(k for k, v in coll.items() if abs(v["delta"]) <= nat.get(k, float("inf"))) or "none")
                       + "; beyond same-letter variation: "
                       + (", ".join(k for k, v in coll.items() if abs(v["delta"]) > nat.get(k, float("inf"))) or "none"))
        out.append(f"- ELSEWHERE IN THE AYAH: {r['collateral']['letters_elsewhere_flipped']} verdict flips "
                   + (str(r['collateral']['flips']) if r['collateral']['flips'] else ""))
        out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------- the smoke matrix -----------------------------

def smoke() -> int:
    h = Harness("Husary_128kbps")
    recs: list[dict[str, Any]] = []
    # 1. zay with fatha (jahr fricative): voicing and duration, on the consonant
    z = h.find_letter("ز")
    zs, ss = h.find_letter("ز", skip=1), h.find_letter("س")
    if z and zs and ss:
        u, idx = h.letter_unit(*z, span="consonant")
        donors = {}
        for kind, (d, i) in (("same-letter", zs), ("neighbour", ss)):
            L = d["measurements"]["letters"]
            w = h.wave(d["surah"], d["ayah"])
            donors[kind] = (w[int(L[i]["onset_s"] * SR):int((L[i]["onset_s"] + L[i]["duration_s"]) * SR)],
                            f"{L[i]['symbol']} {d['surah']}:{d['ayah']} {L[i]['id']}")
        recs += h.run_unit("zay_fatha", u, idx, [("voicing", [{"level": 0.6}, {"level": 0.3}, {"level": 0.0}]),
                                                 ("duration", [{"k": 0.5}, {"k": 1.5}, {"k": 2.0}])], donors)
    # 2. sad with fatha (heavy, whistling): formant warp and F0, on the consonant + vowel
    sd = h.find_letter("ص")
    sd2, sn = h.find_letter("ص", skip=1), h.find_letter("س")
    if sd and sd2 and sn:
        u, idx = h.letter_unit(*sd, span="letter+vowel")
        donors = {}
        for kind, (d, i) in (("same-letter", sd2), ("neighbour", sn)):
            L = d["measurements"]["letters"]
            w = h.wave(d["surah"], d["ayah"])
            donors[kind] = (w[int(L[i]["onset_s"] * SR):int((L[i + 1]["onset_s"] + L[i + 1]["duration_s"]) * SR)],
                            f"{L[i]['symbol']}{L[i + 1]['symbol']} {d['surah']}:{d['ayah']} {L[i]['id']}")
        recs += h.run_unit("sad_fatha", u, idx, [("formant", [{"a": 0.9}, {"a": 0.8}, {"a": 0.7}]),
                                                 ("f0", [{"semitones": -3}, {"semitones": 3}, {"semitones": 6}])], donors)
    # 3. ghunnah mushaddadah: nasality and duration, on the rule's letters (no same-letter / neighbour swap)
    g = h.find_rule("ghunnah")
    if g:
        u, idx = h.rule_unit(*g)
        recs += h.run_unit("ghunnah", u, idx, [("nasal", [{"g_db": -6}, {"g_db": -12}, {"g_db": -18}]),
                                               ("duration", [{"k": 0.5}, {"k": 1.5}, {"k": 2.0}])], {})
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "records.json").write_text(json.dumps(recs, ensure_ascii=False, indent=1, default=float))
    return tables(recs)


def tables(recs: list[dict[str, Any]] | None = None) -> int:
    """(Re)build the committed artifacts from the stored records -- no audio, no engine run."""
    recs = recs if recs is not None else json.loads((DATA / "records.json").read_text())
    RESULTS.mkdir(parents=True, exist_ok=True)
    table = causal_table(recs)
    (RESULTS / "causal_table.json").write_text(json.dumps(table, ensure_ascii=False, indent=1, default=float))
    par = {"spans": parity(recs), "julia_rederived_deltas": julia_table_parity(recs)}
    (RESULTS / "parity.json").write_text(json.dumps(par, ensure_ascii=False, indent=1, default=float))
    (RESULTS / "stability.md").write_text(stability_report(recs))
    print(f"{len(recs)} records; table {len(table)} rows; span disagreements {len(par['spans']['disagreements'])}; "
          f"julia delta mismatches {len(par['julia_rederived_deltas'].get('mismatches', []))}")
    return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["smoke"]:
        sys.exit(smoke())
    if sys.argv[1:] == ["tables"]:
        sys.exit(tables())
    print(__doc__)
