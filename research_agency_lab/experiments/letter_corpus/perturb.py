#!/usr/bin/env python3
"""The perturbation lab: a master's letter or rule altered one characteristic at a time, inside its own
ayah, and measured again by the whole engine.

If a controlled change to real audio moves exactly the measurement it targets -- and nothing else --
then (a) the engine is shown to respond to that characteristic, with a measured dose-response, and (b)
the same change is a generator of labelled data: every clean master instance becomes a known mistake
(or, run the other way, a learner's mistake becomes a corrected reading). See DESIGN.md.

Each case: find an instance in the reciter's measurements, alter its span in the ayah's waveform
(10 ms crossfades at the seams), run Engine.analyze on the altered ayah, and compare the target
letter's checks and the rule verdicts before and after.

  echo_off     qalqalah   the release echo (ڇ) faded to -30 dB                  qalqla head, qalqalah rule
  stretch x k  length     the span time-stretched by k (phase vocoder, pitch kept)  counts, short / long
  devoice      hams/jahr  the letter high-passed at 1.5 kHz: no voicing bar left   hams_or_jahr head
  formant +/-  weight     the letter + vowel resampled by 2^(s/12), then stretched
                          back to length: every formant moves s semitones             tafkheem_or_taqeeq
  delete       identity   the letter cut out                                      identity (heard ∅)
  swap         identity   the letter replaced by the same reciter's neighbour     identity, itbaq, makhraj
                          letter in the same form, from another ayah -- the consonant alone, and the
                          whole syllable (consonant + vowel)

    .venv/bin/python research_agency_lab/experiments/letter_corpus/perturb.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

HERE = Path(__file__).parent
DATA = HERE / "data"
SR = 16000
OUT = DATA / "clips" / "perturb"


def splice(w: np.ndarray, t0: float, t1: float, seg: np.ndarray, xf: float = 0.01) -> np.ndarray:  # type: ignore[type-arg]
    a, b, n = int(t0 * SR), int(t1 * SR), int(xf * SR)
    seg = np.asarray(seg, dtype=np.float32).copy()
    if len(seg) > 2 * n and n > 0:
        ramp = np.linspace(0, 1, n, dtype=np.float32)
        seg[:n] = seg[:n] * ramp + w[a:a + n] * (1 - ramp) if b - a >= n else seg[:n]
        tail = w[b - n:b] if b - a >= n else seg[-n:]
        seg[-n:] = seg[-n:] * ramp[::-1] + tail * (1 - ramp[::-1])
    return np.concatenate([w[:a], seg, w[b:]]).astype(np.float32)


def stretch(x: np.ndarray, k: float) -> np.ndarray:  # type: ignore[type-arg]
    import librosa
    return librosa.effects.time_stretch(x.astype(np.float32), rate=1.0 / k)


def devoice(x: np.ndarray) -> np.ndarray:  # type: ignore[type-arg]
    from scipy.signal import butter, sosfiltfilt
    y = sosfiltfilt(butter(6, 1500, btype="highpass", fs=SR, output="sos"), x)
    return (y * (np.sqrt(np.mean(x ** 2)) / (np.sqrt(np.mean(y ** 2)) + 1e-9))).astype(np.float32)


def formant(x: np.ndarray, semis: float) -> np.ndarray:  # type: ignore[type-arg]
    import librosa
    f = 2 ** (semis / 12)
    y = librosa.resample(x.astype(np.float32), orig_sr=SR, target_sr=int(SR / f))   # every frequency x f
    return librosa.util.fix_length(librosa.effects.time_stretch(y, rate=len(y) / len(x)), size=len(x))


def echo_off(x: np.ndarray) -> np.ndarray:  # type: ignore[type-arg]
    return (x * 10 ** (-30 / 20)).astype(np.float32)


def letter_view(m: dict[str, Any], lid: str) -> dict[str, Any]:
    l = next((x for x in m["letters"] if x["id"] == lid), None)
    if l is None:
        return {}
    return {"identity": (l["identity"]["confirmed"], l["identity"]["competitor"], l["identity"]["margin"]),
            "heads": {h: (c["realised"], c["observed"], c["margin"]) for h, c in l["characteristics"].items()},
            "makhraj": (l.get("makhraj") or {}).get("neighbours") or {}}


def rule_view(m: dict[str, Any], word: int, prefix: str) -> list[tuple[str, str, Any]]:
    return [(r["rule"], r["status"], r["observed_counts"]) for r in m["rules"]
            if r["id"].startswith(prefix) and r["word"] == word]


def diff(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    out = []
    if before.get("identity") and after.get("identity"):
        (c0, _q0, m0), (c1, q1, m1) = before["identity"], after["identity"]
        if c0 != c1 or (m0 is not None and m1 is not None and abs(m1 - m0) > 2):
            out.append(f"identity {m0} -> {m1}" + ("" if c1 else f" (heard {q1})"))
    for h, (r0, _o0, m0) in before.get("heads", {}).items():
        r1, o1, m1 = after.get("heads", {}).get(h, (None, None, None))
        if m1 is None:
            continue
        if r0 != r1 or abs((m1 or 0) - (m0 or 0)) > 2:
            out.append(f"{h} {m0} -> {m1}" + ("" if r1 else f" (heard {o1})"))
    for q, m0 in before.get("makhraj", {}).items():
        m1 = after.get("makhraj", {}).get(q)
        if m1 is not None and (m1 <= 0 < m0 or abs(m1 - m0) > 3):
            out.append(f"makhraj vs {q} {m0} -> {m1}")
    return out


class Lab:
    def __init__(self, reciter: str) -> None:
        from app.engine import Engine
        from app.webapp import decode_upload
        from datastore.review_queue import audio_path
        self.eng, self.dec, self.path, self.rec = Engine(), decode_upload, audio_path, reciter
        self.docs = [json.loads(p.read_text()) for p in sorted((DATA / "measure" / reciter).glob("*.json"))]
        self.cases: list[dict[str, Any]] = []

    def wave(self, s: int, a: int) -> np.ndarray:  # type: ignore[type-arg]
        return self.dec(self.path(self.rec, s, a).read_bytes())

    def find_letter(self, sym: str, form: str, echo: bool = False) -> tuple[dict, int] | None:  # type: ignore[type-arg]
        from research_agency_lab.experiments.letter_corpus.assemble import letter_cells, letter_score
        best = None
        for d in self.docs:
            L = d["measurements"]["letters"]
            for c, f, i in letter_cells(L):
                if c != sym or f != form or L[i]["edge"]:
                    continue
                if echo and not (i + 1 < len(L) and L[i + 1]["symbol"] == "ڇ"):
                    continue
                w = letter_score(L, [i])["weakest"]
                if w is not None and (best is None or w > best[2]):
                    best = (d, i, w)
        return (best[0], best[1]) if best else None

    def find_rule(self, rule: str, status: str = "pass") -> tuple[dict, dict] | None:  # type: ignore[type-arg]
        for d in self.docs:
            for r in d["measurements"]["rules"]:
                if r["rule"] == rule and r["status"] == status and r["observed_counts"]:
                    return d, r
        return None

    def run(self, title: str, how: str, d: dict, t0: float, t1: float, variants: list[tuple[str, Any]],  # type: ignore[type-arg]
            lid: str | None, word: int) -> None:
        import soundfile as sf
        s, a = d["surah"], d["ayah"]
        w = self.wave(s, a)
        prefix = f"{s}:{a}:"
        base = d["measurements"]
        lo, hi = max(0.0, t0 - 0.35), min(len(w) / SR, t1 + 0.35)
        OUT.mkdir(parents=True, exist_ok=True)
        slug = f"{len(self.cases):02d}"
        sf.write(OUT / f"{slug}_orig.wav", w[int(lo * SR):int(hi * SR)], SR, subtype="PCM_16")
        versions = [{"label": "original", "clip": f"perturb/{slug}_orig.wav",
                     "summary": "; ".join(f"{r} {st} {c}" for r, st, c in rule_view(base, word, prefix))
                     or "as recited"}]
        for label, fn in variants:
            seg = fn(w[int(t0 * SR):int(t1 * SR)])
            w2 = splice(w, t0, t1, seg)
            m2 = self.eng.analyze(w2, [(s, a)], makhraj=True)["measurements"]
            changed = diff(letter_view(base, lid), letter_view(m2, lid)) if lid else []
            r0, r1 = rule_view(base, word, prefix), rule_view(m2, word, prefix)
            if r0 != r1:
                changed.append("rules " + ", ".join(f"{x[0]} {x[1]} {x[2]}" for x in r0) + " -> "
                               + ", ".join(f"{x[0]} {x[1]} {x[2]}" for x in r1))
            d_len = (len(seg) - (int(t1 * SR) - int(t0 * SR))) / SR
            name = f"perturb/{slug}_{len(versions)}.wav"
            sf.write(OUT / Path(name).name, w2[int(lo * SR):int((hi + d_len) * SR)], SR, subtype="PCM_16")
            versions.append({"label": label, "clip": name, "summary": "; ".join(changed) or "no measurement moved"})
            print(f"  {title} / {label}: {versions[-1]['summary']}", flush=True)
        self.cases.append({"title": title, "how": how, "reciter": self.rec, "ref": f"{s}:{a}", "versions": versions})


def main() -> None:
    import warnings
    warnings.filterwarnings("ignore")
    lab = Lab("Husary_128kbps")
    # qalqalah: the echo of a sakin qalqalah letter
    for sym in "قدب":
        hit = lab.find_letter(sym, "sakin", echo=True)
        if hit:
            d, i = hit
            L = d["measurements"]["letters"]
            e = L[i + 1]
            lab.run(f"Qalqalah on {sym} sakin — the echo taken away", "The release echo (ڇ) after the closure "
                    "faded to -30 dB; the closure itself untouched.", d, e["onset_s"], e["onset_s"] + e["duration_s"],
                    [("echo -30 dB", echo_off)], L[i]["id"], L[i]["word"])
    # length: madd and ghunnah, both directions
    for rule, ks in (("madd_tabii", (0.5, 2.0)), ("madd_munfasil", (0.5, 1.5)), ("ghunnah", (0.4, 1.8))):
        hit = lab.find_rule(rule)
        if hit:
            d, r = hit
            ls = {x["id"]: x for x in d["measurements"]["letters"]}
            us = [ls[x] for x in r["letters"] if x in ls]
            t0, t1 = min(u["onset_s"] for u in us), max(u["onset_s"] + u["duration_s"] for u in us)
            lab.run(f"{rule.replace('_', ' ')} — held shorter and longer", "The rule's span time-stretched by a "
                    "phase vocoder (pitch kept), spliced back with 10 ms crossfades.", d, t0, t1,
                    [(f"x{k}", (lambda k: lambda x: stretch(x, k))(k)) for k in ks], None, r["word"])
    # voicing: a jahr fricative devoiced
    for sym in "زذ":
        hit = lab.find_letter(sym, "fatha")
        if hit:
            d, i = hit
            l = d["measurements"]["letters"][i]
            lab.run(f"{sym} with fatha — the voice taken out (jahr to hams)", "The consonant high-passed at 1.5 kHz: "
                    "the voicing bar gone, the frication kept, the level restored.", d, l["onset_s"],
                    l["onset_s"] + l["duration_s"], [("devoiced", devoice)], l["id"], l["word"])
    # weight: formants moved on a heavy and a light letter with its vowel
    for sym, semis in (("ص", (+2.0, +4.0)), ("س", (-2.0, -4.0)), ("ط", (+3.0,))):
        hit = lab.find_letter(sym, "fatha")
        if hit:
            d, i = hit
            L = d["measurements"]["letters"]
            t0, t1 = L[i]["onset_s"], L[i + 1]["onset_s"] + L[i + 1]["duration_s"]
            lab.run(f"{sym} with fatha — formants moved {'up (lighter)' if semis[0] > 0 else 'down (heavier)'}",
                    "Letter and vowel resampled by 2^(s/12) and stretched back to length: every formant moves s "
                    "semitones (the voice's pitch with them).", d, t0, t1,
                    [(f"{s:+g} semitones", (lambda s: lambda x: formant(x, s))(s)) for s in semis], L[i]["id"], L[i]["word"])
    # identity: a letter deleted, and a letter swapped for its neighbour from the same reciter
    hit = lab.find_letter("ل", "sakin")
    if hit:
        d, i = hit
        l = d["measurements"]["letters"][i]
        lab.run("ل sakin — deleted", "The letter's span cut out (the vowels either side joined).", d, l["onset_s"],
                l["onset_s"] + l["duration_s"], [("deleted", lambda x: x[:int(0.02 * SR)])], l["id"], l["word"])
    for sym, other in (("ط", "ت"), ("ض", "د"), ("ص", "س"), ("ق", "ك")):
        hit, src = lab.find_letter(sym, "fatha"), lab.find_letter(other, "fatha")
        if hit and src:
            d, i = hit
            l = d["measurements"]["letters"][i]
            sd, si = src
            sl = sd["measurements"]["letters"][si]
            donor = lab.wave(sd["surah"], sd["ayah"])[int(sl["onset_s"] * SR):int((sl["onset_s"] + sl["duration_s"]) * SR)]
            lab.run(f"{sym} with fatha — swapped for the reciter's own {other}", f"The {sym} replaced by the same "
                    f"reciter's {other} with fatha from {sd['surah']}:{sd['ayah']}; the vowel after it untouched.", d,
                    l["onset_s"], l["onset_s"] + l["duration_s"], [(f"{sym} → {other}", lambda x, dn=donor: dn)],
                    l["id"], l["word"])
    # the same swaps, whole syllable: the consonant AND its vowel (the transition that carries the
    # letter's colour), from the reciter's own neighbour syllable
    for sym, other in (("ط", "ت"), ("ض", "د"), ("ص", "س"), ("ق", "ك")):
        hit, src = lab.find_letter(sym, "fatha"), lab.find_letter(other, "fatha")
        if hit and src:
            d, i = hit
            L = d["measurements"]["letters"]
            sd, si = src
            SL = sd["measurements"]["letters"]
            donor = lab.wave(sd["surah"], sd["ayah"])[int(SL[si]["onset_s"] * SR):
                                                      int((SL[si + 1]["onset_s"] + SL[si + 1]["duration_s"]) * SR)]
            lab.run(f"{sym}َ — the whole syllable swapped for the reciter's own {other}َ", f"The consonant and its "
                    f"fatha replaced by the same reciter's {other}َ from {sd['surah']}:{sd['ayah']}.", d,
                    L[i]["onset_s"], L[i + 1]["onset_s"] + L[i + 1]["duration_s"],
                    [(f"{sym}َ → {other}َ", lambda x, dn=donor: dn)], L[i]["id"], L[i]["word"])
    # the control: the same syllable from another ayah of the same reciter. A splice that moves nothing
    # here shows the swaps above are heard for the letter, not for the seam
    for sym in "طضصق":
        hits = []
        for d in lab.docs:
            from research_agency_lab.experiments.letter_corpus.assemble import letter_cells
            L = d["measurements"]["letters"]
            hits += [(d, i) for c, f, i in letter_cells(L) if c == sym and f == "fatha" and not L[i]["edge"]]
        if len(hits) >= 2:
            (d, i), (sd, si) = hits[0], hits[1]
            L, SL = d["measurements"]["letters"], sd["measurements"]["letters"]
            donor = lab.wave(sd["surah"], sd["ayah"])[int(SL[si]["onset_s"] * SR):
                                                      int((SL[si + 1]["onset_s"] + SL[si + 1]["duration_s"]) * SR)]
            lab.run(f"Control: {sym}َ swapped for the reciter's own {sym}َ from another ayah", "The same splice as "
                    "the swaps, with the same syllable: nothing should move.", d, L[i]["onset_s"],
                    L[i + 1]["onset_s"] + L[i + 1]["duration_s"], [(f"{sym}َ → {sym}َ", lambda x, dn=donor: dn)],
                    L[i]["id"], L[i]["word"])
    # the other direction: a fast reciter's short madd stretched into the band
    fast = Lab("Abdurrahmaan_As-Sudais_192kbps")
    if fast.docs:
        hit = fast.find_rule("madd_munfasil", "short") or fast.find_rule("madd_tabii", "short")
        if hit:
            d, r = hit
            ls = {x["id"]: x for x in d["measurements"]["letters"]}
            us = [ls[x] for x in r["letters"] if x in ls]
            t0, t1 = min(u["onset_s"] for u in us), max(u["onset_s"] + u["duration_s"] for u in us)
            want = (r["expected_counts"][0] + r["expected_counts"][1]) / 2 / (r["observed_counts"] or 1)
            fast.run(f"Improving: a fast reciter's short {r['rule'].replace('_', ' ')} stretched into the band",
                     f"Sudais held it {r['observed_counts']} counts; the span stretched x{want:.2f} toward the "
                     "band's centre.", d, t0, t1, [(f"x{want:.2f}", lambda x, k=want: stretch(x, k))], None, r["word"])
    cases = lab.cases + fast.cases
    (DATA / "perturb.json").write_text(json.dumps({
        "about": "A master's letter or rule altered one characteristic at a time inside its own ayah, and the whole "
                 "engine run again. Each row says which measurements moved (margins in nats, lengths in counts): "
                 "the target should move, nothing else should.", "cases": cases}, ensure_ascii=False, indent=1))
    print(f"{len(cases)} cases -> {DATA / 'perturb.json'}")


if __name__ == "__main__" and "--sweep" not in sys.argv:
    main()


# -- the sweep: every letter against its neighbours, with controls -----------------------------------

def _syllables(docs: list[dict[str, Any]], sym: str, n: int) -> list[tuple[dict, int]]:  # type: ignore[type-arg]
    """Up to n clean, non-edge instances of `sym` with fatha, clearest first."""
    from research_agency_lab.experiments.letter_corpus.assemble import letter_cells, letter_score
    out = []
    for d in docs:
        L = d["measurements"]["letters"]
        for c, f, i in letter_cells(L):
            if c == sym and f == "fatha" and not L[i]["edge"] and not L[i + 1]["edge"]:
                sc = letter_score(L, [i, i + 1])
                if sc["clean"] and sc["weakest"] is not None:
                    out.append((sc["weakest"], d["surah"], d["ayah"], i, d))
    out.sort(key=lambda x: -x[0])
    return [(x[4], x[3]) for x in out[:n]]


_W = None


def _sweep_case(job: dict[str, Any]) -> dict[str, Any]:
    import warnings
    warnings.filterwarnings("ignore")
    global _W
    if _W is None:
        _W = Lab.__new__(Lab)
        from app.engine import Engine
        from app.webapp import decode_upload
        from datastore.review_queue import audio_path
        _W.eng, _W.dec, _W.path, _W.rec = Engine(), decode_upload, audio_path, job["reciter"]
    w = _W.wave(job["s"], job["a"])
    dw = _W.wave(job["ds"], job["da"])
    donor = dw[int(job["d0"] * SR):int(job["d1"] * SR)]
    m2 = _W.eng.analyze(splice(w, job["t0"], job["t1"], donor), [(job["s"], job["a"])], makhraj=True)["measurements"]
    after = next((x for x in m2["letters"] if x["id"] == job["lid"]), None)
    if after is None:
        return {**job, "error": "letter not found after splice"}
    heads = {h: c["realised"] for h, c in after["characteristics"].items()}
    return {"reciter": job["reciter"], "letter": job["letter"], "donor": job["donor"], "control": job["control"],
            "ref": f"{job['s']}:{job['a']}", "donor_ref": f"{job['ds']}:{job['da']}", "should_flip": job["should_flip"],
            "identity_confirmed": after["identity"]["confirmed"], "heard": after["identity"]["competitor"],
            "identity_margin": after["identity"]["margin"], "heads_realised": heads,
            "makhraj": (after.get("makhraj") or {}).get("neighbours") or {}}


def sweep(reciter: str, per: int, workers: int) -> None:
    """Syllable swaps for every letter x neighbour (`per` instances each) and same-letter controls; the
    engine's recall per characteristic head and its false alarms, from real master audio alone."""
    from concurrent.futures import ProcessPoolExecutor

    from app.letters import CONSONANTS, NEIGHBOURS
    docs = [json.loads(p.read_text()) for p in sorted((DATA / "measure" / reciter).glob("*.json"))]
    inst = {c: _syllables(docs, c, per + 1) for c in CONSONANTS}
    jobs = []
    for x in CONSONANTS:
        targets = inst[x][:per]
        if not targets:
            continue
        for y in [*NEIGHBOURS.get(x, ()), x]:
            control = y == x
            donors = inst[y][per:per + 1] if control else inst[y][:1]
            if not donors:
                continue
            dd, di = donors[0]
            DL = dd["measurements"]["letters"]
            for d, i in (targets if not control else targets[:per]):
                L = d["measurements"]["letters"]
                should = [] if control else sorted(h for h, c in L[i]["characteristics"].items()
                                                    if c["expected"] != DL[di]["characteristics"].get(h, {}).get("expected"))
                jobs.append({"reciter": reciter, "letter": x, "donor": y, "control": control, "s": d["surah"],
                             "a": d["ayah"], "lid": L[i]["id"], "t0": L[i]["onset_s"],
                             "t1": L[i + 1]["onset_s"] + L[i + 1]["duration_s"], "ds": dd["surah"], "da": dd["ayah"],
                             "d0": DL[di]["onset_s"], "d1": DL[di + 1]["onset_s"] + DL[di + 1]["duration_s"],
                             "should_flip": should})
    print(f"{len(jobs)} splices ({sum(j['control'] for j in jobs)} controls) for {reciter}", flush=True)
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        for k, r in enumerate(ex.map(_sweep_case, jobs, chunksize=2), 1):
            rows.append(r)
            if k % 25 == 0:
                print(f"  {k}/{len(jobs)}", flush=True)
    rows = [r for r in rows if "error" not in r]
    swaps, ctrl = [r for r in rows if not r["control"]], [r for r in rows if r["control"]]
    heads = sorted({h for r in rows for h in r["heads_realised"]})
    per_head = {}
    for h in heads:
        should = [r for r in swaps if h in r["should_flip"] and h in r["heads_realised"]]
        shared = [r for r in swaps if h not in r["should_flip"] and h in r["heads_realised"]]
        cs = [r for r in ctrl if h in r["heads_realised"]]
        per_head[h] = {"swaps_that_should_flip": len(should),
                       "recall": round(sum(not r["heads_realised"][h] for r in should) / len(should), 3) if should else None,
                       "held_when_shared": round(sum(r["heads_realised"][h] for r in shared) / len(shared), 3) if shared else None,
                       "control_false_alarm": round(sum(not r["heads_realised"][h] for r in cs) / len(cs), 3) if cs else None}
    ident = {"swaps": len(swaps), "heard_as_the_donor": sum((not r["identity_confirmed"]) and r["heard"] == r["donor"] for r in swaps),
             "not_confirmed": sum(not r["identity_confirmed"] for r in swaps),
             "makhraj_lost_to_donor": sum((r["makhraj"].get(r["donor"]) is not None and r["makhraj"][r["donor"]] <= 0) for r in swaps),
             "makhraj_tested": sum(r["donor"] in r["makhraj"] for r in swaps),
             "controls": len(ctrl), "controls_not_confirmed": sum(not r["identity_confirmed"] for r in ctrl),
             "controls_any_head_flipped": sum(not all(r["heads_realised"].values()) for r in ctrl)}
    by_pair: dict[str, list[int]] = {}
    for r in swaps:
        k = f"{r['letter']}→{r['donor']}"
        by_pair.setdefault(k, [0, 0])
        by_pair[k][0] += (not r["identity_confirmed"]) or (r["makhraj"].get(r["donor"], 1) <= 0)
        by_pair[k][1] += 1
    out = {"reciter": reciter, "identity": ident, "per_head": per_head,
           "pairs_caught": {k: f"{a}/{b}" for k, (a, b) in sorted(by_pair.items())}, "rows": rows}
    (DATA / f"sweep_{reciter}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, ensure_ascii=False, indent=1))


if __name__ == "__main__" and "--sweep" in sys.argv:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--reciter", default="Husary_128kbps")
    ap.add_argument("--per", type=int, default=3)
    ap.add_argument("--workers", type=int, default=3)
    a_ = ap.parse_args()
    sweep(a_.reciter, a_.per, a_.workers)
