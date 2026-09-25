#!/usr/bin/env python3
"""Characteristic transforms on real letters through the WORLD vocoder, measured by the engine.

WORLD splits a window of speech into f0 (the voice's pitch; 0 = unvoiced), the spectral envelope sp
(the vocal tract: formants, nasal resonance, frication shape) and the aperiodicity ap (how much of each
band is noise rather than harmonics). Each characteristic gets its own knob on the frames of one letter
(and, where the characteristic lives in the transition, the vowel after it):

  null        analysis -> resynthesis, nothing changed: the control; nothing may move
  devoice d   ap -> ap + d (1 - ap), f0 -> 0 at d = 1 on the consonant: jahr toward hams
  voice d     f0 carried through the consonant from its neighbours, ap -> (1 - d) ap: hams toward jahr
  warp a      the envelope read at f * a on consonant + vowel onset (tapered over the vowel):
              a < 1 moves every formant up (lighter), a > 1 down (heavier) -- tafkheem / itbaq
  nasal g     a nasal pole near 250 Hz (+g dB) and an anti-resonance near 900 Hz (-g dB) on the
              letter and vowel onset: ghunnah added (g < 0 removes)

A window of +-0.3 s around the letter is analysed, changed, resynthesised and spliced back (10 ms
crossfades), and Engine.analyze runs on the whole ayah; the target letter's heads are compared.

    .venv/bin/python research_agency_lab/experiments/synthesis/world_lab.py --reciter Husary_128kbps
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

SR = 16000
FP = 5.0                                    # WORLD frame period, ms
OUT = Path(__file__).with_name("world_lab.json")


def world(x: np.ndarray):  # type: ignore[no-untyped-def,type-arg]
    import pyworld as pw
    x = x.astype(np.float64)
    f0, t = pw.harvest(x, SR, frame_period=FP, f0_floor=60, f0_ceil=500)
    sp = pw.cheaptrick(x, f0, t, SR)
    ap = pw.d4c(x, f0, t, SR)
    return f0, sp, ap


def synth(f0, sp, ap, n: int) -> np.ndarray:  # type: ignore[no-untyped-def,type-arg]
    import pyworld as pw
    y = pw.synthesize(f0, sp, ap, SR, frame_period=FP)
    return np.pad(y, (0, max(0, n - len(y))))[:n].astype(np.float32)


def taper(nf: int, c: tuple[int, int], v: tuple[int, int]) -> np.ndarray:  # type: ignore[type-arg]
    """1 over the consonant, falling to 0 across the vowel: where a transform applies, and how much."""
    w = np.zeros(nf)
    w[c[0]:c[1]] = 1
    if v[1] > v[0]:
        w[v[0]:v[1]] = np.maximum(w[v[0]:v[1]], np.linspace(1, 0, v[1] - v[0]))
    return w


def devoice(f0, sp, ap, w, d):  # type: ignore[no-untyped-def]
    ap = ap + (d * w)[:, None] * (1 - ap)
    f0 = np.where((w >= 1) & (d >= 1), 0.0, f0)
    return f0, sp, ap


def voice(f0, sp, ap, w, d):  # type: ignore[no-untyped-def]
    v = f0 > 0
    if v.sum() < 2:
        return f0, sp, ap
    idx = np.arange(len(f0))
    fill = np.interp(idx, idx[v], f0[v])
    m = w >= 1
    f0 = np.where(m, fill, f0)
    ap = np.where(m[:, None], ap * (1 - d), ap)
    return f0, sp, ap


def warp(f0, sp, ap, w, a):  # type: ignore[no-untyped-def]
    nb = sp.shape[1]
    f = np.arange(nb)
    sp = sp.copy()
    for i in np.nonzero(w)[0]:
        k = 1 + (a - 1) * w[i]
        sp[i] = np.exp(np.interp(np.clip(f * k, 0, nb - 1), f, np.log(sp[i] + 1e-12)))
    return f0, sp, ap


def nasal(f0, sp, ap, w, g):  # type: ignore[no-untyped-def]
    nb = sp.shape[1]
    hz = np.linspace(0, SR / 2, nb)
    shape = g * np.exp(-0.5 * ((hz - 250) / 90) ** 2) - g * np.exp(-0.5 * ((hz - 900) / 150) ** 2)
    return f0, sp * (10 ** ((np.outer(w, shape)) / 10)), ap


EXPERIMENTS = {
    "devoice": (devoice, (0.5, 0.8, 1.0), "zdjebg", "hams_or_jahr"),
    "voice": (voice, (0.5, 0.8, 1.0), "stfkhc", "hams_or_jahr"),
    "warp_lighter": (warp, (0.9, 0.8, 0.7), "SDTZqx", "tafkheem_or_taqeeq"),
    "warp_heavier": (warp, (1.12, 1.25, 1.4), "stdkhz", "tafkheem_or_taqeeq"),
    "nasal_add": (nasal, (6.0, 12.0, 18.0), "ldbrzw", "ghonna"),
    "nasal_remove": (nasal, (-6.0, -12.0, -18.0), "nmnmnm", "ghonna"),
}
LATIN = dict(zip("zdjebgstfkhcSDTZqxlrwnmy", "زدجعبغستفكحثصضطظقخلرونمي"))


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--reciter", default="Husary_128kbps")
    ap_.add_argument("--only", nargs="*")
    args = ap_.parse_args()
    from research_agency_lab.experiments.letter_corpus.assemble import letter_cells, letter_score
    from research_agency_lab.experiments.letter_corpus.perturb import Lab, splice
    lab = Lab(args.reciter)
    results: list[dict[str, Any]] = []
    for name, (fn, doses, letters, head) in EXPERIMENTS.items():
        if args.only and name not in args.only:
            continue
        seen = []
        for ch in letters:
            sym = LATIN[ch]
            cands = []
            for d in lab.docs:
                L = d["measurements"]["letters"]
                for c, f, i in letter_cells(L):
                    if c == sym and f == "fatha" and not L[i]["edge"] and not L[i + 1]["edge"]:
                        sc = letter_score(L, [i, i + 1])
                        if sc["clean"]:
                            cands.append((sc["weakest"], d, i))
            cands.sort(key=lambda x: -x[0])
            pick = next(((d, i) for _w, d, i in cands if (d["surah"], d["ayah"], i) not in seen), None)
            if not pick:
                continue
            d, i = pick
            seen.append((d["surah"], d["ayah"], i))
            L = d["measurements"]["letters"]
            s, a = d["surah"], d["ayah"]
            wav = lab.wave(s, a)
            c0, c1 = L[i]["onset_s"], L[i]["onset_s"] + L[i]["duration_s"]
            v1 = L[i + 1]["onset_s"] + L[i + 1]["duration_s"]
            lo, hi = max(0.0, c0 - 0.3), min(len(wav) / SR, v1 + 0.3)
            seg = wav[int(lo * SR):int(hi * SR)]
            f0, sp, apr = world(seg)
            fr = lambda t: int(round((t - lo) * 1000 / FP))  # noqa: E731
            w = taper(len(f0), (fr(c0), fr(c1)), (fr(c1), fr(v1)))
            before = L[i]
            row = {"experiment": name, "letter": sym, "ref": f"{s}:{a}", "lid": before["id"], "head": head,
                   "before": {h: c["margin"] for h, c in before["characteristics"].items()},
                   "identity_before": before["identity"]["margin"], "doses": []}
            for dose in (None, *doses):
                if dose is None:
                    y = synth(f0, sp, apr, len(seg))
                else:
                    y = synth(*fn(f0.copy(), sp.copy(), apr.copy(), w, dose), len(seg))
                m = lab.eng.analyze(splice(wav, lo, hi, y), [(s, a)], makhraj=True)["measurements"]
                aft = next((x for x in m["letters"] if x["id"] == before["id"]), None)
                if aft is None:
                    continue
                hs = {h: (c["realised"], c["margin"]) for h, c in aft["characteristics"].items()}
                moved = {h: round(hs[h][1] - row["before"][h], 2) for h in hs if h in row["before"]}
                row["doses"].append({"dose": "null" if dose is None else dose, "target_margin": hs.get(head, (None, None))[1],
                                     "target_realised": hs.get(head, (None, None))[0],
                                     "identity": (aft["identity"]["confirmed"], aft["identity"]["competitor"], aft["identity"]["margin"]),
                                     "others_moved_most": sorted(((h, v) for h, v in moved.items() if h != head), key=lambda x: -abs(x[1]))[:3],
                                     "heads_not_realised": [h for h, (r, _m) in hs.items() if not r]})
            results.append(row)
            tgt = [(x["dose"], x["target_margin"], x["target_realised"]) for x in row["doses"]]
            print(f"{name:13} {sym} {s}:{a}  {head} before {row['before'].get(head)} -> {tgt}", flush=True)
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
