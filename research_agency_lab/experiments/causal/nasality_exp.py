#!/usr/bin/env python3
"""Nasality specificity control on the decisive 20 ms window -- one unit, run once.

    .venv/bin/python research_agency_lab/experiments/causal/nasality_exp.py

Question: at 11:2 L24 ن (ghunnah, Husary), is Muqri's response to the -12 dB nasal edit specific to the
nasal dimension, or would any comparable alteration of the decisive window [10.30, 10.32] s do?
(results/ALIGNMENT_DISCRIMINATING_EXPERIMENT.md: with the rest of the span nasal-edited, editing or not
editing that window moves makhraj:م by ~7 nats; it carries the ن's final CTC spike.)

Design: every variant but V0 and W is ONE WORLD pass over the SAME context [9.18, 10.62] (span +/- 0.3 s,
exactly as V1 / transforms._world_edit), with the SAME interior edit -- world_lab.nasal g = -12 dB on
[9.48, 10.30] -- and differs ONLY in what is done to the WORLD frames of [10.30, 10.32]:

  B    nothing (the window is resynthesised, unedited)                   no-op control
  N    world_lab.nasal, g = -12 dB (pole 250 Hz / anti-resonance 900 Hz)  the nasal edit; == stored V1, asserted
  C1   world_lab.nasal's formula, g = -12 dB, both centres moved +1750 Hz (2000 / 2650 Hz), widths unchanged:
       the same shape and dB magnitude, away from the nasal bands          shape-matched non-nasal control
  C2   flat envelope gain of g2 dB, g2 = N's measured window RMS change against B (set from the physics
       of N and B before C2 is synthesised; no Muqri value is used)       energy-matched non-nasal control
  W    noop_world on the span (context [9.18, 10.62], no edit at all)     WORLD footprint, reported apart
  V0   the untouched ayah

No new transform family: C1 is world_lab.nasal's own expression with other centre frequencies; C2 is
the same envelope multiplication with a constant shape. WORLD analysis/synthesis, mask, context and
splice are those of transforms._world_edit.

Matching criteria, declared before the run (window [10.30, 10.32], each control against B):
  energy       |RMS change - N's RMS change| <= 0.5 dB
  shape        spectral-change magnitude (mean |dB change| over 100 Hz bands, 100-4000 Hz) within
               0.67-1.5 x N's
  non-nasal    |change of the nasal ratio (150-400 / 750-1100 Hz, rectangular)| <= 1 dB and
               |change of either nasal band| <= 1.5 dB
  location, duration, boundary, resynthesis footprint: identical by construction (same mask/context)

Muqri: Engine.analyze (unmodified) per variant, and each variant rescored with V0's Viterbi path held
(align_exp.viterbi: in-process rebinding of app.submission/app.analysis.ctc_viterbi).

Writes results/nasality_experiment.json; data/nasality_exp/ (local, gitignored): WAVs and posteriors.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from research_agency_lab.experiments.causal import align_exp as AX  # noqa: E402
from research_agency_lab.experiments.causal import physics as PH  # noqa: E402
from research_agency_lab.experiments.causal import transforms as T  # noqa: E402
from research_agency_lab.experiments.causal.harness import Harness, elsewhere, muqri_view, numeric  # noqa: E402
from research_agency_lab.experiments.causal.schema import deltas  # noqa: E402
from research_agency_lab.experiments.letter_corpus.perturb import splice  # noqa: E402
from research_agency_lab.experiments.synthesis import world_lab as W  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE / "results/nasality_experiment.json"
LOCAL = HERE / "data/nasality_exp"
SR = 16000
S, A, LID, IDX, WORD = 11, 2, "11:2:L24", ["11:2:L24"], 4
SPAN0, CUT, SPAN1 = 9.48, 10.30, 10.32          # interior [9.48, 10.30], decisive window [10.30, 10.32]
G = -12.0
SHIFT_HZ = 1750.0
BANDS = [(0, 150), (150, 400), (400, 750), (750, 1100), (1100, 1500), (1500, 2000), (2000, 2400),
         (2400, 3000), (3000, 4000), (4000, 8000)]


def off_band(f0, sp, ap, w, g, shift=SHIFT_HZ):  # type: ignore[no-untyped-def]
    """world_lab.nasal's expression with both centres moved by `shift` Hz (widths and signs unchanged)."""
    nb = sp.shape[1]
    hz = np.linspace(0, SR / 2, nb)
    shape = g * np.exp(-0.5 * ((hz - 250 - shift) / 90) ** 2) - g * np.exp(-0.5 * ((hz - 900 - shift) / 150) ** 2)
    return f0, sp * (10 ** ((np.outer(w, shape)) / 10)), ap


def flat_gain(f0, sp, ap, w, g):  # type: ignore[no-untyped-def]
    """The same envelope multiplication with a constant (frequency-independent) g dB."""
    return f0, sp * (10 ** ((w * g) / 10))[:, None], ap


def composite(wave: np.ndarray, window_edit: Callable | None) -> np.ndarray:  # type: ignore[type-arg]
    """transforms._world_edit over [9.48, 10.32], with nasal -12 on the interior mask and `window_edit` on
    the window mask. N passes window_edit=None and nasal on the union (bit-identical to T.apply)."""
    lo, hi = max(0.0, SPAN0 - T.CTX), min(len(wave) / SR, SPAN1 + T.CTX)
    seg = wave[int(lo * SR):int(hi * SR)]
    f0, sp, ap = W.world(seg)
    fr = lambda t: int(round((t - lo) * 1000 / W.FP))  # noqa: E731
    a, m, b = fr(SPAN0), fr(CUT), fr(SPAN1)
    w_int, w_win = np.zeros(len(f0)), np.zeros(len(f0))
    w_int[a:m], w_win[m:b] = 1.0, 1.0
    if window_edit == "nasal":
        f0, sp, ap = W.nasal(f0.copy(), sp.copy(), ap.copy(), w_int + w_win, G)
    else:
        f0, sp, ap = W.nasal(f0.copy(), sp.copy(), ap.copy(), w_int, G)
        if window_edit is not None:
            f0, sp, ap = window_edit(f0, sp, ap, w_win)
    return splice(wave, lo, hi, W.synth(f0, sp, ap, len(seg)))


# ------------------------------------------------------------------ physics ----------------------------------------
def band_levels(x: np.ndarray) -> dict[str, float]:  # type: ignore[type-arg]
    return {f"{lo}-{hi}": AX.band_db(x, lo, hi) for lo, hi in BANDS}


def fine_levels(x: np.ndarray) -> np.ndarray:  # type: ignore[type-arg]
    return np.array([10 * np.log10(PH._band_energy(x, f, f + 100) + 1e-12) for f in range(100, 4000, 100)])


def win_phys(w: np.ndarray, a: float, b: float) -> dict[str, Any]:  # type: ignore[type-arg]
    x = np.asarray(w[int(round(a * SR)):int(round(b * SR))], dtype=np.float64)
    out = AX.phys(w, a, b)
    out["bands_db"] = band_levels(x)
    pr = PH.measure(w, a, b)                        # Praat on the window, 0.2 s context for analysis
    out["praat"] = {k: pr.get(k) for k in ("voiced_fraction", "f0_hz", "hnr_db", "f1_hz", "f2_hz", "f3_hz")}
    return out


def contrast(x: np.ndarray, y: np.ndarray, a: float, b: float) -> dict[str, Any]:  # type: ignore[type-arg]
    """Change of window [a, b] from audio y (reference) to audio x."""
    sx = np.asarray(x[int(round(a * SR)):int(round(b * SR))], dtype=np.float64)
    sy = np.asarray(y[int(round(a * SR)):int(round(b * SR))], dtype=np.float64)
    fx, fy = fine_levels(sx), fine_levels(sy)
    bx, by = band_levels(sx), band_levels(sy)
    rms = lambda s: 20 * np.log10(np.sqrt(np.mean(s ** 2)) + 1e-12)  # noqa: E731
    return {"rms_db": round(float(rms(sx) - rms(sy)), 2),
            "nasal_ratio_rect_db": round(AX.rect_ratio(sx) - AX.rect_ratio(sy), 2),
            "low_150_400_db": round(bx["150-400"] - by["150-400"], 2),
            "anti_750_1100_db": round(bx["750-1100"] - by["750-1100"], 2),
            "bands_db": {k: round(bx[k] - by[k], 2) for k in bx},
            "spectral_change_mean_abs_db_100_4000": round(float(np.mean(np.abs(fx - fy))), 2),
            "spectral_change_mean_abs_db_outside_nasal_bands": round(float(np.mean(np.abs(
                (fx - fy)[[not (150 <= f < 400 or 750 <= f < 1100) for f in range(100, 4000, 100)]]))), 2)}


# ------------------------------------------------------------------ the run ----------------------------------------
def main() -> int:
    prov = {"commit": AX.git("rev-parse", "HEAD"), "working_tree_porcelain": AX.git("status", "--porcelain"),
            "script_sha256_16": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
            "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    prov["working_tree_clean"] = prov["working_tree_porcelain"] == ""
    h = Harness("Husary_128kbps")
    eng = h.eng
    wave0 = h.wave(S, A)
    phonemes = eng.reference(S, A).phonemes
    LOCAL.mkdir(parents=True, exist_ok=True)
    import soundfile as sf

    # 1. audio: B and N first; C2's gain comes from their measured window RMS difference
    waves: dict[str, np.ndarray] = {"V0": wave0}  # type: ignore[type-arg]
    waves["B"] = composite(wave0, None)
    waves["N"] = composite(wave0, "nasal")
    v1, _, _ = T.apply(wave0, SPAN0, SPAN1, "nasal", {"g_db": G})
    n_equals_v1 = bool(np.array_equal(waves["N"], v1))
    if not n_equals_v1:
        raise RuntimeError("N is not bit-identical to the stored V1 construction")
    g2 = contrast(waves["N"], waves["B"], CUT, SPAN1)["rms_db"]
    waves["C1"] = composite(wave0, lambda f0, sp, ap, w: off_band(f0, sp, ap, w, G))
    waves["C2"] = composite(wave0, lambda f0, sp, ap, w: flat_gain(f0, sp, ap, w, g2))
    waves["W"], _, _ = T.apply(wave0, SPAN0, SPAN1, "noop_world", {})

    spec = {
        "V0": {"transform": "none", "parameters": {}, "window_s": None, "implementation": "—"},
        "B": {"transform": "window no-op (interior nasal only)", "parameters": {"interior_g_db": G},
              "window_s": [CUT, SPAN1], "implementation": "world_lab.world/nasal(interior)/synth + perturb.splice"},
        "N": {"transform": "nasal", "parameters": {"g_db": G, "pole_hz": 250, "anti_hz": 900},
              "window_s": [CUT, SPAN1], "implementation": "world_lab.nasal on interior+window (== transforms.apply nasal [9.48,10.32])"},
        "C1": {"transform": "off-band resonance edit", "parameters": {"g_db": G, "pole_hz": 250 + SHIFT_HZ,
               "anti_hz": 900 + SHIFT_HZ}, "window_s": [CUT, SPAN1],
               "implementation": "nasality_exp.off_band (world_lab.nasal expression, centres +1750 Hz)"},
        "C2": {"transform": "flat envelope gain", "parameters": {"g_db": g2}, "window_s": [CUT, SPAN1],
               "implementation": "nasality_exp.flat_gain (constant envelope multiplication)"},
        "W": {"transform": "noop_world", "parameters": {}, "window_s": None,
              "implementation": "transforms.apply noop_world [9.48,10.32] (world_lab.world -> synth)"}}

    # 2. physics (independent of Muqri)
    physics = {}
    for k, w in waves.items():
        physics[k] = {"window": win_phys(w, CUT, SPAN1),
                      "interior": AX.phys(w, SPAN0, CUT),
                      "vs_B_window": contrast(w, waves["B"], CUT, SPAN1),
                      "vs_V0_window": contrast(w, wave0, CUT, SPAN1),
                      "vs_B_interior": contrast(w, waves["B"], SPAN0, CUT),
                      "vs_B_neighbours": {f"[{a},{b}]": contrast(w, waves["B"], a, b)["rms_db"]
                                          for a, b in ((10.28, 10.30), (10.32, 10.34), (10.34, 10.40))},
                      "samples_changed_vs_V0_s": (lambda d: [round(d[0] / SR, 4), round(d[-1] / SR, 4)] if d.size else None)(
                          np.nonzero(np.abs(w - wave0) > 1e-6)[0])}
    n = physics["N"]["vs_B_window"]
    matching = {}
    for k in ("C1", "C2"):
        c = physics[k]["vs_B_window"]
        ratio = c["spectral_change_mean_abs_db_100_4000"] / max(n["spectral_change_mean_abs_db_100_4000"], 1e-9)
        matching[k] = {"energy_matched": abs(c["rms_db"] - n["rms_db"]) <= 0.5,
                       "shape_magnitude_ratio_to_N": round(ratio, 2), "shape_matched": 0.67 <= ratio <= 1.5,
                       "non_nasal": abs(c["nasal_ratio_rect_db"]) <= 1.0 and abs(c["low_150_400_db"]) <= 1.5
                       and abs(c["anti_750_1100_db"]) <= 1.5,
                       "rms_db": c["rms_db"], "N_rms_db": n["rms_db"], "nasal_ratio_change_db": c["nasal_ratio_rect_db"],
                       "N_nasal_ratio_change_db": n["nasal_ratio_rect_db"]}

    # 3. Muqri: independent and held (V0's path)
    lp0 = eng.posteriors(wave0)
    with AX.viterbi("record") as log:
        rep0 = eng.analyze(wave0, [(S, A)], posteriors=lp0, makhraj=True)
        calls0 = [dict(c) for c in log]
    al0 = AX.alignment(rep0, calls0, phonemes)
    f24 = al0["letters"][LID]["sifat_frames"]
    muqri: dict[str, Any] = {}
    reports: dict[str, Any] = {}
    for k, w in waves.items():
        lp = lp0 if k == "V0" else eng.posteriors(w)
        if k == "V0":
            rep_i, calls_i = rep0, calls0
        else:
            with AX.viterbi("record") as log:
                rep_i = eng.analyze(w, [(S, A)], posteriors=lp, makhraj=True)
                calls_i = [dict(c) for c in log]
        with AX.viterbi("fixed", calls0) as log:
            rep_h = eng.analyze(w, [(S, A)], posteriors=lp, makhraj=True)
            if len(log) != len(calls0):
                raise RuntimeError("held alignment: not every baseline call was replayed")
        sf.write(LOCAL / f"{k}.wav", w, SR, subtype="FLOAT")
        np.save(LOCAL / f"{k}_posteriors.npy", lp)
        ali = AX.alignment(rep_i, calls_i, phonemes)
        reports[k] = (rep_i["measurements"], rep_h["measurements"])
        muqri[k] = {"independent": muqri_view(rep_i["measurements"], LID, IDX, WORD),
                    "held": muqri_view(rep_h["measurements"], LID, IDX, WORD),
                    "alignment_L20_L30": {l: v["sifat_frames"] for l, v in ali["letters"].items()},
                    "L24_moved": ali["letters"][LID]["sifat_frames"] != f24,
                    "letters_moved_vs_V0": [l for l, v in ali["letters"].items()
                                            if v["sifat_frames"] != al0["letters"][l]["sifat_frames"]],
                    "frame_posteriors_255_258": [r for r in AX.frame_posteriors(lp, eng._layout, 255, 259)["frames"]],
                    "audio_sha256_16": AX.sha(w), "audio_local": f"data/nasality_exp/{k}.wav"}
        print(f"{k}: ghonna indep {muqri[k]['independent'].get('head:ghonna:margin')} held "
              f"{muqri[k]['held'].get('head:ghonna:margin')}; identity {muqri[k]['independent'].get('identity:margin')}; "
              f"makhraj:م {muqri[k]['independent'].get('makhraj:م')}", flush=True)

    comparisons = {}
    for ref in ("B", "V0"):
        for k in waves:
            if k == ref:
                continue
            mi_r, mh_r = reports[ref]
            mi, mh = reports[k]
            comparisons[f"{k}_vs_{ref}"] = {
                "independent": deltas(numeric(muqri[ref]["independent"]), numeric(muqri[k]["independent"])),
                "held": deltas(numeric(muqri[ref]["held"]), numeric(muqri[k]["held"])),
                "elsewhere_independent": elsewhere(mi_r, mi, set(IDX)),
                "elsewhere_held": elsewhere(mh_r, mh, set(IDX))}

    out = {"experiment": "nasality specificity control on the decisive window (one unit)",
           "unit": {"reciter": "Husary_128kbps", "surah": S, "ayah": A, "letter_id": LID, "rule": "ghunnah",
                    "source_audio": "everyayah:Husary_128kbps/011002.mp3", "audio_sha256_16": AX.sha(wave0)},
           "window_s": [CUT, SPAN1], "interior_s": [SPAN0, CUT], "world_context_s": [SPAN0 - T.CTX, SPAN1 + T.CTX],
           "provenance": {**prov, "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                          "python": platform.python_version(), "engine": "app.engine.Engine (makhraj=True), unmodified",
                          "model": "obadx/muaalem-model-v3_2", "N_bit_identical_to_V1_construction": n_equals_v1,
                          "prior_artifacts": {"smoke_records": "7bef873 dirty tree",
                                              "alignment_experiment": "1ea5dd9 clean (results 1a6a560)"}},
           "variants": spec, "c2_gain_db_from_N_minus_B_rms": g2, "matching": matching, "physics": physics,
           "muqri": muqri, "comparisons": comparisons,
           "baseline_alignment_L24_sifat_frames": f24}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    print(json.dumps(matching, indent=1))
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
