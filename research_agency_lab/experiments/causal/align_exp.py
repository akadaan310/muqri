#!/usr/bin/env python3
"""The minimum discriminating experiment of results/ALIGNMENT_INVESTIGATION.md section 9 -- run once.

    .venv/bin/python research_agency_lab/experiments/causal/align_exp.py

One unit (11:2 L24 ن, ghunnah, Husary), one transform (world_lab.nasal, g = -12 dB), six variants:

  V0  untouched                                  V3  nasal -12 on [9.48, 10.30]  (transition left unedited)
  V1  nasal -12 on [9.48, 10.32]  (the span)     V4  nasal -12 on [9.46, 10.32]  (kasra tail also edited)
  V2  nasal -12 on [9.46, 10.30]  (-20 ms)       V5  noop_world on [9.46, 10.30] (WORLD context shifted -20 ms)

Seven independent Engine.analyze runs, each with its own acoustic-model forward pass: V0 twice (the
second is the determinism check), then V1-V5. Each variant is also scored a second time from the SAME
posteriors, with Muqri's alignment held at V0's (no new forward pass).

How the alignment is held fixed without changing production code. The engine gets every alignment
from `ctc_viterbi`, through two module-level names: `app.submission.ctc_viterbi` (walk_alignment,
which places the ayah) and `app.analysis.ctc_viterbi` (analyse_clip, the per-phoneme path from which
every letter window is cut). `app.engine.ctc_viterbi` is used only by the basmala check, which does
not run for ayah 2. This script rebinds those two names in THIS process only:

  record  -> the real ctc_viterbi runs and its (T, seq, score, first, last) is kept
  fixed   -> V0's recorded result is returned instead; T and seq must match V0's call exactly, or it raises

Everything else -- posteriors, CTC likelihoods, pooling, rules, measurements.build -- is the
unmodified engine code. No file under app/ is modified.

Writes results/alignment_experiment.json (committed), and data/align_exp/ (local, gitignored): the
full-ayah WAVs and the posterior matrices.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Iterator

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import app.analysis as AN  # noqa: E402
import app.submission as SUB  # noqa: E402
from app.rule_bind import ph_units  # noqa: E402
from research_agency_lab.experiments.causal import physics as PH  # noqa: E402
from research_agency_lab.experiments.causal import transforms as T  # noqa: E402
from research_agency_lab.experiments.causal.harness import Harness, elsewhere, muqri_view, numeric  # noqa: E402
from research_agency_lab.experiments.causal.schema import deltas  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE / "results/alignment_experiment.json"
LOCAL = HERE / "data/align_exp"
SR, FRAME = 16000, 0.04
S, A, LID, IDX, WORD = 11, 2, "11:2:L24", ["11:2:L24"], 4
G = {"g_db": -12}
VARIANTS: list[tuple[str, str, str | None, float, float]] = [
    ("V0", "untouched", None, 9.48, 10.32),
    ("V1", "nasal -12 on the span", "nasal", 9.48, 10.32),
    ("V2", "nasal -12 shifted -20 ms", "nasal", 9.46, 10.30),
    ("V3", "nasal -12, transition [10.30,10.32] left unedited", "nasal", 9.48, 10.30),
    ("V4", "nasal -12, kasra tail [9.46,9.48] also edited", "nasal", 9.46, 10.32),
    ("V5", "WORLD pass-through, context shifted -20 ms", "noop_world", 9.46, 10.30),
]
EDGE_WINDOWS = [(9.44, 9.46), (9.46, 9.48), (9.48, 9.50), (9.50, 9.52),
                (10.28, 10.30), (10.30, 10.32), (10.32, 10.34), (10.34, 10.36)]
LETTERS = range(20, 31)          # L20..L30: the identity/makhraj context of L24 and a margin around it
STORED = {"V1": 'ghunnah/nasal|{"g_db": -12}', "V2": "ghunnah/boundary-0.02|nasal", "V0": "ghunnah/determinism"}


# ------------------------------------------------------------------ holding the alignment ---------------------------
_REAL = AN.ctc_viterbi
_LOG: list[dict[str, Any]] = []


def _recording(lp, seq, blank):  # type: ignore[no-untyped-def]
    score, first, last = _REAL(lp, seq, blank)
    _LOG.append({"T": int(lp.shape[0]), "seq": list(map(int, seq)), "score": float(score),
                 "first": list(map(int, first)), "last": list(map(int, last))})
    return score, first, last


def _fixed(calls: list[dict[str, Any]]):  # type: ignore[no-untyped-def]
    it = iter(calls)

    def f(lp, seq, blank):  # type: ignore[no-untyped-def]
        c = next(it)
        if c["T"] != lp.shape[0] or c["seq"] != list(map(int, seq)):
            raise RuntimeError(f"fixed alignment: call mismatch (T {lp.shape[0]} vs {c['T']})")
        _LOG.append({**c, "held": True})
        return c["score"], list(c["first"]), list(c["last"])
    return f


@contextlib.contextmanager
def viterbi(mode: str, calls: list[dict[str, Any]] | None = None) -> Iterator[list[dict[str, Any]]]:
    _LOG.clear()
    fn = _recording if mode == "record" else _fixed(calls or [])
    AN.ctc_viterbi, SUB.ctc_viterbi = fn, fn
    try:
        yield _LOG
    finally:
        AN.ctc_viterbi, SUB.ctc_viterbi = _REAL, _REAL


# ------------------------------------------------------------------ extraction -------------------------------------
def alignment(report: dict[str, Any], calls: list[dict[str, Any]], phonemes: str) -> dict[str, Any]:
    """The analyse_clip Viterbi path, as absolute model frames and seconds, for L20..L30."""
    clip0, clip1 = report["ayahs"][0]["frames"]
    clip = next(c for c in reversed(calls) if c["T"] == clip1 - clip0)   # analyse_clip runs after walk_alignment
    units = ph_units(phonemes)
    first, last = clip["first"], clip["last"]
    out: dict[str, Any] = {"ayah_span_frames": [int(clip0), int(clip1)], "path_calls": [
        {"T": c["T"], "first": c["first"], "last": c["last"]} for c in calls]}
    letters = {}
    for i in LETTERS:
        sym, a, b = units[i]
        f0, f1 = clip0 + first[a] - 1, clip0 + last[b]         # sifat window [first-1, last) -> absolute
        lo, hi = max(0, i - 2), min(len(units) - 1, i + 2)
        w0, w1 = clip0 + max(0, first[units[lo][1]] - 1 - 3), clip0 + min(clip1 - clip0, last[units[hi][2]] + 3)
        letters[f"11:2:L{i}"] = {"symbol": sym, "phonemes": [a, b], "sifat_frames": [f0, f1],
                                 "sifat_s": [round(f0 * FRAME, 2), round(f1 * FRAME, 2)],
                                 "identity_window_frames": [w0, w1],
                                 "identity_window_s": [round(w0 * FRAME, 2), round(w1 * FRAME, 2)],
                                 "phoneme_first_last_abs": [[clip0 + first[k] - 1, clip0 + last[k]] for k in range(a, b + 1)]}
    out["letters"] = letters
    return out


def frame_posteriors(lp: np.ndarray, lay: dict[str, Any], f0: int, f1: int) -> dict[str, Any]:  # type: ignore[type-arg]
    ph, gh = lay["levels"]["phonemes"], lay["levels"]["ghonna"]
    pv = {t: i for i, t in enumerate(ph["vocab"])}
    rows = []
    for t in range(f0, f1):
        p = lp[t, ph["first"]:ph["first"] + ph["width"]]
        order = np.argsort(-p)
        top = [(ph["vocab"][j] or "<blank>", round(float(p[j]), 3)) for j in order[:3]]
        g = lp[t, gh["first"]:gh["first"] + gh["width"]]
        rows.append({"frame": t, "t_s": round(t * FRAME, 2), "logp_noon": round(float(p[pv["ن"]]), 3),
                     "logp_blank": round(float(p[0]), 3), "top3": top,
                     "ghonna_logp": {gh["vocab"][j]: round(float(g[j]), 3) for j in range(1, gh["width"])}})
    return {"frames": rows}


def band_db(x: np.ndarray, lo: float, hi: float) -> float:  # type: ignore[type-arg]
    return round(10 * float(np.log10(PH._band_energy(x, lo, hi) + 1e-12)), 2)


def rect_ratio(x: np.ndarray) -> float:  # type: ignore[type-arg]
    n = max(512, 1 << int(np.ceil(np.log2(max(len(x), 2)))))
    s, f = np.abs(np.fft.rfft(x, n)) ** 2, np.fft.rfftfreq(n, 1 / SR)
    return round(10 * float(np.log10((s[(f >= 150) & (f < 400)].sum() + 1e-12) / (s[(f >= 750) & (f < 1100)].sum() + 1e-12))), 2)


def phys(wave: np.ndarray, a: float, b: float) -> dict[str, Any]:  # type: ignore[type-arg]
    x = np.asarray(wave[int(round(a * SR)):int(round(b * SR))], dtype=np.float64)
    out = {"window_s": [round(a, 3), round(b, 3)], "low_150_400_db": band_db(x, 150, 400),
           "anti_750_1100_db": band_db(x, 750, 1100), "nasal_ratio_rect_db": rect_ratio(x),
           "rms_db": round(20 * float(np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)), 2)}
    if b - a >= 0.1:
        out["praat_hann"] = PH.measure(wave, a, b)
    return out


def sha(x: np.ndarray) -> str:  # type: ignore[type-arg]
    return hashlib.sha256(np.asarray(x, dtype=np.float32).tobytes()).hexdigest()[:16]


def git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()


# ------------------------------------------------------------------ the run ----------------------------------------
def main() -> int:
    t_start = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    prov = {"commit": git("rev-parse", "HEAD"), "working_tree_porcelain": git("status", "--porcelain"),
            "script_sha256_16": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16], "started": t_start}
    prov["working_tree_clean"] = prov["working_tree_porcelain"] == ""
    h = Harness("Husary_128kbps")
    eng = h.eng
    wave0 = h.wave(S, A)
    phonemes = eng.reference(S, A).phonemes
    LOCAL.mkdir(parents=True, exist_ok=True)
    import soundfile as sf

    def independent(w: np.ndarray) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:  # type: ignore[type-arg]
        lp = eng.posteriors(w)
        with viterbi("record") as log:
            rep = eng.analyze(w, [(S, A)], posteriors=lp, makhraj=True)
            calls = [dict(c) for c in log]
        return lp, rep, calls

    def held(w: np.ndarray, lp: np.ndarray, calls0: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:  # type: ignore[type-arg]
        with viterbi("fixed", calls0) as log:
            rep = eng.analyze(w, [(S, A)], posteriors=lp, makhraj=True)
            used = [dict(c) for c in log]
        if len(used) != len(calls0):
            raise RuntimeError(f"fixed alignment: {len(used)} of {len(calls0)} baseline calls consumed")
        return rep, used

    lp0, rep0, calls0 = independent(wave0)
    m0 = rep0["measurements"]
    al0 = alignment(rep0, calls0, phonemes)
    lp0b, rep0b, calls0b = independent(wave0)                       # engine run 2: determinism
    determinism = {"posteriors_identical": bool(np.array_equal(lp0, lp0b)),
                   "paths_identical": [c["first"] for c in calls0] == [c["first"] for c in calls0b]
                   and [c["last"] for c in calls0] == [c["last"] for c in calls0b],
                   "muqri_delta_nonzero": {k: v for k, v in deltas(numeric(muqri_view(m0, LID, IDX, WORD)),
                                           numeric(muqri_view(rep0b["measurements"], LID, IDX, WORD))).items() if v}}
    lay = eng._layout
    f24 = al0["letters"][LID]["sifat_frames"]
    win = (f24[0] - 6, f24[1] + 6)
    mb = muqri_view(m0, LID, IDX, WORD)
    stored = {r["experiment_id"]: r for r in json.loads((HERE / "data/records.json").read_text())} \
        if (HERE / "data/records.json").is_file() else {}

    rep0h, _ = held(wave0, lp0, calls0)                             # the replay must reproduce V0 exactly
    replay_check = {k: v for k, v in deltas(numeric(mb), numeric(muqri_view(rep0h["measurements"], LID, IDX, WORD))).items() if v}
    if replay_check:
        raise RuntimeError(f"held-alignment replay does not reproduce V0: {replay_check}")
    variants = []
    for vid, desc, name, t0, t1 in VARIANTS:
        if name is None:
            w, e1, impl = wave0, t1, "none"
            lp, rep_i, calls_i = lp0, rep0, calls0
        else:
            w, e1, impl = T.apply(wave0, t0, t1, name, G if name == "nasal" else {})
            lp, rep_i, calls_i = independent(w)                      # engine runs 3..7
        rep_h, used_h = held(w, lp, calls0)
        sf.write(LOCAL / f"{vid}.wav", w, SR, subtype="FLOAT")
        np.save(LOCAL / f"{vid}_posteriors.npy", lp)
        mi, mh = rep_i["measurements"], rep_h["measurements"]
        vi, vh = muqri_view(mi, LID, IDX, WORD), muqri_view(mh, LID, IDX, WORD)
        ali = alignment(rep_i, calls_i, phonemes)
        moved = {k: {"baseline": al0["letters"][k]["sifat_frames"], "independent": v["sifat_frames"]}
                 for k, v in ali["letters"].items() if v["sifat_frames"] != al0["letters"][k]["sifat_frames"]}
        path_changed_frames = sum(int(a != b) for c0, ci in zip(calls0, calls_i)
                                  for a, b in zip(c0["first"] + c0["last"], ci["first"] + ci["last"]))
        ctx = (max(0.0, t0 - T.CTX), min(len(wave0) / SR, t1 + T.CTX)) if name in ("nasal", "noop_world") else None
        # the samples that differ from V0: where the audio actually changed
        diff = np.nonzero(np.abs(np.asarray(w) - np.asarray(wave0)) > 1e-6)[0]
        L_i = ali["letters"][LID]
        variants.append({
            "id": vid, "description": desc, "transform": name, "parameters": G if name == "nasal" else {},
            "implementation": impl, "edit_span_s": [t0, round(e1, 3)], "world_context_s": ctx,
            "samples_changed_s": [round(diff[0] / SR, 4), round(diff[-1] / SR, 4)] if diff.size else None,
            "audio_sha256_16": sha(w), "audio_local": str((LOCAL / f"{vid}.wav").relative_to(HERE)),
            "posteriors_local": str((LOCAL / f"{vid}_posteriors.npy").relative_to(HERE)),
            "physics": {
                "edge_windows_20ms": [phys(w, a, b) for a, b in EDGE_WINDOWS],
                "edit_span": phys(w, t0, t1),
                "fixed_baseline_L24_region": phys(w, al0["letters"][LID]["sifat_s"][0], al0["letters"][LID]["sifat_s"][1]),
                "independent_L24_sifat_window": phys(w, *L_i["sifat_s"]),
                "baseline_identity_window": phys(w, *al0["letters"][LID]["identity_window_s"]),
                "independent_identity_window": phys(w, *L_i["identity_window_s"])},
            "alignment_independent": ali, "alignment_held": {"source": "V0", "calls_replayed": len(used_h)},
            "alignment_change": {"letters_whose_sifat_window_moved": moved,
                                 "path_boundaries_changed": path_changed_frames,
                                 "ayah_span_frames": ali["ayah_span_frames"]},
            "muqri_independent": vi, "muqri_held": vh,
            "delta_independent": deltas(numeric(mb), numeric(vi)),
            "delta_held": deltas(numeric(mb), numeric(vh)),
            "alignment_component": deltas(numeric(vh), numeric(vi)),
            "elsewhere_independent": elsewhere(m0, mi, set(IDX)),
            "elsewhere_held": elsewhere(m0, mh, set(IDX)),
            "frame_posteriors": frame_posteriors(lp, lay, *win),
            "stored_record_check": ({"stored": STORED.get(vid), "delta_vs_stored": {
                k: v for k, v in deltas(numeric(stored[STORED[vid]]["muqri_transformed"]), numeric(vi)).items() if v}}
                if STORED.get(vid) in stored else None)})
        print(f"{vid}: ghonna held {vh.get('head:ghonna:margin')} indep {vi.get('head:ghonna:margin')}; "
              f"identity held {vh.get('identity:margin')} indep {vi.get('identity:margin')}; moved {list(moved)}", flush=True)

    out = {"experiment": "alignment discriminating experiment (ALIGNMENT_INVESTIGATION.md section 9)",
           "unit": {"reciter": "Husary_128kbps", "surah": S, "ayah": A, "letter_id": LID, "rule": "ghunnah",
                    "source_audio": "everyayah:Husary_128kbps/011002.mp3", "audio_sha256_16": sha(wave0)},
           "provenance": {**prov, "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                          "python": platform.python_version(), "engine": "app.engine.Engine (makhraj=True), unmodified",
                          "model": "obadx/muaalem-model-v3_2",
                          "engine_runs_independent": 7, "held_alignment_rescorings": len(VARIANTS),
                          "earlier_smoke_records": "7bef873 dirty tree (see ALIGNMENT_INVESTIGATION.md section 7)"},
           "method": {"held_alignment": "app.submission.ctc_viterbi and app.analysis.ctc_viterbi rebound in-process to "
                      "replay V0's recorded paths; T and seq asserted equal; production files untouched",
                      "frame_s": FRAME, "frame_posterior_window": list(win)},
           "baseline": {"alignment": al0, "muqri": mb, "determinism": determinism, "replay_reproduces_V0": not replay_check,
                        "frame_posteriors": frame_posteriors(lp0, lay, *win)},
           "variants": variants}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
