"""Independent physical measurements of one span of audio -- none of them from the Muqri model.

Python / Praat (parselmouth), the primary layer:

  span_duration_s     the span's length in the audio as it now is (bookkeeping, exact)
  voiced_duration_s   Praat pitch track (5 ms step): voiced frames x step inside the span
  voiced_fraction     voiced frames / frames inside the span
  f0_hz               median Praat F0 over the span's voiced frames
  hnr_db              Praat harmonicity (cross-correlation), mean over frames with HNR > -199 dB
  f1_hz f2_hz f3_hz   Praat Burg formants (5 formants, ceiling 5.5 kHz), median over the span's voiced frames
  nasal_ratio_db      10 log10(E[150-400 Hz] / E[750-1100 Hz]), Hann-windowed FFT of the span
                      (nasal murmur: a pole near 250 Hz, an anti-resonance near 900 Hz)
  rms_db              span RMS in dBFS

Independent cross-checks (different code, different methods -- disagreement is data):

  octave   research_agency_lab/substrate_library/octave/qaari_features.m: autocorrelation voicing,
           cepstral F0, autocorrelation HNR (Boersma), LPC formants at the core, nasal_db as 0-400 Hz
           minus 400-2500 Hz (a different band definition), burst, HF ratio
  julia    causal/julia/xcheck.jl on the raw float32 span: its own autocorrelation voicing, its own
           radix-2 FFT for the same nasal bands as Python, RMS, duration
"""

from __future__ import annotations

import csv
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

SR = 16000
ROOT = Path(__file__).resolve().parents[3]
OCTAVE_DIR = ROOT / "research_agency_lab/substrate_library/octave"
JULIA = Path.home() / "julia-1.11.5/bin/julia"
JULIA_PROJECT = ROOT / "research_agency_lab/substrate_library/julia"
XCHECK = Path(__file__).with_name("julia") / "xcheck.jl"
CONTEXT_S = 0.2


def _band_energy(x: np.ndarray, lo: float, hi: float) -> float:  # type: ignore[type-arg]
    n = max(512, 1 << int(np.ceil(np.log2(max(len(x), 2)))))
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)), n)) ** 2
    f = np.fft.rfftfreq(n, 1 / SR)
    return float(spec[(f >= lo) & (f < hi)].sum())


def measure(wave: np.ndarray, t0: float, t1: float) -> dict[str, Any]:  # type: ignore[type-arg]
    """Praat / numpy measurements of [t0, t1] of `wave` (16 kHz), with context around it for analysis."""
    import parselmouth
    lo, hi = max(0.0, t0 - CONTEXT_S), min(len(wave) / SR, t1 + CONTEXT_S)
    win = np.asarray(wave[int(lo * SR):int(hi * SR)], dtype=np.float64)
    seg = np.asarray(wave[int(t0 * SR):int(t1 * SR)], dtype=np.float64)
    out: dict[str, Any] = {"span_duration_s": round(t1 - t0, 4)}
    if len(seg) < 160 or len(win) < 800:
        return {**out, "error": "span too short"}
    snd = parselmouth.Sound(win, SR)
    a, b = t0 - lo, t1 - lo
    pitch = snd.to_pitch(time_step=0.005, pitch_floor=60, pitch_ceiling=500)
    ts, f0 = pitch.xs(), pitch.selected_array["frequency"]
    m = (ts >= a) & (ts <= b)
    voiced = m & (f0 > 0)
    out["voiced_fraction"] = round(float(voiced.sum() / m.sum()), 4) if m.any() else None
    out["voiced_duration_s"] = round(float(voiced.sum()) * 0.005, 4)
    out["f0_hz"] = round(float(np.median(f0[voiced])), 2) if voiced.any() else None
    harm = snd.to_harmonicity_cc(time_step=0.005, minimum_pitch=60)
    hv = np.array([harm.get_value(t) for t in ts[m]], dtype=float) if m.any() else np.array([])
    hv = hv[np.isfinite(hv) & (hv > -199)]
    out["hnr_db"] = round(float(hv.mean()), 2) if hv.size else None
    fm = snd.to_formant_burg(time_step=0.005, max_number_of_formants=5, maximum_formant=5500)
    for k in (1, 2, 3):
        vals = [fm.get_value_at_time(k, t) for t in ts[voiced]] if voiced.any() else []
        vals = [v for v in vals if v and np.isfinite(v)]
        out[f"f{k}_hz"] = round(float(np.median(vals)), 1) if vals else None
    low, anti = _band_energy(seg, 150, 400), _band_energy(seg, 750, 1100)
    out["nasal_ratio_db"] = round(float(10 * np.log10((low + 1e-12) / (anti + 1e-12))), 2)
    out["rms_db"] = round(float(20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-12)), 2)
    return out


def octave(pairs: list[tuple[str, np.ndarray, float, float]]) -> dict[str, dict[str, float]]:  # type: ignore[type-arg]
    """qaari_features.m on (id, wave, t0, t1) spans; returns {id: {field: value}}; {} if Octave fails."""
    import soundfile as sf
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        jobs = dp / "jobs.csv"
        with jobs.open("w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["span_id", "wav_path", "start_s", "end_s"])
            for i, (sid, wave, t0, t1) in enumerate(pairs):
                p = dp / f"{i}.wav"
                sf.write(p, np.asarray(wave, dtype=np.float32), SR)
                w.writerow([sid, str(p), f"{t0:.4f}", f"{t1:.4f}"])
        out = dp / "out.csv"
        r = subprocess.run(["octave", "--no-gui", "--quiet", "--eval",
                            f"addpath('{OCTAVE_DIR}'); qaari_features('{jobs}', '{out}')"],
                           capture_output=True, text=True, timeout=600)
        if r.returncode != 0 or not out.is_file():
            return {"_error": {"stderr": r.stderr[-400:]}}  # type: ignore[dict-item]
        res = {}
        with out.open() as f:
            for row in csv.DictReader(f):
                sid = row.pop("span_id")
                res[sid] = {k: (float(v) if v not in ("", "NaN", "nan") else None) for k, v in row.items()}
        return res


def julia(pairs: list[tuple[str, np.ndarray, float, float]]) -> dict[str, dict[str, float]]:  # type: ignore[type-arg]
    """xcheck.jl `spans` mode on the raw float32 spans; returns {id: {metric: value}}."""
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        index = []
        for i, (sid, wave, t0, t1) in enumerate(pairs):
            seg = np.asarray(wave[int(t0 * SR):int(t1 * SR)], dtype="<f4")
            p = dp / f"{i}.f32"
            seg.tofile(p)
            index.append({"id": sid, "file": str(p), "n": int(seg.size)})
        (dp / "index.json").write_text(json.dumps(index))
        r = subprocess.run([str(JULIA), f"--project={JULIA_PROJECT}", str(XCHECK), "spans", str(dp / "index.json"),
                            str(dp / "out.json")], capture_output=True, text=True, timeout=900)
        if r.returncode != 0:
            return {"_error": {"stderr": r.stderr[-400:]}}  # type: ignore[dict-item]
        return json.loads((dp / "out.json").read_text())
