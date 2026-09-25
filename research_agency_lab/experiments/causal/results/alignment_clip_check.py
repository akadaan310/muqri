#!/usr/bin/env python3
"""Read-only re-measurement of the STORED ghunnah clips, for ALIGNMENT_INVESTIGATION.md.

    .venv/bin/python research_agency_lab/experiments/causal/results/alignment_clip_check.py

Reads data/records.json and data/audio/ghunnah/*.wav (PCM16 clips of span +/- 0.4 s, written by the
smoke run). Runs no engine, no WORLD, no synthesis; writes nothing. Uses physics._band_energy, the
function behind the stored nasal_ratio_db.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

C = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(C.parents[2]))
from research_agency_lab.experiments.causal.physics import _band_energy  # noqa: E402

SR = 16000
R = {r["experiment_id"]: r for r in json.loads((C / "data/records.json").read_text())}
KEYS = ["determinism", "noop_world", 'nasal|{"g_db": -12}', "boundary-0.02|nasal", "boundary+0.02|nasal"]
LABEL = {"determinism": "untouched", "noop_world": "noop_world", 'nasal|{"g_db": -12}': "-12 dB at span",
         "boundary-0.02|nasal": "-12 dB, -20 ms", "boundary+0.02|nasal": "-12 dB, +20 ms"}


def span(k: str) -> tuple[float, float]:
    a, b = R[f"ghunnah/{k}"]["transform"]["span"].strip("[] s").split(",")
    return float(a), float(b)


def seg(k: str, a: float, b: float) -> np.ndarray:  # type: ignore[type-arg]
    x, sr = sf.read(C / R[f"ghunnah/{k}"]["transformed_audio"], dtype="float64")
    assert sr == SR
    lo = max(0.0, span(k)[0] - 0.4)          # harness._save_audio: clip starts 0.4 s before the span
    return x[int(round((a - lo) * SR)):int(round((b - lo) * SR))]


def ratio(x: np.ndarray, hann: bool = True) -> float:  # type: ignore[type-arg]
    if hann:
        return 10 * np.log10((_band_energy(x, 150, 400) + 1e-12) / (_band_energy(x, 750, 1100) + 1e-12))
    n = max(512, 1 << int(np.ceil(np.log2(len(x)))))
    s, f = np.abs(np.fft.rfft(x, n)) ** 2, np.fft.rfftfreq(n, 1 / SR)
    return 10 * np.log10(s[(f >= 150) & (f < 400)].sum() / s[(f >= 750) & (f < 1100)].sum())


def level(x: np.ndarray, lo: float, hi: float) -> float:  # type: ignore[type-arg]
    return 10 * np.log10(_band_energy(x, lo, hi) + 1e-12)


def main() -> int:
    print("1. stored nasal_ratio_db (own span) vs the same value re-measured from the PCM16 clip")
    for k in KEYS:
        a, b = span(k)
        print(f"   {LABEL[k]:16s} [{a:.2f},{b:.2f}]  stored {R[f'ghunnah/{k}']['physics_transformed']['nasal_ratio_db']:7.2f}"
              f"  clip {ratio(seg(k, a, b)):7.2f}")
    print("2. the FIXED baseline L24 region [9.48, 10.32]: Hann (as physics.py) and rectangular window")
    h0, r0 = ratio(seg("determinism", 9.48, 10.32)), ratio(seg("determinism", 9.48, 10.32), hann=False)
    for k in KEYS:
        x = seg(k, 9.48, 10.32)
        print(f"   {LABEL[k]:16s} hann {ratio(x):8.3f} (d {ratio(x) - h0:+7.2f})   rect {ratio(x, False):8.3f} "
              f"(d {ratio(x, False) - r0:+7.2f})")
    w, n = np.hanning(int(0.84 * SR)), int(0.02 * SR)
    print(f"   Hann energy weight of the first / last 20 ms of a 0.84 s span: "
          f"{np.sum(w[:n] ** 2) / np.sum(w ** 2):.2e} / {np.sum(w[-n:] ** 2) / np.sum(w ** 2):.2e}")
    print("3. per-20 ms band-level change vs untouched, dB (low 150-400 / anti 750-1100); untouched ratio")
    edits = KEYS[2:]
    print("   window          untouched  " + "  ".join(f"{LABEL[k]:>15s}" for k in edits))
    for a in [9.42, 9.44, 9.46, 9.48, 9.50, 9.52, 10.26, 10.28, 10.30, 10.32, 10.34, 10.36]:
        b = round(a + 0.02, 2)
        y = seg("determinism", a, b)
        cells = [f"{level(seg(k, a, b), 150, 400) - level(y, 150, 400):+6.1f}/"
                 f"{level(seg(k, a, b), 750, 1100) - level(y, 750, 1100):+6.1f}" for k in edits]
        print(f"   [{a:.2f},{b:.2f}]  {ratio(y):7.2f}    " + "  ".join(f"{c:>15s}" for c in cells))
    print("4. extent of the resynthesised region (first/last 20 ms window whose samples differ from untouched)")
    edges = np.round(np.arange(9.10, 10.70, 0.02), 2)
    for k in KEYS[1:]:
        diff = [a for a in edges if np.max(np.abs(seg(k, a, a + 0.02) - seg("determinism", a, a + 0.02))) > 1e-4]
        print(f"   {LABEL[k]:16s} [{diff[0]:.2f}, {diff[-1] + 0.02:.2f}]   (WORLD context: span +/- 0.3 s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
