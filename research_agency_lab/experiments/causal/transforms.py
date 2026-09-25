"""Single-variable transforms on one span of an ayah, built only from implementations that already exist:

  duration   letter_corpus/perturb.py::stretch (librosa phase vocoder, pitch kept)       param k (x length)
  voicing    synthesis/world_lab.py::devoice (WORLD aperiodicity -> 1, f0 -> 0 at 0 %)  param level (1 = as is)
  formant    synthesis/world_lab.py::warp (envelope read at f x a; a < 1 moves formants up)   param a
  f0         WORLD f0 track scaled by 2^(s/12) on the span (the vocoder's own pitch input)  param semitones
  nasal      synthesis/world_lab.py::nasal (pole ~250 Hz +g dB, anti-resonance ~900 Hz -g dB)  param g_db

and the controls:

  noop_splice   the untouched samples spliced back (the harness itself must change nothing)
  noop_world    WORLD analysis -> resynthesis with no parameter changed (the vocoder's own footprint)
  noop_pv       the phase vocoder at k = 1.0 (the stretcher's own footprint)
  swap          the span replaced by a donor span (same letter elsewhere / a neighbouring letter)

Every WORLD edit is made on the span plus 0.3 s of context either side, resynthesised, and spliced
back with 10 ms crossfades (perturb.splice). The span's new end is returned: only `duration` moves it.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

SR = 16000
CTX = 0.3


def _world_edit(wave: np.ndarray, t0: float, t1: float, edit: Callable | None) -> np.ndarray:  # type: ignore[type-arg]
    from research_agency_lab.experiments.letter_corpus.perturb import splice
    from research_agency_lab.experiments.synthesis.world_lab import FP, synth, world
    lo, hi = max(0.0, t0 - CTX), min(len(wave) / SR, t1 + CTX)
    seg = wave[int(lo * SR):int(hi * SR)]
    f0, sp, ap = world(seg)
    if edit is not None:
        w = np.zeros(len(f0))
        a, b = int(round((t0 - lo) * 1000 / FP)), int(round((t1 - lo) * 1000 / FP))
        w[a:b] = 1.0
        f0, sp, ap = edit(f0.copy(), sp.copy(), ap.copy(), w)
    return splice(wave, lo, hi, synth(f0, sp, ap, len(seg)))


def apply(wave: np.ndarray, t0: float, t1: float, name: str, p: dict[str, Any],  # type: ignore[type-arg]
          donor: np.ndarray | None = None) -> tuple[np.ndarray, float, str]:  # type: ignore[type-arg]
    """-> (new wave, new span end, implementation name)."""
    from research_agency_lab.experiments.letter_corpus.perturb import splice, stretch
    from research_agency_lab.experiments.synthesis import world_lab as W
    seg = wave[int(t0 * SR):int(t1 * SR)]
    if name == "noop_splice":
        return splice(wave, t0, t1, seg.copy()), t1, "perturb.splice(original samples)"
    if name in ("duration", "noop_pv"):
        k = 1.0 if name == "noop_pv" else float(p["k"])
        new = stretch(seg, k)
        return splice(wave, t0, t1, new), t0 + len(new) / SR, "perturb.stretch (librosa phase vocoder)"
    if name == "swap":
        assert donor is not None
        return splice(wave, t0, t1, donor), t0 + len(donor) / SR, "perturb.splice(donor span)"
    if name == "noop_world":
        return _world_edit(wave, t0, t1, None), t1, "world_lab.world -> synth (no edit)"
    if name == "voicing":
        d = 1.0 - float(p["level"])
        return _world_edit(wave, t0, t1, lambda f0, sp, ap, w: W.devoice(f0, sp, ap, w, d)), t1, "world_lab.devoice"
    if name == "formant":
        a = float(p["a"])
        return _world_edit(wave, t0, t1, lambda f0, sp, ap, w: W.warp(f0, sp, ap, w, a)), t1, "world_lab.warp"
    if name == "nasal":
        g = float(p["g_db"])
        return _world_edit(wave, t0, t1, lambda f0, sp, ap, w: W.nasal(f0, sp, ap, w, g)), t1, "world_lab.nasal"
    if name == "f0":
        r = 2 ** (float(p["semitones"]) / 12)
        return _world_edit(wave, t0, t1, lambda f0, sp, ap, w: (np.where(w > 0, f0 * r, f0), sp, ap)), t1, \
            "WORLD f0 x 2^(s/12) (world_lab.world/synth)"
    raise ValueError(f"unknown transform {name}")
