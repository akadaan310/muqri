"""What makes synthetic recitation sound unnatural, measured: pitch wobble and voicing breaks (Praat
pitch, 10 ms), next to the engine's letter and word checks. Real Husary is the reference."""

from __future__ import annotations

import numpy as np
import parselmouth
import soundfile as sf


def pitch_steadiness(path, max_s: float | None = None) -> dict[str, float]:  # type: ignore[no-untyped-def]
    y, sr = sf.read(path, dtype="float32")
    if max_s:
        y = y[: int(max_s * sr)]
    f = parselmouth.Sound(y, sr).to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=500).selected_array["frequency"]
    v = f > 0
    both = v[1:] & v[:-1]
    st = np.abs(12 * np.log2(f[1:][both] / f[:-1][both]))
    return {"voiced": float(v.mean()), "breaks_per_s": float(np.sum(v[1:] != v[:-1]) / 2 / (len(y) / sr)),
            "wobble_st": float(np.median(st)), "jumps_gt1st_pct": float((st > 1).mean() * 100)}
