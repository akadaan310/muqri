"""Acoustic environment estimation and dereverberation for live (mosque / stadium) recordings.

* **RT60 estimation** (blind): after a speech offset, a reverberant room's energy decays
  exponentially, i.e. linearly in dB. The decay slopes after many offsets are fitted in the
  500–4000 Hz band and the median RT is reported. Recitation pauses are short, so decays are
  truncated and the estimate is biased low for rooms above ~1 s (synthetic check: true 1.0 / 1.5 s
  → 0.7 / 0.8 s); it is reliable for detecting *whether* a room is reverberant.
* **WPE dereverberation** (Weighted Prediction Error, Nakatani et al. 2010, via ``nara_wpe``):
  a delayed linear predictor per frequency bin estimates the late reverberation from past
  frames and subtracts it, leaving the direct sound and early reflections. It runs in 30 s
  chunks so arbitrarily long Taraweeh sessions fit in memory.
* **Late-reverberation suppression** (Lebart 2001 / Habets 2009): the late-reverb power in each
  bin is predicted as the smoothed power 50 ms earlier, attenuated by the room's exponential
  decay, and removed with a floored spectral gain. On synthetic rooms (RT60 0.6–2 s) the chain
  lowers the reverberant energy in pauses by 3–5 dB; WPE alone manages about 1 dB.
* **Proximity EQ**: handheld dynamic microphones boost the bass when held close. A 120 Hz
  high-pass removes it before formant analysis.
* **Environment profile** (8-d, part of the fingerprint): RT60, SNR, room volume estimate, bass
  boost, spectral tilt, background (crowd) noise level, echo density and clipping ratio.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from app.audio import AudioSignal, estimate_snr_db, frame_rms_db, highpass

logger = logging.getLogger(__name__)

FloatArray: TypeAlias = npt.NDArray[np.float32]
PROXIMITY_HPF_HZ = 120.0
WPE_CHUNK_S = 30.0
WPE_OVERLAP_S = 1.0


# --------------------------------------------------------------------------- RT60
def _band_envelope_db(x: npt.NDArray[np.floating], sr: int, hop_ms: float = 10.0) -> npt.NDArray[np.float64]:
    from scipy.signal import butter, sosfiltfilt

    sos = butter(4, [500.0, min(4000.0, sr / 2 - 100)], btype="bandpass", fs=sr, output="sos")
    y = sosfiltfilt(sos, np.asarray(x, dtype=np.float64))
    return frame_rms_db(y, sr, frame_ms=20.0, hop_ms=hop_ms).astype(np.float64)


def estimate_rt60(x: npt.NDArray[np.floating], sr: int) -> float:
    """Blind RT60 (seconds) from free-decay regions after speech offsets; NaN if none found.

    Each offset's decay is fitted from 3 dB below the peak down to at most 25 dB (a T20-style
    fit that skips the plateau of the ending vowel). Offsets cut short by the next syllable still
    count when they fall at least 8 dB.
    """
    hop = 0.01
    env = _band_envelope_db(x, sr)
    if env.size < 50:
        return float("nan")
    floor = float(np.percentile(env, 5))
    estimates: list[float] = []
    i, n = 1, len(env)
    while i < n - 5:
        if env[i] >= env[i - 1] and env[i] >= env[i + 1] and env[i] > floor + 15:
            j = i + 1
            while j < n and env[j] <= env[j - 1] + 0.5:
                j += 1
            seg = env[i:j] - env[i]
            below3 = np.flatnonzero(seg <= -3.0)
            if below3.size:
                a = int(below3[0])
                deepest = max(-25.0, float(seg.min()), floor + 3 - env[i])
                b_idx = np.flatnonzero(seg <= deepest)
                b = int(b_idx[0]) if b_idx.size else len(seg) - 1
                if b - a >= 3 and seg[a] - seg[b] >= 8.0:
                    t = np.arange(a, b + 1) * hop
                    slope = np.polyfit(t, seg[a: b + 1], 1)[0]
                    if slope < -5:
                        estimates.append(-60.0 / slope)
            i = max(j, i + 1)
        else:
            i += 1
    if len(estimates) < 3:
        return float("nan")
    return float(np.median(estimates))


# --------------------------------------------------------------------------- WPE
def wpe_dereverb(x: npt.NDArray[np.floating], sr: int, *, taps: int = 10, delay: int = 3,
                 iterations: int = 3) -> FloatArray:
    """Single-channel WPE dereverberation, chunked with cross-faded overlaps."""
    try:
        from nara_wpe.utils import istft, stft
        from nara_wpe.wpe import wpe
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Dereverberation requires `nara_wpe` (pip install nara_wpe)") from exc
    x = np.asarray(x, dtype=np.float64)
    size, shift = 512, 128

    def run(seg: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        spec = stft(seg[None, :], size=size, shift=shift)  # (D=1, T, F)
        y = wpe(spec.transpose(2, 0, 1), taps=taps, delay=delay, iterations=iterations)  # (F, D, T)
        out = istft(y.transpose(1, 2, 0), size=size, shift=shift)[0]
        return out[: len(seg)] if len(out) >= len(seg) else np.pad(out, (0, len(seg) - len(out)))

    chunk, ov = int(WPE_CHUNK_S * sr), int(WPE_OVERLAP_S * sr)
    if len(x) <= chunk + ov:
        return run(x).astype(np.float32)
    out = np.zeros_like(x)
    weight = np.zeros_like(x)
    start = 0
    while start < len(x):
        end = min(len(x), start + chunk + ov)
        seg_out = run(x[start:end])
        w = np.ones(end - start)
        ramp = min(ov, end - start)
        if start > 0:
            w[:ramp] = np.linspace(0, 1, ramp)
        if end < len(x):
            w[-ramp:] = np.linspace(1, 0, ramp)
        out[start:end] += seg_out * w
        weight[start:end] += w
        if end == len(x):
            break
        start += chunk
    return (out / np.maximum(weight, 1e-6)).astype(np.float32)


def suppress_late_reverb(x: npt.NDArray[np.floating], sr: int, rt60: float, *, late_ms: float = 50.0,
                         floor_db: float = -15.0, smoothing: float = 0.9) -> FloatArray:
    """Statistical late-reverberation suppression given an RT60 estimate (seconds)."""
    from scipy.signal import istft, stft

    nper, hop = 512, 128
    _, _, z = stft(np.asarray(x, dtype=np.float64), fs=sr, nperseg=nper, noverlap=nper - hop)
    power = np.abs(z) ** 2
    smooth = np.copy(power)
    for k in range(1, power.shape[1]):
        smooth[:, k] = smoothing * smooth[:, k - 1] + (1 - smoothing) * power[:, k]
    nd = max(1, int(late_ms / 1000 * sr / hop))
    decay = np.exp(-2 * (6.9 / max(rt60, 0.05)) * nd * hop / sr)
    late = np.zeros_like(power)
    late[:, nd:] = decay * smooth[:, :-nd]
    gain = np.maximum(1 - late / (power + 1e-12), 10 ** (floor_db / 10))
    _, y = istft(z * gain, fs=sr, nperseg=nper, noverlap=nper - hop)
    y = y[: len(x)]
    return np.pad(y, (0, len(x) - len(y))).astype(np.float32) if len(y) < len(x) else y.astype(np.float32)


# --------------------------------------------------------------------------- environment profile
def bass_boost_db(x: npt.NDArray[np.floating], sr: int) -> float:
    """Level of 80–250 Hz relative to 250–1000 Hz in the long-term spectrum (dB)."""
    from app.acoustic.features import band_power_db

    return band_power_db(x, sr, 80.0, 250.0) - band_power_db(x, sr, 250.0, 1000.0)


def spectral_tilt_db_per_octave(x: npt.NDArray[np.floating], sr: int) -> float:
    """Slope of the long-term average spectrum between 100 Hz and 5 kHz (dB/octave)."""
    from scipy.signal import welch

    f, p = welch(np.asarray(x, dtype=np.float64), fs=sr, nperseg=2048)
    sel = (f >= 100) & (f <= min(5000.0, sr / 2 - 1))
    if sel.sum() < 4:
        return float("nan")
    return float(np.polyfit(np.log2(f[sel]), 10 * np.log10(p[sel] + 1e-20), 1)[0])


def background_noise_db(x: npt.NDArray[np.floating], sr: int) -> float:
    """Noise level in the quietest frames relative to the speech level (dB, negative)."""
    db = frame_rms_db(x, sr, 25.0, 10.0)
    return float(np.percentile(db, 10) - np.percentile(db, 95)) if db.size >= 10 else float("nan")


def echo_density(x: npt.NDArray[np.floating], sr: int) -> float:
    """Discrete-echo density: significant envelope-autocorrelation peaks per second at 30–300 ms lags."""
    env = frame_rms_db(x, sr, 10.0, 5.0).astype(np.float64)
    if env.size < 200:
        return float("nan")
    env = env - env.mean()
    ac = np.correlate(env, env, mode="full")[len(env) - 1:]
    ac /= ac[0] + 1e-12
    lo, hi = 6, 60  # 30–300 ms at 5 ms hop
    seg = ac[lo:hi]
    peaks = [k for k in range(1, len(seg) - 1) if seg[k] > seg[k - 1] and seg[k] > seg[k + 1] and seg[k] > 0.1]
    return float(len(peaks) / ((hi - lo) * 0.005))


def gap_reverb_ratio_db(x: npt.NDArray[np.floating], speech_mask: npt.NDArray[np.bool_], sr: int) -> float:
    """Energy in the pauses relative to the energy in speech (dB), given a 10 ms-hop speech mask.

    Reverberation fills the pauses with the decaying tail of the preceding sound, so this ratio
    drops when dereverberation works.
    """
    db = frame_rms_db(x, sr, 20.0, 10.0).astype(np.float64)
    n = min(len(db), len(speech_mask))
    db, mask = db[:n], speech_mask[:n]
    if mask.all() or not mask.any():
        return float("nan")
    power = 10 ** (db / 10)
    return float(10 * np.log10(power[~mask].mean() / power[mask].mean()))


def room_volume_estimate(rt60: float) -> float:
    """Very rough room volume (m³) from RT60 assuming average absorption (V ≈ (RT60 / 0.07)³)."""
    return float((rt60 / 0.07) ** 3) if np.isfinite(rt60) and rt60 > 0 else float("nan")


@dataclass(slots=True)
class EnvironmentProfile:
    rt60_reverberation_time: float
    snr_db: float
    room_volume_estimate: float
    mic_proximity_bass_boost: float
    spectral_tilt: float
    background_crowd_noise_level: float
    echo_density: float
    audio_clipping_ratio: float

    def as_array(self) -> npt.NDArray[np.float64]:
        return np.array([getattr(self, f) for f in ENV_FIELDS], dtype=np.float64)

    def to_dict(self) -> dict[str, float | None]:
        return {k: (round(float(v), 4) if np.isfinite(v) else None) for k, v in asdict(self).items()}

    @property
    def reverberant(self) -> bool:
        return bool(np.isfinite(self.rt60_reverberation_time) and self.rt60_reverberation_time > 0.6)


ENV_FIELDS = (
    "rt60_reverberation_time", "snr_db", "room_volume_estimate", "mic_proximity_bass_boost", "spectral_tilt",
    "background_crowd_noise_level", "echo_density", "audio_clipping_ratio",
)


def environment_profile(audio: AudioSignal) -> EnvironmentProfile:
    x, sr = audio.samples, audio.sr
    rt60 = estimate_rt60(x, sr)
    return EnvironmentProfile(
        rt60_reverberation_time=rt60, snr_db=estimate_snr_db(x, sr), room_volume_estimate=room_volume_estimate(rt60),
        mic_proximity_bass_boost=bass_boost_db(x, sr), spectral_tilt=spectral_tilt_db_per_octave(x, sr),
        background_crowd_noise_level=background_noise_db(x, sr), echo_density=echo_density(x, sr),
        audio_clipping_ratio=audio.quality.clipping_ratio,
    )


# --------------------------------------------------------------------------- the adapter step
@dataclass(slots=True)
class DereverbResult:
    audio: AudioSignal
    rt60_before: float
    rt60_after: float
    applied: list[str]


def adapt_acoustics(audio: AudioSignal, *, dereverb: bool = True, proximity_eq: bool = True,
                    min_rt60: float = 0.3) -> DereverbResult:
    """Strip reverberation and proximity bass so live audio behaves like a dry studio take."""
    x = audio.samples
    rt_before = estimate_rt60(x, audio.sr)
    applied: list[str] = []
    if proximity_eq:
        x = highpass(x, audio.sr, PROXIMITY_HPF_HZ)
        applied.append(f"high-pass {PROXIMITY_HPF_HZ:.0f} Hz (proximity EQ)")
    # NaN means no free decay was measurable at all, i.e. a dry recording.
    if dereverb and np.isfinite(rt_before) and rt_before >= min_rt60:
        try:
            x = wpe_dereverb(x, audio.sr)
            applied.append("WPE dereverberation")
        except RuntimeError as exc:
            logger.warning("WPE skipped: %s", exc)
        x = suppress_late_reverb(x, audio.sr, rt_before)
        applied.append(f"late-reverb suppression (RT60 {rt_before:.2f}s)")
    x = np.asarray(x, dtype=np.float32)
    x = x / (np.max(np.abs(x)) + 1e-9) * 0.891
    out = AudioSignal(samples=x, sr=audio.sr, quality=audio.quality)
    return DereverbResult(out, rt_before, estimate_rt60(x, audio.sr), applied)
