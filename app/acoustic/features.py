"""Shared DSP primitives: band energies, formant tracking, pitch tracking, spectral flux.

Praat (via ``praat-parselmouth``) is used for formants and F0 when installed, which gives the
Burg-LPC formant tracker phoneticians rely on. A pure numpy/scipy LPC fallback keeps the engine
functional without it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import cached_property
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from app.audio import AudioSignal

logger = logging.getLogger(__name__)

try:  # pragma: no cover - availability depends on the environment
    import parselmouth

    HAVE_PARSELMOUTH = True
except ImportError:  # pragma: no cover
    parselmouth = None
    HAVE_PARSELMOUTH = False

FloatArray: TypeAlias = npt.NDArray[np.float64]
NUCLEUS_DB = 6.0


@dataclass(slots=True)
class FormantEstimate:
    f1: float
    f2: float
    f3: float
    b1: float
    b2: float
    n_frames: int

    @property
    def valid(self) -> bool:
        return self.n_frames > 0 and np.isfinite(self.f1) and np.isfinite(self.f2)


NAN_FORMANTS = FormantEstimate(np.nan, np.nan, np.nan, np.nan, np.nan, 0)


def band_energy(x: npt.NDArray[np.floating], sr: int, lo: float, hi: float) -> float:
    """Mean power in [lo, hi) Hz from a Hann-windowed periodogram."""
    if len(x) < 16:
        return 0.0
    win = np.hanning(len(x))
    n_fft = int(2 ** np.ceil(np.log2(max(len(x), 512))))
    spec = np.abs(np.fft.rfft(x * win, n=n_fft)) ** 2
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    mask = (freqs >= lo) & (freqs < hi)
    return float(spec[mask].mean()) if mask.any() else 0.0


def band_power_db(x: npt.NDArray[np.floating], sr: int, lo: float, hi: float) -> float:
    return float(10.0 * np.log10(band_energy(x, sr, lo, hi) + 1e-20))


def spectral_flatness(x: npt.NDArray[np.floating], sr: int, lo: float, hi: float) -> float:
    """Wiener entropy (geometric / arithmetic mean of the power spectrum) inside [lo, hi) Hz.

    1.0 is white noise spread evenly across the band; values near 0 mean a peaky spectrum.
    """
    if len(x) < 64:
        return float("nan")
    n_fft = int(2 ** np.ceil(np.log2(max(len(x), 512))))
    spec = np.abs(np.fft.rfft(np.asarray(x, dtype=np.float64) * np.hanning(len(x)), n=n_fft)) ** 2
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    band = spec[(freqs >= lo) & (freqs < hi)] + 1e-20
    if band.size == 0:
        return float("nan")
    return float(np.exp(np.mean(np.log(band))) / np.mean(band))


def band_energy_ratio_db(x: npt.NDArray[np.floating], sr: int, num: tuple[float, float],
                         den: tuple[float, float]) -> float:
    a = band_energy(x, sr, *num)
    b = band_energy(x, sr, *den)
    return float(10.0 * np.log10((a + 1e-12) / (b + 1e-12)))


def short_time_rms(x: npt.NDArray[np.floating], sr: int, frame_ms: float = 5.0, hop_ms: float = 1.0) -> FloatArray:
    frame = max(2, int(sr * frame_ms / 1000))
    hop = max(1, int(sr * hop_ms / 1000))
    if len(x) < frame:
        return np.zeros(0)
    n = 1 + (len(x) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    return np.sqrt(np.mean(np.square(np.asarray(x, dtype=np.float64)[idx]), axis=1) + 1e-14)


def highband_flux(x: npt.NDArray[np.floating], sr: int, cutoff_hz: float = 1500.0, frame_ms: float = 8.0,
                  hop_ms: float = 1.0) -> FloatArray:
    """Half-wave rectified spectral flux restricted to frequencies above ``cutoff_hz``."""
    frame = max(16, int(sr * frame_ms / 1000))
    hop = max(1, int(sr * hop_ms / 1000))
    if len(x) < frame + hop:
        return np.zeros(0)
    n = 1 + (len(x) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    frames = np.asarray(x, dtype=np.float64)[idx] * np.hanning(frame)[None, :]
    spec = np.abs(np.fft.rfft(frames, axis=1))
    freqs = np.fft.rfftfreq(frame, 1.0 / sr)
    spec = np.log1p(spec[:, freqs >= cutoff_hz] * 100.0)
    flux = np.maximum(np.diff(spec, axis=0), 0.0).sum(axis=1)
    return np.concatenate([[0.0], flux])


def _lpc_formants(x: FloatArray, sr: int, max_formant: float) -> tuple[list[float], list[float]]:
    """Formants of one frame by LPC root solving (Burg-free autocorrelation method)."""
    from scipy.linalg import solve_toeplitz
    from scipy.signal import resample_poly

    target_sr = int(2 * max_formant)
    if sr != target_sr:
        g = np.gcd(sr, target_sr)
        x = resample_poly(x, target_sr // g, sr // g)
    x = np.append(x[0], x[1:] - 0.97 * x[:-1]) * np.hamming(len(x))
    order = 2 + target_sr // 1000
    r = np.correlate(x, x, mode="full")[len(x) - 1: len(x) + order]
    if r[0] <= 0:
        return [], []
    r[0] *= 1.0 + 1e-9
    try:
        a = solve_toeplitz(r[:order], -r[1: order + 1])
    except np.linalg.LinAlgError:
        return [], []
    roots = np.roots(np.concatenate([[1.0], a]))
    roots = roots[np.imag(roots) > 0]
    freqs = np.angle(roots) * target_sr / (2 * np.pi)
    bws = -np.log(np.abs(roots) + 1e-12) * target_sr / np.pi
    keep = (freqs > 90) & (bws < 600) & (freqs < max_formant - 50)
    order_idx = np.argsort(freqs[keep])
    return list(freqs[keep][order_idx]), list(bws[keep][order_idx])


class AcousticContext:
    """Lazily computes and caches whole-utterance Praat analyses for an :class:`AudioSignal`."""

    def __init__(self, audio: AudioSignal) -> None:
        self.audio = audio
        self.sr = audio.sr
        self.x = audio.samples.astype(np.float64)

    @cached_property
    def _sound(self):  # type: ignore[no-untyped-def]
        return parselmouth.Sound(self.x, sampling_frequency=self.sr) if HAVE_PARSELMOUTH else None

    @cached_property
    def f0_track(self) -> tuple[FloatArray, FloatArray]:
        """(times_s, f0_hz) with NaN for unvoiced frames."""
        if HAVE_PARSELMOUTH:
            pitch = self._sound.to_pitch_ac(time_step=0.01, pitch_floor=60.0, pitch_ceiling=600.0)
            f0 = np.asarray(pitch.selected_array["frequency"], dtype=np.float64)
            f0[f0 <= 0] = np.nan
            return np.asarray(pitch.xs(), dtype=np.float64), f0
        import librosa

        f0, voiced, _ = librosa.pyin(self.x, fmin=60, fmax=600, sr=self.sr, frame_length=1024, hop_length=160)
        f0 = np.where(voiced, f0, np.nan)
        return librosa.times_like(f0, sr=self.sr, hop_length=160), f0

    @cached_property
    def median_f0(self) -> float:
        _, f0 = self.f0_track
        return float(np.nanmedian(f0)) if np.any(np.isfinite(f0)) else float("nan")

    @cached_property
    def max_formant(self) -> float:
        """Praat's recommended formant ceiling: 5000 Hz for low voices, 5500 Hz for high voices."""
        f0 = self.median_f0
        return 5500.0 if np.isfinite(f0) and f0 > 165.0 else 5000.0

    @cached_property
    def _formant_obj(self):  # type: ignore[no-untyped-def]
        if not HAVE_PARSELMOUTH:
            return None
        return self._sound.to_formant_burg(
            time_step=0.005, max_number_of_formants=5, maximum_formant=self.max_formant,
            window_length=0.025, pre_emphasis_from=50.0,
        )

    def formants(self, start_s: float, end_s: float) -> FormantEstimate:
        """Median F1/F2/F3 and bandwidths over the voiced frames of ``[start_s, end_s]``."""
        if end_s - start_s < 0.01:
            return NAN_FORMANTS
        rows: list[tuple[float, float, float, float, float, float]] = []
        times = np.arange(start_s + 0.0125, end_s - 0.0125 + 1e-9, 0.005)
        if times.size == 0:
            times = np.array([(start_s + end_s) / 2])
        f0_t, f0 = self.f0_track
        if HAVE_PARSELMOUTH:
            fo = self._formant_obj
            for t in times:
                vals = [fo.get_value_at_time(k, t) for k in (1, 2, 3)]
                bws = [fo.get_bandwidth_at_time(k, t) for k in (1, 2)]
                rows.append((t, *vals, *bws))  # type: ignore[arg-type]
        else:
            half = int(0.0125 * self.sr)
            for t in times:
                c = int(t * self.sr)
                seg = self.x[max(0, c - half): c + half]
                if len(seg) < 2 * half:
                    continue
                fr, bw = _lpc_formants(seg, self.sr, self.max_formant)
                if len(fr) >= 2:
                    rows.append((t, fr[0], fr[1], fr[2] if len(fr) > 2 else np.nan, bw[0], bw[1]))
        if not rows:
            return NAN_FORMANTS
        arr = np.asarray(rows, dtype=np.float64)
        # Keep voiced frames only when the pitch track marks enough of the window as voiced.
        if f0_t.size:
            voiced = np.interp(arr[:, 0], f0_t, np.isfinite(f0).astype(float)) > 0.5
            if voiced.sum() >= 2:
                arr = arr[voiced]
        # Measure the vowel nucleus: frames within NUCLEUS_DB of the window's loudest frame.
        # Low-energy transitions into/out of neighbouring consonants mis-track formants.
        if arr.shape[0] >= 3:
            half = int(0.0125 * self.sr)
            centres = (arr[:, 0] * self.sr).astype(int)
            level = np.array([
                10 * np.log10(np.mean(np.square(self.x[max(0, c - half): c + half])) + 1e-12) for c in centres
            ])
            nucleus = level >= level.max() - NUCLEUS_DB
            if nucleus.sum() >= 2:
                arr = arr[nucleus]
        arr = arr[np.isfinite(arr[:, 1]) & np.isfinite(arr[:, 2]), 1:]
        if arr.size == 0:
            return NAN_FORMANTS
        f1, f2, f3, b1, b2 = (float(v) for v in np.nanmedian(arr, axis=0))
        return FormantEstimate(f1, f2, f3, b1, b2, n_frames=int(arr.shape[0]))

    @cached_property
    def _harmonicity(self):  # type: ignore[no-untyped-def]
        if not HAVE_PARSELMOUTH:
            return None
        return self._sound.to_harmonicity_cc(time_step=0.005, minimum_pitch=75.0, silence_threshold=0.1,
                                             periods_per_window=1.0)

    def hnr(self, start_s: float, end_s: float) -> float:
        """Median harmonics-to-noise ratio (dB) over ``[start_s, end_s]``.

        Praat's cross-correlation harmonicity is used when available; otherwise the normalised
        autocorrelation peak r in the 75–500 Hz lag range gives HNR = 10·log10(r / (1 − r)).
        """
        if end_s - start_s < 0.02:
            return float("nan")
        if HAVE_PARSELMOUTH:
            h = self._harmonicity
            times = np.arange(start_s + 0.01, end_s - 0.01 + 1e-9, 0.005)
            vals = np.array([h.get_value(t) for t in times], dtype=np.float64) if times.size else np.zeros(0)
            vals = vals[np.isfinite(vals) & (vals > -150)]
            return float(np.median(vals)) if vals.size else float("nan")
        seg = self.audio.segment(start_s, end_s).astype(np.float64)
        frame, hop = int(0.04 * self.sr), int(0.01 * self.sr)
        if len(seg) < frame:
            frame = len(seg)
        win = np.hanning(frame)
        # Boersma (1993): divide the frame autocorrelation by the window's own autocorrelation.
        w_ac = np.correlate(win, win, mode="full")[frame - 1:]
        lo, hi = int(self.sr / 500), min(frame - 1, int(self.sr / 75))
        vals = []
        for a in range(0, len(seg) - frame + 1, hop):
            x = seg[a: a + frame] - seg[a: a + frame].mean()
            if not np.any(x):
                continue
            ac = np.correlate(x * win, x * win, mode="full")[frame - 1:]
            ac = ac / (ac[0] + 1e-12) / (w_ac / w_ac[0] + 1e-12)
            if hi > lo:
                r = float(np.clip(np.max(ac[lo:hi]), 1e-4, 0.9999))
                vals.append(10 * np.log10(r / (1 - r)))
        return float(np.median(vals)) if vals else float("nan")

    def f0_in(self, start_s: float, end_s: float) -> FloatArray:
        t, f0 = self.f0_track
        sel = (t >= start_s) & (t <= end_s)
        vals = f0[sel]
        return vals[np.isfinite(vals)]
