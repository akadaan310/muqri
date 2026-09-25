"""Cleaning a letter's audio: as crisp as the recording allows, without changing what was recited.

Applied to the whole ayah first (a filter or a noise estimate on a 0.2 s clip has nothing to work
with), then cut:

  1. DC offset removed.
  2. High-pass, 4th-order Butterworth at 60 Hz, zero-phase (sosfiltfilt): rumble, handling noise and
     mains hum go; the lowest voice fundamental of these reciters (~80 Hz) stays.
  3. Spectral gating: the noise floor is learned per frequency bin from the recording's own quietest
     10 % of frames (its pauses and breaths), and each STFT bin is attenuated where it is not clearly
     above that floor -- a Wiener-style gain max(1 - a * N / |X|, floor), smoothed over three frames so
     it does not warble. Over-subtraction a = 1.5, floor 0.1 (-20 dB): hiss and room tone drop, the
     weak parts of a letter (a hams release, a qalqalah echo) keep their shape.
  4. The cut is snapped to the nearest zero crossing within 2 ms, and faded in and out over 5 ms
     (raised cosine): no clicks.
  5. Loudness: RMS to -20 dBFS, peak held under -1 dBFS -- every clip at the same level, so reciters
     are compared by what they recite, not how loud their microphone was.

What is NOT done, and why: no pitch or time change, no compression, no de-essing (it would take the
whistle out of ص ز س), no de-reverberation (it smears the burst of a stop) -- anything that could move a
characteristic the engine measures.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

SR = 16000


def clean_ayah(wave: npt.NDArray[np.floating]) -> npt.NDArray[np.float32]:
    from scipy.signal import butter, sosfiltfilt
    x = np.asarray(wave, dtype=np.float64)
    x = x - x.mean()
    x = sosfiltfilt(butter(4, 60, btype="highpass", fs=SR, output="sos"), x)
    return spectral_gate(x).astype(np.float32)


def spectral_gate(x: npt.NDArray[np.floating], n_fft: int = 512, hop: int = 128, over: float = 1.5,
                  floor: float = 0.1) -> npt.NDArray[np.floating]:
    import librosa
    from scipy.ndimage import median_filter
    X = librosa.stft(x, n_fft=n_fft, hop_length=hop)
    mag = np.abs(X)
    energy = mag.sum(axis=0)
    quiet = energy <= np.quantile(energy, 0.10)
    noise = np.median(mag[:, quiet], axis=1, keepdims=True) if quiet.any() else np.zeros((mag.shape[0], 1))
    gain = np.maximum(1.0 - over * noise / np.maximum(mag, 1e-12), floor)
    gain = median_filter(gain, size=(1, 3))
    return librosa.istft(X * gain, hop_length=hop, length=len(x))


def _snap(x: npt.NDArray[np.floating], i: int, reach: int = 32) -> int:
    lo, hi = max(1, i - reach), min(len(x) - 1, i + reach)
    zc = np.where(np.signbit(x[lo - 1:hi - 1]) != np.signbit(x[lo:hi]))[0]
    return int(lo + zc[np.argmin(np.abs(zc + lo - i))]) if zc.size else i


def cut(x: npt.NDArray[np.floating], t0: float, t1: float, level: bool = True) -> npt.NDArray[np.float32]:
    a, b = _snap(x, int(t0 * SR)), _snap(x, int(t1 * SR))
    seg = np.array(x[a:max(b, a + 1)], dtype=np.float64)
    n = min(int(0.005 * SR), len(seg) // 2)
    if n > 1:
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n))
        seg[:n] *= ramp
        seg[-n:] *= ramp[::-1]
    if level and seg.size:
        rms = np.sqrt(np.mean(seg ** 2)) or 1e-9
        seg *= 10 ** (-20 / 20) / rms
        peak = np.abs(seg).max()
        if peak > 10 ** (-1 / 20):
            seg *= 10 ** (-1 / 20) / peak
    return seg.astype(np.float32)
