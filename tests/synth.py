"""Klatt-style source-filter synthesis of test signals with known acoustic ground truth."""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

SR = 16_000


def _resonator(x: np.ndarray, freq: float, bw: float, sr: int = SR) -> np.ndarray:
    r = np.exp(-np.pi * bw / sr)
    theta = 2 * np.pi * freq / sr
    a = [1.0, -2 * r * np.cos(theta), r * r]
    b = [1.0 - r]  # rough gain normalisation
    return lfilter(b, a, x)


def _anti_resonator(x: np.ndarray, freq: float, bw: float, sr: int = SR) -> np.ndarray:
    r = np.exp(-np.pi * bw / sr)
    theta = 2 * np.pi * freq / sr
    return lfilter([1.0, -2 * r * np.cos(theta), r * r], [1.0], x)


def glottal_source(duration_s: float, f0: float = 120.0, sr: int = SR, jitter: float = 0.01,
                   seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(duration_s * sr)
    x = np.zeros(n)
    t = 0.0
    while t < n:
        x[int(t)] = 1.0
        t += sr / (f0 * (1 + jitter * rng.standard_normal()))
    # Spectral tilt of the glottal flow (-12 dB/oct) with lip radiation (+6 dB/oct).
    x = lfilter([1.0], [1.0, -0.97], x)
    return x


def vowel(duration_s: float, formants: tuple[float, ...] = (700, 1750, 2600), bws: tuple[float, ...] = (80, 100, 120),
          f0: float = 120.0, amp: float = 0.5, seed: int = 0) -> np.ndarray:
    """A voiced vowel; F4/F5 are fixed at 3500/4500 Hz like an adult male vocal tract."""
    x = glottal_source(duration_s, f0, seed=seed)
    y = x
    for f, b in zip((*formants, 3500.0, 4500.0), (*bws, 200.0, 250.0), strict=False):
        y = _resonator(y, f, b)
    y = y - y.mean()
    y = y / (np.max(np.abs(y)) + 1e-9) * amp
    ramp = min(len(y) // 4, int(0.01 * SR))
    if ramp:
        env = np.ones(len(y))
        env[:ramp] = np.linspace(0, 1, ramp)
        env[-ramp:] = np.linspace(1, 0, ramp)
        y = y * env
    return y


def nasal_murmur(duration_s: float, f0: float = 120.0, amp: float = 0.35, seed: int = 0) -> np.ndarray:
    x = glottal_source(duration_s, f0, seed=seed)
    y = _resonator(x, 250, 60)
    y = _anti_resonator(y, 900, 80)
    y = y + 0.05 * _resonator(x, 2200, 200)
    y = y - y.mean()
    return y / (np.max(np.abs(y)) + 1e-9) * amp


def silence(duration_s: float, level: float = 0.0005, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return level * rng.standard_normal(int(duration_s * SR))


def burst(duration_s: float = 0.012, amp: float = 0.4, seed: int = 1) -> np.ndarray:
    """A broadband release transient (high-passed noise with fast decay)."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * SR)
    noise = rng.standard_normal(n)
    noise = lfilter([1.0, -0.95], [1.0], noise)  # emphasise high frequencies
    env = np.exp(-np.linspace(0, 5, n))
    return amp * noise * env / (np.max(np.abs(noise)) + 1e-9)


def concat(*parts: np.ndarray) -> np.ndarray:
    return np.concatenate([np.asarray(p, dtype=np.float64) for p in parts]).astype(np.float32)
