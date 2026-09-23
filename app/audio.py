"""Robust audio loading and conditioning.

Handles WAV/FLAC/OGG/MP3 input, multichannel down-mixing, arbitrary sample rates, NaN/Inf
samples, DC offset, clipping detection, SNR estimation and optional spectral-subtraction
denoising. Timestamps are preserved (no trimming) so alignments map directly onto the file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

TARGET_SR = 16_000
MIN_DURATION_S = 0.3
MAX_DURATION_S = 600.0
CLIP_LEVEL = 0.999

FloatArray: TypeAlias = npt.NDArray[np.float32]


class AudioError(RuntimeError):
    pass


@dataclass(slots=True)
class AudioQuality:
    original_sr: int
    channels: int
    duration_s: float
    peak_dbfs: float
    clipping_ratio: float
    snr_db: float
    denoised: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "original_sample_rate": self.original_sr,
            "channels": self.channels,
            "duration_s": round(self.duration_s, 3),
            "peak_dbfs": round(self.peak_dbfs, 2),
            "clipping_ratio": round(self.clipping_ratio, 5),
            "estimated_snr_db": round(self.snr_db, 1),
            "denoised": self.denoised,
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class AudioSignal:
    samples: FloatArray
    sr: int
    quality: AudioQuality

    @property
    def duration_s(self) -> float:
        return len(self.samples) / self.sr

    def segment(self, start_s: float, end_s: float) -> FloatArray:
        a = max(0, int(round(start_s * self.sr)))
        b = min(len(self.samples), int(round(end_s * self.sr)))
        return self.samples[a:b] if b > a else np.zeros(0, dtype=np.float32)


def _read(path: Path) -> tuple[npt.NDArray[np.float64], int]:
    """Return (samples[frames, channels], sr)."""
    try:
        import soundfile as sf

        data, sr = sf.read(str(path), dtype="float64", always_2d=True)
        return data, int(sr)
    except Exception as sf_exc:  # noqa: BLE001 - fall through to librosa/audioread
        try:
            import librosa

            data, sr = librosa.load(str(path), sr=None, mono=False)
        except Exception as lr_exc:  # noqa: BLE001
            raise AudioError(f"Could not decode {path}: {sf_exc}; {lr_exc}") from lr_exc
        data = np.atleast_2d(np.asarray(data, dtype=np.float64))
        return data.T, int(sr)


def resample(x: npt.NDArray[np.floating], orig_sr: int, target_sr: int) -> FloatArray:
    if orig_sr == target_sr:
        return np.asarray(x, dtype=np.float32)
    try:
        import soxr

        return np.asarray(soxr.resample(x, orig_sr, target_sr, quality="HQ"), dtype=np.float32)
    except ImportError:
        from scipy.signal import resample_poly

        g = np.gcd(orig_sr, target_sr)
        return np.asarray(resample_poly(x, target_sr // g, orig_sr // g), dtype=np.float32)


def highpass(x: npt.NDArray[np.floating], sr: int, cutoff_hz: float = 40.0) -> FloatArray:
    """Zero-phase high-pass removing DC offset and handling/HVAC rumble below the voice range."""
    from scipy.signal import butter, sosfiltfilt

    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    if len(x) < 64:
        return x.astype(np.float32)
    sos = butter(4, cutoff_hz, btype="highpass", fs=sr, output="sos")
    return sosfiltfilt(sos, x).astype(np.float32)


def frame_rms_db(x: npt.NDArray[np.floating], sr: int, frame_ms: float = 25.0, hop_ms: float = 10.0) -> FloatArray:
    frame = max(1, int(sr * frame_ms / 1000))
    hop = max(1, int(sr * hop_ms / 1000))
    if len(x) < frame:
        x = np.pad(x, (0, frame - len(x)))
    n = 1 + (len(x) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    rms = np.sqrt(np.mean(np.square(x[idx], dtype=np.float64), axis=1) + 1e-12)
    return (20.0 * np.log10(rms + 1e-12)).astype(np.float32)


def estimate_snr_db(x: npt.NDArray[np.floating], sr: int) -> float:
    db = frame_rms_db(x, sr)
    if db.size < 4:
        return 0.0
    noise = float(np.percentile(db, 10))
    signal = float(np.percentile(db, 95))
    return max(0.0, signal - noise)


def spectral_subtract(x: FloatArray, sr: int, *, noise_percentile: float = 10.0, over_sub: float = 1.5) -> FloatArray:
    """Conservative spectral-subtraction denoiser using the quietest frames as noise profile."""
    from scipy.signal import istft, stft

    nper = int(0.032 * sr)
    _, _, z = stft(x, fs=sr, nperseg=nper, noverlap=nper * 3 // 4)
    mag, phase = np.abs(z), np.angle(z)
    energy = mag.sum(axis=0)
    quiet = energy <= np.percentile(energy, noise_percentile)
    if not np.any(quiet):
        return x
    noise = mag[:, quiet].mean(axis=1, keepdims=True)
    floor = 0.05 * mag
    clean = np.maximum(mag - over_sub * noise, floor)
    _, y = istft(clean * np.exp(1j * phase), fs=sr, nperseg=nper, noverlap=nper * 3 // 4)
    y = y[: len(x)]
    if len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    return y.astype(np.float32)


def condition(
    data: npt.NDArray[np.floating],
    sr: int,
    *,
    target_sr: int = TARGET_SR,
    denoise: str = "auto",
    snr_threshold_db: float = 15.0,
) -> AudioSignal:
    """Condition raw samples (frames x channels, or 1-D) into a normalized mono signal."""
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise AudioError("Audio contains no samples")
    channels = arr.shape[1]
    warnings: list[str] = []

    bad = ~np.isfinite(arr)
    if bad.any():
        warnings.append(f"Replaced {int(bad.sum())} non-finite samples with zeros")
        arr = np.where(bad, 0.0, arr)

    # Clipping is measured per channel before down-mixing. Integer PCM decodes to [-1, 1]; float
    # files may exceed it, in which case the observed peak is treated as full scale.
    full_scale = max(1.0, float(np.max(np.abs(arr))))
    clip_ratio = float(np.mean(np.abs(arr) >= CLIP_LEVEL * full_scale))
    if clip_ratio > 0.001:
        warnings.append(
            f"Clipping detected on {clip_ratio * 100:.2f}% of samples; formant/burst measurements may be biased"
        )

    mono = arr.mean(axis=1)
    duration = len(mono) / sr
    if duration < MIN_DURATION_S:
        raise AudioError(f"Audio is too short ({duration:.2f}s < {MIN_DURATION_S}s)")
    if duration > MAX_DURATION_S:
        raise AudioError(f"Audio is too long ({duration:.0f}s > {MAX_DURATION_S:.0f}s); split it per ayah")

    peak = float(np.max(np.abs(mono)))
    if peak < 1e-5:
        raise AudioError("Audio is silent (peak amplitude below -100 dBFS)")
    peak_dbfs = 20 * np.log10(peak / full_scale)
    if sr < 8000:
        warnings.append("Sample rate below 8 kHz: high-frequency cues (qalqalah bursts) are unreliable")
    y = highpass(resample(mono, sr, target_sr), target_sr)
    y = y / (np.max(np.abs(y)) + 1e-9) * 0.891  # -1 dBFS headroom

    snr = estimate_snr_db(y, target_sr)
    denoised = False
    if denoise == "always" or (denoise == "auto" and snr < snr_threshold_db):
        y = spectral_subtract(y, target_sr)
        y = y / (np.max(np.abs(y)) + 1e-9) * 0.891
        denoised = True
        warnings.append(f"Background noise detected (SNR≈{snr:.0f} dB); applied spectral subtraction")
    if snr < 10:
        warnings.append("Low signal-to-noise ratio; results should be treated as indicative only")

    quality = AudioQuality(
        original_sr=sr, channels=channels, duration_s=duration, peak_dbfs=float(peak_dbfs),
        clipping_ratio=clip_ratio, snr_db=snr, denoised=denoised, warnings=warnings,
    )
    return AudioSignal(samples=y.astype(np.float32), sr=target_sr, quality=quality)


def load_audio(path: str | Path, *, target_sr: int = TARGET_SR, denoise: str = "auto") -> AudioSignal:
    p = Path(path)
    if not p.exists():
        raise AudioError(f"Audio file not found: {p}")
    if p.stat().st_size == 0:
        raise AudioError(f"Audio file is empty: {p}")
    data, sr = _read(p)
    return condition(data, sr, target_sr=target_sr, denoise=denoise)
