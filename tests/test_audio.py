"""Audio conditioning edge cases: clipping, stereo, sample rates, silence, NaNs."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from app.audio import AudioError, condition, load_audio
from tests.synth import SR, vowel


def test_resamples_and_downmixes_stereo(tmp_path) -> None:  # type: ignore[no-untyped-def]
    x = vowel(1.0)
    stereo = np.stack([x, 0.5 * x], axis=1)
    path = tmp_path / "stereo.wav"
    sf.write(path, stereo, 44_100)
    audio = load_audio(path, denoise="never")
    assert audio.sr == 16_000
    assert audio.quality.channels == 2 and audio.quality.original_sr == 44_100
    assert audio.duration_s == pytest.approx(len(x) / 44_100, abs=0.01)


def test_clipping_is_reported() -> None:
    x = np.clip(vowel(1.0) * 6, -1.0, 1.0)
    audio = condition(x, SR, denoise="never")
    assert audio.quality.clipping_ratio > 0.01
    assert any("Clipping" in w for w in audio.quality.warnings)
    assert np.max(np.abs(audio.samples)) < 1.0


def test_noise_triggers_denoising() -> None:
    rng = np.random.default_rng(0)
    x = np.concatenate([vowel(0.5), np.zeros(8000), vowel(0.5)]) + 0.06 * rng.standard_normal(24000)
    audio = condition(x, SR, denoise="auto")
    assert audio.quality.denoised


def test_rejects_silence_short_and_missing(tmp_path) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(AudioError):
        condition(np.zeros(SR), SR)
    with pytest.raises(AudioError):
        condition(vowel(0.1), SR)
    with pytest.raises(AudioError):
        load_audio(tmp_path / "missing.wav")
    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    with pytest.raises(AudioError):
        load_audio(empty)


def test_nan_samples_are_sanitised() -> None:
    x = vowel(1.0)
    x[100:110] = np.nan
    audio = condition(x, SR, denoise="never")
    assert np.all(np.isfinite(audio.samples))
    assert any("non-finite" in w for w in audio.quality.warnings)
