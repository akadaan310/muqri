"""Taraweeh adapter: reverberation estimation/removal, pace normalization, breath and fatigue."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import fftconvolve

from app.acoustic.features import AcousticContext
from app.audio import frame_rms_db
from app.models import AlignedUnit, Alignment
from app.tajweed_rules import parse_text
from app.taraweeh_adapter.dereverb import (
    adapt_acoustics,
    environment_profile,
    estimate_rt60,
    gap_reverb_ratio_db,
    suppress_late_reverb,
    wpe_dereverb,
)
from app.taraweeh_adapter.fatigue_detector import detect_pauses, dynamic_stops, fatigue_report
from app.taraweeh_adapter.pace_normalizer import build_local_tempo, classify_pace, hadr_to_tahqeeq_ratio
from tests.synth import SR, concat, silence, vowel

rng = np.random.default_rng(7)


@pytest.fixture(scope="module")
def dry() -> np.ndarray:
    parts = []
    for k in range(30):
        parts.append(vowel(rng.uniform(0.12, 0.35), (rng.uniform(300, 800), rng.uniform(900, 2200), 2600),
                           f0=rng.uniform(100, 160), seed=k))
        parts.append(silence(rng.uniform(0.15, 0.6)))
    return concat(*parts)


def room(x: np.ndarray, rt60: float) -> np.ndarray:
    n = int(rt60 * 1.2 * SR)
    t = np.arange(n) / SR
    h = np.random.default_rng(1).standard_normal(n) * np.exp(-6.9 * t / rt60)
    h[:80] = 0
    h[0] = 4.0
    y = fftconvolve(x, h / np.abs(h).max())[: len(x)]
    return (y / np.abs(y).max() * 0.8).astype(np.float32)


def test_rt60_detects_reverberant_rooms(dry: np.ndarray) -> None:
    assert not np.isfinite(estimate_rt60(dry, SR)) or estimate_rt60(dry, SR) < 0.25
    short, long_ = estimate_rt60(room(dry, 0.3), SR), estimate_rt60(room(dry, 1.2), SR)
    assert 0.15 < short < 0.45
    assert long_ > 0.5 and long_ > short


def test_dereverberation_reduces_reverb_in_pauses(dry: np.ndarray) -> None:
    db = frame_rms_db(dry, SR, 20, 10)
    speech = db > db.max() - 30
    wet = room(dry, 1.0)
    before = gap_reverb_ratio_db(wet, speech, SR)
    after = gap_reverb_ratio_db(suppress_late_reverb(wpe_dereverb(wet, SR), SR, 1.0), speech, SR)
    assert after < before - 2.0  # at least 2 dB less reverberant energy in the pauses


def test_adapter_applies_eq_and_dereverb(dry: np.ndarray, make_signal) -> None:  # type: ignore[no-untyped-def]
    res = adapt_acoustics(make_signal(room(dry, 0.9)))
    assert any("WPE" in a for a in res.applied) and any("high-pass" in a for a in res.applied)
    assert res.audio.samples.shape == (len(dry),)
    dry_res = adapt_acoustics(make_signal(dry))
    assert not any("WPE" in a for a in dry_res.applied)  # dry audio is left alone


def test_environment_profile(dry: np.ndarray, make_signal) -> None:  # type: ignore[no-untyped-def]
    env = environment_profile(make_signal(room(dry, 1.0)))
    assert env.reverberant and env.as_array().shape == (8,)
    assert env.room_volume_estimate > environment_profile(make_signal(room(dry, 0.3))).room_volume_estimate


def test_pace_classification_and_ratio() -> None:
    assert classify_pace(120) == "hadr" and classify_pace(190) == "tadweer" and classify_pace(260) == "tahqeeq"
    assert hadr_to_tahqeeq_ratio(125) == pytest.approx(2.0)


def test_local_tempo_tracks_acceleration() -> None:
    parsed = parse_text(" ".join(["قَالَ كَذَٰلِكَ قَالَ رَبُّكَ"] * 4), include_sifaat=False)
    units = [u for u in parsed.units if u.pronounced]
    spans, t = {}, 0.0
    for k, u in enumerate(units):
        d = 0.25 if k < len(units) // 2 else 0.12  # Tahqeeq, then Hadr
        spans[u.index] = AlignedUnit(u.index, t, t + d)
        t += d
    local = build_local_tempo(parsed, Alignment(spans, "test"), 180.0)
    assert local(0.5) == pytest.approx(250, rel=0.15)
    assert local(t - 0.5) == pytest.approx(120, rel=0.15)
    assert local.variability > 0.1


def test_pause_breath_detection_and_dynamic_waqf(make_signal) -> None:  # type: ignore[no-untyped-def]
    breath = (0.03 * np.random.default_rng(2).standard_normal(int(0.45 * SR))).astype(np.float32)
    samples = concat(vowel(0.8), breath, vowel(0.8, seed=3), silence(0.4), vowel(0.8, seed=4))
    ctx = AcousticContext(make_signal(samples))
    pauses = detect_pauses(ctx)
    assert len(pauses) == 2
    assert pauses[0].breath and not pauses[1].breath

    parsed = parse_text("قُلْ هُوَ ٱللَّهُ أَحَدٌ", include_sifaat=False)
    pron = [u for u in parsed.units if u.pronounced]
    first_word_last = max(i for i in parsed.words[1].unit_indices if parsed.units[i].pronounced)
    spans = {u.index: AlignedUnit(u.index, 0.1 * k, 0.1 * k + 0.1) for k, u in enumerate(pron)}
    stop_at = spans[first_word_last].end_s
    for p in pauses:
        p.after_unit = first_word_last
        p.start_s, p.end_s = stop_at, stop_at + 0.4
    assert dynamic_stops(parsed, Alignment(spans, "test"), pauses[:1]) == {1}


def test_fatigue_report_counts_breaths(make_signal) -> None:  # type: ignore[no-untyped-def]
    breath = (0.03 * np.random.default_rng(3).standard_normal(int(0.4 * SR))).astype(np.float32)
    samples = concat(*[x for k in range(6) for x in (vowel(1.2, seed=k), breath)], vowel(1.0))
    ctx = AcousticContext(make_signal(samples))
    rep = fatigue_report(ctx, detect_pauses(ctx))
    assert rep.breaths >= 5 and rep.breaths_per_minute > 20 and rep.oxygen_depletion
