"""The causal-experiment harness: classification rules, physics on synthetic signals, transforms."""

from __future__ import annotations

import numpy as np
import pytest

from research_agency_lab.experiments.causal import schema as S

SR = 16000


def test_noise_floor_is_the_largest_noop_delta_but_never_below_the_unit_minimum() -> None:
    n = S.noise_floors([{"head:itbaq:margin": 0.3, "letter:duration_s": 0.0}, {"head:itbaq:margin": -1.7}])
    assert n["head:itbaq:margin"] == 1.7 and n["letter:duration_s"] == S.MIN_NOISE["s"]


def test_classification_separates_expected_unexpected_stable_insufficient() -> None:
    noise = {"head:hams_or_jahr:margin": 1.0, "identity:margin": 1.0, "head:itbaq:margin": 1.0}
    d = {"head:hams_or_jahr:margin": -3.0, "identity:margin": 0.4, "head:itbaq:margin": 2.5, "vowel:identity:margin": None}
    c = S.classify(d, noise, {"head:hams_or_jahr:margin": -1})
    assert c["head:hams_or_jahr:margin"] == S.EXPECTED
    assert c["identity:margin"] == S.STABLE
    assert c["head:itbaq:margin"] == S.UNEXPECTED                       # collateral
    assert c["vowel:identity:margin"] == S.INSUFFICIENT
    assert S.classify({"head:hams_or_jahr:margin": 3.0}, noise, {"head:hams_or_jahr:margin": -1})[
        "head:hams_or_jahr:margin"] == S.UNEXPECTED                     # moved, but the wrong way
    assert S.classify({}, noise, {"head:x:margin": -1})["head:x:margin"] == S.INSUFFICIENT


def test_physics_verification_needs_a_measured_change_not_a_requested_one() -> None:
    assert S.verify_physics("f0_hz", 1, 30.0, 28.0, 15.0)["status"] == S.VERIFIED
    assert S.verify_physics("f0_hz", 1, 30.0, 5.0, 15.0)["status"] == S.NOT_VERIFIED   # within noise
    assert S.verify_physics("f0_hz", 1, 30.0, -40.0, 15.0)["status"] == S.NOT_VERIFIED  # wrong way
    assert S.verify_physics("f0_hz", 1, 30.0, 90.0, 15.0)["status"] == S.PARTIAL       # 3x the prediction
    assert S.verify_physics("f0_hz", 1, 30.0, None, 15.0)["status"] == S.UNMEASURABLE


def _tone(sec: float = 0.6) -> np.ndarray:  # type: ignore[type-arg]
    t = np.arange(int(sec * SR)) / SR
    x = sum(np.sin(2 * np.pi * 150 * k * t) / k for k in range(1, 15)) * 0.1
    pad = np.zeros(int(0.4 * SR))
    return np.concatenate([pad, x, pad]).astype(np.float32)


def test_physics_tells_voiced_from_noise() -> None:
    pytest.importorskip("parselmouth")
    from research_agency_lab.experiments.causal.physics import measure
    tone = measure(_tone(), 0.4, 1.0)
    rng = np.random.default_rng(0)
    noise = measure(np.concatenate([np.zeros(6400), rng.standard_normal(9600) * 0.05, np.zeros(6400)]).astype(np.float32), 0.4, 1.0)
    assert tone["voiced_fraction"] > 0.9 and abs(tone["f0_hz"] - 150) < 3
    assert noise["voiced_fraction"] < 0.1 and noise["f0_hz"] is None


def test_transforms_do_what_the_physics_says() -> None:
    pytest.importorskip("pyworld")
    pytest.importorskip("parselmouth")
    from research_agency_lab.experiments.causal.physics import measure
    from research_agency_lab.experiments.causal.transforms import apply
    w = _tone()
    base = measure(w, 0.4, 1.0)
    same, e, _ = apply(w, 0.4, 1.0, "noop_splice", {})
    assert np.allclose(same, w, atol=1e-6) and e == 1.0
    longer, e2, _ = apply(w, 0.4, 1.0, "duration", {"k": 1.5})
    assert abs((e2 - 0.4) - 0.9) < 0.02
    up, _, _ = apply(w, 0.4, 1.0, "f0", {"semitones": 12})
    assert measure(up, 0.4, 1.0)["f0_hz"] > 1.8 * base["f0_hz"]
    unv, _, _ = apply(w, 0.4, 1.0, "voicing", {"level": 0.0})
    assert measure(unv, 0.4, 1.0)["voiced_fraction"] < 0.3 * base["voiced_fraction"]


def test_physical_durations_get_a_finer_floor_than_muqri_frames() -> None:
    assert S.noise_floors([{"span_duration_s": 0.0}], layer="physics")["span_duration_s"] == 0.005
    assert S.noise_floors([{"letter:duration_s": 0.0}])["letter:duration_s"] == S.MIN_NOISE["s"]
    assert S.verify_physics("span_duration_s", -1, -0.04, -0.04, 0.005)["status"] == S.VERIFIED
