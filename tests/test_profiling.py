"""Reciter profile vectors and FAISS similarity search."""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.profiling import (
    PROFILE_DIM,
    STYLE_WEIGHT,
    TIMBRE_DIM,
    TIMBRE_WEIGHT,
    MfccStatsEmbedder,
    ProfileError,
    ReciterIndex,
    ReciterProfile,
    StyleVector,
    merge_profiles,
)
from tests.synth import concat, vowel

pytest.importorskip("faiss")

BACKEND = "test-backend"


def _unit(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(TIMBRE_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def _profiles() -> list[ReciterProfile]:
    styles = {
        "husary": StyleVector(170, 1.25, 2.7, 9.6),
        "minshawi": StyleVector(240, 1.05, 2.0, 18.8),
        "afasy": StyleVector(270, 1.00, 4.2, 20.1),
        "basit": StyleVector(265, 0.95, 1.9, 26.0),
    }
    return [ReciterProfile(rid, rid.title(), _unit(i), s, BACKEND) for i, (rid, s) in enumerate(styles.items())]


@pytest.fixture
def index() -> ReciterIndex:
    idx = ReciterIndex(BACKEND)
    for p in _profiles():
        idx.add(p)
    idx.build()
    return idx


def test_profile_vector_is_132_dimensional() -> None:
    assert _profiles()[0].vector().shape == (PROFILE_DIM,) == (132,)


def test_exact_match_ranks_first(index: ReciterIndex) -> None:
    target = _profiles()[2]
    matches = index.search(target.timbre, target.style, k=3)
    assert [m.reciter_id for m in matches][0] == "afasy"
    top = matches[0]
    assert top.timbre_similarity == pytest.approx(1.0, abs=1e-5)
    assert top.style_similarity == pytest.approx(1.0, abs=1e-6)
    assert top.combined_similarity == pytest.approx(STYLE_WEIGHT * 1.0 + TIMBRE_WEIGHT * 1.0)
    assert len(matches) == 3


def test_combined_score_formula(index: ReciterIndex) -> None:
    # Timbre of Husary with Minshawi's style: combined score must follow 0.6*style + 0.4*timbre.
    husary, minshawi = _profiles()[0], _profiles()[1]
    for m in index.search(husary.timbre, minshawi.style, k=4):
        expected = STYLE_WEIGHT * m.style_similarity + TIMBRE_WEIGHT * m.timbre_similarity
        assert m.combined_similarity == pytest.approx(expected)
    by_id = {m.reciter_id: m for m in index.search(husary.timbre, minshawi.style, k=4)}
    assert by_id["husary"].timbre_similarity > 0.99
    assert by_id["minshawi"].style_similarity > 0.99


def test_style_dominates_when_timbre_is_uninformative(index: ReciterIndex) -> None:
    neutral = _unit(99)
    matches = index.search(neutral, StyleVector(172, 1.24, 2.6, 10.0), k=1)
    assert matches[0].reciter_id == "husary"
    assert "Measured Murattal tempo" in matches[0].matched_traits


def test_missing_style_dimensions_are_ignored(index: ReciterIndex) -> None:
    husary = _profiles()[0]
    style = StyleVector(husary.style.tempo_harakat_per_min, math.nan, husary.style.pitch_range_semitones, math.nan)
    top = index.search(husary.timbre, style, k=1)[0]
    assert top.reciter_id == "husary" and top.style_similarity == pytest.approx(1.0)


def test_save_and_load_roundtrip(index: ReciterIndex, tmp_path) -> None:  # type: ignore[no-untyped-def]
    p = _profiles()[3]
    p.style = StyleVector(265, math.nan, 1.9, 26.0)  # NaN must survive persistence
    index.add(p)
    index.save(tmp_path)
    assert (tmp_path / "reciters_faiss.index").exists()
    loaded = ReciterIndex.load(tmp_path)
    assert len(loaded) == 4 and loaded.backend == BACKEND
    restored = {q.reciter_id: q for q in loaded.profiles}["basit"]
    assert math.isnan(restored.style.madd_stretch_bias)
    np.testing.assert_allclose(restored.timbre, p.timbre, atol=1e-6)
    before = [m.reciter_id for m in index.search(p.timbre, _profiles()[1].style, k=4)]
    after = [m.reciter_id for m in loaded.search(p.timbre, _profiles()[1].style, k=4)]
    assert before == after


def test_backend_mismatch_and_missing_index(tmp_path) -> None:  # type: ignore[no-untyped-def]
    idx = ReciterIndex(BACKEND)
    with pytest.raises(ProfileError):
        idx.add(ReciterProfile("x", "X", _unit(1), StyleVector(1, 1, 1, 1), "other-backend"))
    with pytest.raises(ProfileError):
        ReciterIndex.load(tmp_path)


def test_merge_profiles_averages_segments() -> None:
    parts = [(_unit(1), StyleVector(200, 1.0, 2.0, math.nan)), (_unit(1), StyleVector(220, 1.2, math.nan, 10.0))]
    prof = merge_profiles("r", "R", parts, BACKEND)
    assert prof.n_segments == 2
    assert prof.style.tempo_harakat_per_min == pytest.approx(210)
    assert prof.style.pitch_range_semitones == pytest.approx(2.0)
    assert prof.style.nasal_resonance_db == pytest.approx(10.0)


def test_mfcc_timbre_embedding_separates_voices(make_signal) -> None:  # type: ignore[no-untyped-def]
    emb = MfccStatsEmbedder()
    low_a = emb.embed(make_signal(concat(vowel(0.6, f0=110), vowel(0.6, (400, 2000, 2600), f0=115))))
    low_b = emb.embed(make_signal(concat(vowel(0.6, f0=112, seed=5), vowel(0.6, (400, 2000, 2600), f0=113, seed=6))))
    high = emb.embed(make_signal(concat(vowel(0.6, (900, 2100, 3100), f0=240), vowel(0.6, (500, 2500, 3300), f0=250))))
    assert low_a.shape == (TIMBRE_DIM,)
    assert float(low_a @ low_b) > float(low_a @ high)
