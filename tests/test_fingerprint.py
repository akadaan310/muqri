"""232-d fingerprint, FAISS indices, benchmark comparison."""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.fingerprint import (
    ENV_DIM,
    FINGERPRINT_DIM,
    TAJWEED_DIM,
    TAJWEED_FIELDS,
    TIMBRE_DIM,
    Fingerprint,
    MfccStatsEmbedder,
    ProfileError,
    ReciterIndex,
    ReciterProfile,
    benchmark_comparison,
    compute_tajweed_vector,
    merge_fingerprints,
)
from app.models import RuleDiagnostic, RuleType, Status, Tareeq
from tests.synth import concat, vowel

pytest.importorskip("faiss")
B = "test-backend"


def unit(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(TIMBRE_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def taj(**kw: float) -> np.ndarray:
    v = np.full(TAJWEED_DIM, np.nan)
    for k, x in kw.items():
        v[TAJWEED_FIELDS.index(k)] = x
    return v


def fp(seed: int, **kw: float) -> Fingerprint:
    return Fingerprint(unit(seed), taj(**kw), np.zeros(ENV_DIM), B)


@pytest.fixture
def index() -> ReciterIndex:
    idx = ReciterIndex(B, "studio")
    idx.add(ReciterProfile("husary", "Mahmoud Khalil Al-Hussary", fp(0, haraka_base_ms=300, madd_tabii_accuracy=0.95)))
    dosari = fp(1, haraka_base_ms=150, madd_tabii_accuracy=0.8)
    idx.add(ReciterProfile("dosari", "Yasser Al-Dosari", dosari, "taraweeh"))
    idx.add(ReciterProfile("afasy", "Mishary Alafasy", fp(2, haraka_base_ms=220, madd_tabii_accuracy=0.9)))
    idx.build()
    return idx


def test_dimensions_are_232() -> None:
    assert (TIMBRE_DIM, TAJWEED_DIM, ENV_DIM, FINGERPRINT_DIM) == (192, 32, 8, 232)
    assert fp(0).vector().shape == (232,)
    assert TAJWEED_FIELDS[0] == "haraka_base_ms" and TAJWEED_FIELDS[-1] == "overall_tajweed_perfection_index"


def test_tajweed_vector_from_diagnostics() -> None:
    diags = [RuleDiagnostic(RuleType.MADD_TABII, "w", 0, 1, Status.PASS, "", score=1.0),
             RuleDiagnostic(RuleType.MADD_TABII, "w", 0, 1, Status.WARNING, "", score=0.5),
             RuleDiagnostic(RuleType.QALQALAH, "w", 0, 1, Status.PASS, "", score=1.0,
                            metrics={"energy_rise_db": 14.0}, detail="kubra"),
             RuleDiagnostic(RuleType.SAKT, "w", 0, 1, Status.PASS, "", score=1.0,
                            metrics={"sakt_silence_duration_ms": 280.0})]
    v = compute_tajweed_vector(diags, 250.0, Tareeq.TAYYIBAH, 91.0)
    get = dict(zip(TAJWEED_FIELDS, v, strict=True))
    assert get["madd_tabii_accuracy"] == pytest.approx(0.75)
    assert get["qalqalah_kubra_burst_db"] == 14.0 and get["sakt_silence_duration_ms"] == 280.0
    assert get["is_tayyibah_qasr_flag"] == 1.0 and get["overall_tajweed_perfection_index"] == 91.0
    assert math.isnan(get["madd_lazim_accuracy"])  # rule absent from the text


def test_timbre_weight_identifies_speaker(index: ReciterIndex) -> None:
    q = fp(1, haraka_base_ms=300, madd_tabii_accuracy=0.95)  # Dosari's voice, Hussary's style
    assert index.search(q, 1, style_weight=0.0)[0].reciter_id == "dosari"
    assert index.search(q, 1, style_weight=1.0)[0].reciter_id == "husary"
    with pytest.raises(ProfileError):
        index.search(q, style_weight=2.0)


def test_save_load_roundtrip_keeps_nan(index: ReciterIndex, tmp_path) -> None:  # type: ignore[no-untyped-def]
    index.save(tmp_path)
    assert (tmp_path / "masterclass_reciters.faiss").exists()
    loaded = ReciterIndex.load(tmp_path, "masterclass_reciters")
    prof = loaded.find("hussary")
    assert prof is not None and math.isnan(prof.fingerprint.tajweed[TAJWEED_FIELDS.index("sakt_silence_duration_ms")])
    np.testing.assert_allclose(prof.fingerprint.timbre, unit(0), atol=1e-6)
    assert loaded.find("dosari") is not None and loaded.find("Yasser Ad-Dossary") is not None


def test_merge_and_benchmark_comparison(index: ReciterIndex) -> None:
    merged = merge_fingerprints([fp(3, haraka_base_ms=200), fp(3, haraka_base_ms=240, madd_tabii_accuracy=1.0)])
    assert merged.tajweed[0] == pytest.approx(220) and merged.tajweed[TAJWEED_FIELDS.index("madd_tabii_accuracy")] == 1
    ref = index.find("dosari")
    assert ref is not None
    cmp = benchmark_comparison(merged, ref)
    assert cmp["benchmark"] == "Yasser Al-Dosari"
    assert {r["feature"] for r in cmp["features"]} == {"haraka_base_ms", "madd_tabii_accuracy"}


def test_backend_mismatch_is_rejected() -> None:
    idx = ReciterIndex(B)
    with pytest.raises(ProfileError):
        idx.add(ReciterProfile("x", "X", Fingerprint(unit(0), taj(), np.zeros(ENV_DIM), "other")))


def test_mfcc_embedder_is_192d_and_discriminative(make_signal) -> None:  # type: ignore[no-untyped-def]
    emb = MfccStatsEmbedder()
    low_a = emb.embed(make_signal(concat(vowel(0.6, f0=110), vowel(0.6, (400, 2000, 2600), f0=115))))
    low_b = emb.embed(make_signal(concat(vowel(0.6, f0=112, seed=5), vowel(0.6, (400, 2000, 2600), f0=113, seed=6))))
    high = emb.embed(make_signal(concat(vowel(0.6, (900, 2100, 3100), f0=240), vowel(0.6, (500, 2500, 3300), f0=250))))
    assert low_a.shape == (192,) and float(low_a @ low_b) > float(low_a @ high)
