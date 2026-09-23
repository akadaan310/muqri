"""232-dimensional multi-vector recitation fingerprint and FAISS reciter indices.

The fingerprint keeps three things apart:

* ``timbre`` (192-d): voice biometrics, SpeechBrain ECAPA-TDNN (native 192-d). Without
  SpeechBrain a 192-d MFCC-statistics embedding is used; the back-end is stored with every index
  and must match at query time.
* ``tajweed`` (32-d): the masterclass Tajweed vector — tempo, sukoon spectrum, per-rule accuracy,
  formant, burst, HNR and spectral measurements (``TAJWEED_FIELDS``). Rules absent from the
  recited text are NaN and are ignored when comparing.
* ``environment`` (8-d): room and recording conditions (``app.taraweeh_adapter.dereverb``). It
  describes the context and is never used to decide who a reciter sounds like.

Similarity: ``S = w·S_tajweed + (1 − w)·S_timbre`` with the cosine for timbre and
``exp(−rms_z² / 2)`` over the Tajweed dimensions both vectors measured (z-normalised by the index
statistics). ``w`` defaults to 0.6 as in the v1 spec; held-out tests show timbre alone identifies
the reciter far more reliably (see README), so use a small ``w`` for identification.

Two indices are kept: ``masterclass_reciters`` (dry studio ijaazah recordings) and
``taraweeh_reciters`` (imams whose public recordings are mostly live / Taraweeh).
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypeAlias

import numpy as np
import numpy.typing as npt

from app.audio import AudioSignal
from app.models import RuleDiagnostic, RuleType, Status, Tareeq

logger = logging.getLogger(__name__)

FloatArray: TypeAlias = npt.NDArray[np.float32]

TIMBRE_DIM = 192
TAJWEED_FIELDS: tuple[str, ...] = (
    "haraka_base_ms", "hadr_to_tahqeeq_ratio", "shiddah_duration_ratio", "tawassut_duration_ratio",
    "rakhawah_duration_ratio", "madd_tabii_accuracy", "madd_muttasil_accuracy", "madd_munfasil_accuracy",
    "madd_lazim_accuracy", "ghunnah_envelope_ratio", "ikhfa_formant_anticipation_score",
    "tafkheem_f2_depression_mean", "tarqeeq_f2_elevation_mean", "qalqalah_sughra_burst_db",
    "qalqalah_kubra_burst_db", "qalqalah_akbar_hold_ratio", "hams_hnr_db_mean", "jahr_hnr_db_mean",
    "safir_high_band_power", "tafashhi_spectral_spread", "istitaalah_duration_ratio",
    "takreer_single_strike_flag", "lam_allah_tafkheem_accuracy", "raa_tafkheem_tarqeeq_accuracy",
    "izhaar_halqi_cleanliness", "iqlab_nasal_transition_score", "idghaam_kamil_assimilation_score",
    "idghaam_naqis_sifah_retention", "waqf_madd_aaridh_extension_ratio", "sakt_silence_duration_ms",
    "is_tayyibah_qasr_flag", "overall_tajweed_perfection_index",
)
ENV_FIELDS: tuple[str, ...] = (
    "rt60_reverberation_time", "snr_db", "room_volume_estimate", "mic_proximity_bass_boost", "spectral_tilt",
    "background_crowd_noise_level", "echo_density", "audio_clipping_ratio",
)
TAJWEED_DIM = len(TAJWEED_FIELDS)
ENV_DIM = len(ENV_FIELDS)
FINGERPRINT_DIM = TIMBRE_DIM + TAJWEED_DIM + ENV_DIM  # 232
assert FINGERPRINT_DIM == 232

DEFAULT_STYLE_WEIGHT = 0.6
MIN_PROFILES_FOR_STATS = 10
NAN_SENTINEL = np.float32(-1e30)
# Rough population priors (mean, std) used until an index has enough profiles for statistics.
_PRIORS: dict[str, tuple[float, float]] = {
    "haraka_base_ms": (250, 70), "hadr_to_tahqeeq_ratio": (1.0, 0.3), "shiddah_duration_ratio": (1.0, 0.4),
    "tawassut_duration_ratio": (1.2, 0.4), "rakhawah_duration_ratio": (1.5, 0.5),
    "ghunnah_envelope_ratio": (1.0, 0.3), "ikhfa_formant_anticipation_score": (0.6, 0.3),
    "tafkheem_f2_depression_mean": (0.25, 0.15), "tarqeeq_f2_elevation_mean": (0.0, 0.15),
    "qalqalah_sughra_burst_db": (12, 5), "qalqalah_kubra_burst_db": (14, 5), "qalqalah_akbar_hold_ratio": (1.5, 0.6),
    "hams_hnr_db_mean": (2, 4), "jahr_hnr_db_mean": (12, 5), "safir_high_band_power": (3, 5),
    "tafashhi_spectral_spread": (0.4, 0.15), "istitaalah_duration_ratio": (1.5, 0.6),
    "takreer_single_strike_flag": (0.9, 0.15), "waqf_madd_aaridh_extension_ratio": (2.0, 0.8),
    "sakt_silence_duration_ms": (300, 100), "is_tayyibah_qasr_flag": (0, 0.5),
    "overall_tajweed_perfection_index": (80, 15),
}
_ACCURACY_PRIOR = (0.8, 0.2)
INDEX_NAMES = {"studio": "masterclass_reciters", "taraweeh": "taraweeh_reciters"}


class ProfileError(RuntimeError):
    pass


# --------------------------------------------------------------------------- tajweed vector
def _scores(diags: list[RuleDiagnostic], pred) -> list[float]:  # type: ignore[no-untyped-def]
    return [d.score for d in diags if pred(d) and d.score is not None and d.status not in
            (Status.SKIPPED, Status.VALID_NECESSARY_PAUSE)]


def _metric(diags: list[RuleDiagnostic], rule_types: set[RuleType], key: str, pred=None) -> list[float]:  # type: ignore[no-untyped-def]
    return [d.metrics[key] for d in diags if d.rule_type in rule_types and key in d.metrics
            and (pred is None or pred(d)) and np.isfinite(d.metrics[key])]


def _mean(vals: list[float]) -> float:
    return float(np.mean(vals)) if vals else float("nan")


def compute_tajweed_vector(diags: list[RuleDiagnostic], haraka_ms: float, tareeq: Tareeq,
                           perfection: float | None,
                           lam_allah_words: set[str] | None = None) -> npt.NDArray[np.float64]:
    """The 32-d masterclass Tajweed vector, in ``TAJWEED_FIELDS`` order."""
    from app.taraweeh_adapter.pace_normalizer import hadr_to_tahqeeq_ratio

    lam_words = lam_allah_words or set()

    def acc(rt: RuleType) -> float:
        return _mean(_scores(diags, lambda d: d.rule_type is rt))

    madd_measured = [d for d in diags if d.rule_type is RuleType.MADD_ARID and d.measured_harakat]
    nasal = {RuleType.GHUNNAH, RuleType.IKHFA, RuleType.IDGHAM_GHUNNAH, RuleType.IQLAB, RuleType.IKHFA_SHAFAWI,
             RuleType.IDGHAM_SHAFAWI}
    envelope = [d.measured_harakat / d.expected_range[0] for d in diags if d.rule_type in nasal and d.measured_harakat
                and d.expected_range and d.expected_range[0] > 0]
    vec = {
        "haraka_base_ms": haraka_ms,
        "hadr_to_tahqeeq_ratio": hadr_to_tahqeeq_ratio(haraka_ms),
        "shiddah_duration_ratio": _mean(_metric(diags, {RuleType.SHIDDAH}, "duration_ratio")),
        "tawassut_duration_ratio": _mean(_metric(diags, {RuleType.TAWASSUT}, "duration_ratio")),
        "rakhawah_duration_ratio": _mean(_metric(diags, {RuleType.RAKHAWAH}, "duration_ratio")),
        "madd_tabii_accuracy": acc(RuleType.MADD_TABII),
        "madd_muttasil_accuracy": acc(RuleType.MADD_MUTTASIL),
        "madd_munfasil_accuracy": acc(RuleType.MADD_MUNFASIL),
        "madd_lazim_accuracy": acc(RuleType.MADD_LAZIM),
        "ghunnah_envelope_ratio": _mean(envelope),
        "ikhfa_formant_anticipation_score": _mean(_metric(diags, {RuleType.IKHFA}, "ikhfa_formant_anticipation_score")),
        "tafkheem_f2_depression_mean": _mean(_metric(diags, {RuleType.TAFKHEEM}, "heaviness_index")),
        "tarqeeq_f2_elevation_mean": _mean([-v for v in _metric(diags, {RuleType.TARQEEQ}, "heaviness_index")]),
        "qalqalah_sughra_burst_db": _mean(_metric(diags, {RuleType.QALQALAH}, "energy_rise_db",
                                                  lambda d: d.detail == "sughra")),
        "qalqalah_kubra_burst_db": _mean(_metric(diags, {RuleType.QALQALAH}, "energy_rise_db",
                                                 lambda d: d.detail == "kubra")),
        "qalqalah_akbar_hold_ratio": _mean(_metric(diags, {RuleType.QALQALAH}, "shiddah_hold_harakat")),
        "hams_hnr_db_mean": _mean(_metric(diags, {RuleType.HAMS}, "hnr_db")),
        "jahr_hnr_db_mean": _mean(_metric(diags, {RuleType.JAHR}, "hnr_db")),
        "safir_high_band_power": _mean(_metric(diags, {RuleType.SAFIR}, "safir_high_band_spike_db")),
        "tafashhi_spectral_spread": _mean(_metric(diags, {RuleType.TAFASHHI}, "tafashhi_spectral_flatness")),
        "istitaalah_duration_ratio": _mean(_metric(diags, {RuleType.ISTITAALAH}, "istitaalah_duration_ratio")),
        "takreer_single_strike_flag": _mean(_metric(diags, {RuleType.TAKREER}, "takreer_single_strike_flag")),
        "lam_allah_tafkheem_accuracy": _mean(_scores(diags, lambda d: d.letter == "ل" and d.rule_type in
                                                     (RuleType.TAFKHEEM, RuleType.TARQEEQ)
                                                     and (not lam_words or d.word in lam_words))),
        "raa_tafkheem_tarqeeq_accuracy": _mean(_scores(diags, lambda d: d.letter == "ر" and d.rule_type in
                                                       (RuleType.TAFKHEEM, RuleType.TARQEEQ, RuleType.JAWAZ_WAJHAYN))),
        "izhaar_halqi_cleanliness": acc(RuleType.IZHAR_HALQI),
        "iqlab_nasal_transition_score": acc(RuleType.IQLAB),
        "idghaam_kamil_assimilation_score": _mean(_scores(diags, lambda d: d.rule_type in (
            RuleType.IDGHAM_MITHLAYN, RuleType.IDGHAM_MUTAJANISAYN, RuleType.IDGHAM_MUTAQARIBAYN)
            and d.detail != "naqis")),
        "idghaam_naqis_sifah_retention": _mean(_metric(diags, {RuleType.IDGHAM_MUTAJANISAYN}, "itbaq_retention_index")),
        "waqf_madd_aaridh_extension_ratio": _mean([d.measured_harakat / 2.0 for d in madd_measured
                                                   if d.measured_harakat is not None]),
        "sakt_silence_duration_ms": _mean(_metric(diags, {RuleType.SAKT}, "sakt_silence_duration_ms")),
        "is_tayyibah_qasr_flag": 1.0 if tareeq is Tareeq.TAYYIBAH else 0.0,
        "overall_tajweed_perfection_index": perfection if perfection is not None else float("nan"),
    }
    return np.array([vec[f] for f in TAJWEED_FIELDS], dtype=np.float64)


# --------------------------------------------------------------------------- timbre embedders
class TimbreEmbedder(Protocol):
    backend: str

    def embed(self, audio: AudioSignal) -> FloatArray: ...


def _l2(v: npt.ArrayLike) -> FloatArray:
    a = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(a))
    return a / n if n > 0 else a


class MfccStatsEmbedder:
    """192-d timbre descriptor: mean/std of 48 MFCCs and their deltas over voiced frames."""

    backend = "mfcc-stats-192-v2"

    def embed(self, audio: AudioSignal) -> FloatArray:
        import librosa

        y = audio.samples
        mfcc = librosa.feature.mfcc(y=y, sr=audio.sr, n_mfcc=49, n_fft=512, hop_length=160, n_mels=80)[1:]
        rms = librosa.feature.rms(y=y, frame_length=512, hop_length=160)[0]
        n = min(mfcc.shape[1], rms.size)
        mfcc, rms = mfcc[:, :n], rms[:n]
        voiced = rms > np.percentile(rms, 40)
        if voiced.sum() < 10:
            voiced = np.ones(n, dtype=bool)
        feats = mfcc[:, voiced]
        width = min(9, feats.shape[1] - (1 - feats.shape[1] % 2))
        delta = librosa.feature.delta(feats, width=width) if width >= 3 else np.zeros_like(feats)
        scale = 1.0 / (1.0 + np.arange(feats.shape[0]))
        vec = np.concatenate([
            feats.mean(axis=1) * scale, feats.std(axis=1) * scale,
            delta.mean(axis=1) * scale * 10, delta.std(axis=1) * scale * 4,
        ])
        return _l2(vec)


class EcapaEmbedder:
    """SpeechBrain ECAPA-TDNN speaker embedding (192-d)."""

    backend = "ecapa-voxceleb-192-v2"

    def __init__(self, source: str = "speechbrain/spkrec-ecapa-voxceleb", savedir: str | None = None) -> None:
        try:
            import torch
            try:
                from speechbrain.inference.speaker import EncoderClassifier
            except ImportError:  # speechbrain < 1.0
                from speechbrain.pretrained import EncoderClassifier  # type: ignore[no-redef]
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ProfileError("ECAPA embeddings require `speechbrain` and `torch`") from exc
        self._torch = torch
        try:
            self.model = EncoderClassifier.from_hparams(
                source=source, savedir=savedir or str(Path.home() / ".cache" / "qaari-eval" / "ecapa"),
                run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            )
        except Exception as exc:  # noqa: BLE001
            raise ProfileError(f"Could not load ECAPA model {source!r}: {exc}") from exc

    def embed(self, audio: AudioSignal) -> FloatArray:
        torch = self._torch
        wav = torch.from_numpy(audio.samples).float().unsqueeze(0)
        with torch.inference_mode():
            emb = self.model.encode_batch(wav).squeeze().cpu().numpy().astype(np.float32)
        if emb.shape[0] != TIMBRE_DIM:
            raise ProfileError(f"ECAPA returned {emb.shape[0]}-d embeddings, expected {TIMBRE_DIM}")
        return _l2(emb)


def get_timbre_embedder(backend: str = "auto") -> TimbreEmbedder:
    if backend in ("mfcc", MfccStatsEmbedder.backend):
        return MfccStatsEmbedder()
    if backend in ("ecapa", EcapaEmbedder.backend):
        return EcapaEmbedder()
    if backend == "auto":
        try:
            return EcapaEmbedder()
        except ProfileError:
            logger.info("SpeechBrain unavailable; using MFCC-statistics timbre embedding")
            return MfccStatsEmbedder()
    raise ProfileError(f"Unknown timbre backend {backend!r}")


# --------------------------------------------------------------------------- fingerprint & profiles
@dataclass(slots=True)
class Fingerprint:
    timbre: FloatArray
    tajweed: npt.NDArray[np.float64]
    environment: npt.NDArray[np.float64]
    backend: str

    def vector(self) -> FloatArray:
        v = np.concatenate([_l2(self.timbre), self.tajweed, self.environment]).astype(np.float32)
        return np.where(np.isfinite(v), v, NAN_SENTINEL).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        def clean(names: tuple[str, ...], arr: npt.NDArray[np.float64]) -> dict[str, float | None]:
            return {n: (round(float(x), 4) if np.isfinite(x) else None) for n, x in zip(names, arr, strict=True)}

        return {
            "timbre_vector_192d": {"backend": self.backend, "norm": round(float(np.linalg.norm(self.timbre)), 3)},
            "masterclass_tajweed_vector_32d": clean(TAJWEED_FIELDS, self.tajweed),
            "acoustic_environment_vector_8d": clean(ENV_FIELDS, self.environment),
            "dimensions": FINGERPRINT_DIM,
        }


@dataclass(slots=True)
class ReciterProfile:
    reciter_id: str
    name: str
    fingerprint: Fingerprint
    category: str = "studio"
    n_segments: int = 1
    source: str = ""


def merge_fingerprints(parts: list[Fingerprint]) -> Fingerprint:
    """Average per-ayah fingerprints (NaN-aware) into one reciter profile."""
    if not parts:
        raise ProfileError("No fingerprints to merge")
    backend = parts[0].backend
    if any(p.backend != backend for p in parts):
        raise ProfileError("Cannot merge fingerprints from different timbre back-ends")
    with np.errstate(all="ignore"):
        taj = np.array([p.tajweed for p in parts])
        env = np.array([p.environment for p in parts])
        taj_m = np.array([np.nanmean(c) if np.isfinite(c).any() else np.nan for c in taj.T])
        env_m = np.array([np.nanmean(c) if np.isfinite(c).any() else np.nan for c in env.T])
    return Fingerprint(_l2(np.mean([_l2(p.timbre) for p in parts], axis=0)), taj_m, env_m, backend)


@dataclass(slots=True)
class Match:
    reciter_id: str
    reciter_name: str
    category: str
    combined_similarity: float
    style_similarity: float
    timbre_similarity: float
    shared_dimensions: int
    matched_traits: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "reciter_name": self.reciter_name, "reciter_id": self.reciter_id, "index": self.category,
            "combined_similarity_pct": round(self.combined_similarity * 100, 1),
            "style_similarity_pct": round(self.style_similarity * 100, 1),
            "timbre_similarity_pct": round(self.timbre_similarity * 100, 1),
            "shared_tajweed_dimensions": self.shared_dimensions, "matched_traits": self.matched_traits,
        }


_TRAIT_FIELDS = {
    "hadr_to_tahqeeq_ratio": ("Measured Tahqeeq tempo", "Balanced Tadweer tempo", "Brisk Hadr tempo"),
    "waqf_madd_aaridh_extension_ratio": ("Short waqf madds", "Moderate waqf madds", "Long, dramatic waqf madds"),
    "ghunnah_envelope_ratio": ("Light Ghunnah", "Balanced Ghunnah", "Strong Ghunnah holds"),
    "qalqalah_kubra_burst_db": ("Soft Qalqalah", "Clear Qalqalah", "Explosive Qalqalah"),
    "tafkheem_f2_depression_mean": ("Light Tafkheem", "Balanced Tafkheem", "Deep Tafkheem"),
    "overall_tajweed_perfection_index": ("Relaxed precision", "Solid precision", "Ijaazah-level precision"),
}


class ReciterIndex:
    """FAISS-backed store of 232-d reciter fingerprints."""

    def __init__(self, backend: str, category: str = "studio") -> None:
        self.backend = backend
        self.category = category
        self.profiles: list[ReciterProfile] = []
        self._mean = np.array([_PRIORS.get(f, _ACCURACY_PRIOR)[0] for f in TAJWEED_FIELDS], dtype=np.float64)
        self._std = np.array([_PRIORS.get(f, _ACCURACY_PRIOR)[1] for f in TAJWEED_FIELDS], dtype=np.float64)
        self._timbre_index: Any = None

    def __len__(self) -> int:
        return len(self.profiles)

    def add(self, profile: ReciterProfile) -> None:
        if profile.fingerprint.backend != self.backend:
            raise ProfileError(f"Profile backend {profile.fingerprint.backend!r} != index backend {self.backend!r}")
        if profile.fingerprint.timbre.shape != (TIMBRE_DIM,):
            raise ProfileError(f"Timbre vector must be {TIMBRE_DIM}-d, got {profile.fingerprint.timbre.shape}")
        self.profiles = [p for p in self.profiles if p.reciter_id != profile.reciter_id]
        self.profiles.append(profile)
        self._timbre_index = None

    def build(self) -> None:
        import faiss

        if not self.profiles:
            raise ProfileError("Index is empty")
        taj = np.array([p.fingerprint.tajweed for p in self.profiles])
        for d in range(TAJWEED_DIM):
            col = taj[:, d][np.isfinite(taj[:, d])]
            if col.size >= MIN_PROFILES_FOR_STATS:
                prior_std = _PRIORS.get(TAJWEED_FIELDS[d], _ACCURACY_PRIOR)[1]
                self._mean[d] = float(col.mean())
                self._std[d] = max(float(col.std()), 0.5 * prior_std)
        self._timbre_index = faiss.IndexFlatIP(TIMBRE_DIM)
        self._timbre_index.add(np.stack([_l2(p.fingerprint.timbre) for p in self.profiles]).astype(np.float32))

    def z(self, tajweed: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return (tajweed - self._mean) / self._std

    def search(self, fp: Fingerprint, k: int = 3, style_weight: float = DEFAULT_STYLE_WEIGHT) -> list[Match]:
        if not 0.0 <= style_weight <= 1.0:
            raise ProfileError(f"style_weight must be in [0, 1], got {style_weight}")
        if fp.backend != self.backend:
            raise ProfileError(f"Index built with {self.backend!r} but query used {fp.backend!r}")
        if self._timbre_index is None:
            self.build()
        q_t = _l2(fp.timbre).reshape(1, -1).astype(np.float32)
        sims, ids = self._timbre_index.search(q_t, len(self.profiles))
        timbre_sim = {int(i): float(np.clip(s, 0.0, 1.0)) for s, i in zip(sims[0], ids[0], strict=True) if i >= 0}
        q_z = self.z(fp.tajweed)
        out: list[Match] = []
        for i, prof in enumerate(self.profiles):
            r_z = self.z(prof.fingerprint.tajweed)
            shared = np.isfinite(q_z) & np.isfinite(r_z)
            n = int(shared.sum())
            s_style = math.exp(-float(np.mean((q_z[shared] - r_z[shared]) ** 2)) / 2) if n else 0.0
            t = timbre_sim.get(i, 0.0)
            traits = describe_traits(q_z, r_z, t)
            out.append(Match(prof.reciter_id, prof.name, self.category, style_weight * s_style + (1 - style_weight) * t,
                             s_style, t, n, traits))
        out.sort(key=lambda m: m.combined_similarity, reverse=True)
        return out[: max(1, k)]

    def find(self, name: str) -> ReciterProfile | None:
        key = _name_key(name)
        for p in self.profiles:
            if key in _name_key(p.name) or key in _name_key(p.reciter_id):
                return p
        return None

    # -- persistence ----------------------------------------------------------------------
    def save(self, directory: str | Path, stem: str | None = None) -> Path:
        import faiss

        if self._timbre_index is None:
            self.build()
        stem = stem or INDEX_NAMES.get(self.category, self.category)
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        flat = faiss.IndexFlatIP(FINGERPRINT_DIM)
        flat.add(np.stack([p.fingerprint.vector() for p in self.profiles]).astype(np.float32))
        faiss.write_index(flat, str(out / f"{stem}.faiss"))
        meta = {
            "format": "qaari-eval-fingerprint-index/2", "backend": self.backend, "category": self.category,
            "dimensions": {"timbre": TIMBRE_DIM, "tajweed": TAJWEED_DIM, "environment": ENV_DIM},
            "tajweed_fields": list(TAJWEED_FIELDS), "environment_fields": list(ENV_FIELDS),
            "reciters": [{"reciter_id": p.reciter_id, "name": p.name, "category": p.category,
                          "n_segments": p.n_segments, "source": p.source} for p in self.profiles],
        }
        (out / f"{stem}_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return out / f"{stem}.faiss"

    @classmethod
    def load(cls, directory: str | Path, stem: str) -> ReciterIndex:
        import faiss

        d = Path(directory)
        idx_path, meta_path = d / f"{stem}.faiss", d / f"{stem}_meta.json"
        if not idx_path.exists() or not meta_path.exists():
            raise ProfileError(f"No reciter index {stem!r} in {d} (run datasets/index_reciters.py)")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        flat = faiss.read_index(str(idx_path))
        if flat.d != FINGERPRINT_DIM or flat.ntotal != len(meta["reciters"]):
            raise ProfileError(f"Index {stem!r} is inconsistent with its metadata; rebuild it")
        vectors = flat.reconstruct_n(0, flat.ntotal).astype(np.float64)
        vectors = np.where(vectors <= -1e29, np.nan, vectors)
        index = cls(meta["backend"], meta.get("category", "studio"))
        for vec, info in zip(vectors, meta["reciters"], strict=True):
            fp = Fingerprint(vec[:TIMBRE_DIM].astype(np.float32), vec[TIMBRE_DIM:TIMBRE_DIM + TAJWEED_DIM],
                             vec[TIMBRE_DIM + TAJWEED_DIM:], meta["backend"])
            index.profiles.append(ReciterProfile(info["reciter_id"], info["name"], fp, info.get("category", "studio"),
                                                 int(info.get("n_segments", 1)), info.get("source", "")))
        index.build()
        return index


def load_indices(directory: str | Path) -> dict[str, ReciterIndex]:
    out: dict[str, ReciterIndex] = {}
    for category, stem in INDEX_NAMES.items():
        try:
            out[category] = ReciterIndex.load(directory, stem)
        except ProfileError as exc:
            logger.info("%s", exc)
    return out


_ARTICLE = re.compile(r"\b(?:a[lntdsrzh]|as[h]?|ash)[-\s]+")


def _name_key(s: str) -> str:
    """Spelling-insensitive key: drop the article (Al-/Ad-/As-/Ash-…), collapse vowel/consonant doubling."""
    s = _ARTICLE.sub("", s.lower()).replace("ss", "s").replace("aa", "a").replace("oo", "u").replace("ee", "i")
    for a, b in (("dosary", "dusari"), ("dosari", "dusari"), ("dusary", "dusari"), ("hussary", "husary"),
                 ("shuraim", "shuraym"), ("minshawi", "minshawy"), ("juhani", "juhaynee"), ("juhany", "juhaynee")):
        s = s.replace(a, b)
    return "".join(c for c in s if c.isalnum())


def _level(z: float) -> int:
    return 0 if z < -0.5 else 2 if z > 0.5 else 1


def describe_traits(q_z: npt.NDArray[np.float64], r_z: npt.NDArray[np.float64], timbre_sim: float) -> list[str]:
    traits: list[str] = []
    for f, labels in _TRAIT_FIELDS.items():
        d = TAJWEED_FIELDS.index(f)
        if np.isfinite(q_z[d]) and np.isfinite(r_z[d]) and abs(q_z[d] - r_z[d]) <= 0.5:
            traits.append(labels[_level(float(r_z[d]))])
    if timbre_sim >= 0.6:
        traits.append("Similar vocal timbre")
    return traits


def benchmark_comparison(fp: Fingerprint, ref: ReciterProfile) -> dict[str, Any]:
    """Field-by-field comparison of a recitation with a benchmark reciter's profile."""
    rows = []
    for name, a, b in zip(TAJWEED_FIELDS, fp.tajweed, ref.fingerprint.tajweed, strict=True):
        if np.isfinite(a) and np.isfinite(b):
            rows.append({"feature": name, "you": round(float(a), 3), "benchmark": round(float(b), 3),
                         "difference": round(float(a - b), 3)})
    return {
        "benchmark": ref.name, "index": ref.category,
        "timbre_similarity_pct": round(float(np.clip(np.dot(_l2(fp.timbre), _l2(ref.fingerprint.timbre)), 0, 1)) * 100,
                                       1) if fp.backend == ref.fingerprint.backend else None,
        "features": rows,
        "benchmark_environment": {n: (round(float(v), 3) if np.isfinite(v) else None)
                                  for n, v in zip(ENV_FIELDS, ref.fingerprint.environment, strict=True)},
    }
