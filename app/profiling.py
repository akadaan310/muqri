"""Reciter fingerprinting and FAISS similarity search.

A recitation is summarised by a 132-dimensional **Reciter Profile Vector**:

* ``timbre`` (128-d): speaker embedding of the voice. With SpeechBrain installed this is the
  ECAPA-TDNN embedding (192-d, projected to 128-d by a fixed orthonormal matrix, which keeps
  cosine similarities approximately intact). Without it a 128-d MFCC-statistics embedding is
  used. The back-end name is stored in the index and must match at query time.
* ``style`` (4-d): Tajweed style — tempo (harakat/min), Madd stretch bias, pitch dynamic range
  (std of F0 in semitones) and nasal resonance (mean NER over Ghunnah windows). The Madd stretch
  bias is the median ratio of measured to canonical minimum length over non-phrase-final Madds
  (for Madd Tabi'i this is exactly duration / 2·T_haraka), so it does not depend on which ayah
  was recited.

Similarity: ``S = 0.6·S_style + 0.4·S_timbre`` where ``S_timbre`` is the (clipped) cosine
similarity and ``S_style = exp(−d² / 2σ²)`` (σ = 2) for the Euclidean distance ``d`` between
z-normalised style vectors.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypeAlias

import numpy as np
import numpy.typing as npt

from app.acoustic.features import AcousticContext
from app.audio import AudioSignal
from app.models import MADD_RULES, NASAL_RULES, RuleDiagnostic

logger = logging.getLogger(__name__)

TIMBRE_DIM = 128
STYLE_DIM = 4
PROFILE_DIM = TIMBRE_DIM + STYLE_DIM
STYLE_WEIGHT = 0.6
TIMBRE_WEIGHT = 0.4
STYLE_FIELDS = ("tempo_harakat_per_min", "madd_stretch_bias", "pitch_range_semitones", "nasal_resonance_db")
# Population priors used to z-normalise style when the index is too small to supply statistics
# (and as a floor on the index spread, so a handful of similar reciters cannot inflate z-scores).
STYLE_PRIOR_MEAN = np.array([250.0, 1.1, 2.5, 14.0])
STYLE_PRIOR_STD = np.array([60.0, 0.25, 1.0, 7.0])
MIN_PROFILES_FOR_STATS = 10
STYLE_KERNEL_SIGMA = 2.0
INDEX_FILE = "reciters_faiss.index"
META_FILE = "reciters_meta.json"

FloatArray: TypeAlias = npt.NDArray[np.float32]


class ProfileError(RuntimeError):
    pass


# --------------------------------------------------------------------------- style vector
@dataclass(slots=True)
class StyleVector:
    tempo_harakat_per_min: float
    madd_stretch_bias: float
    pitch_range_semitones: float
    nasal_resonance_db: float

    def as_array(self) -> npt.NDArray[np.float64]:
        return np.array([getattr(self, f) for f in STYLE_FIELDS], dtype=np.float64)

    def to_dict(self) -> dict[str, float | None]:
        values = zip(STYLE_FIELDS, self.as_array(), strict=True)
        return {f: (round(v, 3) if math.isfinite(v) else None) for f, v in values}

    @classmethod
    def from_array(cls, arr: npt.ArrayLike) -> StyleVector:
        a = np.asarray(arr, dtype=np.float64)
        return cls(*(float(v) for v in a))


def compute_style_vector(diagnostics: list[RuleDiagnostic], haraka_ms: float, ctx: AcousticContext,
                         speech_span: tuple[float, float] | None = None,
                         final_words: frozenset[str] = frozenset()) -> StyleVector:
    """Derive the 4-d Tajweed style vector from rule diagnostics and the pitch track.

    ``final_words`` are excluded from the Madd bias because phrase-final lengthening before a stop
    reflects the pause, not the reciter's Madd habit.
    """
    tempo = 60_000.0 / haraka_ms if haraka_ms > 0 else float("nan")

    ratios = [
        d.measured_harakat / d.expected_range[0]
        for d in diagnostics
        if d.rule_type in MADD_RULES and d.measured_harakat is not None and d.expected_range
        and d.expected_range[0] > 0 and d.word not in final_words
    ]
    madd_bias = float(np.median(ratios)) if ratios else float("nan")

    start, end = speech_span if speech_span is not None else (0.0, ctx.audio.duration_s)
    f0 = ctx.f0_in(start, end)
    if f0.size >= 10:
        semis = 12.0 * np.log2(f0 / np.median(f0))
        pitch_range = float(np.std(semis))
    else:
        pitch_range = float("nan")

    ners = [
        d.metrics["nasal_energy_ratio_db"]
        for d in diagnostics
        if d.rule_type in NASAL_RULES and "nasal_energy_ratio_db" in d.metrics
    ]
    nasal = float(np.mean(ners)) if ners else float("nan")
    return StyleVector(tempo, madd_bias, pitch_range, nasal)


# --------------------------------------------------------------------------- timbre embedders
class TimbreEmbedder(Protocol):
    backend: str

    def embed(self, audio: AudioSignal) -> FloatArray: ...


def _l2(v: npt.ArrayLike) -> FloatArray:
    a = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(a))
    return a / n if n > 0 else a


class MfccStatsEmbedder:
    """128-d timbre descriptor: mean/std of 32 MFCCs and their deltas over voiced frames."""

    backend = "mfcc-stats-v1"

    def embed(self, audio: AudioSignal) -> FloatArray:
        import librosa

        y = audio.samples
        mfcc = librosa.feature.mfcc(y=y, sr=audio.sr, n_mfcc=33, n_fft=512, hop_length=160, n_mels=64)[1:]
        rms = librosa.feature.rms(y=y, frame_length=512, hop_length=160)[0]
        n = min(mfcc.shape[1], rms.size)
        mfcc, rms = mfcc[:, :n], rms[:n]
        voiced = rms > np.percentile(rms, 40)
        if voiced.sum() < 10:
            voiced = np.ones(n, dtype=bool)
        feats = mfcc[:, voiced]
        width = min(9, feats.shape[1] - (1 - feats.shape[1] % 2))
        delta = librosa.feature.delta(feats, width=width) if width >= 3 else np.zeros_like(feats)
        # Cepstral mean normalisation per utterance would remove timbre; instead standardise
        # each statistic by fixed scales so that no single coefficient dominates the cosine.
        scale = 1.0 / (1.0 + np.arange(feats.shape[0]))
        vec = np.concatenate([
            feats.mean(axis=1) * scale, feats.std(axis=1) * scale,
            delta.mean(axis=1) * scale * 10, delta.std(axis=1) * scale * 4,
        ])
        return _l2(vec)


class EcapaEmbedder:
    """SpeechBrain ECAPA-TDNN speaker embedding projected to 128-d."""

    backend = "ecapa-voxceleb-proj128-v1"

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
        self._proj: FloatArray | None = None

    def _projection(self, dim_in: int) -> FloatArray:
        if self._proj is None or self._proj.shape[0] != dim_in:
            rng = np.random.default_rng(20240613)
            q, _ = np.linalg.qr(rng.standard_normal((dim_in, TIMBRE_DIM)))
            self._proj = q.astype(np.float32)
        return self._proj

    def embed(self, audio: AudioSignal) -> FloatArray:
        torch = self._torch
        wav = torch.from_numpy(audio.samples).float().unsqueeze(0)
        with torch.inference_mode():
            emb = self.model.encode_batch(wav).squeeze().cpu().numpy().astype(np.float32)
        return _l2(emb @ self._projection(emb.shape[0]))


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


# --------------------------------------------------------------------------- profiles & index
@dataclass(slots=True)
class ReciterProfile:
    reciter_id: str
    name: str
    timbre: FloatArray
    style: StyleVector
    backend: str
    n_segments: int = 1
    source: str = ""

    def vector(self) -> FloatArray:
        """The persisted 132-d profile vector (timbre ⊕ raw style)."""
        return np.concatenate([_l2(self.timbre), self.style.as_array().astype(np.float32)]).astype(np.float32)


def merge_profiles(reciter_id: str, name: str, parts: list[tuple[FloatArray, StyleVector]], backend: str,
                   source: str = "") -> ReciterProfile:
    """Average several per-ayah measurements into one reciter profile (NaN-aware for style)."""
    if not parts:
        raise ProfileError(f"No segments to build a profile for {name}")
    timbre = _l2(np.mean([_l2(t) for t, _ in parts], axis=0))
    styles = np.array([s.as_array() for _, s in parts])
    with np.errstate(all="ignore"):
        style = np.nanmean(styles, axis=0) if np.isfinite(styles).any() else np.full(STYLE_DIM, np.nan)
    return ReciterProfile(reciter_id, name, timbre, StyleVector.from_array(style), backend, len(parts), source)


@dataclass(slots=True)
class Match:
    reciter_id: str
    reciter_name: str
    combined_similarity: float
    style_similarity: float
    timbre_similarity: float
    matched_traits: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "reciter_name": self.reciter_name,
            "reciter_id": self.reciter_id,
            "combined_similarity_pct": round(self.combined_similarity * 100, 1),
            "style_similarity_pct": round(self.style_similarity * 100, 1),
            "timbre_similarity_pct": round(self.timbre_similarity * 100, 1),
            "matched_traits": self.matched_traits,
        }


def _level(z: float) -> str:
    return "low" if z < -0.5 else "high" if z > 0.5 else "mid"


_TRAITS = {
    0: {"low": "Measured Murattal tempo", "mid": "Balanced Tadweer tempo", "high": "Brisk Hadr tempo"},
    1: {"low": "Concise Madd timing", "mid": "Precise Madd timing", "high": "Extended Madd holds"},
    2: {"low": "Steady, level intonation", "mid": "Moderate melodic contour", "high": "Wide melodic range"},
    3: {"low": "Light Ghunnah", "mid": "Balanced Ghunnah resonance", "high": "Strong Ghunnah holds"},
}


def describe_traits(query_z: npt.NDArray[np.float64], ref_z: npt.NDArray[np.float64], timbre_sim: float,
                    query_mask: npt.NDArray[np.bool_]) -> list[str]:
    traits: list[str] = []
    for d in range(STYLE_DIM):
        if query_mask[d] and abs(query_z[d] - ref_z[d]) <= 0.5:
            traits.append(_TRAITS[d][_level(float(ref_z[d]))])
    if timbre_sim >= 0.6:
        traits.append("Similar vocal timbre")
    return traits


class ReciterIndex:
    """FAISS-backed store of reciter profiles with combined style/timbre similarity search."""

    def __init__(self, backend: str) -> None:
        self.backend = backend
        self.profiles: list[ReciterProfile] = []
        self._style_mean = STYLE_PRIOR_MEAN.copy()
        self._style_std = STYLE_PRIOR_STD.copy()
        self._timbre_index: Any = None
        self._style_index: Any = None
        self._style_z: npt.NDArray[np.float32] | None = None

    def __len__(self) -> int:
        return len(self.profiles)

    # -- building -------------------------------------------------------------------------
    def add(self, profile: ReciterProfile) -> None:
        if profile.backend != self.backend:
            raise ProfileError(f"Profile backend {profile.backend!r} does not match index backend {self.backend!r}")
        if profile.timbre.shape != (TIMBRE_DIM,):
            raise ProfileError(f"Timbre vector must be {TIMBRE_DIM}-d, got {profile.timbre.shape}")
        self.profiles = [p for p in self.profiles if p.reciter_id != profile.reciter_id]
        self.profiles.append(profile)
        self._timbre_index = None

    def _fit_stats(self) -> None:
        styles = np.array([p.style.as_array() for p in self.profiles]) if self.profiles else np.zeros((0, STYLE_DIM))
        mean, std = STYLE_PRIOR_MEAN.copy(), STYLE_PRIOR_STD.copy()
        for d in range(STYLE_DIM):
            col = styles[:, d][np.isfinite(styles[:, d])] if styles.size else np.zeros(0)
            if col.size >= MIN_PROFILES_FOR_STATS:
                mean[d] = float(col.mean())
                std[d] = max(float(col.std()), 0.5 * STYLE_PRIOR_STD[d])
        self._style_mean, self._style_std = mean, std

    def normalize_style(self, style: StyleVector) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
        raw = style.as_array()
        mask = np.isfinite(raw)
        z = np.where(mask, (raw - self._style_mean) / self._style_std, 0.0)  # impute missing at mean
        return z, mask

    def build(self) -> None:
        import faiss

        if not self.profiles:
            raise ProfileError("Index is empty")
        self._fit_stats()
        timbres = np.stack([_l2(p.timbre) for p in self.profiles]).astype(np.float32)
        self._style_z = np.stack([self.normalize_style(p.style)[0] for p in self.profiles]).astype(np.float32)
        self._timbre_index = faiss.IndexFlatIP(TIMBRE_DIM)
        self._timbre_index.add(timbres)
        self._style_index = faiss.IndexFlatL2(STYLE_DIM)
        self._style_index.add(self._style_z)

    # -- search ---------------------------------------------------------------------------
    def search(self, timbre: FloatArray, style: StyleVector, k: int = 3,
               style_weight: float = STYLE_WEIGHT) -> list[Match]:
        """Rank reciters by ``style_weight·S_style + (1 − style_weight)·S_timbre``."""
        if not 0.0 <= style_weight <= 1.0:
            raise ProfileError(f"style_weight must be in [0, 1], got {style_weight}")
        if self._timbre_index is None:
            self.build()
        assert self._timbre_index is not None and self._style_index is not None and self._style_z is not None
        n = len(self.profiles)
        k = max(1, min(k, n))
        q_t = _l2(timbre).reshape(1, -1).astype(np.float32)
        q_z, mask = self.normalize_style(style)
        # Distances over the dimensions the query actually measured.
        q_zm = np.where(mask, q_z, 0.0).astype(np.float32).reshape(1, -1)
        pool = min(n, max(64, 8 * k))
        _, t_ids = self._timbre_index.search(q_t, pool)
        style_query_index = self._style_index
        if not mask.all():
            import faiss

            style_query_index = faiss.IndexFlatL2(STYLE_DIM)
            style_query_index.add(np.where(mask, self._style_z, 0.0).astype(np.float32))
        _, s_ids = style_query_index.search(q_zm, pool)
        candidates = {int(i) for i in np.concatenate([t_ids[0], s_ids[0]]) if i >= 0}

        results: list[Match] = []
        for i in candidates:
            prof = self.profiles[i]
            t_sim = float(np.clip(np.dot(q_t[0], _l2(prof.timbre)), 0.0, 1.0))
            ref_z = self._style_z[i].astype(np.float64)
            dist = float(np.linalg.norm((q_z - ref_z)[mask])) if mask.any() else float("inf")
            s_sim = math.exp(-(dist**2) / (2 * STYLE_KERNEL_SIGMA**2)) if math.isfinite(dist) else 0.0
            combined = style_weight * s_sim + (1.0 - style_weight) * t_sim
            results.append(Match(prof.reciter_id, prof.name, combined, s_sim, t_sim,
                                 describe_traits(q_z, ref_z, t_sim, mask)))
        results.sort(key=lambda m: m.combined_similarity, reverse=True)
        return results[:k]

    # -- persistence ----------------------------------------------------------------------
    def save(self, directory: str | Path) -> Path:
        import faiss

        if self._timbre_index is None:
            self.build()
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        flat = faiss.IndexFlatIP(PROFILE_DIM)
        vectors = np.stack([p.vector() for p in self.profiles]).astype(np.float32)
        # FAISS cannot store NaN reliably; missing style values are persisted as a sentinel.
        vectors = np.where(np.isfinite(vectors), vectors, np.float32(-1e30))
        flat.add(vectors)
        faiss.write_index(flat, str(out / INDEX_FILE))
        meta = {
            "format": "qaari-eval-reciter-index/1",
            "backend": self.backend,
            "profile_dim": PROFILE_DIM,
            "style_fields": list(STYLE_FIELDS),
            "style_mean": self._style_mean.tolist(),
            "style_std": self._style_std.tolist(),
            "reciters": [
                {"reciter_id": p.reciter_id, "name": p.name, "n_segments": p.n_segments, "source": p.source,
                 "style": p.style.to_dict()}
                for p in self.profiles
            ],
        }
        (out / META_FILE).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return out / INDEX_FILE

    @classmethod
    def load(cls, directory: str | Path) -> ReciterIndex:
        import faiss

        d = Path(directory)
        idx_path, meta_path = d / INDEX_FILE, d / META_FILE
        if not idx_path.exists() or not meta_path.exists():
            raise ProfileError(f"No reciter index found in {d} (run datasets/index_reciters.py first)")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        flat = faiss.read_index(str(idx_path))
        if flat.d != PROFILE_DIM or flat.ntotal != len(meta["reciters"]):
            raise ProfileError("Reciter index and metadata are inconsistent; rebuild the index")
        vectors = flat.reconstruct_n(0, flat.ntotal)
        vectors = np.where(vectors <= -1e29, np.nan, vectors)
        index = cls(meta["backend"])
        for vec, info in zip(vectors, meta["reciters"], strict=True):
            index.profiles.append(ReciterProfile(
                reciter_id=info["reciter_id"], name=info["name"], timbre=vec[:TIMBRE_DIM].astype(np.float32),
                style=StyleVector.from_array(vec[TIMBRE_DIM:]), backend=meta["backend"],
                n_segments=int(info.get("n_segments", 1)), source=info.get("source", ""),
            ))
        index.build()
        return index
