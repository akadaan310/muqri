"""End-to-end analysis pipeline: audio + target text -> Tajweed diagnostics + reciter matches."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.acoustic.features import AcousticContext
from app.acoustic.tempo import TempoEstimate, estimate_tempo
from app.aligner import Aligner, AlignmentError, HeuristicAligner, export_alignment, get_aligner
from app.audio import AudioSignal, load_audio
from app.models import Alignment, RuleDiagnostic
from app.profiling import (
    ProfileError,
    ReciterIndex,
    StyleVector,
    TimbreEmbedder,
    compute_style_vector,
    get_timbre_embedder,
)
from app.quran_text import get_ayah_text
from app.scoring import TajweedScorer, consistency_notes
from app.tajweed_rules import ParsedText, TajweedParser

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AnalysisOptions:
    aligner: str = "auto"
    aligner_model: str | None = None
    alignment_json: str | Path | None = None
    index_dir: str | Path | None = "index"
    top_k: int = 3
    style_weight: float = 0.6
    timbre_backend: str = "auto"
    stop_at_end: bool = True
    denoise: str = "auto"
    include_alignment: bool = False
    allow_network: bool = True


@dataclass(slots=True)
class AnalysisResult:
    parsed: ParsedText
    alignment: Alignment
    tempo: TempoEstimate
    diagnostics: list[RuleDiagnostic]
    style: StyleVector
    timbre: Any
    timbre_backend: str
    report: dict[str, Any] = field(default_factory=dict)


class QaariEvaluator:
    """Reusable evaluator; heavy models (aligner, embedder, index) are loaded once."""

    def __init__(self, options: AnalysisOptions | None = None) -> None:
        self.options = options or AnalysisOptions()
        self._aligner: Aligner | None = None
        self._embedder: TimbreEmbedder | None = None
        self._index: ReciterIndex | None = None
        self._index_error: str | None = None
        self._index_loaded = False
        self.scorer = TajweedScorer()

    # -- lazily built components --------------------------------------------------------------
    @property
    def aligner(self) -> Aligner:
        if self._aligner is None:
            kwargs: dict[str, Any] = {"alignment_json": self.options.alignment_json}
            if self.options.aligner_model:
                kwargs["model_name"] = self.options.aligner_model
            self._aligner = get_aligner(self.options.aligner, **kwargs)
        return self._aligner

    @property
    def index(self) -> ReciterIndex | None:
        if not self._index_loaded:
            self._index_loaded = True
            if self.options.index_dir is not None:
                try:
                    self._index = ReciterIndex.load(self.options.index_dir)
                except ProfileError as exc:
                    self._index_error = str(exc)
                    logger.info("Reciter matching disabled: %s", exc)
        return self._index

    @property
    def embedder(self) -> TimbreEmbedder:
        if self._embedder is None:
            backend = self.options.timbre_backend
            if self.index is not None and backend == "auto":
                backend = self.index.backend  # queries must use the same embedding space
            self._embedder = get_timbre_embedder(backend)
        return self._embedder

    # -- analysis -----------------------------------------------------------------------------
    def analyze_signal(self, audio: AudioSignal, text: str, *, reference: str | None = None) -> AnalysisResult:
        parser = TajweedParser(stop_at_end=self.options.stop_at_end)
        parsed = parser.parse(text)
        warnings = list(audio.quality.warnings)

        try:
            alignment = self.aligner.align(audio, parsed)
        except AlignmentError as exc:
            if isinstance(self.aligner, HeuristicAligner) or self.options.aligner not in ("auto",):
                raise
            warnings.append(f"Forced alignment failed ({exc}); used heuristic alignment")
            alignment = HeuristicAligner().align(audio, parsed)
        if alignment.method == "heuristic":
            warnings.append(
                "Heuristic alignment in use: letter timings are approximate. Install the ML extras "
                "(requirements-ml.txt) for CTC forced alignment."
            )

        speech = _speech_span(alignment)
        tempo = estimate_tempo(parsed, alignment, voiced_duration_s=speech[1] - speech[0] if speech else None)
        ctx = AcousticContext(audio)
        diagnostics = self.scorer.evaluate(parsed, alignment, ctx, tempo)
        summary = self.scorer.summarize(diagnostics)
        final_words = frozenset({parsed.words[-1].text}) if parsed.stop_at_end else frozenset()
        style = compute_style_vector(diagnostics, tempo.haraka_ms, ctx, speech, final_words)

        matches_block: dict[str, Any]
        timbre = None
        backend = ""
        try:
            embedder = self.embedder
            timbre = embedder.embed(audio)
            backend = embedder.backend
            index = self.index
            if index is None:
                matches_block = {"top_matches": [], "note": self._index_error or "No reciter index configured"}
            elif index.backend != backend:
                matches_block = {"top_matches": [], "note": (
                    f"Index built with {index.backend!r} embeddings but query used {backend!r}")}
            else:
                matches = index.search(timbre, style, k=self.options.top_k, style_weight=self.options.style_weight)
                matches_block = {"top_matches": [m.to_dict() for m in matches], "index_size": len(index),
                                 "style_weight": self.options.style_weight}
        except ProfileError as exc:
            matches_block = {"top_matches": [], "note": str(exc)}

        report: dict[str, Any] = {
            "recitation_summary": {
                "reference": reference,
                "text": parsed.text,
                "overall_tajweed_score": round(summary.overall, 1) if summary.overall is not None else None,
                "category_scores": summary.by_category,
                "tempo_bpm_harakat": round(tempo.harakat_per_minute, 1),
                "base_haraka_duration_ms": round(tempo.haraka_ms, 1),
                "total_rules_evaluated": summary.evaluated,
                "status_counts": summary.status_counts,
                "tempo": tempo.to_dict(),
                "alignment": {
                    "method": alignment.method,
                    "mean_confidence": round(alignment.mean_confidence, 3),
                    "reliable": alignment.method != "heuristic",
                },
                "audio_quality": audio.quality.to_dict(),
                "notes": consistency_notes(diagnostics),
                "warnings": warnings,
            },
            "detailed_rule_diagnostics": [d.to_dict() for d in diagnostics],
            "reciter_profile": {
                "style_vector": style.to_dict(),
                "timbre_backend": backend or None,
            },
            "reciter_similarity_match": matches_block,
        }
        if self.options.include_alignment:
            report["alignment"] = export_alignment(alignment, parsed)
        return AnalysisResult(parsed, alignment, tempo, diagnostics, style, timbre, backend, report)

    def analyze_file(self, audio_path: str | Path, *, surah: int | None = None, ayah: int | None = None,
                     text: str | None = None) -> AnalysisResult:
        if text is None:
            if surah is None or ayah is None:
                raise ValueError("Provide either --text or both --surah and --ayah")
            text = get_ayah_text(surah, ayah, allow_network=self.options.allow_network)
        audio = load_audio(audio_path, denoise=self.options.denoise)
        ref = f"{surah}:{ayah}" if surah is not None and ayah is not None else None
        return self.analyze_signal(audio, text, reference=ref)


def _speech_span(alignment: Alignment) -> tuple[float, float] | None:
    if not alignment.units:
        return None
    return (min(u.start_s for u in alignment.units.values()), max(u.end_s for u in alignment.units.values()))
