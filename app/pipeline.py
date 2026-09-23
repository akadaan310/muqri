"""End-to-end analysis pipeline (v2).

audio + Uthmani text -> [Taraweeh adapter] -> alignment -> dynamic waqf re-parse -> Ahkaam & Sifaat
validators -> perfection index, Sifaat score, 232-d fingerprint, reciter matches, benchmark.

Modes:

* ``studio``: dry recording; global harakah, no dereverberation.
* ``taraweeh_adapted``: WPE + late-reverb suppression + proximity EQ, local (pace-normalised)
  harakah, and breath-aware Waqf al-Dharoori handling.
* ``auto``: ``taraweeh_adapted`` when the measured RT60 is above 0.6 s or background noise is
  within 25 dB of the voice, ``studio`` otherwise.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.acoustic.features import AcousticContext
from app.acoustic.tempo import TempoEstimate, estimate_tempo
from app.aligner import Aligner, AlignmentError, HeuristicAligner, export_alignment, get_aligner
from app.audio import AudioSignal, load_audio
from app.fingerprint import (
    DEFAULT_STYLE_WEIGHT,
    Fingerprint,
    ProfileError,
    ReciterIndex,
    TimbreEmbedder,
    benchmark_comparison,
    compute_tajweed_vector,
    get_timbre_embedder,
    load_indices,
)
from app.models import Alignment, RuleDiagnostic
from app.quran_text import AYAH_COUNTS, get_ayah_text
from app.scoring import TajweedScorer, consistency_notes
from app.tajweed_rules.base import EvalContext, Pause
from app.tajweed_rules.parser import ParsedText, TajweedParser
from app.taraweeh_adapter.dereverb import EnvironmentProfile, adapt_acoustics, environment_profile
from app.taraweeh_adapter.fatigue_detector import detect_pauses, dynamic_stops, fatigue_report, pitch_profile
from app.taraweeh_adapter.pace_normalizer import build_local_tempo, classify_pace, hadr_to_tahqeeq_ratio

logger = logging.getLogger(__name__)

MODES = ("studio", "taraweeh_adapted", "auto")
NOISY_BACKGROUND_DB = -25.0


@dataclass(slots=True)
class AnalysisOptions:
    aligner: str = "auto"
    aligner_model: str | None = None
    alignment_json: str | Path | None = None
    index_dir: str | Path | None = "index"
    top_k: int = 3
    style_weight: float = DEFAULT_STYLE_WEIGHT
    timbre_backend: str = "auto"
    stop_at_end: bool = True
    denoise: str = "auto"
    include_alignment: bool = False
    allow_network: bool = True
    tareeq: str = "shatibiyyah"
    mode: str = "auto"
    benchmark: str | None = None
    dynamic_waqf: bool = True
    include_sifaat: bool = True
    compute_fingerprint: bool = True


@dataclass(slots=True)
class AnalysisResult:
    parsed: ParsedText
    alignment: Alignment
    tempo: TempoEstimate
    diagnostics: list[RuleDiagnostic]
    fingerprint: Fingerprint | None
    mode: str
    environment: EnvironmentProfile
    report: dict[str, Any] = field(default_factory=dict)


class QaariEvaluator:
    """Reusable evaluator; heavy models (aligner, embedder, indices) are loaded once."""

    def __init__(self, options: AnalysisOptions | None = None) -> None:
        self.options = options or AnalysisOptions()
        if self.options.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self._aligner: Aligner | None = None
        self._embedder: TimbreEmbedder | None = None
        self._indices: dict[str, ReciterIndex] | None = None
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
    def indices(self) -> dict[str, ReciterIndex]:
        if self._indices is None:
            self._indices = load_indices(self.options.index_dir) if self.options.index_dir is not None else {}
        return self._indices

    @property
    def embedder(self) -> TimbreEmbedder:
        if self._embedder is None:
            backend = self.options.timbre_backend
            if backend == "auto" and self.indices:
                backend = next(iter(self.indices.values())).backend  # queries must match the index space
            self._embedder = get_timbre_embedder(backend)
        return self._embedder

    # -- helpers ------------------------------------------------------------------------------
    def resolve_mode(self, env: EnvironmentProfile) -> str:
        if self.options.mode != "auto":
            return self.options.mode
        noise = env.background_crowd_noise_level
        noisy = math.isfinite(noise) and noise > NOISY_BACKGROUND_DB
        return "taraweeh_adapted" if env.reverberant or noisy else "studio"

    def _align(self, audio: AudioSignal, parsed: ParsedText, warnings: list[str]) -> Alignment:
        try:
            return self.aligner.align(audio, parsed)
        except AlignmentError as exc:
            if isinstance(self.aligner, HeuristicAligner) or self.options.aligner != "auto":
                raise
            warnings.append(f"Forced alignment failed ({exc}); used heuristic alignment")
            return HeuristicAligner().align(audio, parsed)

    # -- analysis -----------------------------------------------------------------------------
    def analyze_signal(self, audio: AudioSignal, text: str | list[str], *, reference: str | None = None
                       ) -> AnalysisResult:
        opts = self.options
        warnings = list(audio.quality.warnings)
        env = environment_profile(audio)
        mode = self.resolve_mode(env)
        work = audio
        adapter_info: dict[str, Any] = {}
        if mode == "taraweeh_adapted":
            res = adapt_acoustics(audio)
            work = res.audio
            adapter_info = {"applied": res.applied, "rt60_before_s": _r(res.rt60_before),
                            "rt60_after_s": _r(res.rt60_after)}

        parser = TajweedParser(stop_at_end=opts.stop_at_end, tareeq=opts.tareeq, include_sifaat=opts.include_sifaat)
        parsed = parser.parse(text)
        alignment = self._align(work, parsed, warnings)
        ctx = AcousticContext(work)
        pauses: list[Pause] = detect_pauses(ctx, alignment)
        added_stops: set[int] = set()
        if opts.dynamic_waqf and alignment.method != "heuristic":
            added_stops = dynamic_stops(parsed, alignment, pauses)
            if added_stops:
                # The reciter stopped where the text continues: waqf rules apply there.
                stop_words = sorted(parsed.words[i].text for i in added_stops)
                parsed = parser.parse(text, stops=_word_positions(parser, text, added_stops))
                alignment = self._align(work, parsed, warnings)
                pauses = detect_pauses(ctx, alignment)
                warnings.append(f"Stops detected mid-ayah after: {', '.join(stop_words)} (re-read with waqf rules)")
        if alignment.method == "heuristic":
            warnings.append("Heuristic alignment in use: letter timings are approximate. Install the ML extras "
                            "(requirements-ml.txt) for CTC forced alignment.")

        speech = _speech_span(alignment)
        tempo = estimate_tempo(parsed, alignment, voiced_duration_s=speech[1] - speech[0] if speech else None)
        local = build_local_tempo(parsed, alignment, tempo.haraka_ms)
        ev = EvalContext(parsed=parsed, alignment=alignment, ctx=ctx, tempo=tempo, mode=mode, pauses=pauses,
                         local_haraka_fn=local if mode == "taraweeh_adapted" else None)
        diagnostics = self.scorer.evaluate(ev)
        summary = self.scorer.summarize(diagnostics)
        fatigue = fatigue_report(ctx, pauses)

        fp: Fingerprint | None = None
        matches: dict[str, Any] = {"top_matches": []}
        bench: dict[str, Any] | None = None
        if opts.compute_fingerprint:
            try:
                embedder = self.embedder
                tajweed = compute_tajweed_vector(diagnostics, tempo.haraka_ms, parser.tareeq, summary.overall)
                fp = Fingerprint(embedder.embed(work), tajweed, env.as_array(), embedder.backend)
                for category, index in self.indices.items():
                    if index.backend != fp.backend:
                        matches.setdefault("notes", []).append(
                            f"{category} index uses {index.backend!r} embeddings; query used {fp.backend!r}")
                        continue
                    matches["top_matches"] += [m.to_dict() for m in index.search(fp, opts.top_k, opts.style_weight)]
                matches["top_matches"].sort(key=lambda m: m["combined_similarity_pct"], reverse=True)
                matches["top_matches"] = matches["top_matches"][: opts.top_k]
                matches["style_weight"] = opts.style_weight
                if not self.indices:
                    matches["note"] = "No reciter index found (run datasets/index_reciters.py)"
                if opts.benchmark:
                    ref = next((p for idx in self.indices.values() if (p := idx.find(opts.benchmark)) is not None),
                               None)
                    bench = benchmark_comparison(fp, ref) if ref is not None else {
                        "benchmark": opts.benchmark, "note": "Benchmark reciter not found in the indices"}
            except ProfileError as exc:
                matches["note"] = str(exc)

        report: dict[str, Any] = {
            "recitation_summary": {
                "reference": reference,
                "text": parsed.text,
                "tareeq": parser.tareeq.value,
                "mode": mode,
                "overall_tajweed_score": _r(summary.overall, 1),
                "tajweed_perfection_index": _r(summary.overall, 1),
                "sifaat_score": _r(summary.sifaat, 1),
                "category_scores": summary.by_category,
                "tempo_bpm_harakat": round(tempo.harakat_per_minute, 1),
                "base_haraka_duration_ms": round(tempo.haraka_ms, 1),
                "pace": classify_pace(tempo.haraka_ms),
                "hadr_to_tahqeeq_ratio": _r(hadr_to_tahqeeq_ratio(tempo.haraka_ms), 3),
                "local_tempo_variability": _r(local.variability, 3),
                "total_rules_evaluated": summary.evaluated,
                "status_counts": summary.status_counts,
                "tempo": tempo.to_dict(),
                "alignment": {"method": alignment.method, "mean_confidence": round(alignment.mean_confidence, 3),
                              "reliable": alignment.method != "heuristic"},
                "audio_quality": audio.quality.to_dict(),
                "acoustic_environment": env.to_dict(),
                "taraweeh_adapter": adapter_info or None,
                "fatigue": fatigue.to_dict(),
                "pitch_style": pitch_profile(ctx, speech),
                "notes": consistency_notes(diagnostics),
                "warnings": warnings,
            },
            "detailed_rule_diagnostics": [d.to_dict() for d in diagnostics],
            "fingerprint": fp.to_dict() if fp is not None else None,
            "reciter_similarity_match": matches,
        }
        if bench is not None:
            report["benchmark_comparison"] = bench
        if opts.include_alignment:
            report["alignment"] = export_alignment(alignment, parsed)
        return AnalysisResult(parsed, alignment, tempo, diagnostics, fp, mode, env, report)

    def analyze_file(self, audio_path: str | Path, *, surah: int | None = None, ayah: int | None = None,
                     ayah_end: int | None = None, text: str | list[str] | None = None) -> AnalysisResult:
        ref: str | None = None
        if text is None:
            if surah is None:
                raise ValueError("Provide --text or --surah (optionally with --ayah)")
            first = ayah or 1
            last = ayah_end or (ayah if ayah is not None else AYAH_COUNTS[surah - 1])
            text = [get_ayah_text(surah, a, allow_network=self.options.allow_network) for a in range(first, last + 1)]
            ref = f"{surah}:{first}" if first == last else f"{surah}:{first}-{last}"
            if len(text) == 1:
                text = text[0]
        audio = load_audio(audio_path, denoise=self.options.denoise)
        return self.analyze_signal(audio, text, reference=ref)


def _word_positions(parser: TajweedParser, text: str | list[str], parsed_word_indices: set[int]) -> set[int]:
    """Map parsed-word indices (muqatta'at expand into several) back to source-word positions."""
    parsed = parser.parse(text)
    source_of: list[int] = []
    src = -1
    last_text = None
    for w in parsed.words:
        if w.text != last_text or not any(parsed.units[i].synthetic for i in w.unit_indices):
            src += 1
        last_text = w.text
        source_of.append(src)
    return {source_of[i] for i in parsed_word_indices if i < len(source_of)}


def _r(v: float | None, n: int = 3) -> float | None:
    return round(float(v), n) if v is not None and math.isfinite(v) else None


def _speech_span(alignment: Alignment) -> tuple[float, float] | None:
    if not alignment.units:
        return None
    return (min(u.start_s for u in alignment.units.values()), max(u.end_s for u in alignment.units.values()))
