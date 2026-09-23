"""Qalqalah release-burst detector.

Qalqalah letters (ق ط ب ج د) with sukun are plosives: the articulators close completely
(occlusion: a sharp RMS drop), then release with an audible "bounce". Acoustically the release
is a broadband transient visible as a peak in high-frequency spectral flux and an energy rise
within ~60 ms of the end of the occlusion. Qalqalah Kubra (stopping on the letter) should be
more pronounced than Sughra (mid-word).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from app.acoustic.features import AcousticContext, highband_flux, short_time_rms
from app.models import Alignment, RuleDiagnostic, RuleInstance, Status

CLOSURE_DROP_DB = 12.0
MIN_CLOSURE_MS = 20.0
SEARCH_AFTER_RELEASE_MS = 60.0
# The burst often precedes the return of voicing (which is where RMS says the closure ends).
SEARCH_BEFORE_RELEASE_MS = 50.0
FLUX_RATIO_MIN = 3.0
RISE_MIN_DB = 6.0
RISE_STRONG_DB = {"sughra": 8.0, "kubra": 12.0}
PRE_PAD_S = 0.15  # long enough to include the preceding vowel as level reference
POST_PAD_S = 0.08

LETTER_NAMES = {"ق": "Qaf", "ط": "Taa", "ب": "Baa", "ج": "Jeem", "د": "Dal"}


@dataclass(slots=True)
class BurstResult:
    present: bool
    closure_found: bool
    closure_ms: float = 0.0
    release_s: float = float("nan")
    burst_latency_ms: float = float("nan")
    flux_ratio: float = 0.0
    rise_db: float = 0.0

    def metrics(self) -> dict[str, float]:
        out = {"closure_ms": self.closure_ms, "flux_ratio": self.flux_ratio, "energy_rise_db": self.rise_db}
        if np.isfinite(self.burst_latency_ms):
            out["burst_latency_ms"] = self.burst_latency_ms
        return out


def _runs(mask: npt.NDArray[np.bool_]) -> list[tuple[int, int]]:
    """[start, end) index pairs of consecutive True values."""
    if not mask.any():
        return []
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True))


def detect_release_burst(x: npt.NDArray[np.floating], sr: int, *, letter_start_s: float = 0.0,
                         letter_end_s: float | None = None) -> BurstResult:
    """Find an occlusion followed by a release transient in ``x``.

    ``letter_start_s``/``letter_end_s`` (relative to ``x``) tell the detector where the aligned
    letter lies so that the closure of a neighbouring plosive is not picked up by mistake.
    """
    hop_s = 0.001
    rms = short_time_rms(x, sr, frame_ms=5.0, hop_ms=1.0)
    if rms.size < 20:
        return BurstResult(present=False, closure_found=False)
    db = 20 * np.log10(rms)
    ref = float(np.percentile(db, 98))
    closed = db < ref - CLOSURE_DROP_DB
    runs = [(a, b) for a, b in _runs(closed) if (b - a) * hop_s * 1000 >= MIN_CLOSURE_MS]
    # Ignore leading/trailing silence that touches the segment edges without a release after it.
    runs = [(a, b) for a, b in runs if b < len(db) - 5]
    if letter_end_s is None:
        letter_end_s = len(x) / sr
    lo_i, hi_i = int(letter_start_s / hop_s), int(letter_end_s / hop_s)
    # CTC units start at the release, so the occlusion may end shortly before the aligned letter.
    in_letter = [(a, b) for a, b in runs if b >= lo_i - int(0.06 / hop_s) and a <= hi_i + int(0.03 / hop_s)]
    candidates = in_letter or runs
    if not candidates:
        return BurstResult(present=False, closure_found=False)

    flux = highband_flux(x, sr, cutoff_hz=1500.0, frame_ms=8.0, hop_ms=1.0)
    positive = flux[flux > 0]
    global_med = float(np.median(positive)) if positive.size else 1e-9
    # Evaluate every closure candidate and keep the one followed by the strongest release: at a
    # stop (Kubra) the post-release silence is itself a long low-energy run with no burst after it.
    results = [_release_after(db, flux, a, b, hop_s, global_med) for a, b in candidates]
    return max(results, key=lambda r: (r.present, r.rise_db * min(r.flux_ratio, 20.0)))


def _release_after(db: npt.NDArray[np.float64], flux: npt.NDArray[np.float64], a: int, b: int, hop_s: float,
                   global_med: float) -> BurstResult:
    closure_ms = (b - a) * hop_s * 1000
    closure_level = float(np.median(db[a:b]))  # robust to the fade-in/out edges of the run
    # Align the flux frames (8 ms) with the RMS frames (5 ms) by their centres.
    shift = 2
    s0 = max(a + shift, b - int(SEARCH_BEFORE_RELEASE_MS) + shift)
    s1 = min(len(flux), b + int(SEARCH_AFTER_RELEASE_MS) + shift)
    if s1 <= s0:
        return BurstResult(present=False, closure_found=True, closure_ms=closure_ms, release_s=b * hop_s)
    peak_i = int(np.argmax(flux[s0:s1])) + s0
    inner = flux[a + shift: b + shift]
    baseline = float(np.percentile(inner, 25)) if inner.size >= 5 else global_med
    flux_ratio = float(flux[peak_i] / (max(baseline, global_med * 0.25) + 1e-9))
    e1 = min(len(db), b + int(SEARCH_AFTER_RELEASE_MS))
    rise = float(np.max(db[b:e1]) - closure_level) if e1 > b else 0.0
    return BurstResult(
        present=flux_ratio >= FLUX_RATIO_MIN and rise >= RISE_MIN_DB, closure_found=True,
        closure_ms=closure_ms, release_s=b * hop_s,
        burst_latency_ms=float((peak_i - shift - b) * hop_s * 1000), flux_ratio=flux_ratio, rise_db=rise,
    )


def analyze_qalqalah(rule: RuleInstance, alignment: Alignment, ctx: AcousticContext) -> RuleDiagnostic:
    letter = rule.letter or ""
    letter_name = LETTER_NAMES.get(letter, letter)
    span = alignment.span(rule.unit_indices)
    if span is None:
        return RuleDiagnostic(
            rule_type=rule.rule_type, word=rule.word, start_ms=0, end_ms=0, status=Status.SKIPPED,
            feedback=f"Qalqalah on '{rule.word}' could not be located in the audio.", letter=letter,
        )
    start, end = span
    seg_start = max(0.0, start - PRE_PAD_S)
    seg_end = min(ctx.audio.duration_s, end + POST_PAD_S)
    seg = ctx.audio.segment(seg_start, seg_end)
    result = detect_release_burst(seg, ctx.sr, letter_start_s=start - seg_start, letter_end_s=end - seg_start)
    kind = "kubra" if rule.detail == "kubra" else "sughra"
    strong = RISE_STRONG_DB[kind]

    if result.present and result.rise_db >= strong:
        status, score = Status.PASS, 1.0
        feedback = f"Clear acoustic release burst detected on letter {letter_name} ({letter})."
    elif result.present:
        status, score = Status.WARNING, 0.6
        advice = ("stopping on it calls for a stronger (Kubra) bounce." if kind == "kubra"
                  else "give it a crisper bounce.")
        feedback = f"Release on {letter_name} ({letter}) is present but faint ({result.rise_db:.0f} dB rise); {advice}"
    elif not result.closure_found:
        status, score = Status.FAIL, 0.1
        feedback = (
            f"No occlusion found on {letter_name} ({letter}): the letter was not fully stopped "
            "(it may have been voweled or turned into a fricative)."
        )
    else:
        status, score = Status.FAIL, 0.2
        feedback = (
            f"{letter_name} ({letter}) was closed but not released: the Qalqalah bounce is missing."
        )
    return RuleDiagnostic(
        rule_type=rule.rule_type, word=rule.word, start_ms=int(round(start * 1000)),
        end_ms=int(round(end * 1000)), status=status, feedback=feedback, score=score,
        metrics={k: v for k, v in result.metrics().items() if np.isfinite(v)}, letter=letter,
    )
