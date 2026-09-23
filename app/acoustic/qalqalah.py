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

CLOSURE_DROP_DB = 8.0  # voiced stops in reverberant rooms keep a voice bar + reverb tail
MIN_CLOSURE_MS = 15.0
SEARCH_AFTER_RELEASE_MS = 60.0
# The burst often precedes the return of voicing (which is where RMS says the closure ends).
SEARCH_BEFORE_RELEASE_MS = 50.0
FLUX_RATIO_MIN = 3.0
RISE_MIN_DB = 6.0
# Weaker evidence of a release (e.g. masked by room reverb) is reported as a WARNING, not a FAIL.
PARTIAL_FLUX_RATIO = 2.0
PARTIAL_RISE_DB = 4.0
RISE_STRONG_DB = {"sughra": 8.0, "kubra": 10.0}
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
    """Measure the release of the occlusion that starts at frame ``a``.

    A weak post-release echo can stay below the closure threshold, merging the occlusion, the
    release and the following pause into one low-energy run ``[a, b)``. The occlusion level is
    therefore taken from the start of the run, and the release is searched both inside the run
    and just after it.
    """
    min_closure = int(MIN_CLOSURE_MS)
    shift = 2  # flux frames (8 ms) vs RMS frames (5 ms): align centres
    s0 = min(len(flux) - 1, a + min_closure + shift)
    s1 = min(len(flux), min(b, a + 250) + int(SEARCH_AFTER_RELEASE_MS) + shift)
    if s1 <= s0:
        return BurstResult(present=False, closure_found=True, closure_ms=(b - a) * hop_s * 1000,
                           release_s=b * hop_s)
    # Score the strongest few flux peaks: inside a long run the largest spike may be noise.
    window = flux[s0:s1]
    order = np.argsort(window)[::-1]
    peaks: list[int] = []
    for k in order:
        if all(abs(int(k) - p) > 8 for p in peaks):
            peaks.append(int(k))
        if len(peaks) == 5:
            break
    best: BurstResult | None = None
    for k in peaks:
        res = _measure_release(db, flux, a, b, k + s0, shift, hop_s, global_med)
        if best is None or (res.present, res.rise_db * min(res.flux_ratio, 20.0)) > (
            best.present, best.rise_db * min(best.flux_ratio, 20.0)
        ):
            best = res
    assert best is not None
    return best


def _measure_release(db: npt.NDArray[np.float64], flux: npt.NDArray[np.float64], a: int, b: int, peak_i: int,
                     shift: int, hop_s: float, global_med: float) -> BurstResult:
    release = peak_i - shift
    occl = db[a: max(a + 5, release - 3)]
    closure_level = float(np.percentile(occl, 20))  # the deepest part of the occlusion
    inner = flux[a + shift: max(a + shift + 1, release - 3 + shift)]
    baseline = float(np.percentile(inner, 25)) if inner.size >= 5 else global_med
    flux_ratio = float(flux[peak_i] / (max(baseline, global_med * 0.25) + 1e-9))
    lo = max(a, release - 5)
    hi = min(len(db), release + int(SEARCH_AFTER_RELEASE_MS))
    rise = float(np.max(db[lo:hi]) - closure_level) if hi > lo else 0.0
    # The acoustic release is where energy returns: the end of the run if it comes soon after the
    # burst, otherwise the burst itself.
    end = b if 0 <= b - release <= SEARCH_BEFORE_RELEASE_MS else release
    return BurstResult(
        present=flux_ratio >= FLUX_RATIO_MIN and rise >= RISE_MIN_DB, closure_found=True,
        closure_ms=max(0, release - a) * hop_s * 1000, release_s=end * hop_s,
        burst_latency_ms=float((release - end) * hop_s * 1000), flux_ratio=flux_ratio, rise_db=rise,
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
    elif result.closure_found and result.flux_ratio >= PARTIAL_FLUX_RATIO and result.rise_db >= PARTIAL_RISE_DB:
        status, score = Status.WARNING, 0.4
        feedback = (
            f"Only a weak release was detected on {letter_name} ({letter}) ({result.rise_db:.0f} dB rise); "
            "make the Qalqalah bounce clearly audible."
        )
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
