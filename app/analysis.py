"""The serving path: per-unit acoustic analysis in numpy.

The research numerics live in Julia (`substrate_library/julia/src/CtcGop.jl`, `SifatGop.jl`) and stay
the reference — that is where the methods were validated against 2 M measured decisions. This module
mirrors the parts a *request* needs, in numpy, so serving a submission is a library call rather than
a subprocess per recording. `tests/test_analysis_parity.py` pins it to the Julia output.

What one call produces for one ayah:

* **identity** of every unit, with the competing letter named and the likelihood ratio against it
* **the ten sifat** with the expected class, the model's best, and the LLR
* **timing** — each unit's onset-to-onset duration in the reciter's own counts

`ctc_log_likelihood` in `app/lahn/gop.py` already implements the CTC forward and is reused; the
Viterbi alignment needed for timing did not exist in Python and is added here.

**Timing has two modes.** The model emits one frame per 40 ms and a short vowel is one or two frames,
so a Viterbi onset quantises every duration to a whole frame. `timing="centroid"` instead takes each
unit's onset as the posterior-weighted centre of its CTC occupancy (forward-backward), which is
continuous: on 60 anchor clips unique fatha durations went from 56/1245 to 1106/1245. Viterbi stays
the default until the centroid path is mirrored in Julia and pinned by parity test.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from app.lahn.gop import ctc_log_likelihood
from app.rule_bind import ph_units

NEG = -np.inf
FRAME_S = 0.04
SHORT_V = frozenset("َُِ")
MADD = frozenset("اۥۦ")
SPECIAL = frozenset("ں۾ڇۜٲ۪ـؙ")
# a sifah belongs to a consonant: during a vowel the folds always vibrate
SIFAT_SKIP = SHORT_V | MADD

# classical confusion sets, the same as CtcGop.CONFUSIONS
CONFUSIONS: dict[str, list[str]] = {
    "ض": ["د", "ظ"], "ظ": ["ز", "ذ", "ض"], "ذ": ["ز", "د"], "ث": ["س", "ت"], "ص": ["س"],
    "س": ["ص", "ث"], "ط": ["ت"], "ت": ["ط"], "ح": ["ه"], "ه": ["ح"], "ع": ["ء"], "ء": ["ع"],
    "ق": ["ك"], "ك": ["ق"], "غ": ["خ"], "خ": ["غ", "ح"], "ز": ["ذ", "ظ"], "د": ["ض", "ذ"],
    "َ": ["ِ", "ُ"], "ِ": ["َ", "ُ"], "ُ": ["َ", "ِ"],
    "ا": ["ۦ", "ۥ"], "ۦ": ["ا", "ۥ"], "ۥ": ["ا", "ۦ"],
}
DELETABLE = frozenset("بتثجحخدذرزسشصضطظعغفقكلمنهويء")


def ctc_viterbi(lp: npt.NDArray[np.floating], seq: list[int], blank: int
                ) -> tuple[float, list[int], list[int]]:
    """Best CTC path; returns its score and the first/last frame of every symbol of `seq`.

    A direct mirror of `CtcGop.ctc_viterbi`, including its back-pointer convention (0 stay,
    1 from s-1, 2 from s-2) so the two produce identical alignments.
    """
    T, L = lp.shape[0], len(seq)
    if L == 0:
        return 0.0, [], []
    S = 2 * L + 1
    ext = [blank if s % 2 == 0 else seq[s // 2] for s in range(S)]
    delta = np.full((T, S), NEG)
    bp = np.zeros((T, S), dtype=np.int8)
    delta[0, 0] = lp[0, blank]
    if S > 1:
        delta[0, 1] = lp[0, ext[1]]
    for t in range(1, T):
        prev = delta[t - 1]
        stay = prev
        one = np.concatenate(([NEG], prev[:-1]))
        skip = np.full(S, NEG)
        if S > 2:
            ok = np.array([ext[s] != blank and ext[s] != ext[s - 2] for s in range(2, S)])
            skip[2:] = np.where(ok, prev[:-2], NEG)
        stacked = np.vstack([stay, one, skip])
        arg = np.argmax(stacked, axis=0)
        best = stacked[arg, np.arange(S)]
        delta[t] = np.where(np.isneginf(best), NEG, best + lp[t, ext])
        bp[t] = arg.astype(np.int8)
    s = S - 2 if (S > 1 and delta[T - 1, S - 2] > delta[T - 1, S - 1]) else S - 1
    score = float(delta[T - 1, s])
    first, last = [0] * L, [0] * L
    for t in range(T - 1, -1, -1):
        if s % 2 == 1:                       # 0-based: odd index is a real symbol
            k = s // 2
            if last[k] == 0:
                last[k] = t + 1
            first[k] = t + 1
        if t > 0:
            s -= int(bp[t, s])
    return score, first, last


def ctc_occupancy(lp: npt.NDArray[np.floating], seq: list[int], blank: int
                  ) -> npt.NDArray[np.floating]:
    """Posterior probability that each symbol of `seq` occupies each frame: a T x len(seq) matrix.

    CTC forward-backward in log space over the blank-interleaved sequence; the occupancy of symbol k
    is gamma at extended state 2k+1. Mirrors `CtcGop.ctc_occupancy`.
    """
    T, L = lp.shape[0], len(seq)
    if L == 0:
        return np.zeros((T, 0))
    S = 2 * L + 1
    ext = np.array([blank if s % 2 == 0 else seq[s // 2] for s in range(S)])
    skip = np.zeros(S, dtype=bool)
    skip[2:] = (ext[2:] != blank) & (ext[2:] != ext[:-2])
    e = lp[:, ext]
    a = np.full((T, S), NEG)
    a[0, :min(2, S)] = e[0, :min(2, S)]
    for t in range(1, T):
        p = a[t - 1]
        one = np.concatenate(([NEG], p[:-1]))
        two = np.where(skip, np.concatenate(([NEG, NEG], p[:-2])), NEG)
        a[t] = np.logaddexp(np.logaddexp(p, one), two) + e[t]
    b = np.full((T, S), NEG)
    b[T - 1, max(0, S - 2):] = 0.0
    for t in range(T - 2, -1, -1):
        n = b[t + 1] + e[t + 1]
        one = np.concatenate((n[1:], [NEG]))
        two = np.concatenate((np.where(skip, n, NEG)[2:], [NEG, NEG]))
        b[t] = np.logaddexp(np.logaddexp(n, one), two)
    log_z = np.logaddexp(a[T - 1, S - 1], a[T - 1, S - 2]) if S > 1 else a[T - 1, S - 1]
    return np.exp(np.clip(a[:, 1::2] + b[:, 1::2] - log_z, -60.0, 0.0))


def centroid_onsets(lp: npt.NDArray[np.floating], seq: list[int], blank: int
                    ) -> npt.NDArray[np.floating]:
    """Continuous onset of every symbol, in 1-based frames to match `ctc_viterbi`'s `first`.

    The centre of mass of the symbol's occupancy. CTC is peaky, so the centre sits where the model
    commits to the symbol; consecutive centres give the interval between articulations with no 40 ms
    quantisation. A symbol with no occupancy mass (only possible on a degenerate clip) is NaN.
    """
    g = ctc_occupancy(lp, seq, blank)
    mass = g.sum(axis=0)
    t = np.arange(1, lp.shape[0] + 1, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(mass > 1e-12, (g * t[:, None]).sum(axis=0) / mass, np.nan)


@dataclass(slots=True)
class Unit:
    """One phoneme run and everything the acoustic model says about it."""

    index: int
    symbol: str
    kind: str                      # consonant | harakah | madd | ikhfa_noon | iqlab_meem | qalqala_release
    run_length: int
    char_span: tuple[int, int]
    frames: tuple[int, int]
    onset_s: float
    duration_s: float
    duration_counts: float | None
    gop: float
    best_competitor: str
    competitor_llr: float | None
    confirmed: bool
    sifat: dict[str, dict] = field(default_factory=dict)  # type: ignore[type-arg]
    # the recording starts or ends inside this letter's test window (`margin` frames either side), so
    # its identity and characteristics were tested partly on audio that is not there. Learner
    # recordings are cut this tightly far more often than masters' (the last letter ends within
    # 0.12 s of the file end in 58 % of sobolev210 learner ayahs, 2.7 % of T300 masters), and there it
    # failed 46 % of the time against 11 % with room after it. It is REPORTED, not used to withhold a
    # verdict: reviewers' tagged errors sit on these letters too, and excluding them traded recall
    # for false alarms one for one (lift over chance 2.06 -> 2.02..2.13 across thresholds).
    edge: bool = False


def _plausible(counts: float, cap: float) -> float | None:
    """A count beyond any tajweed requirement is a measurement failure, not a reading."""
    return round(counts, 2) if 0 < counts <= cap else None


def _kind(sym: str) -> str:
    if sym in SHORT_V:
        return "harakah"
    if sym in MADD:
        return "madd"
    return {"ں": "ikhfa_noon", "۾": "iqlab_meem", "ڇ": "qalqala_release"}.get(sym, "consonant")


def _pooled(x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Mean posterior of each column in log space (log of the average probability)."""
    m = x.max(axis=0)
    return m + np.log(np.exp(x - m).sum(axis=0)) - np.log(x.shape[0])


def analyse_clip(lp_full: npt.NDArray[np.floating], phonemes: str, vocab: dict[str, int], blank: int,
                 ph_first: int, ph_width: int, sifat_blocks: dict[str, tuple[int, int, list[str]]],
                 expected_sifat: dict[str, list[int]] | None = None,
                 ctx: int = 2, margin: int = 3, timing: str = "viterbi",
                 haraka_s: float | None = None) -> list[Unit]:
    """Everything the engine knows about every unit of one ayah.

    `lp_full` is the T x C log-posterior matrix of all levels; `ph_first`/`ph_width` locate the
    phoneme block within it and `sifat_blocks` the attribute heads (from the dump's layout.json).
    `timing` is "viterbi" (whole frames) or "centroid" (continuous; see the module docstring). Only
    onsets and durations change with it: `frames`, the sifat windows and the GOP contexts stay on the
    Viterbi path, which is what they were validated on.

    `haraka_s` is the count unit to fall back on when this ayah is too short to measure its own
    (fewer than five vowelled letters -- الٓمٓ has none). Without it every duration in such an ayah is
    unknown, and a madd lazim of six counts cannot be judged at all.
    """
    if timing not in ("viterbi", "centroid"):
        raise ValueError(f"timing must be 'viterbi' or 'centroid', not {timing!r}")
    lp = lp_full[:, ph_first:ph_first + ph_width]
    units = ph_units(phonemes)
    nu = len(units)
    if nu == 0:
        return []
    seq = [vocab[c] for c in phonemes]
    _score, first, last = ctc_viterbi(lp, seq, blank)
    T = lp.shape[0]
    start: list[float] = [float(f) for f in first]
    if timing == "centroid":
        cen = centroid_onsets(lp, seq, blank)
        # fall back per symbol, never per clip, so one degenerate symbol cannot unset the rest
        start = [float(c) if np.isfinite(c) else f for c, f in zip(cen, start)]

    def onset(i: int) -> float:
        return start[units[i][1]]

    def dur(i: int) -> float:
        end = onset(i + 1) if i < nu - 1 else last[units[i][2]] + 1
        return (end - onset(i)) * FRAME_S

    # The LAST unit of a clip runs to the end of the audio, so it absorbs whatever trailing silence
    # or breath follows. That is where madd 'arid li-s-sukun sits, and it read 37 counts before this
    # guard. Its duration is not measurable from a clip boundary, so it is reported as unknown rather
    # than as a wild number a verdict might act on.
    unreliable_tail = nu - 1
    # No tajweed duration exceeds six counts (madd lazim is the longest), so a measured value far
    # beyond that is not a very long madd — it is a measurement failure, almost always segmentation
    # slop at an ayah edge stretching the last madd before the stop. Measured: madd 'arid read 37
    # counts. Report it as unknown rather than letting a verdict act on it.
    MAX_PLAUSIBLE_COUNTS = 12.0

    # One count = a VOWELLED LETTER: the consonant contact plus its vowel, so the span runs from this
    # unit's onset to the onset two units later, not one. Matching sukoon_timing.jl exactly matters —
    # spanning only the consonant gave a haraka of 0.08 s against Julia's 0.28 s and inflated every
    # count by 3.5x, while frames and duration_s were already identical.
    harakas = [(onset(min(i + 2, nu - 1)) - onset(i)) * FRAME_S for i in range(nu - 1)
               if units[i][0] not in SHORT_V and units[i][0] not in MADD
               and units[i + 1][0] in SHORT_V]
    haraka = float(np.median(harakas)) if len(harakas) >= 5 else haraka_s

    out: list[Unit] = []
    for i, (sym, a, b) in enumerate(units):
        lo, hi = max(0, i - ctx), min(nu - 1, i + ctx)
        ca, cb = units[lo][1], units[hi][2]
        t0 = max(0, first[ca] - 1 - margin)
        t1 = min(T, last[cb] + margin)
        x = lp[t0:t1] if t1 > t0 else lp
        left, right = seq[ca:a], seq[b + 1:cb + 1]
        n = b - a + 1
        ref = ctc_log_likelihood(x, [*left, *seq[a:b + 1], *right], blank)
        alts: list[tuple[str, float]] = []
        for q in CONFUSIONS.get(sym, []):
            if q in vocab:
                alts.append((q, ctc_log_likelihood(x, [*left, *([vocab[q]] * n), *right], blank) - ref))
        # The ي / و of a madd leen (sakin after fatha: بَيْن، يَوْم) is a glide into the vowel, not a
        # consonant with a closure; testing it against deletion called Husary's leen in ٱلْمَغْرِبَيْنِ
        # "dropped" -- a certified reviewer rejected it. Neither glide has a classical confusion, so a
        # leen unit carries no identity test at all. Mirrors CtcGop.gop_sf.
        leen = (sym in "يو" and i > 0 and units[i - 1][0] == "َ"
                and (i == nu - 1 or units[i + 1][0] not in SHORT_V))
        if sym in DELETABLE and not leen:
            alts.append(("∅", ctc_log_likelihood(x, [*left, *right], blank) - ref))
        if alts:
            lrs = [v for _q, v in alts]
            m = max(lrs)
            z = (np.exp(-m) + sum(np.exp(v - m) for v in lrs)) if m > 0 else (1 + sum(np.exp(v) for v in lrs))
            gop = -(m + float(np.log(z))) if m > 0 else -float(np.log(z))
            k = int(np.argmax(lrs))
            best, lr = alts[k][0], float(lrs[k])
        else:
            gop, best, lr = 0.0, "", None

        u = Unit(index=i, symbol=sym, kind=_kind(sym), run_length=n, char_span=(a, b),
                 frames=(first[a], last[b]), onset_s=round((onset(i) - 1) * FRAME_S, 3),
                 duration_s=round(dur(i), 3),
                 duration_counts=_plausible(dur(i) / haraka, MAX_PLAUSIBLE_COUNTS)
                 if haraka and i != unreliable_tail else None,
                 gop=round(float(gop), 3), best_competitor=best,
                 competitor_llr=None if lr is None else round(lr, 3),
                 confirmed=(lr is None or lr < 0),
                 edge=first[a] - 1 - margin < 0 or last[b] + margin >= T)

        if expected_sifat and sym not in SIFAT_SKIP:
            t0s, t1s = first[a] - 1, last[b]
            if 0 <= t0s < t1s <= T:
                for lvl, (lfirst, width, names) in sifat_blocks.items():
                    ids = expected_sifat.get(lvl, [])
                    if a >= len(ids) or ids[a] <= 0:
                        continue
                    s = _pooled(lp_full[t0s:t1s, lfirst:lfirst + width])
                    ecol = ids[a]
                    if ecol >= width:
                        continue
                    others = [(j, s[j]) for j in range(1, width) if j != ecol]
                    if not others:
                        continue
                    bj, bv = max(others, key=lambda kv: kv[1])
                    u.sifat[lvl] = {"expected": names[ecol], "model_best": names[bj],
                                    "llr": round(float(s[ecol] - bv), 3),
                                    "realised": bool(s[ecol] > bv)}
        out.append(u)
    return out
