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
                 ctx: int = 2, margin: int = 3) -> list[Unit]:
    """Everything the engine knows about every unit of one ayah.

    `lp_full` is the T x C log-posterior matrix of all levels; `ph_first`/`ph_width` locate the
    phoneme block within it and `sifat_blocks` the attribute heads (from the dump's layout.json).
    """
    lp = lp_full[:, ph_first:ph_first + ph_width]
    units = ph_units(phonemes)
    nu = len(units)
    if nu == 0:
        return []
    seq = [vocab[c] for c in phonemes]
    _score, first, last = ctc_viterbi(lp, seq, blank)
    T = lp.shape[0]

    def onset(i: int) -> int:
        return first[units[i][1]]

    def dur(i: int) -> float:
        end = onset(i + 1) if i < nu - 1 else last[units[i][2]] + 1
        return (end - onset(i)) * FRAME_S

    # One count = a VOWELLED LETTER: the consonant contact plus its vowel, so the span runs from this
    # unit's onset to the onset two units later, not one. Matching sukoon_timing.jl exactly matters —
    # spanning only the consonant gave a haraka of 0.08 s against Julia's 0.28 s and inflated every
    # count by 3.5x, while frames and duration_s were already identical.
    harakas = [(onset(min(i + 2, nu - 1)) - onset(i)) * FRAME_S for i in range(nu - 1)
               if units[i][0] not in SHORT_V and units[i][0] not in MADD
               and units[i + 1][0] in SHORT_V]
    haraka = float(np.median(harakas)) if len(harakas) >= 5 else None

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
        if sym in DELETABLE:
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
                 duration_counts=round(dur(i) / haraka, 2) if haraka else None,
                 gop=round(float(gop), 3), best_competitor=best,
                 competitor_llr=None if lr is None else round(lr, 3),
                 confirmed=(lr is None or lr < 0))

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
