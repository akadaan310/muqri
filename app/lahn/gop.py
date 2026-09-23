"""Substitution-aware goodness of pronunciation (GOP) on the CTC emissions: lahn jali detection.

The forced aligner only asks *where* the target letters are, never *whether* they were read. This
module asks the second question. For every pronounced unit it compares the CTC likelihood of a
local window of the target token sequence against the same window with that unit's letter (or
short/long vowel) replaced by each of its classical confusions (ض→د/ظ, ص→س, ط→ت, ح→ه, ع→ء,
ق→ك, ث→س/ت, ذ→ز/د, ظ→ز/ذ, غ→خ, fatha/kasra/damma, …)::

    LLR(u) = log P(X_w | window with target) − max_v log P(X_w | window with variant v)

Segmentation-free within the window (the full CTC forward sum, not the Viterbi frames), so a
boundary error does not decide the verdict. LLR > 0 means the audio supports the written letter;
LLR ≪ 0 means it sounds like the competitor ``best_alt``. Research basis: deep-research reports
00 §rank-1 and 06 P5 (graph nodes ``algo:gop_subst_aware``, ``algo:gop_sf``, ``phen:lahn:jali``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from app.aligner import _PHONETIC, phonetic_tokens
from app.models import Alignment
from app.tajweed_rules import ParsedText

# Classical lahn-jali confusion sets, on the romanised consonant of the phonetic vocabulary.
CONSONANT_CONFUSIONS: dict[str, tuple[str, ...]] = {
    "ḍ": ("d", "ẓ"), "ẓ": ("z", "dh", "ḍ"), "dh": ("z", "d"), "th": ("s", "t"),
    "ṣ": ("s",), "s": ("ṣ", "th"), "ṭ": ("t",), "t": ("ṭ",), "ḥ": ("h",), "h": ("ḥ",),
    "ʿ": ("'",), "'": ("ʿ",), "q": ("k",), "k": ("q",), "gh": ("kh",), "kh": ("gh", "ḥ"),
    "z": ("dh", "ẓ"), "d": ("ḍ", "dh"),
}
SHORT_VOWELS = ("a", "i", "u")
LONG_VOWELS = ("ā", "ī", "ū")
_ARABIC = {v: k for k, v in _PHONETIC.items() if k not in "ىةأإؤئ"} | {"'": "ء"}
_VOWEL_NAME = {"a": "fatha", "i": "kasra", "u": "damma", "ā": "alif madd", "ī": "yaa madd", "ū": "waw madd"}
CONTEXT_UNITS = 2  # units of context on each side of the judged unit


@dataclass(slots=True)
class UnitGOP:
    unit_index: int
    kind: str          # "consonant" | "vowel"
    target: str        # romanised target (e.g. "ḍ", "a")
    llr: float         # nats; > 0 supports the written text
    best_alt: str      # strongest competitor
    frames: int

    @property
    def llr_per_frame(self) -> float:
        return self.llr / max(1, self.frames)

    def label(self, sym: str) -> str:
        return _ARABIC.get(sym) or _VOWEL_NAME.get(sym, sym)


def ctc_log_likelihood(lp: npt.NDArray[np.floating], seq: list[int], blank: int) -> float:
    """log P(seq | frames) summed over all CTC paths (forward algorithm, log space)."""
    n_frames = lp.shape[0]
    if not seq:
        return float(np.sum(lp[:, blank]))
    ext = np.full(2 * len(seq) + 1, blank, dtype=np.int64)
    ext[1::2] = seq
    n = ext.size
    neg = -np.inf
    alpha = np.full(n, neg)
    alpha[0] = lp[0, blank]
    alpha[1] = lp[0, ext[1]]
    skip = np.zeros(n, dtype=bool)
    skip[2:] = (ext[2:] != blank) & (ext[2:] != ext[:-2])
    for t in range(1, n_frames):
        a1 = np.concatenate([[neg], alpha[:-1]])
        a2 = np.where(skip, np.concatenate([[neg, neg], alpha[:-2]]), neg)
        alpha = np.logaddexp(np.logaddexp(alpha, a1), a2) + lp[t, ext]
    return float(np.logaddexp(alpha[-1], alpha[-2]))


def unit_variants(toks: list[str], cons: str) -> list[tuple[str, str, str, list[str]]]:
    """(kind, target, alternative, substituted tokens) for one unit's token list."""
    out: list[tuple[str, str, str, list[str]]] = []
    joined = "".join(toks)
    if cons and cons in CONSONANT_CONFUSIONS and cons in joined:
        prefix = "l-" if joined.startswith("l-") and cons != "l" else ""
        body = joined[len(prefix):]
        for alt in CONSONANT_CONFUSIONS[cons]:
            out.append(("consonant", cons, alt, list(prefix + body.replace(cons, alt))))
    for group in (SHORT_VOWELS, LONG_VOWELS):
        pos = [i for i, t in enumerate(toks) if t in group]
        if pos:
            i = pos[-1]
            for alt in group:
                if alt != toks[i]:
                    out.append(("vowel", toks[i], alt, [*toks[:i], alt, *toks[i + 1:]]))
    return out


def letter_gop(lp: npt.NDArray[np.floating], frame_s: float, vocab: dict[str, int], blank: int,
               parsed: ParsedText, alignment: Alignment,
               context: int = CONTEXT_UNITS, units: set[int] | None = None) -> list[UnitGOP]:
    """Per-unit substitution LLRs over a local window of ``context`` units on each side.

    ``units`` restricts the judged units (their context still comes from the whole text).
    """
    tokens = phonetic_tokens(parsed)
    order = [i for i in tokens if i in alignment.units and all(t in vocab for t in tokens[i]) and tokens[i]]
    by_index = {u.index: u for u in parsed.units}
    only = units
    out: list[UnitGOP] = []
    n_frames = lp.shape[0]
    for pos, idx in enumerate(order):
        if only is not None and idx not in only:
            continue
        u = by_index[idx]
        cons = "" if u.madd_letter else ("h" if (u.char == "ة" and u.vowel is None) else _PHONETIC.get(u.char, ""))
        variants = unit_variants(tokens[idx], cons)
        if not variants:
            continue
        lo_u, hi_u = order[max(0, pos - context)], order[min(len(order) - 1, pos + context)]
        f0 = max(0, int(np.floor(alignment.units[lo_u].start_s / frame_s)))
        f1 = min(n_frames, int(np.ceil(alignment.units[hi_u].end_s / frame_s)))
        if f1 - f0 < 2:
            continue
        win = lp[f0:f1]
        left = [t for j in order[max(0, pos - context):pos] for t in tokens[j]]
        right = [t for j in order[pos + 1:pos + 1 + context] for t in tokens[j]]
        ids = lambda seq: [vocab[t] for t in seq]  # noqa: E731
        base = ctc_log_likelihood(win, ids(left + tokens[idx] + right), blank)
        if not np.isfinite(base):
            continue
        by_kind: dict[str, tuple[float, str, str]] = {}
        for kind, target, alt, sub in variants:
            if any(t not in vocab for t in sub):
                continue
            ll = ctc_log_likelihood(win, ids(left + sub + right), blank)
            if kind not in by_kind or ll > by_kind[kind][0]:
                by_kind[kind] = (ll, target, alt)
        for kind, (ll, target, alt) in by_kind.items():
            out.append(UnitGOP(idx, kind, target, float(base - ll), alt, f1 - f0))
    return out


# --------------------------------------------------------------------------- reference calibration
# The CTC model is biased per letter even on expert audio (ذ is heard as د, ق leans to ك, ح to ه),
# so a raw LLR threshold is meaningless. Each target symbol is standardised against the reference
# reciters' LLR distribution for that same symbol (robust z, one-sided: only a *low* LLR is
# evidence of a substitution). Symbols with few reference samples shrink to their kind's pool.
MIN_SYMBOL_N = 20
MAD_TO_SD = 1.4826
SCALE_FLOOR = 0.75  # nats


@dataclass(slots=True)
class GopBand:
    median: float
    scale: float
    n: int


def build_reference(rows: list[tuple[str, str, float]]) -> dict[str, GopBand]:
    """Bands keyed ``kind`` and ``kind:target`` from (kind, target, llr) rows of correct recitations."""
    by: dict[str, list[float]] = {}
    for kind, target, llr in rows:
        by.setdefault(kind, []).append(llr)
        by.setdefault(f"{kind}:{target}", []).append(llr)
    out: dict[str, GopBand] = {}
    for key, vals in by.items():
        v = np.asarray(vals, dtype=np.float64)
        med = float(np.median(v))
        out[key] = GopBand(med, max(SCALE_FLOOR, MAD_TO_SD * float(np.median(np.abs(v - med)))), int(v.size))
    return out


def gop_z(bands: dict[str, GopBand], kind: str, target: str, llr: float) -> float | None:
    """How far below the reference the LLR is, in robust SDs (≤ 0 means at or above the reference)."""
    pool = bands.get(kind)
    sym = bands.get(f"{kind}:{target}")
    if pool is None and sym is None:
        return None
    if sym is None or sym.n < MIN_SYMBOL_N:
        w = 0.0 if sym is None else sym.n / MIN_SYMBOL_N
        med = w * (sym.median if sym else 0.0) + (1 - w) * pool.median  # type: ignore[union-attr]
        scale = w * (sym.scale if sym else 0.0) + (1 - w) * pool.scale  # type: ignore[union-attr]
    else:
        med, scale = sym.median, sym.scale
    return (med - llr) / scale
