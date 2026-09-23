"""Ayah locator: find which part of a passage each stretch of a long recording contains.

Live Taraweeh sessions (and whole-surah uploads) interleave Qur'an with takbeer, du'a and
silence, repeat Al-Fatiha every rak'ah, and pause in the middle of long ayahs. The locator:

1. cuts the audio at pauses (≥ 0.35 s) into chunks of at most ~30 s;
2. aligns every chunk against the *whole* passage with a **semi-global CTC Viterbi**: the path may
   start at any word and end at any later word, so a chunk that holds half an ayah, or the end of
   one ayah and the start of the next, is located exactly;
3. keeps chunks whose mean log-posterior per frame and matched length are plausible, and drops
   non-Qur'anic speech (takbeer, du'a) that aligns poorly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from app.aligner import AlignmentError, CTCForcedAligner
from app.audio import AudioSignal, frame_rms_db
from app.tajweed_rules.parser import ParsedText, TajweedParser

logger = logging.getLogger(__name__)

SPLIT_PAUSE_S = 0.35
MAX_CHUNK_S = 30.0
MIN_CHUNK_S = 0.8
MIN_SCORE = -1.6  # mean log-posterior per frame along the best path
TOKEN_BONUS = 2.0
MIN_WORDS = 1
TOKENS_PER_S = (2.0, 25.0)


@dataclass(slots=True)
class Chunk:
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(slots=True)
class LocatedChunk:
    chunk: Chunk
    word_start: int  # passage word indices, inclusive
    word_end: int
    score: float

    @property
    def n_words(self) -> int:
        return self.word_end - self.word_start + 1


def split_on_pauses(audio: AudioSignal, *, pause_s: float = SPLIT_PAUSE_S, max_chunk_s: float = MAX_CHUNK_S,
                    min_chunk_s: float = MIN_CHUNK_S) -> list[Chunk]:
    """Speech chunks separated by pauses; long chunks are cut at their longest internal dip."""
    db = frame_rms_db(audio.samples, audio.sr, 20.0, 10.0).astype(np.float64)
    if db.size == 0:
        return []
    floor, peak = float(np.percentile(db, 5)), float(np.percentile(db, 95))
    speech = db >= max(floor + 6.0, peak - 30.0)
    chunks: list[Chunk] = []
    start, silent = None, 0
    min_gap = int(pause_s / 0.01)
    for k, sp in enumerate(np.append(speech, [False] * (min_gap + 1))):
        if sp:
            if start is None:
                start = k
            silent = 0
        elif start is not None:
            silent += 1
            if silent >= min_gap:
                chunks.append(Chunk(start * 0.01, (k - silent + 1) * 0.01))
                start, silent = None, 0
    # Merge fragments that are too short to align on their own.
    merged: list[Chunk] = []
    for c in chunks:
        if merged and (c.duration_s < min_chunk_s or merged[-1].duration_s < min_chunk_s) \
                and c.end_s - merged[-1].start_s <= max_chunk_s:
            merged[-1] = Chunk(merged[-1].start_s, c.end_s)
        else:
            merged.append(c)
    out: list[Chunk] = []
    for c in merged:
        out.extend(_cut_long(c, db, max_chunk_s))
    pad = 0.08
    return [Chunk(max(0.0, c.start_s - pad), min(audio.duration_s, c.end_s + pad)) for c in out]


def _cut_long(c: Chunk, db: npt.NDArray[np.float64], max_s: float) -> list[Chunk]:
    if c.duration_s <= max_s:
        return [c]
    a, b = int(c.start_s / 0.01), int(c.end_s / 0.01)
    lo, hi = a + int(0.3 * (b - a)), a + int(0.7 * (b - a))
    cut = lo + int(np.argmin(db[lo:hi])) if hi > lo else (a + b) // 2
    return _cut_long(Chunk(c.start_s, cut * 0.01), db, max_s) + _cut_long(Chunk(cut * 0.01, c.end_s), db, max_s)


def ctc_semiglobal_align(log_probs: npt.NDArray[np.floating], targets: list[int], blank: int,
                         starts: npt.NDArray[np.bool_], ends: npt.NDArray[np.bool_],
                         token_bonus: float = TOKEN_BONUS) -> tuple[int, int, float]:
    """Best CTC path that begins at a token flagged in ``starts`` and ends at one in ``ends``.

    ``token_bonus`` (nats) is added whenever the path enters a new token. Without it a peaky CTC
    model prefers a short span padded with blanks; the bonus makes the path cover what was said.

    Returns (first token index, last token index, mean log-probability per frame without the bonus).
    """
    lp = np.asarray(log_probs, dtype=np.float64)
    n_frames = lp.shape[0]
    n_tok = len(targets)
    ext = np.full(2 * n_tok + 1, blank, dtype=np.int64)
    ext[1::2] = targets
    n_states = ext.size
    neg = -np.inf
    start_states = np.zeros(n_states, dtype=bool)
    tok_start = np.flatnonzero(starts)
    start_states[2 * tok_start] = True  # blank before the token
    start_states[2 * tok_start + 1] = True
    end_states = np.zeros(n_states, dtype=bool)
    tok_end = np.flatnonzero(ends)
    end_states[2 * tok_end + 1] = True
    end_states[2 * tok_end + 2] = True
    can_skip = np.zeros(n_states, dtype=bool)
    can_skip[2:] = (ext[2:] != blank) & (ext[2:] != ext[:-2])

    dp = np.where(start_states, lp[0, ext], neg)
    origin = np.where(start_states, np.arange(n_states), -1)  # start state of the best path so far
    is_token = (np.arange(n_states) % 2 == 1).astype(np.float64) * token_bonus
    dp = dp + np.where(start_states, is_token, 0.0)
    bonus_count = np.where(start_states & (np.arange(n_states) % 2 == 1), 1, 0)
    for t in range(1, n_frames):
        stay = dp
        prev1 = np.concatenate([[neg], dp[:-1]]) + is_token
        prev2 = np.where(can_skip, np.concatenate([[neg, neg], dp[:-2]]), neg) + is_token
        best = np.maximum(np.maximum(stay, prev1), prev2)
        from_stay, from1 = best == stay, best == prev1
        origin = np.where(from_stay, origin, np.where(from1, np.concatenate([[-1], origin[:-1]]),
                                                      np.concatenate([[-1, -1], origin[:-2]])))
        bonus_count = np.where(from_stay, bonus_count,
                               np.where(from1, np.concatenate([[0], bonus_count[:-1]]),
                                        np.concatenate([[0, 0], bonus_count[:-2]])) + (np.arange(n_states) % 2))
        dp = best + lp[t, ext]
    final = np.where(end_states, dp, neg)
    s_end = int(np.argmax(final))
    if not np.isfinite(final[s_end]):
        raise AlignmentError("No semi-global CTC path")
    s_start = int(origin[s_end])
    raw = float(final[s_end] - token_bonus * bonus_count[s_end])
    return s_start // 2, (s_end - 1) // 2, raw / n_frames


class AyahLocator:
    """Locate the passage words spoken in each chunk of a long recording.

    Pass 1 aligns every chunk against the whole passage independently. Pass 2 keeps the
    largest forward-moving chain of matches (a score-weighted longest increasing subsequence),
    because a reciter moves forward through the passage, and uses it as anchors. Pass 3
    re-searches every other chunk only between its neighbouring anchors. Chunks that still
    align poorly (takbeer, du'a, coughs) are dropped.
    """

    def __init__(self, aligner: CTCForcedAligner, passage: ParsedText) -> None:
        self.aligner = aligner
        self.passage = passage
        targets, owner, _ = aligner.targets(passage)
        self.targets = targets
        self.token_word = np.array([passage.units[u].word_index for u in owner])
        self.word_starts = np.r_[True, self.token_word[1:] != self.token_word[:-1]]
        self.word_ends = np.r_[self.token_word[1:] != self.token_word[:-1], True]

    def _search(self, lp: npt.NDArray[np.float32], lo_word: int = 0, hi_word: int | None = None
                ) -> tuple[int, int, float] | None:
        hi = len(self.passage.words) - 1 if hi_word is None else hi_word
        in_range = (self.token_word >= lo_word) & (self.token_word <= hi)
        starts, ends = self.word_starts & in_range, self.word_ends & in_range
        if not starts.any() or not ends.any():
            return None
        try:
            first, last, score = ctc_semiglobal_align(lp, self.targets, self.aligner.blank, starts, ends)
        except AlignmentError:
            return None
        return int(self.token_word[first]), int(self.token_word[last]), score

    def locate(self, audio: AudioSignal, chunks: list[Chunk]) -> list[LocatedChunk]:
        emissions: list[npt.NDArray[np.float32] | None] = []
        first_pass: list[tuple[int, int, float] | None] = []
        for c in chunks:
            seg = AudioSignal(samples=audio.segment(c.start_s, c.end_s), sr=audio.sr, quality=audio.quality)
            if seg.duration_s < 0.3:
                emissions.append(None)
                first_pass.append(None)
                continue
            lp, _ = self.aligner.emissions(seg)
            emissions.append(lp)
            hit = self._search(lp)
            first_pass.append(hit if hit is not None and hit[2] >= MIN_SCORE else None)

        anchors = _forward_chain(first_pass)
        located: list[LocatedChunk] = []
        for k, c in enumerate(chunks):
            lp = emissions[k]
            if lp is None:
                continue
            if k in anchors:
                hit = first_pass[k]
            else:
                prev = max((a for a in anchors if a < k), default=None)
                nxt = min((a for a in anchors if a > k), default=None)
                before = first_pass[prev] if prev is not None else None
                after = first_pass[nxt] if nxt is not None else None
                lo = max(0, before[1] - 3) if before else 0
                hi = after[0] + 3 if after else None
                if hi is not None and hi < lo:
                    continue
                hit = self._search(lp, lo, hi)
            if hit is None or hit[2] < MIN_SCORE:
                logger.debug("Chunk %.1f-%.1f not located", c.start_s, c.end_s)
                continue
            located.append(LocatedChunk(c, hit[0], hit[1], hit[2]))
        return located


def _forward_chain(hits: list[tuple[int, int, float] | None], backtrack_words: int = 3) -> set[int]:
    """Indices of the score-weighted longest chain of chunks whose passage position moves forward."""
    idx = [k for k, h in enumerate(hits) if h is not None]
    if not idx:
        return set()
    weight = {k: (hits[k][1] - hits[k][0] + 1) for k in idx}  # type: ignore[index]
    best: dict[int, float] = {}
    back: dict[int, int | None] = {}
    for j, k in enumerate(idx):
        best[k], back[k] = weight[k], None
        for m in idx[:j]:
            if hits[k][0] >= hits[m][1] - backtrack_words and best[m] + weight[k] > best[k]:  # type: ignore[index]
                best[k], back[k] = best[m] + weight[k], m
    end = max(idx, key=lambda k: best[k])
    chain: set[int] = set()
    cur: int | None = end
    while cur is not None:
        chain.add(cur)
        cur = back[cur]
    return chain


def passage_parser(ayahs: list[str], tareeq: str) -> ParsedText:
    """Parse a multi-ayah passage with a stop after every ayah (the usual way it is recited)."""
    return TajweedParser(tareeq=tareeq, include_sifaat=False).parse(ayahs)
