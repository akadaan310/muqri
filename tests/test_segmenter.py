"""Semi-global CTC locator for long recordings."""

from __future__ import annotations

import numpy as np

from app.audio import condition
from app.segmenter import _forward_chain, ctc_semiglobal_align, split_on_pauses
from tests.synth import SR, concat, silence, vowel

BLANK = 0


def emissions(tokens: list[int], vocab: int = 6, frames_per_token: int = 3, gap: int = 2) -> np.ndarray:
    """Peaky log-posteriors: each token spikes for a few frames, blanks in between."""
    rows = []
    for tok in tokens:
        rows += [BLANK] * gap + [tok] * frames_per_token
    rows += [BLANK] * gap
    lp = np.full((len(rows), vocab), np.log(0.01))
    for t, k in enumerate(rows):
        lp[t, k] = np.log(0.95)
    return lp - np.log(np.exp(lp).sum(axis=1, keepdims=True))


def test_semiglobal_finds_the_spoken_sub_span() -> None:
    targets = [1, 2, 3, 4, 5, 1, 2]
    lp = emissions([3, 4, 5])  # the middle of the passage
    every = np.ones(len(targets), dtype=bool)
    first, last, score = ctc_semiglobal_align(lp, targets, BLANK, every, every)
    assert (first, last) == (2, 4)
    assert score > -0.5


def test_semiglobal_respects_word_boundaries() -> None:
    targets = [1, 2, 3, 4, 5]
    lp = emissions([2, 3, 4])
    starts = np.array([True, False, True, False, False])  # words start at tokens 0 and 2
    ends = np.array([False, True, False, False, True])
    first, last, _ = ctc_semiglobal_align(lp, targets, BLANK, starts, ends)
    assert starts[first] and ends[last]


def test_forward_chain_drops_out_of_order_hits() -> None:
    hits = [(0, 5, -0.5), (6, 12, -0.5), (40, 44, -0.9), (13, 20, -0.5), None, (21, 30, -0.5)]
    assert _forward_chain(hits) == {0, 1, 3, 5}


def test_split_on_pauses() -> None:
    x = concat(vowel(1.5), silence(0.6), vowel(1.2, seed=2), silence(0.6), vowel(1.0, seed=3))
    chunks = split_on_pauses(condition(x, SR, denoise="never"))
    assert len(chunks) == 3
    assert abs(chunks[1].start_s - 2.1) < 0.2
