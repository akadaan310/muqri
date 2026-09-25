"""walk_alignment places each ayah where its phonemes are, even when tempo varies across ayahs."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")

from app.submission import walk_alignment  # noqa: E402

VOCAB = {"a": 1, "b": 2, "c": 3, "d": 4}
BLANK = 0


def _posteriors(segments: list[tuple[str, int]]) -> np.ndarray:  # type: ignore[type-arg]
    """Log-posteriors holding each symbol (or blank, '_') for its frame count."""
    rows = []
    for sym, n in segments:
        k = BLANK if sym == "_" else VOCAB[sym]
        row = np.full(5, np.log(0.01))
        row[k] = np.log(0.96)
        rows += [row] * n
    return np.array(rows)


def _refs(*texts: str) -> list[SimpleNamespace]:
    return [SimpleNamespace(phonemes=t) for t in texts]


# Ayah 2 has one phoneme for every four of ayah 1 but is held twelve times as long: the
# proportional prior puts it far from where it is, as a master's al-Qadr 97:4 did.
LAYOUT = [("_", 5), ("a", 3), ("b", 3), ("c", 3), ("d", 3), ("_", 10),
          ("a", 180), ("_", 10), ("b", 3), ("c", 3), ("_", 5)]
TRUE = [(5, 17), (27, 207), (217, 223)]


def test_joint_alignment_follows_the_audio_not_the_prior() -> None:
    spans = walk_alignment(_posteriors(LAYOUT), _refs("abcd", "a", "bc"), VOCAB, BLANK, 0, 5)
    assert spans == TRUE


def test_windowed_walk_misplaces_the_same_passage() -> None:
    """The long-submission path, forced here, is what the joint path replaces for short ones."""
    spans = walk_alignment(_posteriors(LAYOUT), _refs("abcd", "a", "bc"), VOCAB, BLANK, 0, 5,
                           joint_cells=0)
    assert spans != TRUE
