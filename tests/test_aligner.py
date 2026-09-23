"""CTC Viterbi forced alignment, phonetic mapping and alignment loaders."""

from __future__ import annotations

import json

import numpy as np
import pytest

from app.aligner import (
    AlignmentError,
    HeuristicAligner,
    JsonAlignmentLoader,
    ctc_forced_align,
    phonetic_tokens,
    spans_from_path,
)
from app.quran_text import get_ayah_text
from app.tajweed_rules import parse_text
from tests.synth import concat, silence, vowel


def _emissions(frame_labels: list[int], vocab: int = 5, sharp: float = 8.0) -> np.ndarray:
    logits = np.zeros((len(frame_labels), vocab))
    logits[np.arange(len(frame_labels)), frame_labels] = sharp
    return logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))


def test_ctc_viterbi_recovers_frame_segmentation() -> None:
    blank = 0
    frames = [0, 1, 1, 0, 2, 2, 2, 0, 0, 3, 0]
    path = ctc_forced_align(_emissions(frames), [1, 2, 3], blank)
    assert path.tolist() == [-1, 0, 0, -1, 1, 1, 1, -1, -1, 2, -1]


def test_ctc_handles_repeated_tokens_and_too_short_audio() -> None:
    frames = [1, 0, 1, 2]
    path = ctc_forced_align(_emissions(frames), [1, 1, 2], blank=0)
    assert path.tolist() == [0, -1, 1, 2]
    with pytest.raises(AlignmentError):
        ctc_forced_align(_emissions([1, 1]), [1, 1, 2], blank=0)


def test_spans_are_contiguous_per_unit() -> None:
    path = np.array([-1, 0, 0, -1, 1, 1, 1, -1, -1, 2, -1])
    spans = spans_from_path(path, token_owner=[10, 10, 11], frame_s=0.02)
    assert spans[10].start_s == pytest.approx(0.02) and spans[10].end_s == pytest.approx(0.18)
    assert spans[11].start_s == pytest.approx(0.18) and spans[11].end_s == pytest.approx(0.20)


def test_phonetic_tokens_follow_quranic_corpus_conventions() -> None:
    parsed = parse_text(get_ayah_text(1, 1))
    toks = phonetic_tokens(parsed)
    words = []
    for w in parsed.words:
        words.append("".join("".join(toks.get(i, [])) for i in w.unit_indices))
    assert words == ["bismi", "l-lahi", "l-raḥmāni", "l-raḥīm"]
    fatiha2 = parse_text(get_ayah_text(1, 2))
    t2 = phonetic_tokens(fatiha2)
    assert "".join("".join(t2[i]) for i in fatiha2.words[3].unit_indices if i in t2) == "lʿālamīn"


def test_heuristic_aligner_covers_units_monotonically(make_signal) -> None:  # type: ignore[no-untyped-def]
    parsed = parse_text("مِن شَرِّ مَا خَلَقَ")
    audio = make_signal(concat(silence(0.3), vowel(1.5), silence(0.3), vowel(1.2), silence(0.3)))
    align = HeuristicAligner().align(audio, parsed)
    pron = [u.index for u in parsed.units if u.pronounced]
    assert set(align.units) == set(pron)
    starts = [align.units[i].start_s for i in pron]
    assert starts == sorted(starts)
    assert starts[0] >= 0.25 and align.units[pron[-1]].end_s <= 3.4
    assert align.method == "heuristic"


def test_json_alignment_loader(tmp_path, make_signal) -> None:  # type: ignore[no-untyped-def]
    parsed = parse_text("مِن شَرِّ")
    pron = [u.index for u in parsed.units if u.pronounced]
    path = tmp_path / "align.json"
    path.write_text(json.dumps({"units": [
        {"index": k, "start_ms": 100 * k, "end_ms": 100 * (k + 1)} for k in range(len(pron))
    ]}))
    align = JsonAlignmentLoader(path).align(make_signal(vowel(1.0)), parsed)
    assert align.units[pron[1]].start_s == pytest.approx(0.1)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"unit_index": 0, "start_ms": 200, "end_ms": 100}]))
    with pytest.raises(AlignmentError):
        JsonAlignmentLoader(bad).align(make_signal(vowel(1.0)), parsed)
