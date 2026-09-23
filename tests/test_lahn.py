"""Lahn-jali GOP: CTC forward likelihood, confusion variants, and substitution verdicts."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from app.aligner import phonetic_tokens
from app.lahn.detector import LahnReference, lahn_diagnostics
from app.lahn.gop import build_reference, ctc_log_likelihood, gop_z, letter_gop, unit_variants
from app.models import AlignedUnit, Alignment, RuleType, Status
from app.tajweed_rules.parser import parse_text


def _collapse(path: tuple[int, ...], blank: int) -> list[int]:
    out: list[int] = []
    prev = None
    for p in path:
        if p != prev and p != blank:
            out.append(p)
        prev = p
    return out


def test_ctc_forward_matches_brute_force() -> None:
    rng = np.random.default_rng(0)
    T, V, blank = 5, 3, 0
    lp = np.log(rng.dirichlet(np.ones(V), size=T))
    for seq in ([1], [1, 2], [1, 1], [2, 1, 2]):
        total = -np.inf
        for path in itertools.product(range(V), repeat=T):
            if _collapse(path, blank) == seq:
                total = np.logaddexp(total, sum(lp[t, p] for t, p in enumerate(path)))
        assert ctc_log_likelihood(lp, seq, blank) == pytest.approx(total, abs=1e-9)


def test_variants_cover_consonant_confusions_and_vowels() -> None:
    v = unit_variants(["ḍ", "ḍ", "a"], "ḍ")
    assert ("consonant", "ḍ", "d", ["d", "d", "a"]) in v
    assert ("vowel", "a", "i", ["ḍ", "ḍ", "i"]) in v
    # sun-letter article prefix is kept; digraph consonants are replaced whole
    assert ("consonant", "th", "s", ["l", "-", "s", "a"]) in unit_variants(["l", "-", "t", "h", "a"], "th")


def _synthetic(parsed, read_as: dict[int, list[str]], vocab: dict[str, int], frames_per_tok: int = 3):  # type: ignore[no-untyped-def]
    """Peaky emissions that 'say' each unit's tokens (optionally substituted) with blanks between."""
    toks = phonetic_tokens(parsed)
    blank = vocab["<pad>"]
    rows: list[int] = []
    spans: dict[int, AlignedUnit] = {}
    for idx, ts in toks.items():
        start = len(rows)
        for t in read_as.get(idx, ts):
            rows += [vocab[t]] * frames_per_tok + [blank]
        spans[idx] = AlignedUnit(idx, start * 0.02, len(rows) * 0.02, 0.9)
    lp = np.full((len(rows), len(vocab)), np.log(0.02 / (len(vocab) - 1)))
    lp[np.arange(len(rows)), rows] = np.log(0.98)
    return lp, Alignment(spans, "ctc:test"), blank


VOCAB = {s: i for i, s in enumerate(["<pad>", "'", "-", "a", "b", "d", "h", "i", "k", "l", "m", "n", "q", "r", "s",
                                     "t", "u", "w", "y", "z", "ā", "ī", "ū", "ʿ", "ḍ", "ḥ", "ṣ", "ṭ", "ẓ"])}


def test_substituted_letter_fails_and_correct_text_passes() -> None:
    parsed = parse_text("وَلَا ٱلضَّآلِّينَ")
    dad = next(u.index for u in parsed.units if u.char == "ض")
    ok_lp, ok_al, blank = _synthetic(parsed, {}, VOCAB)
    toks = phonetic_tokens(parsed)
    bad_lp, bad_al, _ = _synthetic(parsed, {dad: [("d" if t == "ḍ" else t) for t in toks[dad]]}, VOCAB)
    g_ok = {g.unit_index: g for g in letter_gop(ok_lp, 0.02, VOCAB, blank, parsed, ok_al) if g.kind == "consonant"}
    g_bad = {g.unit_index: g for g in letter_gop(bad_lp, 0.02, VOCAB, blank, parsed, bad_al) if g.kind == "consonant"}
    assert g_ok[dad].llr > 5 and g_bad[dad].llr < -5 and g_bad[dad].best_alt == "d"

    ref = LahnReference("test", {"consonant": {"fail": -3.0, "warn": -1.0}, "vowel": {"fail": -3.0, "warn": -1.0}})
    diags = lahn_diagnostics(bad_lp, 0.02, VOCAB, blank, parsed, bad_al, ref)
    fails = [d for d in diags if d.status is Status.FAIL]
    assert [(d.rule_type, d.letter) for d in fails] == [(RuleType.LAHN_LETTER, "ض")]
    assert "د" in fails[0].feedback
    assert all(d.status is Status.PASS for d in lahn_diagnostics(ok_lp, 0.02, VOCAB, blank, parsed, ok_al, ref))


def test_gop_z_shrinks_rare_symbols_to_their_kind() -> None:
    rows = [("consonant", "t", float(x)) for x in np.linspace(4, 12, 50)] + [("consonant", "dh", -7.0)] * 3
    bands = build_reference(rows)
    # three ذ samples are too few to trust: its z sits between its own band and the pooled one
    z = gop_z(bands, "consonant", "dh", -7.0)
    assert z is not None and 0 < z < gop_z(bands, "consonant", "t", -7.0)  # type: ignore[operator]
