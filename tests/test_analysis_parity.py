"""The numpy serving path must agree exactly with the Julia reference.

The research numerics were validated in Julia against ~2 M measured decisions. `app/analysis.py`
re-implements the request path in numpy so serving is a library call rather than a subprocess, which
is only safe while the two produce the same numbers. This pins that.

Skips when the T300 dump is absent (it is 1.8 GB and gitignored), so the suite still runs on a clean
checkout — but on the research box it is a hard parity gate.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from app.analysis import analyse_clip, ctc_viterbi  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DUMP = ROOT / "research_agency_lab/experiments/qaari_keys/modal_T300"
CLIP = "everyayah:Husary_Muallim_128kbps/023041"
REFERENCE = ROOT / "tests/data/letter_report_reference.json"

pytestmark = pytest.mark.skipif(not (DUMP / "index.jsonl").is_file(),
                                reason="T300 dump not present (gitignored, 1.8 GB)")


def load_clip():  # type: ignore[no-untyped-def]
    lay = json.loads((DUMP / "layout.json").read_text())
    ph = next(l for l in lay["levels"] if l["level"] == "phonemes")
    vocab = {t: i for i, t in enumerate(ph["vocab"]) if len(t) == 1}
    blocks = {l["level"]: (l["first"], l["width"], l["vocab"])
              for l in lay["levels"] if l["level"] != "phonemes"}
    rec = next(r for r in map(json.loads, (DUMP / "index.jsonl").open()) if r.get("id") == CLIP)
    raw = np.fromfile(DUMP / rec["file"], dtype="<f4").reshape(rec["frames"], lay["columns"])
    sif = next((x["levels"] for x in map(json.loads, (DUMP / "sifat.jsonl").open())
                if x["id"] == CLIP), None) if (DUMP / "sifat.jsonl").is_file() else None
    return raw, rec, vocab, blocks, ph, lay["blank"], sif


def test_viterbi_is_monotonic_and_covers_every_symbol() -> None:
    raw, rec, vocab, blocks, ph, blank, _sif = load_clip()
    lp = raw[:, ph["first"]:ph["first"] + ph["width"]]
    seq = [vocab[c] for c in rec["ref_ph"]]
    _score, first, last = ctc_viterbi(lp, seq, blank)
    assert len(first) == len(seq)
    assert all(f >= 1 for f in first), "every symbol must be aligned to a frame"
    assert all(l >= f for f, l in zip(first, last)), "a symbol cannot end before it starts"
    assert first == sorted(first), "alignment must advance monotonically"


def test_sifat_are_scored_on_consonants_only() -> None:
    raw, rec, vocab, blocks, ph, blank, sif = load_clip()
    if sif is None:
        pytest.skip("sifat.jsonl not generated")
    units = analyse_clip(raw, rec["ref_ph"], vocab, blank, ph["first"], ph["width"], blocks, sif)
    for u in units:
        if u.kind in {"harakah", "madd"}:
            assert not u.sifat, f"{u.symbol} is a {u.kind} and carries no sifah"
        elif u.kind == "consonant":
            assert len(u.sifat) == 10, f"{u.symbol} should carry all ten judged attributes"


@pytest.mark.skipif(not REFERENCE.is_file(), reason="Julia reference not captured")
def test_matches_the_julia_reference_exactly() -> None:
    """GOP, frames, durations and identity must be identical, not merely close.

    A mismatch here means the serving path has drifted from the validated numerics. The one bug this
    caught: the haraka span covered only the consonant instead of consonant+vowel, which left frames
    and seconds correct but inflated every count by 3.5x.
    """
    raw, rec, vocab, blocks, ph, blank, sif = load_clip()
    got = analyse_clip(raw, rec["ref_ph"], vocab, blank, ph["first"], ph["width"], blocks, sif)
    want = json.loads(REFERENCE.read_text())["units"]
    assert len(got) == len(want)

    from app.analysis import FRAME_S  # noqa: PLC0415

    suppressed = 0
    for i, (p, j) in enumerate(zip(got, want)):
        # the raw numerics must be identical — these are the validated quantities
        assert p.symbol == j["symbol"]
        assert list(p.frames) == j["frames"]
        assert p.best_competitor == j["identity"]["best_competitor"]
        assert abs(p.gop - j["identity"]["gop"]) < 1e-6
        assert abs(p.duration_s - j["duration_s"]) < 1e-6

        # The serving path adds two measurement guards Julia (the research reference) does not: the
        # last unit of a clip absorbs trailing silence, and any count beyond every tajweed
        # requirement is a measurement failure rather than a very long madd. Both suppress a value —
        # they never change one — so where Python reports a count it must match Julia exactly.
        if p.duration_counts is None:
            suppressed += 1
            assert i == len(got) - 1 or (j["duration_counts"] or 0) > 12.0, (
                f"unit {i} suppressed for no reason (julia={j['duration_counts']})")
        else:
            assert abs(p.duration_counts - (j["duration_counts"] or 0)) < 1e-6
    assert suppressed <= 3, f"{suppressed} units suppressed; the guards should be rare"


def test_centroid_timing_is_continuous_and_leaves_everything_else_alone() -> None:
    """`timing="centroid"` must break the 40 ms quantisation and change nothing but timing.

    Under Viterbi every onset is a whole frame, so a one-frame vowel and a 1.4-frame vowel read the
    same. The centroid is continuous. Identity, sifat and frames are validated on the Viterbi path
    and must be identical in both modes.
    """
    raw, rec, vocab, blocks, ph, blank, sif = load_clip()
    vit = analyse_clip(raw, rec["ref_ph"], vocab, blank, ph["first"], ph["width"], blocks, sif)
    cen = analyse_clip(raw, rec["ref_ph"], vocab, blank, ph["first"], ph["width"], blocks, sif,
                       timing="centroid")
    assert len(vit) == len(cen)
    for v, c in zip(vit, cen):
        assert (v.symbol, v.frames, v.best_competitor, v.gop, v.sifat) == \
               (c.symbol, c.frames, c.best_competitor, c.gop, c.sifat)
    from app.analysis import FRAME_S  # noqa: PLC0415

    body = cen[:-1]                        # the tail runs to a Viterbi end, not a centroid
    assert all(c.duration_s > 0 for c in body), "centroid onsets must advance"
    off_grid = [c for c in body if abs(c.duration_s / FRAME_S - round(c.duration_s / FRAME_S)) > 0.02]
    assert len(off_grid) > 0.8 * len(body), "centroid durations should not sit on the 40 ms grid"
    assert all(abs(c.onset_s - v.onset_s) < 3 * FRAME_S for v, c in zip(vit, cen)), \
        "a centroid should sit within a few frames of the Viterbi onset"


def test_unknown_timing_mode_is_refused() -> None:
    raw, rec, vocab, blocks, ph, blank, sif = load_clip()
    with pytest.raises(ValueError):
        analyse_clip(raw, rec["ref_ph"], vocab, blank, ph["first"], ph["width"], blocks, sif,
                     timing="cubic")
