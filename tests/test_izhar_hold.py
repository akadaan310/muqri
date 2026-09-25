"""A clear nūn sākin (iẓhār ḥalqī) is held only as long as the consonant: hidden with a ghunnah, it is long."""

from __future__ import annotations

from app.analysis import Unit
from app.submission import IZHAR_HOLD_CEILING, _izhar_hold


def _u(sym: str, counts: float | None) -> Unit:
    return Unit(index=0, symbol=sym, kind="consonant", run_length=1, char_span=(0, 0), frames=(0, 1), onset_s=0.0,
                duration_s=0.1, duration_counts=counts, gop=0.0, best_competitor="", competitor_llr=None,
                confirmed=True)


def test_a_clear_nun_passes_and_a_hidden_one_is_long() -> None:
    from app.submission import to_counts
    clear, hidden = _u("ن", 1.3), _u("ن", 3.6)
    assert _izhar_hold([clear, _u("ء", 0.3)], {})[0] == "pass"
    status, ev = _izhar_hold([hidden, _u("ء", 0.3)], {})
    assert to_counts(3.6) > IZHAR_HOLD_CEILING and status == "long" and ev["hold_ceiling"] == IZHAR_HOLD_CEILING


def test_an_unmeasured_hold_is_not_judged() -> None:
    assert _izhar_hold([_u("ن", None)], {}) == ("pass", {})
