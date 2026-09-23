"""Tajweed Ahkaam: rule derivation for every family and the acoustic validators on synthetic audio."""

from __future__ import annotations

import numpy as np
import pytest

from app.models import RuleInstance, RuleType, Status, Tareeq
from app.quran_text import get_ayah_text
from app.scoring import VALIDATORS, TajweedScorer
from app.tajweed_rules import TajweedParser, parse_text
from app.tajweed_rules.base import Pause, band_status
from app.tajweed_rules.mudood_engine import TOLERANCE, validate_madd
from app.tajweed_rules.noon_sakinah import (
    validate_ghunnah_mushaddadah,
    validate_idgham_no_ghunnah,
    validate_izhar_halqi,
)
from app.tajweed_rules.qalqalah_engine import count_releases, detect_release_burst, validate_qalqalah
from app.tajweed_rules.sakt_wasl import validate_hamzat_wasl, validate_sakt
from tests.synth import SR, burst, concat, nasal_murmur, silence, voice_bar, vowel


def rule_types(text: str, **kw) -> set[str]:  # type: ignore[no-untyped-def]
    return {r.rule_type.value for r in parse_text(text, include_sifaat=False, **kw).rules}


# --------------------------------------------------------------------------- coverage
def test_every_rule_type_has_a_validator() -> None:
    assert set(RuleType) == set(VALIDATORS)


def test_noon_sakinah_family() -> None:
    assert "izhar_halqi" in rule_types("مَنْ ءَامَنَ")
    assert "idgham_ghunnah" in rule_types("مَن يَقُولُ")
    assert "idgham_no_ghunnah" in rule_types("مِن رَّبِّهِمْ")
    assert "iqlab" in rule_types("مِنۢ بَعْدِ")
    ikhfa = [r for r in parse_text("مِن قَبْلِ وَمَن تَابَ", include_sifaat=False).rules if r.rule_type is RuleType.IKHFA]
    assert {r.detail.split("; ")[-1] for r in ikhfa} == {"heavy", "light"}


def test_meem_sakinah_family() -> None:
    assert "ikhfa_shafawi" in rule_types("تَرْمِيهِم بِحِجَارَةٍ")
    assert "idgham_shafawi" in rule_types("لَهُم مَّا")
    izhar = [r for r in parse_text("عَلَيْهِمْ وَلَا", include_sifaat=False).rules if r.rule_type is RuleType.IZHAR_SHAFAWI]
    assert izhar and izhar[0].detail == "before و/ف"


def test_all_madd_types_and_tareeq() -> None:
    types = set()
    for ref in [(1, 7), (3, 58), (106, 4), (112, 4), (19, 1), (78, 21), (2, 255)]:
        types |= rule_types(get_ayah_text(*ref))
    assert {"madd_tabii", "madd_muttasil", "madd_munfasil", "madd_lazim", "madd_arid_lissukun", "madd_leen",
            "madd_badal", "madd_iwad", "madd_silah_sughra"} <= types
    kubra = [r for r in parse_text("إِنَّهُۥٓ إِلَيْهِ", include_sifaat=False).rules
             if r.rule_type is RuleType.MADD_SILAH_KUBRA]
    assert kubra and kubra[0].expected_harakat == (4, 5)
    shat = [r for r in parse_text(get_ayah_text(108, 1)).rules if r.rule_type is RuleType.MADD_MUNFASIL]
    tayy = [r for r in parse_text(get_ayah_text(108, 1), tareeq=Tareeq.TAYYIBAH).rules
            if r.rule_type is RuleType.MADD_MUNFASIL]
    assert shat[0].expected_harakat == (4, 5) and tayy[0].expected_harakat == (2, 2)


def test_muqattaat_are_read_as_letter_names() -> None:
    rules = parse_text(get_ayah_text(19, 1), include_sifaat=False).rules
    lazim = [r for r in rules if r.rule_type is RuleType.MADD_LAZIM]
    assert len(lazim) == 2 and all("harfi" in r.detail for r in lazim)  # kāf, ṣād
    assert any(r.rule_type is RuleType.MADD_LEEN and r.expected_harakat == (4, 6) for r in rules)  # ʿayn
    assert any(r.rule_type is RuleType.IKHFA for r in rules)  # ʿayn's nūn before ṣād
    # alif-lām-mīm: lām and mīm are both 6 counts, lām's mīm merges into mīm (idgham shafawi).
    alm = parse_text("الٓمٓ", include_sifaat=False).rules
    assert [r.rule_type for r in alm].count(RuleType.MADD_LAZIM) == 2
    assert any(r.rule_type is RuleType.IDGHAM_SHAFAWI for r in alm)


def test_qalqalah_levels() -> None:
    def levels(text: str) -> set[str]:
        return {r.detail for r in parse_text(text, include_sifaat=False).rules if r.rule_type is RuleType.QALQALAH}

    assert levels(get_ayah_text(111, 1)) == {"akbar"}
    assert "kubra" in levels(get_ayah_text(113, 1))
    assert "sughra" in levels(get_ayah_text(103, 3))


def test_idghaam_classes() -> None:
    assert {"idgham_mithlayn"} <= rule_types("ٱضْرِب بِّعَصَاكَ")
    assert {"idgham_mutajanisayn"} <= rule_types("قَد تَّبَيَّنَ")
    assert {"idgham_mutaqaribayn"} <= rule_types("أَلَمْ نَخْلُقكُّم")
    naqis = [r for r in parse_text("أَحَطتُ بِمَا", include_sifaat=False).rules
             if r.rule_type is RuleType.IDGHAM_MUTAJANISAYN]
    assert naqis and naqis[0].detail == "naqis"
    # The ط of an idghaam naqis is not given a Qalqalah.
    assert not any(r.rule_type is RuleType.QALQALAH and r.word == "أَحَطتُ"
                   for r in parse_text("أَحَطتُ بِمَا", include_sifaat=False).rules)
    # The article's lam before a sun letter is not a class idghaam.
    assert "idgham_mutaqaribayn" not in rule_types("ٱلرَّحْمَٰنِ")


def test_raa_conditions() -> None:
    def raa(text: str) -> list[str]:
        return [r.rule_type.value for r in parse_text(text, include_sifaat=False).rules if r.letter == "ر"]

    assert raa("لَبِٱلْمِرْصَادِ") == ["tafkheem"]  # sakin after kasrah, before isti'la
    assert raa("فِرْقٍ كَٱلطَّوْدِ") == ["jawaz_wajhayn"]  # isti'la with kasrah
    assert raa("ٱرْجِعِىٓ") == ["tafkheem"]  # temporary ('aaridha) kasrah of hamzat al-wasl
    assert raa("وَٱلذِّكْرِ") == ["tarqeeq"]
    assert raa("قَدِيرٌ") == ["tarqeeq"]  # at a stop after yaa


def test_sakt_and_hamzat_wasl() -> None:
    rules = parse_text(get_ayah_text(75, 27), include_sifaat=False).rules
    assert [r.word for r in rules if r.rule_type is RuleType.SAKT] == ["مَنْ"]
    # Sakt blocks the idghaam of the noon into raa.
    assert not any(r.rule_type is RuleType.IDGHAM_NO_GHUNNAH for r in rules)
    assert any(r.rule_type is RuleType.HAMZAT_WASL for r in parse_text("بِسْمِ ٱللَّهِ", include_sifaat=False).rules)


def test_dynamic_stop_applies_waqf_mid_ayah() -> None:
    text = get_ayah_text(2, 255)
    parser = TajweedParser(include_sifaat=False)
    plain = parser.parse(text)
    stopped = parser.parse(text, stops={4})  # stop after ٱلْقَيُّومُ
    assert len(stopped.rules) != len(plain.rules)
    assert stopped.words[4].stop_after


def test_whole_quran_parses() -> None:
    from app.quran_text import get_full_quran

    try:
        quran = get_full_quran()
    except Exception:  # noqa: BLE001 - network-less environments
        pytest.skip("full Qur'an text unavailable offline")
    parser = TajweedParser()
    for text in quran.values():
        parser.parse(text)


# --------------------------------------------------------------------------- validators
def _uniform(parsed, haraka: float, overrides: dict[int, float] | None = None) -> dict[int, tuple[float, float]]:  # type: ignore[no-untyped-def]
    t, spans = 0.1, {}
    for u in parsed.units:
        if u.pronounced:
            d = (overrides or {}).get(u.index, haraka)
            spans[u.index] = (t, t + d)
            t += d
    return spans


def test_band_status_uses_absolute_tolerance() -> None:
    assert band_status(2.2, 2, 2, 0.25)[0] is Status.PASS
    assert band_status(2.4, 2, 2, 0.25)[0] is Status.WARNING
    assert band_status(2.6, 2, 2, 0.25)[0] is Status.FAIL


def test_madd_is_proportional_to_local_tempo(eval_ctx) -> None:  # type: ignore[no-untyped-def]
    """A 4-count Madd scores the same in Hadr (100 ms) and Tahqeeq (240 ms)."""
    parsed = parse_text("إِذَا جَآءَ نَصْرُ", include_sifaat=False)
    rule = next(r for r in parsed.rules if r.rule_type is RuleType.MADD_MUTTASIL)
    results = []
    for haraka in (0.1, 0.24):
        spans = _uniform(parsed, haraka, {rule.unit_indices[1]: 3 * haraka})
        ev = eval_ctx(vowel(2.0), spans, parsed, haraka_ms=haraka * 1000)
        results.append(validate_madd(rule, ev))
    assert [d.measured_harakat for d in results] == pytest.approx([4.0, 4.0])
    assert all(d.status is Status.PASS for d in results)


def test_madd_tolerances_follow_spec(eval_ctx) -> None:  # type: ignore[no-untyped-def]
    assert TOLERANCE[RuleType.MADD_TABII] == 0.25 and TOLERANCE[RuleType.MADD_LAZIM] == 0.30
    parsed = parse_text("وَلَا ٱلضَّآلِّينَ", include_sifaat=False)
    lazim = next(r for r in parsed.rules if r.rule_type is RuleType.MADD_LAZIM)
    spans = _uniform(parsed, 0.2, {lazim.unit_indices[1]: 5 * 0.2})
    assert validate_madd(lazim, eval_ctx(vowel(3.0), spans, parsed)).status is Status.PASS  # 6 counts
    spans = _uniform(parsed, 0.2, {lazim.unit_indices[1]: 3 * 0.2})
    assert validate_madd(lazim, eval_ctx(vowel(3.0), spans, parsed)).status is Status.FAIL  # 4 counts


def test_waqf_dharoori_breath_is_a_valid_pause(eval_ctx) -> None:  # type: ignore[no-untyped-def]
    parsed = parse_text("إِذَا جَآءَ نَصْرُ", include_sifaat=False)
    rule = next(r for r in parsed.rules if r.rule_type is RuleType.MADD_MUTTASIL)
    spans = _uniform(parsed, 0.2)  # muttasil held only 2 counts
    end = spans[rule.unit_indices[-1]][1]
    breath = [Pause(end, end + 0.4, breath=True, after_unit=rule.unit_indices[-1])]
    studio = validate_madd(rule, eval_ctx(vowel(2.0), spans, parsed, pauses=breath))
    live = validate_madd(rule, eval_ctx(vowel(2.0), spans, parsed, mode="taraweeh_adapted", pauses=breath))
    assert studio.status is Status.FAIL
    assert live.status is Status.VALID_NECESSARY_PAUSE


def _nasal_setup(eval_ctx, nasal_s: float, nasal: bool = True):  # type: ignore[no-untyped-def]
    parsed = parse_text("إِنَّ", include_sifaat=False)
    rule = next(r for r in parsed.rules if r.rule_type is RuleType.GHUNNAH)
    mid = nasal_murmur(nasal_s) if nasal else vowel(nasal_s, seed=3)
    samples = concat(vowel(0.2), vowel(0.2, seed=1), mid, vowel(0.2, seed=2))
    spans = {0: (0.0, 0.2), rule.unit_indices[0]: (0.4, 0.4 + nasal_s)}
    ev = eval_ctx(samples, spans, parsed)
    ev.__dict__["oral_ner_reference"] = float(np.median([
        __import__("app.sifaat.formants", fromlist=["x"]).nasal_energy_ratio(vowel(0.2, seed=k), SR) for k in (0, 1)]))
    return rule, ev


def test_ghunnah_duration_and_nasality(eval_ctx) -> None:  # type: ignore[no-untyped-def]
    rule, ev = _nasal_setup(eval_ctx, 0.44)
    good = validate_ghunnah_mushaddadah(rule, ev)
    assert good.status is Status.PASS and good.measured_harakat == pytest.approx(2.2, abs=0.1)
    rule, ev = _nasal_setup(eval_ctx, 0.3)
    assert validate_ghunnah_mushaddadah(rule, ev).status in (Status.WARNING, Status.FAIL)
    rule, ev = _nasal_setup(eval_ctx, 0.44, nasal=False)
    assert "Weak nasalization" in validate_ghunnah_mushaddadah(rule, ev).feedback


def test_izhar_halqi_rejects_held_noon(eval_ctx) -> None:  # type: ignore[no-untyped-def]
    parsed = parse_text("مَنْ ءَامَنَ", include_sifaat=False)
    rule = next(r for r in parsed.rules if r.rule_type is RuleType.IZHAR_HALQI)
    noon = rule.unit_indices[0]
    for hold, expected in ((0.2, Status.PASS), (0.5, Status.FAIL)):
        spans = _uniform(parsed, 0.2, {noon: hold})
        samples = concat(silence(0.1), *[nasal_murmur(e - s) if i == noon else vowel(e - s, seed=i)
                                         for i, (s, e) in spans.items()])
        assert validate_izhar_halqi(rule, eval_ctx(samples, spans, parsed)).status is expected


def test_idgham_without_ghunnah_detects_retained_nasal(eval_ctx) -> None:  # type: ignore[no-untyped-def]
    parsed = parse_text("مِن رَّبِّهِمْ", include_sifaat=False)
    rule = next(r for r in parsed.rules if r.rule_type is RuleType.IDGHAM_NO_GHUNNAH)
    target = rule.unit_indices[1]
    for nasal, ok in ((False, True), (True, False)):
        spans = _uniform(parsed, 0.2)
        samples = concat(silence(0.1), *[(nasal_murmur if nasal and i == target else vowel)(e - s)
                                         for i, (s, e) in spans.items()])
        ev = eval_ctx(samples, spans, parsed)
        ev.__dict__["oral_ner_reference"] = 0.0
        status = validate_idgham_no_ghunnah(rule, ev).status
        assert (status is Status.PASS) is ok


def test_qalqalah_burst_and_akbar_hold(eval_ctx) -> None:  # type: ignore[no-untyped-def]
    res = detect_release_burst(concat(vowel(0.15), silence(0.05), burst(), vowel(0.05, (500, 1500, 2500), amp=0.2),
                                      silence(0.1)), SR, letter_start_s=0.15, letter_end_s=0.2)
    assert res.present and res.rise_db >= 12
    assert not detect_release_burst(concat(vowel(0.15), silence(0.2)), SR, letter_start_s=0.15).present
    faint = concat(vowel(0.15), voice_bar(0.05, amp=0.1), burst(amp=0.03), vowel(0.05, (500, 1500, 2500), amp=0.1),
                   silence(0.1))
    assert detect_release_burst(faint, SR, letter_start_s=0.15, letter_end_s=0.21).present
    parsed = parse_text(get_ayah_text(111, 1), include_sifaat=False)
    rule = next(r for r in parsed.rules if r.rule_type is RuleType.QALQALAH)
    samples = concat(vowel(0.3), silence(0.35), burst(), vowel(0.06, (500, 1500, 2500), amp=0.25), silence(0.15))
    ev = eval_ctx(samples, {rule.unit_indices[0]: (0.3, 0.72)}, parsed, haraka_ms=200)
    diag = validate_qalqalah(rule, ev)
    assert diag.status is Status.PASS and diag.metrics["shiddah_hold_harakat"] >= 1.5


def test_single_vs_double_release() -> None:
    one = concat(vowel(0.15), silence(0.08), burst(), vowel(0.1, amp=0.3))
    two = concat(vowel(0.15), silence(0.06), burst(), vowel(0.08, amp=0.3), silence(0.06), burst(),
                 vowel(0.1, amp=0.3))
    assert count_releases(one, SR) == 1
    assert count_releases(two, SR) == 2


def test_sakt_requires_breathless_short_silence(eval_ctx) -> None:  # type: ignore[no-untyped-def]
    parsed = parse_text(get_ayah_text(75, 27), include_sifaat=False)
    rule = next(r for r in parsed.rules if r.rule_type is RuleType.SAKT)
    a, b = rule.unit_indices

    def run(gap: np.ndarray) -> Status:
        samples = concat(vowel(0.3), gap, vowel(0.3, seed=2))
        ev = eval_ctx(samples, {a: (0.1, 0.3), b: (0.3 + len(gap) / SR, 0.6 + len(gap) / SR)}, parsed)
        return validate_sakt(rule, ev).status

    rng = np.random.default_rng(0)
    assert run(silence(0.3)) is Status.PASS
    assert run(silence(0.03)) is Status.FAIL  # no sakt
    breath = (0.05 * rng.standard_normal(int(0.3 * SR))).astype(np.float32)
    assert run(breath) is Status.FAIL  # a breath was taken


def test_hamzat_wasl_glottal_insertion(eval_ctx) -> None:  # type: ignore[no-untyped-def]
    parsed = parse_text("بِسْمِ ٱللَّهِ", include_sifaat=False)
    rule = next(r for r in parsed.rules if r.rule_type is RuleType.HAMZAT_WASL)
    prev, nxt = rule.unit_indices
    smooth = concat(vowel(0.3), vowel(0.3, seed=2))
    ev = eval_ctx(smooth, {prev: (0.0, 0.3), nxt: (0.3, 0.6)}, parsed)
    assert validate_hamzat_wasl(rule, ev).status is Status.PASS
    glottal = concat(vowel(0.3), silence(0.06), vowel(0.3, seed=2))
    ev = eval_ctx(glottal, {prev: (0.0, 0.3), nxt: (0.36, 0.66)}, parsed)
    assert validate_hamzat_wasl(rule, ev).status is Status.FAIL


def test_scorer_separates_perfection_and_sifaat() -> None:
    from app.models import RuleDiagnostic

    diags = [RuleDiagnostic(RuleType.MADD_TABII, "w", 0, 1, Status.PASS, "", score=1.0),
             RuleDiagnostic(RuleType.HAMS, "w", 0, 1, Status.FAIL, "", score=0.2),
             RuleDiagnostic(RuleType.MADD_ARID, "w", 0, 1, Status.VALID_NECESSARY_PAUSE, "", score=1.0)]
    summary = TajweedScorer.summarize(diags)
    assert summary.overall == pytest.approx(100.0)
    assert summary.sifaat == pytest.approx(20.0)
    assert summary.status_counts["VALID_NECESSARY_PAUSE"] == 1
    assert isinstance(RuleInstance, type)
