"""Formant-domain measurements: Isti'la/Istifal (Tafkheem/Tarqeeq), Itbaq, and nasal resonance.

**Weight (Isti'la / Tafkheem).** Velarisation/pharyngealisation retracts the tongue root, which
lowers the second formant of the adjacent vowel (F2 ≈ 1000–1400 Hz after heavy letters versus
> 1700 Hz after light ones for an open /a/) and raises F1, collapsing the **F2 − F1 distance**.
Absolute formants depend on the vocal tract, so each rule is scored against a per-vowel **light
reference** built from the same reciter's coronal/dorsal letters (excluding gutturals, labials,
raa, lam and letters next to a heavy one)::

    H = 1 − (F2 − F1) / (F2_ref − F1_ref)

Absolute thresholds (F2 < 1300–1400 Hz heavy, > 1700 Hz light) are used when no reference exists.

**Itbaq.** The tongue body presses against the palate for ص ض ط ظ: F2 falls towards F1 while F3
stays, widening F3 − F2, and energy above 3.5 kHz is attenuated relative to light letters.

**Nasality (Fn).** A nasal murmur has a strong nasal formant (Fn ≈ 250 Hz) and an anti-formant
around 750–1100 Hz. The Nasal Energy Ratio ``NER = 10·log10(E[150–400] / E[750–1100])`` is
compared with the reciter's own open oral vowels (the nasal contrast).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from app.acoustic.features import AcousticContext, FormantEstimate, band_energy_ratio_db, band_power_db
from app.models import Alignment, Status, Vowel
from app.tajweed_rules.parser import HEAVY_LETTERS, ParsedText

HEAVY_F2_MAX_HZ = 1400.0
LIGHT_F2_MIN_HZ = 1700.0
HEAVY_INDEX_PASS = 0.15
HEAVY_INDEX_WARN = 0.07
LIGHT_INDEX_PASS = 0.10
LIGHT_INDEX_FAIL = 0.20
VOWEL_START, VOWEL_END = 0.15, 0.75  # fraction of a CV unit used as the vowel core
MIN_REFERENCE_TOKENS = 2

NASAL_BAND = (150.0, 400.0)
ZERO_BAND = (750.0, 1100.0)
NER_STRONG_DB = 8.0
NER_WEAK_DB = 3.0

# Gutturals, labials, raa, lam and hamza have their own F2 effects and make poor references.
_NON_REFERENCE = HEAVY_LETTERS | frozenset("رلعحهءأإؤئبمفو")
_F2_LOWERING = HEAVY_LETTERS | frozenset("رعح")


# --------------------------------------------------------------------------- windows & references
def vowel_window(unit_indices: list[int], alignment: Alignment) -> tuple[float, float] | None:
    """Core of the vowel carried by the first unit (and any madd letter after it).

    Aligned CV units start at the consonant release, so the vowel fills most of the unit; its
    last quarter is coarticulated with the following letter and is excluded.
    """
    present = [alignment.units[i] for i in unit_indices if i in alignment.units]
    if not present or present[0].unit_index != unit_indices[0]:
        return None
    first, last = present[0], present[-1]
    start = first.start_s + VOWEL_START * first.duration_s
    end = last.end_s - (1.0 - VOWEL_END) * last.duration_s
    return (start, end) if end > start else None


def _near_heavy(parsed: ParsedText, index: int) -> bool:
    """True if a heavy, pharyngeal or raa letter is adjacent to ``index`` (their F2 lowering spreads)."""
    for j in (index - 1, index + 1, index + 2):
        if 0 <= j < len(parsed.units):
            other = parsed.units[j]
            if other.pronounced and other.char in _F2_LOWERING:
                return True
    return False


@dataclass(slots=True)
class WeightReference:
    f1_by_vowel: dict[Vowel, float] = field(default_factory=dict)
    f2_by_vowel: dict[Vowel, float] = field(default_factory=dict)
    f3_by_vowel: dict[Vowel, float] = field(default_factory=dict)
    hf_db_by_vowel: dict[Vowel, float] = field(default_factory=dict)
    counts: dict[Vowel, int] = field(default_factory=dict)

    def distance(self, vowel: Vowel | None) -> float | None:
        if vowel is None or vowel not in self.f2_by_vowel:
            return None
        return self.f2_by_vowel[vowel] - self.f1_by_vowel[vowel]


def build_reference(parsed: ParsedText, alignment: Alignment, ctx: AcousticContext) -> WeightReference:
    """Median F1/F2/F3 and >3.5 kHz level per short vowel over light, non-coarticulated letters."""
    samples: dict[Vowel, list[tuple[float, float, float, float]]] = {}
    for u in parsed.units:
        if not u.pronounced or u.vowel is None or u.madd_letter or u.char in _NON_REFERENCE:
            continue
        if _near_heavy(parsed, u.index):
            continue
        win = vowel_window([u.index], alignment)
        if win is None:
            continue
        fm = ctx.formants(*win)
        if fm.valid:
            hf = high_band_level_db(ctx.audio.segment(*win), ctx.sr)
            samples.setdefault(u.vowel, []).append((fm.f1, fm.f2, fm.f3, hf))
    ref = WeightReference()
    for vowel, vals in samples.items():
        if len(vals) >= MIN_REFERENCE_TOKENS:
            arr = np.asarray(vals, dtype=np.float64)
            ref.f1_by_vowel[vowel] = float(np.nanmedian(arr[:, 0]))
            ref.f2_by_vowel[vowel] = float(np.nanmedian(arr[:, 1]))
            ref.f3_by_vowel[vowel] = float(np.nanmedian(arr[:, 2]))
            ref.hf_db_by_vowel[vowel] = float(np.nanmedian(arr[:, 3]))
            ref.counts[vowel] = len(vals)
    return ref


# --------------------------------------------------------------------------- weight judgement
def judge_weight(expect_heavy: bool, fm: FormantEstimate, ref_distance: float | None,
                 vowel: Vowel | None) -> tuple[Status, float, float | None]:
    """Return (status, score, heaviness_index) for an expected heavy or light letter.

    A relative reference is used when available; otherwise absolute F2 thresholds apply to
    fathah only (other vowels shift F2 too much to judge without a reference).
    """
    if ref_distance is not None and ref_distance > 100.0:
        h = 1.0 - (fm.f2 - fm.f1) / ref_distance
        if expect_heavy:
            if h >= HEAVY_INDEX_PASS or fm.f2 < 1300.0 and vowel == Vowel.FATHA:
                return Status.PASS, 1.0, h
            if h >= HEAVY_INDEX_WARN:
                return Status.WARNING, 0.6, h
            return Status.FAIL, 0.2, h
        if h < LIGHT_INDEX_PASS:
            return Status.PASS, 1.0, h
        if h < LIGHT_INDEX_FAIL:
            return Status.WARNING, 0.6, h
        return Status.FAIL, 0.2, h
    if vowel == Vowel.FATHA:
        if expect_heavy:
            if fm.f2 <= HEAVY_F2_MAX_HZ:
                return Status.PASS, 1.0, None
            if fm.f2 <= LIGHT_F2_MIN_HZ:
                return Status.WARNING, 0.6, None
            return Status.FAIL, 0.2, None
        if fm.f2 >= LIGHT_F2_MIN_HZ:
            return Status.PASS, 1.0, None
        if fm.f2 > HEAVY_F2_MAX_HZ:
            return Status.WARNING, 0.6, None
        return Status.FAIL, 0.2, None
    return Status.SKIPPED, 0.0, None


def unit_vowel(parsed: ParsedText, unit_indices: list[int]) -> Vowel | None:
    unit = parsed.units[unit_indices[0]]
    if unit.vowel is not None:
        return unit.vowel
    if unit.orig_vowel is not None:
        return unit.orig_vowel
    if len(unit_indices) > 1:
        madd = parsed.units[unit_indices[1]]
        return {"ا": Vowel.FATHA, "و": Vowel.DAMMA, "ي": Vowel.KASRA}.get(madd.char)
    return None


# --------------------------------------------------------------------------- itbaq
def high_band_level_db(x: npt.NDArray[np.floating], sr: int, cutoff: float = 3500.0) -> float:
    """Energy above ``cutoff`` relative to the 300–3500 Hz band (dB)."""
    if len(x) < 64:
        return float("nan")
    return band_power_db(x, sr, cutoff, sr / 2) - band_power_db(x, sr, 300.0, cutoff)


@dataclass(slots=True)
class ItbaqMeasurement:
    f3_minus_f2: float
    ref_f3_minus_f2: float | None
    hf_attenuation_db: float | None  # positive: quieter above 3.5 kHz than light letters

    @property
    def convergence(self) -> float | None:
        """Relative F2→F1 approach expressed as F3−F2 widening versus the light reference."""
        if self.ref_f3_minus_f2 is None or self.ref_f3_minus_f2 <= 0:
            return None
        return self.f3_minus_f2 / self.ref_f3_minus_f2 - 1.0


def measure_itbaq(fm: FormantEstimate, seg: npt.NDArray[np.floating], sr: int, ref: WeightReference,
                  vowel: Vowel | None) -> ItbaqMeasurement:
    ref_gap = None
    if vowel is not None and vowel in ref.f3_by_vowel and np.isfinite(ref.f3_by_vowel[vowel]):
        ref_gap = ref.f3_by_vowel[vowel] - ref.f2_by_vowel[vowel]
    hf = high_band_level_db(seg, sr)
    att = None
    if vowel is not None and vowel in ref.hf_db_by_vowel and np.isfinite(hf):
        att = ref.hf_db_by_vowel[vowel] - hf
    return ItbaqMeasurement(fm.f3 - fm.f2, ref_gap, att)


# --------------------------------------------------------------------------- nasality
@dataclass(slots=True)
class NasalMeasurement:
    ner_db: float
    low_freq_ratio: float
    f1_hz: float
    f2_hz: float
    b1_hz: float


def nasal_energy_ratio(x: npt.NDArray[np.floating], sr: int) -> float:
    return band_energy_ratio_db(x, sr, NASAL_BAND, ZERO_BAND)


def measure_nasality(ctx: AcousticContext, start_s: float, end_s: float) -> NasalMeasurement:
    seg = ctx.audio.segment(start_s, end_s)
    ner = nasal_energy_ratio(seg, ctx.sr) if len(seg) >= 64 else float("nan")
    low = band_energy_ratio_db(seg, ctx.sr, (80.0, 500.0), (500.0, 4000.0)) if len(seg) >= 64 else float("nan")
    fm = ctx.formants(start_s, end_s)
    return NasalMeasurement(ner_db=ner, low_freq_ratio=low, f1_hz=fm.f1, f2_hz=fm.f2, b1_hz=fm.b1)


def oral_reference_ner(windows: list[tuple[float, float]], ctx: AcousticContext) -> float | None:
    """Median NER over oral open-vowel windows of the same recitation (speaker baseline)."""
    vals = []
    for start, end in windows:
        seg = ctx.audio.segment(start, end)
        if len(seg) >= 64:
            vals.append(nasal_energy_ratio(seg, ctx.sr))
    return float(np.median(vals)) if len(vals) >= 2 else None


def nasal_verdict(contrast_db: float) -> tuple[bool | None, float]:
    """(nasal_ok, score): True strong, None moderate/unknown, False oral."""
    if not np.isfinite(contrast_db):
        return None, 0.5
    if contrast_db >= NER_STRONG_DB:
        return True, 1.0
    if contrast_db >= NER_WEAK_DB:
        return None, 0.6
    return False, 0.2
