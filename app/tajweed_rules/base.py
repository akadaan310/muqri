"""Shared evaluation context and helpers for the rule validators."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from app.acoustic.features import AcousticContext, short_time_rms
from app.acoustic.tempo import TempoEstimate, short_syllable_units
from app.models import Alignment, RuleDiagnostic, RuleInstance, RuleType, Status, Vowel
from app.sifaat.formants import WeightReference, build_reference, oral_reference_ner, vowel_window
from app.tajweed_rules.parser import ParsedText


@dataclass(slots=True)
class Pause:
    """A silent interval between two aligned units (see ``taraweeh_adapter.fatigue_detector``)."""

    start_s: float
    end_s: float
    breath: bool = False
    after_unit: int | None = None

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass
class EvalContext:
    """Everything a validator needs to judge one rule instance."""

    parsed: ParsedText
    alignment: Alignment
    ctx: AcousticContext
    tempo: TempoEstimate
    mode: str = "studio"
    pauses: list[Pause] = field(default_factory=list)
    local_haraka_fn: Callable[[float], float] | None = None
    cache: dict[str, object] = field(default_factory=dict)

    # -- tempo -----------------------------------------------------------------------------------
    def haraka_ms(self, t: float | None = None) -> float:
        """Harakah length (ms) at time ``t``: the local tempo when a pace normalizer is active."""
        if t is not None and self.local_haraka_fn is not None:
            return float(self.local_haraka_fn(t))
        return self.tempo.haraka_ms

    # -- spans -----------------------------------------------------------------------------------
    def span(self, unit_indices: list[int]) -> tuple[float, float] | None:
        return self.alignment.span(unit_indices)

    def unit_span(self, index: int) -> tuple[float, float] | None:
        u = self.alignment.units.get(index)
        return (u.start_s, u.end_s) if u is not None else None

    def pause_after(self, unit_index: int, tolerance_s: float = 0.15) -> Pause | None:
        span = self.unit_span(unit_index)
        if span is None:
            return None
        for p in self.pauses:
            if p.after_unit == unit_index or abs(p.start_s - span[1]) <= tolerance_s or span[0] <= p.start_s <= span[1]:
                return p
        return None

    # -- references (lazy) -----------------------------------------------------------------------
    @cached_property
    def weight_reference(self) -> WeightReference:
        return build_reference(self.parsed, self.alignment, self.ctx)

    @cached_property
    def oral_ner_reference(self) -> float | None:
        windows = [
            w for i in short_syllable_units(self.parsed)
            if self.parsed.units[i].vowel == Vowel.FATHA and (w := vowel_window([i], self.alignment)) is not None
        ]
        return oral_reference_ner(windows, self.ctx)

    @cached_property
    def rms_db(self) -> tuple[np.ndarray, float]:
        """(dB envelope at 1 ms hop, 95th-percentile speech level)."""
        rms = short_time_rms(self.ctx.x, self.ctx.sr, frame_ms=10.0, hop_ms=1.0)
        db = 20 * np.log10(rms + 1e-12)
        return db, float(np.percentile(db, 95)) if db.size else 0.0

    def voiced_extent(self, start_s: float, end_s: float, drop_db: float = 20.0) -> tuple[float, float]:
        """Trim leading/trailing frames more than ``drop_db`` below the window's peak.

        CTC units run until the next token's spike, so a sakin/nasal letter's span often absorbs the
        silent closure of the following stop consonant. Durations of held sounds use this extent.
        """
        db, _ = self.rms_db
        a, b = max(0, int(start_s * 1000)), min(len(db), int(end_s * 1000))
        if b - a < 10:
            return start_s, end_s
        seg = db[a:b]
        loud = np.flatnonzero(seg >= seg.max() - drop_db)
        start, end = (a + int(loud[0])) / 1000, (a + int(loud[-1]) + 1) / 1000
        # Held nasals and sonorants are voiced: drop unvoiced edges (a following stop's closure).
        t, f0 = self.ctx.f0_track
        sel = (t >= start) & (t <= end) & np.isfinite(f0)
        if sel.sum() >= 2:
            vt = t[sel]
            start, end = max(start, float(vt[0]) - 0.005), min(end, float(vt[-1]) + 0.005)
        return start, end

    def frication_window(self, index: int, min_ms: float = 30.0) -> tuple[float, float] | None:
        """Locate the frication noise of a fricative near its aligned unit.

        CTC tends to place a fricative's token spike at the end of its noise, so the search starts
        100 ms before the unit and keeps the stretch whose share of energy above 2.5 kHz is highest.
        """
        span = self.unit_span(index)
        if span is None:
            return None
        lo = max(0.0, span[0] - 0.1)
        hi = span[0] + 0.6 * (span[1] - span[0])
        x = self.ctx.audio.segment(lo, hi).astype(np.float64)
        sr, hop, frame = self.ctx.sr, int(0.005 * self.ctx.sr), int(0.02 * self.ctx.sr)
        if len(x) < frame * 2:
            return span
        freqs = np.fft.rfftfreq(frame, 1 / sr)
        ratios = []
        for k in range(0, len(x) - frame, hop):
            spec = np.abs(np.fft.rfft(x[k: k + frame] * np.hanning(frame))) ** 2
            ratios.append(spec[freqs >= 2500].sum() / (spec.sum() + 1e-12))
        r = np.asarray(ratios)
        hot = r >= max(0.5, r.max() - 0.15)
        best, run, best_run = (0, 0), 0, 0
        for k, h in enumerate(np.append(hot, False)):
            run = run + 1 if h else 0
            if run > best_run:
                best_run, best = run, (k - run + 1, k)
        if best_run * 5 < min_ms:
            return None
        a = lo + best[0] * hop / sr
        return a, a + (best[1] - best[0] + 1) * hop / sr + frame / sr

    @cached_property
    def vowel_band_envelope_db(self) -> tuple[np.ndarray, int]:
        """Hilbert envelope (dB, 5 ms hop) of the 100-1000 Hz band: the analytic-signal magnitude
        of the vowel/F1 region, smoothed over 20 ms. Mirrors ``qaari_features.m`` (Octave reference)."""
        from scipy.signal import butter, hilbert, sosfiltfilt

        sr = self.ctx.sr
        sos = butter(4, [100, 1000], btype="bandpass", fs=sr, output="sos")
        env = np.abs(hilbert(sosfiltfilt(sos, self.ctx.x.astype(np.float64))))
        k = int(0.02 * sr)
        env = np.convolve(env, np.ones(k) / k, mode="full")[: len(env)]  # causal, as Octave's filter()
        hop = int(0.005 * sr)
        return 20 * np.log10(env[::hop] + 1e-9), hop

    def vowel_core_ms(self, start_s: float, end_s: float, drop_db: float = 10.0) -> float:
        """Longest voiced run whose vowel-band envelope stays within ``drop_db`` of the span peak.

        CTC spans absorb neighbouring closures and transitions; the core is the sustained vowel
        (or nasal murmur) itself, the quantity a harakah count is defined on.
        """
        env_db, hop = self.vowel_band_envelope_db
        sr = self.ctx.sr
        a, b = int(start_s * sr / hop), int(end_s * sr / hop)
        if b - a < 3:
            return 0.0
        seg = env_db[a:b]
        t_frames = (np.arange(a, b) * hop) / sr
        t, f0 = self.ctx.f0_track
        voiced = np.interp(t_frames, t, np.isfinite(f0).astype(float), left=0, right=0) > 0.5 if t.size else \
            np.zeros(b - a, dtype=bool)
        good = (seg > seg.max() - drop_db) & voiced
        best = run = 0
        for g in good:
            run = run + 1 if g else 0
            best = max(best, run)
        return best * hop / sr * 1000.0

    def voicing_fraction(self, start_s: float, end_s: float) -> float:
        t, f0 = self.ctx.f0_track
        sel = (t >= start_s) & (t <= end_s)
        return float(np.isfinite(f0[sel]).mean()) if sel.any() else float("nan")

    def silence_runs(self, start_s: float, end_s: float, below_peak_db: float = 30.0,
                     min_ms: float = 30.0) -> list[tuple[float, float]]:
        """Silent intervals (RMS more than ``below_peak_db`` under the speech level) in a window."""
        db, peak = self.rms_db
        a, b = max(0, int(start_s * 1000)), min(len(db), int(end_s * 1000))
        if b <= a:
            return []
        quiet = db[a:b] < peak - below_peak_db
        out: list[tuple[float, float]] = []
        run_start = None
        for k, q in enumerate(np.append(quiet, False)):
            if q and run_start is None:
                run_start = k
            elif not q and run_start is not None:
                if k - run_start >= min_ms:
                    out.append(((a + run_start) / 1000, (a + k) / 1000))
                run_start = None
        return out


Validator = Callable[[RuleInstance, EvalContext], RuleDiagnostic]


def ms(t: float) -> int:
    return int(round(t * 1000))


def skipped(rule: RuleInstance, reason: str, span: tuple[float, float] | None = None) -> RuleDiagnostic:
    start, end = span if span is not None else (0.0, 0.0)
    return RuleDiagnostic(
        rule_type=rule.rule_type, word=rule.word, start_ms=ms(start), end_ms=ms(end), status=Status.SKIPPED,
        feedback=reason, letter=rule.letter,
    )


def band_status(value: float, lo: float, hi: float, tol: float) -> tuple[Status, float]:
    """PASS inside [lo−tol, hi+tol], WARNING inside twice the tolerance, FAIL beyond (tol absolute)."""
    if lo - tol <= value <= hi + tol:
        return Status.PASS, 1.0
    dist = (lo - tol - value) if value < lo - tol else (value - hi - tol)
    if dist <= tol:
        return Status.WARNING, max(0.4, 1.0 - dist / (2 * tol))
    return Status.FAIL, max(0.0, 0.4 - (dist - tol) / (4 * tol))


def worst(*statuses: Status) -> Status:
    order: list[Status] = [Status.PASS, Status.VALID_NECESSARY_PAUSE, Status.SKIPPED, Status.WARNING, Status.FAIL]
    real: list[Status] = [s for s in statuses if s is not Status.SKIPPED] or [Status.SKIPPED]
    return max(real, key=order.index)


RULE_NAMES: dict[RuleType, str] = {
    RuleType.MADD_TABII: "Madd Tabi'i",
    RuleType.MADD_MUTTASIL: "Madd Muttasil",
    RuleType.MADD_MUNFASIL: "Madd Munfasil",
    RuleType.MADD_LAZIM: "Madd Lazim",
    RuleType.MADD_ARID: "Madd 'Aaridh lis-Sukoon",
    RuleType.MADD_LEEN: "Madd Leen",
    RuleType.MADD_BADAL: "Madd Badal",
    RuleType.MADD_IWAD: "Madd 'Iwadh",
    RuleType.MADD_SILAH_SUGHRA: "Silah Sughra",
    RuleType.MADD_SILAH_KUBRA: "Silah Kubra",
    RuleType.IZHAR_HALQI: "Izhaar Halqi",
    RuleType.IKHFA: "Ikhfa' Haqiqi",
    RuleType.IDGHAM_GHUNNAH: "Idghaam with Ghunnah",
    RuleType.IDGHAM_NO_GHUNNAH: "Idghaam without Ghunnah",
    RuleType.IQLAB: "Iqlab",
    RuleType.IKHFA_SHAFAWI: "Ikhfa' Shafawi",
    RuleType.IDGHAM_SHAFAWI: "Idghaam Shafawi",
    RuleType.IZHAR_SHAFAWI: "Izhaar Shafawi",
    RuleType.GHUNNAH: "Ghunnah Mushaddadah",
    RuleType.QALQALAH: "Qalqalah",
    RuleType.IDGHAM_MITHLAYN: "Idghaam Mithlayn",
    RuleType.IDGHAM_MUTAJANISAYN: "Idghaam Mutajanisayn",
    RuleType.IDGHAM_MUTAQARIBAYN: "Idghaam Mutaqaribayn",
    RuleType.TAFKHEEM: "Tafkheem",
    RuleType.TARQEEQ: "Tarqeeq",
    RuleType.JAWAZ_WAJHAYN: "Jawaz al-Wajhayn",
    RuleType.HAMZAT_WASL: "Hamzat al-Wasl",
    RuleType.SAKT: "Sakt",
}


def name_of(rule: RuleInstance) -> str:
    return RULE_NAMES.get(rule.rule_type, rule.rule_type.value.replace("_", " ").title())
