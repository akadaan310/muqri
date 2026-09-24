"""Server-side submission: a recitation at any scale in, a full mastery report out.

A user submits a recording — one word, one verse, a page, a juz' — together with the verses it
covers, which the client always knows. The engine returns every letter's complete state, every
located tajweed rule graded against what it requires, and the cross-ayah mastery statistics.

Two design points that are forced rather than chosen:

* **Segment per ayah.** `ctc_viterbi` is O(T x 2S+1); a whole surah is ~27,700 phonemes, so aligning a
  long recording as one string is quadratic and impossible. Instead the reference is walked ayah by
  ayah through the recording, each alignment starting where the previous one ended. That is linear in
  the submission's length, which is what makes a juz' cost ~12 s of analysis.
* **Cross-ayah statistics belong to the submission, not the clip.** Taswiyah (is every madd of a
  category held equally), the sukoon spectrum and tempo drift are variance and trend statistics over
  the *whole* recitation. A per-clip report cannot express them, which is exactly why a master is
  distinguished across a passage rather than in a single verse.

Unvalidated quantities are reported with `confidence: "unvalidated"` rather than omitted or silently
shipped — currently the absolute madd count scale and any learner-facing grade.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.analysis import FRAME_S, Unit, analyse_clip, ctc_viterbi
from app.ghunnah import grade_ghunnah, letter_strength
from app.mudud import drift
from app.waqf import endurance, roll_up as waqf_roll_up
from app.rule_bind import ph_units

SCHEMA_VERSION = "1.4"
ROOT = Path(__file__).resolve().parents[1]
# how much audio one ayah may consume, as a multiple of its phoneme count, when walking a long
# recording: generous enough for the slowest mujawwad, tight enough to stay linear
FRAMES_PER_PHONEME_MAX = 12.0

# The absolute count scale, fitted by Theil-Sen over 10,600 durational instances on the anchors
# (substrate_library/julia/madd_scale.jl). measured = intercept + slope x nominal, so a measured
# duration converts to counts by the inverse. Four of the five nominal levels recover within 0.15
# counts (1.5 -> 1.50, 2 -> 2.04, 4 -> 3.90, 4.5 -> 4.35); the six-count level does not (7.21 from
# n=105), because madd lazim is genuinely stretched beyond six in mujawwad style, so verdicts there
# stay flagged.
_SCALE_PATH = ROOT / "research_agency_lab/experiments/calibration/madd_scale.json"
WELL_FITTED_NOMINALS = (1.5, 2.0, 4.0, 4.5)
NOMINAL_TOLERANCE = 0.75          # counts; a verdict fires only outside this band


def count_scale() -> tuple[float, float]:
    """(intercept, slope) of measured-against-nominal, or the identity when unfitted."""
    try:
        d = json.loads(_SCALE_PATH.read_text())
        return float(d["intercept"]), float(d["slope"])
    except Exception:  # noqa: BLE001 - an unfitted deployment must still serve
        return 0.0, 1.0


def to_counts(measured: float) -> float:
    """Convert a measured own-counts duration into tajweed counts using the fitted scale."""
    a, b = count_scale()
    return (measured - a) / b if b else measured


@dataclass(slots=True)
class AyahRef:
    surah: int
    ayah: int
    uthmani: str = ""
    phonemes: str = ""
    word_ph: list[list[int]] = field(default_factory=list)
    expected_sifat: dict[str, list[int]] = field(default_factory=dict)


@dataclass(slots=True)
class RuleVerdict:
    """One located rule instance, graded."""

    rule: str
    word_index: int
    word: str
    detail: str
    mechanism: str
    units: list[int]
    expected_counts: tuple[float, float] | None
    measured_counts: float | None
    shadda: str | None
    status: str                      # pass | short | long | wrong | unconfirmed | no_evidence
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {"rule": self.rule, "word_index": self.word_index, "word": self.word,
             "detail": self.detail, "mechanism": self.mechanism, "units": self.units,
             "status": self.status, "shadda": self.shadda,
             "expected_counts": list(self.expected_counts) if self.expected_counts else None,
             "measured_counts": self.measured_counts, "evidence": self.evidence}
        if self.mechanism == "durational" and self.measured_counts is not None:
            # The scale is fitted and recovers four of five nominal levels within 0.15 counts, so
            # those verdicts are trustworthy. The six-count level recovers at 7.21 (n=105), because
            # madd lazim is genuinely stretched beyond six in mujawwad, so it stays flagged.
            nominal = (self.expected_counts[0] + self.expected_counts[1]) / 2 \
                if self.expected_counts else None
            d["confidence"] = "validated" if nominal in WELL_FITTED_NOMINALS else "unvalidated"
        return d


def grade_rule(b, units: list[Unit]) -> RuleVerdict:  # type: ignore[no-untyped-def]
    """Grade one bound rule from the acoustic evidence at its units."""
    us = [units[i] for i in b.unit_indices if 0 <= i < len(units)]
    measured = round(sum(u.duration_counts for u in us if u.duration_counts is not None), 2) \
        if us and all(u.duration_counts is not None for u in us) else None
    status, ev = "no_evidence", {}

    if b.mechanism == "durational" and b.expected_counts and measured is not None:
        lo, hi = b.expected_counts
        # convert the measured duration into tajweed counts with the fitted scale, so the verdict can
        # be stated as "you gave 2.1 counts where 4 are required" instead of a bare ratio
        got = round(to_counts(measured), 2)
        status = "pass" if lo - NOMINAL_TOLERANCE <= got <= hi + NOMINAL_TOLERANCE else \
            ("short" if got < lo else "long")
        ev = {"expected": [lo, hi], "given_counts": got, "raw_own_counts": measured}
    elif b.mechanism == "attribute" and b.head:
        llrs = [u.sifat[b.head]["llr"] for u in us if b.head in u.sifat]
        if llrs:
            worst = min(llrs)
            status = "pass" if worst > 0 else "wrong"
            bad = [u.sifat[b.head] for u in us if b.head in u.sifat and u.sifat[b.head]["llr"] <= 0]
            ev = {"head": b.head, "llr": round(worst, 3),
                  "expected": us[0].sifat[b.head]["expected"] if b.head in us[0].sifat else None,
                  "heard": bad[0]["model_best"] if bad else None}
    else:  # segmental: did the reference letters survive against their competitors
        if us:
            status = "pass" if all(u.confirmed for u in us) else "wrong"
            bad = [u for u in us if not u.confirmed]
            ev = {"heard": bad[0].best_competitor if bad else None,
                  "llr": bad[0].competitor_llr if bad else None}
    return RuleVerdict(rule=b.rule_type, word_index=b.word_index, word=b.word, detail=b.detail,
                       mechanism=b.mechanism, units=list(b.unit_indices),
                       expected_counts=b.expected_counts, measured_counts=measured,
                       shadda=b.shadda, status=status, evidence=ev)


def walk_alignment(lp: np.ndarray, refs: list[AyahRef], vocab: dict[str, int], blank: int,
                   ph_first: int, ph_width: int, band: float = 0.6) -> list[tuple[int, int]]:
    """Frame span of each ayah inside one long recording.

    `ctc_viterbi` is O(T x 2S+1), so a long submission cannot be aligned as one string (a whole surah
    is ~27,700 phonemes). It has to be split — but splitting *sequentially* drifts: aligning ayah N
    inside a window that also holds N+1's audio lets N's last phoneme stretch over it, and once one
    ayah over-consumes every later one starts late and the error compounds. Measured on an 8-ayah
    passage, sequential walking ended 2,530 frames out by ayah 6 and produced 150 phantom errors.

    So each ayah is anchored to its **global proportional position** instead: with total frames T and
    total phonemes P, ayah i is expected around T x (phonemes before i) / P, and is aligned inside a
    band around that. Errors cannot accumulate because every window is placed from the whole, not
    from its predecessor. Speech rate is near-constant within one submission, which is what makes the
    proportional prior sound.
    """
    block = lp[:, ph_first:ph_first + ph_width]
    T = block.shape[0]
    lens = [max(1, len(r.phonemes)) for r in refs]
    total = sum(lens)
    starts: list[int] = []
    acc = 0
    for n in lens:
        starts.append(acc)
        acc += n

    spans: list[tuple[int, int]] = []
    prev_end = 0
    for i, r in enumerate(refs):
        seq = [vocab[c] for c in r.phonemes if c in vocab]
        if not seq or T == 0:
            spans.append((prev_end, prev_end))
            continue
        exp_start = int(T * starts[i] / total)
        exp_end = int(T * (starts[i] + lens[i]) / total)
        slack = int(band * (exp_end - exp_start)) + 8
        lo = max(0, min(exp_start - slack, prev_end) if i else 0)
        hi = min(T, exp_end + slack) if i < len(refs) - 1 else T
        if hi <= lo:
            spans.append((prev_end, prev_end))
            continue
        _s, first, last = ctc_viterbi(block[lo:hi], seq, blank)
        a = lo + (min(first) - 1 if first else 0)
        b = lo + (max(last) if last else 0)
        a, b = max(0, a), min(T, max(b, a + 1))
        spans.append((a, b))
        prev_end = b
    return spans


def _roll_up(verdicts: list[RuleVerdict]) -> dict[str, Any]:
    """Per-rule aggregates, the shape a curriculum system needs to pick weak rules directly."""
    by: dict[str, dict[str, Any]] = {}
    for v in verdicts:
        d = by.setdefault(v.rule, {"attempted": 0, "pass": 0, "errors": 0, "statuses": {}})
        d["attempted"] += 1
        d["statuses"][v.status] = d["statuses"].get(v.status, 0) + 1
        if v.status == "pass":
            d["pass"] += 1
        elif v.status not in {"no_evidence", "unconfirmed"}:
            d["errors"] += 1
    for d in by.values():
        d["accuracy"] = round(d["pass"] / d["attempted"], 4) if d["attempted"] else None
    return by


def _ghunnah_roll_up(gs: list[Any]) -> dict[str, Any]:
    """Maratib al-Ghunnah: how each grade of nasalisation was held."""
    by: dict[str, dict[str, Any]] = {}
    for g in gs:
        d = by.setdefault(g.grade, {"n": 0, "pass": 0, "short": 0, "long": 0, "given": []})
        d["n"] += 1
        if g.status in d:
            d[g.status] += 1
        if g.given_counts is not None:
            d["given"].append(g.given_counts)
    for d in by.values():
        vals = d.pop("given")
        d["median_counts"] = round(statistics.median(vals), 2) if vals else None
        d["accuracy"] = round(d["pass"] / d["n"], 4) if d["n"] else None
    return by


def _strength_roll_up(ayahs: list[dict[str, Any]]) -> dict[str, Any]:
    """Letter quwwa: strong sifat expected against realised, and which are most often weak."""
    exp = real = 0
    missing: dict[str, int] = {}
    for a in ayahs:
        for l in a["letters"]:
            st = l.get("strength") or {}
            exp += st.get("expected", 0)
            real += st.get("realised", 0)
            for m in st.get("missing", []):
                missing[m] = missing.get(m, 0) + 1
    return {"strong_sifat_expected": exp, "realised": real,
            "ratio": round(real / exp, 4) if exp else None,
            "weakest": dict(sorted(missing.items(), key=lambda kv: -kv[1])[:5])}


def _mastery(per_ayah: list[dict[str, Any]]) -> dict[str, Any]:
    """Cross-ayah statistics: consistency, tempo and the sukoon spectrum over the whole submission."""
    by_class: dict[str, list[float]] = {}
    harakas: list[float] = []
    for a in per_ayah:
        for u in a["_units"]:
            if u.duration_counts is None:
                continue
            if u.kind == "madd" and u.run_length >= 2:
                by_class.setdefault(f"madd_{u.run_length}", []).append(u.duration_counts)
            elif u.symbol in "ںنم" and u.run_length >= 3:
                by_class.setdefault("ghunnah", []).append(u.duration_counts)
        if a.get("haraka_s"):
            harakas.append(a["haraka_s"])

    def rcv(v: list[float]) -> float | None:
        if len(v) < 4:
            return None
        med = statistics.median(v)
        mad = statistics.median([abs(x - med) for x in v])
        return round(1.4826 * mad / max(abs(med), 1e-9), 4)

    tempo = statistics.median(harakas) if harakas else None
    mode = None
    if tempo:   # the three canonical tempos, ~350 / ~250 / ~150 ms per count
        mode = "tahqiq" if tempo >= 0.30 else ("tadwir" if tempo >= 0.20 else "hadr")
    return {
        "tempo_haraka_s": round(tempo, 3) if tempo else None, "tempo_mode": mode,
        "taswiyah": {k: {"n": len(v), "median_counts": round(statistics.median(v), 2), "cv": rcv(v)}
                     for k, v in sorted(by_class.items()) if len(v) >= 4},
        "note": "taswiyah CV is validated; median_counts uses an uncalibrated absolute scale",
    }


def build_report(per_ayah: list[dict[str, Any]], rule_filter: str | None = None) -> dict[str, Any]:
    """Assemble the submission report from per-ayah results."""
    all_v: list[RuleVerdict] = []
    all_gh: list[Any] = []
    ayahs = []
    for a in per_ayah:
        vs = [v for v in a["verdicts"] if not rule_filter or v.rule.startswith(rule_filter)]
        all_v.extend(vs)
        gh = a.get("ghunnah", [])
        all_gh.extend(gh)
        units = a["_units"]
        ayahs.append({
            "surah": a["surah"], "ayah": a["ayah"], "frames": a["frames"],
            "haraka_s": a.get("haraka_s"),
            "letters": [{"i": u.index, "symbol": u.symbol, "kind": u.kind,
                         "onset_s": u.onset_s, "duration_s": u.duration_s,
                         "duration_counts": u.duration_counts,
                         "identity": {"gop": u.gop, "heard_instead": u.best_competitor,
                                      "llr": u.competitor_llr, "confirmed": u.confirmed},
                         "sifat": u.sifat, "strength": letter_strength(u)}
                        for u in units],
            "rules": [v.to_dict() for v in vs],
            "ghunnah": [g.to_dict() for g in gh],
            "stops": [s.to_dict() for s in a.get("stops", [])],
        })
    errors = [v.to_dict() for v in all_v if v.status in {"short", "long", "wrong"}]
    letters = sum(len(a["letters"]) for a in ayahs)
    judgments = sum(len(l["sifat"]) + 2 for a in ayahs for l in a["letters"])
    return {
        "schema_version": SCHEMA_VERSION,
        "summary": {
            "ayahs": len(ayahs), "letters": letters, "judgments": judgments,
            "rules_located": len(all_v), "errors": len(errors),
            "accuracy": round(1 - len(errors) / len(all_v), 4) if all_v else None,
        },
        "mastery": _mastery(per_ayah),
        "by_rule": _roll_up(all_v),
        "ghunnah_grades": _ghunnah_roll_up(all_gh),
        "madd_drift": drift([a["verdicts"] for a in per_ayah]),
        "waqf": waqf_roll_up([a.get("stops", []) for a in per_ayah]),
        "endurance": endurance([a.get("stops", []) for a in per_ayah]),
        "resolutions": [{"governing": r.governing, "overridden": r.overridden,
                         "expected_counts": list(r.expected_counts) if r.expected_counts else None}
                        for a in per_ayah for r in a.get("resolutions", [])],
        "letter_strength": _strength_roll_up(ayahs),
        "errors": errors,
        "ayahs": ayahs,
    }
