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
from app.tafkhim import roll_up as tafkhim_roll_up
from app.itmam import itmam, roll_up as seq_roll_up
from app.waqf import endurance, roll_up as waqf_roll_up
from app.rule_bind import ph_units

SCHEMA_VERSION = "1.5"
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

# Takrir is a characteristic the reciter must CONCEAL (ikhfa' al-takrir): a master's ر is not audibly
# trilled, least of all at a stop. The head only says trilled / not trilled, so "not trilled" is the
# correct outcome and was being scored as the characteristic missing -- Husary's correct Al-Kawthar
# lost three words to it. Reported, never scored, until an over-trill measure is calibrated.
DESCRIPTIVE_HEADS = frozenset({"tikraar"})


def judged(head: str, run_length: int, symbol: str = "") -> bool:
    """Whether a head's verdict on this letter counts as a fault.

    * A doubled letter (shaddah) is held twice as long, and the shiddah/rakhawah head reads that hold
      as flow: Bukhatir's رَبُّ was called rakhw, and a certified reviewer rejected it -- "it sounds
      held because of the shaddah".
    * Istitalah as a RULE is the ض's alone. When the head hears it on another letter -- the doubled or
      sakin ب of رَبُّ above all, and د -- it is not an error but a latent characteristic a certified
      reviewer recognised: the sound keeps flowing and builds against the closure before the
      release, where ت ك ق ط simply stop. It is recorded as a phenomenon (app/review.py, "voiced
      hold"), never scored as a fault.
    """
    if head in DESCRIPTIVE_HEADS:
        return False
    if head == "istitala" and symbol and symbol != "ض":
        return False
    return not (head == "shidda_or_rakhawa" and run_length >= 2)


def count_scale() -> tuple[float, float]:
    """(intercept, slope) of measured-against-nominal, or the identity when unfitted."""
    try:
        d = json.loads(_SCALE_PATH.read_text())
        return float(d["intercept"]), float(d["slope"])
    except Exception:  # noqa: BLE001 - an unfitted deployment must still serve
        return 0.0, 1.0


def six_count_ceiling() -> float:
    """Longest reading of a six-count madd that still matches how the anchors recite it.

    The fitted scale does not recover the six-count level: the anchors themselves measure 7.21
    counts there (n = 105, CV 0.22), because madd lazim is stretched past six in practice. A fixed
    6 + tolerance ceiling therefore marks the masters "long" -- on Baqarah 2:1-2 it flagged 7 of 41
    professional reciters' final madd 'arid. The ceiling is the anchors' own level plus one CV.
    """
    try:
        d = json.loads(_SCALE_PATH.read_text())["per_nominal"]["6.0"]
        return float(d["implied_counts"]) * (1 + float(d["cv"]))
    except Exception:  # noqa: BLE001 - unfitted deployment: fall back to the nominal band
        return 6.0 + NOMINAL_TOLERANCE


def anchor_count_s() -> float:
    """One count at the anchors' pace, in seconds: the median haraka of the Husary recordings."""
    try:
        d = json.loads((ROOT / "research_agency_lab/experiments/qaari_keys/sukoon_T300.json").read_text())
        hs = [r["haraka_s"] for r in d["reciters"] if r.get("anchor")]
        return float(statistics.median(hs)) if hs else 0.30
    except Exception:  # noqa: BLE001 - an unmeasured deployment falls back to a murattal count
        return 0.30


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
    # for every character of `phonemes`, the Uthmani character that produced it (-1: none), so a
    # judged letter can be shown -- and coloured -- in the script the reciter reads
    ph_to_uth: list[int] = field(default_factory=list)
    # a submission may cover part of an ayah: word indices here are local to the slice, and
    # word_offset is the index of its first word in the whole ayah
    word_offset: int = 0


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
        ceiling = six_count_ceiling() if hi >= 6 else hi + NOMINAL_TOLERANCE
        status = "pass" if lo - NOMINAL_TOLERANCE <= got <= ceiling else \
            ("short" if got < lo else "long")
        ev = {"expected": [lo, hi], "given_counts": got, "raw_own_counts": measured}
        # At hadr the count unit is short, so a ghunnah or madd of normal duration reads as many of the
        # reciter's own counts: Ghamdi's idgham ghunnah read 3.9 counts in 0.63 s, and a certified
        # reviewer ruled "it's allowed at this speed". "Too long" therefore also needs the length to
        # be long in time, against the anchors' count unit. Dossary's 9.5 counts in 1.9 s -- which the
        # reviewer confirmed -- still fires. "Too short" stays relative: speed never excuses a clipped
        # madd.
        secs = sum(u.duration_s for u in us)
        if status == "long" and secs <= ceiling * anchor_count_s():
            status = "pass"
            ev["tempo_allowance"] = {"seconds": round(secs, 2),
                                     "anchor_limit_s": round(ceiling * anchor_count_s(), 2)}
    elif b.mechanism == "attribute" and b.head in DESCRIPTIVE_HEADS:
        ev = {"head": b.head, "reason": "takrir must be concealed; a trill that is not heard is the "
                                        "correct reading, so this attribute is reported, not graded"}
        status = "unconfirmed"
    elif b.mechanism == "attribute" and b.head:
        llrs = [u.sifat[b.head]["llr"] for u in us if b.head in u.sifat]
        if llrs:
            worst = min(llrs)
            status = "pass" if worst > 0 else "wrong"
            bad = [u.sifat[b.head] for u in us if b.head in u.sifat and u.sifat[b.head]["llr"] <= 0]
            ev = {"head": b.head, "llr": round(worst, 3),
                  "expected": us[0].sifat[b.head]["expected"] if b.head in us[0].sifat else None,
                  "heard": bad[0]["model_best"] if bad else None}
    elif b.mechanism == "durational":
        # A durational rule is judged by its length. When the length is unknown, the letters being
        # present says nothing about it -- falling through to the identity check below passed every
        # madd lazim in الٓمٓ, read at any speed, because the ayah had no count unit.
        ev = {"reason": "duration not measurable here"}
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


def ph_to_uthmani(uthmani: str, mappings: list) -> list[int]:  # type: ignore[type-arg]
    """Invert the phonetizer's per-character mappings: phoneme index -> Uthmani character index."""
    n = max((m.pos[1] for m in mappings if m is not None), default=0)
    out = [-1] * n
    for i, m in enumerate(mappings):
        if m is not None and not getattr(m, "deleted", False):
            for p in range(m.pos[0], m.pos[1]):
                out[p] = i
    return out


def _uth_chars(ph_to_uth: list[int], span: tuple[int, int]) -> list[int]:
    """The Uthmani characters behind one unit's phoneme span (inclusive)."""
    return sorted({ph_to_uth[p] for p in range(span[0], span[1] + 1)
                   if 0 <= p < len(ph_to_uth) and ph_to_uth[p] >= 0})


def _word_index(word_ph: list[list[int]]):  # type: ignore[no-untyped-def]
    """Map a phoneme-string character index to the word it belongs to."""
    def of(c: int) -> int | None:
        for w, span in enumerate(word_ph):
            if span and span[0] is not None and span[0] >= 0 and span[0] <= c < span[1]:
                return w
        return None
    return of


def _word_faults(ayah: dict[str, Any], words: list[str]) -> list[dict[str, Any]]:
    """Every word of the ayah, with every fault the engine found in it."""
    out = [{"index": i, "word": w, "faults": []} for i, w in enumerate(words)]
    for v in ayah["rules"]:
        if v["status"] in {"short", "long", "wrong"} and 0 <= v["word_index"] < len(out):
            out[v["word_index"]]["faults"].append({"type": "rule", "rule": v["rule"], "status": v["status"]})
    for l in ayah["letters"]:
        w = l.get("word")
        if w is None or not 0 <= w < len(out):
            continue
        if not l["identity"]["confirmed"]:
            out[w]["faults"].append({"type": "letter", "letter": l["symbol"],
                                     "heard": l["identity"]["heard_instead"]})
        for head, sv in (l.get("sifat") or {}).items():
            if not sv.get("realised") and judged(head, l.get("run_length", 1), l["symbol"]):
                out[w]["faults"].append({"type": "sifah", "letter": l["symbol"], "sifah": head,
                                         "expected": sv.get("expected"), "heard": sv.get("model_best")})
    return out


def _word_accuracy(ayahs: list[dict[str, Any]]) -> float | None:
    ws = [w for a in ayahs for w in a.get("words", [])]
    return round(sum(not w["faults"] for w in ws) / len(ws), 4) if ws else None


def _letter_accuracy(ayahs: list[dict[str, Any]]) -> float | None:
    """Share of letter-level judgments (identity + every sifah) that were realised."""
    n = ok = 0
    for a in ayahs:
        for l in a["letters"]:
            n += 1
            ok += bool(l["identity"]["confirmed"])
            for head, sv in (l.get("sifat") or {}).items():
                if not judged(head, l.get("run_length", 1), l["symbol"]):
                    continue
                n += 1
                ok += bool(sv.get("realised"))
    return round(ok / n, 4) if n else None


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
        # a borrowed unit is the others' median again; counting it would weight them twice
        if a.get("haraka_s") and a.get("haraka_source", "own") == "own":
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
        word_of = _word_index(a.get("word_ph") or [])
        ayahs.append({
            "surah": a["surah"], "ayah": a["ayah"], "frames": a["frames"],
            "word_offset": a.get("word_offset", 0),
            "haraka_s": a.get("haraka_s"), "haraka_source": a.get("haraka_source", "own"),
            "letters": [{"i": u.index, "symbol": u.symbol, "kind": u.kind, "run_length": u.run_length,
                         "word": word_of(u.char_span[0]),
                         "uth": _uth_chars(a.get("ph_to_uth") or [], u.char_span),
                         "onset_s": u.onset_s, "duration_s": u.duration_s,
                         "duration_counts": u.duration_counts,
                         "identity": {"gop": u.gop, "heard_instead": u.best_competitor,
                                      "llr": u.competitor_llr, "confirmed": u.confirmed},
                         "sifat": u.sifat, "strength": letter_strength(u)}
                        for u in units],
            "rules": [v.to_dict() for v in vs],
            "ghunnah": [g.to_dict() for g in gh],
            "stops": [s.to_dict() for s in a.get("stops", [])],
            "heaviness": [h.to_dict() for h in a.get("heaviness", [])],
        })
        ayahs[-1]["words"] = _word_faults(ayahs[-1], a.get("words") or [])
        ayahs[-1]["uthmani"] = a.get("uthmani", "")
    errors = [v.to_dict() for v in all_v if v.status in {"short", "long", "wrong"}]
    letters = sum(len(a["letters"]) for a in ayahs)
    judgments = sum(len(l["sifat"]) + 2 for a in ayahs for l in a["letters"])
    return {
        "schema_version": SCHEMA_VERSION,
        "summary": {
            "ayahs": len(ayahs), "letters": letters, "judgments": judgments,
            "rules_located": len(all_v), "errors": len(errors),
            # the headline: a word is right only when every rule, every characteristic and every
            # letter in it is right -- the way a teacher marks. Rule accuracy alone ignored the ~15
            # sifat judged per letter: a careless reading with 10 lost characteristics scored 83 %.
            "accuracy": _word_accuracy(ayahs),
            "rule_accuracy": round(1 - len(errors) / len(all_v), 4) if all_v else None,
            "letter_accuracy": _letter_accuracy(ayahs),
        },
        "mastery": _mastery(per_ayah),
        "by_rule": _roll_up(all_v),
        "ghunnah_grades": _ghunnah_roll_up(all_gh),
        "madd_drift": drift([a["verdicts"] for a in per_ayah]),
        "waqf": waqf_roll_up([a.get("stops", []) for a in per_ayah]),
        "endurance": endurance([a.get("stops", []) for a in per_ayah]),
        "tafkhim_levels": tafkhim_roll_up([a.get("heaviness", []) for a in per_ayah]),
        "itmam": itmam([a["_units"] for a in per_ayah]),
        "vowel_sequences": seq_roll_up([a.get("sequences", []) for a in per_ayah]),
        "resolutions": [{"governing": r.governing, "overridden": r.overridden,
                         "expected_counts": list(r.expected_counts) if r.expected_counts else None}
                        for a in per_ayah for r in a.get("resolutions", [])],
        "letter_strength": _strength_roll_up(ayahs),
        "errors": errors,
        "ayahs": ayahs,
    }
