"""The one call a server makes: recitation in, mastery report out.

    from app.engine import Engine
    report = Engine().analyze(audio_or_posteriors, [(23, 41), (23, 42)])

Everything below it is already validated: `app/analysis.py` (numpy, pinned to the Julia reference),
`app/rule_bind.py` (98.9 % of located rules bound), `app/submission.py` (grading and aggregation).
This module only resolves the reference text, produces posteriors and walks the recording.
"""

from __future__ import annotations

import json
import sys
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

from app.analysis import analyse_clip, ctc_viterbi
from app.rule_bind import bind
from app.ghunnah import grade_ghunnah
from app.mudud import resolve
from app.tafkhim import grade as grade_tafkhim
from app.itmam import sequences as vowel_sequences
from app.waqf import find_stops, madd_at_stops
from app.submission import (AyahRef, build_report, grade_rule, ph_to_uthmani, to_counts,
                            walk_alignment)
from app.tajweed_rules.parser import TajweedParser

ROOT = Path(__file__).resolve().parents[1]


def _np_slice(wave, t0: int, t1: int, frame_s: float = 0.04, sr: int = 16000):  # type: ignore[no-untyped-def]
    """The waveform samples belonging to a frame span."""
    import numpy as np
    w = np.asarray(wave)
    return w[int(t0 * frame_s * sr):int(t1 * frame_s * sr)]
LEARNER = ROOT / "research_agency_lab/experiments/learner_eval"


def _haraka_of(units) -> float | None:  # type: ignore[no-untyped-def]
    """The count unit an ayah's durations were measured in, recovered from its units."""
    hs = [u.duration_s / u.duration_counts for u in units if u.duration_counts not in (None, 0)]
    return float(np.median(hs)) if hs else None


# The munfasil (and the silah kubra, which is read as a munfasil) has more than one authentic length
# in Hafs: tawassut, 4-5 counts, by al-Shatibiyyah; qasr, 2 counts, by the Tayyibah. Over all 41 T300
# reciters the choice is bimodal -- 17 hold it at ~2 counts throughout (Shuraym, Sudais, Budair,
# Mustafa Ismail, ...), the rest at 4 and beyond -- and graded against the Shatibiyyah alone it read
# as the "hardest rule" in the dataset (pass 0.52). The standard is consistency within the reading
# the reciter chose, so the wajh is inferred from all of the submission's instances and each instance
# is graded against it.
WAJH_RULES = ("madd_munfasil", "madd_silah_kubra")
QASR_MAX_COUNTS = 3.0          # the gap between the two modes: 1.8-2.2 vs 3.2+


def _apply_wajh(verdict_lists, declared: str | None = None) -> dict:  # type: ignore[type-arg,no-untyped-def]
    """Grade munfasil / silah kubra against one wajh: the one the app DECLARES (the learner's chosen
    path), or else the one inferred from the recording. Inference alone cannot catch a munfasil read
    short on purpose -- a single qasr-length instance is read as the qasr wajh and passes."""
    from app.submission import NOMINAL_TOLERANCE
    vs = [v for vl in verdict_lists for v in vl
          if v.rule in WAJH_RULES and v.evidence.get("given_counts") is not None]
    if not vs:
        return {"declared": declared} if declared else {}
    counts = [v.evidence["given_counts"] for v in vs]
    med = float(np.median(counts))
    qasr = (declared == "qasr") if declared else med <= QASR_MAX_COUNTS
    lo, hi = (2.0, 2.0) if qasr else (4.0, 5.0)
    for v in vs:
        got = v.evidence["given_counts"]
        v.expected_counts = (lo, hi)
        v.status = "pass" if lo - NOMINAL_TOLERANCE <= got <= hi + NOMINAL_TOLERANCE else \
            ("short" if got < lo else "long")
        v.evidence = {**v.evidence, "expected": [lo, hi],
                      "wajh": "qasr (Tayyibah)" if qasr else "tawassut (Shatibiyyah)"}
    return {"munfasil": "qasr (Tayyibah), 2 counts" if qasr else "tawassut (Shatibiyyah), 4-5 counts",
            "choice": "qasr" if qasr else "tawassut", "expected_counts": [lo, hi],
            "source": "declared" if declared else "inferred",
            "instances": len(vs), "median_counts": round(med, 2),
            "note": "graded for consistency with the wajh the reciter chose"}


def _unconfirm_durations(verdicts) -> None:  # type: ignore[no-untyped-def]
    """Durations in an ayah whose count unit was borrowed are reported, not judged.

    An ayah like الٓمٓ is recited at its own pace, not the pace of the ayah after it. Graded on the
    borrowed unit, 41 professional reciters on Baqarah 2:1 read the madd lazim anywhere from 3.9 to
    beyond 12 counts and the idgham shafawi "long" in 37 of 41 -- a verdict that marks every master
    wrong is not a verdict. The measured counts stay in the report for the learner to see.
    """
    for v in verdicts:
        if v.mechanism == "durational" and v.status in {"pass", "short", "long"}:
            v.evidence = {**v.evidence, "reason": "count unit borrowed from the neighbouring ayahs; "
                          "this ayah is recited at its own pace, so its lengths are shown but not "
                          "graded"}
            v.status = "unconfirmed"


class Engine:
    """Resolves reference text, runs the acoustic model, and assembles the report."""

    def __init__(self, layout: dict[str, Any] | None = None) -> None:
        self._layout = layout
        self.parser = TajweedParser()

    # -- reference -------------------------------------------------------------------------------
    @cached_property
    def _phonetizer(self):  # type: ignore[no-untyped-def]
        sys.path.insert(0, str(LEARNER))
        from muaalem_eval import MOSHAF
        from quran_transcript import Aya, quran_phonetizer
        import muaalem_dump as md
        return Aya, quran_phonetizer, MOSHAF, md

    @cached_property
    def _sifat_maps(self) -> dict[str, dict[str, int]]:
        from research_agency_lab.experiments.learner_eval.sifat_ref import class_maps  # noqa: PLC0415
        return class_maps()

    def reference(self, surah: int, ayah: int, words: tuple[int, int] | None = None) -> AyahRef:
        """Everything the engine needs to know about what *should* be recited.

        `words` (first, last; 0-based, inclusive) restricts it to part of the ayah. The slice is
        phonetised on its own, because that is what a reader who starts or stops there must say:
        hamzat al-wasl read at the start, the waqf form of the last word at the end.
        """
        Aya, phonetize, moshaf, md = self._phonetizer
        uthmani = Aya(surah, ayah).get().uthmani
        offset = 0
        if words is not None:
            ws = uthmani.split()
            w0, w1 = words
            if not 0 <= w0 <= w1 < len(ws):
                raise ValueError(f"{surah}:{ayah} has {len(ws)} words; asked for {w0}..{w1}")
            if (w0, w1) != (0, len(ws) - 1):
                uthmani, offset = " ".join(ws[w0:w1 + 1]), w0
        r = phonetize(uthmani, moshaf, remove_spaces=True)
        from research_agency_lab.experiments.learner_eval.sifat_ref import LEVELS  # noqa: PLC0415
        maps = self._sifat_maps
        cols: dict[str, list[int]] = {lvl: [] for lvl in LEVELS}
        for e in r.sifat:
            n = len(e.phonemes)
            for lvl in LEVELS:
                cols[lvl].extend([maps[lvl].get(getattr(e, lvl), 0)] * n)
        return AyahRef(surah=surah, ayah=ayah, uthmani=uthmani, phonemes=r.phonemes,
                       word_ph=md.word_spans(uthmani, r.mappings), expected_sifat=cols,
                       ph_to_uth=ph_to_uthmani(uthmani, r.mappings), word_offset=offset)

    # -- acoustics -------------------------------------------------------------------------------
    @cached_property
    def _model(self):  # type: ignore[no-untyped-def]
        sys.path.insert(0, str(LEARNER))
        import torch
        from muaalem_eval import Muaalem
        return Muaalem(device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float32)

    def posteriors(self, audio) -> np.ndarray:  # type: ignore[no-untyped-def]
        """Log-posteriors of every CTC level for one recording, concatenated as the dump stores them."""
        sys.path.insert(0, str(LEARNER))
        import muaalem_dump as md
        wave = audio if isinstance(audio, np.ndarray) else md.load_16k(Path(audio))
        lp = md.posteriors(self._model, wave)
        if self._layout is None:
            self._layout = self._layout_from(lp)
        return np.concatenate([lp[k] for k in sorted(lp, key=md._level_order)], axis=1)

    def _layout_from(self, lp) -> dict[str, Any]:  # type: ignore[no-untyped-def]
        sys.path.insert(0, str(LEARNER))
        import muaalem_dump as md
        cols, c0, levels = [], 0, {}
        for lvl in sorted(lp, key=md._level_order):
            w = int(lp[lvl].shape[1])
            vocab = self._model.multi_level_tokenizer.id_to_vocab[lvl]
            levels[lvl] = {"first": c0, "width": w,
                           "vocab": [vocab.get(i, "") for i in range(w)]}
            c0 += w
        return {"columns": c0, "blank": 0, "levels": levels}

    # -- the call --------------------------------------------------------------------------------
    def analyze(self, audio, verses: list[tuple[int, ...]], *,  # type: ignore[no-untyped-def]
                rule_filter: str | None = None, posteriors: np.ndarray | None = None,
                wajh: str | None = None) -> dict[str, Any]:
        """Score a submission covering `verses`, in order, against the audio.

        Each verse is (surah, ayah) or (surah, ayah, first_word, last_word) for part of an ayah
        (0-based, inclusive; reported word indices stay those of the whole ayah).

        `rule_filter` restricts the report to one rule family, for rule-practice submissions.
        `wajh` ("qasr" | "tawassut") declares the munfasil length the learner follows; without it the
        wajh is inferred from the recording.
        """
        lp = posteriors if posteriors is not None else self.posteriors(audio)
        audio = audio if isinstance(audio, __import__("numpy").ndarray) else None
        lay = self._layout
        if lay is None:
            raise RuntimeError("no layout: pass one to Engine() or let posteriors() build it")
        ph = lay["levels"]["phonemes"] if "levels" in lay and isinstance(lay["levels"], dict) \
            else next(l for l in lay["levels"] if l["level"] == "phonemes")
        blocks = {k: (v["first"], v["width"], v["vocab"])
                  for k, v in (lay["levels"].items() if isinstance(lay["levels"], dict)
                               else ((l["level"], l) for l in lay["levels"])) if k != "phonemes"}
        vocab = {t: i for i, t in enumerate(ph["vocab"]) if len(t) == 1}
        blank = lay["blank"]

        refs = [self.reference(v[0], v[1], (v[2], v[3]) if len(v) > 2 else None) for v in verses]
        spans = walk_alignment(lp, refs, vocab, blank, ph["first"], ph["width"])
        if len(spans) > 1:
            from app.submission import settle_boundaries
            from app.waqf import pause_intervals
            spans = settle_boundaries(spans, pause_intervals(audio) if audio is not None else None)
        basmala = self._basmala(lp, refs, spans, vocab, blank, ph["first"], ph["width"])
        if basmala["present"]:
            spans = basmala.pop("_spans")
        basmala.pop("_spans", None)

        # Pass 1: units per ayah. An ayah too short to measure its own count unit (الٓمٓ has no
        # vowelled letters at all) borrows the unit measured on the rest of the submission, so its
        # six-count madd lazim is judged instead of silently skipped.
        clips = [(r, t0, t1) for r, (t0, t1) in zip(refs, spans) if t1 > t0]
        units_of = [analyse_clip(lp[t0:t1], r.phonemes, vocab, blank, ph["first"], ph["width"],
                                 blocks, r.expected_sifat) for r, t0, t1 in clips]
        own = [_haraka_of(u) for u in units_of]
        known = [h for h in own if h]
        borrowed = float(np.median(known)) if known else None
        for k, (r, t0, t1) in enumerate(clips):
            if own[k] is None and borrowed:
                units_of[k] = analyse_clip(lp[t0:t1], r.phonemes, vocab, blank, ph["first"],
                                           ph["width"], blocks, r.expected_sifat, haraka_s=borrowed)

        per_ayah = []
        for (r, t0, t1), units, h_own in zip(clips, units_of, own):
            parsed = self.parser.parse(r.uthmani)
            bounds, resolutions = resolve(bind(parsed, r.phonemes, r.word_ph))
            # silence is acoustic: pass the audio segment, not the posteriors
            seg = None if audio is None else _np_slice(audio, t0, t1)
            stops = find_stops(units, r.word_ph, seg)
            # where the reciter actually stopped changes which madd the text requires
            bounds = madd_at_stops(bounds, stops, units)
            verdicts = [grade_rule(b, units) for b in bounds]
            if not h_own:
                _unconfirm_durations(verdicts)
            ghunnah = [g for g in (grade_ghunnah(b, units, to_counts) for b in bounds) if g]
            heaviness = grade_tafkhim(units)
            seqs = vowel_sequences(units)
            h = _haraka_of(units)
            per_ayah.append({"surah": r.surah, "ayah": r.ayah, "frames": [t0, t1],
                             "word_offset": r.word_offset,
                             "haraka_s": round(h, 3) if h else None,
                             "haraka_source": "own" if h_own else ("borrowed" if h else None),
                             "words": r.uthmani.split(), "word_ph": r.word_ph,
                             "uthmani": r.uthmani, "ph_to_uth": r.ph_to_uth,
                             "verdicts": verdicts, "ghunnah": ghunnah,
                             "resolutions": resolutions, "stops": stops,
                             "heaviness": heaviness, "sequences": seqs, "_units": units})
        if wajh not in (None, "qasr", "tawassut"):
            raise ValueError(f"wajh must be 'qasr' or 'tawassut', not {wajh!r}")
        wajh_report = _apply_wajh([a["verdicts"] for a in per_ayah], wajh)
        report = build_report(per_ayah, rule_filter)
        report["basmala"] = basmala
        report["wajh"] = wajh_report
        if audio is not None:
            report["audio_seconds"] = round(float(np.asarray(audio).size) / 16000, 2)
        from app.stretch import DECIDES_BOTH, DECIDES_SHORT, apply as stretch
        report["stretch"] = stretch(report)
        by_place = {(s["surah"], s["ayah"], s["word"], s["rule"]): s for s in report["stretch"]["stretchings"]}
        for a in report["ayahs"]:
            for v in a["rules"]:
                s = by_place.get((a["surah"], a["ayah"], v["word_index"], v["rule"]))
                if not s:
                    continue
                v["stretch"] = {k: s.get(k) for k in ("unit_s", "stretch", "expected_stretch", "ratio", "level",
                                                      "equivalent_counts", "z", "verdict")}
                verdict = s.get("verdict")
                decides = (v["rule"] in DECIDES_BOTH and verdict in ("pass", "short", "long")
                           and v["status"] in ("pass", "short", "long")) or \
                          (v["rule"] in DECIDES_SHORT and verdict == "short" and v["status"] == "pass")
                if decides and v["status"] != verdict:
                    v["status"] = verdict
                    v["evidence"] = {**(v.get("evidence") or {}), "decided_by": "stretch calculus"}
        from app.measurements import build as measurements
        report["measurements"] = measurements(report)
        return report

    # Recordings of a surah's first ayah often open with the basmala (in T300, 60 of 766 verse-1 clips,
    # all from four reciters). Graded as part of the ayah, its بِسْمِ was heard as the ayah's first
    # vowels -- a certified reviewer caught it. The two readings are tested against the audio: if
    # "basmala + ayah" explains the opening far better than the ayah alone, the basmala is aligned
    # and cut off before grading. Measured separation is wide (LR +362 with it, -540 without; only 3
    # of 766 fall within +-20 nats).
    BASMALA_LR = 20.0

    def _basmala(self, lp, refs, spans, vocab, blank, first, width) -> dict[str, Any]:  # type: ignore[no-untyped-def]
        from app.lahn.gop import ctc_log_likelihood
        out: dict[str, Any] = {"checked": False, "present": False}
        if not refs or refs[0].ayah != 1 or refs[0].word_offset or refs[0].surah in (1, 9) or not spans:
            return out
        bas = self.reference(1, 1)
        region = lp[:spans[0][1], first:first + width]
        seq = [vocab[c] for c in refs[0].phonemes]
        lr = ctc_log_likelihood(region, [vocab[c] for c in bas.phonemes] + seq, blank) \
            - ctc_log_likelihood(region, seq, blank)
        out.update(checked=True, log_likelihood_ratio=round(float(lr), 1) if np.isfinite(lr) else None)
        if np.isfinite(lr) and lr > self.BASMALA_LR:
            # one joint alignment of basmala + ayah over the opening: the cut is where the basmala's
            # last phoneme ends. (Aligning them as separate ayahs let a short ayah like نٓ slide back
            # onto the ن of ٱلرَّحْمَـٰنِ.)
            nb = len(bas.phonemes)
            _s, f_, l_ = ctc_viterbi(region, [vocab[c] for c in bas.phonemes] + seq, blank)
            cut = int(l_[nb - 1])
            new = [(max(cut, spans[0][0]), spans[0][1]), *spans[1:]]
            out.update(present=True, end_s=round(cut * 0.04, 2), _spans=new)
        return out
