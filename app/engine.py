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

from app.analysis import analyse_clip
from app.rule_bind import bind
from app.submission import AyahRef, build_report, grade_rule, walk_alignment
from app.tajweed_rules.parser import TajweedParser

ROOT = Path(__file__).resolve().parents[1]
LEARNER = ROOT / "research_agency_lab/experiments/learner_eval"


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

    def reference(self, surah: int, ayah: int) -> AyahRef:
        """Everything the engine needs to know about what *should* be recited."""
        Aya, phonetize, moshaf, md = self._phonetizer
        uthmani = Aya(surah, ayah).get().uthmani
        r = phonetize(uthmani, moshaf, remove_spaces=True)
        from research_agency_lab.experiments.learner_eval.sifat_ref import LEVELS  # noqa: PLC0415
        maps = self._sifat_maps
        cols: dict[str, list[int]] = {lvl: [] for lvl in LEVELS}
        for e in r.sifat:
            n = len(e.phonemes)
            for lvl in LEVELS:
                cols[lvl].extend([maps[lvl].get(getattr(e, lvl), 0)] * n)
        return AyahRef(surah=surah, ayah=ayah, uthmani=uthmani, phonemes=r.phonemes,
                       word_ph=md.word_spans(uthmani, r.mappings), expected_sifat=cols)

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
    def analyze(self, audio, verses: list[tuple[int, int]], *,  # type: ignore[no-untyped-def]
                rule_filter: str | None = None, posteriors: np.ndarray | None = None
                ) -> dict[str, Any]:
        """Score a submission covering `verses`, in order, against the audio.

        `rule_filter` restricts the report to one rule family, for rule-practice submissions.
        """
        lp = posteriors if posteriors is not None else self.posteriors(audio)
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

        refs = [self.reference(s, a) for s, a in verses]
        spans = walk_alignment(lp, refs, vocab, blank, ph["first"], ph["width"])

        per_ayah = []
        for r, (t0, t1) in zip(refs, spans):
            if t1 <= t0:
                continue
            units = analyse_clip(lp[t0:t1], r.phonemes, vocab, blank, ph["first"], ph["width"],
                                 blocks, r.expected_sifat)
            parsed = self.parser.parse(r.uthmani)
            verdicts = [grade_rule(b, units) for b in bind(parsed, r.phonemes, r.word_ph)]
            harakas = [u.duration_s / u.duration_counts for u in units
                       if u.duration_counts not in (None, 0)]
            per_ayah.append({"surah": r.surah, "ayah": r.ayah, "frames": [t0, t1],
                             "haraka_s": round(float(np.median(harakas)), 3) if harakas else None,
                             "verdicts": verdicts, "_units": units})
        return build_report(per_ayah, rule_filter)
