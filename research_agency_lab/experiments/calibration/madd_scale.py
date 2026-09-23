#!/usr/bin/env python3
"""Fit the absolute count scale: what does "one count" actually measure?

Every durational verdict has shipped with `confidence: "unvalidated"` because the measured scale did
not match the notation — anchors read madd_2 = 2.25, madd_4 = 6.20, madd_6 = 12.0 own-counts against
a nominal 2 : 4 : 6. The tasāwī CV (is every instance held *equally*) was trustworthy; the absolute
number was not, so "you gave 2 counts, it requires 4" could not be said.

The rule-instance join makes this fittable. The parser states `expected_harakat` for every located
madd, and the engine measures that instance's duration in the reciter's own counts, so the corpus
gives thousands of (nominal, measured) pairs on reciters we already trust. This collects them from
the anchors and writes them for the robust fit in
`substrate_library/julia/madd_scale.jl` (the numerics stay in Julia).

    .venv/bin/python research_agency_lab/experiments/calibration/madd_scale.py DUMP_DIR OUT.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research_agency_lab/experiments/learner_eval"))

from app.analysis import analyse_clip  # noqa: E402
from app.rule_bind import DURATIONAL, bind  # noqa: E402
from app.tajweed_rules.parser import TajweedParser  # noqa: E402

ANCHORS = {"Husary_Muallim_128kbps", "Husary_128kbps", "Husary_128kbps_Mujawwad"}


def main(argv: list[str]) -> int:
    dump, out = Path(argv[0]), Path(argv[1])
    only_anchors = "--all" not in argv
    lay = json.loads((dump / "layout.json").read_text())
    ph = next(l for l in lay["levels"] if l["level"] == "phonemes")
    vocab = {t: i for i, t in enumerate(ph["vocab"]) if len(t) == 1}
    blocks = {l["level"]: (l["first"], l["width"], l["vocab"])
              for l in lay["levels"] if l["level"] != "phonemes"}
    parser = TajweedParser()

    from muaalem_eval import MOSHAF
    from quran_transcript import quran_phonetizer
    import muaalem_dump as md

    n = 0
    rows = 0
    with out.open("w") as fh:
        for line in (dump / "index.jsonl").open():
            rec = json.loads(line)
            if "file" not in rec or "uthmani" not in rec:
                continue
            spk = str(rec.get("speaker", ""))
            if only_anchors and spk not in ANCHORS:
                continue
            try:
                r = quran_phonetizer(rec["uthmani"], MOSHAF, remove_spaces=True)
                word_ph = md.word_spans(rec["uthmani"], r.mappings)
                lp = np.fromfile(dump / rec["file"], dtype="<f4").reshape(rec["frames"], lay["columns"])
                units = analyse_clip(lp, r.phonemes, vocab, lay["blank"], ph["first"], ph["width"],
                                     blocks, None)
                bounds = bind(parser.parse(rec["uthmani"]), r.phonemes, word_ph)
            except Exception:  # noqa: BLE001 - a malformed ayah must not stop the sweep
                continue
            for b in bounds:
                rt = b.rule_type
                if b.expected_counts is None or not b.unit_indices:
                    continue
                if not any(d.value == rt for d in DURATIONAL):
                    continue
                us = [units[i] for i in b.unit_indices if i < len(units)]
                if not us or any(u.duration_counts is None for u in us):
                    continue
                fh.write(json.dumps({
                    "speaker": spk, "rule": rt, "shadda": b.shadda,
                    "nominal": (b.expected_counts[0] + b.expected_counts[1]) / 2,
                    "measured": round(sum(u.duration_counts for u in us), 4),
                    "run": sum(u.run_length for u in us),
                }) + "\n")
                rows += 1
            n += 1
            if n % 200 == 0:
                print(f"{n} clips, {rows} instances", file=sys.stderr)
    print(f"{rows} durational instances from {n} clips -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
