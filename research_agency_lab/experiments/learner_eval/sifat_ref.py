#!/usr/bin/env python3
"""Expected sifat labels per reference phoneme, for the sifat GOP layer.

muaalem emits ten *sifat* CTC heads besides the phoneme head (columns 43–74 of every dump), and
``quran_transcript``'s phonetizer states, for each reference phoneme, which value of each sifah the
reciter is *supposed* to realise. Together those give a per-unit likelihood-ratio test for every
tajweed attribute — hams/jahr, shidda/rakhawa, tafkheem/tarqeeq, itbaq, safeer, qalqala, tikraar,
tafashie, istitala, ghunnah — with no hand-written DSP detector in the path.

This writes ``sifat.jsonl`` next to a dump: one record per clip holding, for each level, the expected
class id (the model's own column index within that level) for every character of ``ref_ph``.

    .venv/bin/python research_agency_lab/experiments/learner_eval/sifat_ref.py DUMP_DIR
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from muaalem_eval import MOSHAF  # noqa: E402  (applies the transformers shim)
from quran_transcript import quran_phonetizer  # noqa: E402

LEVELS = ["ghonna", "hams_or_jahr", "istitala", "itbaq", "qalqla", "safeer",
          "shidda_or_rakhawa", "tafashie", "tafkheem_or_taqeeq", "tikraar"]


def class_maps(model: str = "obadx/muaalem-model-v3_2") -> dict[str, dict[str, int]]:
    """level -> {english sifah value -> class id}, taken from the model's own tokenizer."""
    from quran_muaalem.inference import MultiLevelTokenizer

    t = MultiLevelTokenizer(model)
    out: dict[str, dict[str, int]] = {}
    for lvl, ids in t.sifat_level_to_id_to_en_vocab.items():
        out[lvl] = {name: int(i) for i, name in ids.items() if name != "[PAD]"}
    return out


def per_char(uthmani: str, maps: dict[str, dict[str, int]]) -> tuple[str, dict[str, list[int]]]:
    """(phoneme string, level -> expected class id for each of its characters)."""
    r = quran_phonetizer(uthmani, MOSHAF, remove_spaces=True)
    cols: dict[str, list[int]] = {lvl: [] for lvl in LEVELS}
    for entry in r.sifat:
        n = len(entry.phonemes)
        for lvl in LEVELS:
            cols[lvl].extend([maps[lvl].get(getattr(entry, lvl), 0)] * n)
    return r.phonemes, cols


def main(argv: list[str]) -> int:
    dump = Path(argv[0])
    maps = class_maps()
    out = dump / "sifat.jsonl"
    n = skipped = 0
    with out.open("w") as fh:
        for line in (dump / "index.jsonl").open():
            rec = json.loads(line)
            if "file" not in rec or "uthmani" not in rec:
                continue
            ph, cols = per_char(rec["uthmani"], maps)
            if ph != rec["ref_ph"]:  # phonetizer drift would misalign every label
                skipped += 1
                continue
            fh.write(json.dumps({"id": rec["id"], "n": len(ph), "levels": cols}) + "\n")
            n += 1
    print(f"sifat.jsonl: {n} clips ({skipped} skipped on phoneme mismatch) -> {out}")
    print("levels:", {lvl: maps[lvl] for lvl in LEVELS[:2]}, "...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
