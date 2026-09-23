"""Dump muaalem-v3.2 frame posteriors once, so every lahn decision rule can be designed offline.

Two sources, one output format (``OUT_DIR/<id>.npz`` + one line per clip in ``OUT_DIR/index.jsonl``):

* ``quranmb``   — the 1642 QuranMB.v2 learner clips (planted errors, phone ground truth)
* ``everyayah`` — master recitations: ``~/.cache/qaari-eval/everyayah/<reciter>/SSSAAA.mp3``,
  whole ayahs, the reference text taken from the mushaf (correct by construction)
* ``qdc``       — quran.com surah recordings cut per verse by ``datasets/qaari_keys/fetch.py``
  (``~/.cache/qaari-eval/qdc/<reciter_id>/SSSAAA.wav``), which carry open word timestamps

``--tier T10`` restricts the master sources to the verses of that QaariKeys tier.

Each clip is one raw little-endian float32 file ``<id>.f32``: the log-posteriors of every level
(``phonemes`` 43 columns, then the ten sifat levels) concatenated column-wise, stored row-major
T × C, so Julia (``read!``) and Octave (``fread``) load it without extra packages. The column layout
(level, first column, width) and the vocabularies are written once to ``OUT_DIR/layout.json``; the
index line holds the ids, Uthmani text, Hafs reference phonemes, T and (for QuranMB) the
ground-truth edits. Resumable: clips already in the index are skipped.

    .venv/bin/python research_agency_lab/experiments/learner_eval/muaalem_dump.py quranmb OUT_DIR
    .venv/bin/python research_agency_lab/experiments/learner_eval/muaalem_dump.py everyayah OUT_DIR [--reciters A,B] [--tier T10]
    .venv/bin/python research_agency_lab/experiments/learner_eval/muaalem_dump.py qdc OUT_DIR --tier T10
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from muaalem_eval import MOSHAF, Muaalem, locate, to_16k  # noqa: E402  (muaalem_eval applies the transformers shim)

from quran_transcript import Aya, quran_phonetizer  # noqa: E402

EVERYAYAH = Path.home() / ".cache/qaari-eval/everyayah"
QDC = Path.home() / ".cache/qaari-eval/qdc"
_PARENTS = Path(__file__).resolve().parents
# only the repo checkout has a grandparent tree; on Modal this file sits alone in /root/lab
TIERS = (_PARENTS[3] / "datasets/qaari_keys/build/tiers.json") if len(_PARENTS) > 3 else None


def tier_verses(tier: str) -> set[tuple[int, int]] | None:
    if not tier:
        return None
    if TIERS is None or not TIERS.is_file():
        raise SystemExit(f"tiers.json not available next to {__file__}; pass explicit items instead of --tier")
    t = next(x for x in json.loads(TIERS.read_text())["tiers"] if x["name"] == tier)
    return {(v["surah"], v["ayah"]) for v in t["verses"]}


def word_spans(uthmani: str, mappings: list) -> list[list[int]]:  # type: ignore[type-arg]
    """[p0, p1) phoneme span of every Uthmani word, from the phonetizer's per-character mappings."""
    spans: list[list[int]] = [[]]
    for ch, m in zip(uthmani, mappings):
        if ch == " ":
            spans.append([])
            continue
        if m is not None and not getattr(m, "deleted", False) and m.pos[1] > m.pos[0]:
            spans[-1].extend(m.pos)
    return [[min(x), max(x)] if x else [-1, -1] for x in spans]


def quranmb_items() -> Iterator[tuple[str, dict[str, object], bytes | Path, str]]:
    import pyarrow.parquet as pq

    from quranmb_eval import PARQUET, phone_edits  # imports the engine; only QuranMB runs need it
    pf = pq.ParquetFile(PARQUET)
    for rg in range(pf.num_row_groups):
        for row in pf.read_row_group(rg).to_pylist():
            meta = {"speaker": row["id"].split("_")[0], "text": row["reference_arabic_string"],
                    "match_type": row["match_type"],
                    "edits": phone_edits(row["reference_phoneme_string"], row["annotation_phoneme_string"])}
            yield row["id"], meta, row["audio"]["bytes"], ""


def master_items(root: Path, pattern: str, reciters: list[str] | None,
                 verses: set[tuple[int, int]] | None) -> Iterator[tuple[str, dict[str, object], bytes | Path, str]]:
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if reciters and d.name not in reciters:
            continue
        for f in sorted(d.glob(pattern)):
            if len(f.stem) != 6:
                continue  # whole-surah files next to the cut clips
            sura, aya = int(f.stem[:3]), int(f.stem[3:])
            if verses is not None and (sura, aya) not in verses:
                continue
            uthmani = Aya(sura, aya).get().uthmani
            yield f"{root.name}:{d.name}/{f.stem}", {"speaker": d.name, "sura": sura, "aya": aya}, f, uthmani


def load_16k(src: bytes | Path) -> np.ndarray:
    if isinstance(src, bytes):
        return to_16k(src)
    import librosa
    x, _ = librosa.load(str(src), sr=16000, mono=True)
    return x.astype(np.float32)


@torch.no_grad()
def posteriors(model: Muaalem, wave: np.ndarray) -> dict[str, np.ndarray]:
    """Log-softmax of every CTC level, float32, T × vocab."""
    feats = model.processor([wave], sampling_rate=16000, return_tensors="pt")
    feats = {k: v.to(model.device, dtype=model.dtype) for k, v in feats.items()}
    outs = model.model(**feats, return_dict=False)[0]
    return {lvl: torch.log_softmax(outs[lvl][0].float(), dim=-1).cpu().numpy().astype(np.float32) for lvl in outs}


def _level_order(level: str) -> tuple[int, str]:
    return (level != "phonemes", level)


def write_layout(path: Path, model: Muaalem, lp: dict[str, np.ndarray]) -> None:
    cols, c0 = [], 0
    for lvl in sorted(lp, key=_level_order):
        w = int(lp[lvl].shape[1])
        vocab = model.multi_level_tokenizer.id_to_vocab[lvl]
        cols.append({"level": lvl, "first": c0, "width": w, "vocab": [vocab.get(i, "") for i in range(w)]})
        c0 += w
    path.write_text(json.dumps({"columns": c0, "blank": 0, "dtype": "float32-le", "order": "row-major T x C",
                                "levels": cols}, ensure_ascii=False, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=["quranmb", "everyayah", "qdc"])
    ap.add_argument("--tier", default="")
    ap.add_argument("out_dir")
    ap.add_argument("--reciters", default="")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    index, layout = out / "index.jsonl", out / "layout.json"
    done = {json.loads(line)["id"] for line in index.open()} if index.exists() else set()
    model = Muaalem(device=args.device, dtype=torch.float32)
    reciters = [r for r in args.reciters.split(",") if r] or None
    if args.source == "quranmb":
        items = quranmb_items()
    elif args.source == "everyayah":
        items = master_items(EVERYAYAH, "*.mp3", reciters, tier_verses(args.tier))
    else:
        items = master_items(QDC, "*.wav", reciters, tier_verses(args.tier))
    n, t0 = 0, time.time()
    with index.open("a") as fh:
        for cid, meta, src, uthmani in items:
            if cid in done:
                continue
            if args.limit and n >= args.limit:
                break
            rec: dict[str, object] = {"id": cid, "source": args.source, **meta}
            try:
                uthmani = uthmani or locate(str(meta["text"]))
                ref = quran_phonetizer(uthmani, MOSHAF, remove_spaces=True)
                wave = load_16k(src)
                lp = posteriors(model, wave)
                if not layout.exists():
                    write_layout(layout, model, lp)
                fname = cid.replace("/", "__").replace(":", "__") + ".f32"
                np.concatenate([lp[k] for k in sorted(lp, key=_level_order)], axis=1).astype("<f4").tofile(out / fname)
                rec.update({"uthmani": uthmani, "ref_ph": ref.phonemes, "file": fname,
                            "frames": int(lp["phonemes"].shape[0]), "duration_s": round(len(wave) / 16000, 4),
                            "word_ph": word_spans(uthmani, ref.mappings)})
            except Exception as exc:  # noqa: BLE001
                rec["error"] = f"{type(exc).__name__}: {exc}"[:300]
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            fh.flush()
            n += 1
            if n % 25 == 0:
                print(f"{n} clips, {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
