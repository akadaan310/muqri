"""Every consonant the 41 reciters produced, with what all ten characteristic heads heard in it.

The input to the calculus of characteristics (calculus.jl). For each consonant unit of every T300 clip:

* the letter, its run length (2+ = shaddah), what follows it (a vowel and which, a madd, nothing = sakin),
  the vowel before it, whether it ends the clip (the stop), the reciter;
* for each of the ten heads, the posterior probability of every class, pooled over the unit's
  frames exactly as the serving path pools them (`app.analysis._pooled`) and renormalised without
  [PAD]; 22 classes in all;
* the class the phonetizer expects there -- already context-resolved (tafkhim follows the vowel,
  ghunnah the rule), i.e. what the books say this letter should carry in this place.

Written as raw little-endian arrays plus a JSON header, which Julia reads with no dependencies.

    .venv/bin/python research_agency_lab/experiments/calculus/extract.py [procs]
"""

from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from app.analysis import MADD, SHORT_V, _pooled, ctc_viterbi  # noqa: E402
from app.rule_bind import ph_units  # noqa: E402
from datastore.ingest import ladder  # noqa: E402

DUMP = ROOT / "research_agency_lab/experiments/qaari_keys/modal_T300"
OUT = ROOT / "research_agency_lab/experiments/calculus/data"
HEADS = ["ghonna", "hams_or_jahr", "istitala", "itbaq", "qalqla", "safeer", "shidda_or_rakhawa",
         "tafashie", "tafkheem_or_taqeeq", "tikraar"]
LETTERS = "ءبتثجحخدذرزسشصضطظعغفقكلمنهوي"
NEXT = {"َ": 1, "ُ": 2, "ِ": 3}            # 0 = sakin, 4 = madd letter, 5 = end of clip

_lay = _ph = _heads = _sif = None


def _init() -> None:
    global _lay, _ph, _heads, _sif
    _lay = json.loads((DUMP / "layout.json").read_text())
    _ph = next(lv for lv in _lay["levels"] if lv["level"] == "phonemes")
    _heads = [next(lv for lv in _lay["levels"] if lv["level"] == h) for h in HEADS]
    _sif = {}
    for line in open(DUMP / "sifat.jsonl"):
        x = json.loads(line)
        _sif[x["id"]] = x["levels"]


def _clip(r: dict):  # type: ignore[no-untyped-def]
    ref = _sif.get(r["id"])  # type: ignore[union-attr]
    if ref is None:
        return None
    full = np.fromfile(DUMP / r["file"], dtype="<f4").reshape(r["frames"], _lay["columns"])  # type: ignore[index]
    lp = full[:, _ph["first"]:_ph["first"] + _ph["width"]]  # type: ignore[index]
    vocab = {t: i for i, t in enumerate(_ph["vocab"]) if len(t) == 1}  # type: ignore[index]
    ph = r["ref_ph"]
    if not all(c in vocab for c in ph):
        return None
    _s, first, last = ctc_viterbi(lp, [vocab[c] for c in ph], _lay["blank"])  # type: ignore[index]
    units = ph_units(ph)
    meta, probs, expect = [], [], []
    for i, (sym, a, b) in enumerate(units):
        if sym not in LETTERS:
            continue
        t0, t1 = first[a] - 1, last[b]
        if not 0 <= t0 < t1 <= full.shape[0]:
            continue
        nxt = units[i + 1][0] if i + 1 < len(units) else None
        nk = 5 if nxt is None else NEXT.get(nxt, 4 if nxt in MADD else 0)
        prv = units[i - 1][0] if i > 0 else None
        pk = NEXT.get(prv, 4 if prv in MADD else 0) if prv else 5
        row_p, row_e = [], []
        for h, lv in zip(HEADS, _heads):
            s = _pooled(full[t0:t1, lv["first"]:lv["first"] + lv["width"]])[1:]   # drop [PAD]
            p = np.exp(s - s.max())
            row_p.extend(p / p.sum())
            ids = ref.get(h, [])
            row_e.append(ids[a] - 1 if a < len(ids) and ids[a] > 0 else -1)
        meta.append([LETTERS.index(sym), b - a + 1, nk, pk, int(nxt is None), t1 - t0])
        probs.append(row_p)
        expect.append(row_e)
    return r["speaker"], r["id"], meta, probs, expect


def main(procs: int = 2) -> None:
    _init()
    recs = [json.loads(line) for line in open(DUMP / "index.jsonl")]
    recs = [r for r in recs if "file" in r]
    OUT.mkdir(parents=True, exist_ok=True)
    speakers: list[str] = []
    clips: list[str] = []
    M, P, E, S, C = [], [], [], [], []
    with Pool(procs, initializer=_init) as pool:
        for k, res in enumerate(pool.imap(_clip, recs, chunksize=8)):
            if not res:
                continue
            spk, cid, meta, probs, expect = res
            if spk not in speakers:
                speakers.append(spk)
            clips.append(cid)
            M += meta
            P += probs
            E += expect
            S += [speakers.index(spk)] * len(meta)
            C += [len(clips) - 1] * len(meta)
            if k % 2000 == 0:
                print(f"{k}/{len(recs)} clips, {len(M)} consonants", flush=True)
    np.asarray(M, dtype="<i4").tofile(OUT / "meta.i32")
    np.asarray(P, dtype="<f4").tofile(OUT / "probs.f32")
    np.asarray(E, dtype="<i4").tofile(OUT / "expect.i32")
    np.asarray(S, dtype="<i4").tofile(OUT / "speaker.i32")
    np.asarray(C, dtype="<i4").tofile(OUT / "clip.i32")
    head_classes = []
    for lv in (next(x for x in json.loads((DUMP / "layout.json").read_text())["levels"] if x["level"] == h)
               for h in HEADS):
        head_classes.append(lv["vocab"][1:])
    (OUT / "header.json").write_text(json.dumps({
        "n": len(M), "meta_cols": ["letter", "run_length", "next", "prev", "at_end", "frames"],
        "next_codes": {"0": "sakin", "1": "fatha", "2": "damma", "3": "kasra", "4": "madd", "5": "end"},
        "letters": LETTERS, "heads": HEADS, "head_classes": head_classes,
        "speakers": speakers, "tiers": [ladder(s) for s in speakers], "clips": clips,
    }, ensure_ascii=False, indent=1))
    print(f"{len(M)} consonant units from {len(clips)} clips, {len(speakers)} reciters -> {OUT}")


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:]))
