"""muaalem-v3.2 (obadx, w2v-BERT 2.0 multi-level CTC) on QuranMB.v2 — the second judge, run locally.

Per clip: locate the verse span from the plain text (``quran_transcript.search``), phonetise the
Uthmani reference for Hafs, decode the produced phonemes *freely* (no forcing to the text), and let
``explain_error`` list the recitation errors. Rows share the ground truth format of
``quranmb_eval.py`` so ``analyze_muaalem.py`` can score both engines on the same clips.

    .venv/bin/python research_agency_lab/experiments/learner_eval/muaalem_eval.py OUT.jsonl [--limit N]
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
import transformers.models.wav2vec2_bert.modeling_wav2vec2_bert as _w2vb

_w2vb._HIDDEN_STATES_START_POSITION = 2  # removed in transformers 5; quran-muaalem 0.2.2 still imports it

from quran_muaalem import Muaalem  # noqa: E402
import unicodedata  # noqa: E402

from quran_transcript import Aya, MoshafAttributes, WordSpan, explain_error, quran_phonetizer  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

MOSHAF = MoshafAttributes(rewaya="hafs", madd_monfasel_len=4, madd_mottasel_len=4, madd_mottasel_waqf=4,
                          madd_aared_len=4)


def _skel(word: str) -> str:
    base = "".join(c for c in unicodedata.normalize("NFD", word) if unicodedata.category(c) != "Mn")
    return base.translate(str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه", "ـ": None}))


_INDEX: list[tuple[int, int, list[str]]] = []


def locate(text: str) -> str:
    """Uthmani script of the verse span whose imlaey words match ``text`` (diacritics ignored)."""
    if not _INDEX:
        a = Aya(1, 1)
        for _ in range(6236):
            g = a.get()
            _INDEX.append((g.sura_idx, g.aya_idx, [_skel(w) for w in g.imlaey_words]))
            a = a.step(1)
    q = [_skel(w) for w in text.split()]
    for sura, aya, words in _INDEX:
        for i in range(len(words) - len(q) + 1):
            if words[i:i + len(q)] == q:
                return Aya(sura, aya).imlaey_to_uthmani(WordSpan(start=i, end=i + len(q)))
    raise LookupError("verse span not found")


def to_16k(wav_bytes: bytes) -> np.ndarray:
    x, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != 16000:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=16000)
    return x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--threads", type=int, default=3)
    args = ap.parse_args()
    from quranmb_eval import PARQUET, phone_edits  # the engine import is only needed for QuranMB runs
    torch.set_num_threads(args.threads)
    model = Muaalem(device="cpu", dtype=torch.float32)
    pf = pq.ParquetFile(PARQUET)
    done, t0, idx = 0, time.time(), -1
    with open(args.out, "a") as fh:
        for rg in range(pf.num_row_groups):
            for row in pf.read_row_group(rg).to_pylist():
                idx += 1
                if idx < args.start:
                    continue
                if args.limit and done >= args.limit:
                    return 0
                rec: dict[str, object] = {"id": row["id"], "speaker": row["id"].split("_")[0],
                                          "text": row["reference_arabic_string"],
                                          "edits": phone_edits(row["reference_phoneme_string"],
                                                               row["annotation_phoneme_string"])}
                try:
                    uthmani = locate(row["reference_arabic_string"])
                    ref = quran_phonetizer(uthmani, MOSHAF, remove_spaces=True)
                    out = model([to_16k(row["audio"]["bytes"])], [ref], sampling_rate=16000)[0]
                    errs = explain_error(uthmani, ref.phonemes, out.phonemes.text, ref.mappings)
                    rec.update({"uthmani": uthmani, "ref_ph": ref.phonemes, "pred_ph": out.phonemes.text,
                                "errors": [{"type": e.error_type, "speech": e.speech_error_type,
                                            "expected": e.expected_ph, "predicted": e.preditected_ph,
                                            "rules": [str(r) for r in (e.ref_tajweed_rules or [])]} for e in errs]})
                except Exception as exc:  # noqa: BLE001
                    rec["error"] = f"{type(exc).__name__}: {exc}"[:300]
                fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                fh.flush()
                done += 1
                if done % 10 == 0:
                    print(f"{done} clips, {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
