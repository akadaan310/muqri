"""Run the full engine on QuranMB.v2 (IqraEval): 18 non-master readers, planted mistakes, phone GT.

For every clip: the perfection / sifaat index, coverage, status counts, every lahn verdict, and the
ground-truth edit list between the reference and the annotated (actually produced) phones. This is
the local "non-master" side of the master-vs-learner comparison; the master side is studio-all.

    .venv/bin/python research_agency_lab/experiments/learner_eval/quranmb_eval.py OUT.jsonl [--limit N] [--start K]
"""

from __future__ import annotations

import argparse
import difflib
import io
import json
import sys
import tempfile
import time
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from app.audio import load_audio  # noqa: E402
from app.pipeline import AnalysisOptions, QaariEvaluator  # noqa: E402

PARQUET = Path.home() / ".cache/qaari-eval/datasets/quranmb_v2_full/data/train-00000-of-00001.parquet"


def phone_edits(ref: str, ann: str) -> list[dict[str, object]]:
    """Substitutions / insertions / deletions between the reference and the produced phones."""
    r, a = ref.split(), ann.split()
    out: list[dict[str, object]] = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=r, b=a, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        out.append({"op": op, "ref_pos": i1, "ref": r[i1:i2], "ann": a[j1:j2]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()
    ev = QaariEvaluator(AnalysisOptions(aligner="ctc", compute_fingerprint=False))
    pf = pq.ParquetFile(PARQUET)
    done = 0
    t0 = time.time()
    with open(args.out, "a") as fh:
        idx = -1
        for rg in range(pf.num_row_groups):
            for row in pf.read_row_group(rg).to_pylist():
                idx += 1
                if idx < args.start:
                    continue
                if args.limit and done >= args.limit:
                    break
                rec: dict[str, object] = {"id": row["id"], "speaker": row["id"].split("_")[0],
                                          "text": row["reference_arabic_string"], "match_type": row["match_type"],
                                          "edits": phone_edits(row["reference_phoneme_string"],
                                                               row["annotation_phoneme_string"]),
                                          "n_ref_phones": len(row["reference_phoneme_string"].split())}
                try:
                    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
                        tmp.write(io.BytesIO(row["audio"]["bytes"]).read())
                        tmp.flush()
                        audio = load_audio(tmp.name, denoise="never")
                    res = ev.analyze_signal(audio, row["reference_arabic_string"])
                    s = res.report["recitation_summary"]
                    rec.update({
                        "perfection": s["tajweed_perfection_index"], "sifaat": s["sifaat_score"],
                        "coverage": s["coverage"], "status_counts": s["status_counts"],
                        "category_scores": s["category_scores"],
                        "lahn": [{"rule": d.rule_type.value, "letter": d.letter, "word": d.word, "detail": d.detail,
                                  "llr": d.metrics.get("gop_llr"), "status": d.status.value,
                                  "unit": i}
                                 for i, d in enumerate(res.diagnostics) if d.rule_type.value.startswith("lahn")],
                        "rules": [{"rule": d.rule_type.value, "detail": d.detail, "status": d.status.value,
                                   "score": d.score}
                                  for d in res.diagnostics if not d.rule_type.value.startswith("lahn")],
                    })
                except Exception as exc:  # noqa: BLE001
                    rec["error"] = f"{type(exc).__name__}: {exc}"[:300]
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                done += 1
                if done % 25 == 0:
                    print(f"{done} clips, {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
