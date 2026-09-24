"""sobolev210/quran-recitation-errors: learners' ayahs, reviewed word by word.

Each chunk is one whole ayah recorded by a learner; a reviewer tagged the words read wrongly with
one or more of Letters / Tajweed / Wording / Tashkeel. Only the Hafs chunks are scored (451 of 1,042;
the rest are Qalun, which this engine does not grade). The labels are the reviewer's, and a reviewer
marks what they noticed -- so an untagged word the engine flags is a candidate false alarm, not a
proven one.

    python -m research_agency_lab.experiments.external.sobolev dump      # posteriors (local CPU)
    python -m research_agency_lab.experiments.external.sobolev grade     # engine from the dump
    python -m research_agency_lab.experiments.external.sobolev compare   # engine vs reviewer, per word

Data: https://huggingface.co/datasets/sobolev210/quran-recitation-errors -> DATA
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

DATA = Path.home() / "data/sprint4/hf/sobolev210__quran-recitation-errors"
DUMP = Path.home() / "data/sprint4/dumps/sobolev"
HERE = Path(__file__).resolve().parent
OUT = HERE / "sobolev_engine.jsonl"
_LETTER = re.compile(r"[ء-ي]")


def clips() -> list[dict[str, Any]]:
    rs = [json.loads(l) for l in (DATA / "metadata.jsonl").open()]
    out = []
    for r in rs:
        if r["riwayah"] != "Hafs":
            continue
        cid = r["file_name"].split("chunks_recording_", 1)[1].replace(".wav", "")
        out.append({"id": cid, "path": DATA / r["file_name"], "verses": [[int(r["surah"]), int(r["ayah"])]],
                    "text": r["text"], "errors": r["errors"], "reviewer": r["reviewer_id"],
                    "recording": r["recording_id"]})
    return out


def word_labels(c: dict[str, Any], n_words: int) -> dict[int, list[str]] | None:
    """Reviewer tags by word index of the Uthmani ayah; None when the imla'i text cannot be lined up
    word for word (e.g. يا أيها written as two words where the mushaf has one)."""
    toks = [t for t in c["text"].split() if _LETTER.search(t)]
    if len(toks) != n_words:
        return None
    out: dict[int, list[str]] = {}
    for w, tags in c["errors"].items():
        hits = [i for i, t in enumerate(toks) if t == w]
        if not hits:
            return None
        for i in hits:                     # a repeated word: every occurrence carries the tag
            out.setdefault(i, []).extend(tags)
    return out


def dump() -> None:
    from research_agency_lab.experiments.external.dump import dump as go
    print(go(clips(), DUMP), "clips dumped ->", DUMP)


def grade() -> None:
    from research_agency_lab.experiments.external.dump import grade as go
    from research_agency_lab.experiments.external.runner import reduce
    n = 0
    with OUT.open("w") as fh:
        for rec, rep in go(DUMP):
            row = {"id": rec["id"], "verses": rec["verses"], **(reduce(rep) if "error" not in rep else rep)}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(n, "graded ->", OUT)


def compare() -> dict[str, Any]:
    from research_agency_lab.experiments.external.qdat_bench import auc
    meta = {c["id"]: c for c in clips()}
    eng = {r["id"]: r for r in map(json.loads, OUT.open()) if "measurements" in r}
    words: list[dict[str, Any]] = []
    unmapped = 0
    for cid, r in eng.items():
        m = r["measurements"]
        lab = word_labels(meta[cid], len(m["words"]))
        if lab is None:
            unmapped += 1
            continue
        L = {}
        for l in m["letters"]:
            L.setdefault(l["word"], []).append(l)
        for k, w in enumerate(m["words"]):
            ls = L.get(k, [])
            idm = [l["identity"]["margin"] for l in ls if l["identity"]["margin"] is not None]
            chm = [c["margin"] for l in ls for c in l["characteristics"].values() if c["scored"] and c["margin"] is not None]
            words.append({"clip": cid, "word": k, "tags": lab.get(k, []), "flag": not w["all_correct"],
                          "failing": w["failing"], "min_identity_margin": min(idm) if idm else None,
                          "min_char_margin": min(chm) if chm else None})
    res: dict[str, Any] = {"clips_graded": len(eng), "clips_unmapped": unmapped, "words": len(words),
                           "words_tagged": sum(bool(w["tags"]) for w in words)}
    clean = [w for w in words if not w["tags"]]
    res["flag_rate_untagged"] = round(sum(w["flag"] for w in clean) / len(clean), 4) if clean else None
    by: dict[str, Any] = {}
    for tag in ("any", "Letters", "Tajweed", "Wording", "Tashkeel"):
        pos = [w for w in words if w["tags"] and (tag == "any" or tag in w["tags"])]
        if not pos:
            continue
        # the engine's own word verdict, and two threshold-free separations (lower margin = worse)
        by[tag] = {"n": len(pos), "recall": round(sum(w["flag"] for w in pos) / len(pos), 4),
                   "auc_min_identity_margin": auc([-w["min_identity_margin"] for w in pos if w["min_identity_margin"] is not None],
                                                  [-w["min_identity_margin"] for w in clean if w["min_identity_margin"] is not None]),
                   "auc_min_char_margin": auc([-w["min_char_margin"] for w in pos if w["min_char_margin"] is not None],
                                              [-w["min_char_margin"] for w in clean if w["min_char_margin"] is not None])}
    res["by_tag"] = by
    res["missed"] = [{k: w[k] for k in ("clip", "word", "tags")} for w in words if w["tags"] and not w["flag"]]
    return res


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "compare":
        r = compare()
        (HERE / "sobolev_compare.json").write_text(json.dumps(r, indent=1, ensure_ascii=False))
        print(json.dumps({k: v for k, v in r.items() if k != "missed"}, indent=1, ensure_ascii=False))
    else:
        {"dump": dump, "grade": grade}[cmd]()
