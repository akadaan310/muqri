"""Head-to-head on QuranMB.v2 planted errors: our CTC GOP detector vs muaalem-v3.2.

Ground truth: letter substitutions between QuranMB's reference and annotated phones (e.g. g→x = غ→خ).
  recall    — share of GT letter substitutions the engine names correctly (expected → produced letter)
  flags     — letter-error reports per clip on clips with no GT letter substitution (false-alarm proxy)
  clip AUROC — clips with vs without a GT letter substitution, scored by the number of letter reports

    .venv/bin/python research_agency_lab/experiments/learner_eval/head_to_head.py GOP.jsonl MUAALEM.jsonl [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_learners import auroc  # noqa: E402

# IqraEval (QuranMB) phone -> Arabic letter; doubled phones are geminates of the same letter
PHONE_AR = {"<": "ء", "b": "ب", "t": "ت", "^": "ث", "j": "ج", "H": "ح", "x": "خ", "d": "د", "*": "ذ", "r": "ر",
            "z": "ز", "s": "س", "$": "ش", "S": "ص", "D": "ض", "T": "ط", "Z": "ظ", "E": "ع", "g": "غ", "f": "ف",
            "q": "ق", "k": "ك", "l": "ل", "m": "م", "n": "ن", "h": "ه", "w": "و", "y": "ي"}
# our romanised GOP alternatives -> Arabic letter
OURS_AR = {"'": "ء", "ʿ": "ع", "kh": "خ", "gh": "غ", "ḥ": "ح", "ṣ": "ص", "ḍ": "ض", "ṭ": "ط", "ẓ": "ظ", "dh": "ذ",
           "th": "ث", "q": "ق", "k": "ك", "s": "س", "t": "ت", "d": "د", "z": "ز", "h": "ه"}


def letter(phone: str) -> str | None:
    return PHONE_AR.get(phone[0]) if phone else None


def base_letters(s: str) -> set[str]:
    return {c for c in unicodedata.normalize("NFD", s) if "ء" <= c <= "ي"}


def gt_letter_subs(edits: list[dict]) -> list[tuple[str, str]]:  # type: ignore[type-arg]
    out = []
    for e in edits:
        if e["op"] != "replace" or len(e["ref"]) != 1 or len(e["ann"]) != 1:
            continue
        a, b = letter(e["ref"][0]), letter(e["ann"][0])
        if a and b and a != b:
            out.append((a, b))
    return out


def gop_reports(r: dict) -> list[tuple[str, str]]:  # type: ignore[type-arg]
    return [(x["letter"], OURS_AR.get(x["detail"].split(">")[1], "?")) for x in r.get("lahn", [])
            if x["rule"] == "lahn_letter" and x["status"] != "PASS"]


def muaalem_reports(r: dict) -> list[tuple[str, str]]:  # type: ignore[type-arg]
    out = []
    for e in r.get("errors", []):
        if e["type"] != "normal":
            continue
        exp, pred = base_letters(e["expected"]), base_letters(e["predicted"])
        for a in exp - pred:
            for b in (pred - exp) or {"∅"}:
                out.append((a, b))
    return out


def score(rows: dict[str, dict], reports) -> dict[str, float]:  # type: ignore[no-untyped-def,type-arg]
    hit = n = 0
    counts, has = [], []
    for r in rows.values():
        gt = gt_letter_subs(r["edits"])
        rep = reports(r)
        for a, b in gt:
            n += 1
            hit += int((a, b) in rep)
        counts.append(len(rep))
        has.append(bool(gt))
    c, h = np.array(counts, dtype=float), np.array(has)
    return {"gt_letter_subs": n, "recall": round(hit / n, 3) if n else float("nan"),
            "reports_per_clean_clip": round(float(c[~h].mean()), 2),
            "reports_per_error_clip": round(float(c[h].mean()), 2), "clip_auroc": round(auroc(c[h], c[~h]), 3)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gop")
    ap.add_argument("muaalem")
    ap.add_argument("--json")
    args = ap.parse_args()
    gop = {r["id"]: r for r in map(json.loads, open(args.gop)) if "error" not in r}
    mu = {r["id"]: r for r in map(json.loads, open(args.muaalem)) if "error" not in r}
    common = sorted(set(gop) & set(mu))
    out = {"clips": len(common),
           "gop_ctc": score({k: gop[k] for k in common}, gop_reports),
           "muaalem": score({k: mu[k] for k in common}, muaalem_reports)}
    text = json.dumps(out, ensure_ascii=False, indent=1)
    if args.json:
        Path(args.json).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
