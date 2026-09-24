"""Blind spots of the acoustic model, served: for every check on every letter, the probability that a
professional reciter would "fail" it in this context -- learned from all 41 T300 reciters
(substrate_library/julia/blindspots.jl). Above TAU the check cannot be trusted here: it is reported,
not scored. The context is the same as the fit's: the letter, the two sounds before, the sound after,
voweled / saakin / doubled / stop, position in the word, ayah-final word, and the letter x previous
and letter x next pairs.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

MODEL = Path(__file__).resolve().parents[1] / "research_agency_lab/experiments/quran/blindspots.json"


@lru_cache(maxsize=1)
def model() -> dict[str, Any]:
    if not MODEL.is_file():
        return {}
    d = json.loads(MODEL.read_text())
    return {"tau": d["tau"], "features": d["features"],
            "coef": {h: v["coef"] for h, v in d["checks"].items()},
            # the second layer: word positions where the masters fail a check >= 30 % (n >= 10) that
            # the context features do not capture (e.g. the ta' of صِرَٰطٍ: 40 of 40 master recitations)
            "words": {w["key"]: w["rate"] for w in d.get("words", [])}}


def features(letters: list[dict[str, Any]], i: int, words: dict[int, int], last_word: int) -> list[str]:
    """The fit's context of letter i (letters of one ayah, in order; words: letters per word index)."""
    from app.letter_matrix import _context
    l = letters[i]
    sym = l["symbol"]
    prev = letters[i - 1]["symbol"] if i else "^"
    prev2 = letters[i - 2]["symbol"] if i > 1 else "^"
    nxt = letters[i + 1]["symbol"] if i + 1 < len(letters) else "$"
    w = l["word"]
    k = sum(1 for x in letters[:i] if x["word"] == w)
    wpos = "initial" if k == 0 else ("final" if k == words.get(w, 0) - 1 else "medial")
    ctx = _context(letters, i) if l["kind"] in ("consonant", "ikhfa_noon", "iqlab_meem") else l["kind"]
    stop = "ayah_final_word" if w == last_word else "inner_word"
    return [sym, prev2, prev, nxt, ctx, wpos, stop, f"{sym}>{prev}", f"{sym}<{nxt}"]


def probability(check: str, f: list[str]) -> float | None:
    m = model()
    c = (m.get("coef") or {}).get(check)
    if not c:
        return None
    z = c["intercept"] + sum(c["terms"].get(f"{name}={v}", 0.0) for name, v in zip(m["features"], f))
    return 1 / (1 + math.exp(-z))


def annotate(letters: list[dict[str, Any]], word_text: dict[int, str] | None = None) -> list[dict[str, float | None]]:
    """For each letter of one ayah: {check: probability a professional fails it here} -- the larger of
    the context model's and, where the masters recited this word, the word position's own rate."""
    from app.verse_detect import normalise
    table = model().get("words", {})
    norm = {w: normalise(t) for w, t in (word_text or {}).items()}
    kpos: dict[int, int] = {}
    words: dict[int, int] = {}
    for l in letters:
        if l["word"] is not None:
            words[l["word"]] = words.get(l["word"], 0) + 1
    last = max(words) if words else -1
    out = []
    for i, l in enumerate(letters):
        if l["word"] is None:
            out.append({})
            continue
        f = features(letters, i, words, last)
        w = l["word"]
        k = kpos.get(w, 0)
        kpos[w] = k + 1
        heads = list((l.get("characteristics") or {}).keys()) + ["identity"]
        row = {}
        for h in heads:
            p = probability(h, f)
            wr = table.get(f"{norm.get(w, '')}|{k}|{l['symbol']}|{h}") if norm else None
            row[h] = max(x for x in (p, wr) if x is not None) if (p is not None or wr is not None) else None
        out.append(row)
    return out
