"""Counterfactual text-swap test of the lahn-jali GOP detector (graph node ``algo:text_swap_h1``).

Expert audio is (almost) error free, so on the true text every unit is a negative. Swapping one
letter (or short vowel) of the *text* to a classical confusion makes the audio disagree with the
text at exactly that unit: a positive with known location and known error type. Recall is
measured on swaps, false alarms on the untouched text, per reciter and per confusion pair.

    .venv/bin/python research_agency_lab/experiments/lahn_gop/text_swap.py OUT.jsonl [--reciters ...]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from app.aligner import AlignmentError, CTCForcedAligner  # noqa: E402
from app.audio import load_audio  # noqa: E402
from app.lahn.gop import letter_gop  # noqa: E402
from app.quran_text import get_ayah_text  # noqa: E402
from app.tajweed_rules.parser import parse_text  # noqa: E402

CACHE = Path.home() / ".cache/qaari-eval/everyayah"
# audio letter -> letters the text is swapped to (the audio then "sounds like" the key)
LETTER_SWAPS = {
    "ض": "دظ", "د": "ض", "ص": "س", "س": "صث", "ط": "ت", "ت": "ط", "ح": "ه", "ه": "ح", "ع": "ء",
    "ق": "ك", "ك": "ق", "ث": "س", "ذ": "زد", "ز": "ذ", "ظ": "ز", "غ": "خ", "خ": "غ",
}
FATHA, DAMMA, KASRA = "َ", "ُ", "ِ"
VOWEL_SWAPS = {FATHA: (KASRA, DAMMA), KASRA: (FATHA, DAMMA), DAMMA: (FATHA, KASRA)}


def swap_candidates(parsed, text: str):  # type: ignore[no-untyped-def]
    """(unit index, char offset in text, new char, kind, label) for swappable units."""
    out = []
    offsets = {}
    pos = 0
    for u in parsed.units:  # units are in text order; locate each unit's letter
        if getattr(u, "synthetic", False) or u.char not in LETTER_SWAPS and u.char not in text:
            continue  # normalised units (ayah-initial wasl hamza, dagger alif) have no text offset
        q = text.find(u.char, pos)
        if q < 0 or q - pos > 6:
            continue
        offsets[u.index] = q
        pos = q + 1
    for u in parsed.units:
        if not u.pronounced or u.index not in offsets:
            continue
        off = offsets[u.index]
        for alt in LETTER_SWAPS.get(u.char, ""):
            out.append((u.index, off, alt, "consonant", f"{u.char}->{alt}"))
        # the short vowel mark directly follows the letter (possibly after a shadda)
        j = off + 1
        if j < len(text) and text[j] == "ّ":
            j += 1
        if j < len(text) and text[j] in VOWEL_SWAPS and not u.tanween:
            for alt in VOWEL_SWAPS[text[j]]:
                out.append((u.index, j, alt, "vowel", f"{text[j]}->{alt}"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--reciters", nargs="*")
    ap.add_argument("--swaps-per-ayah", type=int, default=6)
    ap.add_argument("--max-ayahs", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(0)
    al = CTCForcedAligner()
    reciters = args.reciters or sorted(p.name for p in CACHE.iterdir() if p.is_dir())
    with open(args.out, "w") as fh:
        for rec in reciters:
            files = sorted((CACHE / rec).glob("*.mp3"))
            if args.max_ayahs:
                files = files[: args.max_ayahs]
            t0 = time.time()
            for f in files:
                s, a = int(f.stem[:3]), int(f.stem[3:])
                try:
                    text = get_ayah_text(s, a, allow_network=False)
                    audio = load_audio(f, denoise="never")
                    parsed = parse_text(text)
                    alignment = al.align(audio, parsed)
                except Exception as exc:  # noqa: BLE001
                    print(f"skip {rec} {s}:{a}: {exc}", file=sys.stderr)
                    continue
                lp, fs = al.emissions(audio)
                units = {u.index: u for u in parsed.units}
                for g in letter_gop(lp, fs, al.vocab, al.blank, parsed, alignment):
                    fh.write(json.dumps({"reciter": rec, "surah": s, "ayah": a, "unit": g.unit_index,
                                         "char": units[g.unit_index].char, "kind": g.kind, "target": g.target,
                                         "alt": g.best_alt, "llr": round(g.llr, 3), "frames": g.frames,
                                         "positive": False}, ensure_ascii=False) + "\n")
                cands = swap_candidates(parsed, text)
                for idx, off, new, kind, label in rng.sample(cands, min(args.swaps_per_ayah, len(cands))):
                    swapped = text[:off] + new + text[off + 1:]
                    try:
                        p2 = parse_text(swapped)
                        if len(p2.units) != len(parsed.units):
                            continue
                        a2 = al.align(audio, p2)
                    except (AlignmentError, Exception):  # noqa: BLE001
                        continue
                    for g in letter_gop(lp, fs, al.vocab, al.blank, p2, a2, units={idx}):
                        if g.kind != kind:
                            continue
                        fh.write(json.dumps({"reciter": rec, "surah": s, "ayah": a, "unit": idx, "swap": label,
                                             "kind": kind, "target": g.target, "alt": g.best_alt,
                                             "llr": round(g.llr, 3), "frames": g.frames, "positive": True},
                                            ensure_ascii=False) + "\n")
                fh.flush()
            print(f"{rec}: {len(files)} ayahs in {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
