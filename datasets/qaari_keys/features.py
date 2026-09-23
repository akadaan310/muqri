#!/usr/bin/env python3
"""QaariKeys stage A: one sparse key-count vector per ayah, over all 6236 ayahs of Hafs.

A *key* is something a test verse can exercise. Five families, each grounded in a research report:

* ``rule:<key>``      every Tajweed / Sifah rule instance the parser emits, bucketed by
                      ``app.calibration.key_of`` (rule type + the sub-type that matters).
* ``letter:<c>``      pronounced consonants, plus ``letter:<c>:shadda`` / ``:sukun`` (lahn jali
                      coverage; 00 Tier 1 #1, #3).
* ``pair:<a>|<b>``    both letters of a classical confusion pair in the same ayah: a minimal-pair
                      context for GOP (01 §3.5 priority pairs; ``app/lahn/gop.py`` confusion sets).
                      Count = min of the two letter counts.
* ``rep:<rule>``      a madd / ghunnah / qalqalah family repeated ≥ 3 times in one ayah: the
                      within-ayah evidence tasawi (equal lengths) needs (00 Tier 1 #5, 03).
                      Count = occurrences − 2.
* ``special:<id>``    the Hafs special-case words of 00 §3 (tas-heel, imala, ishmam, sakt, seen/sad,
                      madd al-farq, rectangular-zero alif, …), by their references.
* ``waqf:<sign>``     the Uthmani pause marks U+06D6–U+06DB (04 §1).

Output ``build/ayah_keys.json``: ``keys`` (sorted), ``ayahs`` ([surah, ayah, words, letters]) and
``counts`` (per ayah, {key index: count}); plus ``build/ayah_keys.tsv`` (ayah, key, count triplets)
for Julia and Octave.

    .venv/bin/python datasets/qaari_keys/features.py
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.calibration import key_of  # noqa: E402
from app.models import MADD_RULES, RuleType  # noqa: E402
from app.quran_text import get_full_quran  # noqa: E402
from app.tajweed_rules import TajweedParser  # noqa: E402

BUILD = Path(__file__).resolve().parent / "build"

# 01_makharij §3.5 priority pairs + gop.py confusion sets (Arabic letters).
CONFUSION_PAIRS: list[tuple[str, str]] = [
    ("ض", "د"), ("ض", "ظ"), ("ظ", "ز"), ("ظ", "ذ"), ("ذ", "ز"), ("ذ", "د"), ("ث", "س"), ("ث", "ت"),
    ("ص", "س"), ("ط", "ت"), ("ح", "ه"), ("ع", "ء"), ("ق", "ك"), ("غ", "خ"), ("خ", "ح"),
]
HAMZA_FORMS = {"أ": "ء", "إ": "ء", "ؤ": "ء", "ئ": "ء", "ٔ": "ء"}

# 00_curriculum_audit §3: Hafs special-case words, id -> references (surah, ayah).
SPECIAL: dict[str, list[tuple[int, int]]] = {
    "ishmam_tamanna": [(12, 11)],
    "tasheel_aajami": [(41, 44)],
    "imala_majraha": [(11, 41)],
    "daf_fath_damm": [(30, 54)],
    "seen_yabsut": [(2, 245)],
    "seen_bastah": [(7, 69)],
    "sad_or_seen_musaytirun": [(52, 37)],
    "sad_musaytir": [(88, 22)],
    "sakt_iwaja": [(18, 1)],
    "sakt_marqadina": [(36, 52)],
    "sakt_man_raq": [(75, 27)],
    "sakt_bal_ran": [(83, 14)],
    "sakt_maliyah": [(69, 28)],
    "madd_farq_dhakarayn": [(6, 143), (6, 144)],
    "madd_farq_alaan": [(10, 51), (10, 91)],
    "madd_farq_allah": [(10, 59), (27, 59)],
    "salasila": [(76, 4)],
    "qawarira_alif": [(76, 15)],
    "qawarira_sukun": [(76, 16)],
    "zero_alif_33": [(33, 10), (33, 66), (33, 67)],
    "lakinna_huwa": [(18, 38)],
    "ana_waqf_alif": [(7, 12), (12, 90)],
    "thamuda": [(11, 68), (25, 38), (29, 38), (53, 51)],
    "atani_yaa": [(27, 36)],
    "alif_lam_mim_allah": [(3, 1), (3, 2)],
    "ayn_4_or_6": [(19, 1), (42, 2)],
    "izhar_yasin_nun": [(36, 1), (36, 2), (68, 1)],
    "yalhath_dhalika": [(7, 176)],
    # 00 Tier 0.7: raa at waqf, both readings allowed
    "raa_waqf_yasr": [(89, 4), (54, 16)],
    # 00 Tier 0.9: hamzat al-wasl with damm / kasr at ibtida
    "wasl_imshu": [(38, 6)],
}
REPEAT_FAMILIES: dict[str, set[RuleType]] = {
    "madd_tabii": {RuleType.MADD_TABII} if hasattr(RuleType, "MADD_TABII") else set(),
    "madd_munfasil": {RuleType.MADD_MUNFASIL},
    "madd_muttasil": {RuleType.MADD_MUTTASIL},
    "madd_any": set(MADD_RULES),
    "ghunnah": {RuleType.GHUNNAH},
}
WAQF_SIGNS = {"ۖ": "sili", "ۗ": "qili", "ۘ": "mim_lazim", "ۙ": "la", "ۚ": "jim",
              "ۛ": "muanaqah"}


def _repeat_families() -> dict[str, set[RuleType]]:
    fam = {k: v for k, v in REPEAT_FAMILIES.items() if v}
    for rt in RuleType:
        if rt.value.startswith("qalqalah"):
            fam.setdefault("qalqalah", set()).add(rt)
    return fam


def ayah_counts(parser: TajweedParser, text: str) -> tuple[collections.Counter[str], int, int]:
    parsed = parser.parse(text)
    c: collections.Counter[str] = collections.Counter()
    for r in parsed.rules:
        c[f"rule:{key_of(r)}"] += 1
    letters: collections.Counter[str] = collections.Counter()
    for u in parsed.units:
        if not u.pronounced or u.madd_letter or u.char in "اىـ":
            continue
        ch = HAMZA_FORMS.get(u.char, u.char)
        if not ("ء" <= ch <= "ي"):
            continue
        letters[ch] += 1
        if u.shadda:
            c[f"letter:{ch}:shadda"] += 1
        if u.sukun:
            c[f"letter:{ch}:sukun"] += 1
    for ch, n in letters.items():
        c[f"letter:{ch}"] += n
    for a, b in CONFUSION_PAIRS:
        if letters[a] and letters[b]:
            c[f"pair:{a}|{b}"] += min(letters[a], letters[b])
    for fam, types in _repeat_families().items():
        n = sum(r.rule_type in types for r in parsed.rules)
        if n >= 3:
            c[f"rep:{fam}"] += n - 2
    for ch, name in WAQF_SIGNS.items():
        if ch in text:
            c[f"waqf:{name}"] += text.count(ch)
    return c, len(parsed.words), sum(letters.values())


def main() -> int:
    BUILD.mkdir(exist_ok=True)
    quran = get_full_quran()
    parser = TajweedParser(include_sifaat=True)
    special_at: dict[tuple[int, int], list[str]] = collections.defaultdict(list)
    for sid, refs in SPECIAL.items():
        for ref in refs:
            special_at[ref].append(sid)
    rows, all_keys = [], set()
    failed = 0
    for (s, a), text in sorted(quran.items()):
        try:
            c, words, nlet = ayah_counts(parser, text)
        except Exception as exc:  # noqa: BLE001
            print(f"{s}:{a} parse failed: {exc}", file=sys.stderr)
            c, words, nlet = collections.Counter(), len(text.split()), 0
            failed += 1
        for sid in special_at.get((s, a), []):
            c[f"special:{sid}"] += 1
        rows.append(((s, a), words, nlet, c))
        all_keys.update(c)
    keys = sorted(all_keys)
    kidx = {k: i for i, k in enumerate(keys)}
    out = {"keys": keys, "ayahs": [[s, a, w, n] for (s, a), w, n, _ in rows],
           "counts": [{kidx[k]: v for k, v in c.items()} for *_, c in rows],
           "special": {k: v for k, v in SPECIAL.items()}, "confusion_pairs": CONFUSION_PAIRS,
           "parse_failures": failed}
    (BUILD / "ayah_keys.json").write_text(json.dumps(out, ensure_ascii=False))
    with (BUILD / "ayah_keys.tsv").open("w") as fh:  # 1-based indices for Julia / Octave
        for i, (*_, c) in enumerate(rows, start=1):
            for k, v in sorted(c.items()):
                fh.write(f"{i}\t{kidx[k] + 1}\t{v}\n")
    (BUILD / "keys.txt").write_text("\n".join(keys) + "\n")
    (BUILD / "ayahs.tsv").write_text("".join(f"{s}\t{a}\t{w}\t{n}\n" for (s, a), w, n, _ in rows))
    fam = collections.Counter(k.split(":")[0] for k in keys)
    print(f"{len(rows)} ayahs, {len(keys)} keys {dict(fam)}, parse failures {failed} -> {BUILD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
