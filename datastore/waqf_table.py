#!/usr/bin/env python3
"""Every stop the Quran allows, and exactly what it does to the recitation.

The rule binder reads each ayah in wasl (continuous). A reciter may stop at any word, and a stop
changes what is correct: the last word's final vowel falls silent, a natural madd before it becomes
madd 'arid li-s-sukun (2/4/6), a sakin yaa or waw after fatha becomes madd leen, a qalqalah letter
now bounces, tanwin fath becomes an alif, taa marbuta becomes haa, the pronoun haa loses its sila.
Restarting changes the next word too: a hamzat al-wasl is pronounced, an idgham across the boundary
comes undone. A learner who stopped and applied these correctly must not be marked wrong -- that is
what happened on Baqarah 2:2 at فِيهِ.

These changes are not hand-coded here. The phonetizer (quran_transcript, the one the acoustic model
was trained against) implements the waqf rules from the tajweed literature, and a text that ENDS at
a word is phonetized in waqf. So for every internal word boundary k of every ayah:

    wasl    = phonetize(whole ayah)             -- the continuous reading
    waqf    = phonetize(words[:k])              -- the reading that stops after word k
    ibtida  = phonetize(words[k:])              -- the reading that restarts at word k+1

and the difference between them, word by word, IS the effect of that stop. Each difference is
tagged with the classical rule it realises; anything no tag explains is kept as a raw signature so
nothing is silently dropped.

Lengths the tradition leaves free (madd 'arid, leen: 2, 4 or 6) are fixed by the phonetizer's moshaf
settings to one value (4). The table therefore records the rule and its ALLOWED range, never the
phonetizer's chosen length as if it were the only correct one.

    .venv/bin/python -m datastore.waqf_table build [procs]     # ~71k boundaries, a few minutes
    .venv/bin/python -m datastore.waqf_table summary
"""

from __future__ import annotations

import collections
import gzip
import json
import re
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEARNER = ROOT / "research_agency_lab/experiments/learner_eval"
OUT_DIR = ROOT / "research_agency_lab/experiments/waqf"
TABLE = OUT_DIR / "waqf_table.jsonl.gz"
SUMMARY = OUT_DIR / "waqf_summary.json"

SHORT_V = "َُِ"
MADD = "اۥۦ"
QALQALAH_RELEASE = "ڇ"
# allowed lengths, in counts, of the madd a stop creates (Hafs, tariq al-Shatibiyyah)
ALLOWED = {"madd_arid_lissukun": [2, 4, 6], "madd_leen": [2, 4, 6], "tanween_fath_to_alif": [2, 2],
           "madd_muttasil_at_waqf": [4, 5, 6]}

_ph = None


def _init() -> None:
    global _ph
    sys.path.insert(0, str(LEARNER))
    from muaalem_eval import MOSHAF  # noqa: PLC0415
    from quran_transcript import quran_phonetizer  # noqa: PLC0415

    import muaalem_dump as md  # noqa: PLC0415
    _ph = (quran_phonetizer, MOSHAF, md)


def _words(text: str):  # type: ignore[no-untyped-def]
    """Phonetize `text` and cut the phoneme string into its words (with the sifat of each group)."""
    phonetize, moshaf, md = _ph  # type: ignore[misc]
    r = phonetize(text, moshaf, remove_spaces=True)
    spans = md.word_spans(text, r.mappings)
    # tafkheem of the phoneme group covering each character of the phoneme string
    heavy, i = [], 0
    for e in r.sifat:
        heavy.extend([e.tafkheem_or_taqeeq] * len(e.phonemes))
        i += len(e.phonemes)
    out = []
    for sp in spans:
        if sp and sp[0] is not None and sp[0] >= 0 and sp[1] > sp[0]:
            out.append((r.phonemes[sp[0]:sp[1]], heavy[sp[1] - 1] if sp[1] - 1 < len(heavy) else None))
        else:
            out.append(("", None))       # a word absorbed into its neighbour (e.g. a full idgham)
    return out


def _run(s: str, chars: str) -> int:
    """Length of the last run of any of `chars` in `s`."""
    m = re.findall(f"[{chars}]+", s)
    return len(m[-1]) if m else 0


def tag_waqf(wasl: str, waqf: str, wasl_heavy, waqf_heavy) -> list[str]:  # type: ignore[no-untyped-def]
    """Name what stopping on a word changed in it."""
    if wasl == waqf and wasl_heavy == waqf_heavy:
        return ["unchanged"]
    tags = []
    # the vowel is silenced -- unless it survives as the fatha before an 'iwad alif (هُدًى -> هُدَاا)
    if (wasl and wasl[-1] in SHORT_V and not (waqf and waqf[-1] in SHORT_V)
            and not waqf.rstrip(QALQALAH_RELEASE).endswith("َاا")):
        tags.append("final_harakah_dropped")
    if waqf.endswith(QALQALAH_RELEASE) and not wasl.endswith(QALQALAH_RELEASE):
        tags.append("qalqalah_at_stop")
    core_waqf = waqf.rstrip(QALQALAH_RELEASE)
    # a madd letter run lengthened before the now-sakin final consonant
    if re.search(f"[{MADD}]{{3,}}[^{MADD}{SHORT_V}]$", core_waqf) and _run(core_waqf, MADD) > _run(wasl, MADD):
        tags.append("madd_arid_lissukun")
    # fatha + yaa/waw run lengthened before the final consonant: leen
    if re.search("َ[يو]{2,}[^" + MADD + SHORT_V + "]$", core_waqf):
        tags.append("madd_leen")
    if re.search("ء$", core_waqf) and _run(core_waqf, MADD) != _run(wasl, MADD):
        tags.append("madd_muttasil_at_waqf")
    if core_waqf.endswith("اا") and not wasl.endswith("اا"):
        tags.append("tanween_fath_to_alif")
    if core_waqf.endswith("ه") and re.search("ت[" + SHORT_V + "]?ن?$", wasl):
        tags.append("taa_marbuta_to_haa")
    if core_waqf.endswith("ه") and re.search("ه[ُِ][ۥۦ]+$", wasl):
        tags.append("haa_sila_dropped")
    # tanwin read as ن, or already assimilated into the next word (ںںں / ۾۾۾ / a doubled letter)
    if (re.search("[" + SHORT_V + "](ن|ں+|۾+|و{2,}|ي{2,}|ل{2,}|ر{2,}|م{2,})$", wasl)
            and not core_waqf.endswith("ن") and "tanween_fath_to_alif" not in tags):
        tags.append("tanween_dropped")
    # taa marbuta whose tanwin had already assimilated: ...تُوو / ...تَںںں -> ...ه
    if core_waqf.endswith("ه") and re.search("ت[" + SHORT_V + "].+$", wasl) and "taa_marbuta_to_haa" not in tags:
        tags.append("taa_marbuta_to_haa")
    # a munfasil madd exists only because the next word starts with hamza; stopping removes it
    if (_run(wasl, MADD) > _run(core_waqf, MADD) and re.search(f"[{MADD}]+$", wasl)
            and re.search(f"[{MADD}]+$", core_waqf)):
        tags.append("madd_munfasil_reverts_to_tabii")
    # a nasal or assimilation carried across the boundary (ikhfa, idgham, iqlab of a final ن / م)
    # comes undone when the next word is not recited with this one
    if re.search("(ں+|۾+|[يولرم]{2,})$", wasl) and core_waqf.endswith(("ن", "م")):
        tags.append("cross_word_nasal_undone")
    if wasl.endswith("ۜ") and not core_waqf.endswith("ۜ"):
        tags.append("sakt_dropped_at_stop")
    if wasl_heavy != waqf_heavy and wasl_heavy and waqf_heavy:
        tags.append(f"tafkheem_changed:{wasl_heavy}->{waqf_heavy}")
    return tags or ["other:" + _signature(wasl, waqf)]


def tag_ibtida(wasl: str, ibtida: str) -> list[str]:
    """Name what restarting on a word changed in it."""
    if wasl == ibtida:
        return ["unchanged"]
    tags = []
    if ibtida.startswith("ء") and not wasl.startswith("ء"):
        tags.append("hamzat_wasl_pronounced")
    if wasl and ibtida and len(wasl) > len(ibtida) and wasl.endswith(ibtida[1:]) and wasl[0] == wasl[1:2]:
        tags.append("cross_word_assimilation_undone")
    if not wasl:
        tags.append("word_restored_from_idgham")
    # two hamzas at a restart: the second, sakin, becomes the madd letter of the first's vowel (ibdal)
    if re.match("ء[ُِ][ۥۦ]{2}", ibtida) and wasl.startswith("ء") and not re.match("ء[ُِ][ۥۦ]", wasl):
        tags.append("hamzat_wasl_with_ibdal")
    # lam al-amr after ثم / و / ف is sakin in wasl and takes kasra when the reading starts on it
    if ibtida.startswith("لِ") and wasl.startswith("ل") and not wasl.startswith("لِ"):
        tags.append("lam_amr_kasra_on_restart")
    return tags or ["other:" + _signature(wasl, ibtida)]


def _signature(a: str, b: str) -> str:
    """A coarse, letter-independent picture of a change, for grouping what no tag explains."""
    def cls(s: str) -> str:
        return "".join("V" if c in SHORT_V else "M" if c in MADD else "Q" if c == QALQALAH_RELEASE
                       else "C" for c in s)
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    return f"{cls(a[i:])}>{cls(b[i:])}"


def _ayah(args):  # type: ignore[no-untyped-def]
    surah, ayah, text = args
    words = text.split()
    wasl = _words(text)
    rows = []
    if len(wasl) != len(words):
        return [{"surah": surah, "ayah": ayah, "error": f"{len(wasl)} spans for {len(words)} words"}]
    for k in range(1, len(words)):
        try:
            pre, post = _words(" ".join(words[:k])), _words(" ".join(words[k:]))
        except Exception as exc:  # noqa: BLE001 - record, never lose the row
            rows.append({"surah": surah, "ayah": ayah, "k": k, "error": repr(exc)})
            continue
        (w_ph, w_h), (q_ph, q_h) = wasl[k - 1], pre[-1]
        (n_ph, _), (i_ph, _) = wasl[k], post[0]
        wt, it = tag_waqf(w_ph, q_ph, w_h, q_h), tag_ibtida(n_ph, i_ph)
        rows.append({"surah": surah, "ayah": ayah, "k": k, "word": words[k - 1], "next": words[k],
                     "wasl": w_ph, "waqf": q_ph, "waqf_tags": wt,
                     "next_wasl": n_ph, "ibtida": i_ph, "ibtida_tags": it,
                     "allowed": {t: ALLOWED[t] for t in wt if t in ALLOWED}})
    return rows


def build(procs: int = 3) -> None:
    sys.path.insert(0, str(LEARNER))
    from quran_transcript import Aya  # noqa: PLC0415
    jobs, a = [], Aya(1, 1)
    for _ in range(6236):
        g = a.get()
        jobs.append((g.sura_idx, g.aya_idx, g.uthmani))
        a = a.step(1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    with Pool(procs, initializer=_init) as p, gzip.open(TABLE, "wt", encoding="utf-8") as f:
        for rows in p.imap(_ayah, jobs, chunksize=8):
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
    print(f"{n} rows -> {TABLE}")
    summary()


def summary() -> None:
    waqf, ibt, errors, examples = collections.Counter(), collections.Counter(), 0, {}
    total = 0
    for line in gzip.open(TABLE, "rt", encoding="utf-8"):
        r = json.loads(line)
        if "error" in r:
            errors += 1
            continue
        total += 1
        for t in r["waqf_tags"]:
            waqf[t] += 1
            examples.setdefault("waqf:" + t, f"{r['surah']}:{r['ayah']} {r['word']}  {r['wasl']} -> {r['waqf']}")
        for t in r["ibtida_tags"]:
            ibt[t] += 1
            examples.setdefault("ibtida:" + t, f"{r['surah']}:{r['ayah']} {r['next']}  {r['next_wasl']} -> {r['ibtida']}")
    changed = total - waqf["unchanged"]
    res = {"boundaries": total, "errors": errors,
           "waqf_changes_the_word": changed, "waqf_tags": dict(waqf.most_common()),
           "ibtida_tags": dict(ibt.most_common()), "examples": examples}
    SUMMARY.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"{total} internal word boundaries ({errors} errors); stopping changes the word at {changed}"
          f" ({100 * changed / max(total, 1):.1f} %)")
    print("WAQF (the stopped word):")
    for t, c in waqf.most_common(25):
        print(f"  {c:6d}  {t:40s} e.g. {examples.get('waqf:' + t, '')}")
    print("IBTIDA (the restarted word):")
    for t, c in ibt.most_common(12):
        print(f"  {c:6d}  {t:40s} e.g. {examples.get('ibtida:' + t, '')}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["build"]:
        build(*(int(a) for a in sys.argv[2:]))
    else:
        summary()
