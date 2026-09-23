#!/usr/bin/env python3
"""Static per-letter reference: makhraj and the sifāt that no acoustic head can judge.

muaalem gives a trained head for ten attributes, and those *vary by realisation* — whether this ص was
actually produced with iṭbāq is a question about the audio. But several classical sifāt are
**inherent**: ر is muṣtafil, mudhlaq and munḥarif in every recitation by every reciter, and its
makhraj never moves. They are properties of the letter, not of the performance, so they belong in a
lookup table rather than a model — and without them the per-letter record is incomplete.

Combining the two gives the full classical count. For ر:

    judged   (heads)    jahr · tawassuṭ · mufakhkham(contextual) · mukarrar
    inherent (here)     istifāl · infitāḥ · idhlāq · inḥirāf        → seven sifāt

The inherent sets are the standard Hafs teaching:

* **Isti'lā'** (elevated): خ ص ض ط ظ غ ق — the mnemonic خُصَّ ضَغْطٍ قِظْ. Everything else is **istifāl**.
* **Idhlāq** (fluent, articulated at the lip or tongue tip): ف ر م ن ل ب — فِرَّ مِنْ لُبٍّ.
  Everything else is **iṣmāt**.
* **Inḥirāf** (the sound deviates along the tongue): ل ر
* **Līn** (softness): و ي when sākin after a fatḥah

    .venv/bin/python -m datastore.letter_reference
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datastore.store import connect, run  # noqa: E402

# makhraj: (zone, precise articulation point)
MAKHRAJ: dict[str, tuple[str, str]] = {
    "ا": ("jawf", "oral cavity — prolongation, no contact"),
    "ء": ("halq", "deepest throat (aqṣā al-ḥalq)"), "ه": ("halq", "deepest throat (aqṣā al-ḥalq)"),
    "ع": ("halq", "middle throat (wasaṭ al-ḥalq)"), "ح": ("halq", "middle throat (wasaṭ al-ḥalq)"),
    "غ": ("halq", "nearest throat (adnā al-ḥalq)"), "خ": ("halq", "nearest throat (adnā al-ḥalq)"),
    "ق": ("lisan", "deepest tongue with the soft palate"),
    "ك": ("lisan", "deepest tongue, slightly forward, with the hard palate"),
    "ج": ("lisan", "middle of the tongue with the hard palate"),
    "ش": ("lisan", "middle of the tongue with the hard palate"),
    "ي": ("lisan", "middle of the tongue with the hard palate"),
    "ض": ("lisan", "edge of the tongue with the upper molars"),
    "ل": ("lisan", "both edges of the tongue, forward, with the gums"),
    "ن": ("lisan", "tip of the tongue with the gums above the incisors"),
    "ر": ("lisan", "tip of the tongue, slightly further back, with the gums"),
    "ط": ("lisan", "tip of the tongue with the roots of the upper incisors"),
    "د": ("lisan", "tip of the tongue with the roots of the upper incisors"),
    "ت": ("lisan", "tip of the tongue with the roots of the upper incisors"),
    "ص": ("lisan", "tip of the tongue between the incisors"),
    "ز": ("lisan", "tip of the tongue between the incisors"),
    "س": ("lisan", "tip of the tongue between the incisors"),
    "ظ": ("lisan", "tip of the tongue with the tips of the upper incisors"),
    "ذ": ("lisan", "tip of the tongue with the tips of the upper incisors"),
    "ث": ("lisan", "tip of the tongue with the tips of the upper incisors"),
    "ف": ("shafatan", "inside of the lower lip with the tips of the upper incisors"),
    "ب": ("shafatan", "both lips, closed"), "م": ("shafatan", "both lips, lightly closed"),
    "و": ("shafatan", "both lips, rounded, not closed"),
}

# Weight (tafkhīm/tarqīq) is INHERENT for most letters — an isti'lā' letter is always heavy, an
# istifāl letter always light — but for ر and for the lām of the Divine Name it is CONTEXTUAL, and
# those are the strictest conditional rules in tajweed. They are recorded here as data so the
# expected label can be audited; `quran_transcript` applies them and tests/test_raa_lam_weight.py
# pins every condition. Corpus check over 301 ayahs: ر 79.3 % heavy / 20.7 % light,
# ل 13.7 % heavy / 86.3 % light — both as the rules predict.
WEIGHT_RULE: dict[str, list[tuple[str, str]]] = {
    "ر": [("heavy", "ر with fatḥah or ḍammah"),
          ("heavy", "ر sākin after fatḥah or ḍammah"),
          ("heavy", "ر sākin after an INCIDENTAL ('āriḍ) kasrah, e.g. hamzat al-waṣl"),
          ("heavy", "ر sākin after an original kasrah when a ḥarf isti'lā' follows in the same word "
                    "and is not itself kasrah'd — مِرْصَاد، قِرْطَاس"),
          ("light", "ر with kasrah"),
          ("light", "ر sākin after an ORIGINAL kasrah with no isti'lā' following — فِرْعَوْن"),
          ("light", "ر sākin at waqf preceded by a sākin yā' (leen) — خَيْر"),
          ("either", "jawāz al-wajhayn: ر sākin after kasrah with a kasrah'd isti'lā' after — فِرْقٍ; "
                     "and some waqf cases — مِصْر، ٱلْقِطْر، يَسْر، أَنْ أَسْر، نُذُر")],
    "ل": [("heavy", "lām of the Divine Name ٱللَّه after a fatḥah or ḍammah, or ayah-initial"),
          ("light", "lām of the Divine Name after a kasrah — بِسْمِ ٱللَّهِ"),
          ("light", "every other lām, always")],
}

ISTILA = set("خصضطظغق")      # خُصَّ ضَغْطٍ قِظْ — everything else is istifāl
IDHLAQ = set("فرمنلب")        # فِرَّ مِنْ لُبٍّ — everything else is iṣmāt
INHIRAF = set("لر")
LIN = set("وي")               # when sākin after a fatḥah

SCHEMA = """
DROP TABLE IF EXISTS letter_reference;
CREATE TABLE letter_reference (
    letter        TEXT PRIMARY KEY,
    makhraj_zone  TEXT,
    makhraj       TEXT,
    istila        TEXT,   -- isti'la | istifal
    weight        TEXT,   -- inherent-heavy | inherent-light | contextual (see WEIGHT_RULE)
    weight_rules  TEXT,   -- JSON list of (verdict, condition) for the contextual letters
    idhlaq        TEXT,   -- idhlaq | ismat
    inhiraf       BOOLEAN,
    lin           BOOLEAN,
    n_inherent    INTEGER -- inherent sifāt that carry a positive (marked) value
);
"""


def rows() -> list[tuple]:  # type: ignore[type-arg]
    out = []
    for letter, (zone, point) in MAKHRAJ.items():
        istila = "isti'la" if letter in ISTILA else "istifal"
        idhlaq = "idhlaq" if letter in IDHLAQ else "ismat"
        inh, lin = letter in INHIRAF, letter in LIN
        # marked = the positive member of each pair (isti'lā' and idhlāq are the marked ones)
        n_marked = sum([letter in ISTILA, letter in IDHLAQ, inh, lin])
        wr = WEIGHT_RULE.get(letter)
        weight = "contextual" if wr else ("inherent-heavy" if letter in ISTILA else "inherent-light")
        out.append((letter, zone, point, istila, weight, json.dumps(wr, ensure_ascii=False) if wr else None,
                    idhlaq, inh, lin, n_marked))
    return out


def fragment(rs: list[tuple], path: Path | None = None) -> Path:  # type: ignore[type-arg]
    path = path or ROOT / "research_agency_lab/knowledge_map/fragments/10_letter_reference.json"
    REF = "datastore/letter_reference.py"
    nodes: dict[str, dict] = {}
    edges: list[dict] = []

    def node(nid, typ, label, **kw):  # type: ignore[no-untyped-def]
        nodes.setdefault(nid, {"id": nid, "type": typ, "label": label, "refs": [REF], **kw})

    node("ref:letters", "artifact", "Per-letter makhraj and inherent sifāt", status="implemented",
         file=REF, note=f"{len(rs)} letters; the sifāt no acoustic head can judge")
    for letter, zone, point, istila, _w, _wr, idhlaq, inh, lin, _n in rs:
        node(f"letter:{letter}", "phenomenon", letter, status="implemented", note=point)
        node(f"makhraj:{zone}", "phenomenon", f"Makhraj zone: {zone}", status="implemented")
        edges.append({"from": f"letter:{letter}", "to": f"makhraj:{zone}", "rel": "articulated_at"})
        for sif in (istila, idhlaq, *(["inhiraf"] if inh else []), *(["lin"] if lin else [])):
            node(f"sifah:{sif}", "phenomenon", sif, status="implemented")
            edges.append({"from": f"letter:{letter}", "to": f"sifah:{sif}", "rel": "has_inherent"})
        edges.append({"from": "ref:letters", "to": f"letter:{letter}", "rel": "describes"})
    path.write_text(json.dumps({"nodes": list(nodes.values()), "edges": edges},
                               ensure_ascii=False, indent=1))
    return path


def export_json(rs: list[tuple], path: Path | None = None) -> Path:  # type: ignore[type-arg]
    """The form `letter_report.jl` reads, so the Julia side has no table of its own to drift from."""
    path = path or ROOT / "app/data/letter_reference.json"
    path.write_text(json.dumps(
        {r[0]: {"makhraj_zone": r[1], "makhraj": r[2], "istila": r[3], "weight": r[4],
                "weight_rules": json.loads(r[5]) if r[5] else None, "idhlaq": r[6],
                "inhiraf": r[7], "lin": r[8], "n_inherent": r[9]} for r in rs},
        ensure_ascii=False, indent=1))
    return path


def main() -> int:
    rs = rows()
    con = connect()
    con.execute(SCHEMA)
    with run(con, "catalogue:letter_reference", "loader", {"letters": len(rs)}):
        con.execute("DELETE FROM letter_reference")
        con.executemany("INSERT INTO letter_reference VALUES (?,?,?,?,?,?,?,?,?,?)", rs)
    frag, js = fragment(rs), export_json(rs)
    print(f"letter_reference: {len(rs)} letters -> DuckDB, {frag.relative_to(ROOT)}, {js.relative_to(ROOT)}\n")
    print(f"{'letter':7}{'zone':11}{'istila':10}{'weight':16}{'idhlaq':9}{'inhiraf':9}lin")
    for letter, zone, _p, istila, weight, _wr, idhlaq, inh, lin, _n in rs:
        print(f"{letter:6} {zone:11}{istila:10}{weight:16}{idhlaq:9}"
              f"{'yes' if inh else '-':9}{'yes' if lin else '-'}")
    n_ctx = sum(1 for r in rs if r[4] == "contextual")
    print(f"\n{n_ctx} letters have CONTEXTUAL weight (ر, ل) — "
          f"{sum(len(json.loads(r[5])) for r in rs if r[5])} conditions, pinned by tests/test_raa_lam_weight.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
