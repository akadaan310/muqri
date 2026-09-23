#!/usr/bin/env python3
"""The tajweed rule catalogue: every rule the parser locates, and how the engine can actually judge it.

This is the map from *what the parser finds in the text* (41 `RuleType`s, 1.16 M located instances in
`engine_diag`) to *how the acoustic model can test it*. Three mechanisms cover everything:

* **segmental** — the rule changes which phonemes are recited (idgham merges a letter, iqlab turns
  noon into meem, izhar keeps it clear). `quran_phonetizer` already emits the correct Hafs phonemes,
  so the counterfactual is a phoneme-string edit and the test is the CTC likelihood ratio that scored
  99.7 % / 97.1 % on letter swaps (`textswap_gop.jl`).
* **durational** — the rule sets a *length* (madd 2/4/6 counts, ghunnah hold, sakt pause). The
  phonetizer encodes length as **character repetition** (`ۥ×4` = 4 counts, `ا×2` = madd tabii,
  `ں×3` = ikhfa noon held), so a length error is *also* a phoneme-string edit and uses the same test.
* **attribute** — the rule asserts a quality of a letter (hams/jahr, shidda/rakhawa, itbaq, safeer,
  qalqala, tikraar, tafashie, istitala, tafkheem/tarqeeq, ghunnah). muaalem emits a dedicated CTC head
  for each of these; `SifatGop.jl` scores them and measured ~1 % anchor false-alarm, replacing DSP
  detectors that false-FAILed known-good peers at 10–19 %.

`legacy_pass_rate` is the *current* engine's PASS fraction over `engine_diag` — on master reciters. A
rate near 0.05 does not mean the masters are wrong; it means that detector is broken and fails
everyone, which is what makes this table a work list rather than a description.

    .venv/bin/python -m datastore.rule_catalogue          # build + store + print
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datastore.store import connect, run  # noqa: E402

SEG, DUR, ATTR = "segmental", "durational", "attribute"

# rule_type -> (family, mechanism, phoneme-string signature, counterfactual test, muaalem head)
CATALOGUE: dict[str, tuple[str, str, str, str, str | None]] = {
    # ---- Mudood: length is character repetition, so every madd is testable by re-length ----
    "madd_tabii": ("mudood", DUR, "madd letter ا/ۥ/ۦ run of 2", "run 2 → 1 or 4 (wrong count)", None),
    "madd_muttasil": ("mudood", DUR, "madd run of 4–5 before hamza in the same word", "4 → 2 or 6", None),
    "madd_munfasil": ("mudood", DUR, "madd run of 4–5 across a word boundary", "4 → 2 or 6", None),
    "madd_lazim": ("mudood", DUR, "madd run of 6", "6 → 2 or 4", None),
    "madd_arid_lissukun": ("mudood", DUR, "madd run of 2/4/6 before a final sukoon", "vary the count", None),
    "madd_leen": ("mudood", DUR, "و/ي leen run before a stop", "vary the count", None),
    "madd_badal": ("mudood", DUR, "hamza + madd run of 2", "2 → 4", None),
    "madd_iwad": ("mudood", DUR, "tanween fath → ا run of 2 at a stop", "2 → 0 (no 'iwad)", None),
    "madd_silah_sughra": ("mudood", DUR, "ۥ/ۦ run of 2 on the pronoun ha", "2 → 0 (no silah)", None),
    "madd_silah_kubra": ("mudood", DUR, "ۥ/ۦ run of 4 on the pronoun ha before hamza", "4 → 2 or 0", None),
    # ---- Noon sakinah / tanween: the four hukms differ in the phonemes actually produced ----
    "izhar_halqi": ("noon_sakinah", SEG, "clear ن before a throat letter", "ن → ں (wrongly hidden)", "ghonna"),
    "ikhfa": ("noon_sakinah", DUR, "ں run of ~3 (nasalised, held)", "ں → ن (izhar) or run → 1", "ghonna"),
    "idgham_ghunnah": ("noon_sakinah", SEG, "ن absorbed into ي/ن/م/و, held run", "restore the ن (no idgham)", "ghonna"),
    "idgham_no_ghunnah": ("noon_sakinah", SEG, "ن absorbed into ل/ر, no ghunnah", "restore ن / add ghunnah", "ghonna"),
    "iqlab": ("noon_sakinah", SEG, "ن → م before ب, held", "م → ن (no iqlab)", "ghonna"),
    # ---- Meem sakinah ----
    "ikhfa_shafawi": ("meem_sakinah", DUR, "م held before ب", "shorten the hold / no ikhfa", "ghonna"),
    "idgham_shafawi": ("meem_sakinah", SEG, "م + م merged into a held run", "split the run (no idgham)", "ghonna"),
    "izhar_shafawi": ("meem_sakinah", SEG, "clear م, not held", "hold it (wrong ikhfa)", "ghonna"),
    # ---- Ghunnah mushaddadah ----
    "ghunnah": ("ghunnah", DUR, "ن/م shadda as a run of ~4", "run → 1 (ghunnah dropped)", "ghonna"),
    # ---- Qalqalah ----
    "qalqalah": ("qalqalah", ATTR, "ق ط ب ج د at sukoon", "expect moqalqal vs not_moqalqal", "qalqla"),
    # ---- Idgham of other letters ----
    "idgham_mithlayn": ("idgham", SEG, "identical letters merged into one run of 2", "split the run", None),
    "idgham_mutajanisayn": ("idgham", SEG, "same-makhraj pair merged", "keep both letters distinct", None),
    "idgham_mutaqaribayn": ("idgham", SEG, "near-makhraj pair merged", "keep both letters distinct", None),
    # ---- Raa / Lam ----
    "tafkheem": ("raa_lam", ATTR, "heavy realisation of ر/ل", "expect mofakham vs moraqaq", "tafkheem_or_taqeeq"),
    "tarqeeq": ("raa_lam", ATTR, "light realisation of ر/ل", "expect moraqaq vs mofakham", "tafkheem_or_taqeeq"),
    "jawaz_wajhayn": ("raa_lam", ATTR, "both realisations permitted", "either accepted (never FAIL)", "tafkheem_or_taqeeq"),
    # ---- Wasl / waqf / sakt ----
    "hamzat_wasl": ("wasl_waqf", SEG, "initial hamza dropped when joined", "restore/drop the hamza", None),
    "sakt": ("wasl_waqf", DUR, "brief silence without breath", "remove the pause / make it a waqf", None),
    # ---- Sifaat: one muaalem head each ----
    "hams": ("sifaat", ATTR, "voiceless letter", "expect hams vs jahr", "hams_or_jahr"),
    "jahr": ("sifaat", ATTR, "voiced letter", "expect jahr vs hams", "hams_or_jahr"),
    "shiddah": ("sifaat", ATTR, "full stop of the airflow", "expect shadeed", "shidda_or_rakhawa"),
    "tawassut": ("sifaat", ATTR, "between shidda and rakhawa", "expect between", "shidda_or_rakhawa"),
    "rakhawah": ("sifaat", ATTR, "continuant", "expect rikhw", "shidda_or_rakhawa"),
    "itbaq": ("sifaat", ATTR, "tongue raised to the palate", "expect motbaq vs monfateh", "itbaq"),
    "safir": ("sifaat", ATTR, "whistle (ص ز س)", "expect safeer", "safeer"),
    "tafashhi": ("sifaat", ATTR, "spreading (ش)", "expect motafashie", "tafashie"),
    "istitaalah": ("sifaat", ATTR, "elongation (ض)", "expect mostateel", "istitala"),
    "takreer": ("sifaat", ATTR, "trill of ر (must not be exaggerated)", "expect not_mokarar", "tikraar"),
    # ---- Lahn jali: already measured at 99.7 % / 97.1 % ----
    "lahn_letter": ("lahn", SEG, "the consonant itself", "swap to a classical confusion", None),
    "lahn_harakah": ("lahn", SEG, "the short vowel itself", "swap fatha/damma/kasra", None),
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS rule_catalogue (
    rule_type        TEXT PRIMARY KEY,
    family           TEXT,
    mechanism        TEXT,      -- segmental | durational | attribute
    ph_signature     TEXT,      -- how the rule shows up in the phoneme string
    counterfactual   TEXT,      -- the perturbation that tests it
    muaalem_head     TEXT,      -- sifat head that judges it, when there is one
    n_instances      BIGINT,    -- located instances in engine_diag
    n_reciters       INTEGER,
    legacy_pass_rate DOUBLE,    -- current engine PASS fraction, on masters
    status           TEXT       -- broken | suspect | plausible | unmeasured
);
"""


def status_of(pass_rate: float | None) -> str:
    if pass_rate is None:
        return "unmeasured"
    if pass_rate <= 0.10:
        return "broken"      # fails nearly every master: the detector, not the recitation
    if pass_rate <= 0.50:
        return "suspect"
    return "plausible"


def build(con) -> list[tuple]:  # type: ignore[no-untyped-def,type-arg]
    con.execute(SCHEMA)
    stats = {r[0]: (r[1], r[2], r[3]) for r in con.execute(
        "SELECT rule_type, count(*), count(DISTINCT reciter_id), "
        "sum(CASE WHEN status='PASS' THEN 1 ELSE 0 END)*1.0/count(*) FROM engine_diag GROUP BY 1").fetchall()}
    rows = []
    for rule, (fam, mech, sig, cf, head) in CATALOGUE.items():
        n, recs, pr = stats.get(rule, (0, 0, None))
        rows.append((rule, fam, mech, sig, cf, head, n, recs, pr, status_of(pr)))
    con.execute("DELETE FROM rule_catalogue")
    con.executemany("INSERT INTO rule_catalogue VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    return rows


def main() -> int:
    con = connect()
    with run(con, "catalogue:tajweed_rules", "loader", {"rules": len(CATALOGUE)}):
        rows = build(con)
    frag = fragment(rows)
    by_status: dict[str, list[tuple]] = {}
    for r in rows:
        by_status.setdefault(r[9], []).append(r)
    print(f"graph fragment -> {frag.relative_to(ROOT)}")
    print(f"rule_catalogue: {len(rows)} rules, {sum(r[6] for r in rows):,} located instances\n")
    for st in ("broken", "suspect", "plausible", "unmeasured"):
        rs = sorted(by_status.get(st, []), key=lambda r: -r[6])
        if not rs:
            continue
        print(f"== {st.upper()} ({len(rs)} rules, {sum(r[6] for r in rs):,} instances)")
        for r in rs:
            pr = "   -  " if r[8] is None else f"{r[8]:.3f}"
            print(f"   {r[0]:24s} {r[2]:11s} n={r[6]:>8,}  pass={pr}  head={r[5] or '-'}")
        print()
    return 0




def fragment(rows: list[tuple], path: Path | None = None) -> Path:  # type: ignore[type-arg]
    """Project the catalogue into a knowledge-map fragment so the graph and the store never drift.

    One node per rule, linked to its family, its judging mechanism and (where there is one) the
    muaalem head that judges it. `status` carries the measured verdict, so `build_map.py gaps` surfaces
    the broken detectors as graph gaps rather than prose buried in a report.
    """
    path = path or ROOT / "research_agency_lab/knowledge_map/fragments/08_tajweed_rules.json"
    REF = "datastore/rule_catalogue.py"
    nodes: dict[str, dict] = {}
    edges: list[dict] = []

    def node(nid, typ, label, **kw):  # type: ignore[no-untyped-def]
        nodes.setdefault(nid, {"id": nid, "type": typ, "label": label, "refs": [REF], **kw})

    node("rule:catalogue", "artifact", "Tajweed rule catalogue", status="implemented",
         file=REF, note=f"{len(rows)} rule types, {sum(r[6] for r in rows):,} located instances in engine_diag")
    for mech, lbl in (("segmental", "Segmental test (phoneme-string counterfactual)"),
                      ("durational", "Durational test (length as character repetition)"),
                      ("attribute", "Attribute test (muaalem sifat head)")):
        node(f"mech:{mech}", "method", lbl, status="implemented")

    for rule, fam, mech, sig, cf, head, n, _recs, pr, status in rows:
        node(f"fam:{fam}", "phenomenon", fam.replace("_", " ").title(), status="partial")
        node(f"rule:{rule}", "rule", rule, status=status, note=f"{sig}; counterfactual: {cf}",
             instances=n, legacy_pass_rate=None if pr is None else round(pr, 4), mechanism=mech)
        edges.append({"from": f"rule:{rule}", "to": f"fam:{fam}", "rel": "belongs_to"})
        edges.append({"from": f"rule:{rule}", "to": f"mech:{mech}", "rel": "tested_by"})
        edges.append({"from": "rule:catalogue", "to": f"rule:{rule}", "rel": "catalogues"})
        if head:
            node(f"head:{head}", "model_output", f"muaalem head: {head}", status="implemented",
                 file="research_agency_lab/substrate_library/julia/src/SifatGop.jl")
            edges.append({"from": f"rule:{rule}", "to": f"head:{head}", "rel": "judged_by"})

    path.write_text(json.dumps({"nodes": list(nodes.values()), "edges": edges},
                               ensure_ascii=False, indent=1))
    return path


if __name__ == "__main__":
    raise SystemExit(main())
