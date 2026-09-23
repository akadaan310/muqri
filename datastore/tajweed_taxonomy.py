#!/usr/bin/env python3
"""The Tajweed taxonomy from the treatises, mapped to what the engine can actually measure.

Source: `Tajweed.txt` (Treatise I — micro-timing and sound architecture; Treatise II — advanced sifāt
dynamics; Treatise III — Itmām al-Ḥarakāt; Treatise IV — comparative qirā'āt and the unified
diagnostic matrix). This encodes its rule taxonomy from the 101 level up to ijazah mastery, and for
each concept records **which of our mechanisms measures it, in which file, and how well** — so the
gap between "what tajweed requires" and "what the engine checks" is a queryable table rather than
prose.

Mechanisms (the same three that cover the parser's rules, plus two the treatises add):

* ``segmental``   the phoneme sequence changes → CTC likelihood ratio (`textswap_gop.jl`)
* ``durational``  a length changes → the same test, since length is phoneme repetition (`ruleswap_gop.jl`)
* ``attribute``   a letter quality → muaalem's sifat heads (`SifatGop.jl`)
* ``consistency`` a *variance* over instances → Taswiyah / tasāwī (`tasawi_run.jl`)
* ``tempo``       the reciter's own baseline and class scales (`sukoon_timing.jl`)
* ``spectral``    formant / energy-envelope evidence, not a CTC question

Status: ``covered`` (measured, with numbers), ``partial`` (mechanism exists, not fully wired or
graded), ``blocked`` (built and run, but the model's 40 ms frame rate makes it unmeasurable),
``uncovered`` (measurable in principle, nothing built), ``out_of_scope`` (not an acoustic
question, or outside Hafs).

    .venv/bin/python -m datastore.tajweed_taxonomy
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datastore.store import connect, run  # noqa: E402

SEG, DUR, ATTR, CONS, TEMPO, SPEC, NONE = (
    "segmental", "durational", "attribute", "consistency", "tempo", "spectral", "-")

# (id, treatise, section, name, mechanism, implementation, status, note)
TAXONOMY: list[tuple[str, str, str, str, str, str, str, str]] = [
    # ---------------- Treatise I: the metronome — micro-timing ----------------
    ("zaman_proportionality", "I", "1.1", "Zamān/Harakah = K constant across tempo", TEMPO,
     "sukoon_timing.jl", "covered",
     "every duration is expressed in the reciter's OWN counts, so the ratio is what is judged"),
    ("three_tempos", "I", "1.2", "Taḥqīq / Tadwīr / Ḥadr baseline (≈350/250/150 ms per harakah)", TEMPO,
     "sukoon_timing.jl", "covered",
     "measured haraka_s spans 0.16–0.40 s over 41 reciters, which straddles all three bands; "
     "classifying each reciter into a mode is a one-line addition"),
    ("physiology", "I", "1.3", "Respiratory / laryngeal / supraglottal mechanics", NONE, "", "out_of_scope",
     "anatomical explanation, not an acoustic measurement"),
    ("harakah_anatomy", "I", "2.1", "Fatḥah / Kasrah / Ḍammah articulatory posture", SPEC,
     "app/sifaat/formants.py", "partial",
     "the phoneme head distinguishes the three vowels and letter_report.jl confirms every expected "
     "vowel is recorded (id_vowel = 1.0000 for all 41 reciters, every tempo); F1/F2 purity is not wired"),
    ("harakah_isochrony", "I", "2.2", "Zamān(Fatḥah) = Zamān(Kasrah) = Zamān(Ḍammah)", CONS,
     "harakat_timing.jl", "blocked",
     "BUILT AND RUN, but not measurable at the model's 40 ms frame rate: a short vowel is 1–2 frames, so "
     "the per-vowel medians all quantise to exactly 1.0 and the isochrony spread is 0.0 for all 41 "
     "reciters. Needs sub-frame onset estimation or a finer-rate alignment, not more data"),
    ("harakah_weight_independence", "I", "2.2", "Vowel length independent of the consonant's weight", CONS,
     "harakat_timing.jl", "blocked",
     "same 40 ms quantisation: heavy-vs-light vowel bias reads exactly 0.0 for every reciter"),
    ("ikhtilas", "I", "2.3", "Ikhtilās — vowel truncated below 1U (≈⅔U)", DUR,
     "harakat_timing.jl", "covered",
     "measured as the short tail of the vowel-duration distribution: anchors 5.18 %, others 4.54 %"),
    ("ishba", "I", "2.3", "Ishbā' — vowel stretched beyond 1U into a madd", DUR,
     "harakat_timing.jl", "covered",
     "the long tail: anchors 1.47 % vs others 2.22 % — anchors over-stretch vowels less, the expected "
     "direction (ishbā' is the error the treatise warns of at slow tempo)"),
    ("sukoon_shiddah", "I", "3.2", "Shiddah — complete stop, shortest sākin duration", TEMPO,
     "sukoon_timing.jl", "covered", "measured per reciter in own counts"),
    ("sukoon_tawassut", "I", "3.3", "Tawassuṭ / bayniyyah (ل ن ع م ر) — medium duration", TEMPO,
     "sukoon_timing.jl", "covered", "the 'لن عمر' set; measured"),
    ("sukoon_rakhawah", "I", "3.4", "Rakhāwah — sustained flow, longest sākin duration", TEMPO,
     "sukoon_timing.jl", "covered",
     "anchor separation rikhw−shadeed 0.861 vs 0.617 for others; mujawwad styles top the ranking"),
    ("madd_tabii_2u", "I", "4.2", "Madd Ṭabī'ī locked at 2 counts", DUR,
     "ruleswap_gop.jl + tasawi_run.jl", "covered",
     "madd_short 93.7 % / madd_long 99.0 %; anchors measure 2.25 own-counts"),
    ("madd_4_5_6", "I", "4.3", "The 4, 5 and 6-count scales", DUR,
     "substrate_library/julia/madd_scale.jl", "covered",
     "the absolute scale is FITTED: measured = −0.920 + 1.360 × nominal by Theil–Sen over 10,600 anchor "
     "instances, recovering four of five levels within 0.15 counts (1.5→1.50, 2→2.04, 4→3.90, 4.5→4.35). "
     "The six-count level reads 7.21 and stays flagged — madd lāzim is genuinely stretched past six in "
     "mujawwad. The earlier 2.25 : 6.20 : 12.0 figure was keyed on phoneme RUN LENGTH, not the nominal"),
    ("ghunnah_four_levels", "I", "5.2", "Marātib al-Ghunnah: Akmal > Kāmilah > Nāqiṣah > Anqaṣ", DUR,
     "app/ghunnah.py", "covered",
     "graded by the structural context the rule-instance join names. Anchor: akmal 2.33 counts (n=25), "
     "kāmilah 2.43 (n=18), nāqiṣah 1.41 (n=18) — nāqiṣah correctly the short one. Akmal and kāmilah "
     "are not separated by length and should not be: the treatise separates them by oral articulation"),
    ("ghunnah_vs_madd", "I", "5.3/6.1", "2U ghunnah ≠ 2U madd; no nasal bleed into the madd", ATTR,
     "SifatGop.jl (ghonna head)", "partial",
     "the head judges nasality; the 'velum snaps shut instantly' cut-off is not measured"),
    ("madd_arid", "I", "6.2", "Madd 'Āriḍ li-s-Sukoon at 2/4/6 with terminal decay", DUR,
     "rule_catalogue", "partial", "located by the parser; not yet a tasāwī class of its own"),
    ("aqwa_al_mudud", "I", "7.1", "Strength order Lāzim > Muttaṣil > 'Āriḍ > Munfaṣil > Badal", NONE,
     "app/mudud.py", "covered",
     "resolved before judging, so the engine grades against what actually governs: with badal (2) and "
     "lāzim (6) on one letter, grading against badal would call a correct six-count hold a gross "
     "over-lengthening. Six resolutions fired on a ten-ayah passage"),
    ("taswiyat_al_mudud", "I", "7.2", "Taswiyah — every instance of a madd category held identically", CONS,
     "tasawi_run.jl", "covered",
     "exactly what the CV measures; anchors more consistent in all six held classes"),
    ("madd_drift", "I", "7.2", "Drift error — a category shrinking over a passage through fatigue", CONS,
     "app/mudud.py", "covered",
     "a slope of given-counts over ayah position, which the tasāwī CV cannot see. Requires a real shift "
     "in the median as well as a slope: on the anchor, madd tabii showed slope −0.10 with start and end "
     "medians both 1.87, which is outliers tilting the line, not fatigue"),
    ("tafkhim_five_levels", "I/II", "3.2", "Five graded levels of tafkhīm by vowel context", ATTR,
     "SifatGop.jl (tafkheem head)", "partial",
     "the head has mofakham / moraqaq / adnā-l-mofakham (3 classes); the treatise grades 5 by whether the "
     "letter carries fatḥah+alif, fatḥah, ḍammah, sukoon or kasrah"),
    ("tafkhim_nisbi", "II", "3.2", "Relative heaviness: isti'lā' without iṭbāq + kasrah", ATTR,
     "SifatGop.jl", "partial", "the distinction exists in the head's 'adnā' class but is not conditioned on the vowel"),
    ("raa_lam_weight", "II", "3.2/12", "Conditional tafkhīm/tarqīq of ر and of the lām of ٱللَّه", ATTR,
     "datastore/letter_reference.py + tests/test_raa_lam_weight.py", "covered",
     "the strictest conditional rules in tajweed — 11 conditions stored as data and each pinned by a test. "
     "The phonetizer applies them correctly, including ر sākin after an original kasrah with an isti'lā' "
     "following (مِرْصَاد heavy) vs without (فِرْعَوْن light), and the jalālah after fatḥah/ḍammah vs kasrah. "
     "Corpus: ر 79.3 % heavy / 20.7 % light, ل 13.7 % / 86.3 %. NOTE the jalālah rule depends on the "
     "PRECEDING word, so it must be validated on full ayahs, never isolated words"),
    ("letter_completeness", "II", "3.1", "Every letter's full 5–7 classical sifāt, applied and not applied", ATTR,
     "letter_report.jl + datastore/letter_reference.py", "covered",
     "10 judged attributes from the heads plus the inherent ones no model can judge (isti'lā'/istifāl, "
     "idhlāq/iṣmāt, inḥirāf, līn) and the makhraj; median 7.36 characteristics per unit"),
    ("hadr_integrity", "I", "8.2", "Under Ḥadr, tawassuṭ/rakhāwah must not collapse into shiddah", TEMPO,
     "sukoon_timing.jl", "covered",
     "this IS the separation metric: fast imams show the smallest rikhw−shadeed gap (Shuraym 0.250)"),
    # ---------------- Treatise II: sifāt ----------------
    ("hams_jahr", "II", "2.1", "Hams vs Jahr — breath vs vocal-fold vibration", ATTR,
     "SifatGop.jl (hams_or_jahr)", "covered", "1.0 % anchor false alarm"),
    ("shidda_rakhawa_attr", "II", "2.2", "Shiddah / Tawassuṭ / Rakhāwah as a letter quality", ATTR,
     "SifatGop.jl (shidda_or_rakhawa)", "covered", "three classes, 1.0 % anchor false alarm"),
    ("istila_istifal", "II", "2.3", "Isti'lā' vs Istifāl — tongue-root elevation", ATTR,
     "SifatGop.jl (istitala)", "partial",
     "the head covers istiṭāla (ض); isti'lā'/istifāl proper has no dedicated head and is inferred from tafkhīm"),
    ("itbaq_infitah", "II", "2.4", "Iṭbāq vs Infitāḥ — trapping sound against the palate", ATTR,
     "SifatGop.jl (itbaq)", "covered", "1.0 % anchor false alarm, n = 2,108 at T300"),
    ("idhlaq_ismat", "II", "2.5", "Idhlāq vs Iṣmāt — ease of production", NONE, "", "out_of_scope",
     "a lexical/morphological constraint, not an audible per-instance quality; no muaalem head"),
    ("letter_strength", "II", "3.1", "Composite letter strength (quwwa) from its sifāt", ATTR,
     "app/ghunnah.py", "covered",
     "a letter's quwwa is the sum of the strong members it carries, judged plus inherent. Anchor: 560 "
     "of 572 strong sifāt realised (0.979); reports which LETTERS were weak, not only which sifāt"),
    ("makharij_clinicals", "II", "4–16", "Per-letter articulation clinics (ء ه ع ح غ خ ق ك ج ش ي ض ل ر ط د ت ص ز س ظ ذ ث ف ب م و)",
     SEG, "textswap_gop.jl + datastore/letter_reference.py", "covered",
     "letter identity and its classical confusions at T300: 216,577 consonant swaps, 99.91 % detected, "
     "99.06 % NAMED; makhraj recorded per letter"),
    ("compound_collisions", "II", "17", "Adjacent-letter collisions and assimilation", SEG,
     "ruleswap_gop.jl (idgham_undo)", "partial", "idghām/shadda covered; not every collision type"),
    ("endurance", "II", "18", "Muscular endurance and self-correction", CONS, "", "uncovered",
     "overlaps madd_drift: degradation over a long pass"),
    # ---------------- Treatise III: Itmām al-Ḥarakāt ----------------
    ("itmam_universal", "III", "1.1", "Universal law of vowel perfection", CONS, "", "uncovered",
     "every harakah fully formed and equal; the measurement is harakah_isochrony plus ikhtilās/ishbā'"),
    ("neutral_sukoon_zero", "III", "1.2", "The neutral sukoon state as the absolute zero point", TEMPO,
     "sukoon_timing.jl", "partial",
     "we measure sakin durations by class; 'zero point' as a reference posture is not modelled"),
    ("vowel_sequences", "III", "3.1–3.3", "Consecutive ḍammah, alternating vowels, ḍammah→sukoon", CONS,
     "", "uncovered",
     "sequence-level vowel integrity (كُتُبُهُ, يَعِدُكُمُ, كُنتُمْ); measurable from the same durations"),
    # ---------------- Treatise IV: qirā'āt and the diagnostic matrix ----------------
    ("ten_readers", "IV", "I–III", "The ten readers' structural systematics", NONE, "", "out_of_scope",
     "the engine is Hafs 'an 'Āṣim only; other readings would need their own reference text and thresholds"),
    ("metric_temporal_pulse", "IV", "IV.1", "Temporal Pulse Isochrony — 1U/2U/4U/5U/6U ratios hold across tempo",
     CONS, "tasawi_run.jl + sukoon_timing.jl", "covered",
     "the treatise's own first evaluation metric; this is the tasāwī CV plus own-counts normalisation"),
    ("metric_formant_integrity", "IV", "IV.2", "Formant Track Integrity — pure /a/ /i/ /u/, zero nasalisation on vowels",
     SPEC, "app/sifaat/formants.py", "partial",
     "formant code exists in the legacy pipeline; not wired to the muaalem path, and 'zero nasalisation "
     "on pure vowels' is checkable with the ghonna head"),
    ("metric_consonantal_envelope", "IV", "IV.3", "Consonantal Envelope — complete closure, clean release",
     ATTR, "SifatGop.jl (qalqla) + ruleswap_gop.jl", "partial",
     "qalqala release covered at 99.84 %; energy-drop during the stop phase is not measured"),
    ("metric_nasal_isolation", "IV", "IV.4", "Nasal Channel Isolation — clear ghunnah, instant cut-off",
     ATTR, "SifatGop.jl (ghonna)", "partial",
     "nasal presence covered; the cut-off transient is not"),
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS tajweed_taxonomy (
    concept_id     TEXT PRIMARY KEY,
    treatise       TEXT,
    section        TEXT,
    name           TEXT,
    mechanism      TEXT,     -- segmental | durational | attribute | consistency | tempo | spectral | -
    implementation TEXT,     -- file that measures it, when there is one
    status         TEXT,     -- covered | partial | blocked | uncovered | out_of_scope
    note           TEXT
);
"""


def fragment(path: Path | None = None) -> Path:
    """Project the taxonomy into a knowledge-map fragment so graph and store cannot drift."""
    path = path or ROOT / "research_agency_lab/knowledge_map/fragments/09_tajweed_treatises.json"
    REF = "datastore/tajweed_taxonomy.py"
    nodes: dict[str, dict] = {}
    edges: list[dict] = []

    def node(nid, typ, label, **kw):  # type: ignore[no-untyped-def]
        nodes.setdefault(nid, {"id": nid, "type": typ, "label": label, "refs": [REF], **kw})

    node("doc:tajweed_treatises", "source", "Tajweed treatises I–IV", status="implemented", file=REF,
         note=f"{len(TAXONOMY)} concepts mapped to engine mechanisms")
    for cid, tr, sec, name, mech, impl, status, note in TAXONOMY:
        node(f"treatise:{tr}", "source", f"Treatise {tr}", status="reference")
        node(f"mech:{mech}", "method", f"Mechanism: {mech}", status="implemented")
        node(f"concept:{cid}", "phenomenon", name, status=status, note=note, section=sec,
             file=impl or "")
        edges.append({"from": f"concept:{cid}", "to": f"treatise:{tr}", "rel": "from_source"})
        edges.append({"from": f"concept:{cid}", "to": f"mech:{mech}", "rel": "measured_by"})
        edges.append({"from": "doc:tajweed_treatises", "to": f"concept:{cid}", "rel": "defines"})
    path.write_text(json.dumps({"nodes": list(nodes.values()), "edges": edges},
                               ensure_ascii=False, indent=1))
    return path


def main() -> int:
    con = connect()
    con.execute(SCHEMA)
    with run(con, "catalogue:tajweed_taxonomy", "loader", {"concepts": len(TAXONOMY)}):
        con.execute("DELETE FROM tajweed_taxonomy")
        con.executemany("INSERT INTO tajweed_taxonomy VALUES (?,?,?,?,?,?,?,?)", TAXONOMY)
    frag = fragment()

    by_status: dict[str, list] = {}
    for t in TAXONOMY:
        by_status.setdefault(t[6], []).append(t)
    total = len(TAXONOMY)
    scoped = total - len(by_status.get("out_of_scope", []))
    cov = len(by_status.get("covered", []))
    part = len(by_status.get("partial", []))

    print(f"tajweed_taxonomy: {total} concepts from the treatises ({scoped} in scope)")
    print(f"graph fragment -> {frag.relative_to(ROOT)}\n")
    for st in ("covered", "partial", "blocked", "uncovered", "out_of_scope"):
        rows = by_status.get(st, [])
        if not rows:
            continue
        print(f"== {st.upper()} ({len(rows)})")
        for cid, tr, sec, name, mech, impl, _s, _n in rows:
            print(f"   [{tr:3s} {sec:8s}] {cid:26s} {mech:12s} {impl or '-'}")
            print(f"        {name}")
        print()
    print(f"COVERAGE: {cov}/{scoped} fully covered ({100*cov/scoped:.0f} %), "
          f"{part} partial ({100*(cov+part)/scoped:.0f} % at least partially), "
          f"{len(by_status.get('uncovered', []))} uncovered")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
