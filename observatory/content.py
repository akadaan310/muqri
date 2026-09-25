"""The Observatory's statements about the current implementation, as data.

Every entry was written from the code and the runtime of this repository (branch
claude/qaari-eval-engine-btwkjt), not from the README. Status vocabulary:

  IMPLEMENTED            runs in the Sessions engine path served by app/webapp.py
  TESTED                 implemented and covered by tests/ or by recorded measurements on real audio
  PARTIALLY IMPLEMENTED  some of it runs; the gap is named
  EXPERIMENTAL           research code, run and measured, not in the served path
  PLANNED                written down (docs / design files), no code
  CONCEPTUAL             an idea in the documents, no design yet
  NOT VERIFIED           could not be verified from current code/runtime
"""

from __future__ import annotations

NV = "Not verified from current code/runtime."

# ---------------------------------------------------------------- two engines -----------------------
ENGINES = [
    {"name": "Sessions engine (Engine)", "module": "app/engine.py + app/analysis.py + app/submission.py + app/measurements.py",
     "status": "IMPLEMENTED",
     "used_by": "/sessions, /protocol, /analyze, /letters, the letter corpus, the synthesis lab",
     "model": "obadx/muaalem-model-v3_2 (wav2vec2-BERT CTC, multi-level: phonemes + 10 sifat heads), CPU on the VM",
     "reference_data": "research_agency_lab/experiments/quran/reference_stats.json (T300: 41 reciters; masters = the "
                       "anchor tier, 3 Husary recordings), quran/blindspots.json, timing/stretch_model.json",
     "note": "Does NOT load app/data/calibration.json."},
    {"name": "Benchmark pipeline (QaariEvaluator)", "module": "app/pipeline.py + app/aligner.py + app/tajweed_rules/* + app/scoring.py",
     "status": "IMPLEMENTED (batch / benchmarks only)",
     "used_by": "benchmarks/run_benchmark.py (Kaggle and Modal full-Qur'an runs), the README results table",
     "model": "TBOGamer22/wav2vec2-quran-phonetics CTC aligner; speechbrain ECAPA (232-d fingerprint); Praat formants",
     "reference_data": "app/data/calibration.json (v3: Julia calibrate.jl over the studio-all rows)",
     "note": "The README's 'Husary 71 -> 95' calibrated scores are this pipeline, not the Sessions engine."},
]

# ---------------------------------------------------------------- architecture -------------------------
PIPELINE = [
    {"stage": "Audio input", "module": "app/webapp.py::decode_upload (ffmpeg), app/audio.py", "language": "Python",
     "library": "ffmpeg, soundfile, librosa", "model": "-", "input": "any browser/phone recording",
     "output": "16 kHz mono float32", "compute": "CPU, VM", "status": "IMPLEMENTED"},
    {"stage": "Acoustic model", "module": "app/engine.py::posteriors -> research_agency_lab/experiments/learner_eval/muaalem_dump.py",
     "language": "Python", "library": "torch 2.14 (CPU), transformers 5.17, quran-muaalem 0.2.2",
     "model": "obadx/muaalem-model-v3_2", "input": "waveform", "output": "log-posteriors every 40 ms: 43 phoneme "
     "symbols + 10 sifat heads", "compute": "CPU on the VM (GPU on Modal for batch dumps)", "status": "IMPLEMENTED"},
    {"stage": "Reference text", "module": "app/engine.py::reference / reference_text", "language": "Python",
     "library": "quran-transcript 0.6.1 (phonetiser, Hafs, madd lengths 4)", "model": "deterministic",
     "input": "surah:ayah, part of an ayah, or free text", "output": "phoneme string, expected sifat per phoneme, "
     "word spans, Uthmani map", "compute": "CPU", "status": "IMPLEMENTED"},
    {"stage": "Alignment / segmentation", "module": "app/submission.py::walk_alignment, settle_boundaries; "
     "app/engine.py::_with_context", "language": "Python", "library": "numpy CTC Viterbi (app/analysis.py::ctc_viterbi)",
     "model": "deterministic over the model's posteriors", "input": "posteriors + references",
     "output": "frame span per ayah; per-unit frames (40 ms)", "compute": "CPU", "status": "TESTED",
     "evidence": "tests/test_walk_alignment.py; before/after on master audio in the commit log (192e5be, 0ef2ef7)"},
    {"stage": "Letter-level analysis", "module": "app/analysis.py::analyse_clip", "language": "Python",
     "library": "numpy; CTC likelihoods (app/lahn/gop.py)", "model": "GOP-style likelihood ratios over the posteriors",
     "input": "clip posteriors + phonemes", "output": "Unit per letter/vowel: onset, duration, counts, identity margin, "
     "competitor, sifat margins, edge flag, makhraj neighbour margins (drills)", "compute": "CPU",
     "status": "TESTED", "evidence": "tests/test_analysis_parity.py pins it to the Julia CtcGop reference"},
    {"stage": "Makhraj analysis", "module": "app/analysis.py (CONFUSIONS; neighbours), app/letters.py",
     "language": "Python", "library": "numpy", "model": "likelihood ratio vs confusable / neighbouring letters",
     "input": "unit", "output": "identity margin (scored); neighbour margins (reported, drills only)",
     "compute": "CPU", "status": "PARTIALLY IMPLEMENTED",
     "evidence": "identity: scored and used by sessions; neighbour test: reported only, not validated"},
    {"stage": "Sifat analysis", "module": "app/analysis.py (sifat blocks), app/submission.py::judged, app/blindspots.py",
     "language": "Python", "library": "numpy", "model": "muaalem sifat heads (learned)",
     "input": "unit frames", "output": "per head: expected, observed, margin, realised, scored, blind_spot_p",
     "compute": "CPU", "status": "IMPLEMENTED", "evidence": "see the sifat table; synthesis lab shows heads track identity"},
    {"stage": "Vowel / duration analysis", "module": "app/analysis.py (durations, count unit), app/itmam.py, app/stretch.py",
     "language": "Python", "library": "numpy", "model": "deterministic timing; stretch model fitted in Julia",
     "input": "units", "output": "duration_s, duration_counts, count unit per ayah, stretch vs masters",
     "compute": "CPU", "status": "IMPLEMENTED"},
    {"stage": "Tajwid rule analysis", "module": "app/tajweed_rules/parser.py, app/rule_bind.py, app/submission.py::grade_rule, "
     "app/mudud.py, app/waqf.py, app/ghunnah.py, app/tafkhim.py", "language": "Python", "library": "-",
     "model": "deterministic rule parser + measured lengths / heads", "input": "Uthmani text + units",
     "output": "per rule: status pass/short/long/wrong/unconfirmed/no_evidence, counts, band, evidence",
     "compute": "CPU", "status": "TESTED", "evidence": "tests/test_tajweed_rules.py, test_rule_bind.py, test_mudud.py, sessions"},
    {"stage": "Statistical reference", "module": "app/measurements.py (reference_stats), app/blindspots.py, app/stretch.py",
     "language": "Python (fitted offline in Julia)", "library": "-", "model": "empirical quantiles; logistic blind-spot "
     "model; stretch calculus", "input": "measurements", "output": "percentile_masters, percentile_cohort, z_masters, "
     "scored flags", "compute": "CPU", "status": "IMPLEMENTED"},
    {"stage": "Knowledge graph", "module": "datastore/kg.py, datastore/graph.py, datastore/kg_discover.py", "language": "Python + Cypher",
     "library": "Neo4j 2026.09 Community + GDS", "model": "graph analyses (Louvain, FastRP, PMI)",
     "input": "offline datasets", "output": "graph + discoveries.json", "compute": "CPU, VM, 127.0.0.1 only",
     "status": "EXPERIMENTAL", "note": "Not read by the engine at runtime; research layer."},
    {"stage": "Session results", "module": "app/sessions.py::score, app/letter_matrix.py, app/webapp.py (/sessions/*)",
     "language": "Python", "library": "FastAPI", "model": "deterministic scoring against scripted expectations",
     "input": "measurements + exercise definition", "output": "score card (expectations / mistakes / false alarms / "
     "letter matrix); files beside the recording; a line in sessions_results.jsonl", "compute": "CPU",
     "status": "TESTED", "evidence": "tests/test_sessions.py; 5 recorded rounds"},
    {"stage": "Evaluation / iteration", "module": "scripts/rescore_round.py, scripts/validate_round.py", "language": "Python",
     "library": "-", "model": "-", "input": "stored recordings / EveryAyah masters", "output": "rescored cards; master "
     "validation", "compute": "CPU", "status": "IMPLEMENTED",
     "note": "Improvement is manual: a person reads the results, changes code or thresholds, rescores. No training loop."},
]

# ---------------------------------------------------------------- Part 4: what the engine measures ---------
MEASURES = [
    {"item": "Letter identity", "status": "TESTED", "module": "app/analysis.py::analyse_clip (CONFUSIONS, DELETABLE)",
     "algorithm": "CTC likelihood of the reference vs the same context with the letter replaced by each classical "
                  "confusion or deleted; margin = -max log-ratio (nats); confirmed when every alternative loses",
     "features": "muaalem phoneme posteriors over the letter +/- 2 units, 3 frames margin",
     "uncertainty": "margin; blind_spot_p (context model from 41 T300 reciters); scored=false when p >= 0.25",
     "output": "letters[].identity {confirmed, competitor, margin, scored, blind_spot_p}",
     "evidence": "sessions: ḍād->dāl, ṭāʾ->tāʾ voiced cases caught in round 5; sākin ṭāʾ->tāʾ missed; QuranMB v2 "
                 "learner letter substitutions: recall 0.562 at 1.33 reports per clean clip (learner_eval/results/head_to_head.json)"},
    {"item": "Makhārij", "status": "PARTIALLY IMPLEMENTED", "module": "app/analysis.py (CONFUSIONS; neighbours), app/letters.py",
     "algorithm": "identity test above against classical confusions (18 letters have one); a neighbour test against "
                  "letters at adjacent points (all 28) -- reported only, on by default for drill text",
     "features": "same posteriors", "uncertainty": "margins", "output": "letters[].makhraj {point, neighbours, held}",
     "evidence": "no validation of the neighbour test on labelled data; whole-syllable splices flip it (synthesis lab)"},
    {"item": "Ṣifāt", "status": "IMPLEMENTED", "module": "muaalem sifat heads; app/submission.py::judged; app/blindspots.py",
     "algorithm": "expected class (from the phonetiser) vs the head's posterior over the letter's frames; margin "
                  "= log p(expected) - log p(best other)", "features": "10 learned heads",
     "uncertainty": "margin, percentile vs masters, blind_spot_p; takrīr never scored; shiddah on doubled letters and "
                    "istiṭālah off ض not scored", "output": "letters[].characteristics{head}",
     "evidence": "synthesis lab (research_agency_lab/experiments/synthesis/world_lab_*.json): physically devoicing ز د ج "
                 "ع ب غ to 0-6 % voiced did not flip hams/jahr; the heads track letter identity more than the acoustics"},
    {"item": "Vowels", "status": "IMPLEMENTED", "module": "app/analysis.py (short vowels in CONFUSIONS); app/itmam.py",
     "algorithm": "identity test of each short vowel against the other two; itmām (full vowel articulation) measured "
                  "descriptively, not scored", "features": "posteriors; durations",
     "uncertainty": "margin", "output": "letters[] of kind harakah; report.vowel_sequences",
     "evidence": "a dropped ḍammah (كُفُوًا, round 2) was missed; ishbāʿ is not scored (itmam.py is descriptive)"},
    {"item": "Duration", "status": "IMPLEMENTED", "module": "app/analysis.py (Viterbi path), app/submission.py::to_counts",
     "algorithm": "units' frame spans (40 ms); count unit per ayah = median of vowelled-letter spans; counts through a "
                  "fitted scale", "features": "frame counts", "uncertainty": "40 ms resolution; the last unit of a clip "
                  "is not measured (unreliable tail); implausible values (>12 counts) dropped",
     "output": "letters[].duration_s, duration_counts; recording.seconds_per_count", "evidence": "tests/test_mudud.py, test_stretch.py"},
    {"item": "Madd", "status": "TESTED", "module": "app/tajweed_rules/parser.py, app/submission.py::grade_rule, "
     "app/engine.py::_apply_wajh, app/mudud.py, app/stretch.py", "algorithm": "located from text; measured counts vs "
     "nominal band ± tolerance; munfaṣil / ṣilah kubrā judged against a declared or inferred wajh (qaṣr 2 / tawassuṭ "
     "4-5); stretch calculus against masters at the reader's tempo",
     "features": "unit durations", "uncertainty": "z_masters, percentile, tempo allowance",
     "output": "rules[] {status, observed_counts, expected_counts, deviation_counts, stretch}",
     "evidence": "sessions: muttaṣil and ṣilah kubrā cut to 2 counts caught; madd ʿiwaḍ at a stop not measurable"},
    {"item": "Ghunnah", "status": "TESTED", "module": "app/submission.py (NASAL_BAND_OF, nasal_band), app/ghunnah.py",
     "algorithm": "held length of the nasal units vs [2.0, masters' 95th percentile] counts; grades (akmal/kāmilah/"
                  "nāqiṣah/anqaṣ) reported with textbook bands, not scored", "features": "durations; ghonna head",
     "uncertainty": "z_masters", "output": "rules[] ghunnah/idgham_ghunnah/idgham_shafawi/ikhfa/ikhfa_shafawi/iqlab",
     "evidence": "as recorded: ghunnah dropped at إِنَّ missed (r1e1); shaddah ghunnah cut in جَآنٌّ flagged for "
                 "another reason (r1e2); merges without ghunnah caught (r4e1, r4e2, r5e1, r5e3); ghunnah ADDED to "
                 "idghām without ghunnah not caught (r2e3 flagged for another reason, r5e2 missed)"},
    {"item": "Qalqalah", "status": "TESTED", "module": "parser (sughrā/kubrā/akbar), qalqla head on the letter and "
     "its release unit (ڇ)", "algorithm": "head margin on the letter / release", "features": "qalqla head",
     "uncertainty": "margin", "output": "rules[] qalqalah; letters[] ڇ", "evidence": "as recorded: dropped qalqalah "
     "caught at وَقَبَ, يَلِدْ, ٱلصَّمَدُ (r2) and ٱلْقَدْرِ (r5); on the qāf of وَٱلطَّارِقِ (r1e3) and ٱلْفَلَقِ "
     "(r2e2) flagged for another reason, not by the qalqalah check"},
    {"item": "Nūn sākinah / tanwīn", "status": "TESTED", "module": "parser: izhar_halqi, idgham_ghunnah, idgham_no_ghunnah, "
     "iqlab, ikhfa; submission.py::_izhar_hold", "algorithm": "iẓhār: letters present (segmental) + nūn sākin hold "
     "<= 2.5 counts (calibrated on 804 master instances); idghām/iqlāb/ikhfāʾ: nasal band; idghām without ghunnah: "
     "segmental only", "features": "durations, identity", "uncertainty": "counts vs band",
     "output": "rules[]", "evidence": "hidden iẓhār caught after the hold check (3.62 counts); ghunnah added to "
     "idghām without ghunnah not detectable (no evidence path)"},
    {"item": "Mīm sākinah", "status": "TESTED", "module": "parser: ikhfa_shafawi, idgham_shafawi, izhar_shafawi",
     "algorithm": "ikhfāʾ / idghām shafawī: nasal band; iẓhār shafawī: segmental", "features": "durations",
     "uncertainty": "counts vs band", "output": "rules[]", "evidence": "clear mīm in place of ikhfāʾ shafawī caught "
     "(round 5); idghām shafawī without ghunnah caught (round 5)"},
    {"item": "Rāʾ", "status": "IMPLEMENTED", "module": "parser (tafkheem/tarqeeq/jawaz_wajhayn raa cases); tafkheem head; "
     "app/tafkhim.py", "algorithm": "context rule from text; tafkheem_or_taqeeq head margin", "features": "head",
     "uncertainty": "margin; rā' at a stop after kasrah is a known unreliable context (expert labels round 3)",
     "output": "rules[] tafkheem/tarqeeq; letters[]", "evidence": "rā' made heavy with kasrah: see round 6 design "
     "(unrecorded); listening labels in review/labels.jsonl"},
    {"item": "Lām of the Divine Name", "status": "IMPLEMENTED", "module": "parser (tafkheem/tarqeeq lam of the Divine Name)",
     "algorithm": "context rule; tafkheem head", "features": "head", "uncertainty": "margin",
     "output": "rules[]", "evidence": "TTS probe: light lām detected as wrong in 4/4 voices (synthesis/tts_probe_112.json)"},
    {"item": "Waqf / stopping", "status": "PARTIALLY IMPLEMENTED", "module": "app/waqf.py", "algorithm": "pause "
     "detection from the waveform (energy); a stop inside a word is an error; madd at stops re-evaluated; stop "
     "PERMISSIBILITY not judged -- the text source has no waqf signs", "features": "waveform energy",
     "uncertainty": "-", "output": "recording.stops[]", "evidence": "tests/test_protocol.py; waqf.py docstring"},
    {"item": "Pace", "status": "IMPLEMENTED", "module": "app/submission.py::_mastery, app/stretch.py",
     "algorithm": "count unit per ayah; tempo class ḥadr < 0.20 s <= tadwīr < 0.30 s <= taḥqīq; 'too long' also "
                  "needs to be long in time vs the anchors' count unit", "features": "durations",
     "uncertainty": "-", "output": "recording.seconds_per_count, tempo_class, unit_by_ayah_s",
     "evidence": "round 2e4 (two speeds of one passage)"},
    {"item": "Other tajwīd rules", "status": "PARTIALLY IMPLEMENTED", "module": "parser + rule_bind",
     "algorithm": "hamzat al-waṣl, idghām mithlayn/mutajānisayn/mutaqāribayn, sifat rules (hams, jahr, shiddah, "
                  "rakhāwah, ṣafīr, itbāq, tafashshī, istiṭālah, takrīr) located and judged by heads; sakt: the "
                  "text has no sakt mark, so no sakt rule is located", "features": "heads, identity",
     "uncertainty": "margins", "output": "rules[]", "evidence": "tests/test_tajweed_rules.py"},
]

# ---------------------------------------------------------------- Part 5: the 17 makharij -----------------
_MK = [(1, "al-jawf", "ا و ي (madd)"), (2, "aqṣā al-ḥalq", "ء ه"), (3, "wasaṭ al-ḥalq", "ع ح"), (4, "adnā al-ḥalq", "غ خ"),
       (5, "aqṣā al-lisān (soft palate)", "ق"), (6, "aqṣā al-lisān (a little forward)", "ك"), (7, "wasaṭ al-lisān", "ج ش ي"),
       (8, "ḥāfat al-lisān", "ض"), (9, "edge of the tongue, front", "ل"), (10, "tip, below ل", "ن"),
       (11, "tip and its back", "ر"), (12, "tip, upper incisor roots", "ط د ت"), (13, "tip, near lower incisors", "ص ز س"),
       (14, "tip, upper incisor edges", "ظ ذ ث"), (15, "lower lip + upper incisors", "ف"), (16, "the two lips", "ب م و"),
       (17, "al-khayshūm", "ghunnah of ن م")]
_CLASSICAL = {"ض": "د ظ", "ظ": "ز ذ ض", "ذ": "ز د", "ث": "س ت", "ص": "س", "س": "ص ث", "ط": "ت", "ت": "ط", "ح": "ه",
              "ه": "ح", "ع": "ء", "ء": "ع", "ق": "ك", "ك": "ق", "غ": "خ", "خ": "غ ح", "ز": "ذ ظ", "د": "ض ذ"}


def makharij() -> list[dict[str, str]]:
    rows = []
    for n, name, letters in _MK:
        ls = [c for c in letters.split() if len(c) == 1]
        classical = ", ".join(f"{c}: {_CLASSICAL[c]}" for c in ls if c in _CLASSICAL)
        if n == 1:
            rep, det, ev = ("madd units ا ۦ ۥ; their duration", "identity vs the other madd letters; length rules",
                            "madd rules in sessions")
        elif n == 17:
            rep, det, ev = ("ghonna head; nasal hold duration", "ghunnah rules (nasal band); ghonna head",
                            "ghunnah length tested in sessions; nasality itself not independently measured")
        else:
            rep = "muaalem phoneme posteriors (no articulatory or formant feature in the served path)"
            det = (f"identity vs classical confusions ({classical}) -- scored" if classical else
                   "identity vs deletion only (no classical confusion) -- scored") + \
                  "; neighbour-point test (app/letters.NEIGHBOURS) -- reported, drills only"
            ev = "no per-makhraj validation set; whole-syllable swaps in the synthesis lab flip it"
        rows.append({"n": str(n), "makhraj": name, "letters": letters, "representation": rep, "detection": det,
                     "evidence": ev, "score": "identity margin (nats); neighbour margins (nats)",
                     "status": "PARTIALLY IMPLEMENTED",
                     "note": "The makhraj is represented as a label on each letter (app/letters.py) and in the "
                             "graph (16 Makhraj nodes). It is not measured independently of letter identity."})
    return rows


# ---------------------------------------------------------------- Part 6: the 17 sifat ---------------------
SIFAT = [
    ("hams", "همس", "ف ح ث ه ش خ ص س ك ت", "voicelessness / breath", "hams_or_jahr head", "margin (nats)", "IMPLEMENTED",
     "head tracks letter identity: removing voicing (Praat 0-6 %) did not flip it"),
    ("jahr", "جهر", "the other 19", "voicing", "hams_or_jahr head", "margin", "IMPLEMENTED", "as above"),
    ("shiddah", "شدة", "ء ج د ق ط ب ك ت", "closure + burst", "shidda_or_rakhawa head (3 classes); not scored on doubled letters",
     "margin", "IMPLEMENTED", "a French-j ج (no closure) is a round 6 scripted mistake, unrecorded"),
    ("tawassuṭ", "توسط", "ل ن ع م ر", "partial flow", "shidda_or_rakhawa head, middle class", "margin", "IMPLEMENTED", NV),
    ("rakhāwah", "رخاوة", "the rest", "continuous frication/flow", "shidda_or_rakhawa head", "margin", "IMPLEMENTED", NV),
    ("istiʿlāʾ", "استعلاء", "خ ص ض غ ط ق ظ", "tongue-root raising (low F2 of vowel)", "tafkheem_or_taqeeq head "
     "(the audible face of istiʿlāʾ)", "margin", "IMPLEMENTED", "formant warps moved the head monotonically; flipped only ص"),
    ("istifāl", "استفال", "the rest", "no raising", "tafkheem_or_taqeeq head", "margin", "IMPLEMENTED", NV),
    ("iṭbāq", "إطباق", "ص ض ط ظ", "tongue body to palate", "itbaq head", "margin", "IMPLEMENTED",
     "whole-syllable swaps ط->ت, ض->د, ص->س flip it; ṭāʾ made light caught when voweled"),
    ("infitāḥ", "انفتاح", "the rest", "-", "itbaq head", "margin", "IMPLEMENTED", NV),
    ("idhlāq", "إذلاق", "ف ر م ن ل ب", "none -- a classification of letters by fluency of articulation",
     "none (declared only, app/letters.py)", "-", "NOT MEASURED (by design)", "-"),
    ("iṣmāt", "إصمات", "the rest", "none", "none (declared only)", "-", "NOT MEASURED (by design)", "-"),
    ("ṣafīr", "صفير", "ص ز س", "sibilant high-frequency peak", "safeer head", "margin", "IMPLEMENTED",
     "ṣ->s syllable swap kept ṣafīr (shared), as expected"),
    ("qalqalah", "قلقلة", "ق ط ب ج د", "release echo after closure", "qalqla head on letter and release unit (ڇ)",
     "margin", "TESTED", "caught at وَقَبَ, يَلِدْ, ٱلصَّمَدُ (r2), ٱلْقَدْرِ (r5); qāf cases flagged for another reason (r1e3, r2e2)"),
    ("līn", "لين", "و ي sākinah after fatḥah", "glide", "no head; madd_leen rule times it; the identity test is "
     "skipped for a leen glide", "counts", "PARTIALLY IMPLEMENTED", NV),
    ("inḥirāf", "انحراف", "ل ر", "deviation of the airflow", "no head (declared only)", "-", "NOT MEASURED", "-"),
    ("takrīr", "تكرير", "ر", "trill (to be concealed)", "tikraar head -- DESCRIPTIVE, never scored "
     "(submission.DESCRIPTIVE_HEADS)", "margin (reported)", "IMPLEMENTED (reported only)", NV),
    ("tafashshī", "تفشي", "ش", "spread frication", "tafashie head", "margin", "IMPLEMENTED",
     "doubled shīn: 100 % 'failure' across professionals -> treated as a blind spot (GRAPH.md)"),
    ("istiṭālah", "استطالة", "ض", "lateral lengthening", "istitala head; scored on ض only", "margin", "IMPLEMENTED",
     "ض->د syllable swap flips it"),
    ("ghunnah", "غنة", "ن م (not one of al-Jazari's 17; the model has a head)", "nasal resonance + hold",
     "ghonna head + nasal-hold duration rules", "margin; counts", "TESTED", "hold length tested; nasality added to a "
     "non-nasal context not detected"),
]

# ---------------------------------------------------------------- models ------------------------------------
MODELS = [
    {"name": "obadx/muaalem-model-v3_2", "version": "v3.2 (quran-muaalem 0.2.2)", "purpose": "phoneme + 10 sifat posteriors",
     "input": "16 kHz audio", "output": "CTC log-posteriors per 40 ms", "where": "VM CPU (sessions); Modal T4 (batch dumps), "
     "Modal 16-CPU (letter corpus)", "training": "pretrained (third party); not fine-tuned here",
     "interaction": "the Sessions engine's only learned component; every identity / sifat / duration measurement is "
                    "computed deterministically from its posteriors"},
    {"name": "TBOGamer22/wav2vec2-quran-phonetics", "version": "HF latest at install", "purpose": "CTC forced aligner",
     "input": "audio + text", "output": "phone alignment", "where": "benchmark pipeline (Kaggle/Modal CPU)",
     "training": "pretrained", "interaction": "benchmark pipeline only (app/aligner.py)"},
    {"name": "speechbrain/spkrec-ecapa-voxceleb", "version": "SpeechBrain >= 1.0", "purpose": "speaker embedding in the "
     "232-d fingerprint", "input": "audio", "output": "192-d embedding", "where": "benchmark pipeline",
     "training": "pretrained", "interaction": "reciter-similarity features in the pipeline; not in Sessions"},
    {"name": "tarteel-ai/whisper-tiny-ar-quran", "version": "HF", "purpose": "verse detection (/detect)",
     "input": "audio head", "output": "transcript -> candidate verses", "where": "VM CPU",
     "training": "pretrained", "interaction": "suggests verses; does not score"},
    {"name": "tarteel-ai/whisper-base-ar-quran", "version": "HF", "purpose": "transcription of external datasets",
     "input": "audio", "output": "text", "where": "Modal (modal_external.py)", "training": "pretrained",
     "interaction": "research"},
    {"name": "obadx/recitation-segmenter-v2", "version": "HF", "purpose": "pause segmentation for the TTS pilot",
     "input": "audio", "output": "segments", "where": "Modal (modal_synth.py prep)", "training": "pretrained",
     "interaction": "synthesis pilot only"},
    {"name": "Matcha-TTS (trained here) + NVIDIA BigVGAN-v2 44 kHz", "version": "pilot, epoch 824", "purpose": "Husary-voice "
     "TTS pilot", "input": "phoneme symbols", "output": "44.1 kHz audio", "where": "Modal H100 (train), T4 (synth)",
     "training": "Matcha trained from scratch on 47.8 min of Husary; BigVGAN pretrained", "interaction": "experimental; "
     "graded by the Sessions engine (SYNTHESIS.md)"},
    {"name": "WORLD vocoder (pyworld)", "version": "pip latest", "purpose": "characteristic transforms (synthesis lab)",
     "input": "audio", "output": "audio", "where": "VM CPU", "training": "signal processing (no learning)",
     "interaction": "experimental"},
    {"name": "Microsoft Edge neural TTS (edge-tts)", "version": "service", "purpose": "TTS probe", "input": "text",
     "output": "mp3", "where": "external web service", "training": "third party", "interaction": "experimental probe"},
]

REASONING = {
    "DETERMINISTIC RULES": "tajwīd rule location (app/tajweed_rules/parser.py), the phonetiser (quran-transcript), "
                           "rule binding, scoring against bands, session scoring",
    "SIGNAL PROCESSING": "decoding (ffmpeg), pause/stop detection from the waveform (app/waqf.py), audio cleaning "
                         "(letter corpus), Praat formants and Octave features (pipeline / research), WORLD and "
                         "phase-vocoder edits (synthesis lab)",
    "STATISTICAL MODELS": "count-unit scaling, master reference quantiles and robust z (reference_stats.json), "
                          "logistic blind-spot model (blindspots.json), stretch calculus (stretch_model.json), "
                          "calibration bands (pipeline, calibration.json); fitted offline in Julia",
    "ML MODELS": "muaalem v3.2 posteriors (the Sessions engine), wav2vec2 aligner and ECAPA (pipeline), "
                 "Whisper (verse detection)",
    "LLMs / AGENTS": "none in the engine or the web app (no LLM API client in app/ or datastore/). The code was "
                     "written with an AI coding agent; that is development, not runtime.",
    "NEURAL AUDIO MODELS": "Matcha-TTS + BigVGAN pilot (experimental, Modal), edge-tts (external service, probe)",
}

# ---------------------------------------------------------------- Julia / Octave ------------------------------
JULIA_OCTAVE = {
    "summary": "Offline only. No runtime call from the engine (no subprocess / juliacall / oct2py in app/). Julia fits "
               "models and writes JSON the Python engine reads; Octave mirrors and cross-checks algorithms and holds a "
               "feature engine used by the research bridge.",
    "flow": "datasets / benchmark rows / posteriors -> Julia scripts -> JSON artifacts (committed) -> app/*.py reads them "
            "at start-up; Python ports of Julia algorithms are pinned by parity tests",
    "julia": [
        ("src/CtcGop.jl", "CTC Viterbi, GOP, confusion sets -- the reference app/analysis.py mirrors (tests/test_analysis_parity.py)"),
        ("src/QaariLab.jl, calibrate.jl", "robust bands per rule (median/MAD), leave-one-peer-out validation, weight "
                                          "search -> app/data/calibration.json (benchmark pipeline)"),
        ("blindspots.jl", "logistic model of where professionals fail each check -> experiments/quran/blindspots.json (served)"),
        ("stretch.jl", "the stretch calculus / tempo model -> experiments/timing/stretch_model.json (served)"),
        ("src/Frontier.jl, frontier.jl", "extreme-value tails (GPD peaks-over-threshold, Grimshaw), Bures-Wasserstein "
                                         "barycentres, conformal checks; tested 29/29 per commit log"),
        ("src/Discovery.jl, discover.jl", "sparse regression for duration laws (research)"),
        ("ra.jl, letter_report.jl, reciter_sets.jl, causal.jl, structures.jl, calculus.jl, harakat_timing.jl, "
         "sukoon_timing.jl, madd_scale.jl, ...", "research analyses (rā' contexts, letter reference, reciter sets, "
                                                  "causal sketch, timing)"),
    ],
    "octave": [
        ("qaari_features.m", "LPC formants, autocorrelation HNR/voicing, cepstral F0, Hilbert vowel core, nasal "
                             "contrast, burst energy, HF ratio (research bridge: compute_bridge/octave_bridge.py)"),
        ("ctc_viterbi.m, ctc_crosscheck.m, lpc.m", "cross-checks of the CTC and LPC implementations"),
        ("fr_*.m", "the Frontier maths mirrored: GPD fit, POT threshold, Bures-Wasserstein barycentre, AIRM, quantiles"),
    ],
}

# ---------------------------------------------------------------- Modal ---------------------------------------
MODAL = [
    {"app": "qaari-batch", "file": "research_agency_lab/compute_bridge/modal_batch.py", "compute": "CPU",
     "what": "benchmark pipeline over EveryAyah (studio-all: 6 reciters x 6,236 ayahs, 37,411 rows)"},
    {"app": "qaari-muaalem", "file": "compute_bridge/modal_muaalem.py", "compute": "GPU T4 (50 containers max)",
     "what": "muaalem posterior dumps (T300: 42 speakers x 300 ayahs)"},
    {"app": "qaari-profile", "file": "compute_bridge/modal_profile.py", "compute": "CPU 1 core, 60 containers",
     "what": "Sessions engine over the T300 posteriors (reference stats, reciter profiles)"},
    {"app": "qaari-external", "file": "compute_bridge/modal_external.py", "compute": "GPU T4 + CPU",
     "what": "external datasets through the engine (posteriors on GPU, grading on CPU)"},
    {"app": "qaari-synth", "file": "compute_bridge/modal_synth.py", "compute": "GPU T4 / L4 / H100",
     "what": "Matcha-TTS pilot on Husary: prep, train (H100, 75 min), synth"},
    {"app": "qaari-corpus", "file": "compute_bridge/modal_corpus.py", "compute": "CPU 16 cores x 32 GiB x 10",
     "what": "letter corpus measuring: 1,562 ayahs in 183 s"},
    {"app": "qaari-eval", "file": "app/modal_endpoint.py", "compute": "GPU T4, scale to zero", "what": "a serverless "
     "endpoint definition; whether it is deployed now: " + NV},
]

# ---------------------------------------------------------------- synthesis -----------------------------------
SYNTHESIS = {
    "already_implemented": [
        "Letter/rule corpus audio chain (letter_corpus/clean.py): DC removal, 60 Hz zero-phase high-pass, spectral "
        "gating from the recording's quietest 10 % of frames, zero-crossing cuts, 5 ms fades, -20 dBFS -- runs in "
        "assemble.py; its effect on the engine's margins has NOT been measured",
    ],
    "experimentally_tested": [
        "Time-scale modification (librosa phase vocoder) of madd / ghunnah spans spliced back into the ayah: engine "
        "counts move as intended (ghunnah 3.07 -> 1.54 short at x0.4, 5.00 long at x1.8; madd ṭabīʿī 2.06 -> 3.43 at x2)",
        "Syllable substitution (consonant + vowel from the same reciter's neighbour letter): ط->ت, ض->د, ص->س, ق->ك "
        "flip identity and exactly the differing heads; consonant-only substitution does not",
        "WORLD vocoder transforms: devoicing / voicing (aperiodicity + f0), formant warp, nasal pole/zero -- physics "
        "confirmed by Praat for voicing; the model's heads mostly do not follow",
        "Echo removal (-30 dB on the release unit), formant shift by resampling, deletion: little or no effect on the model",
        "Matcha-TTS + BigVGAN-v2 Husary-voice pilot: held-out letters failing identity 9.5 % (real Husary 0.5 %), "
        "words fully correct 38.2 % (89.8 %); 'choppy' by ear",
        "edge-tts probe: 4 voices on al-Ikhlāṣ; letters/vowels confirmed in 3 of 4 verses; no qalqalah, madds 1.2-1.6 counts",
    ],
    "prototype": ["perturb.py --sweep (every letter x neighbours, with same-letter controls): written, run to 50/291 "
                  "splices, not completed"],
    "planned": ["physics verification layer; 10 ms boundary refinement; delta calculus in Julia; per-characteristic "
                "transforms fitted to the deltas; labelled-data generator; TTS -> rule-applied recitation "
                "(docs/epics/EPIC-1.md)"],
    "not_implemented": ["PSOLA / WSOLA (only the phase vocoder is used)", "LPC formant surgery in the synthesis path",
                        "frication shaping", "closure / burst insertion", "voice conversion", "speech inpainting",
                        "articulatory synthesis", "neural vocoder fine-tuning"],
}

QIRAAT = {
    "represented": "Ḥafs ʿan ʿĀṣim only (quran-transcript MoshafAttributes rewaya='hafs'; app/models.py; parser docstring)",
    "variation_within_hafs": "two ṭuruq for the munfaṣil and ṣilah kubrā length: al-Shāṭibiyyah (tawassuṭ 4-5) and "
                             "al-Ṭayyibah (qaṣr 2); declared by the app or inferred per recording (app/engine.py::_apply_wajh). "
                             "The pipeline also carries a tareeq flag.",
    "acoustically_modelled": "only as a length choice (counts); no other reading differences",
    "ten_qiraat": "Not implemented. No other riwāyah (Warsh, Qālūn, al-Dūrī, ...) is represented in code or data.",
    "evidence": "munfaṣil wajh inference over 41 T300 reciters is described in app/engine.py (17 read ~2 counts); " + NV,
}

FACTORY = [
    ("MASTER AUDIO", "EXISTS", "EveryAyah / Quran-MD recordings; ~11 reciters with full-Qur'an measurements"),
    ("LETTER/RULE CORPUS", "EXISTS (pilot)", "298 cells x 5 masters + learner, local"),
    ("CONTROLLED PERTURBATION", "EXPERIMENTAL", "perturb.py, world_lab.py"),
    ("SYNTHETIC AUDIO", "EXPERIMENTAL", "spliced / stretched / vocoded clips (local, gitignored)"),
    ("PHYSICS VERIFICATION", "PARTIAL", "Praat voicing check in one experiment; no general layer yet"),
    ("MUQRI ENGINE", "EXISTS", "Engine.analyze"),
    ("COMPARE EXPECTED vs DETECTED", "EXPERIMENTAL", "perturb.py diff; sessions scoring for human takes"),
    ("TRAIN / IMPROVE", "NOT IMPLEMENTED", "no model is trained on synthetic data; improvements so far are code/threshold changes"),
    ("RETEST", "EXISTS (manual)", "rescore_round.py, validate_round.py"),
    ("NEW MASTER DISTRIBUTIONS", "PARTIAL", "reference_stats / naqisah calibration rebuilt by scripts, by hand"),
]

LIMITATIONS = [
    "The Sessions engine and the benchmark pipeline are different code paths with different models and reference "
    "data; published calibration numbers belong to the pipeline.",
    "The characteristic heads largely follow letter identity (synthesis lab): an independent acoustic test of a "
    "characteristic does not exist in the served path.",
    "40 ms frames; the last unit of a clip is not length-measured (madd ʿiwaḍ at a stop).",
    "Waqf permissibility is not judged (no waqf signs in the text source); sakt is not located.",
    "Round 1-5 recordings are a single certified reader on a phone; expectations and signatures were written by "
    "the same team that tuned the engine, and several fixes were made after seeing a round's results (see the "
    "commit log and the rescore history).",
    "The Neo4j graph's session layer holds 7 exercises / 37 takes -- it was not reloaded after round 2.",
    "Only Ḥafs is represented.",
    "Session recordings live on this VM only (not in git).",
]
