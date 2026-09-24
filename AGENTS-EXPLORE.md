# For AI agents: what this repository is, what it knows, and where you can build

This is an invitation. Read it, explore the repository, and come back with what you want to work on.
It is written for capable AI agents: research, data, the consumer apps, or recitation synthesis.
The detailed state and the rules live in `SPRINT4.md`. Read that too.

## 1. What this is

**muqri** is a measuring instrument for Quran recitation in the riwāya of Ḥafṣ ʿan ʿĀṣim. Given audio
and the verses recited, it returns everything measurable about how each letter, characteristic (ṣifah)
and tajwīd rule was realised. The numbers are placed against professional reciters. It is the "brain"
behind consumer learning apps: **the engine measures; the apps decide what to tell the learner.**

## 2. The technology, layer by layer

| layer | what it does | where |
|---|---|---|
| acoustic model | multi-level CTC (muaalem): phonemes plus ten characteristic heads (ghunnah, hams/jahr, shiddah/rakhāwah, tafkhīm/tarqīq, qalqalah, iṭbāq, ṣafīr, tafashshī, istiṭālah, takrīr), 40 ms frames | `research_agency_lab/experiments/learner_eval/` |
| phonemizer | `quran_transcript`: Uthmani text to a phoneme script, with character mappings and context-resolved expected characteristics | used in `app/engine.py` |
| alignment and grading | Viterbi / forward-backward (sub-frame centroids), per-letter identity against classical confusions (GOP), per-letter characteristic margins, rule binding and durational grading in the reciter's own count unit | `app/analysis.py`, `app/submission.py`, `app/rule_bind.py` |
| context awareness | basmala detection; stops from the waveform; the stop table (every one of 71,197 word boundaries: what a stop changes); wajh inference (qaṣr / tawassuṭ of the munfaṣil) | `app/engine.py`, `app/waqf.py`, `datastore/waqf_table.py` |
| **measurement contract** | `measurements/1`: numbers only, per recording, word, rule and letter; percentiles and robust z against masters and cohort | `app/measurements.py`, `research_agency_lab/experiments/quran/reference_stats.json` |
| learner model | 45 skills; cohort prior (Ledoit–Wolf); Laplace posterior; basis measured / inferred / prior; whole-Quran projection; knowledge-space learning paths | `app/learner.py`, `app/knowledge.py` |
| anatomy of sound | waveform acts per letter (collision, voiced hold, separation burst, qalqalah echo); Julia, with an Octave mirror | `research_agency_lab/substrate_library/julia/anatomy.jl` |
| calculus of characteristics | Formal Concept Analysis of the ṣifāt (derives qalqalah = jahr ∩ shadīd, minus the hamza); off-target residuals by tier; coupling | `…/julia/calculus.jl` |
| causal and structural layer | causal edges tested within (letter, reciter); knowledge space (Doignon–Falmagne); Chow–Liu; PC; d-separation with reciter/verse decomposition; a latent "precision state" | `…/julia/causal.jl`, `structures.jl`, `reciter_sets.jl` |
| recitation graph | Reciter, Letter, Characteristic, Feature, Rule, Measure and Concept nodes; realises / keeps / prerequisite-of / ready-for / fails-with / travels-with / causes / near edges; Cypher queries (embedded Kùzu), Neo4j export | `datastore/recitation_graph.py` |
| compute | Modal CPU fan-out grades 11,996 verses in about 5 minutes; Modal GPU dumps posteriors for new audio | `research_agency_lab/compute_bridge/` |
| human in the loop | `/review` (a certified reciter confirms or rejects predicted mistakes by ear, with Uthmani live highlight); `/protocol` (paired correct / scripted-mistake recordings) | `app/review.py`, `app/protocol.py`, `app/webapp.py` |

## 3. The data you can use

- **T300:** 41 professional reciters × ~300 verses = 11,996 verses of per-frame posteriors for every
  head. They're on the Modal volume `qaari-runs` (`muaalem/T300/`) and local
  (`research_agency_lab/experiments/qaari_keys/modal_T300/`).
- **Graded per-verse records** for all of them: `research_agency_lab/experiments/profiles/clips.jsonl.gz`.
  Reciter profiles, covering tempo class, own madd lengths, characteristics and steadiness:
  `reciter_profiles.json`.
- **Every consonant** (545,592) with all 22 characteristic-class posteriors and context: regenerate with
  `research_agency_lab/experiments/calculus/extract.py`.
- **Quran-wide inventory:** 184,014 rule instances and 264,324 consonants in context, all 6,236 ayahs:
  `research_agency_lab/experiments/quran/inventory.jsonl.gz`.
- **Stop table:** what stopping at each of 71,197 boundaries changes, tagged by classical rule:
  `research_agency_lab/experiments/waqf/`.
- **Expert labels:** a certified reciter's verdicts and notes, in `research_agency_lab/experiments/review/labels.jsonl`.
- **Results to build on:**
  - `research_agency_lab/experiments/calculus/*.txt|json` (calculus, sets, structures, causal edges, rule layer);
  - `learner_validation.json`;
  - `deep_research/00–09` (surveys: makhārij, ṣifāt, timing, waqf, mistake datasets, synthesis).

## 4. What is established (see `SPRINT4.md` for numbers and caveats)

- **The learner model beats the cohort baseline**, leave-one-reciter-out, even on skills the reciter never
  recited: +4.1% from 1 verse, +27.1% from 10.
- **Masters are distinguishable by anatomy, not only by labels.** Shadda collision and separation are
  +5–6 dB stronger for masters than for fast imams. ب and د keep their voicing through the closure (the
  "voiced hold", a latent characteristic a certified reciter recognised).
- **"Hardest rule" findings can be reading choices.** The munfaṣil's low pass rate was the qaṣr/tawassuṭ
  wajh, bimodal across reciters.
- **The ten heads are near-saturated for professionals.** Mastery shows mostly in timing and anatomy.

## 5. Open territory

**Research on data only this corpus has**
- Per-reciter "fingerprints": letter × context × characteristic tensors, plus timing and anatomy. Clustering,
  embeddings, spectral / topological structure of reciters and schools (Egyptian, Gulf, mujawwad vs murattal).
- Learning-path science: extend the knowledge space to characteristics and letters; test the paths on learner data.
- Latent characteristics: search, like the voiced hold, for patterns the masters share that no rule names,
  and present them for expert confirmation.
- Resolve the precision-state confound (performance vs recording condition) with the anatomy measures.
- Why the ghunnah family is hardest: a reading choice, calibration, or genuine difficulty.
- Difficulty maps of the Quran: hardest verses, words and rule co-occurrences across reciters (after the
  known engine blind spots are removed: sakt, muqaṭṭaʿāt, surah openings).

**For the consumer apps**
- Rare views for learners: letter-level comparison with a chosen master on the same verse; a learner's
  own timing signature over time; which masters they sound most like; verse difficulty for *this* learner.
- Consumer-side logic on top of `measurements/1`: priority, severity and path policies (the engine
  supplies margins, percentiles, z and skill posteriors; the policy is yours to design and evaluate).
- Scoring the world's recitations (Sprint 4): learner datasets, see `deep_research/08`.

**For Quran recitation synthesis (TTS / voice)**
- **The engine as a verifier.** Run any generated recitation through `Engine.analyze` with the verse. The
  `measurements/1` report checks every letter's identity, every characteristic's margin, every rule's
  length in counts, stops and wajh, and places each against the masters. A synthesis system can use this
  as an automatic evaluator, a reward signal (percentiles / z vs masters), or a rejection filter.
- **Targets to condition on.** The phonemizer plus the stop table give the exact expected phoneme string,
  allowed lengths (2/4/6 where free) and the stop/restart forms. Reciter profiles give each master's own
  count unit, madd lengths and anatomy, so a synthesis can be asked to "read like Husary murattal at 0.32 s per count".
- **Fine-grained supervision.** Sub-frame letter timings, characteristic posteriors and anatomy acts on
  real masters are frame-level training targets that plain audio–text pairs don't provide.
- **Mistake synthesis** (`deep_research/09`): controlled errors (time-scale a madd, swap a letter with the
  same voice's confusion, remove a qalqalah burst, drop a ghunnah) to train and test detectors, validated
  against real expert-labelled mistakes.
- **Caution:** synthetic Quran recitation carries religious and ethical weight. Keep generated audio
  clearly labelled, never present it as a human reciter's, and involve qualified reciters in evaluation.

## 6. How to contribute

- Read `SPRINT4.md` (rules, traps, compute), then the files above.
- Standing rules:
  - commit small and often, every commit working;
  - numerics in Julia (`~/julia-1.11.5/bin/julia`), with the numpy path pinned by parity tests;
  - never report a number you haven't measured;
  - beat a baseline before making a claim;
  - no advice text in engine output;
  - push only when the owner approves.
- Start by proposing: what you want to explore, why it matters (research, the apps, or synthesis), what data and
  tools here you'd use, what you'd measure, and what baseline it must beat.
