# qaari-eval Codebase Atlas: what the code computes, how accurate it is, what comes next

**Read this first.** This is the single map of the qaari-eval repository (branch
`claude/qaari-eval-engine-btwkjt`) for agents and humans. For every module it gives the algorithm and
its constants (with `file:line`), what goes in and out, which Tajweed phenomenon it measures, how it is
validated, and the **measured** evidence on its accuracy. It ends with the ranked research frontier, the
knowledge-graph hygiene issues and the commands that re-check everything.

- Written 2026-09-23 from a full read of the code. Line numbers are from `HEAD 7a37469`. Four files are
  being edited by another agent right now (`app/calibration.py`, `app/scoring.py`, `app/models.py`,
  `app/pipeline.py`, plus tests): they are documented as committed at HEAD, and the pending changes are
  described in [In flight](#9-in-flight-2026-09-23).
- Evidence sources: deep-research reports `research_agency_lab/experiments/deep_research/00–07_*.md`
  (cited as "00 §2", "06 D2", …), `research_agency_lab/experiments/*.json`, `benchmarks/results/**.jsonl`,
  the session log `~/c3.md`, and checks re-run while writing this (marked **[re-run]**).
- Graph: `graph.json` / `INDEX.md` in this folder. Node ids are given in `backticks` like
  `algo:gop_subst_aware`.

**Status legend**

| Tag | Meaning |
|---|---|
| **implemented** | Built, wired, and working as designed |
| **partial** | Built, but covers only part of the phenomenon, or is built and not wired in |
| **buggy** | Built and wired, but measured to give wrong verdicts on correct recitation, or known to be wrong in the code |
| **proposed** | Designed in a report, not built |

---

## Contents

1. [Headline facts (read these first)](#1-headline-facts-read-these-first)
2. [Data flow end-to-end](#2-data-flow-end-to-end)
3. [app/: the engine, module by module](#3-app-the-engine-module-by-module)
4. [research_agency_lab/: Julia, Octave, compute bridge, experiments](#4-research_agency_lab-julia-octave-compute-bridge-experiments)
5. [benchmarks/, datasets/, tests/, main.py](#5-benchmarks-datasets-tests-mainpy)
6. [Measured accuracy ledger](#6-measured-accuracy-ledger)
7. [Known bugs register](#7-known-bugs-register)
8. [Coverage of the Tajweed curriculum](#8-coverage-of-the-tajweed-curriculum)
9. [In flight (2026-09-23)](#9-in-flight-2026-09-23)
10. [Frontier: what's next, ranked](#10-frontier-whats-next-ranked)
11. [Graph hygiene](#11-graph-hygiene)
12. [How to verify](#12-how-to-verify)

---

## 1. Headline facts (read these first)

1. **The engine cannot detect a wrong letter, and wrong letters raise the score.** Alignment is forced
   onto the canonical text (`app/aligner.py:102`), so a substituted letter still aligns. Its only trace
   is a low CTC posterior. `Calibration.unreliable()` (`app/calibration.py:101`) turns that into
   `SKIPPED`, and `scoring._EXCLUDED` (`app/scoring.py:57`) drops `SKIPPED` from the index. Clear
   errors therefore shrink the denominator instead of lowering the score (missing-not-at-random, 06 D2,
   00 row 1). A `REVIEW` status and a substitution-aware GOP are being built (§9).
2. **Production calibration is off by default.** `app/calibration.py:30` loads
   `app/data/calibration.json`, which is not in the repo. Without it every verdict is the hand-set
   textbook verdict. The only calibration artefact is
   `research_agency_lab/experiments/calibration_husary_strategic_59.json` (Husary only, 59 ayahs,
   13 rule keys, `count_scale` 0.914).
3. **Master reciters score about 60–75, not ≥ 98.** Raw textbook median perfection per ayah (studio
   mode, 59 strategic ayahs): Husary 75.5, Ayyoub 69.1, Muaiqly 62.9, Tablawi 61.1, Budair 59.0,
   Matroud 45.9 (recomputed from `benchmarks/results/local/*.jsonl` **[re-run]**). Husary calibrated
   in-sample: 77.9 (`calibration_husary_strategic_59.json`). The acceptance targets (Husary ≥ 98,
   Dosari adapted ≥ 95, adapter FP reduction ≥ 90 %) have never passed. Those tests are opt-in and
   skipped (`tests/test_benchmarks.py:73`).
4. **Hams/jahr is measured wrongly.** Hams letters: 186 FAIL vs 90 PASS on experts, and every sakin
   ت/ك/ح/ف fails (02 §0.1). Voicing is the wrong correlate; hams is airflow (aspiration/VOT).
5. **Shiddah never sees a closure.** `occlusion_ms` median 0 (94 % WARNING). The sukoon duration ratios
   1.0/1.5/2.2 are falsified: expert medians are about 0.54–0.67 / 0.91–1.0 / 0.76–0.8 (02 §0.2;
   recount **[re-run]**: shiddah 0.667, tawassut 1.0, rakhawah 0.8).
6. **Formants disagree between the two DSP stacks.** Octave vs Python/Praat on 308 spans: F1 and F2
   r = 0.34, p90 absolute error 465 Hz (F1) and 1089 Hz (F2). `core_ms` agrees at r = 0.9975, median
   error 20 ms (`experiments/validation_20260923_115514.json`, 07 §0.1). Treat formant-based discoveries
   as unreliable until the tracker is fixed.
7. **CTC collapse hides long madds.** 25–76 % of long madds measure under half their target for 4 of 5
   reciters (03 §0). The short-syllable beat inflates counts by 1.095, which is exactly the learned
   `count_scale` 0.914 (03 §0.1).
8. **The parser has 12 Hafs orthography bugs** that make a correct reciter FAIL (00 §1, 04 §0). The main
   ones: waqf signs and Hafs marks are deleted by `_DROP`; U+06DC inside يبصط/بصطة creates a false
   sakt; the U+06E0 alif is always silent; a sakt clashes with a valid ayah-end stop. Full list in §7.
9. **The frontier math layer is exact but not wired in.** `Frontier.jl` passes 15/15 identity tests,
   and Julia and Octave agree to 5.49e-15 on synthetic fixtures and 1.02e-13 on 11 real rule keys
   **[re-run for synthetic]**. No Python code consumes it yet. Its B1 beat-normalisation test was
   falsified on the local sample (Fisher separation 0.71 → 0.53, c3.md).
10. **Raw verdicts measure ruler difficulty, not reciter skill.** A Rasch fit on 9,679 instances
    explains 32 % of deviance with rule-key difficulty and 0.4 % with reciter ability. 12 of 48 rule
    keys are near-degenerate; `madd_iwad`, `hamzat_wasl`, `safir` and `jawaz_wajhayn` have zero verdict
    entropy (06 §0.2–0.3).

---

## 2. Data flow end-to-end

```
audio file ──► app/audio.load_audio ─► AudioSignal (16 kHz mono, HPF 40 Hz, −1 dBFS, SNR, opt. denoise)
                                              │
text: surah:ayah ─► app/quran_text ─► Uthmani string(s)
                                              │
pipeline.QaariEvaluator.analyze_signal  (app/pipeline.py:149)
  1 environment_profile(audio) ─► 8-d env; resolve_mode: taraweeh_adapted if RT60>0.6 s or noise>−25 dB  (:132)
  2 [taraweeh_adapted] adapt_acoustics: HPF 120 Hz + WPE + late-reverb suppression           (dereverb.py:258)
  3 TajweedParser.parse(text) ─► ParsedText{units, words, rules}                              (parser.py:278)
  4 aligner.align(audio, parsed) ─► Alignment{unit → (start, end, conf)}   CTC Viterbi         (aligner.py:258)
  5 detect_pauses ─► dynamic_stops (pause ≥0.30 s at a word end) ─► re-parse with stops, re-align  (:167–177)
  6 estimate_tempo ─► global harakah; build_local_tempo ─► ±4 s local harakah (adapted mode only) (:183–186)
  7 TajweedScorer.evaluate: for each RuleInstance ─► VALIDATORS[rule_type] ─► RuleDiagnostic
        + _attach_alignment_metrics (align_conf, align_min_ms, haraka_ms, counts, core_ms, core_counts)
        + Calibration.apply (if app/data/calibration.json exists)                            (scoring.py)
  8 summarize: perfection index (ahkaam, weighted) + sifaat score; consistency_notes; fatigue; pitch
  9 fingerprint: 192 timbre (ECAPA or MFCC stats) + 32 tajweed + 8 env ─► FAISS search (top-k)
 10 report JSON: recitation_summary, detailed_rule_diagnostics, fingerprint, reciter_similarity_match
```

Offline research loop:

```
benchmarks/run_benchmark.py (calibration=None, raw textbook verdicts)  ─► runs*.jsonl
   (local VM, or Modal: compute_bridge/modal_batch.py, or Kaggle: compute_bridge/kaggle_bridge.py)
      │
      ├─► compute_bridge/octave_bridge.py ─► octave/qaari_features.m (independent DSP) ─► agreement r
      ├─► julia/calibrate.jl (QaariLab.calibrate) ─► calibration.json (bands, count_scale, weights, LOO)
      ├─► julia/discover.jl (Discovery.jl) ─► duration law (STLSQ), tempo ODE
      ├─► julia/frontier.jl (Frontier.jl) ─► per-key BW barycenter reference, conformal thresholds, B1 test
      │         └─► --export ─► octave/fr_real_crosscheck.m (numerical agreement)
      └─► benchmarks/summarize.py ─► summary.json/md + index/*.faiss  (re-judges rows with calibration)
compute_bridge/validate_loop.py runs the Octave and Julia legs and writes experiments/validation_*.json
```

Key couplings:

- The `rule_key` function (`app/calibration.py:35`) is the join key everywhere: benchmark rows
  (`run_benchmark.compact`), strategic verse selection, Julia bands, Frontier clouds.
- Duration metrics come in two rulers: `counts` = span / harakah and `core_counts` = `vowel_core_ms` /
  harakah. The core is the Hilbert-envelope vowel core, the same definition as in Octave.
- Stops are an **input**: ayah ends always stop (unless `--no-stop`), and mid-ayah stops come from pauses.
  Nothing judges whether a stop was a good choice.

---

## 3. app/: the engine, module by module

### 3.1 `app/audio.py`: loading and conditioning — **implemented**

- **Algorithm.** Decode (soundfile, else librosa) → replace NaN/Inf → clipping ratio per channel
  (|x| ≥ 0.999·full scale; warn if > 0.1 %) → down-mix → length check 0.3–600 s → resample to 16 kHz
  (soxr HQ, else `resample_poly`) → 4th-order Butterworth high-pass at 40 Hz, zero-phase (`:104`) →
  peak-normalise to 0.891 (−1 dBFS). SNR = p95 − p10 of 25 ms frame RMS dB (`:127`). If SNR < 15 dB
  (`denoise="auto"`), spectral subtraction (`:136`): the noise profile is the mean magnitude of the
  quietest 10 % of STFT frames (32 ms, 75 % overlap), subtracted ×1.5 with a floor of 0.05·|X|.
- **I/O.** Path → `AudioSignal(samples float32, sr=16000, quality)`; quality warnings flow into the report.
- **Validated by.** `tests/test_audio.py` (resample/down-mix, clipping, denoise trigger, silence/short/
  missing rejection, NaN sanitising).
- **Notes.** The benchmark always uses `denoise="never"`.

### 3.2 `app/quran_text.py`: Uthmani text — **implemented**

- Bundled sample (`app/data/quran_uthmani_sample.json`, a few short surahs), else the alquran.cloud
  `quran-uthmani` (Tanzil) edition, cached under `~/.cache/qaari-eval/text/`. `strip_basmala` removes
  a prefixed basmala on ayah 1 (not 1 or 9) by consonant skeleton. `AYAH_COUNTS` validates references;
  `get_full_quran` checks all 6,236 ayahs.
- The Tanzil text carries every Madani pause mark as a separate token. The parser throws them away (§7 B1).

### 3.3 `app/models.py`: shared types — **implemented** (in flight: `REVIEW`)

- `RuleType` (`:10`): 39 rules: 10 madd, 5 noon, 3 meem, ghunnah, qalqalah, 3 idgham classes, 3 weight
  (tafkheem/tarqeeq/jawaz), 2 wasl (hamzat_wasl, sakt), 10 sifaat. `rule_category` (`:112`) maps them to
  9 scoring families. `Status` (`:133`): PASS, WARNING, FAIL, SKIPPED, VALID_NECESSARY_PAUSE.
- `LetterUnit` holds the resolved orthography (vowel, tanween, shadda, sukun, silent, madd_letter,
  synthetic, assimilated, elided, wasla, orig_vowel). `Alignment.span()` = min start / max end of the
  present units.

### 3.4 `app/tajweed_rules/parser.py`: text → letter units → rule instances — **implemented, buggy**

- **Purpose.** Deterministic Hafs ʿan ʿĀṣim rule derivation from Uthmani text. The text is split into
  phrases at every stop or sakt; waqf changes apply at phrase ends, hamzat al-wasl is pronounced at a
  phrase start, and no rule crosses a phrase boundary.
- **Pipeline** (`parse` `:278` → `_parse_phrase` `:341`):
  1. `normalize_text` / `_NORMALIZE` (`:71`): Madani sukun U+06E1 → U+0652, open tanween U+08F0–08F2 →
     tanween, strip tatweel and joiners. `_DROP` (`:85`) deletes U+06D6–06DC, 06DD, 06DE, 06E2, 06E3,
     06E7–06E9, 06EA–06ED, digits and brackets. **This deletes Hafs information (§7 B1).**
  2. `_prepare_words` (`:318`): a token that *contains* U+06DC marks the **previous** word `sakt_after`
     (`:324`, §7 B2). `_expand_muqattaat` (`:209`) spells letter names from `_LETTER_NAMES` (`:102`),
     merging mīm–mīm and nūn + يَنْمُو.
  3. `_make_unit` (`:237`): marks → vowel/tanween/shadda/sukun/maddah; U+06DF **and U+06E0** set
     `silent` (`:254–255`, §7 B3). Dagger alif becomes a synthetic madd alif unit.
  4. `_resolve_wasla` (`:398`): phrase-initial ٱ → ء with fatha before ل, damma if the **third letter
     has damma**, else kasra (§7 B6: no noun or ʿāriḍ-damma lexicon).
  5. `_resolve_madd_letters` (`:427`), `_resolve_assimilation` (`:461`, via `_IDGHAM_TARGETS` `:54`),
     `_resolve_implicit_sukun`, `_apply_waqf` (`:484`: tanween fath + alif → ʿiwaḍ; drop silah;
     remove the last vowel; ة → ه), `_resolve_elision` (`:512`: a word-final madd letter is dropped
     before a sakin/mushaddad next word).
  6. `_detect_rules` (`:526`):
     - Madd (`_detect_madd` `:638`), in "stronger cause" order: silah kubra/sughra (small waw/yaa, by
       a following hamza) → muttasil (4,5), or (4,6) when stopping on the hamza → munfasil → lazim (6,6)
       → ʿāriḍ (2,6) at a stop → ʿiwaḍ → badal (carrier is hamza) → ṭabīʿī (2,2). Targets in
       `_MADD_TARGETS` (`:121`); Tayyibah qasr makes munfasil and silah kubra (2,2) (`:133`). Leen:
       (4,6) before an explicit sukun (ʿayn of كهيعص), 2–6 before a stop.
     - Noon/tanween: throat letters → izhar (1,1); ينمو across words → idgham with ghunnah (2,2)
       (naqis before و/ي); لر across words → idgham without ghunnah; ب → iqlab (2,2); the 15
       `IKHFA_LETTERS` → ikhfa (2,2), heavy if the target is in `HEAVY_LETTERS`. Hafs izhar of يس/ن
       before و via `izhar_exceptions`.
     - Meem sakinah: ب → ikhfa shafawi; م → idgham shafawi; else izhar shafawi ("before و/ف" detail).
     - Idgham classes for assimilated non-ن/م letters: mithlayn, `MUTAJANISAYN_PAIRS` (`:67`),
       `MUTAQARIBAYN_PAIRS` (`:69`); naqis ط → ت handled first.
     - Ghunnah mushaddadah on نّ/مّ not already covered by an idgham.
     - Qalqalah on sakin قطبجد: sughra mid-phrase, kubra at the stop, akbar if also mushaddad.
     - Weight (`_detect_weight` `:703`): HEAVY_LETTERS with fatha/damma → tafkheem (**not** with kasra
       or sukun); raa via `_raa_verdict` (`:740`) → heavy/light/jawaz; lam of the Divine Name by
       skeleton, heavy unless the previous vowel is kasra (or tanween/sukun helper kasra).
     - Hamzat al-wasl dropped in wasl; sakt (`_sakt_rules` `:798`, مَالِيَهْ optional).
     - Sifaat (`_detect_sifaat` `:774`), **sakin letters only** except safir/tafashhi/itbaq: hams/jahr and
       shiddah/tawassut/rakhawah with the expected duration ratio `SUKOON_RATIOS` (`:135`) =
       1.0/1.5/2.2; safir (صسز), tafashhi (ش), istitaalah (ض sakin), takreer (ر mushaddad or sakin),
       itbaq (صضطظ with fatha/damma).
- **Phenomena.** Every rule the engine can judge starts here. Graph: `code:app/tajweed_rules/parser.py`.
- **Validated by.** `tests/test_tajweed_rules.py` (madd classes on 1:7, munfasil/muttasil, elision, noon,
  meem, qalqalah levels, tafkheem/tarqeeq, orthography, basmala), `tests/test_ahkaam.py`
  (muqattaʿat, idgham classes, raa conditions, sakt, dynamic stop, `test_whole_quran_parses` over
  6,236 ayahs when the text is cached).
- **Evidence.** 00 §1 static and dynamic probes: 46 curriculum rows COVERED, 31 PARTIAL, 30 MISSING,
  10 BUG rows + 2 sakt bugs, 3 not observable (00 "Coverage counts").
- **Known bugs.** §7 B1–B9.

### 3.5 `app/aligner.py`: CTC forced alignment — **implemented** (no error detection)

- **Model.** `TBOGamer22/wav2vec2-quran-phonetics` (env `QAARI_ALIGNER_MODEL`), 38-token romanised
  phonetic vocabulary (01 §0). wav2vec2-base, trained on word-level Quran-MD clips (06 D7).
- **Targets** (`phonetic_tokens` `:57`): Quranic-Arabic-Corpus romanisation via `_PHONETIC` (`:48`);
  madd letters → ā/ī/ū (absorbing the carrier vowel); shaddah doubles the consonant; sun-letter
  article written `l-`; tanween adds `n`; **word-initial hamza gets no symbol** (`:79`). Multi-character
  symbols are split into characters (`toks = list(cons)`), so **ث ذ خ ش غ become two tokens** (t+h, d+h,
  k+h, s+h, g+h). Units without any vocab token get a zero-length span with confidence 0
  (`_interpolate_missing` `:268`).
- **Emissions** (`:206`): log-softmax (T×V); audio > 30 s is processed in 20 s windows with 2 s context;
  > 180 s is rejected ("segment it first"). Cached per buffer.
- **Viterbi** (`ctc_forced_align` `:102`): exact best path over the blank-interleaved lattice
  (`ext` length 2N+1; skip transition allowed when `ext[s] ≠ blank` and `ext[s] ≠ ext[s−2]`); requires
  T ≥ N + repeats.
- **Spans** (`spans_from_path` `:146`): a unit starts at the first frame of its first token and ends where
  the next unit starts (trailing blanks belong to the previous unit); the last unit extends toward the
  RMS speech end. **Confidence = mean exp(log p(target)) over the unit's non-blank frames only.** That is a
  target posterior with no competitor normalisation. It saturates because CTC is peaky (01 headline).
- **Other back-ends.** `HeuristicAligner` (`:286`, uniform split over voiced frames snapped to MFCC
  novelty, confidence 0.3); `JsonAlignmentLoader` (`:352`); `get_aligner("auto")` prefers JSON > CTC >
  heuristic.
- **Validated by.** `tests/test_aligner.py` (Viterbi recovers a known segmentation, repeated tokens,
  too-short audio, contiguous spans, phonetic conventions, heuristic monotonicity, JSON loader).
- **Accuracy.** No boundary ground truth exists. Indirect evidence: long madd collapse (03 §0), core_ms
  agreement with Octave (which uses the same spans, so it validates the DSP, not the spans).
- **Known limits.** §7 A1, A3, A4.

### 3.6 `app/segmenter.py`: ayah locator for long recordings — **partial** (not wired)

- `split_on_pauses` (`:59`): speech = frame dB ≥ max(p5 + 6, p95 − 30); cut at pauses ≥ 0.35 s; merge
  pieces < 0.8 s; split chunks > 30 s at the deepest dip in their middle 40 %; pad 80 ms.
- `ctc_semiglobal_align` (`:104`): CTC Viterbi that may start at any word start and end at any word end,
  with a +2 nats **token bonus** on entering a token (otherwise a peaky model prefers a short span padded
  with blanks). Returns the mean log-prob per frame without the bonus; chunks below −1.6 are dropped.
- `AyahLocator.locate` (`:190`): pass 1 whole-passage search per chunk; pass 2 score-weighted longest
  forward chain (`_forward_chain` `:229`, backtrack ≤ 3 words); pass 3 re-search the rest between anchors.
- **Not called by `app/pipeline.py`** (only tests). `tests/test_segmenter.py` covers the DP and chaining.

### 3.7 `app/acoustic/features.py`: shared DSP — **implemented** (formants **buggy**)

| Function | Math | Constants |
|---|---|---|
| `band_energy` `:51` | mean Hann periodogram power in [lo, hi) | n_fft ≥ 512 |
| `spectral_flatness` `:67` | Wiener entropy exp(mean log S)/mean S in band | |
| `short_time_rms` `:90` | frame RMS | default 5 ms / 1 ms |
| `highband_flux` `:100` | half-wave-rectified flux of log1p(100·|X|) above cutoff | 1500 Hz, 8 ms / 1 ms |
| `f0_track` `:158` | Praat `to_pitch_ac` (10 ms, 60–600 Hz), else librosa pYIN | |
| `max_formant` `:177` | 5500 Hz if median F0 > 165 Hz else 5000 Hz | |
| `formants` `:191` | Praat Burg (5 ms step, 5 formants, 25 ms window, pre-emphasis 50 Hz); median F1–F3, B1–B2 over voiced frames that lie within `NUCLEUS_DB` = 6 dB of the window's loudest frame. Fallback: autocorrelation LPC after resampling to 2×ceiling, order 2 + sr/1000, keep roots with bandwidth < 600 Hz | |
| `hnr` `:248` | Praat cross-correlation harmonicity (5 ms, 75 Hz, 1 period); fallback Boersma autocorrelation HNR = 10 log10(r/(1−r)) | |

- **Evidence.** Formants vs Octave: r = 0.34 (F1 and F2), p90 errors 465/1089 Hz on 308 Husary spans;
  errors near 1 kHz are formant-slot swaps (07 §0.1). The two stacks use different settings: Praat Burg
  with ceiling 5000/5500 Hz vs Octave autocorrelation LPC at 16 kHz, order 18, bandwidth < 400 Hz, no
  ceiling (07 §0.2). Neither uses frame-to-frame continuity.

### 3.8 `app/acoustic/tempo.py`: the harakah (beat) — **implemented** (biased)

- `short_syllable_units` (`:49`): pronounced units with a short vowel, no tanween, no shadda, not a madd
  letter, not followed by a madd letter, not in a madd rule or a nasal carrier, and not the first or last
  pronounced unit.
- `estimate_tempo` (`:92`): durations of those CTC spans kept in [40, 900] ms; if ≥ 3, drop outside
  [Q1 − 1.5·IQR, Q3 + 1.5·IQR] and take the median, clipped to [80, 900] ms. Fallback: voiced span ÷
  `expected_total_harakat` (vowel = 1, sukun = 0.5, plus each rule's (lo+hi)/2 − 1).
- **Evidence.** The CTC span of a CV syllable absorbs closures, so the beat is inflated. Husary's natural
  madd reads 2.19 counts, i.e. ×1.095 (03 §0.1); `calibrate.jl` learned `count_scale` = 0.914 = 2/2.19
  independently. The tabīʿī madd is a better ruler (03 §0.1). Husary median beat 320 ms (03 §0 table).

### 3.9 `app/tajweed_rules/base.py`: evaluation context — **implemented**

- `EvalContext.haraka_ms(t)` (`:46`): the local tempo if a pace function is set (adapted mode), else global.
- `voiced_extent` (`:89`): trim the span to frames within 20 dB of its peak (1 ms RMS), then to the voiced
  F0 frames ± 5 ms. Used for nasal holds and izhar.
- `frication_window` (`:110`): searches from 100 ms before the unit to 60 % into it; 20 ms Hann frames,
  5 ms hop; keeps the longest run whose share of energy above 2.5 kHz ≥ max(0.5, peak − 0.15); needs
  ≥ 30 ms.
- `vowel_band_envelope_db` / `vowel_core_ms` (`:143`, `:156`): 4th-order Butterworth 100–1000 Hz
  (zero-phase), Hilbert magnitude, 20 ms **causal** boxcar (mirrors Octave `filter()`), 5 ms hop.
  core_ms = the longest run within 10 dB of the span peak **and** voiced. This is the `core_ms` metric
  cross-checked with Octave (r = 0.9975).
- `silence_runs` (`:184`): runs of 1 ms RMS dB below p95 − `below_peak_db` of at least `min_ms`.
- `band_status` (`:219`): PASS in [lo − tol, hi + tol] (score 1); WARNING if the distance d ≤ tol
  (score max(0.4, 1 − d/2tol)); FAIL beyond (score max(0, 0.4 − (d − tol)/4tol)). Tolerance is absolute.
- `worst` (`:229`): PASS < NECESSARY_PAUSE < SKIPPED < WARNING < FAIL, ignoring SKIPPED.

### 3.10 Rule validators (`app/tajweed_rules/*`)

All durations are in harakat of the local beat. "Raw verdict counts" are PASS/WARNING/FAIL/SKIPPED over
6 reciters (Husary + Ayyoub, Tablawi, Budair, Matroud, Muaiqly), studio mode, 59 strategic ayahs, 354
rows, recomputed from `benchmarks/results/local/*.jsonl` **[re-run]**. These reciters are experts, so
every FAIL is a false positive of the textbook threshold or the measurement.

| Module:function | Rules | Measurement and thresholds | Raw verdicts on experts (P/W/F/S) | Status |
|---|---|---|---|---|
| `mudood_engine.validate_madd` `:47` | 10 madd types | counts = (carrier + madd letter span) / local beat; `band_status` with tol `TOLERANCE` (`:21`): tabii/badal/iwad 0.25, muttasil/munfasil 0.40, lazim 0.30, silah 0.35; ʿāriḍ/leen flexible: over-length judged against hi + 0.5 with tol 1.0. Over-extension before a stop is capped at WARNING. Adapted mode: a madd cut short before a breath = VALID_NECESSARY_PAUSE | tabii 385/255/410/0; muttasil 8/10/72/0; munfasil 16/3/110/0; lazim kalimi 3/5/28/0; ʿāriḍ 90/49/87/0; ʿiwaḍ 0/66/0/0 | implemented; **buggy in effect** (collapse, beat bias; §7 A3) |
| `noon_sakinah.nasal_hold` `:50` | ghunnah, idgham ghunnah, idgham shafawi (2–2.5); ikhfa, iqlab, ikhfa shafawi (2–2) | duration = `voiced_extent` / beat, tol 0.25; nasality = NER − oral reference (median NER of fatha short syllables); NER = 10 log10(E[150–400]/E[750–1100]); contrast ≥ 8 dB strong, ≥ 3 moderate, < 3 oral. Oral with WARNING duration → FAIL. Score 0.6·duration + 0.4·nasal | ghunnah 16/14/135/0; idgham_ghunnah kamil 9/4/75, naqis 12/11/87; ikhfa light 4/29/172, heavy 1/10/33; iqlab 5/14/59; ikhfa_shafawi 0/2/34; idgham_shafawi 3/2/48 | implemented; **thresholds far off for experts** |
| `noon_sakinah.validate_ikhfa` `:140` | ikhfa | + F2 anticipation over the second half: heavy target F2 < 1300 Hz, light > 1700 Hz, else WARNING and score ×0.8 | (above) | implemented (formant-dependent) |
| `noon_sakinah.validate_iqlab_or_ikhfa_shafawi` `:124` | iqlab, ikhfa shafawi | + labial closure = longest silence run 35 dB below peak (≥ 20 ms); > 40 ms turns PASS into WARNING | (above) | implemented |
| `noon_sakinah.clear_letter` `:169` | izhar halqi, izhar shafawi | counts ≤ 1.3 PASS; ≤ 1.6 WARNING (halqi only); else FAIL. Inserted silence > max(150 ms, 0.8 beat) → FAIL | izhar_halqi 66/6/15; izhar_shafawi 129/0/44; before و/ف 46/0/18 | implemented |
| `noon_sakinah.validate_idgham_no_ghunnah` `:210` | idgham without ghunnah | nasal contrast on the target: < 3 dB PASS, < 8 WARNING, else FAIL | 23/5/20 | implemented |
| `meem_sakinah` | 3 shafawi rules | reuse of the noon functions above | (above) | implemented |
| `qalqalah_engine.validate_qalqalah` `:194` | qalqalah sughra/kubra/akbar | segment = span − 150 ms … + 80 ms; `detect_release_burst` (`:74`): closure = 1 ms RMS (5 ms frames) below p98 − 8 dB for ≥ 15 ms; for each closure, the 5 strongest high-band (> 1.5 kHz) flux peaks; release present if flux ratio ≥ 3 and energy rise ≥ 6 dB. PASS if rise ≥ 8 (sughra) or 10 dB (kubra/akbar); weak release (ratio ≥ 2, rise ≥ 4) → WARNING; no closure → FAIL. Akbar: closure hold ≥ 0.75 × 1.5 harakat | sughra 126/36/18; kubra 28/22/3; akbar 3/7/0 | implemented; **level model wrong** (§7 C4) |
| `idghaam_classes.validate_kamil` `:21` | mithlayn, mutajanisayn, mutaqaribayn (kamil) | for stop targets (بتدطقكج), `count_releases` (releases ≥ 40 ms apart) ≥ 2 → FAIL; merged length < 0.9 harakat → WARNING | mithlayn 34/2/0; mutajanisayn 27/3/6; mutaqaribayn 34/2/0 | implemented |
| `idghaam_classes.validate_naqis` `:52` | ط → ت naqis | releases ≥ 2 → FAIL; itbaq retention via `judge_weight` on the vowel before ط (FAIL → WARNING) | 18/2/4 | implemented |
| `raa_lam_rules.validate_weight` `:23` | tafkheem, tarqeeq, jawaz al-wajhayn | formants on the vowel window (15–75 % of the unit); `judge_weight` (§3.11); jawaz always PASS and reports the realised variant | tafkheem 365/15/32/99; raa heavy 265/14/45/60; raa light 27/1/8/42; lam heavy 60/6/18; lam light 11/21/10; jawaz 6/0/0 | implemented (formant caveat) |
| `sakt_wasl.validate_hamzat_wasl` `:27` | hamzat al-wasl | window −120…+60 ms around the next unit; gap > 250 ms → WARNING; gap ≥ 15 ms, next letter not a stop, onset rise ≥ 15 dB → FAIL (glottal stop inserted) | 442/0/0/0 (zero entropy, 06 §0.3) | implemented; uninformative |
| `sakt_wasl.validate_sakt` `:64` | sakt | longest silence (30 dB below peak, ≥ 30 ms) between the two units; breath → FAIL; 200–400 ms PASS; 100–800 WARNING; < 100 FAIL; > 800 FAIL ("full stop"); مَالِيَهْ with no silence = PASS (idgham) | 1/2/27/0 | implemented; **buggy** (ms not beats; §7 B2, B4) |

### 3.11 Sifaat validators (`app/sifaat/*`)

| Module:function | Sifah | Measurement and thresholds | Raw verdicts (P/W/F/S) | Status |
|---|---|---|---|---|
| `formants.build_reference` `:92` / `judge_weight` `:120` | isti'la/istifal (weight), used by raa/lam, itbaq, naqis | Per-vowel light reference: median F1/F2/F3 (and > 3.5 kHz level) over letters not in heavy/guttural/labial/ر/ل/hamza and not within one unit of a heavy, pharyngeal or ر letter, ≥ 2 tokens (`MIN_REFERENCE_TOKENS`). **H = 1 − (F2 − F1)/(F2_ref − F1_ref)** when the reference distance > 100 Hz. Heavy: PASS if H ≥ 0.15 or (fatha and F2 < 1300); WARNING if H ≥ 0.07. Light: PASS if H < 0.10; WARNING < 0.20. No reference: fatha only, F2 ≤ 1400 heavy / ≥ 1700 light (`:33–38`) | see raa/lam rows | implemented. **Evidence:** H orders the tafkheem maratib monotonically: fatha + alif 0.595 > fatha 0.481 > damma 0.201, Spearman ρ = −0.42, p = 6.5e−27, n = 600; Husary 0.77 > 0.63 > 0.20; monotone in 5 of 6 reciters (02 §0.3) |
| `formants.measure_nasality` `:215`, `nasal_verdict` `:233` | ghunnah | NER bands 150–400 / 750–1100 Hz (`:42–45`); strong ≥ 8 dB, weak ≥ 3 dB contrast | (nasal rules) | implemented; confounded by vowel quality, F0 and room bass (07 §A4) |
| `itbaq.validate_itbaq` `:17` | itbaq | `judge_weight(heavy)`; secondary cues F3−F2 widening ≥ 0.1 vs reference and HF (> 3.5 kHz) attenuation ≥ 2 dB; if both are known and both absent, PASS → WARNING | 83/6/13/30 | implemented; ط vs ق contrast weak (H 0.55 vs 0.51, 02 §2) |
| `hams_jahr.validate_hams_jahr` `:76` | hams, jahr (sakin letters only) | Fricatives (`FRICATIVES` `:71`): voicing fraction of the frication window, hams < 0.3, jahr ≥ 0.5. Others: Praat HNR over the central 70 % of the span, hams < 3 dB, jahr > 12 dB, near band ±6 dB → WARNING | hams 70/24/117/79; jahr 302/154/46/182 | **buggy**: 186 FAIL vs 90 PASS on experts; every sakin ت/ك/ح/ف fails; median voicing fraction 1.0 (02 §0.1). Recount **[re-run]**: non-sibilant hams letters (ت ك ح ف) over both modes: 19 PASS, 30 WARNING, 136 FAIL, 78 SKIPPED |
| `hams_jahr.detect_breath` `:37` | breath (used by sakt, pauses) | 20 ms frames; loud = above floor + 8 dB, or within 40 dB of speech and 8 dB below it; breath if loud ≥ 120 ms, voiced fraction < 0.3 and flatness (500–5000 Hz) ≥ 0.15 | – | implemented (no ground truth) |
| `sukoon_spectrum.validate_sukoon_class` `:22` | shiddah, tawassut, rakhawah (sakin) | occlusion = longest silence 25 dB below peak (≥ 15 ms) inside the central 70 %; shiddah PASS if occlusion ≥ 15 ms else WARNING; tawassut/rakhawah PASS if < 30 ms else WARNING/FAIL. Duration ratio span/beat vs 1.0/1.5/2.2 ± 60 % → WARNING | shiddah 5/65/0; tawassut 204/160/0; rakhawah 189/313/38 | **buggy**: occlusion median 0 (94 % WARNING); ratios falsified (medians shiddah 0.54–0.67, tawassut 0.91–1.0, rakhawah 0.76–0.8); 485 spurious rakhawah WARNINGs (02 §0.2) |
| `ghair_mutadhaddah.validate_safir` `:37` | safir | tilt = level above 5 kHz − level 1–4 kHz; spike = frication-window tilt − utterance tilt; ≥ 6 dB PASS, ≥ 0 WARNING; SKIP if the utterance tilt < −45 dB (low bitrate) or no frication window | 104/0/4/204 | implemented; **mostly unscored**: 363/497 SKIPPED, cause undiagnosed (02 §2); zero verdict entropy (06 §0.3) |
| `ghair_mutadhaddah.validate_tafashhi` `:61` | tafashhi (ش) | spectral flatness 2.5–6 kHz ≥ 0.3 PASS, ≥ 0.15 WARNING | 35/7/1/29 | implemented; OK on experts (02 §2) |
| `ghair_mutadhaddah.validate_istitaalah` `:80` | istitaalah (ض sakin) | ratio = span/beat < 1.0 → WARNING; any release (`count_releases`) → WARNING | 5/45/0 | **buggy threshold**: expert median ratio 0.67, 68/75 WARNING (02 §2) |
| `ghair_mutadhaddah.count_taps` `:99` / `validate_takreer` `:120` | takreer (ر mushaddad or sakin) | taps = dips 6 dB below the ±20 ms running max, lasting 6–45 ms (4 ms frames, 1 ms hop); ≤ 1 PASS; mushaddad > 1 FAIL (70 % of the span), sakin > 1 WARNING | 143/25/15 | implemented (195/262 PASS in 02 §2) |

Not implemented at all: idhlaq/ismat (lexical, not observable), inhiraf, leen as a sifah, khafa, hams
of voweled ت/ك, heavy letters with kasra or sukun, tafkheem levels, infitah (02 §0.7, 00 §1C-ii).

### 3.12 `app/calibration.py`: reference-reciter calibration — **implemented, buggy, inactive by default**

- **Math** (HEAD). For a rule key (`rule_key` `:35`: qalqalah level, idgham kamil/naqis, lazim
  harfi/kalimi, ikhfa heavy/light, idgham_ghunnah kamil/naqis, raa vs lam_allah, izhar shafawi before
  و/ف) and each judged metric, a `MetricBand` (`:60`) holds the hull [lo, hi] = [min(m_H, m_C),
  max(m_H, m_C)] (m_H = Husary median, m_C = median of peer medians) and a scale
  s = max(1.4826·MAD pooled, floor). z = distance outside the hull / s, one-sided for "upper"/"lower"
  metrics. `judge` (`:113`) takes **z = max over bands** and maps it with `verdict` (`:153`): z ≤ 2 PASS
  (score 1); ≤ 3 WARNING (1 − 0.6(z − 2)); else FAIL (max(0, 0.4(1 − (z − 3)/3))). `apply` also multiplies
  `measured_harakat` by `count_scale`.
- **Reliability gate** `unreliable` (`:101`): align_conf < `align_conf_min` (Julia sets
  max(0.2, anchor p2)), shortest unit < `align_min_ms` (40 ms), or a duration rule with core_ms ≤ 0 →
  **SKIPPED** (HEAD). This is the skip-bias mechanism (§7 A1).
- **Validated by.** `tests/test_calibration.py`: rule keys, band sidedness, verdict thresholds, override of
  the textbook verdict, one-sided izhar, unreliable → skipped (HEAD; being changed to REVIEW), uncalibrated
  rule passthrough, offline recalibration = live path, calibrated weights.
- **Evidence.** Husary-only calibration on 59 ayahs: 1,276 observations, 13 keys got bands, count_scale
  0.914, raw 71.7 → calibrated 77.9 in-sample (`experiments/calibration_husary_strategic_59.json`). No
  peers were present, so LOO and the weight search returned NaN (`validation_20260923_115514.json`).
  06 D3/D4: max-z has no stated error rate; the piecewise score is not a probability.

### 3.13 `app/scoring.py`: aggregation — **implemented, buggy**

- `VALIDATORS` (`:37`) maps all 39 rule types. `evaluate` runs each validator (exceptions → SKIPPED),
  attaches alignment metrics, then applies calibration.
- `_attach_alignment_metrics` (`:145`): `align_conf` = min confidence over the rule's units,
  `align_min_ms` = shortest unit, `haraka_ms`, `counts`; for madd/noon/meem/ghunnah rules also `core_ms`
  and `core_counts` = core_ms / haraka.
- `summarize_diagnostics` (`:104`): per-category mean score; **perfection index** = weighted mean over the
  8 ahkaam categories with `CATEGORY_WEIGHTS` (`:53`: madd 1.0, noon 1.0, meem 0.8, ghunnah 1.0,
  qalqalah 0.8, idgham 0.8, weight 0.6, wasl 0.5), weighted by the number of instances; **sifaat score**
  separately (weight 1.0). SKIPPED and VALID_NECESSARY_PAUSE are excluded (`:57`).
- `consistency_notes` (`:129`): a text note if muttasil, munfasil or ʿāriḍ counts span > 1.5; not scored.
  This is the only tasāwī check.
- **Evidence against the aggregation.** 06 D5/D6: compensatory mean, ignores item difficulty and
  correlated instances; Rasch analysis in §1 fact 10.

### 3.14 `app/taraweeh_adapter/`: live-recording adaptation — **implemented**

- `dereverb.estimate_rt60` (`:50`): 500–4000 Hz band envelope (20 ms / 10 ms); at each local peak > floor
  + 15 dB, fit the decay from −3 dB down to at most −25 dB (≥ 8 dB range); slope < −5 dB/s → RT = −60/slope;
  median of ≥ 3. Docstring: biased low above ~1 s (true 1.0/1.5 s → 0.7/0.8 s); reliable as a
  reverberant/dry detector. `reverberant` = RT60 > 0.6 s (`:228`).
- `wpe_dereverb` (`:90`, nara_wpe, taps 10, delay 3, 3 iterations, 512/128 STFT, 30 s chunks with 1 s
  cross-fade); `suppress_late_reverb` (`:130`): late power = exp(−2·6.9·Δ/RT60)·(smoothed power 50 ms
  earlier), gain = max(1 − late/P, −15 dB); docstring reports 3–5 dB less reverberant energy in pauses on
  synthetic rooms (WPE alone about 1 dB).
- `adapt_acoustics` (`:258`): 120 Hz high-pass (proximity EQ), then WPE + suppression if RT60 ≥ 0.3 s.
- `environment_profile` (`:238`): the 8-d env vector (RT60, SNR, room volume ≈ (RT60/0.07)³, 80–250 vs
  250–1000 Hz bass boost, spectral tilt dB/octave 100 Hz–5 kHz, background p10 − p95 dB, echo density,
  clipping).
- `fatigue_detector.detect_pauses` (`:45`): non-phonation = quiet (below max(p5 + 6, p95 − 30)) or
  unvoiced flat noise (flatness > 0.25, 12 dB below speech); pauses ≥ 150 ms get a breath flag.
  `dynamic_stops` (`:92`): pauses ≥ 0.30 s at a word end where the text continues → re-parse with a stop.
  `fatigue_report` (`:131`): breaths/min, spectral tilt drift (last third − first third), F0 drift;
  "oxygen depletion" if > 6 breaths/min or tilt drift < −2 dB/oct. `pitch_profile`: reported, never scored.
- `pace_normalizer`: `classify_pace` (`:29`) hadr < 150 ms ≤ tadweer ≤ 220 ms < tahqeeq;
  `hadr_to_tahqeeq_ratio` = 250/beat; `LocalTempo` (`:43`) = median short-syllable duration within ±4 s
  (at least the 5 nearest), clipped [80, 900]; `variability` = CV of the local curve.
- **Validated by.** `tests/test_taraweeh_adapter.py` (synthetic rooms: RT60 detection, reverb reduction in
  pauses, EQ + dereverb applied, environment profile, pace classes, local tempo tracks acceleration,
  breath detection + dynamic waqf, fatigue counts).
- **Evidence on real audio.** The local benchmark shows little difference between modes on EveryAyah
  studio recordings (e.g. Matroud median perfection 45.9 studio vs 54.0 adapted; others within ±2), which
  is expected because these recordings are dry. The ≥ 90 % timing-FP reduction target is untested on real
  Taraweeh audio.

### 3.15 `app/fingerprint.py`: 232-d fingerprint and FAISS — **implemented**

- **Vector.** 192-d timbre (SpeechBrain ECAPA `spkrec-ecapa-voxceleb`, else `MfccStatsEmbedder`: mean/std
  of 48 MFCCs and deltas over voiced frames, 1/(1+k) scaling, L2-normalised) + 32-d Tajweed vector
  (`TAJWEED_FIELDS` `:44`; e.g. haraka_ms, sukoon duration ratios, madd accuracies, mean heaviness index,
  qalqalah burst dB, HNR means, safir spike, sakt ms, perfection) + 8-d environment.
- **Similarity** (`ReciterIndex.search` `:373`): timbre cosine from `faiss.IndexFlatIP(192)`; style =
  exp(−mean(z_q − z_r)²/2) over the Tajweed dims both vectors have, z-scored by index statistics (priors
  `_PRIORS` until ≥ 10 profiles; std floored at half the prior); combined = w·style + (1 − w)·timbre,
  w = 0.6 by default. Environment is stored, never used for matching. The saved 232-d `.faiss` file is a
  storage format (NaN → −1e30 sentinel), not the search index.
- **Evidence.** README "Validation and limitations": on 24 held-out clips (8 reciters × Al-Fatiha 5–7),
  timbre alone ranked the true reciter first 92 % (top-3 96 %); the default 0.6 style weight dropped this
  to 33 % (top-3 50 %). The roadmap C2 "latent bug" premise is false: no SPD covariance reaches FAISS.
- **Validated by.** `tests/test_fingerprint.py` (232 dims, Tajweed vector, timbre identifies the speaker,
  NaN round-trip, merge + benchmark comparison, backend mismatch, MFCC discriminative).

### 3.16 `app/pipeline.py`, `app/api.py`, `app/modal_endpoint.py`, `main.py` — **implemented**

- `QaariEvaluator` (`pipeline.py`) orchestrates §2. `AnalysisOptions.calibration="default"` loads
  `app/data/calibration.json` if present; the benchmark passes `None` to collect raw verdicts.
- `api.create_app`: FastAPI `GET /health`, `POST /analyze` (multipart audio ≤ 50 MB + surah/ayah or text,
  tareeq, mode, benchmark). `modal_endpoint.py`: T4, scale-to-zero, `/warmup`, weights baked at image
  build.
- `main.py`: CLI `analyze`, `rules` (list the rules the parser derives), `serve`.
- **Validated by.** `tests/test_pipeline.py` (report matches spec, CLI writes a report, CLI errors),
  `tests/test_api.py` (skipped here: fastapi not installed).

### 3.17 `app/data/`

- `strategic_verses.json`: 59 ayahs, 515 words, 48 rule keys, `per_key` 6 (built by
  `datasets/strategic_verses.py`).
- `quran_uthmani_sample.json`: offline text for a few surahs.
- `calibration.json`: **absent** (see §1 fact 2).

---

## 4. research_agency_lab/: Julia, Octave, compute bridge, experiments

Julia **1.11.5** at `~/julia-1.11.5/bin/julia` is required; the system Julia 1.10.9 does not match the
Manifest (07 §0.5). Octave 6.4 with `signal` 1.4.1 and `control` 3.4.0.

### 4.1 `julia/src/QaariLab.jl`: calibration — **implemented**

- `load_rows` (`:40`): benchmark JSONL → `Obs` (reciter, mode, surah, ayah, key, rule_type, status, score,
  metrics).
- `robust` (`:77`): median, σ = 1.4826·MAD, p5/p25/p75/p95. `rcv` = σ/|median|.
- `build_bands` (`:132`): per rule key and metric group (`data/metric_spec.json` "alts"), choose the ruler
  with the **lowest anchor rCV** among the alternatives (e.g. `core_counts` vs `counts`), require
  n_anchor ≥ 15 and n_peer ≥ 5; hull = [min, max](anchor median, median of peer medians); scale =
  max(median of σ over anchor and peers, unit floor) with floors `counts` 0.15, `_db` 1.0, `_hz` 40,
  `_ms` 15, `fraction` 0.05, `ratio` 0.05, `taps`/`releases` 0.5, `flatness` 0.02.
- `calibrate` (`:227`): count_scale = 2 / anchor median tabii counts; reliability floor
  max(0.2, anchor p2 of align_conf) and 40 ms; bands; **leave-one-peer-out** re-scoring; weight search.
- `optimise_weights` (`:270`): Nelder–Mead on log category weights, objective −(mean peers − mean imams)
  + 10·Σ max(0, 95 − peer)² + 2·‖log w − log w0‖² (400 iterations).
- `verdict` (`:111`) is identical to Python's.
- **Evidence.** Only Husary data so far; peers/imams NaN. The full `studio-all` Modal collection (Husary +
  peers) was still running at the last session check (c3.md).

### 4.2 `julia/src/Discovery.jl`: model discovery — **implemented** (under-identified)

- `tempo_dynamics` (`:54`): per surah with ≥ 20 ayahs, fit dT/dτ = κ(T∞ − T) + φ (Tsit5) by L-BFGS with
  ForwardDiff on a Huber loss (δ = 15 ms); ΔAIC vs constant tempo. **Never fitted**: the strategic set
  has no surah with ≥ 20 consecutive ayahs (07 §0.4, 03 §0.5).
- `duration_law` (`:114`): STLSQ over Θ = [1, T, nT, n, T², nT², final, final·T] (T in 100 ms units),
  λ ∈ {0, 1, 2, 5, 10, 20, 40, 80} by BIC. Husary: **D_core = −552 + 150·T + 394·n ms, R² 0.68**;
  D_span = −395 + 119·T + 449·n, R² 0.52; implied count scale 0.81 (`experiments/discovery_latest.json`).
  λ = 80 is the edge of the grid; `n` and `n·T` are collinear at near-constant tempo; columns are not
  z-scored (07 §0.3). Interpreted as Klatt-type incompressibility: counts = D/T is biased for large n
  (03 §0.4).

### 4.3 `julia/src/Frontier.jl`: distributional calibration — **implemented, not wired**

Rule-agnostic: every rule key's instances form a multi-metric cloud per reciter.

| Step | Function | Math |
|---|---|---|
| Feature choice | `select_features` `:199` | judged metrics first, then any metric the anchor has on ≥ 90 % of instances (not bookkeeping, not `_flag`), kept only if it adds rank (smallest singular value > 1e-6·σ₁·√n), at most 4 |
| Robust scale | `rscale` `:46` | 1.4826·MAD → IQR/1.349 → SD, never 0 |
| Robust scatter | `ogk` `:62` | Orthogonalised Gnanadesikan–Kettenring (Maronna–Zamar 2002): pairwise U_jk = (s(Y_j+Y_k)² − s(Y_j−Y_k)²)/4, eigenbasis, robust variances; one reweight: scale Σ by median(d²)/χ²_d(0.5), keep d² ≤ χ²_d(0.975), then classical mean/cov |
| Whitening | `build_ref` `:245` | pooled within-reciter OGK scatter S_w of median-centred clouds (anchor + peers); z = S_w^{−½}(x − anchor median) |
| Per-reciter Gaussian | `fit_gauss` `:238` | OGK if n ≥ 2d + 3, else median/cov; shrink C ← (nC + (d+1)I)/(n + d + 1) |
| Reference | `bw_barycenter` `:102` | m̄ = Σwᵢmᵢ; C̄ fixed point of C = Σwᵢ(C^½CᵢC^½)^½ via S ← S^{−½}(Σwᵢ(S^½CᵢS^½)^½)²S^{−½} (Álvarez-Esteban 2016); anchor weight 0.5, peers share 0.5 |
| Distance | `distance` `:266`, `w2_gauss` `:121`, `airm_dist` `:129` | loc² = ‖m − m̄‖²; Bures shape² = tr C + tr C̄ − 2 tr(C̄^½CC̄^½)^½; AIRM = ‖log(C̄^{−½}CC̄^{−½})‖_F; total² = loc² + β·AIRM² (β = 1) |
| Thresholds | `conformal_threshold` `:163`, `frontier_calibrate` `:285` | leave-one-peer-out distances as the null; cut = ⌈(P+1)(1−α)⌉-th order statistic (α_pass 0.2, α_warn 0.1); instance level: Mahalanobis² of single instances vs the LOO reference (α 0.05 / 0.01); "attainable" flag when k > n |
| SPD extras | `le_vec` `:136`, `le_mean`, `karcher_mean` `:145` | log-Euclidean embedding (√2 off-diagonals); AIRM Karcher mean by Riemannian gradient from the log-Euclidean mean |
| OT | `sinkhorn_divergence` `:193` | log-domain Sinkhorn, ε = 0.05; S = OT(a,b) − ½OT(a,a) − ½OT(b,b) |
| B1 test | `beat_normalization` `:368` | local counts = core_ms / (½ median tabii core_ms in the same ayah, excluding itself); robust CV raw/global/local; Fisher separation of log counts across madd keys |

- **Validated by.** `julia/test/test_frontier.jl`: 15/15 **[re-run]** (commuting-case barycenter, fixed
  point, W2 = 0 on itself, 1-D W2, AIRM symmetry and affine invariance, log-Euclidean isometry, Karcher
  midpoint = AIRM geodesic midpoint, OGK vs classical on clean data and with 5 % outliers, conformal order
  statistics, Sinkhorn respects the ground metric). Octave mirror agrees to 5.49e-15 (synthetic)
  **[re-run]** and 1.02e-13 over 11 real keys (c3.md).
- **Evidence.** Run once on local rows: 3 keys at first, 11 keys with `--export` (too little data for
  conclusions). **B1 falsified on the local sample**: Fisher separation 0.71 (global) → 0.53 (local beat)
  (c3.md).
- **Caveat.** With ≤ 9 peers, reciter-level conformal coverage moves in 10 % steps; α = 0.05 needs ≥ 19
  references (06 §0.5).

### 4.4 Julia drivers and data

- `calibrate.jl` → calibration JSON (anchor `Husary_128kbps`, peers and imams from `data/taxonomy.json`).
- `discover.jl` → duration law and tempo ODE per reciter (expected counts from taxonomy: tabii 2, lazim 6,
  muttasil/munfasil/silah kubra 4.5, ghunnah family 2).
- `frontier.jl OUT RUNS... [--export DIR]` → per-key model, summary, B1 test; `--export` writes clouds.
- `frontier_crosscheck.jl DIR` → synthetic fixtures (seed 20260923) for `fr_crosscheck.m`.
- `data/metric_spec.json`: judged metrics per rule type, sides, scale floors, `min_anchor_n` 15,
  `min_peer_n` 5, `collapsed_unit_ms` 40. `data/taxonomy.json`: rule categories, default weights,
  expected counts, roster.

### 4.5 `octave/`

- `qaari_features.m` — **implemented**: independent per-span DSP. core_ms (same Hilbert-core definition),
  voiced fraction (window-corrected normalised autocorrelation peak 70–400 Hz > 0.45), cepstral F0,
  autocorrelation HNR, LPC formants (order 2 + fs/1000 = 18 at 16 kHz, bandwidth < 400 Hz, middle 40 %
  of the core), nasal_db (< 400 vs 400–2500 Hz), burst_db (max mean positive log-spectral flux),
  hf_ratio (> 2.5 kHz share). Agreement with Python: core_ms r = 0.9975, median error 20 ms (n = 485);
  F1/F2 r = 0.34 (n = 308).
- `lpc.m`: `lpc()` via `levinson` (octave-signal 1.4.1 has no `lpc`).
- `fr_*.m` — **implemented**: exact mirror of Frontier.jl (`fr_psd_fun`, `fr_rscale` with type-7
  quantile `fr_quantile7`, `fr_ogk`, `fr_bw_barycenter`, `fr_w2_gauss`, `fr_airm`, `fr_karcher`,
  `fr_conformal`, `fr_sinkhorn_div`, `fr_fit_gauss`), plus `fr_crosscheck.m` (exit 1 if any rel. error
  ≥ 1e-6) and `fr_real_crosscheck.m` (rebuilds every exported key's reference).

### 4.6 `compute_bridge/`

- `octave_bridge.py` — **implemented**: decodes each row's audio, batches spans to `qaari_features.m`
  (one Octave process per worker), writes `diag["octave"]`.
- `validate_loop.py` — **implemented**: samples rows, runs the Octave cross-check (Pearson r, median and
  p90 absolute error), then `calibrate.jl` and `discover.jl`; writes
  `experiments/validation_<ts>.json`. Note: its default `--calibration-out` is `app/data/calibration.json`.
- `modal_batch.py` — **implemented**: Modal app `qaari-batch`; CPU shards (2 CPU, 6 GB, ≤ 90 containers,
  6 h) or T4 (≤ 10); results on Volume `qaari-runs`; entrypoints `smoke`, `launch`, `fire`, `collect`.
- `kaggle_bridge.py` + `kaggle_worker.py` — **implemented**: pushes the code as a private dataset, runs
  shards on Quran-MD WAVs, optional Octave pass.

### 4.7 `experiments/`

| File | What it holds |
|---|---|
| `deep_research/00_curriculum_audit.md` | 120-row Hafs curriculum × engine coverage audit; 12 parser bugs; ranked gap list (Tier 0 bugs, Tier 1) |
| `01_makharij.md` | makhraj verification: segmentation-free GOP, articulatory ground metric, cue verifiers, muaalem second judge |
| `02_sifaat.md` | [DATA] hams/shiddah/qalqalah/tafkheem findings, letter lattice, sifaat build plan |
| `03_timing.md` | beat, collapse mixture, tasāwī models, duration law, timing build plan |
| `04_waqf_ibtida.md` | waqf kinds and signs, 3 parser bugs, boundary posterior, stop-choice DP, raum |
| `05_topology_geometry_graphs.md` | persistence, sheaves, graphs, Lévy area; ten ranked items |
| `06_information_neural.md` | skip bias (D2), Rasch analysis, IRT/Bayes decision stack, muaalem + GOP, P0–P13 plan |
| `07_dsp_julia_discovery.md` | DSP per phenomenon (DO/LATER/SKIP), formant disagreement, discovery architecture |
| `frontier_math_roadmap.md` | A1–A6 calibration, B1–B3 discovery, C1–C3 fingerprint; C2 correction recorded |
| `calibration_husary_strategic_59.json` | Husary-only calibration (13 keys, count_scale 0.914) |
| `discovery_latest.json` / `discovery_husary_strategic_59.json` | duration laws (above); tempo ODE empty |
| `validation_20260923_115514.json` | Octave vs Python agreement numbers |
| `lahn_gop/` | in-flight GOP analysis scripts and results (§9) |

---

## 5. benchmarks/, datasets/, tests/, main.py

- `benchmarks/roster.py`: 8 studio reciters (Husary, Husary Muallim, Minshawy, Hudhaify, Abdul Basit,
  Alafasy, Tablawi, Ayyoub) and 8 taraweeh imams (Dosari, Qatami, Shuraym, Sudais, Juhany, Budair,
  Matroud, Muaiqly); EveryAyah URL; Quran-MD ids for 11 of them.
- `benchmarks/run_benchmark.py`: one JSON row per (reciter, ayah, mode) with compact diagnostics
  (`key` added), summary scores and the fingerprint (studio). Always `calibration=None`,
  `denoise="never"`, CTC aligner; resumable; `--shard i/n`.
- `benchmarks/summarize.py`: re-judges rows with the calibration (`recalibrate` uses the same
  `Calibration.judge` as the live path), aggregates perfection/sifaat/timing FAILs per reciter and mode,
  `rule_gaps` (signed z per reciter and key), builds both FAISS indices.
- `benchmarks/results/`: `local/husary_strategic_raw.jsonl` (59 rows), `local/everyayah_only_strategic.jsonl`
  (590 rows: Ayyoub, Tablawi, Budair, Matroud, Muaiqly × 59 × 2 modes), `runs_shard0/1.jsonl` (214 partial
  rows: Husary, Husary Muallim, Minshawy). Modal `studio-all` results are not yet collected here.
- `datasets/strategic_verses.py`: parses all 6,236 ayahs, buckets rule instances by key, and runs a
  weighted greedy set cover (gain = newly covered / (1 + words/10), anchors Al-Fatiha and 111–114, rare keys
  taken regardless of length). `datasets/index_reciters.py`: studio benchmark + summarize.
- `tests/`: 84 tests, **80 passed, 4 skipped** on 2026-09-23 **[re-run]** (skips: fastapi missing, three
  opt-in acceptance tests). `tests/synth.py` builds Klatt-style source-filter signals with known ground
  truth. Most acoustic tests are synthetic; no test uses real expert audio with labelled errors.

---

## 6. Measured accuracy ledger

Every number that says how good (or bad) a part of the engine is, with its source.

| # | Component | Measurement | Result | Source |
|---|---|---|---|---|
| E1 | Octave vs Python timing | core_ms Pearson r, median / p90 abs err | 0.9975, 20 / 30 ms (n = 485) | `validation_20260923_115514.json` |
| E2 | Octave vs Python formants | F1, F2 r; p90 abs err | 0.34, 0.34; 465 Hz, 1089 Hz (n = 308) | same; 07 §0.1 |
| E3 | Frontier Julia vs Octave | worst relative error | 5.49e-15 synthetic **[re-run]**; 1.02e-13 over 11 real keys | `fr_crosscheck.m`; c3.md |
| E4 | Frontier identities | Julia tests | 15/15 pass **[re-run]** | `julia/test/test_frontier.jl` |
| E5 | Python unit tests | pytest | 80 pass, 4 skip **[re-run]** | `tests/` |
| E6 | Hams validator on experts | PASS vs FAIL | 90 vs 186; all sakin ت/ك/ح/ف fail; median voicing 1.0 | 02 §0.1 |
| E7 | Shiddah validator | occlusion_ms | median 0; 94/100 WARNING | 02 §0.2 |
| E8 | Sukoon duration model | expert medians vs 1.0/1.5/2.2 | 0.54–0.67 / 0.91–1.0 / 0.76–0.8; 485 spurious rakhawah WARNINGs | 02 §0.2; recount **[re-run]** |
| E9 | Qalqalah levels | energy rise kubra vs sughra; akbar | 13.0 vs 13.7 dB (n 98/330); akbar 18 dB, 174 ms hold | 02 §0.6 |
| E10 | Tafkheem maratib by H | Spearman ρ (level, H) | −0.42, p = 6.5e−27, n = 600; monotone in 5/6 reciters | 02 §0.3 |
| E11 | Safir | share SKIPPED | 363/497 (cause undiagnosed) | 02 §2 |
| E12 | Istitaalah | WARNING rate; expert ratio | 68/75; median 0.67 vs threshold 1.0 | 02 §2 |
| E13 | Takreer | PASS rate | 195/262 | 02 §2 |
| E14 | Long madd collapse | share < ½ target | 0–23 % Husary; 25–76 % for 4/5 reciters | 03 §0 |
| E15 | Husary tasāwī reference | robust CV, align_conf > 0.5 | muttasil 0.09 (n 15), tabii 0.14 (n 96) | 03 §0.2 |
| E16 | Beat bias | Husary tabii counts; learned scale | 2.19 → count_scale 0.914 | 03 §0.1; calibration JSON |
| E17 | Relative madd scale across masters | muttasil in tabii units | Husary ≈ 5; Ayyoub/Tablawi ≈ 7–10 | 03 §0.3 |
| E18 | Duration law (Husary) | STLSQ | D_core = −552 + 150T + 394n, R² 0.68; λ at grid edge | `discovery_latest.json`; 07 §0.3 |
| E19 | Tempo ODE | fit | not fitted (no surah ≥ 20 ayahs) | 07 §0.4 |
| E20 | Raw verdict information | McFadden R² key vs reciter | 0.321 vs 0.004; key SD 2.08 vs reciter SD 0.20 logits | 06 §0.2 |
| E21 | Degenerate keys | zero verdict entropy | madd_iwad, hamzat_wasl, safir, jawaz_wajhayn (12/48 near-degenerate) | 06 §0.3 |
| E22 | Perfection on experts (raw) | median per ayah, studio | Husary 75.5, Ayyoub 69.1, Muaiqly 62.9, Tablawi 61.1, Budair 59.0, Matroud 45.9 | local jsonl **[re-run]** |
| E23 | Husary calibrated (in-sample) | perfection | 71.7 raw → 77.9 | `calibration_husary_strategic_59.json` |
| E24 | Skip rate by reciter (raw) | SKIPPED share, studio | Husary 6.2 %, Ayyoub 7.5 %, Muaiqly 8.5 %, Budair 9.8 %, Tablawi 10.2 %, Matroud 14.7 % | local jsonl **[re-run]** |
| E25 | B1 beat normalisation | Fisher separation global → local | 0.71 → 0.53 (falsified) | c3.md |
| E26 | Fingerprint identification | top-1 / top-3 on 24 held-out clips | timbre only 92 % / 96 %; default w = 0.6: 33 % / 50 % | README |
| E27 | RT60 estimator | synthetic bias | true 1.0/1.5 s → 0.7/0.8 s | `dereverb.py` docstring |
| E28 | Late-reverb suppression | pause energy reduction, synthetic | 3–5 dB (WPE alone ≈ 1 dB) | `dereverb.py` docstring |
| E29 | Rasch reciter abilities | θ | Ayyoub +0.30, Tablawi +0.07 above Matroud −0.30, Budair −0.13 (5 reciters; directional only) | 06 §0.2 |
| E30 | Coverage of the curriculum | 120 rows | 46 covered, 31 partial, 30 missing, 10 bug rows (+2 sakt bugs), 3 not observable | 00 |

Reading E24 with fact 1: the reciter with the highest skip rate (Matroud) also has the lowest score, so
skips do not explain the ranking here, but for a learner with real substitutions the mechanism works
against detection (06 D2).

---

## 7. Known bugs register

IDs are stable; cite them in commits and fragments.

### A. Measurement and scoring (engine-wide)

| ID | Bug | Where | Evidence | Fix direction |
|---|---|---|---|---|
| A1 | **Skip bias (MNAR)**: low alignment confidence → SKIPPED → excluded, so wrong letters raise the score; lahn jali is undetectable | `calibration.py:101,127`; `scoring.py:57` | 06 §0.1, D2; 00 row 1 | P0 (06 §8): REVIEW status counted in the index + skip-rate reporting; **in flight** (§9) |
| A2 | No lahn jali detection at all: forced alignment has no competing hypothesis | `aligner.py:102` | 06 D1; 00 rows 1–3 | substitution-aware / segmentation-free GOP (01 §3.1, 06 P5); **in flight** (`app/lahn/gop.py`) |
| A3 | CTC collapse: long madds emitted as 1–2 spike frames; 25–76 % measure < ½ target | aligner spans → `mudood_engine` | 03 §0.2 | collapse-mixture gate (03 §2.2), HSMM re-segmentation (03 §4.3) |
| A4 | Tokenizer: ث ذ خ ش غ split into 2 tokens; word-initial hamza has no token | `aligner.py:75–81` | 01 §0 | sequence-level GOP; second judge `obadx/muaalem-model-v3_2` (single symbols) |
| A5 | Beat inflated ×1.095 by consonant closures in CV spans | `tempo.py:92` | 03 §0.1 | tabīʿī-anchored two-ruler beat (03 §3.1) |
| A6 | Production calibration file absent: textbook verdicts everywhere | `calibration.py:30` | repo state | run `calibrate.jl` on full studio-all data, commit `app/data/calibration.json` |
| A7 | Max-z verdict and piecewise score have no error-rate meaning; compensatory weighted mean | `calibration.py:128–134`, `scoring.py:104` | 06 D3–D6 | surprisal + Cauchy combination, IRT, Bayes decision (06 P2/P4/P7) |
| A8 | Formant tracker disagrees with Octave (slot swaps) | `features.py:191` vs `qaari_features.m` | E2 | harmonise ceiling/order, then KARMA Kalman tracker (07 §A3) |
| A9 | Sakt judged in ms (200–400), not beats | `sakt_wasl.py:20` | 03 §1, §4.8 | beats (1 h fix) |
| A10 | Tasāwī only as an unscored note (> 1.5 counts; 3 madd types) | `scoring.py:129` | 00 row 104; 03 §5 | variance-component tasāwī with TOST (03 §5.2–5.3) |

### B. Parser (Hafs orthography) — the "12 parser bugs"

00 counts **10 BUG rows (#28, 45, 46, 51, 99, 109, 110, 112, 119, 120) plus 2 sakt bugs = 12**. Several
rows are the same defect seen at two levels (#46/#110, #99/#120, #28/#119). Distinct defects:

| ID | Bug | Where | Refs (ayahs) | Source |
|---|---|---|---|---|
| B1 | `_DROP` deletes waqf signs U+06D6–06DB and the Hafs marks U+06E3 (small low seen), U+06EA (imala), U+06EB (ishmam), U+06EC (tas-heel) | `parser.py:85` | all waqf signs; 52:37, 11:41, 12:11, 41:44 | 00 §3 end; 04 §0 |
| B2 | U+06DC read as sakt when it is the "read seen" mark inside a word → false mandatory sakt after the previous word (Bug A) | `parser.py:324` | 2:245 (after يَقْبِضُ), 7:69 (after ٱلْخَلْقِ) | 00 row 54, 112; 04 bug 1 |
| B3 | U+06E0 (rectangular zero) treated like U+06DF → alif always silent, even at waqf (66 occurrences) | `parser.py:254–255`, `_apply_waqf` `:484` | أَنَا۠ (×60), 18:38, 33:10, 33:66, 33:67, 76:15 | 00 rows 99/120; 04 bug 2 |
| B4 | Sakt at an ayah end conflicts with the assumed stop: a valid waqf FAILs "full stop instead of sakt" (Bug B) | `_sakt_rules` `:798` + `validate_sakt` | 18:1, 36:52, 69:28 | 00 row 54, 0.5; 04 bug 3 |
| B5 | ءَا۬عْجَمِىٌّ gets madd lazim 6 instead of tas-heel (no madd) | `_detect_madd` `:638` (+B1) | 41:44 | 00 row 109 |
| B6 | Hamzat al-wasl ibtida' vowel: ٱمْشُوا، ٱقْضُوا، ٱمْرُؤ… get damma (should be kasra); no noun lexicon | `_resolve_wasla` `:398` | e.g. 67:15 area words; nouns ٱبْن، ٱسْم | 00 row 51 |
| B7 | Imala of مَجْر۪ىٰهَا: mark dropped, raa judged heavy | B1 + `_raa_verdict` `:740` | 11:41 | 00 rows 46/110 |
| B8 | Raa at waqf in يَسْرِ / نُذُرِ / أَسْرِ: tafkheem only, tarqeeq (preferred) FAILs | `_raa_verdict` | 89:4, 54:16…, 20:77… | 00 row 45 |
| B9 | الٓمٓ ٱللَّهُ joined: mim must allow 2 or 6; engine requires 6 | `_detect_madd` | 3:1–2 | 00 rows 28/119 |

Related PARTIAL items that come from the same causes (not counted in the 12): madd al-farq tas-heel not
accepted (#34; 6:143–144, 10:51/91, 10:59, 27:59), تَأْمَنَّا raum (#108), ٱلْمُصَيْطِرُونَ sad/seen
(#113), سَلَٰسِلَا two ways (#100), فَمَآ ءَاتَىٰنِۦَ (#101).

### C. Sifaat and rule-specific

| ID | Bug | Where | Evidence |
|---|---|---|---|
| C1 | Hams/jahr uses voicing/HNR; hams is airflow (aspiration/VOT); window contains the next vowel | `hams_jahr.py:76` | E6 (02 §0.1) |
| C2 | Shiddah occlusion uses a −25 dB silence run that never occurs; should reuse the burst detector | `sukoon_spectrum.py:22` | E7 |
| C3 | Sukoon duration ratios 1.0/1.5/2.2 falsified | `parser.py:135` | E8 |
| C4 | Qalqalah kubra assumed louder than sughra | `qalqalah_engine.py:36` | E9 |
| C5 | Istitaalah duration threshold 1.0 harakah vs expert 0.67 | `ghair_mutadhaddah.py:80` | E12 |
| C6 | Safir mostly SKIPPED, cause undiagnosed | `ghair_mutadhaddah.py:37` | E11 |
| C7 | Heavy letters with kasra/sukun get no tafkheem instance; tarqeeq leakage near heavy letters never judged (they are excluded from the reference) | `parser.py:709–712`; `formants.py:68` | 00 rows 48–49 |
| C8 | Zero-entropy keys carry no information (iwad, hamzat_wasl, safir, jawaz) | validators | E21 |

### D. Research-lab issues

| ID | Issue | Evidence |
|---|---|---|
| D1 | Duration law under-identified (one reciter, near-constant tempo; λ at grid edge; unscaled Θ columns) | 07 §0.3 |
| D2 | Tempo ODE cannot be fitted on the strategic set | 07 §0.4 |
| D3 | Julia version drift: Manifest is 1.11.5; `/usr/local/bin/julia` is 1.10.9 | 07 §0.5 |
| D4 | Frontier layer has no consumer in `app/` and too little data (≤ 11 keys) | c3.md |
| D5 | `frontier_math_roadmap.md` C2 premise ("covariance fed to FAISS") was false; corrected in the file | roadmap C2 |

---

## 8. Coverage of the Tajweed curriculum

From 00 (120 rows, Hafs Shatibiyyah), summarised by level:

| Level | Strong (covered) | Weak (partial) | Absent (missing) |
|---|---|---|---|
| 101 | madd tabii, noon/meem sakinah rules, ghunnah mushaddadah, qalqalah, lam of Allah, tanween | sukoon, shaddah (only ن/م and idgham targets), alif weight | letter identity, harakah quality and length, lam shamsiyyah/qamariyyah |
| Intermediate | all madd types (targets), idgham classes, core raa, hamzat al-wasl dropping, 4 saktat (with bugs) | leen, tafkheem levels, madd al-farq, madd tamkeen, haa al-sakt | madd relations (muttasil ≥ munfasil, leen ≤ ʿāriḍ), izhar of verb lam, istifal leakage |
| Advanced | khayshum (nasality), safir, tafashhi, takreer, maratib (descriptive), tanasub | hams/jahr, shiddah (buggy), isti'la, itbaq, istitaalah, most makharij | throat and interdental makharij, inhiraf, khafa, waqf kinds and signs, raum, ibtida' |
| Ijazah | – | tasāwī (notes), ghunnah maratib | wujuh consistency / talfeeq, qabih-stop grading |

The engine is strong on **timing and nasality** and nearly blind to **lahn jali**, Hafs-specific marks and
waqf/ibtida' quality (00 "Big picture").

---

## 9. In flight (2026-09-23)

Another agent is working on these now. Do not duplicate; check `git status` before editing.

- **P0 skip-bias fix (REVIEW status).** Working tree changes add `Status.REVIEW` (`app/models.py`),
  `REVIEW_SCORE = 0.5` and a REVIEW verdict for unreliable spans in `Calibration.judge` (only
  align_conf = 0, a letter outside the model vocabulary, stays SKIPPED), count REVIEW in the index, and add
  `ScoreSummary.coverage` (judged share) to the report (`app/scoring.py`, `app/pipeline.py`). Tests in
  `tests/test_calibration.py` and `tests/test_ahkaam.py` are being updated. This implements 06 §8 P0 and
  node `math:mnar`.
- **Substitution-aware GOP detector** `app/lahn/gop.py`: for each pronounced unit,
  LLR(u) = log P(X_w | window with the target) − max_v log P(X_w | window with variant v), using the full
  CTC forward sum over a local window (segmentation-free), with classical confusion sets (ض→د/ظ, ص→س,
  ط→ت, ح→ه, ع→ء, ق→ك, ث→س/ت, ذ→ز/د, ظ→ز/ذ, غ→خ, short/long vowels). Analysis scripts in
  `research_agency_lab/experiments/lahn_gop/` (`text_swap.py`, `analyze.py`, results
  `text_swap_6reciters.jsonl`). Graph: `algo:gop_subst_aware` (partial), `phen:lahn:jali`. Not yet wired
  into the pipeline or calibrated.
- **Data collection.** Modal `studio-all` (Husary + peers, 150 shards) was running at the last check; the
  imams collection is not launched (c3.md todos 3, 5, 10).

---

## 10. Frontier: what's next, ranked

Ranking = correctness first (bugs that penalise correct recitation or hide errors), then value × effort.
"Report" points to the section with the design and acceptance tests.

| Rank | Item | Why now | Report | Graph nodes | Effort |
|---|---|---|---|---|---|
| 1 | Finish P0 skip-bias (REVIEW + coverage + skip rate per reciter) | Without it every other improvement can be gamed by errors | 06 §8 P0, D2 | `math:mnar`, `code:app/calibration.py` | in flight |
| 2 | Substitution-aware / segmentation-free GOP for lahn jali, calibrated by conformal peer-LOO on text-swap and splice counterfactuals | The biggest curriculum gap (letter identity, harakah) | 01 §3.1, §3.1e; 06 §6.1, P3, P5 | `algo:gop_subst_aware`, `algo:gop_sf`, `algo:gop_segmentation_free`, `algo:text_swap_h1`, `algo:perturbation`, `phen:lahn:jali` | in flight + 3–4 d |
| 3 | Fix the parser Tier 0 bugs B1–B9 (standalone-ۜ sakt, `waqf_only` U+06E0, ayah-head sakt only in wasl, keep U+06E3/06EA–06EC, lexicons for ibtida' vowel and raa at waqf, 3:1–2) with regression tests | Deterministic false FAILs on correct recitation | 00 §2 Tier 0 (0.1–0.11); 04 §7 item 1 | `code:app/tajweed_rules/parser.py`, `phen:waqf:waqf_only_alif`, `phen:hafs:yabsut`, `phen:hafs:aajami`, `phen:raa:imala` | ½–2 d |
| 4 | Fix hams/jahr (landmark window, VOT for ت/ك, CPP for fricatives, never voicing-test ط ق ء) and shiddah (reuse `detect_release_burst`, dip-dB ordinal, drop duration ratios) | 186 + 485 spurious verdicts on experts | 02 §5 items 1–2 | `algo:vot_aspiration`, `algo:cpp_aperiodicity`, `algo:landmark_closure_burst`, `phen:sifah:hams`, `phen:sifah:shiddah` | 2 d |
| 5 | Collapse-mixture gate + tabīʿī-anchored beat + sakt in beats | Long madd verdicts are mostly measurement noise today | 03 §11 items 1–3 | `algo:collapse_gate`, `algo:two_ruler_beat`, `algo:sakt_in_beats` | 2–3 d |
| 6 | Collect full studio-all + imams, then `calibrate.jl`, `frontier.jl`, and commit `app/data/calibration.json` | Turns calibration on; enables LOO and weight search | c3.md todo 10; roadmap A1–A4 | `math:robust_median_mad`, `math:bures_wasserstein`, `math:conformal_loo` | compute + ½ d |
| 7 | Harmonise formant analysis (same ceiling/order in both stacks, Escudero ceiling per reciter, slot-swap QC; target r ≥ 0.85), then KARMA tracker | Everything formant-based (tafkheem, itbaq, ikhfa anticipation, makharij) depends on it | 07 §0.1–0.2, §A3 | `algo:lpc_harmonised`, `algo:karma_kalman` | ½ d + 3 d |
| 8 | Tasāwī v1 (per-type robust log-CV with measurement-error deconvolution, TOST vs Husary; ʿāriḍ 2/4/6 choice consistency; madd hierarchy check) | Core ijazah requirement; only notes today | 03 §5.2–5.4, §11 item 4; 00 row 104 | `phen:timing:tasawi`, `math:variance_components_tasawi`, `math:latent_class_mixture`, `math:order_constraint_test` | 2–3 d |
| 9 | Synthetic counterfactual validation set (WSOLA madd stretch, ghunnah truncation, burst removal, letter splices; LPC resynthesis for sifaat) | Gives detection power (H1); today only expert false-FAIL rates exist | 03 §8; 06 §6.6 P3; 02 §4 | `algo:perturbation`, `algo:lpc_counterfactual` | 2–3 d |
| 10 | Information health check + IRT (Rasch/LLTM) on existing rows | Separates ruler difficulty from reciter skill; flags degenerate keys | 06 P1–P2 | `math:mutual_information`, `math:irt` | 4 d |
| 11 | Waqf: keep pause marks, `WAQF_CHOICE` validator, boundary posterior, restart-capable alignment | Waqf/ibtida' quality is entirely missing | 04 §7 items 2–6 | `algo:pause_mark_parse`, `algo:boundary_classifier`, `algo:waqf_dp`, `algo:restart_align` | 1–2 wk |
| 12 | Second judges (muaalem-v3.2, espeak IPA) + makhraj table/graph + pairwise cue verifiers | Makharij verification; removes the digraph problem | 01 §5 P1–P3; 06 §6.9 | `algo:second_judge_ensemble`, `algo:pairwise_cue_llr`, `math:articulatory_ground_metric` | 1–2 wk |
| 13 | Shaddah engine, short-vowel ishbaʿ/ikhtilas, sukoon tahrik | Missing rule families | 03 §4.4–4.6; 00 Tier 1 #3, #6 | `phen:timing:shaddah`, `phen:timing:short_vowel`, `phen:timing:sukoon_tahreek` | 3–4 d |
| 14 | Tafkheem maratib (ordinal/isotonic) + emit sakin/kasra levels | Data already supports it (E10) | 02 §5 item 4; 00 row 48 | `phen:weight:maratib`, `math:isotonic_ordinal` | 1.5 d |
| 15 | Port the Frontier layer to Python (or export its model to JSON consumed by `calibration.py`) | Makes A1–A4 usable in production | roadmap build order 1 | `code:research_agency_lab/substrate_library/julia/src/Frontier.jl` | 2 d |
| 16 | Fingerprint: default style weight for identification (≈ 0–0.2), then C1 path signatures | Default setting loses 59 points of top-1 | README; roadmap C1 | `phen:fingerprint`, `math:path_signature` | ½ d + 2 d |
| 17 | Discovery on contiguous surahs (tempo ODE, z-scored STLSQ, wider λ grid) | Needs data, not new code | 07 §0.3–0.4, §B.7 | `math:tempo_relaxation_ode`, `algo:sindy_stlsq`, `algo:esindy` | data + 1 d |

Speculative or deferred (keep in the graph, do not build yet): persistent homology of F0 (roadmap C3,
05 "explicit non-recommendation" for letter paths), S4/Mamba prosody (06 P13), GNNs (05 G7).

---

## 11. Graph hygiene

Rebuilt 2026-09-23 with the updated `core.json`: **774 nodes, 1231 edges**, 0 stubs, 0 problems
(before: 738 / 1155). By type: phenomenon 267, algorithm 118, math 115, source 136, code 63, package 54,
dataset 13, model 8.

**Duplicate phenomenon ids (same concept, different spelling or fragment). Not merged; pick one
canonical id and alias the other in a later cleanup:**

| Canonical candidate | Duplicate(s) | Fragments |
|---|---|---|
| `phen:letter:ayn` | `phen:letter:ain` | 02 / 01 |
| `phen:letter:dad` | `phen:letter:daad` | 01 / 02 |
| `phen:letter:ghayn` | `phen:letter:ghain` | 02 / 01 |
| `phen:letter:nun` | `phen:letter:noon` | 01 / 02 |
| `phen:letter:sad` | `phen:letter:saad` | 01 / 02 |
| `phen:letter:taa_emph` (ط) | `phen:letter:taa_t` (ط) | 01 / 02 |
| `phen:letter:zaa_emph` (ظ) | `phen:letter:dhaa` (label ظ; the id reads like ذ) | 01 / 02 |
| `phen:sifah:istitaalah` | `phen:sifah:istitalah` | 00 / 02 |
| `phen:sifah:tafashhi` | `phen:sifah:tafashshi` | 00 / 02 |
| `phen:waqf:taa_marbuta` | `phen:waqf:ta_marbutah` | 04 / 00 |
| `phen:waqf:waqf_only_alif` | `phen:waqf:waqf_alifs` | 04 / 00 |
| `phen:waqf:sukoon` | `phen:waqf:sukun_mahd` | 04 / 00 |
| `phen:ibtida` | `phen:waqf:ibtida` | 04 / 00 |
| `phen:makharij` | `phen:makhraj` | 00,01,05,07,core / 06 |
| `phen:lahn:jali` + `phen:lahn:khafi` | `phen:lahn_jali_khafi` | 00,06 / 01 |
| `phen:timing:tasawi` | `phen:madd:tasawi`, `phen:madd_length` | 00,core / 05 / 06 |
| `phen:timing:madd_hierarchy` | `phen:madd:relations` | 03 / 00 |
| `phen:timing:shaddah` | `phen:letters:shaddah` | 03 / 00 |
| `phen:timing:short_vowel` | `phen:letters:haraka_length` | 03 / 00 |
| `phen:timing:sukoon_tahreek` | `phen:letters:sukoon` | 03 / 00 |
| `phen:timing:harakah` | `phen:harakah` | core / 07 |
| `phen:weight:maratib` | `phen:grad:tafkheem_maratib` | 00 / 02 |
| `phen:noon:ghunnah_maratib` | `phen:grad:ghunnah_maratib` | 00 / 02 |
| `phen:qalqalah:levels` | `phen:grad:qalqalah_levels` | 00 / 02 |
| `phen:sifah:takreer` | `phen:takreer` | 00,02,05 / 07 |
| `phen:sifah:jahr` | `phen:jahr` | 00,02 / 07 |
| `phen:wasl:saktat` | `phen:sakt` | 00 / 04,05 (both also overlap `phen:sakt_hamzat_wasl`, core) |
| `phen:wasl:dropped` | `phen:hamzat_wasl` | 00 / 04 |
| `phen:timing:wujuh_consistency` | `phen:tareeq:talfiq` | 00 / 05 |
| makhraj nodes from 00 (`phen:makhraj:qaf`, `:kaf`, `:lam`, `:noon`, `:raa`, `:faa`, `:dad`, `:asaliyyah`, `:lathawiyyah`, `:nitiyyah`, `:wasat_lisan`, `:shafatan`) | 01 equivalents (`phen:makhraj:lisan_aqsa_qaf`, `:lisan_aqsa_kaf`, `:lisan_hafah_lam`, `:lisan_taraf_nun`, `:lisan_taraf_raa`, `:shafah_faa`, `:lisan_hafah_dad`, `:lisan_asaliyya`, `:lisan_lithawiyya`, `:lisan_nitiyya`, `:lisan_wasat`, `:shafatan_both`) | 00 / 01 |

Near-duplicate algorithm/math ids: `algo:gop_sf` ≈ `algo:gop_segmentation_free`;
`algo:blahut_arimoto` ≈ `algo:blahut_arimoto_fano` ≈ `math:channel_capacity`; `algo:conformal_thresholds`
≈ `math:conformal_loo`; `math:hierarchical_bayes` ≈ `math:bayes_hierarchical`; `math:koopman_spectrum` ≈
`math:koopman_hankel_dmd` (+ `algo:dmd`); `math:hsmm` ≈ `math:hsmm_explicit_duration` ≈ `algo:hsmm`;
`algo:temperature_scaling` ≈ `math:temperature_scaling`; `math:spectral_clustering` ≈
`algo:spectral_clustering_confusion`; `algo:ksg` ≈ part of `math:mutual_information`.

**Other issues:**

1. **Status merge cannot downgrade.** `build_map.py` keeps the most advanced status across fragments, so
   `core.json` marking `code:app/sifaat/hams_jahr.py` and `sukoon_spectrum.py` as `partial` (buggy) is
   overridden by `implemented` from 00/02. The bug is visible only in the node `note`. Likewise
   `phen:lahn:jali` shows `partial` although nothing detects it. A fix would be a `status_override` field
   or core precedence in `build_map.py` (not changed here).
2. **Over-optimistic letter statuses.** 02 marks most `phen:letter:*` nodes `implemented` (the sifaat
   profile exists as data), while letter *identity* verification is missing (`phen:letters:identity`).
3. **Labels and notes are first-writer-wins** by fragment file order (`00…07` before `core`), so core's
   descriptive labels lose to bare file names on code nodes (e.g. label "hams_jahr.py").
4. **Non-path code ids in 04:** `code:parser`, `code:sakt_wasl`, `code:segmenter`, `code:fatigue_detector`,
   `code:hams_jahr`, `code:waqf_choice` should be repo paths (`code:app/tajweed_rules/parser.py`, …).
   Proposed code ids that do not exist yet: `code:app/makharij/`, `code:app/mdd/`,
   `code:app/information/channel.py`.
5. **Reports registered as code:** `code:research_agency_lab/experiments/deep_research/02_sifaat.md` and
   `03_timing.md` are typed `code` with status `implemented`.
6. **`fragments/03_timing.json` is a 25-node reconstruction**; the research agent's 126-node version was
   overwritten (c3.md item 8). The full report `03_timing.md` is intact; regenerate the fragment from it.
7. `phen:madd:hierarchy` (00, implemented: parser branch order = aqwa al-sababayn) and
   `phen:timing:madd_hierarchy` (03, missing: measured ordering across instances) are different concepts
   with similar names; keep both but relabel.

---

## 12. How to verify

```bash
cd ~/github-director/repos/muqri
J=~/julia-1.11.5/bin/julia; P=research_agency_lab/substrate_library

# Python unit tests (80 pass, 4 skip on 2026-09-23); avoid writing caches into the repo
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider -rs

# Opt-in acceptance (needs benchmarks/results/summary.json from summarize.py); currently fails the targets
QAARI_ACCEPTANCE=1 .venv/bin/python -m pytest tests/test_benchmarks.py -q

# The rules the parser derives for an ayah (check a bug, e.g. the false sakt in 2:245 or 41:44 madd lazim)
.venv/bin/python main.py rules --surah 2 --ayah 245
.venv/bin/python main.py rules --surah 41 --ayah 44

# Frontier math: identities, then Julia vs Octave on synthetic fixtures (expect 15/15 and WORST ~5e-15 AGREE)
$J --project=$P/julia $P/julia/test/test_frontier.jl
$J --project=$P/julia $P/julia/frontier_crosscheck.jl /tmp/frx && octave-cli --path $P/octave --eval "fr_crosscheck('/tmp/frx')"

# Frontier on real rows + Octave re-computation of every key (expect WORST ~1e-13 AGREE)
$J --project=$P/julia $P/julia/frontier.jl /tmp/frontier.json benchmarks/results/local/*.jsonl --export /tmp/frexp
octave-cli --path $P/octave --eval "fr_real_crosscheck('/tmp/frexp')"

# Calibration and discovery from rows (writes JSON; point OUT at a scratch path, not app/data, unless intended)
$J --project=$P/julia $P/julia/calibrate.jl /tmp/cal.json benchmarks/results/local/husary_strategic_raw.jsonl
$J --project=$P/julia $P/julia/discover.jl  /tmp/disc.json benchmarks/results/local/husary_strategic_raw.jsonl

# Octave DSP cross-check vs Python (core_ms r≈0.9975, formants r≈0.34). Needs EveryAyah audio in the cache;
# --calibration-out defaults to app/data/calibration.json, so redirect it
.venv/bin/python research_agency_lab/compute_bridge/validate_loop.py \
    --runs 'benchmarks/results/local/husary_strategic_raw.jsonl' --sample 59 --calibration-out /tmp/cal.json

# Re-derive the raw verdict tables in §3.10–3.11 and §6 (E22, E24)
.venv/bin/python - <<'EOF'
import json, glob, collections
rows = [json.loads(l) for p in glob.glob('benchmarks/results/local/*.jsonl') for l in open(p) if l.strip()]
st = collections.defaultdict(collections.Counter)
for r in rows:
    if r['mode'] == 'studio':
        for d in r['diagnostics']:
            st[d['key']][d['status']] += 1
for k in sorted(st): print(k, dict(st[k]))
EOF

# Knowledge graph
.venv/bin/python research_agency_lab/knowledge_map/build_map.py            # expect 774 nodes, 1231 edges
.venv/bin/python research_agency_lab/knowledge_map/build_map.py q hams     # neighbourhood
.venv/bin/python research_agency_lab/knowledge_map/build_map.py gaps       # phenomena with no implemented model
```

Do not launch Modal or Kaggle jobs from a documentation task; the collection commands are in `~/c3.md`.
