# 08 — Recitation datasets that contain mistakes (for scoring the error detector)

*Surveyed 2026-09-24. Everything below was checked at the source (dataset card, file listing or
paper) unless marked "reported". Licensing is noted but not treated as a filter: the organisation
will clear licences, so every usable set is listed.*

## Why this matters

The engine is validated on masters: 41 professional reciters (T300) tell us its **false-alarm** rate
(Baqarah 2:1–2: 25/41 masters now score 100 %). What we cannot yet measure is its **miss rate** —
how many real mistakes it catches. A certified reciter's deliberately careless reading of 2:1–2
scored 83 %, and the engine caught only two of its errors with confidence (tafkhim of ر in رَيْبَ,
a stretched madd in ذَٰلِكَ). A miss rate needs recordings where the mistakes are **known and
located**. That is what this survey looks for.

What makes a dataset useful here, in order:

1. **Located errors** (which word/letter/rule, what was produced instead) — not just a clip label.
2. **Deliberate or controlled errors** — the error type is known by construction.
3. **Tajweed errors**, not only letter/word errors — madd lengths, ghunnah, qalqalah, tafkhim.
4. Hafs 'an 'Asim, and the verse is known.

## Datasets with mistakes

| dataset | size | how the errors arose | what is labelled | located? | access / licence |
|---|---|---|---|---|---|
| **IqraEval `Iqra_Extra_IS26`** | 1,333 utterances | **deliberate**: native speakers instructed to produce specified phoneme substitutions (reported, IQRA 2026 paper) | reference vs. produced phoneme string | yes, phoneme level | HF, **gated** (request access) |
| **IqraEval `Iqra_TTS`** | 55,369 clips, ~52 h | **synthetic**: TTS with injected phoneme substitutions | `phoneme_ref`, `phoneme_mis`, `sentence_ref`, `sentence_aug`, `label` | yes, phoneme level | HF, open |
| **IqraEval `Iqra_train`** | 71,391 + 2,588 dev, ~79 h | mostly correct readings, with an augmented phoneme track | `phoneme_ref`, `phoneme_aug` | phoneme level | HF, open |
| **IqraEval `QuranMB.v2`** (+ `IqraEval_Test_GT`, gated) | 1,642 test clips | real learner speech, annotated | ground truth held in the gated GT set | phoneme level | HF; GT gated |
| **obadx `qdat_bench`** | 159 clips, one verse (المائدة 109) | real, varied reciters | **per-rule lengths in counts** (qalo alif/waw, laa alif, munfasil, 'allam alif, 'arid — 0–8), noon mushaddadah/mukhfah ghunnah grade, qalqalah present, full 10-sifat transcript | **yes, rule level with counts** | HF, open |
| **obadx `qdat`** (QDAT) | 1,505 clips, 160 reciters × ~10 takes, one verse | real (label source not stated in paper) | binary per rule: munfasil madd, noon mushaddadah, ikhfa'; ~19–47 % incorrect per rule | rule level, binary | HF, MIT |
| **sobolev210 `quran-recitation-errors`** (+ `-test`) | ~hundreds of ayah chunks | **real learners**, reviewed by named reviewers | per word: `Wording`, `Tajweed`, `Letters`, `Tashkeel`; Hafs and Qālūn | **word level** | HF, MIT |
| **MuazAhmad7 `Surah_Ikhlas-Labeled_Dataset`** | 1,506 clips (851 error / 655 correct), surah 112 | real recitations | error type (qalqalah, letter, madd …), **error location**, explanation (Arabic) | word level + type | HF, CC BY 4.0 |
| **RetaSy `quranic_audio_dataset`** | 6,828 clips, 1,287 non-Arabic speakers, 11+ countries | real learners | `correct` / `in_correct` / not-Quran / wrong ayah / multiple / incomplete; 3 annotators each with quality scores; golden subset | clip level only | HF |
| "What Counts as a Mistake?" (arXiv 2609.12085) | 100 recordings, 10 learners, 162 located events | real memorisation | substitution / omission / insertion (+ corrected, benign) on the transcript | word level | dev release only (127 cases); gold kept private |
| Tarteel mistake data | ~9 k h private (reported) | real app users | word-level wrong / skipped / order | word | **private** — partnership only |
| "Mispronunciation Detection of Basic Quranic Recitation Rules" (arXiv 2305.06429) | reported | reported | basic rules | — | check authors |
| arXiv 2503.23470 (DNN tajweed evaluation) | reported 3,071 files, 8 rules, correct/incorrect | reported | one rule per file | rule level | check authors |

**Adjacent (MSA / non-Quranic) mispronunciation corpora**, useful for letter-level (makharij)
errors but not tajweed: Arabic-CAPT (62 non-native speakers, 2.36 h), AraVoiceL2 (11 speakers,
5.5 h), ASMDD (100 Egyptian children, top-100 words).

## What is still missing everywhere

No dataset found has **deliberate tajweed errors across rules with located counts at scale**. The
closest are `qdat_bench` (counts, but one verse and 159 clips) and `Iqra_Extra_IS26` (deliberate,
but phoneme substitutions, not tajweed durations). Nothing covers waqf/ibtida errors.

## Recommendation

1. **Now, open data:** score the engine on `qdat_bench` (per-rule counts give a direct miss/false
   alarm matrix for madd, ghunnah, qalqalah), `sobolev210` (word-level Tajweed/Letters tags, real
   learners), `Surah_Ikhlas` (typed, located errors), and `Iqra_TTS` (letter substitutions at scale).
2. **Request access:** `Iqra_Extra_IS26` and `IqraEval_Test_GT` (gated, auto-approval).
3. **Commission our own controlled set** — the only way to get what nobody has. A protocol:
   certified reciters (like the user who supplied 2:1–2) record each passage **once correctly and
   then once per scripted error** — madd tabii shortened to 1, lazim cut to 2–4, ghunnah dropped,
   qalqalah omitted, tafkhim inverted, idgham read as izhar, waqf mid-word, stop on a word followed
   by the wrong ending, restart without hamzat al-wasl. The script is the label, so every error is
   located and typed by construction. 20 reciters × 30 passages × ~12 scripted errors ≈ 7,000 clips.
4. **Synthesise the rest from our own masters:** the engine already edits time and phoneme strings;
   time-stretching or compressing a master's madd segment by a known factor produces a labelled
   duration error on real voices, with the master's untouched clip as the control.

## Sources

- IqraEval shared task: https://aclanthology.org/2025.arabicnlp-sharedtasks.61/ ; benchmark paper
  https://arxiv.org/abs/2506.07722 ; IQRA 2026 challenge https://arxiv.org/html/2603.29087
- HF datasets: IqraEval/Iqra_train, IqraEval/Iqra_TTS, IqraEval/QuranMB.v2, IqraEval/Iqra_Extra_IS26,
  IqraEval/IqraEval_Test_GT, obadx/qdat, obadx/qdat_bench, sobolev210/quran-recitation-errors,
  MuazAhmad7/Surah_Ikhlas-Labeled_Dataset, RetaSy/quranic_audio_dataset
- QDAT (Osman et al. 2021):
  https://www.researchgate.net/publication/350785609_QDAT_A_data_set_for_Reciting_the_Quran
- Quranic Audio Dataset (non-Arabic speakers): https://arxiv.org/abs/2405.02675
- Annotating recitation events: https://arxiv.org/abs/2609.12085
- muaalem / Quran Phonetic Script: https://arxiv.org/abs/2509.00094
- Tarteel mistake detection: https://tarteel.ai/blog/introducing-mistake-detection/
- Basic rules MDD: https://arxiv.org/abs/2305.06429 ; DNN tajweed: https://arxiv.org/html/2503.23470
- ASMDD: https://arxiv.org/pdf/2111.01136
- Quran-Lab quran-g2p (phonemizer with mid-ayah waqf variants; No-Profit licence):
  https://github.com/Quran-Lab/quran-g2p , https://huggingface.co/datasets/Quran-Lab/quran-tajweed-phonetics
