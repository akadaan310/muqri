# Full tajweed coverage: from parser rules to a testable verdict

*Written 2026-09-23. Sources: `datastore/rule_catalogue.py` (the machine-readable version of this
document), DuckDB tables `rule_catalogue` / `engine_diag`, knowledge-map fragment
`08_tajweed_rules.json`, and the measurements in `textswap_T10.json` / `sifat_T10.json`.*

## The claim this document supports

The engine can locate **and name** a tajweed mistake across every rule family, given the reference
text — which the product always has, because the app knows the verse before the recording is
submitted. Scoring is therefore never open-vocabulary recognition; it is always a **likelihood-ratio
test against a known reference**, and that is what makes high accuracy reachable.

## Three mechanisms cover all 40 rules

The parser (`app/tajweed_rules/`, 41 `RuleType`s) already tells us *where* every rule applies —
1,160,912 located instances over 38,274 ayah rows in `engine_diag`. What was missing was a principled
way to judge each one. Every rule falls into exactly one of three testable mechanisms:

| mechanism | what the rule changes | how it is tested | measured |
|---|---|---|---|
| **segmental** | which phonemes are recited (idgham merges, iqlab turns ن→م, izhar keeps ن clear) | edit the phoneme string to the wrong realisation, compare CTC likelihoods | **99.7 % detected / 97.1 % named** on letter swaps |
| **durational** | a *length* (madd 2/4/6, ghunnah hold, sakt) | same test — because length **is** phoneme repetition | inherits the segmental machinery |
| **attribute** | a quality of a letter (hams/jahr, shidda, itbaq, …) | muaalem's dedicated CTC head vs the expected class | **~1 % anchor false-alarm**, vs 10–19 % for the DSP detectors it replaces |

### The key structural fact

`quran_phonetizer` encodes **duration as character repetition** in the phoneme string:

```
ۥ×4   4-count madd (muttasil / munfasil)      ا×2   madd tabii, 2 counts
ا×4   4 counts                                 ں×3   ikhfa noon, nasalised and held
ل×2   idgham / shadda                          ن×4, م×4   ghunnah held
```

So a madd recited at 2 counts instead of 4, or a ghunnah dropped, is **not** a separate duration
model — it is a phoneme-string edit, scored by the same CTC likelihood ratio that already achieves
99.7 % on letters. This is what makes uniform coverage of all rule families tractable.

## What the current engine actually does (measured, not assumed)

`legacy_pass_rate` is the existing engine's PASS fraction over `engine_diag` — computed on **master
reciters**. A rate near 0.05 does not mean the masters are wrong; it means the detector fails
everyone and is broken.

### BROKEN — 11 rules, 131,047 instances (≤10 % pass on masters)

`ikhfa` 0.048 · `ghunnah` 0.045 · `idgham_ghunnah` 0.068 · `madd_munfasil` 0.058 ·
`madd_muttasil` 0.052 · `madd_iwad` 0.019 · `idgham_shafawi` 0.034 · `ikhfa_shafawi` 0.058 ·
`madd_silah_kubra` 0.079 · `madd_lazim` 0.043 · `sakt` 0.048

Every one is **durational or nasalisation-related** — exactly the class the repetition encoding and
the `ghonna` head now make testable. This is the single highest-value work list in the project.

### SUSPECT — 10 rules, 248,350 instances (10–50 %)

`rakhawah` 0.425 · `safir` 0.499 · `hams` 0.392 · `tarqeeq` 0.416 · `madd_silah_sughra` 0.493 ·
`madd_badal` 0.182 · `shiddah` 0.301 · `idgham_no_ghunnah` 0.290 · `iqlab` 0.188 · `istitaalah` 0.243

Seven of these are **attribute** rules with a muaalem head already available, so they are covered by
`SifatGop.jl` today.

### PLAUSIBLE — 17 rules, 781,515 instances (>50 %)

`hamzat_wasl` 0.998 · `idgham_mutaqaribayn` 0.981 · `idgham_mithlayn` 0.941 · `qalqalah` 0.817 ·
`idgham_mutajanisayn` 0.811 · `izhar_halqi` 0.740 · `takreer` 0.725 · `tafkheem` 0.715 ·
`izhar_shafawi` 0.710 · `itbaq` 0.706 · `tawassut` 0.640 · `tafashhi` 0.623 · `madd_leen` 0.569 ·
`jahr` 0.566 · `madd_arid_lissukun` 0.559 · `madd_tabii` 0.522 · `jawaz_wajhayn` 1.000

## Where the data lives

| artefact | what it holds |
|---|---|
| `rule_catalogue` (DuckDB) | 40 rules × family, mechanism, phoneme signature, counterfactual, muaalem head, instance count, legacy pass rate, status |
| `engine_diag` (DuckDB) | 1,160,912 located rule instances with status, score, word/letter and ms span |
| `engine_ayah` (DuckDB) | 38,274 per-ayah perfection / sifaat / environment rows |
| `posterior_file`, `unit_gop`, `deviation` | per-clip posteriors and GOP output |
| `fragments/08_tajweed_rules.json` | the same mapping as graph nodes/edges (rule → family, → mechanism, → muaalem head) |
| `<dump>/sifat.jsonl` | expected sifat class per reference phoneme (`sifat_ref.py`) |

The catalogue and the graph fragment are generated from **one** source (`CATALOGUE` in
`datastore/rule_catalogue.py`), so the store and the knowledge graph cannot drift apart.

## Verified measurements behind this

* **Lahn jali** (`textswap_T10.json`, 414 clips): 6,898 consonant swaps → 99.7 % detected, 97.1 %
  named; 26,378 harakah swaps → 99.8 % detected, 98.9 % named. Anchor false-alarm 0.63–0.94 %
  against a 1 % conformal target.
* **Sifat attributes** (`sifat_T10.json`, 414 clips, 6.8 s): all ten heads land at 0.9 % anchor
  false-alarm. Replaces DSP detectors measured at takreer 19 %, jahr 14 %, shiddah 10 % false FAIL.
* **Data-quality gate**: free decode vs reference, normalised edit distance per reciter. Three
  reciters score 0.000 (perfect transcription); `Abdullaah_3awwaad_Al-Juhaynee_128kbps` scores
  **0.849** — its EveryAyah files do not match their claimed verses. It must be excluded before T300
  or it poisons calibration and the score table.

### Rule-layer counterfactuals (`ruleswap_T10.json`, 414 clips, 8,605 edits, 4.6 s)

Built as `ruleswap_gop.jl`: edit the *length or symbol* of a rule site in the reference and ask whether
the true text still explains the audio better. `anchor win` = the engine prefers the correct
realisation (so it would flag the wrong one); `anchor flag` = it prefers the edited version, i.e. it
believes the anchor actually deviated.

| family | what it simulates | edits | anchor win | median margin | anchor flag |
|---|---|---|---|---|---|
| `idgham_undo` | failing to merge / no shadda | 1,283 | **100 %** | 12.19 | 0.0 % |
| `madd_long` | over-stretching a madd | 3,062 | **99.1 %** | 23.83 | 0.9 % |
| `ghunnah_drop` | nasalisation dropped | 866 | **96.8 %** | 27.09 | 3.2 % |
| `madd_short` | 4 counts recited as 2 | 3,063 | **94.6 %** | 11.90 | 5.4 % |
| `ikhfa_izhar` | ikhfa read as izhar | 331 | **91.7 %** | 45.58 | 8.3 % |

Non-anchor win rates track the anchors within ~2 points (87.5–100 %), so this is not an anchor-only
effect. These are precisely the rules the legacy engine passed at 2–8 %. `ikhfa_izhar` has only 331
sites at T10 (~7.7 per reciter) — the thinnest family, and the clearest case for T300.

## T300 results (11,996 clips, 41 reciters — the $3.84 spend)

### Rule layer — 8 families, 301,150 counterfactual edits

| family | edits | anchor win | others win |
|---|---|---|---|
| `qalqala_drop` (no release on a sakin ق ط ب ج د) | 8,806 | **99.84 %** | 99.31 % |
| `idgham_undo` | 40,833 | **99.16 %** | 99.18 % |
| `madd_long` | 105,390 | **98.99 %** | 98.58 % |
| `ghunnah_drop` | 24,889 | **97.45 %** | 95.89 % |
| `iqlab_undo` (ن never became م before ب) | 2,677 | **97.33 %** | 97.06 % |
| `iqlab_no_ghunnah` (converted but not held) | 2,677 | **97.33 %** | 95.59 % |
| `ikhfa_izhar` | 10,479 | **95.62 %** | 94.34 % |
| `madd_short` | 105,399 | **93.71 %** | 92.42 % |

`iqlab` and `qalqala` are *exact*, not inferred from context: the phonetizer gives them their own
symbols (`۾` for the iqlab meem, `ڇ` for the qalqala release), so the counterfactual targets the rule
itself. At T10 iqlab read 66.7 % on ~9 anchor instances — a small-sample artefact, not a result.

### Sifat — the rare classes are now statistically valid

Every anchor flag rate lands on **0.0100** against α = 1 %. The classes that were too thin at T10 now
clear the n ≥ 99 that conformal exactness requires: tikraar مكرر 117→**3,412**, safeer صفير 99→**2,694**,
itbaq مطبق 102→**2,108**, qalqala مقلقل 60→**1,253**, tafashie متفشي **685**, tafkheem adna **386**.

### Tasāwī — consistency, the mastery question (`tasawi_run.jl`)

"Was this ghunnah correct?" is not what separates a master. A master holds *every* 4-count madd for the
same 4 counts. That is a **variance** statistic, so no per-instance threshold can express it. Durations
come from CTC **onset-to-onset** (the Viterbi span `l−f` is an alignment artefact — peaky CTC gives a
symbol one frame and dumps the rest into blank; using it read madd_2 = 3.0 counts with a CV of exactly
0), divided by the clip's own median short vowel so tempo cancels.

| held class | anchor counts | anchor CV | others CV |
|---|---|---|---|
| madd_2 | 2.25 | **0.135** | 0.185 |
| madd_4 | 6.20 | **0.392** | 0.475 |
| madd_6 | 12.0 | **0.209** | 0.239 |
| ghunnah م | 3.20 | **0.174** | 0.212 |
| ghunnah ن | 3.29 | **0.199** | 0.233 |
| ikhfa noon | 3.20 | **0.135** | 0.227 |

Anchors are more consistent in **all six** classes. Caveat: the absolute count scale is not calibrated —
measured ratios are 2.25 : 6.20 : 12.0 where the notation implies 2 : 4 : 6. The **CV is trustworthy;
the median-counts figure is not yet fit to drive a verdict.**

### Sukoon timing — the master's temporal signature (`sukoon_timing.jl`)

A sakin letter's duration depends on its sifah: **rikhw** (flowing) held longest, **bayniyya** (ل ن م ر ع)
intermediate, **shadeed** (blocked) shortest. Masters realise the ordering; fast reciters compress it.
Each reciter is measured in **their own counts**, with their own vowelled-letter duration as the unit
(`haraka_s` spans 0.16 s for fast imams to 0.40 s for Husary Mujawwad).

Top of the separation ranking is mujawwad and teaching styles — Minshawy Mujawwad 1.175, Abdul Basit
Mujawwad 1.133, Karim Mansoori 1.072, Ayman Sowaid 1.000, Husary Mujawwad 0.943; the bottom is fast
prayer imams — Shuraym 0.250, Hudhaify/Ad-Dussary/Fares Abbad 0.400. Anchor median **0.861 vs 0.617**.

Caveat: values quantise onto {1.0, 0.8, 0.667, 0.4, …} because the model's frames are **40 ms** and a
sakin consonant is only 1–3 frames. The *separation* survives (a difference of medians over thousands
of instances); the strict three-way ordering does not — it holds for 2/3 anchors and 14/38 others
because rikhw and between frequently tie. Finer resolution needs a higher-rate alignment, not more data.

### Ranking reciters — measured, and why no single ordering is trustworthy yet

Two formulations were built and neither is a mastery ranking:

* **Flag-rate score** (`datastore/reciter_table.py`) cannot put the anchors on top *by construction*:
  thresholds are conformal at α on the anchors, so an anchor is flagged at exactly α and anyone more
  conservative outranks it. Husary 128k came 3rd, Muallim 12th.
* **Distributional distance** (`reciter_distance.jl`, Frontier A1/A2: robust OGK whitening over 45
  shared features, Mahalanobis to the anchor centroid) puts the anchors 1-2-3 in sample — but that is
  circular, the centroid *is* their mean. **Leave-one-anchor-out** gives the honest number: ranks
  **4, 5 and 11** of 41 (chance ≈ 21). Real signal, not a podium.

The deeper issue: the distance metric ranks *stylistic similarity to Husary*, not mastery. It places
**Minshawy Mujawwad 40th and Ayman Sowaid 41st** — the two most distinctively mujawwad/teaching
reciters, and Minshawy Mujawwad has the **best sukoon separation of all 41**. A different but equally
valid school scores badly. Per-capability verdicts are solid; collapsing them into one scalar is not.
A style-conditioned reference (one centroid per school) is the way forward.

## Per-letter completeness — the record, not just the errors

The app must be able to open **any single letter** and see its entire state, every characteristic
applied and every one correctly absent. `letter_report.jl` emits that record; `datastore/letter_reference.py`
supplies what no model can judge.

| source | what it gives | example (ص) |
|---|---|---|
| **judged** — the ten muaalem heads | attributes that vary by realisation, each with an LLR | shidda=[رخو] safeer=[صفير] tafkheem=[مفخم] hams=[همس] itbaq=[مطبق] |
| **inherent** — static table | isti'lā'/istifāl, idhlāq/iṣmāt, inḥirāf, līn, makhraj | isti'la, ismat, "tip of the tongue between the incisors" |

The classical count is five opposing pairs (hams/jahr · shidda|tawassuṭ|rakhāwa · isti'lā'/istifāl ·
iṭbāq/infitāḥ · idhlāq/iṣmāt) plus the non-opposing sifāt when present — so ر = 5 + takrīr + inḥirāf =
**seven**. Two of the five pairs have no acoustic head and come from the table. Median **7.36
characteristics per unit**.

### A bug this exposed: sifāt were being scored on vowels

`sifat_llr` had no vowel filter, and **50.2 % of units are consonants — 40.2 % are short vowels and
9.6 % madd**. A sifah belongs to a *consonant*; during a vowel the folds always vibrate, so "expected
hams" over a vowel produced a systematic false violation. Half of every sifat measurement was on units
that carry no sifah.

Fixing it removed exactly **50 %** of scored instances (776,610 → 391,580). Anchor flag rates were
unchanged (0.0100 → 0.0100, as conformal calibration guarantees), but **discrimination improved
sharply** because the vowel noise had been diluting the signal:

| class | others before | others after |
|---|---|---|
| shidda `[رخو]` | 0.0100 | **0.0238** |
| hams_or_jahr `[جهر]` | 0.0097 | **0.0160** |
| safeer `[لا صفير]` | 0.0117 | **0.0190** |

`qalqla.[مقلقل]` is unchanged at 1,253 (+0 %) — qalqala only ever occurs on consonants, confirming the
filter removes only what it should.

## ر and ل: the conditional weight rules

Every other sifah is a fixed property of the letter, so its *expected* value is trivially right.
Tafkhīm/tarqīq of ر and of the lām of the Divine Name is not — it depends on the vowel on the letter,
the vowel **before** it, whether that kasrah is original or incidental, and whether a ḥarf isti'lā'
follows in the same word. If the expected label is wrong the verdict is wrong however good the model is.

`quran_transcript` applies all of it correctly (verified, not assumed):

| case | rule | result |
|---|---|---|
| فِرْعَوْنَ | ر sākin, original kasrah, no isti'lā' after | **light** ✓ |
| مِرْصَادًا | same but ص (isti'lā') follows | **heavy** ✓ |
| قِرْطَاسٍ | same but ط follows | **heavy** ✓ |
| خَيْرٍ | ر after a sākin yā' (leen) | **light** ✓ |
| قَالَ ٱللَّهُ | jalālah after fatḥah | **heavy** ✓ |
| بِسْمِ ٱللَّهِ | jalālah after kasrah | **light** ✓ |

فِرْعَوْن and مِرْصَاد differ *only* in the letter after the sākin ر, so that pair is the sharpest
possible check. The 11 conditions are stored as data in `letter_reference.weight_rules` and each is
pinned by `tests/test_raa_lam_weight.py` (15 tests). Corpus over 301 ayahs: **ر 79.3 % heavy /
20.7 % light**, **ل 13.7 % / 86.3 %** — as the rules predict.

**Scope trap, encoded as a test:** phonetising `قَالَ ٱللَّهُ` as an isolated fragment returns a
*light* lām, because the jalālah rule depends on the **preceding word's** vowel. Weight tests for the
Divine Name must use full ayahs.

## Coverage validation — is every expected unit actually recorded?

`letter_report.jl` in coverage mode over all 11,996 T300 clips, grouped by ladder tier and tempo:

| tier | n | haraka_s range | id_vowel | id_madd | id_consonant | attributes |
|---|---|---|---|---|---|---|
| anchor | 3 | 0.20–0.32 | **1.0000** | 1.0000 | 1.0000 | 0.9980 |
| studio | 10 | 0.12–0.28 | **1.0000** | 1.0000 | 1.0000 | 0.9978 |
| imam | 7 | 0.08–0.16 | **1.0000** | 1.0000 | 1.0000 | 0.9974 |
| fast | 21 | 0.12–0.28 | **1.0000** | 1.0000 | 1.0000 | 0.9977 |

**Vowel and madd coverage is 1.0000 at every level and every speed**, including the fastest imams.
No vowel the phonetizer expects goes unrecorded. The only place identity is ever unconfirmed is
consonants, lowest for Minshawy Mujawwad (0.9767) — the most distinctive style, consistent with the
distance metric also ranking him last.

## Lahn jali at T300 (the full text-swap pass)

| | swaps | detected | named | anchor clean flag |
|---|---|---|---|---|
| consonants | 216,577 | **99.91 %** | **99.06 %** | 0.73–1.30 % |
| harakāt | 907,520 | **99.35 %** | **98.73 %** | 0.86–1.14 % |

1,124,097 counterfactuals. Named recall *improved* at 30× scale (consonants 97.1 → 99.1 %).

## Open work, in priority order

1. ~~Rule-level counterfactuals for the broken rules~~ — **done** (table above). Remaining: `iqlab`
   (ن→م before ب), `sakt`, `madd_iwad` and the meem-sakinah pair need their own edit generators.
2. **Rare-class thresholds** — istitala `مستطيل` n=21, qalqala `مقلقل` n=60, safeer `صفير` n=99,
   itbaq `مطبق` n=102, tikraar `مكرر` n=117. Conformal exactness at α=1 % needs n≥99, so several are
   not yet guaranteed. **This is what the T300 spend buys** (12,900 clips ≈ $4.50).
3. **Retire the DSP sifaat detectors** in `app/sifaat/` in favour of the heads.
4. **Waqf/ibtida** — located by the parser but not yet in the counterfactual framework.
