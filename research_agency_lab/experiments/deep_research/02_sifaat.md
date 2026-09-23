# Deep Research 02: The 17 Sifaat (+ Ghunnah), from 101 to Ijazah Level

Scope: all sifaat of the letters in Hafs 'an 'Asim (Jazari school: 5 opposite pairs + 7 without opposites,
plus ghunnah and khafa as al-Mursafi counts them). Covers letter strength, the maratib of tafkheem, and the
qalqalah levels. Builds on `frontier_math_roadmap.md` (A1–A6 calibration, B, C). Date: 2026-09-23.

The report mixes three kinds of claim, and each one says which it is:
**[DATA]** = computed from the repo's own reference-reciter runs (`benchmarks/results/local/*.jsonl`: Husary,
Tablawi, Ayyoub, Budair, Matroud, Muaiqly; 502 ayah rows). **[LIT]** = a cited source. **[PROPOSAL]** = a design
that hasn't been tested. Scratch scripts (in the session scratchpad, not the repo) recompute every [DATA] number.

---

## 0. Top findings

1. **The hams validator is mostly wrong on expert reciters. [DATA]** Hams letters get 186 FAIL against 90 PASS.
   Every single sakin ت/ك/ح/ف (the letters that aren't sibilants) fails, and the median voicing fraction is 1.0. There are
   two causes. (a) Praat's F0 track smears voicing from the neighbouring vowels across a short consonant, and
   `consonant_window()` (the central 70% of a CTC span) contains the next vowel. (b) Conceptually, **hams/jahr in
   tajweed is about airflow, not vocal-fold vibration.** Heselwood & Maghrabi (2015, doi:10.1093/jss/fgu035) show
   that majhur consonants have *low airflow at release*. That's why ط, ق and hamza are majhur even though they're
   voiceless today. For the hams stops ت/ك the correct correlate is **post-release aspiration (VOT)**: Arabic
   [t] has VOT ≈ 51 ms and emphatic [ṭ] ≈ 14 ms (Kulikov 2021/22, doi:10.1177/0023830920986821).
   Fix: score hams by *aspiration noise after the release / breath noise in the frication*, and don't score it by F0 voicing.
2. **The shiddah detector never sees the closure. [DATA]** `occlusion_ms` has median 0 and 94/100 are WARNING. The
   qalqalah engine measures a median closure of 77 ms on the same reciters. `sukoon_spectrum.py` should reuse
   `qalqalah_engine.detect_release_burst` (an 8 dB drop plus a burst landmark) and stop relying on a −25 dB silence run. The
   duration ratios 1.0/1.5/2.2 are falsified. Measured medians are shiddah 0.54–0.61, tawassut 0.91–1.0 and rakhawah 0.76,
   against a model value of 2.2. That produces 485 spurious rakhawah WARNINGs, so drop the ratio test or calibrate it.
3. **The maratib of tafkheem are measurable today. [DATA]** The existing heaviness index H orders the text-derived levels
   monotonically. Fatha+alif has median 0.595 (n=140), fatha 0.481 (n=381) and damma 0.201 (n=70), with Spearman ρ=−0.42,
   p=6.5e−27, n=600. Husary gives 0.77 > 0.63 > 0.20. The ordering is monotone in 5 of 6 reciters; the exception is
   Tablawi, where levels 1 and 2 are inverted. This justifies **ordinal / isotonic modelling of tafkheem gradation** (§3.5).
4. **Letter strength follows from counting, and the counting reproduces the classical table exactly.** Use al-Mursafi's rule:
   there are 11 strong sifaat (including ghunnah) and 6 weak ones (including leen and khafa). Idhlaq, ismat and tawassut are
   *neutral* and not counted. The formula `strong > weak → qawiyy` reproduces all 29 letters of *Hidayat al-Qari*'s
   5-class table [DATA/LIT].
5. **Sifaat alone don't identify a letter.** Six pairs or triples of letters share an identical sifaat profile:
   ت=ك, ث=ح=ه, ج=د, م=ن, و=ي, ذ=alif. The context has only 22 distinct profiles and 33 formal concepts, and
   H(letter | all sifaat) = 0.53 bits of H(letter) = 4.21 bits. So a joint profile score has to be
   **makhraj-conditioned**: sifaat separate letters that share a makhraj, which is the classical doctrine.
6. **Kubra isn't acoustically stronger than sughra. [DATA]** Energy rise is 13.0 dB for kubra against 13.7 dB for
   sughra (n=98/330). Akbar is distinct: 18 dB rise and a 174 ms hold. Kubra differs from sughra by *prolongation /
   waqf context*, not by burst energy. A level-specific band is needed, and the code's `RISE_STRONG_DB` assumption
   (kubra > sughra) doesn't hold for these experts.
7. **Nothing measures idhlaq/ismat, inhiraf, leen (as a sifah) or khafa, and nothing jointly.**
   Idhlaq/ismat is a phonotactic/lexical class (the "fluent" letters فرمنلب), not a reciter-controllable gesture.
   Treat it as neutral and derived, and don't score it (§2).
8. The **Quran-specific sifaat CTC model already exists**: *muaalem* (Abdelfattah, Khalil, Abbas, arXiv:2509.00094;
   HF `obadx/muaalem-model-v3_2`, Wav2Vec2-BERT with 10 sifaat CTC levels, PER 0.08–0.17%). It's a strong
   second-opinion teacher, with a caveat. Sifaat are a deterministic function of the letter, so a low PER may just mean
   the model decodes the canonical label. It has to be tested on *deviant* realizations before it's trusted (§3.7).

---

## 1. Canonical letter × sifah table

Letter sets follow Ibn al-Jazari's Muqaddimah as used in `app/tajweed_rules/parser.py` (lines 37–50):
hams فحثهشخصسكت; shiddah أجدقطبكت; tawassut لنعمر; rakhawah = the rest; isti'la خصضغطقظ; itbaq صضطظ;
idhlaq فرمنلب; safir صسز; qalqalah قطبجد; leen و/ي sakinah after fatha; inhiraf ل ر; takreer ر;
tafashhi ش (a minority adds ف, ض, ث, ر); istitalah ض; ghunnah ن م (with tanween → ن); khafa ه and the madd letters.

Bit order: jahr, shiddah, tawassut, rakhawah, isti'la, itbaq, idhlaq, safir, qalqalah, leen, inhiraf, takreer,
tafashhi, istitalah, ghunnah, khafa. The "S/W" columns count strong/weak sifaat (al-Mursafi rule). "Class" is the
resulting strength class.

| Letter | Profile bits | S | W | S−W | Class | Distinctive sifaat vs same-makhraj siblings |
|---|---|---|---|---|---|---|
| ء | 1100000000000000 | 2 | 2 | 0 | mutawassit | shiddah+jahr (vs ه: hams, rakhawah, khafa) |
| ب | 1100001010000000 | 3 | 2 | +1 | qawiyy | shiddah, qalqalah (vs م ghunnah) |
| ت | 0100000000000000 | 1 | 3 | −2 | da'if | hams (vs د), istifal/infitah (vs ط) |
| ث | 0001000000000000 | 0 | 4 | −4 | ad'af | hams (vs ذ), istifal/infitah (vs ظ) |
| ج | 1100000010000000 | 3 | 2 | +1 | qawiyy | shiddah/qalqalah (vs ش tafashhi, ي leen) |
| ح | 0001000000000000 | 0 | 4 | −4 | ad'af | hams+rakhawah (vs ع jahr+tawassut) |
| خ | 0001100000000000 | 1 | 3 | −2 | da'if | hams (vs غ) |
| د | 1100000010000000 | 3 | 2 | +1 | qawiyy | jahr (vs ت), istifal (vs ط) |
| ذ | 1001000000000000 | 1 | 3 | −2 | da'if | jahr (vs ث), istifal/infitah (vs ظ) |
| ر | 1010001000110000 | 3 | 2 | +1 | qawiyy | takreer, inhiraf (vs ل, ن) |
| ز | 1001000100000000 | 2 | 3 | −1 | da'if | jahr (vs س), infitah (vs ص) |
| س | 0001000100000000 | 1 | 4 | −3 | da'if | hams (vs ز), istifal (vs ص) |
| ش | 0001000000001000 | 1 | 4 | −3 | da'if | tafashhi, rakhawah (vs ج) |
| ص | 0001110100000000 | 3 | 2 | +1 | qawiyy | isti'la+itbaq (vs س), hams (vs ز) |
| ض | 1001110000000100 | 4 | 1 | +3 | qawiyy | istitalah (unique makhraj; confusable with ظ, د) |
| ط | 1100110010000000 | 5 | 0 | +5 | **aqwa** | isti'la+itbaq+jahr+qalqalah (vs ت, د) |
| ظ | 1001110000000000 | 3 | 1 | +2 | qawiyy | isti'la+itbaq (vs ذ), jahr (vs ث) |
| ع | 1010000000000000 | 1 | 2 | −1 | da'if | jahr, tawassut (vs ح) |
| غ | 1001100000000000 | 2 | 2 | 0 | mutawassit | jahr (vs خ) |
| ف | 0001001000000000 | 0 | 4 | −4 | ad'af | (labio-dental; unique makhraj) |
| ق | 1100100010000000 | 4 | 1 | +3 | qawiyy | isti'la, qalqalah (vs ك) |
| ك | 0100000000000000 | 1 | 3 | −2 | da'if | hams, istifal (vs ق) |
| ل | 1010001000100000 | 2 | 2 | 0 | mutawassit | inhiraf (lateral) (vs ن, ر) |
| م | 1010001000000010 | 2 | 2 | 0 | mutawassit | ghunnah, tawassut (vs ب) |
| ن | 1010001000000010 | 2 | 2 | 0 | mutawassit | ghunnah (vs ل, ر) |
| ه | 0001000000000001 | 0 | 5 | −5 | ad'af | khafa, hams (vs ء) |
| و (leen) | 1001000001000000 | 1 | 4 | −3 | da'if | leen (when sakin after fatha) |
| ي (leen) | 1001000001000000 | 1 | 4 | −3 | da'if | leen (when sakin after fatha) |
| ا (madd) | 1001000000000001 | 1 | 4 | −3 | da'if | (khafa; madd letter) |

**Strength classes ([LIT] al-Mursafi, *Hidayat al-Qari* ch. 2 §3; reproduced 29/29):**
aqwa ط; qawiyy ب ج د ر ص ض ظ ق; mutawassit ء غ ل م ن; da'if ت خ ذ ز س ش ع ك و ي;
ad'af ث ح ف ه. The S−W margin gives a finer ordinal *strength scale* (−5 … +5). Use it as a prior weight for how
severely a lost sifah is penalised: losing a strong letter's defining sifah (for example ط → ت) is a *lahn jali*-class
change of letter.

**Positional and conditional sifaat.** These are contextual, and the parser already resolves most of them:
- ر tafkheem/tarqeeq (`_raa_verdict`); ل tafkheem only in the Divine Name after fatha/damma; alif follows the
  preceding letter.
- Leen is a sifah only for sakin و/ي after fatha. The madd-leen *rule* is duration; the leen *sifah* is the glide quality.
- Ghunnah is a sifah of ن/م, graded in 5 levels by *mushaddad > mudgham > mukhfa > sakin-izhar > mutaharrik*
  ([LIT] classical maratib al-ghunnah).
- Qalqalah levels: sughra (sakin mid-word) < kubra (sakin at waqf) < akbar (mushaddad at waqf). Some schools use
  only 2 levels or put the mid-word sakin in wasl at the bottom. The parser emits `detail ∈ {sughra, kubra, akbar}`.

**Maratib al-tafkheem** ([LIT] al-Mutawalli school; quran-tajweed.net, alukah.net #70330):
1. Fatha followed by alif (طَا).
2. Fatha (طَ).
3. Damma (طُ).
4. Sakin. This level inherits from the preceding vowel: after fatha it is like level 2, after damma like level 3,
   after kasra it is lowest.
5. Kasra (طِ).

A 3-level (Ibn al-Jazari) view is also taught. **Isti'la strength order**: the itbaq letters come first, ط > ض > ص > ظ,
then ق > غ > خ. The [DATA] letter medians of H at fatha are ط .55, ق .51, ض .49, ظ .50, ص .46, خ .47 and غ .28.
These are partly consistent: ط is top and غ is weakest.

---

## 2. Per-sifah physical and acoustic correlates, algorithms, and engine status

Notation: frame-level spectrum |X(f)|², band power `P[a,b]`, F0 period T0, Bark `Z(f)=26.81f/(1960+f)−0.53`
(Traunmüller 1990).

| Sifah | Physical definition | Primary acoustic correlate (equation) | Secondary | Engine now | Gap / fix |
|---|---|---|---|---|---|
| **Hams / Jahr** | Breath runs vs breath held (airflow), *not* voicing (Heselwood & Maghrabi 2015) | Stops ت/ك: **aspiration VOT** = t(voicing onset) − t(burst), expect ≳35 ms (Kulikov: 51 ms vs ṭ 14 ms). Fricatives: **breath-noise share** `HNR`, **CPP** (Hillenbrand 1994) and low-band voicing ratio `P[0,500]/P[500,8k]` in the frication core | H1−H2 (Hanson 1997) at the following vowel onset (breathy onset after hams), zero-crossing rate | `hams_jahr.py`: F0 voicing fraction or HNR over a coarse window. [DATA] fails 186/304 on experts | Replace with a landmark-anchored window (burst/frication from §shiddah), VOT for ت/ك, CPP + aperiodicity (WORLD D4C, Morise 2016) for fricatives. Exclude ط ق ء from any "voicing" test (majhur by low airflow). ح is often realised with partial voicing: use a per-reciter reference, not HNR<3 |
| **Shiddah / Tawassut / Rakhawah** | Sound locked (full occlusion) / partial / running | **Closure landmark** (Stevens 2002; Liu 1996): an abrupt −ΔE in 0–8 kHz of ≥ 8 dB within 10 ms, then a burst `+Δ` highband flux. Closure duration `t_burst − t_close`. Tawassut: continuous voicing with an energy dip of 3–10 dB (sonorant: `E_min/E_vowel`). Rakhawah: no dip below −8 dB, frication present | Burst spectral COG (Forrest 1988); `P[3k,8k]` continuity | `sukoon_spectrum.py`: silence-run < −25 dB. [DATA] occlusion 0 for 94% | Reuse `qalqalah_engine.detect_release_burst`. Make the metric a 3-class **degree-of-constriction** ordinal `c = max dip dB`, with ordered cut-points learned from reference (§3.5). Drop the 1.0/1.5/2.2 duration ratios |
| **Isti'la / Istifal** | Tongue dorsum raised → uvular/pharyngeal retraction | Vowel F2 lowering / F1 raising: `H = 1 − (F2−F1)/(F2_ref−F1_ref)`. Better in Bark: `h_B = (Z2−Z1)_ref − (Z2−Z1)` and `Z3−Z2` rise (Al-Tamimi 2017, doi:10.5334/labphon.19) | F2 at onset (locus; Kulikov: boundary ≈1500 Hz), tense voice quality, lowered burst COG | `formants.py` `judge_weight` (works: [DATA] graded ordering) | Bark-difference and onset/mid two-point measurement; ordinal gradation model (§3.5) |
| **Itbaq / Infitah** | Tongue body clamped to palate (ص ض ط ظ) | Stronger F2 depression than isti'la alone plus **F3−F2 widening**, HF attenuation >3.5 kHz | Z3−Z2 divergence, lowered F2 *onset* locus (locus equation slope, Sussman 1991) | `itbaq.py` (heaviness + 2 secondary cues) | Itbaq vs isti'la-only letters (ق غ خ) should differ: learn the contrast `H_itbaq − H_isti'la` per reciter. [DATA] ط .55 vs ق .51 is weak, so needs onset-locus measurement |
| **Idhlaq / Ismat** | Letters from lip/tongue tip, easy and fluent (فرمنلب); ismat = "withheld" (phonotactic: quadriliteral roots need a dhalaq letter) | No distinct articulatory gesture. At most a **transition-speed proxy**: formant velocity `|dF2/dt|` at the CV boundary, shorter transitions | – | none | **Don't score it.** It's a lexical/phonotactic class (al-Mursafi counts it as neutral). Keep it as a lattice attribute for FCA/strength only |
| **Safir** | Whistle: high-frequency jet noise (ص س ز) | Spectral moments in the frication (Jongman et al. 2000): **COG** `μ1=Σf·S/ΣS` high (س ≳ 6 kHz, ص lower through emphasis), **kurtosis** `μ4` peaky; `P[5k,11k] − P[1k,4k]` | ز: voicing bar (low-band energy); ص: lower COG than س | `ghair_mutadhaddah.validate_safir`. [DATA] 363/497 SKIPPED (empty feedback, so the cause is undiagnosed) | Diagnose the skip. Add COG/SD/kurtosis; judge ص–س–ز as 3 points in (COG, voicing, H) space, not a single threshold |
| **Tafashhi** | Air spreads in the mouth (ش) | Broad, flat spectrum: **spectral SD** μ2 large, COG ≈ 3–4.5 kHz (lower than س), low kurtosis, spectral flatness `exp(mean log S)/mean S` in 2.5–6 kHz | Wiener entropy | `validate_tafashhi` (flatness). [DATA] 44 PASS / 6 fail-warn: OK | Add COG/SD to separate ش from س (the common confusion) |
| **Istitalah** | ض: sound extends along the tongue edge through its whole makhraj | Sustained constriction without a burst, long **F2 transition** (slow, `Δt(F2 settle)` ≳ 60 ms), no qalqalah release | Absence of the ظ-like interdental frication (low HF noise) | `validate_istitaalah`. [DATA] 68/75 WARNING (duration ratio 0.67 < 1 threshold) | The duration threshold is wrong for experts (median 0.67). Use transition duration, no-release, and the ض-vs-ظ/د posterior ratio from SSL (§3.7) |
| **Takreer** | ر: tongue tip ready to repeat; ideal = a single tap (repetition is avoided) | Tap count = energy dips of 6–45 ms (existing); a trill has regular dips at 25–35 Hz (Solé 2002, doi:10.1006/jpho.2002.0179) → **amplitude-modulation spectrum peak** 20–40 Hz | Dip depth | `count_taps` ([DATA] 195/262 PASS) | Add AM-spectrum periodicity and require ≥2 *periodic* dips for "rolled" |
| **Inhiraf** | ل/ر deviate from their makhraj (lateral for ل, toward the back for ر) | ل: lateral **antiresonance** (spectral zero ≈ 2–3 kHz), low F1, abrupt F1 transitions; ر: low F3 (<2.2 kHz) | Lateral: F3 relative to /n/ | none | Score as ل ↔ ن/د and ر ↔ ل confusions: posterior ratio + F3/antiformant check |
| **Leen** | و/ي sakin after fatha pronounced with ease | Smooth diphthong: F2 trajectory `a→i` rising (`aj`) or `a→u` falling (`aw`), **monotone and unconstricted** (no frication, no glottalization). Measure the glide endpoint F2 and trajectory smoothness (2nd derivative energy) | CPP stability | only the madd-leen duration | New `validate_leen`: diphthong endpoints vs the reciter's own /i/,/u/ (Lobanov), plus a smoothness penalty |
| **Qalqalah** | Release bounce of sakin قطبجد | Closure + burst `flux_ratio`, `energy_rise_db`, echo (existing) | Burst COG per letter | `qalqalah_engine.py` | Ordinal level model: [DATA] akbar distinct (18 dB, 174 ms hold), kubra ≈ sughra in burst |
| **Ghunnah** | Nasal resonance of ن/م | NER `P[150,400]/P[750,1100]` (existing), **A1−P0** (Chen 1997, doi:10.1121/1.419620), F1 bandwidth | Nasal murmur duration | `formants.py` nasal functions | 5-level maratib al-ghunnah as ordinal, like tafkheem |
| **Khafa** | ه and the madd letters are "hidden" | Low intensity and breathy (CPP low) but *present*: `E_h − E_floor > 8 dB` | – | none (hams only) | Detect *dropping* of ه at word end (presence test), not strength |

---

## 3. Mathematical models

### 3.1 Letters as points in a feature lattice (FCA, Hamming and poset geometry)
- **Formal context** K = (G = 29 letters, M = 15 binary sifaat (tawassut as a 3rd manner value), I).
  The concept lattice B(K) has **33 concepts** [DATA]. Concepts are exactly the *natural classes* used in teaching
  (for example {ق ط} = isti'la ∧ qalqalah). Ganter & Wille 1999, doi:10.1007/978-3-642-59830-2.
- **Similarity**: Frisch–Pierrehumbert–Broe natural-class similarity
  `sim(a,b) = |C(a)∩C(b)| / (|C(a)∩C(b)| + |C(a)△C(b)|)` over the concepts containing each letter
  (NLLT 22:179, 2004). It's better than Hamming because it discounts redundant features.
- **Makhraj conditioning (key).** Sifaat distinguish letters that share a makhraj. Define the **distinctive set**
  `D(ℓ) = ⋃_{ℓ'∈makhraj(ℓ), ℓ'≠ℓ} (s(ℓ) △ s(ℓ'))`. Only sifaat in D(ℓ) are phonemically decisive for ℓ. The others
  (for example istifal of ت vs ك, which are in different makharij) are redundant. Use D(ℓ) to choose which detectors
  vote for a letter (last column of §1).
- **Strength poset**: ℓ ⪰ ℓ' iff strong(ℓ) ⊇ strong(ℓ') and weak(ℓ) ⊆ weak(ℓ'). The count S−W is a linear
  extension of this order. It reproduces al-Mursafi exactly.
- **Information** (weights from Quran letter frequencies, N=325,665 letters [DATA]): H(L)=4.21 bits. Single-sifah
  informativeness: idhlaq 0.96, jahr 0.72, ghunnah 0.65, inhiraf 0.62, leen 0.57, qalqalah 0.43, isti'la 0.29,
  itbaq 0.13, tafashhi 0.06, istitalah 0.05 bits. Manner (3-way) gives 1.53 bits. Greedy selection:
  manner → jahr → idhlaq → ghunnah → leen → isti'la → takreer → safir reaches H(L|S)=0.68, and all sifaat reach 0.53.
  Implication: rare sifaat (tafashhi, istitalah) carry little letter information but a lot of *pedagogical*
  information. Weight verdicts by pedagogical risk, not by MI.

### 3.2 Joint profile scoring (replaces one-at-a-time judgement)
For an aligned letter token with evidence e = (e_1..e_k) from the detectors in D(ℓ):
1. **Calibrated per-sifah likelihoods.** The lattice gives *negatives for free*. In reference recitations, the same
   detector applied to letters lacking sifah j gives the null class, so each detector gets a 2-class (or ordinal)
   likelihood `p(e_j | s_j=1)`, `p(e_j | s_j=0)` per reciter-normalised feature. Calibrate with Platt/temperature
   scaling (Guo et al., arXiv:1706.04599) fitted on reference reciters.
2. **Product of experts over the lattice** (Hinton 2002, doi:10.1162/089976602760128018):
   `log P(ℓ' | e) = Σ_j β_j log p(e_j | s_j(ℓ')) + log π(ℓ' | makhraj(ℓ)) − log Z`,
   enumerated over the ≤ 5 same-makhraj candidates ℓ' (exact inference, since the output space is tiny).
   Outputs: realised letter (MAP), **which sifaat flipped** (s(ℓ̂) △ s(ℓ)), and a letter-level margin
   `log P(ℓ|e) − max_{ℓ'≠ℓ} log P(ℓ'|e)`. The temperatures β_j correct over-confident, correlated experts.
3. **Dependencies.** Isti'la/itbaq/tafkheem share the F2 cue, and jahr/qalqalah share the burst cue. Either fit a
   **Chow–Liu tree** over detector residuals (Chow & Liu 1968, doi:10.1109/TIT.1968.1054142), or use a Gaussian
   copula on the probit-transformed scores. Use this Bayesian network instead of the naive product when
   residual correlation |ρ| > 0.3.
4. **Conformal letter sets** (Angelopoulos & Bates, arXiv:2107.07511): non-conformity `−log P(ℓ|e)`, peer-LOO
   calibration (roadmap A4). A FAIL means the intended letter isn't in the 1−α set.

### 3.3 Information-theoretic feature selection
- Per sifah, choose acoustic features by **mRMR** (Peng, Long, Ding 2005, doi:10.1109/TPAMI.2005.159) with
  **KSG** mutual-information estimates (Kraskov et al. 2004, doi:10.1103/PhysRevE.69.066138):
  `max I(f; s_j) − (1/|S|) Σ_{g∈S} I(f; g)`.
- Model order / number of features by **MDL**. The two-part code `L(model) + L(data|model)` (Rissanen 1978) is
  equivalent to BIC for logistic detectors. For neural probes use **MDL probing** (Voita & Titov, arXiv:2003.12298).
- Candidate pool per token: formants (onset, mid) in Hz and Bark differences, HNR, CPP, H1−H2, H1−A1, H1−A3,
  spectral moments μ1..μ4, flatness, ZCR, subband energies (8 bands), VOT, closure duration, burst flux/COG,
  AM-spectrum peak, NER, A1−P0.

### 3.4 Robust per-reciter normalisation
- **Lobanov** (JASA 49:606, 1971, doi:10.1121/1.1912396) made robust: `z = (F − median_r,v(F)) / (1.4826·MAD_r,v)`.
  Stats are per reciter r and per vowel v, over *light* reference tokens. The existing `build_reference` is the
  median half of this.
- **Bark differences** (Syrdal & Gopal 1986, doi:10.1121/1.393381) are speaker-intrinsic and need no reference, so
  use them as a fallback when the light reference has < 2 tokens.
- **VTLN**: warp α_r = median F3_r(light /a/) / F3_pop. Apply F → F/α_r before absolute thresholds
  (Lee & Rose 1998, doi:10.1109/89.650310).
- Choose among them with Adank et al.'s criterion (JASA 116:3099, 2004, doi:10.1121/1.1795335): maximise
  between-category / within-category variance across reciters. That means heavy vs light tokens across the 6 reference
  reciters [PROPOSAL].
- **Hierarchical**: roadmap A5 random intercept per reciter×vowel for H. The band = posterior predictive of a new
  reference-like reciter.

### 3.5 Gradations as ordinal / isotonic models
- The text gives the *level* k (tafkheem maratib 1–5, qalqalah sughra/kubra/akbar, ghunnah maratib, manner
  shiddah/tawassut/rakhawah). The audio gives the strength y (H, rise dB, NER, dip dB).
- **Forward model (scoring):** isotonic mixed model `y = f(k) + u_r + ε`, f non-increasing in k
  (pool-adjacent-violators, Barlow et al. 1972). Band per level = predictive quantiles. The student is scored on
  (a) each token vs its level band, and (b) **within-student order consistency**: Kendall τ between k and y over
  their tokens. Test (b) is invariant to microphone and speaker, since it's rank-only.
- **Inverse model (diagnosis):** proportional-odds cumulative link `P(k ≤ j | y) = σ(θ_j − β·y)` (McCullagh 1980,
  JRSS-B 42:109). For multi-feature input use CORAL (Cao, Mirjalili, Raschka, arXiv:1901.07884). This yields feedback
  like "your طَا sounded like a damma-level tafkheem".
- [DATA] support: level medians 0.595 / 0.481 / 0.201 (see §0.3). Levels 4 (sakin) and 5 (kasra) aren't yet emitted
  with H. The parser only emits TAFKHEEM on letters with a vowel window, so emitting for sakin (using the *preceding*
  vowel window) is needed.

### 3.6 Neural probing of SSL representations
- The aligner already computes wav2vec2 hidden states (`CTCForcedAligner._forward`). Train **linear probes per layer**
  for each sifah on reference-reciter frames. Probe on aligned letter cores.
- Controls: selectivity vs control tasks (Hewitt & Liang, arXiv:1909.03368), MDL codelength (Voita & Titov). Layer
  analysis per Pasad et al., arXiv:2107.04734. SSL encodes articulatory kinematics (Cho et al., arXiv:2210.11723;
  arXiv:2310.10788, r≈0.8 to EMA).
- **Leakage caveat (critical):** a sifah is a deterministic function of letter identity. A probe can reach 100% by
  decoding the letter. Valid probing uses **within-letter contrasts** where the sifah varies and the letter is fixed:
  - ر tafkheem vs tarqeeq
  - lam of Allah heavy vs light
  - alif after heavy vs light letters
  - ن ghunnah (ikhfa/idgham) vs izhar
  - qalqalah letters sakin vs mutaharrik
  - regression onto the continuous acoustic targets (H, VOT, NER) within a letter
- The same caveat applies to the muaalem sifaat heads (arXiv:2509.00094): test them on counterfactual audio (§4).

---

## 4. Engine integration, calibration, validation

| Item | Where | Calibration | Validation |
|---|---|---|---|
| Landmark detector (closure, burst, VOT, frication onset) | new `app/acoustic/landmarks.py`; reuse `qalqalah_engine.detect_release_burst`, `EvalContext.frication_window` | per-reciter median/MAD of closure dip, VOT | [DATA] hams FAIL rate on 6 experts must drop from 61% to <10%; shiddah occlusion > 0 on ≥ 90% |
| Hams/jahr as airflow | `app/sifaat/hams_jahr.py::validate_hams_jahr` | VOT and CPP bands from references (roadmap A1/A4) | Imams vs peers contrast; synthetic: add aspiration noise / voice bar |
| Manner ordinal | `app/sifaat/sukoon_spectrum.py` | cut-points θ from references | Monotone ordering shiddah < tawassut < rakhawah in dip dB on references |
| Bark/Lobanov heaviness + ordinal maratib | `app/sifaat/formants.py::judge_weight`, `build_reference`; parser `_detect_weight` to emit sakin/kasra levels | isotonic per-level bands | ρ(level, H) on held-out peer; Kendall τ per reciter ≥ 0.3 |
| Spectral moments for safir/tafashhi | `app/acoustic/features.py` (add `spectral_moments`), `ghair_mutadhaddah.py` | COG/SD bands per letter | ص/س/ز/ش separability (AUC) on references |
| New validators: leen, inhiraf, khafa | `app/sifaat/leen.py`, `inhiraf.py` (+ `RuleType`s, parser `_detect_sifaat`) | per-reciter /i/,/u/ endpoints | Reference pass ≥ 95% (peer-LOO) |
| Joint profile (PoE + makhraj) | new `app/sifaat/profile.py`, invoked after per-sifah diagnostics in `pipeline.py`; letter table as `app/data/sifaat_lattice.json` | β_j, Chow–Liu from references | Letter-substitution detection on synthetic swaps |
| Probing / teacher | `research_agency_lab/.../probe_sifaat.py` (offline) | – | Within-letter contrasts only |

**Counterfactual validation (no human labels needed) [PROPOSAL].** Use `octave/lpc.m` to resynthesise reference
tokens. Each manipulation should be flagged by exactly the intended detector and should leave the others unchanged.
That tests the *specificity* of the joint profile.
- **Tafkheem loss**: raise F2 by 300–500 Hz.
- **Hams loss**: remove the aspiration segment.
- **Rakhawah violation**: insert a 40 ms closure.
- **Safir loss**: low-pass the frication at 4 kHz.
- **Nasality loss**: add a zero at 250 Hz.

**Julia** (`research_agency_lab/substrate_library/julia`, current deps suffice for most):
- The lattice/FCA, S−W strength and MI table are ~80 lines, using no new packages.
- Isotonic PAVA and the proportional-odds model use Optim + ForwardDiff (already there).
- Chow–Liu uses a hand-written maximum spanning tree.
- KSG MI: add `Associations.jl` (optional).
- Calibration of β_j reuses the `calibrate.jl` LOO machinery.

**Octave cross-check**: extend `qaari_features.m` with VOT, CPP, spectral moments, A1−P0 and Bark differences.
The Python/Octave agreement target is |Δ| < 10%, as for the existing core_ms/HNR.

**GPU**: none needed for everything except probing / muaalem inference. Probing uses cached aligner states on CPU
(strategic 59-ayah subsets are fine). Muaalem (Wav2Vec2-BERT ~600M) runs on CPU at ~1–2× real time for subsets.
Full-Quran runs would want a GPU, but cloud compute is out of scope here.

---

## 5. Prioritised build plan

| # | Item | Why | Effort |
|---|---|---|---|
| 1 | **Fix hams/jahr** (landmark-anchored window, VOT for ت/ك, CPP/aperiodicity for fricatives, ط ق ء never voicing-tested) | 186 false FAILs on experts; conceptual error | 1–1.5 d |
| 2 | **Fix shiddah/tawassut/rakhawah** (reuse burst detector, dip-dB ordinal, drop duration ratios) | 94% + 61% spurious WARNINGs | 1 d |
| 3 | **Diagnose safir skips** (363/497) + add spectral moments to safir/tafashhi | Most safir tokens unscored | 0.5–1 d |
| 4 | **Tafkheem maratib ordinal/isotonic** + emit sakin/kasra levels + Bark/Lobanov | Ijazah-level gradation, data already supports it | 1.5 d |
| 5 | **Lattice data file + strength classes + makhraj-distinctive sets** (Julia + JSON) | Foundation for joint scoring and feedback | 0.5 d |
| 6 | **Joint profile PoE** with calibrated detectors and lattice negatives; conformal letter sets | Moves from 1-at-a-time to letter-level verdicts | 3 d |
| 7 | Qalqalah level bands (akbar hold, kubra by context) + ghunnah maratib ordinal | Level mismatch found | 1 d |
| 8 | New validators: leen (diphthong), inhiraf (lateral/ر), khafa (ه presence); idhlaq explicitly unscored | Close the known gaps | 2 d |
| 9 | LPC counterfactual validation suite (Octave) | Specificity without labels | 1.5 d |
| 10 | SSL probing with within-letter contrasts; muaalem as teacher, tested on counterfactuals | Robust detectors for noisy/masjid audio | 3–4 d (CPU, subsets) |
| 11 | mRMR/KSG feature selection + Chow–Liu dependencies | Refinement after 6 | 1.5 d |

---

## 6. Sources
- Abdelfattah, Khalil, Abbas. *Automatic Pronunciation Error Detection and Correction of the Holy Quran's Learners Using Deep Learning* (QPS, muaalem). arXiv:2509.00094. HF `obadx/muaalem-model-v3_2`, `obadx/muaalem-annotated-v3`.
- Heselwood & Maghrabi (2015). J. Semitic Studies 60(1):131. doi:10.1093/jss/fgu035.
- Kulikov (2021/22). *Voice and Emphasis in Arabic Coronal Stops*. Language and Speech. doi:10.1177/0023830920986821.
- Al-Tamimi (2017). Laboratory Phonology 8(1):28. doi:10.5334/labphon.19.
- Jongman, Herd, Al-Masri, Sereno, Combest (2011). J. Phonetics 39(1):85. doi:10.1016/j.wocn.2010.11.007.
- Jongman, Wayland, Wong (2000). Fricative acoustics. JASA 108:1252. doi:10.1121/1.1288413.
- Forrest et al. (1988). Spectral moments of stops. JASA 84:115. doi:10.1121/1.396977.
- Stevens (2002). Landmarks and distinctive features. JASA 111:1872. doi:10.1121/1.1458026.
- Liu (1996). Landmark detection. JASA 100:3417. doi:10.1121/1.416983.
- Hillenbrand, Cleveland, Erickson (1994). CPP. JSHR 37:769. doi:10.1044/jshr.3704.769.
- Hanson (1997). H1−H2. JASA 101:466. doi:10.1121/1.417991.
- Morise (2016). WORLD/D4C. Speech Comm. 84:57. doi:10.1016/j.specom.2016.09.001.
- Chen (1997). A1−P0 nasality. JASA 102:2360. doi:10.1121/1.419620.
- Pruthi & Espy-Wilson (2004). Speech Comm. 43:225. doi:10.1016/j.specom.2004.06.001.
- Solé (2002). Trills. J. Phonetics 30:655. doi:10.1006/jpho.2002.0179.
- Lobanov (1971). doi:10.1121/1.1912396.
- Syrdal & Gopal (1986). doi:10.1121/1.393381.
- Adank et al. (2004). doi:10.1121/1.1795335.
- Lee & Rose (1998). VTLN. doi:10.1109/89.650310.
- Ganter & Wille (1999). FCA. doi:10.1007/978-3-642-59830-2.
- Frisch, Pierrehumbert, Broe (2004). NLLT 22:179.
- Hinton (2002). PoE. doi:10.1162/089976602760128018.
- Chow & Liu (1968). doi:10.1109/TIT.1968.1054142.
- Guo et al. Calibration. arXiv:1706.04599.
- Angelopoulos & Bates. Conformal. arXiv:2107.07511.
- Peng, Long, Ding (2005). mRMR. doi:10.1109/TPAMI.2005.159.
- Kraskov et al. (2004). KSG. doi:10.1103/PhysRevE.69.066138.
- Voita & Titov. MDL probing. arXiv:2003.12298.
- Hewitt & Liang. Control tasks. arXiv:1909.03368.
- Pasad et al. arXiv:2107.04734.
- Cho et al. arXiv:2210.11723 and arXiv:2310.10788.
- McCullagh (1980). JRSS-B 42:109.
- Cao, Mirjalili, Raschka. CORAL. arXiv:1901.07884.
- Algabri et al. (2022). Articulatory-level MDD for Arabic. Mathematics 10:2727. doi:10.3390/math10152727.
- al-Mursafi. *Hidayat al-Qari* ch. 2 §3 (strength classes); maratib al-tafkheem: quran-tajweed.net, alukah.net/sharia/0/70330.
- Qalqalah spectrogram study: CEUR-WS Vol-1539 paper 2 (IIUM).
