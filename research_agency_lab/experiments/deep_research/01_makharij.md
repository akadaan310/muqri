# 01 · Makharij (Articulation Points): Verification Research for qaari-eval

Deep-research report, 2026-09-23. Scope: all 17 makharij (Hafs ʿan ʿĀṣim, Ibn al-Jazarī's scheme), from
beginner (101) to ijazah level, and the mathematics needed to **verify** them. The report builds on
`frontier_math_roadmap.md` (Bures–Wasserstein, conformal, Sinkhorn, SPD, elastic FDA, signatures, TDA)
and does not restate it. Where this report needs thresholds or reference distributions, it reuses A1–A4
(`Frontier.jl`: `bw_barycenter`, `w2_gauss`, `conformal_threshold`, `sinkhorn_cost`).

**Headline gap:** the engine has no makhraj verification. Today the only letter-identity signal is
`AlignedUnit.confidence`, the mean of `exp(lp[t, target])` over the non-blank Viterbi frames
(`app/aligner.py::spans_from_path`). That value is a **target posterior with no competitor
normalisation**. Three things follow. (a) It cannot tell *which* letter was produced. (b) It
saturates, because CTC is peaky and the frames that count are usually one or two. (c) It inherits
the Quran-trained model's implicit text prior, so it leans toward "heard the canonical letter".

---

## 0. Engine facts that constrain everything below (verified in code + model vocab)

| Fact | Consequence for makhraj verification |
|---|---|
| Model `TBOGamer22/wav2vec2-quran-phonetics` vocab = 38 tokens: `' - H a b d e f g h i j k l m n o q r s t u w y z ā ī ū ʿ ḍ ḥ ṣ ṭ ẓ`, plus space, `\|`, PAD and UNK | **No single token for ث ذ خ ش غ.** `phonetic_tokens` maps ث→`th` and then `list(cons)`, which gives **two tokens `t`,`h`**. The same happens for ذ→`d,h`, خ→`k,h`, ش→`s,h`, غ→`g,h`. |
| So ث vs ت+ه, ذ vs د+ه and خ vs ك+ه are string-ambiguous at the token level | A per-token GOP is ill-defined for 5 letters. The GOP must be **sequence-level** (CTC likelihood ratio over token *strings*, §3.1b). That handles multi-token units natively. |
| Word-initial hamza gets **no token** (`first_in_word → cons = ""`) | The makhraj of initial ء (aqṣā al-ḥalq) cannot be verified from CTC. It needs an acoustic glottal-closure test (§2.3, §3.5). |
| Model trained on Quran audio with Quran text | Strong implicit LM, i.e. "linguistic bias" (Geng et al., arXiv:2604.22133). Posteriors are pushed toward canonical. We need a less-biased second judge (§3.6). |
| Parser already has `MUTAJANISAYN_PAIRS`, `MUTAQARIBAYN_PAIRS`, `HEAVY/ITBAQ/HAMS/SAFIR/TAWASSUT/QALQALAH` sets (`app/tajweed_rules/parser.py`) | The sifat half of the letter feature vector already exists. What is missing is the **makhraj coordinate** and the **graph**. |
| `app/sifaat/formants.py` already measures F1/F2/F3, H-index, itbaq convergence, NER; `ghair_mutadhaddah.py` measures HF tilt (safir), tap count (takrir) | The acoustic cue verifiers (§3.5) can reuse `AcousticContext`, `frication_window`, `silence_runs` and `voicing_fraction`. |

---

## 1. Scholarship frame (why 17, and levels 101 → ijazah)

- **17 makharij**: al-Khalīl (Kitāb al-ʿAyn), adopted by Ibn al-Jazarī (*al-Nashr*; *al-Muqaddimah
  al-Jazariyyah*, v. 10–18). This is the Hafs mainstream and our target.
- **16**: Sībawayh (*al-Kitāb*, bāb al-idghām) and al-Shāṭibī. They drop al-jawf and distribute the
  madd letters to ḥalq/lisān/shafatān.
- **14**: al-Farrāʾ, Quṭrub, al-Jarmī. They merge ل ن ر into one ṭaraf makhraj.
  *(This debate becomes a testable hypothesis in §3.3: the eigengap of the confusion Laplacian.)*
- **Laḥn jalī vs laḥn khafī**: substituting one letter for another (ض→ظ, ح→ه) is *laḥn jalī*
  regardless of articulatory closeness. Incomplete sifāt is *laḥn khafī*. **Design rule:** the
  articulatory ground metric (§3.2) is used for *diagnosis, partial credit and learner-progress
  curves*, never to soften a PASS/FAIL on letter identity.
- **Levels**:
  - **101**: region-level (throat vs tongue vs lips); gross substitutions (ح→ه, ع→ء/ا, ق→ك, ث→س/ت, ذ→ز/د, ض/ظ→د/ز).
  - **Intermediate**: exact makhraj within a region (the 3 ḥalq levels; ق vs ك), plus sifāt that
    separate same-makhraj letters (ت/ط/د, س/ص/ز).
  - **Advanced**: idghām between adjacent makharij with the sifah retained (بسطت, أحطت, نخلقكم),
    ḍād istiṭālah, rāʾ tafkhīm/takrīr control, lām of Allāh.
  - **Ijazah**: consistency across a whole khatm segment, ḍād vs ẓāʾ separation in every context,
    hamza clarity without over-stress (nabr), no leakage of pharyngealisation to neighbours
    (tafkhīm spread), khayshūm duration/strength per rank (ghunnah levels).

---

## 2. Canonical makhraj table with measurable acoustic correlates

Notation: IPA is the contemporary Hafs realisation (Husary-style). Numbers are typical adult-male
ranges from the phonetic literature. They are **starting priors only**. All operational bands must be
re-estimated from the anchor + ijazah peers with the roadmap A1/A4 machinery.

### 2.1 Table

| # | Region | Makhraj (Ar.) | Letters | IPA | Anatomy | Primary acoustic correlates (what to measure) |
|---|---|---|---|---|---|---|
| 1 | Jawf | الجوف | ا و ي (madd) | aː uː iː | Oral + pharyngeal cavity; no point constriction | Steady formants over the madd core. /iː/: F1≈280–350, F2≥2100 Hz. /uː/: F1≈300–380, F2≈700–950 Hz. /aː/: F1≈650–800, F2 ≈1100–1300 (heavy) vs 1500–1800 (light). Checks: no diphthongisation (F2 slope ≈0 across core), no nasalisation (A1−P0; Chen 1997), duration in ḥarakāt (beat-normalised, roadmap B1). |
| 2 | Ḥalq | أقصى الحلق | ء ه | ʔ h | Glottis/larynx (deepest throat) | ء: full closure → silence 20–80 ms or creaky offset/onset; abrupt energy discontinuity; **no formant transitions** into the neighbour vowel; low H1−H2 / jitter at edges. ه: aperiodic noise that keeps the formant pattern of the neighbour vowel; low intensity; high H1−H2 (breathy); low HNR. |
| 3 | Ḥalq | وسط الحلق | ع ح | ʕ ħ | Mid-pharynx (epiglottal/pharyngeal constriction) | ع: often an approximant or "tight approximant" with laryngealisation (Heselwood 2007). **High F1** (≥700–900 Hz even next to /i/), F2 lowered towards F1, intensity dip, creak. ح: voiceless frication with energy concentrated <2 kHz, noise F1 high. Neighbour vowels: F1↑, F2↓. Stronger frication and higher F1 than ه. |
| 4 | Ḥalq | أدنى الحلق | غ خ | ʁ~ɣ χ~x | Upper pharynx / uvula region (nearest throat) | Frication with spectral peak ≈1–2 kHz (lower than velar x). Possible uvular-trill amplitude modulation (25–35 Hz). Neighbour vowels: F1↑, F2↓ (musta'liyah). غ has a voicing bar; خ is voiceless (hams). |
| 5 | Lisān | أقصى اللسان مع ما فوقه من الحنك الأعلى | ق | q | Tongue back + soft palate/uvula (deeper than ك) | Stop closure, then a **low, compact burst** (~0.8–1.6 kHz, below the following vowel's F2). **Unaspirated** in Hafs (short-lag VOT). Neighbour vowel F1↑ F2↓ (istiʿlāʾ). Qalqalah on sukūn: release + short vocalic echo. |
| 6 | Lisān | أسفل من مخرج القاف قليلاً | ك | k | Tongue back slightly forward of ق, on hard/soft palate | Compact mid-frequency burst tracking the following vowel's F2 (~1.5–3 kHz). **Aspirated** (VOT ≈ 40–80 ms, hams). **Velar pinch** (F2–F3 convergence at the vowel edge). No F1 raising. |
| 7 | Lisān | وسط اللسان مع الحنك الأعلى | ج ش ي (non-madd) | dʒ ʃ j | Tongue middle + hard palate | ج: stop closure + frication release (peak ≈2.5–4 kHz), voiced, must not lenite to [ʒ] (shiddah + qalqalah). ش: intense fricative, peak ≈2.5–4.5 kHz, M1 ≈3.5–4.5 kHz, spreading (tafashshī = broad spectrum). ي: glide, F2 ≥2000 Hz, low F1. High F2 locus (~2.2 kHz) for all three. |
| 8 | Lisān | إحدى حافتي اللسان مع الأضراس العليا | ض | dˤ (hist. ɮˤ) | Side(s) of tongue against upper molars, with istiṭālah (lengthening along the side) | Modern Hafs: voiced **stop** (closure + burst; shiddah), itbāq/istiʿlāʾ: **strong F2 lowering** of the neighbour vowel (F2 onset ~1000–1250 for /a/), F1↑. Burst centroid lower than د. Short-lag or prevoiced VOT. Istiṭālah is hypothesised to show as a **longer F2 transition** (slower release trajectory) than ط. This is speculative and must be validated on the anchor. The historical lateral fricative (Sībawayh) is rarely heard today. |
| 9 | Lisān | أدنى حافتي اللسان إلى منتهى طرفها مع اللثة | ل | l (ɫ in Allāh after a/u) | Front edges of tongue to tip, against upper gum | Lateral approximant: F1 ≈300–450; F2 ≈1300–1700 (clear) vs ≈800–1100 (dark ɫ); antiformant ≈2–3 kHz; abrupt amplitude edges at onset/offset. |
| 10 | Lisān | طرف اللسان تحت مخرج اللام قليلاً | ن | n | Tongue tip + gum (slightly below/forward of ل) | Nasal murmur: N1 ≈250 Hz, **alveolar antiformant ≈1.4–2.2 kHz** (labial ≈0.75–1.25 kHz; Kurowski & Blumstein 1987). Abrupt murmur→vowel spectral change. F2 locus ≈1700–1800 Hz. Ghunnah duration (khayshūm, #17). |
| 11 | Lisān | طرف اللسان مع ظهره قليلاً | ر | r / ɾ | Tongue tip plus a little of its dorsum, against gum (behind ن) | Tap 20–30 ms closure. Takrīr to be *minimised*: ≤1 dominant tap (existing `count_taps`). Tafkhīm: F2 of neighbour /a/ lowered. F3 not strongly lowered (unlike English r). |
| 12 | Lisān | طرف اللسان مع أصول الثنايا العليا | ط د ت | tˤ d t | Tip of tongue + roots of upper central incisors (nitʿiyyah) | Diffuse-rising burst, peak ≈3.5–5 kHz. F2 locus ≈1600–1800 Hz (locus-equation slope ≈0.35–0.5). **ت aspirated** (VOT ≈ 40–80 ms). **ط unaspirated** (≈10–25 ms) + F2 lowering + F1↑. د voiced (prevoicing / short lag). Gulf VOT data: plain 72 ms vs emphatic 17 ms (Kulikov et al. 2024). |
| 13 | Lisān | طرف اللسان مع أطراف الثنايا العليا | ظ ذ ث | ðˤ ð θ | Tip of tongue + edges of upper incisors (lithawiyyah; interdental) | Non-sibilant: **low relative amplitude**, flat diffuse spectrum, low kurtosis, M1 broad/high (≥5 kHz but low peak prominence). ث voiceless; ذ/ظ voiced. ظ adds itbāq F2 lowering (≈ equal to ض's). The ظ vs ض contrast is continuous noise (rikhwah) vs closure + burst (shiddah). |
| 14 | Lisān | طرف اللسان مع ما بين الثنايا (صفير) | ص س ز | sˤ s z | Tip of tongue near the inner plates of the incisors, with a small gap (asaliyyah) | Sibilant: **high amplitude**, spectral peak ≈4–8 kHz, M1 ≈5.5–7.5 kHz, high kurtosis, negative skew. ص: lower M1 (~200–600 Hz below س) + neighbour F2↓ F1↑. ز: voicing bar, weaker noise. Ṣafīr = high-frequency prominence (existing `validate_safir`). |
| 15 | Shafatān | بطن الشفة السفلى مع أطراف الثنايا العليا | ف | f | Inner lower lip + tips of upper incisors | Labiodental non-sibilant: weak, flat spectrum. Low-mid F2 locus (~1000–1300 Hz); rising F2 into front vowels. |
| 16 | Shafatān | الشفتان | ب م و | b m w | Both lips (closed for ب م; rounded for و) | Low F2/F3 locus (~800–1100 Hz) with rising transitions. ب: diffuse-flat/falling burst, voiced, qalqalah on sukūn. م: murmur with **labial antiformant ≈0.75–1.25 kHz**. و: F1 and F2 low (~300 / 700–900 Hz), lip rounding lowers all formants. |
| 17 | Khayshūm | الخيشوم | ghunnah of ن م (shaddah, ikhfāʾ, idghām bi-ghunnah, iqlāb) | nasal murmur / nasalised vowel | Nasal cavity | Nasal pole ≈250 Hz + zero ≈750–1100 Hz (existing **NER**). **A1−P0** and **A1−P1** nasalisation indices on the neighbour vowel (Chen 1997). F1 bandwidth broadening. Duration ≈2 ḥarakāt. Ikhfāʾ: murmur place follows the *next* letter's makhraj (antiformant shifts), which gives a direct makhraj test of ikhfāʾ quality. |

**Classical ↔ phonetic mismatch to encode.** Classical *jahr/hams* is about breath (nafas) flow, not
vocal-fold vibration. ق and ط are *majhūr* in the classical sense, yet in Hafs recitation both are
voiceless, unaspirated stops. The verifier for ق/ط should therefore test **absence of aspiration**
(short-lag VOT). It should not test voicing. The same applies to ء (majhūr, realised as a glottal
stop). `app/sifaat/hams_jahr.py` should be checked against this.

### 2.2 Confusable pairs: discriminating cues, in priority order

| Pair (target ↔ error) | Makhraj relation | Sifāt difference | Best cues (ordered) |
|---|---|---|---|
| ض ↔ ظ | #8 vs #13 | shiddah vs rikhwah (+ istiṭālah) | closure/silence + burst present? ; frication continuity (spectral flux) ; burst centroid |
| ض ↔ د | #8 vs #12 | itbāq/istiʿlāʾ, istiṭālah | F2-onset lowering (locus residual) ; F1↑ ; burst centroid ; VOT |
| ظ ↔ ذ | same #13 | itbāq/istiʿlāʾ | F2 lowering / H-index (already in `formants.py`) |
| ظ/ذ ↔ ز | #13 vs #14 | non-sibilant vs ṣafīr | relative amplitude, peak prominence, kurtosis, M1 |
| ث ↔ س | #13 vs #14 | no ṣafīr vs ṣafīr | relative amplitude (fricative vs vowel in F4–F5 band), kurtosis, M1 |
| ث ↔ ت | same region, #13 vs #12 | rikhwah vs shiddah | closure + burst vs continuous noise ; VOT |
| ذ ↔ د, ذ ↔ ز | #13 vs #12/#14 | as above | as above |
| ح ↔ ه | #3 vs #2 | (both hams, rikhwah) | noise F1 / LPC pole < 1 kHz ; HNR ; energy <2 kHz ; neighbour-vowel F1↑ |
| ح ↔ خ | #3 vs #4 | istiʿlāʾ | spectral peak location (χ higher, 1–2 kHz, plus possible trill AM) ; neighbour F2↓ |
| ع ↔ ء | #3 vs #2 | tawassuṭ vs shiddah | formant transitions (F1↑ F2↓) present vs abrupt closure without transitions ; creak |
| ع ↔ ا (deletion) | #3 vs #1 | – | intensity dip + F1 excursion present? |
| ق ↔ ك | #5 vs #6 | istiʿlāʾ, qalqalah vs hams | burst peak freq normalised by F2 ; VOT (q short, k long) ; neighbour F1↑F2↓ |
| ق ↔ غ/ء (dialectal) | #5 vs #4/#2 | shiddah | closure + burst present ; burst frequency |
| ص ↔ س | same #14 | itbāq/istiʿlāʾ | neighbour F2↓ F1↑ ; M1 lower |
| ط ↔ ت | same #12 | itbāq/istiʿlāʾ, qalqalah, jahr (classical) | VOT short vs aspirated ; F2 lowering ; burst centroid |
| ج ↔ ʒ/ي | same #7 | shiddah | closure present ; frication duration |

**Adjacent-makhraj idghām (advanced/ijazah).** These are the pairs where the makhraj *must* merge but
a sifah may *persist*:

- **Mutajānisayn (same makhraj):**
  - ت→ط (قالت طائفة): full.
  - ط→ت (بسطت، أحطت، فرّطتم): **idghām nāqiṣ**. A single closure, but itbāq survives, so there is
    F2 lowering on the preceding vowel with a *plain* ت release.
  - د→ت (قد تبيّن), ت→د (أثقلت دعوا), ذ→ظ (إذ ظلموا), ث→ذ (يلهث ذلك), ب→م (اركب معنا).
- **Mutaqāribayn (adjacent makharij):**
  - ل→ر (قل ربّ).
  - ق→ك (ألم نخلقكم): full idghām preferred in Hafs. Test: no uvular burst, geminate ك closure,
    residual istiʿlāʾ on the preceding vowel only if the reciter chose nāqiṣ.
  - ن→ل/ر: idghām without ghunnah.
  - ن→و/ي: nāqiṣ with ghunnah, so the nasal murmur persists while the oral makhraj moves.

**Verification model for all of these:** (i) a **geminate-closure test** (one closure of duration ≈
2× singleton, no intermediate release burst); (ii) a **persistent-sifah test** (itbāq H-index or
ghunnah NER on the correct side of the junction); (iii) **absence of the first makhraj's release
cue** (e.g. no low q-burst in نخلقكم).

---

## 3. Mathematical and computational models

### 3.1 GOP from CTC posteriors (the backbone), with a makhraj-hierarchical decomposition

**Notation.**
- X = audio.
- P_t(q) = softmax posterior at frame t (T × V from `CTCForcedAligner.emissions`).
- y = (y_1…y_N) = canonical **unit** sequence. Each unit maps to a token *string* τ(y_i) via
  `phonetic_tokens`.
- S_i = substitution set for unit i: letter-units q with their token strings τ(q), plus deletion ε.
- y^{(i←q)} = canonical sequence with unit i replaced by q.

**(a) Frame-posterior GOP (Witt & Young 2000; DNN form: Hu et al. 2015).** Let 𝒯_i be the frames
owned by unit i. Weight them by w_t = 1 − P_t(∅) (∅ = CTC blank, index `self.blank`). This fixes
CTC peakiness: blank frames carry no evidence. Renormalise p̃_t(q) = P_t(q)/(1−P_t(∅)). Then

```
LPP_i      = Σ_t w_t log p̃_t(y_i) / Σ_t w_t                       (log posterior probability)
LPR_i(q)   = Σ_t w_t [log p̃_t(y_i) − log p̃_t(q)] / Σ_t w_t         (log posterior ratio vs competitor q)
GOP_post_i = LPP_i − max_q Σ_t w_t log p̃_t(q)/Σ_t w_t
```

This only works for single-token units. It fails for ث ذ خ ش غ (two tokens), which is why (b) is
the primary method.

**(b) Segmentation-free / alignment-free sequence GOP (Cao et al., arXiv:2507.16838, IEEE TASLP; restricted
substitutions: Parikh et al., arXiv:2506.02080).** Use the CTC sequence likelihood
P_CTC(z|X) = Σ_{π∈B⁻¹(z)} Π_t P_t(π_t) (forward algorithm):

```
LR_i(q)  = log P_CTC(τ(y^{(i←q)}) | X) − log P_CTC(τ(y) | X)          q ∈ S_i
P̂_i(q)   = exp LR_i(q) / Σ_{r∈S_i∪{y_i}} exp LR_i(r)                  (unit-i posterior, context clamped)
GOP_SF_i = log P̂_i(y_i)
```

- P̂_i is a proper distribution over *letter hypotheses* at position i. Multi-token letters (ث=`t,h`)
  are handled exactly, because the comparison is between strings.
- The **diagnosis** is argmax_q P̂_i(q).
- **Efficiency.** Run full-utterance forward α and backward β once. For each q, recompute only a
  local window: the frames of unit i ± Δ (Δ≈10 frames, i.e. 200 ms), glued to the prefix α at the
  window's left edge and the suffix β at its right edge. Cost: O(|S_i|·W·|τ|) per unit instead of
  O(|S_i|·T·N).
- **Restrict S_i** to the k nearest letters under the articulatory ground metric (§3.2), plus
  deletion, plus the classical confusable list (§2.2). k≈6 is enough: this is the "RPS" of
  Parikh et al., with phonology supplied by the makharij.

**(c) Makhraj-hierarchical GOP (new; chain rule on a tree).** Let m(q) be the makhraj of letter q and
r(q) its region. By the chain rule on P̂_i:

```
log P̂_i(y) = log P̂_i(R = r(y)) + log P̂_i(M = m(y) | r(y)) + log P̂_i(y | m(y))
             └ region score ┘   └ makhraj-within-region ┘   └ sifah-within-makhraj ┘
where P̂_i(M=m) = Σ_{q: m(q)=m} P̂_i(q)
```

This is the level-appropriate feedback:
- **101 learner:** region term only ("you produced it from the lips, not the throat").
- **Intermediate:** makhraj term (ح vs ه: same region, wrong makhraj).
- **Advanced:** the within-makhraj term is the **sifah error** (ت vs ط share makhraj #12; the loss
  is itbāq).

The three terms sum exactly to GOP_SF, so nothing is double counted.

**(d) Uncertainty measures.**
- Posterior entropy H_i = −Σ_q P̂_i(q) log P̂_i(q), normalised by log|S_i|. High entropy combined
  with high P̂(y) is "indistinct but acceptable". High entropy with low P̂(y) is "unclear
  articulation". Low entropy with low P̂(y) is a "confident substitution" (laḥn jalī).
- Margin: log P̂(y) − max_{q≠y} log P̂(q).
- **Logit-based GOP** (max-logit, logit margin; Parikh et al., arXiv:2506.12067) is less
  overconfident than softmax. Compute both, because the logits are available before `log_softmax`
  in `_forward`.

**(e) Calibration against reference reciters.**
- *H0 (correct) distribution:* GOP_SF_i over all instances of letter y in anchor + peers. Per-letter
  and per-context (vowel, position, shaddah) bands via roadmap A1/A4 conformal LOO.
- *H1 (substitution) distribution without learner data (counterfactual text-swap, new):* feed the
  **reference audio** of a word containing letter b, but score it against a text where b is replaced
  by a. That is exactly "a produced as b" in real, expert-quality acoustics, and it can be generated
  for every confusable pair from Husary alone. Caveat: lexical context no longer matches, and the
  text-prior of the Quran model will fight the swap (useful: it *measures* the linguistic bias).
- *Temperature scaling* (Guo et al., arXiv:1706.04599) of the logits, fit on H0 ∪ swap-H1 per model.
- Decision: PASS if GOP ≥ conformal quantile q_{α}(H0); FAIL-with-diagnosis if P̂(q*) for some
  q* ≠ y exceeds its H1-calibrated threshold.

**Plug-in.** Add a new `app/makharij/gop.py` that consumes `CTCForcedAligner.emissions()`,
`targets()` and `phonetic_tokens()`. Emit one `RuleInstance` of type `MAKHRAJ` per pronounced
consonant unit, created in `TajweedParser._detect_rules`. Calibration key:
`rule_key("makhraj", detail=<context>, letter=<char>)`. Keep `spans_from_path.confidence` as the
alignment-reliability gate (`Calibration.unreliable`).

**Julia/Octave/GPU.** The emissions are a T×38 matrix, so export them to `.npy`/JSON once (GPU for
the wav2vec2 forward only). Implement CTC forward/backward in log space in Julia (`logsumexp` DP,
~80 lines, ForwardDiff-compatible). Cross-check in Octave with the same DP on 3 utterances, requiring
≤1e−9 agreement against PyTorch `ctc_loss`. Everything after the forward pass is CPU-cheap.

### 3.2 Articulatory ground metric and optimal transport

**Letter feature map φ(ℓ) ∈ ℝ^d:**
- **Makhraj coordinate** κ(ℓ) ∈ [0,1]: an ordinal position along the tract, from lips (0) to
  glottis (1), using the 17-makhraj order.
- **Active-articulator one-hot**: lips, tongue tip, tongue side, tongue middle, tongue back,
  pharynx, glottis.
- **Sifāt vector**:
  - Opposites: jahr/hams; shiddah / tawassuṭ / rikhwah as ordinal 0/½/1; istiʿlāʾ; iṭbāq;
    idhlāq/iṣmāt.
  - Singular sifāt: ṣafīr, qalqalah, līn, inḥirāf, takrīr, tafashshī, istiṭālah, ghunnah.
- Most of these sets already exist in `parser.py`. Only κ and the articulator are new.

**Three candidate ground metrics c(a,b):**
1. *Weighted feature distance* (PanPhon-style; Mortensen et al., COLING 2016):
   c(a,b) = Σ_k w_k |φ_k(a)−φ_k(b)|.
2. *Graph metric:* graph G with nodes = letters and edge weights for same makhraj (small), adjacent
   makhraj (medium) and one-sifah difference (small–medium). Use the **commute-time distance**
   c(a,b) = vol(G)·(e_a−e_b)ᵀ L⁺ (e_a−e_b), with L = D − W. It is a squared-Euclidean embedding and
   more robust to single-edge mistakes than the shortest path.
3. **Tree metric (recommended):** a hierarchy root → region → makhraj → letter with edge lengths
   ℓ_region > ℓ_makhraj > ℓ_letter. The sifāt differences inside a makhraj go into the leaf lengths.

**OT verdict.** Compare μ_i = P̂_i (produced) with ν_i. The target ν_i is either the point mass δ_{y_i}
or, better, the **reference posterior** at the same text position averaged over anchor + peers.
Husary's P̂ is not a delta either, and this cancels model bias position by position.

```
W_c(μ,ν) = min_{π∈Π(μ,ν)} Σ_{a,b} π_ab c(a,b)
ν = δ_y  ⇒  W_c = Σ_a μ_a c(a,y)          (expected articulatory cost; closed form)
tree c   ⇒  W_1 = Σ_{edges e} ℓ_e |μ(Sub_e) − ν(Sub_e)|     (tree-Wasserstein, exact, O(|V|))
```

The tree-W₁ decomposes *additively* into region, makhraj and letter edges. It is the transport
counterpart of the chain-rule GOP in §3.1c and gives an interpretable "where the mass went" report.
With the tree metric, ẓāʾ-for-ḍād costs ℓ_makhraj + ℓ_letter-ish, while bāʾ-for-ḍād costs
2ℓ_region + …, as required.

**Sequence level.** Expected articulatory edit distance under the posterior:
E_{ẑ∼P(·|X)}[Lev_c(ẑ, y)]. Approximate it by the per-position W_c sum plus deletion/insertion costs,
or by a 10-best CTC beam. This gives a continuous per-ayah "articulatory distance" for learner
progress curves.

**Learning the ground metric.** Ground-metric learning (Cuturi & Avis, JMLR 15, 2014,
arXiv:1110.2306) fits the tree edge lengths from (i) teacher severity ratings of error pairs or
(ii) human confusion data. The Wasserstein loss with a ground metric (Frogner et al.,
arXiv:1506.05439) can train a makhraj head so that its errors are "articulatorily smooth".

**Gromov–Wasserstein (research).** Compare the learner's *internal* confusion geometry with the
reference geometry without shared coordinates (Mémoli 2011, FoCM 11:417). This detects systematic
"collapsed" makhraj regions, e.g. an L1 speaker whose ḥalq letters all map onto one point.

**Julia/Octave.** A tree-W₁ is a few lines. General W_c (|S|≤38) is solved exactly as an LP, or with
the existing `Frontier.sinkhorn_cost` and Octave `fr_sinkhorn_div.m` as the cross-check.

### 3.3 Graph-theoretic confusion analysis

1. **Confusion matrix** from soft counts: C_ab = Σ_{i: y_i=a} P̂_i(b). Aggregate over a learner, a
   cohort, or the model-on-reference (the "instrument" matrix).
2. **Shepard similarity** (Shepard 1957, Psychometrika 22:325): s_ab = √(p_ab p_ba / (p_aa p_bb)).
   Build W = S, the normalised Laplacian L_sym = I − D^{−½} W D^{−½}, and run spectral clustering
   (Ng–Jordan–Weiss 2002; von Luxburg arXiv:0711.0189).
   - **Test A (instrument validity):** do the model-on-reference clusters recover the classical
     makharij? Measure the adjusted Rand index / variation of information against the 17-partition.
   - **Test B (scholarly curiosity, falsifiable):** the eigengap of L_sym, i.e. whether the data
     favour 14, 16 or 17 groups (al-Farrāʾ / Sībawayh / al-Khalīl). Most informative for ل ن ر.
3. **Graph-signal diagnosis for sparse learner data.** Let g ∈ ℝ^{28} be per-letter mean GOP with
   counts n_ℓ. Use the Tikhonov-on-graph estimate x̂ = argmin Σ n_ℓ(x_ℓ − g_ℓ)² + λ xᵀ L_art x, i.e.
   x̂ = (N + λL_art)^{−1} N g, where L_art is the Laplacian of the articulatory graph from §3.2.
   - It shares strength between articulatorily adjacent letters, so a learner with 3 ع tokens
     borrows from ح.
   - The high graph-frequency energy x̂ᵀ L_art x̂ separates an "isolated letter problem" from a
     "whole-region weakness" (low-frequency deficit on the ḥalq subgraph).
4. Markov view: row-normalised C as a transition matrix. Absorbing sets identify one-way mergers,
   e.g. ظ→ز and ذ→ز but never the reverse (dialect transfer).

All of this is Julia `LinearAlgebra` (eigen on 28–38 nodes); cross-check the eigenvectors in Octave.

### 3.4 Information-theoretic measures

- **Letter channel** Y→Ŷ with transition P(Ŷ|Y) = row-normalised C and input p_Y = Quranic letter
  frequencies. Transmitted information I(Y;Ŷ).
- **Feature transmission (Miller & Nicely 1955, JASA 27:338).** For each feature F ∈ {region,
  makhraj, jahr, shiddah, istiʿlāʾ, iṭbāq, ṣafīr, ghunnah}: T_F = I(F(Y);F(Ŷ)) / H(F(Y)). This is the
  learner's **feature-transmission profile** (e.g. jahr 0.97, iṭbāq 0.62, ḥalq-makhraj 0.55), which
  is exactly the pedagogical diagnosis.
- **Channel capacity** C* = max_p I(Y;Ŷ) via Blahut–Arimoto (Blahut 1972, IEEE-IT 18:460;
  Arimoto 1972, IEEE-IT 18:14). Computed for the **model-on-reference** channel, C* is the *ceiling*
  of any verifier built on these posteriors. **Fano:** P_e ≥ (H(Y|Ŷ) − 1)/log(|Y|−1) lower-bounds
  the achievable error per confusable group.
  - Any group whose model-on-reference MI is low (expected: ث/ت+h, ذ/د+h; possibly ض/ظ, because the
    Quran model rarely heard ظ-for-ض) **must not be claimed as verified** from posteriors. Route it
    to the acoustic verifiers (§3.5). This gives a principled go/no-go per pair.
- **Divergences.** Per position, D_KL(ν_i‖μ_i) and Jensen–Shannon (bounded, symmetric). Use JS for
  display and KL for likelihood-ratio logic. With few tokens, use a bias-corrected entropy (Miller–
  Madow; NSB, Nemenman et al. 2002).
- **Cue selection.** Per confusable pair, pick acoustic cues by the **Chernoff information** (the
  Bayes error exponent) between the class-conditional Gaussians fitted on reference reciters. This
  makes "use VOT for ط/ت in this reciter population, F2 for ص/س" a computed choice.

### 3.5 Knowledge-based acoustic cue verifiers (pairwise likelihood-ratio tests)

For each confusable pair (a,b) of §2.2, take the cue vector z from the aligned consonant window
(`EvalContext.frication_window`, `silence_runs`, `voicing_fraction`, `AcousticContext` formants).

**Cue definitions:**
- **Spectral moments** (Forrest et al. 1988, JASA 84:115; Jongman et al. 2000, JASA 108:1252) on a
  20–40 ms Hamming window at fricative mid or burst: M1 = Σf P(f)/ΣP, M2 = variance, M3 = skewness,
  M4 = kurtosis. Use a multitaper spectrum for stability.
- **Relative amplitude:** fricative-band energy minus vowel energy in the F3 (sibilants) or F5
  (non-sibilants) region.
- **VOT:** burst onset (energy jump + HF flatness) → periodicity onset (HNR / autocorrelation).
- **Burst spectrum:** first 10–20 ms after release, peak frequency normalised by the following
  vowel's F2 (for ق/ك).
- **Locus equation** (Sussman et al. 1991, JASA 90:1309): per letter, fit F2_onset = k·F2_mid + c on
  the reference. The learner's verdict is the Mahalanobis residual to the reference line.
- **Glottal/pharyngeal:** H1−H2 and H1−A3, jitter/creak fraction, HNR, noise-LPC first pole.
- **Nasal:** existing NER plus A1−P0 (Chen 1997).
- **Closure test:** silence run ≥ 25 ms with a burst transient (stop) vs spectral-flux continuity
  (fricative).

**Model.** Class-conditional densities come from the **reference reciters' correct productions of
both a and b**. Both letters occur thousands of times in any khatm, so **negatives come for free
from the anchor**:

```
LLR_ab(z) = log N_t(z; m_a, Σ_a, ν) − log N_t(z; m_b, Σ_b, ν)     (Student-t, whitened; roadmap A1/A5)
```

Speaker normalisation: express z relative to the learner's own vowel space (Lobanov z-scoring of
formants; M1 relative to that speaker's س). Thresholds: conformal LOO over peers (A4). Report LLR
with the cue that contributed most.

**Plug-in.** Add `app/makharij/cues.py`, with pair tests registered by letter in `scoring.py`.
**Octave** reproduces the moments/LPC/VOT on the same windows (extend `qaari_features.m`:
`span_features` already computes band energies). **Julia** fits the densities (`Distributions`,
`Frontier.ogk` for robust covariance).

**Area-function cross-check (research-grade, Octave-native).** Take LPC reflection coefficients on
the adjacent vowel (existing `octave/lpc.m` + `levinson`). The Wakita (1973) lossless-tube area
function is A_k/A_{k+1} = (1−r_k)/(1+r_k). It gives a crude **pharyngeal-constriction index** for
ḥalq letters and iṭbāq: the ratio of back-cavity to front-cavity area. Use it as a *secondary* cue
only, because it is fragile to glottal source and nasal coupling.

### 3.6 Articulatory inversion (acoustic → articulator)

- **Theory bridge (Articulatory Phonology; Browman & Goldstein 1992):** makhraj ≅ **constriction
  location (CL)**; shiddah/tawassuṭ/rikhwah ≅ **constriction degree (CD)**. Tract variables LA/LP
  (lips), TTCL/TTCD (tongue tip), TBCL/TBCD (tongue body) map to makharij 12–16 and 5–7 directly.
- **SPARC** (Cho et al., arXiv:2406.12998; pip `speech-articulatory-coding`) gives 12 EMA channels
  (UL, LL, LI, TT, TB, TD × x/y) + loudness + pitch at 50 Hz, from WavLM features. SSL inversion
  transfers across languages with an affine map (Cho et al., arXiv:2310.10788, ICASSP 2024).
  Tract-variable inversion is being extended to unseen languages (Tabatabaee et al.,
  arXiv:2607.05060, Interspeech 2026; code availability not stated).
- **Blind spot:** EMA/TV channels see nothing behind the tongue dorsum. There is **no tongue root,
  epiglottis or larynx**, so inversion cannot verify ḥalq makharij (#2–4) or the pharyngeal part of
  iṭbāq. Use it for **lisān + shafatān** only.
- **Verification model.** At each consonant's acoustic landmark, take the inferred articulator
  vector a ∈ ℝ^{12}. Speaker-normalise with a Procrustes/affine map fitted on the learner's own
  vowels to the anchor's vowels. Score the Mahalanobis distance to the reference cloud for the target
  letter, and the LLR vs confusable letters (as in §3.5 but in articulator space).
- **Face-validity experiment (falsifiable, 1 day).** On Husary, compute TT_x (tongue-tip
  advancement) at the landmarks of ث ذ ظ / ت د ط / س ص ز / ن / ل / ر, and TD_x for ق vs ك. Test the
  Spearman correlation between inferred constriction location and the classical makhraj ordinal. If
  ρ < 0.6 on the anchor, drop inversion from the verdict path.
- **Analysis-by-synthesis (speculative):** fit VocalTractLab/Maeda parameters by `Optim.jl` so that
  synthetic formants match observed ones. The problem is ill-posed and needs strong priors, so it is
  research only.
- **GPU:** the SPARC/WavLM forward pass (batch the reference corpus once and cache articulator
  traces; Modal is available but not used here per instructions). Everything downstream is CPU.

### 3.7 Neural approaches and the Arabic/Quranic MDD literature

| Resource | What it is | Use for us |
|---|---|---|
| **QuranMB.v1** (El Kheir et al., arXiv:2506.07722, Interspeech 2025) | 98 verses read by 18 native speakers (14 F, 4 M), ~2.2 h, **deliberate** mispronunciations. 68-phoneme MSA set, gemination as doubled symbols. Training: CMV-Ar 82 h + 52 h TTS (26 h with simulated errors, confusion pairs incl. ت/ط, س/ص, غ/خ). Best baseline F1 ≈ 0.30 (mHuBERT). | External **H1 test set** for §3.1 GOP and §3.5 cues. MSA reading, *not* tajweed, so use it for makhraj only. |
| **Iqra'Eval shared task** (ArabicNLP 2025, ACL Anthology 2025.arabicnlp-sharedtasks.61) and Iqra'Eval 2 (Interspeech 2026) | Open MDD benchmark (localisation + diagnosis). Top IS26 systems reach F1 ≈ 0.72 (Zhang et al. arXiv:2606.24086: XLS-R + dilated TCN, 2-stage native→synthetic→real, ensembles; Geng et al. arXiv:2604.22133: prompt-free CROTTC, 71.70%) | Upper reference for achievable phoneme-level MDD. Evidence that **decoupling from canonical text priors** matters. |
| **Muaalem / Quran Phonetic Script** (Abdelfattah et al., arXiv:2509.00094; HF `obadx/muaalem-model-v3_2`, w2v-BERT 2.0, MIT; data `obadx/muaalem-annotated-v3`, 848 h, 22 reciters, CC BY 4.0) | Multi-level CTC: 43-symbol phoneme level + 10 sifāt levels (hams/jahr, shiddah/rikhwah/tawassuṭ, tafkhīm/tarqīq, iṭbāq, ṣafīr, qalqalah, takrīr, tafashshī, istiṭālah, ghunnah). PER 0.21% on test; 75.8% tajweed F1 on learner bench `qdat_bench`. Makhraj **not** a separate head. | **Strongest drop-in second judge.** Its phoneme inventory has single symbols for ث ذ خ ش غ, which removes our digraph problem. Its sifāt heads give the within-makhraj term of §3.1c directly. Makhraj posterior = marginalise its phoneme posterior over §2.1 groups. |
| Multilingual phoneme recogniser `facebook/wav2vec2-xlsr-53-espeak-cv-ft` (Xu, Baevski, Auli, arXiv:2109.11680) | IPA posteriors, not Quran-trained | **Low-text-prior judge** for the linguistic-bias ensemble (disagreement with the Quran model is itself a flag). |
| Speech-attribute MDD (Shahin, Epps, Ahmed, arXiv:2311.07037; Speech Commun. 2025) | wav2vec2 + multi-label CTC (SCTC-SB) over manner/place/voicing attributes; lower FAR/FRR than phoneme MDD | Template for a **makhraj-attribute head**: 17 makhraj labels + sifāt as a multi-label CTC. |
| Isolated-letter work (Zaatiti et al., arXiv:2508.19587; Kucukmanisa et al., arXiv:2511.17477; Akhtar et al. Appl. Sci. 12:238, 2022; MFCC-SVM makhraj recognition) | Isolated letters: wav2vec2 35% → 65% with a head; fragile to perturbation | 101-level "say the letter" drills. Warning: isolated letters lack coarticulatory cues, so use the §3.5 cues. |
| Tajweed-rule MDD (arXiv:2305.06429) | Rule classifiers | Not makhraj; related work only. |

**Training-free neural verifier (build first).**
- Take hidden states h_t^{(ℓ)} from layer ℓ of the existing wav2vec2 (set `output_hidden_states=True`
  in `_forward`).
- Pool each letter over its span: u_i = mean_t h_t.
- Build **per-letter prototypes** from the reference reciters: robust mean and OGK covariance per
  letter and per coarse context.
- Score: s_i(q) = −½ Mahalanobis(u_i; m_q, Σ_q) (shared Σ, i.e. LDA, is stabler), giving a neural
  LLR vs confusable prototypes.
- Choose the layer by a linear probe for makhraj labels on reference data, cross-validated by
  reciter. SSL mid layers carry place/manner information (layer-wise analyses, e.g. Pasad et al.
  arXiv:2107.04734).
- This needs no learner data and bypasses the output-layer text prior.

**Trained verifier (later).** Fine-tune a multi-head CTC (letter + 17-makhraj + sifāt), in the style
of muaalem / SCTC-SB, on muaalem-annotated-v3. Add error augmentation by **text-swap TTS** (the
QuranMB recipe) restricted to §2.2 pairs. An optional Wasserstein loss with the §3.2 ground metric
makes errors articulatorily graded. GPU required (a few A100-hours).

---

## 4. Validation protocol (all models)

1. **Instrument validity on reference (H0).** On anchor + peers, per letter: the false-reject rate at
   the chosen threshold, from conformal LOO over peers (roadmap A4). Target ≤5% per letter-context
   cell. Imams serve as the contrast set, and should show a *higher* but still low reject rate.
2. **Detection power (H1).**
   - (a) Counterfactual text-swap on reference audio (§3.1e).
   - (b) QuranMB.v1 real deliberate errors.
   - (c) muaalem `qdat_bench` learner errors.
   - (d) An in-house micro-corpus: one qāriʾ records ~200 minimal-pair words **deliberately wrong**
     (ض→د/ظ, ح→ه, ع→ء, ق→ك, ث→س/ت, ذ→ز/د, ص→س, ط→ت), ~1 h. This is the single most valuable data
     asset.
   - Report ROC-AUC and precision/recall per pair, and diagnosis accuracy (argmax correct).
3. **Information ceiling.** Blahut–Arimoto capacity and Fano bound per confusable group on the
   model-on-reference channel. Groups below the bound are marked *unverifiable-by-posterior* and
   routed to §3.5 cues.
4. **Robustness.** Channel/codec (phone mic, MP3 64 k), ±20% tempo, reverberation. The GOP drift on
   reference audio must stay inside the conformal band.
5. **Agreement with teachers.** On ≥300 learner letters rated by two ijazah holders, report
   Cohen's κ between the teachers and the system's region/makhraj/sifah decomposition.
6. **Ablation of the text prior.** Compare the Quran model vs espeak-IPA vs muaalem on H1 recall.
   The gap quantifies linguistic bias.

---

## 5. Prioritised build plan

| P | Item | What | Where | Effort | GPU |
|---|---|---|---|---|---|
| **1** | Makhraj/sifāt table + graph | §2.1 as data (κ, articulator, region, sifāt vectors, adjacency, confusable list); tree metric; commute-time metric | `app/makharij/tables.py`; Julia `src/Makharij.jl` | 1 d | – |
| **1** | Segmentation-free GOP + chain-rule decomposition | §3.1b–d: log-space CTC α/β, local-window substitution LRs, P̂_i, region/makhraj/sifah terms, entropy, logit margin | `app/makharij/gop.py` (numpy); Julia port for calibration; Octave CTC check | 3–4 d | forward pass only |
| **1** | Counterfactual text-swap H1 + conformal thresholds | §3.1e using roadmap A4 `conformal_threshold` | `research_agency_lab/.../julia` + calibration JSON | 1–2 d | – |
| **2** | Pairwise acoustic cue verifiers | §3.5 for the 8 priority pairs (ض/ظ/د, ث/س/ت, ذ/ز/ظ, ح/ه, ع/ء, ق/ك, ص/س, ط/ت) + idghām nāqiṣ tests (بسطت، نخلقكم) | `app/makharij/cues.py`; extend `octave/qaari_features.m` | 4–6 d | – |
| **2** | Tree-Wasserstein articulatory distance + ayah-level expected articulatory edit distance | §3.2 | `gop.py` output → `scoring.py` summary | 1 d | – |
| **2** | Information audit | §3.4 channel MI, capacity, Fano, feature-transmission profile; publishes the "verifiable pairs" list | Julia (`LinearAlgebra`, Blahut–Arimoto ~40 lines) | 1–2 d | – |
| **2** | Training-free SSL prototype verifier | §3.7 layer probe + LDA prototypes from reference reciters | `app/makharij/prototypes.py` | 2–3 d | cache hidden states once |
| **3** | Second judges | muaalem-v3.2 (sifāt heads, single-symbol ث ذ خ ش غ) and espeak-IPA; disagreement flag | `app/aligner.py` new emissions backend | 2–3 d | inference only |
| **3** | Minimal-pair error micro-corpus | 1 h deliberate errors by a qāriʾ; annotation sheet | data | 2 d (human) | – |
| **3** | Graph confusion analysis + graph-Tikhonov learner profile | §3.3 spectral clustering, eigengap test, x̂ = (N+λL)^{−1}Ng | Julia | 1–2 d | – |
| **4** | Articulatory inversion (SPARC) for lisān/shafatān + face-validity test | §3.6 | research only first | 2–3 d | yes (cache) |
| **5** | Trained multi-head makhraj CTC with Wasserstein loss + text-swap TTS augmentation | §3.7 | new training repo | 2–3 wk | yes |
| **5** | Area-function (Wakita) pharyngeal index; Gromov–Wasserstein confusion geometry; analysis-by-synthesis | §3.2, §3.5, §3.6 | Octave/Julia research | open | – |

**Critical path to first shippable makhraj verdict:** P1 rows (≈6 days) → cue verifiers for the 8
pairs (≈1 week) → information audit decides, per pair, whether the verdict comes from GOP, cues, or
both (fusion = sum of calibrated LLRs, conformal threshold).

---

## 6. References (primary)

- Ibn al-Jazarī, *al-Nashr fī al-Qirāʾāt al-ʿAshr*; *al-Muqaddimah al-Jazariyyah*. Sībawayh, *al-Kitāb* (bāb al-idghām). al-Khalīl, *Kitāb al-ʿAyn*.
- Witt & Young (2000) Speech Commun. 30(2–3):95–108, doi:10.1016/S0167-6393(99)00044-8.
- Hu, Qian, Soong, Wang (2015) Speech Commun. 67:154–166 (DNN-GOP).
- Cao, Fan, Svendsen, Salvi, *Segmentation-free GOP*, arXiv:2507.16838 (IEEE TASLP); Cao et al. Interspeech 2024 (CTC GOP framework).
- Parikh et al., *Enhancing GOP in CTC-based MDD with phonological knowledge*, arXiv:2506.02080 (Interspeech 2025).
- Parikh et al., *Evaluating logit-based GOP scores*, arXiv:2506.12067 (Interspeech 2025).
- El Kheir et al., *Towards a unified benchmark for Arabic pronunciation assessment: Quranic recitation as case study* (QuranMB), arXiv:2506.07722.
- El Kheir et al., *Iqra'Eval shared task*, ArabicNLP 2025, aclanthology.org/2025.arabicnlp-sharedtasks.61.
- Zhang et al., *Fusion-aware two-stage MDD for low-resource MSA*, arXiv:2606.24086.
- Geng et al., *Beyond acoustic sparsity and linguistic bias: prompt-free MDD*, arXiv:2604.22133.
- Abdelfattah, Khalil, Abbas, *Automatic pronunciation error detection and correction of the Holy Quran's learners* (QPS, muaalem), arXiv:2509.00094.
- Shahin, Epps, Ahmed, *Phonological-level wav2vec2-based MDD*, arXiv:2311.07037 (Speech Commun. 2025).
- Xu, Baevski, Auli, *Simple and effective zero-shot cross-lingual phoneme recognition*, arXiv:2109.11680.
- Zaatiti et al., arXiv:2508.19587; Kucukmanisa et al., arXiv:2511.17477.
- Cho et al., *Coding speech through vocal tract kinematics* (SPARC), arXiv:2406.12998; Cho et al., *SSL models infer universal articulatory kinematics*, arXiv:2310.10788; Tabatabaee et al., *Towards language-agnostic speech inversion*, arXiv:2607.05060.
- Browman & Goldstein (1992) Phonetica 49:155–180 (Articulatory Phonology).
- Pasad, Chou, Livescu, *Layer-wise analysis of a SSL speech model*, arXiv:2107.04734.
- Frogner et al., *Learning with a Wasserstein loss*, arXiv:1506.05439; Cuturi & Avis, *Ground metric learning*, arXiv:1110.2306; Le et al., *Tree-sliced Wasserstein*, arXiv:1902.00342; Mémoli (2011) FoCM 11:417.
- Mortensen et al., *PanPhon*, COLING 2016, aclanthology.org/C16-1328.
- Ng, Jordan, Weiss (2002) NeurIPS; von Luxburg, *A tutorial on spectral clustering*, arXiv:0711.0189; Shepard (1957) Psychometrika 22:325.
- Miller & Nicely (1955) JASA 27:338; Blahut (1972) IEEE-IT 18:460; Arimoto (1972) IEEE-IT 18:14; Nemenman, Shafee, Bialek (2002) NeurIPS (NSB entropy).
- Guo et al., *On calibration of modern neural networks*, arXiv:1706.04599.
- Forrest et al. (1988) JASA 84:115; Jongman, Wayland, Wong (2000) JASA 108:1252; Sussman, McCaffrey, Matthews (1991) JASA 90:1309; Stevens & Blumstein (1978) JASA 64:1358; Kurowski & Blumstein (1987) JASA 81:1917; Chen (1997) JASA 102:2360; Wakita (1973) IEEE Trans. Audio Electroacoust. AU-21:417.
- Jongman, Herd, Al-Masri, Sereno, Combest (2011) J. Phonetics 39:85, doi:10.1016/j.wocn.2010.11.007 (emphasis: F1↑ F2↓ F3↑).
- Heselwood (2007) *The "tight approximant" variant of Arabic ʿayn*, JIPA 37:1, doi:10.1017/S0025100306002799.
- Kulikov et al. (2024) *Phonetic realization of the emphasis contrast in voiceless stops… F2 and VOT*, Arabic Linguistics, doi:10.1075/arli.00001.kul.
- Al-Khairy (2005) *Acoustic characteristics of Arabic fricatives*, PhD thesis, Univ. Florida.
- Al-Tamimi, Alzoubi, Tarawnah (2009) (videofluoroscopy of emphatics).
