# 06 — Information Theory, Statistical Learning & Neural Networks for qaari-eval

Deep-research report, 2026-09-23. Scope: the whole engine, from a 101-level learner to ijazah
level. This report builds on `frontier_math_roadmap.md` (A1–A6, B, C) and does not repeat it. Where
it relies on the roadmap it says so by item (e.g. "roadmap A5").
Status: research proposal. No code in the engine was changed.

---

## 0. Summary

1. **The engine cannot detect lahn jali yet, and it gives a perverse reward to bad recitation.**
   - `CTCForcedAligner` runs Viterbi only through the *canonical* token lattice (`app/aligner.py:102`).
     A substituted letter (ض→د, ص→س, ح→ه, ق→ك) is still forced onto the canonical token.
   - The only trace of the error is a low `align_conf`. `Calibration.unreliable()` then turns the
     instance into **SKIPPED**, and `summarize_diagnostics` drops SKIPPED from the index.
   - Result: *the more clear errors a reciter makes, the more of their recitation is excluded.* That
     can raise the score.
   - Fixing this is the highest-priority item in this report (§6.1, §4.3).
2. **Measured on our own rows (491 ayah-rows, 5 reciters, 48 rule keys, 9 679 judged instances,
   raw verdicts).** A Rasch-style logit `P(PASS) = σ(θ_reciter − b_key)` explains
   - **32 % of deviance with rule-key difficulty alone**, and
   - **0.4 % with reciter ability alone** (McFadden R² 0.321 vs 0.004).
   - Key difficulty SD = 2.08 logits; reciter SD = 0.20.
   - So today's raw verdicts mostly measure *how hard each ruler is*, not *who recites well*. This is
     exactly what item response theory separates (§4.2).
   - Even so, the fitted abilities already order the studio reciters (Ayyoub +0.30, Tablawi +0.07)
     above the imams Matroud (−0.30) and Budair (−0.13). That is directional evidence only: 5 reciters.
3. **12 of 48 rule keys are near-degenerate** (item information p(1−p) < 0.05). `madd_iwad`,
   `hamzat_wasl`, `safir` and `jawaz_wajhayn` have **zero verdict entropy**: the same status for
   every reciter and every instance. They carry no information about the reciter. They are ruler
   defects or constant pass-throughs, and an information-theoretic health check flags them
   automatically (§3.4).
4. **Aggregation should be a measurement model, not a weighted mean.** The recommended stack is:
   - a calibrated per-instance likelihood (reference model), then
   - multidimensional IRT (abilities θ_jali, θ_khafi,timing, θ_khafi,sifaat), then
   - a Bayes decision with an explicit **lahn jali / khafi loss matrix**, then
   - ordinal regression onto the grading scale.
   - A log predictive score is additive in nats. This removes the heterogeneous-units problem and
     the hand-set category weights.
5. **Small-N limit.** With P = 9 reference peers, reciter-level conformal can only certify coverage
   in steps of 1/(P+1) = 10 %. **α = 0.05 at reciter level needs ≥ 19 exchangeable references.**
   Use two-layer hierarchical conformal (Dunn–Wasserman–Ramdas) at instance level, plus conformal
   risk control on the *expert false-FAIL rate* (§5).
6. **Neural: there is a strong ready-made model.** `obadx/muaalem-model-v3_2` is w2v-BERT-2.0 with
   an 11-level CTC head: phonemes plus 10 sifaat. It uses a Quran Phonetic Script that encodes madd
   length, ghunnah, ikhfa and qalqalah (arXiv:2509.00094; MIT licence; about 13.8 k downloads).
   - It gives exactly the jali (phoneme level) vs khafi (sifa level) split the scoring lacks.
   - Pair it with segmentation-free CTC GOP (arXiv:2507.16838) and substitution-aware GOP
     (arXiv:2506.02080).
   - Evaluate on `obadx/qdat_bench` (159 learner clips with per-rule madd/ghunnah/qalqalah labels)
     and the Iqra'Eval QuranMB sets.
   - **Caveat:** it was trained on 22 world-class Hafs reciters, which probably include our
     reference set. It must not be validated on Husary or the peers.
7. **Validation data we lack can be made cheaply.** Controlled **counterfactual perturbations** of
   expert audio give *labeled* khafi errors with known magnitude:
   - WSOLA-shortened or lengthened madd cores,
   - truncated nasal murmur,
   - qalqalah burst removal,
   - letter splicing between peers' aligned units (jali).
   This gives the missing half of validation: detection power. Today we only measure the false-FAIL
   rate on experts.

---

## 1. Diagnosis of the current engine through this lens

| # | Where | Mechanism today | Problem (IT / statistics / ML view) |
|---|---|---|---|
| D1 | `aligner.py` `ctc_forced_align` | Viterbi forced to the canonical sequence | Hypothesis-constrained decoding. There is no competing hypothesis, so no likelihood ratio and no jali detection. |
| D2 | `calibration.py` `unreliable()` → SKIPPED; `scoring.py` excludes SKIPPED | Low CTC posterior means "don't judge" | The data are **missing not at random**: missingness is caused by the error. The score is biased upward for bad recitations. |
| D3 | `Calibration.judge` | `z = max_m z_m` over the metrics | The max of dependent statistics has no stated error rate. The PASS/WARN/FAIL cut-offs 2 and 3 are Gaussian lore (roadmap A6). |
| D4 | `verdict()` | Piecewise-linear map z → score in [0, 1] | The score is not a probability, not a utility and not a proper score. Averages of it have no semantics. |
| D5 | `summarize_diagnostics` | Category-weighted mean of instance scores | A compensatory aggregate: many easy PASSes buy back a jali error. It ignores item difficulty (§0.2) and treats correlated instances within an ayah as independent. |
| D6 | Hand weights `CATEGORY_WEIGHTS` | Fixed, or L2-pulled optimisation | No decision-theoretic meaning. The jali/khafi hierarchy is not encoded. |
| D7 | CTC posteriors | Softmax of `TBOGamer22/wav2vec2-quran-phonetics` (wav2vec2-base, trained on word-level Quran-MD clips) | Domain shift (isolated words vs continuous tarteel). CTC peaky/blank-dominated posteriors are mis-calibrated as confidences (§5.3). |
| D8 | Fingerprint | ECAPA 192-d plus a 32-d Tajweed vector | The timbre embedding is trained for speaker ID on non-Quranic data. With 16 reciters, contrastive fine-tuning would overfit unless a larger pool is used (Tadabur, §6.7). |

---

## 2. The target architecture in one picture

```
audio ──► SSL encoder (w2v-BERT 2.0 / muaalem) ──► frame posteriors: phonemes (43) + 10 sifa tiers
   │                                                      │
   │                        ┌─────────────────────────────┴───────────────┐
   │                        ▼                                             ▼
   │     L0  jali evidence: GOP-SF / LPR vs confusion set      khafi evidence: sifa-tier GOP
   │                        │                                             │
   ├─► acoustic metrics (durations, formants, nasal, bursts) ──► L1 reference model per rule key
   │                                                         (Bayesian predictive, roadmap A1/A5)
   │                                                          ⇒ per-instance log-likelihood ratios
   ▼                                                                      │
 alignment confidence ──► measurement-error / MNAR model ─────────────────┤
                                                                          ▼
                              L2  multidimensional IRT: θ_r = (θ_jali, θ_timing, θ_nasal, θ_sifaat)
                                  items = rule instances, difficulty b_i from item covariates
                                                                          ▼
                              L3  Bayes decision (loss matrix) per instance  +  ordinal grade G_r
                                  + conformal risk control on expert false-FAIL rate
```

The principle: **every layer passes calibrated probabilities up.** Scores are derived only at the
end, as posterior expected utilities.

---

## 3. Information theory

### 3.1 Recitation as a noisy channel; the letter channel and its capacity

Define the channels:
- X ∈ 𝒳 is the canonical letter-unit sequence, with context (the parser output).
- The reciter is a channel X → A (audio).
- The recogniser is a channel A → Ŷ: an *unconstrained* phone decode, or the arg-max of the
  muaalem phoneme tier.
- The observable end-to-end channel is X → Ŷ, with confusion matrix `P_r(ŷ | x)`.

**Quantities**
- Per-letter equivocation: `H_r(X | Ŷ) = − Σ_{x,ŷ} p(x) P_r(ŷ|x) log P(x|ŷ)`.
- Transmitted information: `I_r(X; Ŷ) = H(X) − H_r(X | Ŷ)`.
- **Reciter-attributable loss.** The recogniser has its own confusion. Measure the same quantity on
  the held-out reference reciters and define
  `ΔH_r = H_r(X|Ŷ) − H_ref(X|Ŷ)` (bits per letter),
  or better the KL form per letter class:
  `ΔD_r(x) = D_KL( P_r(·|x) ‖ P_ref(·|x) )`.
  - This is the **jali index**. It is in bits and it is interpretable: for example, "your ض carries
    0.4 bit less information than the reference; 60 % of the loss is confusion with د".
  - Report it per makhraj group (halq, lisan, shafatan, jawf, khayshum).
- **Channel capacity** `C = max_{p(x)} I(X;Ŷ)` (Blahut–Arimoto) of the *reference* channel bounds
  what any detector can resolve. Letters with low reference capacity (e.g. ظ/ذ in fast hadr) are
  where the recogniser, not the reciter, limits us. Down-weight jali claims there automatically.
- **Plug-in:** a new module `app/information/channel.py` consuming muaalem phoneme posteriors.
  Estimate `P(ŷ|x)` from *soft* counts `Σ_t γ_t(x) p_t(ŷ)`, where γ is the forced-alignment
  occupancy. Use Dirichlet smoothing (Jeffreys α = ½) because the counts are small.
- **Data:** the reference = our expert rows. Errors are needed only for power, from §6.6
  perturbations and `qdat_bench`.
- **Julia:** trivial, in `Statistics`/`LinearAlgebra`. Blahut–Arimoto is about 15 lines.
- Sources: Shannon (1948); Blahut, IEEE TIT 18(4):460 (1972) doi:10.1109/TIT.1972.1054855;
  Arimoto, IEEE TIT 18(1):14 (1972) doi:10.1109/TIT.1972.1054753; Cover & Thomas, *Elements of
  Information Theory* (2006).

### 3.2 Surprisal of a reciter under the reference model: the common currency

Let `p_ref(y_i | c_i)` be the **posterior-predictive density** of instance i's metric vector y_i
given its context c_i (rule key, letter, neighbours, local tempo), under the reference model
(roadmap A5 hierarchical model, or A1 Gaussian).

```
s_i  = − log p_ref(y_i | c_i)                         (nats; the instance surprisal)
e_i  = s_i − E_ref[s | c_i]                            (excess surprisal; E_ref from LOO on peers)
S_r  = (1/N_r) Σ_i e_i                                 (mean excess surprisal, nats/instance)
```

Why this matters:
- The log score is the **only local proper scoring rule** (Bernardo 1979). It is additive across
  independent evidence.
- Nats are the same unit for a madd duration, a formant difference and a nasal-energy contrast. The
  heterogeneous-units problem that forces hand category weights (D6) **disappears**.
- A category's weight becomes its information content, not an opinion.

**Dependence correction.** Instances in one ayah share tempo and alignment errors. Summing
surprisals then over-counts evidence. Two remedies:
- (a) model the ayah jointly (multivariate predictive, or a random ayah effect, roadmap A5); or
- (b) temper by the effective sample size: `S_r^adj = (N_eff/N) S_r`. Estimate
  `N_eff = N / (1 + (m−1)ρ)` from the intra-ayah correlation ρ of the e_i (the design-effect
  formula, Kish 1965).

**Surprisal ↔ GOP.** Classic GOP (Witt & Young 2000) is exactly a normalised surprisal of the
canonical phone relative to the best competitor. So §3.2 (continuous acoustic metrics, khafi) and
§6.3 (GOP, jali) are two faces of one quantity. Both can be summed in nats if both are calibrated.

- **Plug-in:** `Calibration.judge` returns `s_i` and `e_i` next to z. The Julia calibrator stores
  the per-key predictive (a Student-t with location, scale and ν is enough to start).
- Sources: Bernardo, Ann. Stat. 7(3):686 (1979) doi:10.1214/aos/1176344689; Gneiting & Raftery,
  JASA 102:359 (2007) doi:10.1198/016214506000001437; Witt & Young, Speech Commun. 30:95 (2000)
  doi:10.1016/S0167-6393(99)00044-8.

### 3.3 KL / Jensen–Shannon verdicts: where they fit, and where they don't

**Per instance:** use the likelihood ratio or p-value (§3.2, §4.3), not a divergence. A single
point has no distribution.

**Per reciter × rule key** (distributional feedback across many instances):
- `JS(P_r ‖ P_ref) = ½ KL(P_r‖M) + ½ KL(P_ref‖M)` with `M = ½(P_r + P_ref)`, in bits.
- It is bounded in [0, 1] bit. √JS is a metric. It stays finite when the supports barely overlap,
  which KL does not. That matters because an over-consistent reciter has a very narrow P_r, and KL
  then diverges.
- Estimate it with the k-NN divergence estimator (Wang, Kulkarni & Verdú 2009). For a quick UI
  number, use a Gaussian closed form.
- **Division of labour with the roadmap:**
  - **W₂/Bures (roadmap A1)** is best for *feedback*. It splits location from shape: "you hold madd
    too long" vs "you are inconsistent".
  - **JS** is best as a *bounded same-distribution test statistic*, with a permutation null over
    ayahs.
  - Use **KL(P_ref‖P_r)** only inside the Bayesian model; it is the expected log-LR.
- Sources: Lin, IEEE TIT 37(1):145 (1991) doi:10.1109/18.61115; Endres & Schindelin, IEEE TIT
  49(7) (2003) doi:10.1109/TIT.2003.813506; Wang, Kulkarni & Verdú, IEEE TIT 55(5):2392 (2009)
  doi:10.1109/TIT.2009.2016060.

### 3.4 Mutual information for feature relevance and ruler health

We have no labelled errors, but the expert data contain **natural minimal pairs**:
- ikhfa vs izhar contexts should differ in nasality;
- madd_tabii vs muttasil/lazim should differ in duration;
- tafkheem vs tarqeeq should differ in F2−F1.

For each metric m and each contrast K (a binary rule-class label fixed by the text), estimate
`I(m ; K | reference reciters)` with the KSG k-NN estimator (mixed discrete–continuous version).

- **Ruler selection.** Replace "lowest robust CV on Husary" (`metric_spec.json` `alts`) with
  **max I(m;K)**, or with the MDL criterion (§3.5).
  - CV only measures the stability of one class.
  - MI measures the ruler's *ability to see the rule*.
  - A ruler with tiny CV but no contrast (e.g. a constant) is exactly what produced the zero-entropy
    keys in §0.3.
- **Health check.** Two automatic flags:
  - verdict-entropy check `H(status | key)` on reference data. A value of 0 means the ruler is dead
    or the band is mis-set. Current offenders: `madd_iwad`, `hamzat_wasl`, `safir`,
    `jawaz_wajhayn`.
  - MI check `I(m;K) < ε`.
- **Multi-metric relevance** without the redundancy of the current `max z`: use conditional MI
  `I(m₂; K | m₁)`, i.e. mRMR-style greedy selection.
- **Julia:** a KSG implementation is about 40 lines with `NearestNeighbors.jl`. `CausalityTools.jl`
  (now `Associations.jl`) ships KSG estimators.
- Sources: Kraskov, Stögbauer & Grassberger, Phys. Rev. E 69:066138 (2004) arXiv:cond-mat/0305641;
  Gao, Kannan, Oh & Viswanath, NeurIPS 2017 arXiv:1709.06212 (mixed discrete-continuous MI);
  Peng, Long & Ding, IEEE TPAMI 27(8) (2005) doi:10.1109/TPAMI.2005.159 (mRMR).

### 3.5 MDL / Bayesian model selection for rule models

Candidate generative models per rule key, fitted on the reference data:

| Model | Example use |
|---|---|
| M1 Gaussian | Baseline |
| M2 Student-t | Heavy tail from deliberate elongation |
| M3 skew-t | Asymmetric duration tails |
| M4 two/three-component mixture | madd_munfasil 4↔5, madd_arid 2/4/6, tayyibah qasr |
| M5 regression on local tempo | Count-in-beats, roadmap B1 |

**Criterion:** PSIS-LOO elpd (Vehtari et al.). This is the Bayesian code length of held-out data,
i.e. prequential MDL.
- For mixtures, BIC is invalid because the models are singular. Use **WBIC** (Watanabe 2013) or
  PSIS-LOO.
- NML/two-part MDL is the frequentist twin: `L(D) = −log p(D|θ̂) + COMP(M)`.
- This yields a principled answer to the carry-over's open question (madd_arid and madd_leen "stay
  textbook"). Model them as mixtures whose **component membership is the reciter's declared
  choice**, and judge **consistency** as a within-reciter component-stability test (Hafs requires
  a consistent length). That is an entropy: `H(component | reciter, rule)` should be ≈ 0 bits.
- The SINDy/STLSQ selection in `Discovery.jl` already uses information criteria. Extend it with
  PSIS-LOO through `ParetoSmooth.jl`.
- Sources: Rissanen, Automatica 14:465 (1978) doi:10.1016/0005-1098(78)90005-5; Grünwald & Roos,
  "MDL revisited", arXiv:1908.08484; Watanabe, JMLR 14 (2013) arXiv:1208.6338 (WBIC); Vehtari,
  Gelman & Gabry arXiv:1507.04544 (PSIS-LOO).

### 3.6 Rate–distortion view of "perfection"

Let the canonical recitation be a source X. A reciter is an encoder producing X̂. Define a
Tajweed **distortion** with an explicit hierarchy:

```
d(x, x̂) = 0                      if correct
        = w_K(k) ∈ (0, 1)        if lahn khafi of type k (ghunnah length, madd deviation, sifa)
        = 1 + w_J(j) (≥ 1)       if lahn jali (letter/haraka change; w_J larger if meaning changes)
```

- **Perfection index:** `𝒫_r = 1 − E[d] / d_ref`, where E[d] is the *posterior expected
  distortion* per letter, from L3, and d_ref normalises.
- **Rate–distortion connection.** `R(D) = min_{p(x̂|x): E d ≤ D} I(X; X̂)` is the minimal
  information per letter a recitation must faithfully carry to stay within distortion D.
  - Operational use: the ijazah threshold is a **distortion budget D\***.
  - The feedback "what to fix first" is the greedy step that most reduces E[d] per unit of practice
    effort. That is a reverse water-filling over rule families: fix the families where the
    *marginal* distortion is largest.
- **Why this is better than a weighted mean of PASS scores:**
  - The distortion measure *is* the lahn hierarchy.
  - The expected distortion under a calibrated posterior is a Bayes risk. That is the same object as
    §4.3, and it is internally consistent.
- Sources: Shannon, IRE Nat. Conv. Rec. (1959); Berger, *Rate Distortion Theory* (1971); Cover &
  Thomas ch. 10.

### 3.7 Information bottleneck for invariant scoring features and fingerprints

- **Scoring features** should keep information about *correctness* and discard *who / where*
  (timbre, room, maqam, tempo): maximise `I(T; correctness) − β I(T; nuisance)`.
  - This is the variational IB of Alemi et al.
  - In practice, add an adversarial nuisance head (reciter ID, room class from the 8-d environment
    vector) with gradient reversal when fine-tuning the SSL head.
- **Fingerprint** is the opposite IB: keep reciter identity, discard content (the text) and room.
- **Diagnostic.** Report `I(T; reciter)` of the scoring representation. If it is high, the "error"
  detector is partly a *style* detector. Style detection is the main false-FAIL risk for imams with
  legitimate maqam and tempo variation.
- Sources: Tishby, Pereira & Bialek arXiv:physics/0004057; Alemi et al., VIB, arXiv:1612.00410;
  Ganin et al. JMLR 17 (2016) arXiv:1505.07818 (gradient reversal).

---

## 4. Scoring: from heterogeneous evidence to one grade

### 4.1 Requirements
The grade must:
- (i) be non-compensatory for jali, since no amount of perfect madd excuses a changed letter;
- (ii) correct for item difficulty and ruler bias;
- (iii) carry an uncertainty interval;
- (iv) map to a human scale (101 → ijazah);
- (v) be learnable from ~16 expert reciters plus a small amount of labelled learner data.

### 4.2 Item Response Theory: deep evaluation (recommended core)

**Why IRT fits well.** A recitation is a test whose **items** are rule instances. Items differ in
difficulty: madd_iwad ≠ hamzat_wasl. They also differ in discrimination: a ghunnah item on a
noisy ruler discriminates less. **Examinees** are reciters, or a student at time t. Our data show
item effects dominate (§0.2), which is IRT's home ground.

**Model family** (build up in this order):

1. **Rasch / 1PL on binary PASS vs not.** Baseline, already fitted in §0.2.
   `logit P(y_{ri}=1) = θ_r − b_i`.
2. **Explanatory IRT with item covariates (LLTM).** Items are too many and too sparse to have a
   free b_i each. Model the difficulty:
   `b_i = x_iᵀ β + ε_i`, with `ε_i ~ N(0, σ_b²)`.
   Here x_i is rule key, letter, makhraj, position (waqf/wasl), local tempo, `align_conf` and
   recording mode.
   - `β_key` then **is** the ruler-bias correction, learned jointly across all reciters.
   - This replaces much of the hull logic for categorical verdicts.
   - Source: De Boeck & Wilson (eds.), *Explanatory Item Response Models* (2004)
     doi:10.1007/978-1-4757-3990-9.
3. **Graded response model (Samejima)** for PASS/WARN/FAIL, keeping the ordinal information:
   `P(y ≥ k) = σ(a_i(θ_r − b_{ik}))`.
   Better still, use **continuous-response IRT** on the calibrated e_i or p-values. That uses the
   full metric, not thresholded verdicts; thresholding throws away ~ (1 − 2/π) of the information
   in the Gaussian case.
4. **Multidimensional IRT (MIRT).** Use a fixed, confirmatory loading structure:
   - `θ_jali` (letters/harakat: phoneme-tier GOP items),
   - `θ_timing` (madd, ghunnah length, sakt),
   - `θ_nasal` (ikhfa/idgham/iqlab quality),
   - `θ_sifaat` (hams/jahr, itbaq, qalqalah, tafkheem).
   Correlations between dimensions are estimated. This is where the jali/khafi hierarchy becomes
   *structural*.
5. **Testlet effects.** Instances in one ayah share tempo and alignment errors. Add a random
   `γ_{r,ayah}` (Bradlow, Wainer & Wang 1999). Without it, abilities are overconfident: the same
   dependence problem as §3.2.
6. **Many-facet Rasch (Linacre)** adds a **facet for recording condition** (studio / taraweeh /
   phone mic) and for the **measurement pipeline version**. Imams are then not penalised for
   reverberation that the dereverb adapter did not fully remove.

**Identification and anchoring.**
- Fix the scale by `θ_Husary := 0` and `sd(θ_reference) := 1`, or by setting the mean of the
  ijazah reference group to 0.
- The ijazah cut-score on θ is then set by standard-setting (§4.4).
- Because item parameters are shared, **any student reciting any ayah is placed on the same
  scale**. This is test *linking* for free: the ayah need not be one the references recited, as
  long as its item covariates are covered.

**Information and adaptive testing (ijazah exam design).**
- Item Fisher information: `I_i(θ) = a_i² P_i(θ)(1 − P_i(θ))`.
- Test information adds up; `SE(θ̂) = 1/√ΣI_i`.
- Two immediate uses:
  - (a) **Retire zero-information items.** The degenerate keys of §0.3 contribute I ≈ 0 and only
    add noise to the current mean.
  - (b) **Computerised adaptive testing.** Choose the next ayah or passage for a student to
    maximise expected information at the current θ̂ (the maximum-information rule), stopping at
    SE < 0.25. Consequences:
    - a 101 learner is tested on high-contrast madd/ghunnah items;
    - an ijazah candidate is routed to hard, discriminating passages (idgham naqis, ikhfa before
      ق/ك, ض vs ظ);
    - this cuts test length by typically 30–50 % (standard CAT result, Wainer 2000).

**Data needs vs what we have.**
- Item parameters (β for about 50 rule keys × context covariates): identifiable from about
  16 reciters × thousands of instances, if items vary enough. With experts only, most responses are
  "correct", so **discriminations a_i are weakly identified**.
- Start with Rasch/LLTM (a ≡ 1) and hierarchical priors. Add a_i only after learner data arrive:
  `qdat_bench`, Iqra'Eval, Quranic Audio Dataset (arXiv:2405.02675), plus §6.6 perturbations.
- Expert-only fitting still gives the most valuable part: **ruler-bias difficulties**.

**Estimation.**
- **Julia:** `Turing.jl` (NUTS, non-centred θ and ε; about 10⁴ responses × about 60 parameters in
  minutes on CPU), or `Pathfinder.jl` for fast variational initialisation.
- **Python:** PyMC / NumPyro; brms (R) supports all of the above with formula syntax.
- **Validation:**
  - PSIS-LOO over ayahs (group-LOO);
  - posterior predictive checks of the verdict distribution per key;
  - person-fit statistics (l_z) to flag reciters the model misfits, e.g. declared tayyibah
    choices;
  - **DIF analysis** (differential item functioning, studio vs taraweeh). DIF items are
    condition-biased rulers.
- Sources: Rasch (1960); Samejima, Psychometrika Monogr. 17 (1969) doi:10.1007/BF03372160; Reckase,
  *Multidimensional IRT* (2009) doi:10.1007/978-0-387-89976-3; Bradlow, Wainer & Wang, Psychometrika
  64:153 (1999) doi:10.1007/BF02294533; Linacre, *Many-Facet Rasch Measurement* (1989); Bürkner,
  "Bayesian IRT with brms and Stan" arXiv:1905.09501; Wainer (ed.), *Computerized Adaptive Testing*
  (2000); Drasgow, Levine & Williams (1985) person-fit l_z, doi:10.1111/j.2044-8317.1985.tb00817.x.

**Assessment: IRT is the right backbone.**
- It is the only candidate that jointly solves D5 (difficulty), D6 (weights become
  discriminations), uncertainty (posterior on θ), the jali/khafi structure (MIRT) and test design
  (CAT).
- Its weaknesses:
  - (a) it needs *some* non-expert responses to learn discrimination;
  - (b) it assumes local independence (mitigated by testlets);
  - (c) it is only as good as the per-instance evidence it consumes. It does **not** fix D1/D2;
    that needs §6.

### 4.3 Bayesian decision theory with a lahn jali/khafi loss matrix (per instance)

**States:** s ∈ {C: correct, K: lahn khafi, J: lahn jali}.
**Actions:** a ∈ {PASS, NOTE (khafi feedback), FAIL (jali), REVIEW (defer to a human teacher)}.

Example loss matrix (to be elicited from teachers; the numbers are placeholders):

| L(a, s) | C | K | J |
|---|---|---|---|
| PASS | 0 | 2 | 20 |
| NOTE | 1 | 0 | 8 |
| FAIL | 6 | 3 | 0 |
| REVIEW | 0.7 | 0.7 | 1.5 |

**Posterior per instance**
```
p(s | x_i) ∝ p(x_i | s, c_i) · π(s | θ_r, c_i)
```
- **Likelihood `p(x|C)`:** the reference predictive of §3.2 (roadmap A1/A5).
- **Likelihood `p(x|K)`:** from perturbation-generated khafi errors (§6.6).
- **Likelihood `p(x|J)`:** from GOP against the jali confusion set (§6.3).
- **Prior π:** from IRT: `π(C | θ, b) = σ(θ − b)`. The split of the remaining mass K:J comes from
  base rates by learner level.
- The coupling is powerful: the same acoustic evidence is judged more sceptically for a learner with
  low θ_jali, which is how human teachers listen.

**Bayes action:** `a* = argmin_a Σ_s L(a,s) p(s|x)`.
- With the table above, FAIL beats PASS when `6p_C + 3p_K < 2p_K + 20p_J`.
- For example, with p_K = 0 the break-even point is p_J > 0.23.
- REVIEW wins in the uncertain middle. This replaces "SKIPPED on low alignment confidence" with an
  explicit abstain option. **It fixes D2**: a low-confidence span becomes evidence for J (through
  GOP) or goes to REVIEW. It is never silently dropped.

**Recitation-level decision (non-compensatory).**
- Jali counts follow a Poisson process with rate λ_J per 1 000 letters.
- The posterior is Gamma(α₀ + n_J, β₀ + n_letters · s_det), with detector recall s_det. The recall
  is measured on §6.6 data.
- **Ijazah-level jali criterion:** `P(λ_J < λ₀ | data) ≥ 0.95`.
- With zero detected jali, the "rule of three" gives the required evidence:
  `n_letters ≥ 3 / (s_det · λ₀)`.
- Example: λ₀ = 1 per 1 000 letters and s_det = 0.8 require **≥ 3 750 judged letters**. That is
  about one third of a juz (the Qur'an has ≈ 320 k letters / 30 juz ≈ 10.7 k letters per juz).
- **This is a concrete, defensible minimum exam length.**
- The khafi dimension then grades *within* the jali-clean set (lexicographic order).
- Sources: Berger, *Statistical Decision Theory and Bayesian Analysis* (1985)
  doi:10.1007/978-1-4757-4286-2; Chow, IEEE TIT 16(1):41 (1970) doi:10.1109/TIT.1970.1054406
  (reject option); Hanley & Lippman-Hand, JAMA 249:1743 (1983) (rule of three);
  Wald, *Sequential Analysis* (1947) (SPRT for early stopping).

### 4.4 Ordinal regression onto the grading scale (101 → ijazah)

Define ordinal grades G ∈ {1: beginner (makharij), 2: basic ahkam, 3: fluent murattal, 4:
mujawwad-competent, 5: ijazah-ready}. The final number of levels should come from teachers.

Cumulative-logit model on the IRT posterior:
```
P(G ≤ g | θ_r) = σ(τ_g − wᵀθ_r),   τ_1 < … < τ_4
```
- **Constraint for the hierarchy:** grade ≥ 4 requires θ_jali above a cut. Use a hurdle: the
  sequential/continuation-ratio ordinal model (Tutz), where passing level g requires clearing the
  jali hurdle first.
- **Data:** about 50–150 teacher-graded students give τ within about ±0.3 logits. Until then, set τ
  by **standard setting**: Angoff/bookmark with 2–3 teachers rating anchor recordings. This is
  cheap and well established.
- **Evaluation:** ranked probability score (a proper ordinal score), quadratic weighted κ against
  teachers, and calibration of `P(G = 5)`.
- Sources: McCullagh, JRSS-B 42:109 (1980); Bürkner & Vuorre, AMPPS 2(1) (2019)
  doi:10.1177/2515245918823199; Tutz, *Regression for Categorical Data* (2012); Cizek (ed.),
  *Setting Performance Standards* (2012).

### 4.5 Proper scoring rules: evaluating the engine itself

Every probability the engine emits should be scored with a proper rule on held-out data:

| Output | Score |
|---|---|
| Per-instance verdict probabilities | Log score and Brier; Brier decomposition into reliability / resolution / uncertainty (Murphy 1973) |
| Continuous predictive of metrics | CRPS (closed form for Gaussian/t) |
| Ordinal grade | RPS |

The current `score ∈ [0,1]` weighted mean is not a scoring rule. Keep it only as a derived display
number: `100 × (1 − posterior expected distortion)` (§3.6).

When several p-values must be combined per instance (multi-metric rules, D3), use the **Cauchy
combination test**. It stays valid under arbitrary dependence, whereas `max z` has no stated level.
- Sources: Gneiting & Raftery (2007) above; Murphy, J. Appl. Meteor. 12:595 (1973); Liu & Xie,
  JASA 115:393 (2020) doi:10.1080/01621459.2018.1554485.

---

## 5. Calibration and uncertainty

### 5.1 Conformal prediction, done right for our data structure

- **Mondrian conformal per rule key** (and per key × context bin, e.g. waqf vs wasl): compute the
  conformity score within each category.
  - This gives **class-conditional coverage per rule**, which a global threshold cannot.
  - Minimum category size for α: `n ≥ 1/α − 1`. That means ≥ 19 calibration instances per key for
    α = 0.05. Most keys have that at *instance* level in the full-Qur'an runs; few do at 59 ayahs.
- **The exchangeability unit is the reciter, not the instance.** Instances are clustered.
  - Roadmap A4 (LOO over peers) is correct but coarse: with P = 9 the achievable levels are
    multiples of 1/(P+1) = 0.1.
  - **Fix: two-layer hierarchical conformal** (Dunn, Wasserman & Ramdas). It pools instances
    across reciters with valid coverage for a *new reciter*, exploiting within-reciter
    exchangeability. This is the correct theorem for "held-out peer passes".
- **Conformal risk control (CRC)** chooses the FAIL threshold λ so that the *expected expert
  false-FAIL rate* stays ≤ ε (e.g. 2 %), with a finite-sample guarantee:
  `λ̂ = inf{λ : (n/(n+1)) R̂_n(λ) + B/(n+1) ≤ ε}`.
- **Learn-then-Test** controls several risks at once, e.g. expert false-FAIL ≤ 2 % **and**
  perturbation-miss rate ≤ 20 %, through multiple testing over a grid of (λ_WARN, λ_FAIL).
- **Conformal anomaly p-values:** `p_i = (1 + #{j: s_j ≥ s_i})/(n+1)` on the neural anomaly scores
  (§6.5) turns any detector into a valid test. Bates et al. give FDR control across the many
  instances of one recitation: Benjamini–Hochberg on conformal p-values stays valid thanks to PRDS.
- **Julia:** `ConformalPrediction.jl` covers split, jackknife+ and CV+ for MLJ models. Mondrian, CRC
  and hierarchical conformal are each < 50 lines by hand (order statistics). Python: `MAPIE`,
  `crepes` (Mondrian).
- Sources: Vovk, Gammerman & Shafer (2005/2022) doi:10.1007/978-3-031-06649-8; Vovk et al.
  Mondrian CP (2003); Dunn, Wasserman & Ramdas arXiv:1809.07441; Angelopoulos et al. CRC
  arXiv:2208.02814; Angelopoulos et al. Learn-then-Test arXiv:2110.01052; Bates et al. "Testing for
  outliers with conformal p-values" arXiv:2104.08279; Barber et al. jackknife+ arXiv:1905.02928.

### 5.2 LOO-PIT for the Bayesian reference model
For the roadmap A5 model:
- compute PIT_i = F_{−i}(y_i) with PSIS weights;
- test uniformity per rule key: ECDF envelope plots (Säilynoja et al.);
- a U-shape means over-confident bands; a hump means bands too wide.
Do this per key and per reciter group (anchor, peers, imams). The imams' PIT *should* deviate; the
peers' must not.
- Sources: Gelfand et al. (1992); Säilynoja, Bürkner & Vehtari, Stat. Comput. 32 (2022)
  arXiv:2103.10522; LOO-PIT arXiv:2410.03507 (cited in the roadmap).

### 5.3 Calibrating neural posteriors (CTC and sifa tiers)
- CTC posteriors are **peaky**: spikes plus blank. Frame max-softmax is a poor confidence.
- Use **segment-level** confidence instead:
  - the GOP-SF posterior of the canonical phone (sums over all alignments), or
  - the forced-alignment path posterior ratio.
- Then **temperature-scale** the logits:
  `p = softmax(z/T)`, with T fitted by NLL on held-out *labelled* segments.
  - Labels come from §6.6 perturbations (positives) plus unperturbed expert audio (negatives).
  - **Vector scaling**, i.e. a per-class T, suits ~43 phone classes. Use it for the jali confusion
    classes.
- **Metrics:** ECE with adaptive binning, and reliability diagrams per makhraj group. Report
  before and after.
- **Domain check:** `TBOGamer22/...` was trained on *isolated word* clips (Buraaq/quran-md-words,
  77 k words). Its confidence on continuous tarteel with idgham across word boundaries is expected
  to be lower and mis-calibrated. That alone may explain part of the SKIPPED mass.
- Sources: Guo et al. ICML 2017 arXiv:1706.04599; Kumar, Liang & Ma arXiv:1909.10155 (verified
  calibration / ECE bias); Graves et al. ICML 2006 (CTC).

### 5.4 Small-N practice (≈ 10 reference reciters)
- Always report **intervals**: perfection index and θ with 90 % credible intervals, via posterior
  draws or the **cluster (ayah-block) Bayesian bootstrap** (Rubin 1981).
- Hierarchical shrinkage of per-key scales τ_k toward a family-level τ (roadmap A5). This matters
  because rare keys (madd_silah_kubra, jawaz_wajhayn) otherwise get arbitrary bands.
- **Nested LOO over reciters** for anything tuned: weights, thresholds, T. Never tune and test on the
  same peer.
- **Leakage audit.** Both candidate neural models may have seen our reference reciters in training.
  Hold out imams and non-EveryAyah reciters (e.g. from Tadabur) as the *model* test set.

---

## 6. Neural networks

### 6.1 The central gap: an alternative hypothesis for every letter
Jali detection needs `p(audio | canonical)` **versus** `p(audio | best alternative)`. Options, from
cheapest to most complete:

1. **Free phone decoding** (greedy CTC) with the existing TBOGamer model, then Levenshtein-align
   to canonical. This gives substitution/deletion/insertion counts, i.e. the channel of §3.1.
   Cheap, but it has a high false-alarm rate on a word-trained model.
2. **muaalem-v3_2 multi-level CTC** (w2v-BERT 2.0, 43 phoneme symbols in the Quran Phonetic Script
   plus 10 sifa tiers).
   - Its madd symbols encode *counts* (e.g. 4-beat = 4 madd symbols) and ghunnah as tripled noon.
     So **timing errors appear as phoneme-tier insertions and deletions.** This is a neural second
     opinion on our DSP counts.
   - Sifa tiers give per-letter hams/jahr, shidda/rakhawa, tafkheem/tarqeeq, itbaq, safir,
     qalqalah, takrir, tafashshi, istitala and ghunnah. That is a direct counterpart to
     `app/sifaat/*`.
   - Reported test PER is 0.54 % (phonemes) and about 0.1–0.2 % (sifa tiers) on held-out *expert*
     mushafs.
   - The authors report qualitative error detection only. **No learner-error benchmark exists yet.**
     Our §6.6 perturbation set plus `qdat_bench` would be a real contribution.
3. **GOP with competitor sets (recommended for jali).**
   - **GOP-SF** (segmentation-free): sum over all CTC alignments of the canonical vs the substituted
     sequence,
     `GOP_SF(p_i) = log P_CTC(ℓ | A) − log max_{q ≠ p_i} P_CTC(ℓ[i ← q] | A)`.
     Computed with the CTC forward algorithm; no reliance on a single Viterbi path. This fixes the
     boundary brittleness behind D2.
   - **Substitution-aware GOP:** restrict q to phonological neighbours. For Qur'an this is the
     **classical lahn-jali confusion set**: ض/ظ/د/ذ, ص/س, ط/ت, ق/ك, ح/ه, ع/ء, ث/س/ت, ذ/ز/د, غ/خ, and
     short-vowel swaps (fatha/kasra/damma) which are jali by definition.
     - This cuts cost from O(|V|·L) to O(|confusion set|·L).
     - It gives the diagnosis for free ("said د instead of ض").
   - Sources: arXiv:2507.16838 (Cao et al., GOP-SA/SF); arXiv:2506.02080 (Parikh et al.,
     Interspeech 2025).
4. **Attribute-level MDD** (Shahin, Epps & Ahmed): detect speech *attributes*, i.e. manner, place,
   voicing. These map almost one-to-one onto makharij and sifaat. They show lower FAR/FRR/DER than
   phoneme-level MDD. The muaalem sifa tiers are the Qur'anic instance of this idea.
   arXiv:2311.07037 (Speech Commun. 2025).

**Plug-in points**
- The new `app/mdd/` (Python, transformers) produces, per letter unit: `gop_canonical`,
  `gop_best_alt`, `alt_symbol`, and a sifa-tier posterior vector.
- These attach to `RuleDiagnostic.metrics` (the same pattern as `_attach_alignment_metrics`) and
  new jali items for IRT (`θ_jali`).
- The aligner keeps TBOGamer for time stamps (fast, already integrated). Alternatively, take
  alignments from muaalem's phoneme tier; that yields better madd/ghunnah boundaries if it
  validates.

**GPU:**
- muaalem (w2v-BERT 2.0, about 600 M parameters) inference: a Kaggle T4/P100 does roughly 50–100×
  real time in fp16. The full Qur'an × 16 reciters (≈ 16 × 130 h) takes about 25–40 GPU-hours,
  inside Kaggle's free 30 h/week over 1–2 weeks.
- A strategic subset (a few thousand ayahs) takes < 2 h.
- CPU-only inference is feasible but ~10× slower.

### 6.2 SSL backbones: what to use for what

| Model | Pretraining | Use here | Source |
|---|---|---|---|
| wav2vec 2.0 | 53 k h English (base) | Current aligner base; weakest for Arabic | arXiv:2006.11477 |
| HuBERT / mHuBERT | Masked prediction of cluster targets | Iqra'Eval baselines | arXiv:2106.07447 |
| WavLM | Denoising masked prediction, 94 k h | Best SUPERB phone recognition + robustness to reverb (taraweeh!) | arXiv:2110.13900 |
| w2v-BERT 2.0 | 4.5 M h, 143 langs incl. ~110 k h Arabic | **Best choice**; muaalem backbone | arXiv:2108.06209; Seamless arXiv:2312.05187 |
| Whisper-large-v3 encoder | 5 M h weakly supervised | Iqra'Eval entry adapted it as speech-to-phoneme; heavy, 30 s windows | arXiv:2212.04356; ACL 2025.arabicnlp-sharedtasks.64 |
| MMS | 1 400 langs | Fallback multilingual CTC | arXiv:2305.13516 |

**Layer choice.** For phonetic and articulatory probing, middle layers carry the most phone
information (Pasad et al.). Use a learned weighted layer sum (SUPERB-style) for any downstream
head, and freeze the encoder when data are small.
Sources: Pasad, Chou & Livescu arXiv:2107.04734; Yang et al. SUPERB arXiv:2105.01051.

### 6.3 GOP variants: summary of equations
- **Classic GOP (Witt–Young):** `GOP(p) = (1/D_p) log [P(O|p) / max_q P(O|q)]` over the aligned
  segment.
- **DNN-GOP / LPR:** `LPR(p) = log p(p|O) − max_{q≠p} log p(q|O)` (frame-averaged posteriors). Hu
  et al. added logistic-regression calibration per phone.
- **GOP feature vector (Shi et al.; Gong et al. GOPT):** feed [LPP, LPR per competitor, duration]
  into a small classifier or transformer. This is the natural input to the IRT/Bayes layer.
- **GOP-SF (CTC):** see §6.1. It is the right choice for CTC models, which have no reliable frame
  segmentation.
- Sources: Witt & Young (2000) above; Hu et al., Speech Commun. 67:154 (2015)
  doi:10.1016/j.specom.2014.12.008; Gong et al., GOPT, ICASSP 2022 arXiv:2205.03432.

### 6.4 Phoneme recognition with articulatory attributes (sifaat)
- The multi-task CTC design (phoneme + attribute tiers) is directly the jali/khafi decomposition:
  - a wrong phoneme with correct attributes suggests a makhraj slip;
  - a correct phoneme with a wrong attribute is khafi-type (e.g. insufficient istila/tafkheem).
- **Consistency constraint:** the phoneme and sifa tiers must agree via the deterministic
  letter → sifaat table. **Disagreement is itself a detection signal**: mutual information between
  tiers drops under error.
- **Plug-in:** replaces or augments the DSP detectors in `app/sifaat/`. Keep DSP as interpretable
  evidence and treat both as items in IRT with their own discriminations; the model learns which
  to trust.

### 6.5 Self-supervised anomaly detection trained only on experts

The setting matches what we have: many hours of *correct* recitation, few labelled errors. It is
one-class, conditional on the canonical letter and context.

**Features:** SSL embeddings (w2v-BERT/WavLM mid-layer, mean-pooled over the aligned unit)
`h_i ∈ ℝ^d`, plus the DSP metrics. Condition on c_i = (letter, harakah, sifaat, rule key,
neighbours).

| Method | Score | Remarks |
|---|---|---|
| Class-conditional Gaussian (Mahalanobis) | `(h − μ_c)ᵀ Σ⁻¹ (h − μ_c)` with shared Σ (Ledoit–Wolf) | **Strong baseline**, closed form, CPU. Lee et al. arXiv:1807.03888 |
| Deep k-NN | Distance to k-th nearest expert unit of the same class | No density assumption. Sun et al. arXiv:2204.06507 |
| Conditional normalising flow | `−log p_φ(h | c)` | Exact likelihood; plugs into §3.2 surprisal. Papamakarios et al. arXiv:1912.02762; RealNVP arXiv:1605.08803 |
| Deep SVDD | `‖f(h) − c‖²` | Needs care against collapse. Ruff et al. ICML 2018 |
| Energy-based / JEM | `E(h, c)` | Heavier; low priority. Grathwohl et al. arXiv:1912.03263 |

**Critical risk: style vs error.** Expert-only one-class models flag *everything unusual*:
maqam, a new voice, a new room. Mitigations:
- (i) condition on context;
- (ii) use peers from *different* schools (Egyptian vs Hijazi) in the reference set;
- (iii) nuisance-invariant features (§3.7);
- (iv) normalise per reciter: score residuals after removing the reciter's own global offset (a
  random effect) so that timbre does not score as error;
- (v) **conformalise** the scores per class (§5.1), making false-alarm rates explicit.

**Evaluation:** AUROC/AUPRC on §6.6 perturbations and `qdat_bench`; false-alarm rate on held-out
peers and on non-EveryAyah expert reciters (Tadabur, arXiv:2604.18932: 1 400 h, 600+ reciters).

**Compute:** embedding extraction on GPU (as in §6.1). Flows and kNN on CPU in Julia (`Lux.jl` or
`Flux.jl` for small RealNVP; `NearestNeighbors.jl`) or Python (`zuko`, `nflows`, FAISS already
present).

### 6.6 Labelled errors without annotators: counterfactual perturbation
This is the cheapest way to get *positives* with **known ground-truth magnitude**, for power
analysis, temperature scaling, CRC/LTT and IRT discriminations.

| Error class (lahn) | Perturbation on expert audio | Label |
|---|---|---|
| Madd too short/long (khafi, jali if the madd disappears) | WSOLA time-scale of the aligned vowel core by factor κ ∈ [0.3, 2] (keep formants) | Δcounts = (κ−1)·counts |
| Ghunnah too short / absent | Truncate or crossfade the nasal murmur; notch the ~250 Hz nasal pole region | Δcounts, nasal Δ |
| Qalqalah missing | Attenuate the release burst (−20 dB over 30 ms) | Binary |
| Idgham/izhar swap | Splice the peer's izhar realisation into the idgham context (same words exist across reciters) | Rule-class swap |
| Letter substitution (jali) | Splice the aligned unit of the confusable letter (ص→س) from the *same reciter* elsewhere, with PSOLA pitch matching | Phone swap |
| Harakah swap (jali) | TTS or splice of the vowel from matched context | Vowel swap |

**Caveat.** Splices contain concatenation artifacts that a detector can learn as a shortcut. Use
the same splice process for *identity* splices (a unit replaced by itself from another occurrence)
as negatives. Reserve real learner data (`qdat_bench`, QuranMB, Quranic Audio Dataset) as the final
test.

Precedent: Iqra'Eval used a 52 h TTS-synthesised error set that was "competitive with
wild-collected data" (arXiv:2506.07722).

**Tools:** `librosa`/`pyrubberband` in Python, or the existing Octave DSP. CPU only.

### 6.7 Contrastive reciter embeddings (fingerprint)
- 16 reciters are too few for metric learning. **Pretrain on Tadabur** (600+ reciters,
  arXiv:2604.18932) or on Quran-MD (30 reciters). Use supervised-contrastive / GE2E loss on SSL
  layer-weighted features with **ayah-disjoint** batches (so content is not the shortcut).
- The IB objective (§3.7) removes text content: add an adversarial "which ayah" head.
- **Evaluate** with ayah-disjoint retrieval mAP plus tempo/channel perturbation curves, per
  roadmap C. Keep ECAPA as the baseline.
- Sources: Khosla et al. SupCon arXiv:2004.11362; Wan et al. GE2E arXiv:1710.10467; Desplanques et
  al. ECAPA arXiv:2005.07143.

### 6.8 Neural ODE / state-space models (S4, Mamba) for prosody: honest assessment
- **What they would add:** long-context tempo, pitch and maqam dynamics across a surah (thousands
  of frames). S4/Mamba handle length linearly. Neural CDEs pair naturally with the roadmap's path
  signatures (C1).
- **What we have:** about 16 reciters. A 3-parameter relaxation ODE already fits the tempo drift
  (Discovery.jl). The roadmap's Koopman/GP/latent-force items (B2/B3) give interpretable
  generalisations with small data.
- **Verdict:** low priority. It is worth doing only (a) as a **sequence-level anomaly model**
  trained on Tadabur (hundreds of reciters), or (b) as the temporal head of a fine-tuned MDD
  model, where Conformer/Transformer already suffice.
- If done, use a **Neural CDE on log-signature features**: `DiffEqFlux.jl` / `Lux.jl` in Julia fit
  the existing SciML stack; `torchcde` in Python.
- Sources: Gu et al. S4 arXiv:2111.00396; Gu & Dao Mamba arXiv:2312.00752; Chen et al. Neural ODE
  arXiv:1806.07366; Kidger et al. Neural CDE arXiv:2005.08926; Morrill et al. log-ODE
  arXiv:2009.08295.

### 6.9 Available Qur'anic datasets and models (verified on the HF Hub, 2026-09-23)

| Resource | Content | Use |
|---|---|---|
| `obadx/muaalem-model-v3_2` | w2v-BERT 2.0, 11-level CTC (phonemes + 10 sifaat), MIT | Jali + sifa evidence (§6.1) |
| `obadx/muaalem-annotated-v3` | ≈ 850 h, 286 k utterances, 22 expert reciters, QPS transcripts | Fine-tuning; phone/sifa references |
| `obadx/qdat_bench` | 159 learner clips of Al-Ma'idah 109 with per-rule labels (madd lengths 0–8, ghunnah, qalqalah) plus sifa transcripts | **Real learner test set** for madd/ghunnah counts (khafi) |
| `obadx/recitation-segmenter-v2` | w2v-BERT waqf segmenter | Pause/sakt evidence |
| `IqraEval/QuranMB.v2`, `IqraEval/Iqra_train`, `Iqra_TTS`, `Iqra_Extra_IS26` | MSA-style Qur'an reading MDD test (1 642 clips) and training; TTS error data | Jali (phoneme) MDD benchmark; **no Tajweed labels** |
| `IqraEval/Iqra_{wavlm,hubert,mhubert,wav2vec2}_base` | Baseline MDD models | Baselines |
| `TBOGamer22/wav2vec2-quran-phonetics` | wav2vec2-base, trained on word-level Quran-MD | Current aligner |
| `Buraaq/quran-md-ayahs` (+ Kaggle WAV mirror) | 30 reciters × 6 236 ayahs | Our reference source |
| Tadabur (arXiv:2604.18932) | 1 400 h, 600+ reciters | Embedding pretraining; expert anomaly negatives; leakage-free tests |
| Quranic Audio Dataset (arXiv:2405.02675) | ≈ 7 k crowd recitations, 1 166 annotated by non-Arabic speakers | Learner-level data for IRT discriminations |

Related work to position against:
- the Iqra'Eval 2025 shared task (ArabicNLP 2025, ACL Anthology 2025.arabicnlp-sharedtasks.61)
  and its system papers (Whisper-large-v3 speech-to-phoneme; multi-stage domain adaptation);
- earlier small-scale Tajweed classifiers (arXiv:2305.06429).

---

## 7. Julia vs Python and GPU

| Component | Language | Packages | GPU |
|---|---|---|---|
| Channel/MI/JS/KSG, Blahut–Arimoto | Julia | `Statistics`, `NearestNeighbors.jl`, `Associations.jl` | No |
| MDL/PSIS-LOO/WBIC per rule | Julia | `Turing.jl`, `ParetoSmooth.jl`, `Distributions.jl` | No |
| IRT (Rasch→LLTM→GRM→MIRT+testlet+facets) | Julia (primary), brms for cross-check | `Turing.jl` (+ `Pathfinder.jl`) | No (minutes on CPU) |
| Bayes decision, ordinal, rule-of-three | Julia → exported tables read by `app/calibration.py` | `Distributions.jl` | No |
| Conformal (Mondrian, hierarchical, CRC, LTT) | Julia | `ConformalPrediction.jl` plus hand-rolled order statistics | No |
| SSL inference, GOP-SF, muaalem tiers | Python | `transformers`, `torchaudio.functional.forced_align`/CTC forward | Kaggle T4 (free) |
| Temperature/vector scaling | Python or Julia (tiny) | `Optim.jl` | No |
| Anomaly (Mahalanobis/kNN/flows) | Julia or Python | `Lux.jl`/`Flux.jl`; `zuko`; FAISS | No (flows: optional GPU) |
| Perturbation generator | Python / Octave | `pyrubberband`, `librosa`, `qaari_features.m` | No |
| Contrastive embeddings, S4/Mamba/NCDE | Python (Julia NCDE optional) | PyTorch, `torchcde`, `DiffEqFlux.jl` | Kaggle/Modal free tier, 5–20 GPU-h |

**Interface contract** (keeps collection and judgement separate, per the carry-over decision):
- Python writes per-instance evidence into the existing benchmark JSONL rows: GOP, sifa
  posteriors, embeddings as `.npy` sidecars.
- Julia reads the rows and writes
  - `calibration.json` v2: per-key predictive parameters, IRT item parameters, conformal
    thresholds, T, and
  - a loss matrix.
- `app/calibration.py` applies them. No audio is re-run when models change.

---

## 8. Prioritised build plan

Effort is in focused person-days. Acceptance tests are stated up front, per the carry-over rule of
honest reporting.

| # | Item | Effort | Depends | Acceptance test |
|---|---|---|---|---|
| **P0** | **Stop the MNAR bias (D2):** report the SKIPPED rate per reciter next to the index; add a REVIEW status; count low-confidence spans as evidence, not exclusions | 0.5 d | – | The skip rate is reported; on §6.6 letter-splice data the index **drops** monotonically with the number of splices (today it may rise) |
| **P1** | **Information health check:** verdict entropy per key, item information, KSG MI of each ruler vs its textbook contrast | 1 d | – | Flags `madd_iwad`, `hamzat_wasl`, `safir`, `jawaz_wajhayn`; ruler choice by MI agrees with or beats CV on peer LOO |
| **P2** | **Rasch/LLTM IRT in Turing.jl** on existing rows (binary, then graded); testlet per ayah; facet for mode | 3 d | P1 | Group-LOO elpd beats the key-only model; peers' θ intervals overlap Husary; imams' DIF items listed |
| **P3** | **Perturbation generator** (madd/ghunnah/qalqalah first, then splices) | 2 d | – | Measured Δcounts track the injected κ (r > 0.9); gives positives for P4–P6 |
| **P4** | **Surprisal scoring + Cauchy p-value combination** replacing `max z` and the linear score map; add CRC for expert false-FAIL ≤ 2 % | 2 d | roadmap A1/A5, P3 | Held-out peer false-FAIL ≤ 2 % (CRC guarantee); AUROC on P3 perturbations ≥ current z |
| **P5** | **muaalem-v3_2 + GOP-SF with the lahn-jali confusion set** (Kaggle T4) | 4 d | P3 | On splice data: jali recall ≥ 0.8 at expert false-alarm ≤ 1 %; `qdat_bench` madd-length MAE reported; leakage audit done |
| **P6** | **Temperature/vector scaling** of CTC and GOP posteriors; ECE before/after | 1 d | P5 | ECE < 0.05 on held-out perturbations |
| **P7** | **Bayes decision layer** with the jali/khafi loss matrix, IRT prior, and a recitation-level Poisson jali rate with the rule-of-three exam length | 2 d | P2, P4, P5 | Teachers review 30 flagged instances; loss matrix elicited (2 teachers); minimum exam length published |
| **P8** | **Hierarchical conformal + LTT** (two risks) per rule key | 2 d | P4 | Coverage on nested reciter-LOO within ±3 % of nominal |
| **P9** | **MIRT (4 dimensions) + ordinal grade** with Angoff standard setting; CAT item selector | 4 d | P2, P7, teacher input | Quadratic weighted κ vs teachers ≥ 0.7 on ≥ 40 graded students; CAT reaches SE < 0.25 in ≤ 60 % of fixed-test length (simulation) |
| **P10** | **Conditional anomaly detector** (Mahalanobis → kNN → flow) on SSL embeddings, conformalised | 3 d | P5 embeddings | False alarms on Tadabur expert reciters ≤ nominal; AUROC on P3 ≥ 0.85 |
| **P11** | **Channel-capacity / ΔKL jali index** per makhraj for feedback | 1 d | P5 | Stable (bootstrap CI) on references; increases with splice rate |
| **P12** | Contrastive fingerprint on Tadabur; IB nuisance heads | 5 d | – | mAP gain over ECAPA on ayah-disjoint splits |
| **P13** | S4/Mamba/NCDE prosody | 5+ d | P12 | Only if P12 shows sequence-level signal |

**Critical path to a defensible ijazah-level claim:**
P0 → P3 → P5 → P4/P6 → P2 → P7 → P8 → P9, about 5–6 focused weeks. The first three items (P0, P1,
P3) need no GPU and change what the current number means the most.

---

## 9. Risks and cautions

- **Authority.** The engine *estimates* the probability of lahn; it does not grant ijazah. The
  REVIEW action and the teacher-elicited loss matrix keep a qualified teacher in the loop by design.
- **Legitimate variation.** These are part of Hafs, not errors:
  - munfasil 4/5 (Shatibiyyah) vs 2 (Tayyibah qasr);
  - madd arid 2/4/6;
  - rawm/ishmam (ishmam is inaudible; the muaalem script excludes it).
  Model them as declared mixture components with a consistency requirement (§3.5), never as
  deviations from a single mean.
- **Leakage.** Both neural models were likely trained on the reciters we treat as references. Use
  imams and Tadabur reciters for *model* evaluation.
- **Style ≠ error.** One-class and IRT models can absorb maqam, tempo and school differences.
  Monitor `I(features; reciter)` and DIF.
- **Evidence double-counting** across correlated instances (the same ayah or tempo) inflates
  confidence. Testlet effects and N_eff tempering are mandatory, not optional.

---

## 10. Reproducibility note for §0 numbers
- **Data:** `benchmarks/results/local/everyayah_only_strategic.jsonl`: 491 rows, reciters Tablawi,
  Ayyoub, Budair, Matroud, Muaiqly; raw (uncalibrated) verdicts; 9 679 PASS/WARN/FAIL instances;
  PASS rate 0.49.
- **Model:** L2-regularised (λ = 0.5) logistic regression with reciter and rule-key dummies, fitted
  by Newton's method.
- **Deviance:**
  - McFadden R²: reciter-only 0.004, key-only 0.321, both 0.326.
  - Reciter effects (centred): Ayyoub +0.30, Tablawi +0.07, Muaiqly +0.05 (19 rows only),
    Budair −0.13, Matroud −0.30.
- **Item information:** mean p(1−p) per instance = 0.157 (max 0.25); 12/48 keys below 0.05.
- These are raw verdicts on 118 strategic ayahs. They show the *structure*: item effects dominate.
  They are not final reciter rankings.

---

## References (primary)
Information theory:
- Shannon 1948/1959;
- Cover & Thomas 2006;
- Blahut doi:10.1109/TIT.1972.1054855;
- Arimoto doi:10.1109/TIT.1972.1054753;
- Lin doi:10.1109/18.61115;
- Endres & Schindelin doi:10.1109/TIT.2003.813506;
- Wang–Kulkarni–Verdú doi:10.1109/TIT.2009.2016060;
- Kraskov et al. arXiv:cond-mat/0305641;
- Gao et al. arXiv:1709.06212;
- Rissanen doi:10.1016/0005-1098(78)90005-5;
- Grünwald & Roos arXiv:1908.08484;
- Watanabe arXiv:1208.6338;
- Tishby et al. arXiv:physics/0004057;
- Alemi et al. arXiv:1612.00410.

Scoring and psychometrics:
- Gneiting & Raftery doi:10.1198/016214506000001437;
- Bernardo doi:10.1214/aos/1176344689;
- Liu & Xie doi:10.1080/01621459.2018.1554485;
- Samejima doi:10.1007/BF03372160;
- De Boeck & Wilson doi:10.1007/978-1-4757-3990-9;
- Reckase doi:10.1007/978-0-387-89976-3;
- Bradlow et al. doi:10.1007/BF02294533;
- Bürkner arXiv:1905.09501;
- Bürkner & Vuorre doi:10.1177/2515245918823199;
- McCullagh JRSS-B 42:109 (1980);
- Berger doi:10.1007/978-1-4757-4286-2;
- Chow doi:10.1109/TIT.1970.1054406.

Calibration:
- Vovk et al. doi:10.1007/978-3-031-06649-8;
- Dunn et al. arXiv:1809.07441;
- Angelopoulos et al. arXiv:2208.02814, arXiv:2110.01052;
- Bates et al. arXiv:2104.08279;
- Barber et al. arXiv:1905.02928;
- Vehtari et al. arXiv:1507.04544;
- Säilynoja et al. arXiv:2103.10522;
- Guo et al. arXiv:1706.04599;
- Kumar et al. arXiv:1909.10155.

Neural / MDD:
- Witt & Young doi:10.1016/S0167-6393(99)00044-8;
- Hu et al. doi:10.1016/j.specom.2014.12.008;
- Gong et al. arXiv:2205.03432;
- Cao et al. GOP-SF arXiv:2507.16838;
- Parikh et al. arXiv:2506.02080;
- Shahin et al. arXiv:2311.07037;
- Abdelfttah et al. (muaalem / QPS) arXiv:2509.00094;
- El Kheir et al. QuranMB arXiv:2506.07722;
- Iqra'Eval ACL Anthology 2025.arabicnlp-sharedtasks.61;
- Quran-MD arXiv:2601.17880;
- Tadabur arXiv:2604.18932;
- Quranic Audio Dataset arXiv:2405.02675;
- wav2vec 2.0 arXiv:2006.11477;
- HuBERT arXiv:2106.07447;
- WavLM arXiv:2110.13900;
- w2v-BERT arXiv:2108.06209;
- Seamless arXiv:2312.05187;
- Whisper arXiv:2212.04356;
- MMS arXiv:2305.13516;
- Pasad et al. arXiv:2107.04734;
- SUPERB arXiv:2105.01051;
- Lee et al. arXiv:1807.03888;
- Sun et al. arXiv:2204.06507;
- Papamakarios et al. arXiv:1912.02762;
- RealNVP arXiv:1605.08803;
- Grathwohl et al. arXiv:1912.03263;
- SupCon arXiv:2004.11362;
- GE2E arXiv:1710.10467;
- ECAPA arXiv:2005.07143;
- Ganin et al. arXiv:1505.07818;
- S4 arXiv:2111.00396;
- Mamba arXiv:2312.00752;
- Neural ODE arXiv:1806.07366;
- Neural CDE arXiv:2005.08926;
- log-ODE arXiv:2009.08295.
