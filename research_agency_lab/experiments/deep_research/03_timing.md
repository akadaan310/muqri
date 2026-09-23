# Deep Research 03 — Timing: harakat, sukoon, shaddah, mudood, ghunnah, sakt, tempo and tasāwī

Scope: every durational object in Hafs 'an 'Āṣim recitation, from beginner ("101": is the madd
roughly 2/4/6?) to ijāzah level ("is every instance of each madd type held equally long across the
whole reading, is the chosen wajh of 'āriḍ held consistently, does the madd hierarchy hold, is the
pace one of the three marātib and is it balanced?"). Builds on `frontier_math_roadmap.md` (B1 SRVF,
B2 Hankel-DMD, B3 GP latent force, A5 hierarchical Bayes, A6 EVT) and makes those concrete for timing.

Engine touch-points read for this report: `app/acoustic/tempo.py` (beat = median of plain CV
syllables, IQR rejection), `app/taraweeh_adapter/pace_normalizer.py` (local beat = median in a
±4 s window; fixed-ms mirtaba thresholds 150/220 ms), `app/tajweed_rules/mudood_engine.py` (counts
= span_ms / local beat, absolute tolerances in harakāt), `app/tajweed_rules/sakt_wasl.py` (sakt in
**absolute ms** 200–400), `app/scoring.py:160` (`core_ms`, `core_counts` via `vowel_core_ms`),
`research_agency_lab/substrate_library/julia/src/Discovery.jl` (relaxation ODE, STLSQ duration law).

---

## 0. Findings from our own data (computed for this report, CPU, local jsonl only)

Source: `benchmarks/results/local/everyayah_only_strategic.jsonl` + `husary_strategic_raw.jsonl`,
studio mode, 59 strategic ayahs, 5 reciters. "Counts" = engine `metrics.counts`.

| reciter | beat (median ms) | tabīʿī median counts | muttaṣil / munfaṣil / lāzim / ʿāriḍ in **tabīʿī=2 units** | long-madd "collapsed" share (counts < ½ target) |
|---|---|---|---|---|
| Husary (anchor) | 320 | 2.19 | 4.9 / 5.5 / 6.0 / 4.6 | 0–23 % |
| Ayyoub | 220 | 1.91 | 10.5 / 9.5 / 10.2 / 8.7 | 0–33 % |
| Tablawi | 224 | 2.10 | 7.2 / 7.6 / 5.4 / 0.6* | 27–76 % |
| Budair | 181 | 2.11 | 0.8* / 2.5* / 0.8* / 2.1* | 32–60 % |
| Matroud | 231 | 2.00 | 1.6* / 3.1* / 5.1 / 1.2* | 25–64 % |

(* = dominated by collapsed spans, not a real reading.)

Five things follow, and they drive the priorities below:

1. **The tabīʿī madd is a better ruler than the short syllable.** On Husary the ratio to tabīʿī
   (tabīʿī≡2) gives muttaṣil 4.9, munfaṣil 5.5, lāzim 6.0, ʿāriḍ 4.6, līn 4.6 — almost textbook
   (4–5, 4–5, 6, 4 tawassuṭ). The raw short-syllable beat inflates everything by 2.19/2 = 1.095, which
   is exactly the `count_scale = 0.914` that `calibrate.jl` already learned (2/2.19 = 0.913).
   Independent confirmation that the calibration ruler is right and should be promoted into the beat itself.
2. **Measurement, not recitation, dominates long-madd variance for 4 of 5 reciters.** 25–76 % of
   long madds measure below half their target; for Husary with `align_conf>0.5` muttaṣil has robust
   CV **0.09** (n=15) and tabīʿī **0.14** (n=96). These are the first ijāzah-grade tasāwī reference
   numbers we have — but they exist only where alignment is confident. Every consistency model below
   must therefore be an **errors-in-variables mixture** (true-reading component + collapse component),
   or it will certify alignment noise as "lack of tasāwī". This is the characteristic CTC peaky-spike
   failure (Zeyer et al., arXiv:2105.14849): a long vowel is emitted as one or two spike frames with the
   rest absorbed by blank/neighbours.
3. **Relative madd scale differs ~2× across master reciters** (Ayyoub/Tablawi read muttaṣil ≈ 7–10
   tabīʿī-units in these ayahs; Husary ≈ 5). Absolute count bands from textbook are therefore a
   *pedagogical* verdict (anchor-referenced), while ijāzah verdicts must be **internal**: tasāwī
   (dispersion), ordinal hierarchy, and choice consistency — all scale-free.
4. **The SINDy duration law already found Klatt-type incompressibility.** Husary:
   `D_core = −552 + 150·(T/100) + 394·n` ms, R²=0.68, implied count scale 0.81: duration is affine,
   not proportional, in count. A negative intercept + sub-proportional slope is exactly Klatt's
   (1976) `D = D_min + k(D_inh − D_min)` shape seen from the other side (span boundaries eat into the
   held vowel). "Counts = D/T" is biased for large n; the inverse of the fitted law must be used.
5. **Tempo ODE could not be fitted** (`tempo_dynamics` pooled `{}`): strategic ayahs are scattered, no
   surah has ≥20 consecutive ayahs. Tempo dynamics need contiguous-surah runs (e.g. full Al-Mulk,
   Al-Kahf 1–30) — a data-collection item, not a modelling one.

---

## 1. The timing ontology (what must be checked, in beat units)

Notation: `T` = harakah (beat) in ms at time t; every target is in beats `c = D/T`.

| object | target (Hafs, Shāṭibiyyah default) | measured on | engine status |
|---|---|---|---|
| short vowel (ḥaraka) | 1 | vowel core of CV | used only to *estimate* T; **no check** of ishbāʿ / ikhtilās |
| sukūn consonant | ~½ (no vowel, no epenthesis) | consonant span | only via qalqalah; **no duration check** |
| shaddah (non-nasal) | geminate ≈ 2× singleton (sukūn + ḥaraka) | closure/frication | **missing** |
| ghunnah (mushaddad nūn/mīm, idghām bi-ghunnah, ikhfāʾ, iqlāb) | 2 | nasal murmur core | present (`core_counts` Husary median 3.38 raw ⇒ 3.1 tabīʿī-units: check core definition) |
| madd ṭabīʿī, badal, ʿiwaḍ, ṣila ṣughrā | 2 | vowel core | present |
| muttaṣil | 4–5 (Shāṭibiyyah), *≥ munfaṣil* | vowel core | present, absolute band |
| munfaṣil, ṣila kubrā | 4–5 Shāṭibiyyah / 2 Ṭayyibah qaṣr | vowel core | present |
| lāzim (kalimī/ḥarfī, muthaqqal/mukhaffaf) | 6 | vowel core | present |
| ʿāriḍ li-s-sukūn | 2 / 4 / 6 free, **consistent** | vowel core at waqf | present, consistency **missing** |
| līn (ʿāriḍ) | 2 / 4 / 6, **≤ chosen ʿāriḍ** | diphthong core at waqf | present, relation **missing** |
| sakt (4 places) | ≈ 2 beats silence, no breath | silence + breath detector | present in **absolute ms** (200–400) — should be beats |
| waqf pause / ibtidāʾ | free; breath allowed | silence | pause detection exists |
| mirtaba (taḥqīq/tadwīr/ḥadr) | reciter-constant regime | beat level & drift | fixed thresholds 150/220 ms |
| tasāwī al-mudūd | equal length within each type | dispersion across recitation | **missing** |
| hierarchy (aqwā al-sababayn) | lāzim ≥ muttaṣil ≥ ʿāriḍ ≥ munfaṣil ≥ badal | ordinal | **missing** |
| rhythm balance | stable beat, balanced ḥarakāt | syllable-duration process | **missing** |

The hierarchy ordering above is the common teaching of "أقوى المدود" (lāzim, then muttaṣil, then
ʿāriḍ, then munfaṣil, then badal) and the līn ≤ ʿāriḍ relation; both must be implemented as a
**configurable policy table** (`app/data/timing_policy.json`) reviewed by the project's shaykh,
since teachers differ on the corner cases (e.g. ʿāriḍ 2 with munfaṣil 4).

---

## 2. Measurement layer first: errors-in-variables for aligned durations

Everything downstream consumes `D_obs`. Model it explicitly.

### 2.1 Boundary-noise model
A segment duration is the difference of two boundary estimates `b̂ = b + η`:
`D_obs,i = D_i + η_{i+1} − η_i`.
With iid boundary noise of variance σ_b² and frame quantisation Δ (=20 ms here, `align_min_ms`),
`Var(D_obs,i) = Var(D_i) + 2σ_b² + Δ²/6`, and **adjacent segments share a boundary**, so

`Cov(D_obs,i, D_obs,i+1) = Cov(D_i, D_{i+1}) − σ_b² − Δ²/12`.

This is the Wing–Kristofferson (1973, *Percept. Psychophys.* 14:5, doi:10.3758/BF03198607) identity
from motor timing (clock + motor delay ⇒ negative lag-1 autocovariance) transplanted to alignment.
**Identification:** on runs of plain CV syllables (the ones `short_syllable_units` already selects,
true durations nearly independent after detrending tempo), `σ̂_b² = −γ̂(1) − Δ²/12` with `γ̂(1)` the
lag-1 autocovariance of detrended durations. This gives a per-recording, per-aligner measurement-error
estimate *with no hand labels*. Validate once against ~200 hand-marked Husary boundaries in Praat.

### 2.2 Collapse mixture (CTC peakiness)
`c_obs = c_true·(1 + ε)` with prob `1−π_i`, and `c_obs ~ Uniform/Beta on (0, ½·c_target)` with prob π_i;
`logit π_i = α₀ + α₁·align_conf_i + α₂·log(align_min_ms_i) + α₃·[rule is long madd]`.
Fit by EM across the reference set; the posterior `P(collapse | data)` becomes a gate: instances with
`P>0.5` are **SKIPPED as unmeasurable** instead of FAIL, and are removed from tasāwī statistics.
Current data show `corr(align_conf, counts)` is weak (−0.3…+0.4), so `align_conf` alone is not a gate;
the mixture uses the *value* too.

**Better cure (root cause):** re-measure long vowels with a *signal* criterion instead of CTC span:
steady-state detection on F0/energy/formant stationarity inside a widened window (the existing
`vowel_core_ms` with a 10 dB drop is the right direction; widen the search window to the neighbouring
unit boundaries when the CTC span is < 1 beat). Or re-align with an **explicit-duration HSMM** (§4.3).

### 2.3 Propagation to counts
`log c = log D − log T`. With T the median of n syllables of log-SD s,
`Var(log T̂) ≈ (π/2)·s²/n` (median efficiency 2/π). Then
`SE(c) ≈ c·√(Var(log D_obs) + (π/2)s²/n)`, with `Var(log D_obs) ≈ (2σ_b²+Δ²/6)/D²`.
At ḥadr (T≈110 ms), a tabīʿī (D≈220 ms) with σ_b=10 ms has SE ≈ 0.14 beats — comparable to the 0.25
tolerance. **Tolerances must be heteroscedastic**: replace the fixed `TOLERANCE` table in
`mudood_engine.py` by `tol_i = √(τ_k² + z²·SE_i²)` (τ_k = reference within-type spread, §5).

### 2.4 Robust regression for laws under noisy covariates
The duration law regresses D on n·T with T itself noisy → attenuation bias (regression dilution).
Use Deming / orthogonal regression with variance ratio from §2.1, or SIMEX (Cook & Stefanski 1994,
JASA 89:1314, doi:10.1080/01621459.1994.10476871); in the Bayesian model (§5) put T latent.

---

## 3. The beat (harakah) — a better estimator for `tempo.py`

### 3.1 Two-ruler estimator
Current: `T_syll = median(CV durations)`. The data (§0.1) show that the stable ruler is the tabīʿī.
Model on log scale, one ayah a:

`log D_CV,j = log T_a + log ρ_CV + e_j`,  `log D_tab,j = log T_a + log 2 + e'_j`,

with ρ_CV (CV-syllable length in beats, ≈1.1 for Husary) a **reciter-level** parameter learned from
reference data. The combined estimator is the precision-weighted average of the two log-medians
(robust: Hodges–Lehmann or Huber M-estimate instead of mean). Plug: `estimate_tempo` returns
`T` and `se_logT`; add `method="two_ruler"`; keep `count_scale` in calibration as a check (should → 1).

*Caveat:* using tabīʿī to set T makes tabīʿī counts ≡ 2 by construction for that ayah; use a
**leave-one-out** ruler per instance (exclude the instance under test), and keep a CV-only estimate for
judging tabīʿī itself. For a learner, T must come from the learner, never from the reference.

### 3.2 Log-normal / gamma duration distributions
Segment durations are right-skewed; log-normal is standard (Rosen 2005, *Clin. Linguist. Phon.*
19:597; Crystal & House 1988, JASA 83:1553 used gamma). Model choice per segment class by LOO
(`Distributions.jl` fit_mle for LogNormal, Gamma, InverseGaussian; Octave `lognfit`, `gamfit`).
Consequence: work in **log-counts**; bands become multiplicative (e.g. tabīʿī 2 ×/÷ 1.14), which matches
the observation that spread scales with length (Weber-like timing; Gibbon's scalar timing, *Psychol.
Rev.* 84:279, 1977).

### 3.3 Continuous local beat: OU state-space smoother (replaces ±4 s window median)
Let `x(t) = log T(t)`. Ornstein–Uhlenbeck:
`dx = κ(μ − x) dt + σ dW` (Uhlenbeck & Ornstein 1930, *Phys. Rev.* 36:823).
Exact discretisation between observation times t_k, t_{k+1} (Δ_k):
`x_{k+1} = μ + e^{−κΔ_k}(x_k − μ) + w_k`,  `w_k ~ N(0, σ²(1−e^{−2κΔ_k})/(2κ))`.
Observation per plain syllable j: `y_j = log D_j − log ρ_CV = x(t_j) + v_j`,
`v_j ~ t_ν(0, s_v²)` (heavy-tailed; handle via iteratively reweighted Kalman or a Student-t filter),
with `s_v²` ≥ the boundary-noise term from §2.1.
- Kalman filter + RTS smoother gives `T(t)` **and its variance** at every madd location → feeds §2.3.
- Parameters (μ, κ, σ, s_v) by maximum likelihood (Kalman innovations likelihood) per recording, with
  hierarchical priors from reference reciters.
- `κ⁻¹` = tempo memory (s); `σ²/(2κ)` = stationary tempo variance = a **tempo-balance (i'tidāl)** metric.
- Relation to the existing ODE: the ODE `dT/dτ = κ(T∞ − T) + φ` is the drift of this SDE; the SDE adds
  the diffusion that the ODE forces into residuals. Add φ as a linear drift in μ(t) = μ₀ + φt
  (fatigue/acceleration). This is also the simplest latent-force model (roadmap B3 with a white-noise
  force); Matérn-3/2 GP (`TemporalGPs.jl`, state-space O(n)) is the smooth-velocity upgrade.
- Plug: `pace_normalizer.LocalTempo.__call__` → `OUTempo.__call__(t) -> (T, se)`; `EvalContext.haraka_ms`
  unchanged signature, add `haraka_se(t)`.

### 3.4 Mirtaba classification (taḥqīq/tadwīr/ḥadr)
Replace fixed 150/220 ms thresholds by a 3-state Gaussian HMM on `x(t)` with state means learned from
reference reciters labelled by mirtaba (EveryAyah has Husary murattal vs. mujawwad vs. Ḥadr sets), with
sticky transitions (Fox et al. sticky HDP-HMM, arXiv:0905.2592, for the "regimes rarely switch" prior).
Output: regime posterior per ayah, and a **regime-consistency** flag (switching mirtaba mid-passage is
a balance fault at ijāzah level; Tarāwīḥ mode tolerates it).

---

## 4. Per-object duration models

### 4.1 Madd — counts via the inverse of a calibrated law
Law (from §0.4, Klatt 1976 JASA 59:1208, doi:10.1121/1.380986):
`D = a_r + b_r·n·T^{γ_r}·exp(ε)` (a_r ≤ 0 absorbs span-boundary loss; γ≈1 means pure beat scaling).
Counts reported to the learner: `ĉ = ((D − a_r)/b_r)/T^{γ_r}`, with `(a,b,γ)` from the **anchor** for
pedagogical verdicts and from the **learner's own tabīʿī** (§3.1) for internal checks. Refit the law with
`SymbolicRegression.jl` (Cranmer, arXiv:2305.01582) over operators {+,−,×,/,^,log} with features
(n, T, phrase_final, word_position, pre-pausal, madd type one-hot, align_conf) and a complexity-penalised
Pareto front; keep STLSQ (`DataDrivenSparse.jl`, Brunton et al. PNAS 113:3932, 2016,
doi:10.1073/pnas.1517384113) as the linear baseline. Accept a symbolic law only if it wins held-out
(surah-disjoint) MAE and is stable across ≥3 reference reciters.

### 4.2 Discrete choice madds (ʿāriḍ, līn, munfaṣil 4/5 vs Ṭayyibah 2)
Latent-class model per reciter/recitation:
`c_i | z_i=m ~ LogNormal(log(m·s_r), σ_m²)`, `m ∈ {2,4,6}` (ʿāriḍ), `z_i ~ Categorical(π)`,
`π ~ Dirichlet(α)`, plus the collapse component (§2.2).
- **Choice**: `m* = argmax π_m`; reported "you chose tawassuṭ (4)".
- **Choice consistency (ijāzah)**: `H(π)` (entropy) and posterior `P(z_i ≠ m* | c_i)` per instance →
  instance-level "inconsistent wajh" flags. A reading is consistent iff `P(π_{m*} > 0.9 | data) > 0.95`.
- **Līn ≤ ʿāriḍ** and **muttaṣil ≥ munfaṣil**: order constraints between latent class means
  (§5.4).
- Borderline instances (c between classes) are what the tolerance currently mis-scores: the mixture
  gives a soft assignment instead of "nearest of 2/4/6" in `mudood_engine.py`.
- Julia: Turing.jl with `MixtureModel` and marginalised z (no discrete sampling), or EM in 40 lines.

### 4.3 Explicit-duration HSMM (count decoding directly from audio)
For a rule span, states = [onset consonant, held vowel, following consonant]; the held-vowel state has
duration pmf `p(u) = DiscretisedLogNormal(log(n·T(t)/Δ), σ²)`; emissions from the aligner's frame
posteriors (or MFCC GMM). Explicit-duration Viterbi/forward (Yu 2010, *Artif. Intell.* 174:215,
doi:10.1016/j.artint.2009.11.011; Ostendorf et al. 1996 segment models) gives:
(a) a re-segmentation robust to CTC peakiness; (b) **Bayes factors between count hypotheses**
`log p(audio | n=4) − log p(audio | n=6)`, a verdict that does not pass through a hard boundary.
Complexity O(frames × D_max) per span ≈ trivial on CPU. Julia: `HiddenMarkovModels.jl` (Dalle, JOSS 2024)
does HMMs; the explicit-duration forward is ~80 lines custom. Octave cross-check: same recursion.

### 4.4 Short vowels: ishbāʿ and ikhtilās (missing today)
Two-class self-calibrated model on log duration of **vowel cores** (not CV syllables):
`log d | short ~ N(μ_S, σ_S²)`, `log d | long ~ N(μ_L, σ_L²)` with μ_L from the learner's tabīʿī cores and
μ_S from plain short vowels (leave-one-out). For each short-vowel instance compute
`LR = p(d|long)/p(d|short)`; **ishbāʿ** if `log LR > λ` *and* a spectral cue agrees (ḍamma→ū / kasra→ī
formant trajectory reaching the long-vowel target, from `app/sifaat/formants.py`). **Ikhtilās/khaṭf**
(vowel reduced) if `d < q_{0.02}` of the short-class predictive and/or vowel centralisation. Contexts
with legitimate short-vowel lengthening are excluded: phrase-final (converted to ʿāriḍ), ṣila
contexts, pre-pausal. Arabic phonetics support the approach: vowel-length contrast is robust (long/short
≈ 1.6–2.2 in MSA/Classical Arabic productions; e.g. Tsukada 2009, and HMM duration-based vowel-length
studies on Qurʾānic chapters); in murattal recitation the short CV syllable is the rhythmic "pulse".
Plug: new validator `app/tajweed_rules/harakat_engine.py`, rule type `HARAKA_LENGTH` (details
`ishbaa`, `ikhtilas`), exclusion list shared with `short_syllable_units`. Circularity guard: the
estimator in §3.1 must exclude flagged instances (two-pass: estimate → flag → re-estimate).

### 4.5 Shaddah on non-nasal letters (missing today)
Phonetics: Arabic geminates ≈ 2× singleton closure/frication duration, with a shortened preceding
vowel (Khattab & Al-Tamimi 2014, *Lab. Phonol.* 5(2):231, doi:10.1515/lp-2014-0009; Al-Tamimi &
Khattab 2015, JASA 138:344, doi:10.1121/1.4922514). Tajweed defines shaddah as two letters (sākin +
mutaḥarrik), i.e. an added ~½–1 beat of consonant hold.
Model per manner class k (stop, fricative, liquid, glide; nasals handled by ghunnah):
`log D_gem,i − log T(t_i) = log ρ_k + β·x_i + e_i`, reference `ρ_k` and spread from Husary + peers;
a **within-recitation contrast** check `log(D_gem/D_sing)` against same-letter singletons (the learner's
own), which is scale-free. Measurement: stops → closure (silence/low-energy) duration from the energy
envelope between the two vowel cores; fricatives → frication (high-band energy) duration; ل/ر → sonorant
steady-state. Failure modes: *takhfīf* (geminate read as singleton: ratio ≈ 1) and over-hold (ratio > 3,
or an epenthetic vowel = splitting). Qalqalah on a mushaddad letter at waqf (e.g. الحقّ) is joint with
`qalqalah_engine.py`. Plug: `app/tajweed_rules/shaddah_engine.py`, rule `SHADDAH` (parser already marks
`u.shadda`).

### 4.6 Sukūn
Sākin consonant hold in beats `D/T ∈ [0.3, 0.8]` (reference-calibrated); failure = *taḥrīk* (an
epenthetic vowel: voiced periodic segment with formants between two consonants, detectable as a
vowel-core > 40 ms) — this is also how ishbāʿ of sukūn manifests. Ties to `app/sifaat/sukoon_spectrum.py`.

### 4.7 Ghunnah
Same machinery as madd with n=2 and the nasal-murmur core (low F1, nasal formant ~250 Hz, anti-formant).
Our data: Husary ghunnah core ≈ 3.1 tabīʿī-units — either Husary's ghunnah is genuinely longer than the
textbook 2 (common in mujawwad-leaning murattal) or the nasal core includes the transition; calibrate the
target against the anchor (the calibration file already carries `core_counts` centre 3.23) rather than
hard-coding 2. Tasāwī applies to ghunnah too (§5).

### 4.8 Sakt and pauses
Sakt target in **beats**: `silence/T ∈ [1.5, 2.5]` (≈ 2 ḥarakāt, the standard teaching "سكتة لطيفة
بمقدار حركتين"), no breath. Current 200–400 ms is right only near T≈150 ms; at Husary's T=320 ms a correct
2-beat sakt (640 ms) is flagged as "full stop". Change `SAKT_MS` to beat-relative with the ms band as a
fallback. Pauses (waqf) are a **point process** (§6.3); breath-at-waqf probabilities and pause-length
distributions come from the reference set.

---

## 5. Tasāwī al-mudūd — consistency models (the ijāzah core)

### 5.1 Definition
For madd type k in one recitation, the tasāwī parameter is the **within-type, within-recitation
spread of log-counts after removing only Tajweed-legitimate covariates** (tempo is already divided out;
phrase-final/waqf changes the *type*, e.g. to ʿāriḍ, so it is not a covariate; the legitimate ʿāriḍ
choice is handled by §4.2). Anything else — position in ayah, word length, fatigue — must stay *inside*
σ_k, because the ijāzah requirement is exactly that it should not matter.

### 5.2 Variance-component model (hierarchical, partial pooling)
For instance i of type k in ayah a, surah s, reciter r:

`y_i = log c_i = μ_{r,k} + u_{r,s,k} + v_{r,a} + e_i + m_i`
- `u_{r,s,k} ~ N(0, ω_k²)` (between-surah drift of the type's length),
- `v_{r,a} ~ N(0, ψ²)` (ayah-level tempo mis-estimate shared by all rules in the ayah — absorbs residual
  beat error; identifiable because many types share the ayah),
- `e_i ~ t_ν(0, σ_{r,k}²)` — **σ_{r,k} is the tasāwī parameter**,
- `m_i ~ N(0, SE_i²)` known measurement variance from §2.3 (errors-in-variables; deconvolves
  aligner noise out of σ), plus the collapse mixture of §2.2.
- Partial pooling: `log σ_{r,k} = λ_k + δ_r + ζ_{r,k}` so types with 6 instances borrow strength from the
  reciter's other types (reciters who are consistent tend to be consistent across types).
- Reference distribution: posterior predictive of σ for a *new reference-like reciter*
  (`δ_new ~ N(0, s_δ)`), i.e. roadmap A5 applied to dispersion instead of location.
Fit: `MixedModels.jl` (REML, fast, Gaussian — first pass, no measurement error) → `Turing.jl` NUTS
(non-centred; Betancourt & Girolami arXiv:1312.0906) for the full model. Priors on scales: half-t /
exponential (Gelman 2006, *Bayesian Anal.* 1:515, doi:10.1214/06-BA117A).
Size: ≤ 10⁴ instances, ~50 parameters/reciter → minutes on CPU; no GPU.

Reference numbers to start priors (Husary, confident alignments): σ(log c) ≈ 0.09 muttaṣil, ≈0.14
tabīʿī, ≈0.15 ʿāriḍ. A learner with σ_muttaṣil = 0.25 holds some muttaṣil at 4 and others at 6.

### 5.3 Verdicts: equivalence testing, not difference testing
"Consistent" is a claim of *equivalence*, so the null must be *non*-consistency:
- **Dispersion TOST** (Schuirmann 1987, *J. Pharmacokinet. Biopharm.* 15:657, doi:10.1007/BF01068419;
  Lakens 2017, *SPPS* 8:355, doi:10.1177/1948550617697177): H0: `σ_learner,k / σ_ref,k ≥ θ_k` vs
  H1: `< θ_k`, θ_k = reference-predictive 90th percentile ratio. Test on log scale with a bootstrap or
  posterior interval; "tasāwī achieved" iff the 90 % upper bound < log θ_k. With few instances the verdict
  is honestly "insufficient evidence" instead of PASS/FAIL.
- **Location TOST between halves / surahs**: `|μ_first − μ_second| < Δ` (Δ = 0.5 count) — checks the
  type's length did not drift.
- **Bayesian equivalent**: `P(σ_{r,k} < σ*_k | data) > 0.95` using §5.2 posterior; ROPE logic (Kruschke
  2018, *AMPPS* 1:270, doi:10.1177/2515245918771304).
- Multiple types → Holm or hierarchical shrinkage already handles it in the Bayesian version.

### 5.4 Ordinal hierarchy & cross-type rules (order-restricted inference)
Constraints (policy table): `μ_lāzim ≥ μ_muttaṣil ≥ μ_munfaṣil`, `μ_muttaṣil ≥ μ_ʿāriḍ-choice?` (policy),
`μ_līn ≤ μ_ʿāriḍ`, badal = tabīʿī in Hafs. Test: unconstrained vs isotonic (PAVA) fit likelihood ratio,
chi-bar-squared null (Robertson, Wright & Dykstra 1988; Silvapulle & Sen 2005), or directly
`P(μ_munfaṣil > μ_muttaṣil + δ | data)` from the posterior. Instance-level: a munfaṣil held longer than the
preceding muttaṣil in the same ayah is the classic ijāzah correction and should be named as such.

### 5.5 Drift and change points
Sequence per type k in recitation order: `y_1..y_N` (log counts, collapse-gated).
- **Offline**: PELT (Killick, Fearnhead & Eckley 2012, JASA 107:1590, doi:10.1080/01621459.2012.737745)
  with a Normal-mean/variance cost and penalty MBIC; also on the OU tempo residuals. `Changepoints.jl`
  (PELT) or 60-line implementation; Octave cross-check implementation.
- **Online (live/Tarāwīḥ)**: BOCPD (Adams & MacKay 2007, arXiv:0710.3742) with Normal–Gamma conjugate
  model; run-length posterior → "your munfaṣil changed from 4 to 5 around ayah 23".
- **Control charts** for teaching dashboards: EWMA (Roberts 1959, *Technometrics* 1:239) with λ=0.2 and
  limits from reference σ_k; CUSUM (Page 1954, *Biometrika* 41:100) for small persistent shifts. Chart
  limits are the reference-predictive quantiles (conformal, roadmap A4), not ±3σ Gaussian.

### 5.6 Tasāwī as elastic-FDA phase variance (roadmap B1 made concrete)
The Qurʾān gives a **canonical time axis**: cumulative expected beats `b` (what
`expected_total_harakat` computes, extended per unit). The recitation is a monotone map `t(b)`; its slope
`dt/db = T(b)` is the local beat and a madd of target n occupies `[b, b+n]`. Registering `t(b)` across
reciters with SRVF (Srivastava et al. arXiv:1103.3817; Tucker et al. arXiv:1212.1791; `fdasrsf` or a
Julia port via `PythonCall.jl`) decomposes timing into amplitude (global tempo level) and phase (local
rubato). Per madd, `∫_{madd} (dt/db)/T̄ db − n` is the **count error in a warp-aware frame**; tasāwī is the
variance of these residuals for the type. Because the canonical axis is known (text = score), this is
closer to **score following** than blind registration — a constrained monotone spline or a
`DynamicAxisWarping.jl` DTW (Sakoe & Chiba 1978; soft-DTW Cuturi & Blondel arXiv:1703.01541) against the
beat grid suffices for the first version.

---

## 6. Rhythm, balance and tempo dynamics

### 6.1 Rhythm metrics in beat units
On syllable/interval durations **divided by T(t)** (so metrics are tempo-free):
- `nPVI = 100/(m−1) Σ |d_k − d_{k+1}| / ((d_k + d_{k+1})/2)` (Grabe & Low 2002, *Lab. Phonol.* 7,
  doi:10.1515/9783110197105.515), computed separately over **plain short syllables** (should be low:
  balanced ḥarakāt) and over all vocalic intervals (driven by madd pattern, text-determined).
- `%V, ΔV, ΔC` (Ramus, Nespor & Mehler 1999, *Cognition* 73:265, doi:10.1016/S0010-0277(99)00058-X);
  `VarcoV = 100·ΔV/mean V` (White & Mattys 2007, *J. Phon.* 35:501).
- Key point: most of the variance in these metrics is **text-determined** (which syllables are madd).
  The balance metric that matters is the **residual** rhythm: nPVI of `d_k / d̂_k` where `d̂_k` is the
  expected duration of unit k from the calibrated law. `nPVI_resid → 0` is perfect i'tidāl.
  Rhythm metrics are known to be rate- and text-sensitive (Arvaniti 2012, *J. Phon.* 40:351), which is
  precisely why the residual form is required.

### 6.2 Renewal-process view of the pulse
Onsets of plain CV syllables as a renewal process with gamma inter-onset intervals in beats:
shape k̂ measures pulse regularity (CV = 1/√k). Wing–Kristofferson decomposition (clock vs motor/boundary
variance) from lag-1 autocovariance (§2.1) separates **reciter timing variance** from **aligner noise**.
Report the clock-variance component as the rhythm-stability score.

### 6.3 Pauses and breaths as a point process
Waqf/breath events on the recited-time axis: inhomogeneous Poisson with intensity depending on
syntactic waqf marks (ۖ ۗ ۚ ۘ), phrase length since last breath, and fatigue; **Hawkes** self-excitation
(Hawkes 1971, *Biometrika* 58:83) captures breath clustering in fatigue (Tarāwīḥ). Use: (i) detect
abnormal breath placement (breath where waqf is qabīḥ is a Tajweed fault, handled elsewhere) and
(ii) model inter-breath phrase length vs. mirtaba. Custom log-likelihood (~40 lines) + `Optim.jl`.

### 6.4 Tempo dynamics: ODE → SDE → UDE
- Existing: `dT/dτ = κ(T∞ − T) + φ` (`Discovery.jl`).
- Upgrade 1 (§3.3): OU/SDE in log T; `StochasticDiffEq.jl` for simulation and posterior predictive checks;
  inference by Kalman likelihood (exact for linear SDE) — no need for SDE solvers in the fit.
- Upgrade 2: **Universal differential equation** (Rackauckas et al. arXiv:2001.04385):
  `dx/dτ = κ(μ − x) + NN_θ(τ, ayah_len, waqf_density, surah_position)` trained with `Lux.jl` +
  `SciMLSensitivity.jl` + `Optimization.jl` on all reference surahs; then **sparsify NN_θ** with
  SINDy (`DataDrivenDiffEq.jl`) or `SymbolicRegression.jl` to recover an interpretable forcing law
  (e.g. slowing before sajdah verses / at surah end). Tiny network (2×16), CPU.
- Upgrade 3 (roadmap B2): Hankel-DMD of `x(t)` + energy + F0 for quasi-periodic phrasing; eigenvalue
  moduli <1 quantify tempo settling (Arbabi & Mezić arXiv:1611.06664).
- Neural ODEs (Chen et al. arXiv:1806.07366) as pure black boxes are **not** recommended: few surahs,
  need interpretability; UDE+SINDy dominates.

---

## 7. Calibration against reference reciters

1. Anchor Husary + ijāzah peers provide: `ρ_CV,r` (§3.1), law `(a,b,γ)` (§4.1), OU hyperpriors (§3.3),
   σ_k reference predictive (§5.2), shaddah ratios ρ_k (§4.5), ʿāriḍ class spread σ_m (§4.2), sakt
   beat band (§4.8), residual-nPVI band (§6.1).
2. **Only confident, non-collapsed instances** enter calibration (collapse posterior < 0.2).
3. Bands are posterior-predictive for a new reference-like reciter (A5), thresholds from peer-LOO
   conformal quantiles (A4), tail via GPD where n allows (A6).
4. Scale-free checks (tasāwī ratio, hierarchy, choice consistency, geminate/singleton ratio,
   residual nPVI) are calibrated per *reciter-relative* quantity so that Ayyoub's long madd style is not
   penalised while Husary-referenced pedagogical bands remain available as a separate "textbook" lens.

## 8. Validation protocol

- **Synthetic ground truth (highest value, CPU)**: take confident Husary madds and time-stretch only the
  held vowel core by known factors (PSOLA/WSOLA; Octave or `librosa`), producing known counts
  {1.5,2,3,4,5,6,7} and known inconsistency patterns (σ injected). Metrics: count MAE, tasāwī-detection
  ROC vs injected σ, ʿāriḍ-choice switch detection latency (BOCPD). Also shaddah shortening (splice out
  closure) and ishbāʿ (stretch short vowels ×1.8).
- **Boundary truth**: 200 hand-labelled Husary madd/shaddah boundaries (Praat) → σ_b, validates §2.1.
- **Peer LOO coverage**: each ijāzah peer must PASS tasāwī and hierarchy at ≥95 % (conformal).
- **Impostor separation**: Tarāwīḥ imams and synthetic-perturbed reciters must be separated (AUC).
- **Calibration**: LOO-PIT of the §5.2 model (Vehtari et al. arXiv:1507.04544).
- **Tempo**: one-step-ahead predictive log score of OU vs window-median on contiguous surahs.
- **Octave cross-checks** (`research_agency_lab/substrate_library/octave/`): `timing_ou_kalman.m`,
  `timing_pelt.m`, `timing_npvi.m`, `timing_wk_boundary.m` reproduce Julia numbers on the same CSV to
  1e-6 (deterministic parts) — mirrors the existing `lpc.m` / `qaari_features.m` pattern.

## 9. Julia implementation map (SciML and friends)

| component | packages | file |
|---|---|---|
| boundary noise σ_b (WK), collapse mixture EM | Statistics, Distributions.jl, Optim.jl | `src/Timing.jl` |
| log-normal/gamma selection | Distributions.jl | `src/Timing.jl` |
| OU Kalman/RTS smoother + MLE | hand-written (or LowLevelParticleFilters.jl), ForwardDiff.jl, Optim.jl; StochasticDiffEq.jl for PPC | `src/Tempo.jl` |
| mirtaba HMM | HiddenMarkovModels.jl | `src/Tempo.jl` |
| duration law | DataDrivenSparse.jl (STLSQ, exists), SymbolicRegression.jl | `src/Discovery.jl` |
| explicit-duration HSMM | custom forward/Viterbi | `src/HSMM.jl` |
| tasāwī variance components | MixedModels.jl (REML), Turing.jl (NUTS) | `src/Tasawi.jl` |
| TOST / bootstrap | Bootstrap.jl, HypothesisTests.jl | `src/Tasawi.jl` |
| hierarchy (PAVA) | hand-written | `src/Tasawi.jl` |
| PELT / BOCPD / EWMA | Changepoints.jl or custom | `src/Drift.jl` |
| UDE tempo | Lux.jl, SciMLSensitivity.jl, Optimization.jl, OrdinaryDiffEq.jl | `src/Discovery.jl` |
| SRVF / DTW | fdasrsf via PythonCall.jl; DynamicAxisWarping.jl | `src/Registration.jl` |
| DMD | DataDrivenDiffEq.jl | `src/Discovery.jl` |

Minimal OU filter (log-beat), the core new primitive:

```julia
function ou_filter(t, y, μ, κ, σ, sv)          # t sorted times (s), y = log D − log ρ
    m, P = μ, σ^2/(2κ); ll = 0.0; ms = similar(y); Ps = similar(y)
    for k in eachindex(y)
        if k > 1
            a = exp(-κ*(t[k]-t[k-1])); m = μ + a*(m-μ); P = a^2*P + σ^2*(1-a^2)/(2κ)
        end
        S = P + sv^2; K = P/S; r = y[k]-m
        ll += -0.5*(log(2π*S) + r^2/S); m += K*r; P *= (1-K); ms[k] = m; Ps[k] = P
    end
    return ms, Ps, ll                             # add RTS backward pass for smoothing
end
```

Tasāwī model sketch (Turing, measurement error + t-likelihood):

```julia
@model function tasawi(y, se, k, s, K, S)
    λ ~ filldist(Normal(log(0.12), 0.5), K)           # log σ_k prior centred on Husary-like
    ω ~ filldist(Exponential(0.1), K); μ ~ filldist(Normal(0, 1), K)
    zs ~ filldist(Normal(), S, K)                      # non-centred surah effects
    ν ~ Gamma(2, 0.1) ; ytrue ~ arraydist([TDist(ν+2)*exp(λ[k[i]]) + μ[k[i]] + ω[k[i]]*zs[s[i],k[i]] for i in eachindex(y)])
    y .~ Normal.(ytrue, se)
end
```

**GPU needs: none.** All models are ≤10⁴ observations; NUTS in minutes on CPU; SymbolicRegression.jl
uses multi-threaded CPU; UDE is a 2×16 network. The only GPU-relevant step is re-running the
wav2vec2/CTC aligner on more audio, which the engine already does.

## 10. Engine integration (Python side, read-only proposals)

| change | file / function |
|---|---|
| `estimate_tempo` → two-ruler log-scale estimator, returns `se_logT`; LOO ruler per instance | `app/acoustic/tempo.py` |
| `LocalTempo` → OU smoother with variance; mirtaba HMM replaces `classify_pace` thresholds | `app/taraweeh_adapter/pace_normalizer.py` |
| heteroscedastic tolerance; inverse duration law; soft 2/4/6 class posterior for ʿāriḍ/līn; collapse gate → SKIPPED | `app/tajweed_rules/mudood_engine.py` |
| sakt band in beats | `app/tajweed_rules/sakt_wasl.py` (`SAKT_MS`) |
| new `harakat_engine.py` (ishbāʿ/ikhtilās, sukūn taḥrīk), `shaddah_engine.py` | `app/tajweed_rules/` + `RuleType` in `app/models.py` |
| recitation-level post-pass: tasāwī, hierarchy, ʿāriḍ consistency, drift, rhythm residual nPVI | new `app/consistency.py`, called after per-rule scoring in `app/pipeline.py` |
| reference parameters | `app/data/calibration.json` (new `timing` block written by `calibrate.jl`) |

Python runtime needs only closed-form pieces (Kalman loop, TOST on bootstrap, PAVA, BOCPD, mixture
posterior with fixed parameters); all fitting stays in Julia and ships as JSON parameters.

## 11. Prioritised build plan

| # | item | why | effort |
|---|---|---|---|
| 1 | Collapse mixture gate + WK boundary-noise estimate (§2) | 25–76 % of long madds are unmeasurable for 4/5 reciters; nothing else is trustworthy until gated | 1–1.5 d |
| 2 | Tabīʿī-anchored two-ruler beat with `se` (§3.1) + heteroscedastic tolerances (§2.3) | empirically validated (Husary ratios textbook-exact; matches count_scale 0.914) | 1 d |
| 3 | Sakt in beats (§4.8) | one-line class of false "full stop" verdicts at taḥqīq | 1 h |
| 4 | Tasāwī v1: per-type robust log-CV with measurement-error deconvolution + bootstrap TOST vs Husary-predictive; ʿāriḍ 2/4/6 mixture + choice consistency; hierarchy PAVA check (§4.2, 5.3, 5.4) | the core ijāzah gap; closed-form, Python-deployable | 2–3 d |
| 5 | Synthetic time-stretch validation set (§8) | gives ground truth for 1–4 and all later items | 1–2 d |
| 6 | Shaddah engine (§4.5) + short-vowel ishbāʿ/ikhtilās + sukūn taḥrīk (§4.4, 4.6) | two missing rule families | 3–4 d |
| 7 | OU Kalman local beat + mirtaba HMM (§3.3–3.4) | replaces ad-hoc window median; gives beat uncertainty | 2 d |
| 8 | Hierarchical variance-component model in Turing (§5.2), priors → calibration.json | principled tasāwī bands with partial pooling | 3 d |
| 9 | Drift: PELT offline, BOCPD + EWMA online (§5.5) | Tarāwīḥ fatigue, "changed wajh mid-reading" | 1–2 d |
| 10 | Duration law via SymbolicRegression.jl + inverse-law counts (§4.1) | removes count inflation for long madds | 2 d |
| 11 | Residual nPVI / renewal pulse regularity (§6.1–6.2) | rhythm balance metric | 1 d |
| 12 | Explicit-duration HSMM re-segmentation + count Bayes factors (§4.3) | root-cause fix for peaky CTC spans | 4–5 d |
| 13 | Canonical-beat-axis registration (score following / SRVF) (§5.6) | unified phase view; roadmap B1 | 3–4 d |
| 14 | UDE tempo + SINDy of the NN forcing; Hawkes pauses (§6.3–6.4) | discovery research; needs contiguous-surah data | 4–5 d |

Data prerequisite for 7, 9, 14: benchmark **contiguous** surahs (≥ 20 ayahs) for the anchor and peers.

## 12. References (primary)

- Adams, R. P. & MacKay, D. J. C. (2007). Bayesian online changepoint detection. arXiv:0710.3742.
- Al-Tamimi, J. & Khattab, G. (2015). Acoustic cue weighting in the singleton vs geminate contrast in Lebanese Arabic. JASA 138(1):344. doi:10.1121/1.4922514.
- Arbabi, H. & Mezić, I. (2017). Ergodic theory, dynamic mode decomposition and computation of spectral properties of the Koopman operator. arXiv:1611.06664.
- Arvaniti, A. (2012). The usefulness of metrics in the quantification of speech rhythm. J. Phonetics 40:351.
- Betancourt, M. & Girolami, M. (2013). Hamiltonian Monte Carlo for hierarchical models. arXiv:1312.0906.
- Brunton, S., Proctor, J., Kutz, J. N. (2016). SINDy. PNAS 113:3932. doi:10.1073/pnas.1517384113.
- Chen, R. T. Q. et al. (2018). Neural ordinary differential equations. arXiv:1806.07366.
- Cook, J. R. & Stefanski, L. A. (1994). Simulation-extrapolation estimation in parametric measurement error models. JASA 89:1314.
- Cranmer, M. (2023). Interpretable ML for science with PySR and SymbolicRegression.jl. arXiv:2305.01582.
- Crystal, T. H. & House, A. S. (1988). Segmental durations in connected-speech signals. JASA 83:1553.
- Cuturi, M. & Blondel, M. (2017). Soft-DTW. arXiv:1703.01541.
- Fox, E. et al. (2011). A sticky HDP-HMM with application to speaker diarization. arXiv:0905.2592.
- Gelman, A. (2006). Prior distributions for variance parameters in hierarchical models. Bayesian Analysis 1:515. doi:10.1214/06-BA117A.
- Gibbon, J. (1977). Scalar expectancy theory and Weber's law in animal timing. Psychol. Rev. 84:279.
- Grabe, E. & Low, E. L. (2002). Durational variability in speech and the rhythm class hypothesis. Laboratory Phonology 7. doi:10.1515/9783110197105.515.
- Hawkes, A. G. (1971). Spectra of some self-exciting and mutually exciting point processes. Biometrika 58:83.
- Khattab, G. & Al-Tamimi, J. (2014). Geminate timing in Lebanese Arabic. Laboratory Phonology 5(2):231. doi:10.1515/lp-2014-0009.
- Killick, R., Fearnhead, P., Eckley, I. (2012). Optimal detection of changepoints with a linear computational cost (PELT). JASA 107:1590. doi:10.1080/01621459.2012.737745.
- Klatt, D. H. (1976). Linguistic uses of segmental duration in English. JASA 59(5):1208. doi:10.1121/1.380986.
- Kruschke, J. K. (2018). Rejecting or accepting parameter values in Bayesian estimation. AMPPS 1:270.
- Lakens, D. (2017). Equivalence tests: a practical primer. Soc. Psychol. Personal. Sci. 8:355.
- Page, E. S. (1954). Continuous inspection schemes. Biometrika 41:100.
- Rackauckas, C. et al. (2020). Universal differential equations for scientific machine learning. arXiv:2001.04385.
- Ramus, F., Nespor, M., Mehler, J. (1999). Correlates of linguistic rhythm in the speech signal. Cognition 73:265.
- Roberts, S. W. (1959). Control chart tests based on geometric moving averages. Technometrics 1:239.
- Rosen, K. M. (2005). Analysis of speech segment duration with the lognormal distribution. Clin. Linguist. Phon. 19:597.
- Schuirmann, D. J. (1987). A comparison of the two one-sided tests procedure… J. Pharmacokinet. Biopharm. 15:657. doi:10.1007/BF01068419.
- Srivastava, A. et al. (2011). Registration of functional data using Fisher–Rao metric. arXiv:1103.3817; Tucker, Wu, Srivastava (2013) arXiv:1212.1791.
- Uhlenbeck, G. E. & Ornstein, L. S. (1930). On the theory of Brownian motion. Phys. Rev. 36:823.
- Vehtari, A., Gelman, A., Gabry, J. (2017). Practical Bayesian model evaluation using LOO-CV and WAIC. arXiv:1507.04544.
- White, L. & Mattys, S. (2007). Calibrating rhythm: first and second language studies. J. Phonetics 35:501.
- Wing, A. M. & Kristofferson, A. B. (1973). Response delays and the timing of discrete motor responses. Percept. Psychophys. 14:5. doi:10.3758/BF03198607.
- Yu, S.-Z. (2010). Hidden semi-Markov models. Artificial Intelligence 174:215. doi:10.1016/j.artint.2009.11.011.
- Zeyer, A., Schlüter, R., Ney, H. (2021). Why does CTC result in peaky behavior? arXiv:2105.14849.
- Quranic context: *Automatic Pronunciation Error Detection and Correction of the Holy Quran's Learners* (Quran Phonetic Script encodes madd counts symbolically), arXiv:2509.00094; *A Critical Review of the Need for Knowledge-Centric Evaluation of Quranic Recitation*, arXiv:2510.12858; *Vowel lengthening in the Quranic recitation: duration and pitch contours* (six Egyptian/Saudi reciters, 4- vs 6-beat madd; academia.edu 470390 — venue to verify before citing externally).

Notes on verification: bibliographic details above were checked by web search where marked in the
session (Zeyer, Klatt, Wing–Kristofferson, Khattab & Al-Tamimi, Grabe & Low, Ramus et al., Cranmer,
the two Quranic arXiv papers); the remainder are standard references cited from memory — confirm page
numbers before external publication. The Tajweed policy constraints in §1/§5.4 must be confirmed by the
project's ijāzah holder.
