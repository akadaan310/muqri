# Frontier-Mathematics Roadmap for qaari-eval

Synthesis of four commissioned deep-research reports (2026-09-23) on high-level mathematics that
can advance the calibration, discovery, and fingerprinting engine. Each item lists the exact
mathematical object, what it replaces/improves, and primary citations. Ordered by impact/effort.

The unifying diagnosis: our current calibration is a **non-parametric method-of-moments estimator of
a Gaussian tail (median/MAD hull + robust z-score) applied to non-Gaussian, hierarchically
structured, heteroscedastically measured, correlated data.** Almost every upgrade below fixes one of
those four mismatches.

---

## A. Reference calibration — replace the median/MAD axis-aligned hull

### A1 (highest impact, ~1–2 days). Bures–Wasserstein Gaussian reference + distributional verdict
Model each reciter's per-rule metric cloud as `N(mᵢ, Cᵢ)` in a robustly **whitened** space (pooled
within-reciter covariance via MCD/OGK — the multivariate generalization of our 1.4826·MAD).
- Reference = anchor-weighted **Bures–Wasserstein barycenter**: mean `m̄=Σwᵢmᵢ`, covariance the
  fixed point `C̄=Σ wᵢ (C̄^½ Cᵢ C̄^½)^½` (Agueh–Carlier), converges dimension-free in ~5 iters at d≤8.
- Verdict = closed-form W₂ between Gaussians `D²=‖m_c−m̄‖² + Tr(C_c)+Tr(C̄) − 2Tr((C̄^½ C_c C̄^½)^½)`,
  which **splits location error (wrong duration/formant) from shape/consistency error** — pedagogically
  valuable ("madd length right but inconsistent").
- Why it beats the hull: with ~9 peers and d≈5 features the hull uses ~2d scalars and **ignores all
  d(d−1)/2 feature correlations** (core_ms↔counts, F1↔F2), causing diagonal-outlier over-acceptance.
- Sources: Agueh & Carlier, SIAM J. Math. Anal. 43(2):904 (2011), doi:10.1137/100805741;
  Bures–Wasserstein averaging, arXiv:2106.08502; large-sample theory arXiv:2305.15592.

### A2 (~½ day). Affine-invariant SPD (Fisher–Rao) *shape* term for heterogeneous units
For the covariance/consistency question use `d_AI(C_c,C̄)=‖log(C̄^{-½}C_c C̄^{-½})‖_F`. Between
zero-mean Gaussians this **equals** the Fisher–Rao distance and is **unit-invariant** (ms↔s, Hz↔kHz),
solving heterogeneous units for the shape part exactly. Combine `D²_total = D²_Bures(loc) + β·d²_AI(shape)`.
- Sources: Miyamoto et al., *Closed-Form Fisher–Rao Distances*, arXiv:2304.14885; Pennec et al. IJCV 66 (2006).

### A3 (selective). Sinkhorn / unbalanced OT for non-Gaussian & robustness-critical rules
Route rules that fail a Gaussianity test (bimodal **madd**: legitimate 2- vs 4-count mixture) to the
debiased **Sinkhorn divergence** `S_ε` on the empirical clouds; use **unbalanced OT** (KL marginal
penalties, transport radius √τ ≈ our z≤2–3 boundary) for outlier robustness inside the OT framework.
ε in whitened units ≈ 0.05–0.1·median pairwise sq-distance.
- Sources: Cuturi arXiv:1306.0895; Feydy et al. AISTATS 2019 arXiv:1810.08278; Chizat et al.
  arXiv:1508.05216 / 1607.05816; Séjourné et al. arXiv:1910.12958; Peyré & Cuturi arXiv:1803.00567.

### A4 (universal). Conformal peer-LOO thresholds — a finite-sample guarantee
For each held-out peer j, rebuild the reference from anchor+remaining peers and record its distance
`d_j`. The `{d_j}` **are** the null distribution; set PASS/WARN/FAIL from their **ceiling (conformal)
quantiles** `⌈(P+1)(1−α)⌉/P`. This turns our "held-out peer ≥95" target into a *coverage theorem*
rather than the arbitrary `|z|≤2`. Category weights → maximize peer/impostor separation under nested LOO.
- Sources: Vovk et al. *Algorithmic Learning in a Random World*; Gretton et al. JMLR 13 (2012);
  Székely & Rizzo, energy statistics, doi:10.1016/j.jspi.2013.03.018.

### A5 (foundation, alternative to A1). Bayesian hierarchical partial-pooling model
Nested random effects reciter→surah→ayah→rule with **Student-t (or skew-t) likelihood** and per-rule
scale `τ_k`; non-centered parameterization. The reference band = **posterior predictive of a new
reference-like reciter** (`β_r_new ~ N(0,σ_r)`), which **auto-widens where peers disagree (σ_r) and
where data are scarce** — the principled replacement for the hull. Fold forced-alignment confidence in
as **heteroscedastic measurement error** (`y_obs ~ N(y_true, s_meas)`) so low-confidence instances are
down-weighted, not spuriously FAILed. Verdicts = **posterior-predictive tail probability (PIT)**.
- Sources: Betancourt & Girolami arXiv:1312.0906; West (1984) JRSS-B; Juárez & Steel (2010) JBES;
  Stan measurement-error guide; Vehtari et al. LOO-CV arXiv:1507.04544; LOO-PIT arXiv:2410.03507.

### A6 (tail control). Extreme-Value Theory for PASS/WARN/FAIL thresholds
PASS/FAIL are **tail events**; a Gaussian z=3 is ~0.13% but a t(ν=5) z=3 is ~1.5% — an order of
magnitude miscalibration that drifts per rule. Fit a **Generalized Pareto Distribution (POT)** to the
reference-predictive tail (Pickands–Balkema–de Haan) and set the FAIL cut at a controlled exceedance
risk z (Siffer et al. SPOT/DSPOT formula `q_z=u+(β/ξ)[(zn/N_u)^{−ξ}−1]`). ξ per rule *quantifies*
non-Gaussianity (durations likely ξ>0, heavy right tail from deliberate elongation).
- Sources: Siffer et al. KDD 2017 doi:10.1145/3097983.3098144; Coles (2001); Scarrott & MacDonald (2012).

---

## B. Model discovery — beyond the relaxation ODE + SINDy

### B1. Elastic FDA — phase/amplitude separation (fixes the madd count confound directly)
Model each surah's tempo/pitch/energy as a **function of recited time**; use **SRVF + Fisher–Rao**
where warping acts by isometries, so the elastic distance is warp-invariant. Separate **phase**
(timing/rubato = the warping γ) from **amplitude** (tempo level). The relaxation ODE is the degenerate
rank-1, γ=identity corner of this model.
- **Highest-value payoff:** evaluate `core_ms` against the *registered* local beat (or use
  `core_ms/haraka_ms` = count-in-beats) so the 2/4/6 madd counts collapse to tight clusters with
  reduced within-count variance — the crispest falsifiable win.
- The warping γ itself is a Tajweed object: `γ̇>1` = local dwell = madd over-stretch / rubato.
- "Warped-time SINDy" (register first, then SINDy on registered curves) de-biases STLSQ/BIC — flagged
  as a novel low-risk *composition*, validate empirically. Or a nonlinear mixed-effects registration
  (Raket et al.) as one coherent estimator.
- Software: `fdasrsf` (Python), `fdasrvf` (R); call from Julia via PythonCall.
- Sources: Srivastava et al. arXiv:1103.3817; Tucker et al. CSDA 61:50 (2013) arXiv:1212.1791;
  Marron et al. Statist. Sci. 30(4):468 (2015) doi:10.1214/15-STS524; Raket et al. arXiv:1712.07265.

### B2. Koopman / Hankel-DMD — generalizes the ODE, extracts rhythm/melody spectra
Our ODE `dT/dτ=κ(T∞−T)+φ` is Koopman-exact with a single real eigenvalue `−κ`. **Hankel-DMD**
(delay-embed the per-frame `[F0,F1,F2,F3,energy]`, d≈50–300 frames) estimates the whole Koopman
spectrum: `Im log λ` = rhythmic/melodic frequency, `Re log λ` = damping. Arbabi–Mezić prove
convergence to true Koopman eigenfunctions. Yields interpretable, reciter-characteristic spectral
features and forecast skill the scalar ODE lacks.
- Software: `DataDrivenDiffEq.jl` (DMD, next to our SINDy), PyDMD (HankelDMD/HAVOK).
- Sources: Williams et al. J. Nonlin. Sci. 25 (2015) doi:10.1007/s00332-015-9258-5; Arbabi & Mezić
  arXiv:1611.06664; Brunton et al. HAVOK Nat. Commun. 8 (2017); Colbrook arXiv:2312.00137.

### B3. Gaussian-process / latent-force hybrid for tempo drift
Replace or augment the ODE with a GP (Matérn-3/2) over ayah position, or best: a **Latent-Force Model**
(ODE backbone + GP forcing → solution is a GP with a physics-derived kernel). Recovers the ODE when
forcing is negligible and flexes where it misfits; residual magnitude = automatic model criticism.
Feed the drift estimate into DSPOT residualization (score `y − μ_drift`).
- Sources: Álvarez et al. arXiv:1107.2699; Deep LFM arXiv:2311.14828; GP-constrained-to-ODE arXiv:2208.12515.

---

## C. Reciter fingerprint — beyond the 232-d Euclidean/FAISS vector

### C1 (build first). Path signatures / rough paths — tempo-warp-invariant style features
The **signature** of the multivariate acoustic path (log-F0 cents, formants, energy) is
**reparameterization-invariant** (tempo-warp invariance *for free* — exactly our style-ID need) and,
by Hambly–Lyons uniqueness + universal-linear-approximation, a principled feature set. Use **depth-3
log-signature** on sliding windows (~50 dims/segment), pooled via Chen's identity; concatenate to the
232-d vector. Choose invariance by including/omitting the time channel (style-ID vs timing-scoring).
- Validate with a ±20% time-stretch test: signature features flat, current vector degrades.
- Software: `signatory` (PyTorch/GPU), `iisignature`, `signax` (JAX).
- Sources: Chevyrev & Kormilitzin arXiv:1603.03788; Hambly & Lyons Ann. Math. 171 (2010)
  doi:10.4007/annals.2010.171.109; Kidger & Lyons *Signatory* arXiv:2001.00706.

### C2 (very low effort, fixes a latent bug). Riemannian SPD geometry for spectral covariance
> **Correction (2026-09-23):** the latent-bug premise is false. `app/fingerprint.py` indexes only the 192-d
> timbre embedding (ECAPA or L2-normalised MFCC statistics), the Tajweed metrics and the environment fields in
> `faiss.IndexFlatIP`. No SPD covariance reaches FAISS, so nothing is broken today. C2 stays useful as a *new*
> feature. The SPD machinery (AIRM, log-Euclidean, Karcher mean) now exists in `julia/src/Frontier.jl` and
> `octave/fr_*.m`, where the frontier calibration layer uses it on per-rule covariance clouds.
We already compute spectral covariance `C` (SPD) but (likely) feed it to **Euclidean** FAISS — wrong
geometry (swelling effect). Switch to **log-Euclidean**: index `vech(log C)` in FAISS (near-zero code
change) for geometric correctness + affine (gain/EQ) invariance; use per-reciter **Fréchet mean** as a
scoring reference (Barachant MDM). Full AIRM re-rank on top-k if desired.
- Software: `pyriemann` (Python), `Manifolds.jl` (SPD, AIRM/LEM, Fréchet mean).
- Sources: Barachant et al. IEEE TBME 59(4) 2012 doi:10.1109/TBME.2011.2172210; Arsigny et al. MRM 56
  (2006) doi:10.1002/mrm.20965; Pennec et al. IJCV 66 (2006).

### C3 (experimental). Topological data analysis of the maqam contour
Sliding-window embed log-F0 → persistent homology; **H₁ max persistence = quasi-periodicity/recurrence
score** (Perea–Harer), stable and tempo-invariant. Vectorize via persistence images to *augment* the
fingerprint; gate behind an incremental-mAP test (must add orthogonal info beyond signatures).
- Software: `ripser`/`giotto-tda` (Python), `Ripserer.jl`.
- Sources: Perea & Harer FoCM 15 (2015) doi:10.1007/s10208-014-9206-z; SW1PerS BMC Bioinf. 16 (2015);
  Bauer *Ripser* doi:10.1007/s41468-021-00071-5.

---

## Recommended build order (dependency-aware)
1. **A1+A2+A4** (Bures–Wasserstein reference + SPD shape term + conformal thresholds) — biggest, cheapest
   calibration win; recovers correlations, principled thresholds with a coverage guarantee.
2. **B1 quick win** — beat-normalized madd counts (no registration), immediate testable payoff; then full
   elastic registration + phase/amplitude variance decomposition.
3. **C2** — log-Euclidean FAISS + Fréchet-mean reference (optional new feature; the "existing geometric error" was not real, see correction above).
4. **C1** — path-signature fingerprint features + tempo-stretch robustness A/B.
5. **A3/A5/A6** — Sinkhorn/unbalanced OT for bimodal rules; or the full Bayesian hierarchical + EVT stack
   if we want calibrated tail probabilities with LOO-PIT validation.
6. **B2/B3** — Koopman/DMD spectra and GP/latent-force drift as discovery upgrades.
7. **C3** — TDA maqam descriptor, gated on incremental value.

**Validation discipline (all items):** with only ~10 reciters, prefer **retrieval mAP with ayah-disjoint
splits** and **tempo-/channel-perturbation robustness curves** over raw reciter-ID accuracy; gate every
calibration change on **leave-one-peer-out coverage** and **LOO-PIT** calibration.
