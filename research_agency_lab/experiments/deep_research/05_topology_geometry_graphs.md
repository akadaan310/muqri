# 05 — Algebraic Topology, Geometry and Graph Theory for qaari-eval

Scope: the whole engine (makharij, sifaat, timing, stops, maqam, text hierarchy), not only fingerprints.
This report extends `frontier_math_roadmap.md` (Bures–Wasserstein, AIRM/log-Euclidean SPD, Perea–Harer
maqam TDA, depth-3 log-signatures, Hankel-DMD, elastic FDA). It does **not** repeat those items. It adds
new objects and new plug-in points.

**Engine facts that constrain everything below** (checked in the code on 2026-09-23):
- The data are small: about 700 ayah rows (`benchmarks/results/**.jsonl`; 4 EveryAyah reciters × ≤118
  strategic ayahs, plus 59 Husary rows), roughly 9.5k diagnostics, and fewer than 10 reference reciters.
  No GPU is available on this box (`nvidia-smi` shows none). Julia 1.10.9; `Graphs.jl` is already in the
  pinned Manifest as a transitive dependency. Octave with `signal` is installed.
- Rule timing is judged on letter spans from CTC forced alignment (`app/aligner.py`). The per-frame
  emissions are cached. `align_conf` is one number per span, the minimum mean posterior. No boundary
  uncertainty is propagated.
- Events are counted with threshold/run-length heuristics: `count_taps` (6 dB dip under a ±20 ms
  running max, run 6–45 ms), `count_releases` and `detect_release_burst` (percentile closure mask plus a
  flux peak and a 40 ms separation).
- Consistency is checked only by `scoring.consistency_notes`, which flags `max−min > 1.5` counts per madd
  type. The recitation score is a category-weighted mean over diagnostics that are treated as
  independent. A single misaligned unit can therefore trigger several FAILs.
- `parser._DROP` **deletes the waqf signs U+06D6–U+06DC** (ۖ ۗ ۘ ۙ ۚ ۛ ۜ). The ground-truth stop
  annotations of the mushaf are thrown away before any stop logic runs.
- The idgham tables (`MUTAJANISAYN_PAIRS`, `MUTAQARIBAYN_PAIRS`, `_IDGHAM_TARGETS`) are hand-written
  sets. There is no makhraj/sifaat metric behind them.

Legend for payoff: **[L]** likely to pay off, **[M]** medium or data-dependent, **[S]** speculative.

---

## 0. The ten findings, ranked by value / effort

| # | Item | Phenomenon | Payoff | Effort |
|---|------|-----------|--------|--------|
| 1 | G2 CTC-trellis random walk (FFBS) → alignment uncertainty for every verdict | all timing rules | L | 2 d |
| 2 | T1 0-dim persistence (topological prominence) for event counting | takreer taps, qalqalah bounce, idgham releases, sakt | L | 1.5 d |
| 3 | P2 Lévy-area *precedence index* (reparametrisation-invariant "what moved first") | ikhfa anticipation, iqlab, qalqalah, idgham naqis | L | 1.5 d |
| 4 | S2 Haraka-field sheaf on the text graph + one global madd choice per recitation | madd/ghunnah counts, Hafs consistency | L | 3 d |
| 5 | G3 Failure incidents (components of the rule–unit overlap graph) + PageRank root cause | score fairness, feedback | L | 1 d |
| 6 | G1 Makhraj tree × sifaat cube metric → derive idgham tables + tree-Wasserstein makhraj score | makharij, idgham classes | L/M | 3–4 d |
| 7 | R4+T3 Tonic-invariant circular OT + circle persistence of pitch-class density | maqam | M/L | 2–3 d |
| 8 | G5 Waqf/ibtida as cuts of the Quranic dependency graph, validated on the mushaf signs | stops | M | 4–5 d |
| 9 | G4 Repetition (mutashabihat) edges → replicate ICC per metric | ruler choice, measurement noise | M | 2 d |
| 10 | P1 Signature-kernel MMD (Goursat PDE) for reciter path-law tests | style, melody, fatigue | M | 3 d |

Speculative items (T4 multiparameter/torus persistence, T5 zigzag fatigue, H2 Hodge melody flows, R2
Grassmann, R3 Gromov–Wasserstein, S3 tahrirat sheaf, GNN/neural sheaf diffusion) are covered in §§1–5
with an honest note on each.

**Explicit non-recommendation.** Do not run Vietoris–Rips persistent homology on per-letter MFCC or
SSL-embedding paths. A letter lasts 50–150 ms, which is 5–15 frames at a 10 ms hop and 3–8 frames at
the 20 ms SSL rate. A point cloud that small has no meaningful H₁. Its "persistence" only re-measures
excursion size or curvature, and topological noise dominates. Topology pays in this engine in three
places: 1-D functions sampled densely (1 ms envelopes, spectral envelopes along frequency, pitch-class
densities on S¹), and long windows (ayah-scale melody).

---

## 1. Algebraic topology (persistence)

### T1 [L] 0-dimensional sublevel persistence = stable event counting
**Math.** Let f: {1..n} → ℝ be a 1-D signal, such as the dB RMS envelope at a 1 ms hop. The 0-dim
persistence diagram of the superlevel filtration {f ≥ θ} is built by union-find with the elder rule, in
O(n log n). Each local maximum p is born at f(p) and dies at the saddle where its component merges into
an older one. Its persistence is π(p) = f(p) − s(p), which is exactly the *topographic prominence*.
Dips are the maxima of −f.
**Stability** (Cohen-Steiner–Edelsbrunner–Harer 2007): d_B(Dgm f, Dgm g) ≤ ‖f − g‖_∞. Consequence:
the count N_τ(f) = #{p : π(p) ≥ τ} is invariant under any perturbation with ‖f−g‖_∞ < γ/2. Here
γ = min_p |π(p) − τ| is the **gap**, and it should be reported as a certified margin
(`tap_margin_db`, `release_margin_db`).
**Tajweed fit.**
- *Takreer*: a rolled raa is k ≥ 2 brief occlusions at ~20–30 Hz. The taps are dips of −f with
  prominence ≥ τ_dB and a basin width, measured at half-prominence, inside TAP_MS. `count_taps`
  currently defines a dip against a sliding ±20 ms maximum. That definition double-counts shallow
  ripples and misses dips that sit between two unequal peaks. Prominence is the principled
  replacement.
- *Qalqalah bounce*: closure (energy basin) → burst (flux peak) → echo vowel. The bounce strength is
  the persistence of the release peak above its higher saddle. This separates "burst above the closure
  floor" from absolute level, so the reference is local. The kubra/sughra difference then appears as
  π plus basin depth.
- *Idgham kamil*: the number of releases is the number of persistent flux peaks, each preceded by a
  persistent energy basin. This replaces the 40 ms separation heuristic in `count_releases`.
- *Sakt vs waqf*: in the superlevel filtration of energy, a pause is a split of the speech
  component. Its depth is the persistence and its duration is the basin width. A sakt is a shallow,
  short split with no inhalation band energy.

**Plug-in.** `app/sifaat/ghair_mutadhaddah.count_taps`,
`app/tajweed_rules/qalqalah_engine.{detect_release_burst,count_releases}`, and `sakt_wasl`. Add metric
keys `*_prominence_db` and `*_margin_db` so the Julia calibrator can build bands on them.
**Validation.**
1. Synthetic AM trills with k ∈ {1..4} dips at SNR 0–30 dB; compare the count accuracy curve against
   `count_taps`.
2. About 150 hand-labelled raa and qalqalah tokens in Praat (1 day).
3. On Husary, `takreer_single_strike_flag` should go to 1.
4. Python vs Octave agreement. A union-find on a vector is about 30 lines of Octave.

**Julia.** `Ripserer.jl` `Cubical` on a 1-D array gives the same diagram and serves as a cross-check.
Production code is plain numpy. **GPU: none.**

### T2 [M] Topological formant and antiformant tracking along frequency
**Math.** For each frame t, take the smoothed spectral envelope S_t(ν) in dB. Persistence of the
superlevel filtration along ν gives peaks (formants) with prominence. Persistence of the sublevel
filtration gives valleys (antiformants). Following the diagrams over t is a **vineyard**
(Cohen-Steiner–Edelsbrunner–Morozov 2006). In a vineyard a formant is a vine, and a merger is a vine
death governed by the elder rule.
**Tajweed fit.**
1. *Tafkheem* is F2 collapsing toward F1. At the collapse, fixed-order LPC peak-picking can jump to F3
   or split spuriously, so a heavy vowel is reported as light. The topological statement "the F2 vine's
   persistence → 0 because it merged into F1" is a robust heaviness indicator. It needs no pole
   bandwidth threshold (`features._lpc_formants` uses < 400 Hz).
2. *Ghunnah/ikhfa*: nasal coupling adds a **zero**, an antiresonance at roughly 0.5–2 kHz. An all-pole
   LPC envelope *cannot represent zeros*. Use a cepstrally-liftered envelope and take the persistence of
   the deepest valley in 500–2000 Hz as a nasal-zero depth. It complements `nasal_db`, the band ratio.

**Plug-in.** `app/sifaat/formants.py` (new `heaviness_topo`, `antiformant_depth_db`), with Octave
`qaari_features.m` (cepstral envelope + union-find) as the independent path.
**Validation.** Correlation with the existing `heaviness_index` on light letters. Imams should separate
better on heavy letters. For the antiformant, test nasal-vs-oral AUC on known ghunnah spans against
`nasal_db`. **Risk:** envelope smoothing controls everything. Report the persistence *scale-space* over
three lifter orders.

### T3 [M/L] Persistence on the pitch-class circle (maqam scale degrees)
**Math.** Let c = 1200·log₂(F0/F0_ref) mod 1200 on S¹, and let h be the von Mises KDE on the circle.
Take 0-dim superlevel persistence of h on the cycle graph, which is periodic union-find. The circle's
single essential H₀ class is the global mode. Each other persistent peak is a **scale degree** with
microtonal position. The neutral intervals of Bayati and Sikah fall at about 150 and 350 cents, and
their persistence says how strongly each degree is used. Stability: d_B ≤ ‖h − h'‖_∞, so the number of
scale degrees is stable to histogram noise. The kernel bandwidth becomes a scale parameter that you
report instead of tuning.
**Plug-in.** A new `app/acoustic/maqam.py` built on `AcousticContext.f0_track`, with a maqam descriptor
per ayah. This goes beyond the roadmap's Perea–Harer item, which measures *recurrence* (H₁ of a delay
embedding). T3 measures the *interval inventory*.

### T4 [S] Beyond Perea–Harer: tori, DTM and multiparameter persistence
- *Vibrato × phrase recurrence*: a quasi-periodic signal with k incommensurate frequencies embeds densely
  into T^k. Gakhar–Perea (arXiv:2103.04540) give Rips lower bounds and window-parameter rules. H₂ of a
  T² is the signature of *vibrato riding on a recurrent melodic cell*. This is an interesting style
  marker, but the ayah-level F0 is not stationary enough to expect clean tori. Speculative.
- *Robustness to mis-tracked frames*: use a distance-to-measure (DTM) filtration (Anai et al.
  arXiv:1811.04757; Chazal et al. arXiv:1412.7197) instead of plain Rips, so octave errors and unvoiced
  frames do not create spurious bars.
- *Confidence as a second parameter*: a bifiltration by (scale r, CTC confidence κ) is a frame's
  inclusion at κ ≥ κ₀. Its signed-barcode vectorisation (Loiseaux et al. arXiv:2306.03801, `multipers`)
  gives features that do not depend on one `conf_min`. This is conceptually right for the engine's
  hard SKIP rule, but it has no Julia implementation and is costly. Research only.

### T5 [S] Zigzag persistence and vineyards across ayahs and rak'ahs
Zigzag X₁ ⊃ X₁∩X₂ ⊂ X₂ … (Carlsson–de Silva arXiv:0812.0290) over consecutive windows tracks when a
melodic loop or vibrato appears or dies. The BuZZ method (Tymochko et al. arXiv:2009.08972) detects
Hopf-type onset and loss of oscillation in one diagram. Candidate use: `taraweeh_adapter/fatigue_detector`,
to detect loss of vibrato or pitch-cycle regularity in late rak'ahs. This needs long live recordings
that are not yet in the corpus.

### T6 [S, likely redundant] Loops of articulatory trajectories
An oral → nasal → oral excursion in (F1, F2, nasal_db) space closes a loop. Its H₁ persistence is
essentially the excursion's area and diameter, which simple statistics already give. Build it only if
T1 and P2 leave residual error.

**Vectorisation and statistics for all TDA features.** Use persistence landscapes (Bubenik
arXiv:1207.6437; 1-Lipschitz in d_B) for statistics and averaging, and persistence images (Adams et al.
arXiv:1507.06217) for FAISS concatenation. Use the Wasserstein-p distance on diagrams (Skraba–Turner
arXiv:2006.16824 for p-stability; Hera arXiv:1606.03357). Test with permutation tests on diagrams
(Robinson–Turner arXiv:1310.7467). Use bootstrap confidence bands to separate signal bars from noise
(Fasy et al. arXiv:1303.7117). Rule: every TDA feature enters the verdict or the fingerprint only after
passing an **incremental-value gate**, meaning peer/imam separation or ayah-disjoint retrieval mAP beyond
the current metrics plus signatures.

---

## 2. Sheaves and Hodge theory

Notation. G = (V, E) is a graph. A cellular sheaf F assigns stalks F(v), F(e) and linear restriction maps
F_{v⊴e}. The coboundary is (δx)_e = F_{u⊴e}x_u − F_{v⊴e}x_v, the sheaf Laplacian is L_F = δᵀδ, and the
Hodge theorem gives ker L_F = H⁰(G;F), the globally consistent sections (Hansen–Ghrist
arXiv:1808.01513). Robinson's **consistency radius** is c(x) = max_e ‖(δx)_e‖ (arXiv:1805.08927,
arXiv:1603.01446). **Honesty note:** when every restriction map is an identity, L_F reduces to a
graph Laplacian and the "sheaf" is just variance. The framework earns its keep only with non-identity
maps (unit conversions, textbook ratios, projections) and when you want the obstruction *localised* to
an edge.

### S2 [L] Haraka-field sheaf: joint tempo + counts + Hafs consistency
The engine currently estimates the harakah unit from short syllables (`tempo.estimate_tempo`) and a
global `count_scale`, and then divides. That makes counts hostage to the tempo estimate.
**Model.** Word nodes i carry a log-haraka h_i. The text is a path graph whose edge weights w_{i,i+1}
are reduced across waqf and sakt. Duration-rule instance r at word i(r) has measured core d_r and
expected counts e_r:

    log d_r = h_{i(r)} + log e_r + ε_r,
    min_h  Σ_r ρ_Huber((log d_r − log e_r − h_{i(r)})/σ_r) + λ Σ_{(i,j)} w_ij (h_i − h_j)²

The ℓ₁ variant is graph trend filtering (Wang et al. JMLR 2016, arXiv:1410.7690) and allows tempo
breaks at stops. Sheaf reading: the stalk at a word is ℝ (h), the stalk at a rule is ℝ (log d_r), and
the restriction is the affine shift by log e_r. The textbook ratio *is* the restriction map. A global
section is one tempo field that explains every duration. The **residuals ε_r are the tajweed errors in
log-count units**, and they are estimated jointly instead of after a tempo guess.
**Free-choice rules** (madd ʿarid 2/4/6, munfasil 4/5, leen) become *one discrete global parameter per
type per recitation*, e_type ∈ {choices}. Enumerate the ≤ 3×2×2 combinations and pick by penalised
likelihood. Hafs *tasawi* (consistency) then follows directly: an inconsistent instance is one with a
large residual under the chosen global length. This replaces `consistency_notes`' `max−min > 1.5`.
**Plug-in.** Julia `QaariLab` (new `HarakaField.jl`, using SparseArrays and a tridiagonal + low-rank
solve in O(n)). Its output replaces `count_scale` and `haraka_ms` in calibration. The online path is a
Python port (scipy sparse).
**Validation.**
1. Within-count variance of madd_tabii/muttasil/lazim on peers should drop compared with the current
   counts. This is the same falsifiable target as roadmap B1, reached by a discrete, closed-form route.
2. Peer-LOO coverage (roadmap A4).
3. Inject synthetic ±20% tempo ramps and check that counts stay flat.

### S1 [M] Measurement-reconciliation sheaf (Python ↔ Octave ↔ overlapping rules)
**Vertices.** Measurement nodes (pipeline p, span s, quantity q) and latent unit-state nodes
z_u = (log core, F1, F2, nasal, burst, HNR). **Edges.** "Pipeline p measured unit u" with
F_{u⊴e} = P_q (a projection), and F_{meas⊴e} = A_p, the pipeline's affine calibration (dB offset, ms
bias), learned on the references. Add boundary edges (end of unit i = start of unit i+1) and
overlapping-rule edges. Examples: ghunnah and idgham_ghunnah on the same mushaddad span; the madd vowel
reused as the tafkheem formant window; shiddah duration_ratio and qalqalah hold on the same stop.
**Reconciliation.** x* = (Σ⁻¹ + α L_F)⁻¹ Σ⁻¹ y. As α → ∞ this is the Σ-orthogonal projection onto H⁰.
The edge residuals r_e = (δx*)_e *localise* the disagreement: which pipeline, which span, which
quantity. A large r_e on a unit whose rules all FAIL points to an alignment fault, not to the reciter.
**Plug-in.** `research_agency_lab` validation loop (`compute_bridge/validate_loop.py` already pairs
Python and Octave). This makes that cross-check a per-span reliability score and a SKIP reason.
**Effort** 3 d. **Payoff** M: most of the value is also reachable with a weighted mean, and the sheaf adds
the localisation and the unit-conversion bookkeeping.

### S3 [S for content, easy for code] Tareeq/tahrirat as a sheaf of sets (constraint satisfaction)
The stalk at each realised choice (munfasil length, sakt yes/no, ghunnah on lam/raa, …) is the set of
turuq under which it is valid. A global section is a single consistent tareeq. When every pairwise
constraint is locally satisfiable but no global section exists, that is exactly the
Abramsky–Brandenburger contextuality pattern (arXiv:1102.0264) and is the formal content of *talfiq*
(mixing turuq). **Honest:** with unary constraints this is just set intersection. It becomes a real
binary CSP only with scholar-curated conditional tahrirat ("if munfasil is qasr via Tayyibah then …").
Implement it as arc-consistency. The mathematics is trivial; the rule table is the actual work and needs
an expert. The engine already has `Tareeq` and `is_tayyibah_qasr_flag`, so there is a natural home.

### S4 [note] The text hierarchy is a tree GMRF
letter → word → ayah → surah is a rooted tree. The nested random-effects model of roadmap A5 *is* a
Gaussian MRF whose precision is a tree Laplacian plus a diagonal (Rue & Held 2005). A sheaf over the
hierarchy generalises A5 only by allowing non-identity maps between levels, such as the at-waqf
realisation change. Recommendation: implement A5 and do not build a separate "hierarchy sheaf".

### H1 [M] HodgeRank for cross-reciter comparisons with incomplete overlap
Build a graph on reciters with edge flow y_ab = the mean metric difference over the ayahs both a and b
recited (Matroud has only 68 of 118). Solve the least-squares Hodge decomposition
y = grad s + curl* Φ + harmonic (Jiang–Lim–Yao–Ye arXiv:0811.1067; Lim arXiv:1507.05379). s is a global
ordering, such as "distance to Husary's madd style". The **cyclic residual detects content confounding**:
if a metric's ranking is intransitive across ayah subsets, the metric depends on the text, not the
reciter. Julia: Graphs.jl + SparseArrays, 1.5 d. It also aggregates teacher pairwise preferences if they
are ever collected.

### H2 [S] Hodge decomposition of melodic transition flows
Take the antisymmetric part of the scale-degree transition counts per ayah. The gradient part is
ascent/descent tendency (qarar → jawab). The curl on 3-cycles is ornamental turns. It is interpretable,
but its value for maqam ID over T3/R4 is unproven.

---

## 3. Graph theory

### G1 [L/M] The makhraj tree × sifaat cube as a metric space
**Construction.** Take Ibn al-Jazari's 17 makharij as a tree T along the tract:

    halq: ء ه — ع ح — غ خ  → ق → ك → (ج ش ي) ─┬─ ض (lateral branch)
                                              └─ ل → ن → ر → (ط د ت) ─┬─ (ص س ز)
                                                                     └─ (ظ ذ ث) → ف → (ب م و)

Jawf (madd letters) attaches as a leaf to the vowel space. Khayshum (nasality) is **not** a node, because
it would close a cycle ن…م. Model it as a sifah bit instead. Sifaat live in the cube
s ∈ {0,½,1} × {0,1}^{k}: hams/jahr, shiddah/tawassut/rakhawah (ordinal), istiʿla/istifal,
itbaq/infitah, idhlaq/ismat, and the non-opposing safir, qalqalah, leen, inhiraf, takreer, tafashhi,
istitalah, ghunnah. Letter metric:

    d(a,b) = d_T(m(a), m(b)) + λ Σ_k w_k |s_k(a) − s_k(b)|

The classical classes are level sets. Mithlayn is d = 0. Mutajanisayn is d_T = 0 with sifaat differing.
Mutaqaribayn is d_T ≤ 1–2 with sifaat close. **Check against the parser:** (ت,د),(ت,ط),(ث,ذ),(ذ,ظ),(ب,م)
have d_T = 0, (ق,ك),(ت,ث) have d_T = 1, (ل,ر) has d_T = 2, and ن → {ل,ر} has d_T = 1. The hand tables
in `parser.py` should be **generated from and unit-tested against** d, which exposes any
inconsistencies.

**Tree-Wasserstein makhraj score.** Let p̄_u be the blank-renormalised CTC state-occupancy posterior
(forward–backward γ_t, *not* raw emissions) averaged over unit u's frames and mapped token → letter →
makhraj node. On a tree, W₁ has a closed form (Evans–Matsen JRSS-B 2012; Le et al. arXiv:1902.00342):

    W₁^T(p, q) = Σ_{e∈T} w_e |p(Γ_e) − q(Γ_e)|,   Γ_e = subtree below e

Compare the learner's p̄_u with **Husary's p̄ for the same letter in the same context**, not with a delta.
That removes the model's baseline confusions and coarticulation. The *signed* flow on each edge gives
**directional feedback** such as "your ع drifted toward ء (deeper in the throat)" or "your ض leaked
toward ظ". Posteriors have never been interpreted this way in the engine.
**Plug-in.** A new `app/sifaat/makhraj.py` that consumes `CTCForcedAligner.emissions`, and a
`makhraj_w1` metric for calibration.
**Risks.** Check the vocabulary of `TBOGamer22/wav2vec2-quran-phonetics`. If ض/ظ or other pairs share a
token, the score is blind there, and those pairs must abstain. CTC is peaky and overconfident, so
temperature-scale on the reference reciters.
**Validation.** On references, the W₁ distribution per letter should be low. Synthetic substitution
comes from splicing a Husary token of a neighbouring letter into context. Beyond that it needs learner
data.

### G2 [L] Random walks on the CTC trellis → alignment uncertainty for every metric
The CTC trellis over the extended label sequence (length 2L+1) is a DAG. Forward α_t(s) is already
implicit in `ctc_forced_align`. **Forward-filtering backward-sampling** (Carter–Kohn 1994) draws M ≈ 200
alignment paths from the exact posterior in O(M·T) after one O(T·S) forward pass. Each path gives unit
boundaries, which give durations, `core_ms` windows and counts, which give **a verdict distribution
P(PASS/WARN/FAIL)** and a per-instance measurement s.d. s_meas,r.
**Why it matters.** It is the missing input to the heteroscedastic measurement-error term of roadmap A5.
It turns `align_conf` (a min-of-means with no unit) into ms-valued uncertainty. It also makes SKIP
principled: skip when P(verdict) is not decisive, not when conf < 0.x.
**Calibration.** Temperature-scale the emissions so that sampled boundary intervals cover (a) Octave
`core_ms` edges and (b) about 200 Praat-labelled boundaries at the nominal rate.
**Plug-in.** `app/aligner.py` (new `sample_alignments`), `scoring._attach_alignment_metrics`. Cheap
metrics (durations) are recomputed per sample. Formant and HNR use the delta method or the 5
highest-weight distinct paths. CPU only, a few ms per ayah.

### G3 [L] Failure incidents and root-cause attribution
Build a graph on non-PASS diagnostics, with an edge when two share a unit index or overlap by more than
20 ms in time. **Connected components are incidents.** Score one penalty per incident, or weight each
diagnostic by 1/|component|, so that one misaligned unit is not penalised four times. For root cause,
use personalised PageRank on the bipartite rule–unit graph seeded at failing rules
(π = (1−β)(I − βP)⁻¹ s). The unit with the most mass is the root. If that unit's alignment uncertainty
(G2) is high, attribute the incident to alignment and SKIP. **Plug-in.** `scoring.summarize_diagnostics`
(this needs `RuleInstance.unit_indices` threaded into `RuleDiagnostic`). 1 d.

### G4 [M] Text graph with repetition edges; graph signal processing that is not just DSP
On a pure path graph, the graph Fourier transform is the DCT-II. **GSP on the text graph is classical DSP
unless you add non-sequential edges.** The valuable extra edges are **repetition edges** between
corresponding letters of identical phrases (for example the 31× refrain of al-Raḥmān, al-Mursalāt, and
the mutashabihat). Then:
- The quadratic form xᵀL_rep x = Σ_rep (x_i − x_j)² gives the *within-reciter, context-matched replicate
  variance* per metric. The ICC = σ²_between-reciter / (σ²_between + σ²_rep) is a better criterion for
  metric_spec's `alts` ruler choice than the anchor's robust CV. It also gives a direct estimate of
  s_meas for A5.
- High-graph-frequency energy on repetition edges is the learner-facing "you recited the same phrase
  differently".
- Graph wavelets (Hammond et al. arXiv:0912.3848) on path+repetition localise anomalies at multiple scales.

**Data dependency:** the strategic set has few repeats. This needs full-surah runs of 55, 77, 54 and 26.

### G5 [M] Waqf and ibtida as cuts in the dependency graph
**Stop first:** keep the waqf signs that `_DROP` currently deletes. Attach them as edge labels on the word
path: مـ lazim, لا forbidden, ج permitted, صلى continuing preferred, قلى stopping preferred, ∴ muʿanaqa.
Muʿanaqa (stop at exactly one of two positions) is an XOR constraint on two edges.
**Model.** Take arcs A from the Quranic Arabic Dependency Treebank (Dukes & Buckwalter 2010; Dukes et al.
LRE 2013, doi:10.1007/s10579-011-9167-7). Its coverage is partial and it is GPL-licensed. A stop after
word i cuts the path edge (i, i+1). Its cost is

    C(i) = Σ_{(h,d)∈A : min(h,d) ≤ i < max(h,d)} w(rel)

with w large for idafa, naʿt, silah and jar–majrur, and small for ʿatf and discourse. Classical
categories map onto C: tamm (no crossing arcs), kafi (only semantic or discourse arcs cross), hasan
(syntactic arcs cross but the left clause is complete), qabih (core arcs cross). Ibtida at j ≤ i is
scored by the incoming core arcs into [j..]. **Validation is free:** the mushaf's own signs are labels,
so check that C predicts the sign class (AUC). **Plug-in.** `parser.py` (Word.stop_after),
`segmenter.py`, and new stop diagnostics.

### G6 [M] Spectral clustering of the CTC confusion graph vs the classical makharij
Let W_ab be the expected posterior mass on letter b during frames aligned to letter a, over the
reference reciters. Take a normalised-Laplacian spectral embedding and measure the adjusted Rand index
against the makhraj partition, plus a Mantel test or Gromov–Wasserstein between the commute-time metric
and G1's d. Use it as a **model audit**: it tells you which letter pairs the aligner can distinguish, and
therefore where G1 must abstain.

### G7 [S now] Graph neural networks
GraphNeuralNetworks.jl (GAT/GCN, Kipf–Welling arXiv:1609.02907) and neural sheaf diffusion (Bodnar et al.
arXiv:2202.04579; Hansen–Gebhart arXiv:2012.06333) could learn restriction maps and verdicts on the
text graph. There are **no labelled learner errors**, and 700 ayahs from fewer than 10 reciters would
overfit any GNN. Revisit after several thousand annotated learner errors exist. Until then, G1–G5 use
fixed, interpretable operators.

---

## 4. Riemannian geometry, information geometry, optimal transport

### R1 [M/L] Fisher–Rao where categories have no metric; OT where they do
- **Simplex:** d_FR(p,q) = 2 arccos Σ_i √(p_i q_i), which is Čencov-unique (Amari–Nagaoka 2000). Use it
  for a reciter's *choice distributions*: madd ʿarid length, jawaz al-wajhayn heavy/light (the engine
  already reports the realised variant), and qalqalah level. Consistency is the entropy of the choice
  distribution. Comparison is FR to the peers.
- **Univariate location–scale**, for example log-normal durations per rule:
  d_FR = √2 · arccosh(1 + ((μ₁−μ₂)²/2 + (σ₁−σ₂)²)/(2σ₁σ₂)) (hyperbolic half-plane). This covers location
  and scale in one unit-free number, and complements the roadmap's zero-mean AIRM.
- **Posteriorgram DTW** of learner vs Husary for the same ayah, with Hellinger/FR ground cost on the
  frame posteriors (Hazen et al. 2009 posteriorgram templates; GOP, Witt–Young 2000
  doi:10.1016/S0167-6393(99)00044-8).
- Principle: **FR ignores the ground metric, OT uses it.** Pitch classes live on a circle and makharij on
  a tree, so they get OT (R4, G1). Unordered categories get FR.

### R4 [M/L] Tonic-invariant circular OT for maqam
Pitch-class measures live on S¹, where W₁ has a closed form (Rabin–Delon–Gousseau JMIV 2011,
doi:10.1007/s10851-011-0284-0; Delon–Salomon–Sobolevski arXiv:0902.3527):

    W₁(μ,ν) = min_α ∫₀¹ |F_μ(θ) − F_ν(θ) − α| dθ   (α = a median of F_μ − F_ν)

The tonic-invariant distance is D(μ,ν) = min_ρ W₁(μ, R_ρ ν). The argmin ρ **is the relative tonic
(qarar)**. Classify maqam by nearest template: the 8 maqamat of Maqam-478 (Shahriar & Tariq, IEEE Access
9:117271, 2021) and pitch-histogram templates in the style of Gedik–Bozkurt (Signal Processing 90(4),
2010). Combine with T3's persistent scale degrees for the interval inventory, with signatures (P1/P2)
for motion, and with roadmap C3 for recurrence. **Caveat:** Husary's murattal uses little maqam, while
the imams and mujawwad recordings are where this matters. Validation: accuracy on Maqam-478 against its
published CNN baselines, and ρ agreement with a manual tonic on 30 ayahs. 2–3 d, CPU, Octave-portable
(sort + cumsum).

### R2 [S/M] Grassmannians for voice subspaces and dynamics
For centred frame features X ∈ ℝ^{T×n}, take the top-k PCA subspace [U] ∈ Gr(k,n) and the principal
angles θ = arccos σ(U₁ᵀU₂), with the projection kernel ‖U₁ᵀU₂‖²_F (Hamm–Lee ICML 2008). **Real
argument for it:** a convolutive channel is an additive constant in the cepstrum and is removed by
centring, so the subspace is channel-invariant. It is **not** invariant to reverberation longer than the
frame (taraweeh), and ECAPA will beat it for ID. Its better use is comparing *dynamics*: the observability
subspaces of Hankel-DMD/LDS models (roadmap B2) compared with the Martin/Grassmann distance (Martin, IEEE
TSP 48(4) 2000; Turaga et al. TPAMI 2011). Manifolds.jl provides `Grassmann`, the Fréchet mean and
`Manopt` (Axen et al. arXiv:2106.08777).

### R3 [S] Gromov–Wasserstein when there are no shared coordinates
GW²(X,Y) = min_π Σ |D_X(i,k) − D_Y(j,l)|² π_ij π_kl (Mémoli FoCM 2011, doi:10.1007/s10208-011-9093-5).
**Honest:** the engine's letters are labelled, so between reciters the correspondence is *known*. Use
Procrustes/RSA on labelled centroid distance matrices, not GW. GW and fused GW (Vayer et al.
arXiv:1805.09114) are for three cases. (a) Unsupervised units, such as per-reciter HuBERT k-means
clusters. (b) Aligning embedding spaces of two SSL models (Alvarez-Melis–Jaakkola arXiv:1809.00013).
(c) Restart-robust matching: FGW on text graphs can match a learner who *goes back and repeats*, which
monotone DTW cannot. Use POT via PythonCall. OptimalTransport.jl covers Sinkhorn and 1-D but has no
mature GW.

---

## 5. Rough paths beyond the roadmap

### P2 [L] Lévy area as a reparametrisation-invariant *precedence index*
For a 2-D path (x, y), the Lévy area is A = ½∫(x−x₀)dy − (y−y₀)dx, the antisymmetric level-2 signature
term. For a path that first moves x by a and then y by b, A = ab/2. In the reverse order, A = −ab/2.
Define

    ρ = 2A / (Δx·Δy) ∈ [−1, 1]

ρ = +1 means x's move entirely precedes y's, 0 means simultaneous, and −1 means y first. The index is
invariant to tempo and to any reparametrisation, and its sign is invariant to monotone rescaling of
either coordinate for monotone moves. **Tajweed fit:**
- **Ikhfa anticipation.** x = F2 displacement toward the following consonant's locus, y = nasality. A
  correct ikhfa has the tongue anticipating *during* the ghunnah, so ρ takes a specific sign. This
  replaces the binary `ikhfa_formant_anticipation_score` with a continuous, calibratable value.
- **Iqlab.** Lip closure (energy drop in the 1–3 kHz band) against nasal onset.
- **Qalqalah.** Closure before burst.
- **Idgham naqis.** Itbaq F2 lowering retained *before* the ت closure.

**Plug-in.** The noon_sakinah, meem_sakinah, qalqalah and idghaam validators produce new metrics
`*_precedence`. Twenty lines of numpy, plus a Julia/Octave cross-check. Calibrate the bands on peers.

### P1 [M] Signature kernel and MMD two-sample tests on path laws
The kernel k(x,y) = ⟨S(x),S(y)⟩ solves the Goursat PDE ∂²k/∂s∂t = ⟨ẋ_s, ẏ_t⟩ k, with k(0,·) = k(·,0) = 1
(Salvi et al. arXiv:2006.14794). A finite-difference solve costs O(L_x L_y) with no truncation.
Chevyrev–Oberhauser tensor normalisation makes the kernel characteristic for laws of unparameterised
paths (arXiv:1810.10971; Király–Oberhauser arXiv:1601.08169). Then

    MMD²(P,Q) = E k(X,X') + E k(Y,Y') − 2 E k(X,Y)

with a permutation test. Uses: "is this imam's *distribution* of ayah-level (log-F0, F1, energy) paths
the same as the anchor-peer law?", fatigue (early vs late rak'ah), and style retrieval. The roadmap
features describe single paths. MMD tests *laws* and plugs into A4's conformal logic. Cost: about 700
ayahs at 100 points each is 2.5e5 pairs × 1e4 operations, which takes minutes on CPU in Julia. No GPU.

### P3 [note] Invariance catalogue
Signatures of log-F0 *increments* are automatically **tonic-invariant** (translation) and
**tempo-invariant** (reparametrisation). That is the maqam equivalence class of melodic *motion*, but not
of the interval *inventory*, which comes from T3/R4. Add a text-index channel (letter position) for
features that are tempo-free but anchored to the text. Add a time channel only for timing-scoring
features.

---

## 6. Unifying picture
The recited text is a cell complex. Letters are vertices, adjacency and repetition relations are edges,
and words, ayahs and surahs are the hierarchy tree. Each measurement is a cochain on it.
- **Laplacians** handle smoothing and consistency: tempo (S2), replicates (G4), reconciliation (S1) and
  hierarchy (A5 = tree GMRF).
- **Persistence** handles stable counting and extraction of events: taps, releases, scale degrees and
  formant mergers.
- **OT/FR metrics on stalks** handle comparison: tree-W₁ for makharij, circular W₁ for maqam, FR for
  choices, and the roadmap's BW/AIRM for continuous metric clouds.
- **Random walks** propagate uncertainty (G2) and blame (G3).

## 7. Build plan (CPU only; GPU not needed)

| Phase | Items | Effort | Gate to pass |
|------|-------|--------|--------------|
| 1 (wk 1) | T1 prominence counting; G3 incidents; P2 precedence; G2 FFBS | ~6 d | Synthetic-trill accuracy; Praat 150-token set; Husary takreer flag → 1; interval coverage of Octave edges |
| 2 (wk 2–3) | S2 haraka field + global madd choice; G1 makhraj metric + parser-table tests + tree-W₁; R4+T3 maqam | ~9 d | Within-count variance ↓ on peers; peer-LOO coverage (A4); Maqam-478 accuracy |
| 3 (mo 2) | G4 replicate ICC (needs full surahs); S1 sheaf reconciliation; G5 waqf cuts (keep waqf signs!); P1 sig-kernel MMD; R1 FR; T2 topological formants; H1 HodgeRank | ~20 d | Incremental peer/imam separation; mushaf-sign AUC; ICC-based ruler choice vs CV |
| 4 (research) | T4 torus/DTM/multiparameter; T5 zigzag fatigue; R2 Grassmann/Martin; R3 GW/FGW; S3 tahrirat CSP (needs scholar table); G7 GNN (needs labels) | open | Must beat Phase 1–3 features on ayah-disjoint mAP |

**Julia.** Create a *separate* sub-environment, for example `substrate_library/julia/topo/`, so the
pinned QaariLab Manifest stays untouched. Packages: `Ripserer` (JOSS doi:10.21105/joss.02614; Rips,
cubical, alpha), `PersistenceDiagrams` (bottleneck, Wasserstein, landscapes, images), `Graphs` +
`SimpleWeightedGraphs`, `SparseArrays`, `KrylovKit`, `Manifolds`/`Manopt`, and `OptimalTransport`.
Through `PythonCall`: `POT` (GW/FGW), `iisignature`, and `multipers`. Signature kernels (Goursat) and
tree/circle W₁ are under 100 lines of native Julia each.
**Octave cross-check feasibility.**
- Easy: 1-D union-find persistence (T1, T3), tree-W₁, circular W₁, the Lévy area, sparse Laplacian
  solves (S2, S1), and the FR formulas.
- Moderate: the cepstral antiformant (T2), which extends `qaari_features.m`.
- Not feasible: Rips/multiparameter PH. Cross-check those Ripserer.jl ↔ Python `ripser` instead.

**Validation discipline for everything.**
1. Invariance claims are tested by perturbation: time-stretch ±20%, pitch-shift, channel EQ, reverb and
   noise.
2. Every verdict change is gated on peer-LOO conformal coverage (roadmap A4) and on peer-vs-imam
   separation.
3. Python ↔ Octave agreement serves as an independent-ruler check.
4. A small Praat gold set of about 300 tokens covers taps, releases and boundaries.
5. No new fingerprint dimension enters without ayah-disjoint retrieval-mAP gain.

## 8. References (primary)
Cohen-Steiner, Edelsbrunner, Harer, DCG 37:103 (2007) doi:10.1007/s00454-006-1276-5 · Edelsbrunner, Letscher, Zomorodian DCG 28 (2002) doi:10.1007/s00454-002-2885-2 · Chazal, de Silva, Glisse, Oudot arXiv:1207.3674 · Cohen-Steiner, Edelsbrunner, Morozov, *Vines and vineyards*, SoCG 2006 doi:10.1145/1137856.1137877 · Carlsson & de Silva arXiv:0812.0290 · Tymochko, Munch, Khasawneh arXiv:2009.08972 · Gakhar & Perea arXiv:2103.04540 · Anai et al. arXiv:1811.04757 · Chazal et al. arXiv:1412.7197 · Botnan & Lesnick arXiv:2203.14289 · Loiseaux et al. arXiv:2306.03801 · Bubenik arXiv:1207.6437 · Adams et al. arXiv:1507.06217 · Skraba & Turner arXiv:2006.16824 · Kerber, Morozov, Nigmetov arXiv:1606.03357 · Robinson & Turner arXiv:1310.7467 · Fasy et al. arXiv:1303.7117 · Čufar, Ripserer.jl doi:10.21105/joss.02614 · Hansen & Ghrist arXiv:1808.01513 · Robinson arXiv:1805.08927, arXiv:1603.01446 · Abramsky & Brandenburger arXiv:1102.0264 · Bodnar et al. arXiv:2202.04579 · Hansen & Gebhart arXiv:2012.06333 · Lim arXiv:1507.05379 · Jiang, Lim, Yao, Ye arXiv:0811.1067 · Wang, Sharpnack, Smola, Tibshirani arXiv:1410.7690 · Rue & Held, *GMRFs* (2005) · Shuman et al. arXiv:1211.0053 · Hammond, Vandergheynst, Gribonval arXiv:0912.3848 · Kipf & Welling arXiv:1609.02907 · Carter & Kohn, Biometrika 81 (1994) · Graves et al. ICML 2006 doi:10.1145/1143844.1143891 · Evans & Matsen JRSS-B 74 (2012) doi:10.1111/j.1467-9868.2011.01018.x · Le, Yamada, Fukumizu, Cuturi arXiv:1902.00342 · Dukes & Buckwalter, INFOS 2010; Dukes, Atwell, Habash LRE 47 (2013) doi:10.1007/s10579-011-9167-7 · Amari & Nagaoka (2000) · Witt & Young, Speech Comm. 30 (2000) doi:10.1016/S0167-6393(99)00044-8 · Rabin, Delon, Gousseau JMIV 41 (2011) doi:10.1007/s10851-011-0284-0 · Delon, Salomon, Sobolevski arXiv:0902.3527 · Shahriar & Tariq, IEEE Access 9:117271 (2021) doi:10.1109/ACCESS.2021.3098415 · Gedik & Bozkurt, Signal Processing 90(4) (2010) doi:10.1016/j.sigpro.2009.06.017 · Hamm & Lee ICML 2008 doi:10.1145/1390156.1390204 · Martin IEEE TSP 48(4) (2000) doi:10.1109/78.827549 · Turaga et al. TPAMI 33(11) (2011) doi:10.1109/TPAMI.2011.52 · Axen et al. arXiv:2106.08777 · Mémoli FoCM 11 (2011) doi:10.1007/s10208-011-9093-5 · Vayer et al. arXiv:1805.09114 · Alvarez-Melis & Jaakkola arXiv:1809.00013 · Salvi et al. arXiv:2006.14794 · Chevyrev & Oberhauser arXiv:1810.10971 · Király & Oberhauser arXiv:1601.08169.
