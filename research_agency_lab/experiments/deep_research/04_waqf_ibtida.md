# 04 — Waqf & Ibtida' (stopping and restarting): models for qaari-eval

Deep-research report, 2026-09-23. Scope: the kinds of waqf, the Madani waqf signs, the waqf forms (sukoon, raum, ishmam, ibdal), sakt, ibtida', and how ijazah examiners judge them. It builds on `frontier_math_roadmap.md`, reusing its A4 conformal thresholds, A5 hierarchical model and B1 beat normalisation.
Status legend: **[verified]** means checked against the code or data in this repo during this session. **[hyp]** marks a hypothesis or prior that still has to be calibrated.

---

## 0. What the engine does today, and three bugs found

**Where stops come from.** Stops are an *input* to the engine. The inputs are:
- an ayah end, which `passage_parser` and `_prepare_words` always treat as a stop;
- pauses of 0.30 s or more that `fatigue_detector.dynamic_stops` finds in mid-ayah.

In both cases `pipeline.py:170` re-parses the text with the new stop and applies the waqf changes. Nothing scores *where* the reciter stopped. Nothing scores *where the reciter restarted* either: the main pipeline's forced alignment is monotone, so a repeat-back after a stop (the correct ibtida' after a qabih stop) breaks the alignment. The taraweeh `segmenter._forward_chain` tolerates at most 3 words of backtrack between chunks.

**What happens to the waqf signs.** The signs are thrown away. The Tanzil `quran-uthmani` text, cached at `~/.cache/qaari-eval/text/quran-uthmani.json`, carries every Madani pause mark as a **separate space-delimited token** after its word. `parser._DROP` (`[ۖ-ۜ…]`) deletes that token and `_prepare_words` then `continue`s past it. The data is already on disk; we only need to stop discarding it.

**Bugs found (all verified by running the parser in `.venv`):**
1. **False mandatory sakt.** U+06DC (small high seen) also appears *inside* two words, where it marks the ص-read-as-س of وَيَبْصُۜطُ (2:245) and بَصْۜطَةً (7:69). `_prepare_words` checks `if SAKT_MARK in tok`, so both words make the *previous* word carry a "mandatory in Hafs" SAKT rule (يَقْبِضُ in 2:245, ٱلْخَلْقِ in 7:69). Every reciter will FAIL there. **Fix:** set sakt only when the token *is* U+06DC standing alone (`tok == "ۜ"`).
2. **U+06E0 alif is silent even at waqf.** The small high upright rectangular zero (۠, 66 occurrences) marks a letter that is dropped in wasl but **pronounced at waqf**. Examples are أَنَا۠ (×60) and لَّٰكِنَّا۠, plus ٱلظُّنُونَا۠, ٱلرَّسُولَا۠, ٱلسَّبِيلَا۠ and قَوَارِيرَا۠ at the ends of 33:10, 33:66, 33:67 and 76:15. `_make_unit` treats `RECT_ZERO` the same as `ROUNDED_ZERO` and marks the letter `silent`, and `_apply_waqf` never brings it back. The results are wrong:
   - 33:10 at waqf gives "…ẓ-ẓunūn" with a spurious **madd ʿarid** on the waw.
   - 76:15 gives a sakin raa judged *tarqeeq after yaa*.
   - The correct reading of both is fatha + alif (madd tabiʿi, 2 counts), with the raa of 76:15 heavy.
   **Fix:** keep a `waqf_only` flag on the unit and un-silence it in `_apply_waqf`. This applies both to the phrase-final word and to أَنَا۠ when a stop is inserted after it.
3. **Sakt at an ayah end conflicts with the assumed stop.** For عِوَجَا ۜ (18:1→18:2) and مَالِيَهْ ۜ (69:28→29), `passage_parser` stops after every ayah, but `_sakt_rules` still emits a SAKT rule. `validate_sakt` then reports "A full stop was made instead of a brief sakt" when the reciter makes the normal waqf on the ayah head. **Fix:** at an ayah head a sakt rule should exist *only if* the observed boundary is wasl. At 18:1 a full waqf on the ayah head is valid; the sakt applies only when the reciter joins the two ayahs.

**Other gaps:**
- Raum, ishmam, stopping on a hamza, the taa-marbuta → haa *acoustic* check, madd ʿarid consistency (taswiyah), and waqf on marsum al-khatt (for example رَحْمَتَ → t, and the ya'at zawa'id) are all un-modelled beyond the text transform.
- An ayah boundary joined in wasl is never detected, because the ayah-end stop is assumed.

---

## 1. Domain model (what examiners actually judge)

### 1.1 Kinds of waqf

**Kinds by the reciter's situation:**
- **Ikhtibari** (test). The student is asked to stop on a word to show its rasm: maqtuʿ/mawsul, taa maftuha versus marbuta, ya'at zawa'id, and so on.
- **Idtirari** (compulsion). The stop is forced by breath, a cough or forgetting. It is allowed anywhere, but the student **must restart from a sound place**, going back if needed.
- **Intizari** (waiting). Used in jamʿ al-qira'at, to append another riwayah. It does not arise for a single-riwayah Hafs reading.
- **Ikhtiyari** (by choice). This is the only kind graded by *quality*.

**Grades of an ikhtiyari stop** (Ibn al-Jazari, *al-Muqaddimah*, bab al-waqf wa-l-ibtida'; al-Dani, *al-Muktafa*):

| Grade | Link to what follows | Verdict |
|---|---|---|
| **Tamm** | none, neither lafzi (grammatical) nor maʿnawi (meaning) | stop, and begin with what follows |
| **Kafi** | maʿnawi only | stop, and begin with what follows |
| **Hasan** | lafzi | the stop is allowed, but beginning with what follows is *not*, except at a ra's al-ayah (sunnah of stopping on ayah heads: the hadith of Umm Salamah in Abu Dawud and al-Tirmidhi); otherwise go back |
| **Qabih** | the sense is incomplete, or wrong | only under compulsion, and **restart from before it** (وَيُبْدَا قَبْلَهُ) |

Two further points:
- *"وَلَيْسَ فِي القُرْآنِ مِنْ وَقْفٍ وَجَبْ وَلاَ حَرَامٌ غَيْرُ مَا لَهُ سَبَبْ"*: no stop is obligatory or forbidden in the Sharia sense unless meaning is corrupted. So the sign costs below are *soft*, except where a stop or join inverts the meaning.
- The classical graded location catalogues are:
  - Ibn al-Anbari, *Idah al-waqf wa-l-ibtida'*;
  - al-Nahhas, *al-Qatʿ wa-l-iʾtinaf*;
  - al-Dani, *al-Muktafa*, which labels tamm, kafi and hasan per location;
  - al-Sajawandi, *ʿIlal al-wuquf*, the origin of the sign letters;
  - al-Ashmuni, *Manar al-huda*, a grade per location.

  Digitising *al-Muktafa* or *Manar al-huda* would give a **graded gold label** that is finer than the six signs.

### 1.2 Madani mushaf signs (King Fahd Complex; Tanzil "pause marks" doc) — counts from the local Tanzil text [verified]
| sign | Unicode | meaning | count in Tanzil |
|---|---|---|---|
| ۘ (م) | U+06D8 | lazim: must stop (joining distorts meaning, e.g. 36:76, 10:65, 5:73) | 22 |
| ۙ (لا) | U+06D9 | stopping not permitted (unless a restart repairs it) | 68 |
| ۚ (ج) | U+06DA | permissible, both options equal | 1972 |
| ۖ (صلى) | U+06D6 | stop allowed, continuing preferred | 1682 |
| ۗ (قلى) | U+06D7 | continue allowed, stopping preferred | 603 |
| ۛ ۛ (∴) | U+06DB | muʿanaqah: stop on either one, never both | 12 = 6 pairs (2:2, 2:195, 5:26, 5:41, 7:172, 14:9) |
| ۜ | U+06DC | sakt when standalone (4 in Hafs + optional 69:28); in-word, ص read as س | 5 + 2 |
| ۠ | U+06E0 | letter pronounced only at waqf | 66 |

Ayah ends carry no mark; they are implicit. Across the whole Quran (77,649 words), the gap between stop-permitted points (any sign other than لا, or an ayah end) has a median of **6 words**, a p90 of 14, a p99 of 24 and a max of 49. This calibrates the breath budget in §3.

### 1.3 Forms of waqf (Shatibiyyah, bab al-waqf ʿala awakhir al-kalim; Ibn al-Jazari, *al-Nashr*)
- **Sukoon mahd.** This is the default.
- **Raum.** إسماع الحركة بصوت خفي: the vowel is made audible in a faint voice that only a listener close by can hear. Tradition puts it at about ⅓ of a harakah.
  - Allowed on damma and kasra only; never on fatha.
- **Ishmam.** Closing the lips, rounded as for a damma, right after the sukoon, **with no sound**. Only a sighted observer can see it.
  - Allowed on damma only.
- **Where neither raum nor ishmam is allowed:**
  - haa' al-ta'nith;
  - mim al-jamʿ;
  - an ʿarid vowel;
  - haa' al-kinaya after ḍamma/waw or kasra/yaa (the stricter, preferred view).
- **Ibdal:**
  - tanween fath → alif ('iwad);
  - taa marbuta → haa sakinah.
  - Hafs does not change a hamza at waqf: it is kept sakin and clear.
- **Madd at a stop:**
  - madd ʿarid li-l-sukoon takes 2, 4 or 6 counts with sukoon or ishmam, but **only 2 with raum**, because raum behaves like wasl;
  - muttasil at a stop (السَّمَاءُ) takes 4, 5 or 6, and 4 or 5 with raum;
  - leen at a stop takes 2, 4 or 6, and none with raum.
- **Examiners also require *taswiyah*.** The chosen ʿarid length must stay consistent through the reading.
- **Qalqalah kubra** happens at a stop on ق ط ب ج د. Raum on such a letter removes the qalqalah.

### 1.4 Ibtida' (where to restart)
Ibtida' is always ikhtiyari: the restart must be a meaningful, self-contained start.
- A hasan stop that is not on an ayah head needs a go-back.
- A qabih stop needs a go-back to a point before the dependency.
- A restart must never begin with a phrase that distorts the meaning (يَدُ ٱللَّهِ مَغْلُولَةٌ, 5:64, as a new utterance).
- Hamzat al-wasl is pronounced on restart. The parser already does this: see `_resolve_wasla(ibtida)`.

---

## 2. Pause and boundary detection as signal processing

### 2.1 Observation model per candidate boundary b (after word w)
Let t_b be the aligned end of w. The feature vector is
x_b = [ s (silence duration), β (breath evidence: HNR<3 dB, spectral flatness, from `detect_breath`), ℓ (final lengthening of the last rhyme, ℓ = d_rhyme / (local haraka · expected count), using the beat normalisation from roadmap B1), ΔF0 (pitch reset: median log-F0 of the first 150 ms after the pause minus the last 150 ms before it, in cents), ΔE (energy reset, dB), v (voicing offset type: sukoon closure, residual vowel, or glottal) ].

Prosody research finds that pause duration increases *monotonically* with boundary strength. Final lengthening and pre-boundary pitch excursion are *non-monotonic*: they peak at intermediate breaks ("The final lengthening of pre-boundary syllables turns into final shortening as boundary strength levels increase", J. Phonetics 2023, sciencedirect.com/science/article/pii/S0095447023000141; review by Wagner & Watson 2010, *Lang. Cogn. Proc.* 25:905). Pitch reset is the most reliable production cue. Pause length also scales with the length of the *upcoming* phrase, because of breath planning (Krivokapić 2007, *J. Phonetics* 35:162; Fuchs et al. 2013, *J. Phonetics* 41:29).

### 2.2 Boundary classes and likelihoods
The per-boundary hidden class is k ∈ {wasl, sakt, waqf, waqf+breath, hesitation}. Model:
p(x_b | k) = LogNormal(s; μ_k, σ_k) · Bern(β; π_k) · N(ΔF0; m_k, τ_k) · …

Here s = 0 for wasl, and the sakt constraints are the ones already in `sakt_wasl.py`: 200–400 ms, β = 0. Hesitation (tanahhud or uncertainty) differs from waqf by a missing final lengthening, a missing F0 fall or reset, and a mid-word position.

Posterior strength is P(waqf | x_b) ∝ p(x_b|waqf) · P(waqf | sign_b). **The sign prior enters here.** A 380 ms silence after a قلى word is a waqf; the same silence after a لا word is more likely a sakt-like hesitation or a stall.

**Plug-in.** Replace the hard `DYNAMIC_STOP_S = 0.30` threshold in `dynamic_stops` with the posterior. Emit a `BoundaryEvent(word, class, posterior, s, β, ΔF0)` list, which also feeds §3. Sakt already uses a silence without breath, so reuse `silence_runs` and `detect_breath`.

**Calibration.** On reference reciters (Husary murattal and muʿallim, and the existing `everyayah` runs), fit μ_k and σ_k per reciter with hierarchical pooling (roadmap A5). The reciter's tempo enters through s / haraka_ms. Set thresholds with conformal LOO-peer thresholds (roadmap A4).

**Validation.**
- For boundary detection, use frame-level F1 against `obadx/recitation-segmenter-v2`. That is a wav2vec2-BERT pause segmenter trained on about 850 h of expert recitation, with frame F1 of 0.996. It is MIT-licensed and needs about 3 GB of GPU.
- For stop/continue classification, use word-boundary precision and recall against hand labels on 300 boundaries.
- Breath: Ruinskiy & Lavner (IEEE TASLP 15(3):838, 2007) report 98% detection with MFCC templates, which is a better detector than a pure HNR threshold for `detect_breath`.

**Compute.** Everything is CPU except the optional wav2vec2-BERT cross-check, which needs about 3 GB of GPU or runs slowly on CPU.

---

## 3. Decision-theoretic scoring of stop choices

### 3.1 The text as a labelled path/graph
Words 1..N form a chain, and the boundary after word i carries a label σ_i ∈ {none, لا, صلى, ج, قلى, م, ∴, ayah-end}.

The chain is enriched with a **syntactic–semantic dependency graph** G = (V = words, E = i'rab dependencies) from the Quranic Arabic Dependency Treebank (Dukes & Buckwalter 2010; Dukes & Habash, LREC 2010; corpus.quran.com). For boundary i, the *cut set* is C_i = {(u,v) ∈ E : u ≤ i < v}. Its weighted size κ_i = Σ_{e∈C_i} w(type e) is a **graph-theoretic lafzi-link measure**. Mapping to the classical grades:
- κ_i = 0 and no maʿnawi link: tamm;
- κ_i = 0 with a maʿnawi link: kafi;
- κ_i > 0 through a *complete* clause: hasan;
- the cut severs head–complement, muḍaf–muḍaf ilayh, or ṣifa–mawṣuf: qabih.

The edge-type weights w(·) are learned so that κ reproduces the signs; see §3.4.

### 3.2 Costs
- **Stop cost:** c_stop(i) = a[σ_i] + γ·κ_i + η·meaning_inversion_i. The inversion term is 0 or ∞-like, and is set for the 22 lazim continuations and the forbidden restarts.
- **Continue cost:** c_cont(i) = b[σ_i]. It is large for م.
- **Ibtida' cost:** c_start(r) is the cost of beginning at word r. It is high if word r depends on the left (κ-based), or if it starts a meaning-inverting quotation.
- **Repeat cost:** ρ per word re-read on going back.
- **Breath budget:** λ·max(0, T(r..i) − T_max)², where T is the predicted phrase duration and T_max is per reciter, estimated from their longest breathless phrases. This is what makes idtirari stops *justified* instead of penalised.

**Starting table [hyp]:**

| Sign | none | لا | صلى | ج | قلى | م | ∴ | ayah |
|---|---|---|---|---|---|---|---|---|
| a (stop) | 5 | 4 | 0.6 | 0.3 | 0 | 0 | 0.2 | 0 |
| b (continue) | 0 | 0 | 0 | 0.3 | 0.6 | 8 | 0 | 0.4 |

For ∴, the constraint is that the reciter must not stop on both members.

### 3.3 Optimal segmentation = shortest path (DP)
A plan is a sequence of (restart r_k, stop i_k), where r_{k+1} ∈ [i_k+1−K, i_k+1]; a value r ≤ i_k means going back. The recursion is

V(i, m) = min_{j<i, m', r} V(j, m') + c_start(r) + ρ(j+1−r) + c_stop(i) + Σ_{r≤k<i} c_cont(k) + λ·max(0, T(r..i) − T_max)².

Here m ∈ {0,1} is the muʿanaqah state bit, "stopped on the first ∴ of the open pair". The complexity is O(N² K); in practice O(N · 24 · K), using the p99 of 24 words.

This is a finite-horizon **MDP**. The state is (position, breath reserve, m), the actions are {continue, stop and restart at r}, and the reward is −cost.

**Scoring the reciter.** For each observed stop, the decision advantage is
A_k = Q(s_k, a_obs) − V*(s_k).
The feedback, per stop, is:
- **"optimal"** when A = 0;
- **"permitted, not preferred"** when A is small, as with صلى;
- **"stop justified by breath"** when the breath term dominates;
- **"qabih without go-back"** when A is large and r = j+1.

The total **regret** is Σ A_k. Because a reciter who ran out of breath is only compared against plans that were feasible under their own T_max, idtirari stops are not punished.

**Julia (tested in this session, Julia 1.10; to be added to `QaariLab` as `src/Waqf.jl`):**
```julia
@enum Sign NONE LA SLY J QLY M MU AYAH
const CONT = Dict(NONE=>0.0, LA=>0.0, SLY=>0.0, J=>0.3, QLY=>0.6, M=>8.0, MU=>0.0, AYAH=>0.4)
const STOP = Dict(NONE=>5.0, LA=>4.0, SLY=>0.6, J=>0.3, QLY=>0.0, M=>0.0, MU=>0.2, AYAH=>0.0)
function plan(sgn, dur, start_cost; Tmax=9.0, λ=2.0, K=4, rep=0.15, mu=Tuple{Int,Int}[])
    N=length(sgn); cum=[0.0;cumsum(dur)]; cc=[0.0;cumsum(getindex.(Ref(CONT),sgn))]
    openb=zeros(Int,N); isfirst=falses(N)
    for (a,b) in mu; isfirst[a]=true; openb[a:b-1].=b; end
    V=fill(Inf,N+1,2); V[1,1]=0.0; bp=fill((0,0,0),N+1,2)
    for i in 1:N, j in 0:i-1, m in 1:2
        isfinite(V[j+1,m]) || continue
        m==2 && i==openb[j] && continue                      # both members of a mu'anaqah pair
        m2 = isfirst[i] ? 2 : (m==2 && openb[i]!=0 ? 2 : 1)
        for r in (j==0 ? (1:1) : (max(1,j+1-K):j+1))         # r<=j : repeat back on restart
            c = start_cost[r] + rep*(j+1-r) + STOP[sgn[i]] + (cc[i]-cc[r]) +
                λ*max(0.0, cum[i+1]-cum[r]-Tmax)^2
            if V[j+1,m]+c < V[i+1,m2]; V[i+1,m2]=V[j+1,m]+c; bp[i+1,m2]=(j,m,r); end
        end
    end
    m=argmin(V[N+1,:]); v=V[N+1,m]; segs=Tuple{Int,Int}[]; i=N
    while i>0; j,mp,r=bp[i+1,m]; pushfirst!(segs,(r,i)); i,m=j,mp; end
    v, segs
end
```
Sanity runs, at 0.6 s per word:
- **2:2, T_max = 9 s:** the plan is a single phrase (wasl through both ∴).
- **2:2, T_max = 2.5 s:** the plan is `[(1,4),(5,9)]`, stopping on the first ∴ only.
- **2:2, T_max = 1.5 s:** a forced qabih stop appears, but the plan still never stops on both ∴.
- **5:73:** the plan is `[(1,8),(9,14),(15,27)]`, stopping at the lazim ۘ after ثَلَٰثَةٍ. Joining through it costs 8.3.

**Octave cross-check.** The same recursion is about 25 lines of loops. Compare V* and the argmin paths on all 6,236 ayahs with random durations, and require the paths to be identical.

**Plug-in.** Add a new `app/tajweed_rules/waqf_choice.py` validator with a `RuleType.WAQF_CHOICE` per observed boundary, and an `IBTIDA` rule per restart. Its inputs are the `BoundaryEvent`s from §2 and per-word durations from the alignment. Sign labels come from a parser change that keeps the standalone marks as `Word.pause_mark`, instead of dropping them in `_DROP`.

### 3.4 Learning the costs (inverse optimisation)
The costs a, b, γ, ρ and the edge weights w(·) can be treated as a structured-prediction or **inverse reinforcement learning** problem. The reference reciters' observed plans should be near-optimal.
- Fit with the max-margin / structured-perceptron objective min_θ Σ_rec [cost_θ(obs) − min_π (cost_θ(π) − Δ(π, obs))]₊, which is convex in θ for fixed paths.
- Alternatively, fit a CRF over boundaries with the DP as the partition function (a semi-Markov CRF).
- **Calibration data:**
  - Husary *muʿallim*, which uses textbook stops;
  - Husary / Minshawi / al-Banna murattal;
  - the 286k-utterance waqf-segmented corpus of arXiv:2509.00094 (Abdelfattah, Khalil & Abbas, CC BY 4.0). Its segment boundaries are observed reciter stops and are directly usable to estimate empirical P(stop | sign).
- **Text-only sanity data:** the Boundary-Annotated Qur'an corpus (Brierley, Sawalha & Atwell, LREC 2012 L12-1091; *J. Speech Sciences* 2012) already maps the signs to phrase-break labels over 77,430 words, with POS tags.

### 3.5 Joint alignment + pause + restart decoding (HSMM / WFST)
The current monotone CTC alignment cannot represent a go-back. The proposal is a hidden **semi-Markov** decoder (Yu 2010, *Artif. Intell.* 174:215, doi:10.1016/j.artint.2009.11.011) over a graph with the following parts:
- **Word states** with explicit duration pmfs.
- **Boundary states** between words, with three kinds:
  - wasl (duration 0);
  - sakt (200–400 ms, no breath);
  - pause (log-normal duration, breath emission).
- **Backward jump arcs** from each pause state *after* word i to the start of any word r ∈ [i+1−K, i+1] in the same ayah, carrying cost c_start(r) + ρ(i+1−r).

The emission terms are the existing CTC posteriors, used as scaled likelihoods for the word states, and the §2 features for the boundary states. The Viterbi search costs O(T · N · K · D_max). With frame-synchronous pruning it runs on CPU for ayah-length audio. The observed (r_k, i_k) then feed §3.3 directly.

**Plug-in.** This is a new `ctc_restart_align` in `segmenter.py`: the same semi-global CTC code, plus jump arcs placed only at pause frames of 0.3 s or more. That keeps the search tractable.

**Validation.** Build a synthetic go-back test by splicing reference audio so that it repeats 1–4 words after a pause, and measure restart-word accuracy.

---

## 4. Raum and ishmam: detection theory

### 4.1 Raum versus sukoon versus full vowel (a vowel kept at the stop is an error)
**Window.** Take W = [release of the final consonant, +150 ms].

**Features:**
- ρ = d_voc / h, the residual voiced duration over the local haraka;
- Δ = level of the residual minus the level of the preceding vowel, in dB;
- F2 of the residual: low for ḍamma, about 800–1100 Hz; high for kasra, above 1900 Hz;
- voicing fraction.

**Priors [hyp]:**
- sukoon: ρ < 0.1;
- raum: ρ from 0.2 to 0.5, with Δ from −6 to −20 dB;
- full haraka (error): ρ ≥ 0.7 and Δ > −6 dB.

**Confound.** Qalqalah kubra on ق ط ب ج د gives a 30–80 ms schwa-like burst with central F2. Raum carries the *vowel quality*: F2 matches the underlying ḍamma or kasra. So F2 distance to the vowel target is the decisive feature on qalqalah letters.

**Hypothesis test.** Use H0 = sukoon, H1 = raum. Under a Gaussian frame-feature model with a per-frame deflection d′ over n frames (10 ms hop), the Neyman–Pearson likelihood-ratio test gives

P_D = Φ(√n·d′ − Φ⁻¹(1 − P_FA)).

Raum lasts about 50 ms, so n ≈ 5. At P_FA = 0.05, d′ = 1 gives P_D ≈ 0.72, and d′ = 1.5 gives P_D ≈ 0.96. So raum is **detectable in studio audio if per-frame d′ ≥ 1.5**. In reverberant audio (taraweeh), T60 tails can mask a −15 dB residual, so a raum verdict should be *abstained* on when the room-adapted SNR in W is under about 10 dB. For an unknown level, use the GLRT form: the energy detector gives a χ² null and a noncentral-χ² alternative, with ROC from the Marcum Q-function (Kay, *Fundamentals of Statistical Signal Processing*, Vol. II, 1998).

**Rule checks, done in the parser from the text:**
- raum on fatha: error;
- raum on haa' al-ta'nith, mim al-jamʿ or an ʿarid vowel: error;
- raum + madd ʿarid must be 2 counts, so a 4 or 6 count with raum is an error;
- raum on a qalqalah letter: no qalqalah expected.

**Plug-in.** Add a `WAQF_FORM` rule at every phrase-final unit whose `orig_vowel ∈ {DAMMA, KASRA}`. The metrics are {ρ, Δ, F2, posterior}. Madd ʿarid validation branches on the detected form. The Octave cross-check uses the existing `lpc.m` for residual formants.

**Calibration.** Husary muʿallim uses sukoon mahd almost always. Recordings of teachers demonstrating raum give positives; the ijazah channels of Egyptian shuyukh are a source. About 100 labelled positives are needed, and the ROC should be reported per SNR bin.

### 4.2 Ishmam
Ishmam is silent lip-rounding *after* the sukoon, so audio carries no reliable acoustic information. The only traces are faint:
- lowered spectral centroid of any post-release exhalation, as rounded lips act like a [ʍ]-filter;
- coarticulatory F2 lowering at the very end of a sonorant final (ن م ل ر).

With n ≈ 3–10 frames and d′ well below 0.5, P_D stays close to P_FA at any reasonable threshold. So **audio-only: treat ishmam ≡ sukoon and never penalise.** Report it as "unobservable".

A video option would use lip landmarks (MediaPipe FaceMesh): protrusion and the width/height ratio in the 100–300 ms after offset, with the same Neyman–Pearson test on the landmark trajectory. It is CPU-feasible.

### 4.3 Other waqf-form checks (cheap, CPU)
- **Taa marbuta → haa.** At the stop, expect a weak [h]: HNR < 3 dB with frication of 30–120 ms. The error is either a [t] burst (spectral transient) or a kept vowel (the §4.1 full-vowel class).
- **Stopping on a hamza.** Expect a glottal closure at the end: an abrupt energy drop of 20 dB or more within 20 ms, preceded by irregular glottal pulses or creak (jitter up). The error is dropping the hamza or adding an echo vowel. Muttasil + ʿarid length follows §1.3.
- **Taswiyah.** Across a session, the variance of the chosen ʿarid count, in beat-normalised count units (roadmap B1), should be small. Flag a mixture of 2/4/6 choices as inconsistent. A two-component mixture test, or a Sinkhorn test (roadmap A3), handles this.

---

## 5. Information-theoretic completeness at a boundary
For a boundary after word i in the ayah context h_{≤i}, define:
- **Continuation entropy** H_i = −Σ_w p(w | h_{≤i}) log p(w | h_{≤i}), and the surprisal of the true next word S_{i+1} = −log p(w_{i+1} | h_{≤i}). A semantically complete point has high H and high S: the next word is not forced.
- **Pointwise mutual information across the cut**, I_i = log p(w_{i+1..i+k} | h_{≤i}) − log p(w_{i+1..i+k} | ⟨ayah-start⟩). High I means a strong link, the lafzi or maʿnawi taʿalluq. Low I at tamm and kafi.
- **Ibtida' suitability** of word r: −log p(w_r | ⟨bos⟩) relative to its in-context surprisal, together with the dependency-graph test that no head lies to the left (κ).

Prosody and text are heavily redundant: Wolf et al., EMNLP 2023, arXiv:2311.17233, show that LLM features predict pause and duration well. This supports using these LM features as a learned sign prior.

- **Models:** CAMeLBERT-CA (classical Arabic; Inoue et al., arXiv:2103.06678), or any Arabic causal LM for H and S.
- **Precomputing:** H, S and I are computed once for all 77,649 boundaries. That is a one-off GPU job of minutes, or a CPU job overnight, stored as a table. Scoring then needs no GPU.
- **Validation:** AUC of (H, S, I, κ) for predicting the Madani sign classes, and Spearman correlation against the Dani/Ashmuni grades once digitised.

---

## 6. Data sources
- **Tanzil Uthmani text** (already cached locally), with the signs as standalone tokens at U+06D6–U+06DB and U+06DC. It follows the Medina mushaf; see tanzil.net/docs/pause_marks.
  - Caveat: sign placement varies between mushaf editions (Madani, Indo-Pak Sajawandi letters, Maghribi). Hafs is keyed to the Madani edition.
  - Cross-check the 22 lazim positions against a KFGQPC Hafs text.
- **Quranic Arabic Corpus** (corpus.quran.com): morphology, dependency treebank (i'rab), and pause-mark documentation.
- **kaisdukes/quran-neural-chunker:** a chunker that combines pause marks, the i'rab grammar (Salih, *al-Iʿrab al-mufassal*) and translation punctuation. Its features are relevant to κ and the sign prior.
- **Boundary-Annotated Qur'an corpus** (Leeds/Jordan; Sawalha, Brierley & Atwell, LREC 2012/2014): the signs mapped to break labels, with an IPA tier.
- **Audio:**
  - the `obadx/recitation-segmentation` dataset and `recitation-segmenter-v2` model (arXiv:2509.00094, CC BY 4.0 / MIT). These give reciter stop points at scale.
  - EveryAyah per-ayah audio (already in use);
  - quran.com / QUL word-level timing segments for several reciters (verify the licence before use).
- **Classical graded catalogues** (not yet digital in a structured form): al-Dani *al-Muktafa*, al-Ashmuni *Manar al-huda*, Ibn al-Anbari *Idah*, al-Sajawandi *ʿIlal al-wuquf*, and Ibn al-Jazari's *al-Nashr* vol. 1.

---

## 7. Prioritised build plan
| # | Item | Effort | GPU | Payoff |
|---|---|---|---|---|
| 1 | Fix bugs 1–3 in §0: standalone-ۜ only; `waqf_only` flag for U+06E0; ayah-head sakt only in wasl | ½ day | no | Removes certain false FAILs at 2:245, 7:69, 18:1, 69:28, 33:10/66/67, 76:15 and every أَنَا۠ |
| 2 | Keep the pause marks: `Word.pause_mark`; tests over all 6,236 ayahs checking the counts in §1.2 | ½ day | no | Enables everything below |
| 3 | `WAQF_CHOICE` validator: sign-cost table + لا/م hard checks + ∴ pair rule, using existing stops | 1 day | no | First real waqf feedback. The م continue-through and لا stop cases are the flagship ijazah errors |
| 4 | Boundary posterior (§2), replacing the 0.30 s threshold, with sakt/pause/breath classes; calibrate on Husary + everyayah | 2 days | no | Stops are detected, not assumed; ayah-head wasl is detected |
| 5 | `Waqf.jl` DP + advantage/regret scoring + breath-aware T_max per reciter; Octave cross-check | 2 days | no | Principled "permitted versus preferred versus justified by breath" |
| 6 | Restart-capable alignment (§3.5, jump arcs at pauses) + `IBTIDA` rule (go-back required after hasan/qabih) | 3–4 days | no (CPU Viterbi) | Handles the correct repeat-back, which currently misaligns |
| 7 | Waqf-form checks: taa marbuta→h, hamza at the stop, taswiyah of ʿarid | 2 days | no | Common ijazah corrections |
| 8 | Raum detector (§4.1) with SNR-gated abstention; ishmam declared unobservable | 3 days + labelling | no | Advanced/ijazah level |
| 9 | Dependency cut-set κ (QAC treebank) + LM entropy/PMI table; inverse-optimisation fit of costs on reference reciters + the arXiv:2509.00094 corpus | 1–2 weeks | a one-off GPU job of minutes (LM table); optional 3 GB for the wav2vec2-BERT cross-check | Learned, finer-than-sign grading, i.e. an estimated tamm/kafi/hasan label per location |
| 10 | Digitise the al-Dani/al-Ashmuni grades as a gold label; video ishmam | research | no | Gold-standard validation |

**Validation discipline** (consistent with the roadmap):
- Do per-rule LOO-peer conformal thresholds.
- Reference reciters should have regret close to 0 and zero م/لا violations. Imams in taraweeh should show stops that are mostly "justified by breath".
- Build a synthetic error suite: continue through each of the 22 lazim; stop on each of the 68 لا; stop on both ∴; restart without go-back; raum on fatha.
