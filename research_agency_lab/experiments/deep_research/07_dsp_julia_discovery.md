# 07 — Research-grade DSP for every Tajweed phenomenon, and a Julia law-discovery architecture

Deep-research report, 2026-09-23. Scope: (A) signal processing per Tajweed phenomenon, what is
worth building and why; (B) using Julia's scientific-ML stack to discover governing laws, with
Octave as an independent numerical check. Read before writing: `substrate_library/julia/src/{QaariLab,Discovery}.jl`,
`substrate_library/octave/qaari_features.m`, `experiments/frontier_math_roadmap.md`,
`experiments/discovery_latest.json`, `experiments/validation_20260923_115514.json`, and the engine
modules `app/acoustic/{features,tempo}.py`, `app/sifaat/*`, `app/tajweed_rules/{noon_sakinah,qalqalah_engine}.py`,
`app/taraweeh_adapter/*`. This report does not repeat roadmap items A1–C3; it covers what that
roadmap leaves out: the measurement layer and the discovery stack.

---

## 0. What the current artefacts already tell us (read these first)

These come from the repo's own outputs and limit what discovery can reach today.

| # | Observation | Evidence | Consequence |
|---|---|---|---|
| 0.1 | **Formants disagree between the two stacks.** Octave vs Python/Praat on 308 spans: F1 r = 0.34, F2 r = 0.34, p90 abs error 465 Hz (F1) and 1089 Hz (F2). `core_ms` agrees almost perfectly (r = 0.9975, median abs err 20 ms). | `validation_20260923_115514.json` | Errors of about 1 kHz on F2 are **formant-slot swaps** (a spurious pole taken as F1/F2), not noise. **Any discovery on tafkheem/itbaq/makharij from formants is premature** until the tracker is fixed (§A3). Timing laws can go ahead. |
| 0.2 | Likely causes of 0.1: the two paths use different analysis settings. Python uses Praat Burg with the ceiling set to 5000/5500 Hz (resampled to 10–11 kHz), a 25 ms window and 5 formants. Octave uses autocorrelation LPC at 16 kHz with order 18, a bandwidth limit under 400 Hz, no ceiling, the middle 40 % of the "core", and a different time window. Neither path uses frame-to-frame continuity. | `features.py:_formant_obj`, `qaari_features.m:span_features` | Match the ceiling and order first, then add a tracker with a continuity prior (Kalman). Report a "slot-swap rate" metric. |
| 0.3 | **The duration law is under-identified.** Husary core_ms: `D = −552 + 150·T + 394·n` (T in 100 ms units), R² = 0.68. BIC picked λ = 80, **the top of the λ grid**. The `n·T` term (the "counts scale with tempo" hypothesis) was dropped for `n`. | `discovery_latest.json` | (a) With one reciter at nearly constant tempo, `n` and `n·T` are almost collinear, so STLSQ cannot tell them apart. The data needs tempo variation: murattal vs mujawwad vs hadr, several reciters, Taraweeh. (b) A BIC optimum at the edge of the grid means the grid is too narrow. (c) The Θ columns have mixed scales (`n` is raw, `T` is in 100 ms units), so a single λ penalises coefficients by their units. **Z-score the columns before STLSQ** (standard SINDy practice). |
| 0.4 | `tempo_dynamics` returned empty `per_surah` and `pooled`. | `discovery_latest.json` | The 59-row sample never reaches `min_ayahs = 20` for any surah. The tempo ODE needs full-surah runs (EveryAyah gives full surahs for free). |
| 0.5 | **Julia version drift.** `Manifest.toml` was resolved with `julia_version = "1.11.5"`, but `/usr/local/bin/julia` is **1.10.9**. | `Manifest.toml`, `julia --version` | Runs on this host silently re-resolve or fail. Pin with juliaup (`juliaup add 1.11.5`) or a Docker image `julia:1.11.5` on Modal/Kaggle. This is a reproducibility blocker for every protocol below. |
| 0.6 | Octave has only `signal 1.4.1` and `control 3.4.0`, with no `optim`, no `dpss`/`pmtm` and no LTFAT. A local `lpc.m` shim exists. | `pkg list` | The Octave cross-check must use core `\`, `fminsearch` and `ode45`. Multitaper needs a small DPSS routine (the tridiagonal eigenproblem, about 15 lines). `pkg install -forge ltfat` adds reassignment and multitaper-friendly Gabor tools for free. |

---

## Part A — Signal processing, phenomenon by phenomenon

Verdict key: **DO** = build now, high value per effort. **LATER** = worth it after the DO items.
**SKIP** = not worth it for this engine, with the reason.

### A1. Time–frequency front-end (shared by everything)

| Method | What it adds | Verdict | Why |
|---|---|---|---|
| **Multitaper spectra** (Thomson 1982) | Low-variance, low-bias spectra on the **short** spans we measure (qalqalah release 10–40 ms, taps 20–30 ms, fricative noise). A single Hann FFT (what `nasal_db`, `hf_ratio` and `spectral_flatness` use now) has χ²₂ variance of about 100 %. K = 3–5 DPSS tapers cut it about K-fold. | **DO** | Cheap and deterministic. It stabilises every band-energy and flatness sifah metric (hams/jahr, safir, tafashhi, istitalah, itbaq HF attenuation) and the spectral moments of Forrest et al. (1988). Python: `scipy.signal.windows.dpss`. Julia: `DSP.jl` `mt_pgram`. Octave: write DPSS by hand. |
| **Reassigned spectrogram / time-corrected IF** (Auger & Flandrin 1995; Fulop & Fitz 2006) | Moves energy to its centre of gravity, so harmonics and formant ridges become sharp lines. The channelized IF gives **sub-bin harmonic frequencies**. The local group-delay (time) correction localises **bursts to about 1 ms**. | **DO** (for bursts and taps), LATER (for formants) | Qalqalah release and takreer tap onsets are impulse events. The time-reassignment operator ∂φ/∂ω puts them at the true instant, independent of window length. Fulop & Fitz give robust algorithms and show their phonetic use. |
| **Synchrosqueezing (SST, 2nd-order SST)** (Daubechies, Lu & Wu 2011; Oberlin, Meignen & Perrier 2015) | Invertible, sharpened TF with ridge extraction of AM–FM modes. It separates harmonics even under vibrato and portamento. | LATER | Most valuable for **maqam pitch micro-dynamics** (vibrato rate/extent, ornament shape) and for harmonic-based HNR under reverb. It needs roughly 3–10× the FFT compute. `ssqueezepy` (Python, GPU-capable) or a hand-written Julia version on FFTW. |
| **Wavelet scattering transform** (Mallat 2012; Andén & Mallat 2014 "Deep scattering spectrum"; Kymatio, Andreux et al. JMLR 2020) | Translation-stable, deformation-Lipschitz features. 2nd-order coefficients capture amplitude-modulation spectra (roughness, tremolo, **trill rate**, breathy AM). No learning needed, and strong in small-data regimes. | **DO** (as a fingerprint and makhraj feature bank) | We have ~10 reference reciters, which is exactly the small-data regime where scattering beats learned front-ends. Order 2 at J ≈ 2⁸ samples (16 ms) encodes 20–60 Hz AM, which is the takreer tap rate (a single tap is ~25 ms; trill pulses run 25–35 Hz, Ladefoged & Maddieson 1996). It is also a natural input for the nonlinear SINDy libraries below. Kymatio on GPU makes corpus-scale extraction cheap (Kaggle T4). |
| **EMD / EEMD / VMD** (Huang et al. 1998; Dragomiretskiy & Zosso 2014) | Adaptive decomposition into narrow-band modes. | **SKIP** for measurement, LATER only for breath/tremor | EMD suffers mode mixing and has no stable inverse or statistics. VMD needs K and α tuned per span. For the quantities we score, multitaper + reassignment/SST dominate it with theory behind them. One niche: VMD separating slow **vocal tremor / fatigue AM (3–8 Hz)** from syllabic AM in Taraweeh fatigue analysis. |
| **Modulation spectrum / SRMR** (Falk, Zheng & Chan 2010) | Energy per (acoustic band × modulation frequency). SRMR is a non-intrusive reverberation index. | **DO** (Taraweeh) | It gives a per-recording **reliability covariate** (see §A10). The scattering 2nd-order layer is a superset, so both can share code. |

### A2. Source (glottal) analysis — jahr/hams, HNR, fatigue, breathiness

| Method | Verdict | Why / notes |
|---|---|---|
| **IAIF** (Alku 1992) and **QCP** quasi-closed-phase inverse filtering (Airaksinen et al. 2014), **GFM-IAIF** (Perrotin & McLoughlin 2019) | **DO: QCP (or GFM-IAIF) for the glottal parameters, and use the resulting vocal-tract filter as the formant source (§A3)** | Inverse filtering separates the source from the filter. We get **(i)** glottal flow parameters: NAQ (Alku et al. 2002), H1–H2, and the open quotient, which are direct measures of phonation. Pressed vs breathy phonation relates to **jahr** strength and to **fatigue** (Taraweeh night 20 vs night 1). **(ii)** A vocal-tract filter without source ripple, so formants are less biased by F0 harmonics. That bias is a known LPC failure for high-pitched mujawwad (F0 > 250 Hz), where F1 locks onto h2/h3. That failure is one candidate cause of 0.1. QCP was designed for exactly this high-F0 bias. Open implementations: Python `pypevoc`/`covarep` ports. COVAREP (Degottex et al. 2014, MATLAB) runs largely under Octave. |
| **CPP / CPPS** (Hillenbrand, Cleveland & Erickson 1994) | **DO** | This is more robust than autocorrelation HNR for breathy and reverberant voice. Use it as a second jahr/hams metric and as the fatigue driver. Cheap (cepstrum), and Octave can already do it (`rceps`). |
| Autocorrelation HNR (Boersma 1993), current | keep | It is a good cross-check, but it degrades under reverb (the tail adds "harmonicity"). Pair it with CPP. |

### A3. Makharij and tafkheem/tarqeeq/itbaq — formant tracking beyond per-frame LPC

The current approach (median of per-frame Burg/LPC roots over the vowel nucleus) is the weakest
link (0.1). Options, in order:

1. **DO first: harmonise and QC the LPC path (½ day).** Use the same ceiling in Python and Octave: resample to 2 × ceiling and set order = 2 + ceiling/1000 (Praat's convention). Choose the ceiling per reciter, not per F0 bin, by the **Escudero et al. (2009)** method: pick the ceiling that minimises formant variance over a reciter's fathah tokens. Log the per-frame slot-swap rate, defined as |ΔF2| > 400 Hz between frames 5 ms apart. Rerun the Octave validation. **Target: r ≥ 0.85.** Without this, nothing in §B.4.4 is meaningful.
2. **DO: Kalman/ARMA formant *and antiformant* tracker — KARMA (Mehta, Rudoy & Wolfe 2012, JASA 132:1732).** A state-space model of the formant/antiformant frequencies and bandwidths with a random-walk prior, observed through the cepstrum of an ARMA spectrum and tracked by EKF. Three benefits: (i) the continuity prior removes slot swaps; (ii) it gives **per-frame uncertainty** (posterior variance), which feeds the heteroscedastic calibration of roadmap A5 directly; (iii) it **tracks antiformants (zeros)**, which is exactly what nasal consonants and ghunnah need (§A4). Implementation: about 300 lines. Julia `LowLevelParticleFilters.jl` 3.31 provides `ExtendedKalmanFilter`/`UnscentedKalmanFilter` and RTS smoothing. Python `filterpy`.
3. **LATER: mixture-state particle filter** (Zheng & Hasegawa-Johnson 2004) if KARMA still fails on nasal–oral transitions. It needs more compute but handles the discrete "which pole is F2" ambiguity. The same package covers it.
4. **LATER: DNN formant tracker (DeepFormants; Dissen, Goldberger & Keshet 2019, JASA 145:642),** trained on VTR-TIMIT (Deng et al. 2006). It is accurate on English, but domain shift to Classical Arabic chanting at F0 150–400 Hz with long vowels is real. Only worth it if we fine-tune on hand-labelled Arabic tokens. **GPU: one Kaggle T4 session.** Use it as a third opinion in the cross-check, not as the primary tracker.
5. **DO (cheap, directly useful): locus equations** (Sussman, McCaffrey & Matthews 1991): F2_onset = k·F2_vowel + c per consonant. The slope k measures coarticulatory resistance, and the locus is roughly the place of articulation. This makes **makhraj a regression coefficient**, which is interpretable and reference-calibratable. Emphatics (ص ض ط ظ) have characteristically lower loci and flatter slopes (Jongman et al. 2011 on Jordanian Arabic emphasis).
6. **Emphasis/tafkheem metric set** (Jongman et al. 2011; Al-Tamimi & Heselwood 2011): lower F2 **across the whole following vowel**, not only at onset, raised F1, and a larger F3−F2 gap. Measure the full **trajectory** (F2 at 20/50/80 % of the vowel), not only a median. This feeds the dynamics discovery in §B.4.4.

### A4. Ghunnah / nasalisation (ikhfa', idgham bi-ghunnah, iqlab, mushaddad noon/meem)

The engine currently uses a "nasal energy ratio": low-band minus mid-band energy, in dB. It is
confounded by vowel quality, F0 and room bass boost, which is exactly what the Taraweeh `bass_boost_db`
covariate detects.

| Measure | Verdict | Why |
|---|---|---|
| **A1−P0** (Chen 1997, JASA 102:2360): the amplitude of the F1 harmonic minus the amplitude of the ~250 Hz nasal pole P0 (or P1 near 950 Hz for high vowels) | **DO** | This is the standard validated acoustic nasality index for nasalised **vowels**. Styler (2017, JASA 142:2469) found **A1−P0 and F1 bandwidth** among the most robust of 29 features. It is largely normalised for vowel quality because it is a within-spectrum difference. Needs harmonic picking (from the F0 track) plus a peak search in 200–300 Hz. It is the right metric for **ikhfa'**, where the vowel before the hidden noon is nasalised. |
| **F1 bandwidth** (from KARMA/QCP) | **DO** | Nasal coupling widens B1. The tracker already gives it. |
| **Nasal murmur pole–zero** (Fujimura 1962): murmur formant ~250–300 Hz, antiresonance near 750–1250 Hz for /m/ and 1450–2200 Hz for /n/ | **DO via KARMA antiformants** | The **zero frequency distinguishes a labial from an alveolar nasal**. That is exactly the difference between **iqlab** (noon to meem before ب: expect a labial zero) and **ikhfa'** (the noon zero moves towards the place of the next letter). This is a fundamentally better test of iqlab than duration plus an energy ratio. |
| **Pruthi & Espy-Wilson (2007)** automatic vowel-nasalisation parameters (JASA 121:3858) | LATER | A feature set with published detection accuracy. It is useful as the input to a learned nasality classifier once we have labels. |
| **Ghunnah envelope dynamics**: onset rise time, plateau, offset decay of the nasal-band envelope and of A1−P0 over the ghunnah | **DO (feeds §B.4.3)** | The rule is "two harakat of ghunnah". The *shape* (plateau vs decay) is a teachable quality dimension, and a candidate law to discover. |

### A5. Qalqalah (ق ط ب ج د) — burst and release

The current path uses high-band spectral flux plus release counting. Upgrades:

- **DO: burst detection with a statistically defined onset.** (i) Use time-reassigned energy or a **matched Teager–Kaiser energy operator** (Kaiser 1990) for 1 ms burst localisation. (ii) Run **CUSUM/GLRT change-point detection** on a multitaper band-energy sequence (Basseville & Nikiforov 1993), which gives a false-alarm-controlled onset instead of a flux threshold.
- **DO: VOT-style measures** adapted to the qalqalah echo: closure duration, burst-to-voicing-onset lag, and the energy and duration of the **post-release schwa-like echo** (the "bounce"). The AutoVOT structured-prediction method (Sonderegger & Keshet 2012, JASA 132:3965) and Dr.VOT (Shrem, Goldrick & Keshet 2019) are the reference designs. Retraining them needs labelled Arabic tokens, so start with a GLRT plus rules and keep the learned version LATER.
- **Burst spectrum = place cue.** Spectral moments (Forrest et al. 1988) of the first ~10 ms after release separate labial ب (diffuse-falling), coronal د ط (diffuse-rising, with ط lowered by emphasis) and dorsal ق ج (compact, mid-frequency peak; Stevens 1998, ch. 7). This is a makhraj check on the qalqalah letters themselves.
- **Qalqalah kubra vs sughra**: in waqf, the echo is longer and stronger. Model this as an echo-energy ratio conditioned on pause. It is a candidate law for §B.4.2 (release strength vs position).

### A6. Takreer (ر) — tap counting

The current method counts RMS dips with a running-max reference. Upgrades:

- **DO: count taps as periodicity in the 15–40 Hz modulation band.** Use the scattering 2nd-order coefficient or a Hilbert-envelope modulation spectrum over the ر span. A single tap gives no modulation peak. A trill gives a peak near 25–30 Hz (Solé 2002, J. Phonetics 30:655: trill pulses are aerodynamically constrained to about 20–35 Hz). Its harmonic count over the duration estimates the number of taps and is **far more robust to reverberation** than dip counting, because a dip gets filled by the room tail.
- Pair it with reassigned-time impulse localisation (A1) for an exact count on clean audio. The two estimators serve as each other's cross-check.
- F3 lowering is a secondary cue that the segment really is ر (and ر tafkheem lowers F2). Take it from the KARMA track.

### A7. Sifaat with frication (hams, safir, tafashhi, istitalah, rikhawah/shiddah/tawassut)

- **DO: multitaper spectral moments** (centroid, SD, skew, kurtosis; Forrest et al. 1988; Jongman, Wayland & Wong 2000, JASA 108:1252). Also the peak frequency and "M1 in 550 Hz–11 kHz". Safir letters (ص س ز) have a high sibilant peak, typically 4–8 kHz. Tafashhi (ش) is lower (2.5–4.5 kHz) and more diffuse. **Audio must be ≥ 22.05 kHz**; the engine resamples to 16 kHz in places (the Octave path does), which **cuts off the sibilant peak region above 8 kHz**. Keep the original sample rate for these rules.
- **Shiddah/rikhawah/tawassut** (stopped / continuous / intermediate flow): closure-silence duration plus burst presence plus the frication-energy trajectory. A 3-state HMM or HSMM over "closure / burst / frication / sonorant" frames gives a principled segmentation and duration estimates with uncertainty. **LATER.**
- **Hams vs jahr**: voicing fraction during the consonant, measured with pYIN voicing probability (not a hard threshold), CPP, and the low-frequency "voice bar" energy.

### A8. Timing: harakah, madd, ghunnah length, sakt, waqf and breath

- The CTC "core" plus Hilbert-envelope approach already agrees well across stacks (0.1). Two upgrades:
- **DO: explicit-duration HMM (HSMM) segmentation** of each held sound into onset / steady / offset. This gives the **steady-state duration with a posterior** instead of a hard 10 dB threshold, and the posterior SD becomes the measurement-error `s_meas` of roadmap A5. Julia: write it directly (a forward–backward pass over durations is about 150 lines). Python: `pyhsmm` (unmaintained). Implement it in Julia.
- **DO: breath/pause detector upgrade** (Ruinskiy & Lavner 2007) using spectral templates of inhalation noise plus low CPP plus a band-energy pattern. The current `detect_breath` is heuristic. Waqf correctness needs to separate a **breath pause** (waqf/ibtida') from a **sakt** (a brief stop *without* breath, e.g. Hafs's four obligatory saktas). **Sakt = silence without inhalation noise, about 1–2 harakat.** This is a classifier on exactly these features and cannot be done by duration alone.
- **Tempo:** use onset-to-onset syllable intervals (vowel-onset detection from spectral flux plus envelope) instead of "total harakat / total time". It gives a local tempo curve T(τ), the input to §B.4.1. Pair with nPVI/varco rhythm metrics (Grabe & Low 2002) for murattal vs mujawwad style.

### A9. Pitch and maqam

- **Pitch estimator: DO — ensemble pYIN + CREPE (or RMVPE).**
  - **pYIN** (Mauch & Dixon 2014): probabilistic voicing, no GPU, and already the engine's librosa fallback. Make it primary in the Taraweeh path, because it degrades gracefully under reverb.
  - **CREPE** (Kim et al. 2018) or **RMVPE** (Wei et al. 2023, robust to accompaniment and reverb): more accurate on sung, melismatic voice, with ~10-cent resolution. CREPE-full needs a GPU for corpus-scale runs. CREPE-tiny or RMVPE run on CPU in real time.
  - Keep **Praat AC** (current) as the third vote. A **median-of-three vote with octave-error repair** is cheap and removes most octave jumps, the dominant pitch error in chant.
  - **GPU:** corpus extraction (all EveryAyah reciters, about 6236 ayahs × ~10 reciters, several hundred hours) takes about 1–2 A10G-hours with CREPE-full batched. That fits comfortably in one Kaggle weekly quota.
- **Quarter-tone maqam analysis: DO.**
  - Convert F0 to cents relative to a per-recitation **tonic**. Estimate the tonic from the final note of waqf phrases (the qarar), which is the standard Arab and Turkish practice: Bozkurt 2008; Gedik & Bozkurt 2010.
  - Build a **pitch-class distribution at 1-cent or ~5-cent (Holdrian comma, 22.6 cents) resolution**, not a 12-TET chroma. Arab "quarter tones" are **not 50 cents**: the sikah degree in Rast is typically ~340–360 cents above the tonic and varies by region and performer (Marcus 1993, Asian Music 24(2):39). So **learn the scale degrees from the reference reciters** as mixture-model peaks, rather than imposing theory.
  - Classify maqam by distance to learned templates. Use KL or Wasserstein-1 on the circular pitch histogram, since the histogram shift is a circular transport.
  - **Jins (tetrachord) detection** runs over sliding windows, so **modulation (intiqal)** shows up as a change-point in the jins sequence.
  - **Why it matters for this engine:** maqam is not a Tajweed rule, but (i) it is a strong **reciter/style fingerprint** (roadmap C) and (ii) it provides **pitch normalisation for the sifaat measures**, since formant ceilings and HNR both depend on F0.
- **Vibrato/ornament micro-dynamics:** SST ridge plus a sinusoid fit gives rate (Hz) and extent (cents). These are inputs for §B.4.5.

### A10. Reverberation-robust features for Taraweeh live audio

The engine already has WPE (Nakatani et al. 2010; `nara_wpe`), late-reverb suppression, RT60,
DRR and gap-reverb ratio. What to add:

1. **DO: make reverberation a covariate, not only a pre-process.** Per recording, estimate RT60, C50, SRMR and DRR, and include them in the calibration model as measurement-error scale predictors: `s_meas = f(SRMR, align_conf)` (roadmap A5). Dereverberation is imperfect, and knowing *how unreliable* each metric is in a given mosque is worth more than a 1 dB SRMR improvement.
2. **DO: per-feature reverb-robustness audit (a synthetic experiment, no GPU):**
   - Convolve the clean EveryAyah reference with measured large-hall RIRs: the ACE challenge corpus (Eaton et al. 2016), the OpenAIR library and the MIT IR survey. Add Taraweeh-like babble.
   - Sweep RT60 from 0.3 to 3 s. Plot each metric's bias and variance against RT60, with and without WPE.
   - **Drop or down-weight metrics whose bias exceeds the reference band width at typical mosque RT60 (1.5–3 s).** Expected casualties: dip-based tap counting, autocorrelation HNR, the nasal energy ratio (a bass-boost confound), and flux-based qalqalah.
   - Expected survivors: modulation-domain tap counting, CPP, pYIN, A1−P0 (partly), and envelope-onset timing (with onset-based rather than offset-based durations, because **reverb inflates offsets, not onsets**).
3. **DO: onset-anchored timing.** In reverberant audio, measure a duration from vowel onset to the *next* onset, not from onset to energy offset. The tail biases offsets by roughly RT60/6 or more at a 10 dB threshold.
4. **LATER: PNCC-style** power-law nonlinearity and medium-time processing (Kim & Stern 2016) for any learned models run on Taraweeh audio.
5. **LATER (GPU): neural dereverberation** (e.g. a DNN-WPE hybrid). Only worth it if the robustness audit shows WPE is insufficient for the metrics we keep. **Do not** feed neural-dereverberated audio into formant or nasality measures, because generative enhancement can invent spectral detail.

### A11. Summary: build now vs later

| Priority | Item | Effort | Compute |
|---|---|---|---|
| P0 | LPC harmonisation + slot-swap QC + rerun Octave validation (A3.1) | ½ day | CPU |
| P0 | Multitaper everywhere a single-FFT band measure is used (A1) | 1 day | CPU |
| P0 | Reverb robustness audit (A10.2) | 1–2 days | CPU (Modal CPU fan-out optional) |
| P1 | KARMA Kalman formant+antiformant tracker (A3.2); A1−P0, B1, nasal zero for ghunnah/iqlab (A4) | 3–4 days | CPU |
| P1 | QCP/IAIF glottal features + CPP (A2) | 1–2 days | CPU |
| P1 | Modulation-spectrum takreer counter (A6); GLRT burst + spectral moments for qalqalah (A5) | 2 days | CPU |
| P1 | pYIN + CREPE/RMVPE ensemble; cents/tonic/learned-scale maqam analysis (A9) | 2–3 days | 1–2 GPU-h (Kaggle) |
| P2 | Wavelet scattering feature bank (A1) | 1 day | GPU optional (Kaggle T4) |
| P2 | HSMM segmentation with duration posteriors (A8); breath-vs-sakt classifier | 3 days | CPU |
| P3 | DeepFormants fine-tune; SST vibrato; neural dereverb | 1 week+ | Kaggle GPU |
| SKIP | EMD/EEMD as a measurement tool; neural enhancement before spectral metrics | — | — |

---

## Part B — Discovery with Julia

### B.1 What is actually discoverable (the scientific framing)

Tajweed "laws" fall into three mathematical types. Each needs a different tool:

| Type | Examples | Right tool | Wrong tool |
|---|---|---|---|
| **Static constitutive laws** (algebraic maps) | harakah duration vs count, tempo, position; ghunnah length vs rule class; release strength vs pause | Sparse regression on normalised libraries, **symbolic regression with units**, and a **hierarchical Bayesian** final fit | ODE-SINDy: there is no time derivative here. `duration_law` is correctly a static STLSQ. |
| **Fast transient dynamics** (10–300 ms) | formant transitions for tafkheem/itbaq, nasal onset/decay, F0 ornaments, qalqalah echo envelope | 2nd-order ODE / SINDy on **smoothed** trajectories, **weak-form SINDy**, UDEs, Koopman/DMD | Raw finite-difference SINDy on noisy 5 ms formant tracks. Noise amplification destroys it; use the weak form. |
| **Slow drifts** (minutes to hours) | tempo relaxation and fatigue across a surah or a Taraweeh night; pitch declination over a phrase; register shift | the current relaxation ODE, **neural SDEs** or latent-force GPs, SINDy with control (SINDYc) with exogenous phrase and breath inputs | Unconstrained neural ODEs (a black box with no law). |

**Priors from speech and music science** give SINDy something to confirm or refute. That makes
the results *falsifiable*, not just curve fits:

- **Articulatory task dynamics** (Saltzman & Munhall 1989): each articulator gesture is a critically damped point attractor, `ẍ = −ω²(x − x_target) − 2ζω ẋ` with ζ ≈ 1. The hypothesis for formant transitions: `F̈2 = −ω²(F2 − F2*) − 2ζω Ḟ2`, where F2* is lower after emphatics. **Tafkheem = a shifted target F2* plus a slower ω.**
- **Target Approximation (qTA) for pitch** (Prom-on, Xu & Thipakorn 2009, JASA 125:405): F0 approaches a linear pitch target through a 3rd-order critically damped system. **Fujisaki** (Fujisaki & Hirose 1984): phrase and accent commands filtered by 2nd-order critically damped systems. These are the priors for maqam pitch dynamics.
- **Final lengthening / ritardando** (Klatt 1976; Friberg & Sundberg 1999, JASA 105:1469): tempo near a stop follows `v(x) = [1 + (w^q − 1)x]^{1/q}`, the "stopping runner" model with q ≈ 2–3. The hypothesis: **madd before waqf** (madd 'arid lissukun) and pre-pause syllables follow this curve. The engine currently adds `final` as a constant offset, a crude special case.

### B.2 Package set (Julia 1.11.5; versions checked against the local General registry snapshot on 2026-09-23)

Already pinned in `Manifest.toml`: ModelingToolkit 11.45.1, DataDrivenDiffEq 1.16.1, DataDrivenSparse 0.2.3,
OrdinaryDiffEq 7.8.1 (SciMLBase 3.56.0, Symbolics 7.41.0), Optim 2.3.2, ForwardDiff 1.4.6, Distributions 0.25.131.

To add. Compat claims below were checked in the registry `Compat.toml` against the pinned SciMLBase 3 / DataDrivenDiffEq 1.16 where marked with ✓:

| Package | Version (latest registered) | Role | Compat note |
|---|---|---|---|
| SymbolicRegression | 2.4.1 | PySR engine: evolutionary SR, units via DynamicQuantities, template expressions | julia ≥ 1.10 ✓ |
| DataDrivenSR | 0.1.9 | SR as a `DataDrivenDiffEq` solver | requires SymbolicRegression 2 ✓, DataDrivenDiffEq ≥ 1.15 ✓ |
| DataDrivenDMD | 0.1.8 | DMD/EDMD solvers in DataDrivenDiffEq | DataDrivenDiffEq ≥ 1.15 ✓ |
| DynamicQuantities | 1.13.0 | units (ms, Hz, counts) for dimensional SR | — |
| Lux | 1.31.4 | explicit-parameter neural nets for UDEs | julia ≥ 1.10 ✓ |
| ComponentArrays | 0.15.49 | structured parameters (mechanistic + NN) | — |
| SciMLSensitivity | 7.119.11 | adjoints through ODE/SDE solves | 7.119.6+ needs SciMLBase ≥ 3.48 ✓ (we have 3.56) |
| Zygote / Enzyme / Mooncake | 0.7.13 / 0.13.204 / 0.5.60 | reverse-mode AD (Enzyme for adjoint VJPs) | SciMLSensitivity 7.119.9+ requires Enzyme ≥ 0.13.203 ✓ |
| Optimization, OptimizationOptimisers, OptimizationOptimJL | 5.9.1 / 0.3.24 / 0.4.21 | Adam → L-BFGS training | — |
| StochasticDiffEq | 7.2.0 | neural SDEs, fatigue noise | SciMLBase ≥ 3.51 ✓ |
| DiffEqFlux | 4.11.0 | convenience layers (optional; Lux + SciMLSensitivity suffice) | SciMLBase ≥ 3.47 ✓ |
| ModelingToolkitNeuralNets | 2.8.1 | NN blocks inside MTK symbolic models (UDE in MTK) | resolve with MTK 11 |
| Turing | 0.49.0 | hierarchical Bayesian inference (NUTS) | DynamicPPL 0.42.x |
| MCMCChains / PosteriorStats / ParetoSmooth | 7.7.0 / 0.4.11 / 0.7.17 | diagnostics, PSIS-LOO (Vehtari et al. 2017) | — |
| CUDA / DiffEqGPU | 6.4.0 / 3.21.2 | GPU ensemble ODE solves | DiffEqGPU 3.18+ needs SciMLBase ≥ 3.49 ✓ and DiffEqBase 7 ✓ |
| LowLevelParticleFilters | 3.31.1 | EKF/UKF/PF (KARMA formant tracker, state-space tempo) | — |
| DataInterpolations | 10.1.3 | collocation / derivative smoothing for SINDy | — |
| DSP / FFTW / Multitaper | 0.8.6 / 1.10.0 / 1.2.0 | Julia-side DSP (third implementation for cross-checks) | — |
| PythonCall + CondaPkg | 0.9.36 / 0.2.36 | call fdasrsf, pysindy (weak/ensemble SINDy), kymatio if needed | — |
| StableRNGs | 1.0.4 | cross-version reproducible RNG streams | — |

> Compat caveat: registry "latest" ≠ "co-installable". Add these in one `Pkg.add` and let the resolver pick. For MTK-11-coupled packages (ModelingToolkitNeuralNets, DataDrivenSR's Symbolics chain), accept whatever version the resolver selects and **commit the new Manifest**. Do this in a scratch environment first (`Pkg.activate(temp=true)`) so the pinned calibration environment stays intact. Keep two environments: `julia/` (calibration, lean) and `julia/discovery/` (heavy SciML plus GPU).

### B.3 Architecture

```
            Python engine (app/)                Octave (independent DSP)           Julia 1.11 (discovery)
   ┌───────────────────────────────┐   ┌──────────────────────────────┐   ┌───────────────────────────────────┐
   │ benchmark JSONL: per-rule     │   │ qaari_features.m re-measures │   │ Data layer: Arrow/JSONL → tidy    │
   │ metrics + span times + align  │──▶│ the SAME spans (jobs.csv)    │──▶│ tables + measurement-error s_meas │
   │ conf; frame tracks (F0, F1-3, │   │ + fits discovered laws with  │   │ (disagreement Octave vs Python)   │
   │ env, A1-P0) as .npz/Arrow     │   │ \, fminsearch, ode45         │   └──────────────┬────────────────────┘
   └───────────────────────────────┘   └──────────────▲───────────────┘                  │
                                                      │ cross-check                      ▼
                                   ┌──────────────────┴────────────────────────────────────────────────┐
                                   │ 1 HYPOTHESIS registry (TOML): family, prior law, library, units    │
                                   │ 2 SCREEN: normalised STLSQ / E-SINDy / weak-SINDy (DataDrivenSparse)│
                                   │ 3 SEARCH: SymbolicRegression (dimensional, templates) → Pareto front│
                                   │ 4 RESIDUAL: UDE (Lux+SciMLSensitivity) → SINDy on the NN term       │
                                   │ 5 DECIDE: Turing hierarchical fit of top-3 laws, PSIS-LOO + held-out │
                                   │    reciter predictive; DiffEqGPU parametric bootstrap for nulls     │
                                   │ 6 VERIFY: Octave refit + re-measurement; register or reject         │
                                   └────────────────────────────────────────────────────────────────────┘
```

Two design rules:

- **Measurement error is a first-class input.** The Octave-vs-Python disagreement per span (0.1) is an empirical estimate of measurement SD. It goes into the Turing likelihood so laws are not fitted to tracker artefacts.
- **The discovery unit is a hypothesis file, not a notebook.** Each run reads `hypotheses/<id>.toml` (library, units, priors, split, pass criteria) and writes `experiments/discovery/<id>_<ts>.json` with its git SHA, Manifest hash, data hash and RNG seed.

### B.4 Discovery targets, with plausibility and API sketches

Plausibility: ★★★ = a clean law is likely and testable with current data plus EveryAyah;
★★ = plausible, but needs the DSP upgrades first; ★ = exploratory.

#### B.4.1 Harakah / tempo law and tempo dynamics — ★★★

- **H1 (static):** `D = α_r · n · T + β_r + γ · final + ε`, where α ≈ 1 is textbook "counting", β is transition overhead, and final is the pre-waqf lengthening.
- **H1':** a power law `D ∝ n^a T^b` with a = b = 1 as the null. Test this in log space. It is the cleanest falsifiable statement of "counts are proportional".
- **H2 (dynamic):** the existing relaxation ODE `dT/dτ = κ(T∞ − T) + φ`, extended with exogenous inputs: `+ u_breath(τ)·η` (tempo resets after a breath) and `+ u_phrase`.
- **Data:** full surahs for about 10 EveryAyah reciters in murattal **and** mujawwad (Husary and Minshawi have both), plus hadr reciters. Tempo variation is what makes `n·T` identifiable (0.3).

**Screen: normalised STLSQ with DataDrivenDiffEq.** This fixes 0.3(b) and 0.3(c):
```julia
using DataDrivenDiffEq, DataDrivenSparse, ModelingToolkit, StableRNGs
@variables n T fin
X  = permutedims(hcat(n_obs, T_obs ./ 100, fin_obs))        # 3 × N features
Y  = permutedims(D_obs)                                     # 1 × N target (ms)
lib = Basis([1, T, n, n*T, T^2, n*T^2, fin, fin*T, n*fin], [n, T, fin])
prob = DirectDataDrivenProblem(X, Y; name = :harakah_law)
λs  = exp10.(range(-3, 1; length = 60))                     # wide grid; the optimum must be interior
opts = DataDrivenCommonOptions(; normalize = DataNormalization(ZScoreTransform),
                               data_processing = DataProcessing(split = 0.8, batchsize = 256,
                                                                shuffle = true, rng = StableRNG(7)),
                               maxiters = 100)
res = solve(prob, lib, STLSQ(λs), options = opts)
get_basis(res), rss(res), bic(res)
```

**Stability:** use ensemble or bagged SINDy (Fasel et al. 2022), i.e. bootstrap rows and keep terms with inclusion probability > 0.8. That is stability selection (Meinshausen & Bühlmann 2010).

**Search: SymbolicRegression with units and a template.** Constraining the form `D = n·f(T) + g(T, fin)` stops SR from inventing dimensionally absurd laws:
```julia
using SymbolicRegression, MLJ, DynamicQuantities
model = SRRegressor(
    binary_operators = [+, -, *, /], unary_operators = [exp, log, sqrt],
    maxsize = 20, niterations = 300, parsimony = 1e-3,
    elementwise_loss = (p, y) -> abs(p - y) <= 30 ? (p - y)^2 / 2 : 30 * (abs(p - y) - 15),  # Huber, ms
    X_units = ["", "ms", ""], y_units = "ms",          # n, T, final
    dimensional_constraint_penalty = 1e3)
mach = machine(model, (n = n_obs, T = T_obs, fin = fin_obs), D_obs)
fit!(mach); r = report(mach); r.equations[r.best_idx]   # plus the full Pareto front r.equations
```
(Template form in SR ≥ 1.9: `TemplateExpressionSpec(; expressions = (:f, :g), variable_names = ["n","T","fin"]) do ...`. Use it only if the front keeps returning non-separable laws; check the exact macro syntax for SR 2.x in its docs.)

**Decide: Turing hierarchical fit on the log scale.** Partial pooling across reciters directly tests `a = b = 1`:
```julia
using Turing, ParetoSmooth
@model function harakah(logD, logn, logT, fin, rid, R)
    a ~ Normal(1, 0.3);  b ~ Normal(1, 0.3);  γ ~ Normal(0, 0.3)
    μc ~ Normal(0, 1);   σc ~ Exponential(0.3)
    z ~ filldist(Normal(), R)                            # non-centred reciter offsets
    ν ~ Gamma(2, 0.1);   σ ~ Exponential(0.2)
    μ = μc .+ σc .* z[rid] .+ a .* logn .+ b .* logT .+ γ .* fin
    logD ~ arraydist(LocationScale.(μ, σ, TDist.(ν .+ 2)))   # Student-t, heavy right tail
end
chn = sample(harakah(logD, log.(n), log.(T), fin, rid, R), NUTS(0.8), MCMCThreads(), 1000, 4)
psis_loo(harakah(...), chn)        # compare against a = b = 1 fixed, and against an additive-overhead model
```
**A discovery counts as real if:**
- the 95 % credible interval of `a` excludes 1 (counts are *not* proportional), or
- an additive-overhead model beats the pure-proportional model by ΔELPD > 4 with SE < ΔELPD/2,

**and** the effect replicates on held-out reciters (§B.5).

**Dynamic (H2):** Keep the ODE fit and add a **UDE residual** (B.4.6 pattern) with inputs `[T, time_since_breath, phrase_position]`. Distil the NN with SINDy. The prior for waqf approaches is Friberg–Sundberg: test `q` and `w` as SR outputs.

#### B.4.2 Madd elongation law vs position and phrase — ★★★

- **Hypothesis:** realised counts for each madd class (tabi'i 2; muttasil 4–5; munfasil 2/4/5 by reciter choice; lazim 6; 'arid 2/4/6) depend on (i) the class, (ii) the local tempo, (iii) **distance to the next waqf** (final lengthening), (iv) the **reciter's chosen munfasil length**, a latent discrete variable, and (v) the stress/rhythm position.
- **Method:** a **mixture** hierarchical model in Turing, because munfasil and 'arid are legitimately multimodal (roadmap A3 flagged madd bimodality). Use a latent class per reciter × madd type with Dirichlet weights. Pre-waqf position enters through the Friberg–Sundberg curve, whose `q` and `w` get learned.
- **Plausible discovery:** "madd 'arid lissukun length = chosen count × harakah × [1 + ρ·(pre-waqf ritardando)]", with ρ estimated per reciter. This is directly teachable ("you are lengthening early").
- **Requires:** beat-normalised counts (roadmap B1 quick win). No new DSP.

#### B.4.3 Ghunnah decay dynamics — ★★ (needs A4 first)

- **Observable:** the A1−P0 trajectory and the nasal-band envelope across the ghunnah. Also the **nasal zero trajectory** from KARMA, which carries the place of the following consonant in ikhfa'.
- **Hypothesis:** velum/nasal coupling follows a critically damped approach to a target, `c̈ = −ω²(c − c*) − 2ζω ċ`, with onset time t₀ and offset tail. Rule-class differences (idgham vs ikhfa' vs iqlab vs mushaddad) appear as different `c*`, ω, or *zero-frequency targets*.
- **Method:** trajectories are short (≈ 150–600 ms, 30–120 frames) and noisy, so use **weak-form SINDy** (Messenger & Bortz 2021), which integrates against test functions and avoids differentiating noise. Available in `pysindy` (`WeakPDELibrary`, callable via PythonCall), or implement the weak-form Galerkin matrix directly in Julia (~60 lines). Pool across tokens with a per-token target (a hierarchical ODE) in Turing or Optim.
- **Plausible outcome:** ζ ≈ 1 (task dynamics confirmed) and a class-dependent ω. A *failure* of the 2nd-order form is itself informative, because it would mean the nasal envelope is controlled by airflow rather than velum dynamics.

#### B.4.4 Formant transition dynamics for tafkheem/itbaq/makharij — ★★ (blocked by 0.1)

- **Observable:** F1, F2 and F3 trajectories (KARMA-smoothed, with per-frame variance) from consonant release to 80 % of the following vowel, plus locus equations (A3.5).
- **Hypothesis:** the task-dynamic 2nd-order law, where **emphasis changes the target** (F2* lower, F1* higher) and possibly the stiffness. **Itbaq spreads**: the heavy target persists across the syllable, which is a coupled 2-state system in which the consonant's pharyngeal gesture keeps acting. The spread duration is a discoverable parameter.
- **Methods:**
  - **SINDy-PI / implicit SINDy** (Kaheman et al. 2020) if rational terms appear.
  - **Hankel-DMD** (DataDrivenDMD), which gives a data-driven ω and ζ spectrum per consonant class to compare with SINDy's fitted parameters.
  - ModelingToolkit for the symbolic model:
```julia
using ModelingToolkit, OrdinaryDiffEq
using ModelingToolkit: t_nounits as t, D_nounits as Dt
@parameters ω ζ F2star F2₀ v₀
@variables F2(t) V(t)
@named tr = System([Dt(F2) ~ V, Dt(V) ~ -ω^2*(F2 - F2star) - 2ζ*ω*V], t)
sys = mtkcompile(tr)       # MTK ≥ 10: `mtkcompile` (was `structural_simplify`)
prob = ODEProblem(sys, [F2 => F2₀, V => v₀, ω => 40.0, ζ => 1.0, F2star => 1200.0], (0.0, 0.15))
```
    Fit per token (Optim + ForwardDiff, exactly like `Discovery.jl`), then run a Turing hierarchical model on `(ω, ζ, F2*)` by letter class.
- **Plausible outcome:** F2* separates heavy and light letters better than a median F2. That is a better tafkheem metric, and it is robust to tempo because it is a *target*, not a sample.

#### B.4.5 Pitch dynamics per maqam — ★ to ★★

- **Observable:** cents relative to the tonic (A9), voiced segments, and vibrato-removed contours (SST).
- **Hypotheses:**
  1. **qTA/Fujisaki**: each syllable is a linear target approached by a critically damped 3rd-order system.
  2. **Maqam = a set of attractor pitches** (the learned scale degrees), with transitions governed by `ṗ = −∇V_maqam(p) + noise`, where V is a potential with wells at the scale degrees. This is a **neural or polynomial SDE**, and its stationary density equals the pitch histogram, a consistency check.
- **Methods:** fit the drift and diffusion of a **Langevin SDE** from contour increments (a Kramers–Moyal estimate: the conditional mean and variance of Δp/Δt given p). Then **SINDy on the drift** gives a sparse polynomial or Fourier potential per maqam. Neural SDE (StochasticDiffEq + Lux) is only needed if the drift depends on history (ornament grammar).
- **Plausible outcome:** the well depths and locations are a compact, interpretable maqam fingerprint and a style descriptor (roadmap C). This is **not a correctness rule**, so keep it out of scoring.

#### B.4.6 Fatigue / Taraweeh drift — ★★

- **Hypothesis:** over a night (≈ 1–2 h), the harakah drifts (φ), F0 rises or falls, and CPP falls. A latent "fatigue" state follows `ḟ = load(t) − recovery·f` (a fitness–fatigue style model, Banister et al. 1975 as the prior form) and drives all three observables.
- **Method:** a **UDE**, with the mechanistic latent state plus a Lux NN mapping f to observable shifts, then SINDy on the NN to name the map:
```julia
using Lux, ComponentArrays, SciMLSensitivity, OrdinaryDiffEq, Optimization,
      OptimizationOptimisers, OptimizationOptimJL, Random, Zygote
nn = Chain(Dense(2 => 16, tanh), Dense(16 => 16, tanh), Dense(16 => 3))    # (f, load) → ΔT, ΔF0, ΔCPP
θnn, st = Lux.setup(Xoshiro(1), nn)
θ0 = ComponentArray(mech = (r = 0.01,), nn = θnn)
function rhs!(du, u, θ, t)
    du[1] = load(t) - θ.mech.r * u[1]                  # latent fatigue f
end
obs(u, θ, t) = first(nn([u[1], load(t)], θ.nn, st))    # predicted observable shifts
prob = ODEProblem(rhs!, [0.0], (0.0, tend), θ0)
function loss(θ, _)
    sol = solve(prob, Tsit5(); p = θ, saveat = tobs,
                sensealg = InterpolatingAdjoint(autojacvec = ZygoteVJP()))
    sum(abs2, reduce(hcat, [obs(sol.u[i], θ, tobs[i]) for i in eachindex(tobs)]) .- Yobs)
end
optf = OptimizationFunction(loss, AutoZygote())
r1 = solve(OptimizationProblem(optf, θ0), Adam(1e-2); maxiters = 3000)
r2 = solve(OptimizationProblem(optf, r1.u), LBFGS(); maxiters = 500)
# Distil: evaluate nn on a grid of (f, load) → DirectDataDrivenProblem → STLSQ / SR
```
- **Data:** this needs **multi-hour recordings of the same imam** (Taraweeh). Public data is scarce, so gate this on collecting it.

#### B.4.7 Koopman spectra for rhythm — ★★ (roadmap B2, implementation note)

```julia
using DataDrivenDiffEq, DataDrivenDMD
H = hankel(Z, d)                                     # delay-embed [F0,F1,F2,env] frames, d ≈ 50–300
res = solve(DiscreteDataDrivenProblem(H), DMDSVD()) # or TOTALDMD() for noisy data
λ = eigvals(get_operator(res)); ω = log.(λ) ./ Δt   # Im = rhythm/melody freq, Re = damping
```
Cross-check the dominant real eigenvalue against the tempo-ODE κ (they should agree: 0.2 in the roadmap's argument).

### B.5 Reproducible experiment protocol (per hypothesis)

1. **Register** `hypotheses/<id>.toml` *before* fitting. It holds: the family; the prior law; the candidate library or SR operators; units; the primary metric; the **pass criteria**; and the data split, i.e. the reciter IDs for train, validation and **held-out test**, fixed forever.
2. **Freeze the data:** hash the benchmark JSONL rows and the frame-track files used. Record the Manifest hash and git SHA, plus Octave `version` and `pkg list`.
3. **Measure twice:** Python/Praat path plus the Octave path on the same spans. Per-span disagreement becomes `s_meas`. **Exclude** spans whose disagreement exceeds 3× the median (that is tracker failure, not recitation variance), and report how many were excluded.
4. **Screen** with normalised STLSQ or E-SINDy on the train reciters. The λ optimum must be interior, and term inclusion probability must be > 0.8.
5. **Search** with SymbolicRegression (dimensional, same data). Keep the Pareto front and pick by the "score" criterion (the largest drop in log-loss per unit of complexity; Cranmer 2023).
6. **Decide** with the Turing hierarchical fit of the top ≤ 3 candidate laws plus the textbook null. Use PSIS-LOO on train, all Pareto-k < 0.7. Run posterior-predictive checks by reciter and by rule class.
7. **Null calibration (GPU-optional):** parametric bootstrap under the null law, e.g. simulate 10⁴–10⁵ datasets from the fitted *textbook* model with the real design matrix and noise, and re-run steps 4–6 with a cheap fast path (STLSQ plus a Laplace or MAP-level comparison). This gives the **false-discovery rate of the whole pipeline**. For ODE hypotheses, DiffEqGPU's `EnsembleGPUKernel` solves the 10⁵ simulated trajectories:
```julia
using DiffEqGPU, CUDA, StaticArrays, OrdinaryDiffEq
f(u, p, t) = SVector(p[1] * (p[2] - u[1]) + p[3])
prob  = ODEProblem{false}(f, SVector(320f0), (0f0, 900f0), SVector(0.004f0, 300f0, 0f0))
eprob = EnsembleProblem(prob; prob_func = (pr, i, _) -> remake(pr; u0 = SVector(U0[i]), p = SVector{3}(Θ[:, i])),
                        safetycopy = false)
sol = solve(eprob, GPUTsit5(), EnsembleGPUKernel(CUDA.CUDABackend());
            trajectories = 100_000, saveat = τgrid)
```
8. **Held-out reciter test:** refit only the hierarchy-level parameters on held-out reciters (population-level coefficients frozen). Report the out-of-sample ELPD and the **coverage of 90 % predictive intervals**, which must fall in [0.85, 0.95]. The law passes if it beats the textbook null on held-out reciters, not only on train.
9. **Octave cross-check:**
   - (a) **Re-measurement:** refit the discovered law on Octave-measured targets. Coefficients must fall inside the Julia 95 % intervals.
   - (b) **Numerical re-implementation:** Octave refits the law independently. Use `\` for linear laws, `fminsearch` on a Huber loss for nonlinear ones, and `ode45` (Dormand–Prince, with a different error control than Tsit5) for ODE laws. Agreement: relative coefficient difference < 2 % on identical data, and trajectory RMSE < 1 % of range.
   - (c) This catches both *measurement* artefacts (the current F1/F2 disagreement is exactly this class) and *solver/optimiser* artefacts. Octave shares no code with the Julia stack: its LAPACK is reference/OpenBLAS vs Julia's OpenBLAS/MKL builds, and its ODE and optimisers are separate code.
10. **Register or reject:** write `experiments/discovery/<id>_<ts>.json` with the verdict. Rejections are kept, because a known-false law is also knowledge. A registered law may **change the engine** only through the calibration layer (for example, replacing `D/T` counts with `(D − β̂)/(α̂T)`), gated by roadmap A4 conformal LOO coverage.

Octave cross-check sketch for a linear law (no packages needed):
```octave
d = csvread('octave_measured.csv', 1, 0);   % n, T_100ms, fin, D_ms
X = [ones(rows(d),1), d(:,2), d(:,1), d(:,1).*d(:,2), d(:,3)];
xi = X \ d(:,4);                              % compare with Julia E-SINDy selected terms / Turing medians
huber = @(r) sum(min(abs(r),30).^2/2 + 30*max(abs(r)-30,0));
xi_h = fminsearch(@(b) huber(d(:,4) - X*b), xi);
```

### B.6 Compute plan (free tier)

- **Most of Part B is CPU-bound, and GPUs do not help it.** STLSQ is small linear algebra. SymbolicRegression is multi-threaded CPU; it gets faster with **more cores**, not a GPU. Modal CPU containers (e.g. 16–32 vCPU) or a local 8-core box work well. Per-token ODE fits are tiny, and CPU threads beat GPU launch overhead. Turing NUTS is CPU (4 chains = 4 threads). Small-NN UDEs are faster on CPU than on GPU (a known SciML result for small networks).
- **GPU is worth it for:**
  1. CREPE/RMVPE pitch, scattering and DNN-formant extraction over the corpus: **1–3 GPU-hours total**, on one Kaggle T4 session.
  2. DiffEqGPU parametric-bootstrap nulls (§B.5.7): 10⁵ trajectories per hypothesis take **minutes on an A10G**.
  3. Optional neural SDE for B.4.5 if the drift is history-dependent: a few GPU-hours.
- **Budget:** about **5–10 GPU-hours total for the whole programme**, well inside Kaggle's ~30 GPU-h/week free quota. Modal is used for **CPU fan-out**: SR runs over the hypothesis grid, and re-measuring all EveryAyah spans with Octave in parallel containers (Octave + signal in a Debian image, ≤ 100 containers). Modal GPU is reserved for DiffEqGPU nulls, where CUDA.jl setup is easier than in a Kaggle notebook. **Julia on Kaggle** is possible (install juliaup in the notebook and point CUDA.jl at the local toolkit), but the first `Pkg.instantiate` plus precompile of the SciML/CUDA stack costs about 20–40 minutes per session. Prefer a **pre-built Modal image with a PackageCompiler sysimage** for anything run repeatedly.
- **Fix 0.5 before any cloud run:** image `julia:1.11.5`, `JULIA_PROJECT=/work/julia/discovery`, and `Pkg.instantiate()` from the committed Manifest.

### B.7 Prioritised build plan (Part B, after the P0 DSP items)

| Step | Deliverable | Effort | Compute | Depends on |
|---|---|---|---|---|
| 1 | Pin Julia 1.11.5 (juliaup/Docker); create a `discovery/` environment with the B.2 packages; commit the Manifest | ½ day | CPU | — |
| 2 | Fix `duration_law`: z-score the columns, widen the λ grid, E-SINDy bootstrap; data = full surahs, ≥ 6 reciters, murattal + mujawwad | 1 day | CPU | EveryAyah pulls |
| 3 | Hypothesis-file runner + provenance JSON + Octave refit script (B.5 steps 1–3, 9) | 1–2 days | CPU (Modal CPU fan-out for Octave re-measurement) | 1 |
| 4 | Turing hierarchical harakah/madd law, PSIS-LOO, held-out reciters (B.4.1, B.4.2) | 2–3 days | CPU | 2, 3 |
| 5 | SymbolicRegression with units on harakah/madd; compare Pareto front to Turing winners | 1 day | CPU (Modal 32-vCPU, ~1–2 h) | 2 |
| 6 | DiffEqGPU null-calibration harness for the tempo ODE and the pipeline FDR | 1–2 days | Modal A10G, < 1 GPU-h | 1, 4 |
| 7 | Tempo ODE + UDE residual with breath/phrase inputs, SINDy distillation (B.4.1 H2) | 2–3 days | CPU | full surahs |
| 8 | Ghunnah dynamics (weak-SINDy, hierarchical ODE) | 3 days | CPU | A4 (A1−P0, KARMA) |
| 9 | Formant-transition task dynamics + locus equations (B.4.4) | 3–4 days | CPU | A3 fixed (Octave r ≥ 0.85) |
| 10 | Maqam Langevin potential per maqam (B.4.5) | 3 days | 1–2 GPU-h for CREPE extraction | A9 |
| 11 | Fatigue UDE (B.4.6) | 1 week | CPU/GPU optional | multi-hour Taraweeh data |

**Most plausible real discoveries in the next month:**
1. The **harakah law has a non-zero transition overhead and a sub-proportional count exponent**. The current data already hints at it (implied counts scale 0.81, negative intercept), but it is confounded until tempo varies.
2. **Pre-waqf madd lengthening follows a ritardando curve**, not a constant offset.
3. **Tempo relaxation with breath resets** in full-surah murattal.

Ghunnah and tafkheem dynamics become plausible only once the nasality and formant measurements pass the Octave cross-check.

---

## References (primary)

**Time–frequency:** Thomson, Proc. IEEE 70(9):1055 (1982) · Auger & Flandrin, IEEE TSP 43(5):1068 (1995) ·
Fulop & Fitz, JASA 119(1):360 (2006) · Daubechies, Lu & Wu, ACHA 30(2):243 (2011) · Oberlin, Meignen & Perrier,
IEEE TSP 63(5):1335 (2015) · Mallat, CPAM 65(10):1331 (2012) · Andén & Mallat, IEEE TSP 62(16):4114 (2014) ·
Andreux et al., "Kymatio", JMLR 21 (2020) · Huang et al., Proc. R. Soc. A 454:903 (1998) · Dragomiretskiy & Zosso,
IEEE TSP 62(3):531 (2014) · Kaiser, ICASSP 1990 (Teager energy) · Basseville & Nikiforov, *Detection of Abrupt Changes* (1993).

**Source/filter, formants, nasality:** Alku, Speech Commun. 11:109 (1992) · Alku, Bäckström & Vilkman, JASA 112(2):701 (2002) (NAQ) ·
Airaksinen et al., IEEE/ACM TASLP 22(3):596 (2014) (QCP) · Perrotin & McLoughlin, Interspeech 2019 (GFM-IAIF) ·
Degottex et al., ICASSP 2014 (COVAREP) · Hillenbrand, Cleveland & Erickson, JSHR 37:769 (1994) (CPP) ·
Boersma, Proc. IFA 17:97 (1993) · Mehta, Rudoy & Wolfe, JASA 132(3):1732 (2012) (KARMA) ·
Zheng & Hasegawa-Johnson, ICASSP 2004 (particle-filter formants) · Dissen, Goldberger & Keshet, JASA 145(2):642 (2019) ·
Deng et al., ICASSP 2006 (VTR database) · Escudero et al., JASA 126(3):1379 (2009) (formant ceiling optimisation) ·
Sussman, McCaffrey & Matthews, JASA 90(3):1309 (1991) (locus equations) · Chen, JASA 102(4):2360 (1997) (A1−P0) ·
Styler, JASA 142(5):2469 (2017) · Pruthi & Espy-Wilson, JASA 121(6):3858 (2007) · Fujimura, JASA 34(12):1865 (1962).

**Arabic phonetics:** Jongman, Herd, Al-Masri, Sereno & Combest, J. Phonetics 39(1):85 (2011) · Al-Tamimi & Heselwood,
in *Instrumental Studies in Arabic Phonetics* (Benjamins, 2011) · Solé, J. Phonetics 30(4):655 (2002) (trills) ·
Ladefoged & Maddieson, *The Sounds of the World's Languages* (1996) · Stevens, *Acoustic Phonetics* (MIT, 1998) ·
Forrest, Weismer, Milenkovic & Dougall, JASA 84(1):115 (1988) · Jongman, Wayland & Wong, JASA 108(3):1252 (2000).

**Timing, VOT, breath:** Sonderegger & Keshet, JASA 132(6):3965 (2012) · Shrem, Goldrick & Keshet, Interspeech 2019 (Dr.VOT) ·
Ruinskiy & Lavner, IEEE TASLP 15(3):838 (2007) · Klatt, JASA 59(5):1208 (1976) · Grabe & Low, *Lab. Phon. 7* (2002) ·
Friberg & Sundberg, JASA 105(3):1469 (1999).

**Pitch and maqam:** de Cheveigné & Kawahara, JASA 111(4):1917 (2002) · Mauch & Dixon, ICASSP 2014 (pYIN) ·
Kim, Salamon, Li & Bello, ICASSP 2018 (CREPE) · Wei et al., Interspeech 2023 (RMVPE) · Bozkurt, JNMR 37(1):1 (2008) ·
Gedik & Bozkurt, Signal Process. 90(4):1049 (2010) · Marcus, Asian Music 24(2):39 (1993) · Fujisaki & Hirose,
J. Acoust. Soc. Jpn. (E) 5(4):233 (1984) · Prom-on, Xu & Thipakorn, JASA 125(1):405 (2009).

**Reverberation:** Nakatani et al., IEEE TASLP 18(7):1717 (2010) (WPE) · Drude et al., ITG 2018 (nara_wpe) ·
Falk, Zheng & Chan, IEEE TASLP 18(7):1766 (2010) (SRMR) · Kim & Stern, IEEE/ACM TASLP 24(7):1315 (2016) (PNCC) ·
Eaton et al., IEEE/ACM TASLP 24(10):1681 (2016) (ACE challenge).

**Discovery / SciML:** Brunton, Proctor & Kutz, PNAS 113(15):3932 (2016) (SINDy) · Mangan et al., Proc. R. Soc. A 473:20170009 (2017)
(SINDy + information criteria) · Kaheman, Kutz & Brunton, Proc. R. Soc. A 476:20200279 (2020) (SINDy-PI) ·
Messenger & Bortz, Multiscale Model. Simul. 19(3):1474 (2021) (weak SINDy) · Fasel et al., Proc. R. Soc. A 478:20210904 (2022)
(E-SINDy) · Meinshausen & Bühlmann, JRSS-B 72(4):417 (2010) · Rackauckas et al., arXiv:2001.04385 (UDEs) ·
Cranmer, arXiv:2305.01582 (PySR/SymbolicRegression.jl) · Chen et al., NeurIPS 2018 arXiv:1806.07366 (neural ODE) ·
Li et al., AISTATS 2020 arXiv:2001.01328; Kidger et al., ICML 2021 arXiv:2102.03657 (neural SDE) ·
Schmid, JFM 656:5 (2010) (DMD) · Williams, Kevrekidis & Rowley, J. Nonlinear Sci. 25:1307 (2015) (EDMD) ·
Ma et al., arXiv:2103.05244 (ModelingToolkit) · Ma et al., arXiv:1812.01892 (sensitivity analysis/adjoints) ·
Ge, Xu & Ghahramani, AISTATS 2018 (Turing) · Hoffman & Gelman, JMLR 15:1593 (2014) (NUTS) ·
Vehtari, Gelman & Gabry, Stat. Comput. 27:1413 (2017) (PSIS-LOO) · Utkarsh et al., CMAME 429:117185 (2024), arXiv:2304.06835 (DiffEqGPU) ·
Besard, Foket & De Sutter, IEEE TPDS 30(4):827 (2019) (CUDA.jl) · Saltzman & Munhall, Ecol. Psychol. 1(4):333 (1989) ·
Banister et al., Aust. J. Sports Med. 7:57 (1975) (fitness–fatigue).

*Verification note:* package versions were taken from the local General registry snapshot (`~/.julia/registries/General.tar.gz`), and compat claims from its `Compat.toml`. The Julia API sketches follow the documented APIs of those major versions but were **not executed** (read-only brief, and the host Julia is 1.10.9, see 0.5). Run them in the pinned 1.11.5 environment before relying on exact keyword names (notably `DataDrivenCommonOptions` fields, the SR template macro, and `mtkcompile`).
