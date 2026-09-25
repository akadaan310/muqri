# muqri

A measuring instrument for Quran recitation (Ḥafṣ ʿan ʿĀṣim). Send it a recording and the verses recited;
it returns, for every letter, characteristic (ṣifah) and tajwīd rule, what was realised and how it
compares with 41 master reciters. **The engine measures; the apps decide** what to tell a learner.

## The service (muaalem engine, live)

```bash
.venv/bin/python -m app.webapp        # http://localhost:8088 — /analyze, /detect, /capability,
                                      # /sessions (calibration rounds), /review (expert listening)
```

* **Acoustic model**: multi-level CTC (muaalem) over 40 ms frames. It emits phonemes plus ten
  characteristic heads: ghunnah, hams/jahr, shiddah/rakhāwah, tafkhīm/tarqīq, qalqalah, iṭbāq, ṣafīr,
  tafashshī, istiṭālah and takrīr.
* **Text side**:
  * The `quran_transcript` phonemizer gives the exact expected phonemes, with lengths.
  * The Tajweed parser locates 40 rule types, 1.16 M instances across the Qur'an.
  * `app/rule_bind.py` ties each rule to the sounds that realise it.
* **Per recording**:
  * letter timings (sub-frame);
  * letter identity against its classical confusions;
  * each characteristic's margin;
  * every madd / ghunnah / idghām / ikhfāʾ length in the reciter's own count unit;
  * stops, basmala and wajh.
* **Comparison**: every number is placed against the masters (percentile, robust z) in the
  `measurements/1` report (`app/schemas/measurements-1.schema.json`).
* **Rules covered**:
  * all ten mudūd;
  * nūn sākinah / tanwīn (iẓhār, ikhfāʾ, idghām with and without ghunnah, iqlāb);
  * mīm sākinah (three rules);
  * ghunnah mushaddadah;
  * qalqalah;
  * idghām mithlayn / mutajānisayn / mutaqāribayn;
  * rāʾ tafkhīm / tarqīq;
  * hamzat al-waṣl, sakt and stops;
  * the ten ṣifāt;
  * laḥn jalī (wrong letter, wrong vowel).
* **Reliability**: `GET /capability` serves each capability's reliability live: what ships and what is still on caution or hold.
* **Calibration rounds** (`/sessions`): a certified reciter records each exercise twice, once correct
  and once with dictated mistakes. Over rounds 1–4 the engine went to 41/48 dictated mistakes caught and
  145/155 expectations met on the correct takes (`app/sessions.py`,
  `research_agency_lab/experiments/sessions_results.jsonl`).

The rest of this README documents the earlier **qaari-eval v2** pipeline (wav2vec2 aligner + DSP
validators). It is kept for the CLI, the benchmark runs and its calibration on the full-Qur'an Kaggle run.

# qaari-eval (legacy pipeline)

Quranic Tajweed analysis, scoring calibrated on master reciters, and reciter fingerprinting.

`qaari-eval` checks how a recitation was delivered, not only which words were said:

1. **Rules from the text.** It derives every Tajweed rule the Uthmani text requires under Hafs ʿan ʿĀṣim (Shatibiyyah or Tayyibah).
2. **Letter timings.** It force-aligns the audio at letter level.
3. **Acoustic measurement.** It measures each rule: Madd length, Ghunnah nasality, the Qalqalah release, formant weight for Tafkheem/Tarqeeq, and the Sifaat (Hams/Jahr, Shiddah/Rakhawah, Itbaq, Safir, Tafashhi, Istitaalah, Takreer).
4. **Judgement.** It judges each measurement against how Sheikh Mahmoud Khalil Al-Hussary and his ijaazah peers actually recite, rather than against hand-set limits.
5. **Live recordings.** A Taraweeh adapter handles reverberant, fast recordings.
6. **Fingerprint.** A 232-d fingerprint places the reciter among master and Taraweeh reciters.

```
[ audio ] + [ Uthmani text ]
      │
      ▼
 app/aligner.py            CTC forced alignment (wav2vec2 Quran phonetic model, exact Viterbi)
 app/tajweed_rules/        parser.py (Uthmani → letter units → 38 rule types, waqf/sakt phrases)
                           mudood_engine, noon_sakinah, meem_sakinah, qalqalah_engine,
                           idghaam_classes, raa_lam_rules, sakt_wasl        (validators)
 app/sifaat/               hams_jahr, sukoon_spectrum, itbaq, ghair_mutadhaddah, formants
 app/taraweeh_adapter/     dereverb (RT60, WPE, late-reverb suppression, 120 Hz HPF),
                           pace_normalizer (local tempo), fatigue_detector (breath pauses)
      │
      ▼
 app/scoring.py            validators → raw metrics → app/calibration.py (robust z vs. reference)
 app/fingerprint.py        192 ECAPA timbre ⊕ 32 Tajweed ⊕ 8 environment → FAISS indices
```

## Install

```bash
pip install -r requirements.txt           # core: parser, DSP, FAISS, WPE, CLI, tests
pip install --index-url https://download.pytorch.org/whl/cpu torch torchaudio
pip install -r requirements-ml.txt        # CTC aligner + ECAPA embeddings + HTTP API
```

The first run downloads `TBOGamer22/wav2vec2-quran-phonetics` (~360 MB) and
`speechbrain/spkrec-ecapa-voxceleb`.

The research lab (`research_agency_lab/`) additionally uses:
- Julia ≥ 1.10 (`research_agency_lab/substrate_library/julia/Project.toml`)
- GNU Octave with the `signal` package
- The Kaggle CLI

## Usage

```bash
python main.py analyze --audio user_input.wav --surah 1 --tareeq shatibiyyah \
       --mode taraweeh_adapted --benchmark dosari
python main.py rules --surah 1 --ayah 7          # the rules the text requires (no audio)
python main.py serve --port 8000                  # POST /analyze
```

| Option | Meaning |
|---|---|
| `--surah S [--ayah A [--ayah-end B]]` | A whole surah, an ayah, or a range (`--text "…"` for any Uthmani text) |
| `--tareeq shatibiyyah\|tayyibah` | Madd targets and Munfasil rules of the chosen tareeq |
| `--mode auto\|studio\|taraweeh_adapted` | `auto` switches to the adapter when the room is reverberant |
| `--benchmark NAME` | Compare the recitation with a reciter from the indices (e.g. `dosari`, `hussary`) |
| `--no-sifaat` | Skip the articulation-attribute (Sifaat) analysis |
| `--aligner`, `--alignment-json`, `--include-alignment` | Alignment back-end and letter timings |

The report has these fields:
- `tajweed_perfection_index`: the Ahkaam, weighted by category.
- `sifaat_score`: kept separate from the perfection index.
- One diagnostic per rule, with its raw metrics and, when calibrated, its z-score.
- Pace, acoustic environment, adapter actions, fatigue and pitch style.
- The fingerprint, and a comparison with the requested benchmark reciter.

Statuses:
- `PASS`, `WARNING`, `FAIL`.
- `SKIPPED`: the rule could not be measured reliably, for example because the alignment collapsed.
- `VALID_NECESSARY_PAUSE`: a short Madd before a breath pause in Taraweeh mode.

Neither of the last two counts toward the score.

## Scoring: Al-Hussary as the reference scorer

Textbook limits ("a natural Madd is 2 counts ± 0.25") assume a perfect ruler, but the engine's rulers are biased:
- CTC spans absorb consonant closures.
- The harakah is estimated from syllable spans.
- Every acoustic metric has its own offset.

Under textbook limits Al-Hussary himself scored **71**, and the limits did not separate anyone.

Scoring is instead calibrated on the reference reciters (`app/data/calibration.json`, built by `research_agency_lab/substrate_library/julia/calibrate.jl`).

**Anchor and peers.**
- The anchor is Al-Hussary (`Husary_128kbps`).
- The ijaazah peers are Al-Hussary (Muallim), Al-Minshawi, Al-Hudhaify, Abdul Basit and Alafasy.
- Tablawi and Ayyoub are in the roster but have no Quran-MD data yet.

**Ruler selection.** Each duration rule has two candidate rulers:
- `counts`: span duration / local harakah.
- `core_counts`: the Hilbert-envelope voiced vowel core / harakah.

The calibrator keeps the ruler with the lowest robust coefficient of variation on Al-Hussary.

**Reference band.** For each rule key the band is `[min(m_H, m_C), max(m_H, m_C)]`:
- `m_H` is Al-Hussary's median.
- `m_C` is the peers' consensus, the median of their medians.

So where the peers legitimately differ, the band widens to include them.

**Scale.** The scale is the median over reciters of `σ_r = max(1.4826·MAD_r, (p95_r − p5_r)/3.29)`, floored per unit. The quantile term keeps σ from collapsing on saturated or discrete metrics such as voicing ≈ 1 or tap counts.

**Verdict.** `z` is the distance outside the band divided by the scale, one-sided where only one direction is an error (e.g. Izhaar may be crisp but not held).
- PASS: z ≤ 2
- WARNING: z ≤ 3
- FAIL: beyond 3

**Alignment reliability.** A span is SKIPPED rather than failed when:
- its CTC posterior is below max(0.2, Al-Hussary's 2nd percentile), or
- a unit collapsed below 40 ms (two CTC frames), or
- a duration rule has no voiced core.

This covers the collapsed Munfasil spans.

**Category weights.** The weights are searched (Optim.jl) to separate peers from imams:
- Each weight stays within ×/÷2 of its default.
- Tuned weights are kept only if the separation gains ≥ 1 point; the 95 target is never optimised for.

The first, unbounded search reached the peers' 95 target by inflating the wasl weight ×30, which lifted the imams to 95 as well. Both bounded searches since gained < 1 point, so the **default weights stand**.

The live scorer and the offline re-scoring (`benchmarks/summarize.py`) share one `Calibration.judge()`. The Julia implementation reproduces the Python scores to the decimal.

### Results (full-Qur'an Kaggle runs, 114,746 ayah recordings, scored with calibration v3)

**Data.**
- Source: Quran-MD WAVs (the EveryAyah recordings), the `full` and `fill` Kaggle runs together; the
  rows are committed gzipped in `benchmarks/results/kaggle/{full,fill}/`.
- Imams complete: Sudais, Juhaynee, Shuraym, Qatami and Dosari, all 6,236 ayahs in both modes.
- Al-Hussary 6,031 ayahs; peers 3,651–4,352 each.
- Scored with the installed calibration (`app/data/calibration.json`, v3, built from the studio-all run)
  through `Calibration.judge`: `python benchmarks/summarize.py --runs <rows> --calibration app/data/calibration.json`.
  The bands come from the anchor and peers only; these rows add scoring coverage, not calibration data.

| Reciter | Set | Raw textbook | Calibrated (studio) | Adapted mode |
|---|---|---|---|---|
| Al-Hussary (Muallim) | peer | 75.0 | **96.9** | 95.1 |
| Al-Hussary | anchor | 71.2 | **95.1** | 94.1 |
| Abdul Basit (Murattal) | peer | 68.9 | 92.5 | 92.8 |
| Al-Hudhaify | peer | 69.0 | 90.8 | 91.2 |
| Alafasy | peer | 68.1 | 89.7 | 89.7 |
| Al-Minshawi (Murattal) | peer | 61.3 | 87.3 | 90.0 |
| Saud Al-Shuraim | imam | 69.1 | 90.4 | 90.7 |
| Abdul Rahman Al-Sudais | imam | 62.6 | 89.0 | 89.5 |
| Yasser Al-Dosari | imam | 68.0 | 88.3 | 88.0 |
| Nasser Al-Qatami | imam | 65.0 | 85.5 | 84.9 |
| Abdullah Al-Juhany | imam | 38.4 | *67.2 — unscored* | 66.9 |

Al-Hussary's 95.1 and the Muallim recording's 96.9 match the calibration gate's own figures (95.10 and
97.05) on the independent studio-all rows.

**Al-Juhany's score is an alignment failure, not a verdict.** His files hold the right ayahs (duration
tracks word count at r = 0.95, as for every reciter), but the CTC alignment posterior is 0.000 even at
the 75th percentile (0.3–0.99 for everyone else), his harakah estimate collapses to an impossible 80 ms
(160–320 ms for the rest) and the signal-to-noise ratio is 8 dB, the worst of the set. Treat his rows as
unmeasured until the audio or the aligner is dealt with.

**Findings.**
- **The vowel-core ruler recovers "2 harakat".**
  - On 26,956 natural Madds (calibration v3), the Hilbert-envelope vowel core measures Al-Hussary's median at **1.98 counts** (`count_scale` 1.009).
  - The raw span ruler gives 2.2–2.5 because CTC spans take in the next letter's closure; this was the "harakah inflation".
- **Held-out peers barely move** (v3: in-sample − LOO ≤ 0.73 for all seven peers), so the bands are not overfit to the reciters that built them.
- **Imams vs. peers.**
  - The measured imams (Al-Juhany excluded) average 88.3 against the peers' 91.4. The separation is
    real but modest, and Al-Minshawi sits below three of the imams.
  - The per-rule gaps (`summary.json → rule_gaps_vs_reference`, signed z from the reference band) are
    more informative than the single index:
    - **Short Munfasil is the Haramain signature:** Qatami, Al-Shuraim and Al-Sudais all hold it at
      median z ≈ −2.4, with 22–29 % inside the reference band.
    - **Al-Hudhaify**, a peer, goes the other way (z ≈ +2.6): his Munfasil is longer.
    - **Qatami** holds Madd ʿIwad long (z ≈ +4.8; 9 % pass).
    - **Al-Shuraim** cuts the Ṣila Kubrā short (z ≈ −3.7).
    - **Dosari**'s Munfasil mostly passes (78 %); his weakest rule is a long Lāzim Kalimī (z ≈ +2.1).

**Acceptance criteria.** All three fail; the numbers are reported as measured, not tuned.
- Al-Hussary ≥ 98: **95.1**.
- Dosari adapted ≥ 95: **88.0**.
- Timing false positives cut ≥ 90 % by the Taraweeh adapter: **−3.3 %** (25,302 → 26,125 timing FAILs
  over the imams).
  - The adapter was built for reverberant live audio. These imam recordings are studio-quality EveryAyah ayahs, so there is little reverberation for it to remove.
  - It does not help here, and a real Taraweeh test needs live recordings.

Run `QAARI_ACCEPTANCE=1 pytest tests/test_benchmarks.py` to reproduce this check.

### Model discovery (`research_agency_lab/experiments/discovery_full.json`)

- **Duration law.** Sparse regression (STLSQ with BIC) of held-sound duration on count `n` and tempo `T`:
  - The count term dominates for every reciter, at about 335–560 ms per count.
  - For several reciters (Alafasy, Dosari) the selected law has **no tempo term**.
  - The per-ayah harakah estimate is therefore a weak predictor of Madd length. R² is only 0.3–0.6, so treat this as indicative.
- **Tempo dynamics.** Three models of the harakah across a surah compete by AIC:
  - constant
  - linear drift
  - a relaxation ODE, `dT/dτ = (T∞ − T)/τc`, solved with OrdinaryDiffEq and fitted by L-BFGS with ForwardDiff gradients through the solver
  - **Result:** drift is preferred in about 40–60 % of surahs, but the median drift is ≤ 0.3 ms/min for the masters and 0.6 ms/min for Qatami. Tempo is essentially stable, with no fatigue signal yet.
  - **Caveats:** missing ayahs, and separately recorded ayah files.

## Research lab (`research_agency_lab/`)

| Path | What it does |
|---|---|
| `substrate_library/octave/qaari_features.m` | GNU Octave + `signal`. Per diagnostic span it measures: LPC formants (order 2 + fs/1000), autocorrelation HNR and voicing (r > 0.45), cepstral F0, Hilbert vowel core, nasal band contrast, spectral-flux burst, high-frequency share. The Python `EvalContext.vowel_core_ms` is its production port and agrees within 5–15 ms. |
| `compute_bridge/octave_bridge.py` | Runs the Octave engine over benchmark rows (`diag["octave"]`) |
| `substrate_library/julia/` | `QaariLab`: `calibrate.jl` (bands, LOO, weights) and `discover.jl` (duration law, tempo ODE) |
| `compute_bridge/kaggle_bridge.py` | `push-code`, `push-deps`, `launch`, `status`, `collect`. Details below the table. |
| `experiments/` | Calibration and discovery outputs |

`kaggle_bridge.py` notes:
- Workers run sharded `run_benchmark.py` over the Quran-MD WAVs attached as inputs.
- A private deps dataset (wheels plus model snapshots) lets them run with Kaggle internet off.
- `--skip-done` resumes a run.
- `--only K` relaunches a single kernel.

The benchmark rows are raw: validators run with `calibration=None`, and each diagnostic keeps its metrics. So the bands can be re-estimated and everything re-scored offline without touching audio.

```bash
python research_agency_lab/compute_bridge/kaggle_bridge.py launch --tag full --reciters Husary_128kbps … --verses all
python research_agency_lab/compute_bridge/kaggle_bridge.py collect --tag full --kernels 5
julia --project=research_agency_lab/substrate_library/julia research_agency_lab/substrate_library/julia/calibrate.jl \
      app/data/calibration.json benchmarks/results/kaggle/full/*runs_full_*.jsonl
julia --project=research_agency_lab/substrate_library/julia research_agency_lab/substrate_library/julia/discover.jl \
      research_agency_lab/experiments/discovery_full.json benchmarks/results/kaggle/full/*runs_full_*.jsonl
python benchmarks/summarize.py --runs benchmarks/results/kaggle/full/*runs_full_*.jsonl
```

The collected rows are committed gzipped (`*.jsonl.gz`); `gunzip -k` them first.

## Reciter indices

`index/masterclass_reciters.faiss` and `index/taraweeh_reciters.faiss` (each with `_meta.json`) hold one 232-d fingerprint per reciter, merged over that reciter's studio-mode ayahs:
- 192-d ECAPA timbre
- 32 Tajweed fields in the spec order
- 8 environment fields

Search blends timbre cosine similarity with a Tajweed style similarity, `exp(−mean z²/2)`. `--benchmark NAME` resolves spelling variants (Dossary/Dosari/Al-Dussary, Hussary/Husary, …).

## Tests

```bash
pytest                                             # unit tests (synthetic audio has exact ground truth)
QAARI_ACCEPTANCE=1 pytest tests/test_benchmarks.py # the spec's acceptance criteria on benchmarks/results/summary.json
```

## Limitations

- **The CTC model** was trained on word-level audio. Boundaries can drift by a frame (20 ms), and some spans collapse; collapsed spans are SKIPPED, not failed.
- **The anchor's score is in-sample.** Only the peers get held-out (LOO) validation.
- **The Taraweeh adapter** is untested on genuine live recordings. EveryAyah imam recordings are studio-clean, and the blind RT60 estimate is biased low above 1 s.
- **Gaps in the benchmark:**
  - Al-Juhany's audio defeats the aligner (see Results); his rows are unmeasured.
  - Tablawi, Ayyoub, Budair, Matroud and Al-Muaiqly are not in Quran-MD and need an EveryAyah run.
- **Jawaz al-wajhayn** reports the realised variant and always passes. Madd ʿArid and Leen stay on textbook limits, because 2/4/6 counts is the reciter's free choice.
- **Educational aid.** This does not replace a qualified teacher (*mujawwid*).
