# Carry-over: qaari-eval — calibrate on Al-Hussary, validate on peers, score the imams

## 1. Goal
Replace qaari-eval's hand-set Tajweed limits with limits learned from real master reciters.

1. **Calibrate.** Measure Sheikh Al-Hussary across the whole Qur'an (6236 ayahs) and derive each rule's reference band from his measurements. Pass means a robust z-score ≤ 2, warning ≤ 3, fail above that.
2. **Validate.** Check those bands on his ijaazah peers by leaving each one out in turn, and widen them where the peers legitimately differ.
3. **Score.** Score the Taraweeh imams against these reference examples.

Done means all of the following:
- `app/data/calibration.json` is committed.
- `benchmarks/results/summary.json` shows raw vs calibrated scores and each rule's gap from the reference.
- The acceptance numbers are reported honestly: Husary ≥ 98, Dosari adapted ≥ 95, timing false-positive reduction ≥ 90 %.
- The README is updated.
- Everything is pushed to `claude/qaari-eval-engine-btwkjt`.

## 2. Context
- **The problem.** With the textbook limits, Husary, the reference reciter, scores only about 72. The rulers are biased:
  - The harakah unit comes from CTC syllable spans (260–520 ms), which include consonant closures.
  - Madd and ghunnah spans take in neighbouring closures.
  - Some munfasil spans collapse (about 0.4 counts) because of misalignment.
- **The user's direction.** Husary is the *set example scorer*. The ijaazah peers produce the "winning examples" that everyone else is compared with. Heavy mathematics goes to Julia (ODE, optimisation, sparse regression) and to GNU Octave (signal processing), with Kaggle as the compute.

## 3. Current state (verified)
- **Branch and commits:** branch `claude/qaari-eval-engine-btwkjt`, head `c0cbb2e` plus this commit. ruff, mypy and pytest were green at `9dd66c3`. The later commits only add data files, notes and the Julia lockfile.
- **Benchmark rows carry raw metrics.** Every diagnostic in a row from `benchmarks/run_benchmark.py` stores its raw metrics:
  - `align_conf`, `align_min_ms`, `haraka_ms`
  - `counts`, `core_ms`, `core_counts` (Hilbert-envelope vowel core)
  - the span `location` and the rule `key`
  - Rows also store `duration_s`.
- **`app/calibration.py`** is done and tested (`tests/test_calibration.py`):
  - Robust bands and the verdict function.
  - Unreliable alignment becomes SKIPPED instead of FAIL.
  - `judge()` is shared by the live scorer and the offline `benchmarks/summarize.py --calibration`.
- **Octave engine** (`research_agency_lab/substrate_library/octave/qaari_features.m`) plus its bridge (`research_agency_lab/compute_bridge/octave_bridge.py`):
  - Measures LPC formants (order 2 + fs/1000), autocorrelation HNR and voicing, cepstral F0, Hilbert vowel core, nasal contrast and burst energy.
  - Its vowel core matches the Python port within 5–15 ms.
- **Julia `QaariLab`** (`research_agency_lab/substrate_library/julia/`), both scripts verified on a 59-ayah Husary sample:
  - `calibrate.jl` ran end to end: count_scale 0.914 and 13 banded rule keys. Husary's in-sample score was 77.9, low because most keys have too few samples in 59 ayahs.
  - `discover.jl` ran end to end. Its preliminary duration law was D_core ≈ −552 + 150·T + 394·n ms (R² 0.68). This suggests a nearly tempo-independent count unit, still to be confirmed on the full Qur'an. Outputs are in `research_agency_lab/experiments/*_59.json`.
- **Kaggle bridge** (`research_agency_lab/compute_bridge/kaggle_bridge.py` plus `kaggle_worker.py`):
  - The code is uploaded as the private dataset `razanashrafalnajjar/qaari-eval-code`.
  - Kernels read Quran-MD WAVs locally (`husseinzahaki/quran-md-ayahs-wav-part1..3`: all 6236 ayahs × 30 reciters).
  - Mapping in `benchmarks/roster.py:QURAN_MD`.
- **Kaggle smoke tests.** Both finished; logs are in `benchmarks/results/kaggle/smoke*/`.
  - **Root cause:** the kernels have **no internet**. `enable_internet` is ignored until the Kaggle account is phone-verified, so pip and apt both failed.
  - **Glob fix confirmed:** after the fixed-depth change, code was found in 0.0 min instead of 5.2 min.
  - **Worker bugs, now fixed:**
    - The Octave reserve (1.5 h) exceeded the 1 h smoke session, so the processes were stopped at once. The reserve is now at most ¼ of the session.
    - The worker waited 190 min on apt retries. apt now times out after 15 min and is skipped entirely when offline.
- **Offline bundle.** Uploaded as the private dataset `razanashrafalnajjar/qaari-eval-deps`, and the worker now installs from it whether or not the kernel has internet:
  - `wheels/`: parselmouth, faiss-cpu, soxr, speechbrain, nara_wpe and their deps, built for Kaggle's Python 3.12.
  - `hf_hub/`: the wav2vec2 and ECAPA model snapshots.
  - The rebuild recipe is in the `kaggle_bridge.py push-deps` docstring.
- **Kaggle pipeline verified** (smoke5, rows in `benchmarks/results/kaggle/smoke5/`):
  - Offline install works, models load, and 28 rows (Husary and Dosari, Al-Fatiha, both modes) were written in about 2 minutes.
  - Rows match the local run: same harakah, scores within ±2–5. The Quran-MD source is 64 kbps WAV, the local run used 128 kbps MP3.
  - Studio mode takes about 2.7 s per ayah per process.
- **Full run launched** (tag `full`): 11 Quran-MD reciters × 6236 ayahs × {studio, taraweeh_adapted}, about 137k rows.
  - 5 kernels × 4 processes, 20 shards, 11.5 h cap; roughly 5 h expected.
  - Every process takes Husary's ayahs first, so his rows finish first even if the session runs out.
  - Kernels are `qaari-full-0..3`. `qaari-full-4` was auto-launched once smoke4 freed the 5th session slot; if it is missing, run `launch ... --only 4` with the same arguments.
  - Collect with `kaggle_bridge.py collect --tag full --kernels 5`.
- **Octave on Kaggle** needs apt, and so needs internet. Until the account is phone-verified, run `octave_bridge.py` on the VM over the collected rows. It reads audio from the EveryAyah cache, or from `--local-root` if you download Quran-MD parts with `kaggle datasets download`.

## 4. Files
- **VM repo:** `/home/azureuser/github-director/repos/muqri`. Run `git fetch && git checkout claude/qaari-eval-engine-btwkjt && git pull`.
- **Earlier results:**
  - `HANDOFF.md`: previous handoff, textbook-limit findings.
  - `benchmarks/results/local/husary_strategic_raw.jsonl`: 59 Husary rows with raw metrics.
- **Calibration:**
  - Spec: `research_agency_lab/substrate_library/julia/data/metric_spec.json`, which lists the judged metric(s) and failure direction for each rule.
  - Taxonomy: `research_agency_lab/substrate_library/julia/data/taxonomy.json`, exported from Python. Re-export it if the categories or roster change (see the commit `9dd66c3` message for the snippet).
  - Consumers: `app/calibration.py`, `app/scoring.py`, `benchmarks/summarize.py`.

## 5. Decisions (don't re-litigate)
- **Anchor and reference sets.** Anchor = `Husary_128kbps`.
  - Peers = the other studio reciters in `benchmarks/roster.py`, Husary Muallim included.
  - Imams = the `taraweeh` set.
- **Band formula.** Band = hull of [Husary median, peer consensus (median of per-peer medians)]. Scale = median of the within-reciter 1.4826·MAD values, floored per unit.
- **Ruler choice.** The ruler for each rule (`core_counts` vs `counts`) is chosen by the lowest robust coefficient of variation on Husary. `count_scale` = 2 / Husary's madd_tabii median, and is used for display only.
- **Uncalibrated rules.** `madd_arid_lissukun` and `madd_leen` are the reciter's free choice (2/4/6 counts), so they stay on the textbook verdict.
- **Collection vs judgement.** Benchmark collection runs with `calibration=None` (raw verdicts). Calibration is applied offline, so no audio is re-run when bands change.
- **Honest reporting.** No tuning to hit the acceptance thresholds. Category weights may be optimised only as `calibrate.jl` does it: an L2 pull toward the defaults, with every peer held out at ≥ 95.
- **Secrets.** Never write tokens into the repo.

## 6. Next steps
**Done (2026-09-24):**
- **`full` run collected:** 46,285 rows, committed gzipped in `benchmarks/results/kaggle/full/`.
- **Calibration:** `app/data/calibration.json` has 37 rule keys, a quantile-robust scale and default weights. The weight search was rejected by its own guard.
- **Discovery:** `research_agency_lab/experiments/discovery_full.json`.
- **Summary and indices:** `benchmarks/results/summary.json` and `.md`, plus both FAISS indices.
- **Acceptance** (all fail, reported as measured): Husary 97.1 (in-sample), Dosari adapted 89.0, FP reduction 1.2 %.
- **README:** rewritten for v2 with the method and results.
- **Rejected calibration outputs:** the first run's unbounded weight search (wasl ×30) and the MAD-only scale were both rejected. See the README.

**Remaining:**
1. **Collect the `fill` run.** It was launched 2026-09-24 ~05:00 UTC with `--skip-done benchmarks/done_full.json`; Sudais, Juhaynee, Shuraym and Qatami go first. Then:
   - `collect --tag fill --kernels 5`
   - gzip the rows
   - rerun `calibrate.jl`, `discover.jl` and `summarize.py` on `full` + `fill` rows together
   - update the README tables with the new numbers
2. **Reciters outside Quran-MD.** Tablawi, Ayyoub, Budair, Matroud and Al-Muaiqly: run on the VM from EveryAyah (`benchmarks/run_benchmark.py --reciters … --modes studio taraweeh_adapted`, strategic set or more), then re-calibrate.
3. **Fingerprint style similarity** should use the calibrated z-scores. `app/fingerprint.py compute_tajweed_vector` still uses textbook-derived fields.
4. **Octave cross-check at scale.** Kaggle kernels have no internet, so apt can't install Octave. Run `octave_bridge.py` on the VM over a Husary subset: rows from the gz files, audio from EveryAyah or a `kaggle datasets download` of Quran-MD part 2. Report the correlation and MAE of Octave vs Python `core_ms`, and of the formants vs Praat.
5. **A real Taraweeh test.** The adapter's ≥ 90 % FP target can only be judged on reverberant live recordings, not EveryAyah studio ayahs. Candidates are the live files used earlier (Dosari Makkah Hud, Kurdi, Qatami 1440).

## 7. Target instance
**acct3** (razan.ashraf.alnajjar@proton.me) is recommended.
- Its Kaggle account `razanashrafalnajjar` owns the code dataset and the smoke kernels.
- Don't run `all` in parallel: the free tier allows only 5 concurrent Kaggle batch sessions, and every instance would push to the same branch.

## 8. Environment notes
- **Julia version.** The VM has Julia 1.10.9, but `Manifest.toml` was resolved on 1.11.5. Try this first:
  `julia --project=research_agency_lab/substrate_library/julia -e 'using Pkg; Pkg.resolve(); Pkg.instantiate(); Pkg.precompile()'`
  If that fails, re-add the packages listed in `Project.toml`. Precompiling ModelingToolkit and OrdinaryDiffEq takes about 20 minutes.
- **Octave version.** The VM has Octave 6.4 and needs the signal package (`lpc`, `hilbert`, `butter`, `filtfilt`). Check with `octave-cli --eval "pkg load signal"`; if it's missing, install `octave-signal` (apt) or `pkg install -forge signal`.
- **Kaggle quotas.** 12 h per session, 5 concurrent batch CPU sessions, 30 GPU h/week. Kernels need internet on (pip installs and the Hugging Face wav2vec2/ECAPA downloads).
- **Modal.** The $30/month credit can be used from the VM for CPU fan-out; `app/modal_endpoint.py` already exists. It wasn't reachable from the previous container. Modal's list prices are about $0.047 per CPU-core-hour and about $0.59 per T4-hour (before region and non-preemptible multipliers).
- **VM compute.** The VM (4 vCPU) should orchestrate, calibrate, run Julia/Octave and handle the EveryAyah-only reciters. Bulk audio runs belong on Kaggle.
- **Credentials.** The user pasted Modal, Kaggle and Lightning keys in an earlier chat; recommend rotating them.
