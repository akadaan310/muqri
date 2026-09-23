# Handoff: qaari-eval v2 → next session

Branch: `claude/qaari-eval-engine-btwkjt` (v2 code committed; lint, mypy and pytest all green).

## State
- v2 engine is complete: 38 rule types, Sifaat analyzers, Taraweeh adapter, 232-d fingerprint, Modal endpoint, notebooks and tests.
- Benchmark runner: `benchmarks/run_benchmark.py` (EveryAyah, ayah-by-ayah, resumable JSONL, `--shard i/n`, `--verses strategic|all|S:A-B`).
  - Summarizer: `benchmarks/summarize.py`, which builds `index/masterclass_reciters.faiss` and `index/taraweeh_reciters.faiss`.
  - Roster: `benchmarks/roster.py`, with 8 studio/ijaazah reciters (Husary, Husary Muallim, Minshawy, Hudhaify, Abdul Basit, Alafasy, Tablaway, Ayyoub) and 8 imams (Dosari, Qatami, Shuraym, Sudais, Juhaynee, Budair, Matroud, Muaiqly).
- Partial runs were committed in `benchmarks/results/runs_shard*.jsonl` (strategic set, the first ~100+ rows, mostly Husary). The run was interrupted; resume it with the same command. Rows already written are skipped.

## Key finding (why Husary scores ~72, not ≥98)
The rule thresholds are hand-set textbook numbers ("2 counts ±0.25"), and the harakah unit is a per-ayah median of CTC short-syllable spans (260–520 ms for Husary, which is inflated by consonants and closures). Measured Husary distributions in studio mode:

| Rule | Median measured | Expected | FAIL |
|---|---|---|---|
| madd_tabii | 2.20 (p90 4.06) | 2 | 47/194 |
| ghunnah | 3.35 | 2–2.5 | 20/30 |
| idgham_ghunnah | 3.16 | 2–2.5 | 25/34 |
| ikhfa | 2.37 | 2 | 25/46 |
| munfasil | 6.1 (p10 0.4, alignment misses) | 4–5 | 22/28 |
| izhar_shafawi | 1.13 | ≤1.3 | 14/45 |

So these are calibration and measurement errors, not recitation errors.

## Direction from the user (do this next)
**Husary is the SET EXAMPLE SCORER.**
1. Run Husary (`Husary_128kbps`, plus `Husary_Muallim_128kbps`) on **all 6236 ayahs** (`--verses all --reciters Husary_128kbps --modes studio`, sharded; or use the Kaggle notebook).
   - Store the full per-rule metric distributions: add the raw metrics (duration ms, nasal contrast, F2, burst dB, HNR, voicing…) to `compact()` in `run_benchmark.py`, not only the status.
2. **Calibrate from Husary.** For each rule key (use `datasets/strategic_verses.rule_key`), derive the reference distribution. Normalize the measured counts by his tempo, using the same haraka estimator.
   - Set PASS = within Husary's [p5, p95] (or a robust z ≤ 2 from his median/MAD), WARNING up to z ≤ 3, FAIL beyond.
   - Replace the hard-coded `TOLERANCE` and `band_status` targets with a data file `app/data/calibration.json` (per rule key, per tareeq, per metric: median, MAD, pctiles), loaded by the validators. Keep the textbook targets only as a fallback and as a sanity check (e.g. the calibrated madd_tabii centre must stay near 2).
   - Also fit the harakah estimator itself so that Husary's madd_tabii median = 2.0 (a scale factor on the CTC syllable median).
3. **Validate on the ijaazah peers** (Minshawy, Hudhaify, Abdul Basit, Alafasy, Tablaway, Ayyoub, Husary Muallim).
   - Use leave-one-reciter-out: calibrate on Husary plus the peers minus one, then score the held-out peer. Those peers should also land at ≥95.
   - Rules where the peers legitimately differ from Husary (e.g. ghunnah length, munfasil 4 vs 5) mean the band is widened to the peer consensus. Tune `CATEGORY_WEIGHTS` so that the scores separate the peers from the imams while the peers stay high. Report this honestly, and don't overfit.
4. **Score everyone else** against the "winning examples". Husary and the peers per rule key are the comparison metric for each imam reciter, in studio and taraweeh_adapted modes. Per-rule deltas go in `summary.json`, and the fingerprint's style similarity uses the calibrated z-scores.
5. Then rerun the acceptance checks: `QAARI_ACCEPTANCE=1 pytest tests/test_benchmarks.py` (Husary ≥98, Dosari adapted ≥95, FP reduction ≥90%). Report the real numbers.

## Caveats
- The box is a 4-core CPU: about 10 rows/min. Whole-Qur'an runs should go to Kaggle/Colab GPU (see the notebooks) or use many shards.
- The munfasil/muttasil p10 of about 0.4 is a CTC misalignment (a madd span collapsed). Look at alignment confidence and skip low-confidence spans rather than FAIL them.
- The blind RT60 estimate is biased low above 1 s. Raad Al-Kurdi has no ayah-level dataset (only surah-level on archive.org).
- The Write tool turns `\u` escapes into literal combining marks. The scratchpad `escape_marks.py` script is gone in a new session, so write Arabic marks via Python with explicit escapes.
