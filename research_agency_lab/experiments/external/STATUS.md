# Learner datasets through the engine — status at pause (2026-09-24)

Paused to make recitation synthesis the focus of Sprint 4. Everything below is committed.

## Measured

| dataset | clips | graded | result |
|---|---|---|---|
| sobolev210 (Hafs) | 451 ayahs, 1,933 words, 47 reviewer-tagged | Modal, known verses | engine flags 64 % of tagged words, 31 % of untagged (lift 2.06 over chance); worst identity margin AUC 0.67; Tajweed-tagged recall 10/11 |
| qdat_bench | 108 of 159 (webapp run, stopped) | local | munfasil counts track labels (Spearman 0.87) but read ~1.15 counts long; ikhfa noon has no identity test (see below) |
| Surah al-Ikhlas | 1,503 | Modal, known verses | graded (`ikhlas_engine.jsonl`, gitignored); comparison not yet written — almost all labels are qalqalah of the dal at 5 places |
| RetaSy | 5,226 located of 6,828 | dumped on T4; grading launched | 808 labelled correct/in_correct located; 4,193 unlabelled (for the learner prior) |
| QDAT | 1,505 | dumped on T4 | not graded |

## Findings
- **Recordings cut inside the last letter** are the largest learner-only failure: the last letter ends
  within 0.12 s of the file end in 58 % of learner ayahs (2.7 % of masters) and fails there 46 % vs 11 %.
  Withholding those verdicts did not beat the baseline (tagged errors sit there too), so it is reported
  as `letter.edge` / `recording.tail_room_s`, not used.
- **Learner phone audio has no silence**: the last 60 ms sit ~20 dB under speech; loudness cannot mark
  a truncated ending.
- **Ikhfa noon vs plain noon is untested**: ں has no confusion set. Adding ں→ن on 1,039 master
  instances gives median margin −44 nats but 6.9 % > 0 — too many false alarms for a verdict as is.

## Built, not yet run
- `app/verse_locate.py` (committed, tested): transcript → exact verses and words.
- `modal_external.py::submit / grade_sub`: every clip as a user submission — Whisper (tarteel base,
  large-v3-turbo fine-tune) → locate → grade. All 12,523 clips (RetaSy 6,828, sobolev 1,042 incl.
  Qalun, Ikhlas 1,506, QDAT 1,505, QuranMB 1,642 with phone GT) are staged on the `qaari-runs` volume
  under `external/<name>/submissions.jsonl`. Next: smoke `--limit 16`, then all.
- `substrate_library/julia/learner_prior.jl`: learner mean/spread per skill (recovery verified);
  waiting on the RetaSy grades, then a held-out-learner test against the current floor prior.

Modal spend today: ~$1.71 (metered $16.38 this month, covered by credits).
