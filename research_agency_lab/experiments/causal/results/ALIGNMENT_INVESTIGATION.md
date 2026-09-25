# Alignment Causality Investigation

This investigation is read-only. It uses the committed code, `data/records.json` (local, gitignored), the stored clips in `data/audio/ghunnah/`, and the letter-corpus measurement docs. Nothing was re-run: no engine call, no WORLD or phase-vocoder call, no new audio, no Modal.

The one new calculation is `results/alignment_clip_check.py`. It re-measures the **stored** ghunnah clips with the same `physics._band_energy` function that produced the stored `nasal_ratio_db`. It reproduces every stored value exactly (check 1 in its output).

Line numbers refer to the files as they are at `ec241c7`.

---

## 1. Executive finding

The stored evidence cannot say whether the ghunnah placement asymmetry runs through a change in Muqri's alignment. The records never kept the post-transform letter boundaries, the frames, or the posteriors. The one segmentation value they did keep, L24's duration, is unchanged at 0.84 s in all nine WORLD-edited ghunnah records.

What the evidence does establish is this:

- **The three placements do not apply the same physical change to the region Muqri scores.** Outside two 20 ms windows they are identical. The −20 ms edit is the only one that:
  - leaves the last 20 ms of the baseline ن region unedited, [10.30, 10.32]. This is the nasal-to-vowel transition.
  - edits the last 20 ms of the preceding kasra, [9.46, 9.48].
- **The "nearly identical physical nasal change" is an artifact of the physics metric.** `nasal_ratio_db` applies a Hann window over the whole span. That window gives the span's first and last 20 ms an energy weight of about 4×10⁻⁷, so the metric cannot see the only places where the placements differ.
- **The boundary records compare physics across different regions.** Each one measures the shifted edited span against the unshifted baseline span.

So explanation D holds for the physics layer. For the Muqri layer, A, B and C cannot be separated with what was stored.

**Correction to the premise.** The stored physical deltas are −20.69 dB for the edit at the span and −20.89 dB for the −20 ms shift (+20 ms: −20.51 dB). The task statement has these two swapped. `REVIEW_smoke.md` has them the right way round.

---

## 2. Implementation trace

### Answers to the ten questions

| # | Question | Answer (from code) | Where |
|---|---|---|---|
| 1 | How is the source segment selected? | From precomputed letter-corpus docs, `letter_corpus/data/measure/Husary_128kbps/*.json`, not from the causal baseline run. Letters: the clean instance with the highest `letter_score`, excluding edge letters. Rules: the passing, non-edge instance with the smallest \|z_masters\|. The span is the stored `onset_s` → `onset_s + duration_s`. | `harness.py` `Harness.find_letter` l.160–171, `find_rule` l.173–183, `letter_unit` l.185–193, `rule_unit` l.195–201 |
| 2 | How are the perturbation boundaries defined? | `[unit.start_s, unit.end_s]`, i.e. Muqri's Viterbi onsets on the 40 ms grid (ghunnah: 9.48 = frame 237, 10.32 = frame 258). Boundary controls add ±0.02 s to both ends. WORLD edits use a hard 0/1 mask on 5 ms WORLD frames over the span. The ayah is analysed and resynthesised on span ±0.3 s, and spliced back with 10 ms crossfades at the context edges, not at the span edges. | `harness.py` l.234–235; `transforms.py` `_world_edit` l.30–41; `world_lab.py` `FP = 5.0` l.39, `nasal` l.95–99; `perturb.py` `splice` l.45–53 |
| 3 | Is the transformed audio re-aligned by Muqri? | **Yes.** Every variant goes through a full `Engine.analyze` on the whole edited ayah. | `harness.py` `run` l.216–223 → `Harness.analyze` l.156–157 |
| 4 | Do baseline and transformed audio get independently computed Viterbi alignments? | **Yes.** Each `analyze` recomputes posteriors, the ayah span, and the per-phoneme Viterbi path. No alignment is passed in or reused. | `engine.py` l.236, 260, 273–275; `analysis.py` l.224 |
| 5 | Are the same frames compared? | **Not guaranteed, and not recorded.** Comparison is by letter ID (`11:2:L24`): the value from the baseline's alignment is differenced against the value from the variant's own alignment. They cover the same frames only if the two alignments agree, and that was not stored. | `harness.py` `muqri_view` l.86–115; `schema.deltas` l.127–133 |
| 6 | Are Muqri's letter boundaries stored in the causal records? | **No.** `muqri_view` keeps only `letter:duration_s` (the sum over the unit's letters), `vowel:duration_s`, and rule counts. It does not keep `onset_s`, `frames`, or any neighbouring letter's timing. The full measurements (`runs[key]["m"]`) were held in memory and never written. `unit.start_s/end_s` is the baseline span only. | `harness.py` l.103–108, 220, 276–295 |
| 7 | Are frame-level likelihoods around the edit available? | **No.** Posteriors (`Engine.posteriors`) are neither returned by `analyze` nor stored. | `engine.py` l.196–204, 236 |
| 8 | Is there any post-transform segmentation information? | Only indirect information. (a) `letter:duration_s` and `rule:*:counts` of the unit. (b) `vowel:duration_s` for letter+vowel units. (c) `elsewhere()`: identity or characteristic **verdict flips** of other letters, with no margins and no timing. (d) The stored clips hold the audio, not Muqri's segmentation. | `harness.py` l.103–114, 118–131 |
| 9 | Which function produces the alignment? | `app/analysis.py::ctc_viterbi` (l.53). It is called twice per analysis. `app/submission.py::walk_alignment` (l.313; joint path at l.341 for a single ayah) places the ayah in the recording. `engine._with_context` (l.47) pads that by 8 frames. `app/analysis.py::analyse_clip` (l.195) then re-runs `ctc_viterbi` on the clip (l.224); this gives the `first`/`last` frame of every phoneme, and all per-letter windows come from it. | as listed |
| 10 | Which function produces the measurements the harness consumes? | `app/measurements.py::build` (l.98), reached from `Engine.analyze` l.336 through `submission.build_report` (l.537). The harness reads `["measurements"]` and flattens it with `harness.muqri_view`. The numbers come from `analyse_clip`: identity/`makhraj` from `ctc_log_likelihood` over a context window (l.263–294), and characteristic margins from `_pooled` posteriors over the letter's own Viterbi frames (l.306–317). | as listed |

### How each Muqri metric depends on the alignment (`analyse_clip`)

- **identity:margin and makhraj:\*** (l.261–294):
  - These are CTC forward log-likelihood ratios, summed over all paths, on the window `lp[first[ca]−1−3 : last[cb]+3]`.
  - `ca`/`cb` run from two units before to two units after the letter.
  - For L24 that window covers L22 ء … L26 ن plus 3 frames on each side: about 9.08–10.76 s in the baseline. It contains **both** 20 ms windows in which the placements differ.
  - The window's edges come from the Viterbi path. Inside the window, the likelihood marginalises over alignments.
- **head:\*:margin** (l.306–317): computed as `_pooled(lp_full[first[a]−1 : last[b]])`, the log of the **mean probability** over the letter's own Viterbi frames (237–257 for L24 in the baseline). The margin is the expected class minus the best other class.
  - Because the probabilities are averaged, a few frames where the competitor is confident can dominate the pooled competitor.
  - This window is **directly** set by the Viterbi boundaries.
- **letter:duration_s** (l.232–237): `onset(next unit) − onset(unit)`, on the 40 ms grid.
- **Posteriors** (`muaalem_dump.posteriors` l.108–113): the whole edited ayah is encoded in one forward pass. So any posterior frame can, in principle, change when the audio changes anywhere. Nothing in the code limits the edit's influence to the frames it touches.

---

## 3. Ghunnah case

### Data-flow chain (11:2, unit L24 ن, run of 4 phonemes, rule R4 ghunnah)

```
EveryAyah Husary_128kbps/011002.mp3
  └─ app.webapp.decode_upload → 16 kHz float wave (sha256[:16] ae1eec5780e0e76e)       harness.wave l.153
SELECTED UNIT (not from the causal baseline; from the letter-corpus doc 011002.json, written 06:16)
  └─ find_rule("ghunnah"): R4, letters [11:2:L24], z_masters −0.06                         harness l.173
  └─ rule_unit: span [9.48, 10.32] = L24 onset 9.48 + duration 0.84 (frames 237–258)      harness l.195
BASELINE ANALYSIS
  └─ Engine.analyze(wave, [(11,2)], makhraj=True)                                          harness l.209
       posteriors → walk_alignment (joint ctc_viterbi) → _with_context → analyse_clip
       (ctc_viterbi again) → identity / makhraj / sifat / durations → build_report → measurements.build
  └─ muqri_view(m0, "11:2:L24", ["11:2:L24"], word 4) → mb                                 harness l.210
       (the doc's values and mb agree to ≤0.001 on every metric: same alignment and margins)
  └─ physics.measure(wave, 9.48, 10.32) → pb (Hann-windowed nasal_ratio 18.92 dB)          harness l.211
PERTURBATION (boundary−0.02|nasal: g = −12 dB, span shifted to [9.46, 10.30])
  └─ transforms._world_edit: WORLD analysis of [9.16, 10.60] (span ±0.3 s),
       mask w = 1 on WORLD frames of [9.46, 10.30], nasal(): ±12 dB at 250/900 Hz,
       resynthesise [9.16, 10.60], splice with 10 ms crossfades at 9.16 and 10.60           transforms l.30–41, 68–69
TRANSFORMED AUDIO
  └─ full ayah; samples differ from the original over [9.16, 10.60] (check 4)
  └─ clip [9.06, 10.70] saved as PCM16                                                      harness l.296, 313–317
MUQRI ANALYSIS + ALIGNMENT (independent)
  └─ Engine.analyze(new wave) → new posteriors → new walk_alignment → new analyse_clip Viterbi
MEASUREMENT EXTRACTION
  └─ muqri_view(m1, "11:2:L24", …): the letter found BY ID in the new alignment             harness l.221
       stored: identity 44.03, ghonna 10.84, makhraj:م 45.13, letter:duration_s 0.84, counts 2.49
       NOT stored: L24's onset/frames in m1, any posterior, any neighbour's timing
  └─ physics.measure(new, 9.46, 10.30) → Hann nasal_ratio −1.97                             harness l.222
CAUSAL COMPARISON
  └─ delta_muqri   = view(m1: L24 by id) − view(m0: L24 by id)                              harness l.254
  └─ delta_physics = measure(new, [9.46, 10.30]) − measure(orig, [9.48, 10.32]) = −20.89     harness l.255
       ← two different regions, both Hann-weighted to ~0 at their edges
  └─ classify against floors from determinism, noop_splice and noop_world at the UNSHIFTED span  harness l.251–263
```

### Where each confound can enter

| Confound | Entry point |
|---|---|
| Segmentation differences | The second `Engine.analyze` recomputes `walk_alignment` and `analyse_clip`'s `ctc_viterbi`. L24's frames in m1 can differ from m0. Only the duration is stored. |
| Boundary differences | (a) The edit edges are defined on the baseline's 40 ms Viterbi grid; ±20 ms puts them mid-frame. (b) The Muqri windows (identity context, sifat frames) are re-derived per variant. (c) The physics span follows the edit, not Muqri. |
| Resynthesis artifacts | WORLD analysis and resynthesis of span ±0.3 s. That region moves with the placement ([9.18, 10.62] / [9.16, 10.60] / [9.20, 10.64]). The only WORLD no-op control was run at the unshifted placement. |
| Changed acoustic content | The intended edit (interior band levels −8 to −11 dB low band, +9.5 to +13.7 dB anti band). Plus the unintended content: the kasra tail at −20 ms and the fatha onset at +20 ms. |
| Changed analysis windows | Muqri: the identity/makhraj window and the sifat frame window are re-derived from each variant's Viterbi. Physics: `measure` uses the transform's span (shifted for boundary records), with Hann weighting. |

### The −20 ms asymmetry

**Stored Muqri values**, all at g = −12 dB (from `records.json`):

| | untouched | edit at span [9.48, 10.32] | −20 ms [9.46, 10.30] | +20 ms [9.50, 10.34] |
|---|---|---|---|---|
| makhraj:م | 45.576 | 37.977 | 45.132 | 38.514 |
| identity:margin | 44.201 | 40.354 | 44.030 | 40.246 |
| head:ghonna:margin | 11.278 | 9.686 | 10.843 | 9.710 |
| letter:duration_s | 0.84 | 0.84 | 0.84 | 0.84 |
| rule:ghunnah:counts | 2.49 | 2.49 | 2.49 | 2.49 |
| stored Δnasal_ratio_db (own span vs baseline span) | — | −20.69 | −20.89 | −20.51 |

**Re-measured from the stored clips** (`alignment_clip_check.py`):

- **Interior.** Over [9.50, 10.30], all three edits give the same per-20 ms band levels to within about 0.3 dB (check 3; e.g. [10.28, 10.30]: −10.7/+12.8, −10.6/+13.1, −10.8/+12.8).
- **Edges.** The placements differ in exactly these windows:

| 20 ms window | its role in the baseline alignment | untouched nasal ratio | edit at span | −20 ms | +20 ms |
|---|---|---|---|---|---|
| [9.46, 9.48] | end of L23 kasra | 16.67 dB | untouched | **edited** (−8.3/+9.8) | untouched |
| [9.48, 9.50] | start of L24 ن | 17.63 dB | edited | edited | **untouched** |
| [10.30, 10.32] | **end of L24 ن**: the nasal-to-vowel transition, where the ratio falls from 21.17 to 7.33 dB | 7.33 dB | edited (−11.6/+11.1) | **untouched** (+0.4/+0.2) | edited |
| [10.32, 10.34] | start of L25 fatha | −0.67 dB | untouched | untouched | **edited** (−11.2/+11.3) |

- **On the fixed baseline region [9.48, 10.32]:**
  - The stored Hann metric gives **−20.69 dB for all three placements** (−1.767 / −1.769 / −1.766). The Hann weight on each edge's 20 ms is 3.9×10⁻⁷, so the metric is blind to the only regions that differ.
  - An unweighted (rectangular) window separates them: −20.33 (span), −18.43 (−20 ms), −19.59 (+20 ms) dB.

**What this explains, and what it does not:**

1. The claim "same physical change, different Muqri response" does not hold as stated. The physical changes are identical in the interior and differ at the edges, and the physics metric could not register the edges.
2. The span edit and the +20 ms edit give nearly identical Muqri results (\|Δ\| ≤ 0.61 nats). So editing or leaving the first 20 ms of L24 makes little difference, and so does editing the first 20 ms of L25.
3. The −20 ms placement is the only one that (i) leaves [10.30, 10.32] untouched and (ii) edits [9.46, 9.48]. It is also the only one whose WORLD context is shifted earlier. Its Muqri values return to within 0.2–0.5 nats of the untouched baseline on identity, makhraj:م and ghonna. The stored evidence **cannot say which of (i), (ii) or the context shift is responsible**. Each was varied only together with the others.
4. Muqri's segmentation of L24 in the variants is unknown except for its duration, 0.84 s everywhere. That rules out a change in L24's length on the 40 ms grid. It does not rule out L24 and L25 shifting together by whole frames. It says nothing about L22, L23, L25 or L26, whose boundaries set the identity/makhraj window.
5. The mechanism by which one 20 ms window could dominate is at least **possible** from the implementation:
   - `_pooled` averages probabilities, so a few confident competitor frames can dominate a margin.
   - CTC posteriors tend to concentrate a token's evidence in a few frames.

   Whether the ن's evidence actually sits at the transition frame is **not observable** from stored data, because no posteriors were saved. This is a hypothesis, not a finding.

---

## 4. Alignment behaviour

Transformed audio is **independently re-aligned**. It is never fixed to the baseline alignment.

- Every variant runs through `Engine.analyze` from the waveform up: new posteriors, a new `walk_alignment`, and a new `analyse_clip` Viterbi.
- The harness pairs baseline and variant **by letter ID**, not by frames or time.
- The unit's time span (`unit.start_s/end_s`) is used only to place the edit and to measure physics. It is never passed to Muqri.
- Neither the variant's alignment nor the baseline's full alignment was saved. The baseline's L24 alignment can be recovered only because the letter-corpus doc (which supplied the span) matches the causal baseline on every stored metric.

---

## 5. WORLD pass-through

**Stored evidence** (`noop_world` = WORLD analysis and resynthesis of span ±0.3 s, no parameter changed, at the unshifted placement only):

| Unit | Muqri deltas beyond the 1-nat floor | largest \|Δ\| within floor | Elsewhere flips | Letter/vowel duration | Physics deltas (Praat, own span) |
|---|---|---|---|---|---|
| ز 48:19:L50 | none | hams/jahr +0.94, tafkhim −0.89 | 0 | 0.28 / 0.20 s unchanged | rms +1.12 dB, nasal ratio +1.04 dB, F2 +15.8 Hz |
| ص 70:5:L6 | makhraj:س +2.19, makhraj:ز +1.70 | vowel identity +0.64 | **1: 70:5:L12 tafkheem_or_taqeeq** | 0.32 / 0.24 s unchanged | rms +2.24 dB, nasal ratio −3.31 dB, F1 −33.4 Hz, HNR +1.23 dB |
| ghunnah 11:2:L24 | makhraj:م +1.34, ل +1.26, ر +1.23 | identity −0.22 | 0 | 0.84 s unchanged | rms +0.60 dB, F3 +81 Hz, nasal ratio −0.18 dB |

In the ghunnah clip, the resynthesised samples cover exactly [9.18, 10.62], the WORLD context. The waveform differs from the original throughout that region, as expected: WORLD does not preserve phase.

**What is established:**

- WORLD pass-through is **not physically neutral**. Independent Praat measurements change on every unit (RMS by 0.6–2.2 dB, and at ص a nasal ratio of −3.3 dB and F1 of −33 Hz). Here "vocoder artifact" and "actual acoustic change" are **not separable categories**: the artifact *is* an acoustic change, and one that the physics layer measures.
- It moves some Muqri margins beyond the 1-nat floor (makhraj only, up to 2.19 nats). It flips one characteristic verdict on a letter outside the unit (70:5:L12 tafkhim).
- It did not change the unit's Muqri duration on any unit.

**What remains ambiguous:**

- **Alignment change.** The unit's durations are unchanged. Onsets were not stored. For 70:5:L12, only the verdict flip was stored: no margin before or after, no timing. Whether L12's frames moved is unknown.
- **Measurement-window change.** This depends on the alignment above. It is not observable from what was stored.
- **Resynthesis artifact vs changed spectral content in the scored band.** The physics layer shows that the spectrum changed. It cannot say which part of that change Muqri responded to.
- **The resynthesis footprint at shifted placements was never measured.** `noop_world` exists only at the unshifted span, yet it is also the noise floor for the ±20 ms boundary records. In the ghunnah clips, the shifted contexts' band levels match the span edit's within about 0.5 dB outside the edit windows (check 3). That is evidence that the context shift changes little spectrally, but it is not a Muqri-level control.

---

## 6. Causal model

| | Explanation | Verdict | Evidence |
|---|---|---|---|
| **A** | Physical perturbation → Muqri measurement (no segmentation change) | **possible** | Consistent with every stored value. Dose response at the span: ghonna 11.33 / 9.69 / 6.49 at −6 / −12 / −18 dB, with L24's duration and counts unchanged in every WORLD-edited ghunnah record. The implementation guarantees a direct path: the margins are functions of the posteriors of the edited audio. But A requires the frames to be unchanged, which was never recorded. |
| **B** | Physical perturbation → alignment/segmentation change → Muqri measurement (alignment as *the* pathway) | **unsupported** (not excluded) | There is no positive evidence of any segmentation change: duration and counts are invariant, and no flips occurred elsewhere in 11:2. B as the *sole* pathway also contradicts the implementation, because the edited audio changes the posteriors inside every window regardless of the boundaries. What cannot be excluded is a same-duration shift of L24, or a shift of the context letters that set the identity/makhraj window. |
| **C** | Direct acoustic change **and** an additional alignment-mediated change | **indeterminate** | The direct part is certain by construction. Whether an alignment-mediated part exists, and how large it is, cannot be determined without the variant's frames or a fixed-alignment re-score. Nothing stored distinguishes C from A. |
| **D** | The experiment compares different physical regions | **supported** for the physics layer; **indeterminate** for the Muqri layer | Physics: each boundary record measures the shifted edited span against the unshifted baseline span (harness l.222 vs l.211). Separately, the Hann-weighted metric is blind to both spans' outer 20 ms, which is exactly where the placements differ, so the "same physical change" figures (−20.69 / −20.89 / −20.51) do not show equal edits on the scored region. Muqri: the comparison is by letter ID across independent alignments, so it covers the same region only if the alignments agree, which was not stored. |

**Additional explanation (E), outside A–D and not excluded: placement changes *which audio content* is edited.** The −20 ms edit alone leaves the ن-to-fatha transition [10.30, 10.32] untouched and alone attenuates the kasra tail. If Muqri's evidence for L24 depends heavily on the transition window, the asymmetry follows with **no alignment change at all**. This fits the stored values (span ≈ +20 ms, which both edit [10.30, 10.32]; −20 ms ≈ untouched). It is untested, because the three windows were never varied one at a time.

---

## 7. Provenance

**Known from the records** (identical in all 40):

- `code_revision` 7bef87300e41642e7c639fdd4908816fa045c954, `code_dirty` true.
- Python 3.11.16; numpy 2.4.6, librosa 0.11.0, pyworld 0.3.5, praat-parselmouth 0.4.7, torch 2.14.0+cpu, quran-muaalem 0.2.2.
- Model *name* obadx/muaalem-model-v3_2; audio sha256[:16]; timestamps 09:41:57–09:44:30Z.

**What `code_dirty` actually covers.** The flag is `git status --porcelain` over `app`, the causal directory, `world_lab.py` and `perturb.py` (harness l.148–151). At 7bef873 the causal directory was not tracked (`git ls-tree 7bef873` is empty), so the untracked harness alone makes the flag true. The flag therefore **cannot say whether `app/` or the transforms were also modified.**

**Circumstantial reconstruction** (file-system metadata, not cryptographic):

- **The imported modules.** The CPython `.pyc` headers store the source mtime and size at compile time. For all of the following, the current sources match their `.pyc` headers, and every source mtime predates the run:
  - causal `schema.py`, `physics.py`, `transforms.py`, `__init__.py`
  - `synthesis/world_lab.py`, `letter_corpus/perturb.py`, `letter_corpus/assemble.py`
  - `app/engine.py`, `analysis.py`, `submission.py`, `measurements.py`, `letters.py`, `ghunnah.py`, `blindspots.py`

  `schema.py` is the latest of these (09:40:07; `.pyc` compiled at 09:40:09, before the first record at 09:41:57). If mtimes were not altered, these modules ran exactly as they now stand in 48c4c6e.
- **`app/`.** It has no diff between 7bef873 and HEAD, the working tree is clean, and its sources were last written at 05:35:53. So the engine that ran is, with high but non-cryptographic confidence, 7bef873's `app/`.
- **Other inputs.** Also untouched since before the run:
  - `learner_eval/muaalem_eval.py` and `muaalem_dump.py` (09-23)
  - `quran/reference_stats.json` (09-24)
  - `octave/qaari_features.m` (09-23)
  - `causal/julia/xcheck.jl` (09:33:36)

**Not reconstructable:**

1. **`harness.py` as it ran.** The current file was written at 09:45:53, after the last record (09:44:30). A script run as `__main__` leaves no `.pyc`. The claim that only the table-building code changed comes from the session log, not from an artifact. The record structure is consistent with the committed `run_unit`, but that does not prove byte identity.
2. **The dirty-tree content.** No diff, patch, file hashes or snapshot were recorded.
3. **The model weights revision.** Only the model name is recorded. The local HF cache holds one snapshot, `01a1ef9fbe40d144ef845101e89ff924aed3fef5`, but the records do not name it.
4. **The letter-corpus docs that chose the units.** They are gitignored and carry no code revision. Their per-letter metrics do match each unit's causal baseline (to ≤0.001), so for the three units they are equivalent to the run's own baseline.
5. **Engine-internal state.** Posteriors, alignments and full reports were never persisted.

**Conclusion.** The 40 records came from a dirty tree at 7bef873, and they must not be described as generated by 48c4c6e. The imported library code very likely equals 48c4c6e's. The harness driver is the one piece that cannot be verified.

---

## 8. What we still cannot know from stored evidence

- Whether Muqri's Viterbi placed L24, or L22–L26, on the same frames in any variant as in the baseline. Only L24's summed duration is known.
- Whether the identity/makhraj window or the sifat frame window moved in any variant.
- The frame-by-frame posteriors of ن, its competitors and the ghonna head around 9.46–9.50 s and 10.28–10.34 s, and so whether Muqri's L24 evidence concentrates at the nasal-to-vowel transition.
- Which of these three drives the −20 ms reversion:
  - (i) the unedited [10.30, 10.32]
  - (ii) the edited kasra tail [9.46, 9.48]
  - (iii) the 20 ms shift of the WORLD analysis and resynthesis context
- The WORLD resynthesis footprint on Muqri at the shifted placements.
- For 70:5:L12, whether the tafkhim flip under WORLD pass-through came with an alignment change, and its margin before and after.
- The run-to-run determinism of the WORLD transforms. Each was run once; WORLD's harvest, CheapTrick and D4C contain no random step, but this was not tested.
- Byte-exact runtime `harness.py`, and the model weights revision.
- Anything beyond one reciter and one ghunnah instance.

---

## 9. Minimum discriminating experiment (defined, NOT run)

**Scope:** one unit (11:2 L24 ن), one transform (nasal, g = −12 dB), Husary audio only. Seven `Engine.analyze` calls in total. No Modal, no new transform types, no engine change.

| Variant | Edit region | Separates |
|---|---|---|
| V0 | none (untouched) | reference; captures the full baseline state |
| V1 | nasal −12 on [9.48, 10.32] (as stored) | reproducibility of the stored record; reference edit |
| V2 | nasal −12 on [9.46, 10.30] (as stored) | reproducibility of the stored asymmetry |
| V3 | nasal −12 on [9.48, 10.30] | (i) alone: leaves the transition window unedited, with no kasra edit |
| V4 | nasal −12 on [9.46, 10.32] | (ii) alone: edits the kasra tail and keeps the transition edit |
| V5 | noop_world with the context shifted −20 ms ([9.16, 10.60]) | (iii): the resynthesis footprint at the shifted placement |

**Captured per variant (lab-side only; this is new record content, not an engine change):**

1. The full `measurements` report, including every letter's `onset_s`/`duration_s`.
2. The `analyse_clip` `Unit.frames` for L20–L30. Obtaining these without touching `app/` means calling the public `analyse_clip` from the lab on the posteriors the engine used.
3. The posterior matrix (T × C float32, about 500 frames).

**Two scorings of every variant:**

- **(a) Engine as-is:** independent alignment. This is the current method.
- **(b) Fixed alignment:** the variant's posteriors scored on **V0's** Viterbi `first`/`last`. This is a lab wrapper that supplies V0's path in place of `ctc_viterbi` inside a lab copy of the scoring loop. Production code stays unmodified.

**Physics:** measured on the **fixed** baseline L24 region and on each variant's **own** Muqri L24 frames. Use a rectangular window plus per-20 ms band levels (as in `alignment_clip_check.py`), alongside the existing Hann metric.

**Decision rules:**

- (a) ≈ (b) for every variant, and frames equal to V0's → **A** (no alignment mediation). Then V3/V4 decide between (i) and (ii).
- (b) ≈ V0 while (a) moves, with frames differing → **B**.
- Both (a) and (b) move, with (a) − (b) beyond the 1-nat floor and frames differing → **C**. The decomposition is the direct part = (b) − V0 and the alignment part = (a) − (b).
- V5 beyond the floor on the metrics that V2 reverted → **(iii)**: a resynthesis-context confound.
- V1/V2 not reproducing the stored values within the floor → the stored asymmetry is itself unstable, and must be re-established before any interpretation.

This is the smallest set that isolates (i), (ii) and (iii) one at a time and observes alignment directly. Any placement sweep should wait until it has run.

---

## 10. Production safety

This investigation changed **nothing outside `research_agency_lab/experiments/causal/results/`**. Two files were added:

- `results/ALIGNMENT_INVESTIGATION.md`: this document
- `results/alignment_clip_check.py`: a read-only re-measurement of the stored clips; it writes nothing

The following were **not** touched: `app/`, Sessions, rounds 1–5, Neo4j runtime, authentication, `data/records.json` and the stored audio.

- No engine run, synthesis, transform, training, TTS, Modal job or corpus generation was started.
- The production engine was not imported: `alignment_clip_check.py` imports only `physics._band_energy`.
