# Alignment Discriminating Experiment

This is the minimum discriminating experiment defined in `ALIGNMENT_INVESTIGATION.md` §9, run once.

| Item | Location |
|---|---|
| Script | `causal/align_exp.py` |
| Machine-readable results | `results/alignment_experiment.json` |
| Local, gitignored data | `data/align_exp/`: full-ayah float32 WAV per variant, and the full posterior matrix per variant (`V*_posteriors.npy`) |

This is a single source unit (Husary, 11:2, letter L24 ن, ghunnah) at a single level (nasal −12 dB). It supports no generalisation, no significance claim and no production claim.

---

## The question

> Does the observed Muqri response remain when the transformed audio is evaluated against the baseline alignment, or does most of the response appear only after independent re-alignment?

**It remains.** With Muqri's alignment held at the untouched baseline's Viterbi path, the span edit (V1) still moves:

| Metric | Held alignment | Independent re-alignment |
|---|---|---|
| ghonna margin (the target) | −1.798 | −1.592 |
| identity margin | −3.847 | −3.847 |
| makhraj:م | −7.599 | −7.599 |

- **None of the response depends on re-alignment.** Re-alignment slightly **reduces** the target response: its contribution to ghonna is +0.206. It leaves identity and makhraj unchanged, with a contribution of 0.000.
- **Re-alignment adds collateral.** It moves six other characteristic margins of L24 by a further −0.23 to −0.65 nats (−0.23, −0.51, −0.62, −0.62, −0.63, −0.65). Four of those metrics cross the 1-nat line only under independent alignment.

The placement asymmetry is a **direct acoustic effect of one 20 ms window**: [10.30, 10.32] s. That window is the second half of the model frame that carries the ن's final CTC spike. It is not an alignment effect.

---

## 1. Design, as run

| Variant | Edit | WORLD context (analysed and resynthesised) | Purpose |
|---|---|---|---|
| V0 | none | — | reference; the held alignment is V0's path |
| V1 | nasal −12 on [9.48, 10.32] | [9.18, 10.62] | the stored span edit |
| V2 | nasal −12 on [9.46, 10.30] | [9.16, 10.60] | the stored −20 ms edit |
| V3 | nasal −12 on [9.48, 10.30] | [9.18, 10.60] | (i): leaves only the transition [10.30, 10.32] unedited |
| V4 | nasal −12 on [9.46, 10.32] | [9.16, 10.62] | (ii): additionally edits the kasra tail [9.46, 9.48] |
| V5 | WORLD pass-through on [9.46, 10.30] | [9.16, 10.60] | (iii): the resynthesis footprint at the shifted placement |

The edits come from the existing `transforms.apply` (`world_lab.nasal`). The samples that actually changed match the WORLD context exactly in every variant (`samples_changed_s`).

**Engine runs.** There are seven independent `Engine.analyze` runs, each with its own acoustic-model forward pass: V0, V0 again as the determinism check, and V1–V5.

**Held-alignment rescoring.** Each of V0–V5 is scored a second time from the same posteriors, with V0's alignment replayed.

- **How the alignment is held.** Two module-level names are rebound inside the experiment process only: `app.submission.ctc_viterbi` (used by `walk_alignment`) and `app.analysis.ctc_viterbi` (used by `analyse_clip`). They replay V0's recorded paths. `T` and the phoneme sequence are asserted equal on every call. The basmala path does not run for ayah 2.
- **Production code is untouched.** Everything else is the unmodified engine: posteriors, CTC likelihoods, pooling, rules and `measurements.build`. No file under `app/` was changed.

**Sanity checks, all passed:**

| Check | Result |
|---|---|
| Determinism | V0 run twice gives identical posteriors, identical paths, and a Muqri Δ of 0 |
| Replay | Scoring V0's posteriors with the replayed V0 path reproduces V0 on every metric (the script raises otherwise) |
| Reproduction of the earlier smoke run | V1 and V2 reproduce the stored smoke records exactly on every numeric Muqri metric (`stored_record_check.delta_vs_stored = {}`), as does V0 against `ghunnah/determinism` |

**Recorded per variant** (JSON):

- the audio sha256, and the WAV and posteriors (local)
- the edit span, WORLD context and changed-sample range
- the full `ctc_viterbi` paths (`first`/`last` of every phoneme) for both calls
- for L20–L30: the sifat frame window, the identity/makhraj window, and per-phoneme frames, all as absolute model frames and seconds
- the Muqri view under the held and independent alignments, their deltas, and the alignment component (independent − held)
- verdict flips elsewhere in the ayah
- per-frame posteriors for frames 231–263: ن, blank, the top 3, and both ghonna classes
- physics on the eight 20 ms edge windows, the edit span, the fixed baseline L24 region, the independent L24 window, and both identity windows

---

## 2. Baseline alignment (V0)

L24's sifat window is frames [237, 258), i.e. 9.48–10.32 s. Its four phonemes sit on frames [237, 238], [240, 241], [243, 245] and [256, 258].

L24's identity/makhraj window is [9.08, 10.76] s.

---

## 3. Comparison table

Definitions:

- **Physical Δ:** measured on the **fixed** baseline L24 region [9.48, 10.32]. `rect` is the unweighted 150–400 / 750–1100 Hz ratio; the Hann metric reads −20.69 dB for V1–V4 alike and cannot see the edges. "Transition window" is the change in the low and anti band levels in [10.30, 10.32].
- **Baseline-alignment Δ and independent-alignment Δ:** relative to V0. Values are given as ghonna / identity / makhraj:م, in nats.
- **Alignment Δ:** the change in Muqri's alignment, and the resulting component (independent − held).
- **Collateral:** the non-target metrics of L24 with |Δ| > 1 nat (the schema's minimum floor), held / independent.

| Variant | Physical Δ | Baseline-alignment Δ | Independent-alignment Δ | Alignment Δ | Collateral >1 nat | Interpretation |
|---|---|---|---|---|---|---|
| V1 span | rect −20.33 dB; transition −11.55 / +11.14 dB | −1.798 / −3.847 / −7.599 | −1.592 / −3.847 / −7.599 | L24 window [237,258)→[237,257): the transition frame 257 is dropped. L24 duration stays 0.84 s because L25's onset holds. L20, L21, L23 and L25 also move. Component: ghonna +0.206, identity and makhraj 0, six heads −0.23 to −0.65 | 7 / 11 | **Combined, dominated by the direct effect.** The full response exists under the held alignment. Re-alignment trims the target by 11 % and adds 4 collateral crossings. |
| V2 −20 ms | rect −18.43 dB; transition +0.41 / +0.16 (unedited); kasra tail −8.27 / +9.76 | −0.435 / −0.171 / −0.444 | identical | L24 window unchanged. L20, L21 and L23 move. Component **0.000** on every L24 metric | 2 / 2 | **Direct only, and small.** The asymmetry is not an alignment effect. |
| V3 transition unedited | rect −18.47 dB; transition +0.41 / +0.15 (unedited) | −0.249 / −1.010 / −0.846 | identical | L24 window unchanged. L20, L21 and L23 move. Component 0.000 | 3 / 3 | Leaving **only** [10.30, 10.32] unedited removes most of V1's response (makhraj:م −0.85 vs −7.60; ghonna −0.25 vs −1.80). |
| V4 kasra tail also edited | rect −20.31 dB; transition −11.55 / +11.15; kasra tail −8.29 / +9.74 | −1.896 / −2.454 / −6.800 | −1.673 / −2.454 / −6.800 | Same as V1 (frame 257 dropped). Component: ghonna +0.223, identity and makhraj 0, six heads −0.21 to −0.64 | 7 / 11 | Editing the kasra tail does **not** cancel the response. It reduces identity's response by 1.39 nats relative to V1 and leaves ghonna and makhraj near V1. |
| V5 WORLD, shifted context | rect −0.11 dB; transition +0.43 / +0.16 | −0.211 / −0.437 / +0.703 | identical | L24 window unchanged. **L23 moves** ([231,233]→[231,234]). Component 0.000 on L24 | 0 / 0 | Resynthesis is **not neutral**: it changes the alignment of a neighbouring letter and moves L24's margins by up to 0.70 nats. It is too small to explain V2's reversion. |

Full per-metric deltas are in the JSON, under `delta_held`, `delta_independent` and `alignment_component` for each variant.

---

## 4. What the frames show

Values are log-posteriors from `frame_posteriors` in the JSON. For ghonna, [مغن] is the nasal class; [لا غنة] is its competitor.

| Frame (s) | V0 | V1 | V2 | V3 | V4 | V5 |
|---|---|---|---|---|---|---|
| 256 (10.24–10.28): log p(ن) | −0.23 | −0.00 | −0.02 | −0.03 | −0.00 | −0.05 |
| **257 (10.28–10.32): log p(ن)** | **−0.00** | **−3.81** (blank −0.09) | −0.00 | −0.00 | **−3.77** | −0.00 |
| 257: ghonna [مغن] | −0.00 | **−1.85** | −0.00 | −0.00 | **−1.83** | −0.00 |

- **Where the evidence is.** The ن is carried by a few CTC spikes: frame 237 at the onset, and frames 256–257 at the end. Every other frame in L24 is blank-dominated, and both ghonna classes sit near e⁻¹³ there. So the pooled ghonna margin (`_pooled`, a mean in probability space) is governed by the spike frames.
- **What removes the spike.** Only the edits that cover [10.30, 10.32], the second half of frame 257, remove the frame-257 spike (V1, V4).
- **What does not.** V2 and V3 edit the first half of frame 257, [10.28, 10.30], and the spike survives. So does WORLD resynthesis of that frame without an edit (V5).
- **How re-alignment reacts.** Once the spike is gone, the independent Viterbi path ends L24 at frame 256 and drops frame 257 from L24's window. This is the alignment change behind the +0.21 component. It is invisible in `letter:duration_s`, because L25's onset does not move.

---

## 5. Causal model: this unit, this level

| | Explanation | Verdict |
|---|---|---|
| A | Direct: the change remains with the alignment held fixed | **Supported** as the dominant pathway. In V1 and V4 the entire identity and makhraj response, and more than the entire ghonna response, is present under the held alignment. In V2, V3 and V5 the alignment component on L24 is exactly 0. |
| B | Alignment as the pathway | **Unsupported.** The alignment component is 0 for identity and makhraj, and opposes the target on ghonna. |
| C | Both | **Supported for V1 and V4, as a secondary component.** Re-alignment (frame 257 dropped) changes ghonna by +0.21 to +0.22 and six other heads by −0.21 to −0.65. So a small alignment-mediated part exists; it mostly adds collateral. |
| D | Different physical regions compared | **Confirmed and now resolved.** The placements differ physically in the edge windows, which the Hann metric cannot see. The unweighted ratio separates {V1, V4} at −20.3 dB from {V2, V3} at −18.4 to −18.5 dB, and Muqri groups the variants the same way. |

**The −20 ms asymmetry, explained:**

- V2 leaves [10.30, 10.32] unedited. The ن's frame-257 spike survives, so the response mostly disappears.
- Neither the kasra-tail edit (V4) nor the shifted WORLD context (V5) accounts for it.
- The identity and makhraj windows are the same under both alignments in every variant, because L22's start and L26's end do not move. So their responses are purely direct.

---

## 6. WORLD pass-through, separated

V5 is WORLD analysis and resynthesis of [9.16, 10.60] with no parameter changed. Its effects:

| Layer | Change |
|---|---|
| Physics | Negligible on the fixed L24 region (rect −0.11 dB). The transition-window band levels move by +0.43 and +0.16 dB. |
| Alignment | L23's sifat window moves from [231, 233] to [231, 234]. Two path boundaries change in total. |
| Muqri on L24 | ghonna −0.211, identity −0.437, makhraj:م +0.703, ل +0.567, ر +0.572. None exceeds 1 nat. Alignment component 0. |
| Elsewhere | 0 verdict flips |

**Separating the artifact from the nasal edit:**

- V5's L24 deltas are at most 0.70 nats.
- V1's held response is 1.80 / 3.85 / 7.60 nats on ghonna / identity / makhraj:م.
- The difference between V2 and V1 is 1.36 / 3.68 / 7.16 nats.

So the resynthesis footprint is an order of magnitude smaller than the effect under study. It is still not zero, and it does move alignment. That is why V5 exists as its own control instead of being assumed neutral.

---

## 7. Remaining confounds and limits

1. **Nasal content and the spike frame coincide.** The decisive window [10.30, 10.32] is both the physical nasal-to-vowel transition and the second half of the model frame that carries the ن's final CTC spike. This experiment shows the response is direct and that it hinges on that window. It does not show that the response is specific to *nasality*. A non-nasal spectral edit of equal size in the same window was not tested. That is a selectivity question, not an alignment question.
2. **V3 changes two things at once.** V3 differs from V1 in the edited transition window and also in its WORLD context end (10.60 vs 10.62). V5 bounds a shifted-context footprint at ≤0.70 nats, but V5 shifts both context ends, not the end alone.
3. **Alignment changes far from the edit.** In V1–V4, re-alignment moves L21 (a fatha) from frame 229 (9.16 s) to 190 (7.60 s), across the mid-word stop, 1.9 s before the edit. Its own measurements were not in the recorded view. Only verdict flips were checked, and there were 0. The posteriors are stored locally, so it can be quantified later without a model run.
4. **Collateral uses a fixed 1-nat line.** It is not the per-unit noise floors of the smoke run.
5. **Scope.** One unit, one reciter, one level, one run per variant. WORLD and the engine are deterministic here: the posteriors were identical on a re-run, and V1 and V2 reproduced the stored records.

---

## 8. Provenance

| Artifact set | Code state |
|---|---|
| Smoke records (`data/records.json`, 40 records) | 7bef873 with a **dirty** tree. See `ALIGNMENT_INVESTIGATION.md` §7. |
| This experiment | Commit `1ea5dd92b986a08a4a82010088b1b5245d39ccd5`, working tree **clean** (`git status --porcelain` empty at start). Script sha256[:16] `59a7fee386d5125d`. Run 2026-09-25 19:07:23–19:08:47Z, Python 3.11.16, CPU, on this VM (no Modal). |
| Model | `obadx/muaalem-model-v3_2`, loaded from the local HF cache. The records do not state a revision; the cache holds one snapshot, `01a1ef9f…`. |

**A cross-check the investigation could not make.** V1 and V2, run by the clean committed code, reproduce the dirty-tree smoke records `ghunnah/nasal|{"g_db": -12}` and `ghunnah/boundary-0.02|nasal` exactly on every numeric Muqri metric. For those two records, this confirms that the dirty tree's library code behaved as 1ea5dd9's does. It does not prove byte identity of the smoke-run harness.

---

## 9. Is another experiment necessary?

**Not for this question.** Direct versus alignment-mediated is resolved for this unit: mostly direct, with a small alignment-mediated component that mainly adds collateral.

Whether the frame-257 dependence is specific to nasality (item 1) is a separate selectivity question and would need its own experiment. Whether any of this generalises beyond one unit would need more units, which was out of scope here. Neither is required to answer the question posed.

---

## 10. Production safety

**Added:**

- `causal/align_exp.py`
- `results/alignment_experiment.json`
- this report
- local, gitignored: `data/align_exp/`

**Not touched:** `app/`, Sessions, rounds 1–5, Neo4j, authentication, TTS, and the other synthesis code. The `ctc_viterbi` rebinding exists only inside the experiment process and is restored after each scoring.

**Not done:** no Modal, no training, no sweep, no additional letters, no corpus.
