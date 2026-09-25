# Nasality Specificity Experiment

This is a single-unit specificity control on the decisive window, run once. It is not proof of anything.

| Item | Location |
|---|---|
| Script | `causal/nasality_exp.py` |
| Machine-readable results | `results/nasality_experiment.json` |
| Local, gitignored data | `data/nasality_exp/`: WAVs and posteriors |

**Scope.** One source unit: Husary, 11:2, letter L24 ن, ghunnah mushaddadah. One 20 ms window, one level, one run per variant. No generalisation, no significance, no production claim.

---

## Answer

**Does the non-nasal control produce a comparable ghunnah response? No.**

Each variant is measured against B: the same interior nasal edit, with the window resynthesised but not edited.

| Variant | ghonna Δ | identity Δ | makhraj:م Δ |
|---|---|---|---|
| N: nasal −12 dB in the window | **−1.297** (−1.503 with the alignment held) | **−1.467** | **−7.753** |
| C1: same shape and magnitude, moved to 2000/2650 Hz | +0.016 | +0.515 | +0.173 |
| C2: flat gain matched to N's window energy change | −0.041 | +0.038 | −0.683 |

Neither control moves any L24 metric by more than 1 nat. Neither removes the ن's final CTC spike, and neither changes the alignment.

For this unit, this **supports** the conclusion that the response depends on the nasal-band content of the transition window, not on a generic alteration of it. What it does **not** establish is that the relevant dimension is *nasality* as opposed to low-frequency / F1-region spectral balance. The nasal edit changes both (see Unresolved).

---

## 1. Design

**Source.** `everyayah:Husary_128kbps/011002.mp3`. The decoded waveform's sha256[:16] is `ae1eec5780e0e76e`, the same audio as the alignment experiment.

**What every edited variant shares:**

- one WORLD pass over the context [9.18, 10.62] s: span ±0.3 s, `world_lab.world` → `synth`, then `perturb.splice` with 10 ms crossfades
- the same interior edit: `world_lab.nasal`, g = −12 dB on [9.48, 10.30]

The edited variants differ **only** in the four 5 ms WORLD frames of the decisive window [10.30, 10.32].

| Variant | Window edit | Implementation | Role |
|---|---|---|---|
| V0 | none; untouched ayah | — | reference |
| B | none; window resynthesised, unedited | `world_lab.nasal` on the interior mask only | **no-op control**; the reference for every contrast below |
| N | nasal, g = −12 dB (pole 250 Hz, anti-resonance 900 Hz) | `world_lab.nasal` on the interior+window mask | intended perturbation; **bit-identical** to `transforms.apply(nasal, 9.48, 10.32)`, i.e. the alignment experiment's V1 (asserted) |
| C1 | same expression and g = −12 dB, both centres moved +1750 Hz (2000 / 2650 Hz), widths unchanged | `nasality_exp.off_band` (`world_lab.nasal`'s formula) | non-nasal control, **shape-matched** |
| C2 | flat envelope gain g2 = +1.48 dB, set to N's measured window RMS change against B before C2 was synthesised | `nasality_exp.flat_gain` | non-nasal control, **energy-matched** |
| W | none; plain WORLD pass-through of the span context | `transforms.apply(noop_world, 9.48, 10.32)` | WORLD footprint, reported separately |

**No new transform family was introduced.** C1 is `world_lab.nasal`'s own expression with other centre frequencies. C2 is the same envelope multiplication with a constant shape.

**Muqri.** Each variant was scored with the unmodified `Engine.analyze` (`makhraj=True`), and again with V0's Viterbi path held fixed through `align_exp.viterbi` (in-process only; no file under `app/` changed).

**Matching criteria**, declared in the script before the run. Each is for the window, against B:

| Criterion | Test |
|---|---|
| Energy | \|ΔRMS − N's ΔRMS\| ≤ 0.5 dB |
| Shape | mean \|ΔdB\| over 100 Hz bands (100–4000 Hz) within 0.67–1.5× N's |
| Non-nasal | \|Δ nasal ratio\| ≤ 1 dB and \|Δ\| of each nasal band ≤ 1.5 dB |
| Location, duration, boundary, resynthesis context | identical by construction |

---

## 2. Main comparison

All values are for the window [10.30, 10.32] against B (Δ), except the last three columns (nats, independent alignment). Collateral counts non-target L24 metrics with |Δ| > 1 nat.

| Variant | Intended change | Actual nasal change: ratio / 150–400 / 750–1100 Hz (dB) | Actual non-nasal change | Ghunnah Δ | Identity Δ | Makhraj Δ (م / ل / ر) | Collateral |
|---|---|---|---|---|---|---|---|
| N | nasal −12 dB | **−19.94 / −11.93 / +11.03** | RMS +1.48 dB; 400–750 Hz +3.17; 1100–1500 +0.90; Praat F1 +115.6 Hz; HNR +2.63 dB | **−1.297** (held −1.503) | **−1.467** | **−7.753 / −6.973 / −7.296** | 8 (independent) / 7 (held): identity, makhraj ×3, shidda, tafashie, tikraar; istitala in independent only |
| C1 | same shape and magnitude at 2000/2650 Hz | +0.08 / 0.00 / −0.01 | RMS **+2.23** dB; 2000–2400 −3.23, 2400–3000 +8.22; Praat F3 −100.6 Hz, F2 −26.9 Hz | +0.016 | +0.515 | +0.173 / −0.250 / +0.069 | 0 |
| C2 | flat +1.48 dB | +0.17 / +1.48 / +1.48 | RMS +1.35 dB; every band +1.46 to +1.48 (0–150: +3.47) | −0.041 | +0.038 | −0.683 / −0.295 / −0.598 | 0 |
| B | none (no-op) | 0 | 0 | 0 | 0 | 0 | 0 |

**How well the controls matched, by the declared criteria:**

| Control | Energy | Shape | Non-nasal |
|---|---|---|---|
| C1 | **Not matched.** +2.23 vs N's +1.48 dB, over by 0.75 dB, i.e. a *larger* change than N's | Matched: ratio 0.92 | Yes |
| C2 | Matched: +1.35 vs +1.48 | Passes the criterion (ratio 0.74) | Yes |

C2's shape pass is not meaningful. The declared metric averages over 39 bands, which dilutes N's localised ±12 dB edit. N's largest band change is 11.9 dB; C2's is 1.5 dB (3.5 dB below 150 Hz). **C2 is treated as energy-matched only.**

No control matches N on both energy and shape. Each non-nasal control matches it on one, and C1 exceeds it in energy.

---

## 3. Observed

These are direct outputs of this run: Muqri values from the unmodified engine.

- **Muqri values**, independent alignment. The held alignment gives identical values for every variant except N:

  | Variant | ghonna | identity | makhraj:م |
  |---|---|---|---|
  | V0 | 11.278 | 44.201 | 45.576 |
  | B | 10.983 | 41.821 | 45.730 |
  | N | 9.686 (held 9.480) | 40.354 | 37.977 |
  | C1 | 10.999 | 42.336 | 45.903 |
  | C2 | 10.942 | 41.859 | 45.047 |
  | W | 11.320 | 43.982 | 46.918 |

- **L24's duration and the ghunnah counts** were unchanged in every variant (0.84 s, 2.49).
- **Alignment.** Only N moves L24's window, from frames [237, 258) to [237, 257). Every edited variant, including W, moves L20, L21 and L23 relative to V0. There were no verdict flips elsewhere in the ayah, under either alignment.
- **Frame 257 (10.28–10.32 s)** carries the ن's final CTC spike:

  | Variant | log p(ن) | ghonna [مغن] |
  |---|---|---|
  | B | −0.006 | −0.002 |
  | **N** | **−3.810** (blank −0.085) | **−1.846** |
  | C1 | −0.003 | −0.001 |
  | C2 | −0.012 | −0.003 |
  | W | −0.000 | −0.000 |

---

## 4. Independently verified

These are the physics results: Praat and numpy, not Muqri.

- **N changes what it was meant to change.** In the window, the nasal ratio moves −19.94 dB, with the 150–400 Hz band −11.93 and the 750–1100 Hz band +11.03. It also changes things it was not meant to:
  - 400–750 Hz: +3.17 dB
  - 1100–1500 Hz: +0.90 dB
  - RMS: +1.48 dB
  - Praat F1: 735.6 → 851.2 Hz
  - HNR: 11.21 → 13.84 dB

  F0 is unchanged (228.33 vs 228.38 Hz) and the window stays fully voiced.
- **C1 leaves the nasal bands untouched** (≤0.01 dB) and the nasal ratio at +0.08 dB. It changes 2000–2400 Hz by −3.23 and 2400–3000 Hz by +8.22 dB. F3 moves 2857.9 → 2757.3 Hz, and RMS rises by +2.23 dB. F0, voicing and HNR are essentially unchanged (HNR −0.34).
- **C2 raises every band by about 1.47 dB** (0–150 Hz: +3.47). The nasal ratio moves +0.17 dB and RMS +1.35 dB. F0, formants (±10 Hz) and HNR (−0.14) are essentially unchanged.
- **The edits are confined to the window.** The interior [9.48, 10.30] is identical across B, N, C1 and C2 (ΔRMS ≤ 0.02 dB). WORLD's frame smoothing leaks into the preceding 20 ms: RMS +0.09 (N), +0.13 (C1), +0.57 dB (C2). The following 20 ms changes by ≤ 0.07 dB. In every variant the changed samples span exactly the WORLD context, [9.18, 10.62].
- **N reproduces the earlier run.** N is bit-identical to the stored V1 construction, and its independent-alignment Muqri values equal the stored V1 record (ghonna 9.686, identity 40.354, makhraj:م 37.977).

---

## 5. Derived

These are computed from the values above.

- **Window-specific response.** N − B = −1.297 ghonna, −1.467 identity, −7.753 makhraj:م. For the controls, |C − B| ≤ 0.683 on every L24 metric (largest: C2's makhraj:م −0.683; C1's identity +0.515).
- **The response is direct.** With the alignment held, N − B is −1.503 on ghonna: the full response, as in the alignment experiment. Re-alignment is +0.206 on ghonna. The controls have zero alignment component.
- **WORLD footprint.** W against V0 gives makhraj:م +1.342, ل +1.260, ر +1.226, identity −0.219, ghonna +0.042. That reproduces the smoke record `ghunnah/noop_world` exactly. Physically, W adds low-frequency energy in the window (0–150 Hz); so do all WORLD variants against V0.
- **Sensitivity to the WORLD context boundary** (a cross-experiment comparison; both runs are deterministic and from clean commits). B and the alignment experiment's V3 differ only in where the WORLD context ends, 10.62 vs 10.60 s: the same interior edit, and an unedited window. Their identity differs by 1.37 nats (41.821 vs 43.191) and makhraj:م by 1.00 (45.730 vs 44.730).
  - In this experiment the context is identical across all contrasts, so this cancels.
  - It does mean that V3's "−1.01 identity" in the alignment report partly reflects the context position.

---

## 6. Inferred

- **Frame 257 is a threshold.** For this unit, the ن's final CTC spike survives a 2000–3000 Hz reshaping larger in energy than the nasal edit, and a broadband energy change of the same size. It collapses under the nasal-band edit. Muqri's response (ghonna, identity, makhraj) follows the spike. The response is therefore selective for the edit's low-frequency, nasal-band content, and not triggered by alteration of the window as such.
- **Reading against the stated criteria:** "nasal perturbation changes ghunnah while matched non-nasal controls do not" → this **supports specificity to the nasal-band dimension for this unit**. The matching caveats in §2 apply.

---

## 7. Unresolved

1. **Nasality versus F1-region balance.** The −12 dB nasal edit is also an F1-region edit: 400–750 Hz +3.17 dB, Praat F1 +116 Hz, HNR +2.6 dB, in a window where the fatha's F1 sits near 735 Hz. The 750–1100 Hz anti-resonance band overlaps the vowel's F1 range. These experiments cannot tell whether Muqri responds to the nasal murmur (pole and anti-resonance) or to low-frequency / F1 spectral balance. No control available in the existing transforms separates the two without also touching the nasal bands.
2. **Imperfect matching.** C1 is 0.75 dB over N in energy. C2 matches energy but is spectrally much weaker (peak 1.5 vs 11.9 dB). No single control matches N on both energy and shape. The null result holds for each control against the criterion it does match, and C1's excess energy goes against the null (a larger change, still no response).
3. **The finding is conditional on the interior edit.** Every variant carries the interior nasal edit. Whether a window-only nasal edit, with the rest of the ن untouched, produces the response was not tested.
4. **Size of the test.** One unit, one reciter, one window, one level, one run per variant. Collateral uses a fixed 1-nat line, not the per-unit noise floors.
5. **WORLD is not neutral.** W moves makhraj by about 1.3 nats and shifts neighbouring-letter alignment. The context-boundary sensitivity in §5 shows that the footprint depends on where the context ends. That is controlled here by construction, but it limits comparisons across experiments with different contexts.

---

## 8. Provenance

| Artifact set | Code state |
|---|---|
| Smoke records (`data/records.json`) | 7bef873, **dirty** tree (`ALIGNMENT_INVESTIGATION.md` §7) |
| Alignment experiment | Run at 1ea5dd9, clean; results 1a6a560 |
| This experiment | Starting commit `1a6a560` (clean). Experiment commit `d2eafbb769514766ecdc345e39f1fc886e2e311d`, working tree **clean** at run start (`git status --porcelain` empty). Script sha256[:16] `1acd270a2ff6b313`. Run 2026-09-25 19:46:17–19:48:08Z, Python 3.11.16, CPU, this VM. |
| Model | `obadx/muaalem-model-v3_2` from the local HF cache (revision not recorded; single cached snapshot `01a1ef9f…`) |
| Transform code used | `synthesis/world_lab.py` (`world`, `synth`, `nasal`, `FP` = 5 ms); `letter_corpus/perturb.py::splice`; `causal/transforms.py::apply` (for W and the N identity check); `nasality_exp.off_band` and `flat_gain` |

---

## 9. Production safety

**Added:**

- `causal/nasality_exp.py`
- `results/nasality_experiment.json`
- this report
- local, gitignored: `data/nasality_exp/`

**Not touched:** `app/`, Sessions, auth, Neo4j, round results, TTS, and the other synthesis code.

**Not done:** no Modal, no training, no sweep, no other letters, no corpus, no agents.
