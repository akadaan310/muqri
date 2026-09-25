# Causal-experiment harness (EXPERIMENTAL research layer)

The question this harness answers:

> If one acoustic property of an authentic recitation unit is deliberately changed, does Muqri's corresponding measurement change in the expected direction, while unrelated measurements stay within their noise floor?

It wraps the existing engine (`app.engine.Engine`, unchanged) and the existing transforms (`letter_corpus/perturb.py`, `synthesis/world_lab.py`). It changes no engine behaviour, no session scoring, no round result and nothing in Neo4j. Nothing here is a production claim.

## Files

| File | Role |
|---|---|
| `schema.py` | The experiment record, noise floors, the classification of deltas, and physics verification. |
| `physics.py` | Independent measurements. The primary layer is Praat and numpy. The cross-checks are Octave (`qaari_features.m`) and Julia (`julia/xcheck.jl`). |
| `transforms.py` | duration, voicing, formant, f0 and nasal transforms, plus the no-op and swap controls, all built only from existing implementations. |
| `harness.py` | Per unit: baseline, controls, transform levels, the records, and the artifacts. `harness.py smoke` runs the smoke matrix. |
| `julia/xcheck.jl` | Julia's independent span measurements, and its independent re-derivation of every delta. |
| `results/` | Committed small artifacts: `causal_table.json`, `parity.json` and `stability.md`. |
| `data/` | Local and gitignored: the full records (`records.json`) and the altered audio clips. |

## One record keeps three things apart

- **`hypothesis`:** what we intended. The physical metric, its direction and predicted size, and the Muqri targets with their directions. Everything else is expected to stay.
- **`physics_*`, `delta_physics`, `physics_verification`:** what the transform actually did to the audio, measured without Muqri, as requested against measured.
- **`muqri_*`, `delta_muqri`, `classification`, `collateral`:** what Muqri then measured. Each metric is classified as *expected change*, *unexpected change*, *stable* or *insufficient evidence*, against a noise floor taken from the unit's own no-op controls.

A transform is never called successful because a parameter was passed to it. Its status comes from the measured physical delta: VERIFIED, PARTIAL, NOT_VERIFIED or UNMEASURABLE.

## Controls, per unit

| Control | Checks |
|---|---|
| determinism | The same audio run twice through the engine. |
| noop_splice | The original samples spliced back in. The harness itself must change nothing. |
| noop_world, noop_pv | The processing footprint of the WORLD vocoder and of the phase vocoder. These set the noise floor for their transform family. |
| boundary ±20 ms | The middle level of the first transform, applied to a span shifted by half a frame. |
| same-letter | The same reciter's same letter from another ayah. Nothing should move. |
| neighbour | A neighbouring letter. A positive control: identity must fall. |

## Reproduce

```
.venv/bin/python research_agency_lab/experiments/causal/harness.py smoke
```

This needs:

- the letter-corpus measurements (`letter_corpus/data/measure/Husary_128kbps/`), which choose the units
- the EveryAyah audio cache
- Octave with `pkg signal`
- `~/julia-1.11.5/bin/julia` with the `QaariLab` project

Each record states its code revision, whether the working tree was dirty, the audio SHA-256, the unit, the transform and its parameters, and the package versions. The smoke test is small by design: no significance is claimed from it.
