# EPIC 1: The synthesis engine (from real letters and TTS voices to fully applied recitation)

*Carry-over and plan. Each sprint below is one dedicated AI session. Read the whole file before starting any sprint. State as of 2026-09-25, branch `claude/qaari-eval-engine-btwkjt`, head 9babeea or later.*

For the consultant-level overview (no code needed), read `docs/synthesis-engine.md`. This file is the working plan.

---

## Goal

Build the system that can **synthesize any letter, characteristic and tajwīd rule, correct or with a known mistake of a known size, in any voice**, verified by physics and by the engine. Then use it to **turn plain Arabic TTS of a verse into a full recitation with every rule applied**.

The purpose is internal: unlimited labelled data for measuring and training the engine. It is never for publishing as recitation.

**Done means all of these hold, measured:**

1. **A physics verification layer.** Every letter unit can be measured independently of the acoustic model: voicing, formant trajectories, nasal murmur, burst and closure, frication spectrum (sibilant peak, spread), tap rate, duration. Boundaries are refined to about 10 ms.
2. **The delta calculus.** For every minimal pair of letters (differing in one characteristic), across at least the 5 pilot masters: the mean difference and the variance of every physical quantity, as trajectories over consonant plus vowel transition. It is stored as data, not prose.
3. **Characteristic synthesis.** For each of the 10 measured characteristics (hams/jahr, shiddah/rakhāwah/tawassuṭ, tafkhīm/tarqīq, itbāq, ṣafīr, qalqalah, takrīr, tafashshī, istiṭālah, ghunnah), a transform with a dose parameter satisfying all three conditions:
   - moves the physics of ≥ 80% of test instances into the natural range of the target class
   - leaves same-letter controls and pass-through controls unchanged
   - is reported with its measured effect on the engine
   Where the current model doesn't respond, that is recorded, not hidden.
4. **Rule synthesis.**
   - every madd type and ghunnah re-timed to a target count
   - qalqalah added or removed
   - waqf forms rendered
   - all verified by the engine's rule measurements
5. **TTS to recitation.** For at least 3 voices (male and female, two dialects) and at least 20 verses (al-Fātiḥah plus juz' 30 samples):
   - ≥ 95% of the engine's located rules measured in band
   - every consonant and vowel heard as itself
   - a listening file per verse
6. **Scale.**
   - the letter/rule corpus for the 5 pilot masters over the full Qur'an
   - the sweep (every letter × neighbours, with controls), with recall and false-alarm per characteristic
   - a data-factory run that emits labelled synthetic clips with a manifest
   All run on Modal after review.

---

## Current state (carry-over)

### What exists and works

**Engine** (`app/`). It is measured in `docs/synthesis-engine.md` §2:
- 49/61 scripted mistakes caught
- 156/160 take-A expectations met
- master validation clean on round 5

Changes this phase:
- **Joint alignment for short submissions** (`app/submission.py::walk_alignment`)
- **Clip context from pauses** (`app/engine.py::_with_context`)
- **An iẓhār ḥalqī hold limit** of 2.5 counts, calibrated on 804 master instances
- **Free-text drills:** `Engine.reference_text`, and `analyze()` accepts strings (surah 0, line 1..n)
- **The makhraj test:** `app/letters.py` NEIGHBOURS; `makhraj=True`, on by default for drills
- **Full declared profiles:** `app/letters.py` has 17 makharij and 17 ṣifāt per letter, measured or not

**Letter and rule corpus** (`research_agency_lab/experiments/letter_corpus/`):

| File | What it does |
|---|---|
| `plan.py` → `plan.json` | 300 cells; 363 short ayahs hold 5 instances each; 10 cells short |
| `measure.py` | local measuring, and `--sessions` for the learner's takes |
| `research_agency_lab/compute_bridge/modal_corpus.py` | measuring on Modal: 10 containers × 16 CPU × 32 GiB × 8 engine processes; 1,562 ayahs in 183 s, $0.52 |
| `assemble.py` | cuts, cleans, scores and ranks; writes `data/corpus.json` and `data/clips/` |
| `clean.py` | DC, 60 Hz high-pass, spectral gating, zero-cross cuts, 5 ms fades, −20 dBFS |
| `perturb.py` | the lab: length stretches, echo off, devoicing, formant shifts, deletion, consonant-only and whole-syllable swaps, same-syllable controls; `--sweep` mode written and not yet run to completion (it stopped at 50 of 291 when moved to Modal) |
| `DESIGN.md` | design, cleaning techniques, the perturbation catalogue, the data factory |

- **Pilot assembled:** Husary 1,452 instances, Minshawy 1,450, Abdul Basit 1,447, Alafasy 1,449, Sudais 1,449, the learner 611; 298 cells; 292 MB of clips.
- **Data stays local and is gitignored:** `letter_corpus/data/` (measure/, clips/, corpus.json, perturb.json).

**Synthesis lab** (`research_agency_lab/experiments/synthesis/`):
- `world_lab.py`: WORLD-vocoder transforms per characteristic (devoice, voice, formant warp, nasal pole/zero) with pass-through controls. Results: `world_lab_voicing.json`, `world_lab_weight_nasal.json`.
- `tts_probe.py`: edge-tts voices measured by the engine. Results: `tts_probe_112.json`; audio in `synthesis/data/tts/` (gitignored).

**Web app** (`python -m app.webapp`, port 8088, public at `http://40.64.120.87:8088`):
- `/letters`: the corpus (letters, rules, who reads best, perturbation lab)
- `/sessions`: calibration rounds 1–6; round 6 is the letter drill, unrecorded and paused by decision
- `/review`: listening labels

To restart it, find the PID with `pgrep -f "^.venv/bin/python -m app.webapp"`, `kill <pid>`, then run `(setsid nohup .venv/bin/python -m app.webapp > /tmp/webapp.log 2>&1 &)`. **Never** use `pkill -f` with a pattern that also appears in your own command line: it kills your shell.

### What was measured (the facts the sprints build on)

- **Lengths are synthesizable.** A phase-vocoder stretch of a madd or ghunnah span moves the engine's counts as intended:
  - ghunnah 3.07 → 1.54 at ×0.4 (short); → 5.00 at ×1.8 (long)
  - madd ṭabīʿī 2.06 → 3.43 at ×2 (long)
- **Whole-syllable swaps flip exactly the differing characteristics:** طَ→تَ, ضَ→دَ, صَ→سَ (ṣafīr kept), قَ→كَ.
- **Consonant-only DSP edits don't flip the model:**
  - devoicing (Praat: voiced 94–100% → 0–6%) left hams/jahr unchanged in 6/6 letters
  - formant warps moved margins monotonically and flipped only ص
  - nasal pole/zero moved ghonna modestly
- **Pass-through vocoder controls** moved heads by about ±1: the vocoder is clean.
- **Conclusion:** the model's characteristic heads largely encode the letter's identity (it learned them from text-derived labels). Verification of synthetic characteristics must be physics-based, and the synthetic data is what can teach a future model to hear the characteristics themselves.
- **TTS (4 of 32 edge-tts voices, al-Ikhlāṣ):**
  - consonants and vowels heard as themselves in 3 of 4 verses in every voice
  - missing in all voices: qalqalah, madd length (1.2–1.6 counts against 2), the heavy lām of the Name
  - 112:2 read as prose
  - pace 0.16–0.24 s per count

### Tools on the VM

| Tool | Notes |
|---|---|
| Python venv `.venv` | torch 2.14 CPU, transformers 5.17, quran-transcript 0.6.1, quran-muaalem 0.2.2, librosa 0.11, scipy 1.17, soundfile, praat-parselmouth 0.4.7, nara-wpe, **pyworld** (needs `setuptools<81`), **edge-tts** |
| Julia | `~/julia-1.11.5/bin/julia` (the `julia` on PATH, 1.10, stalls) |
| Octave | `/usr/bin/octave`; the feature engine is `research_agency_lab/substrate_library/octave/qaari_features.m` with its bridge `research_agency_lab/compute_bridge/octave_bridge.py` |
| Modal CLI | 1.5.5. Profile `akadaan310` is nearly spent ($25.77 of $30 this month, free tier, 10 containers). **The user has a second account with more budget.** Ask for it at the start of the first Modal job (`modal token set` or `modal profile activate`) |

### Open issues

- Boundaries are 40 ms frames. Some letters get one frame (final letters at a stop, the qalqalah echo). Refinement is sprint 1's first job.
- The model's heads can't verify most characteristic syntheses (see above). Physics must.
- Idghām without ghunnah with a nasal hum added, a sākin emphatic made light, and the last letter before a stop are not caught by the engine. They are synthesis targets, and later detector targets.
- Rights: master audio is for research; synthetic audio is internal only.

---

## Working rules for every sprint

1. **Commit constantly; every commit production-ready; push each commit** to `origin/claude/qaari-eval-engine-btwkjt`. Other agents work on this branch: fetch and merge before pushing; never force-push. Tests must pass (`.venv/bin/python -m pytest -p no:warnings -q`).
2. **Report only numbers you measured.** State misses and false alarms as measured. A negative result is a result: record it.
3. **Heavy compute on Modal only**, on the largest machines (16+ CPU, 32+ GiB; GPU where a model runs). **Before any larger Modal job, show the user the job:** what, how many items, machines, estimated time and cost. Wait for approval. Small experiments (5–6 letters, one verse) run locally; so does math in Julia or Octave.
4. **The engine measures; apps decide.** No advice text in engine output.
5. **Verify every synthetic label by physics,** and include controls in every batch: pass-through (vocoder only) and same-syllable splices. A transform is only "working" if its controls don't move.
6. **Synthetic audio never leaves the research store** and is never presented as a reciter's recitation.

---

## Sprint 1: Physics, boundaries and the delta calculus

**Session goal:** make every letter measurable independently of the model, at about 10 ms precision, and compute what each characteristic *is*, physically, across 5 masters.

**User stories:**
- As the synthesis engine, I need to know a letter's true start and end, so that the edits land on the letter and not its neighbours.
- As a researcher, I need each characteristic described by numbers (mean change and spread), so that transforms can be fitted and synthetic mistakes graded by size.
- As the lab, I need a verdict independent of the model for every characteristic, so that synthetic labels are trustworthy.

**Tasks:**
1. **Boundary refinement** (`synthesis/boundaries.py`). Refine each unit's edges within ±40 ms to spectral-flux peaks, energy minima and voicing onsets (Praat). Validate on 50 hand-checked letters from the pilot, using the reviewer's ear via `/review`. Report the median shift and how many single-frame letters gain real duration.
2. **The physics layer** (`synthesis/physics.py`, reusing `qaari_features.m` through the Octave bridge where it already computes the quantity). Per unit:
   - voiced fraction and harmonics-to-noise ratio
   - F1, F2, F3 trajectories: consonant, vowel onset, vowel midpoint
   - nasal energy ratio (150–400 Hz against 750–1,100 Hz)
   - closure duration and burst energy
   - frication centroid, peak and spread (sibilant 4–8 kHz)
   - tap count and rate for rā'
   - duration
   Unit tests on synthetic signals (a pure tone is voiced; white noise is not).
3. **The delta calculus** (`synthesis/deltas.jl`, Julia). For the minimal pairs:
   - voicing: ز/س, ذ/ث, د/ت, ج/ش, ب/ف
   - stop against flow: د/ذ, ت/ث, ب/ف
   - itbāq and heaviness: ت/ط, د/ض, س/ص, ذ/ظ, ك/ق
   - nasality: ن/ل, م/ب
   - rā' heavy against light
   Take every corpus instance (5 masters × 5 instances × each vowel form). Compute each physical quantity's mean difference, variance and effect size, per vowel context. Output `synthesis/deltas.json` plus a short table.
4. **Modal job (review first): the sweep.** `perturb.py --sweep` for the 5 masters, about 1,450 engine runs: every letter × its neighbours as whole-syllable swaps, with same-letter controls. Estimate about 4 minutes and about $0.50 on 10 × 16 CPU. Output: recall and false-alarm per model head, and makhraj flips per pair.
5. **Modal job (review first): the full-Qur'an corpus** for the 5 masters. 6,236 × 5 = 31,180 ayahs. The pilot's rate (1,562 short ayahs of 36 phonemes in 183 s, $0.52), scaled by the Qur'an's average length (about 104 phonemes, 2.9×), gives about 3 hours on 10 × 16 CPU and about $30. Output Parquet: the per-unit index, plus the physics layer where cheap.

**Acceptance:**
- the boundary validation reported
- the physics layer tested and run on the full pilot
- `deltas.json` with every pair listed above
- the sweep table committed
- the full-Qur'an index on Modal storage, with its size and cost reported

## Sprint 2: Characteristic and rule synthesis, verified

**Session goal:** a transform for every characteristic and every rule, fitted to the deltas, verified by physics with controls, and packaged as a generator of labelled synthetic clips.

**User stories:**
- As the data factory, I can take any clean master or TTS letter and produce it with one characteristic changed by a chosen dose, with a label saying exactly what changed.
- As the evaluator, I can measure the engine's recall and false-alarm per characteristic from synthetic mistakes alone.

**Tasks:**
1. **The syllable library.** Per voice (master or TTS), the donor syllables for every letter × vowel form, from the corpus. For TTS voices, generate the missing ones on demand (the voice speaks بَ, ṭā' with fatḥah, …).
2. **Transforms per characteristic** (`synthesis/transforms.py`), each with a dose, each fitted in Julia to move the physics along its delta:
   - voicing (aperiodicity plus f0 continuation; for stops, voicing through the closure)
   - stop against flow (insert or remove closure silence and burst; continue frication)
   - heaviness and itbāq (time-varying formant warp over consonant plus vowel onset, fitted to the ṭ–t F2 trajectory delta)
   - ṣafīr and tafashshī (frication spectrum reshaping)
   - qalqalah (echo from the same voice's release, or removed)
   - takrīr (tap modulation)
   - istiṭālah (lateral frication lengthening)
   - ghunnah (nasal pole/zero plus hold)
   - madd and ghunnah length (stretch; already proven)
   Where DSP fails the physics test, fall back to syllable substitution, which is proven.
3. **Dose-response and controls.** For each transform on 5–6 letters locally, then at scale on Modal (review first): the physics before and after, the engine before and after, and the controls.
4. **The generator** (`synthesis/factory.py`). Input: a clip or a verse, a target (characteristic or rule, direction, dose). Output: audio plus a JSON label (what, where, how much, physics before and after, engine before and after). A Modal job (review first) emits the first batch, e.g. 10,000 labelled clips, with a manifest.

**Acceptance:**
- for each of the 10 characteristics: the fraction of instances whose physics lands in the target class's natural range, with controls unchanged (target ≥ 80%, reported per characteristic, including those that miss)
- for rules: every madd type and ghunnah at target counts within ±0.25 as measured by the engine; qalqalah added and removed
- the first labelled batch and its manifest
- the engine's recall and false-alarm table from synthetic mistakes

## Sprint 3: TTS to full recitation

**Session goal:** plain Arabic TTS of a verse, turned into a full recitation with every rule applied and verified, in several voices.

**User stories:**
- As the data factory, I can render any verse in any of 32+ voices as a recitation with every rule applied, at tartīl or tadwīr pace, with a full label of every rule and its realised value.
- As a researcher, I can then introduce any mistake at any size into that rendering (sprint 2's generator), in any voice.

**Tasks:**
1. **Text:** the verse fully vowelled. The engine's parser gives every rule, its target (counts, heavy or light, echo, stop form), and the waqf form at verse end.
2. **Voice:** edge-tts (32 Arabic voices, free; generation throttled), plus one open model (e.g. Meta MMS-TTS Arabic) as a fallback. Probe the voices first and keep the clearest (all consonants and vowels confirmed).
3. **Alignment** of the TTS audio by the engine, with boundaries refined (sprint 1).
4. **Rule application,** in order:
   - global tempo to tartīl
   - madds and ghunnahs to counts
   - qalqalah echoes
   - heavy lām and rā' by syllable substitution or formant warp
   - waqf forms
   - optional: the voice spoken more slowly by the TTS itself (rate), before editing
5. **Verification:** the engine measures the rendering, the physics layer checks each characteristic edit, and a listening file goes to the user.
6. **Scale (Modal, review first):** al-Fātiḥah plus a juz' 30 sample × at least 3 voices. Report per verse and voice: rules in band, consonants and vowels confirmed.
7. **Optional:** voice conversion (kNN-VC or Seed-VC) of a master's recitation into several timbres, keeping every rule, for speaker diversity.

**Acceptance:**
- at least 20 verses × at least 3 voices
- ≥ 95% of located rules measured in band
- every consonant and vowel confirmed
- listening files delivered
- the pipeline documented in `docs/synthesis-engine.md`

---

## Where this stands against everything else

This is our assessment, not a formal market survey. Publicly available work for Qur'an recitation covers:
- word-level mistake detection for memorisation
- phoneme-level mispronunciation research on read Arabic
- open acoustic models, including muaalem, which this engine builds on

What this project already has, measured, that we have not seen combined anywhere:
- **Letter-level measurement:** identity, articulation point (all 28 letters against their neighbouring points) and 10 characteristics, with margins.
- **Rule lengths in calibrated counts at the reader's own tempo,** with bands learned from master reciters and validated on their peers (reference 71 → 95).
- **A detection record against a certified reader's deliberate mistakes:** 49/61 caught, 156/160 expectations met, master validation clean.
- **A per-letter, per-rule corpus of master recitation,** built on the cloud in minutes and ranking reciters letter by letter.
- **A synthesis laboratory** that already produces measured, labelled length mistakes and letter/characteristic mistakes (whole-syllable swaps) from real audio. It has also established, with physics, what the current model does and does not hear, which is the specification for the next model.

When this epic is done, the project can generate unlimited, precisely labelled recitation data in any voice, and measure any engine with it. That is the foundation for a model that hears the characteristics themselves, not only the letters.
