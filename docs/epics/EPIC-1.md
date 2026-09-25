# EPIC 1: The synthesis engine

*Carry-over and plan. Read the whole file before starting. Branch `claude/qaari-eval-engine-btwkjt`.*

The platform overview (no code needed) is `docs/synthesis-engine.md`. This file is the working plan.

---

## Goal

Build the system that can **synthesize any letter, characteristic and tajwīd rule, correct or with a known mistake of a known size, in any voice**, verified by physics and by the engine. The same system **turns plain Arabic TTS of a verse into recitation with every rule applied**.

It is for internal use: labelled data for measuring and training the engine, never published as recitation.

---

## Current state (carry-over)

### What exists and works

**Engine** (`app/`). Capabilities in `docs/synthesis-engine.md` §2. Recent changes:
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
| `perturb.py` | the lab: length stretches, echo off, devoicing, formant shifts, deletion, consonant-only and whole-syllable swaps, same-syllable controls; `--sweep` mode written, not yet run to completion |
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

### What was measured (the facts Sprint 1 builds on)

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

## Working rules

1. **Commit constantly.** Every commit is production-ready; push each commit to `origin/claude/qaari-eval-engine-btwkjt`. Other agents work on this branch: fetch and merge before pushing, and never force-push. Tests must pass: `.venv/bin/python -m pytest -p no:warnings -q`.
2. **Report only numbers you measured.** A negative result is a result: record it.
3. **Experiments run on the existing pilot corpus:** 5 masters, about 1,450 instances each, plus the learner. No full-Qur'an corpus in this sprint.
4. **Small, fast experiment loops run locally:** 5–6 letters at a time, one verse at a time, Julia and Octave for the maths and the signal processing.
5. **Anything larger runs on Modal,** on the largest CPU and GPU machines. Before launching, show the user what it is, how many items, which machines, and the estimated time and cost. Wait for approval.
6. **Every synthetic label is verified by physics,** with controls in every batch: pass-through (processing without an edit) and same-syllable splices. A transform counts as working only if its controls don't move.
7. **The engine measures; apps decide.** No advice text in engine output.
8. **Synthetic audio stays internal.**

---

## Sprint 1: The synthesis loop, end to end, on the pilot corpus

**Session goal:** one continuous sweep of experiments that builds and proves every part of the synthesis engine together:
- the physics layer
- clean, precise letter audio
- the delta calculus
- a transform for every characteristic and rule
- the labelled-data generator
- TTS voices turned into rule-applied recitation

All of it runs on the letters and rules already in the pilot corpus. Each part feeds the next inside the same session. Experiments iterate: try, measure, keep what moves the physics correctly, record what doesn't.

### User stories

- As the synthesis engine, I know every letter's true edges, so edits land on the letter and nowhere else.
- As the lab, I can measure every characteristic from the audio's physics, independently of the acoustic model, so synthetic labels are trustworthy.
- As a researcher, every characteristic is described by numbers (how the sound changes, on average and in its spread), so transforms are fitted and mistakes graded by size.
- As the data factory, I can take any letter (master, learner or TTS) and change one characteristic or one rule by a chosen amount, and receive the audio with a full label.
- As the data factory, I can take a TTS voice reading a verse and return it with every rule applied, verified, in that same voice.

### Work, as one experimental loop

**1. Clean, precise letter audio.**
- Refine every unit's edges from the 40 ms frame grid to about 10 ms: spectral flux, energy minima and voicing onsets (Praat). Check against letters heard by ear.
- Extend `clean.py` with:
  - per-clip quality gates (local SNR, clipping, bandwidth)
  - a hum notch when hum is detected
  - log-MMSE denoising as an option for learner audio
- Every cleaning step must leave the engine's margins on master clips unchanged. Measure that.

**2. The physics layer (`synthesis/physics.py`,** reusing the Octave feature engine `qaari_features.m` through its bridge). Per unit:
- voiced fraction and harmonics-to-noise ratio
- F1, F2, F3 trajectories: consonant, vowel onset, vowel midpoint
- nasal energy ratio
- closure duration and burst energy
- frication centroid, peak and spread
- rā' tap count and rate
- duration

Unit tests on synthetic signals. Run it over the pilot corpus.

**3. The delta calculus (`synthesis/deltas.jl`, Julia).** Minimal pairs from the pilot corpus:
- voicing: ز/س, ذ/ث, د/ت, ج/ش, ب/ف
- closure: د/ذ, ت/ث
- heaviness and itbāq: ت/ط, د/ض, س/ص, ذ/ظ, ك/ق
- nasality: ن/ل, م/ب
- heavy against light rā'

For each: per vowel context, the mean difference and variance of every physical quantity as a trajectory, and effect sizes. Output `synthesis/deltas.json`.

**4. Transforms, one per characteristic and rule (`synthesis/transforms.py`),** each with a dose, each fitted in Julia so the physics moves along its delta:
- voicing: aperiodicity, and pitch carried through the consonant; voicing through the closure for stops
- closure and burst insertion or removal
- heaviness and itbāq: a formant warp over the consonant and the vowel onset, following the measured trajectory delta
- ṣafīr and tafashshī: frication spectrum reshaping
- qalqalah: a release echo, from the same voice, added or removed
- takrīr: tap modulation
- istiṭālah: lateral frication lengthening
- ghunnah: nasal resonance plus hold
- madd and ghunnah lengths: time-scale modification, to target counts

**Where signal processing can't move a characteristic's physics, use syllable substitution** (the consonant and its transition from the same voice's neighbouring letter).

For each transform:
- test on 5–6 letters locally, with the physics before and after, the engine before and after, and the controls
- iterate
- then a sweep over the pilot corpus: locally if small, on Modal after review if large

**5. The generator (`synthesis/factory.py`).**
- **Input:** a clip or verse, plus a target (a characteristic or rule, a direction, a dose).
- **Output:** the audio plus a JSON label: what, where, how much, the physics before and after, the engine before and after.
- Produce a first labelled batch from the pilot corpus, with a manifest.
- From it, measure the engine's recall and false alarms per characteristic, from synthetic mistakes alone.

**6. TTS, tested throughout and taken to recitation.**
- Probe many edge-tts voices (32 Arabic voices: male, female, 16 dialects), plus one open model (e.g. Meta MMS-TTS), on several verses. Keep the clearest.
- **Each TTS voice is also a syllable library:** it speaks any letter form on demand, as donor material for substitution and qalqalah echoes.
- Apply the rules to the TTS reading, using the transforms above:
  - tempo to tartīl
  - madds and ghunnahs to their counts
  - qalqalah
  - heavy lām and rā'
  - forms at a stop
- Measure with the engine and the physics, and give the user listening files.
- Then use the generator to put controlled mistakes into the TTS recitations.

### Done when

- Boundaries are refined and checked, and cleaning is shown not to move master margins.
- The physics layer is tested and run on the pilot corpus.
- `deltas.json` covers every pair listed.
- For every characteristic and rule, there is a transform (or a substitution) with measured dose-response, physics landing in the target's natural range, and controls unchanged. The share of instances that succeed is reported per characteristic, including those that don't.
- The generator has produced a labelled batch with a manifest, and the engine's recall and false-alarm table per characteristic comes from it.
- TTS verses in several voices are rendered with every rule applied, with the engine's rule measurements and the listening files delivered.
- Everything is committed. `docs/synthesis-engine.md` is updated with what was proven, and this file with the state for the next session.
