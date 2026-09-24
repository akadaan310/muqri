# Sprint 4 — score the world's recitations, and serve the consumer apps

## Paste this to start the next session

> Continuing the qaari/muqri engine. Same VM (koda-vm), repo `~/github-director/repos/muqri`, branch
> `claude/qaari-eval-engine-btwkjt`. **Read `SPRINT4.md` in the repo root first** — it is the complete
> handover: what is built and measured, the architecture and key files, the rigor rules, the traps
> that cost time, and this sprint's goals.
>
> This is **Sprint 4**. The engine is the brain of consumer apps: for every recording it must return
> the most accurate, most computable measurement report — what was right, what was wrong, how right,
> how wrong — as numbers the apps use to decide priority, severity and learning paths. The engine
> measures; the apps decide what to tell the learner.
>
> Main initiative: **gather as many Quran recitations from online datasets as possible** — official
> or not, reciters, students, learners, anyone — **score them with the engine**, find where it fails,
> and improve it for the consumer apps. Start with the open datasets listed in SPRINT4.md, run them
> through the engine on Modal, and report per dataset what the engine gets right and wrong.
>
> Standing rules: commit constantly, every commit production-ready; numerics in Julia with the numpy
> path pinned by parity test; never report a number not measured; verify against the corpus rather
> than memory; validate any model against a baseline before it becomes a claim; push only on my
> explicit approval. Check `modal app list` and the Modal budget before launching anything.

---

## Where things stand (end of Sprint 3, 2026-09-24)

Tag **`stage-2-engine-brain`** marks the checkpoint; two commits follow it (the knowledge report and
the measurement contract). Nothing is pushed.

### The serving engine — `app/engine.py` → `Engine().analyze(audio, verses)`

Every report carries, besides the per-ayah letters/rules/words:

| section | what | file |
|---|---|---|
| `measurements` (schema `measurements/1`) | **the consumer contract**: numbers only. Recording (seconds per count, tempo class, wajh, basmala, stops); words (all_correct, ids of failing measurements); rules (expected range, observed counts/seconds, signed deviation, status, scored, percentile among masters and cohort, robust `z_masters`); letters (timing, identity competitor + margin, per characteristic expected/observed/margin/realised/scored/percentiles/z) | `app/measurements.py` |
| `knowledge` (schema `knowledge/1`) | learner skill posterior over 45 skills (35 rules, 9 characteristic heads, letter identity) with basis measured / inferred / prior, 90 % interval, gap to masters; the learner's own madd lengths vs masters; whole-Quran projection; knowledge-space learning path. `/analyze` takes an optional learner id and accumulates history | `app/knowledge.py`, `app/learner.py` |
| `basmala` | verse-1 recordings: tests "basmala + ayah" vs "ayah" on the audio, cuts it off by joint alignment | `app/engine.py` |
| `wajh` | munfasil / silah kubra graded against the reciter's inferred wajh: qasr (Tayyibah, ~2) or tawassut (Shatibiyyah, 4–5) | `app/engine.py` |

Reference distributions for percentiles/z: `research_agency_lab/experiments/quran/reference_stats.json`
(`datastore/reference_stats.py`). Cohort model: `research_agency_lab/experiments/quran/cohort_model.json`
(`datastore/learner_validation.py`). Quran-wide inventory (184,014 rule instances, 264,324 consonants
in all 6,236 ayahs): `research_agency_lab/experiments/quran/inventory.jsonl.gz`.

### What is validated (measured, not asserted)

- **Learner model, leave-one-reciter-out** (41 reciters): beats the cohort mean on skills the reciter
  has NOT recited by +4.1 % (1 verse), +15.1 % (3), +27.1 % (10), +9.2 % (30); on seen skills +2 to
  +31 %, and beats the raw observed rate everywhere (from one verse the raw rate errs ~10×).
- **Masters control** on Baqarah 2:1–2 (`control_passage.py`): 25/41 at 100 % after the fixes.
- **Expert review**: 15 verdicts from a certified reciter (`research_agency_lab/experiments/review/labels.jsonl`);
  every fix they drove is in the engine and replays correctly.
- **Anatomy of sound** (`substrate_library/julia/anatomy.jl`, Octave mirror agrees to 2.4e-6 dB):
  voiced hold on ب/د, shaddah = collision + separation, qalqalah stronger at the stop, the hamza does not
  bounce. 11 of 12 causal edges supported (`causal.jl`).
- **Calculus of characteristics** (`calculus.jl`, 545,592 consonants): FCA derives qalqalah = jahr ∩
  shadid minus the hamza.

### Research tooling (all committed)

- `datastore/waqf_table.py` — every stop in the Quran and what it changes (71,197 boundaries).
- `datastore/reciter_profile.py`, `datastore/rule_layer.py` — per-reciter profiles and rule layer.
- `substrate_library/julia/structures.jl` — knowledge space, Chow–Liu, PC, d-separation with
  reciter/verse decomposition, the precision-state factor.
- `datastore/recitation_graph.py` — Cypher-queryable graph (embedded Kùzu; Neo4j CSV + import script).
- `/review` (listening review with Uthmani live highlight), `/protocol` (10 paired recording tests).
- `research_agency_lab/compute_bridge/modal_profile.py` — the full-dataset engine pass on Modal CPU:
  11,996 verses in ~5 min, reading posteriors already on the `qaari-runs` volume; also builds the
  Quran inventory. `modal_muaalem.py` dumps posteriors on GPU (needed for NEW audio).
- Surveys: `research_agency_lab/experiments/deep_research/08_mistake_datasets.md`, `09_mistake_synthesis.md`.

---

## Sprint 4 goals

### Goal A — gather recitations and score them

Sources to start with (checked in Sprint 3; see `deep_research/08_mistake_datasets.md`):

| dataset | what | why |
|---|---|---|
| `RetaSy/quranic_audio_dataset` (HF) | 6,828 clips, 1,287 non-Arabic learners, labelled correct / incorrect by 3 annotators | real learners; labels to score against |
| `sobolev210/quran-recitation-errors` (HF) | learner chunks, word-level tags Wording / Tajweed / Letters / Tashkeel, Hafs + Qalun | located real errors |
| `MuazAhmad7/Surah_Ikhlas-Labeled_Dataset` (HF) | 1,506 clips, typed + located errors | typed errors |
| `obadx/qdat_bench`, `obadx/qdat` (HF) | per-rule lengths in counts, ghunnah / qalqalah grades; 1,505 clips of one verse | rule-level ground truth |
| `IqraEval/Iqra_train`, `Iqra_TTS`, `QuranMB.v2` (HF) | 71k train, 55k synthetic mispronunciations, 1,642 test | phoneme-level substitutions |
| `IqraEval/Iqra_Extra_IS26` (HF, **gated**) | 1,333 deliberate substitutions | request access |
| EveryAyah (other reciters) and quran.com audio | professional recitations beyond T300 | more masters, more styles |

For each: download (HF datasets), dump posteriors on Modal GPU (`modal_muaalem.py` pattern), grade on
Modal CPU (`modal_profile.py` pattern), and compare the engine's findings with the dataset's labels:
per dataset, the agreement, the misses, the false alarms, and what they reveal. Learner audio is
different from professional audio (phones, rooms, non-native voices): expect new failure modes.

### Goal B — serve the consumer apps

- Treat `measurements/1` as an API: document every field, version it, keep it stable, add tests that
  pin its shape. Consider a JSON Schema file.
- Calibrate the learner prior on real learner audio (it is fitted on 41 professionals and widened by
  an assumed floor `LEARNER_FLOOR_SD = 1.0`).
- Measure latency per recording and per page; the target was < 10 s for one page.

### Open issues carried over

1. **Speed allowance vs careless reading** — "too long" at hadr is allowed when the length is normal in
   seconds (the reviewer's ruling on Ghamdi). The same rule now passes the stretched ذَٰلِكَ in a
   deliberately careless fast reading. Needs the reviewer's ruling on where fast-but-correct ends.
2. **The ghunnah family is the hardest rule set** (idgham with ghunnah 0.69, ghunnah 0.76, ikhfa 0.80
   across 41 reciters) — not yet checked for a reading choice or measurement artefact the way the
   munfasil was (that one was the wajh).
3. **Stop detection misses reverberant stops** (32 dB threshold; masters dip 12–34 dB at a real stop,
   continuous readers 2–5 dB). The waqf table is only partly in grading (madd tabii → 'arid, munfasil,
   silah). Planned: test "wasl vs waqf" per boundary acoustically instead of by loudness.
4. **Anatomy scores (collision, separation, hold, echo) are research-only** — not yet in the serving
   report.
5. **Precision state** (verse-level characteristic coupling) cannot yet separate performance from the
   recording condition; needs the anatomy cross-check and a per-file noise covariate.
6. `/review` has 47 unlabelled candidates; the `/protocol` takes are not yet recorded.
7. Kùzu is archived upstream — a bridge; Neo4j is the target for the graph.

---

## Rigor rules — keep these exactly

- Commit constantly; every commit production-ready; never batch. Push only on explicit approval.
- Numerics in Julia (`~/julia-1.11.5/bin/julia`), numpy pinned by parity test; Octave cross-checks DSP.
- Never report a number you have not measured. Verify against the corpus, not memory.
- Any model or metric must beat a baseline (e.g. the cohort mean, leave-one-reciter-out) before it is a
  claim. Two learner-model formulations lost to the baseline in Sprint 3 and were fixed at the cause.
- The engine measures; consumer apps decide what to tell learners. No advice text in the engine.

## Traps that cost time

- `julia` on PATH is 1.10.9 and stalls precompiling this project — always `~/julia-1.11.5/bin/julia`.
- `pkill -f PATTERN` / `pgrep -f PATTERN` match your OWN shell's command line when the pattern appears in
  it — the shell kills itself (exit 144) or a waiter never exits. Match anchored: `pgrep -f "^\.venv/bin/python -m app\.webapp$"`.
- A waiter grepping for `Error` matches Modal's transient `ConnectionError` heartbeat line; wait for the
  final result line instead.
- Julia: Arabic strings index by BYTES — `collect(String(s))` before indexing; JSON3 cannot write NaN.
- Python 3.11: no nested same-type quotes inside f-strings.
- Modal: the local client can lose its heartbeat mid-run and still finish; count the output lines.
- Webapp restart: `.venv/bin/python -m app.webapp` (port 8088, public http://40.64.120.87:8088); the
  model takes ~1 min to warm.

## Compute

Modal: `qaari-runs` volume holds T300 posteriors (`muaalem/T300/<shard>/`). Sprint 3 spent a few dollars
of the $8.53 budget on CPU fan-outs (three full-dataset passes + the inventory); check the dashboard
before GPU work — dumping new posteriors is GPU and the most expensive step.
