# AGENTS.md — for AI agents working in this repository

Read this first. It describes the service, what it computes, which Tajweed rules it covers and how
reliably, where each piece lives, and the working rules. Deeper material: `AGENTS-EXPLORE.md` (research
directions and data), `GRAPH.md` (the knowledge graph), `SPRINT4.md` (numbers and traps), `README.md`
(the legacy qaari-eval CLI pipeline and its calibration on the full-Qur'an run).

## 1. What the service is

**muqri** measures Quran recitation in the riwāya of Ḥafṣ ʿan ʿĀṣim. Input: audio plus the verses
recited (whole ayahs or parts of an ayah). Output: a `measurements/1` report that says, for every
letter, characteristic (ṣifah) and tajwīd rule, what was realised and how it compares with 41 master
reciters. **The engine measures; the apps decide.** Output is numbers, verdicts and reference
comparisons only, never advice text for the learner.

Two code paths exist:

| path | status | entry |
|---|---|---|
| **muaalem engine** (multi-level CTC acoustic model, phonemizer, rule binder) | **live**; used by the web service, sessions, review | `app/engine.py` → `Engine.analyze` |
| qaari-eval v2 (wav2vec2 aligner + DSP validators + calibrated z) | legacy; CLI and benchmark runs | `main.py`, `app/pipeline.py`, `app/scoring.py` |

## 2. Running it

```bash
.venv/bin/python -m app.webapp                 # port 8088; the model warms in ~1 min; GET /health
.venv/bin/python -m pytest -q -p no:warnings   # the full suite; all must pass before a commit
scripts/neo4j.sh start                         # the knowledge graph (see GRAPH.md)
```

Restart the web app after changing `app/sessions.py` or the engine; it loads them once. Start it with
`(setsid nohup .venv/bin/python -m app.webapp > /tmp/claude-1000/webapp.log 2>&1 &)`. Do **not** use
`pkill -f app.webapp` from a shell whose own command line contains that string: it kills the shell.
Kill by pid (`pgrep -f "m app.webapp"`).

Julia: `~/julia-1.11.5/bin/julia` (the `julia` on PATH is 1.10 and stalls).

### HTTP surface (`app/webapp.py`)

| endpoint | what it does |
|---|---|
| `POST /analyze` | audio + verses → the full report and `measurements/1` |
| `POST /detect` | identify which verse(s) a recording contains |
| `GET /capability` | live coverage tables: taxonomy, rule catalogue, launch matrix (see §4) |
| `GET /sessions`, `/sessions/rounds`, `/sessions/round/{n}`, `POST /sessions/submit` | calibration rounds (§5) |
| `GET /review`, `/review/next`, `POST /review/label`, `/review/stats`, `/review/audio/{id}` | expert listening: confirm or reject engine verdicts by ear |
| `GET /protocol`, `/protocol/tests`, `/protocol/status`, `POST /protocol/submit` | paired correct / scripted-mistake recording protocol |

The reviewer records at `http://40.64.120.87:8088/sessions` (plain http; the page offers "Record with
phone" and file upload).

## 3. What is computed, layer by layer

| layer | computation | where |
|---|---|---|
| acoustic model | muaalem multi-level CTC, 40 ms frames: phonemes + ten characteristic heads (ghunnah, hams/jahr, shiddah/rakhāwah, tafkhīm/tarqīq, qalqalah, iṭbāq, ṣafīr, tafashshī, istiṭālah, takrīr) | `research_agency_lab/experiments/learner_eval/` |
| expected text | `quran_transcript` phonemizer: Uthmani → phoneme script; lengths as character repetition (a 4-count madd is the vowel ×4) | `app/engine.py` |
| rule location | `TajweedParser`: Uthmani → letter units → the 40 `RuleType`s with detail (kāmil/nāqiṣ, ṣughrā/kubrā…) | `app/tajweed_rules/parser.py`, `app/models.py` |
| rule binding | each located rule → the phoneme units that realise it (word-final rules bind to the last letter; idghām to the held letter) | `app/rule_bind.py` |
| alignment | Viterbi / forward–backward over the expected phonemes, sub-frame centroid onsets; multi-part submissions settle part boundaries on pauses | `app/analysis.py`, `app/submission.py` |
| letter identity | GOP against the classical confusions of each letter (ح/ه, ع/ء, ظ/ز/ذ, ش/س …): margin in nats | `app/analysis.py`, `app/lahn/` |
| characteristics | per letter, per head: expected vs heard class and margin | `app/analysis.py` |
| durations | every madd, ghunnah, ikhfāʾ, idghām hold in the reciter's **own count unit** (seconds of a plain voweled letter, estimated per recording and per ayah) | `app/stretch.py`, `app/submission.py` |
| context | basmala detection; stops from the waveform; the stop table (what stopping at each of 71,197 word boundaries changes); wajh (qaṣr/tawassuṭ) declared or inferred | `app/engine.py`, `app/waqf.py`, `datastore/waqf_table.py` |
| comparison | percentile and robust z of each rule length and letter×characteristic margin against the masters (`reference_stats.json`) | `app/measurements.py` |
| learner model | 45 skills, cohort prior, Laplace posterior, knowledge-space paths | `app/learner.py`, `app/knowledge.py` |
| graph | Neo4j + GDS: mushaf, phonology, calculus, performance, timing, skills, reliability, sessions, ra' layers | `datastore/kg.py`, `GRAPH.md` |

`measurements/1` (schema in `app/measurements.py`, `app/schemas/`): `recording` (tempo, count unit,
wajh, basmala, stops, edge room), `summary`, `words` (all_correct, failing ids), `rules` (observed
counts, deviation, status, percentile, z_masters), `letters` (identity, characteristics with margins,
onset, duration). Adding an optional field keeps the schema version; changing one needs a new version.

## 4. Rule coverage

The parser locates **40 catalogued rules, 1.16 M instances** over the whole Qur'an. Each is judged by
one of three mechanisms:

* **segmental**: the rule changes which phonemes are said (idghām merges, iqlāb turns nūn into mīm,
  iẓhār keeps it clear). Tested by the CTC likelihood of the correct vs the counterfactual phoneme string.
* **durational**: the rule sets a length (madd 2/4/5/6, ghunnah hold, sakt). Measured in counts.
* **attribute**: the rule asserts a quality of a letter. Judged by the matching muaalem head.

| family | rules | mechanism |
|---|---|---|
| mudūd | ṭabīʿī, muttaṣil, munfaṣil, lāzim, ʿāriḍ li-s-sukūn, līn, badal, ʿiwaḍ, ṣila ṣughrā, ṣila kubrā | durational |
| nūn sākinah / tanwīn | iẓhār ḥalqī, ikhfāʾ, idghām bi-ghunnah, idghām bilā ghunnah, iqlāb | segmental + nasal duration |
| mīm sākinah | ikhfāʾ shafawī, idghām shafawī, iẓhār shafawī | segmental + nasal duration |
| ghunnah mushaddadah | ghunnah on نّ / مّ | durational |
| qalqalah | ṣughrā / kubrā / akbar | attribute (qalqla head) |
| other idghām | mithlayn, mutajānisayn, mutaqāribayn | segmental |
| rāʾ / lām | tafkhīm, tarqīq, jawāz al-wajhayn | attribute (tafkheem_or_taqeeq head) |
| waṣl / waqf | hamzat al-waṣl, sakt, stops | segmental / durational / waveform |
| ṣifāt | hams, jahr, shiddah, tawassuṭ, rakhāwah, iṭbāq, ṣafīr, tafashshī, istiṭālah, takrīr | attribute |
| laḥn jalī | wrong letter, wrong short vowel | segmental (GOP) |

**How reliable each capability is** comes from `GET /capability` → `matrix` (built by
`datastore/launch_matrix.py`; read it live, do not copy numbers into prose that will drift). As of
2026-09-24:

* **ship**: laḥn letter / ḥarakah (named recall 0.99 at 1 % anchor false alarm); all ten ṣifāt heads
  (anchor false alarm ≤ 1 %); counterfactual rule tests at ≥ 0.95 anchor win rate for ghunnah drop,
  idghām undo, ikhfāʾ→iẓhār, iqlāb, madd too long, qalqalah drop; ghunnah four levels; waqf execution;
  sukūn separation; ladder placement (held-out anchor 97.05).
* **caution**: madd too short (0.937 vs 0.95), sakt, ikhtilās / ishbāʿ rates, sukūn ordering.
* **hold**: absolute madd scale against notation, ḥarakah isochrony, vowel sequences, reciter ranking.
* **absent**: verse identification as a gated capability, learner grading policy, positive feedback,
  waqf permissibility.

The `rules` table's `status` (broken / suspect / plausible) is the **legacy DSP detectors'** pass rate on
masters: it is a work list for that path, not the muaalem engine's accuracy.

Out of scope: other qirāʾāt (Ḥafṣ only), anatomical explanation, idhlāq/iṣmāt.

## 5. Calibration rounds (`app/sessions.py`, `/sessions`)

The rounds calibrate the engine on recordings made by a certified reciter (the owner). Each exercise
is recited twice at tadwīr:

* **Take A**, to the spec: every declared expectation (rule, word, count band) must be met and nothing
  else flagged.
* **Take B**, the same reading with **dictated mistakes** at named words: each must be caught with the
  right kind of evidence (`Sig`: rule status, characteristic, identity), with named control words left
  correct.

An `Exercise` declares `expect`, `mistakes`, `controls`, the wajh, and for long ayahs the `stops`
(the reciter does not choose them; the engine must know them, because a stop changes what the text
requires). Results append to `research_agency_lab/experiments/sessions_results.jsonl`; audio goes to
`experiments/session_recordings/` (git-ignored: a person's voice). After an engine fix, re-score every
take by re-posting it to `/sessions/submit` with `note="rescore of <stamp> after <fix>"`.

Validate a new exercise against Husary and Minshawy (`datastore.review_queue.audio_path`) before it
goes live: every expectation must be located, or be dropped with the reason in the commit.

Measured effect of the rounds (original takes, first scoring → current engine):

| round | dictated mistakes caught | take-A expectations met | take-A false alarms |
|---|---|---|---|
| 1 | 4/12 → 9/12 | 14/22 → 19/22 | 4 → 0 |
| 2 | 10/15 → 12/15 | 32/36 → 33/36 | 9 → 4 |
| 3 | 10/12 → 11/12 | 62/72 → 68/72 | 14 → 8 |
| 4 | 9/9 | 25/25 | 1 (plus 2 in take B) |

The fixes behind those moves came from the misses: madd ʿiwaḍ bound to the word's final alif, ṣila
bound to the final letter, idghām timed on the held letter rather than the following vowel, the
calibrated ghunnah band.

## 6. Data

* T300: 41 masters × ~300 verses of per-frame posteriors (Modal volume `qaari-runs`; local
  `experiments/qaari_keys/modal_T300/`, git-ignored). Per-verse records: `experiments/profiles/clips.jsonl.gz`.
* Whole-Qur'an inventory: `experiments/quran/inventory.jsonl.gz`. Stop table: `experiments/waqf/`.
* Reference stats for z / percentiles: `experiments/quran/reference_stats.json`.
* Expert labels: `experiments/review/labels.jsonl`.
* Legacy full-Qur'an Kaggle run (46,285 rows) and calibration: `benchmarks/results/`.
* DuckDB store behind `/capability`: `datastore/store.py`.

## 7. Working rules

* **Commit constantly, every commit working, and push each commit** to
  `origin/claude/qaari-eval-engine-btwkjt`. Other agents work on this branch: `git fetch` and merge
  (never force-push, never overwrite someone else's commits).
* Tests green before every commit. Match the surrounding code: comments explain *why*, in plain prose.
* Never report a number you have not measured; state misses and false alarms as measured.
* No advice text in engine output (§1).
* Numerics in Julia with parity tests against the numpy path.
* Recordings of people's voices never enter git.
* Writing Arabic combining marks: the Write tool can mangle `\u` escapes; write them from Python with
  explicit escapes when it matters.
