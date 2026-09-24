# Sprint 2 — 100 % treatise coverage, with a live upload page

## Paste this to start the next session

> Continuing the qaari/muqri engine. Same VM (koda-vm), repo `~/github-director/repos/muqri`, branch
> `claude/qaari-eval-engine-btwkjt`.
>
> **Read `SPRINT2.md` in the repo root first** — it is the complete handover: what Sprint 1 built and
> measured, the architecture and key files, Sprint 2's goals and KPIs, the rigor rules, every phonetic
> fact verified against the corpus, the traps that cost time, and the compute position.
>
> This is **Sprint 2**, with a 5-hour window. Three goals, in this order:
>
> 1. **Build the upload page first, not last.** A small web service where I open a URL, drop in my own
>    recitation file, pick the verses, and see the full report. `app/api.py` already has FastAPI
>    `POST /analyze` but routes to the legacy pipeline — re-point it at `app/engine.py`. Give me the
>    address early, then keep using it to evaluate the service while you build everything else.
> 2. **Finish the sub-frame timing work (Goal B2).** The 40 ms quantisation ceiling was broken at the
>    end of Sprint 1 — `research_agency_lab/experiments/subframe/` has the working method and the
>    measurements. One question is open and must be settled before anything ships; the file says which.
> 3. **Take treatise coverage from 58 % to 100 %.** Query `tajweed_taxonomy` in DuckDB for the live
>    gap list.
>
> Standing rules: commit constantly and make every commit production-ready; numerics stay in Julia
> with the numpy path pinned to it by parity test; never report a number you have not measured; verify
> against the corpus rather than reasoning from memory. Push only on my explicit approval. $8.53 of
> Modal budget remains — check `modal app list` before launching anything.

---

*Sprint 1 ended 2026-09-23 on branch `claude/qaari-eval-engine-btwkjt`, 14 commits, nothing pushed.
Every commit is production-ready; that is a standing rule for this repo — **commit constantly, never
batch**.*

---

## What Sprint 1 delivered (do not rebuild any of this)

A **server-side submission service**. `Engine().analyze(audio, verses)` takes a recitation at any
scale — a word, a verse, a page, a juz' — and returns every letter's complete state, every located
tajweed rule graded against what it requires, and cross-ayah mastery statistics.

Measured on the Husary Muallim anchor, 10 ayahs: **1,318 letters, 9,396 judgments, 474 rules graded,
96.6 % accuracy**. One verse alone yields ~774 judgments; a five-letter word ~98.

| capability | measured |
|---|---|
| wrong letter, located and **named** | 99.91 % detected / **99.06 % named**, 0.73–1.30 % anchor false alarm |
| wrong harakah, located and named | 99.35 % / **98.73 %** |
| 8 rule families (counterfactual) | 93.7–99.8 % (qalqala 99.84, idgham 99.16, madd_long 98.99, ghunnah 97.45, iqlab 97.33, ikhfa 95.62, madd_short 93.71) |
| 10 sifat heads | ~1.0 % anchor false alarm, all classes n ≥ 99 |
| rule-instance binding | **98.9 %** of 10,529 located rules |
| verse/madd coverage | id_vowel = id_madd = **1.0000** at every tier and tempo |
| ladder placement | held-out anchor **97.05** |

Evidence base: **2,036,974 individually measured decisions** across 41 reciters (T300, $3.84 spent).

### The architecture, in one paragraph

Client knows the verses and submits audio server-side (**not** on-device, **not** real-time).
`muaalem-v3.2` (605 M params, 11 CTC heads: 43 phonemes + 10 sifat, 40 ms frames) produces
posteriors. `app/analysis.py` (numpy, **pinned to the Julia reference exactly**) gives per-unit
identity/GOP/sifat/timing. `app/rule_bind.py` joins the parser's *named* rule instances to phoneme
units via `word_ph`. `app/submission.py` grades and aggregates. Julia
(`substrate_library/julia/`) remains the research numerics and the reference; Python serves.

### Key files

```
app/engine.py        the one call a server makes
app/analysis.py      numpy per-unit core (ctc_viterbi, gop, sifat, timing)
app/rule_bind.py     parser RuleInstance -> phoneme units  (the structural join)
app/submission.py    grading, roll-ups, cross-ayah mastery, walk_alignment
app/ghunnah.py       four maratib + letter quwwa
app/mudud.py         aqwa al-mudud resolution + drift
datastore/           rule_catalogue, tajweed_taxonomy, letter_reference, launch_matrix (all in DuckDB)
substrate_library/julia/   CtcGop, SifatGop, Frontier (EVT/GPD, Bures-W, conformal), madd_scale
```

`~/qaari-store/qaari.duckdb` (382 MB): 1.16 M `engine_diag`, 1,226 `test_result`, 40
`rule_catalogue`, 38 `tajweed_taxonomy`, 939-node knowledge graph.

---

## Sprint 2 goal

**100 % treatise coverage, and a live page where the user uploads their own recitations.**

### Goal A — the upload page (build this FIRST, then keep using it)

A small web service the user can open and drop an MP3/WAV into, pick the verses, and see the full
report. It is how the service gets evaluated while the rest of the sprint is built, so it must exist
early, not at the end.

- `app/api.py` already has FastAPI `POST /analyze` (audio + surah/ayah/ayah_end) — it currently routes
  to the **legacy v2 pipeline**. Re-point it at `app/engine.py`.
- Add a minimal upload page + a rendered report (letters with their sifat, rules with expected vs
  given counts, mastery block). No framework needed.
- Serve it and give the user a URL. It must accept a real recording end to end.
- **KPI:** a user-supplied file returns a full report in < 10 s for one page of text.

### Goal B — close the last treatise gaps to 100 %

Current: **22/38 covered (58 %), 11 partial, 3 uncovered, 2 blocked.** Query the live table:
`SELECT status, concept_id, name FROM tajweed_taxonomy ORDER BY status;`

**Uncovered (3)** — all need the parser's stop/waqf annotations wired through:
- `waqf_ibtida` — stop types, permissible and impermissible starting points. `TajweedParser.parse`
  takes `stops=` and `continue_after=`; `ParsedText` has `phrase_final_words` and `stop_at_end`.
- `sakt` — the brief pause without breath (only 104 instances in the corpus; needs an edit generator
  in `ruleswap_gop.jl`).
- `endurance` / `itmam_universal` / `vowel_sequences` — sequence-level vowel integrity
  (كُتُبُهُ, يَعِدُكُمُ, كُنتُمْ) and degradation over a long pass.

**Partial (11)** — the valuable ones:
- `tafkhim_five_levels` — grade heaviness by vowel context (fatḥah+alif > fatḥah > ḍammah > sukūn >
  kasrah). The join now supplies the context, so this is reachable.
- `tafkhim_nisbi` — isti'lā' without iṭbāq + kasrah reduces heaviness.
- `istila_istifal` — no dedicated head; infer from `tafkheem_or_taqeeq` + `letter_reference`.
- `metric_formant_integrity` / `metric_nasal_isolation` / `metric_consonantal_envelope` — the
  treatise's own diagnostic matrix. `app/sifaat/formants.py` exists but is not wired to the new path.
- `ghunnah_vs_madd` — the velum must snap shut; the cut-off transient is not measured.
- `madd_arid` — give it its own tasāwī class.

**Blocked (2) — SOLVED at the end of Sprint 1; finish the job.** See the next section.

### Goal B2 — sub-frame timing: the 40 ms ceiling is broken, now exploit it

`research_agency_lab/experiments/subframe/` (committed). **Read `occupancy.py` first.**

The two "blocked" concepts were never a coding gap. The model emits one frame per 40 ms
(`add_adapter: true, adapter_stride: 2` halves a 20 ms base), a short vowel is 1–2 frames, so a hard
Viterbi span can only be 1 or 2 and every duration statistic built on it quantises. The three vowel
medians landed on the same value and the spread read exactly **0.0 for all 41 reciters** — an
artefact, not a finding.

**The fix: stop taking a hard path.** CTC forward–backward gives γ_t(s), the posterior probability
that extended state *s* occupies frame *t*. Summed it is a fractional occupancy; its centre of mass
is a continuous onset. Three estimators are implemented and cross-check each other:

| estimator | what it gives | verdict |
|---|---|---|
| `expected_durations` | Σγ per symbol | **confidence mass, not acoustic time** — only 132 of 605 frames carry symbol mass, CTC is peaky |
| `centroid_onsets` | posterior-weighted centre of occupancy | **continuous and acoustic — build on this** |
| `peak_parabolic` | parabolic interpolation of the occupancy peak | cheap cross-check; agrees with the centroid to ~0.004 of a vowel |

**Quantisation is decisively broken** (60 anchor clips):

| vowel | unique values, Viterbi | unique values, centroid |
|---|---|---|
| fatḥah | 56 / 1245 | **1106 / 1245** |
| ḍammah | 37 / 382 | **368 / 382** |
| kasrah | **1 / 440** — every instance identical | **422 / 439** |

#### What it already measured (bootstrap 95 % CIs, 400 clips per group)

**Isochrony** — vowel duration as a fraction of that reciter's own haraka unit:

| group | fatḥah | ḍammah | kasrah | spread |
|---|---|---|---|---|
| anchors | 0.7313 [0.7267, 0.7355] | 0.7598 [0.7512, 0.7663] | 0.7319 [0.7230, 0.7388] | 0.0284 (**3.84 %**) |
| fast imams | 0.6207 [0.6170, 0.6237] | 0.6255 [0.6223, 0.6293] | 0.6123 [0.6043, 0.6191] | 0.0132 (2.12 %) |

On the anchors **fatḥah and kasrah are statistically equal** — their CIs overlap almost exactly,
which is isochrony holding. **Ḍammah is genuinely longer**: its CI [0.7512, 0.7663] does not overlap
fatḥah's, so the ~3.9 % difference is real and not noise. That is a publishable observation about
how Hafs is actually recited, and it could not be seen at all before.

**Weight independence** — the treatise requires a vowel's length to be independent of whether its
consonant is heavy; the pharyngealisation effort belongs to the consonant phase:

| group | heavy-consonant vowel | light | difference |
|---|---|---|---|
| anchors | 0.7093 (n=1594) | 0.7389 (n=11824) | **−0.0296 (−4.0 %)** |
| fast imams | 0.5723 (n=1728) | 0.6235 (n=12947) | **−0.0512 (−8.2 %)** |

The law is **violated by everyone** — vowels on heavy letters are measurably shorter — but **anchors
violate it roughly half as much as fast imams**. That makes weight independence a genuine mastery
discriminator, in the direction the treatise predicts.

#### What Sprint 2 must settle

1. **The isochrony spread direction is counterintuitive and unresolved.** Fast imams show a *smaller*
   spread (2.12 %) than anchors (3.84 %). The likely explanation is compression — everything is
   driven toward a floor, so the vowels converge for the wrong reason — but that is a hypothesis, not
   a result. Test it: normalise by something other than the reciter's own haraka, or compare spread
   against absolute tempo. **Do not ship an isochrony score until this is settled**, or a fast imam
   will outscore Husary on it.
2. **Port the centroid onsets into `app/analysis.py`** behind a flag, re-run the parity test, and
   check what else improves — sukoon three-way ordering, the madd absolute scale, and the segmentation
   boundary error (currently up to ~1,100 frames) all rest on the same quantised onsets.
3. **Mirror in Julia** (`substrate_library/julia/`) and cross-check, per the standing rule.
4. Then flip `harakah_isochrony` and `harakah_weight_independence` in `tajweed_taxonomy` from
   `blocked` to `covered`, with these numbers.

**The wider prize:** continuous timing turns every per-letter duration into a real-valued measurement.
That is what makes it possible to study how individual letters behave, how scoring responds when
particular characteristics or rules co-occur, and where a given reciter's timing signature sits —
none of which is possible while every duration is 1 or 2.

### Goal C — KPIs to move

| KPI | now | target |
|---|---|---|
| treatise coverage | 58 % full / 87 % partial | **100 % full** (the 2 blocked are now solvable — see Goal B2) |
| rule-instance bind rate | 98.9 % | ≥ 99.5 % (`madd_lazim` 0.57 and `idgham_shafawi` 0.87 remain) |
| passage accuracy on the anchor | 96.6 % | ≥ 98 % |
| segmentation boundary error | up to ~1,100 frames | < 300 frames (sub-frame onsets should help) |
| vowel-timing resolution | 1–2 frames, quantised | continuous (**done**: 1106/1245 unique) |
| upload → report latency (one page) | — | < 10 s |

---

## Rigor rules — keep these exactly

1. **Commit constantly; every commit production-ready.** Message carries the measurements that
   justify it and the failures found on the way.
2. **Numerics in Julia**, Octave as an independent mirror, CUDA where warranted. Python serves.
   `tests/test_analysis_parity.py` pins the numpy path to Julia *exactly*; if you change one, prove
   the other still matches.
3. **Never report a number you have not measured.** Small-n results get held back — iqlāb read 66.7 %
   at T10 (9 anchor instances) and 97.3 % at T300.
4. **Verify against the corpus, never reason from memory.** Every phonetic assumption in Sprint 1 that
   was checked turned out to need correction.
5. Unvalidated quantities ship with `confidence: "unvalidated"`, never silently.
6. Push / PR / publication only on **explicit user approval**. Local commits are expected.

### Phonetic facts you will need (all verified on the corpus)

- The phonetizer encodes **duration as character repetition**: `ا×2` = madd ṭabī'ī, `ۥ×4` = 4-count,
  `ں×3` = ikhfā' noon (2-count ghunnah), `۾×3` = iqlāb **and** ikhfā' shafawī, `ڇ` = qalqala release.
- **A shadda letter is two letters** — a sākin half and a voweled half collided. Nasals held `×4`
  (two counts of ghunnah), everything else `×2` (the collision).
- **Idghām crosses the word boundary**: the letter is deleted and the *next word's first* letter is
  doubled (`مِن رَّبِّهِمْ` → `ر×2`). Search the receiving word, not the current one.
- **Hamzat al-waṣl is absent** from the phonemes when joined; bind to the word's first unit.
- **Muqaṭṭaʿāt** are spelled as letter *names* — `الٓمٓ` is one word to the phonetizer, three to the
  parser.
- **ر / ل weight is contextual** (11 conditions, `letter_reference.weight_rules`, 15 tests). The
  jalālah rule depends on the **preceding word**, so it must be validated on full ayahs, never
  isolated words.
- **Sifāt belong to consonants only** — vowels and madd letters carry none.
- **Count scale**: `measured = −0.920 + 1.360 × nominal` (Theil–Sen, 10,600 anchor instances).
  Recovers 1.5→1.50, 2→2.04, 4→3.90, 4.5→4.35; the 6-count level reads 7.21 and stays flagged.

### Traps that cost Sprint 1 time

- **Sequential alignment drifts.** Walking ayah-by-ayah through a recording accumulates error (2,530
  frames by ayah 6, accuracy 0.55). Anchor each ayah to its **global proportional position**.
- **CTC is peaky** — the Viterbi span `l−f` is an alignment artefact, not a duration. Use
  onset-to-onset.
- **The last unit of a clip absorbs trailing silence**, and any count > 12 is a measurement failure,
  not a long madd (madd 'āriḍ read 37 counts).
- **A slope is not drift** without a shift in the median too.
- Julia `Threads.@spawn` **raced inside `gop_sf`** even with per-task buffers — textswap stays
  sequential.
- `INT8` ONNX is worse *and* slower than fp32 (RTF 1.049 vs 0.813). Irrelevant server-side; do not retry.

---

## Measuring anything on this box

**It is shared and frequently loaded.** Observed load average **43 on 4 cores** — three Julia
processes from `~/quran-calligraphy/math` plus `eval_htr.py` and `run_experiments.py`, none of them
this project's. Every latency figure taken under that is meaningless: the engine measured RTF ~2 per
verse, and raising torch threads 1 → 4 made it *slower* (RTF 3.65 → 22.79), which is thrashing at
~10x oversubscription, not model behaviour. The same model measured RTF 0.52 on a quiet box.

Check `uptime` before timing anything, and re-profile when load is under ~4. Accuracy measurements
are unaffected — contention only slows them.

Where the time actually goes, profiled: **analysis is free** (0.26 s per verse, RTF 0.02) and
essentially all of it is the acoustic model's forward pass.

## Compute

- **Modal** (`source ~/qaari_creds.sh; export MODAL_PROFILE=akadaan310`): **$8.53 of the $20 work cap
  remains**; account cap $30. T4 measured at **$0.00035/clip**, 1.623 s/clip. `modal_muaalem.py` now
  runs 50 containers with cpu=4 and a 12-deep prefetch. Whole Quran ≈ 29 min GPU / $2.18.
  **Check `modal app list` before launching — never duplicate GPU spend.**
- **Local**: 4 CPU, 15 GB, no GPU, 93 GB free. Analysis is cheap: sifat 5.2 ms/ayah, rules 3 ms,
  letter report 10.5 ms. A juz' is ~12 s of analysis.
- **Azure $200 / Oracle $300** available; both logins verified, but **GPU quota is 0 on both** and the
  managed identity cannot request it. Modal is the only GPU path.
- Data on disk: `modal_T300` (11,996 clips, 41 reciters, 1.8 GB), `modal_T10` (414), all gitignored.

## Verification before calling Sprint 2 done

- `pytest -q` green (**140 now**), Julia `test_ctcgop.jl` 28/28 and `test_frontier.jl` 29/29.
- The upload page accepts a real user file and returns a full report.
- `tajweed_taxonomy` shows no `uncovered`; anything left is `out_of_scope` **with a stated reason**.
- `launch_matrix` regenerated; every number written to `test_result` in DuckDB.
