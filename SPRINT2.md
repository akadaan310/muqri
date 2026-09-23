# Sprint 2 — 100 % treatise coverage, with a live upload page

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

**Blocked (2) — do not burn the sprint on these without a plan.** `harakah_isochrony` and
`harakah_weight_independence` are unmeasurable at the model's **40 ms frame rate**: a short vowel is
1–2 frames, so the three vowel medians all quantise to 1.0 and the spread reads exactly 0.0 for all
41 reciters. Root cause is `add_adapter: true, adapter_stride: 2` halving a 20 ms base. Either
implement **sub-frame onset interpolation from posterior shapes**, or mark them permanently
out-of-scope with the reason. The same ceiling blocks the strict three-way sukoon ordering.

### Goal C — KPIs to move

| KPI | now | target |
|---|---|---|
| treatise coverage | 58 % full / 87 % partial | **100 % full** (or explicitly out-of-scope with a reason) |
| rule-instance bind rate | 98.9 % | ≥ 99.5 % (`madd_lazim` 0.57 and `idgham_shafawi` 0.87 remain) |
| passage accuracy on the anchor | 96.6 % | ≥ 98 % |
| segmentation boundary error | up to ~1,100 frames | < 300 frames |
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
