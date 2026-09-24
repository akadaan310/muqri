# The Recitation Knowledge Graph

Every dataset we hold, as one connected graph in Neo4j with Graph Data Science. It serves two uses:
the engine's own algorithms (what to trust, what to expect) and research (questions nobody else can
ask, because nobody else has these measurements joined).

    scripts/neo4j.sh start                              # local server, 127.0.0.1 only
    .venv/bin/python -m datastore.graph load-blindspots # the reliability layer
    .venv/bin/python -m datastore.kg build              # every other layer (about 30 s)
    .venv/bin/python -m datastore.kg_discover           # the GDS analyses -> experiments/graph/discoveries.json

The credentials are in `~/.config/qaari/neo4j.env` (chmod 600) and never in the repo. The install is
Neo4j Community 2026.09.0 with GDS 2026.09.0 and Java 21, under `~/opt`. GDS is called through Cypher
procedures (`CALL gds.louvain.stream(...)`) so it doesn't depend on the Python client's API.

## Layers, their structures, and the questions they answer

| layer | nodes / relationships | mathematical structure | what it answers |
|---|---|---|---|
| mushaf | Surah, Ayah, QWord; HAS, NEXT, CARRIES {n}→Rule | a path graph (the text) with a bipartite word–rule incidence | where each rule occurs; which ayahs train the same rules (`SAME_TAJWID`, Jaccard) |
| phonology | Sound, Letter, Sifah, Makhraj, Region; HAS_SIFAH, ARTICULATED_AT, PRECEDES {count, pmi} | a formal context (letters × the 17 sifat, for FCA) and a weighted transition digraph (PMI) | is the text's sound sequencing organised by articulation or by characteristic? |
| calculus | Reciter, Feature, Community, Rule, Measure, Concept; REALISES, KEEPS, PREREQUISITE_OF, FAILS_WITH, DEPENDS, CAUSES | a DAG of prerequisites and a causal sketch over measures | which rules are foundational; what fails together |
| performance | Performance {accuracies, unit}; RECITED, OF, HOLDS {stretch quartiles}→Rule | a reciter × rule matrix of stretch and pass rate | reciter styles and schools, found without being given |
| timing | RuleLevel {a, b, sd, sd_low}; MULTIPLE_OF {ratio}→madd_tabii | a lattice of multiples of the count unit | how many counts each rule really gets, as a function of tempo |
| skills | Skill; COVARIES {partial_r} | a Gaussian graphical model (partial correlations from the precision matrix) | which skills move together once every other skill is held fixed |
| reliability | Context, Check, Word; FAILS {n, failures, rate} | a bipartite context–check graph with failure rates | where the acoustic model cannot be trusted (blind spots) and what else looks like them |
| sessions | Exercise, ScriptedMistake, Take; SCRIPTS, AT, JUDGED {verdict} | ground-truth labels from a certified reciter | what the engine catches and misses on scripted mistakes |
| ra | RaContext {rule, reference_heavy, heard_heavy, margin_median}; SAID_RA {margin}← Reciter | a partition of all 6,891 ra' instances by classical context | where the classical rule, the reference and the masters agree or not |

Learner datasets (sobolev, RetaSy, Ikhlas outputs) are the next layer, as `(:Learner)-[:RECITED]->(:Performance)`
beside the masters' performances.

## First discoveries (measured; `research_agency_lab/experiments/graph/discoveries.json`)

Every NMI below is tested against a permutation null of 2,000 label shuffles.

**Reciter schools, found without labels.** Each master's style vector has 50 features: the log stretch
and the pass rate per rule, standardised. kNN (k = 5) followed by Louvain splits the 41 masters into 6
communities. The communities carry information about the tiers we assigned by hand (NMI 0.279; null p95
0.228; p = 0.009), but they are not the same partition:
- **The mujawwad style separates by itself.** Abdul Basit Mujawwad, Minshawy Mujawwad and Karim
  Mansoori form a community defined by longer tabii (+2.6 sd), tawassut (+2.8 sd) and silah sughra stretches.
- **Alafasy, Hudhaify, Tablawy and Muhammad Ayyoub** form a group defined by lower munfasil and silah
  kubra pass rates (−2.3 sd and −2.2 sd) and longer ikhfa and leen holds.
- **The Haramain imams and the fast Gulf reciters** (Sudais, Shuraym, Budair, Jibreel, Matroud and others;
  11 in all) form a group defined by shorter nasal holds (ikhfa, iqlab, ikhfa shafawi, ghunnah: −0.8 to −1.0 sd).
- **Husary (all three recordings), Minshawy Murattal, Abdul Basit Murattal, Basfar and Maher** fall in the
  20-reciter main community.

**The text's sound sequencing is not organised by articulation.** Louvain on the PMI transition graph
gives 4 clusters. They agree with the makhraj regions no better than chance (NMI 0.145; null p95 0.228;
p = 0.43), and the same holds for hams/jahr (p = 0.36) and isti'la (p = 0.56). This is a null result:
at this granularity, the sequencing of Quranic sounds cuts across the classical classes. By PageRank,
the vowels, the qalqalah echo and the hidden nun are the hubs.

**Blind-spot families.** FastRP embeddings (64 dimensions) of the contexts where the 41 professionals
fail a check (at least 25% of at least 20 checks) give 277 hot contexts. Linking contexts with cosine
above 0.9 gives these families:
- **Doubled shin, tafashie: 20 contexts, 100% failure.** This is a perception limit, not a recitation fault.
- **Hidden nun (ikhfa), tafkhim and ghunnah.** Separate families form before ف (17 contexts, 46%),
  ق (9 contexts, 77%), ك (8 contexts, 51%), س (6) and ت (9). This is one phenomenon that recurs across
  the next letter, not a set of word-specific errors.
- **Qalqalah release after ط, itbaq: 11 contexts, 73%.** The echo inherits the ط's itbaq label, which
  the model does not hear in the release.
- **A word-final doubled waw, identity: 18 contexts, 56%.**

**Rules.** By PageRank on the prerequisite DAG, the foundational rules are tafashhi, tafkhim/tarqeeq,
madd lazim and madd munfasil. Louvain on the skill partial correlations gives 4 skill families. One
family puts the nasal rules together with tabii and silah sughra; another puts itbaq and qalqalah
together with izhar halqi, madd leen and madd lazim.

**Ayahs by tajwid fingerprint.** Node similarity on the ayah–rule graph finds the ayahs that exercise the
same rule set. For example, 55:39 (the second session exercise) is closest to 47:32 (Jaccard 0.71) and
42:33 (0.63). The ayahs with the most rule kinds are 2:282, 2:233 and 24:33 (15 kinds each), which makes
them candidate exercises.

**Sessions.** The latest takes catch 21 of the 27 scripted mistakes.

## The ra' calculus (`research_agency_lab/substrate_library/julia/ra.jl`)

The dataset is all 6,891 ra' instances: al-Qamar × 10 masters, plus T300. The instances split into 21
classical contexts. The classical rule, the engine's reference and what the model hears agree in every
context with at least 40 instances except two, and those two are the model's least confident:

| context | n | rule | reference heavy | heard heavy | median margin |
|---|---|---|---|---|---|
| all other contexts | 6,810 | — | agrees | agrees | 5.2 – 12.4 |
| a doubled ra' at a stop after a kasra (مُسْتَقِرّ 54:3, 54:38; مُسْتَمِرّ 54:2, 54:19) | 40 | light (sakin ra' after kasra) | 1.00 | 1.00 (10/10 masters) | 2.38 |
| يَسْرِ at a stop (ya' dropped) | 41 | either | 0.00 | 0.39 | 1.0 |

For al-Qamar's shadda ra' at the verse end, the reference and the model both say heavy. The model's
agreement is not independent evidence, though: it was trained on labels from the same phonetizer, and
its margin there is a quarter of the typical margin. Whether the masters make it heavy or light needs a
trained ear on these 40 clips.

## Recipes

```cypher
// what else behaves like ṣirāṭ's ṭā' for tafkhim
MATCH (c:Context {letter: 'ط'})-[f:FAILS]->(:Check {name: 'tafkheem_or_taqeeq'}) WHERE f.n >= 20
RETURN c.key, f.rate ORDER BY f.rate DESC LIMIT 10

// a reciter's school and what defines it
MATCH (x:Reciter {name: 'Husary_128kbps'}) MATCH (y:Reciter {school: x.school}) RETURN collect(y.name)

// every ra' a master said at the end of an al-Qamar ayah, with the model's confidence
MATCH (s:Reciter)-[r:SAID_RA]->(c:RaContext) WHERE r.ayah STARTS WITH '54:' AND r.ayah_final
RETURN c.name, r.word, s.name, r.heard, r.margin ORDER BY r.margin
```
