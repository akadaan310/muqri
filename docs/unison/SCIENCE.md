# UNISON/0.1 — General-form science of computational language, indexed dictionaries and Qur'anic rendering

**Owner commission:** Arabic first, every other human language by explicit grammar packs; GLASS register calculus; MUQRI's recitation physics; SURFACE's native observation plane; and address-first data operations.  
**Scientific status:** Mathematical programming specification with bounded initial source modules, NOT a claim that the Qur'an historically encoded these algorithms. No external theory-checking or Google searches are part of this commission.  
**Implementation:** \`unison/\` standard-library Python modular-monolith capabilities, with \`app/\` and existing recitation contracts unchanged.

## 1. A reversible modular monolith, not an anti-microservice slogan

The source is one coherent deployable/research program with strongly typed internal capabilities, stable contracts, and explicit switches: \`legacy\` (default), \`shadow\` (pure capabilities only, original output authoritative), and \`unison\` (explicitly selected). \`UnisonHost.rollback()\` restores dispatch state to legacy, not already executed side effects. Source/version manifests and numeric evidence survive rollback. Do not make a network of microservices a prerequisite for the conceptual system; adapters may later deploy independently without changing types or message semantics.

The engine may "learn" only by ingesting consented and sourced data with explicit provenance; a database registration is not proof of a discovery, pronunciation or universal linguistic rule.

| MUQRI source authority | Responsibility and UNISON adapter |
|---|---|
| \`app/quran_text.py\` | Uthmani reference text, ayah references and edition handling; keep source and basmala provenance |
| \`app/tajweed_rules/parser.py\` | Full contextual word→letter→rule parsing, muqatta'at letter-name readings, wajh and waqf |
| \`app/engine.py\` | Acoustic posteriors, alignment, contextual grading; do not replace with proposed QITṬ rules |
| \`app/letters.py\` | Named articulation locations, phonological classes and legitimate neighbor confusions |
| \`app/measurements.py\` / \`measurements/1\` | Stable contract: letter and rule measurements, uncertainty and calibration |
| \`research_agency_lab/substrate_library/julia/calculus.jl\` | Actual phonological-class lattice / measured residual research, not symbolic extrapolation |
| \`datastore/kg.py\`, \`GRAPH.md\` | Recitation knowledge graph and evidence relations |
| \`unison/registry.py\` | Exact high-order GLASS letter-address lift, typed ṢIFR, projected native weights |
| \`unison/text.py\`, \`unison/ast.py\` | Lossless source bytes and hierarchical indexed document/word/grapheme nodes |
| \`unison/quran.py\` | Fourteen distinct muqatta'at letter identities and optional numeric frames |
| \`unison/dictionary.py\` | Key/root/value/interval/sense indexes; honest heat based on observed counts |
| \`unison/speech.py\` | Explicit language packs, unknown phonemes and speaker/scope operations |
| \`unison/address_ops.py\` | Scoped immutable collection handles, shape/axis sorts, opportunity costs |
| \`unison/surface_scene.py\` | Read-only, source-attributed, separated Qur'an + experimental QITṬ + astronomy scene |
| \`unison/host.py\` | Explicit routing; legacy mode by default, restricted pure-only shadow and rollback |

## 2. The formal calculus

A language-specific UNISON system is described by the typed tuple

\[
\mathscr U_\lambda = (B,\Sigma_\lambda,\mathcal G_\lambda,\mathcal M_\lambda,
\mathcal L_\lambda,\mathcal D_\lambda,\mathcal P_\lambda,
\mathcal A_\lambda,\mathcal O,\mathcal E_\lambda)
\]

where \(B\) is exact source bytes; \(\Sigma_\lambda\) orthographic inventory; \(\mathcal G_\lambda\) grammatical transformations; \(\mathcal M_\lambda\) sourced morphology; \(\mathcal L_\lambda\) lexicon with indexed senses; \(\mathcal D_\lambda\) speaker/discourse roles; \(\mathcal P_\lambda\) pronunciation, dialect and prosody; \(\mathcal A_\lambda\) acoustic evidence or synthesis plan; \(\mathcal O\) typed operations; and \(\mathcal E_\lambda\) versioned epistemic/provenance evidence.

Canonical flow (each arrow has an explicit type, admissibility guard and loss witness):

\[
B \longrightarrow \mathrm{SourceAST}
\longrightarrow \mathrm{Morphology}
\longrightarrow \mathrm{LexicalSense}
\longrightarrow \mathrm{SpeechPlan}
\longrightarrow \mathrm{AcousticObservation}
\longrightarrow \mathrm{SurfaceScene}.
\]

No arrow automatically inverts if it merges information. In particular, a numeric word sum or even a phoneme sequence generally cannot reconstruct a single historical text, inflection, sense, speaker, or recording.

GLASS QITṬ supplies an **orthogonal research view** from a strictly declared 28-letter native register to its integer sum and typed canonical digit tape. Its \(L(p,d)\) and \(V(p,d)=d10^p\) operators are preserved as specified. It does not replace Arabic phonology or the Arabic verbal system.

## 3. مُعْجَم — dictionary as a materialized index and language graph

**Muʿjam** is used as the name for the dictionary operator interface. Distinguish three things: the ordinary Arabic lexical item معجم, a proposed string construction over letters م-ع-ج-م, and a proven morphological/etymological derivation. The prefix-Mīm explanation is a **proposed string grammar**, not evidence of origin.

Example GLASS native values in the bare-grapheme research profile:
- عجم = 70 + 3 + 40 = **113**.
- معجم = 40 + 70 + 3 + 40 = **153**.
- جيم = 3 + 10 + 40 = **53**.
- ميم = 40 + 10 + 40 = **90**.
- لام = 30 + 1 + 40 = **71**.

The fascinating **final-Mīm** pattern does include جيم and ميم; it is **not exclusive** to those two, since لام also ends in م. The correct scientific object is an explicit terminal-letter predicate over a source-defined letter-name dictionary. A glyph's drawing, a stroke decomposition, and a rotated image of a glyph are **different** from its spelling and require a real, versioned font/geometry grammar before any pattern is admitted.

A dictionary entry is a record \((id,\lambda,era,dialect,lemma,POS,root?,sense?,gloss,phones?,source,frequency?)\). Indexes: \(\mathrm{key}=(language,NFCcasefold(lemma))\), root index, native register value index for supported Arabic forms, sorted numeric interval index, manually or corpus-attested **sense ID** cross-language index, and observed heat/frequency groups. Their access methods are:
- hash exact key/root/value: **expected \(O(1)+O(k)\)** to produce \(k\) returned records; hashing/key normalization/index construction are separate costs.
- ordered interval: **\(O(\log n+k)\)** with a maintained sorted key.
- numeric collisions: two or more different words may share a value; numeric equivalence is not semantic equivalence.
- cross-language semantic links: only entries with an attested common sense ID.
- multidialect/era views: a grammar+lexicon family with independently versioned sound/meaning profiles, not a wholesale modernization of classical text.

The phrase "**10–70 languages at a time**" becomes a user-visible **parallel language-lane capacity**. Actual lexical, morphological and speech quality in each lane requires a fitted/sourced pack. The system must render UNKNOWN rather than hallucinating a Turkish word, phoneme string or historical Arabic meaning.

**Heat** is a measured indexed access-frequency or corpus occurrence count with source, sampling window and uncertainty. In a brand-new unsampled dictionary heat is UNKNOWN, not zero and not fabricated “importance.” Content-based recommendations must not guess an individual's age, identity, psychological condition or capacity from language/behavioral stereotypes.

## 4. Indexed AST vs an unindexed story

The syntax tree has typed hierarchical nodes \`Document→Word/Space/Punctuation→GraphemeClusters\` with:
- immutable raw Unicode source and exact UTF-8 byte spans;
- stable source-derived node IDs;
- parent and child IDs;
- lexical and morphology annotations in separate layers;
- a materialized node-ID hash map enabling **expected O(1) retrieval AFTER construction**.

AST *construction* and general *evaluation* are not O(1). A transformation or a search over an unindexed AST can remain O(n). The syntax tree becomes a computational oracle only when it accepts constrained questions, returns referenced source locations, supplies a choice/cost model, and never silently swaps source representation.

Qur'anic Arabic is one profile; Ottoman Turkish, modern Turkish, English, Jordanian colloquial Arabic, Medieval/early Islamic Arabic etc. require **separate** phonologies and source evidence. A 48-form conjugation grid is allowed as an explicitly declared **profile-generated paradigm**, but no universal “48 Arabic forms” theorem is assumed. Enumerated person/number/gender/voice/tense/aspect/mood combinations depend on language, dialect and the stated counting convention.

## 5. The Qur'an reader and the 14 disconnected-letter identities

MUQRI's existing parser registers the following 14 initial-form *sequences* as written units:

\[
\{\text{الم},\text{المص},\text{الر},\text{المر},\text{كهيعص},\text{طه},
\text{طسم},\text{طس},\text{يس},\text{ص},\text{حم},\text{عسق},\text{ق},\text{ن}\}.
\]

The **union of their distinct letters** is exactly the 14-identity set:
\(\{\text{ا،ل،م،ص،ر،ك،ه،ي،ع،ط،س،ح،ق،ن}\}\).
Separate the number of distinct letters from the number of sequence forms and from the number of surahs containing them. Surah/ayah boundaries are sourced, not guessed from a sequence or typography: for example, حم and عسق can occupy separately numbered verses in a given edition.

**Initials are read as their named letters**, not an ordinary unvowelled lexical word. MUQRI already has \`_LETTER_NAMES\`, tajwīd duration rules and source/waqf context. UNISON's optional \`classify_initials\` only produces the QITṬ mathematical projection after an explicit \`verse_initial_candidate\` role and recognized sequence; actual phonemes and measured durations are still delegated to MUQRI. The frame for **الم** carries:
- exact source "الم";
- letter-names Alif, Lām, Mīm and MUQRI's context-aware phonetic target;
- QITṬ native-weight sum **1 + 30 + 40 = 71**, clearly marked as an experimental supplementary view;
- byte-addressed source, version, corpus reference and speaker/voice/reading provenance.

Other sample native register projections:
- الر = 1 + 30 + 200 = **231**.
- المر = 1 + 30 + 40 + 200 = **271**.
- المص = 1 + 30 + 40 + 90 = **161**.
- حم = 8 + 40 = **48**.
- عسق = 70 + 60 + 100 = **230**.
- كهيعص = 20 + 5 + 10 + 70 + 90 = **195**.
- الرقيم = 1 + 30 + 200 + 100 + 10 + 40 = **381**, **not** a disconnected-letter token; ordinary lexical/morphological analysis applies.

These sums are **well-defined inside the GLASS profile**, but they are not established codes for hidden speech, celestial bodies, dates or speaker identity. A root, an identical letter sequence and a related meaning are different relations: الر, الرقيم, الرعد, رقيم and رق are not automatically one lexeme or one program.

**Rasm and recitation** must be independent observation planes:
1. Exact Qur'anic text including orthographic marks, surah/ayah, edition and qirāʾah.
2. A separately declared *noncanonical simplified skeleton* from removing Unicode marks for research, never overwriting the Uthmani Rasm.
3. Sourced root/morphology and word-sequence graph.
4. Speech action and quoted-speaker graph, e.g. قال / قال قائل / speakers and turn annotations.
5. Muhqatta'at letter-name pronunciation, specific to the source/reading.
6. GLASS register plane, numeric and cyclic states separately from sound.
7. MUQRI measured phonological features and audio only when authentic data exists.
8. SURFACE scene projection retaining all source IDs and loss declarations.

**Basmalah** requires an explicit textual state: e.g. a surah-opening invocation versus an embedded quoted formula (such as within the letter of Sulaymān in Sūrat al-Naml) are different discourse/verse roles even when spellings overlap. Do not assume a fixed verse-number treatment independent of edition and recitation convention.

## 6. QAWL: speech acts and non-switching as a guard, not a lexical shortcut

A formal discourse event is \((speaker,utterance,scope,operation,source)\). Operations are SPEAK, CONTINUE, SWITCH and QUOTED_SPEECH. \`CONTINUE\` is admitted only if a prior, matching speaker is established. \`SWITCH\` requires a distinct identified next speaker. A quotation without a sourced speaker remains uncertain.

Thus a repeated قال or قائل may be used to *ask whether* the speaker remains constant in a particular Qur'anic passage; its mere spelling cannot prove no speaker switch. Similarly a geometric letter shape and its consonant's sound do not intrinsically encode an agent identity.

The engine may generate educational conversational narratives (calling a friend, a bar bill, ordinary problem solving) with a role-aware grammar and user-controlled tone, but it cannot infer hidden user identities, years of birth, ages or intentions without evidence.

## 7. Address-first computation and algorithmic opportunity cost

"Give me the address of the array" is implemented as a typed, **authorized collection handle**, not an arbitrary operating-system virtual address. The operation is:

\[
\operatorname{PLAN_SORT}(h, axis,policy,scope)
\to\{(candidate,cost,preconditions,information\ loss)\}.
\]

A scoped in-memory snapshot store issues opaque handles and content digests. A 4D tensor has an explicit shape and sorting axis. Sorting the last scalar axis differs from reordering higher-dimensional slices according to the sum of all elements. Sorting by itself is **not injective**, but the old immutable handle preserves the original state and may be used to undo a change at the *session state* level.

Typical stable comparison sorting is \(O(n\log n)\); binary search on a sorted index is \(O(\log n)\), not O(1). Hash lookup is expected O(1) for an existing key index. An in-memory address is not portable across processes, devices or sessions; capability/scopes/versioned logical IDs are.

**No universal free lunch:** one system cannot sort a hidden dataset merely from a pointer string without access to its data, shape, type, comparator and authorized memory scope. A cost oracle must disclose whether it is materializing an index, changing data, or answering a query.

## 8. Contiguity and continuity are distinct calculi

Memory contiguity means physical layout addresses have a stated stride:
\(a_{i+1}=a_i+s_i\), **when the memory model actually guarantees it**.
Logical contiguity of events means consecutive index or sequence labels under an ordering. Adjacency of lexical AST nodes is a graph relation; two consecutive word IDs are not automatically adjacent physical memory values.

Mathematical continuity requires a topology and \(\lim_{x\to a}f(x)=f(a)\). A finite list, even stored in contiguous RAM, is not automatically a continuum; a wave over continuous time may be sampled as contiguous discrete storage. Sampling adds quantization/aliasing limits.

This yields a typed two-way bridge:
\[
\operatorname{continuous\ physical\ model}
\xrightarrow[\Delta t,quantizer]{sample}
\operatorname{discrete\ samples}
\xrightarrow[\mathrm{interpolation\ rule}]{reconstruct}
\operatorname{approximate\ model}.
\]
Neither arrow is a universal inverse without bandwidth, regularity and error assumptions.

## 9. A seven-axis HyperRuler and the astronomical observer

SURFACE supports independent physical, logical, observer and typography layers. A **HyperRuler** here is a declared coordinate chart of 1–7 axes with named units. It is a research interface for navigating parameter spaces, **not** evidence that the physical universe has exactly seven spatial dimensions or that projecting seven axes to two is injective.

A minimal mechanical wave reference is \(x(t)=A\cos(2\pi f t+\phi)\), with \(A\) in metres, \(f\) in hertz, \(t\) in seconds and phase in radians. Real structural vibration needs material data, boundary conditions, damping, numerical solver and measured comparisons; a simple sine is a **model**, not a measured field.

The astronomy observation plane may display stars (catalogue ID, RA/Dec, frame, epoch, survey source and uncertainty). It is NOT a literal decoding of Qur'anic wording (e.g. \`فلا أقسم بمواقع النجوم\`) into coordinates. Some astronomical objects have measured pulsar timing; most ordinary stars do not have an observed pulsar rate. Current SURFACE evidence identifies a **10,000-source Gaia directional slice**, not a live 1.6-billion-star physical simulation. Large sky data require offline indexed tiles/LOD, validated coordinate/time transformations and carefully budgeted render batches. A deterministic Surface clock is a replay device clock over admitted initial conditions, **not** a deterministic all-universe observer.

Classic bits, addresses, Q-Strings, qubit basis labels, quantum amplitudes and measurement outcomes are distinct types; the fact that physical simulation and quantum simulation are stored on binary hardware does not prove their conceptual equivalence. A legitimate hybrid automaton carries classical control, quantum state, instrument, physical parameters and separately clocked outcomes.

## 10. Discrete anti-calculus, sequences and series

The **Fibonacci numbers** \(F_n\) are individual terms of the **Fibonacci sequence** \((F_0,F_1,\dots)\) where \(F_0=0,F_1=1,F_{n+2}=F_{n+1}+F_n\). A Fibonacci **series** is an infinite sum \(\sum_{n\ge0}F_n\), which diverges in the ordinary real-number sense. Its **partial sums** are \(S_n=\sum_{k=0}^n F_k=F_{n+2}-1\).

Let the discrete difference be \((\Delta S)_n=S_n-S_{n-1}=F_n\). Summation reconstructs a partial-sum sequence from terms given an integration constant/initial condition, a useful discrete “anti-calculus” analogy. Factorization \(3\cdot6=18\) is a property of arithmetic operands; recording intermediate bit/carry/wave trajectories requires a separately declared execution or physical model. No claim that the uncertainty principle literally factors a product is supported by this algebra.

## 11. SURFACE observer, Glass super-OS and permissions

A \`surface.unison.quran-scene/0.1\` contains a versioned exact Qur'anic source, word-node address spans, recognized muqatta'at only under declared context, pronunciation marked DEFERRED, GLASS numeric overlays optional, and astrophysical observations independently sourced. The companion SURFACE implementation provides a **standalone offline HTML review renderer and a native C++ typed scene carrier**. It does NOT yet paint Arabic on the actual native Android Surface screen; shaping-capable Android glyph runs and lifecycle admission remain gated.

GLASS as a "super-OS" can be specified as a capability-governed orchestration/launcher layer. Allowed first-party/official interfaces include launcher grouping and shortcuts with user-selected permissions, accessible settings actions and documented app intents. Root/jailbreak mechanisms can be modeled for security research but are not assumed to confer permission to alter another app or the operating system. iOS application execution needs a compatible authorized runtime/simulator and licensed binaries; rendering a screenshot/scene is not the same as executing the application.

To make a unique GLASS installation, define **a signed installation identity, immutable state roots and scoped capability grants**; mathematical structures and executable algorithms remain reproducible and cloneable as such.

## 12. Gate matrix and status

| Gate | Requirement | Present status |
|---|---|---|
| U01 | 28-register QITṬ exact encoder/decoder | source + implementation specified; test fixtures authored |
| U02 | Typed ṢIFR, lift, shadda alternate profile | implemented in \`unison/registry.py\`; tests authored |
| U03 | Source-preserving indexed AST, correct spans | implemented; actual performance claim limited to indexed hash lookup |
| U04 | Fourteen-letter muqatta'at inventory and exact reading boundary | structural adapter implemented; recitation delegated to existing MUQRI |
| U05 | Muʿjam indexes (key/root/value/interval/sense) | finite in-memory implementation; language corpus still to acquire |
| U06 | Multilingual speech | explicit pack interface; pronunciation UNKNOWN without pack |
| U07 | QAWL speaker-state guards | typed finite control in \`unison/speech.py\` |
| U08 | 4D collection address, cost proposals and immutable sorting | bounded in-process implementation |
| U09 | Static Qur'an visual review in SURFACE | stand-alone HTML generator and fixture; not native Android |
| U10 | Native Android shaping, interactive scene switching, runtime events | not implemented |
| U11 | Authenticated user permission app/OS control | research contract only |
| U12 | Real stellar ephemerides and pulsar pulse-to-observable adapter | not implemented |
| U13 | Physical/quantum hybrid, seven-axis chart verification | bounded scalar reference and chart types; full project not implemented |
| U14 | Independent tests / machine checked scientific results | software tests authored; not executed by this publication run |
| U15 | Arabic historical hypothesis and hidden Qur'anic semantics | proposal only; no external theory testing commissioned |

## 13. Exact source anchors

- [MUQRI](https://github.com/akadaan310/muqri): \`app/engine.py\`, \`app/tajweed_rules/parser.py\`, \`app/letters.py\`, \`app/measurements.py\`, \`docs/synthesis-engine.md\`, \`GRAPH.md\`.
- [GLASS](https://github.com/akadaan310/GLASS): \`docs/QITT_CYCLIC_ARABIC_REGISTER_CALCULUS.md\`, \`atlas/qitt-register-v0.1.json\`, \`docs/QITT-0.1.ebnf\`.
- [SURFACE](https://github.com/akadaan310/SURFACE): \`docs/ARCHITECTURE_AND_HARNESSES.md\`, \`docs/RENDERER_EVOLUTION.md\`, \`docs/MATHEMATICAL_ATLAS.md\`, \`src/renderer.cpp\`, \`include/surface/quran_scene.hpp\`.

**No Google/external theory confirmation or source corpus enlargement.** Owner statements are treated as research hypotheses, with ordinary mathematics derived from explicit definitions and verification gates stated rather than presumed achieved.
