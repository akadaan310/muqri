# The letters corpus: every letter and rule, in every form, for every reciter

This document covers what the corpus is, how it is built, how its audio is cleaned, and how it becomes a generator of labelled synthetic data. It also says what has been measured so far and what that costs.

Code is in `research_agency_lab/experiments/letter_corpus/`:

| File | What it does |
|---|---|
| `plan.py` | chooses the verses |
| `measure.py` | runs the engine locally |
| `research_agency_lab/compute_bridge/modal_corpus.py` | runs the engine on Modal |
| `assemble.py` | cuts, cleans, scores and ranks |
| `clean.py` | the audio chain |
| `perturb.py` | the perturbation lab |

The page is `/letters` on the engine's web app. Audio and measurements stay local in `letter_corpus/data/`, which is gitignored. The recordings belong to their reciters and publishers.

## 1. What a corpus row is

A **cell** is one letter in one form, or one rule in one form.

- **Letter forms:**
  - fatḥah, kasrah and ḍammah (short)
  - the same three carrying a madd (`*_long`)
  - sākin (a consonant follows)
  - shaddah
  - at the stop (the ayah's last letter)

  With 28 letters that makes **241 letter cells** that occur in the Qur'an.
- **Rule forms:** the parser's rule type and detail. Examples: qalqalah ṣughrā, kubrā and akbar; ikhfā' light and heavy; idghām bi-ghunnah nāqiṣ and kāmil; every madd; every rā' case. That makes **59 rule cells**.

An **instance** is one occurrence of a cell in one recording. For every instance the engine keeps:

- **Where:** onset, duration and frames. The clip is cut from the source audio with 50 ms of context either side, because a letter is co-articulated and has no clean edge.
- **What:** the letter, its form, the letters around it, the word, and the ayah.
- **What was measured:**
  - identity, as a CTC margin in nats against the classical confusions and against deletion
  - all 10 characteristic heads: hams/jahr, shiddah/tawassuṭ/rakhāwah, tafkhīm/tarqīq, itbāq, ṣafīr, qalqalah, takrīr, tafashshī, istiṭālah, ghunnah; each as expected, observed and margin
  - the vowel's identity
  - the makhraj test: the letter against every letter at a neighbouring articulation point (`app/letters.py`)
  - for rules: status, length in counts against the band, and z against the masters
- **Declared but not measured:** idhlāq/iṣmāt, līn and inḥirāf. `app/letters.py` says why for each.

Each letter instance has two scores:
- **`weakest`:** the lowest margin among all the checks the engine scores on it, meaning the check it came closest to failing.
- **`clean`:** whether every check held.

Reciters are ranked per cell: letters by the median `weakest`, and rules by pass rate, then by median |z| against the masters.

## 2. Choosing the audio: every cell from as little as possible

`plan.py` reads all 6,236 ayahs as text (phonetiser plus rule parser; about 4 minutes on one CPU). It lists every cell, then greedily picks ayahs by the number of still-needed instances per phoneme, until each cell has 5.

Result (`plan.json`):
- **363 short ayahs, 13,121 phonemes**, cover all 300 cells.
- 10 cells occur only in longer ayahs and have 1–4 instances. Examples: hamza at a stop (3), and rā' sākin after a temporary kasrah (3).

Every reciter reads the same ayahs, so each cell compares the same text positions across reciters.

## 3. Measuring: local and on Modal

| Run | Measured |
|---|---|
| Local, 3 processes | about 5–15 s per ayah; 1,815 jobs would take about 70 min |
| Modal (`modal_corpus.py`): 10 containers (the free tier's cap) × 16 CPU × 32 GiB, 8 engine processes each, model baked into the image | **1,562 ayahs in 183 s; billed $0.52**; 1 ayah failed (Abdul Basit 17:24, a decode error) |

For the reciters in this pilot:

| Reciter | Role |
|---|---|
| Husary | master |
| Minshawy (Murattal) | master |
| Abdul Basit (Murattal) | master |
| Alafasy | master |
| Sudais | the fast (ḥadr) reciter |
| the learner | latest take A of each session exercise |

**Scaling.** The measured rate is $0.52 for 1,562 short ayahs (36 phonemes on average). The Qur'an averages about 104 phonemes per ayah, 2.9× longer. Scaled by length, a full-Qur'an pass costs roughly $6 per reciter, which is about $180 for 30 reciters, and 5 reciters take about 3 hours on 10 containers × 80 processes. The index for 30 full readings is about 20 million units:
- JSON: about 20 GB
- Parquet: about 3–5 GB
- 256-dimensional float16 embeddings per unit: about 10 GB

Store the index, and cut clips on demand. Every clip as its own WAV would be about 140 GB.

## 4. Cleaning a letter's audio

The implemented chain (`clean.py`) runs on the whole ayah, then cuts, because a filter or noise estimate on a 0.2 s clip has nothing to work with:

1. DC offset removal.
2. High-pass filter: 4th-order Butterworth at 60 Hz, zero-phase (`sosfiltfilt`). It removes rumble, handling noise and mains hum; the lowest voice fundamental of these reciters (about 80 Hz) stays.
3. Spectral gating:
   - The noise floor per frequency bin is learned from the recording's own quietest 10% of frames (its pauses and breaths).
   - Each STFT bin gets a Wiener-style gain, max(1 − 1.5·N/|X|, 0.1), smoothed over 3 frames so it doesn't warble.
   - Hiss and room tone go; weak parts of a letter keep their shape: the release of a hams letter, the qalqalah echo.
4. Cut points snapped to the nearest zero crossing within 2 ms, with 5 ms raised-cosine fades. No clicks.
5. Loudness: RMS normalised to −20 dBFS, peak held under −1 dBFS. Reciters are compared on what they recite, not on how loud their microphone was. For example, Husary's qāf with fatḥah comes out at 0.38 s and −20.0 dBFS.

**Deliberately not done**, because each could move a characteristic the engine measures:
- no pitch or time change
- no dynamic compression
- no de-essing (it would remove the whistle of ص ز س)
- no de-reverberation (it smears the burst of a stop)
- no neural enhancement

**Further techniques, in order of value:**

- **Boundary refinement.** Frames are 40 ms and CTC is peaky, so some letters get one frame, final letters at a stop especially. Refine each edge within ±40 ms to the nearest spectral-flux peak or energy minimum. Or re-align at 10 ms hop with a forced aligner on the 16 kHz waveform. This is the largest single gain in crispness.
- **Per-clip quality metrics as gates.** Local SNR against the ayah's noise floor, clipping ratio, estimated bandwidth (MP3 at 64 kbps cuts around 11 kHz, which removes the top of ص and س), and alignment confidence. Rows that fail a gate are kept but flagged, not deleted.
- **Better sources first.** Prefer WAV or high-bitrate recordings (Quran-MD WAVs, 192 kbps EveryAyah folders) over 64 kbps MP3. Codec pre-echo smears stops and bursts, and no filter restores what the codec removed.
- **Denoisers.** Log-MMSE or OM-LSA estimators (smoother than spectral subtraction, less musical noise) for noisy learner recordings. WPE dereverberation (`nara_wpe` is already in the Kaggle bundle) only with a before/after check that the engine's margins on clean masters don't move.
- **Hum notch at 50/60 Hz and harmonics**, only when a spectral peak is detected. The high-pass already removes the fundamental.
- **Neural enhancers (DeepFilterNet, Demucs):** only as a separate, clearly marked variant. They can invent or erase exactly the fine detail being measured (a qalqalah echo, a hams release). Any use needs the same margin-invariance check on master clips.
- **Level matching per cell:** normalise to the median loudness of the masters' instances of that cell, so A/B listening compares like with like.

## 5. The perturbation lab: synthesising labelled data from real audio

`perturb.py` works like this:
1. Take a master's letter or rule inside its own ayah.
2. Alter that span, with 10 ms crossfades at the seams.
3. Run the whole engine on the altered ayah.
4. Record which measurements moved.

If a controlled change moves exactly the measurement it targets, and nothing it doesn't, then:
- (a) the engine is shown to hear that characteristic, with a measured dose-response;
- (b) the change is a label-generating machine: every clean master instance becomes a known mistake, or run the other way, a learner's mistake becomes a corrected reading.

### 5.1 What the first cases measured (Husary, all numbers from `data/perturb.json`)

**Lengths respond precisely:**

| Perturbation | Result |
|---|---|
| madd ṭabīʿī ×0.5 / ×2.0 | 2.06 → 1.50 counts (pass) / 3.43 (long) |
| madd munfaṣil ×0.5 / ×1.5 | 3.79 → 2.24 / 5.35 counts |
| ghunnah (with its iqlāb) ×0.4 / ×1.8 | 3.07 → 1.54 (short) / 5.00 (long) |

**Changing only the consonant's frames barely moves the characteristic heads:**
- **Qalqalah echo faded to −30 dB:** on د the qalqalah margin went 8.4 → 2.2, still realised. On ق and ب nothing moved.
- **Devoicing ز or ذ with a 1.5 kHz high-pass:** the hams/jahr head did not flip.
- **Formant shifts of ±2 to ±4 semitones on ص, س, ط:** every head's margin fell together (the audio is less natural), but heaviness never flipped.
- **ل sākin cut out:** identity still confirmed (16.9 → 10.6).
- **Consonant-only swaps for the same reciter's neighbour** (ط→ت, ض→د, ص→س, ق→ك): identity held in all four.

**Whole-syllable swaps flip exactly the right things.** Here the consonant and its vowel together are replaced by the same reciter's neighbouring syllable:

| Swap | What flipped | What stayed |
|---|---|---|
| طَ→تَ | heard ت; itbāq → open; tafkhīm → light; hams/jahr flipped; itbāq and tafkhīm rules → wrong | — |
| ضَ→دَ | heard د; istiṭālah lost; itbāq lost; heard shadīd (as د is); light | hams/jahr stayed jahr (both are) |
| صَ→سَ | heard س; itbāq and tafkhīm lost | **ṣafīr stayed realised** (both whistle) |
| قَ→كَ | heard ك; hams gained (ك is a hams letter); tafkhīm → light | — |

So the characteristics the two letters share stay, and the ones they differ in flip.

**The finding.** The model decides a letter mostly from its transitions into and out of the vowel, its coarticulation, not from the consonant's own frames. Two consequences:

1. **Synthetic mistakes must be built at the syllable level:** the consonant plus its transition, taken from a real reading of the target letter. Consonant-only DSP edits don't produce the labelled change.
2. **The same fact explains a real miss.** In round 5 the learner read مَطْلَعِ with a sākin ṭā' made light. It was not caught: a sākin letter has no following vowel of its own to carry the change.

The control case (the same syllable from another ayah of the same reciter) checks that the swaps are heard for the letter and not for the seam. It is in `perturb.py`, and its result is in `data/perturb.json`.

### 5.2 The perturbation catalogue

Each row gives the target, the generator, and the expected engine response.

| Target | Generator | Engine response |
|---|---|---|
| identity / makhraj | syllable swap with the same reciter's neighbour at an adjacent point (`letters.NEIGHBOURS`) | identity heard as the neighbour; the makhraj test flips |
| hams ↔ jahr | syllable swap within a voicing pair: ز↔س, ذ↔ث, د↔ت, ظ↔ث (heavy) | hams_or_jahr flips, other heads hold |
| shiddah ↔ rakhāwah | ج↔ش, د↔ذ, ت↔ث; also the closure gap removed (splice out the silent closure) | shidda_or_rakhawa |
| tafkhīm ↔ tarqīq | ص↔س, ط↔ت, ق↔ك, ظ↔ذ, and the heavy/light rā' of one reciter (رَ↔رِ) | tafkheem_or_taqeeq, the tafkhīm/tarqīq rules |
| itbāq | ط↔ت, ص↔س, ض↔د, ظ↔ذ | itbaq |
| ṣafīr | ص/س/ز ↔ ث/ذ (the whistle ↔ the lisp at makhraj 14) | safeer |
| qalqalah | echo removed; or a qalqalah syllable (أَقْ) swapped for the same letter held (أَقّ) | qalqla, the qalqalah rule |
| ghunnah | nasal hold compressed or stretched; idghām with ghunnah ↔ without (splice the reciter's own يَ with and without the nasal onset) | the ghunnah band; ghonna on the merged letter |
| takrīr | a single-tap rā' ↔ a trilled one (from reciters who trill) | tikraar |
| tafashshī / istiṭālah | ش↔س; ض↔د (as above) | tafashie, istitala |
| length, every madd | phase-vocoder stretch ×0.5 to ×3 in steps; WSOLA as a check | counts; short/long, with a dose-response curve per madd type |
| tempo | the whole ayah stretched ×0.7 to ×1.4 | each count unit must stay the same in counts. This is the tempo invariance the stretch calculus claims |
| vowels | a fatḥah syllable swapped for the same consonant with kasrah or ḍammah; a short vowel stretched into a madd (ishbāʿ) | vowel identity; the itmām measures |
| deletion / insertion | a syllable removed; a syllable duplicated (a stutter) | identity ∅; alignment |
| improving | the other direction on learner audio: a short madd stretched into the band, a learner's syllable replaced by a master's | the verdict turns to pass; the page's "improving" case |
| channel robustness (label-preserving) | noise at 0–30 dB SNR, room impulse responses, phone and codec simulation (AMR, Opus 8–24 kbps), gain, clipping | **nothing** must move: the engine should be invariant |

### 5.3 From the lab to a data factory

**Scale.** Take the full corpus: 30 reciters × 330,000 letters × about 10 perturbations each, plus channel augmentation. That is 10⁸ to 10⁹ labelled clips. At about 30 KB per 1 s clip, that is **hundreds of GB of synthetic audio, 300 GB and more**. Every clip carries its label: which characteristic, which direction, how much.

**Uses:**
1. **Measuring the engine, with no human recordings.** For every characteristic: recall (the flip rate on true swaps) and false-alarm rate (the flip rate on same-letter controls and on label-preserving channel perturbations). Per letter × form × reciter, with confidence intervals. This replaces "a person recorded 4 mistakes" with thousands per cell.
2. **Training.** Fine-tune the acoustic model with syllable-swap negatives, which are hard negatives at exactly the confusions that matter. Train dedicated detectors for the gaps sessions exposed: a ghunnah added to idghām without ghunnah, a sākin emphatic made light, a final letter at a stop. Train contrastive letter embeddings: the same letter across reciters as positives, neighbour-point letters as negatives.
3. **Calibrating every band from data**, like the iẓhār nūn ceiling (804 master instances → 2.5 counts) but for every letter × context × head.

**Checks the factory needs:**
- **Seam controls in every batch** (same-letter swaps). A model trained on synthetic data can learn the splice instead of the letter. Also randomise crossfade length and position, and add splice-only augmentation to the positives.
- **Donor matching:** the same reciter, the same vowel, a similar tempo and pitch. Report the donor's own margins, since a weak donor makes a weak label.
- **Listening spot-checks:** a random 1% of each perturbation type, labelled by an expert through `/review`. The labels file is already the ground truth the engine is scored against.
- **Keep train and test apart:** by reciter (never train and test on the same reciter's synthetic data) and by ayah.
- **Label-preserving perturbations are the invariance test set.** If noise or a codec moves a verdict, that is an engine fault, not a mistake.

## 6. What the corpus enables (summary)

- **A reference distribution for every letter × context × head** (quantiles, robust z, conformal bands). Checks move from "margin > 0" to "within the masters' spread".
- **Separating letter, context, reciter style and error** with a mixed-effects model: which parts of a reading are the letter, which are the reciter's school, and which are mistakes.
- **A geometry of letters:**
  - distance from a learner's letter to the masters' cloud (Mahalanobis distance, density)
  - Wasserstein distance between a learner's and a master's whole distribution of a letter
  - the discriminability of every letter pair from real audio
  - the 17 makharij tested against the data
- **Reciter fingerprints and the closest reciter per letter or rule** (a consumer feature: "your qāf is closest to al-Minshawi's").
- **Master references for drills:** round 6 had no master baseline. The corpus holds every بَ each master read, in context.
- **Quality control:** outlying reciters or clips expose alignment failures automatically (the al-Juhany case).

## 7. Rights and respect

- The recordings belong to their reciters and publishers. Research use of derived measurements is one thing; serving cut master audio in a consumer app needs permission.
- The closest-reciter feature can run on features and embeddings alone.
- Spliced and perturbed audio is an internal measuring instrument. It must never be presented as a reciter's recitation, published as recitation, or leave the research store.

## 8. Next steps

1. **Sweep (`perturb.py --sweep`):** every letter × its neighbours × 5 instances, with controls. This gives the recall and false-alarm table per characteristic. About 1,000 engine runs; minutes on Modal.
2. **Boundary refinement** (section 4) before any clip is used for training.
3. **Full-Qur'an pass** for the studio reciters on Modal (about $2 per reciter at the measured rate), writing Parquet plus embeddings.
4. **The embedding index** (faiss is already in the dependency bundle) for closest-reciter search.
5. **Record round 6 (the letter drills)**, and put the learner's drill takes into the corpus as the first learner reference set in every letter × form.
