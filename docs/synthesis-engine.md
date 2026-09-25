# Muqri: the recitation engine, the letter and rule corpus, and the synthesis engine

*A briefing written to be read without the codebase. Every number in it was measured in this project, and it says where. State as of 2026-09-25.*

---

## 1. In one page

**Muqri listens to Qur'an recitation and measures it, letter by letter and rule by rule.** It does more than say whether a word was right. For every letter it reports:
- whether the letter was heard as itself or as a neighbouring letter
- each of its classical characteristics (ṣifāt): voiced or breathy, stopped or flowing, heavy or light, and so on
- how long each madd (vowel lengthening) and ghunnah (nasal hold) was held, in the reciter's own count unit
- how every one of those numbers compares with master reciters

On top of that engine, three new capabilities were built in this phase:

1. **A letter and rule corpus.** Every letter of the Arabic alphabet in every vowel form, and every tajwīd rule in every form, cut from real recitations of several master reciters (plus a learner), cleaned, measured and ranked. It's playable on a web page. The pilot covers 298 cells with about 1,450 instances per reciter. At the measured rate, the full Qur'an for a reciter costs roughly $6 of cloud compute.
2. **A synthesis laboratory.** Controlled edits to real recitation audio, verified by physics and by the engine:
   - rule lengths can already be synthesized precisely
   - whole-syllable edits change exactly the characteristics they should
   - the edit techniques for individual characteristics are mapped and partly proven
3. **Text-to-speech as raw material.** Clear neural Arabic voices (32 free voices, 16 dialects) pronounce the letters and vowels correctly, as the engine measures them. What they lack is the recitation layer: qalqalah, madd lengths, heaviness of the Divine Name, and pace. That's exactly what the synthesis engine adds.

**Why it matters.** Together these make a way to **generate unlimited, precisely labelled recitation data**: every letter and rule, correct or with a known mistake of a known size, in many voices. That is what's needed to train a much stronger engine, to measure it without humans recording mistakes, and to give consumer apps capabilities no current product has. One example is "your qāf is closest to al-Minshawi's; here is his, here is yours, here is exactly what differs."

---

## 2. What the engine does today

### 2.1 From audio to measurements

A recording goes in, together with which verses it should contain. The engine:

1. Runs an open-source acoustic model for Qur'anic Arabic (muaalem v3.2). Every 40 ms it gives probabilities for every phoneme and for 10 characteristic classes.
2. Aligns the expected phonetic text to the audio. Short passages are aligned whole; each verse also gets context from the pauses around it.
3. Tests every letter:
   - **Identity:** is it more likely itself than its classical confusions (ض vs د and ظ, ط vs ت, ح vs ه …) or than nothing at all? The answer is a margin in log-likelihood units.
   - **Makhraj (articulation point):** the letter against every letter at a neighbouring point of articulation. Of the 17 classical points, this gives all 28 letters a competitor, where the classical test covered only 18.
   - **Characteristics:** the 10 model heads (hams/jahr, shiddah/tawassuṭ/rakhāwah, tafkhīm/tarqīq, itbāq, ṣafīr, qalqalah, takrīr, tafashshī, istiṭālah, ghunnah), each as expected against heard, with a margin.
   - **Every vowel:** heard as itself or as one of the other two.
4. Locates every tajwīd rule in the text: madds, ghunnah, iẓhār, idghām, iqlāb, ikhfā', qalqalah, heaviness, the rā' and lām cases, stops. It then judges each one from the audio:
   - lengths in counts of the reciter's own pace
   - nasal holds against bands calibrated from masters
   - the declared munfaṣil reading (tawassuṭ at 4–5 counts, or qaṣr at 2)
5. Compares every number with master reciters: a percentile, and a robust z-score against the masters.

The engine only measures. Deciding what to tell a learner is left to the app that uses it.

### 2.2 Calibrated on master reciters, not textbook limits

With textbook limits, even the reference master (Sheikh al-Husary) scored 71. The limits were re-learned from his full-Qur'an recitation and checked on his peers (by leaving each peer out in turn). On about 115,000 ayah recordings:

| Reciter | Textbook | Calibrated |
|---|---|---|
| al-Husary (Muʿallim) | 75.0 | 96.9 |
| al-Husary (reference) | 71.2 | 95.1 |
| Abdul Basit (Murattal) | 68.9 | 92.5 |
| al-Hudhaify | 69.0 | 90.8 |
| Alafasy | 68.1 | 89.7 |
| al-Minshawi (Murattal) | 61.3 | 87.3 |

Taraweeh imams scored 85–90. One imam's recordings failed alignment and are reported as unscored, not as a low score.

### 2.3 Tested against a certified reader's deliberate mistakes

Across 5 calibration rounds, a certified reader recorded each passage twice:
- **Take A:** to a precise specification.
- **Take B:** with scripted mistakes (one per word). Some are gross, most are subtle: a madd cut to 2 counts, a ghunnah dropped, ḍād read as dāl, ṭā' made light, a qalqalah dropped, a hidden iẓhār.

Every take was then re-scored on the current engine:

| | Result |
|---|---|
| Scripted mistakes caught | **49 of 61 (80%)**; 9/9 and 10/13 in the two latest rounds |
| Take-A expectations met (the right rule, status and length) | **156 of 160 (97.5%)** |
| Words flagged outside the script | 14 across all takes; the reader confirmed one of them was a real, unscripted mistake (an ikhfā' held 1 count) |
| Master recitations (Husary, Minshawy) on the round 5 texts | 52/52 expectations each; 0–3 flagged words per passage |

Engine improvements made this phase, each measured before and after:
- **Short multi-verse submissions are aligned whole.** Before, a slow master's al-Qadr drew 13 false alarms, because one 17 s verse broke a proportional-timing assumption. Now it draws 1.
- **Every verse gets acoustic context from the pauses around it.** This removed false alarms on verse-initial letters.
- **A clear nūn before a throat letter (iẓhār ḥalqī) now has a calibrated length limit.** It's taken from 804 master instances: median 1.29 counts, 99th percentile 1.74, limit set at 2.5. A nūn hidden with a nasal hold (3.62 counts) is now caught.

**What it still misses, and why:**
- **A nasal hum added where there must be none** (idghām without ghunnah). There is no length to measure, and the model hears no nasality on a lām.
- **A sākin ṭā' made light.** Nothing distinguishes it once there is no vowel after it (section 4.2).
- **The length of the very last letter before a stop.** It runs into the silence after it.

### 2.4 Drills, not only verses

The engine accepts any Arabic text, not only verses: letters, syllables, words, rule forms. Proof: a verse passed in as free text gives identical results to the same verse passed by reference. Every letter carries its full declared profile: its makhraj among the 17 points in 5 regions, and all 17 ṣifāt, each marked measured (and by what) or not measurable.

Four ṣifāt have no acoustic test: idhlāq and iṣmāt are classifications, not sounds; līn and inḥirāf have no model head. A 28-letter × 4-form drill round was built (بَ بِ بُ أَبْ for every letter, with 33 scripted subtle mistakes). It's ready to record.

---

## 3. The letter and rule corpus

### 3.1 What it is

A **cell** is one letter in one form, or one rule in one form.

- **Letter forms:**
  - fatḥah, kasrah, ḍammah (short, and carrying a madd)
  - sākin
  - shaddah
  - at a stop

  That's 241 letter cells in the Qur'an.
- **Rule forms:** e.g. qalqalah ṣughrā, kubrā and akbar; ikhfā' light and heavy; idghām with ghunnah, partial and complete; every madd type; every rā' case. That's 59 rule cells.

A planner read all 6,236 verses as text and chose **363 short verses that together hold 5 instances of every cell**. So every reciter is compared on the same words.

For every instance the corpus keeps:
- a clean clip and a raw clip
- where it sits: verse, word, time
- every measurement: identity, vowel, 10 characteristics, makhraj test, or for rules the length and z-score
- a `weakest` score: the check it came closest to failing

Reciters are ranked in every cell.

### 3.2 The pilot, measured

| Source | Instances | Cells |
|---|---|---|
| al-Husary | 1,452 | 298 |
| al-Minshawi | 1,450 | 297 |
| Abdul Basit | 1,447 | 297 |
| Alafasy | 1,449 | 296 |
| as-Sudais (fast reader) | 1,449 | 296 |
| the learner (from their session recordings) | 611 | 192 |

- **Compute:** about 1,560 verses were measured on Modal (a cloud platform) using 10 machines of 16 CPUs each: **183 seconds, $0.52**. The pilot's verses are short (36 phonemes on average; the Qur'an averages about 104). Scaled by length, the full Qur'an costs roughly $6 per reciter: about 3 hours for 5 reciters on 10 machines, about $180 for 30 reciters.
- **Audio cleaning,** designed so nothing it does can change a characteristic:
  - DC offset removal
  - a 60 Hz high-pass filter
  - noise gating learned from the recording's own pauses
  - cuts snapped to zero crossings with 5 ms fades
  - level normalised to −20 dBFS

  No pitch or time change, no compression, no de-essing (it would remove the whistle of ص س ز), no de-reverberation (it smears the burst of a stop).
- **The page `/letters`:** pick a letter or a rule; every form shows each reciter's 5 instances, playable clean or raw. Tap an instance to see its numbers. A ranking shows who reads each cell most clearly.

### 3.3 What the full corpus (every reciter, every letter, the whole Qur'an) would give

The Qur'an has about 330,000 letters, which is about 650,000 measured units per full reading. With 30 reciters that's about 20 million.

- **A reference distribution for every letter × context × characteristic.** Checks move from "the model preferred the right class" to "within the spread of the masters", exactly as was just done for the iẓhār nūn with 804 instances.
- **Separating the letter, the context, the reciter's style and the error,** with mixed-effects statistics. A master's soft final lām is style; a learner's dropped lām is an error.
- **A geometry of pronunciation:**
  - every letter instance as a point in an embedding space
  - a learner's letter placed against the masters' cloud of the same letter (Mahalanobis distance, density)
  - whole distributions compared with optimal transport (Wasserstein distance)
  - the discriminability of every letter pair from real audio
  - the 17 makharij tested against data
- **Reciter fingerprints** and **the closest master for each letter and rule.**
- **Master references for drills:** every بَ each master ever read, in context.
- **Automatic quality control:** outlying reciters or clips expose alignment failures.

---

## 4. The physics and the calculus

### 4.1 What each characteristic is, physically

Each ṣifah has an acoustic signature. These are the quantities the synthesis engine manipulates and the verification layer measures:

| Characteristic | Physical correlate |
|---|---|
| hams / jahr (breathy / voiced) | voicing during the consonant (periodicity, harmonics-to-noise ratio, energy of the voicing bar below 400 Hz) |
| shiddah / rakhāwah / tawassuṭ | a full closure with silence then a burst, against continuous frication, against partial flow |
| istiʿlāʾ, tafkhīm / tarqīq (heavy / light) | the tongue root raised and retracted: F2 of the neighbouring vowel lowered, F1 raised, the F2–F1 gap collapsed |
| itbāq | tongue body pressed to the palate: F2 falls toward F1 while F3 holds; high-frequency energy attenuated |
| ṣafīr | sibilant energy peak at 4–8 kHz |
| tafashshī (spreading, ش) | broad, flat frication spectrum |
| istiṭālah (ض) | lateral frication stretched along the tongue's edge: duration and spectral shape |
| qalqalah | after the closure of ق ط ب ج د, a short voiced release, an echo |
| takrīr (ر) | tap rate; a trill is repeated closures at about 25 Hz |
| ghunnah | nasal murmur: a strong resonance near 250 Hz, an anti-resonance near 750–1,100 Hz, and a held duration |
| madd | duration in counts, where one count is the reciter's own vowelled-letter time |

### 4.2 The central finding: the model hears letters, not characteristics

The lab tested each characteristic by editing it in real master audio and checking two things: the physics (with Praat, independent of the model) and the model's response.

- **Voicing removed** from ز د ج ع ب غ: Praat confirms voicing went from 94–100% to 0–6%. **The model's hams/jahr head still said "voiced" every time.**
- **Voicing added** to ت ف ك (0 / 38 / 31% → 100 / 100 / 62% voiced): nothing moved.
- **Formants moved,** making a letter lighter or heavier: margins fell steadily with strength (س 11.2 → 3.8; د 13.2 → 4.0), but only ص actually flipped (10.9 → −0.5).
- **Nasal resonance** added or removed: modest movement, no flips.
- **Only the consonant's frames replaced** (by a neighbouring letter's): identity held in 4 of 4 cases.
- **The whole syllable replaced** (consonant plus vowel transition, from the same reciter's neighbouring letter): **exact flips**.

| Swap | Result |
|---|---|
| طَ → تَ | heard ت; itbāq lost; heaviness lost |
| ضَ → دَ | heard د; istiṭālah and itbāq lost; heard as "stopped" (as د is) |
| صَ → سَ | heard س; itbāq and heaviness lost; **ṣafīr kept**, because both letters whistle |

So characteristics the two letters share don't move, and characteristics they differ in all flip.

**Interpretation.** The model was trained on correct recitations, and its characteristic labels were derived from the text. So it learned "ز is voiced", not "this sound is voiced". Its characteristic heads largely repeat the letter's identity, and it reads a letter mostly from its transitions into the vowel.

That explains one real miss: a sākin ṭā' made light has no following vowel to carry the change.

**Consequences:**
1. **Synthetic labels must be verified by physics** (voicing, formants, nasal murmur, burst, spectra), not by the current model.
2. **This is exactly why synthetic data is valuable.** A ز whose voicing was removed while everything else stayed intact is an example the current model has never seen. Training on such examples is what makes a future model's heads genuinely hear voicing, heaviness and nasality, independently of the letter.

### 4.3 The calculus: from differences to formulas

The corpus provides **minimal pairs**: letters that differ in exactly one characteristic, spoken by the same reciter.

| Pair | Differs only in |
|---|---|
| ز/س, ذ/ث, د/ت | voicing |
| د/ذ, ت/ث | stop against flow |
| ت/ط, د/ض, س/ص, ذ/ظ | itbāq and heaviness |
| ك/ق | heaviness and point |

For each pair, across reciters and contexts, the lab computes the **delta**: the difference in each physical quantity, as a trajectory over the consonant and its vowel transition, with its mean and its variance. That is a statistical description of "what changes when this one characteristic changes".

Three uses:
- **Synthesis formulas:** a transform that moves a letter along exactly that delta, scaled by a dose. For example, move the F2 trajectory of the vowel onset by the mean ṭ–t difference, and move it by one standard deviation for a "slightly light" ṭ.
- **Verification targets:** a synthetic "light ṭā'" must land inside the natural distribution of tā' on those quantities.
- **Graded mistakes:** a mistake at 25%, 50% or 100% of the natural difference, for training models to detect subtle errors, not only gross ones.

Heavy mathematics runs where it belongs: Julia for optimisation and model fitting, GNU Octave for signal processing (a feature engine already exists there: formants, voicing, harmonics-to-noise ratio, nasal contrast, burst energy, vowel core), and cloud machines for scale.

---

## 5. The synthesis engine

### 5.1 Proven now

**Lengths.** Any madd or ghunnah can be re-timed inside real audio without changing its pitch. The engine measures the result exactly as intended:

| Edit | Before | After | Verdict |
|---|---|---|---|
| ghunnah ×0.4 | 3.07 counts | 1.54 | short |
| ghunnah ×1.8 | 3.07 | 5.00 | long |
| madd ṭabīʿī ×2.0 | 2.06 | 3.43 | long |
| madd munfaṣil ×1.5 | 3.79 | 5.35 | |

That gives graded length mistakes in both directions, and it corrects lengths (a fast reader's clipped madd stretched into the band).

**Syllable substitution.** Replacing a syllable with the same voice's neighbouring syllable creates an exactly labelled letter or characteristic mistake, and the engine reads it as such (section 4.2).

**Pass-through controls.** The vocoder's analysis and resynthesis alone moves nothing (about ±1 margin), so the edits, not the processing, cause the changes.

### 5.2 Proven in the physics, not yet in the model

Using the WORLD vocoder, which splits speech into pitch, vocal-tract envelope and noise, each characteristic has its own control:

| Knob | Effect |
|---|---|
| aperiodicity | breath ↔ voice |
| warping the envelope's formants over the consonant and vowel onset | heavy ↔ light, itbāq |
| a nasal resonance / anti-resonance | ghunnah added or removed |

The physical change is measured to be real (voicing 100% → 0%). As section 4.2 explains, the current model doesn't yet respond to most of them. They are the training data for the one that will.

### 5.3 From TTS to recitation

**What TTS gives.** Four free neural voices read al-Ikhlāṣ, fully vowelled. The engine heard every consonant and every vowel as itself in 3 of the 4 verses, in every voice. What was missing was consistent across voices:
- no qalqalah (every instance absent)
- madds clipped (1.2–1.6 counts against 2)
- the lām of the Divine Name light where it must be heavy
- one verse read as prose, not with the recitation's rules for the start and the stop
- conversational pace (0.16–0.24 s per count, even at −25% rate)

**The pipeline:**
1. **Text:** the verse, fully vowelled, with the recitation's rules computed by the engine's parser (which madd, how many counts, where the ghunnah is, where qalqalah applies, heavy or light).
2. **Voice:** a TTS voice speaks the verse, and also, on demand, any syllable. So each voice is its own donor library.
3. **Alignment:** the engine places every letter and vowel of the TTS audio.
4. **Rule application,** each as a local, measured edit:
   - madds and ghunnahs re-timed to their counts at a tartīl pace
   - qalqalah echoes built from the same voice's syllables (the release of دَ)
   - heaviness by syllable substitution or formant warping
   - stops rendered in their waqf form (the final vowel dropped, a madd ʿiwaḍ where due)
   - global tempo set to tartīl or tadwīr
5. **Verification:** the engine measures the result against the rule targets, and the physics layer checks each characteristic.
6. **Output:** a recitation-like rendering with every rule applied and known, for internal testing and training only.

**Other reciters' voices.** The same edits apply to any master's audio. Voice conversion (kNN-VC, RVC, Seed-VC) can move a performance into many timbres while keeping its rules, multiplying speaker diversity.

### 5.4 Techniques on the roadmap

**Signal processing (Octave, Python):**
- **Pitch-synchronous overlap-add (PSOLA)** and WSOLA for re-timing
- **LPC** analysis and resynthesis for formant surgery
- **Pole/zero insertion** for nasality
- **Burst synthesis and closure insertion** for stop against continuant
- **Amplitude modulation** for trills
- **Spectral shaping** of frication for ṣafīr and tafashshī
- **Boundary refinement** from 40 ms to about 10 ms with spectral flux (the largest single gain in clip quality)
- **Codec-aware cleaning:** prefer lossless sources; per-clip SNR, bandwidth and clipping gates

**Mathematics (Julia):**
- fitting each transform's parameters so the result's physical features land on the natural distribution of the target letter (distribution matching with optimal transport, in the style of Bures–Wasserstein)
- **mixed-effects models** for letter, context and reciter
- **extreme-value tails** (peaks over threshold) for principled pass/fail cuts
- **conformal bounds**
- **dose-response fits** per transform

**Neural methods:**
- **Neural vocoders** (HiFi-GAN, BigVGAN) for higher-fidelity resynthesis
- **Speech inpainting and editing** (VoiceCraft-style) to regenerate one syllable in context
- **Zero-shot TTS** (F5-TTS, XTTS) fine-tuned on recitation, so the voice itself learns madd and qalqalah
- **Voice conversion** for speaker diversity
- **Articulatory synthesis** (VocalTractLab) for the makharij themselves: move the tongue, not the spectrum

**Channel augmentation:** noise, rooms, phone codecs and gain. These must change **nothing**, and they are the engine's robustness test set.

---

## 6. The world this opens

### 6.1 An unlimited, labelled data factory

- Every clean master instance can be turned into a known mistake of a known size: a letter substitution, a characteristic missing, a madd too short or too long by *x* counts, a ghunnah added or dropped, a qalqalah missing.
- Every learner mistake can be turned into its correction.
- Scale: 30 reciters × 330,000 letters × about 10 edits each, plus TTS voices and channel variants, is **10⁸ to 10⁹ labelled clips, hundreds of gigabytes and beyond**. Each carries its label: which characteristic, which direction, how much.

### 6.2 What that powers

**Measuring the engine without human mistake recordings.** Per characteristic, per letter, per reciter: recall (the share of synthetic mistakes caught) and false-alarm rate (flags on untouched controls and on label-preserving channel noise), with confidence intervals. The first design is built: every letter against its neighbours, with same-letter controls.

**Training a new generation of models:**
- heads that hear voicing, heaviness, nasality and articulation point independently of the letter
- dedicated detectors for today's gaps (a nasal hum where none belongs, a light sākin emphatic, the last letter at a stop)
- models that grade mistakes by size, not only yes or no
- robustness to phones, rooms and codecs

**Consumer apps:**
- the closest master per letter and rule, with both recordings side by side
- "hear it from the masters" for any letter form or rule form
- a personal curriculum ranked by distance from the masters, not by pass/fail
- progress measured as that distance shrinking
- choosing a target style (follow al-Husary, or al-Minshawi)
- feedback tied to a named physical dimension (voicing, heaviness, closure, point, length)
- drills in any voice, including the learner's chosen voice, with every rule audibly applied

**Research:**
- the first measured, reciter-by-reciter map of the 17 makharij and 17 ṣifāt
- the variation that belongs to style against the variation that is error
- the laws of tempo (which lengths scale with pace, which do not)
- the same method is transferable to other languages' pronunciation teaching and to clinical speech (articulation disorders)

### 6.3 Position

To our knowledge (this is not a formal market survey), what's publicly available for Qur'an recitation is:
- word-level mistake detection for memorisation
- phoneme-level mispronunciation research on read Arabic
- open acoustic models, including the one this engine builds on

We are not aware of a system that combines:
- letter-level identity, articulation point and characteristic measurement
- rule lengths in calibrated counts at the reader's own tempo
- bands learned from master reciters and validated on their peers
- a per-letter, per-rule corpus of master recitation
- a synthesis layer that creates labelled mistakes and corrections from real and TTS audio

---

## 7. Guardrails

- **Rights:** the recordings belong to their reciters and publishers. Derived measurements serve research; serving cut master audio in a product needs permission. Closest-reciter features can run on numbers and embeddings alone.
- **Synthetic audio stays internal.** It is a measuring and training instrument. It is never presented as anyone's recitation and never published as recitation.
- **Labels are verified by physics** and spot-checked by an expert's ear. Seam controls (same-syllable splices) run in every batch, so no model learns the splice instead of the letter. Training and test data are kept apart by reciter.

---

## 8. Glossary

| Term | Meaning |
|---|---|
| tajwīd | the rules of Qur'anic pronunciation |
| makhraj (pl. makhārij) | point of articulation; classically 17, in 5 regions (the oral cavity, the throat, the tongue, the lips, the nasal passage) |
| ṣifah (pl. ṣifāt) | a characteristic of a letter; classically 17 |
| hams / jahr | breathy (voiceless) / voiced |
| shiddah / rakhāwah / tawassuṭ | stopped / flowing / in between |
| istiʿlāʾ / tafkhīm | tongue-root raising / heaviness |
| itbāq | tongue body pressed to the palate (ص ض ط ظ) |
| ṣafīr | whistle (ص ز س) |
| qalqalah | echo after a stopped ق ط ب ج د |
| takrīr | the rā''s tendency to trill (to be concealed) |
| tafashshī | spreading of the air (ش) |
| istiṭālah | lengthening along the tongue's side (ض) |
| ghunnah | nasal resonance and its hold (ن م) |
| madd | vowel lengthening, measured in counts (ḥarakāt) |
| munfaṣil / muttaṣil | madd before a hamza in the next word / the same word |
| iẓhār / idghām / iqlāb / ikhfāʾ | the four treatments of a nūn sākin: clear / merged / turned to mīm / hidden |
| ḥadr / tadwīr / tartīl | fast / medium / slow recitation |
| waqf | stopping |
