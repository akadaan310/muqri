# Muqri: what the platform does, and the synthesis engine it is building

*A briefing on the platform's power and direction, written to be read without the codebase.*

---

## 1. The platform in brief

Muqri is a measurement engine for Qur'an recitation. Give it a recording and the text that should be in it, and it returns a precise account of how every letter and every tajwīd rule was realised. It measures; it doesn't prescribe. The apps built on it decide what to teach.

On that engine the project is building three things:

- **A letter and rule corpus:** every letter of the alphabet in every form, and every tajwīd rule in every form, as recited by master reciters and by learners, cut out, cleaned, measured and comparable.
- **A synthesis engine:** the ability to produce any letter, any characteristic and any rule on demand, correct or deliberately wrong by a controlled amount, in any voice.
- **TTS to recitation:** plain Arabic text-to-speech in dozens of voices, turned into recitation with every rule applied.

Together these make a closed loop. The engine measures, the corpus supplies the reference, and the synthesis engine generates unlimited, precisely labelled audio. That audio tests the engine and trains the next, stronger one.

---

## 2. What the engine measures

### 2.1 Every letter

For each letter of a recitation, the engine reports:

- **Identity:** was the letter heard as itself, or as a neighbouring letter (ḍād as dāl, ṭā' as tā', ḥā' as hā'), or not at all? It gives a confidence margin, not just a yes or no.
- **Makhraj (point of articulation):** all 17 classical points, in five regions (the oral cavity, the throat, the tongue, the lips, the nasal passage). Each letter is tested against the letters at the neighbouring points, so a letter pulled toward a neighbour is measured, not guessed.
- **Ṣifāt (characteristics):** all 17 classical characteristics are declared for every letter. Those with an acoustic signature are measured:
  - hams / jahr (breathy / voiced)
  - shiddah / tawassuṭ / rakhāwah (stopped / partial / flowing)
  - istiʿlāʾ and tafkhīm / tarqīq (heavy / light)
  - itbāq
  - ṣafīr (whistle)
  - qalqalah (echo)
  - takrīr (trill)
  - tafashshī (spreading)
  - istiṭālah (lateral lengthening)
  - ghunnah (nasality)

  The four that are classifications rather than sounds are declared and explained: idhlāq and iṣmāt, līn, inḥirāf.
- **Vowels:** every fatḥah, kasrah and ḍammah is heard as itself or as another.
- **Duration:** every letter and vowel is timed, in seconds and in the reciter's own count unit.

### 2.2 Every rule

The engine reads the text, locates every tajwīd rule, and judges each one from the audio:

- **Madd,** every type (ṭabīʿī, muttaṣil, munfaṣil, lāzim, ʿāriḍ li-s-sukūn, badal, līn, ṣilah, ʿiwaḍ): its length in counts at the reciter's own pace, against the required length. It also respects the reading the reciter follows (qaṣr or tawassuṭ for the munfaṣil).
- **Nūn and mīm sākinah:** iẓhār, idghām with and without ghunnah, iqlāb, ikhfāʾ, and the shafawī rules. Nasal holds are measured against bands learned from masters, and a clear nūn is checked not to be held like a hidden one.
- **Ghunnah,** in its grades.
- **Qalqalah** (minor, major, strongest).
- **Heaviness and lightness,** including every case of rā' and the lām of the Divine Name.
- **Stops:** where the reciter stopped, whether the stop was legitimate, and how the stop changes what the text requires.

### 2.3 Measured against masters, at the reader's own tempo

- **Pace:** every length is expressed in the reciter's own count unit, so a slow reader and a fast reader are judged fairly. A length that is right at ḥadr (fast) is not called wrong for being short in seconds.
- **Bands from the masters:** acceptable ranges are learned from master reciters over the whole Qur'an and validated on their peers, not taken from textbook numbers. Every measurement comes with its position among the masters: a percentile and a robust z-score.
- **Blind spots are reported:** contexts where the acoustic model is unreliable are flagged as "not scored" rather than turned into false accusations.

### 2.4 Any text, not only verses

The engine accepts any Arabic text, not only verses: a single letter in its forms (بَ بِ بُ أَبْ), a syllable drill, a word, an isolated rule form. The same measurements apply, so drills for every letter, characteristic and rule are measured exactly as recitation is.

### 2.5 What an app receives

A structured record for every recording:
- every letter, with its identity, makhraj, characteristics, vowel and timing
- every rule, with its status, measured length, required band and position among the masters
- every word's overall state
- the reader's pace and chosen reading

These are all numbers and references, ready for any interface: a tutor, a drill app, a teacher's dashboard, a research tool.

---

## 3. The letter and rule corpus

### 3.1 What it is

For each reciter, the corpus collects **every letter in every form** and **every rule in every form**:

- **Letter forms:** each of the 28 letters with fatḥah, kasrah and ḍammah (short and lengthened), sākin, doubled (shaddah), and at a stop.
- **Rule forms:** every madd type, every treatment of nūn and mīm sākinah, qalqalah in its degrees, every rā' and lām case, and so on.

Each instance is:
- cut from the recitation with a little context
- cleaned (hum, rumble and room noise removed; level matched; no processing that could alter a characteristic)
- measured by the engine on every dimension above

A planner chooses verses so that every cell is covered with the least audio. Every reciter is compared on the same words.

### 3.2 What it gives

- **A reference for everything:** for every letter in every context, the natural range of every measurement across the masters. Judgements become "within the masters' spread", not "above an arbitrary threshold".
- **Style against error:** statistical models separate what belongs to the letter, to its context, to a reciter's style and school, and to a mistake. A master's characteristic soft final lām is style; a learner's dropped lām is an error.
- **A geometry of pronunciation:** every letter instance is a point in a measurement space.
  - A learner's letter is placed against the masters' cloud of the same letter, as a distance, not a pass or fail.
  - A learner's whole distribution of a letter across a session is compared with a master's.
  - The discriminability of every pair of letters is known from real audio.
  - The classical 17 makharij can be examined against data.
- **Reciter fingerprints:** every reciter as a profile over all letters and rules. Which reciters sound alike, letter by letter and overall.
- **A playable library:** any letter form or rule form, heard from several masters side by side, with its measurements.

### 3.3 What it enables for users

- **Closest master:** when a learner practises a letter or rule, the engine finds the masters whose realisation is nearest, and shows the difference on the dimensions that matter: "your qāf is closest to al-Minshawi's; here is his and yours".
- **Hear it from the masters:** any letter form or rule form, instantly, from the reciter of the learner's choice.
- **A personal target:** a learner can follow a chosen reciter's style and be measured against that reciter's corpus.
- **A curriculum by distance:** the letters and rules furthest from the masters come first, and progress is the distance shrinking.

---

## 4. The synthesis engine

### 4.1 The idea

If we can **change one thing** in real recitation audio (one characteristic of one letter, the length of one madd, the presence of one ghunnah), by a controlled amount, while everything else stays exactly as it was, then every recording becomes a source of labelled examples:
- the original, correct
- the same with a known mistake of known size
- or, run the other way, a learner's mistake with its correction

Do this across every letter, every rule, every reciter and many synthetic voices, and the supply of labelled data is effectively unlimited.

### 4.2 The physics it rests on

Every characteristic has an acoustic signature, and the synthesis engine works directly on those signatures:

| Characteristic | What changes in the sound |
|---|---|
| hams / jahr | voicing through the consonant: periodic energy against noise |
| shiddah / rakhāwah / tawassuṭ | a full closure, silence and burst, against continuous flow |
| tafkhīm / tarqīq, istiʿlāʾ | the tongue root raised: the second formant of the vowel pulled down, the first raised |
| itbāq | the tongue body against the palate: the second formant falls toward the first, high frequencies are damped |
| ṣafīr | a sharp energy peak high in the frication spectrum |
| tafashshī | a broad, spread frication spectrum |
| istiṭālah | lateral frication sustained along the tongue's edge |
| qalqalah | a brief voiced release after the closure |
| takrīr | repeated taps |
| ghunnah | a nasal resonance with a characteristic anti-resonance, and a held duration |
| madd | the duration of the vowel in counts |

### 4.3 The calculus

The corpus provides **minimal pairs**: letters that differ in exactly one characteristic, spoken by the same reciter in the same context.

| Pair | Differs only in |
|---|---|
| ز / س, ذ / ث, د / ت | voicing |
| د / ذ, ت / ث | closure |
| ت / ط, س / ص, د / ض, ذ / ظ | heaviness and itbāq |
| ن / ل, م / ب | nasality |

For each pair, across reciters and contexts, the engine computes the **delta**: how every physical quantity changes over the consonant and its transition into the vowel, as an average trajectory with its natural variance.

The deltas are the synthesis formulas:
- **To change a characteristic,** move the sound along its delta.
- **To make a subtle mistake,** move it part of the way: 25%, 50%, 75% of the natural difference.
- **To verify the result,** check that its physics now falls within the natural range of the target.

The mathematics (optimisation, distribution matching, mixed-effects statistics, principled tail thresholds) runs in Julia and GNU Octave. The large runs go to cloud compute.

### 4.4 The techniques

**Signal processing:**
- **Source-filter decomposition (vocoders such as WORLD):** separate pitch, vocal-tract envelope and breathiness, then edit each on its own:
  - breathiness for voicing
  - envelope formants for heaviness and itbāq
  - an added or removed nasal resonance for ghunnah
- **Time-scale modification (phase vocoder, WSOLA, PSOLA):** stretch or shorten a madd or ghunnah to any length, pitch unchanged. It changes lengths precisely as the engine measures them.
- **Syllable substitution:** replace a letter together with its transition into the vowel by the same voice's realisation of a neighbouring letter. This produces an exactly labelled letter or characteristic mistake, and the engine reads it as such. The characteristics the two letters share stay; the ones they differ in change.
- **Closure and burst surgery:** insert or remove the silence and burst of a stop; synthesise or remove the qalqalah release.
- **Frication shaping:** reshape the noise spectrum for ṣafīr and tafashshī.
- **Tap modulation:** turn a single rā' tap into a trill, or back.
- **LPC formant surgery** as a second route to heaviness and itbāq.
- **Boundary refinement:** letter edges placed to about 10 ms using spectral change and voicing onsets, so every edit lands on the letter and nowhere else.

**Neural methods:**
- neural vocoders for higher-fidelity resynthesis
- speech inpainting, to regenerate one syllable in context
- voice conversion, to carry a recitation into many voices while keeping every rule
- zero-shot and fine-tuned TTS that learn recitation itself
- articulatory synthesis, which moves a simulated tongue and lips, so makharij are synthesised at their physical source

**Verification, for every synthetic example:**
- **The physics:** voicing, formants, nasal resonance, closure and burst, frication spectrum and duration, measured independently of any model.
- **The engine's own reading**
- **Controls:** processing with no edit, and splices of a letter with itself, which must change nothing. This guarantees the label describes the edit, not an artefact.

### 4.5 From TTS to recitation

Modern Arabic TTS voices (dozens of them, male and female, across dialects) pronounce letters and vowels clearly. What they lack is the recitation layer: the lengths of the madds, the ghunnah holds, qalqalah, the heaviness rules, the forms at a stop, and the pace of tartīl. The synthesis engine supplies exactly that:

1. **Text:** the verse, fully vowelled, and the engine's list of every rule it contains with its target (counts, heavy or light, echo, stop form).
2. **Voice:** the TTS voice reads the verse. It can also speak any syllable on demand, so each voice is its own library of letters for substitution.
3. **Alignment:** the engine places every letter and vowel in the audio.
4. **Rules applied, each as a precise local edit:**
   - madds and ghunnahs set to their counts at a tartīl pace
   - qalqalah echoes added
   - heavy letters and the lām of the Name made heavy
   - stops rendered in their proper form
   - the overall tempo set
5. **Verified:** the engine measures the result against every rule's target, and the physics checks every edit.

The result is recitation-like audio, in any voice, with every rule applied and known. It is for internal testing and training only. The same edits can then add any mistake of any size to that audio.

The same approach applies to **other reciters' voices**. A master's recitation can be carried into many timbres by voice conversion, or edited to contain controlled mistakes, and it keeps its labels throughout.

---

## 5. What this makes possible

### 5.1 An unlimited, labelled data factory

- Every clean instance of every letter and rule, from every reciter and every synthetic voice, can yield:
  - its correct form
  - each of its possible mistakes, at graded sizes
  - its corrections
  - its channel variants (phone microphones, rooms, noise, codecs)
- Every sample carries a complete label: what was changed, where, in which direction, by how much, and what the physics and the engine measured.
- The scale is hundreds of gigabytes and well beyond, generated rather than recorded.

### 5.2 Measuring any engine without human mistake recordings

For every characteristic, letter, rule and voice, the synthetic data measures:
- **Recall:** the share of known mistakes caught.
- **False alarms:** flags on untouched controls, and on audio altered only by the channel.
- **Sensitivity:** the smallest mistake that is reliably caught.

Engines, including future ones, can be compared on the same tests.

### 5.3 Training far stronger models

- **Characteristic detectors that hear the physics itself:** voicing, heaviness, nasality, closure, articulation point, independently of which letter is expected. These are learned from examples where a single characteristic was changed and nothing else.
- **Graded detection:** how large a mistake is, not only whether there is one.
- **Hard cases on demand:** a nasal hum where none belongs, a sākin emphatic made light, the last letter before a stop, the subtle distinctions between neighbouring makharij.
- **Robustness:** to phones, rooms, noise and compression, taught by channel augmentation that must never change a verdict.
- **Recitation-capable voices:** TTS models that learn tajwīd from rendered, rule-perfect examples.

### 5.4 Consumer applications

- **Letter, characteristic and rule drills** in any voice, with every rule audibly applied. Each is measured the moment the learner reads it.
- **Closest master and side-by-side listening** for every letter and rule.
- **Feedback on named physical dimensions:** "your ṣād is not raised enough", "the ghunnah was 1 count", "this qāf is drifting toward kāf". It comes with the master reference to listen to.
- **A curriculum ordered by measured distance** from the masters, and progress shown as that distance closing.
- **A chosen target reciter** to learn from and be measured against.
- **Teacher dashboards:** every student's letters, characteristics and rules at a glance, with recordings.
- **Accessible anywhere:** robust to ordinary phone recordings.

### 5.5 Research

- **A measured map** of the makharij and ṣifāt across reciters and schools.
- **The laws of recitation tempo:** which lengths scale with pace and which stay fixed.
- **Style and school:** what distinguishes reciting traditions, measured.
- **Beyond the Qur'an:** the same method (measure, collect, synthesise, verify) carries over to pronunciation teaching in other languages and to clinical speech, such as articulation therapy.

---

## 6. Principles

- **The engine measures; applications decide** what to tell a learner.
- **Synthetic audio stays internal.** It is an instrument for measurement and training, never presented or published as anyone's recitation.
- **Rights are respected:** master recordings belong to their reciters and publishers. Derived measurements serve research; serving master audio in a product requires permission. Closest-master features can work from measurements alone.
- **Every synthetic label is verified by physics,** with controls in every batch, and spot-checked by an expert's ear.

---

## 7. Glossary

| Term | Meaning |
|---|---|
| tajwīd | the rules of Qur'anic pronunciation |
| makhraj (pl. makhārij) | point of articulation; classically 17, in 5 regions |
| ṣifah (pl. ṣifāt) | characteristic of a letter; classically 17 |
| hams / jahr | breathy (voiceless) / voiced |
| shiddah / tawassuṭ / rakhāwah | stopped / in between / flowing |
| istiʿlāʾ, tafkhīm / tarqīq | tongue-root raising; heavy / light |
| itbāq | tongue body pressed to the palate (ص ض ط ظ) |
| ṣafīr | whistle (ص ز س) |
| qalqalah | echo after a stopped ق ط ب ج د |
| takrīr | the rā''s tendency to trill, to be kept concealed |
| tafashshī | spreading of the air (ش) |
| istiṭālah | lengthening along the side of the tongue (ض) |
| ghunnah | nasal resonance and its hold (ن م) |
| madd | vowel lengthening, measured in counts (ḥarakāt) |
| iẓhār / idghām / iqlāb / ikhfāʾ | clear / merged / turned to mīm / hidden: the treatments of nūn sākinah |
| ḥadr / tadwīr / tartīl | fast / medium / slow recitation |
| waqf | stopping |
