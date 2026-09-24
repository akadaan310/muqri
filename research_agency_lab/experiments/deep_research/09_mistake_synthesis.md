# 09 — Synthesising recitation mistakes the engine must catch

*Design, 2026-09-24. Grounded in how the engine decides (below); to be calibrated on the recording
protocol's paired takes before anything is generated at scale.*

## What a synthetic mistake has to move

The engine reaches a verdict in exactly three ways, so a synthetic error is only realistic if it moves
the same quantity a real error moves:

| engine mechanism | what it measures | the mistakes it decides |
|---|---|---|
| **duration** — CTC alignment onsets, in the reciter's own counts | how long a unit lasts | every madd, ghunnah length, ikhtilās |
| **identity** — CTC likelihood of the reference letter vs its classical confusions (ض↔د/ظ, ص↔س, ح↔ه, ع↔ء, ق↔ك, ط↔ت, ث↔س …) | which letter was produced | makhraj substitutions |
| **characteristics** — ten muaalem heads pooled over the letter's frames | whether each ṣifah is realised | ghunnah present, qalqalah, tafkhīm/tarqīq, hams/jahr, iṭbāq, ṣafīr, tafashshī, istiṭālah |

And a fourth, from this session: **waqf/ibtidā'** — which reading (wasl or stop) the audio matches,
decided against the Quran-wide stop table (`datastore/waqf_table.py`, 71,197 boundaries).

## Operators

Each operator takes a correct recording, its CTC alignment, and a target (ayah, word, unit), and returns
audio plus an exact label. All work on real voices — the 41 masters, and the protocol's take A.

1. **Time-scale a segment** (duration errors). WSOLA/PSOLA-stretch the aligned madd or ghunnah
   segment by a chosen factor (6 counts → 2, 2 → 5), pitch preserved, crossfaded at both joins.
   Label: the new length in counts. Covers madd tabii/munfaṣil/muttaṣil/lāzim/'āriḍ/līn/ṣilah and ghunnah
   length.
2. **Swap a unit** (makhraj substitutions and many characteristics). Every master's clips are already
   CTC-aligned, so each voice has an inventory of its own segments per phoneme and context. Replace
   the target letter with the **same reciter's** realisation of its confusion (their س for their ص),
   matched for neighbouring vowel, energy and F0, and crossfaded. Label: reference → produced. Because
   the donor is the same voice, the detector cannot cheat on speaker identity.
3. **Remove or add a burst** (qalqalah). Cut the release burst after a sākin qalqalah letter and
   extend the closure silence (qalqalah removed), or splice the reciter's own burst onto a letter that
   has none (qalqalah where it does not belong).
4. **Nasal on / off** (ghunnah, iẓhār vs ikhfā'/idghām). Swap the nasal murmur of a doubled nūn/mīm
   for the same voice's plain short nūn/mīm (ghunnah dropped), or splice a nasal murmur into an iẓhār
   position (ghunnah added). A spectral route — suppress the ~250 Hz nasal formant and fill the
   anti-resonance — is the fallback where no donor exists.
5. **Heavy ↔ light** (tafkhīm/tarqīq). Warp the formants of the vowel next to the letter: lower F2 for
   heavy, raise it for light, by the shift measured between the same voice's heavy and light
   instances of that letter. Label: tafkhīm flipped.
6. **Waqf errors.** Insert silence inside a word (mid-word stop); at a real stop, splice back the
   final vowel from the same voice's wasl reading (vowel kept at waqf); at a restart, cut the hamzat
   al-waṣl (ibtidā' without hamza). Labels come straight from the stop table.
7. **Scale** — a phoneme-conditioned TTS fine-tuned on the masters, fed the reference phoneme string
   with the error edited in (the Iqra_TTS route). Used only after 1–6 are validated, and never as the
   only evidence.

## Calibrate on real mistakes before trusting any operator

The recording protocol provides, for one certified voice, the same passage recited correctly (A) and
with scripted mistakes (B), each at a known word. For every scripted mistake:

- apply the matching operator to take A at that word, producing A′;
- run the engine on A′ and on the real B;
- the operator is **accepted** only if the engine's response at that word — duration in counts, the
  identity LLR, the head's LLR — falls inside the range the real mistake produced, and leaves the
  untouched words of A′ scoring the same as A.

An operator that fails is refined or dropped. This matters: a synthetic error the detector finds easier
than the real one inflates every number built on it.

## Then measure, then train, then measure on what was never trained on

1. **Measure:** generate every accepted operator at every eligible position on the 41 masters across
   the T300 set: a miss rate per error type × per voice × per tempo, with the untouched clip as its own
   control (the false-alarm rate on the same audio).
2. **Train:** fine-tune the heads and the confusion scoring with the synthetic negatives (and the
   public mistake sets from 08), holding out whole reciters.
3. **Judge on real mistakes only:** the protocol takes are **never** trained on. They are the test
   set, and the success criterion is the scorecard on them: scripted mistakes caught, false-alarm
   words on take A, and take B of test 10 (fast but correct) scoring the same as its take A.

## Speed and the characteristics

Speed must never excuse a missing characteristic. The heads are already judged absolutely, not scaled
to tempo, and the careless 2:1–2 lost ten of 270 characteristics against zero for the careful reading
— but a certified ear counts 40–50 there. So the miss rate is expected to be worst at speed. Operators
2–5 applied to the fastest masters (haraka 0.16 s) and to hadr takes will measure how much worse, and
test 10 of the protocol checks the opposite failure: a fast reading that is correct must not lose points.
