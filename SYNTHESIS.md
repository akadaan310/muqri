# Recitation synthesis — research and plan (Sprint 4, from 2026-09-24)

Goal: synthesize master-level recitation — a voice a master-trained ear accepts as perfect — with
control deep enough that every tajweed phenomenon can be produced correctly *and* broken on purpose.
Perfect synthetic recitation plus controlled, labelled mistakes is the training set for a detector
aimed at near-total recall of every mistake, characteristic and phenomenon. The audio is for internal
training; it must still sound like a master.

## Where the existing synthesis engine stands (akadaan310/qaari @ 31e3def)

F5-TTS (`IbrahimSalah/Arabic-F5-TTS-v2`, flow-matching DiT, 300 h of diacritised MSA) cloning Husary
zero-shot from a ~10 s prompt. Tajweed is imposed only as the *total* duration of each chunk
(F5 has no per-phoneme duration input), chunks are cross-faded, WORLD smooths the pitch. Mistakes
are DSP edits on the output (WSOLA stretch, burst mute, ghunnah attenuation).

Measured / observed:
- A master's ear: choppy and hard to hear.
- Our engine still scored those files ~90-98 %, perturbed mistakes included, and on the four samples
  checked: 112:2 had 0/13 identity failures, 1:2 118/118 characteristics realised. **The engine is not a
  synthesis judge on its own**: it tests whether each expected phoneme beats its competitors, and a
  choppy but intelligible voice passes that. It has no measure of continuity, naturalness or voice
  quality. And its ceiling on the real voice is not 100 %: Husary's own murattal scores word accuracy
  0.888 over 296 T300 verses.
- The git remote of that checkout embeds a GitHub token in plain text — revoke it.

What to keep: the tajweed-phonetic rendering idea, the mistake-operator taxonomy and labelling
conventions (SYNTHETIC_MISTAKE_*, `synthetic: true`). What to replace: F5 with chunk-level duration
(no per-phoneme control means madd, ghunnah, qalqalah cannot be set, only hoped for) and DSP-edited
mistakes (edits on a waveform leave artefacts a detector learns instead of the mistake).

## The voice: measured, not assumed

Effective bandwidth (highest frequency within 60 dB of the speech peak) and speech-to-floor range on
the same six ayahs (2:2, 2:255, 18:10, 36:1, 67:2, 112:1), EveryAyah sources, 44.1 kHz:

| source | bandwidth | speech-floor | note |
|---|---|---|---|
| **Husary_128kbps** (murattal) | **15.7 kHz** | **44.2 dB** | widest and driest of 16 |
| Alafasy_128kbps | 14.6 kHz | 21.6 dB | reverberant |
| Yasser_Ad-Dussary_128kbps | 13.2 kHz | 19.0 dB | reverberant |
| Abdul_Basit_Murattal_192kbps | 11.7 kHz | 43.8 dB | |
| Abdurrahmaan_As-Sudais_192kbps | 11.1 kHz | 15.5 dB | mosque reverb fills the pauses |
| Husary_Muallim_128kbps | 9.3 kHz | 43.3 dB | |
| Husary_128kbps_Mujawwad | 9.0 kHz | 47.8 dB | mixed 24 / 44.1 kHz |
| Minshawy_Murattal_128kbps | 9.0 kHz | 43.3 dB | |

Husary murattal is the crispest *and* driest source measured, so it is the target voice. Sudais is
recorded in reverberant space; a dry synthetic Sudais would need dereverberation first. (Speech-floor
range mixes noise, reverb and pause length; it is one indicator, not the whole story.)

EveryAyah's Husary murattal reads the munfasil at a median 4.26 counts (tawassut, n=144) at 0.32 s per
count -- the wajh of muaalem-annotated-v3's moshaf 0.1, so the phonetizer settings for its labels are
known exactly.

## Data

| source | what | use |
|---|---|---|
| EveryAyah `Husary_128kbps` | all 6,236 ayahs, one clean voice, verse-segmented | the training voice |
| `obadx/muaalem-annotated-v3` (MIT) | 25 complete mushafs of 22 masters indexed; Husary: 0.0 murattal qasr 29 h, 0.1 tawassut 43 h, 0.2 mujawwad 62 h, 0.3 mu'allim 48 h; each mushaf's wajh choices labelled; segments carry **QPS phonemes and the 10 sifat** | labels at exactly the engine's depth; more Husary; other voices later |
| our T300 (41 reciters x ~300 verses, posteriors on Modal) + the certified reciter's takes | the engine's own corpus | evaluation, multi-voice later |
| `quranlab/quran-audio` | word timings for 245 reciters (CC-BY) | alignment cross-check |
| `Buraaq/quran-audio-text-dataset` (CC0) | verse audio, 32 reciters, word-level alignment | more voices |
| `rabah2026/Quran_TTS` | Google TTS reading Quran text | not recitation; not used |

## The engine as annotator, filter and judge

Everything a controllable synthesizer needs as *input* is what the engine already *measures*:
- **phonemes**: QPS from `quran_transcript` (madd lengths and ghunnah are already explicit symbols:
  ااا, ںںں, the qalqalah release ڇ) with Husary's wajh;
- **durations**: per-phoneme frames from the muaalem CTC Viterbi alignment, counts from the haraka unit;
- **characteristics**: per-letter margins of the ten sifat heads;
- **filter**: train only on ayahs where every judgement passes (the masters control);
- **closed-loop check**: synthesize with counts N, measure with the engine, require N back.

What the engine lacks as a judge -- continuity, naturalness, timbre -- is measured separately: a MOS
predictor (UTMOS), speaker similarity to real Husary, and, because every ayah has a real Husary
recording, direct distances to the truth on held-out ayahs (mel-cepstral distortion, F0 RMSE,
duration error). The final judge is a master's ear.

## Approach: explicit-control acoustic model, singing-style

Recitation is closer to singing than to speech: sustained pitched vowels, melodic contours, lengths
set by rule. So the model family is the explicit-control one used in singing voice synthesis
(FastSpeech2 / DiffSinger lineage; Matcha-TTS for flow matching): **phoneme sequence + per-phoneme
duration + F0 curve (+ energy)** in, mel out, then a vocoder trained for sustained pitched voice
(NSF-HiFiGAN / BigVGAN) at 44.1 kHz, since the source carries 15.7 kHz.

- Correct recitation: phonemes and durations from the text and Husary's measured tempo; F0 from a
  pitch model trained on his contours.
- A mistake is an **edit of the input**, re-synthesized: a madd at 1 count instead of 2, ں read as ن,
  the ڇ removed, a tafkhim letter swapped for its muraqqaq partner. The voice stays natural and the
  label is exact by construction.

## Plan

1. **Annotate** Husary murattal: QPS phonemes, engine alignment (durations), F0 (parselmouth / RMVPE),
   engine verdicts as the filter. Start with Juz' 'Amma (short surahs, ~1 h) for the pilot.
2. **Pilot** a small explicit-duration model (Matcha-TTS style, with an F0 input) + a pretrained
   44.1 kHz vocoder, fine-tuned on the pilot hour. Evaluate on held-out ayahs against the real
   recordings (MCD, F0 RMSE, UTMOS, speaker similarity, engine closed loop), then your ear.
3. **Scale** to all 6,236 ayahs if the pilot sounds right; then mistakes by input edits.

Compute: Modal. The pilot is sized to a few GPU-hours; the full run is sized after the pilot's
learning curve is measured.
