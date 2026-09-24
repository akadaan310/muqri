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

## Status at pause (2026-09-24) — paused to return to detection

**Pilot built and measured.** Coverage-selected hour (`datasets/qaari_keys/synth_select.jl`), cut at
Husary's own pauses by `obadx/recitation-segmenter-v2` with the ayah's words assigned to the pieces
by a DP over the engine's CTC likelihoods (`modal_synth.py::prep`); 259 pieces / 47.8 min trained.
Matcha-TTS on the 42 QPS symbols, 128-band 44.1 kHz mels, BigVGAN-v2 44 kHz vocoder, H100, 75 min
(epoch 824, ~7.4k steps).

Held-out passages (12, 186 words, 1,309 letters), graded by the engine:

| | letters failing identity | words fully correct | characteristics not realised |
|---|---|---|---|
| real Husary | 0.5 % | 89.8 % | 0.2 % |
| epoch 524 | 19.7 % | 12.9 % | 8.8 % |
| epoch 824 | 9.5 % | 38.2 % | 5.1 % |

A master's ear (the user): intelligible, recognisably Husary, "sounds like an alien" -- choppy.

**Where the choppiness comes from (measured).**
- Not mainly the vocoder: Husary's real 17:109-110 through BigVGAN-v2 keeps voicing breaks 0.85 -> 0.83
  per s and spectral flux; the user hears it "crisp and clear, a little choppy" -- a small ceiling to
  fix later by fine-tuning the vocoder on Husary.
- The acoustic model's pitch: frame-to-frame wobble 0.106 semitones vs Husary's 0.067 (+58 %).
- Sampling settings (3 held-out passages, `voice_metrics.py`): lower temperature steadies the pitch
  (wobble 0.080-0.085 at t=0.2-0.4 vs 0.096-0.099 at 0.667; real 0.071); 64 steps at t=0.4 gives the most
  fully correct words (24/55 vs 16-22; real 50/55); letter failures stay ~12 % at every setting --
  sampling does not fix the model, training does.
- The model was still improving steeply when the budget ended (errors halved in the last 25 min).

**Ready, not run:** resuming from epoch 824 with mels precomputed beside every wav
(`_cache_mels`, loader patched) -- the pilot H100 ran ~1.6 steps/s waiting on STFTs.

**Next round, in order of payoff:** precomputed mels + longer training on all 43 h of Husary;
pause symbols from the segmenter so it learns his waqf rhythm; explicit F0 (pitch predictor + pitch-
conditioned decoder, the singing-synthesis approach) against the wobble; an NSF-style vocoder or
BigVGAN fine-tuned on Husary; pre-train on muaalem's ~848 h of masters then fine-tune per voice
(and the multi-reciter model). Costs: ~$4.60 per H100-hour all-in; the pilot hour's prep ~$0.30.

**Spend:** Modal metered $24.18 of the $30 this month (synthesis $8.11 of it); $5.82 left.

**Engine findings from this track:** alignment of one long ayah recording (> ~30 s) smears -- a fatha
stretched over a 4 s pause in 27:36; stop detection found no word-boundary stop in several long ayahs
Husary breathes in; the engine flags ~10 % of Husary's own words (word accuracy 0.888 on 296 verses).
