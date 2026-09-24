# Quran recitation datasets with known mistakes — scout log

Round 1 (Hugging Face + IqraEval challenge), 2026-09-23. All entries verified via the HF API / datasets-server; samples in `~/.cache/qaari-eval/datasets/`.

Categories: (a) deliberate/injected errors, (b) learner recitations with error labels, (c) rule-level / makharij labels, (d) expert negatives / reference.

| # | Dataset | Cat | Size | Errors | Granularity | Score |
|---|---|---|---|---|---|---|
| 1 | [IqraEval/QuranMB.v2](https://huggingface.co/datasets/IqraEval/QuranMB.v2) | a | 1642 clips | deliberate: reciters read vowelized verses with planted misp | phoneme | 5 |
| 2 | [MuazAhmad7/Surah_Ikhlas-Labeled_Dataset](https://huggingface.co/datasets/MuazAhmad7/Surah_Ikhlas-Labeled_Dataset) | b | 1506 clips (851 error / 655 correct) | natural learner errors, labelled by annotators | utterance | 4 |
| 3 | [azizbekphd/ikhlas_recitations](https://huggingface.co/datasets/azizbekphd/ikhlas_recitations) | b | 1505 wavs in one 439 MB zip (same audio as Mendeley sxtmmr6mvk) | natural learner errors | phoneme: custom 25-symbol alphabet where | 4 |
| 4 | [obadx/qdat_bench](https://huggingface.co/datasets/obadx/qdat_bench) | b/c | 159 clips (one per reciter) | natural (learners/varied reciters of one verse, Al-Ma'idah 5 | rule-level: 6 madd lengths | 4 |
| 5 | [obadx/muaalem-annotated-v3](https://huggingface.co/datasets/obadx/muaalem-annotated-v3) | d | ~848 h | none (expert reciters) -> negatives | phoneme string + per-phone 11 sifat | 4 |
| 6 | [obadx/qdat](https://huggingface.co/datasets/obadx/qdat) | c | 1505 clips | natural | rule-level binary: separate_tide | 3 |
| 7 | [sobolev210/quran-recitation-errors](https://huggingface.co/datasets/sobolev210/quran-recitation-errors) | b | 1042 ayah clips from 137 recordings | natural (app users), reviewer-marked | word-level with category: Letters / Tajw | 3 |
| 8 | [RetaSy/quranic_audio_dataset](https://huggingface.co/datasets/RetaSy/quranic_audio_dataset) | b | 6828 clips (1228 labelled: 409 correct | natural non-native learner errors | utterance | 3 |
| 9 | [IqraEval/Iqra_Extra_IS26](https://huggingface.co/datasets/IqraEval/Iqra_Extra_IS26) | a | 1330 clips | real human mispronunciations recorded for IqraEval.2 (Inters | phoneme | 3 |
| 10 | [IqraEval/Iqra_TTS](https://huggingface.co/datasets/IqraEval/Iqra_TTS) | a | 55369 clips (~80 h) | deliberate, text-injected before TTS synthesis (letter subst | phoneme + Arabic text | 3 |
| 11 | [Quran-Lab/quran-tajweed-phonetics](https://huggingface.co/datasets/Quran-Lab/quran-tajweed-phonetics) | d (reference) | 6236 ayat | n/a (prescriptive reference) | phone with madd class & allowed lengths, | 3 |
| 12 | [IqraEval/Iqra_train](https://huggingface.co/datasets/IqraEval/Iqra_train) | d | 71 | none (native speech, pseudo-labelled | phoneme | 2 |
| 13 | [MON3EMPASHA/arabic_mispronunciation_dataset](https://huggingface.co/datasets/MON3EMPASHA/arabic_mispronunciation_dataset) | b (non-Quran) | 4842 word clips (686 mispronounced) | deliberate? files suffixed _N are mispronounced | word, binary | 2 |
| 14 | [Reinjin/Pelafalan_Huruf_Hijaiyah](https://huggingface.co/datasets/Reinjin/Pelafalan_Huruf_Hijaiyah) | c (makharij) | 8 | none labelled (tahfidz students' pronunciations) | isolated letter class | 2 |
| 15 | [mark-muhammad/makharij-huruf](https://huggingface.co/datasets/mark-muhammad/makharij-huruf) | c (makharij) | 317 train + 74 test long drill clips | none labelled | utterance | 2 |
| 16 | [HamzaSidhu786/arabic-alphabet-speech-classification](https://huggingface.co/datasets/HamzaSidhu786/arabic-alphabet-speech-classification) | c (makharij) | 22 | none | isolated letter | 2 |
| 17 | [tarteel-ai/tlog](https://huggingface.co/datasets/tarteel-ai/tlog) | b (unlabelled) | 336 GB original; clean split mirror 55 GB | natural app-user recitations, NOT error-labelled (text = tar | utterance | 2 |
| 18 | [Buraaq/quran-md-words](https://huggingface.co/datasets/Buraaq/quran-md-words) | d | 77 | none (expert) | word | 2 |
| 19 | [maqra-project/mahmoud-al-husary-muallim-128kbps](https://huggingface.co/datasets/maqra-project/mahmoud-al-husary-muallim-128kbps) | d | 6236 ayah files | none (expert teaching style, slow, exaggerated tajweed) | ayah | 2 |
| 20 | [obadx/ood_muaalem_test](https://huggingface.co/datasets/obadx/ood_muaalem_test) | b (unlabelled) | 219 clips | unlabelled OOD user clips (sources: tarteel_log, iqraa_eval_ | none | 1 |

## Per-dataset details

### QuranMB.v2 (IqraEval test set) — `IqraEval/QuranMB.v2` (5/5)

- URL: https://huggingface.co/datasets/IqraEval/QuranMB.v2 (host: huggingface)
- Category: a
- Size: 1642 clips, ~2 h (98 verses x 18 speakers); 221 MB parquet
- Errors: deliberate: reciters read vowelized verses with planted mispronunciations; human-annotated realized phones
- Label granularity: phoneme (ref vs annotated phone strings, alignable to letter/word)
- Label format: audio-only in IqraEval/QuranMB.v2; GT phones in IqraEval/IqraEval_Test_GT (gated=auto, labels_test.csv); ungated mirror safikhan/quran_mbv2_formatted has reference_phoneme_string, annotation_phoneme_string, reference_arabic_string (recovered; 904 exact / 738 fuzzy matches)
- Speakers: 18
- Audio: WAV 16 kHz mono (in parquet)
- Download: load_dataset('safikhan/quran_mbv2_formatted') (audio+GT, 232 MB) or IqraEval/QuranMB.v2 + IqraEval/test_references (refs only, ungated)
- License: none declared (challenge page apache-2.0)
- Citation: IqraEval shared task, ArabicNLP 2025 (Iqra'Eval); IqraEval.2 Interspeech 2026 challenge
- Local sample: ~/.cache/qaari-eval/datasets/quranmb_v2/ (labels.jsonl for all 1642 rows + 2 wavs)
- Why: Only public Quran set with deliberate errors and phone-level GT; 1532/1642 clips have >=1 error; covers س>ص 55, ت>ط 84, ق>ك 31, ض>ظ 19, ص>س 14, غ>خ 14, ذ>ز 12, plus many short-vowel swaps and long>short madd (ii>i 108, uu>u 56)

### Surah Al-Ikhlas Quran Recitation Error Detection (labelled) — `MuazAhmad7/Surah_Ikhlas-Labeled_Dataset` (4/5)

- URL: https://huggingface.co/datasets/MuazAhmad7/Surah_Ikhlas-Labeled_Dataset (host: huggingface (mirror of Mendeley doi:10.17632/sxtmmr6mvk.1))
- Category: b
- Size: 1506 clips (851 error / 655 correct), 4 verses of Surah 112; ~485 MB
- Errors: natural learner errors, labelled by annotators
- Label granularity: utterance (binary) + word location + free-text Arabic explanation + error-type (tajweed / harakat)
- Label format: metadata.csv: file_name,label,label_name,verse_number,verse_text,error_type,error_location,error_explanation,error_count
- Speakers: 384
- Audio: WAV 44.1 kHz 16-bit stereo
- Download: hf download MuazAhmad7/Surah_Ikhlas-Labeled_Dataset --repo-type dataset (or per-file resolve/main/data/IDxVyT|F.wav)
- License: cc-by-4.0
- Citation: Maghraby A. et al. (2024) Surah Al-Ikhlas of the Holy Qur'an Error Detection Dataset, Mendeley Data V1, doi:10.17632/sxtmmr6mvk.1
- Local sample: ~/.cache/qaari-eval/datasets/surah_ikhlas_labeled/ (metadata.csv + 1 wav)
- Why: Real learner errors with word location; dominated by qalqalah on د (929), damma on ف in كفوا (141), izhar/sukun of م, a few makhraj ه/ل errors; narrow (one surah) but clean GT for qalqalah & vowel detectors

### Ikhlas recitations (phoneme-level error transcription) — `azizbekphd/ikhlas_recitations` (4/5)

- URL: https://huggingface.co/datasets/azizbekphd/ikhlas_recitations (host: huggingface)
- Category: b
- Size: 1505 wavs in one 439 MB zip (same audio as Mendeley sxtmmr6mvk)
- Errors: natural learner errors
- Label granularity: phoneme: custom 25-symbol alphabet where error variants are distinct tokens (د lacking qalqalah=ݚ, wrong-makhraj ه=ه vs ھ, prolonged م=ࢧ, prolonged ن=ڽ, 3-4 count ءو=ﱞ, o-coloured fatha=ࣵ)
- Label format: data/metadata.csv inside zip: file_name,transcription; vocab.json
- Speakers: 384
- Audio: WAV (zip)
- Download: hf download azizbekphd/ikhlas_recitations --repo-type dataset; metadata.csv can be range-read from the zip
- License: mit
- Citation: derived from Maghraby et al. 2024 (Mendeley sxtmmr6mvk)
- Local sample: ~/.cache/qaari-eval/datasets/ikhlas_recitations/ (metadata.csv, vocab.json, loader script)
- Why: Same clips as MuazAhmad7 but with per-phone error tokens -> can align and score our qalqalah / madd / makhraj detectors at phone level

### QDAT-Bench — `obadx/qdat_bench` (4/5)

- URL: https://huggingface.co/datasets/obadx/qdat_bench (host: huggingface)
- Category: b/c
- Size: 159 clips (one per reciter), 86 MB
- Errors: natural (learners/varied reciters of one verse, Al-Ma'idah 5:109)
- Label granularity: rule-level: 6 madd lengths (0-8 counts), noon mushaddadah ghunnah (partial/complete), ikhfa noon in أنت (noon/partial/complete), qalqalah in الغيوب (0/1); plus full phonetic_transcript and per-phone sifat
- Label format: parquet columns qalo_alif_len, qalo_waw_len, laa_alif_len, separate_madd, noon_moshaddadah_len, noon_mokhfah_len, allam_alif_len, madd_aared_len, qalqalah, phonetic_transcript, sifat[list of 11 attrs]
- Speakers: 159
- Audio: parquet-embedded audio
- Download: load_dataset('obadx/qdat_bench')
- License: none declared (source qdat MIT)
- Citation: Harere & Jallad / QDAT: A data set for Reciting the Quran (ResearchGate 350785609); relabelled by obadx (quran-muaalem)
- Local sample: rows inspected via datasets-server (not downloaded; 86 MB > limit)
- Why: Only public set with measured madd lengths and ghunnah/ikhfa grades per clip — direct GT for madd-length and ghunnah detectors; has leaderboard (obadx/qdat_bench_leaderboard)

### Muaalem annotated v3 (expert reciters, phoneme+sifat) — `obadx/muaalem-annotated-v3` (4/5)

- URL: https://huggingface.co/datasets/obadx/muaalem-annotated-v3 (host: huggingface)
- Category: d
- Size: ~848 h, 286,537 segments, 27 moshafs, 97 GB
- Errors: none (expert reciters) -> negatives
- Label granularity: phoneme string + per-phone 11 sifat (ghunnah, qalqalah, tafkheem, itbaq, safeer, ...), moshaf-level madd settings (munfasil/muttasil/aared lengths etc.)
- Label format: per moshaf config moshaf_<id>; columns phonemes, sifat[list], uthmani, timestamps, moshaf metadata
- Speakers: unknown
- Audio: 16 kHz mono
- Download: load_dataset('obadx/muaalem-annotated-v3', name='moshaf_0.0', split='train') (per-moshaf; compressed variant obadx/muaalem-annotated-compressed-v3)
- License: mit
- Citation: quran-muaalem (github.com/obadx/quran-muaalem); quran-transcript phonetizer
- Local sample: card only
- Why: Best source of expert negatives with per-phone tajweed attributes in the same scheme as qdat_bench; use to calibrate false-alarm rate per rule

### QDAT (cleaned) — `obadx/qdat` (3/5)

- URL: https://huggingface.co/datasets/obadx/qdat (host: huggingface)
- Category: c
- Size: 1505 clips, ~160 reciters x ~10 takes, 784 MB
- Errors: natural; binary rule correctness
- Label granularity: rule-level binary: separate_tide (madd munfasil), the_tight_noon (ghunnah), concealment (ikhfa), target
- Label format: parquet columns separate_tide, the_tight_noon, concealment, target, age, gender, original_id
- Speakers: 160
- Audio: 16 kHz
- Download: load_dataset('obadx/qdat')
- License: mit
- Citation: QDAT: A data set for Reciting the Quran (Harere & Jallad, IJCDS 2021 / arXiv 2305.06429 related)
- Local sample: card only
- Why: Larger binary madd/ghunnah/ikhfa labels on a single verse; labels known noisy (duplicate conflicts listed in card)

### Quran recitation errors (Hafs + Qaloon) — `sobolev210/quran-recitation-errors` (3/5)

- URL: https://huggingface.co/datasets/sobolev210/quran-recitation-errors (host: huggingface)
- Category: b
- Size: 1042 ayah clips from 137 recordings, 154 MB
- Errors: natural (app users), reviewer-marked
- Label granularity: word-level with category: Letters / Tajweed / Wording / Tashkeel
- Label format: errors = dict{word: [category]} (null when fine); riwayah, surah, ayah, text, reviewer_id ('not reviewed' for 768 rows)
- Speakers: unknown
- Audio: wav (in parquet)
- Download: load_dataset('sobolev210/quran-recitation-errors') (-test repo appears identical)
- License: mit
- Citation: none
- Local sample: all 1042 label rows inspected via datasets-server
- Why: Only 61 rows carry errors (49 Letters, 16 Tajweed, 3 Wording, 2 Tashkeel); word-located but no phone detail; only 274 rows reviewed

### Quranic Audio Dataset - crowdsourced, non-Arabic speakers — `RetaSy/quranic_audio_dataset` (3/5)

- URL: https://huggingface.co/datasets/RetaSy/quranic_audio_dataset (host: huggingface)
- Category: b
- Size: 6828 clips (1228 labelled: 409 correct, 502 in_correct, others off-task), 1.26 GB
- Errors: natural non-native learner errors
- Label granularity: utterance (correct/in_correct/not_match_aya/in_complete/multiple_aya/not_related_quran); 62 expert 'golden'
- Label format: final_label + annotation_metadata JSON (per-annotator labels & quality)
- Speakers: 1287
- Audio: WAV 16 kHz
- Download: load_dataset('RetaSy/quranic_audio_dataset')
- License: none declared
- Citation: Salameh et al., Quranic Audio Dataset: Crowdsourced and Labeled Recitation from Non-Arabic Speakers (arXiv 2405.02675)
- Local sample: stats via datasets-server
- Why: Real non-native learner errors (diacritic-level) for utterance-level correlation of our overall score; no location labels; half of clips are Fatiha; includes non-Quran adhkar

### Iqra_Extra_IS26 (real human mispronunciations) — `IqraEval/Iqra_Extra_IS26` (3/5)

- URL: https://huggingface.co/datasets/IqraEval/Iqra_Extra_IS26 (host: huggingface (gated=auto; needs HF login + accept))
- Category: a
- Size: 1330 clips, ~2 h, 208 MB
- Errors: real human mispronunciations recorded for IqraEval.2 (Interspeech 2026); likely MSA sentences
- Label granularity: phoneme (phoneme_ref vs phoneme_mis)
- Label format: id, audio, sentence, phoneme_ref, phoneme_mis
- Speakers: unknown
- Audio: 16 kHz
- Download: load_dataset('IqraEval/Iqra_Extra_IS26') with HF token after accepting terms
- License: none declared
- Citation: IqraEval.2 Challenge, Interspeech 2026
- Local sample: schema from README only
- Why: Real deliberate/human errors with phone GT, but gated (not accessible anonymously) and probably MSA rather than Quran

### Iqra_TTS (synthetic injected mispronunciations) — `IqraEval/Iqra_TTS` (3/5)

- URL: https://huggingface.co/datasets/IqraEval/Iqra_TTS (host: huggingface)
- Category: a
- Size: 55369 clips (~80 h), 5.9 GB; ~53% augmented / 47% original
- Errors: deliberate, text-injected before TTS synthesis (letter substitutions, vowel swaps, deletions)
- Label granularity: phoneme + Arabic text (sentence_ref vs sentence_aug)
- Label format: sentence_ref, sentence_aug, speaker (7 TTS voices), label (augmented/original), phoneme_ref, phoneme_mis
- Speakers: 7
- Audio: 16 kHz
- Download: load_dataset('IqraEval/Iqra_TTS') (stream a few shards only)
- License: none declared
- Citation: IqraEval shared task 2025
- Local sample: rows via datasets-server
- Why: Huge supply of known letter/vowel injections on Quranic verses (e.g. ح>ه, ق>غ) — good for stress tests, but TTS voice not real recitation and no tajweed realism

### Quran Tajweed Phonetics (text reference, no audio) — `Quran-Lab/quran-tajweed-phonetics` (3/5)

- URL: https://huggingface.co/datasets/Quran-Lab/quran-tajweed-phonetics (host: huggingface)
- Category: d (reference)
- Size: 6236 ayat, 522,475 phones; 189 MB jsonl
- Errors: n/a (prescriptive reference)
- Label granularity: phone with madd class & allowed lengths, ghunna grade, qalqalah class, tafkheem rank, rule ids
- Label format: quran_phonetics.jsonl, rule_index.jsonl, rulings.jsonl
- Speakers: 0
- Audio: none
- Download: hf download Quran-Lab/quran-tajweed-phonetics --repo-type dataset
- License: quran-lab-npl-1.2 (non-profit share-alike)
- Citation: Quran Lab, quran-g2p (2026)
- Local sample: card only
- Why: Expected-rule layer to know WHERE each madd/ghunnah/qalqalah should occur in any ayah (join with audio sets above)

### Iqra_train (MSA + Quran, correct speech) — `IqraEval/Iqra_train` (2/5)

- URL: https://huggingface.co/datasets/IqraEval/Iqra_train (host: huggingface)
- Category: d
- Size: 71,391 train + 2,588 dev (~79 + 3.4 h), 12 GB
- Errors: none (native speech, pseudo-labelled; phoneme_aug == phoneme_ref)
- Label granularity: phoneme
- Label format: id, phoneme_ref, phoneme_aug, sentence, tashkeel_sentence
- Speakers: unknown
- Audio: 16 kHz
- Download: load_dataset('IqraEval/Iqra_train', split='dev')
- License: none declared
- Citation: IqraEval 2025
- Local sample: rows via datasets-server
- Why: Negatives in IqraEval phone set; mirrors: mostafaashahin/IqraEval_Training_Data, KhateebAI/iqraeval (adds 'score' 0/100 = quran/non-quran?), Bisher/iqra_eval_all (train+TTS merged)

### Arabic mispronunciation dataset (words, non-Quran) — `MON3EMPASHA/arabic_mispronunciation_dataset` (2/5)

- URL: https://huggingface.co/datasets/MON3EMPASHA/arabic_mispronunciation_dataset (host: huggingface)
- Category: b (non-Quran)
- Size: 4842 word clips (686 mispronounced), 104 MB
- Errors: deliberate? files suffixed _N are mispronounced
- Label granularity: word, binary
- Label format: arabic_speech_dataset.csv: audio_path, transcript, label(0/1), speaker_id, word_index
- Speakers: 79
- Audio: 16 kHz
- Download: load_dataset('MON3EMPASHA/arabic_mispronunciation_dataset')
- License: none declared
- Citation: none
- Local sample: csv inspected
- Why: Everyday MSA words, binary only, no error type — weak proxy

### Pelafalan Huruf Hijaiyah (letter x haraka) — `Reinjin/Pelafalan_Huruf_Hijaiyah` (2/5)

- URL: https://huggingface.co/datasets/Reinjin/Pelafalan_Huruf_Hijaiyah (host: huggingface)
- Category: c (makharij)
- Size: 8,988 clips, 84 classes (28 letters x fatha/kasra/damma), 106 per class, 238 MB zip
- Errors: none labelled (tahfidz students' pronunciations)
- Label granularity: isolated letter class
- Label format: folder name = class
- Speakers: unknown
- Audio: wav in zip
- Download: hf download Reinjin/Pelafalan_Huruf_Hijaiyah dataset.zip --repo-type dataset
- License: afl-3.0
- Citation: Indonesian undergraduate thesis (Pondok Tahfidz Yanbuul Qur'an Kudus)
- Local sample: zip listing via range reads
- Why: Isolated letter templates incl. ض/د/ظ, ص/س, ط/ت, ح/ه, ع/ء, ق/ك, ث/س, ذ/ز, غ/خ — usable to build confusion tests for letter-substitution classifier

### Makharij huruf drills — `mark-muhammad/makharij-huruf` (2/5)

- URL: https://huggingface.co/datasets/mark-muhammad/makharij-huruf (host: huggingface)
- Category: c (makharij)
- Size: 317 train + 74 test long drill clips, 592 MB
- Errors: none labelled
- Label granularity: utterance (drill text per letter, e.g. 'نَا نِى نُو بَنْ ...')
- Label format: audio, text
- Speakers: unknown
- Audio: parquet audio
- Download: load_dataset('mark-muhammad/makharij-huruf')
- License: mit
- Citation: none
- Local sample: rows via datasets-server
- Why: Letter-drill references per makhraj; negatives for letter classifier

### Arabic alphabet speech classification — `HamzaSidhu786/arabic-alphabet-speech-classification` (2/5)

- URL: https://huggingface.co/datasets/HamzaSidhu786/arabic-alphabet-speech-classification (host: huggingface)
- Category: c (makharij)
- Size: 22,200 clips, 28 letter classes, 1.3 GB (pre-extracted input_values float arrays)
- Errors: none
- Label granularity: isolated letter
- Label format: label (28 letter names), input_values, attention_mask
- Speakers: unknown
- Audio: raw float arrays (wav2vec2 features-ready)
- Download: load_dataset('HamzaSidhu786/arabic-alphabet-speech-classification')
- License: none declared
- Citation: none
- Local sample: card only
- Why: Letter-name utterances; limited relevance to in-context substitutions

### Tarteel TLOG (user recitation logs) — `tarteel-ai/tlog` (2/5)

- URL: https://huggingface.co/datasets/tarteel-ai/tlog (host: huggingface (gated=manual); ungated mirrors deepdml/tlog-clean-sf16k, nour-world/tlog-clean-sf16k (55 GB FLAC))
- Category: b (unlabelled)
- Size: 336 GB original; clean split mirror 55 GB
- Errors: natural app-user recitations, NOT error-labelled (text = target ayah)
- Label granularity: utterance (ayah id only)
- Label format: audio + ayah text/path
- Speakers: unknown
- Audio: 16 kHz FLAC (mirror)
- Download: load_dataset('deepdml/tlog-clean-sf16k', split='clean', streaming=True)
- License: cc-by-4.0
- Citation: Tarteel AI
- Local sample: card only
- Why: Realistic learner/phone audio for unsupervised error mining & false-alarm study; no GT

### Quran-MD (words/ayahs) — `Buraaq/quran-md-words` (2/5)

- URL: https://huggingface.co/datasets/Buraaq/quran-md-words (host: huggingface)
- Category: d
- Size: 77,429 word clips (~20 h); ayah set Buraaq/quran-md-ayahs 35 GB
- Errors: none (expert)
- Label granularity: word
- Label format: surah_id, ayah_id, word_index, word_ar, word_tr, audio(mp3)
- Speakers: unknown
- Audio: MP3
- Download: load_dataset('Buraaq/quran-md-words')
- License: none declared
- Citation: Quran-MD: A Fine-Grained Multimodal Dataset of the Quran (arXiv 2601.17880, MusIML@NeurIPS 2025)
- Local sample: card only
- Why: Word-level expert negatives, handy for per-word detector calibration

### Maqra Husary Muallim (teaching recitation) — `maqra-project/mahmoud-al-husary-muallim-128kbps` (2/5)

- URL: https://huggingface.co/datasets/maqra-project/mahmoud-al-husary-muallim-128kbps (host: huggingface)
- Category: d
- Size: 6236 ayah files, 2.6 GB
- Errors: none (expert teaching style, slow, exaggerated tajweed)
- Label granularity: ayah
- Label format: per-ayah mp3 + metadata
- Speakers: 1
- Audio: MP3 128 kbps
- Download: hf download maqra-project/mahmoud-al-husary-muallim-128kbps --repo-type dataset
- License: everyayah-mirror-notice
- Citation: Maqra (github.com/Abdalla-Eldoumani/maqra)
- Local sample: card only
- Why: Canonical slow negatives with clear madd/ghunnah (sibling sets: husary mujawwad/murattal, minshawi muallim)

### OOD Muaalem test — `obadx/ood_muaalem_test` (1/5)

- URL: https://huggingface.co/datasets/obadx/ood_muaalem_test (host: huggingface)
- Category: b (unlabelled)
- Size: 219 clips, 31 MB
- Errors: unlabelled OOD user clips (sources: tarteel_log, iqraa_eval_open_test)
- Label granularity: none
- Label format: audio, source, id, original_id
- Speakers: unknown
- Audio: 16 kHz
- Download: load_dataset('obadx/ood_muaalem_test')
- License: none declared
- Citation: quran-muaalem
- Local sample: rows via datasets-server
- Why: No labels; only useful as a small OOD robustness probe

## Sample label formats

QuranMB.v2 (safikhan mirror), id 00000_00001:
```
reference : < i nn a m aa y a x $ aa ll a h a m i n E i b aa d i h i l E u l a m aa < u
annotation: < i nn a m aa y a x $ aa ll a h a m i n E i b aa d i h u l E u l a m < u
```
Ikhlas (MuazAhmad7 metadata.csv): `data/ID1V1F.wav,0,error,1,قُلْ هُوَ اللَّهُ أَحَدٌ,تجويد,أَحَدٌ,خطأ في قلقلة حرف الدال,5`

Ikhlas (azizbekphd phone tokens): correct `قﹸلھﹸوﹶڶڶٞھﹸﹶحﹶۮ` vs error `قﹸلھﹸوﹶڶڶٞھﹸﹶحﹶݚ` (ݚ = د lacking qalqalah)

qdat_bench row: `qalo_alif_len=2, qalo_waw_len=2, laa_alif_len=2, separate_madd=3, noon_moshaddadah_len=1, noon_mokhfah_len=2, allam_alif_len=2, madd_aared_len=2, qalqalah=1, phonetic_transcript='قَاالُۥۥ لَاا عِلمَ لَنَااا ءِننننَكَ ءَںںںتَ عَللَاامُ لغُيُۥۥبڇ'`

sobolev210 row: `{"الخناس": ["Letters"]}` for 114:4 (categories: Letters/Tajweed/Wording/Tashkeel)

## QuranMB.v2 error profile (computed from all 1642 GT rows)

- 110 clips error-free, 787 with 1 error, 416 with 2, 329 with 3+.
- Top substitutions: a>A 178, ii>i 108, z>s 91, t>T (ت>ط) 84, n>y 77, a<->u 134, *>Z (ذ>ظ) 59, uu>u 56, s>S (س>ص) 55, q>k (ق>ك) 31, D>Z (ض>ظ) 19, g>x (غ>خ) 14, S>s 14, *>z 12.
- Not present: ض>د, ح>ه, ع>ء, ث>س as direct subs (check deletions: h 57, E 21, H 19).

## Leads / text-only / gated (not scored)

- `aisyahrevolab/mispronounced-phoneme-dataset` — gated=auto; 13,470 rows TEXT ONLY (audio_path, original/mispronounced text + phoneme_ref/mis) — audio not in repo; possibly error-injection recipe
- `omar-abu-hfs-siraj/tasmee3-muaalem-findings` — research notes + scripts on detecting tasmee' errors with muaalem-model-v3_2; NO audio (reports 3/7 letter-substitution detection)
- `IqraEval/leaderboard_data` — leaderboard TSVs: best phone-F1 ~0.72 (whu-iasp, RAM, Utokyo) — external baseline to compare our engine
- `IqraEval/IqraEval_Test_GT` — gated=auto labels_test.csv (QuranMB GT) — mirrored ungated by safikhan/quran_mbv2_formatted
- `Quran-Lab/quranic-asr-benchmark` — gated=manual; 600 clips (everyayah_heldout, qul_alnufais, tlog_holdout) ASR benchmark, no error labels
- `omartariq612/everyayah-with-tajweed-tokens` — EveryAyah expert audio with tajweed-token transcripts (56 GB); negatives with rule tokens
- `obadx/mualem-recitations-annotated / -original` — earlier versions of muaalem-annotated-v3
- `nurlingo/quran-recitation-detect-split` — text-only ASR transcripts of user recitations (no audio)
