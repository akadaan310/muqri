# qaari-eval

Deep Quranic Tajweed analysis, automated scoring and reciter fingerprinting.

`qaari-eval` checks how a recitation was delivered, not only which words were said. It
force-aligns the audio to the canonical Uthmani text at letter level, derives every Tajweed rule
that Hafs ʿan ʿĀṣim requires from the text itself, and measures each rule acoustically:
Madd length in harakat, Ghunnah nasal resonance, the Qalqalah release burst, and the formant
shift of heavy (Tafkheem) versus light (Tarqeeq) letters. It also places the recitation in a
132-dimensional style + timbre space and finds the closest reciters in a FAISS index built from
EveryAyah.com.

```
[ audio ] + [ Uthmani text ]
      │
      ▼
 app/aligner.py        CTC forced alignment (wav2vec2 Quran phonetic model + exact Viterbi)
 app/tajweed_rules.py  Uthmani → letter units → Hafs rule instances
      │
      ▼
 app/acoustic/         tempo.py  madd.py  ghunnah.py  qalqalah.py  tafkheem.py  (Praat/numpy DSP)
      │
      ▼
 app/scoring.py        per-rule PASS / WARNING / FAIL + feedback, weighted score
 app/profiling.py      ECAPA timbre (128-d) ⊕ Tajweed style (4-d) → FAISS k-NN
```

## Install

```bash
pip install -r requirements.txt           # core: parser, DSP, FAISS, CLI, tests
pip install --index-url https://download.pytorch.org/whl/cpu torch torchaudio
pip install -r requirements-ml.txt        # CTC aligner + ECAPA embeddings + HTTP API
```

The core install runs without PyTorch. Letter timings then come from a heuristic aligner and the
report marks them as approximate. For real measurements, install the ML extras. The first run
downloads `TBOGamer22/wav2vec2-quran-phonetics` (~360 MB) and `speechbrain/spkrec-ecapa-voxceleb`.

## Usage

```bash
python main.py analyze --audio user_recitation.wav --surah 113 --ayah 1
```

This prints a report and writes `analysis_report.json`:

```
════════════════════════════════════════════════════════════════════════
 qaari-eval report — 113:1
 قُلْ أَعُوذُ بِرَبِّ ٱلْفَلَقِ
════════════════════════════════════════════════════════════════════════
 Overall Tajweed score : 100.0 / 100
 Category scores       : madd 100.0, weight 100.0, qalqalah 100.0
 Base harakah          : 330.8 ms  (181.4 harakat/min)
 Rules evaluated       : 3  PASS=3 SKIPPED=1
 Alignment             : ctc:TBOGamer22/wav2vec2-quran-phonetics (confidence 0.6)
────────────────────────────────────────────────────────────────────────
 · SKIPPED tafkheem             قُلْ              581-822   ms
     No light reference for this vowel; Tafkheem (heavy) of ق in 'قُلْ' not scored.
 ✔ PASS    madd_tabii           أَعُوذُ          1564-2246  ms [2.1/2]
     Excellent prolongation. Measured 682ms (2.1 counts); target 2.
 ✔ PASS    tafkheem             بِرَبِّ          2967-3288  ms
     ر in 'بِرَبِّ' was correctly heavy (F2−F1 52 Hz).
 ✔ PASS    qalqalah             ٱلْفَلَقِ        5233-5905  ms
     Clear acoustic release burst detected on letter Qaf (ق).
```

(This is Al-Husary's EveryAyah recording of 113:1.)

Useful options:

| Option | Meaning |
|---|---|
| `--text "…"` | Analyze any Uthmani text instead of looking up `--surah/--ayah` |
| `--aligner auto\|ctc\|heuristic\|json` | Alignment back-end (`auto` = CTC if installed) |
| `--alignment-json FILE` | Use letter timings from an external aligner or manual annotation |
| `--no-stop` | The reciter continues into the next ayah (no waqf rules at the end) |
| `--include-alignment` | Add per-letter timings to the JSON |
| `--index-dir DIR` | Reciter index location (default `index/`) |
| `--style-weight W` | Weight of style vs. timbre in reciter matching (default 0.6 per spec; 0 = timbre only) |
| `--denoise auto\|always\|never` | Spectral-subtraction denoising (auto when SNR < 15 dB) |

Other commands:

```bash
python main.py rules --surah 1 --ayah 7     # list the rules the text requires (no audio)
python main.py serve --port 8000            # POST /analyze (multipart: audio, surah, ayah | text)
python datasets/index_reciters.py           # (re)build the reciter index from EveryAyah
```

## JSON report

```json
{
  "recitation_summary": {
    "overall_tajweed_score": 100.0,
    "tempo_bpm_harakat": 181.4,
    "base_haraka_duration_ms": 330.8,
    "total_rules_evaluated": 3,
    "category_scores": {"madd": 100.0, "weight": 100.0, "qalqalah": 100.0},
    "alignment": {"method": "ctc:TBOGamer22/wav2vec2-quran-phonetics", "reliable": true},
    "audio_quality": {"estimated_snr_db": 41.0, "clipping_ratio": 0.0, "warnings": []}
  },
  "detailed_rule_diagnostics": [
    {
      "rule_type": "madd_tabii",
      "word": "أَعُوذُ",
      "location": {"start_ms": 1564, "end_ms": 2246},
      "expected_harakat": 2,
      "measured_harakat": 2.06,
      "status": "PASS",
      "feedback": "Excellent prolongation. Measured 682ms (2.1 counts); target 2."
    }
  ],
  "reciter_profile": {"style_vector": {"tempo_harakat_per_min": 181.4, "madd_stretch_bias": 1.03}},
  "reciter_similarity_match": {
    "top_matches": [
      {"reciter_name": "Husary", "reciter_id": "Husary_128kbps", "combined_similarity_pct": 58.0,
       "style_similarity_pct": 35.4, "timbre_similarity_pct": 58.0, "matched_traits": ["Wide melodic range"]}
    ],
    "index_size": 44,
    "style_weight": 0.0
  }
}
```

(The match block is real output for Husary's held-out recording of 1:7 with `--style-weight 0`.)

Statuses: `PASS`, `WARNING` (close to the target), `FAIL`, and `SKIPPED` (the analyzer could not
measure the rule reliably). Skipped rules are excluded from the score.

## How it works

### Rule derivation (`app/tajweed_rules.py`)
The parser resolves the Tanzil/Madani Uthmani orthography into letter units. It handles the
dagger alif, small waw/yaa (silah), silent-letter marks, hamzat al-wasl, the lam of the article
before sun letters, bare letters that imply sukun or idgham, a Madd dropped before a sakin letter
in the next word, and the tanween-alif. It then applies Hafs (Shatibiyya) with a stop at the end
of the ayah:

* **Madd:** Tabiʿi (2), Muttasil (4–5), Munfasil (4–5), Lazim Kalimi (6), ʿArid lil-Sukun (2–6),
  Silah Sughra.
* **Ghunnah:** Noon/Meem mushaddadah; Noon sakinah/tanween → Ikhfa, Idgham bi-Ghunnah, Iqlab;
  Meem sakinah → Ikhfa/Idgham Shafawi.
* **Qalqalah:** ق ط ب ج د with sukun. Sughra mid-ayah, Kubra when stopping on the letter.
* **Tafkheem/Tarqeeq:** isti'la letters with fathah/dammah, context rules for Raa, and the Lam of
  the Divine Name.

### Tempo (`acoustic/tempo.py`)
`T_haraka` is the median duration of plain open short syllables that no lengthening rule touches
(IQR outlier rejection). A Madd's length is the whole long syllable (carrier + madd letter)
divided by `T_haraka`. A Tabiʿi syllable therefore measures 2 counts, which is how teachers
count. The pass band is ±15 %.

### Ghunnah (`acoustic/ghunnah.py`)
Duration must be ≥ 2 harakat. Nasality is measured with the **Nasal Energy Ratio**
`10·log10(E[150–400 Hz] / E[750–1100 Hz])`, the nasal formant against the nasal anti-formant.
It is compared with the reciter's own open oral vowels, which cancels microphone and voice
differences.

### Qalqalah (`acoustic/qalqalah.py`)
The detector finds the occlusion, an RMS drop of ≥ 12 dB lasting ≥ 20 ms. It then looks for a
high-band (>1.5 kHz) spectral-flux transient and an energy rebound from 50 ms before to 60 ms
after the release. Kubra needs a stronger rebound (≥ 10 dB) than Sughra (≥ 8 dB).

### Tafkheem / Tarqeeq (`acoustic/tafkheem.py`)
Formants come from Praat's Burg tracker via `praat-parselmouth`, with a numpy LPC fallback. They
are measured on the vowel nucleus: the 15–75 % core of the syllable, voiced frames within 6 dB of
its peak. The score is the relative collapse of **F2 − F1** against a light reference built from
the same reciter's coronal/dorsal letters. The reference excludes gutturals, labials, raa and lam,
and any letter next to a heavy one.

### Reciter fingerprint (`app/profiling.py`)
* **Timbre (128-d):** SpeechBrain ECAPA-TDNN (192-d), projected by a fixed orthonormal matrix.
  Without SpeechBrain, a 128-d MFCC-statistics embedding is used. The index records its back-end,
  and queries must use the same one.
* **Style (4-d):**
  * tempo (harakat/min)
  * Madd stretch bias: median of measured/canonical-minimum length over non-final Madds, so it
    does not depend on which ayah was recited
  * pitch dynamic range: std of F0 in semitones
  * mean nasal energy ratio
* **Similarity:** `S = 0.6·S_style + 0.4·S_timbre`, where `S_timbre` is the cosine similarity and
  `S_style = exp(−d²/8)` for the Euclidean distance `d` between z-normalized style vectors. FAISS
  retrieves candidates from both spaces, and they are re-ranked exactly.

## Reciter index

`index/reciters_faiss.index` stores one 132-d vector per reciter (timbre ⊕ raw style) in a FAISS
`IndexFlatIP`. `index/reciters_meta.json` holds names, the embedding back-end and the style
statistics. The shipped index covers every distinct Hafs reciter in the EveryAyah catalogue,
built from Al-Ikhlas and Al-Falaq with CTC alignment and ECAPA timbre: **44 reciters**
(Ibrahim Akhdar was skipped because EveryAyah's MP3s for him failed to decode). EveryAyah hosts
about 45 distinct Hafs reciters. To reach 100+, add other sources with
`--extra-catalog my_sources.json`:

```json
[{"id": "reciter_x", "name": "Reciter X", "url_template": "https://host/x/{surah:03d}{ayah:03d}.mp3"}]
```

The builder downloads in parallel with retry and back-off, caches audio, and checkpoints after
each reciter. Re-running resumes where it stopped.

## Alignment JSON format

```json
{"units": [{"index": 0, "start_ms": 120, "end_ms": 260}, {"index": 1, "start_ms": 260, "end_ms": 410}]}
```

`index` counts pronounced letters in order (silent letters such as hamzat al-wasl are skipped).
Use `"unit_index"` to address the parser's internal unit numbering directly. `main.py analyze
--include-alignment` emits this format, which also makes it convenient for hand-correction.

## Tests

```bash
pytest                # 52 tests: parser, madd/tempo, qalqalah, ghunnah, tafkheem, FAISS, aligner, audio, CLI, API
```

The DSP tests use Klatt-style source-filter synthesis (`tests/synth.py`) with known formants,
nasal zeros and plosive bursts, so every assertion has an exact ground truth. The tests need only
`requirements.txt`, not PyTorch.

## Validation and limitations

Tested on Al-Husary's EveryAyah recordings of Al-Falaq 1–2 and Al-Fatiha 7 with CTC alignment:

* Madd Tabiʿi measured 1.7–2.5 counts.
* The Ikhfa in مِن شَرِّ measured 2.0 counts, with nasal resonance detected.
* Both ayah-final Qalqalah Kubra were detected.
* Every scored heavy letter (ر ط غ ض خ) was judged heavy.

Known limitations:

* **Phrase-final lengthening:** masters stretch the last Madd before a stop far beyond their
  running tempo. Husary holds the Lazim in ٱلضَّآلِّينَ for ~14 short-syllable lengths.
  Over-extension in the final word is therefore capped at `WARNING`.
* **The CTC model** was trained on word-level audio; its card warns about full ayahs. Forced
  alignment copes well because the target text is known, but boundaries can drift by a frame or
  two (20 ms).
* **Style features vary from ayah to ayah.** On 24 held-out recordings (8 reciters × Al-Fatiha
  5–7, none of them in the index), timbre alone ranked the true reciter first 92 % of the time
  (top-3: 96 %). With the spec's `0.6·style + 0.4·timbre` weighting that fell to 33 % (top-3:
  50 %), because tempo and Madd bias change more between passages than between reciters. The
  default keeps the spec's weights, which answer "whose *style* is this closest to"; use
  `--style-weight 0` (or around 0.2) when the goal is to identify the voice.
* **Not covered:** Madd Lazim Harfi (muqattaʿat letters), Madd Leen, Hafs's saktāt, Imalah, the
  Tayyibah 2-count Munfasil, and Tafkheem on letters carrying kasrah. The formant thresholds are
  set from phonetic literature and checked on a small set of recordings, not calibrated on a
  large labelled corpus.
* This is an educational aid and does not replace a qualified teacher (*mujawwid*).
