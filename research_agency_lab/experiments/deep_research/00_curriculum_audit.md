# 00 — Hafs 'an 'Asim Tajweed Curriculum Taxonomy & qaari-eval Coverage Audit

Date: 2026-09-23 · Scope: Hafs 'an 'Asim, Shatibiyyah route (default of the engine), Tayyibah differences noted.
Method: (a) taxonomy built from the classical texts (al-Muqaddimah al-Jazariyyah, Tuhfat al-Atfal, Hidayat al-Mustafid, Nihayat al-Qawl al-Mufid, Hidayat al-Qari chapter 19 "what must be observed for Hafs"), with key points checked against the web sources listed at the end; (b) static reading of `app/models.py`, `app/tajweed_rules/*.py`, `app/sifaat/*.py`, `app/scoring.py`, `app/calibration.py`, `research_agency_lab/substrate_library/julia/data/metric_spec.json`; (c) **dynamic probes**: I ran `TajweedParser` (read-only) on the real Uthmani text in `~/.cache/qaari-eval/text/quran-uthmani.json` for every special case in §3. Probe scripts are in the session scratchpad, not in the repo.

Status legend: **COVERED** = a rule instance is generated *and* an acoustic validator judges it · **PARTIAL** = only some positions or aspects are handled, or the text is resolved correctly but nothing is judged · **MISSING** = no instance or no validator · **N/O** = not acoustically observable (visual, lexical or a convention only) · **BUG** = the engine currently expects the *wrong* pronunciation (a correct reciter gets penalised). The table counts BUG as MISSING and flags it separately.

Lahn column: **J** = lahn jali (a clear error that changes a letter, harakah or word, or drops or adds a letter; prohibited by consensus). **K** = lahn khafi (spoils the letter's full quality without changing the letter; e.g. ghunnah and madd amounts, tafkheem levels, rolling the raa). Nihayat al-Qawl al-Mufid classes leaving out required ghunnah, idgham or ikhfa' as khafi by the older definition, but modern ijazah examiners treat these as major errors. I mark them **K\***.

---

## 1. Taxonomy × engine status

### 1A. Level 101 — letters, harakat, sukoon, shaddah, tanween, basic madd, noon and meem sakinah

| # | Level | Topic | Rule / phenomenon | Hafs specifics | Acoustic observable | Engine status | Evidence (file / function) | Lahn |
|---|---|---|---|---|---|---|---|---|
| 1 | 101 | Letters | Correct letter identity, no substitution (ض→د/ظ, ث→س/ت, ذ→ز/د, ظ→ز, ح→ه, ع→ء, ق→ك/g, ط→ت, ص→س, غ→خ) | — | Free (unforced) phone posteriors / GOP; spectral envelope of the frication; F2 and itbaq cues | **MISSING** | `aligner.py` `CTCForcedAligner` forces the *target* text. Worse, `calibration.Calibration.unreliable()` turns low `align_conf` into SKIPPED, which **hides** exactly the spans where a wrong letter was read | J |
| 2 | 101 | Harakat | Right short vowel (fatha, kasra or damma, not swapped; e.g. أَنْعَمْتَ read with damma) | — | F1/F2 of the vowel core against the reciter's own vowel centroids; unforced phone decoding | **MISSING** | `phonetic_tokens` carries the vowel symbols, but the alignment is forced and nothing compares them | J |
| 3 | 101 | Harakat | Short vowel ≈ 1 harakah (no ishba' that turns it into a madd letter; no clipping) | — | Duration of the vowel core ÷ local harakah | **MISSING** | `acoustic/tempo.short_syllable_units` uses these vowels to *estimate* tempo, but no single vowel is ever judged | J (ishba' adds a letter) |
| 4 | 101 | Sukoon | A sakin letter carries no vowel (no epenthetic vowel, no tahrik) | — | Voiced vowel-formant stretch after the consonant; for stops, a release followed by a vowel | **PARTIAL** | Only indirectly: `sukoon_spectrum.validate_sukoon_class` (occlusion) and qalqalah ("may have been voweled"). Only sakin letters outside qalqalah, ن, م and hamza get these checks | J |
| 5 | 101 | Shaddah | Geminate held about twice as long as a single letter, for **every** letter (e.g. إِيَّاكَ, ٱلرَّحْمَٰنِ, ٱلْحَقُّ) | — | Duration of the consonant closure, frication or nasal against the same letter single | **PARTIAL** | Only ن/م (`validate_ghunnah_mushaddadah`) and idgham targets (`idghaam_classes.validate_kamil`, `MIN_GEMINATE_HARAKAT`). A plain shaddah (يَّ, سَّ, …) is never checked | J (dropped shaddah) / K (short) |
| 6 | 101 | Tanween | Read as noon sakinah in wasl; at waqf, fath becomes an alif ('iwad) and damm/kasr becomes sukun | — | Nasal after the vowel; at waqf, a 2-count /aː/ or no nasal | **COVERED** | Through noon rules (`noon_sakinah.*`), `parser._apply_waqf`, `MADD_IWAD` | J |
| 7 | 101 | Madd | Madd tabi'i = 2 counts (alif, waw, yaa; dagger alif; small waw/yaa) | — | Long-syllable duration ÷ local harakah | **COVERED** | `mudood_engine.validate_madd`; calibrated `core_counts` in `metric_spec.json` | J if dropped, K if off |
| 8 | 101 | Makhraj of madd letters (jawf) | Alif follows the weight of the letter before it (heavy after خ ص ض غ ط ق ظ, light otherwise) | Standard | F2 of the /aː/ | **PARTIAL** | `_detect_weight` puts the following madd alif into the heavy letter's span. No check that alif stays **light** after light letters (a common non-Arab error, e.g. heavy نَار or قَال-style colouring spreading) | K |
| 9 | 101 | Noon sakinah/tanween | Izhar halqi before ء ه ع ح غ خ | — | No nasal hold (≤ 1.3 h); no inserted pause | **COVERED** | `noon_sakinah.clear_letter` / `validate_izhar_halqi` | K\* |
| 10 | 101 | Noon sakinah/tanween | Idgham with ghunnah (ي ن م و), across words only; naqis before و/ي, kamil before ن/م | — | 2 h nasal hold + nasal energy ratio (NER) contrast | **COVERED** | `validate_idgham_ghunnah`; parser `cross_word` | K\* |
| 11 | 101 | Noon sakinah | Izhar mutlaq inside a word: الدُّنْيَا، بُنْيَانٌ، صِنْوَانٌ، قِنْوَانٌ | Hafs: izhar | Short, non-nasal noon | **PARTIAL** | Parser correctly generates *no* idgham (the `cross_word` guard), but no izhar instance is checked | K\* |
| 12 | 101 | Noon sakinah/tanween | Idgham without ghunnah before ل ر | Shatibiyyah: no ghunnah. Tayyibah: ghunnah allowed via some turuq | No nasal on the target (NER contrast < 3 dB) | **COVERED** | `validate_idgham_no_ghunnah` | K\* |
| 13 | 101 | Noon sakinah/tanween | Iqlab before ب | — | 2 h nasal with light labial contact (no silent closure) | **COVERED** | `validate_iqlab_or_ikhfa_shafawi`, `labial_closure_ms` | K\* |
| 14 | 101 | Noon sakinah/tanween | Ikhfa' haqiqi before 15 letters; ghunnah heavy before isti'la letters | — | 2 h nasal; F2 anticipation (< 1300 Hz heavy / > 1700 light) | **COVERED** | `validate_ikhfa` | K\* |
| 15 | 101 | Meem sakinah | Ikhfa' shafawi before ب (with ghunnah) | Common Egyptian ada': a light gap between the lips (infirāj). Closing the lips is also taught | 2 h nasal; labial closure | **COVERED** | `meem_sakinah.validate_ikhfa_shafawi` | K\* |
| 16 | 101 | Meem sakinah | Idgham shafawi (meem into meem) | — | 2–2.5 h nasal | **COVERED** | `validate_idgham_shafawi` | K\* |
| 17 | 101 | Meem sakinah | Izhar shafawi, especially before و ف | — | ≤ 1.3 h, no lingering lip closure | **COVERED** | `validate_izhar_shafawi` (`izhar_shafawi:before_waw_faa` calibration key) | K\* |
| 18 | 101 | Ghunnah | Ghunnah mushaddadah on نّ مّ | — | 2 h nasal + NER | **COVERED** | `validate_ghunnah_mushaddadah` | K\* |
| 19 | 101 | Lam al-ta'reef | Lam qamariyyah: izhar before أ ب غ ح ج ك و خ ف ع ق ي م ه (ابغ حجك وخف عقيمه) | — | An audible lateral /l/ (voiced, F3 dip about 1 h) with no gemination of the next letter | **MISSING** | No `RuleType`. The unit is only aligned | J (dropped lam) |
| 20 | 101 | Lam al-ta'reef | Lam shamsiyyah: idgham into the 14 sun letters | — | No /l/; the sun letter is geminated | **MISSING** | The parser silences the lam (`_resolve_assimilation`) and `_is_article_lam` deliberately suppresses the idgham-class instance. Nothing checks either the lam's absence or the gemination | J |
| 21 | 101 | Lam | Lam of the Divine Name: heavy after fatha/damma, light after kasra (including a helper kasra) | — | F2−F1 against the reciter's light reference | **COVERED** | `parser._detect_weight` → `raa_lam_rules.validate_weight` | K |
| 22 | 101 | Qalqalah | Sughra (mid-word) / kubra (at waqf) / akbar (mushaddad at waqf, e.g. ٱلْحَقّ) | — | Closure plus release burst: high-band flux and energy rise; closure hold for akbar | **COVERED** | `qalqalah_engine.validate_qalqalah`, `detect_release_burst` | K (J if the letter is voweled or lost) |

### 1B. Intermediate — all madd types, idgham classes, lam, raa, tafkheem levels, hamzat al-wasl, saktat

| # | Level | Topic | Rule / phenomenon | Hafs specifics | Acoustic observable | Engine status | Evidence | Lahn |
|---|---|---|---|---|---|---|---|---|
| 23 | INT | Madd | Muttasil | Shatibiyyah 4 or 5. At waqf on the hamza (ٱلسَّمَآءِ) 4/5/6. Tayyibah: 4 with the qasr-al-munfasil routes, longer on others | Duration ÷ harakah | **COVERED** | `_MADD_TARGETS` (4,5); `final_hamza` → (4,6) | K (J when shortened to 2) |
| 24 | INT | Madd | Munfasil | Shatibiyyah 4 or 5; Tayyibah 2 (qasr) or 4/5 | Same | **COVERED** | `_TAYYIBAH_QASR` | K |
| 25 | INT | Madd | Lazim kalimi muthaqqal (ٱلضَّآلِّينَ) / mukhaffaf (ءَآلْـَٰٔنَ) | 6 | Same | **COVERED** | `_followed_by_original_sukun` | K (J when shortened) |
| 26 | INT | Madd | Lazim harfi (muqatta'at: نٓ، قٓ، صٓ، الٓمٓ …); 2-count letters حى طهر | 6; 2 for the two-letter names | Same | **COVERED** | `_LETTER_NAMES`, `_expand_muqattaat` | K |
| 27 | INT | Madd | ʿayn in كٓهيعٓصٓ (19:1) and عٓسٓقٓ (42:2) | 4 or 6 | Same | **COVERED** | `MADD_LEEN` "leen lazim (ʿayn)" (4,6) | K |
| 28 | INT | Madd | الٓمٓ ٱللَّهُ (3:1–2) joined: the mim is moved (fatha) → **6 or 2** | Hafs: both | Same | **BUG / PARTIAL** | Probe: `madd_lazim` (6,6) is always required, so a correct 2-count reading FAILs | K |
| 29 | INT | Madd | 'Arid lis-sukun (2/4/6) | Free choice, but it must stay **consistent** | Same | **COVERED** (range) | `TOLERANCE[MADD_ARID]=None`; `consistency_notes` | K |
| 30 | INT | Madd | Leen at waqf (خَوْفٍ، ٱلْبَيْتِ) 2/4/6 | Must not exceed the chosen 'arid length | Same | **PARTIAL** | The range is checked; the leen ≤ 'arid constraint is not | K |
| 31 | INT | Madd | Badal (ءَامَنُوا۟) | 2 | Same | **COVERED** | `MADD_BADAL` | K |
| 32 | INT | Madd | 'Iwad (tanween fath at waqf) | 2 | Same | **COVERED** | `_apply_waqf`, `MADD_IWAD` | J if dropped |
| 33 | INT | Madd | Silah sughra / kubra of haa' al-kinayah | Kubra: 4/5 (Shatibiyyah), 2 (Tayyibah qasr). Exceptions: فِيهِۦ مُهَانًا silah; يَرْضَهُ (no silah); أَرْجِهْ، فَأَلْقِهْ (sukun); وَيَتَّقْهِ (short kasra); أَنسَىٰنِيهُ، عَلَيْهُ ٱللَّهَ (damma) | Same | **COVERED** | The Madani small waw/yaa encode the exceptions; `SMALL_WAW`, `SMALL_YAA` | J (adding or dropping silah) |
| 34 | INT | Madd | Madd al-farq: ءَآلذَّكَرَيْنِ (6:143, 6:144), ءَآلْـَٰٔنَ (10:51, 10:91), ءَآللَّهُ (10:59, 27:59) | **Ibdal 6 (preferred) or tas-heel** of the second hamza | Duration; tas-heel = a weakened, breathy, shorter hamza-vowel with no 6-count vowel | **PARTIAL** | Only the 6-count reading is accepted; a correct tas-heel reading FAILs | K |
| 35 | INT | Madd | Madd tamkeen (حُيِّيتُم، ٱلنَّبِيِّـۧنَ) and two yaas across words (فِى يَوْمٍ); two waws (ءَامَنُوا۟ وَعَمِلُوا۟) | The madd letter is kept distinct from the following glide (no merging) | Duration plus a formant transition (glide) | **PARTIAL** | Measured as plain tabi'i; no check that there is no idgham | J (merged) |
| 36 | INT | Madd | Aqwa al-sababayn (the stronger cause wins): lazim > muttasil > 'arid > munfasil > badal | e.g. ٱلسَّمَآءِ at waqf ≥ 4; مَـَٔابٍ at waqf is 'arid, not badal | Same | **COVERED** | The order of the branches in `parser._detect_madd` | K |
| 37 | INT | Madd | Relations between madd types: muttasil ≥ munfasil; leen ≤ 'arid; ghunnah and madd lengths proportional to the tempo | Required by the sanad teachers | Cross-instance statistics | **MISSING** | — | K |
| 38 | INT | Idgham | Mithlayn saghir (ٱضْرِب بِّعَصَاكَ، بَل لَّا). Exception: a madd waw/yaa does not merge (قَالُوا۟ وَهُمْ، فِى يَوْمٍ) | — | One closure, one release, geminate length | **COVERED** | `idghaam_classes.validate_kamil` (`count_releases`) | J |
| 39 | INT | Idgham | Mutajanisayn: ت←د، د←ت، ط←ت، ث←ذ، ذ←ظ، ب←م. **Naqis** ط→ت (أَحَطتُ، بَسَطتَ، فَرَّطتُمْ) | Hafs: يَلْهَث ذَّٰلِكَ (7:176) and ٱرْكَب مَّعَنَا (11:42) are **idgham** (Shatibiyyah). Tayyibah also allows izhar | Releases; for naqis, the itbaq F2 lowering before ت | **COVERED** | `MUTAJANISAYN_PAIRS`, `validate_naqis` | J |
| 40 | INT | Idgham | Mutaqaribayn: ل←ر (قُل رَّبِّ، بَل رَّفَعَهُ), ق←ك نَخْلُقكُّم (77:20) | نخلقكم: **kamil** (preferred), naqis also related | Releases; for naqis, qaf isti'la F2 | **COVERED** (kamil) | `MUTAQARIBAYN_PAIRS`. The naqis variant is not modelled | J |
| 41 | INT | Idgham | Izhar of the verb lam / the lam of هل and بل before letters other than ر and ل (جَعَلْنَا، قُلْ نَعَمْ، هَلْ تَعْلَمُ) | Izhar is obligatory | An audible /l/, no gemination | **MISSING** | — | J |
| 42 | INT | Raa | Core raa rules: heavy with fatha/damma; light with kasra; sakin raa follows the vowel before it. Exceptions: an 'aaridh kasra (ٱرْجِعِىٓ، إِنِ ٱرْتَبْتُمْ) or an isti'la letter after it (مِرْصَادًا، قِرْطَاسٍ، فِرْقَةٍ) makes it heavy | — | F2−F1 against the light reference | **COVERED** | `parser._raa_verdict`, `raa_lam_rules.validate_weight` | K |
| 43 | INT | Raa at waqf | Raa sakin at waqf after a sakin letter or yaa (ٱلْقَدْرِ، خَيْرٌ، ٱلذِّكْرِ) | — | Same | **COVERED** | `_raa_verdict` ("sakin after yaa leen at a stop") | K |
| 44 | INT | Raa | Jawaz al-wajhayn: فِرْقٍ (26:63), مِصْرَ at waqf, ٱلْقِطْرِ at waqf (34:12) | Ibn al-Jazari: مِصْر heavy preferred, ٱلْقِطْر light preferred; فِرْق both (light more common in ada') | Report which variant was realised | **COVERED** | `JAWAZ_WAJHAYN` | K |
| 45 | INT | Raa at waqf | يَسْرِ (89:4), وَنُذُرِ (al-Qamar ×6), أَنْ أَسْرِ / فَأَسْرِ (at waqf) | **Both**; tarqeeq preferred because of the deleted yaa | Same | **BUG / PARTIAL** | Probe: 89:4 and 54:16 → `tafkheem` "sakin after fathah/dammah", so tarqeeq, the preferred reading, FAILs | K |
| 46 | INT | Raa | Imala kubra of مَجْر۪ىٰهَا (11:41): the raa is light | Hafs's only imala | F1 fall / F2 rise toward /eː/ on the alif; a light raa | **BUG / MISSING** | Probe: the imala mark U+06EA is dropped by `_DROP`, the raa is treated as sakin and gets `tafkheem` "sakin after fathah". The correct reading FAILs; the imala itself is never checked | J (leaving out the imala changes the riwayah) |
| 47 | INT | Tafkheem | Isti'la letters with fatha/damma are heavy | — | F2−F1 collapse | **COVERED** | `_detect_weight`, `judge_weight` | K |
| 48 | INT | Tafkheem | **5 levels** (maratib): fatha+alif > fatha / sakin after fatha > damma / sakin after damma > sakin after kasra > kasra. Isti'la letters with kasra (قِ خِ غِ صِ) keep a lower-level tafkheem | Common ada' in Egypt and Sham | Graded heaviness index H, rising monotonically across the levels | **PARTIAL** | Binary heavy/light only. Heavy letters with kasra or sukun get **no instance** (`if u.vowel in (FATHA, DAMMA)`). ط/ق sakin get qalqalah only | K |
| 49 | INT | Tarqeeq | Light (istifal) letters stay light next to heavy ones (no leakage: تَطَّلِعُ، ٱلسَّمَٰوَٰتِ، أَحَطتُ is the exception); a light alif after a light letter | — | H ≈ 0 on light-letter vowels near heavy letters | **MISSING** | The reference *excludes* letters near heavy ones (`_near_heavy`), so they are never judged | K |
| 50 | INT | Hamzat al-wasl | Dropped in wasl (no glottal stop) | — | Gap plus an abrupt onset | **COVERED** | `sakt_wasl.validate_hamzat_wasl` | J (inserting a hamza) |
| 51 | INT | Hamzat al-wasl | Ibtida' vowel: fatha on ال; in verbs, damma if the 3rd letter has an **original** damma, otherwise kasra; kasra in the nouns ٱبْن، ٱبْنَت، ٱمْرُؤ، ٱمْرَأَت، ٱثْنَيْن، ٱثْنَتَيْن، ٱسْم | Exceptions: ٱمْشُوا۟، ٱقْضُوٓا۟، ٱبْنُوا۟، ٱئْتُوا۟ (the damma is 'aaridh → kasra) | Vowel quality of the initial syllable | **BUG / PARTIAL** | Probe: ٱمْشُوا۟، ٱقْضُوٓا۟، ٱمْرُؤٌا → **damma** (should be kasra) in `_resolve_wasla`. No validator checks the ibtida' vowel at all | J |
| 52 | INT | Hamzat al-wasl | ٱلِٱسْمُ (49:11) at ibtida': ٱَلِسْمُ or لِسْمُ | Both | Onset | **PARTIAL** | Only the first way is produced; neither is judged | J |
| 53 | INT | Two sakins meeting | Kasra for joining after a tanween or sakin before hamzat al-wasl (نُوحٌ ٱبْنَهُۥ → نُوحُنِ ٱبْنَهُۥ) | — | Vowel after the nasal | **PARTIAL** | The text is resolved (helper kasra in the lam-of-Allah logic); no check | J |
| 54 | INT | Saktat | The 4 mandatory saktat of Shatibiyyah: عِوَجَا ۜ (18:1), مَّرْقَدِنَا ۜ (36:52), مَنْ ۜ رَاقٍ (75:27), بَلْ ۜ رَانَ (83:14) | ~2 h, breathless; waqf at 18:1 and 36:52 is also allowed | A silence of 200–400 ms with no breath (flatness) | **COVERED + 2 BUGS** | `validate_sakt`. **Bug A**: `_prepare_words` marks sakt when U+06DC appears *anywhere* in a token, so the "read seen" mark in وَيَبْصُۜطُ (2:245) and بَصْۜطَةً (7:69) puts a **false sakt after يَقْبِضُ and ٱلْخَلْقِ** (probe confirmed). **Bug B**: at 18:1 and 36:52 a sakt instance is still created when the reciter makes a full, valid waqf, so the waqf FAILs with "A full stop was made instead of a brief sakt" | K (J by some; leaving it out can change the meaning) |
| 55 | INT | Saktat | مَالِيَهْ ۜ هَلَكَ (69:28–29): sakt or idgham | Both | Silence, or a geminate ه | **COVERED** | `optional` branch | K |
| 56 | INT | Between surahs | Basmalah options; al-Anfal→al-Tawbah: waqf, sakt or wasl, with no basmalah | — | Pause length, breath, presence of the basmalah | **MISSING** | — | K |
| 57 | INT | Haa' al-sakt | كِتَٰبِيَهْ، حِسَابِيَهْ (69:19–20, 25–26), مَالِيَهْ، سُلْطَٰنِيَهْ، مَا هِيَهْ (101:10), يَتَسَنَّهْ (2:259), ٱقْتَدِهْ (6:90): the haa is kept in wasl and waqf | — | Audible /h/ (aspiration noise) | **PARTIAL** | Resolved as sakin haa, which gets hams/rakhawah instances; dropping the haa is not detected as an error | J |

### 1C. Advanced — makharij, sifaat, waqf and ibtida', raum, ishmam, ibdal, maratib, tasawi

#### 1C-i. The 17 makharij (al-Khalil / Ibn al-Jazari system)

| # | Level | Region | Makhraj → letters | Acoustic observable | Engine status | Evidence | Lahn |
|---|---|---|---|---|---|---|---|
| 58 | ADV | Jawf | 1. Jawf → the madd letters ا و ي | Vowel quality and duration of the long vowel | **PARTIAL** (duration only) | `mudood_engine` | J/K |
| 59 | ADV | Halq | 2. Aqsa al-halq → ء ه | Glottal stop (silence plus a burst); /h/ aspiration | **MISSING** | — | J |
| 60 | ADV | Halq | 3. Wasat al-halq → ع ح | Pharyngeal: F1 up, F2 down; ع creaky or voiced, ح voiceless noise | **MISSING** | — | J |
| 61 | ADV | Halq | 4. Adna al-halq → غ خ | Uvular fricative noise at 1–3 kHz; voicing tells غ from خ | **MISSING** | — | J |
| 62 | ADV | Lisan | 5. Aqsa al-lisan → ق | Uvular burst, low F2 | **MISSING** (qalqalah only) | — | J |
| 63 | ADV | Lisan | 6. Slightly lower → ك | Velar burst, aspirated (hams) | **MISSING** | — | J |
| 64 | ADV | Lisan | 7. Middle of the tongue → ج ش ي | Affricate burst plus frication (ج); broad noise (ش) | **PARTIAL** (ش tafashhi) | `validate_tafashhi` | J |
| 65 | ADV | Lisan | 8. Edge(s) of the tongue and molars → ض | Lateral-emphatic: long, voiced, low F2, no burst | **PARTIAL** (istitaalah only when sakin) | `validate_istitaalah` | J (ض→د/ظ) |
| 66 | ADV | Lisan | 9. Tip-edge → ل | Lateral: an F3 dip, anti-resonance | **MISSING** | — | J |
| 67 | ADV | Lisan | 10. Tip → ن | Nasal murmur | **COVERED** (nasal rules) | `formants.measure_nasality` | J/K |
| 68 | ADV | Lisan | 11. Tip, back side → ر | Tap (one closure), low F3 | **PARTIAL** (takreer taps, weight) | `validate_takreer` | K |
| 69 | ADV | Lisan | 12. Tip plus upper incisor roots → ت د ط | Alveolar burst; VOT tells ت from د; F2 tells ط | **PARTIAL** (qalqalah د ط; itbaq ط) | — | J |
| 70 | ADV | Lisan | 13. Tip plus the lower incisors → ص س ز | Sibilant peak > 4 kHz (safir); F2 for ص | **PARTIAL** (safir) | `validate_safir` | J |
| 71 | ADV | Lisan | 14. Tip plus the edges of the upper incisors → ث ذ ظ | Weak, flat interdental noise (no sibilant peak) | **MISSING** (the ث/س and ذ/ز confusion is not tested) | — | J |
| 72 | ADV | Shafatan | 15. Inner lower lip plus upper incisors → ف | Labiodental noise | **MISSING** | — | J |
| 73 | ADV | Shafatan | 16. Both lips → ب م و | Bilabial closure; lip rounding for و | **PARTIAL** (labial closure in iqlab/ikhfa' shafawi) | `labial_closure_ms` | J |
| 74 | ADV | Khayshum | 17. Nasal cavity → ghunnah | Nasal energy ratio (NER) | **COVERED** | `nasal_hold`, `nasal_verdict` | K\* |

#### 1C-ii. The 17 sifaat (5 opposing pairs + 7 without opposites; tawassut is the middle category of shiddah/rakhawah)

| # | Level | Sifah | Letters | Acoustic observable | Engine status | Evidence | Lahn |
|---|---|---|---|---|---|---|---|
| 75 | ADV | Hams / Jahr | فحثه شخص سكت / the rest | Voicing fraction, HNR; aspiration of ت and ك **even when voweled** | **PARTIAL**: sakin letters only (and not qalqalah, ن, م or hamza). Hams on a voweled ت/ك (a classic ijazah point) is not checked | `hams_jahr.validate_hams_jahr`; parser `_detect_sifaat` (`u.sukun` guard) | K |
| 76 | ADV | Shiddah / Tawassut / Rakhawah | أجد قط بكت / لن عمر / the rest | Full occlusion / partial / continuous noise | **PARTIAL** (sakin only; the duration ratios are an unvalidated model) | `sukoon_spectrum.validate_sukoon_class`, `SUKOON_RATIOS` | K |
| 77 | ADV | Isti'la / Istifal | خص ضغط قظ / the rest | F2−F1 collapse | **PARTIAL** (fatha/damma only; see #48–49) | `judge_weight` | K |
| 78 | ADV | Itbaq / Infitah | ص ض ط ظ / the rest | F2 lowering, a wider F3−F2, attenuation above 3.5 kHz | **PARTIAL** (fatha/damma only; infitah is not judged) | `itbaq.validate_itbaq` | K |
| 79 | ADV | Idhlaq / Ismat | فر من لب / the rest | Lexical classification (ease of articulation) | **N/O** | — | — |
| 80 | ADV | Safir | ص س ز | Spike above 5 kHz | **COVERED** | `validate_safir` | K |
| 81 | ADV | Qalqalah | قطب جد | Burst | **COVERED** | `qalqalah_engine` | K |
| 82 | ADV | Leen | و ي sakin after fatha | Diphthong glide (F2 trajectory) plus duration | **PARTIAL** (duration only, and only at waqf) | `MADD_LEEN` | K |
| 83 | ADV | Inhiraf | ل ر | Laterality, retroflex F3 | **MISSING** | — | K |
| 84 | ADV | Takreer (to be *restrained*) | ر | Tap count ≤ 1 | **COVERED** (sakin and mushaddad raa only) | `validate_takreer` | K |
| 85 | ADV | Tafashhi | ش | Flat 2.5–6 kHz noise | **COVERED** | `validate_tafashhi` | K |
| 86 | ADV | Istitaalah | ض | Long lateral, no burst | **PARTIAL** (sakin only) | `validate_istitaalah` | J (when it becomes د/ظ) |
| 87 | ADV | Ghunnah (sifah lazimah of ن م) | ن م | NER | **COVERED** | nasal rules | K |
| 88 | ADV | Khafa' (hidden letters) | ه and the madd letters: هُ / هِ clarity, especially at word end (عَلَيْهِ، إِلَيْهِ) | Audible aspiration energy | **MISSING** | — | J (dropped) / K |
| 89 | ADV | Strong/weak letters (quwwah/du'f by sifaat count) | e.g. ط strongest; ه ف weakest | Derived: pedagogical weighting only | **N/O** (could weight errors) | — | — |

#### 1C-iii. Waqf, ibtida', raum, ishmam, ibdal, maratib, tasawi

| # | Level | Topic | Rule / phenomenon | Hafs specifics | Acoustic observable | Engine status | Evidence | Lahn |
|---|---|---|---|---|---|---|---|---|
| 90 | ADV | Waqf kinds | Ikhtiyari: tamm / kafi / hasan / **qabih**; plus idtirari, intizari, ikhtibari | — | Pause location (from the alignment) matched against a waqf-quality lexicon | **MISSING** | `fatigue_detector.dynamic_stops` accepts a stop *anywhere* without judging its quality | K (qabih stops can be J in meaning) |
| 91 | ADV | Waqf signs | ۘ (lazim), ۖ (al-wasl awla), ۗ (al-waqf awla), ۚ (ja'iz), ۙ (la), ۛ mu'anaqah | Madinah mushaf conventions | Same | **MISSING** | `parser._DROP` strips U+06D6–U+06DC | K |
| 92 | ADV | Mu'anaqah | Stop at only one of the two ۛ marks (e.g. 2:2) | — | Pause count | **MISSING** | — | K |
| 93 | ADV | Ibtida' | Resuming at a meaningful place (going back after a qabih stop) | — | Repeated words after a pause (the alignment must allow repetition) | **MISSING** (the forced aligner cannot model a restart) | `aligner.py` | K |
| 94 | ADV | Waqf manner | Sukun mahd: the last letter's vowel is removed completely | Default | No voiced vowel after the last consonant | **MISSING** (the parser *assumes* it; nothing checks it) | `_apply_waqf` | J/K |
| 95 | ADV | Raum | Voicing about ⅓ of a damma/kasra at waqf | Optional (the ada' of the ijazah track) | A short, weak vowel tail (≈ ⅓ h, −10 dB) | **MISSING** | — | K |
| 96 | ADV | Ishmam at waqf | Silent lip rounding after a damma | Optional | Lip rounding with no sound | **N/O** (visual; video only) | — | — |
| 97 | ADV | Ta' marbutah at waqf | Becomes a haa sakinah | — | /h/ noise vs a /t/ burst | **MISSING** (the text converts it; nothing checks it) | `_apply_waqf` sets `ه` | J |
| 98 | ADV | Ibdal | Tanween fath → alif ('iwad); madd farq ibdal (#34) | — | Duration | **COVERED** / **PARTIAL** | #32, #34 | J |
| 99 | ADV | Alifs pronounced **only at waqf** (rectangular zero ۠ U+06E0): أَنَا۠ (everywhere), لَّٰكِنَّا۠ (18:38), ٱلظُّنُونَا۠ (33:10), ٱلرَّسُولَا۠ (33:66), ٱلسَّبِيلَا۠ (33:67), قَوَارِيرَا۠ (76:15) | Waqf **with** the alif, wasl without it | /aː/ at waqf | **BUG** | Probe: `_make_unit` treats U+06E0 exactly like U+06DF, so it is **always silent**. 33:10 at waqf → `madd_arid` on و with a sakin ن; أَنَا۠ at waqf → "an". A correct waqf FAILs and a wrong one passes | J |
| 100 | ADV | سَلَٰسِلَا۟ (76:4) | Waqf with the alif or with sukun (both); قَوَارِيرَا۟ (76:16): sukun only; ثَمُودَا۟ (11:68, 25:38, 29:38, 53:51): sukun | Rounded zero = silent / two ways | Same | **PARTIAL** | The sukun reading is expected (correct for 76:16 and ثمودا). For سلاسلا the alif reading is not recognised as valid (no instance; it passes silently) | J |
| 101 | ADV | فَمَآ ءَاتَىٰنِۦَ (27:36) | Waqf with the yaa or without it | Both | Presence of /iː/ | **MISSING** | — | J |
| 102 | ADV | Maratib al-qira'ah | Tahqeeq / tadweer / hadr (+ tartil) | All valid; proportions must hold within the chosen pace | Harakah ms | **COVERED** (descriptive) | `pace_normalizer.classify_pace` (<150 hadr, ≤220 tadweer); `hadr_to_tahqeeq_ratio` | — |
| 103 | ADV | Tanasub (proportionality) | Madd and ghunnah lengths scale with the pace | — | Counts ÷ **local** harakah | **COVERED** | `EvalContext.haraka_ms(t)`, `build_local_tempo` | K |
| 104 | IJZ | **Tasawi al-mudood** | The same count for every occurrence of each madd type (all munfasil equal; all muttasil equal; all 'arid equal within a sitting) | Ijazah requirement | Within-type variance of the counts | **PARTIAL** | `scoring.consistency_notes`: a note when the spread is > 1.5 counts, for muttasil, munfasil and 'arid only. Not scored; ghunnah, leen and lazim are not included | K |
| 105 | IJZ | Consistency of the chosen wujuh | Once a wajh is chosen (e.g. munfasil 4, 'arid 4, tas-heel vs ibdal, sad vs seen in المصيطرون) it is kept for the whole reading; routes are not mixed (talfeeq), e.g. Tayyibah qasr al-munfasil with Shatibiyyah-only choices | Ijazah requirement | Categorical choice per instance → a consistency check across instances | **MISSING** | `Tareeq` changes only the madd targets (`_TAYYIBAH_QASR`) | K (talfeeq is prohibited in riwayah) |
| 106 | IJZ | Ghunnah maratib | mushaddad ≥ mudgham ≥ mukhfa > izhar sakin > mutaharrik | — | An ordering of the nasal durations and NER | **PARTIAL** (absolute targets 2–2.5; no ordering) | `nasal_hold` | K |
| 107 | IJZ | Stability of pitch and rhythm (no tatrib/tarqis extremes; no tremor) | Not a riwayah rule, but examiners comment on it | F0 contour, tempo variability | **PARTIAL** (reported, not judged) | `pitch_profile`, `local.variability` | — |

### 1D. Hafs special cases (rows here; the verified word list is in §3)

| # | Level | Case | Hafs (Shatibiyyah) | Tayyibah note | Acoustic observable | Engine status | Evidence | Lahn |
|---|---|---|---|---|---|---|---|---|
| 108 | IJZ | لَا تَأْمَ۫نَّا (12:11) | Idgham with **ishmam** (lip rounding at the merged noon) or **raum/ikhtilas** (a partial damma with izhar of both noons) | Same | Ishmam: N/O acoustically (visual). Raum: a short weak vowel between two /n/ + reduced gemination | **MISSING** (the U+06EB mark is dropped; the word is parsed as a plain نّ ghunnah, so a correct raum reading FAILs on the ghunnah duration) | `_DROP` covers U+06EA–U+06ED | K |
| 109 | IJZ | ءَا۬عْجَمِىٌّ (41:44) | **Tas-heel** of the second hamza (between hamza and alif), with no madd | Same | A weak, glottalised, short vowel; **no** 6-count vowel | **BUG** | Probe: `madd_lazim` (6,6) is generated ("kalimi: madd before an original sukun"). Correct recitation FAILs | J |
| 110 | IJZ | مَجْر۪ىٰهَا (11:41) | Imala kubra; a light raa | Same | Formant shift of the alif to /eː/ | **BUG** | See #46 | J |
| 111 | IJZ | ضَعْفٍ، ضَعْفٍ، ضَعْفًا (30:54) | Fath (preferred, Hafs chose it) **or** damm of the ض | Same | F1/F2 of the vowel after ض | **PARTIAL** (only fath is modelled; a damm reading is judged as a tafkheem instance and probably passes, but it is not recognised as a wajh) | parser | J if another vowel is read |
| 112 | IJZ | وَيَبْصُۜطُ (2:245), بَصْۜطَةً (7:69) | Read with **seen** | Tayyibah: sad also allowed | Sibilant without itbaq (high F2, safir) vs ص | **BUG / MISSING** | The mark is interpreted as sakt (#54 bug A); the letter is still treated as ص, so a correct seen reading gets itbaq/tafkheem FAILs | J |
| 113 | IJZ | ٱلْمُصَۣيْطِرُونَ (52:37) | Sad (preferred) or seen | Same | Same | **MISSING** (U+06E3 dropped; only sad accepted) | `_DROP` | J |
| 114 | IJZ | بِمُصَيْطِرٍ (88:22) | Sad only | Tayyibah: seen also related | Same | **COVERED** (as ص) | — | J |
| 115 | IJZ | يسٓ وَٱلْقُرْءَانِ (36:1–2), نٓ وَٱلْقَلَمِ (68:1) | **Izhar** of the noon | Tayyibah: idgham via some turuq | Short noon, no nasal hold before و | **PARTIAL** (idgham correctly suppressed via `izhar_exceptions`; no izhar instance checked) | parser `izhar_exceptions` | K\* |
| 116 | IJZ | مَكَّنِّى (18:95), تَأْمَ۫نَّا | One noon mushaddad (idgham) | — | Geminate nasal | **COVERED** (as ghunnah) | — | J |
| 117 | IJZ | 4 saktat + مَالِيَهْ | See #54–55 | Tayyibah: idraj (no sakt) allowed with qasr al-munfasil via some routes | Silence without breath | **COVERED + BUGS** | #54 | K |
| 118 | IJZ | Madd al-farq words | See #34 | — | — | **PARTIAL** | — | K |
| 119 | IJZ | الٓمٓ ٱللَّهُ (3:1–2) | 6 or 2 on the mim in wasl | — | Duration | **BUG / PARTIAL** | #28 | K |
| 120 | IJZ | Alifs pronounced only at waqf | See #99 | — | — | **BUG** | #99 | J |

---

### Coverage counts (rows 1–120)

| Status | Count | Notes |
|---|---|---|
| COVERED (including "COVERED + bugs", "COVERED (descriptive)" and "COVERED (as …)") | **46** | Madd lengths, noon/meem sakinah, ghunnah, qalqalah, idgham classes, core raa and lam-of-Allah, hamzat al-wasl dropping, safir/tafashhi/takreer, pace |
| PARTIAL (including "PARTIAL (…)") | **31** | Typically: only sakin positions, binary rather than graded, text resolved but not judged, or an alternative wajh not accepted |
| MISSING (not counting the BUG rows) | **30** | Makharij identity, harakat, lam shamsiyyah/qamariyyah, waqf/ibtida', raum, wujuh consistency |
| BUG (the engine expects the wrong reading; includes BUG/PARTIAL and BUG/MISSING) | **10** rows (#28, 45, 46, 51, 99, 109, 110, 112, 119, 120) plus the 2 sakt bugs inside #54 | Several are the *same* defect seen at two levels (#46/#110, #99/#120, #28/#119) |
| N/O (not acoustically observable) | **3** | Idhlaq/ismat, strength ranking, ishmam at waqf |

Engine rule inventory: 39 `RuleType`s (10 madd, 5 noon, 3 meem, ghunnah, qalqalah, 3 idgham classes, 3 weight, 2 wasl/sakt, 10 sifaat). Every one has a validator (`scoring.VALIDATORS`); 34 have calibration bands in `metric_spec.json` (madd_arid and madd_leen deliberately keep the textbook verdict; jawaz_wajhayn always passes; itbaq falls back to the validator).

**Big picture.** The engine is strong on **timing and nasality** (the "ahkam" layer that the 101 and intermediate levels drill). It is almost blind to the **lahn jali layer**: wrong letter, wrong harakah, dropped or added letters, lam al-ta'reef, waqf manner. It also mis-handles **Hafs-specific orthography marks**, because `_DROP` deletes U+06E3 and U+06EA–U+06ED and the parser gives U+06DC and U+06E0 the wrong meaning. An ijazah examiner would stop a student first for lahn jali, then for Hafs special cases, then for tasawi. Those three are the engine's weakest areas.

---

## 2. Gap list, ranked by pedagogical importance × feasibility

Scores are 1–5 (I = importance to a teacher or examiner, F = feasibility with the current stack of CTC alignment, formants, NER and local tempo). Rank = I × F; bugs that penalise correct recitation are ranked first no matter the score, because they make the engine wrong rather than incomplete.

### Tier 0 — correctness bugs (fix before adding anything)

| Rank | Gap | I×F | Measurement / fix design |
|---|---|---|---|
| 0.1 | **False sakt from U+06DC used as the "read seen" mark** (2:245, 7:69) | 5×5 | In `_prepare_words`, treat U+06DC as a sakt only when it is a **stand-alone token** (or the final mark after a completed word followed by a space), not when it sits over ص inside a word. When it sits inside a word over ص, set the letter to س (see 0.6). Add regression tests for 2:245, 7:69, 18:1, 36:52, 75:27, 83:14 |
| 0.2 | **Rectangular zero U+06E0 treated as always silent** (أَنَا، لَّٰكِنَّا، ٱلظُّنُونَا، ٱلرَّسُولَا، ٱلسَّبِيلَا، قَوَارِيرَا 76:15) | 5×5 | In `_make_unit`, mark U+06E0 as `waqf_only=True` (silent in wasl). In `_apply_waqf`, if the tail is `waqf_only`, pronounce it as a madd alif (tabi'i 2; no 'arid on the previous letter). U+06DF stays always silent. سَلَٰسِلَا۟ (rounded zero, but two ways in waqf): add a small `WAQF_TWO_WAYS` lexicon accepting either result |
| 0.3 | **ءَا۬عْجَمِىٌّ gets a 6-count madd lazim** (41:44) | 5×5 | Keep U+06EC (it is removed from `_DROP`). The alif under it becomes `LetterUnit(tasheel=True)`, with no madd. New rule `TASHEEL`: the vowel lasts ≤ 1.3 h, and glottalisation (a jitter or creak burst, weak onset) is present but without a full glottal closure (no silence ≥ 20 ms before it) |
| 0.4 | **Imala of مَجْر۪ىٰهَا: raa judged heavy** (11:41) | 5×4 | Keep U+06EA and give the raa `vowel=IMALA`. `_raa_verdict` → light. New rule `IMALA`: on the alif core, F2 is at least 30% above the reciter's /a/ reference and F1 at least 15% below it, moving toward the reciter's /i/ centroid (target mid-point /eː/) |
| 0.5 | **Sakt demanded when a valid waqf is made** (18:1, 36:52) | 4×5 | When `stop_after` is also set at the sakt position, make the sakt instance "waqf or sakt". Pause ≥ 800 ms or with a breath → treat as waqf (PASS). The fatha tanween of 18:1 is then an 'iwad alif |
| 0.6 | **Seen/sad lexicon**: يَبْصُۜطُ، بَصْۜطَةً → seen; ٱلْمُصَۣيْطِرُونَ → both; بِمُصَيْطِرٍ → sad | 4×4 | Map U+06DC over ص inside a word to "read س" and U+06E3 (small seen below) to "either". Validator: itbaq F2 index H (existing `judge_weight`) near 0 means seen, high means sad; together with the existing safir spike. Report the wajh chosen |
| 0.7 | **Raa at waqf in يَسْرِ / نُذُرِ / أَسْرِ** | 4×5 | Add a waqf-only lexicon in `_raa_verdict` → `JAWAZ_WAJHAYN` (preferred: light) |
| 0.8 | **الٓمٓ ٱللَّهُ, mim 6 or 2 in wasl** | 3×5 | When the muqatta'at mim is followed by ٱللَّه in wasl, use the expected range (2,6) with two bands (2 or 6). Anything near 4 gives a WARNING |
| 0.9 | **Hamzat al-wasl ibtida' vowel** (ٱمْشُوا، ٱقْضُوا، ٱبْنُوا، ٱئْتُوا، ٱمْرُؤ، ٱمْرَأَت، ٱبْن، ٱسْم، ٱثْنَيْن، ٱثْنَتَيْن) | 4×5 | Add a noun lexicon (always kasra) and an 'aaridh-damma lexicon (kasra). Then add a validator `IBTIDA_WASL_VOWEL`: check that the formants of the first vowel match the expected /a/, /i/ or /u/ centroid (from the reciter's own reference, `build_reference`) |
| 0.10 | **Madd al-farq tas-heel not accepted** | 3×4 | Make it two-wajh: 6 counts **or** the tas-heel signature from 0.3. Record the chosen wajh for the consistency check (Tier 1, #5) |
| 0.11 | **تَأْمَ۫نَّا** | 3×3 | Keep U+06EB. Accept (a) the ghunnah geminate (the ishmam is silent; report "ishmam not verifiable acoustically") or (b) raum: two /n/ murmurs separated by a weak vowel of ≤ 0.5 h, ≥ 6 dB below the neighbouring vowels |

### Tier 1 — high importance, feasible now

| Rank | Gap | I×F | Measurement design |
|---|---|---|---|
| 1 | **Letter/harakah correctness (lahn jali) from GOP** | 5×4 = 20 | Run the same CTC model **unconstrained** (or with a confusion-set lattice: for each target letter, allow {target, its confusions such as ض/د/ظ, ث/س, ذ/ز, ح/ه, ع/ء, ق/ك, ط/ت, ص/س, and the vowels a/i/u}). Score each unit by GOP = log P(target) − max log P(competitor) over its aligned frames. Flag GOP < τ (τ calibrated on Husary and peers the way `calibrate.jl` does). This also fixes the current perverse effect: a low `align_conf` must become a **suspected error**, not SKIPPED, whenever the audio quality is good |
| 2 | **Lam shamsiyyah / qamariyyah** | 5×4 = 20 | New `LAM_QAMARIYYAH`: an /l/ interval ≥ 0.5 h, voiced, with an F3 dip relative to the vowel before it. New `LAM_SHAMSIYYAH`: the sun letter's duration ≥ 1.6 × its median singleton duration in the same recitation, and no lateral segment (GOP of "l" < τ) |
| 3 | **Shaddah on all letters** | 5×4 = 20 | For every `u.shadda` not already covered: duration of the consonant (closure for stops, frication for fricatives from `frication_window`, murmur for sonorants) ÷ the median singleton of the same class ≥ 1.6. For stops, also require a single release (`count_releases`) |
| 4 | **Waqf manner (sukun mahd), ta' marbutah → haa, haa' al-sakt kept** | 4×5 = 20 | At every `stop_after`, the last letter: no voiced vowel core > 40 ms after the consonant (reuse `vowel_core_ms`). For ه (ta' marbutah and sakt haa): aspiration noise ≥ 40 ms with a voicing fraction < 0.3 and no /t/ burst (`detect_release_burst` must be absent) |
| 5 | **Tasawi al-mudood / wujuh consistency (scored)** | 5×4 = 20 | Promote `consistency_notes` to a scored `TASAWI` family. For each type (munfasil, muttasil, 'arid, leen, silah kubra, ghunnah): robust CV of the calibrated counts, where CV ≤ reference-reciter CV + 2s means PASS. Add the relations muttasil ≥ munfasil − 0.3 and leen ≤ 'arid + 0.3. For categorical wujuh (tas-heel/ibdal, sad/seen, 'arid 2/4/6 cluster via a nearest-of-{2,4,6} vote), flag any change within one session. Also check the talfeeq matrix per `Tareeq` |
| 6 | **Short vowel length (ishba' / clipping)** | 4×5 = 20 | For every short-vowel unit: `vowel_core_ms / haraka_ms` must fall in [0.6, 1.5]; above 1.6 (an added madd) is J. Exclude the vowels before a waqf (phrase-final lengthening) |
| 7 | **Tafkheem levels (5 maratib) plus heavy letters with kasra or sukun** | 4×4 = 16 | Emit TAFKHEEM for every isti'la letter with its `level` (1–5). Validator: the heaviness index H must fall in a level-specific band calibrated on the reference reciters, and the order must be monotonic within the recitation (Spearman ρ between level and median H > 0). For a sakin heavy letter, measure the vowel before it |
| 8 | **Tarqeeq leakage (istifal letters next to heavy ones, the light alif)** | 4×4 = 16 | Judge light-letter vowels *within* 2 units of a heavy letter (currently excluded from the reference) against the far-context light reference: H < 0.10 is PASS |
| 9 | **Waqf and ibtida' quality** | 5×3 = 15 | Keep the waqf signs (U+06D6–U+06DB) as `Word.waqf_sign`. For each detected stop (`dynamic_stops`): (a) ۙ, or a mid-construction stop from a waqf-quality lexicon (for example the Dani/Ushmuni classes digitised from Mushaf al-Madinah marks plus a qabih list) → WARNING "waqf qabih; resume from a meaningful place"; (b) ۛ: flag stops at both marks. Ibtida': allow repetitions in the aligner (a loop-back arc in the CTC graph at word boundaries) and check that the restart point is ≤ the stop point and is a valid ibtida' |
| 10 | **Izhar checks where the engine now emits nothing** (يس والقرآن، ن والقلم، دنيا/بنيان/صنوان/قنوان; lam of verbs and of هل/بل before non-ر/ل) | 3×5 = 15 | Emit `IZHAR_HALQI`-style `clear_letter` instances with detail "izhar mutlaq / exception"; for lam, the /l/ presence test from #2 |
| 11 | **Hams of voweled ت/ك (aspiration) and khafa' of ه** | 4×3 = 12 | For ت and ك with a vowel: the aspiration interval after the burst (VOT) ≥ 25 ms of noise with a voicing fraction < 0.3. For word-final هُ/هِ: aspiration energy above the room floor + 10 dB, ≥ 30 ms |
| 12 | **Makharij of the throat and interdental letters** (ع ح ء ه غ خ; ث ذ ظ vs س ز) | 5×2 = 10 | Mostly through GOP (#1). Targeted cues: ح/ه spectral centroid and F1 of the adjacent vowel; ع: the voiced pharyngeal F1↑ F2↓ constriction relative to the reference; ث/س: no sibilant peak (ratio of energy in 4–8 kHz to 2–4 kHz, < reference − 6 dB) |
| 13 | **Raum at waqf** | 2×4 = 8 | At a stop on a damma or kasra: a vowel tail of 0.2–0.5 h, ≥ 8 dB weaker than the vowel before it → report "raum" (PASS). Ishmam stays N/O without video |
| 14 | **Ghunnah maratib and the ordering of the nasals** | 3×3 = 9 | Rank test inside the recitation: median NER and duration of mushaddad ≥ mudgham ≥ mukhfa > izhar |
| 15 | **Between-surah options, Anfal→Tawbah, basmalah** | 2×4 = 8 | A text-level check that the basmalah is present or absent (GOP); pause classification (waqf/sakt/wasl) with the existing breath detector |
| 16 | **Leen diphthong quality and inhiraf** | 2×3 = 6 | Leen: an F2 trajectory from /a/ toward /i/ or /u/ over ≥ 60% of the span. Inhiraf/ل: an F3 dip ≥ 200 Hz below the neighbouring vowel |

---

## 3. Hafs special-case word list (Uthmani text copied from the engine's own Tanzil file `~/.cache/qaari-eval/text/quran-uthmani.json`; every reference opened and checked)

| # | Word (Uthmani) | Ref (surah:ayah) | Hafs ruling (Shatibiyyah) | Tayyibah note | Engine today (probe) |
|---|---|---|---|---|---|
| 1 | لَا تَأْمَ۫نَّا | 12:11 | Idgham with ishmam, or raum (ikhtilas) with izhar | Same | Plain ghunnah; mark U+06EB dropped |
| 2 | ءَا۬عْجَمِىٌّ | 41:44 | Tas-heel of the 2nd hamza, no madd | Same | **Madd lazim 6 required (BUG)** |
| 3 | مَجْر۪ىٰهَا | 11:41 | Imala kubra of the raa-alif | Same | **Raa tafkheem required (BUG)** |
| 4 | ضَعْفٍ · ضَعْفٍ · ضَعْفًا | 30:54 (three times) | Fath or damm of the ض (fath preferred) | Same | Fath only |
| 5 | وَيَبْصُۜطُ | 2:245 | Seen | Sad also | Treated as ص; **false sakt after يَقْبِضُ (BUG)** |
| 6 | بَصْۜطَةً | 7:69 | Seen | Sad also | Treated as ص; **false sakt after ٱلْخَلْقِ (BUG)** |
| 7 | ٱلْمُصَۣيْطِرُونَ | 52:37 | Sad (preferred) or seen | Same | Sad only |
| 8 | بِمُصَيْطِرٍ | 88:22 | Sad only | Seen also via some turuq | OK |
| 9 | عِوَجَا ۜ قَيِّمًا | 18:1–2 | Mandatory sakt in wasl (or waqf at the ayah end) | Idraj allowed via some turuq | Sakt checked; **a valid waqf is penalised (BUG)** |
| 10 | مِن مَّرْقَدِنَا ۜ هَٰذَا | 36:52 | Mandatory sakt in wasl (or waqf) | As above | As above |
| 11 | وَقِيلَ مَنْ ۜ رَاقٍ | 75:27 | Mandatory sakt (izhar of the noon) | As above | OK |
| 12 | كَلَّا بَلْ ۜ رَانَ | 83:14 | Mandatory sakt (izhar of the lam) | As above | OK |
| 13 | مَالِيَهْ ۜ هَلَكَ | 69:28–29 | Sakt or idgham | Same | OK (optional) |
| 14 | ءَآلذَّكَرَيْنِ | 6:143, 6:144 | Madd al-farq 6 (ibdal) or tas-heel | Same | 6 only |
| 15 | ءَآلْـَٰٔنَ | 10:51, 10:91 | Madd al-farq 6 or tas-heel (+ badal options on the second hamza) | Same | 6 only |
| 16 | ءَآللَّهُ | 10:59, 27:59 | Madd al-farq 6 or tas-heel | Same | 6 only |
| 17 | سَلَٰسِلَا۟ | 76:4 | Wasl: no alif. Waqf: with the alif **or** with sukun | Same | Sukun only |
| 18 | قَوَارِيرَا۠ | 76:15 | Waqf **with** the alif (rectangular zero) | Same | **Alif silent (BUG)** |
| 19 | قَوَارِيرَا۟ | 76:16 | Waqf with sukun (no alif) | Same | OK |
| 20 | ٱلظُّنُونَا۠ · ٱلرَّسُولَا۠ · ٱلسَّبِيلَا۠ | 33:10, 33:66, 33:67 | Waqf with the alif, wasl without | Same | **Alif silent (BUG)** |
| 21 | لَّٰكِنَّا۠ هُوَ | 18:38 | Alif at waqf only | Same | **Alif silent (BUG)** |
| 22 | أَنَا۠ (all occurrences) | e.g. 7:12, 12:90 | Alif at waqf only | Same | **Alif silent (BUG)** |
| 23 | ثَمُودَا۟ | 11:68, 25:38, 29:38, 53:51 | No alif, in wasl or waqf | Same | OK |
| 24 | فَمَآ ءَاتَىٰنِۦَ | 27:36 | Waqf with or without the yaa | Same | Not modelled |
| 25 | الٓمٓ ٱللَّهُ | 3:1–2 | Mim 6 or 2 when joined | Same | **6 only (BUG)** |
| 26 | كٓهيعٓصٓ · عٓسٓقٓ | 19:1 · 42:2 | ʿayn 4 or 6 | Same | OK |
| 27 | يسٓ وَٱلْقُرْءَانِ · نٓ وَٱلْقَلَمِ | 36:1–2 · 68:1 | Izhar of the noon | Idgham also | Correct text; not verified |
| 28 | يَلْهَث ذَّٰلِكَ | 7:176 | Idgham | Izhar also | OK (idgham) |
| 29 | ٱرْكَب مَّعَنَا | 11:42 | Idgham | Izhar also | OK (idgham) |
| 30 | أَلَمْ نَخْلُقكُّم | 77:20 | Complete idgham (kamil) preferred; naqis also related | Same | Kamil only |
| 31 | فِيهِۦ مُهَانًا | 25:69 | Silah (exception: normally no silah after a sakin) | Same | OK |
| 32 | يَرْضَهُ لَكُمْ | 39:7 | Damma without silah | Same | OK |
| 33 | أَرْجِهْ | 7:111, 26:36 | Sukun on the haa | Same | OK |
| 34 | فَأَلْقِهْ | 27:28 | Sukun on the haa | Same | OK |
| 35 | وَيَتَّقْهِ | 24:52 | Qaf sakin, haa with a short kasra | Same | OK |
| 36 | أَنسَىٰنِيهُ · عَلَيْهُ ٱللَّهَ | 18:63 · 48:10 | Damma on the haa (not kasra) | Same | OK |
| 37 | مَكَّنِّى | 18:95 | One noon mushaddad | Same | OK |
| 38 | بِئْسَ ٱلِٱسْمُ | 49:11 | Ibtida': ٱَلِسْمُ or لِسْمُ | Same | One way only; not judged |
| 39 | يَسْرِ · وَنُذُرِ · أَسْرِ | 89:4 · 54:16 (and 54:18, 21, 30, 37, 39) · 20:77, 26:52, 11:81, 15:65, 44:23 | Waqf: tafkheem or tarqeeq (tarqeeq preferred) | Same | **Tafkheem only (BUG)** |
| 40 | مِصْرَ · ٱلْقِطْرِ · فِرْقٍ | 10:87, 12:21, 12:99, 43:51 · 34:12 · 26:63 | Waqf both (مصر: tafkheem preferred; القطر: tarqeeq preferred); فرق both | Same | OK (jawaz) |
| 41 | كِتَٰبِيَهْ · حِسَابِيَهْ · مَا هِيَهْ · يَتَسَنَّهْ · ٱقْتَدِهْ | 69:19–20, 25–26 · 101:10 · 2:259 · 6:90 | Haa' al-sakt kept in wasl and waqf | Same | Resolved; not judged |
| 42 | مِا۟ئَةَ · أُو۟لَٰٓئِكَ | 2:259 · passim | Silent alif / waw (rounded zero) | Same | OK |
| 43 | al-Anfal → Bara'ah | 8:75 → 9:1 | Waqf, sakt or wasl, without basmalah | Same | Not modelled |

Orthography marks that carry Hafs information and are currently lost or misread by `parser._DROP` / `_make_unit`: U+06DC (small high seen: sakt **or** "read seen"), U+06E0 (rectangular zero: pronounced at waqf only), U+06E3 (small low seen: seen/sad both), U+06EA (imala), U+06EB (ishmam), U+06EC (tas-heel). The waqf signs U+06D6–U+06DB (ۖ ۗ ۘ ۙ ۚ ۛ) are also dropped.

---

## 4. What ijazah/sanad examiners actually listen for (in the order a session usually goes)

1. **Lahn jali, zero tolerance**: wrong letter (especially ض, ظ, ذ, ث, the throat letters and ق), wrong harakah, a dropped or added letter, a sukun read as a vowel or the reverse, a dropped shaddah, a madd dropped or added. → The engine now covers only the madd part (Tier 1 #1, #3, #6).
2. **Makharij and sifaat of every letter in every position**: hams of voweled ت/ك, qalqalah only on the five letters, istitaalah of ض, no rolling of the raa, clean ghunnah without nasalising the vowels next to it (tashreeb al-ghunnah). → Partial, for sakin letters only.
3. **Ahkam of noon/meem, madd and raa/lam**. → The strongest area.
4. **Tasawi and consistency**: equal munfasil, muttasil, 'arid and ghunnah across the whole reading; the chosen wujuh kept; no talfeeq between Shatibiyyah and Tayyibah. → Notes only.
5. **Hafs specifics**: the words in §3, which examiners test deliberately. → Several are bugs today.
6. **Waqf and ibtida'**: no qabih stops, resuming correctly, the manner of waqf (sukun, raum, ishmam, ibdal, haa for ta' marbutah), the waqf-only alifs. → Missing.
7. **Maratib and tanasub**: one pace per sitting, proportions preserved in hadr (a 2-count madd must not collapse). → Measured through the local harakah.

---

## Sources

- [Hidayat al-Qari, Chapter 19: what must be observed for Hafs (islamweb library)](https://www.islamweb.net/ar/library/content/231/101/)
- [Differences between the Shatibiyyah and the Tayyibah in the Hafs riwayah (islamweb fatwa 230116)](https://www.islamweb.net/ar/fatwa/230116/)
- [Seen and sad in pronunciation and in the mushaf's rasm (islamweb fatwa 60507)](https://www.islamweb.net/ar/fatwa/60507/)
- [Things to observe when reading for Hafs (Katara Prize for Quran Recitation)](https://kataraquran.com/%D8%A3%D9%85%D9%88%D8%B1-%D9%8A%D8%AC%D8%A8-%D9%85%D8%B1%D8%A7%D8%B9%D8%A7%D8%AA%D9%87%D8%A7-%D8%B9%D9%86%D8%AF-%D8%A7%D9%84%D9%82%D8%B1%D8%A7%D8%A1%D8%A9%D8%A3%D9%85%D9%88%D8%B1-%D9%8A%D8%AC%D8%A8/)
- [Quranpedia: Hafs via Rawdat al-Mu'addil (qasr routes) and via the Shatibiyyah](https://quranpedia.net/book/1655/1/21)
- [Nihayat al-Qawl al-Mufid: the chapter on lahn jali and khafi (ketabonline)](https://ketabonline.com/ar/books/55066/read?part=1&page=24&index=4089547)
- [al-Tamhid fi 'Ilm al-Tajwid: the definition of lahn (islamweb)](https://www.islamweb.net/ar/library/content/230/11/)
- [Maratib al-tafkheem (alukah)](https://www.alukah.net/sharia/0/70330/%D9%85%D8%B1%D8%A7%D8%AA%D8%A8-%D8%A7%D9%84%D8%AA%D9%81%D8%AE%D9%8A%D9%85-%D9%81%D9%8A-%D8%A7%D9%84%D8%AA%D9%84%D8%A7%D9%88%D8%A9/)
- [Discussion of the tafkheem levels (Multaqa Ahl al-Tafsir)](https://mtafsir.net/threads/%D8%A7%D9%84%D8%AA%D8%AD%D9%82%D9%8A%D9%82-%D9%81%D9%8A-%D9%85%D8%B3%D8%A3%D9%84%D8%A9-%D9%85%D8%B1%D8%A7%D8%AA%D8%A8-%D8%A7%D9%84%D8%AA%D9%81%D8%AE%D9%8A%D9%85.4999/)
- [Waqf on سلاسلا and قواريرا, and the two zero marks (Mabahith fi 'Ilm al-Qira'at, Shamela)](https://shamela.ws/book/38039/94)
- [The seven alifs in the Hafs riwayah (mawdoo3)](https://mawdoo3.com/%D8%A7%D9%84%D8%A3%D9%84%D9%81%D8%A7%D8%AA_%D8%A7%D9%84%D8%B3%D8%A8%D8%B9%D8%A9_%D9%81%D9%8A_%D8%A7%D9%84%D9%82%D8%B1%D8%A2%D9%86_%D8%A7%D9%84%D9%83%D8%B1%D9%8A%D9%85)
- Classical texts relied on for the taxonomy (standard content, not fetched): al-Muqaddimah al-Jazariyyah (makharij, sifaat, the tafkheem of alif following the letter before it, the raa, waqf); Tuhfat al-Atfal (noon/meem, madd); Hidayat al-Mustafid (madd, lam, idgham classes); Nihayat al-Qawl al-Mufid (lahn, maratib, waqf).
