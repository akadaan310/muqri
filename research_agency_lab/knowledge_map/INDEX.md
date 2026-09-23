# Knowledge map index (generated — edit fragments/, then run build_map.py)

939 nodes · 1628 edges · fragments: 00_curriculum, 01_makharij, 02_sifaat, 03_timing, 04_waqf, 05_topology, 06_information_neural, 07_dsp_julia, 08_tajweed_rules, 09_tajweed_treatises, 10_letter_reference, core

## Tajweed phenomena → models

| phenomenon | level | engine status | implemented models / code | proposed models |
|---|---|---|---|---|
| `phen:hamzat_wasl` Hamzat al-wasl (drop in wasl, pronounce on ibtida') | 101 | implemented |  |  |
| `phen:lahn` Lahn | 101 | partial |  |  |
| `phen:lahn:jali` Lahn jali | 101 | partial | `code:app/lahn/gop.py`, `algo:gop_subst_aware`, `code:research_agency_lab/substrate_library/julia/src/CtcGop.jl` |  |
| `phen:lahn:khafi` Lahn khafi | 101 | partial |  |  |
| `phen:lahn_jali_khafi` Lahn jali vs lahn khafi severity | 101 | missing |  | `math:makhraj_chain_rule_gop` |
| `phen:lam` Lam rules | 101 | partial |  |  |
| `phen:lam:allah` Lam of the Divine Name | 101 | implemented | `code:app/tajweed_rules/raa_lam_rules.py` |  |
| `phen:lam:qamariyyah` Lam qamariyyah | 101 | missing |  |  |
| `phen:lam:shamsiyyah` Lam shamsiyyah | 101 | missing | `code:app/tajweed_rules/parser.py` |  |
| `phen:letter:ayn` ع | 101 | implemented |  |  |
| `phen:letter:daad` ض | 101 | implemented |  |  |
| `phen:letter:dhaa` ظ | 101 | implemented |  |  |
| `phen:letter:ghayn` غ | 101 | implemented |  |  |
| `phen:letter:noon` ن | 101 | implemented |  |  |
| `phen:letter:saad` ص | 101 | implemented |  |  |
| `phen:letter:taa_t` ط | 101 | implemented |  |  |
| `phen:letters` Letters & harakat | 101 | partial |  |  |
| `phen:letters:haraka_length` Short vowel length (no ishba') | 101 | missing |  |  |
| `phen:letters:haraka_quality` Short vowel quality | 101 | missing | `code:app/aligner.py` |  |
| `phen:letters:identity` Letter identity (no substitution) | 101 | missing | `code:app/aligner.py`, `code:app/calibration.py` |  |
| `phen:letters:shaddah` Shaddah on all letters | 101 | partial | `code:app/tajweed_rules/idghaam_classes.py`, `code:app/tajweed_rules/noon_sakinah.py` |  |
| `phen:letters:sukoon` Sukoon (no epenthetic vowel) | 101 | partial | `code:app/sifaat/sukoon_spectrum.py`, `code:app/tajweed_rules/qalqalah_engine.py` |  |
| `phen:letters:tanween` Tanween | 101 | implemented | `code:app/tajweed_rules/noon_sakinah.py`, `code:app/tajweed_rules/parser.py` |  |
| `phen:madd` Mudood | 101 | implemented | `math:affine_duration_law`, `code:app/tajweed_rules/mudood_engine.py`, `math:bures_wasserstein`, `algo:beat_normalization`, `algo:sindy_stlsq`, `algo:vowel_core_hilbert`, `algo:anchor_tail_ruler` | `algo:collapse_gate`, `algo:hsmm` |
| `phen:madd:tabii` Madd tabii | 101 | implemented | `code:app/tajweed_rules/mudood_engine.py` |  |
| `phen:madd_length` Madd length & consistency | 101 | partial |  |  |
| `phen:makhraj:shafah_faa` Inner lower lip + upper incisors | 101 | missing |  |  |
| `phen:makhraj:shafatan_both` Both lips | 101 | missing |  |  |
| `phen:meem` Meem sakinah | 101 | partial |  |  |
| `phen:meem:idgham_shafawi` Idgham shafawi | 101 | implemented | `code:app/tajweed_rules/meem_sakinah.py` |  |
| `phen:meem:ikhfa_shafawi` Ikhfa shafawi | 101 | implemented | `code:app/tajweed_rules/meem_sakinah.py` |  |
| `phen:meem:izhar_shafawi` Izhar shafawi | 101 | implemented | `code:app/tajweed_rules/meem_sakinah.py` |  |
| `phen:noon` Noon sakinah & tanween | 101 | partial |  |  |
| `phen:noon:ghunnah_mushaddadah` Ghunnah mushaddadah | 101 | implemented | `code:app/tajweed_rules/noon_sakinah.py` |  |
| `phen:noon:idgham_ghunnah` Idgham with ghunnah | 101 | implemented | `code:app/tajweed_rules/noon_sakinah.py` |  |
| `phen:noon:idgham_no_ghunnah` Idgham without ghunnah | 101 | implemented | `code:app/tajweed_rules/noon_sakinah.py` |  |
| `phen:noon:ikhfa` Ikhfa haqiqi | 101 | implemented | `code:app/tajweed_rules/noon_sakinah.py` |  |
| `phen:noon:iqlab` Iqlab | 101 | implemented | `code:app/tajweed_rules/noon_sakinah.py` |  |
| `phen:noon:izhar_halqi` Izhar halqi | 101 | implemented | `code:app/tajweed_rules/noon_sakinah.py` |  |
| `phen:noon:izhar_mutlaq` Izhar mutlaq (dunya...) | 101 | partial | `code:app/tajweed_rules/parser.py` |  |
| `phen:noon_meem_sakinah` noon/meem sakinah & tanween rules | 101 | implemented | `code:app/tajweed_rules/noon_sakinah.py`, `code:app/tajweed_rules/meem_sakinah.py` |  |
| `phen:qalqalah` Qalqalah | 101 | implemented | `code:app/tajweed_rules/qalqalah_engine.py`, `math:bures_wasserstein`, `algo:release_burst` | `math:persistent_homology_0d`, `algo:levy_precedence`, `algo:reassignment`, `algo:glrt_burst`, `algo:spectral_moments` |
| `phen:qalqalah:levels` Qalqalah sughra/kubra/akbar | 101 | implemented | `code:app/tajweed_rules/qalqalah_engine.py` |  |
| `phen:ras_ayah` Stop at ayah head (sunnah) | 101 | partial |  |  |
| `phen:region:halq` Halq (throat) | 101 | missing |  | `algo:area_function_wakita` |
| `phen:region:jawf` Jawf (oral/pharyngeal cavity) | 101 | missing |  |  |
| `phen:region:khayshum` Khayshum (nasal cavity) | 101 | missing |  |  |
| `phen:region:lisan` Lisan (tongue) | 101 | missing |  | `algo:sparc_inversion` |
| `phen:region:shafatan` Shafatan (lips) | 101 | missing |  | `algo:sparc_inversion` |
| `phen:timing:harakah` harakah beat / tempo | 101 | implemented | `code:app/acoustic/tempo.py`, `code:app/taraweeh_adapter/pace_normalizer.py`, `math:tempo_relaxation_ode`, `algo:median_iqr_beat`, `algo:local_beat_window` | `algo:two_ruler_beat`, `math:ou_kalman_beat` |
| `phen:timing:shaddah` shaddah (geminate ~2x singleton by manner) | 101 | missing |  |  |
| `phen:timing:short_vowel` short vowel length: ishbaa' (over-lengthening) / ikhtilas | 101 | missing |  |  |
| `phen:timing:sukoon_tahreek` sukoon realised as a vowel (tahreek) | 101 | missing |  |  |
| `phen:waqf:ibdal` Ibdal (tanween fath -> alif) | 101 | implemented |  |  |
| `phen:waqf:idtirari` Waqf idtirari (breath) | 101 | partial |  |  |
| `phen:waqf:sukoon` Sukoon mahd at stop | 101 | partial |  |  |
| `phen:waqf:taa_marbuta` Taa marbuta -> haa at stop | 101 | partial |  |  |
| `phen:waqf_sign:jeem` Sign ج permitted (U+06DA, 1972x) | 101 | missing |  |  |
| `phen:waqf_sign:la` Sign لا not permitted (U+06D9, 68x) | 101 | missing |  |  |
| `phen:waqf_sign:lazim` Sign م lazim (U+06D8, 22x) | 101 | missing |  |  |
| `phen:waqf_sign:qly` Sign قلى stop preferred (U+06D7, 603x) | 101 | missing |  |  |
| `phen:waqf_sign:sly` Sign صلى continue preferred (U+06D6, 1682x) | 101 | missing |  |  |
| `phen:weight:alif_follows` Alif follows preceding letter | 101 | partial | `code:app/tajweed_rules/parser.py` |  |
| `phen:grad:qalqalah_levels` Qalqalah sughra/kubra/akbar | intermediate | partial | `code:app/tajweed_rules/qalqalah_engine.py` | `math:isotonic_ordinal` |
| `phen:ibtida` Ibtida' (restart point) | intermediate | missing |  | `math:stop_cost` |
| `phen:idgham` Idgham classes | intermediate | partial |  |  |
| `phen:idgham:mithlayn` Idgham mithlayn | intermediate | implemented | `code:app/tajweed_rules/idghaam_classes.py` |  |
| `phen:idgham:mutajanisayn` Idgham mutajanisayn (+ta naqis) | intermediate | implemented | `code:app/tajweed_rules/idghaam_classes.py` |  |
| `phen:idgham:mutaqaribayn` Idgham mutaqaribayn | intermediate | implemented | `code:app/tajweed_rules/idghaam_classes.py` |  |
| `phen:lam:fil_izhar` Izhar of verb lam / hal / bal | intermediate | missing |  |  |
| `phen:letter:alif_madd` Letter ا madd | intermediate | missing |  |  |
| `phen:letter:baa` Letter ب | intermediate | implemented |  |  |
| `phen:letter:dal` Letter د | intermediate | implemented |  |  |
| `phen:letter:dhal` Letter ذ | intermediate | implemented |  | `algo:pairwise_cue_llr` |
| `phen:letter:faa` Letter ف | intermediate | implemented |  |  |
| `phen:letter:ghain` Letter غ | intermediate | missing |  |  |
| `phen:letter:haa` Letter ه | intermediate | implemented |  |  |
| `phen:letter:haa_h` Letter ح | intermediate | implemented |  | `algo:pairwise_cue_llr` |
| `phen:letter:hamza` Letter ء | intermediate | implemented |  | `algo:pairwise_cue_llr` |
| `phen:letter:jeem` Letter ج | intermediate | implemented |  |  |
| `phen:letter:kaf` Letter ك | intermediate | implemented |  |  |
| `phen:letter:khaa` Letter خ | intermediate | implemented |  |  |
| `phen:letter:lam` Letter ل | intermediate | implemented |  |  |
| `phen:letter:meem` Letter م | intermediate | implemented |  |  |
| `phen:letter:nun` Letter ن | intermediate | missing |  |  |
| `phen:letter:qaf` Letter ق | intermediate | implemented |  | `algo:pairwise_cue_llr` |
| `phen:letter:raa` Letter ر | intermediate | implemented |  |  |
| `phen:letter:sad` Letter ص | intermediate | partial |  | `algo:pairwise_cue_llr` |
| `phen:letter:seen` Letter س | intermediate | implemented |  |  |
| `phen:letter:sheen` Letter ش | intermediate | implemented |  |  |
| `phen:letter:taa` Letter ت | intermediate | implemented |  |  |
| `phen:letter:taa_emph` Letter ط | intermediate | partial |  | `algo:pairwise_cue_llr` |
| `phen:letter:thaa` Letter ث | intermediate | implemented |  | `algo:pairwise_cue_llr` |
| `phen:letter:waw` Letter و | intermediate | implemented |  |  |
| `phen:letter:waw_madd` Letter و madd | intermediate | missing |  |  |
| `phen:letter:yaa` Letter ي | intermediate | implemented |  |  |
| `phen:letter:yaa_madd` Letter ي madd | intermediate | missing |  |  |
| `phen:letter:zay` Letter ز | intermediate | implemented |  |  |
| `phen:madd:alm_allah` Alif-Lam-Mim Allah 6 or 2 (3:1-2) | intermediate | partial | `code:app/tajweed_rules/parser.py` |  |
| `phen:madd:arid` Madd arid lis-sukun | intermediate | implemented | `code:app/tajweed_rules/mudood_engine.py` |  |
| `phen:madd:ayn` Ayn 4/6 (19:1, 42:2) | intermediate | implemented | `code:app/tajweed_rules/mudood_engine.py` |  |
| `phen:madd:badal` Madd badal | intermediate | implemented | `code:app/tajweed_rules/mudood_engine.py` |  |
| `phen:madd:farq` Madd al-farq (6 or tas-heel) | intermediate | partial | `code:app/tajweed_rules/parser.py` |  |
| `phen:madd:hierarchy` Aqwa al-sababayn | intermediate | implemented | `code:app/tajweed_rules/parser.py` |  |
| `phen:madd:iwad` Madd iwad | intermediate | implemented | `code:app/tajweed_rules/mudood_engine.py`, `code:app/tajweed_rules/parser.py` |  |
| `phen:madd:lazim` Madd lazim kalimi/harfi | intermediate | implemented | `code:app/tajweed_rules/mudood_engine.py` |  |
| `phen:madd:leen` Madd leen | intermediate | partial | `code:app/tajweed_rules/mudood_engine.py` |  |
| `phen:madd:munfasil` Madd munfasil | intermediate | implemented | `code:app/tajweed_rules/mudood_engine.py` |  |
| `phen:madd:muttasil` Madd muttasil | intermediate | implemented | `code:app/tajweed_rules/mudood_engine.py` |  |
| `phen:madd:silah` Silah sughra/kubra + exceptions | intermediate | implemented | `code:app/tajweed_rules/mudood_engine.py` |  |
| `phen:madd:tamkeen` Madd tamkeen | intermediate | partial | `code:app/tajweed_rules/mudood_engine.py` |  |
| `phen:makhraj:lisan_aqsa_kaf` Aqsa al-lisan slightly forward (kaf) | intermediate | missing |  |  |
| `phen:makhraj:lisan_aqsa_qaf` Aqsa al-lisan + soft palate (qaf) | intermediate | missing |  |  |
| `phen:makhraj:lisan_asaliyya` Tip near incisors, safir (asaliyya) | intermediate | missing |  |  |
| `phen:makhraj:lisan_hafah_lam` Front tongue edges to tip + gum (lam) | intermediate | missing |  |  |
| `phen:makhraj:lisan_lithawiyya` Tip + edges of upper incisors (lithawiyya) | intermediate | missing |  |  |
| `phen:makhraj:lisan_nitiyya` Tip + roots of upper incisors (nit'iyya) | intermediate | missing |  |  |
| `phen:makhraj:lisan_taraf_nun` Tongue tip + gum below lam (nun) | intermediate | missing |  |  |
| `phen:makhraj:lisan_taraf_raa` Tongue tip + some dorsum (raa) | intermediate | missing |  |  |
| `phen:makhraj:lisan_wasat` Wasat al-lisan + hard palate | intermediate | missing |  |  |
| `phen:qalqalah:no_bounce` No bounce on non-qalqalah letters | intermediate | partial | `code:app/sifaat/ghair_mutadhaddah.py` |  |
| `phen:raa` Raa rules | intermediate | partial |  |  |
| `phen:raa:core` Raa core rules | intermediate | implemented | `code:app/tajweed_rules/raa_lam_rules.py`, `code:app/tajweed_rules/parser.py` |  |
| `phen:raa:jawaz` Raa jawaz (firq, misr, qitr) | intermediate | implemented | `code:app/tajweed_rules/raa_lam_rules.py` |  |
| `phen:raa:waqf` Raa at waqf after sakin/yaa | intermediate | implemented | `code:app/tajweed_rules/raa_lam_rules.py` |  |
| `phen:raa:yasr_nudhur` Raa yasr/nudhur/asr at waqf | intermediate | partial | `code:app/tajweed_rules/parser.py` |  |
| `phen:sakt` Sakt of Hafs | intermediate | implemented | `phen:boundary_prosody`, `algo:breath_flatness` | `math:persistent_homology_0d` |
| `phen:sakt_hamzat_wasl` saktat + hamzat al-wasl | intermediate | implemented | `code:app/tajweed_rules/sakt_wasl.py` | `algo:sakt_in_beats` |
| `phen:sifah:tafashshi` Tafashshi | intermediate | implemented | `code:app/sifaat/ghair_mutadhaddah.py`, `algo:hf_tilt_flatness` | `algo:spectral_moments`, `algo:muaalem_teacher` |
| `phen:tafkheem_tarqeeq` tafkheem/tarqeeq (isti'la, raa, lam of Allah) | intermediate | implemented | `code:app/tajweed_rules/raa_lam_rules.py`, `code:app/sifaat/formants.py`, `math:bures_wasserstein` |  |
| `phen:tokenizer_digraph_gap` CTC vocab lacks ث ذ خ ش غ (split into 2 tokens); initial hamza untokenised | intermediate | missing | `algo:gop_segmentation_free`, `algo:second_judge_ensemble` |  |
| `phen:waqf:hamza` Stopping on hamza (clear glottal closure) | intermediate | missing |  |  |
| `phen:waqf:hasan` Waqf hasan | intermediate | missing |  | `math:dependency_cut` |
| `phen:waqf:kafi` Waqf kafi | intermediate | missing |  |  |
| `phen:waqf:madd_arid` Madd arid li-l-sukun + taswiyah | intermediate | partial |  |  |
| `phen:waqf:qabih` Waqf qabih | intermediate | missing |  |  |
| `phen:waqf:tamm` Waqf tamm | intermediate | missing |  | `math:lm_entropy` |
| `phen:waqf:waqf_only_alif` Alif pronounced only at waqf (U+06E0, 66x) | intermediate | partial |  |  |
| `phen:waqf_sign:muanaqah` Mu'anaqah ∴ (U+06DB, 6 pairs) | intermediate | missing |  |  |
| `phen:wasl` Hamzat al-wasl & saktat | intermediate | partial |  |  |
| `phen:wasl:between_surahs` Basmalah / Anfal-Tawbah | intermediate | missing |  |  |
| `phen:wasl:dropped` Hamzat al-wasl dropped in wasl | intermediate | implemented | `code:app/tajweed_rules/sakt_wasl.py` |  |
| `phen:wasl:haa_sakt` Haa al-sakt kept | intermediate | partial | `code:app/tajweed_rules/parser.py`, `code:app/sifaat/hams_jahr.py` |  |
| `phen:wasl:ibtida_vowel` Hamzat al-wasl ibtida vowel | intermediate | partial | `code:app/tajweed_rules/parser.py` |  |
| `phen:wasl:maliyah` Maliyah sakt or idgham | intermediate | implemented | `code:app/tajweed_rules/sakt_wasl.py` |  |
| `phen:wasl:sakinayn` Kasra for joining two sakins | intermediate | partial | `code:app/tajweed_rules/parser.py` |  |
| `phen:wasl:saktat` 4 saktat of Hafs | intermediate | implemented | `code:app/tajweed_rules/sakt_wasl.py`, `code:app/tajweed_rules/parser.py` |  |
| `phen:weight` Tafkheem/tarqeeq | intermediate | partial |  |  |
| `phen:weight:istifal_leak` Tarqeeq of istifal letters near heavy | intermediate | missing | `code:app/sifaat/formants.py` |  |
| `phen:weight:istila` Tafkheem isti'la letters | intermediate | implemented | `code:app/tajweed_rules/raa_lam_rules.py`, `code:app/sifaat/formants.py` |  |
| `phen:weight:maratib` 5 levels of tafkheem | intermediate | partial | `code:app/sifaat/formants.py` |  |
| `phen:boundary_prosody` Prosodic boundary cues (pause, F0 reset, final lengthening, breath) | advanced | partial |  | `math:boundary_likelihood` |
| `phen:fingerprint` reciter style / identity | advanced | implemented | `code:app/fingerprint.py`, `math:log_euclidean` | `math:path_signature` |
| `phen:grad:letter_strength` Letter strength classes (aqwa..adaaf) | advanced | proposed |  | `math:fca_lattice`, `math:makhraj_distinctive_set`, `math:strength_poset` |
| `phen:hafs_variants` Legit Hafs variants (munfasil 4/5 vs 2, arid 2/4/6) | advanced | partial |  | `model:hf:obadx/recitation-segmenter-v2` |
| `phen:idgham_adjacent_makhraj` Idgham mutajanisayn/mutaqaribayn with retained sifah (بسطت, نخلقكم) | advanced | partial |  | `algo:pairwise_cue_llr` |
| `phen:letter:ain` Letter ع | advanced | missing |  | `algo:pairwise_cue_llr` |
| `phen:letter:ghunnah` Letter ghunnah (ن/م) | advanced | partial |  |  |
| `phen:letter:zaa_emph` Letter ظ | advanced | partial |  | `algo:pairwise_cue_llr` |
| `phen:makharij` 17 Makharij | advanced | partial | `algo:gop_segmentation_free`, `math:sinkhorn_divergence` | `math:makhraj_chain_rule_gop`, `math:articulatory_ground_metric`, `math:tree_wasserstein`, `math:ot_posterior_reference`, `algo:spectral_clustering_confusion`, `algo:graph_tikhonov_profile`, `math:feature_transmission`, `algo:blahut_arimoto_fano`, `algo:ssl_prototype_verifier`, `algo:multihead_makhraj_ctc`, `algo:gop_frame_posterior`, `algo:logit_gop_entropy`, `math:makhraj_metric`, `algo:makhraj_w1_score`, `algo:karma_kalman` |
| `phen:makhraj:asaliyyah` ص س ز | advanced | partial | `code:app/sifaat/ghair_mutadhaddah.py` |  |
| `phen:makhraj:dad` Hafat al-lisan ض | advanced | partial | `code:app/sifaat/ghair_mutadhaddah.py` |  |
| `phen:makhraj:faa` ف | advanced | missing |  |  |
| `phen:makhraj:halq_adna` Adna al-halq غ خ | advanced | missing |  |  |
| `phen:makhraj:halq_aqsa` Aqsa al-halq ء ه | advanced | missing |  |  |
| `phen:makhraj:halq_wasat` Wasat al-halq ع ح | advanced | missing |  |  |
| `phen:makhraj:jawf` Jawf (madd letters) | advanced | partial | `code:app/tajweed_rules/mudood_engine.py` |  |
| `phen:makhraj:kaf` ك | advanced | missing |  |  |
| `phen:makhraj:khayshum` Khayshum (ghunnah) | advanced | implemented | `code:app/sifaat/formants.py` |  |
| `phen:makhraj:lam` ل | advanced | missing |  |  |
| `phen:makhraj:lathawiyyah` ث ذ ظ | advanced | missing |  |  |
| `phen:makhraj:lisan_hafah_dad` Side of tongue + upper molars (dad) | advanced | missing |  |  |
| `phen:makhraj:nitiyyah` ت د ط | advanced | partial | `code:app/tajweed_rules/qalqalah_engine.py`, `code:app/sifaat/itbaq.py` |  |
| `phen:makhraj:noon` ن | advanced | implemented | `code:app/sifaat/formants.py` |  |
| `phen:makhraj:qaf` Aqsa al-lisan ق | advanced | missing | `code:app/tajweed_rules/qalqalah_engine.py` |  |
| `phen:makhraj:raa` ر | advanced | partial | `code:app/sifaat/ghair_mutadhaddah.py` |  |
| `phen:makhraj:shafatan` ب م و | advanced | partial | `code:app/tajweed_rules/noon_sakinah.py` |  |
| `phen:makhraj:wasat_lisan` ج ش ي | advanced | partial | `code:app/sifaat/ghair_mutadhaddah.py` |  |
| `phen:sifaat` 17 Sifaat | advanced | partial | `code:app/sifaat/hams_jahr.py`, `code:app/sifaat/sukoon_spectrum.py`, `code:app/sifaat/ghair_mutadhaddah.py`, `code:app/sifaat/itbaq.py`, `math:bures_wasserstein` |  |
| `phen:sifah:ghunnah` Ghunnah | advanced | implemented | `code:app/tajweed_rules/noon_sakinah.py`, `code:app/sifaat/formants.py`, `algo:nasal_a1p0_ner` | `algo:ssl_probe`, `algo:muaalem_teacher` |
| `phen:sifah:hams` Hams | advanced | partial | `code:app/sifaat/hams_jahr.py`, `algo:hnr_voicing` | `math:poe_joint_profile`, `algo:vot_aspiration`, `algo:cpp_aperiodicity`, `algo:muaalem_teacher` |
| `phen:sifah:idhlaq` Idhlaq | advanced | missing |  |  |
| `phen:sifah:infitah` Infitah | advanced | implemented | `code:app/sifaat/itbaq.py` |  |
| `phen:sifah:inhiraf` Inhiraf | advanced | missing |  | `algo:lateral_antiformant` |
| `phen:sifah:ismat` Ismat | advanced | missing |  |  |
| `phen:sifah:istifal` Istifal | advanced | implemented | `code:app/sifaat/formants.py`, `algo:heaviness_index` |  |
| `phen:sifah:istila` Isti'la | advanced | implemented | `code:app/sifaat/formants.py`, `algo:heaviness_index` | `math:poe_joint_profile`, `math:lobanov_bark_vtln`, `algo:ssl_probe`, `algo:muaalem_teacher` |
| `phen:sifah:istitaalah` Istitaalah | advanced | partial | `code:app/sifaat/ghair_mutadhaddah.py` |  |
| `phen:sifah:istitalah` Istitalah | advanced | partial | `code:app/sifaat/ghair_mutadhaddah.py` | `algo:f2_transition`, `algo:muaalem_teacher` |
| `phen:sifah:itbaq` Itbaq | advanced | implemented | `code:app/sifaat/itbaq.py`, `algo:heaviness_index` | `math:poe_joint_profile`, `math:lobanov_bark_vtln`, `algo:muaalem_teacher` |
| `phen:sifah:jahr` Jahr | advanced | partial | `code:app/sifaat/hams_jahr.py`, `algo:hnr_voicing` | `algo:vot_aspiration`, `algo:cpp_aperiodicity` |
| `phen:sifah:khafa` Khafa (haa clarity) | advanced | missing |  | `algo:cpp_aperiodicity` |
| `phen:sifah:leen` Leen | advanced | partial | `code:app/tajweed_rules/mudood_engine.py` | `math:lobanov_bark_vtln`, `algo:diphthong_trajectory` |
| `phen:sifah:qalqalah` Qalqalah | advanced | implemented | `code:app/tajweed_rules/qalqalah_engine.py` | `algo:landmark_closure_burst`, `algo:ssl_probe`, `algo:muaalem_teacher` |
| `phen:sifah:quwwah` Strong/weak letters | advanced | not_observable |  |  |
| `phen:sifah:rakhawah` Rakhawah | advanced | partial | `code:app/sifaat/sukoon_spectrum.py` | `algo:landmark_closure_burst` |
| `phen:sifah:safir` Safir | advanced | implemented | `code:app/sifaat/ghair_mutadhaddah.py`, `algo:hf_tilt_flatness` | `algo:spectral_moments`, `algo:muaalem_teacher` |
| `phen:sifah:shiddah` Shiddah | advanced | partial | `code:app/sifaat/sukoon_spectrum.py` | `math:poe_joint_profile`, `algo:landmark_closure_burst`, `algo:muaalem_teacher` |
| `phen:sifah:tafashhi` Tafashhi | advanced | implemented | `code:app/sifaat/ghair_mutadhaddah.py` |  |
| `phen:sifah:takreer` Takreer | advanced | implemented | `code:app/sifaat/ghair_mutadhaddah.py`, `algo:tap_count` | `algo:am_spectrum_trill`, `algo:muaalem_teacher`, `math:persistent_homology_0d` |
| `phen:sifah:tawassut` Tawassut | advanced | partial | `code:app/sifaat/sukoon_spectrum.py` | `math:isotonic_ordinal`, `algo:landmark_closure_burst` |
| `phen:timing:madd_hierarchy` madd strength order (lazim >= muttasil >= munfasil >= arid >= tabii) | advanced | missing |  | `math:order_constraint_test` |
| `phen:waqf` Waqf & ibtida' | advanced | partial | `phen:boundary_prosody` | `math:stop_cost` |
| `phen:waqf:atani` Atani 27:36 yaa | advanced | missing |  |  |
| `phen:waqf:ibtida` Ibtida quality/restart | advanced | missing | `code:app/aligner.py` |  |
| `phen:waqf:ishmam` Ishmam at waqf | advanced | missing |  | `math:neyman_pearson` |
| `phen:waqf:muanaqah` Mu'anaqah | advanced | missing |  |  |
| `phen:waqf:raum` Raum | advanced | missing |  | `algo:raum_detector` |
| `phen:waqf:salasila` Salasila two ways | advanced | partial | `code:app/tajweed_rules/parser.py` |  |
| `phen:waqf:signs` Waqf signs | advanced | missing | `code:app/tajweed_rules/parser.py` |  |
| `phen:waqf:sukun_mahd` Waqf with sukun mahd | advanced | missing |  |  |
| `phen:waqf:ta_marbutah` Ta marbutah -> haa | advanced | missing | `code:app/tajweed_rules/parser.py` |  |
| `phen:waqf:types` Waqf kinds (tamm/kafi/hasan/qabih) | advanced | missing | `code:app/taraweeh_adapter/fatigue_detector.py` |  |
| `phen:waqf:waqf_alifs` Alifs pronounced only at waqf (U+06E0) | advanced | missing | `code:app/tajweed_rules/parser.py` |  |
| `phen:grad:ghunnah_maratib` Maratib al-ghunnah (5 levels) | ijazah | missing |  | `math:isotonic_ordinal` |
| `phen:grad:tafkheem_maratib` Maratib al-tafkheem (5 levels) | ijazah | partial | `code:app/sifaat/formants.py` | `math:isotonic_ordinal` |
| `phen:hafs` Hafs special cases | ijazah | partial |  |  |
| `phen:hafs:aajami` A'jamiyy tas-heel 41:44 | ijazah | missing | `code:app/tajweed_rules/parser.py` |  |
| `phen:hafs:bimusaytir` Bimusaytir sad 88:22 | ijazah | implemented | `code:app/tajweed_rules/parser.py` |  |
| `phen:hafs:dad_du'f` Du'f fath/damm 30:54 | ijazah | partial | `code:app/tajweed_rules/parser.py` |  |
| `phen:hafs:irkab` Irkab ma'ana idgham 11:42 | ijazah | implemented | `code:app/tajweed_rules/parser.py` |  |
| `phen:hafs:majraha` Majraha imala 11:41 | ijazah | missing | `code:app/tajweed_rules/parser.py` |  |
| `phen:hafs:musaytirun` Musaytirun sad/seen 52:37 | ijazah | missing | `code:app/tajweed_rules/parser.py` |  |
| `phen:hafs:nakhluqkum` Nakhluqkum kamil 77:20 | ijazah | implemented | `code:app/tajweed_rules/parser.py` |  |
| `phen:hafs:tamanna` Ta'manna ishmam/raum 12:11 | ijazah | missing | `code:app/tajweed_rules/parser.py` |  |
| `phen:hafs:yabsut` Yabsut/bastah seen 2:245 7:69 | ijazah | missing | `code:app/tajweed_rules/parser.py` |  |
| `phen:hafs:yalhath` Yalhath dhalik idgham 7:176 | ijazah | implemented | `code:app/tajweed_rules/parser.py` |  |
| `phen:hafs:yasin_nun` Yasin/Nun izhar | ijazah | partial | `code:app/tajweed_rules/parser.py` |  |
| `phen:ijazah_grade` Ordinal grade 101→ijazah | ijazah | missing |  |  |
| `phen:letter:dad` Letter ض | ijazah | partial |  | `algo:pairwise_cue_llr` |
| `phen:madd:relations` Muttasil>=munfasil, leen<=arid | ijazah | missing |  |  |
| `phen:noon:ghunnah_maratib` Ghunnah maratib ordering | ijazah | partial | `code:app/tajweed_rules/noon_sakinah.py` |  |
| `phen:raa:imala` Imala majraha 11:41 | ijazah | missing | `code:app/tajweed_rules/parser.py` |  |
| `phen:timing` Maratib, tanasub, tasawi | ijazah | partial |  |  |
| `phen:timing:arid_choice` consistent 2/4/6 choice for madd arid | ijazah | missing |  | `math:latent_class_mixture` |
| `phen:timing:maratib` Tahqeeq/tadweer/hadr | ijazah | implemented | `code:app/taraweeh_adapter/pace_normalizer.py` | `algo:pelt_bocpd`, `math:sticky_hmm_maratib` |
| `phen:timing:pitch_rhythm` Pitch/rhythm stability | ijazah | partial | `code:app/pipeline.py` |  |
| `phen:timing:rhythm_balance` balance of timing across letters/phrases | ijazah | missing |  | `math:npvi_residual` |
| `phen:timing:tanasub` Proportionality to local tempo | ijazah | implemented | `code:app/taraweeh_adapter/pace_normalizer.py`, `code:app/tajweed_rules/base.py` |  |
| `phen:timing:tasawi` Tasawi al-mudood | ijazah | partial | `code:app/scoring.py`, `math:airm` | `math:variance_components_tasawi` |
| `phen:timing:wujuh_consistency` Consistency of chosen wujuh / no talfeeq | ijazah | missing |  |  |
| `phen:waqf:ikhtibari` Waqf ikhtibari (rasm test) | ijazah | missing |  |  |
| `phen:waqf:intizari` Waqf intizari | ijazah | missing |  |  |
| `phen:wasl:ism` al-ismu 49:11 | ijazah | partial | `code:app/tajweed_rules/parser.py` |  |
| `concept:aqwa_al_mudud` Strength order Lāzim > Muttaṣil > 'Āriḍ > Munfaṣil > Badal |  | covered | `mech:-` |  |
| `concept:compound_collisions` Adjacent-letter collisions and assimilation |  | partial | `mech:segmental` |  |
| `concept:endurance` Muscular endurance and self-correction |  | uncovered | `mech:consistency` |  |
| `concept:ghunnah_four_levels` Marātib al-Ghunnah: Akmal > Kāmilah > Nāqiṣah > Anqaṣ |  | covered | `mech:durational` |  |
| `concept:ghunnah_vs_madd` 2U ghunnah ≠ 2U madd; no nasal bleed into the madd |  | partial | `mech:attribute` |  |
| `concept:hadr_integrity` Under Ḥadr, tawassuṭ/rakhāwah must not collapse into shiddah |  | covered | `mech:tempo` |  |
| `concept:hams_jahr` Hams vs Jahr — breath vs vocal-fold vibration |  | covered | `mech:attribute` |  |
| `concept:harakah_anatomy` Fatḥah / Kasrah / Ḍammah articulatory posture |  | partial | `mech:spectral` |  |
| `concept:harakah_isochrony` Zamān(Fatḥah) = Zamān(Kasrah) = Zamān(Ḍammah) |  | blocked | `mech:consistency` |  |
| `concept:harakah_weight_independence` Vowel length independent of the consonant's weight |  | blocked | `mech:consistency` |  |
| `concept:idhlaq_ismat` Idhlāq vs Iṣmāt — ease of production |  | out_of_scope | `mech:-` |  |
| `concept:ikhtilas` Ikhtilās — vowel truncated below 1U (≈⅔U) |  | covered | `mech:durational` |  |
| `concept:ishba` Ishbā' — vowel stretched beyond 1U into a madd |  | covered | `mech:durational` |  |
| `concept:istila_istifal` Isti'lā' vs Istifāl — tongue-root elevation |  | partial | `mech:attribute` |  |
| `concept:itbaq_infitah` Iṭbāq vs Infitāḥ — trapping sound against the palate |  | covered | `mech:attribute` |  |
| `concept:itmam_universal` Universal law of vowel perfection |  | uncovered | `mech:consistency` |  |
| `concept:letter_completeness` Every letter's full 5–7 classical sifāt, applied and not applied |  | covered | `mech:attribute` |  |
| `concept:letter_strength` Composite letter strength (quwwa) from its sifāt |  | covered | `mech:attribute` |  |
| `concept:madd_4_5_6` The 4, 5 and 6-count scales |  | covered | `mech:durational` |  |
| `concept:madd_arid` Madd 'Āriḍ li-s-Sukoon at 2/4/6 with terminal decay |  | partial | `mech:durational` |  |
| `concept:madd_drift` Drift error — a category shrinking over a passage through fatigue |  | covered | `mech:consistency` |  |
| `concept:madd_tabii_2u` Madd Ṭabī'ī locked at 2 counts |  | covered | `mech:durational` |  |
| `concept:makharij_clinicals` Per-letter articulation clinics (ء ه ع ح غ خ ق ك ج ش ي ض ل ر ط د ت ص ز س ظ ذ ث ف ب م و) |  | covered | `mech:segmental` |  |
| `concept:metric_consonantal_envelope` Consonantal Envelope — complete closure, clean release |  | partial | `mech:attribute` |  |
| `concept:metric_formant_integrity` Formant Track Integrity — pure /a/ /i/ /u/, zero nasalisation on vowels |  | partial | `mech:spectral` |  |
| `concept:metric_nasal_isolation` Nasal Channel Isolation — clear ghunnah, instant cut-off |  | partial | `mech:attribute` |  |
| `concept:metric_temporal_pulse` Temporal Pulse Isochrony — 1U/2U/4U/5U/6U ratios hold across tempo |  | covered | `mech:consistency` |  |
| `concept:neutral_sukoon_zero` The neutral sukoon state as the absolute zero point |  | partial | `mech:tempo` |  |
| `concept:physiology` Respiratory / laryngeal / supraglottal mechanics |  | out_of_scope | `mech:-` |  |
| `concept:raa_lam_weight` Conditional tafkhīm/tarqīq of ر and of the lām of ٱللَّه |  | covered | `mech:attribute` |  |
| `concept:shidda_rakhawa_attr` Shiddah / Tawassuṭ / Rakhāwah as a letter quality |  | covered | `mech:attribute` |  |
| `concept:sukoon_rakhawah` Rakhāwah — sustained flow, longest sākin duration |  | covered | `mech:tempo` |  |
| `concept:sukoon_shiddah` Shiddah — complete stop, shortest sākin duration |  | covered | `mech:tempo` |  |
| `concept:sukoon_tawassut` Tawassuṭ / bayniyyah (ل ن ع م ر) — medium duration |  | covered | `mech:tempo` |  |
| `concept:tafkhim_five_levels` Five graded levels of tafkhīm by vowel context |  | partial | `mech:attribute` |  |
| `concept:tafkhim_nisbi` Relative heaviness: isti'lā' without iṭbāq + kasrah |  | partial | `mech:attribute` |  |
| `concept:taswiyat_al_mudud` Taswiyah — every instance of a madd category held identically |  | covered | `mech:consistency` |  |
| `concept:ten_readers` The ten readers' structural systematics |  | out_of_scope | `mech:-` |  |
| `concept:three_tempos` Taḥqīq / Tadwīr / Ḥadr baseline (≈350/250/150 ms per harakah) |  | covered | `mech:tempo` |  |
| `concept:vowel_sequences` Consecutive ḍammah, alternating vowels, ḍammah→sukoon |  | uncovered | `mech:consistency` |  |
| `concept:zaman_proportionality` Zamān/Harakah = K constant across tempo |  | covered | `mech:tempo` |  |
| `fam:ghunnah` Ghunnah |  | partial |  |  |
| `fam:idgham` Idgham |  | partial |  |  |
| `fam:lahn` Lahn |  | partial |  |  |
| `fam:meem_sakinah` Meem Sakinah |  | partial |  |  |
| `fam:mudood` Mudood |  | partial |  |  |
| `fam:noon_sakinah` Noon Sakinah |  | partial |  |  |
| `fam:qalqalah` Qalqalah |  | partial |  |  |
| `fam:raa_lam` Raa Lam |  | partial |  |  |
| `fam:sifaat` Sifaat |  | partial |  |  |
| `fam:wasl_waqf` Wasl Waqf |  | partial |  |  |
| `letter:ء` ء |  | implemented |  |  |
| `letter:ا` ا |  | implemented |  |  |
| `letter:ب` ب |  | implemented |  |  |
| `letter:ت` ت |  | implemented |  |  |
| `letter:ث` ث |  | implemented |  |  |
| `letter:ج` ج |  | implemented |  |  |
| `letter:ح` ح |  | implemented |  |  |
| `letter:خ` خ |  | implemented |  |  |
| `letter:د` د |  | implemented |  |  |
| `letter:ذ` ذ |  | implemented |  |  |
| `letter:ر` ر |  | implemented |  |  |
| `letter:ز` ز |  | implemented |  |  |
| `letter:س` س |  | implemented |  |  |
| `letter:ش` ش |  | implemented |  |  |
| `letter:ص` ص |  | implemented |  |  |
| `letter:ض` ض |  | implemented |  |  |
| `letter:ط` ط |  | implemented |  |  |
| `letter:ظ` ظ |  | implemented |  |  |
| `letter:ع` ع |  | implemented |  |  |
| `letter:غ` غ |  | implemented |  |  |
| `letter:ف` ف |  | implemented |  |  |
| `letter:ق` ق |  | implemented |  |  |
| `letter:ك` ك |  | implemented |  |  |
| `letter:ل` ل |  | implemented |  |  |
| `letter:م` م |  | implemented |  |  |
| `letter:ن` ن |  | implemented |  |  |
| `letter:ه` ه |  | implemented |  |  |
| `letter:و` و |  | implemented |  |  |
| `letter:ي` ي |  | implemented |  |  |
| `makhraj:halq` Makhraj zone: halq |  | implemented |  |  |
| `makhraj:jawf` Makhraj zone: jawf |  | implemented |  |  |
| `makhraj:lisan` Makhraj zone: lisan |  | implemented |  |  |
| `makhraj:shafatan` Makhraj zone: shafatan |  | implemented |  |  |
| `phen:alignment_uncertainty` Alignment uncertainty | phenomenon | partial |  | `math:multiparameter_persistence`, `algo:ffbs_alignment_uncertainty` |
| `phen:anchor_gate` Gate philosophy: near-gold anchor (slow studio muallim/mujawwad) ≈97, few % flagged; other masters make real mistakes and are graded, not forced to pass |  | implemented |  |  |
| `phen:fatigue` Taraweeh fatigue | phenomenon | partial | `code:app/taraweeh_adapter/fatigue_detector.py` | `algo:zigzag_fatigue`, `algo:sigkernel_mmd`, `algo:cpp` |
| `phen:ghunnah` Ghunnah nasal antiresonance | phenomenon | implemented | `code:app/sifaat/formants.py`, `code:app/tajweed_rules/noon_sakinah.py`, `math:bures_wasserstein`, `algo:ner_nasal_contrast` | `algo:topo_formant_tracking`, `math:a1_p0` |
| `phen:harakah` Harakah / tempo | phenomenon | partial |  |  |
| `phen:idgham:classes` Mithlayn/mutajanisayn/mutaqaribayn | phenomenon | implemented | `code:app/tajweed_rules/idghaam_classes.py`, `algo:release_burst` | `math:makhraj_metric` |
| `phen:idgham:kamil` Idgham kamil single release | phenomenon | partial |  | `math:persistent_homology_0d` |
| `phen:idgham:naqis` Idgham naqis itbaq retention | phenomenon | partial |  | `algo:levy_precedence` |
| `phen:ikhfa` Ikhfa anticipation | phenomenon | partial |  | `algo:levy_precedence`, `math:a1_p0` |
| `phen:iqlab` Iqlab lip closure vs nasal onset | phenomenon | partial |  | `algo:levy_precedence`, `algo:karma_kalman` |
| `phen:jahr` Jahr / phonation | phenomenon | partial |  | `algo:multitaper`, `algo:iaif_qcp` |
| `phen:madd:counts` Madd/ghunnah counts | phenomenon | partial |  | `algo:haraka_field` |
| `phen:madd:tasawi` Hafs madd consistency (tasawi) | phenomenon | partial |  | `algo:haraka_field`, `algo:replicate_icc`, `algo:fr_choice_consistency` |
| `phen:makhraj` Makharij (letter articulation) | 101-ijazah | missing |  |  |
| `phen:maqam` Maqam (melodic mode) | phenomenon | partial | `algo:pyin` | `algo:maqam_scale_degrees`, `math:sliding_window_torus`, `math:hodge_flow_melody`, `algo:crepe_rmvpe`, `algo:pitch_histogram`, `algo:synchrosqueezing` |
| `phen:reciter_style` Reciter style / melody law | phenomenon | partial |  | `algo:hodgerank_reciters`, `math:grassmannian`, `algo:sigkernel_mmd` |
| `phen:reverb` Taraweeh reverberation | phenomenon | implemented | `algo:wpe`, `code:app/taraweeh_adapter/dereverb.py`, `algo:rt60_blind` | `algo:modulation_spectrum` |
| `phen:safir_tafashhi` Safir / tafashhi / hams | phenomenon | partial |  | `algo:multitaper`, `algo:spectral_moments` |
| `phen:tafkheem` Tafkheem F2 collapse | phenomenon | partial |  | `algo:topo_formant_tracking`, `algo:karma_kalman` |
| `phen:takreer` Takreer (raa trill) | phenomenon | partial |  | `algo:scattering`, `algo:modulation_spectrum` |
| `phen:tareeq:talfiq` Mixing turuq (talfiq) | phenomenon | missing |  | `algo:tahrirat_csp` |
| `phen:text_hierarchy` Letter-word-ayah-surah hierarchy | phenomenon | partial |  | `math:tree_gmrf` |
| `phen:waqf_ibtida` Waqf and ibtida quality | phenomenon | partial | `code:app/tajweed_rules/parser.py`, `code:app/taraweeh_adapter/fatigue_detector.py` | `algo:waqf_cut_cost` |
| `phen:waqf_sakt` Waqf / sakt / breath | phenomenon | partial |  | `algo:breath_detector` |
| `sifah:idhlaq` idhlaq |  | implemented |  |  |
| `sifah:inhiraf` inhiraf |  | implemented |  |  |
| `sifah:ismat` ismat |  | implemented |  |  |
| `sifah:isti'la` isti'la |  | implemented |  |  |
| `sifah:istifal` istifal |  | implemented |  |  |
| `sifah:lin` lin |  | implemented |  |  |

## Mathematical constructs

| id | status | prio | label | where / refs |
|---|---|---|---|---|
| `math:airm` | implemented | 1 | AIRM SPD | research_agency_lab/experiments/deep_research/05_topology_geometry_graphs.md |
| `math:bures_wasserstein` | implemented | 1 | Bures-Wasserstein | research_agency_lab/experiments/deep_research/05_topology_geometry_graphs.md |
| `math:conformal_loo` | implemented | 1 | Conformal peer-LOO thresholds | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `math:tempo_relaxation_ode` | implemented | 1 | dT/dtau = kappa(Tinf-T)+phi | research_agency_lab/experiments/deep_research/07_dsp_julia_discovery.md |
| `math:karcher_mean` | implemented | 2 | Karcher mean | research_agency_lab/experiments/deep_research/05_topology_geometry_graphs.md |
| `math:log_euclidean` | implemented | 2 | Log-Euclidean SPD | research_agency_lab/experiments/deep_research/05_topology_geometry_graphs.md |
| `math:sinkhorn_divergence` | implemented | 2 | Sinkhorn divergence | research_agency_lab/experiments/deep_research/05_topology_geometry_graphs.md |
| `math:ogk` | implemented |  | OGK robust scatter (Maronna-Zamar 2002) + chi2 reweight | Frontier.jl ogk / fr_ogk.m |
| `math:robust_median_mad` | implemented |  | median/MAD hull + robust z (|z|<=2 PASS, <=3 WARN) | app/calibration.py |
| `math:ctc_forward_backward` | partial | 1 | Log-space CTC forward/backward α/β | research_agency_lab/experiments/deep_research/01_makharij.md |
| `math:affine_duration_law` | partial |  | D = a + b T + c n T (sub-proportional, Klatt 1976); counts by inversion | Discovery.jl duration_law |
| `math:a1_p0` | proposed | 1 | A1-P0 nasality index | research_agency_lab/experiments/deep_research/07_dsp_julia_discovery.md |
| `math:articulatory_ground_metric` | proposed | 1 | Articulatory ground metric (feature/commute-time/tree) | research_agency_lab/experiments/deep_research/01_makharij.md |
| `math:bayes_decision` | proposed | 1 | Bayes decision w/ jali/khafi loss matrix + REVIEW | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `math:bottleneck_stability` | proposed | 1 | Bottleneck stability theorem | research_agency_lab/experiments/deep_research/05_topology_geometry_graphs.md |
| `math:conformal_risk_control` | proposed | 1 | Conformal risk control (expert false-FAIL) | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `math:ctc_trellis_ffbs` | proposed | 1 | Random walk on CTC trellis (FFBS) | research_agency_lab/experiments/deep_research/05_topology_geometry_graphs.md |
| `math:graph_components` | proposed | 1 | Connected components (failure incidents) | research_agency_lab/experiments/deep_research/05_topology_geometry_graphs.md |
| `math:graph_trend_filtering` | proposed | 1 | Graph trend filtering / Laplacian regularisation | research_agency_lab/experiments/deep_research/05_topology_geometry_graphs.md |
| `math:harakah_power_law` | proposed | 1 | log D = a log n + b log T + reciter + final (test a=b=1) | research_agency_lab/experiments/deep_research/07_dsp_julia_discovery.md |
| `math:hierarchical_bayes` | proposed | 1 | Hierarchical Student-t model with measurement error | research_agency_lab/experiments/deep_research/07_dsp_julia_discovery.md |
| `math:hierarchical_conformal` | proposed | 1 | Two-layer hierarchical conformal | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `math:irt` | proposed | 1 | Item response theory (Rasch→LLTM→GRM→MIRT) | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `math:isotonic_ordinal` | proposed | 1 | Isotonic mixed model + proportional odds / CORAL | src:arXiv:1901.07884 |
| `math:levy_area_precedence` | proposed | 1 | Levy-area precedence index | research_agency_lab/experiments/deep_research/05_topology_geometry_graphs.md |
| `math:makhraj_chain_rule_gop` | proposed | 1 | Hierarchical GOP: region + makhraj|region + sifah|makhraj | research_agency_lab/experiments/deep_research/01_makharij.md |
| `math:makhraj_distinctive_set` | proposed | 1 | Makhraj-conditioned distinctive sifaat D(l) |  |
| `math:mdp_advantage` | proposed | 1 | MDP advantage/regret of each observed stop |  |
| `math:mutual_information` | proposed | 1 | KSG MI ruler relevance / verdict entropy health | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `math:persistent_homology_0d` | proposed | 1 | 0-dim sublevel persistence (prominence) | research_agency_lab/experiments/deep_research/05_topology_geometry_graphs.md |
| `math:personalized_pagerank` | proposed | 1 | Personalised PageRank root cause | research_agency_lab/experiments/deep_research/05_topology_geometry_graphs.md |
| `math:poe_joint_profile` | proposed | 1 | Product-of-experts joint letter profile | src:arXiv:1706.04599, src:doi:10.1162/089976602760128018 |
| `math:stop_cost` | proposed | 1 | Stop/continue/ibtida'/breath cost functional |  |
| `math:surprisal` | proposed | 1 | Excess surprisal under reference predictive (nats) | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `math:variance_components_tasawi` | proposed | 1 | variance-component tasawi model sigma_{r,k} + TOST/ROPE equivalence |  |
| `math:bayes_hierarchical` | proposed | 2 | Bayesian hierarchical reference (roadmap A5) | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `math:boundary_likelihood` | proposed | 2 | Class-conditional boundary likelihood with sign prior |  |
| `math:cauchy_combination` | proposed | 2 | Cauchy p-value combination (replace max z) | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `math:cellular_sheaf` | proposed | 2 | Cellular sheaf / sheaf Laplacian | research_agency_lab/experiments/deep_research/05_topology_geometry_graphs.md |
| `math:circle_persistence` | proposed | 2 | Persistence on S1 pitch-class density | research_agency_lab/experiments/deep_research/05_topology_geometry_graphs.md |
| `math:circular_ot` | proposed | 2 | Tonic-invariant circular W1 | research_agency_lab/experiments/deep_research/05_topology_geometry_graphs.md |
| `math:conformal_letter_sets` | proposed | 2 | Conformal letter prediction sets | src:arXiv:2107.07511 |
| `math:elastic_fda_srvf` | proposed | 2 | B1 full: SRVF phase/amplitude registration |  |
| `math:fca_lattice` | proposed | 2 | Formal concept lattice of sifaat (33 concepts) | src:doi:10.1007/978-3-642-59830-2 |
| `math:feature_transmission` | proposed | 2 | Miller-Nicely feature transmission, MI, KL/JS | research_agency_lab/experiments/deep_research/01_makharij.md |
| `math:hsmm` | proposed | 2 | Hidden semi-Markov model with jump arcs | src:yu-2010-hsmm |
| `math:latent_class_mixture` | proposed | 2 | latent-class mixture for arid 2/4/6 choice |  |
| `math:letter_channel` | proposed | 2 | Letter channel confusion P(ŷ|x), ΔKL jali index | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `math:lobanov_bark_vtln` | proposed | 2 | Robust Lobanov / Bark-difference / VTLN normalisation | src:doi:10.1109/89.650310, src:doi:10.1121/1.1912396, src:doi:10.1121/1.393381 |
| `math:locus_equation` | proposed | 2 | Locus equations F2onset = k F2vowel + c | research_agency_lab/experiments/deep_research/07_dsp_julia_discovery.md |
| `math:loo_pit` | proposed | 2 | LOO-PIT calibration check | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `math:makhraj_metric` | proposed | 2 | Makhraj tree x sifaat cube metric | research_agency_lab/experiments/deep_research/05_topology_geometry_graphs.md |
| `math:mdl` | proposed | 2 | MDL / PSIS-LOO / WBIC rule-model selection | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `math:mirt` | proposed | 2 | Multidimensional IRT θ_jali/timing/nasal/sifaat | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `math:mondrian_conformal` | proposed | 2 | Mondrian conformal per rule key | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `math:order_constraint_test` | proposed | 2 | isotonic / order-constrained test of madd hierarchy |  |
| `math:ordinal_regression` | proposed | 2 | Cumulative/sequential ordinal grade | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `math:ot_posterior_reference` | proposed | 2 | OT between produced and reference posteriors | research_agency_lab/experiments/deep_research/01_makharij.md |
| `math:path_signature` | proposed | 2 | C1 log-signature features (tempo-warp invariant) |  |
| `math:poisson_jali_rate` | proposed | 2 | Poisson jali rate, rule-of-three exam length | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `math:proper_scoring` | proposed | 2 | Proper scoring (log, Brier, CRPS, RPS) | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `math:ritardando_curve` | proposed | 2 | Friberg-Sundberg final ritardando v(x) | research_agency_lab/experiments/deep_research/07_dsp_julia_discovery.md |
| `math:temperature_scaling` | proposed | 2 | Temperature/vector scaling of CTC posteriors | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `math:testlet` | proposed | 2 | Testlet (ayah) random effects | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `math:tree_wasserstein` | proposed | 2 | Tree-Wasserstein makhraj distance (closed form) | research_agency_lab/experiments/deep_research/01_makharij.md |
| `math:cat` | proposed | 3 | Computerized adaptive testing | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `math:consistency_radius` | proposed | 3 | Consistency radius | research_agency_lab/experiments/deep_research/05_topology_geometry_graphs.md |
| `math:dependency_cut` | proposed | 3 | Dependency-graph cut-set weight κ_i (lafzi link) | research_agency_lab/experiments/deep_research/05_topology_geometry_graphs.md |
| `math:evt_gpd` | proposed | 3 | A6 POT / Generalized Pareto tail thresholds (SPOT) |  |
| `math:fisher_rao_normal` | proposed | 3 | Fisher-Rao univariate normal (hyperbolic) | research_agency_lab/experiments/deep_research/05_topology_geometry_graphs.md |
| `math:fisher_rao_simplex` | proposed | 3 | Fisher-Rao on simplex | research_agency_lab/experiments/deep_research/05_topology_geometry_graphs.md |
| `math:gsp_repetition` | proposed | 3 | GSP on text graph with repetition edges | research_agency_lab/experiments/deep_research/05_topology_geometry_graphs.md |
| `math:hodgerank` | proposed | 3 | HodgeRank / combinatorial Hodge | research_agency_lab/experiments/deep_research/05_topology_geometry_graphs.md |
| `math:js_divergence` | proposed | 3 | Jensen–Shannon reciter×rule verdict | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `math:koopman_spectrum` | proposed | 3 | Koopman / Hankel-DMD spectrum | research_agency_lab/experiments/deep_research/07_dsp_julia_discovery.md |
| `math:learn_then_test` | proposed | 3 | Learn-then-Test multi-risk | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `math:many_facet_rasch` | proposed | 3 | Many-facet Rasch (recording condition) | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `math:neyman_pearson` | proposed | 3 | Neyman-Pearson LRT / GLRT, ROC P_D=Φ(√n d'−Φ⁻¹(1−P_FA)) | src:kay-1998 |
| `math:npvi_residual` | proposed | 3 | nPVI rhythm on duration-law residuals |  |
| `math:ou_kalman_beat` | proposed | 3 | Ornstein-Uhlenbeck Kalman/RTS smoother on log T (local beat + uncertainty) |  |
| `math:persistence_image` | proposed | 3 | Persistence images | research_agency_lab/experiments/deep_research/05_topology_geometry_graphs.md |
| `math:persistence_landscape` | proposed | 3 | Persistence landscapes | research_agency_lab/experiments/deep_research/05_topology_geometry_graphs.md |
| `math:signature_kernel` | proposed | 3 | Signature kernel (Goursat PDE) MMD | research_agency_lab/experiments/deep_research/05_topology_geometry_graphs.md |
| `math:spectral_clustering` | proposed | 3 | Spectral clustering of CTC confusion graph | research_agency_lab/experiments/deep_research/05_topology_geometry_graphs.md |
| `math:sticky_hmm_maratib` | proposed | 3 | sticky 3-state HMM for tahqeeq/tadweer/hadr |  |
| `math:strength_poset` | proposed | 3 | Strength poset / S-W linear extension | src:book:hidayat_al_qari |
| `math:task_dynamics_2nd_order` | proposed | 3 | Critically damped target approach (Saltzman-Munhall) | research_agency_lab/experiments/deep_research/07_dsp_julia_discovery.md |
| `math:tree_gmrf` | proposed | 3 | Tree GMRF (hierarchy) | research_agency_lab/experiments/deep_research/05_topology_geometry_graphs.md |
| `math:vineyards` | proposed | 3 | Vineyards (time-varying persistence) | research_agency_lab/experiments/deep_research/05_topology_geometry_graphs.md |
| `math:channel_capacity` | proposed | 4 | Channel capacity (Blahut–Arimoto) | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `math:chow_liu_bn` | proposed | 4 | Chow-Liu Bayesian network over detector residuals | src:doi:10.1109/TIT.1968.1054142 |
| `math:gp_latent_force` | proposed | 4 | B3 GP / latent-force tempo drift |  |
| `math:hsmm_explicit_duration` | proposed | 4 | explicit-duration HSMM re-segmentation |  |
| `math:info_bottleneck` | proposed | 4 | Information bottleneck (nuisance invariance) | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `math:inverse_opt` | proposed | 4 | Inverse optimisation / structured max-margin for costs |  |
| `math:koopman_hankel_dmd` | proposed | 4 | B2 Hankel-DMD Koopman spectra |  |
| `math:lm_entropy` | proposed | 4 | Continuation entropy / surprisal / cross-cut PMI | src:camelbert, src:wolf-2023 |
| `math:mdl_probe` | proposed | 4 | MDL probing | src:arXiv:2003.12298 |
| `math:mrmr_ksg` | proposed | 4 | mRMR with KSG mutual information | src:doi:10.1103/PhysRevE.69.066138, src:doi:10.1109/TPAMI.2005.159 |
| `math:rate_distortion` | speculative | 3 | Rate–distortion perfection (expected distortion) | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `math:dtm_filtration` | speculative | 4 | DTM filtration | research_agency_lab/experiments/deep_research/05_topology_geometry_graphs.md |
| `math:fitness_fatigue` | speculative | 4 | Latent fatigue ODE (Banister form) | research_agency_lab/experiments/deep_research/07_dsp_julia_discovery.md |
| `math:graph_wavelets` | speculative | 4 | Spectral graph wavelets | research_agency_lab/experiments/deep_research/05_topology_geometry_graphs.md |
| `math:grassmannian` | speculative | 4 | Grassmann / Martin distance | research_agency_lab/experiments/deep_research/05_topology_geometry_graphs.md |
| `math:gromov_wasserstein` | speculative | 4 | Gromov-Wasserstein confusion geometry | research_agency_lab/experiments/deep_research/01_makharij.md |
| `math:langevin_maqam_potential` | speculative | 4 | Langevin SDE with maqam potential wells | research_agency_lab/experiments/deep_research/07_dsp_julia_discovery.md |
| `math:qta_fujisaki` | speculative | 4 | qTA / Fujisaki pitch target models | research_agency_lab/experiments/deep_research/07_dsp_julia_discovery.md |
| `math:sheaf_of_sets_csp` | speculative | 4 | Sheaf of sets / contextuality CSP | research_agency_lab/experiments/deep_research/05_topology_geometry_graphs.md |
| `math:sliding_window_torus` | speculative | 4 | Sliding-window tori (quasiperiodic) | research_agency_lab/experiments/deep_research/05_topology_geometry_graphs.md |
| `math:zigzag` | speculative | 4 | Zigzag persistence | research_agency_lab/experiments/deep_research/05_topology_geometry_graphs.md |
| `math:gnn` | speculative | 5 | GNN / neural sheaf diffusion | research_agency_lab/experiments/deep_research/05_topology_geometry_graphs.md |
| `math:hodge_flow_melody` | speculative | 5 | Hodge decomposition of melodic flows | research_agency_lab/experiments/deep_research/05_topology_geometry_graphs.md |
| `math:multiparameter_persistence` | speculative | 5 | Multiparameter persistence (signed barcodes) | research_agency_lab/experiments/deep_research/05_topology_geometry_graphs.md |
| `math:persistent_homology` | speculative | 5 | C3 sliding-window persistent homology of F0 |  |
| `math:mnar` | missing | 1 | Missing-not-at-random skip modeling | research_agency_lab/experiments/deep_research/06_information_neural.md |

## Algorithms

| id | status | prio | label | where / refs |
|---|---|---|---|---|
| `algo:conformal_thresholds` | implemented | 1 | Conformal LOO thresholds (roadmap A4) | research_agency_lab/experiments/deep_research/01_makharij.md |
| `algo:sindy_stlsq` | implemented | 1 | SINDy / STLSQ | research_agency_lab/experiments/deep_research/07_dsp_julia_discovery.md |
| `algo:text_swap_h1` | implemented | 1 | Counterfactual text-swap H1 from reference audio | research_agency_lab/experiments/deep_research/01_makharij.md |
| `algo:heaviness_index` | implemented | 2 | Heaviness index F2-F1 vs light reference | app/sifaat/formants.py |
| `algo:wpe` | implemented | 2 | WPE dereverberation | research_agency_lab/experiments/deep_research/07_dsp_julia_discovery.md |
| `algo:hf_tilt_flatness` | implemented | 3 | HF spike / spectral flatness (current) | app/sifaat/ghair_mutadhaddah.py |
| `algo:tap_count` | implemented | 4 | Tap dip counting (current) | app/sifaat/ghair_mutadhaddah.py |
| `algo:hnr_voicing` | implemented | 5 | HNR + F0 voicing fraction (current) | app/sifaat/hams_jahr.py |
| `algo:anchor_tail_ruler` | implemented |  | Calibration ruler choice by the near-gold anchor's false-FAIL rate under the final band (not robust CV) | research_agency_lab/substrate_library/julia/src/QaariLab.jl |
| `algo:beat_normalization` | implemented |  | B1 local-beat madd counts (same-ayah tabii as ruler) + Fisher separation test | Frontier.jl beat_normalization |
| `algo:breath_flatness` | implemented |  | breath = unvoiced (<30%), audible, flat (500-5000 Hz flatness >=0.15) frames >=120 ms | app/sifaat/hams_jahr.py detect_breath |
| `algo:ctc_semiglobal` | implemented |  | semi-global CTC Viterbi (free start/end word, 2-nat token bonus) + score-weighted LIS chain | app/segmenter.py |
| `algo:ctc_viterbi` | implemented |  | exact CTC Viterbi forced alignment | app/aligner.py |
| `algo:greedy_set_cover` | implemented |  | weighted greedy set cover of ayahs over rule keys | datasets/strategic_verses.py |
| `algo:late_reverb_suppression` | implemented |  | Lebart/Habets statistical late-reverb spectral gain (floor -15 dB) | app/taraweeh_adapter/dereverb.py |
| `algo:local_beat_window` | implemented |  | local harakah = median short-syllable duration within +-4 s (>=5 samples) | app/taraweeh_adapter/pace_normalizer.py |
| `algo:median_iqr_beat` | implemented |  | harakah = median of open short-syllable spans after 1.5 IQR rejection; global-rate fallback | app/acoustic/tempo.py |
| `algo:ner_nasal_contrast` | implemented |  | NER = 10 log10(E[150-400]/E[750-1100]) minus the reciter's oral-vowel median | app/sifaat/formants.py |
| `algo:praat_burg_formants` | implemented |  | Praat Burg formants (ceiling 5000/5500 Hz), median over voiced nucleus frames within 6 dB | app/acoustic/features.py |
| `algo:reading_path` | implemented |  | Free decode + Needleman–Wunsch against the text: repetitions (iʿādah) and skips typed and inserted into the reading path before GOP |  |
| `algo:release_burst` | implemented |  | closure (RMS < p98-8 dB, >=15 ms) + high-band (>1.5 kHz) flux peak + energy rise | app/tajweed_rules/qalqalah_engine.py |
| `algo:rt60_blind` | implemented |  | blind RT60 from T20-style decay fits after offsets (500-4000 Hz) | app/taraweeh_adapter/dereverb.py |
| `algo:spectral_subtraction` | implemented |  | spectral subtraction denoiser (quietest 10% frames, over-subtraction 1.5) | app/audio.py |
| `algo:vowel_core_hilbert` | implemented |  | core_ms = longest voiced run of 100-1000 Hz Hilbert envelope within 10 dB of span peak | app/tajweed_rules/base.py vowel_core_ms / qaari_features.m |
| `algo:weight_search_nelder_mead` | implemented |  | category-weight search: max peers-imams separation, LOO peers >= 95, L2 pull (Nelder-Mead) | QaariLab.jl optimise_weights |
| `algo:gop_segmentation_free` | partial | 1 | Segmentation-free sequence GOP with restricted substitutions | research_agency_lab/experiments/deep_research/01_makharij.md |
| `algo:gop_subst_aware` | partial | 1 | Substitution-aware GOP (lahn-jali confusion set) | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `algo:pyin` | partial | 1 | pYIN | research_agency_lab/experiments/deep_research/07_dsp_julia_discovery.md |
| `algo:nasal_a1p0_ner` | partial | 3 | NER + A1-P0 nasality | app/sifaat/formants.py |
| `algo:second_judge_ensemble` | partial | 3 | Second judges: muaalem-v3.2 + espeak IPA (linguistic-bias check) | research_agency_lab/experiments/deep_research/01_makharij.md |
| `algo:collapse_gate` | proposed | 1 | errors-in-variables collapse mixture for CTC-peaky long madds | arXiv:2105.14849 |
| `algo:cpp_aperiodicity` | proposed | 1 | CPP + D4C aperiodicity + H1-H2 | src:doi:10.1016/j.specom.2016.09.001, src:doi:10.1044/jshr.3704.769, src:doi:10.1121/1.417991 |
| `algo:esindy` | proposed | 1 | Ensemble SINDy + stability selection | research_agency_lab/experiments/deep_research/07_dsp_julia_discovery.md |
| `algo:failure_incidents` | proposed | 1 | Failure incidents + root cause | research_agency_lab/experiments/deep_research/05_topology_geometry_graphs.md |
| `algo:ffbs_alignment_uncertainty` | proposed | 1 | FFBS alignment sampling -> verdict distribution | research_agency_lab/experiments/deep_research/05_topology_geometry_graphs.md |
| `algo:gop_sf` | proposed | 1 | Segmentation-free CTC GOP | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `algo:haraka_field` | proposed | 1 | Joint haraka field + global madd choice | research_agency_lab/experiments/deep_research/05_topology_geometry_graphs.md |
| `algo:karma_kalman` | proposed | 1 | KARMA Kalman formant+antiformant tracker | research_agency_lab/experiments/deep_research/07_dsp_julia_discovery.md |
| `algo:landmark_closure_burst` | proposed | 1 | Closure/burst landmark detector | app/tajweed_rules/qalqalah_engine.py |
| `algo:levy_precedence` | proposed | 1 | Precedence index for ikhfa/iqlab/qalqalah/naqis | research_agency_lab/experiments/deep_research/05_topology_geometry_graphs.md |
| `algo:lpc_harmonised` | proposed | 1 | Harmonised LPC ceiling + slot-swap QC | research_agency_lab/experiments/deep_research/07_dsp_julia_discovery.md |
| `algo:modulation_spectrum` | proposed | 1 | Modulation spectrum / SRMR | research_agency_lab/experiments/deep_research/07_dsp_julia_discovery.md |
| `algo:multitaper` | proposed | 1 | Multitaper spectra | research_agency_lab/experiments/deep_research/07_dsp_julia_discovery.md |
| `algo:nuts` | proposed | 1 | NUTS + PSIS-LOO | research_agency_lab/experiments/deep_research/07_dsp_julia_discovery.md |
| `algo:pairwise_cue_llr` | proposed | 1 | Pairwise acoustic cue LLR (moments, VOT, burst, locus, closure, A1-P0) | research_agency_lab/experiments/deep_research/01_makharij.md |
| `algo:pause_mark_parse` | proposed | 1 | Keep standalone pause-mark tokens as Word.pause_mark | app/tajweed_rules/parser.py |
| `algo:perturbation` | proposed | 1 | Counterfactual perturbation (WSOLA madd, ghunnah trunc, splices) | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `algo:prominence_event_count` | proposed | 1 | Prominence-based tap/release/burst counting | research_agency_lab/experiments/deep_research/05_topology_geometry_graphs.md |
| `algo:reverb_audit` | proposed | 1 | RIR robustness audit | research_agency_lab/experiments/deep_research/07_dsp_julia_discovery.md |
| `algo:sakt_in_beats` | proposed | 1 | judge sakt length in beats, not 200-400 ms | app/tajweed_rules/sakt_wasl.py |
| `algo:symbolic_regression` | proposed | 1 | Dimensional symbolic regression | research_agency_lab/experiments/deep_research/07_dsp_julia_discovery.md |
| `algo:two_ruler_beat` | proposed | 1 | beat from short syllables + tabii (leave-one-out), log scale, SE |  |
| `algo:vot_aspiration` | proposed | 1 | VOT / post-release aspiration | src:doi:10.1093/jss/fgu035, src:doi:10.1177/0023830920986821 |
| `algo:waqf_dp` | proposed | 1 | Shortest-path DP optimal segmentation (mu'anaqah state bit, go-back) | research_agency_lab/substrate_library/julia/src/Waqf.jl |
| `algo:wing_kristofferson_noise` | proposed | 1 | label-free boundary noise = -lag1 autocov of syllable durations - Delta^2/12 |  |
| `algo:attribute_mdd` | proposed | 2 | Articulatory-attribute MDD | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `algo:blahut_arimoto_fano` | proposed | 2 | Channel capacity (Blahut-Arimoto) + Fano bound = verifiability audit | research_agency_lab/experiments/deep_research/01_makharij.md |
| `algo:boundary_classifier` | proposed | 2 | Boundary posterior (wasl/sakt/waqf/breath/hesitation) |  |
| `algo:cpp` | proposed | 2 | Cepstral peak prominence | research_agency_lab/experiments/deep_research/07_dsp_julia_discovery.md |
| `algo:crepe_rmvpe` | proposed | 2 | CREPE / RMVPE | research_agency_lab/experiments/deep_research/07_dsp_julia_discovery.md |
| `algo:free_phone_decode` | proposed | 2 | Free phone decoding + Levenshtein | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `algo:glrt_burst` | proposed | 2 | CUSUM/GLRT burst onset + Teager energy | research_agency_lab/experiments/deep_research/07_dsp_julia_discovery.md |
| `algo:gop_frame_posterior` | proposed | 2 | Frame-posterior GOP (LPP/LPR, blank-weighted) | research_agency_lab/experiments/deep_research/01_makharij.md |
| `algo:gpu_ensemble_bootstrap` | proposed | 2 | GPU ensemble parametric bootstrap nulls | research_agency_lab/experiments/deep_research/07_dsp_julia_discovery.md |
| `algo:iaif_qcp` | proposed | 2 | IAIF / QCP glottal inverse filtering | research_agency_lab/experiments/deep_research/07_dsp_julia_discovery.md |
| `algo:ksg` | proposed | 2 | KSG MI estimator | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `algo:logit_gop_entropy` | proposed | 2 | Logit GOP, posterior entropy/margin | research_agency_lab/experiments/deep_research/01_makharij.md |
| `algo:lpc_counterfactual` | proposed | 2 | LPC resynthesis counterfactual validation | research_agency_lab/substrate_library/octave/lpc.m |
| `algo:mahalanobis_ood` | proposed | 2 | Class-conditional Mahalanobis anomaly | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `algo:makhraj_w1_score` | proposed | 2 | Tree-W1 makhraj score with directional feedback | research_agency_lab/experiments/deep_research/05_topology_geometry_graphs.md |
| `algo:maqam_scale_degrees` | proposed | 2 | Persistent scale-degree extraction + circular OT maqam | research_agency_lab/experiments/deep_research/05_topology_geometry_graphs.md |
| `algo:pitch_histogram` | proposed | 2 | Tonic + learned-scale pitch histogram | research_agency_lab/experiments/deep_research/07_dsp_julia_discovery.md |
| `algo:psis_loo` | proposed | 2 | PSIS-LOO | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `algo:reassignment` | proposed | 2 | Reassigned spectrogram | research_agency_lab/experiments/deep_research/07_dsp_julia_discovery.md |
| `algo:restart_align` | proposed | 2 | Restart-capable semi-global CTC Viterbi | app/segmenter.py |
| `algo:scattering` | proposed | 2 | Wavelet scattering transform | research_agency_lab/experiments/deep_research/07_dsp_julia_discovery.md |
| `algo:spectral_moments` | proposed | 2 | Spectral moments COG/SD/skew/kurtosis | research_agency_lab/experiments/deep_research/07_dsp_julia_discovery.md |
| `algo:ssl_prototype_verifier` | proposed | 2 | Training-free SSL layer prototypes (LDA/Mahalanobis) | research_agency_lab/experiments/deep_research/01_makharij.md |
| `algo:ude` | proposed | 2 | Universal differential equation + distillation | research_agency_lab/experiments/deep_research/07_dsp_julia_discovery.md |
| `algo:weak_sindy` | proposed | 2 | Weak-form SINDy | research_agency_lab/experiments/deep_research/07_dsp_julia_discovery.md |
| `algo:breath_detector` | proposed | 3 | Breath vs sakt classifier | research_agency_lab/experiments/deep_research/07_dsp_julia_discovery.md |
| `algo:cond_flow` | proposed | 3 | Conditional normalizing flow | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `algo:confusion_audit` | proposed | 3 | CTC confusion-graph makhraj audit | research_agency_lab/experiments/deep_research/05_topology_geometry_graphs.md |
| `algo:deep_knn` | proposed | 3 | Deep kNN anomaly | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `algo:diphthong_trajectory` | proposed | 3 | Diphthong F2 trajectory + smoothness |  |
| `algo:dmd` | proposed | 3 | Hankel DMD | research_agency_lab/experiments/deep_research/07_dsp_julia_discovery.md |
| `algo:f2_transition` | proposed | 3 | F2 transition duration, no-release |  |
| `algo:fr_choice_consistency` | proposed | 3 | FR choice-distribution consistency | research_agency_lab/experiments/deep_research/05_topology_geometry_graphs.md |
| `algo:gop_classic` | proposed | 3 | Classic GOP (Witt–Young) | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `algo:graph_tikhonov_profile` | proposed | 3 | Graph-Tikhonov per-letter learner profile | research_agency_lab/experiments/deep_research/01_makharij.md |
| `algo:hodgerank_reciters` | proposed | 3 | HodgeRank reciter comparison / confound detector | research_agency_lab/experiments/deep_research/05_topology_geometry_graphs.md |
| `algo:hsmm` | proposed | 3 | Explicit-duration HMM segmentation | research_agency_lab/experiments/deep_research/07_dsp_julia_discovery.md |
| `algo:lateral_antiformant` | proposed | 3 | Lateral antiformant / F3 check |  |
| `algo:pelt_bocpd` | proposed | 3 | change points (PELT offline, BOCPD/EWMA online) for tempo drift |  |
| `algo:raum_detector` | proposed | 3 | Raum/sukoon/full-vowel 3-class detector, SNR-gated |  |
| `algo:replicate_icc` | proposed | 3 | Repetition-edge replicate ICC | research_agency_lab/experiments/deep_research/05_topology_geometry_graphs.md |
| `algo:sheaf_reconciliation` | proposed | 3 | Python/Octave/overlapping-rule sheaf reconciliation | research_agency_lab/experiments/deep_research/05_topology_geometry_graphs.md |
| `algo:sigkernel_mmd` | proposed | 3 | Signature-kernel MMD path-law test | research_agency_lab/experiments/deep_research/05_topology_geometry_graphs.md |
| `algo:spectral_clustering_confusion` | proposed | 3 | Shepard similarity + Laplacian spectral clustering; eigengap 14/16/17 test | research_agency_lab/experiments/deep_research/01_makharij.md |
| `algo:ssl_probe` | proposed | 3 | Layerwise SSL linear probes (within-letter contrasts) | app/aligner.py |
| `algo:synchrosqueezing` | proposed | 3 | Synchrosqueezing (2nd order) | research_agency_lab/experiments/deep_research/07_dsp_julia_discovery.md |
| `algo:temperature_scaling` | proposed | 3 | Temperature scaling of CTC logits | research_agency_lab/experiments/deep_research/01_makharij.md |
| `algo:topo_formant_tracking` | proposed | 3 | Topological formant/antiformant tracking | research_agency_lab/experiments/deep_research/05_topology_geometry_graphs.md |
| `algo:waqf_cut_cost` | proposed | 3 | Waqf/ibtida dependency-cut scoring | research_agency_lab/experiments/deep_research/05_topology_geometry_graphs.md |
| `algo:am_spectrum_trill` | proposed | 4 | Amplitude-modulation spectrum trill rate | app/sifaat/ghair_mutadhaddah.py |
| `algo:blahut_arimoto` | proposed | 4 | Blahut–Arimoto | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `algo:breath_mfcc` | proposed | 4 | MFCC-template breath detection | src:ruinskiy-lavner-2007 |
| `algo:contrastive_embed` | proposed | 4 | SupCon/GE2E reciter embedding | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `algo:sparc_inversion` | proposed | 4 | SPARC articulatory inversion (lisan/shafatan only) | research_agency_lab/experiments/deep_research/01_makharij.md |
| `algo:deep_svdd` | speculative | 4 | Deep SVDD | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `algo:deepformants` | speculative | 4 | DNN formant tracker | research_agency_lab/experiments/deep_research/07_dsp_julia_discovery.md |
| `algo:muaalem_teacher` | speculative | 4 | Muaalem multi-level CTC sifaat teacher | src:arXiv:2509.00094 |
| `algo:neural_sde` | speculative | 4 | Neural / Langevin SDE | research_agency_lab/experiments/deep_research/07_dsp_julia_discovery.md |
| `algo:particle_filter_formants` | speculative | 4 | Particle-filter formant tracker | research_agency_lab/experiments/deep_research/07_dsp_julia_discovery.md |
| `algo:tahrirat_csp` | speculative | 4 | Tahrirat arc-consistency checker | research_agency_lab/experiments/deep_research/05_topology_geometry_graphs.md |
| `algo:area_function_wakita` | speculative | 5 | LPC area function pharyngeal index | research_agency_lab/experiments/deep_research/01_makharij.md |
| `algo:emd_vmd` | speculative | 5 | EMD / VMD | research_agency_lab/experiments/deep_research/07_dsp_julia_discovery.md |
| `algo:multihead_makhraj_ctc` | speculative | 5 | Multi-head letter+makhraj+sifat CTC with Wasserstein loss | research_agency_lab/experiments/deep_research/01_makharij.md |
| `algo:ssm_prosody` | speculative | 5 | S4/Mamba/Neural CDE prosody | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `algo:zigzag_fatigue` | speculative | 5 | Zigzag vibrato-loss fatigue detector | research_agency_lab/experiments/deep_research/05_topology_geometry_graphs.md |

## Packages

| id | status | prio | label | where / refs |
|---|---|---|---|---|
| `pkg:julia:DataDrivenDiffEq` | implemented | 2 | DataDrivenDiffEq 1.16.1 | research_agency_lab/experiments/deep_research/07_dsp_julia_discovery.md |
| `pkg:julia:DataDrivenSparse` | implemented | 2 | DataDrivenSparse 0.2.3 | research_agency_lab/experiments/deep_research/07_dsp_julia_discovery.md |
| `pkg:julia:ModelingToolkit` | implemented | 2 | ModelingToolkit 11.45.1 | research_agency_lab/experiments/deep_research/07_dsp_julia_discovery.md |
| `pkg:octave:signal` | implemented | 2 | Octave signal | research_agency_lab/experiments/deep_research/07_dsp_julia_discovery.md |
| `pkg:python:nara_wpe` | implemented | 2 | nara_wpe | research_agency_lab/experiments/deep_research/07_dsp_julia_discovery.md |
| `pkg:julia:Graphs` | implemented | 3 | Graphs.jl |  |
| `pkg:julia:SparseArrays` | implemented | 3 | SparseArrays |  |
| `pkg:julia_linearalgebra_distributions` | implemented | 3 | Julia LinearAlgebra/Distributions/Optim (QaariLab) |  |
| `pkg:transformers` | implemented | 3 | transformers | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `pkg:octave` | implemented | 4 | Octave DSP cross-check (lpc.m) | research_agency_lab/substrate_library/octave/lpc.m |
| `pkg:ForwardDiff.jl` | implemented |  | ForwardDiff.jl |  |
| `pkg:Optim.jl` | implemented |  | Optim.jl |  |
| `pkg:octave-signal` | implemented |  | Octave signal |  |
| `pkg:parselmouth` | implemented |  | Praat/parselmouth |  |
| `pkg:julia-qaarilab` | partial | 2 | QaariLab.jl | research_agency_lab/substrate_library/julia/src/QaariLab.jl |
| `pkg:julia:CUDA` | proposed | 2 | CUDA 6.4.0 | research_agency_lab/experiments/deep_research/07_dsp_julia_discovery.md |
| `pkg:julia:DataDrivenDMD` | proposed | 2 | DataDrivenDMD 0.1.8 | research_agency_lab/experiments/deep_research/07_dsp_julia_discovery.md |
| `pkg:julia:DataDrivenSR` | proposed | 2 | DataDrivenSR 0.1.9 | research_agency_lab/experiments/deep_research/07_dsp_julia_discovery.md |
| `pkg:julia:DiffEqGPU` | proposed | 2 | DiffEqGPU 3.21.2 | research_agency_lab/experiments/deep_research/07_dsp_julia_discovery.md |
| `pkg:julia:DynamicQuantities` | proposed | 2 | DynamicQuantities 1.13.0 | research_agency_lab/experiments/deep_research/07_dsp_julia_discovery.md |
| `pkg:julia:LowLevelParticleFilters` | proposed | 2 | LowLevelParticleFilters 3.31.1 | research_agency_lab/experiments/deep_research/07_dsp_julia_discovery.md |
| `pkg:julia:Lux` | proposed | 2 | Lux 1.31.4 | research_agency_lab/experiments/deep_research/07_dsp_julia_discovery.md |
| `pkg:julia:ParetoSmooth` | proposed | 2 | ParetoSmooth 0.7.17 | research_agency_lab/experiments/deep_research/07_dsp_julia_discovery.md |
| `pkg:julia:SciMLSensitivity` | proposed | 2 | SciMLSensitivity 7.119.11 | research_agency_lab/experiments/deep_research/07_dsp_julia_discovery.md |
| `pkg:julia:StochasticDiffEq` | proposed | 2 | StochasticDiffEq 7.2.0 | research_agency_lab/experiments/deep_research/07_dsp_julia_discovery.md |
| `pkg:julia:SymbolicRegression` | proposed | 2 | SymbolicRegression.jl | research_agency_lab/experiments/deep_research/07_dsp_julia_discovery.md |
| `pkg:julia:Turing` | proposed | 2 | Turing.jl | research_agency_lab/experiments/deep_research/07_dsp_julia_discovery.md |
| `pkg:octave:ltfat` | proposed | 2 | ltfat forge | research_agency_lab/experiments/deep_research/07_dsp_julia_discovery.md |
| `pkg:python:crepe` | proposed | 2 | crepe | research_agency_lab/experiments/deep_research/07_dsp_julia_discovery.md |
| `pkg:python:kymatio` | proposed | 2 | kymatio | research_agency_lab/experiments/deep_research/07_dsp_julia_discovery.md |
| `pkg:quranmb` | proposed | 2 | IqraEval QuranMB.v1 test set |  |
| `pkg:Associations.jl` | proposed | 3 | Associations.jl (KSG MI) | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `pkg:ConformalPrediction.jl` | proposed | 3 | ConformalPrediction.jl | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `pkg:Lux.jl` | proposed | 3 | Lux.jl | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `pkg:NearestNeighbors.jl` | proposed | 3 | NearestNeighbors.jl | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `pkg:ParetoSmooth.jl` | proposed | 3 | ParetoSmooth.jl | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `pkg:Turing.jl` | proposed | 3 | Turing.jl | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `pkg:brms` | proposed | 3 | brms | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `pkg:muaalem_model` | proposed | 3 | HF obadx/muaalem-model-v3_2 + muaalem-annotated-v3 |  |
| `pkg:recitation-segmenter-v2` | proposed | 3 | obadx/recitation-segmenter-v2 (wav2vec2-BERT, MIT) | src:arxiv-2509.00094 |
| `pkg:zuko` | proposed | 3 | zuko | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `pkg:camelbert-ca` | proposed | 4 | CAMeLBERT-CA classical Arabic LM | src:camelbert |
| `pkg:speech_articulatory_coding` | proposed | 4 | pip speech-articulatory-coding (SPARC) |  |
| `pkg:julia:MixedModels` | proposed |  | MixedModels.jl |  |
| `pkg:julia:GraphNeuralNetworks` | missing | 3 | GraphNeuralNetworks.jl |  |
| `pkg:julia:Manifolds` | missing | 3 | Manifolds.jl |  |
| `pkg:julia:Manopt` | missing | 3 | Manopt.jl |  |
| `pkg:julia:OptimalTransport` | missing | 3 | OptimalTransport.jl |  |
| `pkg:julia:PersistenceDiagrams` | missing | 3 | PersistenceDiagrams.jl |  |
| `pkg:julia:Ripserer` | missing | 3 | Ripserer.jl |  |
| `pkg:julia:SimpleWeightedGraphs` | missing | 3 | SimpleWeightedGraphs.jl |  |
| `pkg:python:POT` | missing | 3 | POT (GW/FGW) |  |
| `pkg:python:iisignature` | missing | 3 | iisignature |  |
| `pkg:python:multipers` | missing | 3 | multipers |  |

## Pretrained models

| id | status | prio | label | where / refs |
|---|---|---|---|---|
| `model:hf:TBOGamer22/wav2vec2-quran-phonetics` | implemented | 2 | Current aligner; word-level training (domain shift) | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `model:ecapa` | implemented | 3 | ECAPA-TDNN timbre | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `model:hf:obadx/muaalem-model-v3_2` | proposed | 1 | w2v-BERT 2.0 11-level CTC (phonemes+10 sifaat) | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `model:hf:facebook/w2v-bert-2.0` | proposed | 2 | SSL backbone | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `model:hf:IqraEval/Iqra_wavlm_base` | proposed | 3 | Iqra'Eval MDD baseline | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `model:hf:microsoft/wavlm-large` | proposed | 3 | SSL backbone robust to reverb | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `model:hf:obadx/recitation-segmenter-v2` | proposed | 4 | Waqf segmenter | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `model:hf:openai/whisper-large-v3` | proposed | 4 | Encoder for speech-to-phoneme | research_agency_lab/experiments/deep_research/06_information_neural.md |

## Datasets

| id | status | prio | label | where / refs |
|---|---|---|---|---|
| `data:local:everyayah_only_strategic` | implemented | 2 | 491 rows local raw verdicts | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `data:benchmark_rows_local` | implemented |  | benchmarks/results/local/*.jsonl: Husary 59 + 5 EveryAyah-only reciters x 59 x 2 modes (649 rows) | benchmarks/results/local/ |
| `data:strategic_verses` | implemented |  | app/data/strategic_verses.json: 59 ayahs covering 48 rule keys (>=6 each or all) | app/data/strategic_verses.json |
| `dataset:qaari_keys_tiers` | implemented |  | QaariKeys tiers: T10 (109 words, 171/188 keys), T100 (1133 words, 188/188), T300 (4243 words, 188/188) × 43 reciter-styles + 13 quran.com timed | datasets/qaari_keys/build/tiers.json |
| `dataset:qdc_segments` | implemented |  | quran.com / QDC per-word segments on 13 surah recordings (verse cut + shifted) |  |
| `dataset:quran_align` | implemented |  | quran-align word timings (C. Fair 2016, CC BY 4.0) for 12 EveryAyah reciters (As-Sudais file is a crash log) | ~/.cache/qaari-eval/quran_align |
| `data:hf:Buraaq/quran-md-ayahs` | proposed | 1 | Quran-MD 30 reciters | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `data:hf:obadx/qdat_bench` | proposed | 1 | 159 learner clips, per-rule madd/ghunnah/qalqalah labels | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `data:perturbation_set` | proposed | 1 | Synthetic labelled lahn (to build) | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `data:hf:IqraEval/QuranMB.v2` | proposed | 2 | Iqra'Eval MDD test (phoneme, no Tajweed) | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `data:hf:obadx/muaalem-annotated-v3` | proposed | 2 | 850h expert QPS-annotated | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `data:tadabur` | proposed | 2 | Tadabur 1400h 600+ reciters | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `data:hf:IqraEval/Iqra_TTS` | proposed | 3 | Synthetic error TTS | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `data:hf:IqraEval/Iqra_train` | proposed | 3 | Iqra'Eval train | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `data:quranic_audio_dataset` | proposed | 3 | Crowdsourced learner recitations | research_agency_lab/experiments/deep_research/06_information_neural.md |
| `data:hf:Buraaq/quran-md-words` | proposed | 4 | Word-level audio (aligner training) | research_agency_lab/experiments/deep_research/06_information_neural.md |

## Code

| id | status | prio | label | where / refs |
|---|---|---|---|---|
| `code:app/acoustic/features.py` | implemented | 1 | Python Praat features | app/acoustic/features.py |
| `code:parser` | implemented | 1 | app/tajweed_rules/parser.py | app/tajweed_rules/parser.py |
| `code:research_agency_lab/substrate_library/julia/src/Discovery.jl` | implemented | 1 | Discovery.jl (tempo ODE + STLSQ) | research_agency_lab/substrate_library/julia/src/Discovery.jl |
| `code:research_agency_lab/substrate_library/octave/qaari_features.m` | implemented | 1 | Octave DSP features | research_agency_lab/substrate_library/octave/qaari_features.m |
| `code:app/aligner.py` | implemented | 2 | aligner.py | app/aligner.py |
| `code:app/calibration.py` | implemented | 2 | calibration.py | app/calibration.py |
| `code:app/fingerprint.py` | implemented | 2 | 232-d fingerprint | app/fingerprint.py |
| `code:app/scoring.py` | implemented | 2 | scoring.py | app/scoring.py |
| `code:fatigue_detector` | implemented | 2 | app/taraweeh_adapter/fatigue_detector.py | app/taraweeh_adapter/fatigue_detector.py |
| `code:research_agency_lab/substrate_library/julia/src/Frontier.jl` | implemented | 2 | BW, conformal, sinkhorn | research_agency_lab/substrate_library/julia/src/Frontier.jl |
| `code:research_agency_lab/substrate_library/julia/src/QaariLab.jl` | implemented | 2 | Julia calibrator | research_agency_lab/substrate_library/julia/src/QaariLab.jl |
| `code:sakt_wasl` | implemented | 2 | app/tajweed_rules/sakt_wasl.py | app/tajweed_rules/sakt_wasl.py |
| `code:segmenter` | implemented | 2 | app/segmenter.py | app/segmenter.py |
| `code:app/acoustic/tempo.py` | implemented | 3 | haraka estimate | app/acoustic/tempo.py |
| `code:app/sifaat/formants.py` | implemented | 3 | formants.py | app/sifaat/formants.py |
| `code:app/sifaat/ghair_mutadhaddah.py` | implemented | 3 | ghair_mutadhaddah.py | app/sifaat/ghair_mutadhaddah.py |
| `code:app/tajweed_rules/noon_sakinah.py` | implemented | 3 | noon_sakinah.py | app/tajweed_rules/noon_sakinah.py |
| `code:app/tajweed_rules/parser.py` | implemented | 3 | parser.py | app/tajweed_rules/parser.py |
| `code:app/tajweed_rules/qalqalah_engine.py` | implemented | 3 | qalqalah_engine.py | app/tajweed_rules/qalqalah_engine.py |
| `code:app/taraweeh_adapter/fatigue_detector.py` | implemented | 3 | fatigue_detector.py | app/taraweeh_adapter/fatigue_detector.py |
| `code:hams_jahr` | implemented | 3 | app/sifaat/hams_jahr.py (detect_breath) | app/sifaat/hams_jahr.py |
| `code:app/api.py` | implemented |  | FastAPI /health /analyze |  |
| `code:app/audio.py` | implemented |  | load/condition audio: mono, 16 kHz, 40 Hz HPF, -1 dBFS, SNR, spectral subtraction |  |
| `code:app/modal_endpoint.py` | implemented |  | Modal T4 scale-to-zero deployment of the API |  |
| `code:app/models.py` | implemented |  | RuleType (39), Status, LetterUnit, RuleInstance, Alignment, RuleDiagnostic |  |
| `code:app/pipeline.py` | implemented |  | pipeline.py | app/pipeline.py |
| `code:app/quran_text.py` | implemented |  | Uthmani text provider (bundled sample, alquran.cloud Tanzil cache), basmala stripping |  |
| `code:app/segmenter.py` | implemented |  | ayah locator: pause chunking + semi-global CTC Viterbi + forward chain |  |
| `code:app/sifaat/hams_jahr.py` | implemented |  | hams_jahr.py | app/sifaat/hams_jahr.py |
| `code:app/sifaat/itbaq.py` | implemented |  | itbaq.py | app/sifaat/itbaq.py |
| `code:app/sifaat/sukoon_spectrum.py` | implemented |  | sukoon_spectrum.py | app/sifaat/sukoon_spectrum.py |
| `code:app/tajweed_rules/base.py` | implemented |  | base.py | app/tajweed_rules/base.py |
| `code:app/tajweed_rules/idghaam_classes.py` | implemented |  | idghaam_classes.py | app/tajweed_rules/idghaam_classes.py |
| `code:app/tajweed_rules/meem_sakinah.py` | implemented |  | meem_sakinah.py | app/tajweed_rules/meem_sakinah.py |
| `code:app/tajweed_rules/mudood_engine.py` | implemented |  | mudood_engine.py | app/tajweed_rules/mudood_engine.py |
| `code:app/tajweed_rules/raa_lam_rules.py` | implemented |  | raa_lam_rules.py | app/tajweed_rules/raa_lam_rules.py |
| `code:app/tajweed_rules/sakt_wasl.py` | implemented |  | sakt_wasl.py | app/tajweed_rules/sakt_wasl.py |
| `code:app/taraweeh_adapter/dereverb.py` | implemented |  | blind RT60, WPE, late-reverb suppression, proximity EQ, 8-d environment profile |  |
| `code:app/taraweeh_adapter/pace_normalizer.py` | implemented |  | pace_normalizer.py | app/taraweeh_adapter/pace_normalizer.py |
| `code:benchmarks/roster.py` | implemented |  | 8 studio + 8 taraweeh EveryAyah reciters; Quran-MD ids |  |
| `code:benchmarks/run_benchmark.py` | implemented |  | per (reciter, ayah, mode) JSONL rows with raw textbook verdicts |  |
| `code:benchmarks/summarize.py` | implemented |  | aggregate rows, re-judge with calibration, build FAISS indices |  |
| `code:datasets/index_reciters.py` | implemented |  | run_benchmark (studio) + summarize -> indices |  |
| `code:datasets/qaari_keys/` | implemented |  | QaariKeys rule-targeted verse dataset: per-ayah key counts (rule/letter/confusion pair/repetition/Hafs special case/waqf sign), nested tiers T10⊂T100⊂T300, reciter catalogue, open word timings | datasets/qaari_keys/ |
| `code:datasets/strategic_verses.py` | implemented |  | weighted greedy set cover of ayahs over rule keys (59 ayahs, 515 words, 48 keys) |  |
| `code:research_agency_lab/compute_bridge/kaggle_bridge.py` | implemented |  | Kaggle kernel launcher/collector (+ kaggle_worker.py) |  |
| `code:research_agency_lab/compute_bridge/modal_batch.py` | implemented |  | Modal fan-out benchmark collection (CPU shards, optional T4) |  |
| `code:research_agency_lab/compute_bridge/octave_bridge.py` | implemented |  | run qaari_features.m on benchmark spans |  |
| `code:research_agency_lab/compute_bridge/validate_loop.py` | implemented |  | continuous Octave<->Julia validation loop |  |
| `code:research_agency_lab/experiments/deep_research/02_sifaat.md` | implemented |  | Sifaat deep research report | research_agency_lab/experiments/deep_research/02_sifaat.md |
| `code:research_agency_lab/experiments/deep_research/03_timing.md` | implemented |  | timing deep-research report |  |
| `code:research_agency_lab/experiments/learner_eval/muaalem_dump.py` | implemented |  | muaalem-v3.2 frame posteriors (phonemes + 10 sifat levels, 40 ms frames) as raw float32 for Julia/Octave; QuranMB, EveryAyah, QDC sources |  |
| `code:research_agency_lab/substrate_library/julia/calibrate.jl` | implemented |  | driver: runs -> calibration.json |  |
| `code:research_agency_lab/substrate_library/julia/discover.jl` | implemented |  | driver: duration law + tempo ODE per reciter |  |
| `code:research_agency_lab/substrate_library/julia/frontier.jl` | implemented |  | driver: frontier_calibrate + beat_normalization (+ --export clouds) |  |
| `code:research_agency_lab/substrate_library/julia/frontier_crosscheck.jl` | implemented |  | synthetic fixtures for the Octave cross-check |  |
| `code:research_agency_lab/substrate_library/julia/src/CtcGop.jl` | implemented |  | CTC forward/Viterbi, run units, segmentation-free substitution GOP on muaalem posteriors, repetition/skip detection and reading path | research_agency_lab/substrate_library/julia/src/CtcGop.jl |
| `code:research_agency_lab/substrate_library/julia/textswap_gop.jl` | implemented |  | text-swap test of muaalem GOP: recall and correct-diagnosis rate at conformal anchor-only thresholds |  |
| `code:research_agency_lab/substrate_library/octave/fr_*.m` | implemented |  | Octave mirror of Frontier.jl |  |
| `code:research_agency_lab/substrate_library/octave/lpc.m` | implemented |  | research_agency_lab/substrate_library/octave/lpc.m | research_agency_lab/substrate_library/octave/lpc.m |
| `code:tests/` | implemented |  | pytest suite (84 tests: 80 pass, 4 skip on 2026-09-23) |  |
| `code:research_agency_lab/substrate_library/julia/Manifest.toml` | partial | 1 | Julia Manifest (1.11.5; host julia is 1.10.9) | research_agency_lab/substrate_library/julia/Manifest.toml |
| `code:app/lahn/gop.py` | partial |  | substitution-aware CTC GOP (lahn jali), windowed forward-sum LLR vs confusion set |  |
| `code:app/makharij/` | proposed | 1 | proposed makharij package (tables/gop/cues/prototypes) | app/makharij/ |
| `code:waqf_choice` | proposed | 1 | app/tajweed_rules/waqf_choice.py (WAQF_CHOICE, IBTIDA, WAQF_FORM) | app/tajweed_rules/waqf_choice.py |
| `code:app/information/channel.py` | proposed | 2 | Proposed letter-channel module | app/information/channel.py |
| `code:app/mdd/` | proposed | 2 | Proposed MDD module (GOP, sifa tiers) | app/mdd/ |
