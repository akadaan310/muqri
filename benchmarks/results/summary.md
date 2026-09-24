# qaari-eval v2 benchmark

Up to 4055 ayahs per reciter (the most complete reciter's count; see each row).

| Reciter | Set | Raw textbook | Perfection (studio) | Perfection (adapted) | Sifaat | Timing FAILs studio → adapted | FP reduction |
|---|---|---|---|---|---|---|---|
| Al-Hussary (Muallim) | studio | 75.1 | 98.0 | 97.7 | 95.0 | 743 → 781 (of 32795) | -5.1% |
| Mahmoud Khalil Al-Hussary | studio | 71.2 | 97.1 | 97.1 | 94.1 | 864 → 773 (of 26718) | 10.5% |
| Abdul Basit Abdul Samad (Murattal) | studio | 69.1 | 92.5 | 95.1 | 95.6 | 1156 → 684 (of 11842) | 40.8% |
| Ali Al-Hudhaify | studio | 69.0 | 91.3 | 91.6 | 90.6 | 1976 → 1912 (of 19759) | 3.2% |
| Mishary Alafasy | studio | 68.1 | 89.4 | 90.2 | 96.4 | 1263 → 1188 (of 11062) | 5.9% |
| Siddiq Al-Minshawi (Murattal) | studio | 61.3 | 86.0 | 91.9 | 94.0 | 3036 → 1822 (of 17990) | 40.0% |
| Saud Al-Shuraim | taraweeh | 69.2 | 91.1 | 92.3 | 90.8 | 109 → 88 (of 1139) | 19.3% |
| Yasser Al-Dosari | taraweeh | 68.2 | 88.9 | 89.0 | 95.9 | 1057 → 1028 (of 8561) | 2.7% |
| Nasser Al-Qatami | taraweeh | 64.3 | 87.3 | 86.9 | 95.5 | 789 → 815 (of 4890) | -3.3% |

## Aggregates

- **studio_set_mean_perfection**: 92.4
- **taraweeh_set_mean_perfection_unadapted**: 89.1
- **taraweeh_set_mean_perfection_adapted**: 89.4
- **taraweeh_timing_fails_unadapted**: 1955
- **taraweeh_timing_fails_adapted**: 1931
- **taraweeh_timing_fp_reduction_pct**: 1.2
