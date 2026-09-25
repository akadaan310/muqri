# qaari-eval v2 benchmark

Up to 6236 ayahs per reciter (the most complete reciter's count; see each row).

| Reciter | Set | Raw textbook | Perfection (studio) | Perfection (adapted) | Sifaat | Timing FAILs studio → adapted | FP reduction |
|---|---|---|---|---|---|---|---|
| Al-Hussary (Muallim) | studio | 75.0 | 96.9 | 95.1 | 89.1 | 1381 → 2294 (of 44505) | -66.1% |
| Mahmoud Khalil Al-Hussary | studio | 71.2 | 95.1 | 94.1 | 88.2 | 2670 → 3300 (of 54704) | -23.6% |
| Abdul Basit Abdul Samad (Murattal) | studio | 68.9 | 92.5 | 92.8 | 90.3 | 2733 → 2429 (of 31930) | 11.1% |
| Ali Al-Hudhaify | studio | 69.0 | 90.8 | 91.2 | 86.1 | 3235 → 3079 (of 32206) | 4.8% |
| Mishary Alafasy | studio | 68.1 | 89.7 | 89.7 | 91.5 | 3516 → 3586 (of 31492) | -2.0% |
| Siddiq Al-Minshawi (Murattal) | studio | 61.3 | 87.3 | 90.0 | 89.6 | 3901 → 2882 (of 28138) | 26.1% |
| Saud Al-Shuraim | taraweeh | 69.1 | 90.4 | 90.7 | 85.2 | 3678 → 3579 (of 43239) | 2.7% |
| Abdul Rahman Al-Sudais | taraweeh | 62.6 | 89.0 | 89.5 | 89.9 | 4466 → 4512 (of 35497) | -1.0% |
| Yasser Al-Dosari | taraweeh | 68.0 | 88.3 | 88.0 | 89.6 | 5704 → 6031 (of 46432) | -5.7% |
| Nasser Al-Qatami | taraweeh | 65.0 | 85.5 | 84.9 | 89.1 | 7598 → 8017 (of 44116) | -5.5% |
| Abdullah Al-Juhany | taraweeh | 38.4 | 67.2 | 66.9 | 75.0 | 3856 → 3986 (of 10648) | -3.4% |

## Aggregates

- **studio_set_mean_perfection**: 92.0
- **taraweeh_set_mean_perfection_unadapted**: 84.1
- **taraweeh_set_mean_perfection_adapted**: 84.0
- **taraweeh_timing_fails_unadapted**: 25302
- **taraweeh_timing_fails_adapted**: 26125
- **taraweeh_timing_fp_reduction_pct**: -3.3
