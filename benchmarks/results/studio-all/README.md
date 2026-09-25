# studio-all: the rows calibration v3 was built from

The Modal `studio-all` run (2026-09-23): 150 shards, 37,411 rows, all `mode: studio`, EveryAyah
recordings of the anchor and five peers across the whole Qur'an:

| Reciter | Rows |
|---|---|
| Husary_128kbps (anchor) | 6,234 |
| Husary_Muallim_128kbps | 6,233 |
| Abdul_Basit_Murattal_192kbps | 6,236 |
| Alafasy_128kbps | 6,236 |
| Hudhaify_128kbps | 6,236 |
| Minshawy_Murattal_128kbps | 6,236 |

`app/data/calibration.json` (v3) records `n_rows` 37,765. That is these 37,411 rows plus the studio rows
of `../local/everyayah_only_strategic.jsonl` (295) and `../local/husary_strategic_raw.jsonl` (59);
`QaariLab.load_rows` keeps studio rows only. The count matches, but the exact v3 command line was not
recorded.

Each shard is gzipped (`gzip -9 -n`) from `benchmarks/results/modal/studio-all/runs_NNNN.jsonl`;
decompressed, the shards concatenate byte for byte to the originals. `gunzip -k` them before
passing them to `calibrate.jl` or `benchmarks/summarize.py`.
