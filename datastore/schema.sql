-- qaari persistent store (DuckDB). Location: $QAARI_STORE (default ~/qaari-store/qaari.duckdb).
-- Every table is append/upsert-friendly so loaders can be re-run; `runs` records what produced what.

CREATE TABLE IF NOT EXISTS runs (
    run_id      VARCHAR PRIMARY KEY,          -- e.g. 'quran_phonetic:hafs4', 'modal:muaalem:T10'
    kind        VARCHAR,                      -- loader | compute | test | calibration | cloud
    params      JSON,
    started_at  TIMESTAMP DEFAULT current_timestamp,
    finished_at TIMESTAMP,
    cost_usd    DOUBLE,
    note        VARCHAR
);

-- ------------------------------------------------------------------ the text ---------------------
CREATE TABLE IF NOT EXISTS ayah (
    surah INTEGER, ayah INTEGER, uthmani VARCHAR, n_words INTEGER, n_letters INTEGER,
    PRIMARY KEY (surah, ayah)
);

-- Hafs phonetic script (quran_transcript.quran_phonetizer); one row per ayah per moshaf setting.
CREATE TABLE IF NOT EXISTS ayah_phonetic (
    moshaf      VARCHAR,                      -- e.g. 'hafs_m4' (madd munfasil/muttasil/arid = 4)
    surah INTEGER, ayah INTEGER,
    phonemes    VARCHAR,                      -- muaalem phoneme script, no spaces
    collapsed   VARCHAR,                      -- runs of one symbol collapsed (madd-length invariant)
    word_spans  INTEGER[][],                  -- [p0, p1) phoneme span per Uthmani word
    PRIMARY KEY (moshaf, surah, ayah)
);

-- ------------------------------------------------------------------ QaariKeys dataset -------------
CREATE TABLE IF NOT EXISTS verse_key (key VARCHAR PRIMARY KEY, family VARCHAR);
CREATE TABLE IF NOT EXISTS ayah_key (surah INTEGER, ayah INTEGER, key VARCHAR, n INTEGER, PRIMARY KEY (surah, ayah, key));
CREATE TABLE IF NOT EXISTS tier_verse (tier VARCHAR, rank INTEGER, surah INTEGER, ayah INTEGER, PRIMARY KEY (tier, surah, ayah));

-- ------------------------------------------------------------------ reciters and audio ------------
CREATE TABLE IF NOT EXISTS reciter (
    reciter_id VARCHAR PRIMARY KEY,           -- 'everyayah:<folder>' | 'qdc:<id>' | 'quranicaudio:<id>' | 'quranmb:<speaker>'
    source VARCHAR, name VARCHAR, style VARCHAR, kbps INTEGER,
    ladder VARCHAR,                           -- anchor | studio | fast | imam | learner (grading tier, not a verdict)
    meta JSON
);
CREATE TABLE IF NOT EXISTS audio_clip (
    clip_id VARCHAR PRIMARY KEY,              -- '<reciter_id>/<SSSAAA>' or a QuranMB id
    reciter_id VARCHAR, surah INTEGER, ayah INTEGER, path VARCHAR, duration_s DOUBLE
);
CREATE TABLE IF NOT EXISTS word_timing (
    source VARCHAR, reciter_id VARCHAR, surah INTEGER, ayah INTEGER, word_from INTEGER, word_to INTEGER,
    start_ms DOUBLE, end_ms DOUBLE
);

-- ------------------------------------------------------------------ model outputs -----------------
CREATE TABLE IF NOT EXISTS posterior_file (
    clip_id VARCHAR, model VARCHAR, dump_dir VARCHAR, file VARCHAR, frames INTEGER, duration_s DOUBLE,
    ref_phonemes VARCHAR, word_spans INTEGER[][], run_id VARCHAR,
    PRIMARY KEY (clip_id, model)
);
CREATE TABLE IF NOT EXISTS unit_gop (
    clip_id VARCHAR, model VARCHAR, unit INTEGER, symbol VARCHAR, ref_from INTEGER, ref_to INTEGER,
    gop DOUBLE, best VARCHAR, lr DOUBLE, run_id VARCHAR
);
CREATE TABLE IF NOT EXISTS deviation (
    clip_id VARCHAR, model VARCHAR, kind VARCHAR, ref_from INTEGER, ref_to INTEGER, at_ref INTEGER, run_id VARCHAR
);
CREATE TABLE IF NOT EXISTS test_result (      -- text-swap, learner, head-to-head, gate metrics …
    run_id VARCHAR, test VARCHAR, scope VARCHAR, metric VARCHAR, value DOUBLE, detail JSON
);

-- ------------------------------------------------------------------ engine benchmark rows ---------
CREATE TABLE IF NOT EXISTS engine_ayah (
    run_id VARCHAR, reciter_id VARCHAR, surah INTEGER, ayah INTEGER, perfection DOUBLE, sifaat DOUBLE,
    haraka_ms DOUBLE, environment JSON
);
CREATE TABLE IF NOT EXISTS engine_diag (
    run_id VARCHAR, reciter_id VARCHAR, surah INTEGER, ayah INTEGER, rule_type VARCHAR, rule_key VARCHAR,
    detail VARCHAR, word VARCHAR, letter VARCHAR, status VARCHAR, score DOUBLE, start_ms INTEGER, end_ms INTEGER,
    metrics JSON
);

-- ------------------------------------------------------------------ calibration & knowledge -------
CREATE TABLE IF NOT EXISTS calibration_version (version INTEGER PRIMARY KEY, source VARCHAR, installed BOOLEAN, body JSON, metrics JSON);
CREATE TABLE IF NOT EXISTS kg_node (id VARCHAR PRIMARY KEY, type VARCHAR, label VARCHAR, status VARCHAR, body JSON);
CREATE TABLE IF NOT EXISTS kg_edge (src VARCHAR, dst VARCHAR, rel VARCHAR);
