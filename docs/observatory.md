# Muqri Observatory: read-only inspection layer

A separate process in front of the running engine, for external technical review. It does not import, modify or restart the engine.

## What runs

| Process | Where | What |
|---|---|---|
| engine web app (unchanged) | `python -m app.webapp`, 0.0.0.0:8088, plain HTTP | the source of truth for Sessions |
| Observatory | `python -m observatory.server`, **127.0.0.1:8095** | read-only proxy and pages, token-gated |
| HTTPS quick tunnel | `~/bin/cloudflared tunnel --url http://127.0.0.1:8095` | public `https://<random>.trycloudflare.com`, pointing only at the Observatory |

Start, stop or show the URL with `scripts/observatory.sh start|stop|status|url`.

The quick-tunnel URL changes whenever cloudflared restarts. A permanent hostname needs a named tunnel on a Cloudflare account, which is not configured here.

## Routes (GET/HEAD only; every other method returns 405)

| Route | Content |
|---|---|
| `/sessions`, `/sessions/rounds`, `/sessions/round/{1-5}` | the existing Sessions UI and JSON, proxied from :8088, rounds 1–5 only, recording and upload controls hidden |
| `/observatory/audio/{exercise}/{take}/{file}` | the stored round 1–5 recordings (from `research_agency_lab/experiments/session_recordings/`, local only) |
| `/muqri-observatory` | the technical page |
| `/muqri-observatory/report` (and `.md`) | the full technical report |
| `/api/observatory/manifest` | the machine-readable manifest |

**Access.** The reviewer link carries `?token=…` once. The token is kept in an HttpOnly, Secure cookie, and the token itself lives in `~/.config/qaari/observatory.token` (chmod 600). Without it, every route returns 401.

## Data behind the pages

Everything is precomputed. The server never touches the database, credentials or the shell.

| File | Built by | From |
|---|---|---|
| `observatory/data/snapshot.json` | `observatory/build_snapshot.py` | session definitions, `sessions_results.jsonl` (as recorded), recording files, git, benchmark rows, T300 index, letter-corpus index |
| `observatory/data/current_engine.json` | `observatory/rescore_current.py` | the latest round 1–5 recordings rescored by the current engine; a dry run that writes nothing historical |
| `observatory/data/neo4j_snapshot.json` | a read-only Cypher session (labels, relationship types, counts, patterns, indexes) | the local Neo4j |
| `observatory/data/modal_billing.json` | `modal billing report` | the Modal workspace |

To refresh after engine changes, rebuild with `.venv/bin/python observatory/build_snapshot.py` and `.venv/bin/python observatory/rescore_current.py`.

## Reviewer orientation

1. **Start** at `/muqri-observatory`. Section 1 shows that there are **two engines**: the Sessions engine, and the benchmark pipeline behind the README's calibration numbers.
2. **Rounds 1–5:**
   - `/sessions` is the original UI. Use ☰ History for rounds 1–5; each take shows its last score card.
   - Observatory section 5 lists each exercise with its recordings (playable), per-round metrics (as recorded against the current engine), the engine commits between rounds, and the rescore trail of every recording.
   - Section 6 is the take-B matrix: every scripted mistake, its target check, its verdict and evidence, and both takes' audio.
3. **Recordings and results:** the audio is on the VM only. The as-recorded history is `research_agency_lab/experiments/sessions_results.jsonl` (in git). The per-take `*.report.json` and `*.score.json` files sit beside each recording.
4. **Architecture, what is measured, the makhārij and ṣifāt:** Observatory sections 1–4 and report sections 1–4.
5. **Neo4j:** section 9 (a snapshot) and `GRAPH.md`.
6. **Modal:** section 11. The code is in `research_agency_lab/compute_bridge/modal_*.py` and `app/modal_endpoint.py`.
7. **Julia and Octave:** section 10. The code is in `research_agency_lab/substrate_library/{julia,octave}/`.
8. **Synthesis:** section 13, split into current implementation and plans. Experiments are in `research_agency_lab/experiments/{synthesis,letter_corpus}/` and `SYNTHESIS.md` (the Matcha-TTS pilot). Plans are in `docs/synthesis-engine.md` and `docs/epics/EPIC-1.md`.
