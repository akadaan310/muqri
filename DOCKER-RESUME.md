# Resuming the qaari-eval workspace in Docker

The `Dockerfile` reproduces the research VM: Python 3.11 engine venv, Julia 1.11.5 with the pinned
`QaariLab` environment precompiled, and Octave 6 with `signal` + `control`. **No secrets are in the image.**

```bash
docker build -t qaari-workspace .                    # add --build-arg WITH_ML=1 for torch/transformers/speechbrain
docker run --rm -it -v "$PWD":/work --env-file /path/to/.env qaari-workspace
```

The `.env` file supplies credentials at run time. Only the variable names are listed here:
`GITHUB_TOKEN`, `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`, `KAGGLE_API_TOKEN` (Kaggle CLI ≥1.7 also wants
dummy `KAGGLE_USERNAME`/`KAGGLE_KEY`), `LIGHTNING_USER_ID`, `LIGHTNING_API_KEY`.

## Smoke checks inside the container

```bash
julia --project=research_agency_lab/substrate_library/julia research_agency_lab/substrate_library/julia/test/test_frontier.jl
D=/tmp/frx; julia --project=research_agency_lab/substrate_library/julia research_agency_lab/substrate_library/julia/frontier_crosscheck.jl $D \
  && octave-cli --path research_agency_lab/substrate_library/octave --eval "fr_crosscheck('$D')"   # expect AGREE
pytest -q tests
python research_agency_lab/knowledge_map/build_map.py
```

## Where the state lives

- **Research map:** `research_agency_lab/knowledge_map/` (start with its README; `build_map.py q <term>`).
- **Deep-research reports:** `research_agency_lab/experiments/deep_research/00–07_*.md`.
- **Benchmark rows:** `benchmarks/results/`. The Modal runs are in the Volume `qaari-runs`; collect them
  with `modal run research_agency_lab/compute_bridge/modal_batch.py::collect --tag <tag>`.
- **Compute limits:** Modal free tier allows 100 concurrent containers (the harness caps CPU at 90).
  Never run two large fan-outs at once.
