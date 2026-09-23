# Knowledge map — research ↔ math ↔ code, as a graph

> **Start here: [CODEBASE_ATLAS.md](CODEBASE_ATLAS.md).** It is the central map of the codebase: every
> module's math, constants (file:line), validation and measured accuracy, the known-bugs register, the
> ranked research frontier (tied to report sections and graph node ids), graph hygiene, and verify commands.

**Start here** (Claude Code sessions / agents): one graph linking every Tajweed phenomenon (101 → ijazah)
to the mathematical constructs, algorithms, packages, sources and repo code that measure or model it,
with an implementation status on every node.

```bash
python research_agency_lab/knowledge_map/build_map.py              # merge fragments -> graph.json + INDEX.md
python research_agency_lab/knowledge_map/build_map.py q makharij   # a node's neighbourhood (id or text)
python research_agency_lab/knowledge_map/build_map.py gaps         # phenomena with no implemented model
```

- `INDEX.md` (generated): phenomenon → implemented / proposed models table, then per-type tables.
- `graph.json` (generated): `{nodes, edges, problems}`; load it with networkx/Graphs.jl for path queries.
- `fragments/*.json` (source of truth): `core.json` is hand-maintained (**what is implemented**);
  `00…07_*.json` come from the deep-research reports in `../experiments/deep_research/`.
  Never edit the generated files; add or update a fragment and rebuild.

## Schema

Node: `{"id", "type", "label", "level", "status", "priority", "refs", "file", "note"}`

| id prefix | type | example |
|---|---|---|
| `phen:` | phenomenon | `phen:madd:tabii`, `phen:sifah:hams`, `phen:makhraj:halq_aqsa`, `phen:waqf:raum`, `phen:timing:tasawi` |
| `math:` | math | `math:bures_wasserstein`, `math:persistent_homology` |
| `algo:` | algorithm | `algo:sindy_stlsq`, `algo:gop_ctc` |
| `pkg:` | package | `pkg:julia:DataDrivenDiffEq`, `pkg:octave:signal` |
| `model:` / `data:` | model / dataset | `model:hf:<repo>` |
| `src:` | source | `src:arXiv:1810.08278`, `src:doi:…` |
| `code:` | code | repo-relative path |

- `level` (phenomena): `101 | intermediate | advanced | ijazah`
- `status`: `implemented | partial | missing | proposed | speculative | not_observable`. When fragments
  merge, the most advanced status wins.
- `priority`: 1 (build first) to 5.
- Edge `rel`: `part_of, measured_by, models, measures, implements, implemented_in, requires, validates,
  cross_checks, generalizes, alternative_to, confusable_with, has_sifah, cites, trained_on`.

## Maintenance rule

When you implement something, flip its node to `implemented` in `fragments/core.json`, set `file`, and
add an `implemented_in` edge. Then rebuild. Keep reports in `experiments/deep_research/`; the graph only
points at them.
