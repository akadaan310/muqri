#!/usr/bin/env python3
"""Build the masterclass (studio) and Taraweeh reciter indices from ayah-by-ayah datasets.

This runs the full v2 pipeline on every reciter of ``benchmarks/roster.py`` (or ``--reciters``)
over a verse set and writes the 232-d fingerprints into ``index/masterclass_reciters.faiss`` and
``index/taraweeh_reciters.faiss``:

    python datasets/index_reciters.py                      # strategic verse set (59 ayahs)
    python datasets/index_reciters.py --verses all --shard 3/8   # entire Qur'an, one shard

Entire-dataset runs are meant for a GPU notebook (``notebooks/kaggle_dataset_indexer.ipynb``):
shards write separate JSONL files that ``benchmarks/summarize.py`` merges into the indices.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks import run_benchmark, summarize  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--modes" not in args:
        args += ["--modes", "studio"]  # fingerprints come from studio-mode analysis
    code = run_benchmark.main(args)
    if code:
        return code
    return summarize.main([])


if __name__ == "__main__":
    sys.exit(main())
