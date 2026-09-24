"""The full-dataset engine pass on Modal CPU: every T300 verse graded, fanned out wide.

`datastore.reciter_profile clips` runs `Engine.analyze` on all 11,996 T300 clips -- about 1.4 s each,
four hours on the research box. The posteriors are already on the `qaari-runs` volume (they were
dumped there, muaalem/T300/<shard>/), and grading needs no model, only the engine's numpy path, so the
pass fans out over CPU containers that read the volume directly: 10 shards x 12 chunks = 120 tasks
of ~100 clips. Each returns the same compact per-verse records the local stage writes, and the
local entrypoint writes them to research_agency_lab/experiments/profiles/clips.jsonl and runs the
aggregation.

    modal run research_agency_lab/compute_bridge/modal_profile.py::smoke
    modal run research_agency_lab/compute_bridge/modal_profile.py::run [--chunks 12]
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

_here = Path(__file__).resolve()
ROOT = _here.parents[2] if len(_here.parents) > 2 and (_here.parents[2] / "app").is_dir() else Path("/root/qaari")

app = modal.App("qaari-profile")
vol = modal.Volume.from_name("qaari-runs")
SHARDS = [f"{i:04d}" for i in range(10)]
_IGNORE = ["**/__pycache__", "**/*.pyc"]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1")
    .pip_install("torch==2.14.0", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("transformers==5.17.0", "numpy==2.4.6", "librosa", "soundfile", "pyarrow",
                 "quran-transcript==0.6.1", "diff-match-patch", "rich", "pydantic")
    .pip_install("quran-muaalem==0.2.2", extra_options="--no-deps")
    .add_local_dir(str(ROOT / "app"), "/root/qaari/app", ignore=_IGNORE)
    .add_local_dir(str(ROOT / "datastore"), "/root/qaari/datastore", ignore=_IGNORE)
    .add_local_dir(str(ROOT / "research_agency_lab/experiments/learner_eval"),
                   "/root/qaari/research_agency_lab/experiments/learner_eval", ignore=_IGNORE + ["**/*.jsonl"])
    .add_local_dir(str(ROOT / "research_agency_lab/experiments/calibration"),
                   "/root/qaari/research_agency_lab/experiments/calibration", ignore=_IGNORE)
    .add_local_file(str(ROOT / "research_agency_lab/experiments/qaari_keys/sukoon_T300.json"),
                    "/root/qaari/research_agency_lab/experiments/qaari_keys/sukoon_T300.json")
    .add_local_file(str(ROOT / "research_agency_lab/experiments/qaari_keys/modal_T300/layout.json"),
                    "/root/qaari/layout.json")
)


@app.function(image=image, volumes={"/vol": vol}, cpu=1.0, memory=3072, timeout=1800, max_containers=90,
              retries=1)
def grade(shard: str, chunk: int, chunks: int) -> list[dict]:  # type: ignore[type-arg]
    import sys
    import warnings

    import numpy as np
    warnings.filterwarnings("ignore")
    sys.path.insert(0, "/root/qaari")
    from app.engine import Engine
    from datastore.reciter_profile import record

    lay = json.loads(Path("/root/qaari/layout.json").read_text())
    base = Path(f"/vol/muaalem/T300/{shard}")
    recs = [json.loads(line) for line in open(base / "index.jsonl")]
    recs = [r for r in recs if "file" in r][chunk::chunks]
    eng = Engine(layout=lay)
    out = []
    for r in recs:
        try:
            lp = np.fromfile(base / r["file"], dtype="<f4").reshape(r["frames"], lay["columns"])
            rep = eng.analyze(None, [(r["sura"], r["aya"])], posteriors=lp)
            out.append(record(rep, r["speaker"], "everyayah", r["sura"], r["aya"]))
        except Exception as exc:  # noqa: BLE001 - one bad clip must not lose the chunk
            out.append({"source": "everyayah", "speaker": r["speaker"], "surah": r["sura"], "ayah": r["aya"],
                        "error": repr(exc)})
    return out


@app.local_entrypoint()
def smoke() -> None:
    rows = grade.remote("0000", 0, 400)
    print(f"{len(rows)} records; first: {json.dumps(rows[0], ensure_ascii=False)[:300]}")


@app.local_entrypoint()
def run(chunks: int = 12) -> None:
    import time
    t = time.time()
    jobs = [(s, c, chunks) for s in SHARDS for c in range(chunks)]
    out = ROOT / "research_agency_lab/experiments/profiles/clips.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    n = err = 0
    with open(out, "w") as f:
        for rows in grade.starmap(jobs, order_outputs=False):
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
                err += "error" in r
    print(f"{n} verse records ({err} errors) from {len(jobs)} tasks in {time.time() - t:.0f} s -> {out}")
