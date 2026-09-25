"""The letter corpus's measuring pass on Modal: the planned ayahs x reciters, the engine with the makhraj
test, results written into the same local files measure.py writes (letter_corpus/data/measure/).

The account runs at most 10 containers, so each is wide: 16 CPUs, 32 GiB, and `WORKERS` engine
processes inside it (the model baked into the image, torch on 2 threads each). Jobs already measured
locally are skipped before launch.

    modal run research_agency_lab/compute_bridge/modal_corpus.py::run \\
        --reciters "Husary_128kbps,Minshawy_Murattal_128kbps,Abdul_Basit_Murattal_192kbps,Alafasy_128kbps,Abdurrahmaan_As-Sudais_192kbps"
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

_here = Path(__file__).resolve()
ROOT = _here.parents[2] if len(_here.parents) > 2 and (_here.parents[2] / "app").is_dir() else Path("/root/qaari")
CONTAINERS = 10
WORKERS = 8
MODEL = "obadx/muaalem-model-v3_2"
_IGNORE = ["**/__pycache__", "**/*.pyc"]

app = modal.App("qaari-corpus")


def _download() -> None:
    import transformers.models.wav2vec2_bert.modeling_wav2vec2_bert as w

    w._HIDDEN_STATES_START_POSITION = 2
    from quran_muaalem import Muaalem

    Muaalem(MODEL, device="cpu")
    from quran_transcript import Aya

    Aya(1, 1).get()


_files = ["research_agency_lab/experiments/quran/reference_stats.json",
          "research_agency_lab/experiments/quran/blindspots.json",
          "research_agency_lab/experiments/timing/stretch_model.json",
          "research_agency_lab/experiments/qaari_keys/sukoon_T300.json"]
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1", "ffmpeg")
    .pip_install("torch==2.14.0", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("transformers==5.17.0", "numpy==2.4.6", "scipy==1.17.1", "librosa==0.11.0", "soundfile",
                 "pyarrow", "quran-transcript==0.6.1", "diff-match-patch", "rich", "pydantic")
    .pip_install("quran-muaalem==0.2.2", extra_options="--no-deps")
    .run_function(_download)
    .add_local_dir(str(ROOT / "app"), "/root/qaari/app", ignore=_IGNORE)
    .add_local_dir(str(ROOT / "datastore"), "/root/qaari/datastore", ignore=_IGNORE)
    .add_local_dir(str(ROOT / "research_agency_lab/experiments/learner_eval"),
                   "/root/qaari/research_agency_lab/experiments/learner_eval", ignore=_IGNORE + ["**/*.jsonl"])
    .add_local_dir(str(ROOT / "research_agency_lab/experiments/calibration"),
                   "/root/qaari/research_agency_lab/experiments/calibration", ignore=_IGNORE)
)
for f in _files:
    image = image.add_local_file(str(ROOT / f), f"/root/qaari/{f}")

_ENG = None


def _one(job: tuple[str, int, int]) -> dict:  # type: ignore[type-arg]
    import io
    import sys
    import urllib.request
    import warnings

    warnings.filterwarnings("ignore")
    sys.path.insert(0, "/root/qaari")
    global _ENG
    import torch
    torch.set_num_threads(2)
    if _ENG is None:
        from app.engine import Engine
        _ENG = Engine()
    import librosa
    rec, s, a = job
    url = f"https://everyayah.com/data/{rec}/{s:03d}{a:03d}.mp3"
    try:
        data = None
        for _ in range(3):
            try:
                data = urllib.request.urlopen(url, timeout=60).read()
                break
            except Exception:  # noqa: BLE001 - retry the download
                continue
        if not data:
            return {"reciter": rec, "surah": s, "ayah": a, "error": "no audio"}
        wave = librosa.load(io.BytesIO(data), sr=16000, mono=True)[0]
        rep = _ENG.analyze(wave, [(s, a)], makhraj=True)
        return {"reciter": rec, "surah": s, "ayah": a, "measurements": rep["measurements"]}
    except Exception as e:  # noqa: BLE001 - one bad ayah must not lose the chunk
        return {"reciter": rec, "surah": s, "ayah": a, "error": repr(e)[:300]}


@app.function(image=image, cpu=16.0, memory=32768, timeout=3600, max_containers=CONTAINERS, retries=0)
def measure(jobs: list[tuple[str, int, int]]) -> list[dict]:  # type: ignore[type-arg]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(WORKERS) as ex:
        return list(ex.map(_one, jobs, chunksize=2))


@app.local_entrypoint()
def run(reciters: str) -> None:
    import time
    t = time.time()
    data = ROOT / "research_agency_lab/experiments/letter_corpus/data/measure"
    plan = json.loads((ROOT / "research_agency_lab/experiments/letter_corpus/plan.json").read_text())
    jobs = [(r, x["surah"], x["ayah"]) for r in reciters.split(",") for x in plan["chosen"]
            if not (data / r / f"{x['surah']:03d}{x['ayah']:03d}.json").is_file()]
    print(f"{len(jobs)} ayahs to measure on {CONTAINERS} containers x {WORKERS} workers")
    chunks = [jobs[i::CONTAINERS] for i in range(CONTAINERS)]
    n = err = 0
    for rows in measure.map(chunks, order_outputs=False):
        for r in rows:
            if "error" in r:
                err += 1
                print("error", r["reciter"], r["surah"], r["ayah"], r["error"][:120])
                continue
            out = data / r["reciter"] / f"{r['surah']:03d}{r['ayah']:03d}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(r, ensure_ascii=False))       # audio: the EveryAyah cache (assemble.py)
            n += 1
        print(f"{n} written, {err} errors, {time.time() - t:.0f} s", flush=True)
