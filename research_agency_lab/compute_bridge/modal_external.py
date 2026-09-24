"""External datasets through the engine on Modal: posteriors on GPU, grading on CPU.

A dataset is a directory of audio plus a manifest, `manifest.jsonl`, one line per clip:
{"id", "audio": <path relative to the directory>, "verses": [[surah, ayah(, first_word, last_word)], ...]}.

    # 1. put the audio and manifest on the qaari-runs volume (external/<name>/)
    modal volume put qaari-runs <local dir> external/<name>
    # 2. posteriors on GPU -> external/<name>/dump/<shard>/ (resumable per clip; the dump format of
    #    modal_muaalem.py, index lines also carry "audio" and "verses")
    modal run research_agency_lab/compute_bridge/modal_external.py::dump --name <name> --shards 8
    # 3. the engine on CPU containers reading those posteriors (and the audio, for stops); the
    #    reduced reports land in research_agency_lab/experiments/external/<name>_engine.jsonl,
    #    the same records runner.py writes, so every comparison reads either
    modal run research_agency_lab/compute_bridge/modal_external.py::grade --name <name> --chunks 40
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

_here = Path(__file__).resolve()
ROOT = _here.parents[2] if len(_here.parents) > 2 and (_here.parents[2] / "app").is_dir() else Path("/root/qaari")
GPU = "T4"

# Quran Whisper models that transcribe a submission so its verses can be located (app/verse_locate.py):
# tarteel's base model is the widely used baseline; the large-v3-turbo fine-tune is the challenger
ASR_MODELS = {"base": "tarteel-ai/whisper-base-ar-quran", "turbo": "naazimsnh02/whisper-large-v3-turbo-ar-quran"}

app = modal.App("qaari-external")
vol = modal.Volume.from_name("qaari-runs")
_IGNORE = ["**/__pycache__", "**/*.pyc", "**/*.jsonl", "**/*.f32"]


def _warm() -> None:
    import transformers.models.wav2vec2_bert.modeling_wav2vec2_bert as w

    w._HIDDEN_STATES_START_POSITION = 2
    from quran_muaalem import Muaalem

    Muaalem("obadx/muaalem-model-v3_2", device="cpu")
    from quran_transcript import Aya

    Aya(1, 1).get()
    from transformers import pipeline
    for m in ASR_MODELS.values():
        pipeline("automatic-speech-recognition", model=m, device=-1)


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1", "ffmpeg")
    .pip_install("torch==2.14.0")
    .pip_install("transformers==5.17.0", "numpy==2.4.6", "scipy", "librosa", "soundfile", "pyarrow",
                 "quran-transcript==0.6.1", "diff-match-patch", "rich", "pydantic")
    .pip_install("quran-muaalem==0.2.2", extra_options="--no-deps")
    .pip_install("accelerate")
    .run_function(_warm)
    .add_local_dir(str(ROOT / "app"), "/root/qaari/app", ignore=_IGNORE)
    .add_local_dir(str(ROOT / "datastore"), "/root/qaari/datastore", ignore=_IGNORE)
    .add_local_dir(str(ROOT / "research_agency_lab/experiments/learner_eval"),
                   "/root/qaari/research_agency_lab/experiments/learner_eval", ignore=_IGNORE)
    .add_local_dir(str(ROOT / "research_agency_lab/experiments/calibration"),
                   "/root/qaari/research_agency_lab/experiments/calibration", ignore=_IGNORE)
    .add_local_file(str(ROOT / "research_agency_lab/experiments/qaari_keys/sukoon_T300.json"),
                    "/root/qaari/research_agency_lab/experiments/qaari_keys/sukoon_T300.json")
    .add_local_file(str(ROOT / "research_agency_lab/experiments/quran/reference_stats.json"),
                    "/root/qaari/research_agency_lab/experiments/quran/reference_stats.json")
)


def _wave(path: Path):  # type: ignore[no-untyped-def]
    import sys
    sys.path.insert(0, "/root/qaari/research_agency_lab/experiments/learner_eval")
    import muaalem_dump as md
    return md.load_16k(path)


@app.function(image=image, gpu=GPU, timeout=3 * 3600, volumes={"/vol": vol}, max_containers=20,
              cpu=4.0, memory=8192, retries=1)
def dump_shard(name: str, shard: int, items: list[dict]) -> dict:  # type: ignore[type-arg]
    import sys
    import time
    from concurrent.futures import ThreadPoolExecutor

    sys.path.insert(0, "/root/qaari")
    from app.engine import Engine

    base = Path(f"/vol/external/{name}")
    out = base / "dump" / f"{shard:04d}"
    out.mkdir(parents=True, exist_ok=True)
    index, layout = out / "index.jsonl", out / "layout.json"
    done = {r["id"] for r in map(json.loads, index.open()) if "file" in r} if index.exists() else set()
    todo = [it for it in items if it["id"] not in done]
    eng = Engine()
    t0, gpu_s, n = time.time(), 0.0, 0
    with index.open("a") as fh, ThreadPoolExecutor(max_workers=4) as pool:
        waves = pool.map(lambda it: _safe_wave(base / it["audio"]), todo)
        for it, wave in zip(todo, waves):
            rec = {"id": it["id"], "audio": it["audio"], "verses": it["verses"]}
            try:
                if isinstance(wave, Exception):
                    raise wave
                g = time.time()
                lp = eng.posteriors(wave)
                gpu_s += time.time() - g
                if not layout.exists():
                    layout.write_text(json.dumps(eng._layout, ensure_ascii=False))
                fname = it["id"].replace("/", "__").replace(":", "__") + ".f32"
                lp.astype("<f4").tofile(out / fname)
                rec.update({"file": fname, "frames": int(lp.shape[0]), "duration_s": round(wave.size / 16000, 4)})
            except Exception as exc:  # noqa: BLE001
                rec["error"] = f"{type(exc).__name__}: {exc}"[:300]
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            if n % 200 == 0:
                fh.flush()
                vol.commit()
    vol.commit()
    return {"shard": shard, "clips": n, "wall_s": round(time.time() - t0, 1), "gpu_s": round(gpu_s, 1)}


def _safe_wave(p: Path):  # type: ignore[no-untyped-def]
    try:
        return _wave(p)
    except Exception as exc:  # noqa: BLE001
        return exc


@app.local_entrypoint()
def dump(name: str, shards: int = 8, manifest: str = "") -> None:
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        if manifest:
            items = [json.loads(l) for l in open(manifest)]
        else:
            subprocess.run(["modal", "volume", "get", "qaari-runs", f"external/{name}/manifest.jsonl", tmp],
                           check=True, capture_output=True)
            items = [json.loads(l) for l in open(Path(tmp) / "manifest.jsonl")]
    chunks = [items[i::shards] for i in range(shards)]
    print(f"{name}: {len(items)} clips over {shards} {GPU} shards")
    wall = 0.0
    for r in dump_shard.starmap([(name, i, c) for i, c in enumerate(chunks)]):
        wall += r["wall_s"]
        print(r)
    print(f"container wall total {wall:.0f} s")


@app.function(image=image, volumes={"/vol": vol}, cpu=1.0, memory=3072, timeout=3600, max_containers=80,
              retries=1)
def grade_chunk(name: str, shard: str, chunk: int, chunks: int) -> list[dict]:  # type: ignore[type-arg]
    import sys
    import time
    import warnings

    import numpy as np
    warnings.filterwarnings("ignore")
    sys.path.insert(0, "/root/qaari")
    from app.engine import Engine
    from datastore.reciter_profile import record

    base = Path(f"/vol/external/{name}")
    d = base / "dump" / shard
    lay = json.loads((d / "layout.json").read_text())
    recs = [r for r in map(json.loads, (d / "index.jsonl").open()) if "file" in r]
    best = {r["id"]: r for r in recs}          # a retried clip appears twice; one record each
    recs = sorted(best.values(), key=lambda r: r["id"])[chunk::chunks]
    eng = Engine(layout=lay)
    out = []
    for r in recs:
        t = time.time()
        try:
            lp = np.fromfile(d / r["file"], dtype="<f4").reshape(r["frames"], lay["columns"])
            w = _wave(base / r["audio"])[: r["frames"] * 640]
            wave = np.pad(w, (0, r["frames"] * 640 - w.size)).astype("float32")
            rep = eng.analyze(wave, [tuple(v) for v in r["verses"]], posteriors=lp)
            out.append({"id": r["id"], "verses": r["verses"], "measurements": rep["measurements"],
                        "ghunnah": [g for a in rep.get("ayahs", []) for g in a.get("ghunnah", [])],
                        "summary": rep.get("summary"), "audio_seconds": r["duration_s"],
                        # the per-verse skill record the cohort model is fitted from, same function
                        "record": record(rep, r["id"], name, *r["verses"][0][:2]),
                        "grade_seconds": round(time.time() - t, 3)})
        except Exception as exc:  # noqa: BLE001 - one bad clip must not lose the chunk
            out.append({"id": r["id"], "verses": r["verses"], "error": f"{type(exc).__name__}: {exc}"[:300]})
    return out


@app.local_entrypoint()
def grade(name: str, chunks: int = 20) -> None:
    import subprocess
    import time
    t = time.time()
    ls = subprocess.run(["modal", "volume", "ls", "qaari-runs", f"external/{name}/dump"], check=True,
                        capture_output=True, text=True).stdout.split()
    shards = sorted({Path(x).name for x in ls if Path(x).name.isdigit()})
    jobs = [(name, s, c, chunks) for s in shards for c in range(chunks)]
    out = ROOT / f"research_agency_lab/experiments/external/{name}_engine.jsonl"
    n = err = 0
    with open(out, "w") as f:
        for rows in grade_chunk.starmap(jobs, order_outputs=False):
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
                err += "error" in r
    print(f"{n} clips ({err} errors) from {len(jobs)} tasks in {time.time() - t:.0f} s -> {out}")


# ------------------------------------------------------------------ submissions: verses unknown
# Every clip is treated as a consumer app would receive it: audio only. Whisper transcribes it, the
# transcript is located in the Quran (app/verse_locate.py), and the engine grades the located span.
# A dataset's own verse, when it has one, is kept only to score the identification.

@app.function(image=image, gpu=GPU, timeout=4 * 3600, volumes={"/vol": vol}, max_containers=20,
              cpu=4.0, memory=12288, retries=1)
def submit_shard(name: str, shard: int, items: list[dict]) -> dict:  # type: ignore[type-arg]
    import sys
    import time
    from concurrent.futures import ThreadPoolExecutor

    import torch
    from transformers import pipeline

    sys.path.insert(0, "/root/qaari")
    from app.engine import Engine

    base = Path(f"/vol/external/{name}")
    have = {}                                   # posteriors already dumped for this dataset
    for idx in (base / "dump").glob("*/index.jsonl"):
        for r in map(json.loads, idx.open()):
            if "file" in r:
                have[r["id"]] = {"posteriors": str((idx.parent / r["file"]).relative_to(base)),
                                 "layout": str((idx.parent / "layout.json").relative_to(base)),
                                 "frames": r["frames"]}
    out = base / "sub" / f"{shard:04d}"
    out.mkdir(parents=True, exist_ok=True)
    index, layout = out / "index.jsonl", out / "layout.json"
    done = {r["id"] for r in map(json.loads, index.open()) if "transcripts" in r} if index.exists() else set()
    todo = [it for it in items if it["id"] not in done]
    asr = {k: pipeline("automatic-speech-recognition", model=m, device=0, torch_dtype=torch.float16)
           for k, m in ASR_MODELS.items()}
    eng = Engine()
    t0, n = time.time(), 0
    with index.open("a") as fh, ThreadPoolExecutor(max_workers=4) as pool:
        waves = pool.map(lambda it: _safe_wave(base / it["audio"]), todo)
        for it, wave in zip(todo, waves):
            rec = {k: it[k] for k in ("id", "audio")}
            try:
                if isinstance(wave, Exception):
                    raise wave
                long = wave.size > 30 * 16000
                rec["transcripts"] = {
                    k: str(p(wave.copy(), generate_kwargs={"num_beams": 1, "language": "ar", "task": "transcribe"}
                             if k == "turbo" else {"num_beams": 1},
                             **({"chunk_length_s": 30} if long else {})).get("text", "")).strip()
                    for k, p in asr.items()}
                rec["duration_s"] = round(wave.size / 16000, 4)
                if it["id"] in have:
                    rec.update(have[it["id"]])
                else:
                    lp = eng.posteriors(wave)
                    if not layout.exists():
                        layout.write_text(json.dumps(eng._layout, ensure_ascii=False))
                    fname = it["id"].replace("/", "__").replace(":", "__") + ".f32"
                    lp.astype("<f4").tofile(out / fname)
                    rec.update({"posteriors": str((out / fname).relative_to(base)),
                                "layout": str(layout.relative_to(base)), "frames": int(lp.shape[0])})
            except Exception as exc:  # noqa: BLE001
                rec["error"] = f"{type(exc).__name__}: {exc}"[:300]
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            if n % 200 == 0:
                fh.flush()
                vol.commit()
    vol.commit()
    return {"shard": shard, "clips": n, "wall_s": round(time.time() - t0, 1)}


@app.local_entrypoint()
def submit(name: str, shards: int = 8, limit: int = 0) -> None:
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["modal", "volume", "get", "qaari-runs", f"external/{name}/submissions.jsonl", tmp],
                       check=True, capture_output=True)
        items = [json.loads(l) for l in open(Path(tmp) / "submissions.jsonl")]
    items = items[:limit] if limit else items
    chunks = [items[i::shards] for i in range(shards)]
    print(f"{name}: {len(items)} submissions over {shards} {GPU} shards")
    wall = 0.0
    for r in submit_shard.starmap([(name, i, c) for i, c in enumerate(chunks)]):
        wall += r["wall_s"]
        print(r)
    print(f"container wall total {wall:.0f} s")


@app.function(image=image, volumes={"/vol": vol}, cpu=1.0, memory=3072, timeout=3600, max_containers=100,
              retries=1)
def grade_sub_chunk(name: str, shard: str, chunk: int, chunks: int, asr: str) -> list[dict]:  # type: ignore[type-arg]
    import sys
    import time
    import warnings

    import numpy as np
    warnings.filterwarnings("ignore")
    sys.path.insert(0, "/root/qaari")
    from app.engine import Engine
    from app.verse_locate import locate
    from datastore.reciter_profile import record

    base = Path(f"/vol/external/{name}")
    d = base / "sub" / shard
    recs = [r for r in map(json.loads, (d / "index.jsonl").open()) if "transcripts" in r]
    recs = sorted({r["id"]: r for r in recs}.values(), key=lambda r: r["id"])[chunk::chunks]
    engines: dict[str, Engine] = {}
    out = []
    for r in recs:
        t = time.time()
        row = {"id": r["id"], "transcripts": r["transcripts"], "audio_seconds": r["duration_s"]}
        try:
            locs = {k: locate(v) for k, v in r["transcripts"].items()}
            row["located"] = {k: (l.to_dict() if l else None) for k, l in locs.items()}
            loc = locs.get(asr)
            if loc is None:
                raise ValueError("no transcript to locate")
            if r["layout"] not in engines:
                engines[r["layout"]] = Engine(layout=json.loads((base / r["layout"]).read_text()))
            eng = engines[r["layout"]]
            lay = eng._layout
            lp = np.fromfile(base / r["posteriors"], dtype="<f4").reshape(r["frames"], lay["columns"])
            w = _wave(base / r["audio"])[: r["frames"] * 640]
            wave = np.pad(w, (0, r["frames"] * 640 - w.size)).astype("float32")
            rep = eng.analyze(wave, loc.verses, posteriors=lp)
            row.update({"verses": [list(v) for v in loc.verses], "measurements": rep["measurements"],
                        "ghunnah": [g for a in rep.get("ayahs", []) for g in a.get("ghunnah", [])],
                        "summary": rep.get("summary"),
                        "record": record(rep, r["id"], name, *loc.verses[0][:2]),
                        "grade_seconds": round(time.time() - t, 3)})
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"[:300]
        out.append(row)
    return out


@app.local_entrypoint()
def grade_sub(name: str, chunks: int = 20, asr: str = "turbo") -> None:
    import subprocess
    import time
    t = time.time()
    ls = subprocess.run(["modal", "volume", "ls", "qaari-runs", f"external/{name}/sub"], check=True,
                        capture_output=True, text=True).stdout.split()
    shards = sorted({Path(x).name for x in ls if Path(x).name.isdigit()})
    jobs = [(name, s, c, chunks, asr) for s in shards for c in range(chunks)]
    out = ROOT / f"research_agency_lab/experiments/external/{name}_submissions_{asr}.jsonl"
    n = err = 0
    with open(out, "w") as f:
        for rows in grade_sub_chunk.starmap(jobs, order_outputs=False):
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
                err += "error" in r
    print(f"{n} submissions ({err} errors) from {len(jobs)} tasks in {time.time() - t:.0f} s -> {out}")
