"""Modal Labs fan-out batch runner for qaari-eval benchmark collection.

Unlike ``kaggle_bridge.py`` (5 free 12 h CPU sessions), this uses Modal's serverless autoscaling to
run *many* containers at once, so a full-Qur'an collection finishes in hours, not days. The per-clip
cost is CPU-bound (Praat formants, Hilbert vowel core, WPE dereverb), so CPU containers are both the
cheapest and — fanned out wide — the fastest; a ``--gpu`` path is offered for the wav2vec2 aligner.

The image bakes in the wav2vec2 CTC aligner + ECAPA weights and the full Uthmani text cache, so a
container only pulls its shard's EveryAyah audio over HTTP. Each shard runs ``benchmarks/run_benchmark.py
--shard i/n`` and appends ``runs_*.jsonl`` to a Modal Volume (resumable: finished rows are skipped).

    modal run   research_agency_lab/compute_bridge/modal_batch.py::smoke
    modal run   research_agency_lab/compute_bridge/modal_batch.py::launch \
        --tag husary-all --reciters "Husary_128kbps,Husary_Muallim_128kbps" --verses all \
        --modes studio --shards 60
    modal run   research_agency_lab/compute_bridge/modal_batch.py::collect --tag husary-all
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import modal

# Client-side this file lives at research_agency_lab/compute_bridge/modal_batch.py; inside a Modal
# container the entrypoint is remounted at /root/modal_batch.py (no repo parents), so fall back to
# the baked-in copy path when the usual parents[2] does not resolve to the repo root.
_here = Path(__file__).resolve()
if len(_here.parents) > 2 and (_here.parents[2] / "app").is_dir():
    ROOT = _here.parents[2]
else:
    ROOT = Path("/root/qaari")

app = modal.App("qaari-batch")
vol = modal.Volume.from_name("qaari-runs", create_if_missing=True)
VOL_ROOT = "/out"

# Cache the HF/torch model files under a directory baked into the image at build time.
_IGNORE = ["**/.git", "**/.venv", "**/__pycache__", "**/*.pyc", "**/node_modules",
           "benchmarks/results/**", "notebooks/**", "**/*.jsonl"]


def _download_models() -> None:
    """Runs at image build so containers start with the weights already on local disk."""
    from speechbrain.inference.speaker import EncoderClassifier
    from transformers import AutoModelForCTC, AutoProcessor

    from app.aligner import DEFAULT_CTC_MODEL

    AutoProcessor.from_pretrained(DEFAULT_CTC_MODEL)
    AutoModelForCTC.from_pretrained(DEFAULT_CTC_MODEL)
    EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                   savedir="/root/.cache/qaari-eval/ecapa")


def _base(torch_index: str) -> modal.Image:
    img = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("ffmpeg", "libsndfile1")
        # Install torch first from the requested wheel index; then the rest WITHOUT re-listing torch
        # (requirements-ml.txt pins torch and would drag in the default CUDA build over a CPU image).
        .pip_install("torch", "torchaudio", index_url=torch_index)
        .pip_install_from_requirements(str(ROOT / "requirements.txt"))
        .pip_install("transformers>=4.40", "speechbrain>=1.0")
        # NB: do not set HF_HUB_OFFLINE here — it would block the build-time model download in
        # _download_models. The weights are baked into the cache, so runtime loads never hit the net.
        .env({"PYTHONPATH": "/root/qaari", "QAARI_CACHE_DIR": "/root/.cache/qaari-eval"})
        .workdir("/root/qaari")
        .add_local_dir(str(ROOT), "/root/qaari", copy=True, ignore=_IGNORE)
    )
    # Bake the pre-built full Uthmani text cache so no container hits the alquran.cloud API.
    text = Path.home() / ".cache" / "qaari-eval" / "text" / "quran-uthmani.json"
    if text.exists():
        img = img.add_local_file(str(text), "/root/.cache/qaari-eval/text/quran-uthmani.json", copy=True)
    return img.run_function(_download_models)


CPU_IMAGE = _base("https://download.pytorch.org/whl/cpu")
GPU_IMAGE = _base("https://download.pytorch.org/whl/cu121")


def _run_shard(tag: str, reciters: list[str], verses: str, modes: list[str],
               shard_i: int, shard_n: int, threads: int) -> dict:
    """Body shared by the CPU and GPU functions: run one shard, append to the Volume."""
    import time

    out_dir = Path(VOL_ROOT) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"runs_{shard_i:04d}.jsonl"
    cache = Path("/tmp/everyayah")  # ephemeral per-container audio cache
    cmd = [sys.executable, "/root/qaari/benchmarks/run_benchmark.py",
           "--reciters", *reciters, "--verses", verses, "--modes", *modes,
           "--shard", f"{shard_i}/{shard_n}", "--output", str(out),
           "--cache-dir", str(cache), "--threads", str(threads)]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd="/root/qaari", capture_output=True, text=True)
    rows = sum(1 for _ in open(out)) if out.exists() else 0
    vol.commit()
    tail = (proc.stdout + proc.stderr).splitlines()[-15:]
    return {"shard": shard_i, "rc": proc.returncode, "rows": rows, "secs": round(time.time() - t0, 1),
            "tail": tail}


# Free-tier container ceiling is 100; keep CPU fan-out under it with headroom for retries.
@app.function(image=CPU_IMAGE, volumes={VOL_ROOT: vol}, cpu=2.0, memory=6144, timeout=6 * 3600,
              max_containers=90, retries=2)
def process_shard_cpu(tag, reciters, verses, modes, shard_i, shard_n, threads=2):  # type: ignore[no-untyped-def]
    return _run_shard(tag, reciters, verses, modes, shard_i, shard_n, threads)


@app.function(image=GPU_IMAGE, gpu="T4", volumes={VOL_ROOT: vol}, cpu=2.0, memory=6144, timeout=6 * 3600,
              max_containers=10, retries=2)
def process_shard_gpu(tag, reciters, verses, modes, shard_i, shard_n, threads=2):  # type: ignore[no-untyped-def]
    return _run_shard(tag, reciters, verses, modes, shard_i, shard_n, threads)


@app.function(image=CPU_IMAGE, volumes={VOL_ROOT: vol}, timeout=24 * 3600)
def orchestrate(tag, reciters, verses, modes, shards, gpu=False, threads=2):  # type: ignore[no-untyped-def]
    """Server-side driver: runs the whole shard fan-out ON Modal so it survives client disconnects.

    Deploy the app then ``spawn`` this (see the `fire` entrypoint); the fan-out then completes on
    Modal's infrastructure regardless of the local machine. Results land in the qaari-runs Volume.
    """
    fn = process_shard_gpu if gpu else process_shard_cpu
    argv = [(tag, reciters, verses, modes, i, shards, threads) for i in range(shards)]
    ok, total = 0, 0
    per_shard = []
    for res in fn.starmap(argv):
        total += res["rows"]
        ok += res["rc"] == 0
        per_shard.append({"shard": res["shard"], "rc": res["rc"], "rows": res["rows"], "secs": res["secs"]})
    return {"tag": tag, "shards": shards, "ok": ok, "rows": total, "per_shard": per_shard}


@app.local_entrypoint()
def fire(tag: str, reciters: str, verses: str = "all", modes: str = "studio",
         shards: int = 150, gpu: bool = False, threads: int = 2) -> None:
    """Spawn the server-side orchestrator and return immediately (use after ``modal deploy``)."""
    rec = [r.strip() for r in reciters.split(",") if r.strip()]
    mod = [m.strip() for m in modes.split(",") if m.strip()]
    call = orchestrate.spawn(tag, rec, verses, mod, shards, gpu, threads)
    print(f"spawned orchestrate for tag={tag}: call_id={call.object_id}")


@app.local_entrypoint()
def launch(tag: str, reciters: str, verses: str = "all", modes: str = "studio",
           shards: int = 60, gpu: bool = False, threads: int = 2) -> None:
    """Fan a collection out over ``shards`` containers and print a per-shard summary."""
    rec = [r.strip() for r in reciters.split(",") if r.strip()]
    mod = [m.strip() for m in modes.split(",") if m.strip()]
    fn = process_shard_gpu if gpu else process_shard_cpu
    argv = [(tag, rec, verses, mod, i, shards, threads) for i in range(shards)]
    total_rows = 0
    ok = 0
    for res in fn.starmap(argv):
        total_rows += res["rows"]
        ok += res["rc"] == 0
        flag = "ok" if res["rc"] == 0 else f"RC={res['rc']}"
        print(f"shard {res['shard']:>4}/{shards}  {flag}  rows={res['rows']:<5} {res['secs']}s")
        if res["rc"] != 0:
            print("   " + "\n   ".join(res["tail"]))
    print(f"\n{ok}/{shards} shards ok, {total_rows} rows into volume qaari-runs:/{tag}")


@app.function(image=CPU_IMAGE, volumes={VOL_ROOT: vol}, timeout=1200)
def _gather(tag: str) -> dict:
    import json as _json

    d = Path(VOL_ROOT) / tag
    files = sorted(d.glob("runs_*.jsonl")) if d.exists() else []
    payload = {f.name: f.read_text(encoding="utf-8") for f in files}
    rows = sum(len(v.splitlines()) for v in payload.values())
    return {"files": payload, "rows": rows, "count": len(files),
            "reciters": sorted({_json.loads(l)["reciter"] for v in payload.values()
                                for l in v.splitlines() if l.strip()})}


@app.local_entrypoint()
def collect(tag: str) -> None:
    """Download all shard jsonl for ``tag`` from the Volume into benchmarks/results/modal/<tag>/."""
    res = _gather.remote(tag)
    dest = ROOT / "benchmarks" / "results" / "modal" / tag
    dest.mkdir(parents=True, exist_ok=True)
    for name, text in res["files"].items():
        (dest / name).write_text(text, encoding="utf-8")
    print(f"collected {res['count']} files, {res['rows']} rows, reciters={res['reciters']}")
    print(f"-> {dest}")


@app.local_entrypoint()
def smoke() -> None:
    """Tiny end-to-end check: Al-Fatiha, Husary + Dosari, both modes, 2 shards."""
    res = list(process_shard_cpu.starmap(
        [("smoke", ["Husary_128kbps", "Yasser_Ad-Dussary_128kbps"], "1:1-7",
          ["studio", "taraweeh_adapted"], i, 2, 2) for i in range(2)]))
    for r in res:
        print(f"shard {r['shard']} rc={r['rc']} rows={r['rows']} {r['secs']}s")
        if r["rc"] != 0:
            print("\n".join(r["tail"]))
