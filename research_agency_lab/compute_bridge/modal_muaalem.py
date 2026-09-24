"""muaalem-v3.2 posteriors at scale on Modal GPUs — the same dump format as the local
``research_agency_lab/experiments/learner_eval/muaalem_dump.py`` (raw float32 ``<id>.f32``, ``layout.json``,
``index.jsonl``), so every Julia/Octave analysis runs unchanged on the result.

Work items are (EveryAyah folder, surah, ayah). ``launch`` splits a QaariKeys tier (or ``all``) over the
catalogued reciters into shards; each GPU container fetches its audio over HTTP, runs the model and
writes ``/out/muaalem/<tag>/<shard>/`` on the ``qaari-runs`` volume (resumable per clip). ``collect``
merges the shards into one local dump directory with ``modal volume get`` (no compute).

    modal run research_agency_lab/compute_bridge/modal_muaalem.py::launch --tag T10 --tier T10 --shards 4
    modal run research_agency_lab/compute_bridge/modal_muaalem.py::collect --tag T10 --dest research_agency_lab/experiments/qaari_keys/modal_T10
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

_here = Path(__file__).resolve()
ROOT = _here.parents[2] if len(_here.parents) > 2 and (_here.parents[2] / "datasets").is_dir() else Path("/root/qaari")
GPU = "T4"
MODEL = "obadx/muaalem-model-v3_2"

app = modal.App("qaari-muaalem")
vol = modal.Volume.from_name("qaari-runs", create_if_missing=True)


def _download() -> None:
    import transformers.models.wav2vec2_bert.modeling_wav2vec2_bert as w

    w._HIDDEN_STATES_START_POSITION = 2
    from quran_muaalem import Muaalem

    Muaalem(MODEL, device="cpu")
    from quran_transcript import Aya

    Aya(1, 1).get()


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1")
    .pip_install("torch==2.14.0")
    .pip_install("transformers==5.17.0", "numpy", "librosa", "soundfile", "pyarrow", "quran-transcript==0.6.1",
                 "diff-match-patch", "rich", "pydantic")
    .pip_install("quran-muaalem==0.2.2", extra_options="--no-deps")
    .run_function(_download)
    .add_local_file(str(ROOT / "research_agency_lab/experiments/learner_eval/muaalem_eval.py"), "/root/lab/muaalem_eval.py")
    .add_local_file(str(ROOT / "research_agency_lab/experiments/learner_eval/muaalem_dump.py"), "/root/lab/muaalem_dump.py")
)


# Measured on the T10 run: 278 GPU-s out of 760 container-s, so the GPU idles 63 % of the time while
# one thread downloads, decodes and phonetises. Modal bills per container-second and the total work
# is fixed, so more containers cut wall-clock almost linearly at the SAME price. cpu/memory are for
# the preprocessing half; PREFETCH overlaps it with inference.
@app.function(image=image, gpu=GPU, timeout=3 * 3600, volumes={"/out": vol}, max_containers=50,
              cpu=4.0, memory=8192, retries=1)
def dump_shard(tag: str, shard: int, items: list[tuple[str, int, int]]) -> dict[str, float]:
    import sys
    import tempfile
    import time
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import torch

    sys.path.insert(0, "/root/lab")
    import muaalem_dump as md  # applies the transformers shim via muaalem_eval
    from quran_transcript import Aya, quran_phonetizer

    PREFETCH = 12                      # clips fetched ahead of the GPU; ~1 MB of audio each

    out = Path(f"/out/muaalem/{tag}/{shard:04d}")
    out.mkdir(parents=True, exist_ok=True)
    index, layout = out / "index.jsonl", out / "layout.json"
    # only a record that actually wrote posteriors counts as done — errors must be retried
    done = {r["id"] for r in map(json.loads, index.open()) if "file" in r} if index.exists() else set()
    model = md.Muaalem(MODEL, device="cuda", dtype=torch.float32)
    t0, n, gpu_s = time.time(), 0, 0.0
    def prepare(item):  # runs on a worker thread: the 63 % that is not GPU
        folder, s, a = item
        url = f"https://everyayah.com/data/{folder}/{s:03d}{a:03d}.mp3"
        data = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "qaari"}),  # noqa: S310
                                      timeout=120).read()
        with tempfile.NamedTemporaryFile(suffix=".mp3") as tmp:
            tmp.write(data)
            tmp.flush()
            wave = md.load_16k(Path(tmp.name))
        uthmani = Aya(s, a).get().uthmani
        return wave, uthmani, quran_phonetizer(uthmani, md.MOSHAF, remove_spaces=True)

    todo = [it for it in items if f"everyayah:{it[0]}/{it[1]:03d}{it[2]:03d}" not in done]
    with index.open("a") as fh, ThreadPoolExecutor(max_workers=PREFETCH) as pool:
        pending: dict = {}
        for i, it in enumerate(todo[:PREFETCH]):
            pending[i] = pool.submit(prepare, it)
        for i, (folder, s, a) in enumerate(todo):
            nxt = i + PREFETCH
            if nxt < len(todo):
                pending[nxt] = pool.submit(prepare, todo[nxt])
            cid = f"everyayah:{folder}/{s:03d}{a:03d}"
            rec: dict[str, object] = {"id": cid, "source": "everyayah", "speaker": folder, "sura": s, "aya": a}
            try:
                wave, uthmani, ref = pending.pop(i).result()
                g0 = time.time()
                lp = md.posteriors(model, wave)
                gpu_s += time.time() - g0
                if not layout.exists():
                    md.write_layout(layout, model, lp)
                fname = cid.replace("/", "__").replace(":", "__") + ".f32"
                np.concatenate([lp[k] for k in sorted(lp, key=md._level_order)], axis=1).astype("<f4").tofile(out / fname)
                rec.update({"uthmani": uthmani, "ref_ph": ref.phonemes, "file": fname,
                            "frames": int(lp["phonemes"].shape[0]), "duration_s": round(len(wave) / 16000, 4),
                            "word_ph": md.word_spans(uthmani, ref.mappings)})
            except Exception as exc:  # noqa: BLE001
                rec["error"] = f"{type(exc).__name__}: {exc}"[:300]
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            fh.flush()
            n += 1
    vol.commit()
    return {"shard": shard, "clips": n, "wall_s": round(time.time() - t0, 1), "gpu_s": round(gpu_s, 1)}


def _items(tier: str, reciters: str, exclude: str = "") -> list[tuple[str, int, int]]:
    cat = json.loads((ROOT / "datasets/qaari_keys/build/reciters.json").read_text())
    folders = [r for r in reciters.split(",") if r] or [str(r["folder"]) for r in cat["everyayah"]]
    # a folder whose audio does not match its claimed verses poisons calibration; see
    # experiments/tajweed_coverage.md (free-decode vs reference edit distance per reciter)
    drop = {x for x in exclude.split(",") if x}
    folders = [f for f in folders if f not in drop]
    if tier.startswith("surah:"):          # a whole surah, e.g. surah:54
        sura = int(tier.split(":")[1])
        counts = json.loads((ROOT / "datasets/qaari_keys/build/ayah_keys.json").read_text())["ayahs"]
        verses = [(s, a) for s, a, *_ in counts if s == sura]
    elif tier == "all":
        counts = json.loads((ROOT / "datasets/qaari_keys/build/ayah_keys.json").read_text())["ayahs"]
        verses = [(s, a) for s, a, *_ in counts]
    else:
        t = next(x for x in json.loads((ROOT / "datasets/qaari_keys/build/tiers.json").read_text())["tiers"]
                 if x["name"] == tier)
        verses = [(v["surah"], v["ayah"]) for v in t["verses"]]
    return [(f, s, a) for f in folders for s, a in verses]


@app.local_entrypoint()
def launch(tag: str, tier: str = "T10", reciters: str = "", shards: int = 4, exclude: str = "") -> None:
    items = _items(tier, reciters, exclude)
    chunks = [items[i::shards] for i in range(shards)]
    print(f"{tag}: {len(items)} clips over {shards} {GPU} shards")
    total_gpu = 0.0
    for r in dump_shard.starmap([(tag, i, c) for i, c in enumerate(chunks)]):
        total_gpu += r["wall_s"]
        print(r)
    print(f"container wall total {total_gpu:.0f} s ≈ ${total_gpu * 0.000164 * 1.2:.2f} at T4 list price (+CPU/mem)")


@app.local_entrypoint()
def collect(tag: str, dest: str) -> None:
    """Merge /out/muaalem/<tag>/<shard>/ into DEST (layout.json, index.jsonl, *.f32)."""
    import shutil
    import subprocess
    import tempfile

    d = Path(dest)
    d.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["modal", "volume", "get", "qaari-runs", f"muaalem/{tag}", tmp, "--force"], check=True)
        root = next(Path(tmp).rglob("index.jsonl")).parents[1]
        seen = set()
        if (d / "index.jsonl").exists():
            seen = {json.loads(line)["id"] for line in (d / "index.jsonl").open()}
        n = 0
        with (d / "index.jsonl").open("a") as fh:
            for sh in sorted(p for p in root.iterdir() if p.is_dir()):
                if (sh / "layout.json").exists() and not (d / "layout.json").exists():
                    shutil.copy(sh / "layout.json", d / "layout.json")
                # a retried clip leaves an old error line and a later success line: keep the success
                best: dict[str, dict] = {}
                for line in (sh / "index.jsonl").open():
                    r = json.loads(line)
                    if "file" in r or r["id"] not in best:
                        best[r["id"]] = r
                for rec in best.values():
                    line = json.dumps(rec, ensure_ascii=False) + "\n"
                    if rec["id"] in seen:
                        continue
                    if "file" in rec:
                        shutil.move(str(sh / rec["file"]), d / rec["file"])
                    fh.write(line if line.endswith("\n") else line + "\n")
                    seen.add(rec["id"])
                    n += 1
    print(f"collected {n} clips -> {d}")
