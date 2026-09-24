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


def _audio(r: dict):  # type: ignore[no-untyped-def,type-arg]
    """The verse's EveryAyah recording at 16 kHz, cut to the posteriors' frames (40 ms = 640 samples)."""
    import io
    import urllib.request

    import librosa
    import numpy as np
    url = f"https://everyayah.com/data/{r['speaker']}/{r['sura']:03d}{r['aya']:03d}.mp3"
    for _ in range(3):
        try:
            data = urllib.request.urlopen(url, timeout=60).read()
            w = librosa.load(io.BytesIO(data), sr=16000, mono=True)[0][: r["frames"] * 640]
            return np.pad(w, (0, r["frames"] * 640 - w.size)).astype("float32")
        except Exception:  # noqa: BLE001 - retry, then grade without stops rather than lose the verse
            continue
    return None


@app.function(image=image, volumes={"/vol": vol}, cpu=1.0, memory=3072, timeout=1800, max_containers=90,
              retries=1)
def grade(shard: str, chunk: int, chunks: int, audio: bool = True) -> list[dict]:  # type: ignore[type-arg]
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
            wave = _audio(r) if audio else None       # stops are acoustic: without audio none are seen
            rep = eng.analyze(wave, [(r["sura"], r["aya"])], posteriors=lp)
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
def run(chunks: int = 12, audio: bool = True) -> None:
    import time
    t = time.time()
    jobs = [(s, c, chunks, audio) for s in SHARDS for c in range(chunks)]
    out = ROOT / f"research_agency_lab/experiments/profiles/clips{'' if audio else '_noaudio'}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    n = err = 0
    with open(out, "w") as f:
        for rows in grade.starmap(jobs, order_outputs=False):
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
                err += "error" in r
    print(f"{n} verse records ({err} errors) from {len(jobs)} tasks in {time.time() - t:.0f} s -> {out}")


# ------------------------------------------------------------------ the Quran-wide inventory
@app.function(image=image, cpu=1.0, memory=3072, timeout=1800, max_containers=60, retries=1)
def inventory(ayahs: list[tuple[int, int]]) -> list[dict]:  # type: ignore[type-arg]
    """Every located rule and every consonant-in-context of each ayah, by the grader's own binder."""
    import sys
    import warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, "/root/qaari")
    from app.engine import Engine
    from app.mudud import resolve
    from app.rule_bind import bind, ph_units

    eng = Engine(layout={"columns": 0, "blank": 0, "levels": {}})
    vowels, madd = set("َُِ"), set("اۥۦ")
    out = []
    for s, a in ayahs:
        try:
            r = eng.reference(s, a)
            bounds, _ = resolve(bind(eng.parser.parse(r.uthmani), r.phonemes, r.word_ph))
            units = ph_units(r.phonemes)
            letters = []
            for i, (sym, x, y) in enumerate(units):
                if sym in vowels or sym in madd:
                    continue
                nxt = units[i + 1][0] if i + 1 < len(units) else None
                ctx = ("shaddah" if y > x else "stop") if nxt is None else (
                    "shaddah" if y > x else {"َ": "fatha", "ُ": "damma", "ِ": "kasra"}.get(nxt, "madd" if nxt in madd else "sakin"))
                letters.append([sym, ctx])
            out.append({"surah": s, "ayah": a, "words": len(r.uthmani.split()),
                         "rules": [[b.rule_type, b.word_index,
                                    list(b.expected_counts) if b.expected_counts else None] for b in bounds],
                         "letters": letters})
        except Exception as exc:  # noqa: BLE001
            out.append({"surah": s, "ayah": a, "error": repr(exc)})
    return out


@app.local_entrypoint()
def build_inventory(tasks: int = 60) -> None:
    import gzip
    import sys
    import time
    sys.path.insert(0, str(ROOT / "research_agency_lab/experiments/learner_eval"))
    from quran_transcript import Aya
    t = time.time()
    ayahs, x = [], Aya(1, 1)
    for _ in range(6236):
        g = x.get()
        ayahs.append((g.sura_idx, g.aya_idx))
        x = x.step(1)
    chunks = [ayahs[i::tasks] for i in range(tasks)]
    out = ROOT / "research_agency_lab/experiments/quran/inventory.jsonl.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [r for rs in inventory.map(chunks, order_outputs=False) for r in rs]
    rows.sort(key=lambda r: (r["surah"], r["ayah"]))
    with gzip.open(out, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    err = sum("error" in r for r in rows)
    print(f"{len(rows)} ayahs ({err} errors), "
          f"{sum(len(r.get('rules', [])) for r in rows)} rule instances, "
          f"{sum(len(r.get('letters', [])) for r in rows)} consonants in {time.time() - t:.0f} s -> {out}")
