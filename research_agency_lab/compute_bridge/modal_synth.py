"""Recitation synthesis pilot on Modal: Husary's voice from the coverage-selected hour (SYNTHESIS.md).

    prep     T4   every ayah of datasets/qaari_keys/build/synth/pilot.json: EveryAyah audio at 44.1 kHz,
                  the engine's alignment (muaalem posteriors -> Engine.analyze), cut at Husary's own stops
                  into segments of at most MAX_SEG_S, each with its QPS text phonetised for exactly those
                  words (so a cut ends in the waqf form he actually recited). Train / test filelists and
                  the engine's report for every whole ayah land on the volume under synth/pilot/.
    train    L4   Matcha-TTS (conditional flow matching, monotonic alignment search, explicit per-phoneme
                  durations at inference) on QPS symbols, 128-band 44.1 kHz mels -- the exact feature of
                  NVIDIA's pretrained BigVGAN-v2 44 kHz vocoder, so Husary's 15.7 kHz bandwidth survives.
    synth    T4   the held-out passages (and a few training ones) through Matcha + BigVGAN-v2.

    modal run research_agency_lab/compute_bridge/modal_synth.py::prep
    modal run research_agency_lab/compute_bridge/modal_synth.py::train --hours 2.5
    modal run research_agency_lab/compute_bridge/modal_synth.py::synth --out research_agency_lab/experiments/synthesis/pilot
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

_here = Path(__file__).resolve()
ROOT = _here.parents[2] if len(_here.parents) > 2 and (_here.parents[2] / "app").is_dir() else Path("/root/qaari")
SYN = ROOT / "datasets/qaari_keys/build/synth"
VOICE = "Husary_128kbps"
SR = 44100
MAX_SEG_S = 18.0
MEL = {"n_fft": 2048, "n_feats": 128, "sample_rate": SR, "hop_length": 512, "win_length": 2048, "f_min": 0,
       "f_max": 22050}
BIGVGAN = "nvidia/bigvgan_v2_44khz_128band_512x"

app = modal.App("qaari-synth")
vol = modal.Volume.from_name("qaari-runs")
_IGNORE = ["**/__pycache__", "**/*.pyc", "**/*.jsonl", "**/*.f32"]


def _warm_engine() -> None:
    import transformers.models.wav2vec2_bert.modeling_wav2vec2_bert as w

    w._HIDDEN_STATES_START_POSITION = 2
    from quran_muaalem import Muaalem

    Muaalem("obadx/muaalem-model-v3_2", device="cpu")
    from transformers import AutoFeatureExtractor, AutoModelForAudioFrameClassification
    AutoFeatureExtractor.from_pretrained("obadx/recitation-segmenter-v2")
    AutoModelForAudioFrameClassification.from_pretrained("obadx/recitation-segmenter-v2")


engine_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1", "ffmpeg")
    .pip_install("torch==2.14.0", "torchaudio")
    .pip_install("transformers==5.17.0", "numpy==2.4.6", "scipy", "librosa", "soundfile", "pyarrow",
                 "quran-transcript==0.6.1", "diff-match-patch", "rich", "pydantic")
    .pip_install("quran-muaalem==0.2.2", "recitations-segmenter==1.0.0", extra_options="--no-deps")
    .run_function(_warm_engine)
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


def _patch_matcha() -> None:
    """Matcha reads a fixed English/IPA symbol table; give it the Quran Phonetic Script instead."""
    import pathlib
    base = pathlib.Path("/root/matcha/matcha/text")
    qps = pathlib.Path("/root/qps_symbols.txt").read_text()
    (base / "symbols.py").write_text(
        '"""Quran Phonetic Script (quran_transcript, remove_spaces=False): the symbols of all 6,236 ayahs."""\n'
        f'_pad = "_"\nsymbols = [_pad] + list({qps!r})\nSPACE_ID = symbols.index(" ")\n')
    # newer torchaudio needs torchcodec just to read a wav; read it with soundfile instead
    dm = base.parent / "data" / "text_mel_datamodule.py"
    src = dm.read_text()
    assert "audio, sr = ta.load(filepath)" in src
    dm.write_text(src.replace("audio, sr = ta.load(filepath)",
                              "import soundfile as _sf; _y, sr = _sf.read(filepath, dtype='float32', always_2d=True); "
                              "audio = torch.from_numpy(_y.T.copy())"))
    cl = base / "cleaners.py"
    cl.write_text(cl.read_text() + '\n\ndef qps_cleaners(text):\n    """QPS is already phonetic: no cleaning."""\n    return text\n')


train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "libsndfile1", "ffmpeg", "espeak-ng")  # matcha.text imports phonemizer
    .pip_install("torch==2.14.0", "torchaudio")
    .pip_install("cython", "numpy<2.5")
    # the Hydra configs live in the repository, not the package: install the clone in place
    .run_commands("git clone --depth 1 https://github.com/shivammehta25/Matcha-TTS /root/matcha",
                  "pip install --no-build-isolation -e /root/matcha",
                  "git clone --depth 1 https://github.com/NVIDIA/BigVGAN /root/bigvgan")
    .pip_install("huggingface_hub", "librosa", "soundfile", "ninja", "tensorboard", "matplotlib==3.9.4")  # matcha plots with tostring_rgb (gone in 3.10)
    .add_local_file(str(SYN / "qps_symbols.txt"), "/root/qps_symbols.txt", copy=True)
    .run_function(_patch_matcha)
)


# ---------------------------------------------------------------------------------------------- prep
SEGMENTER = "obadx/recitation-segmenter-v2"   # waqf pauses, 20 ms precision (F1 0.996 per its card)
MIN_SILENCE_MS = 200                          # shorter gaps are not pauses (QuranCaption's default)


def _intervals(wave16, seg) -> list[tuple[float, float]]:  # type: ignore[no-untyped-def]
    """Husary's speech intervals in one ayah, split at his pauses by the recitation segmenter
    that built muaalem's 848 h (and drives QuranCaption / the Quranic Universal Aligner). The engine's
    own stop detection found no word-boundary stop inside 60 s ayahs he breathes in several times
    (open issue 3), so the engine is used only to say which words a pause falls between."""
    import torch
    from recitations_segmenter import clean_speech_intervals, segment_recitations
    model, proc = seg
    out = segment_recitations([torch.from_numpy(wave16)], model, proc, device=torch.device("cuda"),
                              dtype=torch.bfloat16, batch_size=1)[0]
    iv = clean_speech_intervals(out.speech_intervals, out.is_complete, min_silence_duration_ms=MIN_SILENCE_MS,
                                min_speech_duration_ms=30, pad_duration_ms=30, return_seconds=True)
    return [tuple(map(float, x)) for x in iv.clean_speech_intervals.tolist()]


def _assign_words(eng, ref, wave16, intervals) -> list[tuple[int, int]]:  # type: ignore[no-untyped-def]
    """Which words each speech interval holds: the ayah's words split into len(intervals) consecutive,
    non-empty groups maximising the summed CTC log-likelihood of each group's phonemes over its own
    interval's posteriors. Every interval gets its own model call, so no alignment spans a pause --
    aligned as one 60 s file, the engine stretched a fatha over a 4 s pause (27:36)."""
    import numpy as np
    from app.lahn.gop import ctc_log_likelihood
    lay = eng._layout
    ph = lay["levels"]["phonemes"]
    vocab = {t: i for i, t in enumerate(ph["vocab"]) if len(t) == 1}
    blocks = []
    for t0, t1 in intervals:
        lp = eng.posteriors(wave16[int(t0 * 16000):int(t1 * 16000)])
        blocks.append(lp[:, ph["first"]:ph["first"] + ph["width"]])
    W, K = len(ref.word_ph), len(intervals)
    wseq = [[vocab[c] for c in ref.phonemes[p0:p1] if c in vocab] for p0, p1 in ref.word_ph]
    memo: dict[tuple[int, int, int], float] = {}

    def ll(k: int, w0: int, w1: int) -> float:          # interval k holds words w0..w1-1
        if (k, w0, w1) not in memo:
            memo[(k, w0, w1)] = ctc_log_likelihood(blocks[k], [x for w in range(w0, w1) for x in wseq[w]], lay["blank"])
        return memo[(k, w0, w1)]

    NEG = -np.inf
    best = np.full((K + 1, W + 1), NEG)
    back = np.zeros((K + 1, W + 1), dtype=int)
    best[0, 0] = 0.0
    for k in range(1, K + 1):
        for w in range(k, W - (K - k) + 1):
            for v in range(k - 1, w):
                if best[k - 1, v] == NEG:
                    continue
                sc = best[k - 1, v] + ll(k - 1, v, w)
                if sc > best[k, w]:
                    best[k, w], back[k, w] = sc, v
    out, w = [], W
    for k in range(K, 0, -1):
        v = back[k, w]
        out.append((int(v), int(w - 1)))
        w = int(v)
    return out[::-1]


def _segments(intervals, words, seconds: float) -> list[tuple[int, int, float, float]]:  # type: ignore[no-untyped-def]
    """Consecutive speech intervals merged into pieces of at most MAX_SEG_S, each cut in the middle
    of the pause that ends it: (first word, last word, start s, end s)."""
    out: list[tuple[int, int, float, float]] = []
    cur = None
    for i, ((t0, t1), (w0, w1)) in enumerate(zip(intervals, words)):
        start = 0.0 if i == 0 else (intervals[i - 1][1] + t0) / 2
        end = seconds if i == len(intervals) - 1 else (t1 + intervals[i + 1][0]) / 2
        if cur and end - cur[2] <= MAX_SEG_S:
            cur = (cur[0], w1, cur[2], end)
        else:
            if cur:
                out.append(cur)
            cur = (w0, w1, start, end)
    out.append(cur)  # type: ignore[arg-type]
    return out


@app.function(image=engine_image, gpu="T4", timeout=3 * 3600, volumes={"/vol": vol}, cpu=4.0, memory=12288)
def prep_items(items: list[dict]) -> list[dict]:  # type: ignore[type-arg]
    import io
    import sys
    import urllib.request

    import librosa
    import numpy as np
    import soundfile as sf

    sys.path.insert(0, "/root/qaari")
    sys.path.insert(0, "/root/qaari/research_agency_lab/experiments/learner_eval")
    from muaalem_eval import MOSHAF
    from quran_transcript import quran_phonetizer

    from app.engine import Engine

    out_dir = Path("/vol/synth/pilot")
    (out_dir / "wavs").mkdir(parents=True, exist_ok=True)
    (out_dir / "ayahs").mkdir(parents=True, exist_ok=True)
    import torch
    from transformers import AutoFeatureExtractor, AutoModelForAudioFrameClassification
    seg = (AutoModelForAudioFrameClassification.from_pretrained(SEGMENTER).to("cuda", dtype=torch.bfloat16),
           AutoFeatureExtractor.from_pretrained(SEGMENTER))
    eng = Engine()
    rows = []
    for it in items:
        s, a = it["surah"], it["ayah"]
        rec = {**it}
        try:
            url = f"https://everyayah.com/data/{VOICE}/{s:03d}{a:03d}.mp3"
            for attempt in range(3):
                try:
                    data = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "qaari"}),  # noqa: S310
                                                  timeout=120).read()
                    y, sr = librosa.load(io.BytesIO(data), sr=SR, mono=True)
                    break
                except Exception:  # noqa: BLE001
                    if attempt == 2:
                        raise
            y16 = librosa.resample(y, orig_sr=SR, target_sr=16000)
            rep = eng.analyze(y16, [(s, a)])
            m = rep["measurements"]
            uth = eng.reference(s, a).uthmani.split()
            sf.write(out_dir / "ayahs" / f"{s:03d}{a:03d}.wav", y, SR, subtype="PCM_16")
            (out_dir / "ayahs" / f"{s:03d}{a:03d}.json").write_text(json.dumps(m, ensure_ascii=False))
            segs = []
            ref = eng.reference(s, a)
            intervals = _intervals(y16.astype("float32"), seg)
            while len(intervals) > len(uth):       # more pauses than words: merge across the shortest
                g = min(range(len(intervals) - 1), key=lambda i: intervals[i + 1][0] - intervals[i][1])
                intervals[g:g + 2] = [(intervals[g][0], intervals[g + 1][1])]
            words = _assign_words(eng, ref, y16.astype("float32"), intervals) if len(intervals) > 1 else \
                [(0, len(uth) - 1)]
            for k, (w0, w1, t0, t1) in enumerate(_segments(intervals, words, len(y) / SR)):
                piece = y[int(t0 * SR):int(t1 * SR)]
                text = quran_phonetizer(" ".join(uth[w0:w1 + 1]), MOSHAF, remove_spaces=False).phonemes
                name = f"{s:03d}{a:03d}_{k}.wav"
                sf.write(out_dir / "wavs" / name, piece / max(1.0, float(np.abs(piece).max())), SR, subtype="PCM_16")
                segs.append({"file": f"wavs/{name}", "words": [w0, w1], "seconds": round(len(piece) / SR, 3),
                             "text": text})
            rec.update({"seconds": round(len(y) / SR, 3), "segments": segs, "intervals": [[round(p0, 3), round(p1, 3), *w] for (p0, p1), w in zip(intervals, words)],
                        "engine_stops": sum(st["position"] == "word_boundary" for st in m["recording"]["stops"]), "summary": m["summary"],
                        "seconds_per_count": m["recording"]["seconds_per_count"]})
        except Exception as exc:  # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {exc}"[:300]
        rows.append(rec)
    vol.commit()
    return rows


@app.local_entrypoint()
def prep(shards: int = 4) -> None:
    pilot = json.loads((SYN / "pilot.json").read_text())
    items = [{"surah": p["surah"], "ayah": a, "split": split, "passage": f"{p['surah']}:{p['from_ayah']}-{p['to_ayah']}"}
             for split in ("train", "test") for p in pilot[split]["passages"]
             for a in range(p["from_ayah"], p["to_ayah"] + 1)]
    rows = [r for rs in prep_items.map([items[i::shards] for i in range(shards)]) for r in rs]
    bad = [r for r in rows if "error" in r]
    lines = {"train": [], "test": []}
    for r in rows:
        for sg in r.get("segments", []):
            if sg["seconds"] <= MAX_SEG_S + 2:
                lines[r["split"]].append(f"{sg['file']}|{sg['text']}")
    local = ROOT / "research_agency_lab/experiments/synthesis/pilot"
    local.mkdir(parents=True, exist_ok=True)
    (local / "prep.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1, default=float))
    for split, ls in lines.items():
        (local / f"{split}.txt").write_text("\n".join(sorted(ls)) + "\n")
    import subprocess
    for split in lines:
        subprocess.run(["modal", "volume", "put", "--force", "qaari-runs", str(local / f"{split}.txt"),
                        f"synth/pilot/{split}.txt"], check=True, capture_output=True)
    segs = [sg for r in rows for sg in r.get("segments", [])]
    print(f"{len(rows)} ayahs ({len(bad)} errors), {len(segs)} segments, "
          f"{sum(len(v) for v in lines.values())} kept (train {len(lines['train'])}, test {len(lines['test'])}), "
          f"{sum(sg['seconds'] for sg in segs) / 60:.1f} min; longest {max(sg['seconds'] for sg in segs):.1f} s")
    for r in bad:
        print("  error", r["surah"], r["ayah"], r["error"])


# --------------------------------------------------------------------------------------------- train
@app.function(image=train_image, gpu="H100", timeout=24 * 3600, volumes={"/vol": vol}, cpu=12.0, memory=65536)
def train_matcha(hours: float, batch: int, resume: bool) -> str:
    import os
    import subprocess

    base = Path("/vol/synth/pilot")
    work = Path("/vol/synth/matcha")
    work.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        lines = [l for l in (base / f"{split}.txt").read_text().splitlines() if l.strip()]
        (work / f"{split}.txt").write_text("\n".join(f"{base}/{l}" for l in lines) + "\n")
    stats = work / "stats.json"
    common = [f"data.train_filelist_path={work}/train.txt", f"data.valid_filelist_path={work}/test.txt",
              "data.cleaners=[qps_cleaners]", "data.add_blank=True", "data.n_spks=1",
              *(f"data.{k}={v}" for k, v in MEL.items()), f"data.batch_size={batch}", "data.num_workers=10"]
    if not stats.exists():
        stats.write_text(json.dumps(_mel_stats(work / "train.txt")))
    st = json.loads(stats.read_text())
    from matcha.text.symbols import symbols
    cfg = [*common, f"data.data_statistics.mel_mean={st['mel_mean']}", f"data.data_statistics.mel_std={st['mel_std']}",
           "model.n_feats=128", f"model.n_vocab={len(symbols)}", "model.out_size=688", f"+trainer.max_time=00:{int(hours * 60) // 60:02d}:{int(hours * 60) % 60:02d}:00",
           "trainer.check_val_every_n_epoch=25", "run_name=husary_pilot", f"paths.output_dir={work}/run",
           f"hydra.run.dir={work}/run", "callbacks.model_checkpoint.every_n_epochs=25",
           "callbacks.model_checkpoint.save_top_k=-1", "logger=tensorboard"]
    if resume:
        ckpts = sorted((work / "run").rglob("*.ckpt"), key=os.path.getmtime)
        if ckpts:
            cfg.append(f"ckpt_path={ckpts[-1]}")
    train_py = Path("/root/matcha/matcha/train.py")
    print("training:", " ".join(cfg))
    proc = subprocess.Popen(["python", str(train_py), "experiment=ljspeech", *cfg], cwd=work,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    tail: list[str] = []
    for line in proc.stdout:  # type: ignore[union-attr]
        tail = (tail + [line])[-40:]
        if "Epoch" in line and "loss" in line:
            print(line.strip()[-200:], flush=True)
    proc.wait()
    vol.commit()
    return "".join(tail)


def _mel_stats(filelist: Path) -> dict[str, float]:
    """Mean and std of the log-mel over every training frame, with Matcha's own mel function."""
    import soundfile as sf
    import torch
    from matcha.utils.audio import mel_spectrogram
    n, s1, s2 = 0, 0.0, 0.0
    for line in filelist.read_text().splitlines():
        if not line.strip():
            continue
        y, _ = sf.read(line.split("|")[0], dtype="float32")
        m = mel_spectrogram(torch.from_numpy(y)[None], MEL["n_fft"], MEL["n_feats"], SR, MEL["hop_length"],
                            MEL["win_length"], MEL["f_min"], MEL["f_max"])
        n += m.numel()
        s1 += float(m.sum())
        s2 += float((m ** 2).sum())
    mean = s1 / n
    return {"mel_mean": mean, "mel_std": (s2 / n - mean ** 2) ** 0.5}


@app.local_entrypoint()
def train(hours: float = 2.5, batch: int = 16, resume: bool = False) -> None:
    print(train_matcha.remote(hours, batch, resume))


# --------------------------------------------------------------------------------------------- synth
AYAH_GAP_S = 0.8          # silence between synthesized ayahs of a passage


@app.function(image=train_image, gpu="T4", timeout=3600, volumes={"/vol": vol}, cpu=4.0, memory=16384)
def synth_passages(passages: list[dict], ckpt: str, steps: int, temperature: float) -> list[dict]:  # type: ignore[type-arg]
    """Each passage, ayah by ayah (QPS of the whole ayah), through Matcha + BigVGAN-v2 44 kHz."""
    import io
    import sys

    import numpy as np
    import soundfile as sf
    import torch

    sys.path.insert(0, "/root/bigvgan")
    import bigvgan
    from matcha.models.matcha_tts import MatchaTTS
    from matcha.text import text_to_sequence
    from matcha.utils.utils import intersperse

    vol.reload()
    run = Path("/vol/synth/matcha/run")
    ck = Path(ckpt) if ckpt else max(run.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    # our own checkpoint (trusted): it stores its Hydra config, which weights_only=True refuses
    model = MatchaTTS.load_from_checkpoint(str(ck), map_location="cuda", weights_only=False).eval().cuda()
    # BigVGAN.from_pretrained no longer matches huggingface_hub's mixin signature: build it by hand
    from env import AttrDict
    from huggingface_hub import hf_hub_download
    h = AttrDict(json.loads(Path(hf_hub_download(BIGVGAN, "config.json")).read_text()))
    voc = bigvgan.BigVGAN(h, use_cuda_kernel=False)
    voc.load_state_dict(torch.load(hf_hub_download(BIGVGAN, "bigvgan_generator.pt"), map_location="cpu")["generator"])
    voc.remove_weight_norm()
    voc = voc.eval().cuda()
    out = []
    for p in passages:
        parts = []
        for t in p["ayahs"]:
            x = torch.tensor(intersperse(text_to_sequence(t["text"], ["qps_cleaners"])[0], 0), dtype=torch.long,
                             device="cuda")[None]
            with torch.inference_mode():
                o = model.synthesise(x, torch.tensor([x.shape[-1]], device="cuda"), n_timesteps=steps,
                                     temperature=temperature, length_scale=1.0)
                wav = voc(o["mel"]).squeeze().float().cpu().numpy()
            parts += [wav, np.zeros(int(AYAH_GAP_S * SR), dtype=np.float32)]
        y = np.concatenate(parts[:-1])
        y = 0.95 * y / max(1e-6, float(np.abs(y).max()))
        buf = io.BytesIO()
        sf.write(buf, y, SR, format="WAV", subtype="PCM_16")
        out.append({"name": p["name"], "wav": buf.getvalue(), "seconds": round(len(y) / SR, 2), "ckpt": ck.name})
    return out


@app.local_entrypoint()
def synth(out: str = "research_agency_lab/experiments/synthesis/pilot/samples", ckpt: str = "", steps: int = 32,
          temperature: float = 0.667, train_passages: int = 2) -> None:
    """The held-out test passages (and a few training ones) synthesized; the real Husary beside each."""
    import io
    import sys
    import urllib.request

    import numpy as np
    import soundfile as sf
    sys.path.insert(0, str(ROOT / "research_agency_lab/experiments/learner_eval"))
    from muaalem_eval import MOSHAF
    from quran_transcript import Aya, quran_phonetizer
    import librosa

    pilot = json.loads((SYN / "pilot.json").read_text())
    chosen = [("test", p) for p in pilot["test"]["passages"]] + \
        [("train", p) for p in sorted(pilot["train_long"]["passages"], key=lambda p: -p["ayahs"])[:train_passages]]
    passages = []
    for split, p in chosen:
        ays = [{"surah": p["surah"], "ayah": a,
                "text": quran_phonetizer(Aya(p["surah"], a).get().uthmani, MOSHAF, remove_spaces=False).phonemes}
               for a in range(p["from_ayah"], p["to_ayah"] + 1)]
        passages.append({"name": f"{split}_{p['surah']:03d}_{p['from_ayah']:03d}-{p['to_ayah']:03d}", "ayahs": ays})
    d = ROOT / out
    d.mkdir(parents=True, exist_ok=True)
    for r in synth_passages.remote(passages, ckpt, steps, temperature):
        (d / f"{r['name']}_SYNTHETIC.wav").write_bytes(r["wav"])
        print(r["name"], r["seconds"], "s", r["ckpt"])
    for p in passages:                         # the real recitation of the same passage, for comparison
        parts = []
        for a in p["ayahs"]:
            data = urllib.request.urlopen(urllib.request.Request(  # noqa: S310
                f"https://everyayah.com/data/{VOICE}/{a['surah']:03d}{a['ayah']:03d}.mp3",
                headers={"User-Agent": "qaari"}), timeout=120).read()
            y, _ = librosa.load(io.BytesIO(data), sr=SR, mono=True)
            parts += [y, np.zeros(int(AYAH_GAP_S * SR), dtype=np.float32)]
        sf.write(d / f"{p['name']}_REAL.wav", np.concatenate(parts[:-1]), SR, subtype="PCM_16")
