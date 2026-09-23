#!/usr/bin/env python3
"""Offload benchmark runs to Kaggle kernels (free tier: 12 h sessions, 5 concurrent batch CPU sessions).

    export KAGGLE_API_TOKEN=...                      # never commit it
    python research_agency_lab/compute_bridge/kaggle_bridge.py push-code
    python research_agency_lab/compute_bridge/kaggle_bridge.py launch --tag husary-all \
        --reciters Husary_128kbps Husary_Muallim_128kbps --verses all --modes studio --kernels 5 --procs 4 --octave
    python research_agency_lab/compute_bridge/kaggle_bridge.py status --tag husary-all --kernels 5
    python research_agency_lab/compute_bridge/kaggle_bridge.py collect --tag husary-all --kernels 5

``push-code`` uploads the engine (app/, benchmarks/, research_agency_lab/, the cached Uthmani text)
as the private dataset ``<user>/qaari-eval-code``. Each kernel attaches it plus the three Quran-MD WAV
parts, pip-installs the few missing wheels, runs ``--procs`` benchmark processes on its slice of
``--shard`` (global shard = kernel * procs + proc), optionally runs the Octave DSP pass, and leaves
``runs_*.jsonl`` in /kaggle/working for ``collect``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CODE_SLUG = "qaari-eval-code"
DEPS_SLUG = "qaari-eval-deps"  # wheels + model snapshots, for kernels without internet
QURAN_MD = [f"husseinzahaki/quran-md-ayahs-wav-part{i}" for i in (1, 2, 3)]
PIP = "praat-parselmouth faiss-cpu nara_wpe soxr speechbrain"
WORKER = ROOT / "research_agency_lab" / "compute_bridge" / "kaggle_worker.py"


def kaggle(*args: str, check: bool = True) -> str:
    proc = subprocess.run(["kaggle", *args], capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise SystemExit(f"kaggle {' '.join(args)} failed:\n{proc.stdout}\n{proc.stderr}")
    return proc.stdout + proc.stderr


def username() -> str:
    out = kaggle("config", "view")
    for line in out.splitlines():
        if "username:" in line:
            return line.split(":", 1)[1].strip()
    raise SystemExit("Could not determine the Kaggle username (is KAGGLE_API_TOKEN set?)")


def publish(folder: str, message: str, mode: str) -> None:
    """New version of an existing dataset, or create it on the first push."""
    out = kaggle("datasets", "version", "-p", folder, "-m", message, "-r", mode, check=False)
    if "being created" not in out and "successfully" not in out.lower():
        out = kaggle("datasets", "create", "-p", folder, "-r", mode)
    print("\n".join(ln for ln in out.splitlines() if "%|" not in ln)[-600:])


def push_code(args: argparse.Namespace) -> None:
    user = username()
    with tempfile.TemporaryDirectory() as tmp:
        tmpd = Path(tmp)
        with tarfile.open(tmpd / "qaari_eval.tar.gz", "w:gz") as tar:
            for rel in ("app", "benchmarks", "datasets", "research_agency_lab", "main.py", "requirements.txt",
                        "pyproject.toml"):
                tar.add(ROOT / rel, arcname=rel,
                        filter=lambda ti: None if ("__pycache__" in ti.name or ti.name.endswith(".jsonl")) else ti)
        text = Path.home() / ".cache" / "qaari-eval" / "text" / "quran-uthmani.json"
        if text.exists():
            shutil.copy(text, tmpd / "quran-uthmani.json")
        meta = {"title": "qaari-eval code", "id": f"{user}/{CODE_SLUG}", "licenses": [{"name": "other"}]}
        (tmpd / "dataset-metadata.json").write_text(json.dumps(meta))
        publish(tmp, args.message, "tar")


def push_deps(args: argparse.Namespace) -> None:
    """Upload a directory holding ``wheels/`` and ``hf_hub/`` as the private deps dataset.

    Build it with (Kaggle's Python is 3.12):
        pip download --no-deps --only-binary=:all: --python-version 3.12 --implementation cp \
            --platform manylinux2014_x86_64 --platform manylinux_2_28_x86_64 -d DIR/wheels \
            praat-parselmouth faiss-cpu soxr sentencepiece bottleneck ruamel.yaml.clib
        pip download --no-deps --only-binary=:all: -d DIR/wheels speechbrain hyperpyyaml ruamel.yaml
        pip wheel --no-deps -w DIR/wheels nara_wpe
        cp -r ~/.cache/huggingface/hub/models--X/{refs,snapshots} DIR/hf_hub/models--X/  (use cp -L)
    """
    user = username()
    src = Path(args.dir)
    meta = {"title": "qaari-eval deps", "id": f"{user}/{DEPS_SLUG}", "licenses": [{"name": "other"}]}
    (src / "dataset-metadata.json").write_text(json.dumps(meta))
    publish(str(src), args.message, "zip")


def kernel_slug(tag: str, k: int) -> str:
    return f"qaari-{tag}-{k}"


def launch(args: argparse.Namespace) -> None:
    user = username()
    total = args.kernels * args.procs
    for k in range(args.kernels):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = {"reciters": args.reciters, "verses": args.verses, "modes": args.modes, "procs": args.procs,
                   "shard_base": k * args.procs, "shard_total": total, "octave": args.octave, "tag": args.tag,
                   "pip": PIP, "hours": args.hours}
            src = WORKER.read_text(encoding="utf-8").replace("__CONFIG__", json.dumps(cfg))
            (Path(tmp) / "worker.py").write_text(src, encoding="utf-8")
            meta = {
                "id": f"{user}/{kernel_slug(args.tag, k)}", "title": kernel_slug(args.tag, k),
                "code_file": "worker.py", "language": "python", "kernel_type": "script",
                "is_private": True, "enable_gpu": args.gpu, "enable_internet": True,
                "dataset_sources": [f"{user}/{CODE_SLUG}", f"{user}/{DEPS_SLUG}", *QURAN_MD],
                "competition_sources": [], "kernel_sources": [],
            }
            (Path(tmp) / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
            print(kaggle("kernels", "push", "-p", tmp).strip())


def status(args: argparse.Namespace) -> None:
    user = username()
    for k in range(args.kernels):
        print(kaggle("kernels", "status", f"{user}/{kernel_slug(args.tag, k)}", check=False).strip())


def collect(args: argparse.Namespace) -> None:
    user = username()
    dest = ROOT / "benchmarks" / "results" / "kaggle" / args.tag
    dest.mkdir(parents=True, exist_ok=True)
    for k in range(args.kernels):
        with tempfile.TemporaryDirectory() as tmp:
            print(kaggle("kernels", "output", f"{user}/{kernel_slug(args.tag, k)}", "-p", tmp, check=False)[-300:])
            for f in Path(tmp).glob("*.jsonl"):
                shutil.copy(f, dest / f"k{k}_{f.name}")
            for f in Path(tmp).glob("*.log"):
                shutil.copy(f, dest / f"k{k}_{f.name}")
    print(f"Collected into {dest}")


def main(argv: list[str] | None = None) -> int:
    if not os.environ.get("KAGGLE_API_TOKEN") and not (Path.home() / ".kaggle" / "access_token").exists():
        print("Set KAGGLE_API_TOKEN first.", file=sys.stderr)
        return 2
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("push-code")
    p.add_argument("--message", default="update")
    p = sub.add_parser("push-deps")
    p.add_argument("dir", help="directory with wheels/ and hf_hub/")
    p.add_argument("--message", default="update")
    p = sub.add_parser("launch")
    p.add_argument("--tag", required=True)
    p.add_argument("--reciters", nargs="+", required=True)
    p.add_argument("--verses", default="all")
    p.add_argument("--modes", nargs="+", default=["studio"])
    p.add_argument("--kernels", type=int, default=5)
    p.add_argument("--procs", type=int, default=4)
    p.add_argument("--hours", type=float, default=11.5, help="stop cleanly before Kaggle's 12 h limit")
    p.add_argument("--octave", action="store_true")
    p.add_argument("--gpu", action="store_true")
    for name in ("status", "collect"):
        p = sub.add_parser(name)
        p.add_argument("--tag", required=True)
        p.add_argument("--kernels", type=int, default=5)
    args = ap.parse_args(argv)
    commands = {"push-code": push_code, "push-deps": push_deps, "launch": launch, "status": status, "collect": collect}
    commands[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
