"""Modal Labs serverless deployment (scale-to-zero T4) of the qaari-eval HTTP API.

Deploy::

    pip install modal && modal setup
    modal deploy app/modal_endpoint.py

Cost profile: ``min_containers=0`` means no GPU is billed while idle; a container is started on
the first request and kept warm for ``scaledown_window`` seconds. Model weights (wav2vec2 aligner
and ECAPA) are downloaded at *image build* time, so a cold start only loads them from local disk.
Call ``GET /warmup`` (e.g. from a cron or before a class) to pre-start a container.

Endpoints (FastAPI, see ``app/api.py``): ``GET /health``, ``GET /warmup``, ``POST /analyze``
(multipart: ``audio`` + ``surah``/``ayah``/``ayah_end`` or ``text``, optional ``tareeq``, ``mode``,
``benchmark``).
"""

from __future__ import annotations

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
GPU = "T4"
SCALEDOWN_S = 120

app = modal.App("qaari-eval")


def _download_models() -> None:
    """Runs during the image build so containers start with the weights on disk."""
    from speechbrain.inference.speaker import EncoderClassifier
    from transformers import AutoModelForCTC, AutoProcessor

    from app.aligner import DEFAULT_CTC_MODEL

    AutoProcessor.from_pretrained(DEFAULT_CTC_MODEL)
    AutoModelForCTC.from_pretrained(DEFAULT_CTC_MODEL)
    EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                   savedir="/root/.cache/qaari-eval/ecapa")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install("torch", "torchaudio", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install_from_requirements(str(ROOT / "requirements-ml.txt"))
    .add_local_python_source("app", copy=True)
    .run_function(_download_models)
    .add_local_dir(str(ROOT / "index"), remote_path="/root/index")
)


@app.function(image=image, gpu=GPU, min_containers=0, scaledown_window=SCALEDOWN_S, timeout=600, max_containers=4)
@modal.concurrent(max_inputs=4)
@modal.asgi_app()
def api():  # type: ignore[no-untyped-def]
    import torch

    from app.api import create_app
    from app.pipeline import AnalysisOptions

    torch.backends.cudnn.benchmark = True
    web = create_app(AnalysisOptions(index_dir="/root/index", aligner="ctc", mode="auto"))
    evaluator = web.state.evaluator

    @web.get("/warmup")
    def warmup() -> dict[str, str]:
        _ = evaluator.aligner, evaluator.embedder, evaluator.indices  # load everything once
        return {"status": "warm", "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}

    return web


@app.local_entrypoint()
def smoke(audio: str, surah: int = 1, ayah: int = 1) -> None:
    """``modal run app/modal_endpoint.py --audio file.wav --surah 1 --ayah 1`` (prints the score)."""
    import json
    import urllib.request

    url = api.get_web_url()
    boundary = "qaari"
    data = Path(audio).read_bytes()
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"surah\"\r\n\r\n{surah}\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"ayah\"\r\n\r\n{ayah}\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"audio\"; filename=\"{Path(audio).name}\"\r\n"
            "Content-Type: application/octet-stream\r\n\r\n").encode() + data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(f"{url}/analyze", data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=600) as resp:  # noqa: S310
        report = json.loads(resp.read())
    print(json.dumps(report["recitation_summary"], ensure_ascii=False, indent=2))
