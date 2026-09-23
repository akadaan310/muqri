"""Minimal FastAPI service exposing the analysis pipeline (optional dependency).

No ``from __future__ import annotations`` here: FastAPI must resolve the endpoint annotations,
which reference types imported inside :func:`create_app`.
"""

import tempfile
from pathlib import Path
from typing import Any

from app.aligner import AlignmentError
from app.audio import AudioError
from app.pipeline import AnalysisOptions, QaariEvaluator
from app.quran_text import QuranTextError
from app.tajweed_rules import TajweedParseError

MAX_UPLOAD_BYTES = 50 * 1024 * 1024


def create_app(options: AnalysisOptions | None = None):  # type: ignore[no-untyped-def]
    try:
        from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("pip install fastapi uvicorn python-multipart") from exc

    app = FastAPI(title="qaari-eval", version="2.0.0")
    evaluator = QaariEvaluator(options)
    app.state.evaluator = evaluator

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/analyze")
    async def analyze(
        audio: UploadFile = File(...),  # noqa: B008
        surah: int | None = Form(None),  # noqa: B008
        ayah: int | None = Form(None),  # noqa: B008
        ayah_end: int | None = Form(None),  # noqa: B008
        text: str | None = Form(None),  # noqa: B008
        tareeq: str | None = Form(None),  # noqa: B008
        mode: str | None = Form(None),  # noqa: B008
        benchmark: str | None = Form(None),  # noqa: B008
    ) -> dict[str, Any]:
        data = await audio.read()
        if not data:
            raise HTTPException(400, "Empty upload")
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Audio file too large")
        suffix = Path(audio.filename or "upload.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
            tmp.write(data)
            tmp.flush()
            opts = evaluator.options
            saved = (opts.tareeq, opts.mode, opts.benchmark)
            try:
                if tareeq:
                    opts.tareeq = tareeq
                if mode:
                    opts.mode = mode
                opts.benchmark = benchmark
                result = evaluator.analyze_file(tmp.name, surah=surah, ayah=ayah, ayah_end=ayah_end, text=text)
            except (AudioError, AlignmentError, QuranTextError, TajweedParseError, ValueError) as exc:
                raise HTTPException(422, str(exc)) from exc
            finally:
                opts.tareeq, opts.mode, opts.benchmark = saved
        return result.report

    return app
