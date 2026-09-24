"""Score an external dataset with the serving engine, one clip at a time, resumably.

Every external dataset reduces to the same work list: an audio file and the verses it covers
(surah, ayah[, first_word, last_word], 0-based words). Each clip goes through `/analyze` of a running
webapp -- the exact path a consumer app takes, and one model in memory however many runners -- and
its report is reduced to what the comparisons need: the measurement contract, the ghunnah grades,
the timing. Results append to a jsonl keyed by clip id, so an interrupted run resumes.

    from research_agency_lab.experiments.external.runner import run
    run(items, out_path)        # items: [{"id", "path", "verses": [[s, a, w0, w1], ...]}]
"""

from __future__ import annotations

import json
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Iterable

URL = "http://127.0.0.1:8088/analyze"


def _form(fields: dict[str, str], name: str, data: bytes) -> tuple[bytes, str]:
    b = uuid.uuid4().hex
    parts = [f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode() for k, v in fields.items()]
    parts.append(f'--{b}\r\nContent-Disposition: form-data; name="audio"; filename="{name}"\r\n'
                 f"Content-Type: application/octet-stream\r\n\r\n".encode() + data + b"\r\n")
    return b"".join(parts) + f"--{b}--\r\n".encode(), f"multipart/form-data; boundary={b}"


def fields_for(verses: list[list[int]]) -> dict[str, str]:
    """The /analyze form for a run of verses in one surah (words 0-based here, 1-based on the form)."""
    first, last = verses[0], verses[-1]
    f = {"surah": str(first[0]), "ayah": str(first[1])}
    if last[1] != first[1]:
        f["ayah_end"] = str(last[1])
    if len(first) > 2:
        f["word"] = str(first[2] + 1)
    if len(last) > 2:
        f["word_end"] = str(last[3] + 1)
    return f


def reduce(report: dict[str, Any]) -> dict[str, Any]:
    return {"measurements": report.get("measurements"),
            "ghunnah": [g for a in report.get("ayahs", []) for g in a.get("ghunnah", [])],
            "summary": report.get("summary"), "audio_seconds": report.get("audio_seconds"),
            "elapsed_seconds": report.get("elapsed_seconds")}


def analyze(path: Path, verses: list[list[int]], url: str = URL, timeout: float = 600) -> dict[str, Any]:
    body, ctype = _form(fields_for(verses), path.name, path.read_bytes())
    req = urllib.request.Request(url, data=body, headers={"Content-Type": ctype})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - local webapp
            rep = json.loads(r.read())
    except urllib.error.HTTPError as e:
        rep = json.loads(e.read() or b"{}")
    out = reduce(rep) if "error" not in rep else {"error": rep["error"]}
    out["wall_seconds"] = round(time.time() - t0, 3)
    return out


def run(items: Iterable[dict[str, Any]], out: Path, url: str = URL) -> int:
    done = {json.loads(l)["id"] for l in out.open()} if out.is_file() else set()
    n = 0
    with out.open("a") as fh:
        for it in items:
            if it["id"] in done:
                continue
            rec = {"id": it["id"], "verses": it["verses"], **analyze(Path(it["path"]), it["verses"], url)}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            n += 1
    return n
