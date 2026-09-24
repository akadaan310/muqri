"""Posteriors for an external dataset, dumped once, graded as often as the engine changes.

The dump format is the one every Modal / Julia / Octave analysis reads (raw float32 `<id>.f32`,
`layout.json`, `index.jsonl`); each index record also carries the clip's audio path and verses, so
`grade` can re-run `Engine.analyze` on the stored posteriors -- with the waveform for stop detection
-- in about a second a clip, without the acoustic model.

    from research_agency_lab.experiments.external.dump import dump, grade
    dump(items, dest)                  # items: [{"id", "path", "verses"}]; needs the model (CPU ok)
    for rec, report in grade(dest): ...
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
LEARNER = ROOT / "research_agency_lab/experiments/learner_eval"


def _fname(cid: str) -> str:
    return cid.replace("/", "__").replace(":", "__") + ".f32"


def dump(items: Iterable[dict[str, Any]], dest: Path) -> int:
    sys.path.insert(0, str(LEARNER))
    import muaalem_dump as md  # noqa: PLC0415
    from app.engine import Engine  # noqa: PLC0415

    dest.mkdir(parents=True, exist_ok=True)
    idx = dest / "index.jsonl"
    done = {json.loads(l)["id"] for l in idx.open()} if idx.is_file() else set()
    eng, n = Engine(), 0
    with idx.open("a") as fh:
        for it in items:
            if it["id"] in done:
                continue
            rec: dict[str, Any] = {"id": it["id"], "path": str(it["path"]), "verses": it["verses"]}
            try:
                wave = md.load_16k(Path(it["path"]))
                lp = eng.posteriors(wave)
                if not (dest / "layout.json").is_file():
                    (dest / "layout.json").write_text(json.dumps(eng._layout, ensure_ascii=False))
                lp.astype("<f4").tofile(dest / _fname(it["id"]))
                rec.update({"file": _fname(it["id"]), "frames": int(lp.shape[0]),
                            "duration_s": round(wave.size / 16000, 4)})
            except Exception as exc:  # noqa: BLE001 - a bad clip is recorded, not fatal
                rec["error"] = f"{type(exc).__name__}: {exc}"[:300]
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            n += 1
    return n


def load(dest: Path, rec: dict[str, Any], layout: dict[str, Any]) -> np.ndarray:
    return np.fromfile(dest / rec["file"], dtype="<f4").reshape(rec["frames"], layout["columns"])


def grade(dest: Path, audio: bool = True) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """(index record, engine report) for every dumped clip, graded from the stored posteriors."""
    sys.path.insert(0, str(LEARNER))
    import muaalem_dump as md  # noqa: PLC0415
    from app.engine import Engine  # noqa: PLC0415

    layout = json.loads((dest / "layout.json").read_text())
    eng = Engine(layout=layout)
    for rec in map(json.loads, (dest / "index.jsonl").open()):
        if "file" not in rec:
            continue
        lp = load(dest, rec, layout)
        wave = None
        if audio:
            w = md.load_16k(Path(rec["path"]))[: rec["frames"] * 640]
            wave = np.pad(w, (0, rec["frames"] * 640 - w.size)).astype("float32")
        verses = [tuple(v) for v in rec["verses"]]
        try:
            rep = eng.analyze(wave, verses, posteriors=lp)
        except Exception as exc:  # noqa: BLE001
            rep = {"error": f"{type(exc).__name__}: {exc}"}
        yield rec, rep
