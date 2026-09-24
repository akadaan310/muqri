"""RetaSy/quranic_audio_dataset: 6,828 recordings by 1,287 non-Arabic learners in 11+ countries.

911 are labelled `correct` / `in_correct` by three crowd annotators (algorithmic consensus, 0.89 with
expert judgement per the dataset card); the other labels (not Quran, wrong ayah, several ayahs,
incomplete) say the clip is not a gradable recitation of its text. The text is given, not the ayah
number, so each is located: a whole ayah of the named surah, or -- Ayat al-Kursi is recorded phrase
by phrase -- a word range inside 2:255. Du'a, adhan and tashahhud clips and the isti'adha are not
Quran text and are left out.

The unlabelled clips matter as much as the labelled: they are what learner audio actually measures
like, which is what the learner prior needs.

    python -m research_agency_lab.experiments.external.retasy stage     # manifest + audio -> STAGE
    python -m research_agency_lab.experiments.external.retasy compare   # engine vs annotators
"""

from __future__ import annotations

import difflib
import glob
import json
import re
import sys
from pathlib import Path
from typing import Any

DATA = Path.home() / "data/sprint4/hf/RetaSy__quranic_audio_dataset"
STAGE = Path.home() / "data/sprint4/stage/retasy"
HERE = Path(__file__).resolve().parent
OUT = HERE / "retasy_engine.jsonl"
SURAH = {"Al-Faatihah": 1, "Al-Ikhlas": 112, "An-Nas": 114, "Ayat al-Kursi": 2, "Al-Kafiroon": 109,
         "Al-Falaq": 113, "Al-Kauthar": 108, "Al-Asr": 103, "Al-Masad": 111, "Al-Humazah": 104,
         "An-Nasr": 110, "Al-Qadr": 97, "Al-Fil": 105, "Al-Maaoon": 107, "Quraish": 106, "Al-NABAA": 78}
MIN_RATIO = 0.8
COLS = ["Surah", "Aya", "duration_ms", "golden", "final_label", "reciter_id", "reciter_country",
        "reciter_gender", "reciter_age", "reciter_qiraah", "judgments_num", "annotation_metadata"]


def norm(s: str) -> list[str]:
    s = re.sub(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640\u08F0-\u08FF]", "", s)
    s = re.sub("[ٱأإآ]", "ا", s).replace("ى", "ي").replace("ی", "ي").replace("ة", "ه")
    s = s.replace("ؤ", "و").replace("ئ", "ي").replace("ۤ", "")
    return re.sub(r"[^\u0621-\u064A ]", "", s).split()


def _ayahs(surah: int) -> list[tuple[int, list[str], list[str]]]:
    from quran_transcript import Aya
    out, x = [], Aya(surah, 1)
    while True:
        g = x.get()
        if g.sura_idx != surah:
            return out
        out.append((g.aya_idx, norm(g.imlaey), g.uthmani.split()))
        x = x.step(1)


def locate(surah_name: str, text: str, cache: dict[int, Any]) -> tuple[list[int], float] | None:
    """[surah, ayah] or [surah, ayah, w0, w1] for the recorded text, with its match ratio."""
    s = SURAH.get(surah_name or "")
    if s is None:
        return None
    ays = cache.setdefault(s, _ayahs(s))
    q = " ".join(norm(text))
    r = lambda a: difflib.SequenceMatcher(None, q, a).ratio()  # noqa: E731
    best = max(ays, key=lambda a: r(" ".join(a[1])))
    score = r(" ".join(best[1]))
    if score >= MIN_RATIO:
        return [s, best[0]], round(score, 3)
    # a phrase of one ayah (Ayat al-Kursi): the best window of the ayah's words, when the imla'i and
    # the Uthmani word counts agree so the window carries over
    n = len(q.split())
    cands = []
    for a, ws, uth in ays:
        if len(ws) != len(uth):
            continue
        for i in range(0, len(ws) - n + 1):
            cands.append((r(" ".join(ws[i:i + n])), a, i, i + n - 1))
    if cands:
        sc, a, i, j = max(cands)
        if sc >= MIN_RATIO:
            return [s, a, i, j], round(sc, 3)
    return None


def rows() -> list[dict[str, Any]]:
    import pyarrow.parquet as pq
    out = []
    for f in sorted(glob.glob(str(DATA / "data/*.parquet"))):
        t = pq.read_table(f, columns=COLS)
        base = Path(f).stem
        for i, r in enumerate(t.to_pylist()):
            r["id"] = f"{base}-{i:05d}"
            out.append(r)
    return out


def stage() -> None:
    import pyarrow.parquet as pq
    (STAGE / "audio").mkdir(parents=True, exist_ok=True)
    cache: dict[int, Any] = {}
    loc: dict[str, Any] = {}
    meta, kept = [], 0
    with (STAGE / "manifest.jsonl").open("w") as man:
        for f in sorted(glob.glob(str(DATA / "data/*.parquet"))):
            t = pq.read_table(f)
            base = Path(f).stem
            for i, r in enumerate(t.to_pylist()):
                cid = f"{base}-{i:05d}"
                key = (r["Surah"], r["Aya"])
                if key not in loc:
                    loc[key] = locate(r["Surah"], r["Aya"], cache)
                hit = loc[key]
                a = r.pop("audio")
                m = {k: r[k] for k in COLS if k != "annotation_metadata"} | {"id": cid, "verses": None}
                if hit and (r.get("reciter_qiraah") in (None, "hafs")):
                    ext = Path(a.get("path") or "x.wav").suffix or ".wav"
                    (STAGE / "audio" / f"{cid}{ext}").write_bytes(a["bytes"])
                    man.write(json.dumps({"id": cid, "audio": f"audio/{cid}{ext}", "verses": [hit[0]]}) + "\n")
                    m["verses"], m["match"] = [hit[0]], hit[1]
                    kept += 1
                meta.append(m)
    (STAGE / "meta.jsonl").write_text("".join(json.dumps(m, ensure_ascii=False) + "\n" for m in meta))
    print(f"{kept} of {len(meta)} clips located and staged -> {STAGE}")


if __name__ == "__main__":
    {"stage": stage}[sys.argv[1]]()
