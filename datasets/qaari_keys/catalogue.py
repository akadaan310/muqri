#!/usr/bin/env python3
"""QaariKeys stage C: catalogue of open Hafs reciters and the open timing data for them.

Sources (all public, no key):
* EveryAyah (``everyayah.com/data/recitations.js``): ayah-per-file MP3 folders. Kept: Arabic Hafs
  recitations (translations, multi-language and Warsh folders dropped), one folder per reciter and
  style at the highest bitrate. Style = mujawwad / muallim / murattal (default), which sets the
  expected tempo band (mirtaba; 03_timing §calibration).
* QuranicAudio (``quranicaudio.com/api/qaris``): surah-per-file reciters; timestamps must come from
  our own alignment. Listed for the later, larger tiers.
* quran.com v4 (``api.quran.com/api/v4/resources/recitations``): 12 recitations with per-word
  segment timestamps (``/recitations/{id}/by_ayah/{s}:{a}`` → ``segments``) — the word-level
  ground truth our alignments are checked against.

Writes ``build/reciters.json``.

    .venv/bin/python datasets/qaari_keys/catalogue.py
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path

BUILD = Path(__file__).resolve().parent / "build"
EVERYAYAH_LIST = "https://everyayah.com/data/recitations.js"
QURANICAUDIO = "https://quranicaudio.com/api/qaris"
QURANCOM = "https://api.quran.com/api/v4/resources/recitations"
_DROP = re.compile(r"^(English|MultiLanguage|translations|warsh)/|Parhizgar|QuranExplorer", re.I)
# quran.com recitation id -> EveryAyah folder of the same recording (same reciter and style)
QURANCOM_TO_EVERYAYAH = {
    1: "Abdul_Basit_Mujawwad_128kbps", 2: "Abdul_Basit_Murattal_192kbps", 3: "Abdurrahmaan_As-Sudais_192kbps",
    4: "Abu_Bakr_Ash-Shaatree_128kbps", 5: "Hani_Rifai_192kbps", 6: "Husary_128kbps", 7: "Alafasy_128kbps",
    8: "Minshawy_Mujawwad_192kbps", 9: "Minshawy_Murattal_128kbps", 10: "Saood_ash-Shuraym_128kbps",
    11: "Mohammad_al_Tablaway_128kbps", 12: "Husary_Muallim_128kbps",
}


def fetch_json(url: str) -> object:
    req = urllib.request.Request(url, headers={"User-Agent": "qaari-eval-dataset/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310 - fixed public hosts
        body = r.read().decode("utf-8")
    return json.loads(body[body.index("{"):body.rindex("}") + 1] if url.endswith(".js") else body)


def _kbps(s: str) -> int:
    m = re.search(r"(\d+)\s*kbps", s, re.I)
    return int(m.group(1)) if m else 0


def _style(folder: str, name: str) -> str:
    t = f"{folder} {name}".lower()
    if "mujawwad" in t:
        return "mujawwad"
    if "muallim" in t:
        return "muallim"
    return "murattal"


def _person(name: str) -> str:
    base = re.sub(r"\b(mujawwad|murattal|muallim|\(muallim\)|ketaballah\.net|\(iran\))\b", "", name, flags=re.I)
    return re.sub(r"[^a-z]", "", base.lower().replace("minshawy", "menshawi").replace("minshawi", "menshawi"))


def everyayah() -> list[dict[str, object]]:
    data = fetch_json(EVERYAYAH_LIST)
    best: dict[tuple[str, str], dict[str, object]] = {}
    for v in data.values():  # type: ignore[union-attr]
        if not isinstance(v, dict) or "subfolder" not in v or _DROP.search(v["subfolder"]):
            continue
        rec = {"source": "everyayah", "folder": v["subfolder"], "name": v["name"],
               "style": _style(v["subfolder"], v["name"]), "kbps": _kbps(v.get("bitrate", "") or v["subfolder"])}
        key = (_person(v["name"]), str(rec["style"]))
        if key not in best or int(rec["kbps"]) > int(best[key]["kbps"]):  # type: ignore[call-overload]
            best[key] = rec
    return sorted(best.values(), key=lambda r: str(r["folder"]))


def main() -> int:
    BUILD.mkdir(exist_ok=True)
    ea = everyayah()
    qa = fetch_json(QURANICAUDIO)
    qa = qa if isinstance(qa, list) else qa.get("qaris", [])  # type: ignore[union-attr]
    qc = fetch_json(QURANCOM)["recitations"]  # type: ignore[index]
    for r in ea:
        r["qurancom_id"] = next((k for k, f in QURANCOM_TO_EVERYAYAH.items() if f == r["folder"]), None)
    out = {"everyayah": ea,
           "quranicaudio": [{"id": q["id"], "name": q["name"], "path": q.get("relative_path"),
                             "section": q.get("section_id")} for q in qa],
           "qurancom": [{"id": q["id"], "name": q["reciter_name"], "style": q.get("style"),
                         "everyayah": QURANCOM_TO_EVERYAYAH.get(q["id"])} for q in qc]}
    (BUILD / "reciters.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    styles: dict[str, int] = {}
    for r in ea:
        styles[str(r["style"])] = styles.get(str(r["style"]), 0) + 1
    print(f"EveryAyah Hafs reciter-styles {len(ea)} {styles}; QuranicAudio {len(out['quranicaudio'])}; "
          f"quran.com timed {len(qc)} ({sum(r['qurancom_id'] is not None for r in ea)} mapped to EveryAyah)")
    for r in ea:
        print(f"  {r['folder']:45s} {r['style']:9s} {r['kbps']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
