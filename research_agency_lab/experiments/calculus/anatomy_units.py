"""Consonant timings + waveforms for the anatomy measurement (substrate_library/julia/anatomy.jl).

Masters first -- Husary (three recordings) and Minshawy (two), the calibration the tradition itself
points to for collision, separation and qalqalah -- beside three fast imams. Each consonant's span
comes from the Viterbi alignment of the T300 posteriors; its audio is the EveryAyah verse (the same
file the posteriors were computed from, frame drift 0).

Context, per the anatomy of sound: 'voweled' (a separation), 'sakin' (a collision), 'shaddah' (both),
'stop' (the last consonant of the ayah -- where qalqalah is strongest), and 'shaddah_stop'.

    .venv/bin/python research_agency_lab/experiments/calculus/anatomy_units.py [clips_per_reciter]
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from app.analysis import MADD, SHORT_V, ctc_viterbi  # noqa: E402
from app.rule_bind import ph_units  # noqa: E402
from datastore.review_queue import AUDIO  # noqa: E402

DUMP = ROOT / "research_agency_lab/experiments/qaari_keys/modal_T300"
OUT = ROOT / "research_agency_lab/experiments/calculus/data/anatomy"
RECITERS = ["Husary_128kbps", "Husary_Muallim_128kbps", "Husary_128kbps_Mujawwad",
            "Minshawy_Murattal_128kbps", "Minshawy_Mujawwad_192kbps",
            "Saood_ash-Shuraym_128kbps", "Abdurrahmaan_As-Sudais_192kbps", "MaherAlMuaiqly128kbps"]
LETTERS = set("ءبتثجحخدذرزسشصضطظعغفقكلمنهوي")


def context(units, i):  # type: ignore[no-untyped-def]
    sym, a, b = units[i]
    last_cons = all(u[0] in SHORT_V or u[0] in MADD for u in units[i + 1:])
    doubled = b - a + 1 >= 2
    if last_cons:
        return "shaddah_stop" if doubled else "stop"
    if doubled:
        return "shaddah"
    nxt = units[i + 1][0] if i + 1 < len(units) else None
    return "voweled" if nxt in SHORT_V or nxt in MADD else "sakin"


def main(per: int = 60) -> None:
    import librosa
    lay = json.loads((DUMP / "layout.json").read_text())
    ph = next(lv for lv in lay["levels"] if lv["level"] == "phonemes")
    vocab = {t: i for i, t in enumerate(ph["vocab"]) if len(t) == 1}
    OUT.mkdir(parents=True, exist_ok=True)
    taken = collections.Counter()
    units_out, k = [], 0
    for line in open(DUMP / "index.jsonl"):
        r = json.loads(line)
        spk = r.get("speaker")
        if "file" not in r or spk not in RECITERS or taken[spk] >= per:
            continue
        mp3 = AUDIO / spk / f"{r['sura']:03d}{r['aya']:03d}.mp3"
        if not mp3.is_file() or not all(c in vocab for c in r["ref_ph"]):
            continue
        lp = np.fromfile(DUMP / r["file"], dtype="<f4").reshape(r["frames"], lay["columns"])
        lp = lp[:, ph["first"]:ph["first"] + ph["width"]]
        _s, first, last = ctc_viterbi(lp, [vocab[c] for c in r["ref_ph"]], lay["blank"])
        w = librosa.load(str(mp3), sr=16000, mono=True)[0][: r["frames"] * 640].astype("<f4")
        w.tofile(OUT / f"wave_{k}.f32")
        us = ph_units(r["ref_ph"])
        for i, (sym, a, b) in enumerate(us):
            if sym in LETTERS:
                units_out.append({"wave": k, "t0": round((first[a] - 1) * 0.04, 3), "t1": round(last[b] * 0.04, 3),
                                  "letter": sym, "context": context(us, i), "speaker": spk,
                                  "clip": r["id"]})
        taken[spk] += 1
        k += 1
    (OUT / "units.json").write_text(json.dumps(units_out, ensure_ascii=False))
    print(f"{k} clips, {len(units_out)} consonants -> {OUT}  ({dict(taken)})")


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:]))
