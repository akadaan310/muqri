"""Every rā' in a set of master recitations, with what the reference expects, what the model hears,
and its full context -- for the rā' calculus (substrate_library/julia/ra.jl).

    .venv/bin/python -m research_agency_lab.experiments.quran.ra_extract
"""

from __future__ import annotations

import json
import random
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "research_agency_lab/experiments/quran/ra_instances.jsonl"
SOURCES = [ROOT / "research_agency_lab/experiments/qaari_keys/modal_Q54",
           ROOT / "research_agency_lab/experiments/qaari_keys/modal_T300"]


def clips() -> list[tuple[Path, dict]]:  # type: ignore[type-arg]
    out = []
    for d in SOURCES:
        recs = [json.loads(l) for l in (d / "index.jsonl").open()]
        recs = [r for r in recs if "file" in r and "ر" in r.get("ref_ph", "")]
        if d.name == "modal_T300":
            final = [r for r in recs if r["ref_ph"].rstrip("ڇ").endswith("ر")]
            other = [r for r in recs if r not in final]
            random.Random(11).shuffle(other)
            recs = final + other[:1000]
        out += [(d, r) for r in recs]
    return out


def main() -> None:
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(ROOT))
    from app.engine import Engine
    work = clips()
    done = {(json.loads(l)["source"], json.loads(l)["clip"]) for l in OUT.open()} if OUT.is_file() else set()
    engines: dict[str, Engine] = {}
    with OUT.open("a") as fh:
        for d, r in work:
            if (d.name, r["id"]) in done:
                continue
            lay = json.loads((d / "layout.json").read_text())
            eng = engines.setdefault(d.name, Engine(layout=lay))
            lp = np.fromfile(d / r["file"], dtype="<f4").reshape(r["frames"], lay["columns"])
            try:
                m = eng.analyze(None, [(r["sura"], r["aya"])], posteriors=lp)["measurements"]
            except Exception:  # noqa: BLE001
                continue
            L = m["letters"]
            words = {int(w["ref"].split(":")[2]): w["text"] for w in m["words"]}
            last_cons = max((i for i, l in enumerate(L) if l["kind"] == "consonant"), default=-1)
            rows = []
            for i, l in enumerate(L):
                if l["symbol"] != "ر" or "tafkheem_or_taqeeq" not in l["characteristics"]:
                    continue
                t = l["characteristics"]["tafkheem_or_taqeeq"]
                prev = [x["symbol"] for x in L[max(0, i - 4):i]]
                rows.append({"source": d.name, "clip": r["id"], "speaker": r["speaker"], "surah": r["sura"], "ayah": r["aya"],
                             "word": words.get(l["word"], ""), "letter_i": i, "ayah_final": i == last_cons,
                             "run_length": l["run_length"], "next": L[i + 1]["symbol"] if i + 1 < len(L) else "$",
                             "prev": prev, "expected": t["expected"], "observed": t["observed"], "margin": t["margin"],
                             "realised": t["realised"]})
            fh.write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows) or
                     json.dumps({"source": d.name, "clip": r["id"], "none": True}) + "\n")
            fh.flush()


if __name__ == "__main__":
    main()
