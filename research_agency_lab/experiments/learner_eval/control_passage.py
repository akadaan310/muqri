"""Masters as the control group: how does the engine grade professional reciters on one passage?

A learner's report is only as trustworthy as the engine's false-alarm rate on people who recite
correctly. The T300 dump holds posteriors for the same ayahs from 41 professional reciters, so any
passage they all recited is a ready-made control: every error the engine reports on them is either a
genuine slip by a master (rare) or a false alarm (the thing to fix).

Posteriors come from the dump (no model load); where the EveryAyah audio is cached it is passed too,
trimmed to the dump's frame count, so stops are detected from the waveform exactly as in serving.

    python research_agency_lab/experiments/learner_eval/control_passage.py SURAH A1 A2 [OUT.json]
"""
import json
import os
import sys
import warnings
from collections import Counter, defaultdict

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from app.engine import Engine  # noqa: E402

D = "research_agency_lab/experiments/qaari_keys/modal_T300"
AUDIO = os.path.expanduser("~/.cache/qaari-eval/everyayah")
SR, HOP = 16000, 640          # 40 ms frames


def main(surah: int, a1: int, a2: int, out: str | None = None) -> None:
    import librosa
    lay = json.load(open(f"{D}/layout.json"))
    clips = defaultdict(dict)
    for line in open(f"{D}/index.jsonl"):
        r = json.loads(line)
        if "file" in r and r.get("sura") == surah and a1 <= r.get("aya", 0) <= a2:
            clips[r["speaker"]][r["aya"]] = r
    eng = Engine(layout=lay)
    verses = [(surah, a) for a in range(a1, a2 + 1)]
    rows, tally = {}, defaultdict(Counter)
    for spk, by in sorted(clips.items()):
        if set(by) != {a for _s, a in verses}:
            continue
        recs = [by[a] for _s, a in verses]
        lp = np.concatenate([np.fromfile(f"{D}/{r['file']}", dtype="<f4").reshape(r["frames"], lay["columns"])
                             for r in recs])
        waves = []
        for r in recs:
            mp3 = f"{AUDIO}/{spk}/{r['sura']:03d}{r['aya']:03d}.mp3"
            if not os.path.exists(mp3):
                waves = None
                break
            w = librosa.load(mp3, sr=SR, mono=True)[0][: r["frames"] * HOP]
            waves.append(np.pad(w, (0, r["frames"] * HOP - w.size)))
        audio = np.concatenate(waves) if waves else None
        rep = eng.analyze(audio, verses, posteriors=lp)
        errs = [(e["rule"], e["word"], e["status"], (e.get("evidence") or {}).get("given_counts"))
                for e in rep["errors"]]
        rows[spk] = {"accuracy": rep["summary"]["accuracy"], "errors": errs,
                     "tempo": rep["mastery"]["tempo_haraka_s"], "audio": audio is not None}
        for a in rep["ayahs"]:
            for v in a["rules"]:
                tally[(v["rule"], v["word"])][v["status"]] += 1
        print(f"{spk:42s} acc={rep['summary']['accuracy']}  tempo={rep['mastery']['tempo_haraka_s']}"
              f"  audio={'y' if audio is not None else 'n'}  errors={errs}", flush=True)
    print("\nPER RULE INSTANCE (status counts over reciters)")
    for (rule, word), c in sorted(tally.items(), key=lambda kv: -sum(v for k, v in kv[1].items() if k != "pass")):
        print(f"  {rule:22s} {word:14s} {dict(c)}")
    if out:
        json.dump({"verses": verses, "reciters": rows,
                   "tally": {f"{r}|{w}": dict(c) for (r, w), c in tally.items()}},
                  open(out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:4]), *(sys.argv[4:5]))
