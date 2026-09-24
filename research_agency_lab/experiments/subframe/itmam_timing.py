"""Does sub-frame timing lift the itmam inversion, or is it the pause confound?

`app/itmam.py` ranks an anchor BELOW a fast imam (ikhtilas 0.239 vs 0.145, run collapse 0.533 vs
0.367 on six ayahs). Two suspects: 40 ms quantisation (a vowel reads 1 or 2 frames, so the
two-thirds threshold is crossed by rounding), and pauses (onset-to-onset charges a vowel for the
silence after it). This runs the real `analyse_clip` + `itmam` + `sequences` on the ladder in both
timing modes. If the inversion survives the centroid, quantisation is excluded.

    python itmam_timing.py [clips_per_reciter]
"""
import json
import sys

import numpy as np

sys.path.insert(0, ".")
from app.analysis import analyse_clip  # noqa: E402
from app.itmam import itmam, roll_up, sequences  # noqa: E402

D = "research_agency_lab/experiments/qaari_keys/modal_T300"
LADDER = ["Husary_Muallim_128kbps", "Husary_128kbps", "Husary_128kbps_Mujawwad",
          "Minshawy_Mujawwad_192kbps", "Saood_ash-Shuraym_128kbps", "Abdurrahmaan_As-Sudais_192kbps",
          "MaherAlMuaiqly128kbps"]


def main(per=40):
    lay = json.load(open(f"{D}/layout.json"))
    ph = next(lv for lv in lay["levels"] if lv["level"] == "phonemes")
    vocab = {t: i for i, t in enumerate(ph["vocab"]) if len(t) == 1}
    recs = {s: [] for s in LADDER}
    for line in open(f"{D}/index.jsonl"):
        r = json.loads(line)
        if "file" in r and r.get("speaker") in recs and len(recs[r["speaker"]]) < per:
            recs[r["speaker"]].append(r)
    print(f"{'reciter':34s} {'mode':9s} {'n':>5s} {'ikhtilas':>9s} {'ishba':>7s} "
          f"{'alt.collapse':>12s} {'dmm.collapse':>12s}")
    for spk in LADDER:
        clips = []
        for r in recs[spk]:
            raw = np.fromfile(f"{D}/{r['file']}", dtype="<f4").reshape(r["frames"], lay["columns"])
            clips.append((raw, r["ref_ph"]))
        for mode in ("viterbi", "centroid"):
            ua = [analyse_clip(raw, p, vocab, lay["blank"], ph["first"], ph["width"], {}, None,
                               timing=mode) for raw, p in clips]
            it = itmam(ua)
            ru = roll_up([sequences(u) for u in ua])
            col = {k: (ru.get(k) or {}).get("collapse_rate") for k in ("alternating", "consecutive_damma")}
            print(f"{spk:34s} {mode:9s} {it['n']:5d} {it['ikhtilas_rate']:9.3f} {it['ishba_rate']:7.3f} "
                  f"{col['alternating'] or 0:12.3f} {col['consecutive_damma'] or 0:12.3f}", flush=True)


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:]))
