"""Is the vowel-run collapse inversion caused by pauses?

`itmam_timing.py` showed the inversion survives sub-frame timing: anchors' alternating-vowel runs
"collapse" at 0.43-0.46 against 0.26-0.29 for the fast imams, so quantisation is excluded. The
remaining suspect is pauses: onset-to-onset duration charges a vowel for any silence after it, and a
run that spans a breath looks like a long vowel followed by short ones.

Silence is taken from the WAVEFORM (`app.waqf.silent_frames`), never from the posteriors. Each run
is labelled by whether any silent frame falls inside it; collapse rates are then compared on the
pause-free runs alone. If the anchors still collapse more there, pauses are excluded too.

    python itmam_pause.py [clips_per_reciter]
"""
import json
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, ".")
from app.analysis import analyse_clip  # noqa: E402
from app.itmam import _vowel_runs, sequences  # noqa: E402
from app.waqf import silent_frames  # noqa: E402

warnings.filterwarnings("ignore")
D = "research_agency_lab/experiments/qaari_keys/modal_T300"
AUDIO = os.path.expanduser("~/.cache/qaari-eval/everyayah")
LADDER = ["Husary_Muallim_128kbps", "Husary_128kbps", "Husary_128kbps_Mujawwad",
          "Minshawy_Mujawwad_192kbps", "Saood_ash-Shuraym_128kbps", "Abdurrahmaan_As-Sudais_192kbps",
          "MaherAlMuaiqly128kbps"]


def main(per=100):
    import librosa
    lay = json.load(open(f"{D}/layout.json"))
    ph = next(lv for lv in lay["levels"] if lv["level"] == "phonemes")
    vocab = {t: i for i, t in enumerate(ph["vocab"]) if len(t) == 1}
    recs = {s: [] for s in LADDER}
    for line in open(f"{D}/index.jsonl"):
        r = json.loads(line)
        if "file" not in r or r.get("speaker") not in recs or len(recs[r["speaker"]]) >= per:
            continue
        spk, aya = r["id"].split(":")[1].split("/")
        mp3 = f"{AUDIO}/{spk}/{aya}.mp3"
        if os.path.exists(mp3):
            recs[r["speaker"]].append((r, mp3))

    print(f"{'reciter':34s} {'clips':>5s} {'frame|diff|':>11s} {'runs':>5s} {'paused':>7s} "
          f"{'collapse(all)':>13s} {'collapse(no pause)':>18s} {'collapse(paused)':>16s}")
    out = {}
    for spk in LADDER:
        tally = {"clean": [0, 0], "paused": [0, 0]}
        drift = []
        for r, mp3 in recs[spk]:
            raw = np.fromfile(f"{D}/{r['file']}", dtype="<f4").reshape(r["frames"], lay["columns"])
            units = analyse_clip(raw, r["ref_ph"], vocab, lay["blank"], ph["first"], ph["width"], {},
                                 None, timing="centroid")
            sil = silent_frames(librosa.load(mp3, sr=16000, mono=True)[0])
            drift.append(abs(len(sil) - r["frames"]))
            span = {(k, st): idxs for k, st, idxs in _vowel_runs(units)}
            for s in sequences(units):
                if s.kind != "alternating":
                    continue
                # through the unit after the run's last vowel: its onset closes that vowel's duration
                last = min(span[(s.kind, s.start_unit)][-1] + 1, len(units) - 1)
                lo, hi = units[s.start_unit].frames[0] - 1, units[last].frames[1]
                key = "paused" if sil[max(0, lo):hi].any() else "clean"
                tally[key][0] += 1
                tally[key][1] += int(s.collapsing)
        n = tally["clean"][0] + tally["paused"][0]
        rate = {k: (v[1] / v[0] if v[0] else float("nan")) for k, v in tally.items()}
        allr = (tally["clean"][1] + tally["paused"][1]) / n if n else float("nan")
        out[spk] = {"runs": n, "paused": tally["paused"][0], "collapse_all": round(allr, 4),
                    "collapse_clean": round(rate["clean"], 4), "collapse_paused": round(rate["paused"], 4),
                    "frame_drift_median": float(np.median(drift))}
        print(f"{spk:34s} {len(recs[spk]):5d} {np.median(drift):11.1f} {n:5d} {tally['paused'][0]:7d} "
              f"{allr:13.3f} {rate['clean']:18.3f} {rate['paused']:16.3f}", flush=True)
    json.dump(out, open("research_agency_lab/experiments/qaari_keys/itmam_pause_T300.json", "w"), indent=1)


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:]))
