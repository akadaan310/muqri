"""Is the isochrony spread a skill signal or a tempo artefact?

Sprint 1 found fast imams with a SMALLER vowel spread (2.12 %) than the anchors (3.84 %). If spread
were shipped as an isochrony score, Shuraym would outscore Husary. The hypothesis is compression:
at speed every vowel is driven toward a floor, so fatha/damma/kasra converge for the wrong reason.

Two tests, each able to refute it:

* ACROSS reciters (41): does relative spread fall with tempo (haraka unit in seconds)?
* WITHIN reciter: split each reciter's own clips into tempo terciles. A reciter's skill does not
  change between their fast and slow ayahs, so if spread still falls with tempo inside one voice,
  tempo is driving it.

Stage 1 caches per-clip vowel measurements (`extract`), stage 2 analyses them (`analyse`).

    python tempo_test.py extract OUT.jsonl [clips_per_reciter] [procs]
    python tempo_test.py analyse OUT.jsonl
    python tempo_test.py null OUT.jsonl [permutations]
"""
import json
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "research_agency_lab/experiments/subframe")
from occupancy import centroid_onsets  # noqa: E402

from app.rule_bind import ph_units  # noqa: E402

D = "research_agency_lab/experiments/qaari_keys/modal_T300"
FRAME_S = 0.04
V = {"َ": "fatha", "ُ": "damma", "ِ": "kasra"}
MADD = set("اۥۦ")
ANCH = {"Husary_Muallim_128kbps", "Husary_128kbps", "Husary_128kbps_Mujawwad"}
FAST = {"Saood_ash-Shuraym_128kbps", "MaherAlMuaiqly128kbps", "Abdurrahmaan_As-Sudais_192kbps"}

_lay = json.load(open(f"{D}/layout.json"))
_pl = [lv for lv in _lay["levels"] if lv["level"] == "phonemes"][0]
_vocab = {t: i for i, t in enumerate(_pl["vocab"]) if len(t) == 1}


def one_clip(r):
    lp = np.fromfile(f"{D}/{r['file']}", dtype="<f4").reshape(r["frames"], _lay["columns"])
    lp = lp[:, _pl["first"]:_pl["first"] + _pl["width"]]
    ph = r["ref_ph"]
    units = ph_units(ph)
    nu = len(units)
    cen, _ = centroid_onsets(lp, [_vocab[c] for c in ph], _lay["blank"])
    pos = np.array([cen[units[i][1]] for i in range(nu)])
    if nu < 8 or np.any(~np.isfinite(pos)):
        return None
    dur = np.diff(pos)
    har = [pos[i + 2] - pos[i] for i in range(nu - 2)
           if units[i][0] not in V and units[i][0] not in MADD and units[i + 1][0] in V]
    har = [h for h in har if h > 0]
    if len(har) < 5:
        return None
    # the last unit has no successor onset, so it is never measured
    vs = [(V[units[i][0]], float(dur[i])) for i in range(nu - 1) if units[i][0] in V and dur[i] > 0]
    return {"speaker": r["speaker"], "id": r["id"], "haraka_frames": float(np.median(har)), "vowels": vs}


def extract(out, per=150, procs=3):
    recs, seen = [], {}
    for line in open(f"{D}/index.jsonl"):
        r = json.loads(line)
        if "file" not in r:
            continue
        k = seen.get(r["speaker"], 0)
        if k < per:
            seen[r["speaker"]] = k + 1
            recs.append(r)
    with Pool(procs) as p, open(out, "w") as f:
        for x in p.imap_unordered(one_clip, recs, chunksize=4):
            if x:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")


def spread(clips, norm="haraka"):
    """Relative spread of the three vowel medians, pooled over `clips`."""
    per = {"fatha": [], "damma": [], "kasra": []}
    for c in clips:
        h = c["haraka_frames"] if norm == "haraka" else 1.0
        for v, d in c["vowels"]:
            per[v].append(d / h)
    if min(len(x) for x in per.values()) < 20:
        return None
    m = {k: float(np.median(x)) for k, x in per.items()}
    return (max(m.values()) - min(m.values())) / np.mean(list(m.values())), m


def analyse(path):
    by = {}
    for line in open(path):
        c = json.loads(line)
        by.setdefault(c["speaker"], []).append(c)

    print("ACROSS RECITERS -- relative spread vs tempo (haraka unit, seconds)")
    rows = []
    for spk, cl in sorted(by.items()):
        s = spread(cl)
        if s is None:
            continue
        h = float(np.median([c["haraka_frames"] for c in cl])) * FRAME_S
        tag = "ANCHOR" if spk in ANCH else "FAST" if spk in FAST else ""
        rows.append((h, s[0], spk, tag))
    for h, s, spk, tag in sorted(rows):
        print(f"  {h:.3f}s  spread={100 * s:5.2f}%  {spk:42s} {tag}")
    hs, ss = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
    rho = float(np.corrcoef(np.argsort(np.argsort(hs)), np.argsort(np.argsort(ss)))[0, 1])
    print(f"  Spearman(tempo, spread) over {len(rows)} reciters = {rho:+.3f}")

    print("\nWITHIN RECITER -- spread by the reciter's own tempo tercile (fast / mid / slow)")
    diffs = []
    for spk, cl in sorted(by.items()):
        cl = sorted(cl, key=lambda c: c["haraka_frames"])
        n = len(cl) // 3
        parts = [cl[:n], cl[n:2 * n], cl[2 * n:]]
        sp = [spread(p) for p in parts]
        if any(x is None for x in sp):
            continue
        diffs.append(sp[2][0] - sp[0][0])
        print(f"  {spk:42s} " + "  ".join(f"{100 * x[0]:5.2f}%" for x in sp))
    diffs = np.array(diffs)
    rng = np.random.default_rng(7)
    boot = np.median(rng.choice(diffs, (4000, len(diffs))), axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  slow-minus-fast spread, median over {len(diffs)} reciters = {100 * np.median(diffs):+.2f} pp"
          f"  95% CI [{100 * lo:+.2f}, {100 * hi:+.2f}]  ({int((diffs > 0).sum())}/{len(diffs)} positive)")


def equalised_spread(parts, rng, reps=40):
    """Spread per part with every part subsampled to the same vowel count per vowel.

    Spread is a max-minus-min of three medians, so it grows with noise: a part with fewer vowels
    reads a larger spread for no reason. Equalising n removes that bias before parts are compared.
    """
    pools = []
    for part in parts:
        per = {"fatha": [], "damma": [], "kasra": []}
        for c in part:
            for v, d in c["vowels"]:
                per[v].append(d / c["haraka_frames"])
        pools.append({k: np.array(x) for k, x in per.items()})
    n = {k: min(len(p[k]) for p in pools) for k in ("fatha", "damma", "kasra")}
    if min(n.values()) < 20:
        return None
    out = np.zeros(len(parts))
    for _ in range(reps):
        for j, p in enumerate(pools):
            m = [np.median(rng.choice(p[k], n[k], replace=False)) for k in n]
            out[j] += (max(m) - min(m)) / np.mean(m)
    return out / reps


def null_test(path, perms=200):
    """Slow-minus-fast spread with equal n, against clips assigned to terciles at random."""
    by = {}
    for line in open(path):
        c = json.loads(line)
        by.setdefault(c["speaker"], []).append(c)
    rng = np.random.default_rng(11)

    def stat(order_fn):
        d = []
        for cl in by.values():
            cl = order_fn(cl)
            n = len(cl) // 3
            e = equalised_spread([cl[:n], cl[2 * n:]], rng, reps=10)
            if e is not None:
                d.append(e[1] - e[0])
        return float(np.median(d)), d

    obs, d = stat(lambda cl: sorted(cl, key=lambda c: c["haraka_frames"]))
    print(f"EQUAL-n slow-minus-fast spread: median {100 * obs:+.2f} pp over {len(d)} reciters "
          f"({sum(x > 0 for x in d)}/{len(d)} positive)")
    null = [stat(lambda cl: list(rng.permutation(cl)))[0] for _ in range(perms)]
    pval = (1 + sum(abs(x) >= abs(obs) for x in null)) / (1 + perms)
    print(f"  permutation null (random terciles): median {100 * np.median(null):+.2f} pp, "
          f"95% [{100 * np.percentile(null, 2.5):+.2f}, {100 * np.percentile(null, 97.5):+.2f}]  p = {pval:.3f}")


if __name__ == "__main__":
    if sys.argv[1] == "extract":
        extract(sys.argv[2], *(int(a) for a in sys.argv[3:]))
    elif sys.argv[1] == "null":
        null_test(sys.argv[2], *(int(a) for a in sys.argv[3:]))
    else:
        analyse(sys.argv[2])
