"""obadx/qdat_bench: 159 readers of the end of al-Ma'idah 5:109 (قَالُوا۟ ... ٱلْغُيُوبِ, words 7-14),
each labelled by hand with the length of six madd in counts, the ghunnah of the noon mushaddadah
(partial / complete) and of the ikhfa noon (plain noon / partial / complete), and qalqalah on the
final ب.

    python -m research_agency_lab.experiments.external.qdat_bench run      # clips -> engine (webapp)
    python -m research_agency_lab.experiments.external.qdat_bench compare  # engine vs labels

Data: https://huggingface.co/datasets/obadx/qdat_bench -> DATA (labels.jsonl + <id>.wav, unpacked
from the parquet).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DATA = Path.home() / "data/sprint4/qdat_bench"
HERE = Path(__file__).resolve().parent
OUT = HERE / "qdat_bench_engine.jsonl"
VERSES = [[5, 109, 7, 14]]


def labels() -> list[dict]:  # type: ignore[type-arg]
    return [json.loads(l) for l in (DATA / "labels.jsonl").open()]


def run() -> None:
    from research_agency_lab.experiments.external.runner import run as go
    n = go(({"id": r["id"], "path": DATA / r["file"], "verses": VERSES} for r in labels()), OUT)
    print(f"{n} clips scored -> {OUT}")


# The labelled places, as the engine names them: (label, rule, word, nth instance of that rule in the word)
MADD = {"qalo_alif_len": ("madd_tabii", 7, 0), "qalo_waw_len": ("madd_tabii", 7, 1),
        "laa_alif_len": ("madd_tabii", 8, 0), "separate_madd": ("madd_munfasil", 10, 0),
        "allam_alif_len": ("madd_tabii", 13, 0), "madd_aared_len": ("madd_arid_lissukun", 14, 0)}
# what Hafs (al-Shatibiyyah or the Tayyibah) allows at each place, in counts
ALLOWED = {"qalo_alif_len": {2}, "qalo_waw_len": {2}, "laa_alif_len": {2}, "allam_alif_len": {2},
           "separate_madd": {2, 4, 5}, "madd_aared_len": {2, 4, 6}}


def auc(pos: list[float], neg: list[float]) -> float | None:
    """P(a positive scores above a negative), ties half: the threshold-free separation of two groups
    by one number. 0.5 is chance -- the baseline every measure here must beat."""
    if not pos or not neg:
        return None
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return round(wins / (len(pos) * len(neg)), 3)


def _rule(m: dict, rule: str, word: int, nth: int) -> dict | None:  # type: ignore[type-arg]
    hits = sorted((r for r in m["rules"] if r["rule"] == rule and r["word"] == word),
                  key=lambda r: int(r["letters"][0].rsplit("L", 1)[1]) if r["letters"] else 0)
    return hits[nth] if nth < len(hits) else None


def _letter(m: dict, word: int, kind_or_symbol: str) -> dict | None:  # type: ignore[type-arg]
    return next((l for l in m["letters"] if l["word"] == word and kind_or_symbol in (l["kind"], l["symbol"])), None)


def compare() -> dict:  # type: ignore[type-arg]
    import numpy as np
    from scipy.stats import pearsonr, spearmanr
    lab = {r["id"]: r for r in labels()}
    eng = {r["id"]: r for r in map(json.loads, OUT.open()) if "measurements" in r}
    ids = [i for i in lab if i in eng]
    out: dict = {"clips": len(ids), "errors": len(lab) - len(ids), "madd": {}, "ghunnah": {}, "qalqalah": {}}  # type: ignore[type-arg]

    for key, (rule, word, nth) in MADD.items():
        pairs, verdict = [], {"tp": 0, "fn": 0, "fp": 0, "tn": 0, "not_scored": 0}
        for i in ids:
            r = _rule(eng[i]["measurements"], rule, word, nth)
            if r is None or r["observed_counts"] is None:
                verdict["not_scored"] += 1
                continue
            y = lab[i][key]
            pairs.append((y, r["observed_counts"]))
            bad_label, bad_engine = y not in ALLOWED[key], r["status"] in {"short", "long"}
            verdict[("tp" if bad_engine else "fn") if bad_label else ("fp" if bad_engine else "tn")] += 1
        y, x = map(np.array, zip(*pairs))
        out["madd"][key] = {
            "n": len(pairs), "label_counts": {int(k): int((y == k).sum()) for k in np.unique(y)},
            "spearman": round(float(spearmanr(y, x)[0]), 3) if len(set(y)) > 1 else None,
            "pearson": round(float(pearsonr(y, x)[0]), 3) if len(set(y)) > 1 else None,
            "mae_counts": round(float(np.abs(x - y).mean()), 3), "bias_counts": round(float((x - y).mean()), 3),
            "engine_median_by_label": {int(k): round(float(np.median(x[y == k])), 2) for k in np.unique(y)},
            "verdict_vs_label": verdict}

    # ghunnah: the noon mushaddadah of إِنَّكَ (0 partial / 1 complete) and the ikhfa noon of أَنتَ
    # (0 read as a plain noon / 1 partial / 2 complete)
    for key, rule, word, kind in (("noon_moshaddadah_len", "ghunnah", 11, "ن"), ("noon_mokhfah_len", "ikhfa", 12, "ikhfa_noon")):
        feats: dict[str, list[tuple[int, float]]] = {"counts": [], "ghonna_margin": [], "identity_margin": []}
        verdict = {"label_full": {}, "label_not_full": {}}
        for i in ids:
            m, y = eng[i]["measurements"], lab[i][key]
            full = y == (1 if key == "noon_moshaddadah_len" else 2)
            r = _rule(m, rule, word, 0)
            if r and r["observed_counts"] is not None:
                feats["counts"].append((full, r["observed_counts"]))
            st = r["status"] if r else "absent"
            d = verdict["label_full" if full else "label_not_full"]
            d[st] = d.get(st, 0) + 1
            l = _letter(m, word, kind)
            if l:
                g = l["characteristics"].get("ghonna")
                if g and g["margin"] is not None:
                    feats["ghonna_margin"].append((full, g["margin"]))
                if l["identity"]["margin"] is not None:
                    feats["identity_margin"].append((full, l["identity"]["margin"]))
        out["ghunnah"][key] = {
            "labels": {int(k): sum(lab[i][key] == k for i in ids) for k in sorted({lab[i][key] for i in ids})},
            "engine_status": verdict,
            "auc_full_vs_not": {f: auc([v for p, v in xs if p], [v for p, v in xs if not p]) for f, xs in feats.items()},
            "n": {f: len(xs) for f, xs in feats.items()}}

    # qalqalah on the final ب of ٱلْغُيُوبِ
    marg, verdict = [], {"label_yes": {}, "label_no": {}}
    for i in ids:
        m, y = eng[i]["measurements"], lab[i]["qalqalah"]
        l = _letter(m, 14, "ب")
        q = l["characteristics"].get("qalqla") if l else None
        if q and q["margin"] is not None:
            marg.append((y == 1, q["margin"]))
        r = _rule(m, "qalqalah", 14, 0)
        d = verdict["label_yes" if y == 1 else "label_no"]
        st = r["status"] if r else "absent"
        d[st] = d.get(st, 0) + 1
    out["qalqalah"] = {"labels": {"yes": sum(lab[i]["qalqalah"] == 1 for i in ids), "no": sum(lab[i]["qalqalah"] == 0 for i in ids)},
                       "engine_status": verdict,
                       "auc_qalqla_margin": auc([v for p, v in marg if p], [v for p, v in marg if not p]), "n": len(marg)}
    el = [eng[i]["elapsed_seconds"] for i in ids]
    au = [eng[i]["audio_seconds"] for i in ids]
    out["latency"] = {"median_s": round(float(np.median(el)), 2), "p90_s": round(float(np.percentile(el, 90)), 2),
                      "median_audio_s": round(float(np.median(au)), 2),
                      "median_real_time_factor": round(float(np.median(np.array(el) / np.array(au))), 3)}
    return out


def main() -> None:
    if sys.argv[1] == "run":
        run()
    else:
        res = compare()
        (HERE / "qdat_bench_compare.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
        print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
