"""Anatomy measurements that an expert should hear: phenomena and mistake-like departures.

Selected from anatomy.jl's per-consonant acts (masters and fast imams), a few per pattern, largest
magnitude first, and pushed into the listening review:

  latent:voiced_hold      ب / د sakin or doubled whose voicing continues through the closure
  anatomy:release_on_hams  a sakin ت / ك released with a bounce (masters' echo there is ~1.7 dB)
  anatomy:weak_qalqalah    a qalqalah letter at the stop with almost no echo
  anatomy:shaddah_unclear  a doubled stop whose collision or separation falls far short of the masters'

    .venv/bin/python research_agency_lab/experiments/calculus/anatomy_findings.py [per_pattern]
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from datastore.review_queue import unit_candidates  # noqa: E402

DATA = ROOT / "research_agency_lab/experiments/calculus/data"
MASTERS = {"Husary_128kbps", "Husary_Muallim_128kbps", "Husary_128kbps_Mujawwad",
           "Minshawy_Murattal_128kbps", "Minshawy_Mujawwad_192kbps"}
QALQALAH = set("قطبجد")


def main(per: int = 6) -> None:
    units = json.loads((DATA / "anatomy" / "units.json").read_text())
    acts = json.loads((DATA / "anatomy_masters.json").read_text())["units"]
    # anatomy.jl skips a few consonants (t0 < 60 ms); re-pair acts with their unit by walking both lists
    rows, j = [], 0
    for u in units:
        if j < len(acts) and acts[j]["letter"] == u["letter"] and acts[j]["speaker"] == u["speaker"] \
                and acts[j]["context"] == u["context"] and u["t0"] >= 0.06:
            rows.append({**u, **acts[j]})
            j += 1
    master = lambda l, c, f: statistics.median([r[f] for r in rows if r["speaker"] in MASTERS  # noqa: E731
                                                and r["letter"] == l and r["context"] == c and r[f] == r[f]] or [float("nan")])
    picks = []

    def take(sel, key, n, make):  # type: ignore[no-untyped-def]
        for r in sorted(sel, key=key)[:n]:
            picks.append({**r, **make(r)})

    take([r for r in rows if r["letter"] in "بد" and r["context"] in ("sakin", "shaddah") and r["hold_db"] >= 2],
         lambda r: -r["hold_db"], per,
         lambda r: {"kind": "phenomenon", "detector": f"latent:voiced_hold:{r['letter']}", "severity": "phenomenon",
                    "magnitude": round(r["hold_db"], 2),
                    "claim": f"voiced hold on the {r['letter']} ({r['context']}): voicing continues through the closure "
                             f"({r['hold_db']:+.1f} dB against the vowel before). Not a mistake — is the effect there?"})
    take([r for r in rows if r["letter"] in "تك" and r["context"] == "sakin" and r["echo_db"] >= 8],
         lambda r: -r["echo_db"], per,
         lambda r: {"kind": "characteristic", "detector": "anatomy:release_on_hams", "severity": "moderate",
                    "magnitude": round(r["echo_db"] - master(r["letter"], "sakin", "echo_db"), 2),
                    "claim": f"the sakin {r['letter']} is released with a bounce (echo {r['echo_db']:.1f} dB; the masters' "
                             f"median is {master(r['letter'], 'sakin', 'echo_db'):.1f}) — qalqalah where none belongs?"})
    take([r for r in rows if r["letter"] in QALQALAH and r["context"] in ("stop", "shaddah_stop") and r["echo_db"] <= 3],
         lambda r: r["echo_db"], per,
         lambda r: {"kind": "characteristic", "detector": "anatomy:weak_qalqalah", "severity": "moderate",
                    "magnitude": round(master(r["letter"], r["context"], "echo_db") - r["echo_db"], 2),
                    "claim": f"qalqalah on the stopped {r['letter']} is almost silent (echo {r['echo_db']:.1f} dB)"})
    take([r for r in rows if r["context"] == "shaddah" and r["letter"] in "بدتكطقج"
          and r["burst_db"] < 0.6 * master(r["letter"], "shaddah", "burst_db")],
         lambda r: r["burst_db"] / max(1e-6, master(r["letter"], "shaddah", "burst_db")), per,
         lambda r: {"kind": "characteristic", "detector": "anatomy:shaddah_unclear", "severity": "moderate",
                    "magnitude": round(master(r["letter"], "shaddah", "burst_db") - r["burst_db"], 2),
                    "claim": f"the shaddah on the {r['letter']} is not made clear: its separation is {r['burst_db']:.1f} dB "
                             f"against the masters' {master(r['letter'], 'shaddah', 'burst_db'):.1f}"})
    n = unit_candidates(picks)
    print(f"{len(picks)} findings selected, {n} new candidates in the review queue")


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:]))
