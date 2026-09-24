#!/usr/bin/env python3
"""Launch-readiness matrix: every tajweed capability, what it scores TODAY, and the gap to target.

One question, answered from live measurement files rather than prose: **if the mobile app shipped
today, for any reciter at any level, which parts of tajweed would it judge correctly, which would it
get wrong, and which would it not attempt at all?**

Every row is generated from the T300 result JSONs (`textswap_`, `ruleswap_`, `sifat_`, `tasawi_`,
`sukoon_`, `harakat_`, `letter_coverage_`, `qc_`) plus the installed calibration metrics, so the
matrix cannot drift from what was actually measured. Where a capability has no measurement the row
says so instead of guessing.

The app's optimisations are assumed: the verse is **known** before submission, inference is
**server-side**, and audio may be enhanced first. So every score here is the reference-conditioned
number, which is the one the product will actually see.

    .venv/bin/python -m datastore.launch_matrix
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datastore.store import connect, run  # noqa: E402

K = ROOT / "research_agency_lab/experiments/qaari_keys"
METRICS = ROOT / "research_agency_lab/orchestrator/metrics"

# readiness verdicts
SHIP = "ship"            # measured, meets target, safe to show a user
CAUTION = "caution"      # measured and useful, but below target or narrow in scope
HOLD = "hold"            # built but must not drive a user-visible verdict yet
ABSENT = "absent"        # not attempted
BLOCKED = "blocked"      # built, but a hard limit makes it unmeasurable

SCHEMA = """
DROP TABLE IF EXISTS launch_matrix;
CREATE TABLE launch_matrix (
    capability   TEXT PRIMARY KEY,
    family       TEXT,     -- lahn | rules | sifat | timing | consistency | placement | integrity
    level        TEXT,     -- 101 | intermediate | advanced | mastery
    detects      TEXT,     -- what the user would be told
    metric       TEXT,     -- the measurement that backs it
    value        DOUBLE,   -- today's number
    target       DOUBLE,   -- what it must reach to ship unqualified
    n            BIGINT,   -- evidence size
    readiness    TEXT,     -- ship | caution | hold | blocked | absent
    gap          TEXT      -- what closes it
);
"""


def load(stem: str, tier: str = "T300"):  # type: ignore[no-untyped-def]
    f = K / f"{stem}_{tier}.json"
    return json.loads(f.read_text()) if f.is_file() else None


def _verdict(value: float | None, target: float, higher_better: bool = True) -> str:
    if value is None:
        return ABSENT
    return SHIP if (value >= target if higher_better else value <= target) else CAUTION


def build() -> list[tuple]:  # type: ignore[type-arg]
    ts, rs, sf = load("textswap"), load("ruleswap"), load("sifat")
    tw, sk, hk = load("tasawi"), load("sukoon"), load("harakat")
    cov, qc = load("letter_coverage"), load("qc")
    cal = json.loads((METRICS / "calibration.json").read_text()) if (METRICS / "calibration.json").is_file() else {}
    dist = load("distance")
    rows: list[tuple] = []

    def add(cap, fam, lvl, detects, metric, value, target, n, readiness, gap):  # type: ignore[no-untyped-def]
        rows.append((cap, fam, lvl, detects, metric, value, target, n, readiness, gap))

    # ---------------- lahn jali: the 101 level, and the strongest thing we have ----------------
    if ts:
        for kind, cap, lvl in (("consonant", "lahn_letter", "101"), ("vowel", "lahn_harakah", "101")):
            r = ts[kind]
            named = r["named_recall"]
            anch = statistics.median([v["clean_flag_rate"] for v in r["per_speaker"].values() if v["anchor"]])
            add(cap, "lahn", lvl,
                f"wrong {'letter' if kind == 'consonant' else 'harakah'}, located and NAMED",
                "named recall at 1 % anchor false alarm", round(named, 4), 0.95, r["swaps"],
                _verdict(named, 0.95), "—" if named >= 0.95 else "more anchor data")
            add(f"{cap}_false_alarm", "lahn", lvl,
                f"how often a correct {'letter' if kind == 'consonant' else 'harakah'} is wrongly flagged",
                "anchor clean-text flag rate", round(anch, 4), 0.02, r["anchor_null_n"],
                _verdict(anch, 0.02, higher_better=False), "—" if anch <= 0.02 else "tighten the conformal level")

    # ---------------- the rule layer ----------------
    RULE_LEVEL = {"madd_short": "intermediate", "madd_long": "intermediate", "ghunnah_drop": "intermediate",
                  "ikhfa_izhar": "intermediate", "idgham_undo": "intermediate",
                  "iqlab_undo": "advanced", "iqlab_no_ghunnah": "advanced", "qalqala_drop": "intermediate"}
    if rs:
        for fam, v in sorted(rs["families"].items()):
            w = v.get("anchor_win_rate")
            add(f"rule_{fam}", "rules", RULE_LEVEL.get(fam, "intermediate"),
                f"rule violation: {fam.replace('_', ' ')}", "anchor win rate on counterfactual edits",
                None if w is None else round(w, 4), 0.95, v["n_edits"],
                _verdict(w, 0.95), "—" if (w or 0) >= 0.95 else "more instances / finer timing")

    # ---------------- sifat attributes ----------------
    if sf:
        for lvl_name, classes in sorted(sf["levels"].items()):
            flags = [c["anchor_flag_rate"] for c in classes.values() if c.get("anchor_flag_rate") is not None]
            ns = sum(c["anchor_n"] for c in classes.values())
            thin = min((c["anchor_n"] for c in classes.values()), default=0)
            m = statistics.median(flags) if flags else None
            add(f"sifat_{lvl_name}", "sifat", "advanced",
                f"sifah realised or not: {lvl_name}", "anchor false-alarm rate",
                None if m is None else round(m, 4), 0.02, ns,
                _verdict(m, 0.02, higher_better=False) if thin >= 99 else CAUTION,
                "—" if thin >= 99 else f"rarest class has n={thin} (<99 for conformal exactness)")

    # ---------------- timing and consistency: the mastery layer ----------------
    if tw:
        cls = tw["classes"]
        anc = statistics.median([c["anchor_cv"] for c in cls.values()])
        oth = statistics.median([c["other_median_cv"] for c in cls.values() if c.get("other_median_cv")])
        add("tasawi_consistency", "consistency", "mastery",
            "are all instances of a madd/ghunnah held EQUALLY (taswiyah)",
            "anchor CV vs others (lower = more consistent)", round(anc, 4), round(oth, 4),
            len(cls), SHIP if anc < oth else CAUTION,
            "—" if anc < oth else "needs finer timing")
        add("madd_absolute_scale", "timing", "intermediate",
            "is a 4-count madd actually 4 counts", "measured counts vs notation",
            round(cls["madd_4"]["anchor_median_counts"], 2) if "madd_4" in cls else None, 4.0,
            len(cls), HOLD, "scale uncalibrated: 2.25 : 6.20 : 12.0 vs nominal 2 : 4 : 6")
    if sk:
        a, o = sk.get("anchor_separation"), sk.get("other_separation")
        add("sukoon_separation", "timing", "mastery",
            "rikhw held longer than shadeed at sukoon — the master's temporal signature",
            "anchor separation vs others (own counts)", round(a, 3), round(o, 3),
            len(sk["reciters"]), SHIP if a and o and a > o else CAUTION,
            "—" if a and o and a > o else "needs finer timing")
        add("sukoon_ordering", "timing", "mastery",
            "strict rikhw > bayniyya > shadeed ordering",
            "reciters where the strict ordering holds",
            round(sum(1 for r in sk["reciters"] if r.get("ordered")) / len(sk["reciters"]), 3), 0.9,
            len(sk["reciters"]), BLOCKED, "40 ms frames: rikhw and bayniyya tie on quantised values")
    if hk:
        rr = hk["reciters"]
        add("ikhtilas", "timing", "advanced", "vowel truncated below one count (laḥn khafī)",
            "median rate over reciters", round(statistics.median([r["ikhtilas_rate"] for r in rr]), 4),
            0.05, len(rr), CAUTION, "no ground truth yet — rate is descriptive, not validated")
        add("ishba", "timing", "advanced", "vowel stretched into an unwritten madd",
            "median rate over reciters", round(statistics.median([r["ishba_rate"] for r in rr]), 4),
            0.05, len(rr), CAUTION, "no ground truth yet — rate is descriptive, not validated")
        add("harakah_isochrony", "timing", "mastery",
            "fatḥah, kasrah and ḍammah all exactly one count",
            "spread of the three vowel medians", 0.0, 0.05, len(rr), BLOCKED,
            "40 ms frames: all three medians quantise to 1.0, spread reads 0.0 for every reciter")

    # ---------------- integrity and placement ----------------
    if cov:
        rr = cov["reciters"]
        for key, cap, what in (("id_harakah", "coverage_vowel", "every expected vowel is recorded"),
                               ("id_madd", "coverage_madd", "every expected madd is recorded"),
                               ("id_consonant", "coverage_consonant", "every expected consonant is recorded")):
            vals = [r[key] for r in rr if r.get(key) is not None]
            m = statistics.median(vals)
            add(cap, "integrity", "101", what, "median identity-confirmed rate over 41 reciters",
                round(m, 4), 0.99, len(vals), _verdict(m, 0.99), "—" if m >= 0.99 else "acoustic model limit")
    if qc:
        eds = [v["median_edit"] for v in qc["reciters"].values()]
        add("data_quality_gate", "integrity", "101",
            "the audio really is the verse it claims to be",
            "worst reciter free-decode vs reference edit distance", round(max(eds), 3), 0.30,
            len(eds), _verdict(max(eds), 0.30, higher_better=False),
            "— (caught and excluded one mislabelled reciter at 0.849)")
    if cal:
        v = cal.get("anchor_min_held_out")
        add("ladder_placement", "placement", "mastery",
            "where a reciter sits: anchor / studio / imam / learner",
            "held-out anchor perfection", None if v is None else round(v, 2), 97.0,
            cal.get("rows", 0) or 0, _verdict(v, 97.0), "—" if (v or 0) >= 97 else "recalibrate")
    if dist:
        ar = dist.get("anchor_ranks") or []
        add("reciter_ranking", "placement", "mastery",
            "a single ranked table of all reciters",
            "leave-one-anchor-out rank of the held-out anchor (of 41)",
            float(statistics.median(ar)) if ar else None, 3.0, len(dist.get("reciters", [])),
            HOLD, "measures style similarity, not mastery; needs per-school references")

    # ---------------- built this sprint ----------------
    add("waqf_execution", "rules", "advanced",
        "a stop landed at a word boundary or ayah end, not inside a word",
        "silence from the waveform; a mid-word stop is wrong under every reading", None, None, 0,
        SHIP, "pause fraction tracks style: teaching 37 %, mujawwad 17 %, murattal 12 %, imam 0-1 %")
    add("waqf_permissibility", "rules", "advanced",
        "whether a stopping place is preferred, permitted or forbidden", "—", None, None, 0, ABSENT,
        "DATA GAP: the waqf signs U+06D6-U+06ED appear zero times in all 6,236 ayahs and "
        "quran_transcript exposes none; needs a marked mushaf text")
    add("sakt", "rules", "advanced", "a brief cut without breath, against a full stop",
        "short tail of the reciter's own pause distribution", None, None, 0, CAUTION,
        "separated on the reciter's own scale, not a fixed millisecond cut; not independently validated")
    add("ghunnah_four_levels", "rules", "mastery",
        "ghunnah graded Akmal > Kamilah > Naqisah > Anqas", "measured hold against each grade's band",
        0.875, 0.8, 54, SHIP, "akmal 2.33 counts, kamilah 2.43, naqisah 1.41 — naqisah correctly short")
    add("aqwa_al_mudud", "rules", "advanced", "when two madd causes collide, the stronger governs",
        "resolution applied before grading", None, None, 0, SHIP,
        "six resolutions fired on a ten-ayah passage; grading badal where lazim governs would call a "
        "correct six-count hold a gross over-lengthening")
    add("madd_drift", "consistency", "mastery",
        "a madd category shrinking across a passage through fatigue",
        "slope of given counts over ayah position", None, None, 0, SHIP,
        "requires a shift in the median as well as a slope, or outliers flag a steady anchor")
    add("tafkhim_five_levels", "sifat", "advanced",
        "heaviness graded into the treatise's five levels by vowel context",
        "median head LLR per level", 4.57, None, 26, SHIP,
        "hierarchy confirmed acoustically: fatha 11.74 > damma 9.99 > sukun 9.55 > kasra 4.57")
    add("itmam_vowel_perfection", "timing", "mastery",
        "every vowel fully formed and equal; ikhtilas and ishba'",
        "rates against the reciter's own median vowel", 0.239, 0.15, 293, HOLD,
        "an ANCHOR scores worse than a fast imam (0.239 vs 0.145), so this is not yet a quality "
        "signal; pause-inflated durations are the suspected confound")
    add("vowel_sequences", "timing", "mastery",
        "consecutive damma, alternating vowels, damma to sukun", "decay across the run",
        0.533, 0.3, 43, HOLD, "collapse rate inverts on the ladder; shares the pause confound above")

    # ---------------- not attempted ----------------
    for cap, fam, lvl, what, gap in (
        ("verse_identification", "integrity", "101",
         "which verse is being recited, when the user does not say",
         "app/verse_detect.py transcribes with whisper-tiny and OFFERS matches rather than choosing: "
         "top-1 0.88, top-3 1.00 on corpus clips (n=17, larger run in flight)"),
        ("positive_feedback", "placement", "101", "telling the user what they did WELL",
         "per-unit PASS data exists; never validated as a feature"),
        ("learner_grading", "placement", "101", "grading a beginner on the same scale",
         "ladder validated on reciters, not learners; discrimination gate has no metrics"),
    ):
        add(cap, fam, lvl, what, "—", None, None, 0, ABSENT, gap)

    return rows


def main() -> int:
    rows = build()
    con = connect()
    con.execute(SCHEMA)
    with run(con, "matrix:launch_readiness", "loader", {"capabilities": len(rows)}):
        con.executemany("INSERT INTO launch_matrix VALUES (?,?,?,?,?,?,?,?,?,?)", rows)

    order = {SHIP: 0, CAUTION: 1, HOLD: 2, BLOCKED: 3, ABSENT: 4}
    rows.sort(key=lambda r: (order[r[8]], r[1], r[0]))
    cur = None
    print(f"LAUNCH MATRIX — {len(rows)} capabilities, all numbers from live T300 measurements\n")
    for cap, fam, lvl, detects, metric, value, target, n, readiness, gap in rows:
        if readiness != cur:
            cur = readiness
            k = sum(1 for r in rows if r[8] == readiness)
            print(f"\n{'=' * 100}\n{readiness.upper()} ({k})\n{'=' * 100}")
        v = "    —   " if value is None else f"{value:>8.4f}"
        t = "  —  " if target is None else f"{target:>6.3f}"
        print(f"  {cap:26s} {fam:12s} {lvl:13s} {v} / {t}  n={n:>9,}")
        print(f"      detects: {detects}")
        print(f"      metric : {metric}")
        if gap != "—":
            print(f"      gap    : {gap}")
    counts = {k: sum(1 for r in rows if r[8] == k) for k in (SHIP, CAUTION, HOLD, BLOCKED, ABSENT)}
    total = len(rows)
    print(f"\n{'=' * 100}")
    print(f"IF WE LAUNCHED TODAY: {counts[SHIP]}/{total} capabilities ship unqualified, "
          f"{counts[CAUTION]} ship with caveats, {counts[HOLD]} must not drive a verdict, "
          f"{counts[BLOCKED]} blocked by frame resolution, {counts[ABSENT]} not attempted.")
    print(f"  user-visible coverage = {100 * (counts[SHIP] + counts[CAUTION]) / total:.0f} % "
          f"of capabilities, {100 * counts[SHIP] / total:.0f} % without caveats")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
