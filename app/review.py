"""The listening review: mistakes the engine predicts in existing recitations, confirmed by an expert ear.

No dataset has located tajweed mistakes at scale, but the recordings exist -- 41 professional
reciters, including fast prayer imams who, measured against a master, make many small departures
nobody corrects in prayer. If the engine proposes where those departures are and a certified reciter
confirms or rejects each one by listening, every verdict becomes a located, typed, expert-labelled
mistake (or a labelled false alarm), which is exactly the training and calibration data that is
missing.

A **candidate** is one engine finding in one verse, with:

* a plain-language **claim** ("the madd in فِيهِ is held about 6 counts; 2 are required"),
* a **magnitude** -- how far from the requirement, not pass/fail. Lengths are judged in the reciter's
  own counts, and a count or so either way is the reciter's calibration, not a mistake: only
  departures well outside the band are proposed,
* the **time** of the letter in the recording, so the reviewer can play just that word.

Four kinds: a length outside its band (durational rule), a characteristic not realised (one of the
ten heads), a letter heard as its classical confusion (identity), and inconsistency -- the same madd
type held at clearly different lengths within one verse (taswiyah).

Every verdict (yes / not there / unsure, with an optional note) is appended to `labels.jsonl`. The
per-detector precision computed from them is what calibrates the detectors, and the confirmed
mistakes are the seed set for synthesis (research_agency_lab/experiments/deep_research/09).
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from app.submission import judged

ROOT = Path(__file__).resolve().parents[1]
REVIEW_DIR = ROOT / "research_agency_lab/experiments/review"
CANDIDATES = REVIEW_DIR / "candidates.jsonl"
LABELS = REVIEW_DIR / "labels.jsonl"
FRAME_S = 0.04

# thresholds for PROPOSING a candidate -- deliberately above the grading tolerance, so a reviewer's
# time goes to real departures (a count either way is calibration, not a mistake)
MIN_COUNTS_OUTSIDE = 0.75       # beyond the band edge, which already carries 0.75 of tolerance
MIN_SIFAH_NATS = 2.0            # the wrong class at least e^2 ~ 7x more likely than the expected one
MIN_IDENTITY_NATS = 2.0
MIN_SPREAD_COUNTS = 2.0         # same madd type within one verse

HEAD_NAMES = {
    "ghonna": "ghunnah (nasalisation)", "qalqla": "qalqalah (the bounce)",
    "tafkheem_or_taqeeq": "tafkhīm / tarqīq (heavy or light)", "hams_or_jahr": "hams / jahr (breath or voice)",
    "shidda_or_rakhawa": "shiddah / rakhāwah (closure or flow)", "itbaq": "iṭbāq (tongue covering)",
    "safeer": "ṣafīr (whistle)", "tafashie": "tafashshī (spread of shīn)",
    "istitala": "istiṭālah (the ḍād's extension)", "tikraar": "takrīr (trill)",
}


HEAVY_HEADS = frozenset({"itbaq", "tafkheem_or_taqeeq"})
LIGHT = frozenset({"[منفتح]", "[مرقق]"})


def _cid(*parts: Any) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:12]


def _severity(m: float, moderate: float, major: float) -> str:
    return "major" if m >= major else "moderate" if m >= moderate else "minor"


def mine(report: dict[str, Any], speaker: str, source: str = "everyayah",
         audio_url: str | None = None) -> list[dict[str, Any]]:
    """Every proposable mistake in one engine report, with its magnitude and where to listen."""
    out: list[dict[str, Any]] = []
    for a in report.get("ayahs", []):
        t0 = a["frames"][0] * FRAME_S
        letters = a["letters"]
        words = {w["index"]: w["word"] for w in a.get("words", [])}

        def span(idx: list[int]) -> tuple[float, float]:
            ls = [letters[i] for i in idx if 0 <= i < len(letters)]
            if not ls:
                return t0, t0
            return (round(t0 + min(l["onset_s"] for l in ls), 3),
                    round(t0 + max(l["onset_s"] + l["duration_s"] for l in ls), 3))

        def chars(idx: list[int]) -> list[int]:
            return sorted({c for i in idx if 0 <= i < len(letters) for c in letters[i].get("uth", [])})

        # every judged letter's Uthmani characters and when it sounds: the page follows playback on it
        timeline = [[round(t0 + l["onset_s"], 3), round(t0 + l["onset_s"] + l["duration_s"], 3), l.get("uth", [])]
                    for l in letters if l.get("uth")]
        base = {"source": source, "speaker": speaker, "surah": a["surah"], "ayah": a["ayah"],
                "audio_url": audio_url, "haraka_s": a.get("haraka_s"),
                "uthmani": a.get("uthmani", ""), "timeline": timeline}

        # 1. a length outside its band
        by_type: dict[str, list[tuple[int, float, list[int]]]] = defaultdict(list)
        for v in a["rules"]:
            ev = v.get("evidence") or {}
            got = ev.get("given_counts")
            if v["mechanism"] != "durational" or got is None:
                continue
            by_type[v["rule"]].append((v["word_index"], got, v["units"]))
            if v["status"] not in {"short", "long"}:
                continue
            lo, hi = v["expected_counts"]
            outside = (lo - 0.75 - got) if v["status"] == "short" else (got - (hi + 0.75 if hi < 6 else 8.8))
            if outside < MIN_COUNTS_OUTSIDE:
                continue
            s, e = span(v["units"])
            req = f"{lo:g}" if lo == hi else f"{lo:g}–{hi:g}"
            out.append({**base, "kind": "length", "detector": f"length:{v['rule']}",
                        "word_index": v["word_index"], "word": words.get(v["word_index"], v["word"]),
                        "claim": f"{v['rule'].replace('_', ' ')} held about {got:g} counts; {req} required "
                                 f"({'too short' if v['status'] == 'short' else 'too long'})",
                        "magnitude": round(outside, 2), "severity": _severity(outside, 0.75, 2.0),
                        "start_s": s, "end_s": e, "focus": chars(v["units"]),
                        "id": _cid(speaker, a["surah"], a["ayah"], "len", v["rule"], v["word_index"], v["units"])})

        # 2. inconsistency: one madd type at clearly different lengths inside the same verse
        for rule, inst in by_type.items():
            if len(inst) < 2 or not rule.startswith("madd"):
                continue
            counts = [c for _w, c, _u in inst]
            spread = max(counts) - min(counts)
            if spread < MIN_SPREAD_COUNTS:
                continue
            idx = [u for _w, _c, us in inst for u in us]
            s, e = span(idx)
            out.append({**base, "kind": "consistency", "detector": f"consistency:{rule}",
                        "word_index": inst[0][0],
                        "word": " · ".join(words.get(w, "?") for w, _c, _u in inst),
                        "claim": f"{rule.replace('_', ' ')} held at different lengths in one verse: "
                                 + ", ".join(f"{words.get(w, '?')} {c:g}" for w, c, _u in inst) + " counts",
                        "magnitude": round(spread, 2), "severity": _severity(spread, 2.0, 3.0),
                        "start_s": s, "end_s": e, "focus": chars(idx),
                        "id": _cid(speaker, a["surah"], a["ayah"], "cons", rule)})

        # 3 and 4. letter identity and characteristics
        for i, l in enumerate(letters):
            w = l.get("word")
            s, e = round(t0 + l["onset_s"], 3), round(t0 + l["onset_s"] + l["duration_s"], 3)
            idn = l["identity"]
            if not idn["confirmed"] and (idn.get("llr") or 0) >= MIN_IDENTITY_NATS:
                # "dropped" only on strong evidence; below it the letter is there but under-articulated
                # (the reviewer on Muhsin's ع: "it's there, but needs more بينية ... to produce Ayn")
                if idn["heard_instead"] == "∅":
                    claim = (f"the {l['symbol']} is dropped" if idn["llr"] >= 6
                             else f"the {l['symbol']} is weakly articulated — barely produced")
                else:
                    claim = f"the {l['symbol']} sounds like {idn['heard_instead']}"
                out.append({**base, "kind": "letter", "detector": f"letter:{l['symbol']}",
                            "word_index": w, "word": words.get(w, ""),
                            "claim": claim,
                            "magnitude": round(idn["llr"], 2), "severity": _severity(idn["llr"], 2.0, 6.0),
                            "start_s": s, "end_s": e, "letter": l["symbol"], "focus": l.get("uth", []),
                            "id": _cid(speaker, a["surah"], a["ayah"], "id", i)})
            nxt = letters[i + 1] if i + 1 < len(letters) else None
            for head, sv in (l.get("sifat") or {}).items():
                # the voiced hold on ب / د: a latent characteristic to confirm by ear, not a mistake
                if (head == "istitala" and l["symbol"] != "ض" and not sv.get("realised")
                        and -sv["llr"] >= MIN_SIFAH_NATS):
                    ctx = "with shaddah" if l.get("run_length", 1) >= 2 else (
                        "sakin" if nxt is None or nxt["kind"] != "harakah" else "voweled")
                    out.append({**base, "kind": "phenomenon", "detector": f"latent:voiced_hold:{l['symbol']}",
                                "word_index": w, "word": words.get(w, ""),
                                "claim": f"voiced hold on the {l['symbol']} ({ctx}): the sound keeps flowing and "
                                         "builds against the closure before the release. Not a mistake — "
                                         "is the effect there?",
                                "magnitude": round(-sv["llr"], 2), "severity": "phenomenon",
                                "start_s": s, "end_s": e, "letter": l["symbol"], "focus": l.get("uth", []),
                                "context": ctx, "id": _cid(speaker, a["surah"], a["ayah"], "hold", i)})
                    continue
                if (not judged(head, l.get("run_length", 1), l["symbol"]) or sv.get("realised")
                        or -sv["llr"] < MIN_SIFAH_NATS):
                    continue
                # Heaviness heard on a LIGHT letter that carries a vowel is the vowel being coloured,
                # not the letter: "that effect is itmam al-harakat ... the reciter adding heavy deep
                # resonance into the dhamma" (certified reviewer, Matroud's رَبُّ).
                if (head in HEAVY_HEADS and sv["expected"] in LIGHT and nxt is not None
                        and nxt["kind"] == "harakah"):
                    out.append({**base, "kind": "characteristic", "detector": "itmam:heavy_vowel",
                                "word_index": w, "word": words.get(w, ""),
                                "claim": f"the vowel after the {l['symbol']} is coloured heavy — "
                                         "itmām al-ḥarakāt: the ḥarakah must stay pure and light here",
                                "magnitude": round(-sv["llr"], 2), "severity": _severity(-sv["llr"], 2.0, 6.0),
                                "start_s": s, "end_s": round(t0 + nxt["onset_s"] + nxt["duration_s"], 3),
                                "letter": l["symbol"], "focus": l.get("uth", []) + nxt.get("uth", []),
                                "id": _cid(speaker, a["surah"], a["ayah"], "itmam", i)})
                    continue
                out.append({**base, "kind": "characteristic", "detector": f"sifah:{head}",
                            "word_index": w, "word": words.get(w, ""),
                            "claim": f"the {l['symbol']}: {HEAD_NAMES.get(head, head)} — expected "
                                     f"{sv['expected']}, heard {sv['model_best']}",
                            "magnitude": round(-sv["llr"], 2), "severity": _severity(-sv["llr"], 2.0, 6.0),
                            "start_s": s, "end_s": e, "letter": l["symbol"], "focus": l.get("uth", []),
                            "id": _cid(speaker, a["surah"], a["ayah"], "sf", i, head)})
    # two heads (itbaq, tafkhim) can both hear the same coloured vowel: one finding, not two
    seen: set[str] = set()
    return [c for c in out if not (c["id"] in seen or seen.add(c["id"]))]  # type: ignore[func-returns-value]


# ------------------------------------------------------------------ the queue and the labels
def _read(p: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in open(p)] if p.is_file() else []


def labels() -> dict[str, dict[str, Any]]:
    """The latest verdict per candidate (a re-listen overrides an earlier verdict)."""
    return {x["id"]: x for x in _read(LABELS)}


def add_label(cid: str, verdict: str, note: str = "", reviewer: str = "expert") -> dict[str, Any]:
    if verdict not in {"yes", "no", "unsure"}:
        raise ValueError("verdict must be yes, no or unsure")
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    row = {"id": cid, "verdict": verdict, "note": note.strip(), "reviewer": reviewer,
           "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(LABELS, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def queue(limit: int = 20) -> list[dict[str, Any]]:
    """The next candidates to hear, spread across detectors so every rule gets reviewed.

    Round-robin over detectors, largest magnitude first within each, one reciter-verse at a time --
    and every fifth pick from the middle of a detector's range rather than its top, because a
    detector's threshold is calibrated by its borderline cases, not its obvious ones.
    """
    done = labels()
    todo = [c for c in _read(CANDIDATES) if c["id"] not in done]
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in todo:
        by[c["detector"]].append(c)
    for v in by.values():
        v.sort(key=lambda c: -c["magnitude"])
    out, k = [], 0
    order = sorted(by, key=lambda d: -len(by[d]))
    while len(out) < limit and any(by.values()):
        for d in order:
            if by[d] and len(out) < limit:
                pick = by[d].pop(len(by[d]) // 2 if k % 5 == 4 else 0)
                out.append(pick)
                k += 1
    return out


def stats() -> dict[str, Any]:
    """How often each detector is right, by the expert's ear -- the number that calibrates it."""
    cands = {c["id"]: c for c in _read(CANDIDATES)}
    per: dict[str, dict[str, int]] = defaultdict(lambda: {"yes": 0, "no": 0, "unsure": 0})
    for cid, lab in labels().items():
        if cid in cands:
            per[cands[cid]["detector"]][lab["verdict"]] += 1
    rows = {}
    for d, c in sorted(per.items()):
        decided = c["yes"] + c["no"]
        # Beta(1,1) posterior mean: honest with few labels, converges to the raw rate with many
        rows[d] = {**c, "precision": round((c["yes"] + 1) / (decided + 2), 3), "decided": decided}
    return {"candidates": len(cands), "labelled": sum(sum(v.values()) for v in per.values()),
            "detectors": rows}
