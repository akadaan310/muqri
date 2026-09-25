"""The letter matrix: for every letter of a recitation, its articulation point and all 17 classical
characteristics, each with what the engine measured.

The 17 (al-Jazari): five opposed pairs -- hams / jahr, shiddah / rakhawah (with tawassut between),
isti'la / istifal, itbaq / infitah, idhlaq / ismat -- and seven singles -- safir, qalqalah, leen,
inhiraf, takrir, tafashshi, istitalah. Every letter carries one side of each pair and whichever
singles are its own. The engine's heads measure ten of these (hams_or_jahr, shidda_or_rakhawa,
tafkheem_or_taqeeq as the audible face of isti'la / istifal, itbaq, safeer, qalqla, tikraar,
tafashie, istitala, and ghonna beside them); idhlaq / ismat is a property of the letter set, not a
sound, and leen and inhiraf have no head yet -- the matrix says so instead of leaving them out.

The makhraj column is the letter-identity test: the letter against its classical confusions (the
nearest articulation points) and against deletion, with the margin in nats.

Context matters for how much a letter reveals: a voweled letter (separation -- the tongue leaves
the point into the vowel) versus a saakin, doubled or stopped one (collision -- held at the point),
which shows the articulation and the characteristics most plainly.
"""

from __future__ import annotations

from typing import Any

from app.letters import ALIAS, NASAL, PAIRS, SINGLE_AR, SINGLE_HEADS, SINGLES, UNMEASURED_WHY, makhraj_of

VOWELS = {"َ": "fatha", "ِ": "kasra", "ُ": "damma"}


def _context(letters: list[dict[str, Any]], i: int) -> str:
    l = letters[i]
    if l["run_length"] >= 2:
        return "doubled"
    nxt = letters[i + 1] if i + 1 < len(letters) and letters[i + 1]["id"].rsplit(":", 1)[0] == l["id"].rsplit(":", 1)[0] else None
    if nxt is None:
        return "stop"
    return "voweled" if nxt["kind"] in ("harakah", "madd") else "saakin"


def _cell(l: dict[str, Any], head: str | None) -> dict[str, Any]:
    if head is None:
        return {"measured": False}
    c = l["characteristics"].get(head)
    if not c:
        return {"measured": False}
    return {"measured": True, "realised": c["realised"], "margin": c["margin"], "scored": c["scored"],
            "expected": c["expected"], "observed": c["observed"], "percentile_masters": c["percentile_masters"]}


def build(m: dict[str, Any]) -> dict[str, Any]:
    """Per consonant: makhraj (identity test) and its 17 characteristics; per group: accuracies."""
    letters = m["letters"]
    rows = []
    for i, l in enumerate(letters):
        if l["kind"] not in ("consonant", "ikhfa_noon", "iqlab_meem"):
            continue
        c = ALIAS.get(l["symbol"], l["symbol"])
        sifat = []
        for name, ar, head, value in PAIRS:
            sifat.append({"sifah": name, "ar": ar, "value": value(c), **_cell(l, head)})
        for name, members in SINGLES.items():
            if c in members:
                sifat.append({"sifah": name, "ar": SINGLE_AR[name], "value": name, **_cell(l, SINGLE_HEADS[name])})
        if c in NASAL:
            sifat.append({"sifah": "ghunnah", "ar": "الغنة", "value": "ghunnah", **_cell(l, "ghonna")})
        idn = l["identity"]
        mk = makhraj_of(c)
        near = (l.get("makhraj") or {}).get("neighbours") or {}
        nxt = letters[i + 1] if i + 1 < len(letters) else None
        rows.append({"id": l["id"], "word": l["word"], "letter": l["symbol"], "context": _context(letters, i),
                     "vowel": VOWELS.get(nxt["symbol"]) if nxt and nxt["word"] == l["word"] else None,
                     "makhraj": {"point": mk["n"] if mk else None, "ar": mk["ar"] if mk else "",
                                 "region": mk["point"] if mk else "", "confirmed": idn["confirmed"],
                                 "competitor": idn["competitor"], "margin": idn["margin"],
                                 # the articulation-point test (drills): against every neighbouring point
                                 "neighbours": near, "neighbours_held": min(near.values()) > 0 if near else None},
                     "sifat": sifat})
    # every short vowel: was it heard as itself, against the other two
    vowels = []
    for i, l in enumerate(letters):
        if l["symbol"] not in VOWELS:
            continue
        prev = letters[i - 1] if i else None
        idn = l["identity"]
        vowels.append({"id": l["id"], "word": l["word"], "vowel": VOWELS[l["symbol"]],
                       "after": prev["symbol"] if prev and prev["word"] == l["word"] else None,
                       "confirmed": idn["confirmed"], "competitor": idn["competitor"], "margin": idn["margin"],
                       "duration_s": l["duration_s"]})
    # accuracies, separately for the voweled (separation) and the held (collision) letters
    groups = {"voweled": ("voweled",), "held (saakin / doubled / stop)": ("saakin", "doubled", "stop"), "all": None}
    summary: dict[str, Any] = {}
    for g, ctx in groups.items():
        rs = [r for r in rows if ctx is None or r["context"] in ctx]
        mk = [r["makhraj"] for r in rs if r["makhraj"]["margin"] is not None]
        per: dict[str, list[bool]] = {}
        for r in rs:
            for s in r["sifat"]:
                if s.get("measured") and s.get("scored"):
                    per.setdefault(s["sifah"], []).append(bool(s["realised"]))
        nb = [r["makhraj"]["neighbours_held"] for r in rs if r["makhraj"]["neighbours_held"] is not None]
        summary[g] = {"letters": len(rs),
                      "makhraj_confirmed": round(sum(x["confirmed"] for x in mk) / len(mk), 3) if mk else None,
                      "makhraj_neighbours_held": round(sum(nb) / len(nb), 3) if nb else None,
                      "sifat": {k: {"n": len(v), "realised": round(sum(v) / len(v), 3)} for k, v in sorted(per.items())}}
    by_vowel = {}
    for v in VOWELS.values():
        vs = [x for x in vowels if x["vowel"] == v and x["margin"] is not None]
        if vs:
            by_vowel[v] = {"n": len(vs), "confirmed": round(sum(x["confirmed"] for x in vs) / len(vs), 3)}
    return {"letters": rows, "vowels": vowels, "summary": summary, "vowel_summary": by_vowel,
            "not_measured": UNMEASURED_WHY}
