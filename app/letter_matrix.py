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

HAMS = set("فحثهشخصسكت")
SHIDDAH = set("ءجدقطبكت")
TAWASSUT = set("لنعمر")
ISTILA = set("خصضغطقظ")
ITBAQ = set("صضطظ")
IDHLAQ = set("فرمنلب")
SINGLES = {"safir": set("صزس"), "qalqalah": set("قطبجد"), "leen": set("وي"), "inhiraf": set("لر"),
           "takrir": set("ر"), "tafashshi": set("ش"), "istitalah": set("ض")}
NASAL = set("نم")
# phonetic-script letters that are a noon or a meem in another guise
ALIAS = {"ں": "ن", "۾": "م"}

MAKHRAJ = {**{c: "throat, deepest" for c in "ءه"}, **{c: "throat, middle" for c in "عح"},
           **{c: "throat, nearest" for c in "غخ"}, "ق": "tongue root, soft palate",
           "ك": "tongue back, hard palate", **{c: "tongue middle" for c in "جشي"},
           "ض": "tongue side, molars", "ل": "tongue edge, front", "ن": "tongue tip, gum",
           "ر": "tongue tip and back of it, gum", **{c: "tongue tip, upper incisor roots" for c in "طدت"},
           **{c: "tongue tip, lower incisors (whistle)" for c in "صزس"},
           **{c: "tongue tip, upper incisor edges" for c in "ظذث"}, "ف": "lower lip, upper incisors",
           **{c: "the lips" for c in "بمو"}}

# (name, arabic, the engine head measuring it or None, the letter set it applies to, how to state it)
PAIRS = (("hams_jahr", "الهمس/الجهر", "hams_or_jahr", lambda c: "hams" if c in HAMS else "jahr"),
         ("shiddah_rakhawah", "الشدة/التوسط/الرخاوة", "shidda_or_rakhawa",
          lambda c: "shiddah" if c in SHIDDAH else ("tawassut" if c in TAWASSUT else "rakhawah")),
         ("istila_istifal", "الاستعلاء/الاستفال", "tafkheem_or_taqeeq",
          lambda c: "isti'la" if c in ISTILA else "istifal"),
         ("itbaq_infitah", "الإطباق/الانفتاح", "itbaq", lambda c: "itbaq" if c in ITBAQ else "infitah"),
         ("idhlaq_ismat", "الإذلاق/الإصمات", None, lambda c: "idhlaq" if c in IDHLAQ else "ismat"))
SINGLE_HEADS = {"safir": "safeer", "qalqalah": "qalqla", "leen": None, "inhiraf": None, "takrir": "tikraar",
                "tafashshi": "tafashie", "istitalah": "istitala"}
SINGLE_AR = {"safir": "الصفير", "qalqalah": "القلقلة", "leen": "اللين", "inhiraf": "الانحراف",
             "takrir": "التكرير", "tafashshi": "التفشي", "istitalah": "الاستطالة"}


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
        rows.append({"id": l["id"], "word": l["word"], "letter": l["symbol"], "context": _context(letters, i),
                     "makhraj": {"region": MAKHRAJ.get(c, ""), "confirmed": idn["confirmed"],
                                 "competitor": idn["competitor"], "margin": idn["margin"]},
                     "sifat": sifat})
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
        summary[g] = {"letters": len(rs),
                      "makhraj_confirmed": round(sum(x["confirmed"] for x in mk) / len(mk), 3) if mk else None,
                      "sifat": {k: {"n": len(v), "realised": round(sum(v) / len(v), 3)} for k, v in sorted(per.items())}}
    return {"letters": rows, "summary": summary,
            "not_measured": {"idhlaq_ismat": "a property of the letter set, not a sound",
                             "leen": "no per-letter head; the madd leen rule times it",
                             "inhiraf": "no head yet"}}
