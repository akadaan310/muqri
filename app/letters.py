"""Every letter's declared profile: its makhraj (one of the 17 articulation points) and its 17 sifat,
each marked with how the engine measures it -- or that it does not, yet.

The declaration is the reference a drill or a recitation is measured against, and it stands whether
or not a measurement exists: a consumer app can teach a letter's full profile, and the matrix can say
which parts of it the engine hears.

Makharij (Ibn al-Jazari), five regions, seventeen points:

    al-jawf       1  the open mouth and throat           the madd letters ا و ي
    al-halq       2  deepest throat                      ء ه
                  3  middle throat                       ع ح
                  4  nearest throat                      غ خ
    al-lisan      5  tongue root, soft palate            ق
                  6  tongue root, a little forward       ك
                  7  tongue middle, hard palate          ج ش ي
                  8  tongue side(s), upper molars        ض
                  9  tongue edge, front gums             ل
                 10  tongue tip, gum (below ل)            ن
                 11  tongue tip and its back, gum         ر
                 12  tongue tip, upper incisor roots      ط د ت
                 13  tongue tip, near the lower incisors  ص ز س   (asaliyyah, the whistle)
                 14  tongue tip, upper incisor edges      ظ ذ ث
    ash-shafatan 15  lower lip, upper incisor edges       ف
                 16  the two lips                         ب م و
    al-khayshum  17  the nasal passage                    the ghunnah of ن م

How a makhraj is measured: the letter against the letters at the same or a neighbouring point
(`NEIGHBOURS`), each as a CTC likelihood margin in nats. The classical identity test
(analysis.CONFUSIONS) gives only 18 letters a substitution competitor; the neighbour test gives all
of them one. It is reported, not scored, until drills show which margins hold on a correct reading.
"""

from __future__ import annotations

from typing import Any

MAKHARIJ: tuple[dict[str, Any], ...] = (
    {"n": 1, "region": "jawf", "ar": "الجوف", "point": "the open mouth and throat", "letters": "اوي"},
    {"n": 2, "region": "halq", "ar": "أقصى الحلق", "point": "deepest throat", "letters": "ءه"},
    {"n": 3, "region": "halq", "ar": "وسط الحلق", "point": "middle throat", "letters": "عح"},
    {"n": 4, "region": "halq", "ar": "أدنى الحلق", "point": "nearest throat", "letters": "غخ"},
    {"n": 5, "region": "lisan", "ar": "أقصى اللسان مع ما فوقه من الحنك الأعلى", "point": "tongue root, soft palate",
     "letters": "ق"},
    {"n": 6, "region": "lisan", "ar": "أقصى اللسان أسفل مخرج القاف", "point": "tongue root, a little forward",
     "letters": "ك"},
    {"n": 7, "region": "lisan", "ar": "وسط اللسان مع ما فوقه من الحنك الأعلى", "point": "tongue middle, hard palate",
     "letters": "جشي"},
    {"n": 8, "region": "lisan", "ar": "إحدى حافتي اللسان مع ما يليها من الأضراس العليا",
     "point": "tongue side(s), upper molars", "letters": "ض"},
    {"n": 9, "region": "lisan", "ar": "أدنى حافتي اللسان إلى منتهى طرفه", "point": "tongue edge, front gums",
     "letters": "ل"},
    {"n": 10, "region": "lisan", "ar": "طرف اللسان تحت مخرج اللام", "point": "tongue tip, gum (below ل)",
     "letters": "ن"},
    {"n": 11, "region": "lisan", "ar": "طرف اللسان مع ظهره", "point": "tongue tip and its back, gum", "letters": "ر"},
    {"n": 12, "region": "lisan", "ar": "طرف اللسان مع أصول الثنايا العليا", "point": "tongue tip, upper incisor roots",
     "letters": "طدت"},
    {"n": 13, "region": "lisan", "ar": "طرف اللسان مع ما بين الثنايا السفلى", "point": "tongue tip, near the lower incisors",
     "letters": "صزس"},
    {"n": 14, "region": "lisan", "ar": "طرف اللسان مع أطراف الثنايا العليا", "point": "tongue tip, upper incisor edges",
     "letters": "ظذث"},
    {"n": 15, "region": "shafatan", "ar": "باطن الشفة السفلى مع أطراف الثنايا العليا",
     "point": "lower lip, upper incisor edges", "letters": "ف"},
    {"n": 16, "region": "shafatan", "ar": "الشفتان", "point": "the two lips", "letters": "بمو"},
    {"n": 17, "region": "khayshum", "ar": "الخيشوم", "point": "the nasal passage", "letters": "نم"},
)

# the consonants: hamza stands for alif as a letter; ا و ي as madd letters are jawf (makhraj 1)
CONSONANTS = "ءبتثجحخدذرزسشصضطظعغفقكلمنهوي"
# phonetic-script symbols that are a letter in another guise (ikhfa noon, iqlab / ikhfa meem, qalqalah echo)
ALIAS = {"ں": "ن", "۾": "م"}

# letters at the same or a neighbouring point, and the classical confusions (analysis.CONFUSIONS)
NEIGHBOURS: dict[str, tuple[str, ...]] = {
    "ء": ("ه", "ع"), "ه": ("ء", "ح"), "ع": ("ح", "ء", "غ"), "ح": ("ع", "ه", "خ"), "غ": ("خ", "ع", "ق"),
    "خ": ("غ", "ح", "ك"), "ق": ("ك", "غ"), "ك": ("ق", "ت"), "ج": ("ش", "ي", "د"), "ش": ("س", "ج"),
    "ي": ("ج",), "ض": ("د", "ظ", "ل"), "ل": ("ن", "ر"), "ن": ("ل", "م", "ر"), "ر": ("ل", "ن"),
    "ط": ("ت", "د"), "د": ("ت", "ط", "ض", "ذ"), "ت": ("ط", "د"), "ص": ("س", "ز"), "ز": ("س", "ذ"),
    "س": ("ص", "ث", "ز"), "ظ": ("ذ", "ض", "ز"), "ذ": ("ث", "ظ", "ز", "د"), "ث": ("ذ", "س", "ت"),
    "ف": ("ث", "ب"), "ب": ("م", "ف"), "م": ("ب", "ن"), "و": ("ب", "ف"),
    # vowels against each other: a vowel's "makhraj" is the shape of the mouth
    "َ": ("ِ", "ُ"), "ِ": ("َ", "ُ"), "ُ": ("َ", "ِ"),
}

# -- the 17 sifat ----------------------------------------------------------------------------------
HAMS = set("فحثهشخصسكت")
SHIDDAH = set("ءجدقطبكت")
TAWASSUT = set("لنعمر")
ISTILA = set("خصضغطقظ")
ITBAQ = set("صضطظ")
IDHLAQ = set("فرمنلب")
SINGLES = {"safir": set("صزس"), "qalqalah": set("قطبجد"), "leen": set("وي"), "inhiraf": set("لر"),
           "takrir": set("ر"), "tafashshi": set("ش"), "istitalah": set("ض")}
NASAL = set("نم")

# (key, arabic, the engine head that measures it or None, the letter's side of it)
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
# why an unmeasured sifah is unmeasured
UNMEASURED_WHY = {
    "idhlaq_ismat": "a property of the letter set (fluency of the lips and tongue tip), not a sound: no "
                    "recitation can get it wrong",
    "leen": "no model head; the engine times a leen madd (madd_leen), not the softness of the glide",
    "inhiraf": "no model head; the deviation of ل toward the tongue's edge and ر toward the back is heard "
               "only through the letter's identity",
}


def makhraj_of(letter: str) -> dict[str, Any] | None:
    """The articulation point of a consonant (ا و ي as consonants are 7 / 16; as madd letters, 1)."""
    c = ALIAS.get(letter, letter)
    for m in MAKHARIJ[1:16]:
        if c in m["letters"]:
            return m
    return None


def sifat_of(letter: str) -> list[dict[str, Any]]:
    """All 17: the five pairs (one side each) and the seven singles (present or absent)."""
    c = ALIAS.get(letter, letter)
    out = [{"sifah": k, "ar": ar, "value": side(c), "head": head, "measured": head is not None}
           for k, ar, head, side in PAIRS]
    for k, letters in SINGLES.items():
        out.append({"sifah": k, "ar": SINGLE_AR[k], "value": c in letters, "head": SINGLE_HEADS[k],
                    "measured": SINGLE_HEADS[k] is not None})
    return out


def profile(letter: str) -> dict[str, Any]:
    """The full declared profile of one letter, with what the engine measures of it."""
    c = ALIAS.get(letter, letter)
    m = makhraj_of(c)
    return {"letter": c, "makhraj": m and {k: m[k] for k in ("n", "region", "ar", "point")},
            "nasal_passage": c in NASAL,
            "sifat": sifat_of(c),
            "ghunnah": {"value": c in NASAL, "head": "ghonna", "measured": True},
            "neighbours": list(NEIGHBOURS.get(c, ())),
            "unmeasured": {k: v for k, v in UNMEASURED_WHY.items()
                           if k != "leen" or c in SINGLES["leen"]}}


def table() -> list[dict[str, Any]]:
    """Every consonant's profile, in alphabet order."""
    return [profile(c) for c in CONSONANTS]
