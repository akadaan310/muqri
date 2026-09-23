"""Marātib al-Ghunnah — the four grades of nasalisation, and letter strength.

Ghunnah is not a yes/no. The classical teaching grades it by the *structural context* the nūn or mīm
sits in, and the four grades differ in how long the velum stays down:

    1. Akmal    (أكمل)     mushaddad نّ مّ and idghām bi-ghunnah   maximum, ~2 counts and beyond
    2. Kāmilah  (كاملة)    ikhfā' and iqlāb                        complete, ~2 counts, but the oral
                                                                   articulation is softened or moved
    3. Nāqiṣah  (ناقصة)    a clear sākin nūn/mīm (iẓhār)           only the consonant's own tawassuṭ
    4. Anqaṣ    (أنقص)     a voweled nūn/mīm                       a microscopic nasal trait

The engine already knows which structure applies — the rule-instance join names it — and already
measures the hold in the reciter's own counts. Grading is therefore reading the expected grade off
the bound rule and checking the measured duration against that grade's band, rather than asking the
acoustic model a new question.

`letter_strength` is the other composite the treatises describe: a letter's quwwa is not one sifah
but the sum of the strong members it carries. Reported so a report can say *which* letters were
weakly produced, not only which sifāt failed.
"""

from __future__ import annotations

from dataclasses import dataclass

# expected ghunnah grade by the rule that creates it
AKMAL_RULES = frozenset({"ghunnah", "idgham_ghunnah", "idgham_shafawi"})
KAMILAH_RULES = frozenset({"ikhfa", "iqlab", "ikhfa_shafawi"})
NAQISAH_RULES = frozenset({"izhar_halqi", "izhar_shafawi"})

# each grade's expected hold, in tajweed counts. Akmal and Kamilah are both the full two counts; what
# separates them is oral articulation, not length, so the bands overlap deliberately.
GRADE_BANDS: dict[str, tuple[float, float]] = {
    "akmal": (1.5, 3.0),
    "kamilah": (1.4, 2.8),
    "naqisah": (0.6, 1.8),
    "anqas": (0.0, 1.0),
}
NASALS = frozenset("نمں۾")


@dataclass(slots=True)
class GhunnahVerdict:
    grade: str                    # akmal | kamilah | naqisah | anqas
    rule: str
    word: str
    expected_band: tuple[float, float]
    given_counts: float | None
    status: str                   # pass | short | long | no_evidence

    def to_dict(self) -> dict:  # type: ignore[type-arg]
        return {"grade": self.grade, "rule": self.rule, "word": self.word,
                "expected_counts": list(self.expected_band), "given_counts": self.given_counts,
                "status": self.status}


def expected_grade(rule_type: str, shadda: str | None) -> str | None:
    """Which grade of ghunnah this rule instance should produce, or None if it produces none."""
    if rule_type in AKMAL_RULES:
        return "akmal"
    if rule_type in KAMILAH_RULES:
        return "kamilah"
    if rule_type in NAQISAH_RULES:
        return "naqisah"
    # a nasal carrying a shadda is akmal even when no nasal rule was located on it
    if shadda == "nasal":
        return "akmal"
    return None


def grade_ghunnah(bound, units, to_counts) -> GhunnahVerdict | None:  # type: ignore[no-untyped-def]
    """Grade one bound rule's ghunnah, if it creates one."""
    grade = expected_grade(bound.rule_type, bound.shadda)
    if grade is None:
        return None
    us = [units[i] for i in bound.unit_indices if 0 <= i < len(units)]
    us = [u for u in us if u.symbol in NASALS]
    if not us or any(u.duration_counts is None for u in us):
        return GhunnahVerdict(grade, bound.rule_type, bound.word, GRADE_BANDS[grade], None,
                              "no_evidence")
    given = round(to_counts(sum(u.duration_counts for u in us)), 2)
    lo, hi = GRADE_BANDS[grade]
    status = "pass" if lo <= given <= hi else ("short" if given < lo else "long")
    return GhunnahVerdict(grade, bound.rule_type, bound.word, (lo, hi), given, status)


# ---------------------------------------------------------------- letter strength (quwwa) ---------
# The strong member of each opposing pair, plus the standalone sifat that add strength. A letter's
# quwwa is how many of these it carries; the treatises rank letters by exactly this sum.
STRONG_JUDGED: dict[str, frozenset[str]] = {
    "hams_or_jahr": frozenset({"[جهر]"}),
    "shidda_or_rakhawa": frozenset({"[شديد]"}),
    "itbaq": frozenset({"[مطبق]"}),
    "safeer": frozenset({"[صفير]"}),
    "qalqla": frozenset({"[مقلقل]"}),
    "tafashie": frozenset({"[متفشي]"}),
    "istitala": frozenset({"[مستطيل]"}),
    "tikraar": frozenset({"[مكرر]"}),
    "ghonna": frozenset({"[مغن]"}),
    "tafkheem_or_taqeeq": frozenset({"[مفخم]"}),
}


def letter_strength(unit, inherent: dict | None = None) -> dict:  # type: ignore[no-untyped-def,type-arg]
    """Quwwa of one letter: the strong sifāt it should carry, and how many were realised.

    `expected` counts the strong members the letter carries by definition; `realised` counts those
    the model agreed with. A gap between them is a weakly produced letter even when no single sifah
    crossed its threshold.
    """
    expected = realised = 0
    missing: list[str] = []
    for lvl, strong in STRONG_JUDGED.items():
        judged = unit.sifat.get(lvl)
        if not judged or judged["expected"] not in strong:
            continue
        expected += 1
        if judged["realised"]:
            realised += 1
        else:
            missing.append(lvl)
    if inherent:
        if inherent.get("istila") == "isti'la":
            expected += 1
            realised += 1          # inherent: true by definition of the letter
        if inherent.get("idhlaq") == "idhlaq":
            expected += 1
            realised += 1
    return {"expected": expected, "realised": realised, "missing": missing,
            "ratio": round(realised / expected, 3) if expected else None}
