"""Tafkhīm graded, not binary — the five levels, and relative heaviness.

The engine judges heaviness with muaalem's `tafkheem_or_taqeeq` head, which has three classes
(mufakhkham, muraqqaq, adnā-l-mufakhkham). The treatises grade it into **five levels**, and the
grading is not acoustic — it is decided entirely by the vowel the letter carries and what follows:

    1 highest   fatḥah AND followed by an alif      خَالِدِينَ
    2           fatḥah alone                        خَلَقَ
    3           ḍammah                              خُلِقُوا
    4           sukūn                                يَخْلُقُ
    5 lowest    kasrah                               خِيَانَة

That context is exactly what the rule-instance join supplies, so the level is read off the phoneme
string rather than asked of the model. The head then says whether the letter was realised heavy at
all; the level says *how* heavy it should have been.

**Tafkhīm nisbī** is the refinement on level 5. An isti'lā' letter *without* iṭbāq (خ غ ق) loses much
of its weight under a kasrah — the lowered tongue body works against the raised root. An iṭbāq letter
(ص ض ط ظ) keeps it, because the palatal trapping preserves the resonance. So the same level-5 slot
means different things for the two groups, and the model's `adnā-l-mufakhkham` class is the one that
should appear for the first group.

**Istiʿlāʾ / istifāl** has no head of its own. It does not need one: it is inherent to the letter
(`letter_reference.istila`), and what varies is the *degree*, which is what the level captures.
"""

from __future__ import annotations

from dataclasses import dataclass

ISTILA = frozenset("خصضطظغق")      # خُصَّ ضَغْطٍ قِظْ
ITBAQ = frozenset("صضطظ")           # the four that keep their weight under a kasrah
FATHA, DAMMA, KASRA = "َ", "ُ", "ِ"
SHORT_V = frozenset("َُِ")
MADD_ALIF = frozenset("ا")

LEVEL_NAME = {1: "fatha+alif", 2: "fatha", 3: "damma", 4: "sukun", 5: "kasra"}


@dataclass(slots=True)
class Heaviness:
    """One isti'lā' letter, its required level, and what the model heard."""

    unit: int
    letter: str
    level: int                 # 1 highest .. 5 lowest
    level_name: str
    itbaq: bool
    expected_class: str        # what the head should say at this level
    heard_class: str | None
    llr: float | None
    realised: bool | None
    nisbi: bool                # level 5 on a non-iṭbāq letter: heaviness legitimately reduced

    def to_dict(self) -> dict:  # type: ignore[type-arg]
        return {"unit": self.unit, "letter": self.letter, "level": self.level,
                "level_name": self.level_name, "itbaq": self.itbaq,
                "expected_class": self.expected_class, "heard_class": self.heard_class,
                "llr": self.llr, "realised": self.realised, "nisbi": self.nisbi}


def level_of(units, i: int) -> int | None:
    """The tafkhīm level of the isti'lā' letter at unit `i`, from its vowel context.

    Returns None when the unit is not an isti'lā' letter.
    """
    sym = units[i].symbol if hasattr(units[i], "symbol") else units[i][0]
    if sym not in ISTILA:
        return None
    nxt = units[i + 1].symbol if i + 1 < len(units) and hasattr(units[i + 1], "symbol") else (
        units[i + 1][0] if i + 1 < len(units) else "")
    nxt2 = units[i + 2].symbol if i + 2 < len(units) and hasattr(units[i + 2], "symbol") else (
        units[i + 2][0] if i + 2 < len(units) else "")
    if nxt == FATHA:
        # level 1 needs an alif of prolongation right after the fatḥah
        return 1 if nxt2 in MADD_ALIF else 2
    if nxt == DAMMA:
        return 3
    if nxt == KASRA:
        return 5
    return 4                                    # no vowel follows: the letter is sākin


def grade(units) -> list[Heaviness]:  # type: ignore[no-untyped-def]
    """Grade every isti'lā' letter in a clip by level, against what the head heard."""
    out: list[Heaviness] = []
    for i, u in enumerate(units):
        lvl = level_of(units, i)
        if lvl is None:
            continue
        itbaq = u.symbol in ITBAQ
        # level 5 on a non-iṭbāq letter is tafkhīm nisbī: the head's "adnā" class is the correct
        # reading there, not full mufakhkham
        nisbi = lvl == 5 and not itbaq
        expected = "[أدنى المفخم]" if nisbi else "[مفخم]"
        judged = u.sifat.get("tafkheem_or_taqeeq") if getattr(u, "sifat", None) else None
        out.append(Heaviness(
            unit=i, letter=u.symbol, level=lvl, level_name=LEVEL_NAME[lvl], itbaq=itbaq,
            expected_class=expected,
            heard_class=judged["model_best"] if judged else None,
            llr=judged["llr"] if judged else None,
            realised=judged["realised"] if judged else None,
            nisbi=nisbi))
    return out


def roll_up(by_ayah: list[list[Heaviness]]) -> dict:  # type: ignore[type-arg]
    """Per-level summary: how each grade of heaviness was realised across the submission."""
    import statistics
    flat = [h for hs in by_ayah for h in hs]
    per: dict[str, dict] = {}
    for h in flat:
        d = per.setdefault(f"{h.level}_{h.level_name}", {"n": 0, "realised": 0, "llrs": [],
                                                         "itbaq": 0, "nisbi": 0})
        d["n"] += 1
        d["itbaq"] += int(h.itbaq)
        d["nisbi"] += int(h.nisbi)
        if h.realised:
            d["realised"] += 1
        if h.llr is not None:
            d["llrs"].append(h.llr)
    for d in per.values():
        llrs = d.pop("llrs")
        d["median_llr"] = round(statistics.median(llrs), 3) if llrs else None
        d["accuracy"] = round(d["realised"] / d["n"], 4) if d["n"] else None
    return {"levels": dict(sorted(per.items())), "n": len(flat),
            "note": "level is decided by the vowel context, not by the acoustic model; the head says "
                    "whether the letter was heavy, the level says how heavy it had to be"}
