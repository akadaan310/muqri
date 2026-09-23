"""Bind the parser's located rule instances to the acoustic engine's phoneme units.

Two halves of the engine know different things and have never been joined:

* ``TajweedParser`` reads the Uthmani text and says **which named rule applies where** and what it
  requires — "madd munfaṣil at word 3, expected 4–5 counts", "tafkhīm of ر: fatḥah/ḍammah".
* the counterfactual engine reads the audio and says **what happened at each phoneme unit** — its
  identity confidence, its ten sifat LLRs, its duration in the reciter's own counts.

Without the join the engine can only report an error *type* ("a madd length error here"). With it,
every one of the parser's 40 rule types becomes a graded named instance: *"madd munfaṣil in
وَمَا أَنزَلَ — you gave 2.1 counts, it requires 4."* That is the difference between a detector and
mastery-level feedback, and it upgrades every rule at once.

The bridge is the word index. The parser numbers words of the Uthmani text; ``muaalem_dump.word_spans``
gives each of those same words its phoneme character span (verified one-to-one: 7 parser words ↔ 7
spans on 2:2, with رَيْبَ → ``رَيبَ``). Inside a word the rule is matched to units by the symbols that
can realise it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.models import RuleType

# Phoneme symbols that can realise each rule family. The phonetizer gives several rules their own
# symbol, which makes those exact rather than inferred: ں = ikhfa noon, ۾ = iqlab meem,
# ڇ = qalqala release.
MADD_LETTERS = frozenset("اۥۦ")
NASAL = frozenset("نمں۾")
SHORT_V = frozenset("َُِ")

TARGETS: dict[RuleType, frozenset[str]] = {
    # mudood — the madd letter itself; its run length is the count
    RuleType.MADD_TABII: MADD_LETTERS, RuleType.MADD_MUTTASIL: MADD_LETTERS,
    RuleType.MADD_MUNFASIL: MADD_LETTERS, RuleType.MADD_LAZIM: MADD_LETTERS,
    RuleType.MADD_ARID: MADD_LETTERS, RuleType.MADD_LEEN: frozenset("وي"),
    RuleType.MADD_BADAL: MADD_LETTERS, RuleType.MADD_IWAD: MADD_LETTERS,
    RuleType.MADD_SILAH_SUGHRA: frozenset("ۥۦ"), RuleType.MADD_SILAH_KUBRA: frozenset("ۥۦ"),
    # noon sakinah / tanween — each hukm has a distinct realisation
    RuleType.IZHAR_HALQI: frozenset("ن"), RuleType.IKHFA: frozenset("ں"),
    RuleType.IDGHAM_GHUNNAH: NASAL, RuleType.IDGHAM_NO_GHUNNAH: frozenset("لر"),
    RuleType.IQLAB: frozenset("۾م"),
    # meem sakinah
    # the hidden meem before a ba takes the same ۾ symbol as iqlab, held x3 for its two-count ghunnah
    RuleType.IKHFA_SHAFAWI: frozenset("۾م"), RuleType.IDGHAM_SHAFAWI: frozenset("م"),
    RuleType.IZHAR_SHAFAWI: frozenset("م"),
    RuleType.GHUNNAH: frozenset("نم"),
    RuleType.QALQALAH: frozenset("ڇقطبجد"),
    # idgham of other letters: the merged (doubled) consonant
    RuleType.IDGHAM_MITHLAYN: frozenset(), RuleType.IDGHAM_MUTAJANISAYN: frozenset(),
    RuleType.IDGHAM_MUTAQARIBAYN: frozenset(),
    RuleType.HAMZAT_WASL: frozenset("ء"), RuleType.SAKT: frozenset(),
}

# Assimilation moves the letter into the NEXT word. The parser locates idgham at the word holding the
# assimilated letter, but the phonetizer deletes it there and doubles the receiving letter in the word
# that follows, so the target is across the boundary. Measured: without this, idgham_shafawi bound at
# 0.10 and ikhfa_shafawi at 0.06.
BOUNDARY = frozenset({
    RuleType.IDGHAM_GHUNNAH, RuleType.IDGHAM_NO_GHUNNAH, RuleType.IDGHAM_SHAFAWI,
    RuleType.IDGHAM_MITHLAYN, RuleType.IDGHAM_MUTAJANISAYN, RuleType.IDGHAM_MUTAQARIBAYN,
})
# Ikhfa shafawi HIDES the meem before a ba; it is not absorbed and not doubled, so it stays a plain
# meem in its own word (verified on 2:10, where the doubled meem is the idgham shafawi, not this).
# Hamzat al-wasl is DROPPED when joined, so there is no hamza to point at — the rule binds to the
# word's first unit, which is where the hamza would have been (al-kitabu -> "l-kitaabu").
FIRST_UNIT_OF_WORD = frozenset({RuleType.HAMZAT_WASL})

# A shadda letter is TWO letters — a sakin half and a voweled half — collided into one articulation,
# so it occupies real time and the engine must expect it. The phonetizer writes that as a repeated
# run, and the run length says which structure it is:
#     non-nasal shadda / idgham   x2   sakin half + voweled half (the collision)
#     nasal shadda / idgham bi-ghunnah (m, n)   x4   ghunnah held for TWO counts
#     ikhfa noon (ں)              x3   hidden, two-count ghunnah
# Verified on min rabbihim -> r x2, inna -> n x4, alladhina -> l x2.
SHADDA_NASAL_RUN = 4
SHADDA_PLAIN_RUN = 2

# rules whose verdict is a LENGTH (expected_harakat), not a substitution
DURATIONAL = frozenset({
    RuleType.MADD_TABII, RuleType.MADD_MUTTASIL, RuleType.MADD_MUNFASIL, RuleType.MADD_LAZIM,
    RuleType.MADD_ARID, RuleType.MADD_LEEN, RuleType.MADD_BADAL, RuleType.MADD_IWAD,
    RuleType.MADD_SILAH_SUGHRA, RuleType.MADD_SILAH_KUBRA, RuleType.GHUNNAH, RuleType.IKHFA,
    RuleType.IKHFA_SHAFAWI, RuleType.TAWASSUT,
    # the merged letter is held: nasals for two counts of ghunnah, others for the collision
    RuleType.IDGHAM_GHUNNAH, RuleType.IDGHAM_SHAFAWI,
})

# rules judged by a muaalem sifat head rather than by sequence or length
ATTRIBUTE_HEAD: dict[RuleType, str] = {
    RuleType.HAMS: "hams_or_jahr", RuleType.JAHR: "hams_or_jahr",
    RuleType.SHIDDAH: "shidda_or_rakhawa", RuleType.TAWASSUT: "shidda_or_rakhawa",
    RuleType.RAKHAWAH: "shidda_or_rakhawa", RuleType.ITBAQ: "itbaq", RuleType.SAFIR: "safeer",
    RuleType.TAFASHHI: "tafashie", RuleType.ISTITAALAH: "istitala", RuleType.TAKREER: "tikraar",
    RuleType.QALQALAH: "qalqla", RuleType.TAFKHEEM: "tafkheem_or_taqeeq",
    RuleType.TARQEEQ: "tafkheem_or_taqeeq", RuleType.JAWAZ_WAJHAYN: "tafkheem_or_taqeeq",
    RuleType.GHUNNAH: "ghonna",
}


def ph_units(phonemes: str) -> list[tuple[str, int, int]]:
    """Runs of one repeated symbol: ``(symbol, first_char, last_char)``, 0-based inclusive.

    Mirrors ``CtcGop.ph_units`` so Python and Julia index the same units.
    """
    out: list[tuple[str, int, int]] = []
    i = 0
    while i < len(phonemes):
        j = i
        while j + 1 < len(phonemes) and phonemes[j + 1] == phonemes[i]:
            j += 1
        out.append((phonemes[i], i, j))
        i = j + 1
    return out


@dataclass(slots=True)
class BoundRule:
    """One located rule instance, tied to the phoneme units that realise it."""

    rule_type: str
    word_index: int
    word: str
    letter: str | None
    detail: str
    at_waqf: bool
    mechanism: str                      # durational | attribute | segmental
    expected_counts: tuple[float, float] | None
    head: str | None                    # sifat head that judges it, when there is one
    shadda: str | None = None           # "nasal" (2-count ghunnah) | "plain" (sakin+voweled collision)
    unit_indices: list[int] = field(default_factory=list)   # into ph_units(phonemes)
    ph_span: tuple[int, int] | None = None                  # character span in the phoneme string

    def to_dict(self) -> dict:  # type: ignore[type-arg]
        return {"rule": self.rule_type, "word_index": self.word_index, "word": self.word,
                "letter": self.letter, "detail": self.detail, "at_waqf": self.at_waqf,
                "mechanism": self.mechanism, "expected_counts": list(self.expected_counts)
                if self.expected_counts else None, "head": self.head, "shadda": self.shadda,
                "units": self.unit_indices, "ph_span": list(self.ph_span) if self.ph_span else None}


def _mechanism(rt: RuleType) -> str:
    if rt in DURATIONAL:
        return "durational"
    if rt in ATTRIBUTE_HEAD:
        return "attribute"
    return "segmental"


def bind(parsed, phonemes: str, word_ph: list[list[int]]) -> list[BoundRule]:  # type: ignore[no-untyped-def]
    """Attach every rule the parser located to the phoneme units that realise it.

    ``parsed``   result of ``TajweedParser.parse(uthmani)``
    ``phonemes`` the phonetizer's phoneme string for the same text
    ``word_ph``  per-word ``[first, last)`` phoneme character spans (``muaalem_dump.word_spans``);
                 −1 marks a word with no phonemes.
    """
    units = ph_units(phonemes)
    out: list[BoundRule] = []
    # A word can carry the same rule twice (وَيُقِيمُونَ has two madd tabii). Each instance must bind
    # to a DIFFERENT run, otherwise both report on the same audio and one real error is invisible.
    used: dict[tuple[int, str], set[int]] = {}
    for r in parsed.rules:
        rt = r.rule_type
        w = r.word_index
        span = word_ph[w] if 0 <= w < len(word_ph) else None
        if not span or span[0] < 0 or span[1] <= span[0]:
            continue
        lo, hi = int(span[0]), int(span[1])
        if rt in BOUNDARY:
            # The assimilated letter is word-FINAL and lands on the next word's FIRST letter, so look
            # there — not from the start of the current word, which may itself open with an unrelated
            # shadda from the preceding merge (min rabbihim: the mim of "min" is already doubled).
            nxt = word_ph[w + 1] if w + 1 < len(word_ph) else None
            if nxt and nxt[0] >= 0 and nxt[1] > nxt[0]:
                lo, hi = int(nxt[0]), int(nxt[1])
        targets = TARGETS.get(rt, frozenset())
        if r.letter:
            targets = targets | {r.letter} if targets else frozenset({r.letter})

        taken = used.setdefault((w, rt.value), set())
        idxs: list[int] = []
        if rt in FIRST_UNIT_OF_WORD:
            first = next((i for i, (_s, a, _b) in enumerate(units) if a >= lo), None)
            if first is not None and units[first][2] < hi:
                idxs = [first]
        for i, (sym, a, b) in enumerate(units) if not idxs else []:
            if a < lo or b >= hi or i in taken:
                continue
            if rt in BOUNDARY:
                # the merged letter is the doubled run (the shadda), whatever letter it landed on
                if b > a and sym not in SHORT_V and sym not in MADD_LETTERS:
                    idxs.append(i)
            elif not targets:
                if b > a and sym not in SHORT_V and sym not in MADD_LETTERS:
                    idxs.append(i)
            elif sym in targets:
                idxs.append(i)
        if not idxs and rt in BOUNDARY:
            # idgham naqis (into waw / yaa) keeps the receiving letter single but nasalised, so there
            # is no doubled run to find; fall back to that word's first unit
            first = next((i for i, (_s, a, _b) in enumerate(units) if a >= lo and i not in taken), None)
            if first is not None and units[first][2] < hi:
                idxs = [first]
        if not idxs:
            continue
        # the parser may locate several instances of the same rule in one word; keep as many units as
        # it named, preferring the longest runs (a madd is the long run, not an incidental letter)
        want = max(1, len(r.unit_indices) // 2) if rt in DURATIONAL else len(idxs)
        if rt in DURATIONAL and len(idxs) > want:
            idxs = sorted(sorted(idxs, key=lambda i: -(units[i][2] - units[i][1]))[:want])
        if rt in BOUNDARY and len(idxs) > 1:
            idxs = idxs[:1]                     # one assimilation, one receiving letter
        taken.update(idxs)
        run = units[idxs[0]][2] - units[idxs[0]][1] + 1 if idxs else 0
        shadda = ("nasal" if run >= SHADDA_NASAL_RUN else "plain") if run >= SHADDA_PLAIN_RUN and \
            units[idxs[0]][0] not in MADD_LETTERS else None

        out.append(BoundRule(
            rule_type=rt.value, word_index=w, word=r.word, letter=r.letter, detail=r.detail,
            at_waqf=bool(getattr(r, "at_waqf", False)), mechanism=_mechanism(rt),
            expected_counts=r.expected_harakat, head=ATTRIBUTE_HEAD.get(rt),
            shadda=shadda, unit_indices=idxs,
            ph_span=(units[idxs[0]][1], units[idxs[-1]][2])))
    return out
