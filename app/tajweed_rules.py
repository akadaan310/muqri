"""Canonical Tajweed rule parser: Uthmani text -> letter units -> Hafs 'an 'Asim rule instances.

The parser is deliberately deterministic and dependency-free. It resolves the orthography of the
Tanzil/King Fahd Uthmani script (dagger alif, small waw/yaa, silent-letter markers, bare letters
that imply sukun or assimilation) into a sequence of :class:`LetterUnit` objects, then applies
the rules of Hafs via the Shatibiyya path in continuous reading (wasl), stopping (waqf) at the
end of the ayah.

Coverage: Madd Tabi'i / Muttasil / Munfasil / Lazim Kalimi / 'Arid lil-Sukun, Ghunnah
Mushaddadah, Noon Sakinah & Tanween (Ikhfa, Idgham bi-Ghunnah, Iqlab), Meem Sakinah (Ikhfa
Shafawi, Idgham Shafawi), Qalqalah Sughra/Kubra, and Tafkheem/Tarqeeq of the isti'la letters,
Raa and the Lam of the Divine Name.

Not covered (documented limitations): Madd Lazim Harfi of the muqatta'at letters, Madd Leen,
Sakt of Hafs, Imalah in مجريها, and the optional 2-count Munfasil of the Tayyibah path.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from app.models import LetterUnit, RuleInstance, RuleType, Vowel, Word

# --- Unicode inventory ----------------------------------------------------------------------
FATHA, DAMMA, KASRA = "\u064e", "\u064f", "\u0650"
FATHATAN, DAMMATAN, KASRATAN = "\u064b", "\u064c", "\u064d"
SHADDA, SUKUN, MADDAH = "\u0651", "\u0652", "\u0653"
DAGGER_ALIF = "\u0670"
SMALL_WAW, SMALL_YAA = "\u06e5", "\u06e6"
ROUNDED_ZERO, RECT_ZERO = "\u06df", "\u06e0"
ALIF_WASLA = "ٱ"
ALIF, WAW, YAA, ALIF_MAKSURA = "ا", "و", "ي", "ى"
TEH_MARBUTA = "ة"
ALIF_MADDA = "آ"

HAMZA_FORMS = frozenset("ءأإؤئ")
QALQALAH_LETTERS = frozenset("قطبجد")
HEAVY_LETTERS = frozenset("خصضغطقظ")
IKHFA_LETTERS = frozenset("تثجدذزسشصضطظفقك")
IDGHAM_GHUNNAH_LETTERS = frozenset("ينمو")
SHAMSI_LETTERS = frozenset("تثدذرزسشصضطظلن")
MADD_BEARERS = frozenset({ALIF, WAW, YAA, ALIF_MAKSURA, SMALL_WAW, SMALL_YAA})

# Letters that a bare (unmarked) letter can be fully assimilated into when the next letter
# carries a shadda (idgham kamil). Identical letters (mithlayn) always assimilate.
_IDGHAM_TARGETS: dict[str, frozenset[str]] = {
    "ن": frozenset("ينمولر"),
    "م": frozenset("م"),
    "ل": SHAMSI_LETTERS | frozenset("ر"),
    "د": frozenset("ت"),
    "ت": frozenset("دط"),
    "ط": frozenset("ت"),
    "ذ": frozenset("ظ"),
    "ث": frozenset("ذ"),
    "ق": frozenset("ك"),
    "ب": frozenset("م"),
}

_NORMALIZE = str.maketrans(
    {
        "\u06e1": SUKUN,  # small high dotless head of khah (Madani sukun)
        "\u08f0": FATHATAN,  # open (successive) tanween forms
        "\u08f1": DAMMATAN,
        "\u08f2": KASRATAN,
        "\u0640": None,  # tatweel
        "\ufeff": None,
        "\u200c": None,
        "\u200d": None,
    }
)
# Waqf signs, rub el-hizb, sajda marks, ayah-number digits and iqlab/idgham helper marks carry
# no articulatory information for this parser.
_DROP = re.compile(r"[\u06d6-\u06dc\u06dd\u06de\u06e2\u06e3\u06e7\u06e8\u06e9\u06ea-\u06ed٠-٩۰-۹()\[\]﴿﴾0-9]")

_BASE_LETTERS = frozenset(
    [chr(c) for c in range(0x0621, 0x063B)]
    + [chr(c) for c in range(0x0641, 0x064B)]
    + [ALIF_WASLA, SMALL_WAW, SMALL_YAA]
)

_DIVINE_NAME_SKELETONS = frozenset(
    {"الله", "لله", "بالله", "والله", "فالله", "تالله", "ولله", "فلله", "أبالله", "أفالله", "اللهم"}
)


class TajweedParseError(ValueError):
    pass


@dataclass(slots=True)
class ParsedText:
    """The resolved letter sequence for one recitation segment plus the rules it requires."""

    text: str
    units: list[LetterUnit]
    words: list[Word]
    rules: list[RuleInstance] = field(default_factory=list)
    stop_at_end: bool = True

    @property
    def pronounced(self) -> list[LetterUnit]:
        return [u for u in self.units if u.pronounced]

    def word_text(self, word_index: int) -> str:
        return self.words[word_index].text


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).translate(_NORMALIZE)
    text = _DROP.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokenize_word(word: str) -> list[tuple[str, list[str]]]:
    """Split one word into (base letter, [marks]) clusters."""
    clusters: list[tuple[str, list[str]]] = []
    for ch in word:
        if ch == ALIF_MADDA:
            # NFC composes alif + maddah into U+0622. After an open (fatha) letter it is a long
            # alif carrying the madd sign; otherwise it spells hamza + fatha + long alif.
            if clusters and (FATHA in clusters[-1][1]):
                clusters.append((ALIF, [MADDAH]))
            else:
                clusters.append(("ء", [FATHA]))
                clusters.append((ALIF, [MADDAH]))
        elif ch in _BASE_LETTERS:
            clusters.append((ch, []))
        elif unicodedata.category(ch) == "Mn":
            if not clusters:
                raise TajweedParseError(f"Mark {ch!r} (U+{ord(ch):04X}) has no base letter in {word!r}")
            clusters[-1][1].append(ch)
        else:
            raise TajweedParseError(f"Unsupported character {ch!r} (U+{ord(ch):04X}) in {word!r}")
    return clusters


def _make_unit(index: int, word_index: int, base: str, marks: list[str]) -> LetterUnit:
    unit = LetterUnit(index=index, word_index=word_index, char=base, text=base + "".join(marks))
    for m in marks:
        if m in (FATHA, FATHATAN):
            unit.vowel = Vowel.FATHA
        elif m in (DAMMA, DAMMATAN):
            unit.vowel = Vowel.DAMMA
        elif m in (KASRA, KASRATAN):
            unit.vowel = Vowel.KASRA
        if m in (FATHATAN, DAMMATAN, KASRATAN):
            unit.tanween = True
        elif m == SHADDA:
            unit.shadda = True
        elif m == SUKUN:
            unit.sukun = unit.explicit_sukun = True
        elif m == MADDAH:
            unit.maddah_sign = True
        elif m in (ROUNDED_ZERO, RECT_ZERO):
            unit.silent = True
    return unit


class TajweedParser:
    """Turns Uthmani text into :class:`ParsedText` with all applicable rule instances."""

    def __init__(self, *, stop_at_end: bool = True, start_of_recitation: bool = True) -> None:
        self.stop_at_end = stop_at_end
        self.start_of_recitation = start_of_recitation

    # ------------------------------------------------------------------ orthography resolution
    def parse(self, text: str) -> ParsedText:
        norm = normalize_text(text)
        if not norm:
            raise TajweedParseError("Target text is empty after normalization")
        units: list[LetterUnit] = []
        words: list[Word] = []
        for w_idx, raw in enumerate(norm.split(" ")):
            word = Word(index=w_idx, text=raw)
            for base, marks in _tokenize_word(raw):
                has_dagger = DAGGER_ALIF in marks
                marks = [m for m in marks if m != DAGGER_ALIF]
                unit = _make_unit(len(units), w_idx, base, marks)
                units.append(unit)
                word.unit_indices.append(unit.index)
                if has_dagger:
                    # A dagger alif on ى / bare و turns the bearer silent and supplies the madd.
                    if base in (ALIF_MAKSURA, WAW) and unit.vowel is None:
                        unit.silent = True
                    madd = LetterUnit(
                        index=len(units), word_index=w_idx, char=ALIF, text=DAGGER_ALIF,
                        madd_letter=True, synthetic=True, maddah_sign=MADDAH in marks,
                    )
                    units.append(madd)
                    word.unit_indices.append(madd.index)
            if not word.unit_indices:
                raise TajweedParseError(f"Word {raw!r} contains no letters")
            words.append(word)

        parsed = ParsedText(text=norm, units=units, words=words, stop_at_end=self.stop_at_end)
        self._resolve_wasla(parsed)
        self._resolve_madd_letters(parsed)
        self._resolve_assimilation(parsed)
        self._resolve_implicit_sukun(parsed)
        if self.stop_at_end:
            self._apply_waqf(parsed)
        self._resolve_elision(parsed)
        parsed.rules = self._detect_rules(parsed)
        return parsed

    def _resolve_wasla(self, p: ParsedText) -> None:
        for u in p.units:
            if u.char != ALIF_WASLA:
                continue
            first_of_text = u.index == 0
            if first_of_text and self.start_of_recitation:
                # Initial hamzat al-wasl is pronounced: fatha before the definite article,
                # damma when the third letter carries damma, otherwise kasra.
                word = p.words[u.word_index].unit_indices
                pos = word.index(u.index)
                nxt = p.units[word[pos + 1]] if pos + 1 < len(word) else None
                third = p.units[word[pos + 2]] if pos + 2 < len(word) else None
                u.char = "ء"
                if nxt is not None and nxt.char == "ل":
                    u.vowel = Vowel.FATHA
                elif third is not None and third.vowel == Vowel.DAMMA:
                    u.vowel = Vowel.DAMMA
                else:
                    u.vowel = Vowel.KASRA
            else:
                u.silent = True

    def _prev_in_word(self, p: ParsedText, u: LetterUnit, *, pronounced: bool = True) -> LetterUnit | None:
        idxs = p.words[u.word_index].unit_indices
        pos = idxs.index(u.index)
        for j in reversed(idxs[:pos]):
            cand = p.units[j]
            if not pronounced or cand.pronounced:
                return cand
        return None

    def _resolve_madd_letters(self, p: ParsedText) -> None:
        for u in p.units:
            if u.silent or u.synthetic or u.char not in MADD_BEARERS:
                continue
            if u.vowel is not None or u.shadda or u.explicit_sukun:
                continue  # a consonantal waw/yaa (or a voweled alif-seat)
            prev = self._prev_in_word(p, u)
            if prev is None:
                if u.char in (ALIF, ALIF_MAKSURA):
                    u.silent = True
                continue
            if prev.tanween and u.char in (ALIF, ALIF_MAKSURA):
                u.silent = True  # alif after tanween fath is silent in wasl
            elif (
                (u.char in (ALIF, ALIF_MAKSURA) and prev.vowel == Vowel.FATHA)
                or (u.char in (WAW, SMALL_WAW) and prev.vowel == Vowel.DAMMA)
                or (u.char in (YAA, ALIF_MAKSURA, SMALL_YAA) and prev.vowel == Vowel.KASRA)
            ):
                u.madd_letter = True
            elif u.char in (ALIF, ALIF_MAKSURA, SMALL_WAW, SMALL_YAA):
                u.silent = True

    def _next_unit(self, p: ParsedText, i: int, *, pronounced: bool = True) -> LetterUnit | None:
        for j in range(i + 1, len(p.units)):
            if not pronounced or p.units[j].pronounced:
                return p.units[j]
        return None

    def _prev_unit(self, p: ParsedText, i: int) -> LetterUnit | None:
        for j in range(i - 1, -1, -1):
            if p.units[j].pronounced:
                return p.units[j]
        return None

    def _resolve_assimilation(self, p: ParsedText) -> None:
        for u in p.units:
            if u.silent or u.madd_letter or u.vowel is not None or u.explicit_sukun or u.shadda:
                continue
            nxt = self._next_unit(p, u.index)
            if nxt is None or not nxt.shadda:
                continue
            if nxt.char == u.char or nxt.char in _IDGHAM_TARGETS.get(u.char, frozenset()):
                u.silent = True
                u.assimilated = True
                u.sukun = True

    def _resolve_implicit_sukun(self, p: ParsedText) -> None:
        for u in p.units:
            if u.pronounced and not u.madd_letter and u.vowel is None:
                u.sukun = True

    def _last_pronounced(self, p: ParsedText) -> LetterUnit | None:
        for u in reversed(p.units):
            if u.pronounced:
                return u
        return None

    def _apply_waqf(self, p: ParsedText) -> None:
        # Tanween fath + alif at the stop becomes a 2-count madd ('iwad).
        tail = p.units[-1]
        if tail.silent and tail.char in (ALIF, ALIF_MAKSURA) and not tail.assimilated:
            prev = self._prev_in_word(p, tail)
            if prev is not None and prev.tanween and prev.vowel == Vowel.FATHA:
                tail.silent = False
                tail.madd_letter = True
                prev.tanween = False
                return
        last = self._last_pronounced(p)
        if last is None or last.madd_letter:
            return
        last.vowel = None
        last.tanween = False
        last.sukun = True
        if last.char == TEH_MARBUTA:
            last.char = "ه"

    def _resolve_elision(self, p: ParsedText) -> None:
        # A word-final madd letter is dropped before a sakin/mushaddad letter of the next word.
        for u in p.units:
            if not (u.pronounced and u.madd_letter):
                continue
            nxt = self._next_unit(p, u.index)
            if nxt is None or nxt.word_index == u.word_index:
                continue
            if nxt.sukun or nxt.shadda:
                u.silent = True
                u.elided = True
                u.madd_letter = False

    # ------------------------------------------------------------------ rule detection
    def _detect_rules(self, p: ParsedText) -> list[RuleInstance]:
        rules: list[RuleInstance] = []
        last = self._last_pronounced(p)
        last_idx = last.index if last is not None else -1
        covered_shadda: set[int] = set()

        def add(rule: RuleType, units: list[int], expected: tuple[float, float] | None,
                letter: str | None = None, detail: str = "", waqf: bool = False) -> None:
            anchor = p.units[units[0]]
            rules.append(RuleInstance(
                rule_type=rule, word_index=anchor.word_index, word=p.word_text(anchor.word_index),
                unit_indices=units, expected_harakat=expected, letter=letter, detail=detail,
                at_waqf=waqf,
            ))

        for u in p.units:
            # ---- Madd --------------------------------------------------------------------
            if u.pronounced and u.madd_letter:
                carrier = self._prev_unit(p, u.index)
                span = [carrier.index, u.index] if carrier is not None else [u.index]
                nxt = self._next_unit(p, u.index)
                is_word_final = nxt is None or nxt.word_index != u.word_index
                if nxt is not None and nxt.char in HAMZA_FORMS and not is_word_final:
                    at_stop = p.stop_at_end and nxt.index == last_idx
                    add(RuleType.MADD_MUTTASIL, span, (4, 6) if at_stop else (4, 5),
                        detail="madd before hamza in the same word", waqf=at_stop)
                elif nxt is not None and nxt.char in HAMZA_FORMS and is_word_final:
                    add(RuleType.MADD_MUNFASIL, span, (4, 5),
                        detail="madd at word end before hamza of next word")
                elif nxt is not None and not is_word_final and (nxt.shadda or nxt.explicit_sukun) \
                        and not (p.stop_at_end and nxt.index == last_idx and not nxt.shadda):
                    add(RuleType.MADD_LAZIM, span, (6, 6), detail="madd before an original sukun")
                elif p.stop_at_end and nxt is not None and nxt.index == last_idx:
                    add(RuleType.MADD_ARID, span, (2, 6), detail="madd before a sukun caused by stopping",
                        waqf=True)
                else:
                    detail = "madd 'iwad at stop" if u.index == last_idx and u.char in (ALIF, ALIF_MAKSURA) \
                        and not u.synthetic and carrier is not None and carrier.vowel == Vowel.FATHA \
                        and p.stop_at_end else "natural madd"
                    if u.char in (SMALL_WAW, SMALL_YAA):
                        detail = "madd al-silah al-sughra"
                    add(RuleType.MADD_TABII, span, (2, 2), detail=detail)

            # ---- Noon sakinah & tanween -----------------------------------------------
            is_noon_sakinah = u.char == "ن" and u.sukun and not u.shadda
            if (is_noon_sakinah or (u.tanween and u.pronounced)) and not (u.index == last_idx and p.stop_at_end):
                tgt = self._next_unit(p, u.index)
                if tgt is not None:
                    cross_word = tgt.word_index != u.word_index
                    detail = "tanween" if u.tanween else "noon sakinah"
                    if tgt.char in IDGHAM_GHUNNAH_LETTERS and cross_word:
                        add(RuleType.IDGHAM_GHUNNAH, [u.index, tgt.index], (2, 2), letter="ن", detail=detail)
                        covered_shadda.add(tgt.index)
                    elif tgt.char == "ب":
                        add(RuleType.IQLAB, [u.index], (2, 2), letter="ن", detail=detail)
                    elif tgt.char in IKHFA_LETTERS:
                        add(RuleType.IKHFA, [u.index], (2, 2), letter="ن", detail=detail)
                    elif tgt.char in "لر" and cross_word:
                        covered_shadda.add(tgt.index)  # idgham without ghunnah: not a nasal rule

            # ---- Meem sakinah -----------------------------------------------------------
            if u.char == "م" and u.sukun and not u.shadda and not (u.index == last_idx and p.stop_at_end):
                tgt = self._next_unit(p, u.index)
                if tgt is not None and tgt.char == "ب":
                    add(RuleType.IKHFA_SHAFAWI, [u.index], (2, 2), letter="م")
                elif tgt is not None and tgt.char == "م":
                    add(RuleType.IDGHAM_SHAFAWI, [u.index, tgt.index], (2, 2), letter="م")
                    covered_shadda.add(tgt.index)

            # ---- Ghunnah mushaddadah -------------------------------------------------------
            if u.pronounced and u.char in "نم" and u.shadda and u.index not in covered_shadda:
                add(RuleType.GHUNNAH, [u.index], (2, 2), letter=u.char, detail="mushaddad")

            # ---- Qalqalah --------------------------------------------------------------
            if u.pronounced and u.char in QALQALAH_LETTERS and u.sukun:
                kubra = p.stop_at_end and u.index == last_idx
                add(RuleType.QALQALAH, [u.index], None, letter=u.char,
                    detail="kubra" if kubra else "sughra", waqf=kubra)

            # ---- Tafkheem / Tarqeeq --------------------------------------------------------
            if u.pronounced and not u.madd_letter:
                self._detect_weight(p, u, add)

        return rules

    def _detect_weight(self, p: ParsedText, u: LetterUnit, add) -> None:  # type: ignore[no-untyped-def]
        nxt = self._next_unit(p, u.index)
        span = [u.index]
        if nxt is not None and nxt.madd_letter and nxt.word_index == u.word_index:
            span.append(nxt.index)

        if u.char in HEAVY_LETTERS:
            if u.vowel in (Vowel.FATHA, Vowel.DAMMA):
                add(RuleType.TAFKHEEM, span, None, letter=u.char, detail="isti'la letter")
            return

        if u.char == "ر":
            heavy = self._raa_is_heavy(p, u)
            if heavy is not None:
                add(RuleType.TAFKHEEM if heavy else RuleType.TARQEEQ, span, None, letter="ر",
                    detail="raa")
            return

        if u.char == "ل" and u.shadda and nxt is not None and nxt.char == "ه" \
                and nxt.word_index == u.word_index:
            skeleton = "".join(
                p.units[i].char for i in p.words[u.word_index].unit_indices if not p.units[i].synthetic
            )
            if skeleton.startswith("ء"):  # an initial hamzat al-wasl made audible at the start
                skeleton = ALIF + skeleton[1:]
            skeleton = skeleton.replace(ALIF_WASLA, ALIF)
            if skeleton not in _DIVINE_NAME_SKELETONS:
                return
            prev = self._prev_unit(p, u.index)
            if prev is None:
                heavy = True
            elif prev.tanween or prev.sukun:
                heavy = False  # helper kasra (tanween/sakin before hamzat al-wasl)
            elif prev.madd_letter:
                heavy = prev.char not in (YAA, SMALL_YAA)
            else:
                heavy = prev.vowel != Vowel.KASRA
            add(RuleType.TAFKHEEM if heavy else RuleType.TARQEEQ, span, None, letter="ل",
                detail="lam of the Divine Name")

    def _raa_is_heavy(self, p: ParsedText, u: LetterUnit) -> bool | None:
        if u.vowel in (Vowel.FATHA, Vowel.DAMMA):
            return True
        if u.vowel == Vowel.KASRA:
            return False
        prev = self._prev_unit(p, u.index)
        if prev is None:
            return None
        if prev.sukun and not prev.madd_letter:
            if prev.char == YAA:
                return False
            prev = self._prev_unit(p, prev.index)
            if prev is None:
                return None
        if prev.madd_letter:
            return prev.char not in (YAA, ALIF_MAKSURA, SMALL_YAA)
        if prev.vowel == Vowel.KASRA:
            # Raa sakinah after an original kasra is light unless a heavy letter follows in-word.
            follow = self._next_unit(p, u.index)
            if follow is not None and follow.word_index == u.word_index and follow.char in HEAVY_LETTERS \
                    and follow.vowel != Vowel.KASRA:
                return True
            return False
        return True


def parse_text(text: str, *, stop_at_end: bool = True) -> ParsedText:
    return TajweedParser(stop_at_end=stop_at_end).parse(text)
