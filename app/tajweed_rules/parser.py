"""Canonical Tajweed rule parser: Uthmani text -> letter units -> Hafs 'an 'Asim rule instances.

The parser is deterministic and dependency-free. It resolves the orthography of the Tanzil/King
Fahd Uthmani script (dagger alif, small waw/yaa, silent-letter markers, bare letters that imply
sukun or assimilation, the sakt sign, the muqatta'at letters) into :class:`LetterUnit` objects and
derives every rule instance the text requires.

The text is split into **phrases** at every stop (waqf) and sakt. Waqf changes are applied to the
end of each phrase, hamzat al-wasl is pronounced at the start of each phrase (ibtida'), and no
rule crosses a phrase boundary (for example, there is no Munfasil or Ikhfa across a stop).

Rule targets depend on the **tareeq**: Shatibiyyah (Munfasil and Silah Kubra 4–5) or Tayyibah
with qasr al-munfasil (2).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from app.models import LetterUnit, RuleInstance, RuleType, Tareeq, Vowel, Word

# --- Unicode inventory ----------------------------------------------------------------------
FATHA, DAMMA, KASRA = "\u064e", "\u064f", "\u0650"
FATHATAN, DAMMATAN, KASRATAN = "\u064b", "\u064c", "\u064d"
SHADDA, SUKUN, MADDAH = "\u0651", "\u0652", "\u0653"
DAGGER_ALIF = "\u0670"
SMALL_WAW, SMALL_YAA = "\u06e5", "\u06e6"
ROUNDED_ZERO, RECT_ZERO = "\u06df", "\u06e0"
SAKT_MARK = "\u06dc"
ALIF_WASLA = "ٱ"
ALIF, WAW, YAA, ALIF_MAKSURA = "ا", "و", "ي", "ى"
TEH_MARBUTA = "ة"
ALIF_MADDA = "آ"

HAMZA_FORMS = frozenset("ءأإؤئ")
THROAT_LETTERS = HAMZA_FORMS | frozenset("هعحغخ")
QALQALAH_LETTERS = frozenset("قطبجد")
HEAVY_LETTERS = frozenset("خصضغطقظ")
ITBAQ_LETTERS = frozenset("صضطظ")
IKHFA_LETTERS = frozenset("تثجدذزسشصضطظفقك")
IDGHAM_GHUNNAH_LETTERS = frozenset("ينمو")
SHAMSI_LETTERS = frozenset("تثدذرزسشصضطظلن")
MADD_BEARERS = frozenset({ALIF, WAW, YAA, ALIF_MAKSURA, SMALL_WAW, SMALL_YAA})
# Sifaat classes (Ibn al-Jazari): hams فحثه شخص سكت, shiddah أجد قط بكت, tawassut لن عمر.
HAMS_LETTERS = frozenset("فحثهشخصسكت")
SHIDDAH_LETTERS = HAMZA_FORMS | frozenset("جدقطبكت")
TAWASSUT_LETTERS = frozenset("لنعمر")
SAFIR_LETTERS = frozenset("صسز")

# Letters that a bare (unmarked) letter is fully assimilated into when the next letter carries a
# shadda (idgham kamil). Identical letters (mithlayn) always assimilate.
_IDGHAM_TARGETS: dict[str, frozenset[str]] = {
    "ن": frozenset("ينمولر"),
    "م": frozenset("م"),
    "ل": SHAMSI_LETTERS | frozenset("ر"),
    "د": frozenset("ت"),
    "ت": frozenset("دطث"),
    "ط": frozenset("ت"),
    "ذ": frozenset("ظ"),
    "ث": frozenset("ذ"),
    "ق": frozenset("ك"),
    "ب": frozenset("م"),
}
# Same makhraj, different sifaat.
MUTAJANISAYN_PAIRS = frozenset({("ت", "د"), ("ت", "ط"), ("د", "ت"), ("ط", "ت"), ("ث", "ذ"), ("ذ", "ظ"), ("ب", "م")})
# Close makhraj.
MUTAQARIBAYN_PAIRS = frozenset({("ل", "ر"), ("ق", "ك"), ("ت", "ث")})

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
# no articulatory information (the sakt sign is read before this is applied).
_DROP = re.compile(r"[\u06d6-\u06dc\u06dd\u06de\u06e2\u06e3\u06e7\u06e8\u06e9\u06ea-\u06ed٠-٩۰-۹()\[\]﴿﴾0-9]")
_VOWEL_MARKS = frozenset({FATHA, DAMMA, KASRA, FATHATAN, DAMMATAN, KASRATAN, SHADDA, SUKUN})

_BASE_LETTERS = frozenset(
    [chr(c) for c in range(0x0621, 0x063B)]
    + [chr(c) for c in range(0x0641, 0x064B)]
    + [ALIF_WASLA, SMALL_WAW, SMALL_YAA]
)

_DIVINE_NAME_SKELETONS = frozenset(
    {"الله", "لله", "بالله", "والله", "فالله", "تالله", "ولله", "فلله", "أبالله", "أفالله", "اللهم"}
)

Cluster = tuple[str, list[str]]

# Muqatta'at: each letter is read as its name. Three-letter names with a middle madd letter carry
# Madd Lazim Harfi (6); two-letter names a natural madd (2); ʿayn a Madd Leen (4 or 6).
_LETTER_NAMES: dict[str, list[Cluster]] = {
    "ا": [("ء", [FATHA]), ("ل", [SUKUN]), ("ف", [SUKUN])],
    "ل": [("ل", [FATHA]), ("ا", []), ("م", [SUKUN])],
    "م": [("م", [KASRA]), ("ي", []), ("م", [SUKUN])],
    "ص": [("ص", [FATHA]), ("ا", []), ("د", [SUKUN])],
    "ر": [("ر", [FATHA]), ("ا", [])],
    "ك": [("ك", [FATHA]), ("ا", []), ("ف", [SUKUN])],
    "ه": [("ه", [FATHA]), ("ا", [])],
    "ي": [("ي", [FATHA]), ("ا", [])],
    "ع": [("ع", [FATHA]), ("ي", [SUKUN]), ("ن", [SUKUN])],
    "ط": [("ط", [FATHA]), ("ا", [])],
    "س": [("س", [KASRA]), ("ي", []), ("ن", [SUKUN])],
    "ح": [("ح", [FATHA]), ("ا", [])],
    "ق": [("ق", [FATHA]), ("ا", []), ("ف", [SUKUN])],
    "ن": [("ن", [DAMMA]), ("و", []), ("ن", [SUKUN])],
}
MUQATTAAT = frozenset({"الم", "المص", "الر", "المر", "كهيعص", "طه", "طسم", "طس", "يس", "ص", "حم", "عسق", "ق", "ن"})

# Madd targets in harakat per tareeq.
_MADD_TARGETS: dict[RuleType, tuple[float, float]] = {
    RuleType.MADD_TABII: (2, 2),
    RuleType.MADD_MUTTASIL: (4, 5),
    RuleType.MADD_MUNFASIL: (4, 5),
    RuleType.MADD_LAZIM: (6, 6),
    RuleType.MADD_ARID: (2, 6),
    RuleType.MADD_LEEN: (2, 6),
    RuleType.MADD_BADAL: (2, 2),
    RuleType.MADD_IWAD: (2, 2),
    RuleType.MADD_SILAH_SUGHRA: (2, 2),
    RuleType.MADD_SILAH_KUBRA: (4, 5),
}
_TAYYIBAH_QASR = {RuleType.MADD_MUNFASIL: (2.0, 2.0), RuleType.MADD_SILAH_KUBRA: (2.0, 2.0)}
# Sukoon spectrum: expected sakin-letter duration relative to one harakah.
SUKOON_RATIOS: dict[RuleType, float] = {RuleType.SHIDDAH: 1.0, RuleType.TAWASSUT: 1.5, RuleType.RAKHAWAH: 2.2}


def madd_target(rule_type: RuleType, tareeq: Tareeq = Tareeq.SHATIBIYYAH) -> tuple[float, float]:
    if tareeq is Tareeq.TAYYIBAH and rule_type in _TAYYIBAH_QASR:
        return _TAYYIBAH_QASR[rule_type]
    return _MADD_TARGETS[rule_type]


class TajweedParseError(ValueError):
    pass


@dataclass(slots=True)
class ParsedText:
    """The resolved letter sequence for one recitation passage plus the rules it requires."""

    text: str
    units: list[LetterUnit]
    words: list[Word]
    rules: list[RuleInstance] = field(default_factory=list)
    stop_at_end: bool = True
    tareeq: Tareeq = Tareeq.SHATIBIYYAH

    @property
    def pronounced(self) -> list[LetterUnit]:
        return [u for u in self.units if u.pronounced]

    def word_text(self, word_index: int) -> str:
        return self.words[word_index].text

    def phrase_final_words(self) -> set[int]:
        """Indices of words followed by a stop (including the final word when stopping)."""
        out = {w.index for w in self.words if w.stop_after}
        if self.stop_at_end and self.words:
            out.add(self.words[-1].index)
        return out


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).translate(_NORMALIZE)
    text = _DROP.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokenize_word(word: str) -> list[Cluster]:
    """Split one word into (base letter, [marks]) clusters."""
    clusters: list[Cluster] = []
    for ch in word:
        if ch == ALIF_MADDA:
            # NFC composes alif + maddah into U+0622. After an open (fatha) letter it is a long
            # alif carrying the madd sign; otherwise it spells hamza + fatha + long alif.
            if clusters and FATHA in clusters[-1][1]:
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


def _skeleton(word: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", word) if unicodedata.category(c) != "Mn").replace(
        ALIF_WASLA, ALIF
    )


def _expand_muqattaat(word: str, clusters: list[Cluster]) -> list[list[Cluster]] | None:
    """Letter-name pseudo-words for a muqatta'at word (e.g. ال\u0653م\u0653 -> alif, lām, mīm)."""
    if any(m in _VOWEL_MARKS for _, marks in clusters for m in marks):
        return None
    skel = _skeleton(word)
    if skel not in MUQATTAAT:
        return None
    names = [[(c, list(m)) for c, m in _LETTER_NAMES[ch]] for ch in skel]
    # Merges between consecutive names: mīm-mīm (idgham shafawi), nūn into ي ن م و (idgham with
    # ghunnah). The first letter becomes bare and the second receives a shadda.
    for a, b in zip(names, names[1:], strict=False):
        last, first = a[-1], b[0]
        if (last[0] == "م" and first[0] == "م") or (last[0] == "ن" and first[0] in IDGHAM_GHUNNAH_LETTERS):
            a[-1] = (last[0], [])
            b[0] = (first[0], [SHADDA, *first[1]])
    return names


@dataclass(slots=True)
class _Word:
    text: str
    ayah: int
    clusters: list[list[Cluster]]  # one entry, or several pseudo-words for muqatta'at
    stop_after: bool = False
    sakt_after: bool = False
    muqattaat: bool = False


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
    unit.orig_vowel = unit.vowel
    if base == ALIF_WASLA:
        unit.wasla = True
    return unit


class TajweedParser:
    """Turns Uthmani text into :class:`ParsedText` with every applicable rule instance.

    ``stop_at_end``: the reciter stops after the last word. ``stops``: extra word indices
    (global, 0-based) after which the reciter stops, e.g. breath pauses detected in the audio;
    ``continue_after`` lists ayah-final words the reciter joins to the next ayah (wasl).
    """

    def __init__(self, *, stop_at_end: bool = True, start_of_recitation: bool = True,
                 tareeq: Tareeq | str = Tareeq.SHATIBIYYAH, include_sifaat: bool = True) -> None:
        self.stop_at_end = stop_at_end
        self.start_of_recitation = start_of_recitation
        self.tareeq = Tareeq(tareeq)
        self.include_sifaat = include_sifaat

    # ------------------------------------------------------------------ public API
    def parse(self, text: str | list[str], *, stops: set[int] | None = None,
              continue_after: set[int] | None = None) -> ParsedText:
        ayahs = [text] if isinstance(text, str) else list(text)
        words = self._prepare_words(ayahs)
        if not words:
            raise TajweedParseError("Target text is empty after normalization")
        for i, w in enumerate(words):
            if stops and i in stops:
                w.stop_after = True
            if continue_after and i in continue_after:
                w.stop_after = False
        words[-1].stop_after = self.stop_at_end

        units: list[LetterUnit] = []
        out_words: list[Word] = []
        rules: list[RuleInstance] = []
        phrase: list[_Word] = []
        first_phrase = True
        for w in words:
            phrase.append(w)
            if w.stop_after or w.sakt_after or w is words[-1]:
                stop = w.stop_after or w.sakt_after
                ph_units, ph_words, ph_rules = self._parse_phrase(
                    phrase, unit_offset=len(units), word_offset=len(out_words), stop=stop,
                    ibtida=not first_phrase or self.start_of_recitation,
                )
                units += ph_units
                out_words += ph_words
                rules += ph_rules
                phrase = []
                first_phrase = False
        parsed = ParsedText(
            text=" ".join(w.text for w in words), units=units, words=out_words, rules=rules,
            stop_at_end=self.stop_at_end, tareeq=self.tareeq,
        )
        parsed.rules += self._sakt_rules(parsed)
        parsed.rules.sort(key=lambda r: (min(r.unit_indices), r.rule_type.value))
        return parsed

    # ------------------------------------------------------------------ word preparation
    def _prepare_words(self, ayahs: list[str]) -> list[_Word]:
        out: list[_Word] = []
        for a_idx, ayah in enumerate(ayahs):
            raw = unicodedata.normalize("NFC", ayah).translate(_NORMALIZE).split()
            ayah_words: list[_Word] = []
            for tok in raw:
                if SAKT_MARK in tok and ayah_words:
                    ayah_words[-1].sakt_after = True
                clean = _DROP.sub("", tok)
                if not clean:
                    continue
                clusters = _tokenize_word(clean)
                if not clusters:
                    raise TajweedParseError(f"Word {clean!r} contains no letters")
                names = _expand_muqattaat(clean, clusters) if not ayah_words else None
                ayah_words.append(_Word(clean, a_idx, names or [clusters], muqattaat=names is not None))
            if not ayah_words:
                raise TajweedParseError(f"Ayah {a_idx + 1} is empty after normalization")
            ayah_words[-1].stop_after = True
            out += ayah_words
        return out

    # ------------------------------------------------------------------ one phrase
    def _parse_phrase(self, words: list[_Word], *, unit_offset: int, word_offset: int, stop: bool,
                      ibtida: bool) -> tuple[list[LetterUnit], list[Word], list[RuleInstance]]:
        units: list[LetterUnit] = []
        pwords: list[Word] = []
        izhar_exceptions: set[int] = set()
        for w in words:
            for k, clusters in enumerate(w.clusters):
                word = Word(index=len(pwords), text=w.text, ayah=w.ayah)
                for base, marks in clusters:
                    has_dagger = DAGGER_ALIF in marks
                    marks = [m for m in marks if m != DAGGER_ALIF]
                    unit = _make_unit(len(units), word.index, base, marks)
                    if w.muqattaat:
                        unit.synthetic = True
                    units.append(unit)
                    word.unit_indices.append(unit.index)
                    if has_dagger:
                        # A dagger alif on ى / bare و silences the bearer and supplies the madd.
                        if base in (ALIF_MAKSURA, WAW) and unit.vowel is None:
                            unit.silent = True
                        madd = LetterUnit(index=len(units), word_index=word.index, char=ALIF, text=DAGGER_ALIF,
                                          madd_letter=True, synthetic=True, maddah_sign=MADDAH in marks)
                        units.append(madd)
                        word.unit_indices.append(madd.index)
                if w.muqattaat and k == len(w.clusters) - 1 and clusters[-1][0] == "ن":
                    # Hafs reads the nūn of يس\u0653 and ن\u0653 with izhar before the following waw.
                    izhar_exceptions.add(word.unit_indices[-1])
                pwords.append(word)
            pwords[-1].stop_after = w.stop_after
            pwords[-1].sakt_after = w.sakt_after

        p = ParsedText(text=" ".join(w.text for w in words), units=units, words=pwords,
                       stop_at_end=stop, tareeq=self.tareeq)
        self._resolve_wasla(p, ibtida)
        self._resolve_madd_letters(p)
        self._resolve_assimilation(p)
        self._resolve_implicit_sukun(p)
        if stop:
            self._apply_waqf(p)
        self._resolve_elision(p)
        rules = self._detect_rules(p, izhar_exceptions, muqattaat_words={
            wi for wi, w in enumerate(pwords) if any(units[i].synthetic and not units[i].madd_letter
                                                    for i in w.unit_indices)})

        # Re-index into the passage-wide numbering.
        for u in units:
            u.index += unit_offset
            u.word_index += word_offset
        for pw in pwords:
            pw.index += word_offset
            pw.unit_indices = [i + unit_offset for i in pw.unit_indices]
        for r in rules:
            r.unit_indices = [i + unit_offset for i in r.unit_indices]
            r.word_index += word_offset
        return units, pwords, rules

    # ------------------------------------------------------------------ orthography resolution
    def _resolve_wasla(self, p: ParsedText, ibtida: bool) -> None:
        for u in p.units:
            if u.char != ALIF_WASLA:
                continue
            if u.index == 0 and ibtida:
                # Hamzat al-wasl is pronounced at the start of a phrase: fatha before the article,
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
                u.orig_vowel = u.vowel
            else:
                u.silent = True

    def _prev_in_word(self, p: ParsedText, u: LetterUnit) -> LetterUnit | None:
        idxs = p.words[u.word_index].unit_indices
        for j in reversed(idxs[: idxs.index(u.index)]):
            if p.units[j].pronounced:
                return p.units[j]
        return None

    def _resolve_madd_letters(self, p: ParsedText) -> None:
        for u in p.units:
            if u.silent or u.synthetic and u.madd_letter or u.char not in MADD_BEARERS:
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

    def _next_unit(self, p: ParsedText, i: int) -> LetterUnit | None:
        for j in range(i + 1, len(p.units)):
            if p.units[j].pronounced:
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
        if last is None:
            return
        if last.madd_letter and last.char in (SMALL_WAW, SMALL_YAA):
            # Silah is dropped at a stop: ل\u064eه\u064f\u06e5 is read "lah".
            last.silent = True
            last.madd_letter = False
            last = self._last_pronounced(p)
            if last is None:
                return
        if last.madd_letter:
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
            if (nxt.sukun or nxt.shadda) and not (u.synthetic and nxt.synthetic):
                u.silent = True
                u.elided = True
                u.madd_letter = False

    # ------------------------------------------------------------------ rule detection
    def _detect_rules(self, p: ParsedText, izhar_exceptions: set[int],
                      muqattaat_words: set[int]) -> list[RuleInstance]:
        rules: list[RuleInstance] = []
        last = self._last_pronounced(p)
        last_idx = last.index if last is not None else -1
        at_stop = p.stop_at_end
        covered_shadda: set[int] = set()
        naqis_units: set[int] = set()

        def add(rule: RuleType, units: list[int], expected: tuple[float, float] | None = None,
                letter: str | None = None, detail: str = "", waqf: bool = False) -> None:
            anchor = p.units[units[0]]
            w = p.words[anchor.word_index]
            rules.append(RuleInstance(
                rule_type=rule, word_index=anchor.word_index, word=w.text, unit_indices=units,
                expected_harakat=expected, letter=letter, detail=detail, at_waqf=waqf, ayah=w.ayah,
            ))

        # Idghaam naqis of ط into ت (أ\u064eح\u064eطت\u064f, ب\u064eس\u064eطت\u064e):
        # the ط keeps its itbaq, no qalqalah.
        for u in p.units:
            nxt = self._next_unit(p, u.index)
            if u.pronounced and u.char == "ط" and u.vowel is None and not u.explicit_sukun and nxt is not None \
                    and nxt.char == "ت" and not nxt.shadda:
                naqis_units.add(u.index)
                add(RuleType.IDGHAM_MUTAJANISAYN, [u.index, nxt.index], letter="ط", detail="naqis")

        for u in p.units:
            nxt = self._next_unit(p, u.index)
            # ---- Mudood ------------------------------------------------------------------
            if u.pronounced and u.madd_letter:
                self._detect_madd(p, u, nxt, last_idx, at_stop, muqattaat_words, add)
            if u.pronounced and u.char in (WAW, YAA) and u.sukun and not u.madd_letter and not u.shadda:
                prev = self._prev_unit(p, u.index)
                if prev is not None and prev.vowel == Vowel.FATHA and nxt is not None \
                        and nxt.word_index == u.word_index:
                    if nxt.explicit_sukun:
                        add(RuleType.MADD_LEEN, [prev.index, u.index], (4, 6), detail="leen lazim (ʿayn)")
                    elif at_stop and nxt.index == last_idx:
                        add(RuleType.MADD_LEEN, [prev.index, u.index], madd_target(RuleType.MADD_LEEN),
                            detail="leen before a stop", waqf=True)

            # ---- Noon sakinah & tanween ------------------------------------------------
            is_noon_sakinah = u.char == "ن" and u.sukun and not u.shadda and u.index not in izhar_exceptions
            if (is_noon_sakinah or (u.tanween and u.pronounced)) and not (u.index == last_idx and at_stop):
                tgt = nxt
                if tgt is not None:
                    cross_word = tgt.word_index != u.word_index
                    detail = "tanween" if u.tanween else "noon sakinah"
                    if tgt.char in THROAT_LETTERS:
                        add(RuleType.IZHAR_HALQI, [u.index, tgt.index], (1, 1), letter="ن", detail=detail)
                    elif tgt.char in IDGHAM_GHUNNAH_LETTERS and cross_word:
                        add(RuleType.IDGHAM_GHUNNAH, [u.index, tgt.index], (2, 2), letter="ن",
                            detail=f"{detail}; naqis" if tgt.char in "وي" else f"{detail}; kamil")
                        covered_shadda.add(tgt.index)
                    elif tgt.char in "لر" and cross_word:
                        add(RuleType.IDGHAM_NO_GHUNNAH, [u.index, tgt.index], (1, 1), letter="ن", detail=detail)
                        covered_shadda.add(tgt.index)
                    elif tgt.char == "ب":
                        add(RuleType.IQLAB, [u.index], (2, 2), letter="ن", detail=detail)
                    elif tgt.char in IKHFA_LETTERS:
                        weight = "heavy" if tgt.char in HEAVY_LETTERS else "light"
                        add(RuleType.IKHFA, [u.index, tgt.index], (2, 2), letter=tgt.char,
                            detail=f"{detail}; {weight}")

            # ---- Meem sakinah -----------------------------------------------------------
            if u.char == "م" and u.sukun and not u.shadda and not (u.index == last_idx and at_stop) \
                    and nxt is not None:
                if nxt.char == "ب":
                    add(RuleType.IKHFA_SHAFAWI, [u.index], (2, 2), letter="م")
                elif nxt.char == "م":
                    add(RuleType.IDGHAM_SHAFAWI, [u.index, nxt.index], (2, 2), letter="م")
                    covered_shadda.add(nxt.index)
                elif u.pronounced:
                    detail = "before و/ف" if nxt.char in "وف" else ""
                    add(RuleType.IZHAR_SHAFAWI, [u.index, nxt.index], (1, 1), letter="م", detail=detail)

            # ---- Idghaam classes of other letters (kamil) --------------------------------
            if u.assimilated and u.char not in "نم" and nxt is not None and not self._is_article_lam(p, u):
                pair = (u.char, nxt.char)
                if u.char == nxt.char:
                    add(RuleType.IDGHAM_MITHLAYN, [u.index, nxt.index], letter=u.char, detail="kamil")
                elif pair in MUTAJANISAYN_PAIRS:
                    add(RuleType.IDGHAM_MUTAJANISAYN, [u.index, nxt.index], letter=u.char, detail="kamil")
                elif pair in MUTAQARIBAYN_PAIRS:
                    add(RuleType.IDGHAM_MUTAQARIBAYN, [u.index, nxt.index], letter=u.char, detail="kamil")

            # ---- Ghunnah mushaddadah -------------------------------------------------------
            if u.pronounced and u.char in "نم" and u.shadda and u.index not in covered_shadda:
                add(RuleType.GHUNNAH, [u.index], (2, 2), letter=u.char, detail="mushaddad")

            # ---- Qalqalah ------------------------------------------------------------------
            if u.pronounced and u.char in QALQALAH_LETTERS and u.sukun and u.index not in naqis_units:
                final = at_stop and u.index == last_idx
                level = ("akbar" if u.shadda else "kubra") if final else "sughra"
                add(RuleType.QALQALAH, [u.index], letter=u.char, detail=level, waqf=final)

            # ---- Tafkheem / Tarqeeq --------------------------------------------------------
            if u.pronounced and not u.madd_letter:
                self._detect_weight(p, u, add)

            # ---- Hamzat al-wasl in continuous reading ------------------------------------------
            if u.wasla and u.silent and u.index > 0:
                prev, nxt_p = self._prev_unit(p, u.index), nxt
                if prev is not None and nxt_p is not None:
                    add(RuleType.HAMZAT_WASL, [prev.index, nxt_p.index], detail="dropped in wasl")

            # ---- Sifaat -----------------------------------------------------------------
            if self.include_sifaat and u.pronounced and not u.madd_letter and not u.synthetic:
                self._detect_sifaat(u, nxt, naqis_units, add)
        return rules

    def _detect_madd(self, p: ParsedText, u: LetterUnit, nxt: LetterUnit | None, last_idx: int, at_stop: bool,
                     muqattaat_words: set[int], add) -> None:  # type: ignore[no-untyped-def]
        carrier = self._prev_unit(p, u.index)
        span = [carrier.index, u.index] if carrier is not None else [u.index]
        is_word_final = nxt is None or nxt.word_index != u.word_index
        tq = self.tareeq
        harfi = "harfi" if u.word_index in muqattaat_words else ""
        if u.char in (SMALL_WAW, SMALL_YAA):
            if nxt is not None and nxt.char in HAMZA_FORMS:
                add(RuleType.MADD_SILAH_KUBRA, span, madd_target(RuleType.MADD_SILAH_KUBRA, tq),
                    detail="pronoun haa before hamza")
            else:
                add(RuleType.MADD_SILAH_SUGHRA, span, madd_target(RuleType.MADD_SILAH_SUGHRA, tq),
                    detail="pronoun haa between two voweled letters")
            return
        if nxt is not None and nxt.char in HAMZA_FORMS and not is_word_final:
            final_hamza = at_stop and nxt.index == last_idx
            add(RuleType.MADD_MUTTASIL, span, (4, 6) if final_hamza else madd_target(RuleType.MADD_MUTTASIL, tq),
                detail="madd before hamza in the same word", waqf=final_hamza)
        elif nxt is not None and nxt.char in HAMZA_FORMS and is_word_final:
            add(RuleType.MADD_MUNFASIL, span, madd_target(RuleType.MADD_MUNFASIL, tq),
                detail="madd at word end before hamza of next word")
        elif self._followed_by_original_sukun(p, u, nxt, is_word_final):
            add(RuleType.MADD_LAZIM, span, (6, 6), detail=f"{harfi or 'kalimi'}: madd before an original sukun")
        elif at_stop and nxt is not None and nxt.index == last_idx:
            add(RuleType.MADD_ARID, span, madd_target(RuleType.MADD_ARID, tq),
                detail="madd before a sukun caused by stopping", waqf=True)
        elif u.index == last_idx and at_stop and u.char in (ALIF, ALIF_MAKSURA) and not u.synthetic \
                and carrier is not None and carrier.vowel == Vowel.FATHA and carrier.orig_vowel == Vowel.FATHA \
                and self._was_tanween_iwad(p, u):
            add(RuleType.MADD_IWAD, span, madd_target(RuleType.MADD_IWAD, tq), detail="tanween fath at a stop")
        elif carrier is not None and carrier.char in HAMZA_FORMS and not harfi:
            add(RuleType.MADD_BADAL, span, madd_target(RuleType.MADD_BADAL, tq), detail="hamza before the madd")
        else:
            add(RuleType.MADD_TABII, span, madd_target(RuleType.MADD_TABII, tq),
                detail="harfi (two-letter name)" if harfi else "natural madd")

    @staticmethod
    def _followed_by_original_sukun(p: ParsedText, u: LetterUnit, nxt: LetterUnit | None, is_word_final: bool) -> bool:
        """Madd letter followed, in its own word, by an original sukun or shaddah (Madd Lazim).

        The following letter may itself be assimilated (the mīm of لَامْ in الٓمٓ merges into مِّيمْ).
        """
        if nxt is not None and not is_word_final and (nxt.shadda or nxt.explicit_sukun):
            return True
        idxs = p.words[u.word_index].unit_indices
        pos = idxs.index(u.index)
        raw_next = p.units[idxs[pos + 1]] if pos + 1 < len(idxs) else None
        return raw_next is not None and raw_next.assimilated

    @staticmethod
    def _was_tanween_iwad(p: ParsedText, u: LetterUnit) -> bool:
        idxs = p.words[u.word_index].unit_indices
        pos = idxs.index(u.index)
        return pos > 0 and not p.units[idxs[pos - 1]].tanween and "\u064b" in p.units[idxs[pos - 1]].text

    def _is_article_lam(self, p: ParsedText, u: LetterUnit) -> bool:
        """The lam of ال before a sun letter (lam shamsiyyah) is a standard, not a class idgham."""
        if u.char != "ل":
            return False
        idxs = p.words[u.word_index].unit_indices
        pos = idxs.index(u.index)
        prev = p.units[idxs[pos - 1]] if pos > 0 else None
        return prev is not None and (prev.wasla or prev.char in (ALIF, "ء"))

    def _detect_weight(self, p: ParsedText, u: LetterUnit, add) -> None:  # type: ignore[no-untyped-def]
        nxt = self._next_unit(p, u.index)
        span = [u.index]
        if nxt is not None and nxt.madd_letter and nxt.word_index == u.word_index:
            span.append(nxt.index)

        if u.char in HEAVY_LETTERS:
            if u.vowel in (Vowel.FATHA, Vowel.DAMMA):
                add(RuleType.TAFKHEEM, span, letter=u.char, detail="isti'la letter")
            return

        if u.char == "ر":
            verdict, why = self._raa_verdict(p, u)
            rule = {"heavy": RuleType.TAFKHEEM, "light": RuleType.TARQEEQ, "either": RuleType.JAWAZ_WAJHAYN}
            if verdict in rule:
                add(rule[verdict], span, letter="ر", detail=f"raa: {why}")
            return

        if u.char == "ل" and u.shadda and nxt is not None and nxt.char == "ه" and nxt.word_index == u.word_index:
            skeleton = "".join(p.units[i].char for i in p.words[u.word_index].unit_indices if not p.units[i].synthetic)
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
            add(RuleType.TAFKHEEM if heavy else RuleType.TARQEEQ, span, letter="ل",
                detail="lam of the Divine Name")

    def _raa_verdict(self, p: ParsedText, u: LetterUnit) -> tuple[str | None, str]:
        """(heavy|light|either|None, reason) for a raa following the rules of Hafs."""
        if u.vowel in (Vowel.FATHA, Vowel.DAMMA):
            return "heavy", "fathah/dammah"
        if u.vowel == Vowel.KASRA:
            return "light", "kasrah"
        prev = self._prev_unit(p, u.index)
        if prev is None:
            return None, ""
        if prev.sukun and not prev.madd_letter:
            if prev.char == YAA:
                return "light", "sakin after yaa leen at a stop"
            before = self._prev_unit(p, prev.index)
            if before is None:
                return None, ""
            if prev.char in HEAVY_LETTERS and before.vowel == Vowel.KASRA and before.word_index == u.word_index:
                return "either", ("isti'la sakin between kasrah and raa at a stop "
                                  "(م\u0650ص\u0652ر / ٱل\u0652ق\u0650ط\u0652ر)")
            prev = before
        if prev.madd_letter:
            if prev.char in (YAA, ALIF_MAKSURA, SMALL_YAA):
                return "light", "after madd yaa"
            return "heavy", "after madd alif/waw"
        if prev.vowel == Vowel.KASRA:
            if prev.wasla or prev.word_index != u.word_index:
                return "heavy", "sakin after a temporary kasrah ('aaridha)"
            follow = self._next_unit(p, u.index)
            if follow is not None and follow.word_index == u.word_index and follow.char in HEAVY_LETTERS:
                if (follow.orig_vowel or follow.vowel) == Vowel.KASRA:
                    return "either", "sakin after kasrah before a kasrah isti'la letter (ف\u0650ر\u0652ق\u064d)"
                return "heavy", "sakin after kasrah but before an isti'la letter"
            return "light", "sakin after an original kasrah"
        return "heavy", "sakin after fathah/dammah"

    def _detect_sifaat(self, u: LetterUnit, nxt: LetterUnit | None, naqis: set[int], add) -> None:  # type: ignore[no-untyped-def]
        c = u.char
        if u.sukun and not u.assimilated and c not in "نم" and c not in QALQALAH_LETTERS and u.index not in naqis \
                and c not in HAMZA_FORMS:
            add(RuleType.HAMS if c in HAMS_LETTERS else RuleType.JAHR, [u.index], letter=c)
            if c in SHIDDAH_LETTERS:
                cls = RuleType.SHIDDAH
            elif c in TAWASSUT_LETTERS:
                cls = RuleType.TAWASSUT
            else:
                cls = RuleType.RAKHAWAH
            ratio = SUKOON_RATIOS[cls]
            add(cls, [u.index], (ratio, ratio), letter=c)
        if c in SAFIR_LETTERS:
            add(RuleType.SAFIR, [u.index], letter=c)
        if c == "ش":
            add(RuleType.TAFASHHI, [u.index], letter=c)
        if c == "ض" and u.sukun:
            add(RuleType.ISTITAALAH, [u.index], letter=c)
        if c == "ر" and (u.shadda or u.sukun):
            add(RuleType.TAKREER, [u.index], letter=c, detail="mushaddad" if u.shadda else "sakin")
        if c in ITBAQ_LETTERS and u.vowel in (Vowel.FATHA, Vowel.DAMMA):
            add(RuleType.ITBAQ, [u.index], letter=c)

    def _sakt_rules(self, parsed: ParsedText) -> list[RuleInstance]:
        out: list[RuleInstance] = []
        for w in parsed.words:
            if not w.sakt_after:
                continue
            here = [i for i in w.unit_indices if parsed.units[i].pronounced]
            after = [u.index for u in parsed.units[max(w.unit_indices) + 1:] if u.pronounced][:1]
            if not here or not after:
                continue
            optional = _skeleton(w.text) == "ماليه"
            out.append(RuleInstance(
                rule_type=RuleType.SAKT, word_index=w.index, word=w.text, unit_indices=[here[-1], after[0]],
                detail="optional (sakt or idgham)" if optional else "mandatory in Hafs", ayah=w.ayah,
            ))
        return out


def parse_text(text: str | list[str], *, stop_at_end: bool = True,
               tareeq: Tareeq | str = Tareeq.SHATIBIYYAH, include_sifaat: bool = True) -> ParsedText:
    return TajweedParser(stop_at_end=stop_at_end, tareeq=tareeq, include_sifaat=include_sifaat).parse(text)
