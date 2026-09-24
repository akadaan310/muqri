"""Where in the Quran a recording is: exact verses and words, from a transcript.

A consumer app cannot count on the learner typing the verse, and a learner rarely recites exactly one
whole ayah: they start mid-ayah, run on into the next, stop early. `verse_detect` finds the opening
ayah from a Whisper transcript; this finds the whole span -- start ayah and word, end ayah and word --
by aligning the transcript's letters against the Quran's.

The index is every ayah's Uthmani text, word by word, reduced to bare letters (`letters`: no
diacritics, no alif) and collapsed like `verse_id` (a doubled letter counts once). The
search is `verse_id`'s: tf-idf character n-gram retrieval, then a semi-global edit alignment of the
whole transcript against windows of consecutive ayahs. The result's `score` is 1 - edits / length:
low for a recording that is not Quran text at all, which the caller decides on.

    from app.verse_locate import locate
    loc = locate("قل هو الله احد الله الصمد")
    loc.verses   # [(112, 1), (112, 2)]  or (surah, ayah, first_word, last_word) for a partial ayah
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.verse_detect import normalise
from app.verse_id import VerseIndex, VerseMatch

INDEX_PATH = Path(__file__).resolve().parent / "data" / "verse_text_index.json"
SPAN = 6                  # a window covers the candidate ayah and up to SPAN following ones


def letters(text: str) -> str:
    """Bare letters without alif. Uthmani and plain spelling disagree on alif both ways -- the dagger
    alif of عَـٰلَمِينَ is written out (العالمين) but that of ٱلرَّحْمَـٰنِ is not (الرحمن) -- so it
    is left out of both sides; the consonant skeleton carries the identification."""
    return normalise(text).replace("ا", "")


@dataclass(slots=True)
class Location:
    verses: list[tuple[int, ...]]
    score: float
    edits: int
    match: VerseMatch

    def to_dict(self) -> dict[str, object]:
        return {"verses": [list(v) for v in self.verses], "score": round(self.score, 4), "edits": self.edits,
                **{k: v for k, v in self.match.to_dict().items() if k not in ("score", "edits")}}


def build_rows() -> list[tuple[int, int, str, list[list[int]]]]:
    """(surah, ayah, letters, per-word [start, end) spans) for all 6,236 ayahs, from the store."""
    from datastore.store import connect
    con = connect(read_only=True)
    rows = []
    for s, a, u in con.execute("SELECT surah, ayah, uthmani FROM ayah ORDER BY surah, ayah").fetchall():
        text, spans = "", []
        for w in (u or "").split():
            n = letters(w)
            spans.append([len(text), len(text) + len(n)])
            text += n
        rows.append((int(s), int(a), text, spans))
    return rows


def save_index(path: Path = INDEX_PATH) -> None:
    rows = build_rows()
    path.write_text(json.dumps({"rows": rows, "words": [len(sp) for *_x, sp in rows]}, ensure_ascii=False,
                               separators=(",", ":")))


@lru_cache(maxsize=1)
def _load() -> tuple[VerseIndex, dict[tuple[int, int], int]]:
    d = json.loads(INDEX_PATH.read_text())
    rows = [(s, a, t, sp) for s, a, t, sp in d["rows"]]
    return VerseIndex.from_rows(rows), {(s, a): n for (s, a, *_x), n in zip(rows, d["words"])}


def verses_of(m: VerseMatch, n_words: dict[tuple[int, int], int]) -> list[tuple[int, ...]]:
    """The engine's verse list for a matched span: whole ayahs as (surah, ayah), a cut one with its words."""
    out: list[tuple[int, ...]] = []
    ayahs = [(m.surah, a) for a in range(m.ayah, m.end_ayah + 1)] if m.surah == m.end_surah else \
        [(m.surah, m.ayah)]
    for s, a in ayahs:
        n = n_words[(s, a)]
        w0 = m.word if a == m.ayah else 0
        w1 = m.end_word if (s, a) == (m.end_surah, m.end_ayah) else n - 1
        out.append((s, a) if (w0, w1) == (0, n - 1) else (s, a, w0, w1))
    return out


def locate(text: str, top: int = 1) -> Location | None:
    """The best-matching span of the Quran for a transcript (None when nothing matches at all)."""
    idx, n_words = _load()
    ms = idx.identify(letters(text), k=12, span=SPAN)
    if not ms:
        return None
    m = ms[0]
    return Location(verses_of(m, n_words), m.score, m.edits, m)
