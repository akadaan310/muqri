"""Which verse (and which words) is being recited? — from a free phoneme decode, no user selection.

The recitation is decoded freely by the phoneme model (muaalem script). Both the query and the
index are *collapsed* (every run of one symbol folded: ااااا → ا), so madd lengths, ghunnah length
and doubled consonants do not matter for identification.

1. Candidates: character n-gram retrieval (tf-idf weighted hits) over all 6236 ayahs.
2. Verification: semi-global edit alignment of the query against each candidate *window* (the
   ayah and the next ``span`` ayahs concatenated, so a recording may start mid-ayah and run across
   ayah ends); the query must be consumed entirely, the window ends are free.
3. The best window gives surah, ayah and word range of the start and the end, and a match score
   1 − edits / len(query) (learners' mistakes lower it but rarely change the argmax).

The index is built from the persistent store (``datastore``) and saved as a compact JSON bundle
(``app/data/verse_index.json``) that ships with the app.
"""

from __future__ import annotations

import itertools
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

INDEX_PATH = Path(__file__).resolve().parent / "data" / "verse_index.json"
NGRAM = 4


def collapse(ph: str) -> str:
    return "".join(k for k, _ in itertools.groupby(ph))


def _collapse_map(ph: str) -> list[int]:
    """Index in the collapsed string of every character of ``ph``."""
    out, j = [], -1
    prev = None
    for c in ph:
        if c != prev:
            j += 1
        out.append(j)
        prev = c
    return out


@dataclass(slots=True)
class VerseMatch:
    surah: int
    ayah: int                 # ayah where the recitation starts
    word: int                 # 0-based word index where it starts
    end_surah: int
    end_ayah: int
    end_word: int             # 0-based word index where it ends (inclusive)
    score: float              # 1 − edits / len(query)
    edits: int

    def to_dict(self) -> dict[str, object]:
        return {"surah": self.surah, "ayah": self.ayah, "word": self.word, "end_surah": self.end_surah,
                "end_ayah": self.end_ayah, "end_word": self.end_word, "score": round(self.score, 4),
                "edits": self.edits}


class VerseIndex:
    def __init__(self, refs: list[tuple[int, int]], collapsed: list[str], word_starts: list[list[int]]) -> None:
        self.refs = refs
        self.collapsed = collapsed
        self.word_starts = word_starts          # collapsed index where each word begins
        self.pos = {r: i for i, r in enumerate(refs)}
        self.post: dict[str, list[int]] = defaultdict(list)
        for i, s in enumerate(collapsed):
            for g in {s[k:k + NGRAM] for k in range(len(s) - NGRAM + 1)}:
                self.post[g].append(i)
        n = len(refs)
        self.idf = {g: math.log(n / len(v)) for g, v in self.post.items()}

    # -- build / persist ---------------------------------------------------------------------------
    @classmethod
    def from_rows(cls, rows: list[tuple[int, int, str, list[list[int]]]]) -> VerseIndex:
        refs, col, starts = [], [], []
        for s, a, ph, spans in sorted(rows):
            cmap = _collapse_map(ph)
            refs.append((s, a))
            col.append(collapse(ph))
            starts.append([cmap[p0] if 0 <= p0 < len(cmap) else -1 for p0, _ in spans])
        return cls(refs, col, starts)

    @classmethod
    def from_store(cls, moshaf: str = "hafs_m4") -> VerseIndex:
        from datastore.store import connect
        con = connect(read_only=True)
        rows = con.execute("SELECT surah, ayah, phonemes, word_spans FROM ayah_phonetic WHERE moshaf = ?",
                           [moshaf]).fetchall()
        return cls.from_rows([(s, a, ph, spans) for s, a, ph, spans in rows])

    def save(self, path: Path = INDEX_PATH) -> None:
        path.write_text(json.dumps({"ngram": NGRAM, "refs": self.refs, "collapsed": self.collapsed,
                                    "word_starts": self.word_starts}, ensure_ascii=False, separators=(",", ":")))

    @classmethod
    def load(cls, path: Path = INDEX_PATH) -> VerseIndex:
        d = json.loads(path.read_text())
        return cls([tuple(r) for r in d["refs"]], d["collapsed"], d["word_starts"])

    # -- query -------------------------------------------------------------------------------------
    def candidates(self, q: str, k: int = 12) -> list[int]:
        hits: Counter[int] = Counter()
        for g in {q[i:i + NGRAM] for i in range(len(q) - NGRAM + 1)}:
            w = self.idf.get(g)
            if w is None:
                continue
            for i in self.post[g]:
                hits[i] += w
        return [i for i, _ in hits.most_common(k)]

    def _word_at(self, ayah_i: int, cpos: int) -> int:
        ws = self.word_starts[ayah_i]
        w = 0
        for j, st in enumerate(ws):
            if 0 <= st <= cpos:
                w = j
        return w

    def identify(self, phonemes: str, k: int = 12, span: int = 2) -> list[VerseMatch]:
        """Best matches for a free phoneme decode, best first."""
        q = collapse(phonemes)
        if len(q) < NGRAM:
            return []
        out: dict[tuple[int, int, int], VerseMatch] = {}
        for i in self.candidates(q, k):
            # a window may start one ayah early (recording begins at the end of the previous ayah)
            for start in (i - 1, i) if i > 0 else (i,):
                idxs = [j for j in range(start, min(len(self.refs), start + span + 1))
                        if self.refs[j][0] == self.refs[start][0]]
                text = "".join(self.collapsed[j] for j in idxs)
                edits, t0, t1 = semiglobal(q, text)
                offs = np.cumsum([0] + [len(self.collapsed[j]) for j in idxs])
                a0 = int(np.searchsorted(offs, t0, side="right") - 1)
                a1 = int(np.searchsorted(offs, max(t0, t1 - 1), side="right") - 1)
                j0, j1 = idxs[a0], idxs[a1]
                m = VerseMatch(*self.refs[j0], self._word_at(j0, t0 - offs[a0]), *self.refs[j1],
                               self._word_at(j1, max(0, t1 - 1 - offs[a1])), 1 - edits / len(q), edits)
                key = (m.surah, m.ayah, m.word)
                if key not in out or m.edits < out[key].edits:
                    out[key] = m
        return sorted(out.values(), key=lambda m: (m.edits, m.surah, m.ayah))


def _semiglobal_arrays(q: np.ndarray, t: np.ndarray) -> tuple[int, int, int]:
    n, m = q.shape[0], t.shape[0]
    D = np.zeros(m + 1, dtype=np.int32)            # free start in t
    S = np.arange(m + 1, dtype=np.int32)           # start position carried along the best path
    nd = np.empty(m + 1, dtype=np.int32)
    ns = np.empty(m + 1, dtype=np.int32)
    for i in range(1, n + 1):
        nd[0] = i
        ns[0] = 0
        for j in range(1, m + 1):
            v = D[j - 1] + (0 if q[i - 1] == t[j - 1] else 1)
            s = S[j - 1]
            if D[j] + 1 < v:                        # query symbol unmatched
                v = D[j] + 1
                s = S[j]
            if nd[j - 1] + 1 < v:                   # text symbol unmatched
                v = nd[j - 1] + 1
                s = ns[j - 1]
            nd[j] = v
            ns[j] = s
        D, nd = nd, D
        S, ns = ns, S
    end = int(np.argmin(D))
    return int(D[end]), int(S[end]), end


try:  # compiled when numba is available (research box); the pure-Python path is the mobile reference
    from numba import njit

    _semiglobal_arrays = njit(cache=True)(_semiglobal_arrays)
except ImportError:  # pragma: no cover
    pass


def semiglobal(q: str, t: str) -> tuple[int, int, int]:
    """Edit distance of all of ``q`` against its best substring of ``t``; returns (edits, start, end)."""
    return _semiglobal_arrays(np.frombuffer(q.encode("utf-32-le"), dtype=np.uint32),
                              np.frombuffer(t.encode("utf-32-le"), dtype=np.uint32))


_DEFAULT: VerseIndex | None = None


def default_index() -> VerseIndex:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = VerseIndex.load()
    return _DEFAULT
