"""Identify which verses a recording contains, by transcribing it and matching the text.

The engine scores against a known reference, so something has to supply the reference when the user
does not type it. `app/verse_id.py` attempts that from the muaalem phoneme decode; this module takes
the more direct route the user pointed at — a Whisper model trained on Quranic recitation produces
*Arabic text*, and Arabic text can be matched against the 6,236 ayahs in the store.

Two things make it cheap enough to be worth doing:

* **whisper-tiny-ar-quran, unchunked.** Measured on a 14.1 s clip: the default pipeline config
  (`chunk_length_s=30`, beam search) ran at RTF 5.34; tiny with greedy decoding and no chunking runs
  at **RTF 1.00 and returns the identical, perfect transcription** —
  `ذَلِكَ الْكِتَابُ لَا رَيْبَ فِيهِ هُدًى لِلْمُتَّقِينَ`, exactly 2:2.
* **Only the head of the recording is transcribed.** Locating the *starting* verse is enough: once
  that is known the reference walks forward through the known text. So detection costs a fixed ~20 s
  regardless of whether the submission is one verse or a whole juz'.

Matching strips diacritics and normalises the letters that vary in orthography (alif forms, ta
marbuta, alif maqsura), then scores candidate ayahs by shared character 4-grams — the same retrieval
idea as `verse_id`, on letters rather than phonemes, which is what Whisper gives us.
"""

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEAD_SECONDS = 25.0          # enough text to locate the opening verse
NGRAM = 4
MODEL = "tarteel-ai/whisper-tiny-ar-quran"

_DIACRITICS = re.compile(r"[ً-ٰٟۖ-ۭـ]")
_NONLETTER = re.compile(r"[^ء-ي]")


def normalise(text: str) -> str:
    """Strip diacritics and fold the orthographic variants Whisper and the Uthmani script disagree on."""
    t = unicodedata.normalize("NFKD", text)
    t = _DIACRITICS.sub("", t)
    t = (t.replace("آ", "ا").replace("أ", "ا").replace("إ", "ا")
         .replace("ٱ", "ا").replace("ى", "ي").replace("ة", "ه"))
    return _NONLETTER.sub("", t)


@dataclass(slots=True)
class Detection:
    surah: int
    ayah: int
    score: float                 # shared 4-grams / query 4-grams, 0..1
    text: str                    # what Whisper heard
    n_candidates: int

    def to_dict(self) -> dict:
        return {"surah": self.surah, "ayah": self.ayah, "score": round(self.score, 4),
                "transcript": self.text, "candidates": self.n_candidates}


@lru_cache(maxsize=1)
def _index() -> tuple[list[tuple[int, int]], list[set[str]]]:
    """Every ayah as a set of normalised character 4-grams, loaded once."""
    from datastore.store import connect
    con = connect(read_only=True)
    rows = con.execute("SELECT surah, ayah, uthmani FROM ayah ORDER BY surah, ayah").fetchall()
    refs, grams = [], []
    for s, a, u in rows:
        n = normalise(u or "")
        refs.append((int(s), int(a)))
        grams.append({n[i:i + NGRAM] for i in range(len(n) - NGRAM + 1)})
    return refs, grams


@lru_cache(maxsize=1)
def _asr():  # type: ignore[no-untyped-def]
    import warnings
    warnings.filterwarnings("ignore")
    from transformers import pipeline
    # no chunk_length_s: for clips under 30 s it only adds the experimental seq2seq chunking path,
    # which measured 5x slower for an identical transcription
    return pipeline("automatic-speech-recognition", model=MODEL, device=-1)


def transcribe(wave, seconds: float = HEAD_SECONDS) -> str:  # type: ignore[no-untyped-def]
    """Whisper text for the first `seconds` of audio (16 kHz mono float32)."""
    import numpy as np
    head = np.asarray(wave, dtype="float32")[: int(seconds * 16000)]
    out = _asr()(head.copy(), generate_kwargs={"num_beams": 1})
    return str(out.get("text", "")).strip()


def match(text: str, top: int = 5) -> list[Detection]:
    """Best-matching ayahs for a transcript, best first."""
    q = normalise(text)
    if len(q) < NGRAM:
        return []
    qg = {q[i:i + NGRAM] for i in range(len(q) - NGRAM + 1)}
    refs, grams = _index()
    scored = []
    for (s, a), g in zip(refs, grams):
        if not g:
            continue
        hit = len(qg & g)
        if hit:
            scored.append((hit / len(qg), s, a))
    scored.sort(reverse=True)
    return [Detection(surah=s, ayah=a, score=sc, text=text, n_candidates=len(scored))
            for sc, s, a in scored[:top]]


def detect(wave, seconds: float = HEAD_SECONDS) -> list[Detection]:  # type: ignore[no-untyped-def]
    """Transcribe the head of a recording and return the best-matching starting verses."""
    return match(transcribe(wave, seconds))


def span_for(wave, start: Detection, max_ayahs: int = 60) -> list[tuple[int, int]]:  # type: ignore[no-untyped-def]
    """Verses a recording plausibly covers, from its start verse and its duration.

    The reference walks forward through the known text, so only the opening needs recognising. The
    count is bounded by audio length against the corpus mean of ~7.5 s per ayah, then clamped.
    """
    import numpy as np
    from datastore.store import connect
    seconds = len(np.asarray(wave)) / 16000
    n = max(1, min(max_ayahs, int(round(seconds / 7.5)) + 1))
    con = connect(read_only=True)
    rows = con.execute(
        "SELECT surah, ayah FROM ayah WHERE (surah = ? AND ayah >= ?) OR surah > ? "
        "ORDER BY surah, ayah LIMIT ?", [start.surah, start.ayah, start.surah, n]).fetchall()
    return [(int(s), int(a)) for s, a in rows]
