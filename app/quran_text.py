"""Uthmani Quran text provider.

A handful of short surahs are bundled so the engine works offline; any other ayah is fetched
from the public alquran.cloud API (``quran-uthmani`` edition, Tanzil source) and cached on disk.
"""

from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
import urllib.error
import urllib.request
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

_BUNDLED = Path(__file__).parent / "data" / "quran_uthmani_sample.json"
_API = "https://api.alquran.cloud/v1/ayah/{surah}:{ayah}/quran-uthmani"
_BASMALA_SKELETON = ("بسم", "الله", "الرحمن", "الرحيم")
AYAH_COUNTS: tuple[int, ...] = (
    7, 286, 200, 176, 120, 165, 206, 75, 129, 109, 123, 111, 43, 52, 99, 128, 111, 110, 98, 135,
    112, 78, 118, 64, 77, 227, 93, 88, 69, 60, 34, 30, 73, 54, 45, 83, 182, 88, 75, 85, 54, 53,
    89, 59, 37, 35, 38, 29, 18, 45, 60, 49, 62, 55, 78, 96, 29, 22, 24, 13, 14, 11, 11, 18, 12,
    12, 30, 52, 52, 44, 28, 28, 20, 56, 40, 31, 50, 40, 46, 42, 29, 19, 36, 25, 22, 17, 19, 26,
    30, 20, 15, 21, 11, 8, 8, 19, 5, 8, 8, 11, 11, 8, 3, 9, 5, 4, 7, 3, 6, 3, 5, 4, 5, 6,
)


class QuranTextError(RuntimeError):
    pass


def validate_reference(surah: int, ayah: int) -> None:
    if not 1 <= surah <= 114:
        raise QuranTextError(f"Surah must be between 1 and 114, got {surah}")
    if not 1 <= ayah <= AYAH_COUNTS[surah - 1]:
        raise QuranTextError(
            f"Surah {surah} has {AYAH_COUNTS[surah - 1]} ayahs; ayah {ayah} is out of range"
        )


@lru_cache(maxsize=1)
def _bundled() -> dict[str, dict[str, object]]:
    with _BUNDLED.open(encoding="utf-8") as fh:
        data: dict[str, dict[str, object]] = json.load(fh)
    return data


def _cache_dir() -> Path:
    root = Path(os.environ.get("QAARI_CACHE_DIR", Path.home() / ".cache" / "qaari-eval"))
    path = root / "text"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _skeleton(word: str) -> str:
    """Bare consonantal skeleton of a word (marks removed, alif forms unified)."""
    word = "".join(c for c in unicodedata.normalize("NFD", word) if unicodedata.category(c) != "Mn")
    return re.sub("[\u0671\u0622\u0623\u0625]", "\u0627", word)


def strip_basmala(text: str, surah: int, ayah: int) -> str:
    """Remove a basmala prefixed onto ayah 1 (every surah except Al-Fatiha / At-Tawbah)."""
    text = text.replace("\ufeff", "").strip()
    if ayah != 1 or surah in (1, 9):
        return text
    words = text.split()
    if len(words) > 4 and tuple(_skeleton(w) for w in words[:4]) == _BASMALA_SKELETON:
        return " ".join(words[4:])
    return text


_FULL_API = "https://api.alquran.cloud/v1/quran/quran-uthmani"


def get_full_quran(*, allow_network: bool = True, timeout: float = 60.0) -> dict[tuple[int, int], str]:
    """All 6236 ayahs as {(surah, ayah): text}, downloaded once and cached on disk."""
    cache_file = _cache_dir() / "quran-uthmani.json"
    if cache_file.exists():
        raw = json.loads(cache_file.read_text(encoding="utf-8"))
    else:
        if not allow_network:
            raise QuranTextError("Full Qur'an text is not cached and network access is disabled")
        try:
            with urllib.request.urlopen(_FULL_API, timeout=timeout) as resp:  # noqa: S310 - fixed https URL
                payload = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise QuranTextError(f"Could not download the Qur'an text: {exc}") from exc
        raw = {
            f"{s['number']}:{a['numberInSurah']}": strip_basmala(a["text"], s["number"], a["numberInSurah"])
            for s in payload["data"]["surahs"] for a in s["ayahs"]
        }
        cache_file.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    out = {}
    for key, text in raw.items():
        s_, a_ = key.split(":")
        out[(int(s_), int(a_))] = text
    if len(out) != sum(AYAH_COUNTS):
        raise QuranTextError(f"Expected {sum(AYAH_COUNTS)} ayahs, got {len(out)}")
    return out


def get_ayah_text(surah: int, ayah: int, *, allow_network: bool = True, timeout: float = 15.0) -> str:
    """Return the Uthmani text of ``surah:ayah``."""
    validate_reference(surah, ayah)
    bundled = _bundled().get(str(surah))
    if bundled is not None:
        ayahs = bundled["ayahs"]
        assert isinstance(ayahs, dict)
        if str(ayah) in ayahs:
            return strip_basmala(str(ayahs[str(ayah)]), surah, ayah)

    cache_file = _cache_dir() / f"{surah:03d}{ayah:03d}.txt"
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")

    if not allow_network:
        raise QuranTextError(
            f"{surah}:{ayah} is not bundled and network access is disabled; pass --text instead"
        )
    url = _API.format(surah=surah, ayah=ayah)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - fixed https URL
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise QuranTextError(f"Could not fetch {surah}:{ayah} from {url}: {exc}") from exc
    if payload.get("code") != 200 or "data" not in payload:
        raise QuranTextError(f"Unexpected response for {surah}:{ayah}: {payload.get('status')}")
    text = strip_basmala(str(payload["data"]["text"]), surah, ayah)
    try:
        cache_file.write_text(text, encoding="utf-8")
    except OSError as exc:  # cache is best-effort
        logger.debug("Could not cache ayah text: %s", exc)
    return text
