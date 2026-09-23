"""Reciters used for benchmarking and indexing (EveryAyah ayah-by-ayah folders).

``studio``: canonical ijaazah-level studio recordings. ``taraweeh``: imams whose public recordings
are mostly taken from Haramain / Taraweeh prayers (live, reverberant, faster, melodic).
"""

from __future__ import annotations

STUDIO: dict[str, str] = {
    "Husary_128kbps": "Mahmoud Khalil Al-Hussary",
    "Husary_Muallim_128kbps": "Al-Hussary (Muallim)",
    "Minshawy_Murattal_128kbps": "Siddiq Al-Minshawi (Murattal)",
    "Hudhaify_128kbps": "Ali Al-Hudhaify",
    "Abdul_Basit_Murattal_192kbps": "Abdul Basit Abdul Samad (Murattal)",
    "Alafasy_128kbps": "Mishary Alafasy",
    "Mohammad_al_Tablaway_128kbps": "Mohammad Al-Tablawi",
    "Muhammad_Ayyoub_128kbps": "Muhammad Ayyoub",
}
TARAWEEH: dict[str, str] = {
    "Yasser_Ad-Dussary_128kbps": "Yasser Al-Dosari",
    "Nasser_Alqatami_128kbps": "Nasser Al-Qatami",
    "Saood_ash-Shuraym_128kbps": "Saud Al-Shuraim",
    "Abdurrahmaan_As-Sudais_192kbps": "Abdul Rahman Al-Sudais",
    "Abdullaah_3awwaad_Al-Juhaynee_128kbps": "Abdullah Al-Juhany",
    "Salah_Al_Budair_128kbps": "Salah Al-Budair",
    "Abdullah_Matroud_128kbps": "Abdullah Al-Matroud",
    "MaherAlMuaiqly128kbps": "Maher Al-Muaiqly",
}
ROSTER: dict[str, tuple[str, str]] = {
    **{k: (v, "studio") for k, v in STUDIO.items()},
    **{k: (v, "taraweeh") for k, v in TARAWEEH.items()},
}
EVERYAYAH_URL = "https://everyayah.com/data/{folder}/{surah:03d}{ayah:03d}.mp3"
