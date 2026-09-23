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

# Quran-MD (NeurIPS 2025 Muslims-in-ML; Kaggle ``husseinzahaki/quran-md-ayahs-wav-part{1,2,3}``):
# the same EveryAyah recordings as WAV, all 6236 ayahs per reciter, laid out as ``<id>/SSS_AAA.wav``.
QURAN_MD: dict[str, str] = {
    "Husary_128kbps": "husary",
    "Husary_Muallim_128kbps": "hussary.teacher",
    "Minshawy_Murattal_128kbps": "minshawy_murattal",
    "Hudhaify_128kbps": "hudhaify",
    "Abdul_Basit_Murattal_192kbps": "abdul_basit_murattal",
    "Alafasy_128kbps": "alafasy",
    "Yasser_Ad-Dussary_128kbps": "yasser_ad_dussary",
    "Nasser_Alqatami_128kbps": "nasser_alqatami",
    "Saood_ash-Shuraym_128kbps": "saood_ash_shuraym",
    "Abdurrahmaan_As-Sudais_192kbps": "abdurrahmaan_as_sudais",
    "Abdullaah_3awwaad_Al-Juhaynee_128kbps": "abdullaah_3awwaad_al_juhaynee",
}
