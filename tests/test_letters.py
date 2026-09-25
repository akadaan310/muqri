"""The declared letter profiles: 17 makharij, 17 sifat, every consonant placed and described."""

from __future__ import annotations

from app.letters import CONSONANTS, MAKHARIJ, NEIGHBOURS, makhraj_of, profile, sifat_of, table


def test_seventeen_points_cover_every_consonant_once() -> None:
    assert [m["n"] for m in MAKHARIJ] == list(range(1, 18))
    for c in CONSONANTS:
        points = [m["n"] for m in MAKHARIJ[1:16] if c in m["letters"]]
        assert len(points) == 1, (c, points)
    assert len(CONSONANTS) == 28


def test_seventeen_sifat_per_letter_and_what_is_measured() -> None:
    for c in CONSONANTS:
        s = sifat_of(c)
        assert len(s) == 12            # five pairs (one side each) + seven singles, present or absent
        assert {x["sifah"] for x in s if not x["measured"]} == {"idhlaq_ismat", "leen", "inhiraf"}


def test_known_profiles() -> None:
    d = {x["sifah"]: x["value"] for x in sifat_of("ض")}
    assert d["hams_jahr"] == "jahr" and d["shiddah_rakhawah"] == "rakhawah" and d["istila_istifal"] == "isti'la"
    assert d["itbaq_infitah"] == "itbaq" and d["istitalah"] is True and d["qalqalah"] is False
    assert makhraj_of("ض")["n"] == 8 and makhraj_of("ن")["n"] == 10 and makhraj_of("ں")["n"] == 10
    r = {x["sifah"]: x["value"] for x in sifat_of("ر")}
    assert r["takrir"] and r["inhiraf"] and r["shiddah_rakhawah"] == "tawassut"
    assert profile("م")["ghunnah"]["value"] and not profile("ب")["ghunnah"]["value"]


def test_every_consonant_has_a_neighbour_to_be_told_apart_from() -> None:
    for c in CONSONANTS:
        assert NEIGHBOURS[c], c
        assert all(q in CONSONANTS for q in NEIGHBOURS[c])
    assert len(table()) == 28
