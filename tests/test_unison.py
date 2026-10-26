"""Focused UNISON/0.1 acceptance tests; stdlib only, no model/audio/network.

Run: python -m unittest tests.test_unison -v
These are local mathematical/software fixtures, NOT historical-theory tests.
"""
from __future__ import annotations

import unittest

from unison.registry import QittRegistry, AddressCell, RegistryError
from unison.text import parse_text, basic_rasm_projection
from unison.quran import classify_initials, MUQATTAAT, INITIAL_LETTERS, quran_scene_source
from unison.dictionary import DictionaryEntry, MuajamIndex, terminal_meem_letter_names
from unison.address_ops import CollectionStore, CollectionError
from unison.speech import LanguagePack, plan_speech, QawlEvent, qawl_transition
from unison.surface_scene import quran_surface_scene, oscillator_samples, LogicalClock, HyperRuler
from unison.host import UnisonHost


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.r = QittRegistry()

    def test_registry_cardinality(self):
        self.assertEqual(len(self.r.alphabet), 28)
        self.assertEqual(self.r.native["ط"], 9)
        self.assertEqual(self.r.native["ت"], 400)
        self.assertEqual(self.r.native["غ"], 1000)

    def test_positional_round_trips(self):
        nums = [0, 1, 9, 10, 90, 100, 109, 190, 999, 1000, 1009,
                1999, 2000, 9000, 10000, 90000, 100000, 900000,
                1000000, 100000000, 1000000000]
        for n in nums:
            with self.subTest(n=n):
                self.assertEqual(self.r.decode(self.r.encode(n)), n)
        self.assertEqual(self.r.encode(2000)[0].glyph, "ب")
        self.assertEqual(self.r.encode(1000000)[0].glyph, "غ")
        self.assertEqual(self.r.encode(10000)[0].glyph, "ي")

    def test_glyph_collision_requires_place(self):
        self.assertEqual(self.r.bare_projection(self.r.encode(2)),
                         self.r.bare_projection(self.r.encode(2000)))
        self.assertEqual(self.r.bare_projection(self.r.encode(1000)),
                         self.r.bare_projection(self.r.encode(1000000)))
        with self.assertRaises(RegistryError):
            self.r.decode((AddressCell(3, 2, "LETTER", "ا"),))

    def test_typed_sifr_guard(self):
        self.assertEqual(self.r.encode(0)[0].kind, "SIFR_ZERO")
        self.assertEqual(self.r.encode(109)[1].kind, "SIFR_GAP")
        with self.assertRaises(RegistryError):
            self.r.decode((AddressCell(1, 1, "LETTER", "ي"),
                           AddressCell(0, 0, "SIFR_OPERATOR", None)))

    def test_shadda_root_word_values(self):
        self.assertEqual(self.r.word_value("قط"), 109)
        self.assertEqual(self.r.word_value("قِطّ"), 109)
        self.assertEqual(self.r.word_value("قِطّ", profile="shadda_expanded_opt_in"), 118)
        self.assertEqual(self.r.word_value("قصص"), 280)
        self.assertEqual(self.r.word_value("قصقص"), 380)
        self.assertEqual(self.r.word_value("الرقيم"), 381)
        with self.assertRaises(RegistryError):
            self.r.word_value("ألف")


class QuranSyntaxTests(unittest.TestCase):
    def test_source_is_immutable_spanned_and_lossy_rasm_labeled(self):
        s = "قَالَ قَائِلٌ، قِطّ"
        a = parse_text(s, language="ar")
        self.assertEqual("".join(x.raw for x in a.units), s)
        for x in a.units:
            self.assertEqual(s.encode()[x.start_byte:x.end_byte].decode(), x.raw)
        r = basic_rasm_projection(s)
        self.assertTrue(r["lossy"])
        self.assertEqual(r["kind"], "simplified_noncanonical_letter_skeleton")

    def test_muqattaat_are_contextual_not_generic_lexemes(self):
        self.assertEqual(len(MUQATTAAT), 14)
        self.assertEqual(len(INITIAL_LETTERS), 14)
        frame = classify_initials("الم", role="verse_initial_candidate", reference="2:1")
        self.assertTrue(frame.recognized_sequence)
        self.assertEqual(frame.qitt_native_value, 71)
        self.assertEqual(len(frame.letter_nodes), 3)
        self.assertFalse(classify_initials("الم", role="ordinary_word").recognized_sequence)
        self.assertFalse(classify_initials("الرقيم", role="verse_initial_candidate").recognized_sequence)
        self.assertEqual(classify_initials("حم", role="verse_initial_candidate").qitt_native_value, 48)

    def test_quran_verse_source_no_basmala_mutation(self):
        r = quran_scene_source(27, 30, "إِنَّهُ بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ",
                               edition="owner-supplied-unverified")
        self.assertIn("بِسْمِ", r["raw"])
        self.assertEqual(r["basmala"][:6], "source")


class DictionaryTests(unittest.TestCase):
    def test_indexed_dictionary_and_ambiguous_numeric_values(self):
        d = MuajamIndex()
        entries = [
            DictionaryEntry("a", "ar", "عجم", "root", "owner hypothesis", "owner-context", root="عجم"),
            DictionaryEntry("b", "ar", "معجم", "noun", "dictionary", "owner-context", root="عجم"),
            DictionaryEntry("c", "ar", "قط", "noun", "word", "owner-context", root="قط", sense_id="demo"),
            DictionaryEntry("d", "en", "cat", "noun", "a domestic feline", "example-data", sense_id="demo"),
        ]
        for row in entries:
            d.insert(row)
        self.assertEqual(d.lookup("ar", "معجم").rows[0].entry_id, "b")
        self.assertEqual({e.entry_id for e in d.by_root("ar", "عجم").rows}, {"a", "b"})
        self.assertEqual(d.by_native_value(153).rows[0].entry_id, "b")
        self.assertEqual(len(d.polyglot("demo").rows), 2)
        self.assertEqual(len(d.interval(100, 200).rows), 3)
        self.assertIn("لام", terminal_meem_letter_names())
        self.assertFalse(d.heat_map("ar")[0]["frequency_complete"])


class AddressTests(unittest.TestCase):
    def test_reversible_4d_axis_sort_and_scope(self):
        s = CollectionStore(max_cells=32)
        t = [[[[4, 1], [2, 3]], [[6, 5], [8, 7]]]]
        a = s.register(t, scope="research")
        self.assertEqual(a.shape, (1, 2, 2, 2))
        ordered = s.sorted_copy(a.handle, axis=3, scope="research")
        self.assertEqual(s.read(ordered.handle, scope="research"),
                         [[[[1, 4], [2, 3]], [[5, 6], [7, 8]]]])
        self.assertEqual(s.read(a.handle, scope="research"), t)
        self.assertEqual(ordered.parent, a.handle)
        self.assertFalse(s.sort_options(a.handle, axis=2, scope="research")[0]["admissible"])
        with self.assertRaises(PermissionError):
            s.read(a.handle, scope="other")
        with self.assertRaises(CollectionError):
            s.sorted_copy(a.handle, axis=8, scope="research")


class SpeechAndSurfaceTests(unittest.TestCase):
    def test_unknown_language_speech_no_invented_sound(self):
        pack = LanguagePack("tr", "tr-TR", "modern", "curated-demo", {"merhaba": ("m", "e", "r")})
        plan = plan_speech("merhaba dostum!", pack)
        self.assertFalse(plan.complete)
        self.assertFalse(plan.executable_audio)
        self.assertIsNone(plan.units[2].phonemes)

    def test_qawl_switch_guard(self):
        a = QawlEvent("speaker-a", "قال قائل", "direct", "SPEAK", "owner-text")
        self.assertEqual(qawl_transition(None, a)["after"], "speaker-a")
        b = QawlEvent("speaker-a", "كم لبثتم", "direct", "CONTINUE")
        self.assertEqual(qawl_transition(a, b)["operation"], "CONTINUE")
        with self.assertRaises(ValueError):
            qawl_transition(a, QawlEvent("speaker-b", "test", "direct", "CONTINUE"))

    def test_surface_scene_provenance(self):
        scene = quran_surface_scene(surah=2, ayah=1, raw_text="الم", edition="declared-hafs",
                                    role="verse_initial_candidate", clock=LogicalClock(1000, 10, 7))
        self.assertEqual(scene["clock"]["logical_time_ns"], 1070)
        self.assertEqual(scene["initials_layer"]["qitt_native_value"], 71)
        self.assertEqual(scene["pronunciation_layer"]["state"], "DEFERRED")
        self.assertTrue(scene["render_hints"]["arabic_rtl"])
        with self.assertRaises(ValueError):
            quran_surface_scene(surah=2, ayah=1, raw_text="الم", edition="declared-hafs",
                                astronomical_objects=[{"catalog_id": "missing-source"}])

    def test_dimensions_and_mechanical_oscillator(self):
        ruler = HyperRuler(("x", "time"), ("m", "s"), (0.0, 1.5))
        self.assertEqual(len(ruler.axes), 2)
        wave = oscillator_samples(amplitude_m=2, frequency_hz=1,
                                  phase_rad=0, sample_rate_hz=4, count=4)
        self.assertAlmostEqual(wave[0], 2, places=10)
        self.assertAlmostEqual(wave[1], 0, places=10)
        with self.assertRaises(ValueError):
            HyperRuler(("x", "x"), ("m", "m"), (0.0, 1.0))

    def test_legacy_default_and_explicit_shadow(self):
        h = UnisonHost()
        h.register("address", legacy=lambda x: x + 1, unison=lambda x: x + 1, pure=True)
        h.register("recitation", legacy=lambda x: x, unison=lambda x: x * 2)
        self.assertEqual(h.invoke("address", 5).result, 6)
        h.set_mode("shadow")
        self.assertEqual(h.invoke("address", 5).shadow, 6)
        with self.assertRaises(PermissionError):
            h.invoke("recitation", 5)
        h.set_mode("unison")
        self.assertEqual(h.invoke("address", 5).result, 6)
        h.rollback()
        self.assertEqual(h.mode, "legacy")


if __name__ == "__main__":
    unittest.main()
