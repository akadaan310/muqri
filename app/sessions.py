"""Calibration sessions: a master records strategically chosen passages twice -- once to a stated
spec, once with scripted mistakes -- and the engine's measurements are checked against what each
recording SHOULD produce, number by number.

The protocol (app/protocol.py) asks "was the word flagged?". A consumer app needs more: whether the
right rule fired, with the right status, the right sign and size of deviation, on the right letter.
So every exercise states

  goal       what this exercise teaches us about the engine
  spec       how take A is recited: tempo class, the declared wajh, the counts held for each madd
  expect     for take A, per located rule: the status and the range its observed counts must fall in
  mistakes   for take B, one per word: what to do wrong, and the engine signatures that count as
             catching it -- a rule with a status and deviation sign, a characteristic not realised,
             or a letter heard as a named competitor

and every take is scored: take A expectation by expectation, take B mistake by mistake (caught with
the right kind / flagged for another reason / missed), and any flagged word outside the script is a
false alarm. The reciter reads at tadwir; rounds come two or three exercises at a time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

TADWIR = (0.20, 0.30)            # seconds per count: the engine's tadwir band (submission._mastery)
TOL = 0.75                       # counts either side of a spec'd length


@dataclass(frozen=True, slots=True)
class Expect:
    """Take A: a located rule at (ayah, word) passes, and its measured counts sit in a range."""
    ayah: int
    word: int
    rule: str
    counts: tuple[float, float] | None = None
    note: str = ""


@dataclass(frozen=True, slots=True)
class Sig:
    """One engine finding that counts as catching a mistake.

    kind "rule":     rule `name` at the word with a status in `status`; `sign` -1/+1 = the deviation
                     must be negative / positive
    kind "sifah":    characteristic head `name` not realised on a letter of the word (`letter` if set)
    kind "identity": a letter of the word (`letter` if set) not confirmed, heard as one of `heard`
    """
    kind: str
    name: str = ""
    status: tuple[str, ...] = ("short", "long", "wrong")
    sign: int = 0
    letter: str = ""
    heard: tuple[str, ...] = ()
    word_offset: int = 0              # idgham: the merged letter sits in the next word


@dataclass(frozen=True, slots=True)
class Mistake:
    ayah: int
    word: int
    do: str
    catch: tuple[Sig, ...]


@dataclass(frozen=True, slots=True)
class Exercise:
    id: str
    title: str
    surah: int
    ayahs: tuple[int, int]
    goal: str
    spec: str
    wajh: str
    expect: tuple[Expect, ...]
    mistakes: tuple[Mistake, ...]
    learn: str = ""                          # what a miss / false alarm here would tell us
    controls: tuple[tuple[int, int], ...] = field(default_factory=tuple)   # (ayah, word) kept correct in B
    b_is_correct: bool = False               # take B is a second CORRECT reading, at another speed
    b_spec: str = ""
    # (ayah, word) where the reciter STOPS inside an ayah and resumes with the next word. A long ayah
    # must name its stops: the reciter should not choose them, and the engine must know them, since a
    # stop changes what the text requires (قَآئِمًۭا at a stop is madd 'iwad, not ikhfa').
    stops: tuple[tuple[int, int], ...] = ()


def parts(ex: Exercise) -> list[tuple[int, ...]]:
    """The submission's verses for the engine: whole ayahs, or an ayah cut at each declared stop."""
    from quran_transcript import Aya
    out: list[tuple[int, ...]] = []
    for a in range(ex.ayahs[0], ex.ayahs[1] + 1):
        cut = sorted(w for (y, w) in ex.stops if y == a)
        if not cut:
            out.append((ex.surah, a))
            continue
        last = len(Aya(ex.surah, a).get().uthmani.split()) - 1
        lo = 0
        for w in [*cut, last]:
            out.append((ex.surah, a, lo, w))
            lo = w + 1
    return out


def _nasal(rule: str) -> tuple[float, float]:
    """A nasal hold is expected in the engine's calibrated band (app/submission.nasal_band)."""
    from app.submission import nasal_band
    return nasal_band(rule) or (2.0, 4.8)


def _madd(n: float, hi: float | None = None) -> tuple[float, float]:
    """n counts (or n..hi, e.g. tawassut's 4-5), with TOL either side."""
    return (n - TOL, (hi or n) + TOL)


ROUNDS: dict[int, tuple[Exercise, ...]] = {
    1: (
        Exercise(
            "r1e1", "The madd ladder — al-Mutaffifin 83:32", 83, (32, 32),
            goal="Five lengths in six words: tabii 2, munfasil 4 (declared tawassut), muttasil 4, lazim 6, "
                 "plus the shaddah ghunnah of inna. Does the engine measure each length on your count "
                 "unit, and does it catch each one cut short -- including a munfasil it used to excuse "
                 "as qasr?",
            spec="Tadwir, one breath, stop at the end. Hold: وَإِذَا 2 · قَالُوٓا۟ إِنَّ munfasil 4 · the "
                 "ghunnah of إِنَّ 2 · both madds of هَـٰٓؤُلَآءِ 4 · لَضَآلُّونَ the lazim 6.",
            wajh="tawassut",
            expect=(Expect(32, 0, "madd_tabii", _madd(2)),
                    Expect(32, 2, "madd_munfasil", _madd(4, 5), "declared tawassut"),
                    Expect(32, 3, "ghunnah", _nasal("ghunnah")),
                    Expect(32, 4, "madd_muttasil", _madd(4, 5)),
                    Expect(32, 5, "madd_lazim", None, "judged by the stretch calculus at your tempo")),
            mistakes=(Mistake(32, 2, "قَالُوٓا۟ إِنَّ — munfasil at 2 counts (qasr), though you declared 4.",
                              (Sig("rule", "madd_munfasil", ("short",), -1),)),
                      Mistake(32, 3, "إِنَّ — drop the ghunnah: a short plain nūn, no nasal hold.",
                              (Sig("rule", "ghunnah", ("short", "wrong"), -1), Sig("sifah", "ghonna", letter="ن"))),
                      Mistake(32, 4, "هَـٰٓؤُلَآءِ — both madds at 2 counts.",
                              (Sig("rule", "madd_muttasil", ("short",), -1),)),
                      Mistake(32, 5, "لَضَآلُّونَ — the lazim at 2 counts instead of 6.",
                              (Sig("rule", "madd_lazim", ("short",), -1),))),
            controls=((32, 0), (32, 1)),
            learn="A missed munfasil means the declared wajh is not reaching the grade; a missed ghunnah "
                  "means the nasal hold is measured too loosely -- the ghunnah family was the hardest in "
                  "the 41-reciter profiles."),
        Exercise(
            "r1e2", "The ghunnah family — ar-Rahman 55:39", 55, (39, 39),
            goal="Ikhfa twice, iqlab, idgham with ghunnah and the shaddah ghunnah, next to a ṣila kubrā "
                 "and a 6-count lazim. The hardest family in the profiles (idgham with ghunnah passed "
                 "0.69, ghunnah 0.76, ikhfa 0.80 across 41 masters): which of these does the engine "
                 "actually hear, and which can it not?",
            spec="Tadwir, stop at the end. Ghunnah 2 counts on every nasal: عَن ذَنۢبِهِۦٓ (ikhfa), ذَنۢبِهِۦٓ "
                 "(iqlab: meem), إِنسٌ (ikhfa), إِنسٌ وَلَا (idgham with ghunnah), جَآنٌّ (shaddah). "
                 "ذَنۢبِهِۦٓ إِنسٌ ṣila kubrā 4 · جَآنٌّ lazim 6.",
            wajh="tawassut",
            expect=(Expect(39, 0, "idgham_no_ghunnah"),
                    Expect(39, 3, "ikhfa", _nasal("ikhfa")),
                    Expect(39, 4, "iqlab", _nasal("iqlab")),
                    Expect(39, 4, "madd_silah_kubra", _madd(4, 5), "declared tawassut"),
                    Expect(39, 5, "ikhfa", _nasal("ikhfa")),
                    Expect(39, 5, "idgham_ghunnah", _nasal("idgham_ghunnah")),
                    Expect(39, 7, "ghunnah", _nasal("ghunnah")),
                    Expect(39, 7, "madd_lazim", None, "judged by the stretch calculus at your tempo")),
            mistakes=(Mistake(39, 3, "عَن ذَنۢبِهِۦٓ — iẓhār: a clear nūn, tongue on the gum, no hiding, no nasal hold.",
                              (Sig("rule", "ikhfa"), Sig("sifah", "ghonna"), Sig("identity", heard=("ن",)))),
                      Mistake(39, 4, "ذَنۢبِهِۦٓ — no iqlab: a clear nūn before the bā', no meem.",
                              (Sig("rule", "iqlab"), Sig("identity", heard=("ن",)), Sig("sifah", "ghonna"))),
                      Mistake(39, 5, "إِنسٌ وَلَا — iẓhār of the tanwīn: 'insun walā', the nūn clear, no merging.",
                              (Sig("rule", "idgham_ghunnah"), Sig("identity"))),
                      Mistake(39, 7, "جَآنٌّ — keep the 6-count madd, but cut the shaddah ghunnah to a short plain nūn.",
                              (Sig("rule", "ghunnah", ("short", "wrong"), -1), Sig("sifah", "ghonna", letter="ن")))),
            controls=((39, 1), (39, 2), (39, 6)),
            learn="The ikhfa nūn (ں) has no identity test yet -- 'ں read as ن' is tested on masters "
                  "(median margin -44 nats, 6.9 % false alarms) but not graded. A miss on mistake 1 or 2 "
                  "is the measurement that decides whether that test goes in."),
        Exercise(
            "r1e3", "Characteristics and articulation — at-Tariq 86:1–3", 86, (1, 3),
            goal="Qalqalah at three stops, heavy against light (ṭā' / tā', ṣād / sīn), the ra' light "
                 "under kasrah, and thā' against sīn -- across three ayahs, so the tempo and the madd "
                 "lengths must also hold steady from ayah to ayah.",
            spec="Tadwir, stop at the end of each ayah with a clear qalqalah on the qāf and the bā'. "
                 "ٱلسَّمَآءِ muttasil 4 · وَمَآ أَدْرَىٰكَ munfasil 4 · ٱلطَّارِقِ the rā' light (kasrah).",
            wajh="tawassut",
            expect=(Expect(1, 0, "madd_muttasil", _madd(4, 5)),
                    Expect(1, 1, "qalqalah"), Expect(1, 1, "tarqeeq"), Expect(1, 1, "tafkheem"),
                    Expect(2, 0, "madd_munfasil", _madd(4, 5), "declared tawassut"),
                    Expect(2, 3, "qalqalah"), Expect(2, 3, "itbaq"),
                    Expect(3, 1, "qalqalah")),
            mistakes=(Mistake(1, 1, "وَٱلطَّارِقِ — stop dead on the qāf: no echo, no qalqalah.",
                              (Sig("rule", "qalqalah"), Sig("sifah", "qalqla", letter="ق"))),
                      Mistake(1, 0, "وَٱلسَّمَآءِ — make the sīn heavy, like ṣād.",
                              (Sig("identity", letter="س", heard=("ص",)), Sig("sifah", "tafkheem_or_taqeeq", letter="س"),
                               Sig("sifah", "itbaq", letter="س"), Sig("rule", "safir"))),
                      Mistake(2, 3, "ٱلطَّارِقُ — make the ṭā' light, like tā'.",
                              (Sig("identity", letter="ط", heard=("ت",)), Sig("rule", "itbaq"),
                               Sig("sifah", "itbaq", letter="ط"), Sig("sifah", "tafkheem_or_taqeeq", letter="ط"))),
                      Mistake(3, 1, "ٱلثَّاقِبُ — say the thā' as sīn.",
                              (Sig("identity", letter="ث", heard=("س",)),
                               # the whistle of sin heard on the tha' (round 1: safir margin 7.2 -> -0.2)
                               Sig("sifah", "safeer", letter="ث")))),
            controls=((2, 1), (3, 0)),
            learn="Letter substitutions are tested only against each letter's classical confusions; a miss "
                  "on sīn->ṣād or thā'->sīn says the confusion is heard but not decided, and the margin "
                  "shows by how much."),
    ),
    2: (
        Exercise(
            "r2e1", "Articulation points beginners confuse — al-Fatihah 1:5–7", 1, (5, 7),
            goal="The five classic substitutions of non-Arabic learners, one per word: ʿayn as hamza, ṣād as "
                 "sīn, qāf as kāf, ḍād as dāl, ḍād as ẓā'. Al-Fatihah is half of all learner recordings in "
                 "the open datasets. Which substitutions does the articulation test (each letter against its "
                 "nearest-articulation competitor) decide, and by what margin?",
            spec="Tadwir, stop at the end of each ayah. ٱلضَّآلِّينَ lazim 6; the ʿāriḍ at نَسْتَعِينُ and "
                 "ٱلْمُسْتَقِيمَ 4.",
            wajh="tawassut",
            expect=(Expect(7, 8, "madd_lazim", None, "judged by the stretch calculus at your tempo"), Expect(6, 1, "itbaq"), Expect(7, 5, "itbaq"),
                    Expect(7, 8, "itbaq"), Expect(7, 2, "izhar_halqi")),
            mistakes=(Mistake(5, 1, "نَعْبُدُ — say the ʿayn as a hamza: 'na-abudu'.",
                              (Sig("identity", letter="ع", heard=("ء",)),)),
                      Mistake(6, 1, "ٱلصِّرَٰطَ — say the ṣād as sīn: 'as-sirāt'.",
                              (Sig("identity", letter="ص", heard=("س",)), Sig("sifah", "itbaq", letter="ص"),
                               Sig("sifah", "tafkheem_or_taqeeq", letter="ص"), Sig("rule", "itbaq"))),
                      Mistake(6, 2, "ٱلْمُسْتَقِيمَ — say the qāf as kāf: 'al-mustakīm'.",
                              (Sig("identity", letter="ق", heard=("ك",)), Sig("sifah", "tafkheem_or_taqeeq", letter="ق"))),
                      Mistake(7, 5, "ٱلْمَغْضُوبِ — say the ḍād as dāl: 'al-maghdūb'.",
                              (Sig("identity", letter="ض", heard=("د",)), Sig("sifah", "itbaq", letter="ض"),
                               Sig("rule", "itbaq"))),
                      Mistake(7, 8, "ٱلضَّآلِّينَ — keep the 6-count madd, but say the ḍād as ẓā': 'aẓ-ẓāllīn'.",
                              (Sig("identity", letter="ض", heard=("ظ",)), Sig("sifah", "istitala", letter="ض"),
                               Sig("sifah", "shidda_or_rakhawa", letter="ض")))),
            controls=((5, 0), (5, 2), (7, 1), (7, 2), (7, 4)),
            learn="In round 1 a heavy sīn and a light ṭā' barely moved the margins (tafkhim 13.1 -> 13.1, "
                  "9.6 -> 7.4) while thā' -> sīn moved the whistle clearly (7.2 -> -0.2). This round asks the "
                  "same of five other pairs."),
        Exercise(
            "r2e2", "Qalqalah at the stop, the rā', and ikhfā' — al-Falaq 113:1–3", 113, (1, 3),
            goal="Round 1 missed a qalqalah dropped at a stop (its margin barely moved). Is that systematic? "
                 "Two more stops, on qāf and bā'. Plus the rā' made light where it must be heavy, khā' as "
                 "ḥā', and ikhfā' read as a clear nūn in a new context.",
            spec="Tadwir, stop at the end of each ayah with a clear qalqalah. بِرَبِّ the rā' heavy · شَرِّ the "
                 "rā' light (kasrah) · مِن شَرِّ and وَمِن شَرِّ ikhfā' with ghunnah.",
            wajh="tawassut",
            expect=(Expect(1, 2, "tafkheem"), Expect(1, 3, "qalqalah"), Expect(2, 0, "ikhfa", _nasal("ikhfa")),
                    Expect(2, 1, "tarqeeq"), Expect(2, 3, "qalqalah"), Expect(3, 0, "ikhfa", _nasal("ikhfa")),
                    Expect(3, 2, "izhar_halqi"), Expect(3, 4, "qalqalah")),
            mistakes=(Mistake(1, 2, "بِرَبِّ — make the rā' light (it carries fatḥah and must be heavy).",
                              (Sig("rule", "tafkheem"), Sig("sifah", "tafkheem_or_taqeeq", letter="ر"))),
                      Mistake(1, 3, "ٱلْفَلَقِ — stop dead on the qāf: no echo.",
                              (Sig("rule", "qalqalah"), Sig("sifah", "qalqla", letter="ق"), Sig("sifah", "qalqla", letter="ڇ"))),
                      Mistake(2, 0, "مِن شَرِّ — iẓhār: a clear nūn before the shīn, no nasal hold.",
                              (Sig("rule", "ikhfa", ("short", "wrong"), -1), Sig("sifah", "ghonna", letter="ں"))),
                      Mistake(2, 3, "خَلَقَ — say the khā' as ḥā': 'ḥalaq'.",
                              (Sig("identity", letter="خ", heard=("ح", "غ")), Sig("sifah", "tafkheem_or_taqeeq", letter="خ"))),
                      Mistake(3, 4, "وَقَبَ — stop dead on the bā': no echo.",
                              (Sig("rule", "qalqalah"), Sig("sifah", "qalqla", letter="ب"), Sig("sifah", "qalqla", letter="ڇ")))),
            controls=((1, 0), (1, 1), (2, 1), (3, 0), (3, 1), (3, 2)),
            learn="If both stops are missed again, stop-qalqalah needs the waveform echo measure (the anatomy "
                  "research) in grading; the posteriors alone do not decide it."),
        Exercise(
            "r2e3", "The Name, qalqalah ṣughrā and kubrā, idghām, a vowel — al-Ikhlāṣ 112:1–4", 112, (1, 4),
            goal="The surah of the largest labelled learner dataset (1,506 clips, nearly all errors qalqalah "
                 "of the dāl). The lām of the Name heavy after ḍammah, qalqalah on a sākin dāl mid-ayah "
                 "(ṣughrā) and at a stop (kubrā), idghām without ghunnah read with one, and a dropped vowel.",
            spec="Tadwir, stop at the end of each ayah. ٱللَّهُ heavy in both ayahs · يَلِدْ qalqalah ṣughrā "
                 "(no stop) · ٱلصَّمَدُ, يُولَدْ, أَحَدٌۢ qalqalah at the stop · يَكُن لَّهُۥ idghām without ghunnah.",
            wajh="tawassut",
            expect=(Expect(1, 2, "tafkheem"), Expect(1, 3, "qalqalah"), Expect(2, 0, "tafkheem"),
                    Expect(2, 1, "qalqalah"), Expect(3, 1, "qalqalah"), Expect(3, 3, "qalqalah"),
                    Expect(4, 1, "idgham_no_ghunnah"), Expect(4, 2, "madd_silah_sughra", _madd(2)),
                    Expect(4, 4, "qalqalah")),
            mistakes=(Mistake(1, 2, "ٱللَّهُ (ayah 1) — make the lām of the Name light, though it follows a ḍammah.",
                              (Sig("rule", "tafkheem"), Sig("sifah", "tafkheem_or_taqeeq", letter="ل"))),
                      Mistake(2, 1, "ٱلصَّمَدُ — stop on the dāl with no qalqalah.",
                              (Sig("rule", "qalqalah"), Sig("sifah", "qalqla", letter="د"), Sig("sifah", "qalqla", letter="ڇ"))),
                      Mistake(3, 1, "يَلِدْ — no qalqalah on the sākin dāl (keep going into وَلَمْ).",
                              (Sig("rule", "qalqalah"), Sig("sifah", "qalqla", letter="د"), Sig("sifah", "qalqla", letter="ڇ"))),
                      Mistake(4, 1, "يَكُن لَّهُۥ — merge the nūn into the lām WITH a nasal ghunnah (it must be without).",
                              (Sig("rule", "idgham_no_ghunnah"), Sig("sifah", "ghonna", letter="ل", word_offset=1))),
                      Mistake(4, 3, "كُفُوًا — drop the ḍammah on the fā': 'kufwan'.",
                              (Sig("identity"),))),
            controls=((1, 0), (1, 1), (2, 0), (3, 0), (3, 2), (3, 3), (4, 0), (4, 4)),
            learn="Vowel deletion has no identity test (short vowels are tested only against each other), "
                  "so mistake 5 measures whether a dropped ḍammah is visible at all."),
        Exercise(
            "r2e4", "One passage at two speeds — al-ʿAṣr 103:1–3", 103, (1, 3),
            goal="Both takes perfect: take A at your tadwīr, take B clearly faster while keeping every "
                 "rule and characteristic intact. The adaptive timing judges each stretching against your "
                 "own count at each speed; a fast correct reading must score exactly as well as a measured "
                 "one, and every stretching should keep its proportion.",
            spec="Tadwīr, stop at the end of each ayah. إِنَّ ghunnah · ٱلْإِنسَـٰنَ ikhfā' · ءَامَنُوا۟ badal · "
                 "بِٱلصَّبْرِ qalqalah at the stop.",
            wajh="tawassut",
            expect=(Expect(2, 0, "ghunnah", _nasal("ghunnah")), Expect(2, 1, "ikhfa", _nasal("ikhfa")),
                    Expect(3, 2, "madd_badal"), Expect(3, 4, "itbaq"), Expect(3, 8, "qalqalah")),
            mistakes=(),
            b_is_correct=True,
            b_spec="Take B: the same passage, perfect, but clearly FASTER -- as fast as you can while keeping "
                   "every rule and characteristic. It should score the same as take A.",
            learn="The report compares your count unit and every stretching across the two speeds."),
    ),
    3: (
        Exercise(
            "r3e1", "Light letters made heavy — al-Masad 111:1–3", 111, (1, 3),
            goal="Rounds 1–2 caught ṣād→sīn, ḍād→ẓā', qāf→kāf and khā'→ḥā', but missed sīn→ṣād, ṭā'→tā' and "
                 "ḍād→dāl -- the pairs that share a makhraj and differ only in itbāq and tafkhīm. This round "
                 "tests the other direction (tā'→ṭā', dāl→ḍād) and sīn→ṣād again in a new word, with "
                 "ṣād→sīn (caught before) in the same take as the reference point.",
            spec="Tadwīr, stop at the end of each ayah. يَدَآ أَبِى and مَآ أَغْنَىٰ munfasil 4 · لَهَبٍۢ وَتَبَّ "
                 "idghām with ghunnah · نَارًۭا ذَاتَ ikhfā' · qalqalah at the stops on وَتَبَّ, كَسَبَ, لَهَبٍۢ.",
            wajh="tawassut",
            expect=(Expect(1, 1, "madd_munfasil", _madd(4, 5), "declared tawassut"),
                    Expect(1, 3, "idgham_ghunnah", _nasal("idgham_ghunnah")), Expect(1, 4, "qalqalah"),
                    Expect(2, 0, "madd_munfasil", _madd(4, 5), "declared tawassut"),
                    Expect(2, 3, "madd_silah_sughra", _madd(2)), Expect(2, 5, "qalqalah"),
                    Expect(3, 1, "ikhfa", _nasal("ikhfa")), Expect(3, 1, "tafkheem"), Expect(3, 3, "qalqalah")),
            mistakes=(Mistake(1, 0, "تَبَّتْ — say the first tā' as ṭā': 'ṭabbat'.",
                              (Sig("identity", letter="ت", heard=("ط",)), Sig("sifah", "tafkheem_or_taqeeq", letter="ت"),
                               Sig("sifah", "itbaq", letter="ت"))),
                      Mistake(1, 1, "يَدَآ — say the dāl as ḍād: 'yaḍā' (keep the munfasil at 4).",
                              (Sig("identity", letter="د", heard=("ض",)), Sig("sifah", "tafkheem_or_taqeeq", letter="د"),
                               Sig("sifah", "itbaq", letter="د"))),
                      Mistake(2, 5, "كَسَبَ — say the sīn as ṣād: 'kaṣab'.",
                              (Sig("identity", letter="س", heard=("ص",)), Sig("sifah", "tafkheem_or_taqeeq", letter="س"),
                               Sig("sifah", "itbaq", letter="س"))),
                      Mistake(3, 0, "سَيَصْلَىٰ — say the ṣād as sīn: 'sayaslā' (the direction caught before).",
                              (Sig("identity", letter="ص", heard=("س",)), Sig("sifah", "itbaq", letter="ص"),
                               Sig("sifah", "tafkheem_or_taqeeq", letter="ص")))),
            controls=((1, 2), (2, 1), (2, 2), (3, 2)),
            learn="If tā'→ṭā' and dāl→ḍād are caught but sīn→ṣād is missed again, the gap is one letter's "
                  "test, not the itbāq pairs as a class; the margins show how far each moved."),
        Exercise(
            "r3e2", "The nasal family in context — az-Zalzalah 99:7–8", 99, (7, 8),
            goal="Idghām with ghunnah three times, iẓhār ḥalqī and ikhfā' on tanwīn, and heavy rā's. The "
                 "blind-spot graph finds the hidden nūn unreliable before ف ق ك س ت; before shīn it is trusted, "
                 "so this is a clean test of whether an iẓhār of ikhfā' is heard -- and whether a merge "
                 "without ghunnah is.",
            spec="Tadwīr, stop at the end of each ayah (يَرَهْ). فَمَن يَعْمَلْ and وَمَن يَعْمَلْ idghām with "
                 "ghunnah 2 · ذَرَّةٍ خَيْرًۭا iẓhār · خَيْرًۭا يَرَهُۥ and شَرًّۭا يَرَهُۥ idghām with ghunnah · "
                 "ذَرَّةٍۢ شَرًّۭا ikhfā' · every rā' heavy.",
            wajh="tawassut",
            expect=(Expect(7, 0, "idgham_ghunnah", _nasal("idgham_ghunnah")), Expect(7, 3, "izhar_halqi"),
                    Expect(7, 4, "idgham_ghunnah", _nasal("idgham_ghunnah")), Expect(7, 5, "tafkheem"),
                    Expect(8, 0, "idgham_ghunnah", _nasal("idgham_ghunnah")), Expect(8, 3, "ikhfa", _nasal("ikhfa")),
                    Expect(8, 4, "idgham_ghunnah", _nasal("idgham_ghunnah")), Expect(8, 4, "tafkheem")),
            mistakes=(Mistake(7, 0, "فَمَن يَعْمَلْ — merge the nūn into the yā' with NO ghunnah: a plain 'fa-may-yaʿmal'.",
                              (Sig("rule", "idgham_ghunnah", ("short", "wrong"), -1),
                               Sig("sifah", "ghonna", letter="ي", word_offset=1), Sig("sifah", "ghonna"))),
                      Mistake(7, 4, "خَيْرًۭا يَرَهُۥ — iẓhār of the tanwīn: 'khayran yarah', the nūn clear, no merging.",
                              (Sig("rule", "idgham_ghunnah"), Sig("identity"))),
                      Mistake(8, 3, "ذَرَّةٍۢ شَرًّۭا — iẓhār of the tanwīn before shīn: 'dharratin sharran', a clear nūn.",
                              (Sig("rule", "ikhfa"), Sig("sifah", "ghonna", letter="ں"), Sig("identity", heard=("ن",)))),
                      Mistake(8, 4, "شَرًّۭا — make the rā' light (it carries fatḥah and must be heavy).",
                              (Sig("rule", "tafkheem"), Sig("sifah", "tafkheem_or_taqeeq", letter="ر")))),
            controls=((7, 1), (7, 2), (7, 3), (8, 1), (8, 2)),
            learn="A merge without ghunnah and an iẓhār in place of idghām have no letter competitor (the nūn "
                  "is absent from the reference); only the rule's duration and the ghunnah head can see them."),
        Exercise(
            "r3e3", "Vowels: wrong, dropped, stretched — al-Kawthar 108:1–3", 108, (1, 3),
            goal="Round 2's dropped ḍammah (كُفُوًا) went unseen: short vowels are tested only against each "
                 "other, and a missing vowel has no competitor. This round separates four vowel errors -- a "
                 "wrong vowel (has a competitor), a dropped vowel, a short vowel stretched into a madd, and a "
                 "natural madd cut to one count.",
            spec="Tadwīr, stop at the end of each ayah. إِنَّآ ghunnah and munfasil 4 · أَعْطَيْنَـٰكَ and "
                 "شَانِئَكَ madd 2 · فَصَلِّ heavy ṣād · وَٱنْحَرْ and ٱلْأَبْتَرُ heavy rā' at the stop · "
                 "ٱلْأَبْتَرُ qalqalah on the bā'.",
            wajh="tawassut",
            expect=(Expect(1, 0, "ghunnah", _nasal("ghunnah")), Expect(1, 0, "madd_munfasil", _madd(4, 5), "declared tawassut"),
                    Expect(1, 1, "madd_tabii", _madd(2)), Expect(1, 1, "itbaq"), Expect(2, 0, "itbaq"),
                    Expect(2, 1, "tafkheem"), Expect(2, 2, "izhar_halqi"), Expect(2, 2, "tafkheem"),
                    Expect(3, 0, "ghunnah", _nasal("ghunnah")), Expect(3, 1, "madd_tabii", _madd(2)),
                    Expect(3, 3, "qalqalah"), Expect(3, 3, "tafkheem")),
            mistakes=(Mistake(1, 1, "أَعْطَيْنَـٰكَ — cut the madd on the nūn to one count: 'aʿṭaynaka'.",
                              (Sig("rule", "madd_tabii", ("short",), -1), Sig("identity"))),
                      Mistake(2, 0, "فَصَلِّ — end on fatḥah instead of kasrah: 'fa-ṣalla'.",
                              (Sig("identity", heard=("َ",)),)),
                      Mistake(3, 1, "شَانِئَكَ — drop the kasrah of the nūn: 'shān'aka'.",
                              (Sig("identity"),)),
                      Mistake(3, 2, "هُوَ — stretch the ḍammah into a madd: 'hūwa'.",
                              (Sig("identity"),))),
            controls=((1, 2), (2, 1), (2, 2), (3, 0), (3, 3)),
            learn="The wrong vowel should be caught; if the dropped and stretched vowels are missed, the next "
                  "step is to judge each short vowel's length in units of your own count -- the same "
                  "calculus that judges the madds."),
        Exercise(
            "r3e4", "One long ayah — Yūnus 10:12", 10, (12, 12),
            goal="Real learners read long ayahs. The knowledge graph chose this one from the long surahs "
                 "(18–40 words) for coverage: 11 of the 12 rule kinds we target in 27 words, with three ḍāds, "
                 "two sīns and a dāl. Does the engine hold alignment, tempo and every judgement over a "
                 "long ayah, and do the substitutions tested in the short ones behave the same here?",
            spec="Tadwīr, in three breaths with the two stops marked in the text: stop on قَآئِمَا (madd 'iwaḍ, "
                 "two counts) and resume with فَلَمَّا; stop on مَسَّهْ (at the ۚ) and resume with كَذَٰلِكَ; stop "
                 "at the end. ٱلْإِنسَـٰنَ ikhfā' · ٱلضُّرُّ heavy ḍād · لِجَنۢبِهِۦٓ iqlāb and ṣila kubrā 4 · قَآئِمًۭا "
                 "muttaṣil 4 · فَلَمَّا ghunnah · كَأَن لَّمْ idghām without ghunnah · يَدْعُنَآ إِلَىٰ "
                 "munfaṣil 4 and qalqalah on the dāl · ضُرٍّۢ مَّسَّهُۥ idghām with ghunnah · ٱلْمُسْرِفِينَ light rā'.",
            wajh="tawassut",
            expect=(Expect(12, 2, "ikhfa", _nasal("ikhfa")), Expect(12, 3, "tafkheem"), Expect(12, 3, "itbaq"),
                    Expect(12, 5, "iqlab", _nasal("iqlab")),
                    Expect(12, 5, "madd_silah_kubra", _madd(4, 5), "declared tawassut"),
                    Expect(12, 7, "izhar_halqi"), Expect(12, 9, "madd_muttasil", _madd(4, 5)),
                    Expect(12, 9, "madd_iwad", _madd(2), "the stop"), Expect(12, 10, "ghunnah", _nasal("ghunnah")),
                    Expect(12, 15, "idgham_no_ghunnah"),
                    Expect(12, 17, "madd_munfasil", _madd(4, 5), "declared tawassut"), Expect(12, 17, "qalqalah"),
                    Expect(12, 19, "idgham_ghunnah", _nasal("idgham_ghunnah")), Expect(12, 23, "tarqeeq"),
                    Expect(12, 26, "madd_arid_lissukun")),
            mistakes=(Mistake(12, 1, "مَسَّ — say the sīn as ṣād: 'maṣṣa'.",
                              (Sig("identity", letter="س", heard=("ص",)), Sig("sifah", "tafkheem_or_taqeeq", letter="س"),
                               Sig("sifah", "itbaq", letter="س"))),
                      Mistake(12, 4, "دَعَانَا — say the dāl as ḍād: 'ḍaʿānā'.",
                              (Sig("identity", letter="د", heard=("ض",)), Sig("sifah", "tafkheem_or_taqeeq", letter="د"),
                               Sig("sifah", "itbaq", letter="د"))),
                      Mistake(12, 13, "ضُرَّهُۥ — say the ḍād as dāl: 'durrahu'.",
                              (Sig("identity", letter="ض", heard=("د",)), Sig("sifah", "itbaq", letter="ض"),
                               Sig("sifah", "tafkheem_or_taqeeq", letter="ض"), Sig("rule", "itbaq"))),
                      Mistake(12, 19, "ضُرٍّۢ مَّسَّهُۥ — merge the tanwīn into the mīm with NO ghunnah.",
                              (Sig("rule", "idgham_ghunnah", ("short", "wrong"), -1),
                               Sig("sifah", "ghonna", letter="م", word_offset=1))),
                      Mistake(12, 23, "ٱلْمُسْرِفِينَ — make the rā' heavy (it carries kasrah and must be light).",
                              (Sig("rule", "tarqeeq"), Sig("sifah", "tafkheem_or_taqeeq", letter="ر")))),
            controls=((12, 0), (12, 6), (12, 8), (12, 11), (12, 12), (12, 21), (12, 22), (12, 24), (12, 25)),
            stops=((12, 9), (12, 20)),
            learn="The first long ayah in the sessions: a miss here that was caught in a short ayah points at "
                  "alignment or tempo over length, not at the letter's test."),
    ),
    4: (
        Exercise(
            "r4e1", "Idghām with ghunnah, re-measured — al-Mulk 67:22", 67, (22, 22),
            goal="Round 3 found the engine timing the vowel AFTER a merged yā' instead of the held yā', so correct "
                 "idghāms read 'short' and a merge without ghunnah passed. Now it times the held letter "
                 "(Husary: 2.2-3.8 counts) against the ghunnah band. The knowledge graph chose this ayah: two "
                 "idghāms into yā' (أَفَمَن يَمْشِى, أَمَّن يَمْشِى), one into mīm (صِرَٰطٍۢ مُّسْتَقِيمٍۢ) and a "
                 "mushaddad ghunnah, in 12 words. Take B also tests shīn made sīn, a letter not tested yet.",
            spec="Tadwīr, one breath, stop at the end (مُّسْتَقِيمْ). أَفَمَن يَمْشِى, أَمَّن يَمْشِى and "
                 "صِرَٰطٍۢ مُّسْتَقِيمٍۢ idghām with ghunnah · أَمَّن ghunnah on the mīm · وَجْهِهِۦٓ أَهْدَىٰٓ ṣila "
                 "kubrā 4 and qalqalah on the jīm · أَهْدَىٰٓ أَمَّن munfaṣil 4 · مُكِبًّا عَلَىٰ and سَوِيًّا عَلَىٰ iẓhār.",
            wajh="tawassut",
            expect=(Expect(22, 0, "idgham_ghunnah", _nasal("idgham_ghunnah")), Expect(22, 2, "izhar_halqi"),
                    Expect(22, 4, "madd_silah_kubra", _madd(4, 5), "declared tawassut"), Expect(22, 4, "qalqalah"),
                    Expect(22, 5, "madd_munfasil", _madd(4, 5), "declared tawassut"),
                    Expect(22, 6, "ghunnah", _nasal("ghunnah")),
                    Expect(22, 6, "idgham_ghunnah", _nasal("idgham_ghunnah")), Expect(22, 8, "izhar_halqi"),
                    Expect(22, 10, "idgham_ghunnah", _nasal("idgham_ghunnah")), Expect(22, 10, "tafkheem"),
                    Expect(22, 11, "madd_arid_lissukun")),
            mistakes=(Mistake(22, 0, "أَفَمَن يَمْشِى — merge the nūn into the yā' with NO ghunnah: 'afamay-yamshī'.",
                              (Sig("rule", "idgham_ghunnah", ("short", "wrong"), -1),
                               Sig("sifah", "ghonna", letter="ي", word_offset=1), Sig("sifah", "ghonna"))),
                      Mistake(22, 1, "يَمْشِى — say the shīn as sīn: 'yamsī'.",
                              (Sig("identity", letter="ش", heard=("س",)), Sig("sifah", "tafashie", letter="ش"))),
                      Mistake(22, 6, "أَمَّن يَمْشِى — iẓhār: 'amman yamshī', the nūn clear, no merging (keep the "
                                     "ghunnah on the mīm).",
                              (Sig("rule", "idgham_ghunnah"), Sig("identity"))),
                      Mistake(22, 10, "صِرَٰطٍۢ مُّسْتَقِيمٍۢ — merge the tanwīn into the mīm with NO ghunnah.",
                              (Sig("rule", "idgham_ghunnah", ("short", "wrong"), -1),
                               Sig("sifah", "ghonna", letter="م", word_offset=1)))),
            controls=((22, 3), (22, 7), (22, 9)),
            learn="If the merge without ghunnah is caught here and was missed in round 3, the binder fix is what "
                  "changed; if the one into mīm is missed, kāmil and nāqiṣ need separate bands."),
        Exercise(
            "r4e2", "A long ayah with its stops named — Āyat al-Kursī 2:255", 2, (255, 255),
            goal="The first long ayah recited the way a learner recites it: in breaths, stopping where the text "
                 "allows. The stops are declared, so the engine expects the waqf form at each (a madd ʿāriḍ, "
                 "no ṣila) and nothing is left to the reciter's choice. Fifty words, eleven rule kinds, a nūn "
                 "merged with and without ghunnah, and ḥā', ʿayn and ẓā' in take B.",
            spec="Tadwīr, six breaths, stopping ONLY at the five marked stops: ٱلْقَيُّومْ · ٱلْأَرْضْ · خَلْفَهُمْ · "
                 "شَآءْ · حِفْظُهُمَا · and the end, ٱلْعَظِيمْ. Read through نَوْمٌۭ لَّهُۥ (idghām without ghunnah) and "
                 "بِإِذْنِهِۦ يَعْلَمُ (ṣila) without stopping. لَآ إِلَـٰهَ munfaṣil 4 · سِنَةٌۭ وَلَا and بِشَىْءٍۢ مِّنْ "
                 "idghām with ghunnah · مَن ذَا and عِندَهُۥٓ ikhfā' · عِندَهُۥٓ إِلَّا and عِلْمِهِۦٓ إِلَّا ṣila kubrā 4 · "
                 "تَأْخُذُهُۥ, لَّهُۥ, بِإِذْنِهِۦ, يَـُٔودُهُۥ ṣila ṣughrā 2 · مِّنْ عِلْمِهِۦٓ iẓhār.",
            wajh="tawassut",
            expect=(Expect(255, 1, "madd_munfasil", _madd(4, 5), "declared tawassut"),
                    Expect(255, 6, "madd_arid_lissukun", None, "the stop"),
                    Expect(255, 8, "madd_silah_sughra", _madd(2)),
                    Expect(255, 9, "idgham_ghunnah", _nasal("idgham_ghunnah")),
                    Expect(255, 11, "idgham_no_ghunnah"), Expect(255, 12, "madd_silah_sughra", _madd(2)),
                    Expect(255, 19, "ikhfa", _nasal("ikhfa")), Expect(255, 23, "ikhfa", _nasal("ikhfa")),
                    Expect(255, 23, "madd_silah_kubra", _madd(4, 5), "declared tawassut"),
                    Expect(255, 34, "idgham_ghunnah", _nasal("idgham_ghunnah")), Expect(255, 35, "izhar_halqi"),
                    Expect(255, 36, "madd_silah_kubra", _madd(4, 5), "declared tawassut"),
                    Expect(255, 45, "madd_silah_sughra", _madd(2)),
                    Expect(255, 49, "madd_arid_lissukun")),
            mistakes=(Mistake(255, 5, "ٱلْحَىُّ — say the ḥā' as hā': 'al-hayy'.",
                              (Sig("identity", letter="ح", heard=("ه",)), Sig("sifah", "hams_or_jahr", letter="ح"))),
                      Mistake(255, 9, "سِنَةٌۭ وَلَا — merge the tanwīn into the wāw with NO ghunnah: 'sinatuw-wa lā'.",
                              (Sig("rule", "idgham_ghunnah", ("short", "wrong"), -1),
                               Sig("sifah", "ghonna", letter="و", word_offset=1), Sig("sifah", "ghonna"))),
                      Mistake(255, 22, "يَشْفَعُ — say the ʿayn as a hamza: 'yashfaʾu'.",
                              (Sig("identity", letter="ع", heard=("ء",)),)),
                      Mistake(255, 34, "بِشَىْءٍۢ مِّنْ — iẓhār: 'bishayʾin min', the nūn clear, no merging.",
                              (Sig("rule", "idgham_ghunnah"), Sig("identity"),
                               Sig("sifah", "ghonna", letter="م", word_offset=1))),
                      Mistake(255, 49, "ٱلْعَظِيمْ — say the ẓā' as zāy: 'al-ʿazīm'.",
                              (Sig("identity", letter="ظ", heard=("ز", "ذ")), Sig("sifah", "itbaq", letter="ظ"),
                               Sig("sifah", "tafkheem_or_taqeeq", letter="ظ")))),
            controls=((255, 0), (255, 15), (255, 26), (255, 41), (255, 48)),
            stops=((255, 6), (255, 18), (255, 31), (255, 39), (255, 46)),
            learn="Six parts in one recording: a part whose words all fail points at the stop being misplaced "
                  "(alignment), not at the letters."),
    ),
    5: (
        Exercise(
            "r5e1", "The mīm family and two misses retested — al-Fīl 105:1–5", 105, (1, 5),
            goal="No take B has tested mīm sākinah yet: iẓhār shafawī four times, ikhfāʾ shafawī at "
                 "تَرْمِيهِم بِحِجَارَةٍۢ, beside two idghāms with ghunnah. Two substitutions missed in rounds 1–2 "
                 "come back in new words: ḍād as dāl (تَضْلِيلٍۢ, missed in ٱلْمَغْضُوبِ) and ṭāʾ as tāʾ "
                 "(طَيْرًا, missed in ٱلطَّارِقُ).",
            spec="Tadwīr, stop at the end of each ayah. تَرْمِيهِم بِحِجَارَةٍۢ ikhfāʾ shafawī with ghunnah 2 · "
                 "بِحِجَارَةٍۢ مِّن and كَعَصْفٍۢ مَّأْكُولٍۭ idghām with ghunnah · مِّن سِجِّيلٍۢ ikhfāʾ · "
                 "every mīm before a letter other than bāʾ or mīm clear (iẓhār shafawī) · طَيْرًا ṭāʾ and rāʾ heavy.",
            wajh="tawassut",
            expect=(Expect(1, 0, "izhar_shafawi"), Expect(1, 1, "tafkheem"), Expect(1, 4, "tafkheem"),
                    Expect(1, 6, "madd_arid_lissukun"),
                    Expect(2, 0, "izhar_shafawi"), Expect(2, 1, "qalqalah"), Expect(2, 2, "izhar_shafawi"),
                    Expect(2, 4, "madd_arid_lissukun"),
                    Expect(3, 0, "tafkheem"), Expect(3, 1, "izhar_shafawi"), Expect(3, 2, "itbaq"),
                    Expect(3, 2, "izhar_halqi"), Expect(3, 2, "tafkheem"), Expect(3, 3, "madd_arid_lissukun"),
                    # no length expectation on تَرْمِيهِم's ikhfāʾ shafawī: Husary holds it 1.99 counts,
                    # inside the masters' own spread (1st / 5th percentile 1.95 / 2.06) under a floor
                    # of 2.0 that stays, because merges without ghunnah were caught at 1.81-1.92
                    Expect(4, 1, "idgham_ghunnah", _nasal("idgham_ghunnah")), Expect(4, 2, "ikhfa", _nasal("ikhfa")),
                    Expect(4, 3, "madd_arid_lissukun"),
                    Expect(5, 0, "izhar_shafawi"), Expect(5, 1, "idgham_ghunnah", _nasal("idgham_ghunnah")),
                    Expect(5, 2, "madd_arid_lissukun")),
            mistakes=(Mistake(2, 4, "تَضْلِيلٍۢ — say the ḍād as dāl: 'taḍlīl' becomes 'tadlīl'.",
                              (Sig("identity", letter="ض", heard=("د",)), Sig("sifah", "itbaq", letter="ض"),
                               Sig("sifah", "tafkheem_or_taqeeq", letter="ض"))),
                      Mistake(3, 2, "طَيْرًا — make the ṭāʾ light, like tāʾ: 'tayran' (keep the rāʾ heavy).",
                              (Sig("identity", letter="ط", heard=("ت",)), Sig("rule", "itbaq"),
                               Sig("sifah", "itbaq", letter="ط"), Sig("sifah", "tafkheem_or_taqeeq", letter="ط"))),
                      Mistake(4, 0, "تَرْمِيهِم بِحِجَارَةٍۢ — iẓhār shafawī: close the lips on a clear mīm, no hiding, "
                                    "no nasal hold before the bāʾ.",
                              (Sig("rule", "ikhfa_shafawi", ("short", "wrong"), -1),
                               Sig("sifah", "ghonna", letter="م"))),
                      Mistake(5, 1, "كَعَصْفٍۢ مَّأْكُولٍۭ — merge the tanwīn into the mīm with NO ghunnah.",
                              (Sig("rule", "idgham_ghunnah", ("short", "wrong"), -1),
                               Sig("sifah", "ghonna", letter="م", word_offset=1)))),
            controls=((1, 0), (1, 2), (1, 3), (2, 1), (3, 0), (4, 3), (5, 2)),
            learn="If ḍād→dāl and ṭāʾ→tāʾ are missed again in new words, the two pairs that share a makhraj and "
                  "differ only in itbāq are a gap in the identity test itself, not a quirk of one word. Round 5 "
                  "does not retest a short vowel stretched into a madd (ishbāʿ, missed in هُوَ): "
                  "app/itmam.py measures it descriptively and scores nothing, so no signal can catch it yet."),
        Exercise(
            "r5e2", "Idghām without ghunnah against with — al-Balad 90:5–7", 90, (5, 7),
            goal="Round 4 caught every merge made without ghunnah, but at 1.81–1.92 counts against a band "
                 "that starts at 2.0: a thin margin. Here three idghāms WITHOUT ghunnah (أَن لَّن, "
                 "مَالًۭا لُّبَدًا, أَن لَّمْ) sit beside one WITH (لَّن يَقْدِرَ). Round 2 "
                 "missed the reverse error — a ghunnah added where it must not be (يَكُن لَّهُۥ).",
            spec="Tadwīr, stop at the end of each ayah. أَن لَّن, مَالًۭا لُّبَدًا and أَن لَّمْ "
                 "idghām WITHOUT ghunnah, no nasal hold · لَّن يَقْدِرَ idghām with ghunnah 2 · يَرَهُۥٓ "
                 "أَحَدٌ ṣila kubrā 4 · qalqalah on يَقْدِرَ and at the stops on أَحَدٌۭ and أَحَدٌ · "
                 "لُّبَدًا madd ʿiwaḍ 2 at the stop.",
            wajh="tawassut",
            expect=(Expect(5, 1, "idgham_no_ghunnah"), Expect(5, 2, "idgham_ghunnah", _nasal("idgham_ghunnah")),
                    Expect(5, 3, "qalqalah"), Expect(5, 3, "tafkheem"), Expect(5, 5, "qalqalah"),
                    Expect(6, 0, "madd_tabii", _madd(2)), Expect(6, 2, "madd_tabii", _madd(2)),
                    # no madd ʿiwaḍ expectation: at the stop its alif is the clip's last unit, whose
                    # length runs into the silence after it and is not measured (analysis.unreliable_tail)
                    Expect(6, 2, "idgham_no_ghunnah"),
                    Expect(7, 1, "idgham_no_ghunnah"), Expect(7, 2, "izhar_shafawi"),
                    Expect(7, 3, "madd_silah_kubra", _madd(4, 5), "declared tawassut"), Expect(7, 3, "tafkheem"),
                    Expect(7, 4, "qalqalah")),
            mistakes=(Mistake(5, 1, "أَن لَّن — merge the nūn into the lām WITH a nasal ghunnah (it must be without).",
                              (Sig("rule", "idgham_no_ghunnah"), Sig("sifah", "ghonna", letter="ل", word_offset=1))),
                      Mistake(5, 2, "لَّن يَقْدِرَ — merge the nūn into the yāʾ with NO ghunnah.",
                              (Sig("rule", "idgham_ghunnah", ("short", "wrong"), -1),
                               Sig("sifah", "ghonna", letter="ي", word_offset=1), Sig("sifah", "ghonna"))),
                      Mistake(6, 2, "مَالًۭا لُّبَدًا — iẓhār of the tanwīn: 'mālan lubadā', the nūn clear, no merging.",
                              # the nūn read aloud sits at the head of the next word, on the lām
                              (Sig("rule", "idgham_no_ghunnah"), Sig("identity"),
                               Sig("sifah", "ghonna", letter="ل", word_offset=1))),
                      Mistake(7, 3, "يَرَهُۥٓ أَحَدٌ — the ṣila kubrā at 2 counts instead of 4.",
                              (Sig("rule", "madd_silah_kubra", ("short",), -1),))),
            controls=((5, 0), (5, 4), (6, 1), (7, 0)),
            learn="Mistake 1 is round 2's miss in a new word. If it is missed again, a ghunnah added to an "
                  "idghām without ghunnah has no evidence path: the rule has no counted length and the lām "
                  "has no nasal competitor. Mistake 2's deviation, set beside round 4's -0.08 / -0.19, says "
                  "whether the with/without boundary is really where the band puts it."),
        Exercise(
            "r5e3", "Madd too short, the shafawī idghām, qalqalah at the stop — al-Qadr 97:1–5", 97, (1, 5),
            goal="'Madd too short' is still on caution (0.937 against 0.95), so a muttaṣil is cut to two counts. "
                 "Idghām shafawī (رَبِّهِم مِّن) is read without its ghunnah, an iẓhār ḥalqī (مِّنْ "
                 "أَلْفِ) is hidden, and a qalqalah is dropped at a stop on dāl. A third ṭāʾ→tāʾ, sākinah "
                 "this time (مَطْلَعِ), shows whether r5e1's result holds without a vowel on the letter.",
            spec="Tadwīr, stop at the end of each ayah. إِنَّآ ghunnah and munfaṣil 4 · أَنزَلْنَـٰهُ ikhfāʾ · "
                 "وَمَآ munfaṣil 4 · ٱلْمَلَـٰٓئِكَةُ muttaṣil 4 · خَيْرٌۭ مِّنْ idghām with ghunnah · مِّنْ "
                 "أَلْفِ iẓhār · رَبِّهِم مِّن idghām shafawī with ghunnah · مِّن كُلِّ ikhfāʾ · "
                 "qalqalah on the dāl of every ٱلْقَدْرِ and on مَطْلَعِ · ٱلْفَجْرِ heavy rāʾ at the stop.",
            wajh="tawassut",
            # no munfasil expectation on إِنَّآ: Minshawy holds it 5.82 counts, 0.07 past tawassut's
            # 5.75, less than one frame at his pace; وَمَآ keeps the munfasil under test
            expect=(Expect(1, 0, "ghunnah", _nasal("ghunnah")),
                    Expect(1, 1, "ikhfa", _nasal("ikhfa")), Expect(1, 4, "qalqalah"), Expect(1, 4, "tafkheem"),
                    Expect(2, 0, "madd_munfasil", _madd(4, 5), "declared tawassut"), Expect(2, 1, "qalqalah"),
                    Expect(3, 1, "qalqalah"), Expect(3, 1, "tarqeeq"),
                    Expect(3, 2, "idgham_ghunnah", _nasal("idgham_ghunnah")), Expect(3, 3, "izhar_halqi"),
                    Expect(4, 1, "madd_muttasil", _madd(4, 5)),
                    Expect(4, 5, "idgham_shafawi", _nasal("idgham_shafawi")), Expect(4, 6, "ikhfa", _nasal("ikhfa")),
                    Expect(5, 0, "izhar_halqi"), Expect(5, 3, "qalqalah"), Expect(5, 4, "qalqalah"),
                    Expect(5, 4, "tafkheem")),
            mistakes=(Mistake(4, 1, "ٱلْمَلَـٰٓئِكَةُ — the muttaṣil at 2 counts instead of 4.",
                              (Sig("rule", "madd_muttasil", ("short",), -1),)),
                      Mistake(4, 5, "رَبِّهِم مِّن — merge the mīm into the mīm with NO ghunnah.",
                              (Sig("rule", "idgham_shafawi", ("short", "wrong"), -1),
                               Sig("sifah", "ghonna", letter="م", word_offset=1))),
                      Mistake(3, 3, "مِّنْ أَلْفِ — hide the nūn with a ghunnah before the hamza (it must be clear).",
                              (Sig("rule", "izhar_halqi"), Sig("identity", letter="ن", heard=("ں",)))),
                      Mistake(1, 4, "ٱلْقَدْرِ (ayah 1) — stop on the rāʾ with no qalqalah on the dāl.",
                              (Sig("rule", "qalqalah"), Sig("sifah", "qalqla", letter="د"),
                               Sig("sifah", "qalqla", letter="ڇ"))),
                      Mistake(5, 3, "مَطْلَعِ — make the sākin ṭāʾ light, like tāʾ: 'matlaʿ'.",
                              (Sig("identity", letter="ط", heard=("ت",)), Sig("sifah", "itbaq", letter="ط"),
                               Sig("sifah", "tafkheem_or_taqeeq", letter="ط")))),
            controls=((1, 2), (1, 3), (2, 2), (2, 3), (3, 0), (4, 0), (4, 2), (4, 3), (5, 1), (5, 2)),
            learn="A hidden iẓhār (mistake 3) is new: the engine has only tested iẓhār read in place of "
                  "ikhfāʾ or idghām, never the reverse. Sakt was meant for this round and is held back: the "
                  "engine parses the quran_transcript text, which carries no sakt mark (ۜ), so no sakt rule is "
                  "located at 36:52, 75:27 or 83:14 (the Tanzil text has the mark and the parser finds it)."),
    ),
}


@dataclass(frozen=True, slots=True)
class Listen:
    """A question for the expert's ear: one word of a master's recitation, answered yes / no / unsure,
    saved as an expert label (research_agency_lab/experiments/review/labels.jsonl)."""
    id: str
    surah: int
    ayah: int
    word: str
    question: str
    why: str
    reciters: tuple[str, ...]
    comparison: bool = False                 # a known case, to hear the recording's heavy / light first


LISTENERS = ("Husary_128kbps", "Minshawy_Murattal_128kbps", "Abdul_Basit_Murattal_192kbps", "Alafasy_128kbps",
             "Abdurrahmaan_As-Sudais_192kbps")
RA_WHY = ("The reference and the model both call it heavy, 10/10 masters, but at a quarter of the model's "
          "usual confidence -- and the model learned from the same reference, so it is not independent. "
          "The stop makes it a sākin rā' after a kasrah, which the general rule makes light.")

LISTEN: dict[int, tuple[Listen, ...]] = {
    3: (
        Listen("ra-54-2", 54, 2, "مُّسْتَمِرٌّۭ", "At the stop, is the rā' of مُّسْتَمِرّ heavy?", RA_WHY, LISTENERS),
        Listen("ra-54-3", 54, 3, "مُّسْتَقِرٌّۭ", "At the stop, is the rā' of مُّسْتَقِرّ heavy?", RA_WHY, LISTENERS),
        Listen("ra-54-19", 54, 19, "مُّسْتَمِرٍّۢ", "At the stop, is the rā' of مُّسْتَمِرّ heavy?", RA_WHY, LISTENERS),
        Listen("ra-54-38", 54, 38, "مُّسْتَقِرٌّۭ", "At the stop, is the rā' of مُّسْتَقِرّ heavy?", RA_WHY, LISTENERS),
        Listen("ra-89-4", 89, 4, "يَسْرِ", "At the stop, is the rā' of يَسْرِ heavy?",
               "Both readings are allowed (its yā' is dropped); the reference says light and the model "
               "hears heavy in 39 % of 41 masters, at its lowest confidence of any rā'. Which do these masters do?",
               LISTENERS),
        Listen("ra-54-1", 54, 1, "ٱلْقَمَرُ", "At the stop, is the rā' of ٱلْقَمَرُ heavy?",
               "For comparison: a rā' after fatḥah, heavy by every account.", LISTENERS[:2], comparison=True),
        Listen("ra-54-15", 54, 15, "مُّدَّكِرٍۢ", "At the stop, is the rā' of مُّدَّكِرٍ heavy?",
               "For comparison: a rā' after kasrah, light by every account.", LISTENERS[:2], comparison=True),
    ),
}


def listen_json(n: int, answered: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """The round's listening questions, one clip per reciter, with any answer already given."""
    out = []
    for q in LISTEN.get(n, ()):
        clips = []
        for r in q.reciters:
            cid = f"listen:{q.id}:{r}"
            a = answered.get(cid)
            clips.append({"id": cid, "reciter": r.split("_1")[0].split("_4")[0].replace("_", " "),
                          "audio": f"https://everyayah.com/data/{r}/{q.surah:03d}{q.ayah:03d}.mp3",
                          "answer": a["verdict"] if a else None})
        out.append({"id": q.id, "ref": f"{q.surah}:{q.ayah}", "word": q.word, "question": q.question,
                    "why": q.why, "comparison": q.comparison, "clips": clips})
    return out


def exercises() -> dict[str, Exercise]:
    return {e.id: e for r in ROUNDS.values() for e in r}


def _rules_at(m: dict[str, Any], surah: int, ayah: int, word: int, name: str = "") -> list[dict[str, Any]]:
    return [r for r in m["rules"] if r["id"].startswith(f"{surah}:{ayah}:") and r["word"] == word
            and (not name or r["rule"] == name)]


def _letters_at(m: dict[str, Any], surah: int, ayah: int, word: int, symbol: str = "") -> list[dict[str, Any]]:
    return [l for l in m["letters"] if l["id"].startswith(f"{surah}:{ayah}:") and l["word"] == word
            and (not symbol or l["symbol"] == symbol)]


def _matches(sig: Sig, m: dict[str, Any], surah: int, ayah: int, word: int) -> list[str]:
    """The engine's evidence for one signature at one word (empty when it did not fire)."""
    ev = []
    if sig.kind == "rule":
        for r in _rules_at(m, surah, ayah, word, sig.name):
            if r["status"] in sig.status and (not sig.sign or (r["deviation_counts"] or 0) * sig.sign > 0
                                              or r["deviation_counts"] is None):
                ev.append(f"{r['rule']} {r['status']}" + (f" {r['observed_counts']} counts (dev {r['deviation_counts']:+})"
                                                          if r["observed_counts"] is not None else ""))
    elif sig.kind == "sifah":
        for l in _letters_at(m, surah, ayah, word, sig.letter):
            c = l["characteristics"].get(sig.name)
            if c and not c["realised"]:
                ev.append(f"{l['symbol']} {sig.name}: heard {c['observed']} (margin {c['margin']})")
    elif sig.kind == "identity":
        for l in _letters_at(m, surah, ayah, word, sig.letter):
            i = l["identity"]
            if not i["confirmed"] and (not sig.heard or i["competitor"] in sig.heard):
                ev.append(f"{l['symbol']} heard as {i['competitor']} (margin {i['margin']})")
    return ev


def _named(sig: Sig, m: dict[str, Any], surah: int, ayah: int, word: int) -> set[str]:
    """The failing-item ids (as in measurements' words[].failing) a signature speaks for."""
    if sig.kind == "rule":
        return {r["id"] for r in _rules_at(m, surah, ayah, word, sig.name)}
    head = sig.name if sig.kind == "sifah" else "identity"
    return {f"{l['id']}:{head}" for l in _letters_at(m, surah, ayah, word, sig.letter)}


def score(ex: Exercise, take: str, m: dict[str, Any]) -> dict[str, Any]:
    """Take A against its expectations, take B against its script; false alarms for both."""
    s = ex.surah
    spc, tempo = m["recording"]["seconds_per_count"], m["recording"]["tempo_class"]
    out: dict[str, Any] = {"exercise": ex.id, "take": take, "wajh": m["recording"].get("wajh"),
                           "tempo": {"seconds_per_count": spc, "class": tempo,
                                     "ok": spc is not None and TADWIR[0] <= spc <= TADWIR[1]}}
    flagged = {(int(w["ref"].split(":")[1]), int(w["ref"].split(":")[2])): w for w in m["words"] if not w["all_correct"]}
    word_text = {(int(w["ref"].split(":")[1]), int(w["ref"].split(":")[2])): w["text"] for w in m["words"]}
    if take == "A" or ex.b_is_correct:
        rows = []
        for e in ex.expect:
            rs = _rules_at(m, s, e.ayah, e.word, e.rule)
            if not rs:
                rows.append({"ayah": e.ayah, "word": e.word, "text": word_text.get((e.ayah, e.word)), "rule": e.rule,
                             "verdict": "not located", "measured": None})
                continue
            for r in rs:
                measured = r["observed_counts"] is not None
                in_range = e.counts is None or (measured and e.counts[0] <= r["observed_counts"] <= e.counts[1])
                ok = r["status"] == "pass" and in_range
                rows.append({"ayah": e.ayah, "word": e.word, "text": word_text.get((e.ayah, e.word)), "rule": e.rule,
                             "expected_counts": e.counts, "measured": r["observed_counts"], "status": r["status"],
                             "z_masters": r["z_masters"], "verdict": "ok" if ok else (("out of range" if measured else "ok, length not measured")
                                                           if r["status"] == "pass" else r["status"]),
                             "note": e.note})
        out["expectations"] = rows
        out["met"] = sum(r["verdict"] == "ok" for r in rows)
        scripted: set[tuple[int, int]] = set()
    else:
        rows = []
        for mk in ex.mistakes:
            ev = [x for sig in mk.catch for x in _matches(sig, m, s, mk.ayah, mk.word + sig.word_offset)]
            here = flagged.get((mk.ayah, mk.word))
            verdict = "caught" if ev else ("flagged, other reason" if here else "missed")
            rows.append({"ayah": mk.ayah, "word": mk.word, "text": word_text.get((mk.ayah, mk.word)), "do": mk.do,
                         "verdict": verdict, "evidence": ev, "engine_failing": here["failing"] if here else []})
        out["mistakes"] = rows
        out["caught"] = sum(r["verdict"] == "caught" for r in rows)
        scripted = {(mk.ayah, mk.word) for mk in ex.mistakes}
        # a mistake's evidence can sit in the next word (an idghām or iẓhār across the boundary):
        # there, only the items its signatures name are the mistake's; anything else is still a flag
        for mk in ex.mistakes:
            for sig in mk.catch:
                key = (mk.ayah, mk.word + sig.word_offset)
                if sig.word_offset and key in flagged and key not in scripted:
                    named = _named(sig, m, s, *key)
                    rest = [f for f in flagged[key]["failing"] if f not in named]
                    flagged[key] = {**flagged[key], "failing": rest}
        flagged = {k: v for k, v in flagged.items() if v["failing"]}
    from app.letter_matrix import build as letter_matrix
    out["letter_matrix"] = letter_matrix(m)
    out["false_alarms"] = [{"ayah": a, "word": w, "text": v["text"], "failing": v["failing"],
                            "control": (a, w) in ex.controls}
                           for (a, w), v in sorted(flagged.items()) if (a, w) not in scripted]
    return out


def to_json(ex: Exercise) -> dict[str, Any]:
    return {"id": ex.id, "title": ex.title, "surah": ex.surah, "ayahs": list(ex.ayahs), "goal": ex.goal,
            "spec": ex.spec, "wajh": ex.wajh, "learn": ex.learn,
            "expect": [{"ayah": e.ayah, "word": e.word, "rule": e.rule, "counts": e.counts, "note": e.note}
                       for e in ex.expect],
            "mistakes": [{"ayah": mk.ayah, "word": mk.word, "do": mk.do} for mk in ex.mistakes],
            "b_is_correct": ex.b_is_correct, "b_spec": ex.b_spec,
            "stops": [{"ayah": a, "word": w} for a, w in ex.stops]}


def tempo_pair(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Two correct readings of one passage at two speeds: the count unit of each, and every
    stretching's equivalent counts in both. A sound calculus keeps them equal: the masters are the
    standard at any tempo, and a fast correct reading must not measure worse than a slow one."""
    def key(s: dict[str, Any]) -> tuple[Any, ...]:
        return (s["surah"], s["ayah"], s["word"], s["rule"])
    sa = {key(s): s for s in a.get("stretchings", [])}
    rows = []
    for s in b.get("stretchings", []):
        x = sa.get(key(s))
        if x and x.get("equivalent_counts") is not None and s.get("equivalent_counts") is not None:
            rows.append({"ayah": s["ayah"], "word": s["word"], "rule": s["rule"],
                         "seconds": [x["seconds"], s["seconds"]], "equivalent_counts": [x["equivalent_counts"], s["equivalent_counts"]],
                         "verdict": [x["verdict"], s["verdict"]]})
    return {"unit_s": [a.get("unit_s"), b.get("unit_s")],
            "speed_ratio": round(a["unit_s"] / b["unit_s"], 3) if a.get("unit_s") and b.get("unit_s") else None,
            "stretchings": rows}
