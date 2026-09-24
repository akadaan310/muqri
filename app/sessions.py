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
                    Expect(32, 5, "madd_lazim", (5.0, 7.75), "Husary holds it ~7 counts")),
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
                    Expect(39, 5, "idgham_ghunnah", _madd(2)),
                    Expect(39, 7, "ghunnah", _nasal("ghunnah")),
                    Expect(39, 7, "madd_lazim", (5.0, 7.75))),
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
            expect=(Expect(7, 8, "madd_lazim", (5.0, 7.75)), Expect(6, 1, "itbaq"), Expect(7, 5, "itbaq"),
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
    ),
}


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


def score(ex: Exercise, take: str, m: dict[str, Any]) -> dict[str, Any]:
    """Take A against its expectations, take B against its script; false alarms for both."""
    s = ex.surah
    spc, tempo = m["recording"]["seconds_per_count"], m["recording"]["tempo_class"]
    out: dict[str, Any] = {"exercise": ex.id, "take": take, "wajh": m["recording"].get("wajh"),
                           "tempo": {"seconds_per_count": spc, "class": tempo,
                                     "ok": spc is not None and TADWIR[0] <= spc <= TADWIR[1]}}
    flagged = {(int(w["ref"].split(":")[1]), int(w["ref"].split(":")[2])): w for w in m["words"] if not w["all_correct"]}
    word_text = {(int(w["ref"].split(":")[1]), int(w["ref"].split(":")[2])): w["text"] for w in m["words"]}
    if take == "A":
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
            "mistakes": [{"ayah": mk.ayah, "word": mk.word, "do": mk.do} for mk in ex.mistakes]}
