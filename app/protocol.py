"""The recording protocol: paired correct / scripted-mistake takes, scored against the script.

The engine's false-alarm rate is measured on 41 professional reciters. Its MISS rate cannot be, because
no public dataset has deliberate tajweed errors located at known words (see
research_agency_lab/experiments/deep_research/08_mistake_datasets.md). This builds that dataset with a
certified reciter: each test is one passage recited twice --

    take A  "correct"   as perfectly as possible
    take B  "mistakes"  the same passage, perfect except for 3-5 scripted mistakes at named words

The script is the label. Every mistake names its word by exact Uthmani text and the kinds of engine
finding that would count as catching it, so each take B is scored the moment it is uploaded: caught
at the word, caught one word away ("near"), or missed -- and every fault in a word the script left
alone is a false alarm. Take A of the same passage is the control on the same voice, room and mic.

Mistakes are spread across different words and different mechanisms (duration, nasalisation,
articulation point, characteristic, waqf) so each one is attributable, and together the ten tests
cover every family the engine judges. Test 10 is different on purpose: both takes are CORRECT, the
second at hadr speed, because a fast recitation must keep every characteristic -- the masters are
the golden rule at any tempo -- and a fast correct reading must not score lower than a slow one.

These recordings are also the seed for synthesis: each (A, B) pair is a real example of one voice
producing a named error, which is what a perturbation engine has to reproduce on other voices.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

# what counts as catching each kind of scripted mistake: rule names, sifah heads, or "letter"
# (the letter's identity was not confirmed against its classical confusions)
MADD = ["madd_tabii", "madd_arid_lissukun", "madd_lazim", "madd_munfasil", "madd_muttasil",
        "madd_leen", "madd_badal", "madd_silah_sughra"]
NASAL = ["ghonna", "ghunnah", "idgham_ghunnah", "idgham_no_ghunnah", "ikhfa", "iqlab",
         "izhar_halqi", "izhar_shafawi", "idgham_shafawi"]
HEAVY = ["tafkheem_or_taqeeq", "tafkheem", "tarqeeq"]


@dataclass(frozen=True, slots=True)
class Mistake:
    ayah: int
    word: str                      # exact Uthmani word, resolved to an index at startup
    do: str                        # what the reciter is asked to do wrong
    rule: str                      # the rule or characteristic being broken
    catch: tuple[str, ...]         # engine findings that count as catching it
    occurrence: int = 1            # which occurrence of `word` in the ayah


@dataclass(frozen=True, slots=True)
class Test:
    id: str
    title: str
    surah: int
    ayahs: tuple[int, int]
    focus: str
    correct_note: str
    mistakes: tuple[Mistake, ...] = field(default_factory=tuple)
    b_is_correct: bool = False     # test 10: take B is a correct reading at another tempo
    b_note: str = ""


TESTS: tuple[Test, ...] = (
    Test("t01", "Al-Fātiḥah 1–4 · heaviness and the throat letters", 1, (1, 4),
         "tafkhīm/tarqīq of the lām of Allah and of rā', ḥā' and ʿayn, madd ʿāriḍ at the ayah ends",
         "Recite 1:1–4 at your normal murattal pace, stopping at the end of every ayah.",
         (Mistake(1, "ٱللَّهِ", "Make the lām of Allah HEAVY (it follows a kasrah, so it must be light).",
                  "tarqīq of the lām of the Name after kasrah", tuple(HEAVY)),
          Mistake(1, "ٱلرَّحْمَـٰنِ", "Say the ḥā' as a plain hā' (ه) — 'arRahmān' with an English h.",
                  "makhraj of ḥā' (mid-throat)", ("letter", "hams_or_jahr")),
          Mistake(2, "رَبِّ", "Make the rā' LIGHT (tarqīq) — it carries fatḥah and must be heavy.",
                  "tafkhīm of rā' with fatḥah", ("takreer", *HEAVY)),
          Mistake(2, "ٱلْعَـٰلَمِينَ", "Say the ʿayn as a hamza — 'al-ālamīn'.",
                  "makhraj of ʿayn", ("letter",)),
          Mistake(4, "ٱلدِّينِ", "At the stop, do NOT lengthen the yā' at all — clip it to one count.",
                  "madd ʿāriḍ li-s-sukūn (2/4/6)", tuple(MADD)))),
    Test("t02", "Al-Fātiḥah 6–7 · iṭbāq, iẓhār and the madd lāzim", 1, (6, 7),
         "ṣād/ḍād against their light twins, iẓhār of nūn sākinah, the six-count madd lāzim",
         "Recite 1:6–7, stopping at the end of each ayah.",
         (Mistake(6, "ٱلصِّرَٰطَ", "Say the ṣād as a plain sīn — 'as-sirāt', no heaviness.",
                  "iṭbāq and isti'lā' of ṣād", ("letter", "itbaq", "safeer", *HEAVY)),
          Mistake(7, "أَنْعَمْتَ", "Nasalise the nūn (as if ikhfā') — it must be read with clear iẓhār before ʿayn.",
                  "iẓhār ḥalqī", tuple(NASAL)),
          Mistake(7, "ٱلْمَغْضُوبِ", "Say the ḍād as a heavy dāl — 'al-maghdūb'.",
                  "makhraj and istiṭālah of ḍād", ("letter", "istitala", "shidda_or_rakhawa")),
          Mistake(7, "ٱلضَّآلِّينَ", "Cut the madd lāzim to about two counts.",
                  "madd lāzim kalimī muthaqqal (6)", tuple(MADD)))),
    Test("t03", "Al-Ikhlāṣ · qalqalah and ṣafīr", 112, (1, 4),
         "qalqalah on a stopped dāl and a sākin dāl, ṣafīr and iṭbāq of ṣād, idghām without ghunnah",
         "Recite the whole sūrah, stopping at the end of each ayah.",
         (Mistake(1, "أَحَدٌ", "At the stop, end the dāl dead — no qalqalah bounce.",
                  "qalqalah kubrā at waqf", ("qalqla", "qalqalah")),
          Mistake(2, "ٱلصَّمَدُ", "Say the ṣād as sīn — 'as-samad'.",
                  "iṭbāq of ṣād", ("letter", "itbaq", "safeer", *HEAVY)),
          Mistake(3, "يَلِدْ", "No qalqalah on the sākin dāl — stop it flat.",
                  "qalqalah ṣughrā", ("qalqla", "qalqalah")),
          Mistake(4, "يَكُن", "Add a ghunnah on the nūn as it merges into the lām of لَّهُۥ.",
                  "idghām bi-lā ghunnah (nūn into lām)", tuple(NASAL)),
          Mistake(4, "كُفُوًا", "Nasalise the tanwīn into the hamza of أَحَدٌ — it must be clear iẓhār.",
                  "iẓhār ḥalqī of tanwīn", tuple(NASAL)))),
    Test("t04", "Al-Kawthar · ghunnah, iṭbāq, tafashshī", 108, (1, 3),
         "ghunnah of a doubled nūn, ṭā' against tā', thā' against sīn, shīn's spread, qalqalah",
         "Recite the whole sūrah, stopping at the end of each ayah.",
         (Mistake(1, "إِنَّآ", "No ghunnah — say the doubled nūn as a quick plain 'n'.",
                  "ghunnah of nūn mushaddadah (2 counts)", tuple(NASAL)),
          Mistake(1, "أَعْطَيْنَـٰكَ", "Say the ṭā' as a light tā' — 'a'taynāk'.",
                  "iṭbāq of ṭā'", ("letter", "itbaq", *HEAVY)),
          Mistake(1, "ٱلْكَوْثَرَ", "Say the thā' as sīn — 'al-kawsar'.",
                  "makhraj of thā' (tongue tip on the teeth)", ("letter", "safeer")),
          Mistake(3, "شَانِئَكَ", "Say the shīn as sīn — 'sāni'aka', no spread.",
                  "tafashshī of shīn", ("letter", "tafashie", "safeer")),
          Mistake(3, "ٱلْأَبْتَرُ", "No qalqalah on the sākin bā'.",
                  "qalqalah ṣughrā", ("qalqla", "qalqalah")))),
    Test("t05", "Al-Falaq · ikhfā' and the letter pairs", 113, (1, 5),
         "ikhfā' of nūn sākinah, qāf/kāf, ghayn/khā', ḥā'/hā', qalqalah",
         "Recite the whole sūrah, stopping at the end of each ayah.",
         (Mistake(1, "ٱلْفَلَقِ", "Say the final qāf as a kāf — 'al-falak'.",
                  "makhraj and isti'lā' of qāf", ("letter", "qalqla", *HEAVY)),
          Mistake(2, "مِن", "Pronounce the nūn clearly (iẓhār) before shīn — it must be hidden (ikhfā').",
                  "ikhfā' ḥaqīqī", tuple(NASAL)),
          Mistake(3, "غَاسِقٍ", "Say the ghayn as khā' (voiceless) — 'khāsiq'.",
                  "jahr of ghayn", ("letter", "hams_or_jahr")),
          Mistake(3, "وَقَبَ", "At the stop, no qalqalah on the bā'.",
                  "qalqalah kubrā at waqf", ("qalqla", "qalqalah")),
          Mistake(5, "حَاسِدٍ", "Say the ḥā' as hā' — 'hāsid'.",
                  "makhraj of ḥā'", ("letter", "hams_or_jahr")))),
    Test("t06", "Al-Masad · assimilation across words and the munfaṣil", 111, (1, 5),
         "idghām with ghunnah, ikhfā' of tanwīn, madd munfaṣil, ghunnah of mīm mushaddadah",
         "Recite the whole sūrah, stopping at the end of each ayah; hold the munfaṣil 4 counts.",
         (Mistake(1, "يَدَآ", "Cut the madd munfaṣil (يَدَآ أَبِى) to two counts.",
                  "madd munfaṣil (4–5 in Ḥafṣ by al-Shāṭibiyyah)", tuple(MADD)),
          Mistake(3, "نَارًۭا", "Read the tanwīn with a clear 'n' before the dhāl — it must be ikhfā'.",
                  "ikhfā' of tanwīn", tuple(NASAL)),
          Mistake(4, "حَمَّالَةَ", "No ghunnah on the doubled mīm.",
                  "ghunnah of mīm mushaddadah", tuple(NASAL)),
          Mistake(5, "حَبْلٌۭ", "Pronounce the tanwīn clearly before مِّن — it must merge with ghunnah.",
                  "idghām bi-ghunnah", tuple(NASAL)))),
    Test("t07", "Al-Humazah 4–6 · iqlāb, the Name after ḍammah, tā' marbūṭah at the stop", 104, (4, 6),
         "iqlāb, ṭā' against tā', tafkhīm of Allah after ḍammah, the munfaṣil, waqf on tā' marbūṭah",
         "Recite 104:4–6, stopping at the end of each ayah (the tā' marbūṭah becomes a hā' at the stop).",
         (Mistake(4, "لَيُنۢبَذَنَّ", "Say a plain 'n' before the bā' — no iqlāb to mīm, no ghunnah.",
                  "iqlāb", ("letter", *NASAL)),
          Mistake(4, "ٱلْحُطَمَةِ", "Say the ṭā' as a light tā'.",
                  "iṭbāq of ṭā'", ("letter", "itbaq", *HEAVY)),
          Mistake(5, "وَمَآ", "Cut the madd munfaṣil (وَمَآ أَدْرَىٰكَ) to two counts.",
                  "madd munfaṣil", tuple(MADD)),
          Mistake(6, "ٱللَّهِ", "Make the lām of Allah LIGHT — after the ḍammah of نَارُ it must be heavy.",
                  "tafkhīm of the Name after ḍammah", tuple(HEAVY)),
          Mistake(6, "ٱلْمُوقَدَةُ", "At the final stop keep the ending '-tu' instead of stopping on a hā'.",
                  "waqf on tā' marbūṭah (becomes hā')", ("letter", "waqf", *MADD)))),
    Test("t08", "Al-Baqarah 2 · stopping and restarting", 2, (2, 2),
         "waqf on فِيهِ (madd ʿāriḍ), ibtidā' with hamzat al-waṣl, a mid-word stop, rā' with fatḥah",
         "Recite 2:2 once: stop after فِيهِ holding the madd 4 counts, restart at هُدًى, stop at the end.",
         (Mistake(2, "ذَٰلِكَ", "Stop after ذَٰلِكَ, then restart at ٱلْكِتَـٰبُ WITHOUT the opening hamza ('lkitāb').",
                  "hamzat al-waṣl at ibtidā'", ("letter", "hamzat_wasl", "waqf")),
          Mistake(2, "رَيْبَ", "Make the rā' light.",
                  "tafkhīm of rā' with fatḥah", ("takreer", *HEAVY)),
          Mistake(2, "فِيهِ", "Stop on فِيهِ but pronounce the final kasrah ('fīhi') before the pause.",
                  "waqf with sukūn", ("waqf", "letter", *MADD)),
          Mistake(2, "لِّلْمُتَّقِينَ", "Pause briefly INSIDE this word (lil-mut … taqīn), then finish it.",
                  "no stop inside a word", ("waqf",)))),
    Test("t09", "An-Nās · ṣafīr, ghunnah, the heavy letters", 114, (1, 6),
         "ghunnah of نّ, sīn against thā', khā' against ḥā', ṣād against sīn",
         "Recite the whole sūrah, stopping at the end of each ayah.",
         (Mistake(1, "ٱلنَّاسِ", "No ghunnah on the doubled nūn (first ayah only).",
                  "ghunnah of nūn mushaddadah", tuple(NASAL)),
          Mistake(4, "ٱلْوَسْوَاسِ", "Lisp the sīn into thā' — 'al-wathwāth'.",
                  "ṣafīr of sīn", ("letter", "safeer")),
          Mistake(4, "ٱلْخَنَّاسِ", "Say the khā' as ḥā' — 'al-ḥannās'.",
                  "isti'lā' of khā'", ("letter", *HEAVY)),
          Mistake(5, "صُدُورِ", "Say the ṣād as sīn — 'sudūr'.",
                  "iṭbāq of ṣād", ("letter", "itbaq", "safeer", *HEAVY)),
          Mistake(6, "ٱلْجِنَّةِ", "No ghunnah on the doubled nūn.",
                  "ghunnah of nūn mushaddadah", tuple(NASAL)))),
    Test("t10", "Al-ʿAṣr · the same perfection at two speeds", 103, (1, 3),
         "every characteristic must survive speed: the masters are the golden rule at any tempo",
         "Take A: recite the sūrah at a calm murattal pace, perfectly.",
         (), b_is_correct=True,
         b_note="Take B: recite it again PERFECTLY but at ḥadr — as fast as you can while keeping every "
                "rule and every characteristic intact. It should score the same as take A."),
)


def _norm(w: str) -> str:
    """Canonical form for matching: Unicode NFC (shadda/fatha order) without the small Quranic
    annotation marks and tatweel, which vary between text sources but are not the word."""
    return re.sub("[\u06D6-\u06ED\u0640]", "", unicodedata.normalize("NFC", w))


def resolve(words_of) -> dict[str, list[dict[str, Any]]]:  # type: ignore[no-untyped-def]
    """Attach the word index to every scripted mistake; raises if any word is not in its ayah.

    `words_of(surah, ayah)` returns that ayah's Uthmani words.
    """
    out: dict[str, list[dict[str, Any]]] = {}
    for t in TESTS:
        rows = []
        for m in t.mistakes:
            ws = words_of(t.surah, m.ayah)
            hits = [i for i, w in enumerate(ws) if _norm(w) == _norm(m.word)]
            if len(hits) < m.occurrence:
                raise ValueError(f"{t.id}: {m.word!r} not found in {t.surah}:{m.ayah} ({ws})")
            rows.append({"ayah": m.ayah, "word": ws[hits[m.occurrence - 1]], "index": hits[m.occurrence - 1],
                         "do": m.do, "rule": m.rule, "catch": list(m.catch)})
        out[t.id] = rows
    return out


def _fault_keys(f: dict[str, Any]) -> set[str]:
    return {k for k in (f.get("rule"), f.get("sifah"), f.get("type") if f.get("type") == "letter" else None)
            if k} | ({"letter"} if f.get("type") == "letter" else set())


def scorecard(test: Test, take: str, report: dict[str, Any],
              resolved: list[dict[str, Any]]) -> dict[str, Any]:
    """Score one take against the script: hits, near-hits, misses and false alarms."""
    faults: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for a in report.get("ayahs", []):
        for w in a.get("words", []):
            if w["faults"]:
                faults[(a["ayah"], w["index"])] = w["faults"]
    scripted = resolved if (take == "mistakes" and not test.b_is_correct) else []
    rows = []
    # a scripted word's own faults belong to it; a neighbouring fault can credit one mistake only
    claimed = {(m["ayah"], m["index"]) for m in scripted}
    for m in scripted:
        key = (m["ayah"], m["index"])
        here = faults.get(key, [])
        near = [] if here else [k for k in ((m["ayah"], m["index"] - 1), (m["ayah"], m["index"] + 1))
                                if k in faults and k not in claimed][:1]
        typed = any(_fault_keys(f) & set(m["catch"]) for f in here)
        verdict = "caught" if here else ("near" if near else "missed")
        claimed.update(near)
        rows.append({**m, "verdict": verdict, "right_kind": typed, "engine_found": here})
    false_alarms = [{"ayah": k[0], "word_index": k[1], "faults": v}
                    for k, v in sorted(faults.items()) if k not in claimed]
    n = len(scripted)
    return {"test": test.id, "take": take,
            "scripted": n, "caught": sum(r["verdict"] == "caught" for r in rows),
            "near": sum(r["verdict"] == "near" for r in rows),
            "missed": sum(r["verdict"] == "missed" for r in rows),
            "right_kind": sum(r["right_kind"] for r in rows),
            "false_alarm_words": len(false_alarms), "mistakes": rows, "false_alarms": false_alarms,
            "word_accuracy": (report.get("summary") or {}).get("accuracy")}
