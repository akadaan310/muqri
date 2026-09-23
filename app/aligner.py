"""Forced alignment of recitation audio to the letter units of the target text.

Three back-ends are provided:

* :class:`CTCForcedAligner` — wav2vec2 CTC emissions (default model
  ``TBOGamer22/wav2vec2-quran-phonetics``, a Quran-specific phonetic model) + exact CTC Viterbi
  forced alignment. Requires ``torch`` and ``transformers``.
* :class:`JsonAlignmentLoader` — timings produced by any external aligner (MFA, Praat
  TextGrid conversions, manual annotation) supplied as JSON.
* :class:`HeuristicAligner` — a dependency-free fallback that distributes units uniformly over
  the voiced regions and snaps boundaries to spectral change points. Its timings are
  approximate and are flagged as such in the report.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt

from app.audio import AudioSignal, frame_rms_db
from app.models import AlignedUnit, Alignment, Vowel
from app.tajweed_rules import HAMZA_FORMS, ParsedText

logger = logging.getLogger(__name__)

DEFAULT_CTC_MODEL = os.environ.get("QAARI_ALIGNER_MODEL", "TBOGamer22/wav2vec2-quran-phonetics")
MAX_CTC_SECONDS = 120.0


class AlignmentError(RuntimeError):
    pass


class Aligner(Protocol):
    name: str

    def align(self, audio: AudioSignal, parsed: ParsedText) -> Alignment: ...


# --------------------------------------------------------------------------- phonetic mapping
_PHONETIC: dict[str, str] = {
    "ب": "b", "ت": "t", "ث": "th", "ج": "j", "ح": "ḥ", "خ": "kh", "د": "d", "ذ": "dh", "ر": "r",
    "ز": "z", "س": "s", "ش": "sh", "ص": "ṣ", "ض": "ḍ", "ط": "ṭ", "ظ": "ẓ", "ع": "ʿ", "غ": "gh",
    "ف": "f", "ق": "q", "ك": "k", "ل": "l", "م": "m", "ن": "n", "ه": "h", "و": "w", "ي": "y",
    "ى": "y", "ة": "t", "ء": "'", "أ": "'", "إ": "'", "ؤ": "'", "ئ": "'",
}
_LONG = {"ا": "ā", "و": "ū", "\u06e5": "ū", "ي": "ī", "\u06e6": "ī"}


def phonetic_tokens(parsed: ParsedText) -> dict[int, list[str]]:
    """Romanised phone symbols per pronounced unit, matching the Quranic phonetic ASR vocabulary.

    Conventions (Quranic Arabic Corpus transliteration): doubled consonant for shaddah, long
    vowels ``ā ī ū`` for madd letters (absorbing the preceding short vowel), ``n`` for tanween,
    and no symbol for a word-initial hamza.
    """
    pron = [u for u in parsed.units if u.pronounced]
    out: dict[int, list[str]] = {}
    for pos, u in enumerate(pron):
        nxt = pron[pos + 1] if pos + 1 < len(pron) else None
        if u.madd_letter:
            carrier = pron[pos - 1] if pos > 0 else None
            if u.char == "ى":
                out[u.index] = ["ī" if carrier is not None and carrier.vowel == Vowel.KASRA else "ā"]
            else:
                out[u.index] = ["ā" if u.synthetic else _LONG.get(u.char, "ā")]
            continue
        cons = "h" if (u.char == "ة" and u.vowel is None) else _PHONETIC.get(u.char, "")
        first_in_word = parsed.words[u.word_index].unit_indices[0] == u.index or (
            pos > 0 and pron[pos - 1].word_index != u.word_index
        )
        if u.char in HAMZA_FORMS and first_in_word:
            cons = ""
        toks = list(cons)
        prev_raw = parsed.units[u.index - 1] if u.index > 0 else None
        if u.shadda and prev_raw is not None and prev_raw.assimilated and prev_raw.char == "ل" \
                and prev_raw.word_index == u.word_index:
            toks = ["l", "-", *toks]  # the corpus writes the sun-letter article as "l-"
        elif u.shadda:
            toks = toks + toks
        if u.vowel is not None and not (nxt is not None and nxt.madd_letter and nxt.word_index == u.word_index):
            toks.append(str(u.vowel))
        if u.tanween:
            toks.append("n")
        out[u.index] = toks
    return out


def script_tokens(parsed: ParsedText) -> dict[int, list[str]]:
    """Arabic-script tokens (letter + marks) for character-level Arabic CTC models."""
    return {u.index: list(u.text) for u in parsed.units if u.pronounced}


# --------------------------------------------------------------------------- CTC Viterbi
def ctc_forced_align(log_probs: npt.NDArray[np.floating], targets: list[int], blank: int) -> npt.NDArray[np.int64]:
    """Exact CTC forced alignment (Viterbi over the blank-interleaved target lattice).

    Returns, for every frame, the index into ``targets`` of the emitted token, or ``-1`` for
    blank frames.
    """
    lp = np.asarray(log_probs, dtype=np.float64)
    n_frames, _ = lp.shape
    if not targets:
        raise AlignmentError("Empty target sequence")
    ext = np.full(2 * len(targets) + 1, blank, dtype=np.int64)
    ext[1::2] = targets
    n_states = ext.size
    repeats = sum(1 for a, b in zip(targets, targets[1:], strict=False) if a == b)
    if n_frames < len(targets) + repeats:
        raise AlignmentError(
            f"Audio too short for the text: {n_frames} frames < {len(targets) + repeats} required"
        )
    neg = -np.inf
    dp = np.full(n_states, neg)
    back = np.zeros((n_frames, n_states), dtype=np.int8)  # 0 stay, 1 from s-1, 2 from s-2
    dp[0] = lp[0, ext[0]]
    dp[1] = lp[0, ext[1]]
    can_skip = np.zeros(n_states, dtype=bool)
    can_skip[2:] = (ext[2:] != blank) & (ext[2:] != ext[:-2])
    for t in range(1, n_frames):
        stay = dp
        prev1 = np.concatenate([[neg], dp[:-1]])
        prev2 = np.where(can_skip, np.concatenate([[neg, neg], dp[:-2]]), neg)
        stacked = np.stack([stay, prev1, prev2])
        choice = np.argmax(stacked, axis=0)
        dp = stacked[choice, np.arange(n_states)] + lp[t, ext]
        back[t] = choice
    end_state = n_states - 1 if dp[-1] >= dp[-2] else n_states - 2
    if not np.isfinite(dp[end_state]):
        raise AlignmentError("No valid CTC path through the target sequence")
    path = np.empty(n_frames, dtype=np.int64)
    s = end_state
    for t in range(n_frames - 1, -1, -1):
        path[t] = s
        s -= int(back[t, s])
    return np.where(path % 2 == 1, (path - 1) // 2, -1)


def spans_from_path(frame_tokens: npt.NDArray[np.int64], token_owner: list[int], frame_s: float,
                    lp: npt.NDArray[np.floating] | None = None, targets: list[int] | None = None,
                    end_s: float | None = None) -> dict[int, AlignedUnit]:
    """Convert a frame->token path into contiguous unit spans.

    A unit starts at the first frame of its first token and ends where the next unit starts, so
    trailing blank frames (sustained vowels, held nasals) belong to the unit that produced them.
    """
    first: dict[int, int] = {}
    last: dict[int, int] = {}
    conf: dict[int, list[float]] = {}
    for t, tok in enumerate(frame_tokens):
        if tok < 0:
            continue
        unit = token_owner[tok]
        first.setdefault(unit, t)
        last[unit] = t
        if lp is not None and targets is not None:
            conf.setdefault(unit, []).append(float(np.exp(lp[t, targets[tok]])))
    order = sorted(first, key=lambda u: first[u])
    spans: dict[int, AlignedUnit] = {}
    for i, unit in enumerate(order):
        start = first[unit] * frame_s
        if i + 1 < len(order):
            end = first[order[i + 1]] * frame_s
        else:
            end = (last[unit] + 1) * frame_s
            if end_s is not None:
                end = max(end, min(end_s, end + 1.0))
        spans[unit] = AlignedUnit(unit, start, end, float(np.mean(conf[unit])) if conf.get(unit) else 1.0)
    return spans


# --------------------------------------------------------------------------- CTC aligner
class CTCForcedAligner:
    name = "ctc"

    def __init__(self, model_name: str = DEFAULT_CTC_MODEL, device: str | None = None) -> None:
        try:
            import torch
            from transformers import AutoModelForCTC, AutoProcessor
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise AlignmentError(
                "CTC alignment requires `torch` and `transformers` (pip install -r requirements-ml.txt)"
            ) from exc
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForCTC.from_pretrained(model_name).to(self.device).eval()
        except Exception as exc:  # noqa: BLE001 - network/model errors
            raise AlignmentError(f"Could not load CTC model {model_name!r}: {exc}") from exc
        self.model_name = model_name
        tok = self.processor.tokenizer
        self.vocab: dict[str, int] = dict(tok.get_vocab())
        blank = tok.pad_token_id
        self.blank = int(blank if blank is not None else 0)
        self.phonetic = "ā" in self.vocab

    def emissions(self, audio: AudioSignal) -> tuple[npt.NDArray[np.float32], float]:
        if audio.duration_s > MAX_CTC_SECONDS:
            raise AlignmentError(f"Audio longer than {MAX_CTC_SECONDS:.0f}s; analyze one ayah at a time")
        torch = self._torch
        inputs = self.processor(audio.samples, sampling_rate=audio.sr, return_tensors="pt")
        with torch.inference_mode():
            logits = self.model(inputs.input_values.to(self.device)).logits[0]
            lp = torch.log_softmax(logits.float(), dim=-1).cpu().numpy()
        frame_s = audio.duration_s / lp.shape[0]
        return lp, frame_s

    def align(self, audio: AudioSignal, parsed: ParsedText) -> Alignment:
        unit_tokens = phonetic_tokens(parsed) if self.phonetic else script_tokens(parsed)
        targets: list[int] = []
        owner: list[int] = []
        missing: list[int] = []
        for unit_idx, toks in unit_tokens.items():
            ids = [self.vocab[t] for t in toks if t in self.vocab]
            if not ids:
                missing.append(unit_idx)
                continue
            targets.extend(ids)
            owner.extend([unit_idx] * len(ids))
        lp, frame_s = self.emissions(audio)
        path = ctc_forced_align(lp, targets, self.blank)
        spans = spans_from_path(path, owner, frame_s, lp, targets, end_s=_speech_end(audio))
        _interpolate_missing(spans, missing, parsed)
        conf = float(np.mean([s.confidence for s in spans.values()])) if spans else 0.0
        return Alignment(units=spans, method=f"ctc:{self.model_name}", mean_confidence=conf)


def _interpolate_missing(spans: dict[int, AlignedUnit], missing: list[int], parsed: ParsedText) -> None:
    """Give units without any vocabulary token a zero-length span at their neighbour's boundary."""
    order = [u.index for u in parsed.units if u.pronounced]
    for idx in missing:
        pos = order.index(idx)
        prev = next((spans[o] for o in reversed(order[:pos]) if o in spans), None)
        t = prev.end_s if prev is not None else 0.0
        spans[idx] = AlignedUnit(idx, t, t, 0.0)


def _speech_end(audio: AudioSignal) -> float:
    db = frame_rms_db(audio.samples, audio.sr, 25.0, 10.0)
    active = np.flatnonzero(db > np.max(db) - 40.0)
    return float((active[-1] + 1) * 0.01 + 0.015) if active.size else audio.duration_s


# --------------------------------------------------------------------------- heuristic aligner
@dataclass(slots=True)
class HeuristicAligner:
    """Uniform segmentation over voiced speech refined by spectral-change landmarks.

    Useful for smoke tests and environments without PyTorch. Durations measured on top of it
    are approximate; the report marks the alignment as unreliable.
    """

    name: str = "heuristic"
    snap_fraction: float = 0.35
    silence_db: float = 35.0

    def align(self, audio: AudioSignal, parsed: ParsedText) -> Alignment:
        units = [u for u in parsed.units if u.pronounced]
        if not units:
            raise AlignmentError("Target text has no pronounced letters")
        hop_s = 0.01
        db = frame_rms_db(audio.samples, audio.sr, 25.0, 10.0)
        voiced = db > (np.max(db) - self.silence_db)
        idx = np.flatnonzero(voiced)
        if idx.size == 0:
            raise AlignmentError("No speech detected in audio")
        # Remove internal pauses longer than 150 ms from the time budget.
        active_frames = _active_frames(voiced, idx[0], idx[-1], min_pause=15)
        if active_frames.size < len(units):
            raise AlignmentError("Audio too short for the text")
        bounds = np.linspace(0, active_frames.size, len(units) + 1)
        novelty = _spectral_novelty(audio)
        frame_bounds = [int(active_frames[min(int(round(b)), active_frames.size - 1)]) for b in bounds[:-1]]
        frame_bounds.append(int(active_frames[-1]) + 1)
        step = active_frames.size / len(units)
        radius = max(1, int(step * self.snap_fraction))
        for i in range(1, len(frame_bounds) - 1):
            lo = max(frame_bounds[i - 1] + 1, frame_bounds[i] - radius)
            hi = min(frame_bounds[i + 1] - 1, frame_bounds[i] + radius)
            if hi > lo and hi <= novelty.size:
                frame_bounds[i] = lo + int(np.argmax(novelty[lo:hi]))
        spans = {
            u.index: AlignedUnit(u.index, frame_bounds[i] * hop_s, frame_bounds[i + 1] * hop_s, 0.3)
            for i, u in enumerate(units)
        }
        return Alignment(units=spans, method="heuristic", mean_confidence=0.3)


def _active_frames(voiced: npt.NDArray[np.bool_], first: int, last: int, min_pause: int) -> npt.NDArray[np.int64]:
    frames = np.arange(first, last + 1)
    keep = np.ones(frames.size, dtype=bool)
    run_start = None
    for k, f in enumerate(frames):
        if not voiced[f]:
            run_start = k if run_start is None else run_start
        elif run_start is not None:
            if k - run_start >= min_pause:
                keep[run_start:k] = False
            run_start = None
    return frames[keep]


def _spectral_novelty(audio: AudioSignal) -> npt.NDArray[np.float64]:
    import librosa

    mfcc = librosa.feature.mfcc(y=audio.samples, sr=audio.sr, n_mfcc=13, hop_length=160, n_fft=400)
    diff = np.linalg.norm(np.diff(mfcc, axis=1), axis=0)
    return np.concatenate([[0.0], diff])


# --------------------------------------------------------------------------- JSON loader
class JsonAlignmentLoader:
    """Load alignments from JSON.

    Accepted shapes: ``{"units": [{"unit_index": 3, "start_ms": 120, "end_ms": 260}, ...]}`` or a
    bare list. Entries may use ``index`` (position among *pronounced* units) instead of
    ``unit_index``; times may be given as ``start``/``end`` in seconds instead of milliseconds.
    """

    name = "json"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def align(self, audio: AudioSignal, parsed: ParsedText) -> Alignment:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AlignmentError(f"Could not read alignment JSON {self.path}: {exc}") from exc
        entries = data.get("units", []) if isinstance(data, dict) else data
        pron = [u.index for u in parsed.units if u.pronounced]
        spans: dict[int, AlignedUnit] = {}
        for e in entries:
            if "unit_index" in e:
                unit_idx = int(e["unit_index"])
            elif "index" in e:
                pos = int(e["index"])
                if not 0 <= pos < len(pron):
                    raise AlignmentError(f"Alignment index {pos} out of range")
                unit_idx = pron[pos]
            else:
                raise AlignmentError("Alignment entries need 'unit_index' or 'index'")
            if "start_ms" in e:
                start, end = float(e["start_ms"]) / 1000, float(e["end_ms"]) / 1000
            else:
                start, end = float(e["start"]), float(e["end"])
            if end < start:
                raise AlignmentError(f"Alignment entry for unit {unit_idx} ends before it starts")
            spans[unit_idx] = AlignedUnit(unit_idx, start, end, float(e.get("confidence", 1.0)))
        if not spans:
            raise AlignmentError(f"No alignment entries in {self.path}")
        return Alignment(units=spans, method=f"json:{self.path.name}", mean_confidence=1.0)


def export_alignment(alignment: Alignment, parsed: ParsedText) -> list[dict[str, object]]:
    rows = []
    for u in parsed.units:
        span = alignment.units.get(u.index)
        if span is None:
            continue
        rows.append({
            "unit_index": u.index, "letter": u.text, "word": parsed.words[u.word_index].text,
            "start_ms": round(span.start_s * 1000, 1), "end_ms": round(span.end_s * 1000, 1),
            "confidence": round(span.confidence, 3),
        })
    return rows


def get_aligner(kind: str = "auto", *, model_name: str = DEFAULT_CTC_MODEL,
                alignment_json: str | Path | None = None) -> Aligner:
    """Build an aligner. ``auto`` prefers JSON > CTC > heuristic."""
    if alignment_json is not None or kind == "json":
        if alignment_json is None:
            raise AlignmentError("--alignment-json is required for the json aligner")
        return JsonAlignmentLoader(alignment_json)
    if kind == "heuristic":
        return HeuristicAligner()
    if kind in ("ctc", "auto"):
        try:
            return CTCForcedAligner(model_name)
        except AlignmentError:
            if kind == "ctc":
                raise
            logger.warning("CTC aligner unavailable; falling back to the heuristic aligner", exc_info=True)
            return HeuristicAligner()
    raise AlignmentError(f"Unknown aligner {kind!r}")
