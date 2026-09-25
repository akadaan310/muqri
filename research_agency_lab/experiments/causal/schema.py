"""The causal-experiment record, and the rules that classify what changed.

A record keeps three things strictly apart:

  intended   what we meant to change (hypothesis: a physical quantity, a direction, a Muqri target)
  physics    what the signal-processing transform actually did to the audio, measured independently
             of Muqri (Praat in Python; cross-checked by Octave and Julia)
  muqri      what the unchanged Muqri engine then measured

and every measurement delta is classified against a noise floor taken from the record's own no-op
controls, never against an assumed zero. Nothing here judges statistical significance: with a
handful of units per cell there is none to claim.

Classification of one metric's delta d, given its noise floor n (>= a fixed minimum per unit):

  target metric    |d| <= n            -> "stable"            (the hypothesis is NOT supported here)
                   |d| >  n, expected  -> "expected change"
                   |d| >  n, opposite  -> "unexpected change"
  other metric     |d| <= n            -> "stable"
                   |d| >  n            -> "unexpected change" (collateral)
  missing value / unknown noise        -> "insufficient evidence"

Physics verification of the transform itself (requested vs measured):

  VERIFIED       the targeted physical metric moved beyond its noise floor in the requested direction
                 (and within a factor of 2 of the predicted size, where a size is predicted)
  PARTIAL        right direction, beyond noise, but off the predicted size by more than 2x
  NOT_VERIFIED   within noise, or the wrong direction
  UNMEASURABLE   the metric could not be measured on this span
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION = "causal-1"

EXPECTED, UNEXPECTED, STABLE, INSUFFICIENT = "expected change", "unexpected change", "stable", "insufficient evidence"
VERIFIED, PARTIAL, NOT_VERIFIED, UNMEASURABLE = "VERIFIED", "PARTIAL", "NOT_VERIFIED", "UNMEASURABLE"

# minimum noise floors, per unit of measurement, when the no-op controls show less than this. Muqri
# durations live on its 40 ms frame grid; physical durations are measured from the samples (the span)
# or from Praat's 5 ms pitch step, so they get their own, finer floor.
MIN_NOISE = {"nats": 1.0, "s": 0.041, "counts": 0.25, "fraction": 0.05, "hz": 15.0, "db": 1.0, "ratio": 0.02}
MIN_NOISE_PHYSICS = {**MIN_NOISE, "s": 0.005}


@dataclass
class Unit:
    reciter: str
    surah: int
    ayah: int
    letter_id: str            # engine id, e.g. "48:19:L12"
    symbol: str
    form: str                 # fatha / kasra / ... / rule:<type>
    word: int
    unit_type: str            # "letter" | "letter+vowel" | "rule"
    start_s: float
    end_s: float
    boundary_method: str      # how start/end were found
    source_audio: str         # a stable reference (EveryAyah folder/file), not a local path


@dataclass
class Transform:
    name: str                 # duration | voicing | formant | f0 | nasal | noop_world | noop_splice | swap
    parameters: dict[str, Any]
    implementation: str       # the function that did it
    span: str                 # which part of the unit it touched


@dataclass
class Hypothesis:
    physical_metric: str      # e.g. "voiced_fraction"
    physical_direction: int   # +1 / -1 / 0 (0 = must stay)
    predicted_physical_delta: float | None
    muqri_targets: dict[str, int]   # muqri metric -> expected direction (+1/-1); others must stay stable
    statement: str


@dataclass
class Record:
    experiment_id: str
    schema: str
    unit: Unit
    transform: Transform
    hypothesis: Hypothesis
    requested_delta: dict[str, Any]
    physics_baseline: dict[str, Any]
    physics_transformed: dict[str, Any]
    delta_physics: dict[str, float | None]
    physics_verification: dict[str, Any]
    muqri_baseline: dict[str, Any]
    muqri_transformed: dict[str, Any]
    delta_muqri: dict[str, float | None]
    classification: dict[str, str]
    collateral: dict[str, Any]
    crosscheck: dict[str, Any]
    controls: list[str]
    reproducibility: dict[str, Any]
    verification_status: str
    transformed_audio: str | None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def unit_of(metric: str) -> str:
    if metric.endswith(":margin") or metric.startswith("makhraj:"):
        return "nats"
    if metric.endswith("duration_s"):
        return "s"
    if metric.endswith(":counts"):
        return "counts"
    if metric in ("voiced_fraction",):
        return "fraction"
    if metric.startswith(("f0", "f1", "f2", "f3")):
        return "hz"
    if metric.endswith("_db"):
        return "db"
    return "ratio"


def deltas(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for k in sorted(set(a) | set(b)):
        x, y = a.get(k), b.get(k)
        out[k] = round(y - x, 4) if isinstance(x, (int, float)) and isinstance(y, (int, float)) \
            and not (isinstance(x, bool) or isinstance(y, bool)) and math.isfinite(x) and math.isfinite(y) else None
    return out


def noise_floors(noop_deltas: list[dict[str, float | None]], layer: str = "muqri") -> dict[str, float]:
    """Per metric: the largest |delta| any no-op control produced, raised to the unit's minimum."""
    mins = MIN_NOISE_PHYSICS if layer == "physics" else MIN_NOISE
    keys = {k for d in noop_deltas for k in d}
    out = {}
    for k in keys:
        vals = [abs(d[k]) for d in noop_deltas if d.get(k) is not None]
        out[k] = max([mins[unit_of(k)], *vals]) if vals else mins[unit_of(k)]
    return out


def classify(delta: dict[str, float | None], noise: dict[str, float], targets: dict[str, int]) -> dict[str, str]:
    out = {}
    for k, d in delta.items():
        n = noise.get(k)
        if d is None or n is None:
            out[k] = INSUFFICIENT
        elif abs(d) <= n:
            out[k] = STABLE
        elif k in targets:
            out[k] = EXPECTED if math.copysign(1, d) == targets[k] else UNEXPECTED
        else:
            out[k] = UNEXPECTED
    for k in targets:
        out.setdefault(k, INSUFFICIENT)
    return out


def verify_physics(metric: str, direction: int, predicted: float | None, measured: float | None,
                   noise: float) -> dict[str, Any]:
    if measured is None:
        return {"status": UNMEASURABLE, "metric": metric}
    beyond = abs(measured) > noise
    right = direction == 0 or (measured != 0 and math.copysign(1, measured) == direction)
    if direction == 0:
        status = VERIFIED if not beyond else NOT_VERIFIED
    elif not (beyond and right):
        status = NOT_VERIFIED
    elif predicted is not None and predicted != 0 and not (0.5 <= measured / predicted <= 2.0):
        status = PARTIAL
    else:
        status = VERIFIED
    return {"status": status, "metric": metric, "requested_direction": direction, "predicted_delta": predicted,
            "measured_delta": measured, "noise_floor": noise}
