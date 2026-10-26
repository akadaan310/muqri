"""Data scene contract for a future SURFACE Quran renderer.

No Android, iOS, GPU, astrophysics or native SURFACE dependency is loaded here.
The scene is an exact, bounded, serializable instruction structure. Astronomy
is an OPTIONAL, separately sourced scientific layer, never a correspondence
inferred from a verse. Simulation time does not simulate the universe.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from .quran import classify_initials, quran_scene_source


@dataclass(frozen=True, slots=True)
class LogicalClock:
    epoch_ns: int
    tick_ns: int
    generation: int = 0

    def advance(self, ticks: int = 1) -> "LogicalClock":
        if type(ticks) is not int or ticks < 0:
            raise ValueError("Nonnegative integer tick count required")
        return LogicalClock(self.epoch_ns, self.tick_ns, self.generation + ticks)

    def time_ns(self) -> int:
        return self.epoch_ns + self.tick_ns * self.generation


@dataclass(frozen=True, slots=True)
class HyperRuler:
    """An arbitrary coordinate-chart label, not evidence for extra physical dimensions."""
    axes: tuple[str, ...]
    units: tuple[str, ...]
    coordinates: tuple[float, ...]

    def __post_init__(self) -> None:
        if not 1 <= len(self.axes) <= 7:
            raise ValueError("This view supports 1..7 declared axes")
        if not (len(self.axes) == len(self.units) == len(self.coordinates)):
            raise ValueError("Axis, unit and coordinate dimensions must match")
        if len(set(self.axes)) != len(self.axes) or any(not unit for unit in self.units):
            raise ValueError("Distinct axes and explicit units required")
        if any(not math.isfinite(x) for x in self.coordinates):
            raise ValueError("Nonfinite dimension coordinate")


def oscillator_samples(
    *, amplitude_m: float, frequency_hz: float,
    phase_rad: float, sample_rate_hz: float, count: int,
) -> tuple[float, ...]:
    """Simple harmonic displacement model x(t)=A*cos(2*pi*f*t+phi)."""
    if any(not math.isfinite(x) for x in (amplitude_m, frequency_hz, phase_rad, sample_rate_hz)):
        raise ValueError("Nonfinite physical input")
    if amplitude_m < 0 or frequency_hz < 0 or sample_rate_hz <= 0 or not 0 <= count <= 4096:
        raise ValueError("Invalid units or resource budget")
    return tuple(
        amplitude_m * math.cos(2 * math.pi * frequency_hz * (i / sample_rate_hz) + phase_rad)
        for i in range(count)
    )


def quran_surface_scene(
    *, surah: int, ayah: int, raw_text: str, edition: str,
    role: str = "ordinary_word", clock: LogicalClock | None = None,
    astronomical_objects: list[dict] | None = None,
) -> dict:
    passage = quran_scene_source(surah, ayah, raw_text, edition=edition)
    if clock is None:
        clock = LogicalClock(0, 0)
    if len(raw_text) > 100_000:
        raise ValueError("RESOURCE_LIMIT: render source")
    words = passage["word_spans"]
    initials = classify_initials(raw_text, role=role, reference=f"{surah}:{ayah}").as_dict()
    astro = []
    for item in astronomical_objects or []:
        if len(astro) >= 512:
            raise ValueError("RESOURCE_LIMIT: scene object batch; use streaming tile adapter")
        if not all(x in item for x in ("catalog_id", "coordinate_frame", "ra_deg", "dec_deg", "epoch", "source")):
            raise ValueError("Only sourced astronomical objects may enter a scene")
        if not all(math.isfinite(float(item[k])) for k in ("ra_deg", "dec_deg")):
            raise ValueError("Invalid sky coordinate")
        astro.append(dict(item))
    return {
        "schema": "surface.unison.quran-scene/0.1",
        "clock": {"epoch_ns": clock.epoch_ns, "tick_ns": clock.tick_ns,
                  "generation": clock.generation, "logical_time_ns": clock.time_ns()},
        "text_layer": passage,
        "initials_layer": initials,
        "pronunciation_layer": {"provider": "muqri.tajweed_parser", "state": "DEFERRED",
                                "units": "letters+phonemes+seconds/harakat when supplied"},
        "qitt_layer": {
            "provider": "GLASS.QITT/0.1", "research_only": True,
            "does_not_overwrite_meaning_or_tajwid": True,
        },
        "astro_layer": {
            "purpose": "opt-in source-backed astronomical chart, NOT scriptural spatial decoding",
            "objects": astro, "observational_clock": "requires separately sourced ephemerides",
        },
        "render_hints": {"word_count": len(words), "word_ids": [w["id"] for w in words],
                         "arabic_rtl": True, "independent_layers": True,
                         "requires_native_surface_adapter": True},
    }
