"""The QITṬ registry is a pinned GLASS research profile, NOT Tajwid.

Arithmetic and contextual letter projection are independent. A typed,
position-bearing tape decodes exactly; stripping position does not. Source
and optional shadda profiles MUST be preserved through computations.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
import json
import unicodedata


class RegistryError(ValueError):
    """A value does not satisfy the chosen registry/profile contract."""


@dataclass(frozen=True, slots=True)
class AddressCell:
    position: int
    coefficient: int
    kind: str
    glyph: str | None

    def as_dict(self) -> dict:
        return {
            "position": self.position,
            "coefficient": self.coefficient,
            "kind": self.kind,
            "glyph": self.glyph,
        }


@lru_cache(maxsize=1)
def _load() -> dict:
    path = files("unison").joinpath("data/qitt_registry_v0_1.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    alphabet = raw.get("alphabet", ())
    if len(alphabet) != 28 or len({item["glyph"] for item in alphabet}) != 28:
        raise RegistryError("Expected exactly 28 distinct GLASS letter glyphs")
    native = [x["native_value"] for x in alphabet]
    expected = list(range(1, 10)) + list(range(10, 100, 10)) + list(range(100, 1000, 100)) + [1000]
    if native != expected or alphabet[-1]["glyph"] != "غ":
        raise RegistryError("GLASS registry declaration or order does not match QITṬ/0.1")
    return raw


class QittRegistry:
    """Exact nonnegative integer addresses and native letter sums."""

    max_positions = 1024

    def __init__(self) -> None:
        self.data = _load()
        self.alphabet = tuple(self.data["alphabet"])
        self.native = {x["glyph"]: x["native_value"] for x in self.alphabet}
        self.units = tuple(x["glyph"] for x in self.alphabet[:9])
        self.tens = tuple(x["glyph"] for x in self.alphabet[9:18])
        self.hundreds = tuple(x["glyph"] for x in self.alphabet[18:27])

    def lifted_glyph(self, position: int, coefficient: int) -> str:
        if not isinstance(position, int) or isinstance(position, bool) or position < 0:
            raise RegistryError("Place must be a nonnegative integer")
        if not isinstance(coefficient, int) or isinstance(coefficient, bool) or not 1 <= coefficient <= 9:
            raise RegistryError("Nonzero coefficient must be 1..9")
        if position >= self.max_positions:
            raise RegistryError("RESOURCE_LIMIT: decimal place")
        if position == 0:
            return self.units[coefficient - 1]
        if position == 1:
            return self.tens[coefficient - 1]
        if position == 2:
            return self.hundreds[coefficient - 1]
        phase = position % 3
        if phase == 0:
            return "غ" if coefficient == 1 else self.units[coefficient - 1]
        return (self.tens if phase == 1 else self.hundreds)[coefficient - 1]

    def encode(self, number: int) -> tuple[AddressCell, ...]:
        if type(number) is not int or number < 0:
            raise RegistryError("Exact nonnegative integer required")
        if number == 0:
            return (AddressCell(0, 0, "SIFR_ZERO", None),)
        tape = []
        p = 0
        while number:
            if p >= self.max_positions:
                raise RegistryError("RESOURCE_LIMIT: decimal places")
            number, d = divmod(number, 10)
            tape.append(AddressCell(
                p, d, "SIFR_GAP" if d == 0 else "LETTER",
                None if d == 0 else self.lifted_glyph(p, d),
            ))
            p += 1
        return tuple(reversed(tape))

    def decode(self, cells: tuple[AddressCell, ...] | list[AddressCell]) -> int:
        if not cells or len(cells) > self.max_positions:
            raise RegistryError("Empty or oversized tape")
        if len(cells) == 1 and cells[0] == AddressCell(0, 0, "SIFR_ZERO", None):
            return 0
        if cells[0].coefficient == 0:
            raise RegistryError("Leading zero must not appear in canonical address")
        value = 0
        for expected_p, c in zip(range(len(cells) - 1, -1, -1), cells):
            if type(c.position) is not int or c.position != expected_p:
                raise RegistryError("Noncontiguous positional tape")
            d = c.coefficient
            if type(d) is not int or d < 0 or d > 9:
                raise RegistryError("Invalid decimal coefficient")
            if d == 0:
                if c.kind != "SIFR_GAP" or c.glyph is not None:
                    raise RegistryError("Expected typed SIFR_GAP, not an operand/operator holder")
            elif c.kind != "LETTER" or c.glyph != self.lifted_glyph(c.position, d):
                raise RegistryError("Invalid glyph/address combination")
            value = value * 10 + d
        return value

    def word_value(self, source: str, *, profile: str = "bare_graphemes") -> int:
        """A native-weight projection, not a linguistic semantic interpretation.

        Count bare register glyphs; keep all other Unicode combining marks
        orthographically, but count a shadda a second time ONLY in opt-in mode.
        Any unknown base character rejects instead of silently dropping hamza.
        """
        if profile not in {"bare_graphemes", "diacritics_preserved", "shadda_expanded_opt_in"}:
            raise RegistryError("Unknown QITṬ profile")
        if not isinstance(source, str):
            raise RegistryError("Expected source text")
        source = unicodedata.normalize("NFC", source)
        value = 0
        previous: str | None = None
        n = 0
        for c in source:
            if c in self.native:
                value += self.native[c]
                previous = c
                n += 1
            elif c == "\u0651":
                if previous is None:
                    raise RegistryError("Shadda without a letter")
                if profile == "shadda_expanded_opt_in":
                    value += self.native[previous]
                    n += 1
            elif unicodedata.combining(c):
                continue
            else:
                raise RegistryError("Unknown base glyph: source must not be silently coerced")
        if n > 9:
            raise RegistryError("RESOURCE_LIMIT: QITṬ source-word length 9")
        return value

    def word_address(self, source: str, *, profile: str = "bare_graphemes") -> tuple[AddressCell, ...]:
        return self.encode(self.word_value(source, profile=profile))

    def bare_projection(self, cells: tuple[AddressCell, ...]) -> str:
        """Purposely non-injective projection: retain cells to reverse."""
        return "".join(c.glyph or "" for c in cells)
