"""UNISON/0.1 — opt-in, standard-library modular-monolith substrate.

No legacy MUQRI code is imported by default. Preserve original text,
phonetic and recitation measurement semantics; all new transforms are explicit.
"""
from .registry import QittRegistry, AddressCell, RegistryError
from .text import TextAST, TextNode, parse_text, basic_rasm_projection
from .quran import MuqattaatFrame, classify_initials, MUQATTAAT, INITIAL_LETTERS

__all__ = [
    "QittRegistry", "AddressCell", "RegistryError",
    "TextAST", "TextNode", "parse_text", "basic_rasm_projection",
    "MuqattaatFrame", "classify_initials", "MUQATTAAT", "INITIAL_LETTERS",
]
__version__ = "0.1.0"
