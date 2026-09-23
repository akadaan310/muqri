"""Tajweed Ahkaam: the canonical rule parser plus one acoustic validator module per rule family.

``parser`` derives every rule instance from the Uthmani text; the family modules
(``noon_sakinah``, ``meem_sakinah``, ``mudood_engine``, ``qalqalah_engine``, ``idghaam_classes``,
``raa_lam_rules``, ``sakt_wasl``) measure each instance in the audio.
"""

from app.tajweed_rules.parser import (
    HAMZA_FORMS,
    HEAVY_LETTERS,
    ParsedText,
    TajweedParseError,
    TajweedParser,
    madd_target,
    normalize_text,
    parse_text,
)

__all__ = [
    "HAMZA_FORMS",
    "HEAVY_LETTERS",
    "ParsedText",
    "TajweedParseError",
    "TajweedParser",
    "madd_target",
    "normalize_text",
    "parse_text",
]
