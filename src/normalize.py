"""Conservative, country-agnostic normalization of individual text values.

No files are read or written. For large datasets, call these functions within
a streaming or chunked workflow rather than loading all rows into memory.
"""

from numbers import Real
import unicodedata


def normalize_text(value: object) -> str:
    """Lowercase text, normalize Unicode, and replace punctuation with spaces.

    None and numeric NaN become an empty string. Other scalar values are
    converted to text (so numeric values are retained). Literal strings such
    as "NaN" are treated as text, not inferred to be missing values.

    NFC makes composed/decomposed accents equivalent without stripping them.
    Unicode punctuation (category P*) becomes spaces; symbols such as + stay.
    Whitespace is collapsed and trimmed. Numbers, accents, and business
    suffixes are preserved; no country-specific substitutions are applied.
    """
    if value is None:
        return ""

    # NaN is the numeric value that is not equal to itself.
    if isinstance(value, Real) and value != value:
        return ""

    text = unicodedata.normalize("NFC", str(value).lower())
    text = "".join(
        " " if unicodedata.category(character).startswith("P") else character
        for character in text
    )

    # split() handles tabs, newlines, and Unicode whitespace as well as spaces.
    return " ".join(text.split())


def normalize_name(value: object) -> str:
    """Normalize a business name while keeping all business suffixes."""
    return normalize_text(value)


def normalize_address(value: object) -> str:
    """Normalize an address while keeping numbers and country-neutral rules."""
    return normalize_text(value)
