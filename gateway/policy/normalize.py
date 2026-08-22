"""Adversarial text normalization for pattern-based screening.

Regex screening is trivially defeated by obfuscation that leaves the text
perfectly readable to a model: a zero-width space inside a word, a Cyrillic
'a' for a Latin 'a', `1gn0re` for `ignore`. Normalizing to a canonical form
before matching closes that gap deterministically and with no added latency.

This does NOT close the semantic gap — a paraphrase or another language still
reads as different text. That needs a classifier (see `classifier.py`).
Normalization is applied for *matching only*; the original text is what gets
forwarded to the backend.
"""

import re
import unicodedata

# Zero-width and bidirectional-control characters. Invisible to a reader, and
# largely ignored by a model's tokenizer — but to a regex they break a word in
# half.
_INVISIBLE = dict.fromkeys(
    [
        0x00AD,  # soft hyphen
        0x200B,  # zero-width space
        0x200C,  # zero-width non-joiner
        0x200D,  # zero-width joiner
        0x200E,  # left-to-right mark
        0x200F,  # right-to-left mark
        0x2060,  # word joiner
        0xFEFF,  # zero-width no-break space
    ]
    + list(range(0x202A, 0x2030))  # bidi embedding / override
    + list(range(0x2066, 0x206A))  # bidi isolates
)

# Leetspeak / visually-confusable substitutions.
_LEET = {
    "0": "o",
    "1": "i",
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t",
    "@": "a",
    "$": "s",
}

# Non-Latin homoglyphs that render identically to Latin letters.
_HOMOGLYPHS = {
    "а": "a",  # Cyrillic а
    "е": "e",  # Cyrillic е
    "о": "o",  # Cyrillic о
    "р": "p",  # Cyrillic р
    "с": "c",  # Cyrillic с
    "х": "x",  # Cyrillic х
    "ѕ": "s",  # Cyrillic ѕ
    "і": "i",  # Cyrillic і
    "ο": "o",  # Greek ο
    "α": "a",  # Greek α
    "ε": "e",  # Greek ε
    "ρ": "p",  # Greek ρ
}

_WHITESPACE = re.compile(r"\s+")
_INTRA_WORD_SEPARATORS = re.compile(r"(?<=[a-z])[-._*|](?=[a-z])")
# A leet character sitting between two letters. Conservative: leaves ordinary
# numbers ("type 2 diabetes", "400 tokens") untouched.
_LEET_BETWEEN_LETTERS = re.compile(r"(?<=[a-z])([013457@$])(?=[a-z])", re.IGNORECASE)
# A whole token that mixes letters with leet characters, e.g. "1gn0re".
_LEET_TOKEN = re.compile(r"\b(?=[a-z0-9@$]*[a-z])[a-z0-9@$]+\b", re.IGNORECASE)


def _base_normalize(text: str) -> str:
    # NFKC folds compatibility forms: fullwidth i -> i, ligatures, etc.
    out = unicodedata.normalize("NFKC", text)
    out = out.translate(_INVISIBLE)
    out = "".join(_HOMOGLYPHS.get(ch, ch) for ch in out)
    out = out.casefold()
    out = _WHITESPACE.sub(" ", out)
    # Drop separators wedged between letters ("i-g-n-o-r-e", "i.g.n.o.r.e").
    out = _INTRA_WORD_SEPARATORS.sub("", out)
    return out.strip()


def normalize_for_matching(text: str) -> str:
    """Canonicalize `text` so obfuscated variants collapse onto one form.

    For screening only — never for text sent to a backend, as this is lossy.
    """
    if not text:
        return text
    out = _base_normalize(text)
    return _LEET_BETWEEN_LETTERS.sub(lambda m: _LEET[m.group(1).lower()], out)


def normalize_aggressive(text: str) -> str:
    """Like `normalize_for_matching`, but folds leet characters anywhere in a
    token that also contains letters — catching word-initial substitutions
    such as `1gn0re`, which the conservative form leaves alone.

    This deliberately over-folds ordinary tokens ("win32" -> "winea"), so use
    it only as an *additional* candidate alongside the conservative form. A
    bad fold can then only fail to match; it can never corrupt a decision.
    """
    if not text:
        return text
    out = _base_normalize(text)

    def _fold(match: "re.Match[str]") -> str:
        return "".join(_LEET.get(ch, ch) for ch in match.group(0))

    return _LEET_TOKEN.sub(_fold, out)


def matching_variants(text: str) -> set:
    """All forms `text` should be pattern-matched against."""
    if not text:
        return {text}
    return {text, normalize_for_matching(text), normalize_aggressive(text)}
