"""Shared validation for user-supplied broker symbols.

Symbols flow into filesystem paths (BarCache) and generated MQL5 source
(deploy.mql5.generate_ea) without any structural guarantee from MT5 or the
UI. A crafted value could otherwise escape the bars cache directory or
break out of a string literal in generated EA code.
"""

from __future__ import annotations

import re

_SYMBOL_RE = re.compile(r"^[A-Za-z0-9._-]{1,15}$")


def validate_symbol(symbol: str) -> str:
    if not _SYMBOL_RE.match(symbol):
        raise ValueError(
            f"Invalid symbol {symbol!r}: expected 1-15 characters, "
            "letters/digits/dot/underscore/hyphen only."
        )
    return symbol
