"""Estimate password search-space entropy for user-facing diagnostics."""

from __future__ import annotations

import math


_CHARSET_GROUPS = (
    ("lower", set("abcdefghijklmnopqrstuvwxyz")),
    ("upper", set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")),
    ("digits", set("0123456789")),
    ("symbols", set("~`!@#$%^&*()_-+={}[]|\\:;\"'<>,.?/")),
)


def charset_size(password: str) -> int:
    """Return the size of the smallest common character pool covering a password."""
    if not password:
        return 0
    size = 0
    for _, characters in _CHARSET_GROUPS:
        if any(character in characters for character in password):
            size += len(characters)
    return size


def entropy_bits(password: str) -> float:
    """Estimate Shannon-style search entropy under a uniform pool assumption."""
    pool = charset_size(password)
    return len(password) * math.log2(pool) if pool else 0.0


def strength_label(password: str) -> str:
    """Map estimated entropy to a conservative display label."""
    entropy = entropy_bits(password)
    if entropy < 28:
        return "very weak"
    if entropy < 36:
        return "weak"
    if entropy < 60:
        return "moderate"
    if entropy < 90:
        return "strong"
    return "very strong"
