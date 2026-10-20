"""Bounded decimal-integer helpers shared by public text I/O adapters."""

from __future__ import annotations

from decimal import Decimal

PUBLIC_IO_MAX_DECIMAL_DIGITS = 1000
PUBLIC_IO_MAX_ABS_INT = 10**PUBLIC_IO_MAX_DECIMAL_DIGITS - 1


class PublicIntegerTokenTooLong(ValueError):
    """A public decimal token exceeds the packing3d I/O contract."""


def _digit_count(token: str) -> int:
    return len(token) - 1 if token.startswith(("+", "-")) else len(token)


def parse_public_decimal_token(token: str) -> int:
    """Convert a bounded decimal token without changing process-global limits."""
    if _digit_count(token) > PUBLIC_IO_MAX_DECIMAL_DIGITS:
        raise PublicIntegerTokenTooLong(
            f"integer token exceeds {PUBLIC_IO_MAX_DECIMAL_DIGITS} decimal digits"
        )
    try:
        return int(token)
    except ValueError as exc:
        # CPython may be configured with an int<->decimal-string guard below
        # our public 1000-digit contract.  Keep the public contract local and
        # bounded without changing sys.set_int_max_str_digits().
        try:
            return int(Decimal(token))
        except Exception:
            raise exc


def require_public_io_int(value: object, *, name: str) -> int:
    """Validate a Python int against the public 1000-decimal-digit bound."""
    if type(value) is not int:
        raise TypeError(f"{name} must be a Python int")
    if value < -PUBLIC_IO_MAX_ABS_INT or value > PUBLIC_IO_MAX_ABS_INT:
        raise ValueError(
            f"{name} exceeds the public {PUBLIC_IO_MAX_DECIMAL_DIGITS}-decimal-digit limit"
        )
    return value


def public_int_text(value: int, *, name: str) -> str:
    """Render one already-bounded public integer without changing global limits."""
    checked = require_public_io_int(value, name=name)
    try:
        return str(checked)
    except ValueError:
        # Same bounded fallback as parsing. Decimal(int) is exact and does not
        # consult CPython's decimal-string digit guard.
        return format(Decimal(checked), "f")
