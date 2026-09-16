"""Exception types for the MLA parametric model.

Every failure that would otherwise tempt a silent fallback is an explicit exception
carrying the full list of violated constraints so the caller sees all of them at once.
"""
from __future__ import annotations


class MLAModelError(Exception):
    """Base class for all model errors."""


class UnsupportedDtypeError(MLAModelError):
    """A dtype whose storage width is unknown, or a packed/sub-byte dtype the adapter does not model."""


class CapabilityError(MLAModelError):
    """The adapter was asked for a case (packed, quantized, sharded, zero-length, ...) it does not declare support for."""


class MetadataError(MLAModelError):
    """Malformed metadata (missing role, bad shape, unknown axis name, ...)."""


class BindingError(MLAModelError):
    """Contradictory or incomplete bindings. ``violations`` lists every problem found."""

    def __init__(self, violations, message=None):
        self.violations = list(violations)
        text = message or "binding failed"
        detail = "\n".join(f"  - {v}" for v in self.violations)
        super().__init__(f"{text}:\n{detail}" if detail else text)


class TaskIdentityError(MLAModelError):
    """A candidate strategy changed the task (shape, mask, scale, scope, numerics) instead of only the execution strategy."""


class ReferenceSemanticsError(MLAModelError):
    """A small-shape reference was asked to evaluate an ill-defined case (e.g. all-masked row with policy 'error')."""
