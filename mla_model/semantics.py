"""Semantic configuration (task scope): fields that cannot be recovered from shape (spec 1.4).

Everything here is *declared* by the caller.  Missing fields stay ``None`` and are
reported as unknown; they are never inferred from shapes and never filled from
example values.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Mapping

from .errors import MetadataError

MASK_KINDS = ("none", "causal")
SCALE_POLICIES = ("standard", "declared")
PROJECTION_SCOPES = ("full", "new_tokens_only", "none")
EMPTY_ROW_POLICIES = ("error", "zero_output_neg_inf_lse")
LSE_CONVENTIONS = ("natural_log", "log2")


CACHE_KINDS = ("expanded_kv",)   # the only entry scope the seven-input latent graphs model (spec 5.3)


@dataclass
class SemanticConfig:
    # visibility
    mask: str | None = None                       # 'none' | 'causal'
    position_offset: int | None = None            # Delta: key j visible to query i iff j <= i + Delta
    query_position_start: int | None = None       # absolute position of query row 0 (alternative way to give Delta)
    key_position_start: int | None = None         # absolute position of key 0
    # scaling
    scale_policy: str | None = None               # 'standard' -> gamma = (D_n + D_r)^(-1/2); 'declared' -> scale_value
    scale_value: float | None = None
    scale_match_rel_tol: float | None = None      # relative tolerance used ONLY to compare a declared scale_value with the standard value
    # output scope
    outputs: tuple[str, ...] | None = None        # subset of {'O', 'LSE'}; None = undeclared (reported unknown)
    lse_convention: str | None = None             # 'natural_log' | 'log2'
    # cache / projection boundary
    projection_scope: str | None = None           # 'full' | 'new_tokens_only' | 'none'
    cache: dict | None = None                     # e.g. {'kind': 'expanded_kv', 'cached_tokens': 120}
    # degenerate rows
    empty_row_policy: str | None = None           # 'error' | 'zero_output_neg_inf_lse'
    # active lengths (ragged); capacity comes from shapes
    active_lengths: dict | None = None            # {'Sq': int | [int]*B, 'Sk': int | [int]*B}
    per_batch_offsets: list[int] | None = None    # Delta per batch (ragged case)
    segments: Any = None                          # packed-sequence segment info -> capability check
    notes: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.mask is not None and self.mask not in MASK_KINDS:
            raise MetadataError(f"mask must be one of {MASK_KINDS}, got {self.mask!r}")
        if self.scale_policy is not None and self.scale_policy not in SCALE_POLICIES:
            raise MetadataError(f"scale_policy must be one of {SCALE_POLICIES}, got {self.scale_policy!r}")
        for fname in ("scale_value", "scale_match_rel_tol"):
            v = getattr(self, fname)
            if v is None:
                continue
            # a bool is an int in Python; True read as a scale of 1.0 is a silent coercion, not a declaration
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or v in (float("inf"), float("-inf")):
                raise MetadataError(f"{fname} must be a finite number, got {v!r}")
            if fname == "scale_match_rel_tol" and v < 0:
                raise MetadataError(f"scale_match_rel_tol must be non-negative, got {v!r}")
        if self.projection_scope is not None and self.projection_scope not in PROJECTION_SCOPES:
            raise MetadataError(f"projection_scope must be one of {PROJECTION_SCOPES}")
        if self.empty_row_policy is not None and self.empty_row_policy not in EMPTY_ROW_POLICIES:
            raise MetadataError(f"empty_row_policy must be one of {EMPTY_ROW_POLICIES}")
        if self.lse_convention is not None and self.lse_convention not in LSE_CONVENTIONS:
            raise MetadataError(f"lse_convention must be one of {LSE_CONVENTIONS}")
        if self.outputs is not None:
            # a bare string iterates character by character, so "LSE" would silently read as ['L','S','E']
            if isinstance(self.outputs, (str, bytes)) or not isinstance(self.outputs, (list, tuple)):
                raise MetadataError(f"outputs must be a list of output names (e.g. ['O', 'LSE']), got {self.outputs!r}")
            self.outputs = tuple(self.outputs)
            for o in self.outputs:
                if o not in ("O", "LSE"):
                    raise MetadataError(f"unknown output {o!r}; expected 'O' and/or 'LSE'")
        if self.cache is not None and not isinstance(self.cache, dict):
            raise MetadataError("cache must be a mapping (e.g. {'kind': 'expanded_kv', 'cached_tokens': n})")
        if self.cache is not None and self.cache.get("kind") is not None and self.cache["kind"] not in CACHE_KINDS:
            # the model reads cached entries as expanded K^n|V; a latent-only cache is a different byte ledger
            # and a different projection count, so accepting the word without modelling it would mislead
            raise MetadataError(f"cache.kind {self.cache['kind']!r} is not modelled; expected one of {sorted(CACHE_KINDS)}")

    @classmethod
    def from_source(cls, src) -> "SemanticConfig":
        if src is None:
            return cls()
        if isinstance(src, cls):
            return src
        if isinstance(src, (str, Path)):
            with open(src, "r", encoding="utf-8") as fh:
                src = json.load(fh)
        if not isinstance(src, Mapping):
            raise MetadataError("semantic config must be a mapping / JSON object")
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in src.items() if k in known}
        unknown = sorted(set(src) - known)
        cfg = cls(**kwargs)
        if unknown:
            cfg.notes.append(f"ignored unknown semantic fields: {unknown}")
        return cfg

    # -- derived (never guessed) ------------------------------------------------
    def offset(self) -> tuple[int | None, list[str]]:
        """Return (Delta, conflicts).  Delta is derived only from declared fields."""
        conflicts = []
        derived = None
        if self.position_offset is not None and (isinstance(self.position_offset, bool)
                                                 or not isinstance(self.position_offset, int)):
            # checked before any arithmetic, so a non-integer never escapes as a bare ValueError
            conflicts.append(f"position_offset={self.position_offset!r} must be an integer number of positions")
            return None, conflicts
        if self.query_position_start is not None or self.key_position_start is not None:
            for name in ("query_position_start", "key_position_start"):
                v = getattr(self, name)
                if v is not None and (isinstance(v, bool) or not isinstance(v, int)):
                    conflicts.append(f"{name}={v!r} must be an integer position")
                    return None, conflicts
            if self.query_position_start is None or self.key_position_start is None:
                conflicts.append("both query_position_start and key_position_start are needed to derive the offset")
            else:
                derived = int(self.query_position_start) - int(self.key_position_start)
        if self.position_offset is not None and derived is not None and int(self.position_offset) != derived:
            conflicts.append(
                f"position_offset={self.position_offset} conflicts with query_position_start-key_position_start={derived}")
            return None, conflicts
        if self.position_offset is not None:
            return int(self.position_offset), conflicts
        return derived, conflicts

    def wants_lse(self):
        """True / False when outputs are declared, None when undeclared."""
        if self.outputs is None:
            return None
        return "LSE" in self.outputs

    def to_dict(self) -> dict:
        d = asdict(self)
        d["outputs"] = None if self.outputs is None else list(self.outputs)
        return d
