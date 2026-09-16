"""Numeric policy (nu): compute/accumulation/conversion rules, recurrence form, exp base.

Storage dtypes of the *inputs* come from tensor metadata; nothing here is derived
from them (spec 13.2: changing an input storage dtype must not silently alter the
numerical policy).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Mapping

from .dtypes import normalize_dtype, storage_bytes
from .errors import MetadataError

RECURRENCES = ("unnormalized", "normalized")
EXP_BASES = ("e", "2")
FINAL_NORMALIZE = ("divide", "reciprocal_multiply")

# spec 3.3: "different cast locations must generate different vector operation graphs", so a cast location is
# a declared site in the modelled dataflow, not free text.  Each declared site emits its own cast op.
# site -> (stage that performs it, description).  algorithms._CAST_SITE_OPS is derived from this, so a site
# cannot be accepted by the validator and then silently emit no op.
#
# CAST_SITE_DTYPES names the (producer, consumer) dtype fields of each site, so a declared site list can be
# checked against the declared dtypes at every site, not only at two of them.  The producer of the final
# output differs by variant: the attention accumulator in the expanded path, the projection accumulator of
# Z W^v in the absorbed paths (see algorithms._cast_vector_ops).
CAST_SITE_DTYPES = {
    "score_to_exp":          ("score_dtype", "exp_dtype"),
    "exp_to_p":              ("exp_dtype", "p_operand_dtype"),
    "state_update":          ("exp_dtype", "state_dtype"),
    "accumulator_to_output": ("accumulator_dtype", "output_dtype"),
    "projected_operand":     ("projection_accumulator_dtype", "projected_operand_dtype"),
    "z_to_output":           ("accumulator_dtype", "projected_operand_dtype"),
}
CAST_SITES = {
    "score_to_exp":          ("softmax",  "cast X to the exponential's dtype before exp(X - m')"),
    "exp_to_p":              ("pv",       "cast E to the PV / P C_k matmul operand dtype"),
    "state_update":          ("softmax",  "cast the running row state (m, l) at every update"),
    "accumulator_to_output": ("finalize", "cast the final output rows to the output dtype (n*D_v)"),
    "projected_operand":     ("kv_proj",  "cast a projection accumulator to the projected operand dtype"),
    "z_to_output":           ("finalize", "cast the R_k-wide Z before the output projection (absorbed paths only)"),
}


@dataclass
class NumericPolicy:
    matmul_input_dtype: str | None = None      # operand dtype fed to the matrix unit
    accumulator_dtype: str | None = None       # A (attention accumulator) and projection accumulators
    score_dtype: str | None = None             # X
    exp_dtype: str | None = None               # E = exp(X - m)
    p_operand_dtype: str | None = None         # P/E as sent to the PV (or P C_k) matmul
    state_dtype: str | None = None             # m, l row states
    output_dtype: str | None = None            # O
    lse_dtype: str | None = None               # LSE
    projected_operand_dtype: str | None = None # storage dtype of Q^n / K^n / V / Q~ / Z operands (materialized or resident)
    projection_accumulator_dtype: str | None = None  # accumulator of the projection matmuls (may differ from the attention accumulator)
    exp_base: str | None = None                # 'e' | '2'
    recurrence: str | None = None              # 'unnormalized' | 'normalized'
    final_normalize: str | None = None         # 'divide' | 'reciprocal_multiply' (spec 5.5: implementation-dependent)
    cast_points: list[str] | None = None   # declared cast sites (see CAST_SITES); None -> inferred from dtype differences
    acceptance: dict | None = None             # caller-declared tolerances per test level (never defaults)
    dtype_bytes_overrides: dict | None = None  # explicit storage widths for unusual dtypes
    notes: list[str] = field(default_factory=list)

    def __post_init__(self):
        for f in ("matmul_input_dtype", "accumulator_dtype", "score_dtype", "exp_dtype", "p_operand_dtype",
                  "state_dtype", "output_dtype", "lse_dtype", "projected_operand_dtype", "projection_accumulator_dtype"):
            v = getattr(self, f)
            if v is not None:
                setattr(self, f, normalize_dtype(v))
        if self.exp_base is not None:
            self.exp_base = str(self.exp_base)
            if self.exp_base not in EXP_BASES:
                raise MetadataError(f"exp_base must be one of {EXP_BASES}")
        if self.recurrence is not None and self.recurrence not in RECURRENCES:
            raise MetadataError(f"recurrence must be one of {RECURRENCES}")
        if self.final_normalize is not None and self.final_normalize not in FINAL_NORMALIZE:
            raise MetadataError(f"final_normalize must be one of {FINAL_NORMALIZE}")
        if self.cast_points is not None:
            if isinstance(self.cast_points, str) or not isinstance(self.cast_points, (list, tuple)):
                raise MetadataError("cast_points must be a list of cast site names")
            bad = [x for x in self.cast_points if not isinstance(x, str)]
            if bad:
                raise MetadataError(f"cast_points entries must be site names (strings), got {bad!r}")
            unknown = sorted(set(self.cast_points) - set(CAST_SITES))
            if unknown:
                raise MetadataError(f"unknown cast_points {unknown}; expected a subset of {sorted(CAST_SITES)}")
            self.cast_points = list(dict.fromkeys(self.cast_points))
        if self.acceptance is not None and not isinstance(self.acceptance, Mapping):
            raise MetadataError("acceptance must be a mapping of tolerance names to declared values")
        if self.dtype_bytes_overrides is not None:
            if not isinstance(self.dtype_bytes_overrides, Mapping):
                raise MetadataError(f"dtype_bytes_overrides must be a mapping of dtype to byte width, "
                                    f"got {type(self.dtype_bytes_overrides).__name__}")
            for k, v in self.dtype_bytes_overrides.items():
                if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
                    raise MetadataError(f"dtype_bytes_overrides[{k!r}] must be a positive integer byte width, got {v!r}")
            self.dtype_bytes_overrides = {normalize_dtype(k): int(v) for k, v in self.dtype_bytes_overrides.items()}

    @classmethod
    def from_source(cls, src) -> "NumericPolicy":
        if src is None:
            return cls()
        if isinstance(src, cls):
            return src
        if isinstance(src, (str, Path)):
            with open(src, "r", encoding="utf-8") as fh:
                src = json.load(fh)
        if not isinstance(src, Mapping):
            raise MetadataError("numeric policy must be a mapping / JSON object")
        known = set(cls.__dataclass_fields__)
        pol = cls(**{k: v for k, v in src.items() if k in known})
        unknown = sorted(set(src) - known)
        if unknown:
            pol.notes.append(f"ignored unknown numeric fields: {unknown}")
        return pol

    def resolved_dtypes(self) -> tuple[dict[str, str], list[str]]:
        """(symbol -> dtype, assumptions).  Explicit fields win; documented fallbacks are recorded as assumptions."""
        out: dict[str, str] = {}
        assumed: list[str] = []
        direct = {"s_X": self.score_dtype, "s_E": self.exp_dtype, "s_P": self.p_operand_dtype, "s_A": self.accumulator_dtype,
                  "s_state": self.state_dtype, "s_O": self.output_dtype, "s_LSE": self.lse_dtype}
        for sym, dt in direct.items():
            if dt is not None:
                out[sym] = dt
        proj_op = self.projected_operand_dtype
        if proj_op is None and self.matmul_input_dtype is not None:
            proj_op = self.matmul_input_dtype
            assumed.append("projected_operand_dtype undeclared: Q^n/K^n/V/Q~/Z operand widths (s_Qn, s_Kn, s_V, s_Qt, s_Z) assumed = matmul_input_dtype")
        if proj_op is not None:
            for sym in ("s_Qn", "s_Kn", "s_V", "s_Qt", "s_Z"):
                out[sym] = proj_op
        proj_acc = self.projection_accumulator_dtype
        if proj_acc is None and self.accumulator_dtype is not None:
            proj_acc = self.accumulator_dtype
            assumed.append("projection_accumulator_dtype undeclared: projection accumulator width (s_Qacc) assumed = accumulator_dtype")
        if proj_acc is not None:
            out["s_Qacc"] = proj_acc
        return out, assumed

    def storage_bytes_bindings(self) -> dict[str, int]:
        """Symbol bindings (s_X, s_E, ...) for the dtypes that are declared or documented-fallback; undeclared stay unbound."""
        dts, _ = self.resolved_dtypes()
        return {sym: storage_bytes(dt, self.dtype_bytes_overrides) for sym, dt in dts.items()}

    def assumptions(self) -> list[str]:
        return self.resolved_dtypes()[1]

    def to_dict(self) -> dict:
        return asdict(self)
