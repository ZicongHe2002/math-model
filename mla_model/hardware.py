"""Hardware profile (eta) and calibration evidence.

A profile is data loaded from JSON/dict (or an optional capability-checked runtime
adapter); the mathematical formulas never embed device constants (spec 9.1).
Every quantity carries value, unit, scope (per_core / per_tensorcore / per_chip /
per_device), source and optional precision so that mismatched scopes are detected
instead of divided by one another.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Mapping

from .dtypes import normalize_dtype
from .errors import MetadataError

SCOPES = ("per_core", "per_tensorcore", "per_chip", "per_device", "per_host")


@dataclass
class Quantity:
    value: float | None
    unit: str
    scope: str | None = None
    source: str | None = None
    precision: str | None = None        # dtype the figure applies to (throughputs)
    confidence: str | None = None       # 'datasheet' | 'measured' | 'estimated' | ...
    valid_for: dict | None = None       # toolchain / compiler versions the figure applies to

    @classmethod
    def from_any(cls, v, unit: str, default_scope: str | None) -> "Quantity":
        if v is None:
            return cls(None, unit, default_scope)
        if isinstance(v, Quantity):
            return v
        if isinstance(v, Mapping):
            scope = v.get("scope", default_scope)
            if scope is not None and scope not in SCOPES:
                raise MetadataError(f"unknown hardware scope {scope!r}; expected one of {SCOPES}")
            raw = v.get("value")
            if raw is not None and (isinstance(raw, bool) or not isinstance(raw, (int, float))):
                raise MetadataError(f"hardware quantity value must be a number, got {raw!r}")
            return cls(value=None if raw is None else float(raw), unit=v.get("unit", unit),
                       scope=scope, source=v.get("source"), precision=v.get("precision"),
                       confidence=v.get("confidence"), valid_for=v.get("valid_for"))
        if isinstance(v, (int, float)):
            return cls(float(v), unit, default_scope, source="bare number in profile (no provenance)")
        raise MetadataError(f"cannot interpret hardware quantity {v!r}")

    def __post_init__(self):
        # a rate of zero or less is not an upper bound; treat it as a declaration error rather than
        # letting it reach a division and surface as ZeroDivisionError
        if self.value is not None and self.unit in ("FLOP/s", "op/s", "byte/s") and float(self.value) <= 0:
            raise MetadataError(f"hardware quantity with unit {self.unit} must be positive, got {self.value}")
        if self.value is not None and self.unit == "byte" and float(self.value) < 0:
            raise MetadataError(f"hardware capacity must be non-negative, got {self.value}")
        if self.value is not None and self.unit == "s" and float(self.value) < 0:
            raise MetadataError(f"a time term must be non-negative, got {self.value} s")

    def known(self) -> bool:
        return self.value is not None

    def to_dict(self) -> dict:
        return asdict(self)


def device_matches(bucket_device: str | None, profile_name: str | None) -> bool:
    """Whether a bucket's declared device identifies this profile (case-insensitive substring, either way).

    The single predicate both the profile notes and the calibrated prediction use, so the two surfaces of one
    report cannot disagree about whether the evidence belongs to this device.
    """
    if bucket_device is None or profile_name in (None, "unspecified"):
        return False
    a, b = str(bucket_device).strip().lower(), str(profile_name).strip().lower()
    return bool(a) and (a in b or b in a)


@dataclass
class CalibrationBucket:
    """Measured efficiency for a class of nodes; applicability is explicit (spec 9.3)."""
    node_kind: str                       # 'matmul' | 'exp' | 'reduce' | 'layout' | 'hbm_transfer'
    dtype: str | None = None
    epsilon_low: float | None = None     # efficiency interval relative to peak
    epsilon_high: float | None = None
    startup_s: float | None = None        # None = not measured (kept missing, never 0)
    m_range: tuple | None = None         # applicability ranges (inclusive); None = any
    n_range: tuple | None = None
    k_range: tuple | None = None
    source: str | None = None
    device: str | None = None
    toolchain: dict | None = None
    validated_on_unseen_shapes: bool | None = None
    validation_error: float | None = None

    def __post_init__(self):
        if self.node_kind not in ("matmul", "exp", "reduce", "layout", "hbm_transfer", "vector"):
            raise MetadataError(f"calibration bucket node_kind {self.node_kind!r} unknown")
        for name in ("epsilon_low", "epsilon_high", "startup_s", "validation_error"):
            v = getattr(self, name)
            if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float))):
                raise MetadataError(f"calibration bucket {name} must be a number, got {v!r}")
        for name in ("m_range", "n_range", "k_range"):
            rng = getattr(self, name)
            if rng is None:
                continue
            if isinstance(rng, (str, bytes)) or not isinstance(rng, (list, tuple)) or len(rng) != 2:
                raise MetadataError(f"calibration bucket {name} must be [low, high] (either may be null), got {rng!r}")
            for x in rng:
                if x is not None and (isinstance(x, bool) or not isinstance(x, (int, float))):
                    raise MetadataError(f"calibration bucket {name} bounds must be numbers or null, got {rng!r}")
            setattr(self, name, tuple(rng))
        if self.toolchain is not None and not isinstance(self.toolchain, Mapping):
            raise MetadataError(f"calibration bucket toolchain must be a mapping, got {type(self.toolchain).__name__}")
        if self.device is not None and not isinstance(self.device, str):
            raise MetadataError(f"calibration bucket device must be a string, got {self.device!r}")
        if self.dtype is not None:
            self.dtype = normalize_dtype(self.dtype)
        for name in ("epsilon_low", "epsilon_high"):
            v = getattr(self, name)
            if v is not None and not (0 < float(v) <= 1):
                raise MetadataError(f"calibration bucket {name}={v} must lie in (0, 1]")
        if self.epsilon_low is not None and self.epsilon_high is not None and self.epsilon_low > self.epsilon_high:
            raise MetadataError("calibration bucket epsilon_low must not exceed epsilon_high")
        if self.startup_s is not None and float(self.startup_s) < 0:
            raise MetadataError("calibration bucket startup_s must be non-negative")
        if self.validated_on_unseen_shapes and self.validation_error is None:
            # spec 9.3 asks for the validation error on new shapes, so the claim without the number is not
            # validation evidence; the bucket stays usable but the claim does not suppress the caveat
            raise MetadataError("a calibration bucket claiming validated_on_unseen_shapes must also carry "
                                "validation_error (spec 9.3: report the validation error on new shapes)")

    def applies(self, dtype: str | None, M=None, N=None, K=None) -> bool:
        if self.dtype and dtype and self.dtype != normalize_dtype(dtype):
            return False
        for rng, val in ((self.m_range, M), (self.n_range, N), (self.k_range, K)):
            if rng is not None:
                if val is None:
                    return False
                lo, hi = rng
                if (lo is not None and val < lo) or (hi is not None and val > hi):
                    return False
        return True


@dataclass
class HardwareProfile:
    name: str = "unspecified"
    scope_default: str | None = None
    provenance: str | None = None
    retrieved: str | None = None
    toolchain: dict = field(default_factory=dict)
    compute: dict[str, Quantity] = field(default_factory=dict)   # e.g. 'mxu.peak_flops.bfloat16' -> Quantity
    memory: dict[str, Quantity] = field(default_factory=dict)    # e.g. 'vmem.capacity_bytes'
    paths: dict[str, Quantity] = field(default_factory=dict)     # e.g. 'hbm_to_vmem.bandwidth'
    layout: dict[str, tuple[int, int]] = field(default_factory=dict)  # dtype -> (L_s, L_l)
    calibration: list[CalibrationBucket] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    # -- loading ----------------------------------------------------------------
    @classmethod
    def from_source(cls, src) -> "HardwareProfile | None":
        if src is None:
            return None
        if isinstance(src, cls):
            return src
        if isinstance(src, (str, Path)):
            with open(src, "r", encoding="utf-8") as fh:
                src = json.load(fh)
        if not isinstance(src, Mapping):
            raise MetadataError("hardware profile must be a mapping / JSON object")
        if src.get("toolchain") is not None and not isinstance(src.get("toolchain"), Mapping):
            raise MetadataError(f"hardware profile toolchain must be a mapping (e.g. {{'compiler': ..., 'version': ...}}), "
                                f"got {type(src.get('toolchain')).__name__}")
        prof = cls(name=src.get("name", "unspecified"), scope_default=src.get("scope_default"),
                   provenance=src.get("provenance"), retrieved=src.get("retrieved"),
                   toolchain=dict(src.get("toolchain", {}) or {}), notes=list(src.get("notes", []) or []))
        known_top = {"name", "scope_default", "provenance", "retrieved", "toolchain", "notes", "compute", "memory", "paths", "layout", "calibration"}
        unknown_top = sorted(set(src) - known_top)
        if unknown_top:
            prof.notes.append(f"ignored unknown hardware profile fields: {unknown_top}")
        if prof.scope_default is not None and prof.scope_default not in SCOPES:
            raise MetadataError(f"scope_default {prof.scope_default!r} not in {SCOPES}")
        units = {"peak_flops": "FLOP/s", "throughput": "op/s", "capacity_bytes": "byte", "bandwidth": "byte/s",
                 "count": "count", "bytes_per_vreg": "byte", "usable_capacity_bytes": "byte",
                 "scoped_budget_bytes": "byte", "launch_overhead_s": "s", "model_error_s": "s"}

        def _unit_for(key: str) -> str:
            for k, u in units.items():
                if k in key:
                    return u
            return "unknown"

        for section in ("compute", "memory", "paths"):
            flat = _flatten(src.get(section, {}) or {})
            store = getattr(prof, section)
            for key, val in flat.items():
                if key.startswith("mxu.peak_flops."):
                    key = "mxu.peak_flops." + normalize_dtype(key.split(".", 2)[2])
                store[key] = Quantity.from_any(val, _unit_for(key), prof.scope_default)
        for dt, dims in (src.get("layout", {}) or {}).items():
            if dims is None:
                continue
            if not (isinstance(dims, (list, tuple)) and len(dims) == 2):
                raise MetadataError(f"layout tile for {dt} must be [L_s, L_l]")
            if any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in dims):
                raise MetadataError(f"layout tile for {dt} must be two positive integers, got {list(dims)!r}")
            prof.layout[normalize_dtype(dt)] = (int(dims[0]), int(dims[1]))
        allowed = set(CalibrationBucket.__dataclass_fields__)
        for b in src.get("calibration", []) or []:
            if not isinstance(b, Mapping):
                raise MetadataError("calibration entries must be objects")
            unknown = sorted(set(b) - allowed)
            if unknown:
                raise MetadataError(f"calibration bucket has unknown fields {unknown}; allowed: {sorted(allowed)}")
            b = dict(b)
            for r in ("m_range", "n_range", "k_range"):
                if b.get(r) is not None:
                    if isinstance(b[r], (str, bytes)) or not isinstance(b[r], (list, tuple)):
                        raise MetadataError(f"calibration bucket {r} must be a 2-element list [lo, hi] (null bounds allowed), got {b[r]!r}")
                    b[r] = tuple(b[r])
            bucket = CalibrationBucket(**b)
            if bucket.device and not device_matches(bucket.device, prof.name):
                prof.notes.append(f"calibration bucket measured on device {bucket.device!r} attached to profile {prof.name!r}: validity not established")
            prof.calibration.append(bucket)
        return prof

    # -- queries ------------------------------------------------------------------
    def get(self, section: str, key: str) -> Quantity | None:
        store = getattr(self, section)
        return store.get(key)

    def peak_matmul(self, dtype: str | None) -> Quantity | None:
        if dtype is None:
            return None
        return self.compute.get(f"mxu.peak_flops.{normalize_dtype(dtype)}")

    def calibration_summary(self) -> list[dict]:
        # the ranges and the toolchain are what decide whether a bucket matches a node, so they belong in the
        # summary a reader uses to judge the prediction
        return [{"node_kind": b.node_kind, "dtype": b.dtype, "epsilon": [b.epsilon_low, b.epsilon_high], "startup_s": b.startup_s,
                 "m_range": None if b.m_range is None else list(b.m_range), "n_range": None if b.n_range is None else list(b.n_range),
                 "k_range": None if b.k_range is None else list(b.k_range), "toolchain": b.toolchain,
                 "source": b.source, "device": b.device, "validated_on_unseen_shapes": b.validated_on_unseen_shapes,
                 "validation_error": b.validation_error} for b in self.calibration]

    def quantity_summary(self) -> list[dict]:
        """Every declared quantity with its value, unit, scope, source and confidence (spec 9.1)."""
        rows = []
        for section in ("compute", "memory", "paths"):
            for key, q in sorted(getattr(self, section).items()):
                rows.append({"section": section, "key": key, "value": q.value, "unit": q.unit, "scope": q.scope,
                             "source": q.source, "confidence": q.confidence, "precision": q.precision,
                             "valid_for": q.valid_for})
        return rows

    def layout_tile(self, dtype: str | None) -> tuple[int, int] | None:
        if dtype is None:
            return None
        return self.layout.get(normalize_dtype(dtype))

    def bindings(self, compute_dtype: str | None) -> dict[str, float]:
        """Symbol bindings (P_mxu, W_hbm, ...) for known quantities; unknown stay unbound."""
        out: dict[str, float] = {}
        q = self.peak_matmul(compute_dtype)
        if q and q.known():
            out["P_mxu"] = q.value
        for sym, key in (("P_exp", "vpu.exp_throughput"), ("P_vpu", "vpu.elementwise_throughput"),
                         ("P_red", "vpu.reduction_throughput"), ("P_layout", "vpu.layout_throughput")):
            qq = self.compute.get(key)
            if qq and qq.known():
                out[sym] = qq.value
        for sym, key in (("W_hbm", "hbm_to_vmem.bandwidth"), ("W_hbm_w", "vmem_to_hbm.bandwidth"),
                         ("W_vmem", "vmem_to_vreg.bandwidth"), ("W_vmem_w", "vreg_to_vmem.bandwidth")):
            qq = self.paths.get(key)
            if qq and qq.known():
                out[sym] = qq.value
        for sym, key in (("R_vreg", "vreg.usable_capacity_bytes"), ("R_vmem", "vmem.scoped_budget_bytes")):
            qq = self.memory.get(key)
            if qq and qq.known():
                out[sym] = qq.value
        lo = self.compute.get("launch_overhead_s")
        if lo and lo.known():
            out["T_launch"] = lo.value
        # spec 9.3's third term of T_hat; declared with the profile's calibration evidence, never assumed 0
        me = self.compute.get("model_error_s")
        if me and me.known():
            out["eps_model"] = me.value
        return out

    def scope_report(self) -> list[str]:
        """List scope inconsistencies among known quantities (values of different scopes must not be divided)."""
        scopes = {}
        for section in ("compute", "memory", "paths"):
            for key, q in getattr(self, section).items():
                if q.known():
                    scopes.setdefault(q.scope, []).append(f"{section}.{key}")
        if len(scopes) > 1:
            return [f"mixed scopes in profile: {dict((k or 'unspecified', v) for k, v in scopes.items())}"]
        if None in scopes:
            return ["quantities without a declared scope: " + ", ".join(scopes[None])]
        return []

    def to_dict(self) -> dict:
        return {
            "name": self.name, "scope_default": self.scope_default, "provenance": self.provenance,
            "retrieved": self.retrieved, "toolchain": self.toolchain,
            "compute": {k: v.to_dict() for k, v in self.compute.items()},
            "memory": {k: v.to_dict() for k, v in self.memory.items()},
            "paths": {k: v.to_dict() for k, v in self.paths.items()},
            "layout": {k: list(v) for k, v in self.layout.items()},
            "calibration": [asdict(b) for b in self.calibration],
            "notes": list(self.notes),
        }


def _flatten(d: Mapping, prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts to dotted keys, stopping at quantity dicts (those with a 'value' key)."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, Mapping) and "value" not in v and "unit" not in v:
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def from_runtime_jax(device_index: int = 0) -> HardwareProfile | None:
    """Optional runtime adapter: read what ``jax.experimental.pallas.tpu.TpuInfo`` exposes.

    Lazily imports JAX; returns None (never raises) when JAX or a TPU is unavailable.
    Only fields the runtime actually reports are filled; bandwidths and register
    capacities are NOT invented (spec 9.1).  This function has only been exercised
    on a CPU host where it returns None.
    """
    try:  # pragma: no cover - depends on the host
        import jax  # noqa: F401
        from jax.experimental.pallas import tpu as pltpu  # noqa: F401
    except Exception:
        return None
    try:  # pragma: no cover
        devices = jax.devices()
        if not devices or devices[device_index].platform != "tpu":
            return None
        dev = devices[device_index]
        info = None
        get_info = getattr(pltpu, "get_tpu_info", None)
        if callable(get_info):
            info = get_info()
        prof = HardwareProfile(name=f"runtime:{dev.device_kind}", scope_default="per_tensorcore",
                               provenance="jax runtime device query (TpuInfo); fields not reported stay unknown",
                               toolchain={"jax": jax.__version__})
        if info is not None:
            for attr, key in (("vmem_capacity_bytes", "vmem.capacity_bytes"), ("cmem_capacity_bytes", "cmem.capacity_bytes"),
                              ("smem_capacity_bytes", "smem.capacity_bytes"), ("hbm_capacity_bytes", "hbm.capacity_bytes")):
                val = getattr(info, attr, None)
                if val is not None:
                    prof.memory[key] = Quantity(float(val), "byte", "per_tensorcore", source="TpuInfo")
        return prof
    except Exception:
        return None
