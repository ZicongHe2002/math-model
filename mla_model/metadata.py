"""Tensor metadata carriers (metadata-only: shape, dtype, axis names, strides, aliases).

No tensor values are ever read.  ``TensorMeta.from_array_like`` duck-types on
``.shape`` / ``.dtype`` so numpy arrays, ``jax.ShapeDtypeStruct``, torch tensors,
or any user object with those attributes can be described without importing the
owning library (spec 1.2).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .dtypes import normalize_dtype, storage_bytes
from .errors import MetadataError


@dataclass
class TensorMeta:
    role: str
    shape: tuple[int, ...]
    dtype: str
    axes: tuple[str, ...] | None = None           # semantic axis names in physical order
    strides: tuple[int, ...] | None = None        # element strides, optional
    storage_offset: int = 0
    alias_of: str | None = None                   # role of another tensor sharing storage
    layout: str | None = None                     # physical layout hint (free text, adapter-specific)
    quantization: dict | None = None              # presence triggers a capability check
    sharding: dict | None = None                  # presence triggers a capability check
    capacity_shape: tuple[int, ...] | None = None  # allocated capacity if larger than logical shape
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.shape, (tuple, list)):
            raise MetadataError(f"{self.role}: shape must be a sequence, got {type(self.shape).__name__}")
        shape = []
        for d in self.shape:
            if isinstance(d, bool) or not isinstance(d, int):
                raise MetadataError(f"{self.role}: shape entries must be integers, got {d!r}")
            if d < 0:
                raise MetadataError(f"{self.role}: negative extent {d}")
            shape.append(int(d))
        self.shape = tuple(shape)
        self.dtype = normalize_dtype(self.dtype)
        if self.axes is not None:
            self.axes = tuple(str(a) for a in self.axes)
            if len(self.axes) != len(self.shape):
                raise MetadataError(
                    f"{self.role}: {len(self.axes)} axis names for a rank-{len(self.shape)} shape")
        def _ints(name, seq):
            if isinstance(seq, (str, bytes)) or not isinstance(seq, (list, tuple)):
                raise MetadataError(f"{self.role}: {name} must be a list of integers, got {seq!r}")
            out = []
            for x in seq:
                if isinstance(x, bool) or not isinstance(x, int):
                    # a 10.7 coerced to 10 would silently change the allocation; refuse instead
                    raise MetadataError(f"{self.role}: {name} entries must be integers, got {x!r}")
                out.append(int(x))
            return tuple(out)
        if self.strides is not None:
            self.strides = _ints("strides", self.strides)
            if len(self.strides) != len(self.shape):
                raise MetadataError(f"{self.role}: strides rank mismatch")
        if isinstance(self.storage_offset, bool) or not isinstance(self.storage_offset, int) or self.storage_offset < 0:
            raise MetadataError(f"{self.role}: storage_offset must be a non-negative integer, got {self.storage_offset!r}")
        if self.capacity_shape is not None:
            self.capacity_shape = _ints("capacity_shape", self.capacity_shape)
            if len(self.capacity_shape) != len(self.shape):
                raise MetadataError(f"{self.role}: capacity_shape rank mismatch")
            for cap, log in zip(self.capacity_shape, self.shape):
                if cap < log:
                    raise MetadataError(f"{self.role}: capacity {cap} smaller than logical extent {log}")

    # -- constructors -----------------------------------------------------
    @classmethod
    def from_dict(cls, role: str, d: Mapping[str, Any]) -> "TensorMeta":
        if "shape" not in d:
            raise MetadataError(f"{role}: metadata entry has no 'shape'")
        if "dtype" not in d:
            raise MetadataError(f"{role}: metadata entry has no 'dtype' (dtypes are read per tensor, never assumed)")
        known = {"shape", "dtype", "axes", "strides", "storage_offset", "alias_of", "layout",
                 "quantization", "sharding", "capacity_shape"}
        extra = {k: v for k, v in d.items() if k not in known}
        return cls(
            role=role,
            shape=tuple(d["shape"]),
            dtype=d["dtype"],
            axes=tuple(d["axes"]) if d.get("axes") is not None else None,
            strides=tuple(d["strides"]) if d.get("strides") is not None else None,
            storage_offset=d.get("storage_offset", 0) if d.get("storage_offset") is not None else 0,
            alias_of=d.get("alias_of"),
            layout=d.get("layout"),
            quantization=d.get("quantization"),
            sharding=d.get("sharding"),
            capacity_shape=tuple(d["capacity_shape"]) if d.get("capacity_shape") is not None else None,
            extra=extra,
        )

    @classmethod
    def from_array_like(cls, role: str, obj: Any, axes=None, **kw) -> "TensorMeta":
        """Describe any object exposing ``.shape`` and ``.dtype`` without touching its values."""
        shape = getattr(obj, "shape", None)
        dtype = getattr(obj, "dtype", None)
        if shape is None or dtype is None:
            raise MetadataError(f"{role}: object of type {type(obj).__name__} has no shape/dtype metadata")
        sharding = getattr(obj, "sharding", None)
        sharding_info = None
        if sharding is not None and not _is_trivial_sharding(sharding):
            sharding_info = {"repr": repr(sharding)}
        return cls(role=role, shape=tuple(int(x) for x in shape), dtype=dtype, axes=axes, sharding=sharding_info, **kw)

    # -- derived ---------------------------------------------------------------
    @property
    def rank(self) -> int:
        return len(self.shape)

    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    def element_bytes(self, overrides: dict | None = None) -> int:
        return storage_bytes(self.dtype, overrides)

    def logical_bytes(self, overrides: dict | None = None) -> int:
        """bytes(X) = prod(shape(X)) * storage_bytes(dtype(X))  (spec 6.1)."""
        return self.numel() * self.element_bytes(overrides)

    def allocated_bytes(self, overrides: dict | None = None) -> int:
        shape = self.capacity_shape or self.shape
        n = 1
        for d in shape:
            n *= d
        return n * self.element_bytes(overrides)

    def axis_extent(self, axis_name: str, default_axes: tuple[str, ...]) -> int:
        axes = self.axes or default_axes
        if axis_name not in axes:
            raise MetadataError(f"{self.role}: axis {axis_name!r} not among {axes}")
        return self.shape[axes.index(axis_name)]

    def to_dict(self) -> dict:
        d = {"shape": list(self.shape), "dtype": self.dtype}
        if self.axes:
            d["axes"] = list(self.axes)
        for k in ("strides", "capacity_shape"):
            v = getattr(self, k)
            if v is not None:
                d[k] = list(v)
        for k in ("storage_offset", "alias_of", "layout", "quantization", "sharding"):
            v = getattr(self, k)
            if v not in (None, 0):
                d[k] = v
        if self.extra:
            d.update(self.extra)
        return d


def _is_trivial_sharding(sharding) -> bool:
    # Without importing jax we can only recognise the obvious "fully replicated / single device" repr forms.
    text = repr(sharding)
    return "SingleDeviceSharding" in text or text in ("None", "")


def parse_tensor_metadata(source: Any) -> dict[str, TensorMeta]:
    """Turn a JSON path / dict / mapping of role -> (dict | array-like | TensorMeta) into TensorMeta objects.

    A bare list of tensors is rejected: without role labels there is no basis for
    the seven-input semantics (spec 1.2).  Configuration strings are never executed.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise MetadataError(f"metadata file not found: {path}")
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict) and "tensors" in data:
            data = data["tensors"]
        source = data
    if isinstance(source, Mapping) and "tensors" in source and isinstance(source["tensors"], Mapping):
        source = source["tensors"]
    if isinstance(source, (list, tuple)):
        raise MetadataError(
            "tensor metadata is an unlabeled list; the seven-input latent adapter needs role labels "
            "(q_latent, kv_latent, q_pe, k_pe, w_q_nope, w_k_nope, w_v)")
    if not isinstance(source, Mapping):
        raise MetadataError(f"unsupported metadata container {type(source).__name__}")
    out: dict[str, TensorMeta] = {}
    for role, entry in source.items():
        role = str(role)
        if role.startswith("_"):
            continue  # comment / annotation keys are not tensors
        if isinstance(entry, TensorMeta):
            out[role] = entry
        elif isinstance(entry, Mapping):
            out[role] = TensorMeta.from_dict(role, entry)
        elif hasattr(entry, "shape") and hasattr(entry, "dtype"):
            out[role] = TensorMeta.from_array_like(role, entry)
        else:
            raise MetadataError(f"{role}: cannot describe metadata entry of type {type(entry).__name__}")
    return out
