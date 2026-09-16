"""Storage-width table for tensor dtypes.

Only *storage* bytes live here.  Storage width never implies compute precision,
accumulator width, or throughput; those are read from the numeric policy and the
hardware profile respectively (see spec 13.2).

Sub-byte / packed dtypes are rejected unless the caller supplies an explicit
packing model through ``overrides`` (spec 6.1: do not multiply element counts by
fractional bytes and call it an allocation).
"""
from __future__ import annotations

from .errors import UnsupportedDtypeError

# canonical name -> storage bytes per element
STORAGE_BYTES: dict[str, int] = {
    "float64": 8,
    "float32": 4,
    "bfloat16": 2,
    "float16": 2,
    "float8_e4m3fn": 1,
    "float8_e4m3fnuz": 1,
    "float8_e5m2": 1,
    "float8_e5m2fnuz": 1,
    "float8_e4m3b11fnuz": 1,
    "int64": 8,
    "uint64": 8,
    "int32": 4,
    "uint32": 4,
    "int16": 2,
    "uint16": 2,
    "int8": 1,
    "uint8": 1,
    "bool": 1,
}

# spelling variants -> canonical
ALIASES: dict[str, str] = {
    "f64": "float64", "fp64": "float64", "double": "float64",
    "f32": "float32", "fp32": "float32", "float": "float32",
    "bf16": "bfloat16",
    "f16": "float16", "fp16": "float16", "half": "float16",
    "fp8_e4m3": "float8_e4m3fn", "float8_e4m3": "float8_e4m3fn", "e4m3": "float8_e4m3fn",
    "fp8_e5m2": "float8_e5m2", "e5m2": "float8_e5m2",
    "i64": "int64", "i32": "int32", "i16": "int16", "i8": "int8",
    "u8": "uint8", "u16": "uint16", "u32": "uint32", "u64": "uint64",
    "bool_": "bool",
}

# dtypes that exist but are not whole-byte per element; need an explicit packing model
SUB_BYTE_DTYPES: dict[str, int] = {  # name -> bits
    "int4": 4, "uint4": 4, "float4_e2m1fn": 4, "fp4": 4, "int2": 2, "uint2": 2,
    "float6_e2m3fn": 6, "float6_e3m2fn": 6,
}


def normalize_dtype(name) -> str:
    """Return the canonical dtype name for a dtype-like value (string, numpy/jax dtype, ...)."""
    if name is None:
        raise UnsupportedDtypeError("dtype is missing")
    if isinstance(name, type):                      # e.g. numpy scalar type classes such as np.float32
        text = getattr(name, "__name__", str(name))
    else:
        text = getattr(name, "name", None) or str(name)
    if not isinstance(text, str):
        text = str(text)
    text = text.strip().lower()
    for prefix in ("torch.", "np.", "numpy.", "jnp.", "jax.numpy.", "ml_dtypes."):
        if text.startswith(prefix):
            text = text[len(prefix):]
    text = text.strip("<>'\"")
    if text.startswith("dtype(") and text.endswith(")"):
        text = text[6:-1].strip("'\"")
    return ALIASES.get(text, text)


def storage_bytes(name, overrides: dict | None = None) -> int:
    """Storage bytes per element.

    ``overrides`` maps canonical dtype names to integer bytes and is the only way a
    packed/sub-byte dtype or an unknown dtype can obtain a width; the override is
    then recorded by the caller as an assumption.
    """
    canon = normalize_dtype(name)
    if overrides and canon in overrides:
        value = overrides[canon]
        if not isinstance(value, int) or value <= 0:
            raise UnsupportedDtypeError(f"override for dtype {canon!r} must be a positive int, got {value!r}")
        return value
    if canon in STORAGE_BYTES:
        return STORAGE_BYTES[canon]
    if canon in SUB_BYTE_DTYPES:
        raise UnsupportedDtypeError(
            f"dtype {canon!r} is a {SUB_BYTE_DTYPES[canon]}-bit packed type; the seven-input adapter has no packing model. "
            "Supply an explicit packing model (bytes per packed group, scale/zero-point tensors) or a dtype override."
        )
    raise UnsupportedDtypeError(f"unknown dtype {canon!r}; supply its storage width via dtype overrides")


def is_floating(name) -> bool:
    canon = normalize_dtype(name)
    return canon.startswith("float") or canon == "bfloat16"
