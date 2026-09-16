"""Load an MLA workload YAML into the model's explicit configuration bundle.

``shape.seq_len`` declares both sequence lengths. Non-square workloads instead
declare ``seq_len_q`` and ``seq_len_kv`` (``seq_len_k`` is also accepted). All
aliases in ``SHAPE_FIELDS`` are equivalent declarations: supplying different
values for aliases, including a common ``seq_len``, is an error. Shapes declare
both allocation and active extents; an explicit ``semantics.active_lengths``
override can select a smaller active region.

``dtype`` is a storage dtype string or a role-to-dtype mapping, optionally with
``default``. Role names accept the seven-input adapter's aliases. Storage dtypes
never imply compute or accumulation policy. ``tiling`` declares execution tile
sizes, but does not declare scheduled extents. ``tolerance`` accepts ``atol`` and
``rtol`` and becomes ``numerics.acceptance``. ``seed`` is retained for provenance
only: loading a workload neither allocates tensor values nor runs a kernel.

Optional ``semantics``, ``implementation`` and ``numerics`` mappings override
the translated fields; mapping-valued fields merge by key. Undeclared mask,
projection scope, outputs, numerical policy and execution policies stay absent.
Only the single-device, independently scheduled batch model is supported.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml

from .adapters import ROLE_ALIASES, ROLE_AXES, SevenInputLatentAdapter
from .algorithms import ALGORITHMS
from .dtypes import normalize_dtype
from .errors import CapabilityError, MetadataError
from .implementation import ExecutionStrategy
from .metadata import parse_tensor_metadata
from .numerics import NumericPolicy
from .semantics import SemanticConfig


SHAPE_FIELDS = {
    "B": ("batch",),
    "H": ("num_heads",),
    "Sq": ("seq_len_q",),
    "Sk": ("seq_len_kv", "seq_len_k"),
    "Rq": ("q_latent_dim",),
    "Rk": ("kv_latent_dim",),
    "Dn": ("qk_nope_head_dim",),
    "Dr": ("qk_rope_head_dim",),
    "Dv": ("value_head_dim",),
}
TILE_FIELDS = {
    "b_q": ("block_q", "b_q"),
    "b_k": ("block_kv", "b_k"),
    "b_rq": ("block_rq", "b_rq"),
    "b_rk": ("block_rkv", "b_rk"),
}
_TOP_FIELDS = {
    "name", "kernel", "shape", "dtype", "seed", "tiling", "devices",
    "tolerance", "semantics", "implementation", "numerics", "algorithm",
}
_KERNELS = ("mla_fwd", "mla")


def _mapping(value: Any, field: str) -> dict:
    if not isinstance(value, Mapping):
        raise MetadataError(f"workload.{field} must be a mapping, got {type(value).__name__}")
    if any(not isinstance(k, str) for k in value):
        raise MetadataError(f"workload.{field} keys must be strings")
    return deepcopy(dict(value))


def _known(mapping: dict, allowed, field: str) -> None:
    unknown = sorted(set(mapping) - set(allowed))
    if unknown:
        raise MetadataError(f"workload.{field} has unknown fields {unknown}; expected {sorted(allowed)}")


def _integer(value: Any, field: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "non-negative" if minimum == 0 else "positive"
        raise MetadataError(f"workload.{field} must be a {qualifier} integer, got {value!r}")
    return value


def _aliases(mapping: dict, names: tuple[str, ...], field: str, *, minimum: int = 1,
             required: bool = False) -> tuple[int | None, list[str]]:
    declared = [(name, _integer(mapping[name], f"{field}.{name}", minimum=minimum))
                for name in names if name in mapping]
    if not declared:
        if required:
            raise MetadataError(f"workload.{field} is missing {names[0]!r} (accepted aliases: {list(names)})")
        return None, []
    if len({value for _, value in declared}) != 1:
        detail = ", ".join(f"{field}.{name}={value}" for name, value in declared)
        raise MetadataError(f"conflicting workload declarations: {detail}")
    return declared[0][1], [f"{field}.{name}" for name, _ in declared]


def _merge(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge(dict(merged[key]), dict(value))
        else:
            merged[key] = deepcopy(value)
    return merged


def _dtype(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MetadataError(f"workload.{field} must be a non-empty storage dtype string, got {value!r}")
    return normalize_dtype(value)


def load_workload(source: str | Path | Mapping) -> dict:
    """Return ``tensors/semantics/implementation/numerics`` and provenance.

    A path is parsed with ``yaml.safe_load``; a mapping uses the same schema.
    Returned data is independent of the caller's dictionaries. ``algorithm`` is
    included only when declared; ``mla`` names the family for the workflow to
    expand into its algorithm scenarios. ``_workload`` contains the declarations,
    their normalized dimensions and field mappings; it is not model configuration.
    Unknown fields and conflicting aliases fail explicitly.
    """
    source_path = None
    if isinstance(source, (str, Path)):
        path = Path(source).expanduser()
        source_path = str(path.resolve())
        try:
            with path.open("r", encoding="utf-8") as handle:
                source = yaml.safe_load(handle)
        except OSError as exc:
            raise MetadataError(f"cannot read workload YAML {path}: {exc}") from exc
        except yaml.YAMLError as exc:
            raise MetadataError(f"invalid workload YAML {path}: {exc}") from exc
    data = _mapping(source, "root")
    _known(data, _TOP_FIELDS, "root")
    kernel = data.get("kernel", data.get("name"))
    if not isinstance(kernel, str) or kernel not in _KERNELS:
        raise CapabilityError(f"workload.kernel must identify one of {_KERNELS}, got {kernel!r}")
    if "name" in data and (not isinstance(data["name"], str) or not data["name"].strip()):
        raise MetadataError("workload.name must be a non-empty string")

    if "shape" not in data:
        raise MetadataError("workload is missing the required 'shape' mapping")
    shape = _mapping(data["shape"], "shape")
    _known(shape, {name for names in SHAPE_FIELDS.values() for name in names} | {"seq_len"}, "shape")
    dims = {}
    mappings = {}
    for dim, names in SHAPE_FIELDS.items():
        aliases = names + (("seq_len",) if dim in ("Sq", "Sk") else ())
        dims[dim], origins = _aliases(shape, aliases, "shape", minimum=0 if dim == "Dr" else 1,
                                     required=True)
        mappings[f"dimensions.{dim}"] = origins

    if "dtype" not in data:
        raise MetadataError("workload is missing 'dtype'; every input storage dtype must be declared")
    dtype_decl = data["dtype"]
    role_dtypes = {}
    dtype_origins = {}
    if isinstance(dtype_decl, str):
        common_dtype = _dtype(dtype_decl, "dtype")
        role_dtypes = {role: common_dtype for role in ROLE_AXES}
        dtype_origins = {role: ["dtype"] for role in ROLE_AXES}
    else:
        dtype_decl = _mapping(dtype_decl, "dtype")
        default = _dtype(dtype_decl["default"], "dtype.default") if "default" in dtype_decl else None
        for role, dtype in dtype_decl.items():
            if role == "default":
                continue
            canonical = ROLE_ALIASES.get(role.lower())
            if canonical is None:
                raise MetadataError(f"workload.dtype has unknown role {role!r}; expected seven-input role names or 'default'")
            normalized = _dtype(dtype, f"dtype.{role}")
            if canonical in role_dtypes and role_dtypes[canonical] != normalized:
                raise MetadataError(f"conflicting workload dtype aliases for {canonical!r}: {dtype_origins[canonical]} and dtype.{role}")
            role_dtypes[canonical] = normalized
            dtype_origins.setdefault(canonical, []).append(f"dtype.{role}")
        missing = [role for role in ROLE_AXES if role not in role_dtypes and default is None]
        if missing:
            raise MetadataError(f"workload.dtype is missing storage dtypes for {missing}; declare each role or 'default'")
        for role in ROLE_AXES:
            if role not in role_dtypes:
                role_dtypes[role] = default
                dtype_origins[role] = ["dtype.default"]
    tensors = {}
    for role, axes in ROLE_AXES.items():
        tensors[role] = {"shape": [dims[axis] for axis in axes], "dtype": role_dtypes[role]}
        mappings[f"tensors.{role}.shape"] = [mappings[f"dimensions.{axis}"] for axis in axes]
        mappings[f"tensors.{role}.dtype"] = dtype_origins[role]

    tiling = _mapping(data["tiling"], "tiling") if "tiling" in data else {}
    _known(tiling, {name for names in TILE_FIELDS.values() for name in names} | {"block_b"}, "tiling")
    implementation = {}
    for field, names in TILE_FIELDS.items():
        value, origins = _aliases(tiling, names, "tiling")
        if value is not None:
            implementation[field] = value
            mappings[f"implementation.{field}"] = origins
    if "block_b" in tiling and _integer(tiling["block_b"], "tiling.block_b") != 1:
        raise CapabilityError("workload.tiling.block_b must be 1: the model schedules batches independently")
    devices = _mapping(data["devices"], "devices") if "devices" in data else {}
    _known(devices, {"num_devices"}, "devices")
    if "num_devices" in devices and _integer(devices["num_devices"], "devices.num_devices") != 1:
        raise CapabilityError("workload.devices.num_devices must be 1: multi-device sharding is not modeled")
    if "seed" in data:
        _integer(data["seed"], "seed", minimum=0)

    numerics = {}
    if "tolerance" in data:
        tolerance = _mapping(data["tolerance"], "tolerance")
        _known(tolerance, {"atol", "rtol"}, "tolerance")
        for key, value in tolerance.items():
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0
                    or value != value or value in (float("inf"), float("-inf"))):
                raise MetadataError(f"workload.tolerance.{key} must be a finite non-negative number, got {value!r}")
            mappings[f"numerics.acceptance.{key}"] = [f"tolerance.{key}"]
        numerics["acceptance"] = tolerance
    semantics = {"active_lengths": {"Sq": dims["Sq"], "Sk": dims["Sk"]}}
    for axis in ("Sq", "Sk"):
        mappings[f"semantics.active_lengths.{axis}"] = mappings[f"dimensions.{axis}"]

    sections = {"semantics": semantics, "implementation": implementation, "numerics": numerics}
    classes = {"semantics": SemanticConfig, "implementation": ExecutionStrategy, "numerics": NumericPolicy}
    for section, cls in classes.items():
        override = _mapping(data[section], section) if section in data else {}
        _known(override, cls.__dataclass_fields__, section)
        if "notes" in override and (not isinstance(override["notes"], list)
                                    or any(not isinstance(x, str) for x in override["notes"])):
            raise MetadataError(f"workload.{section}.notes must be a list of strings")
        sections[section] = _merge(sections[section], override)
        validated = cls.from_source(sections[section]).to_dict()
        sections[section] = {key: validated[key] for key in sections[section]}
        for field, value in override.items():
            if isinstance(value, Mapping):
                for child in value:
                    mappings[f"{section}.{field}.{child}"] = [f"{section}.{field}.{child}"]
            else:
                for existing in list(mappings):
                    if existing.startswith(f"{section}.{field}."):
                        del mappings[existing]
                mappings[f"{section}.{field}"] = [f"{section}.{field}"]

    # Reuse the interface's semantic/capability checks, including per-batch lengths,
    # offsets and unsupported packed or quantized dtypes; no numerical policy is
    # inferred while checking storage widths.
    SevenInputLatentAdapter().extract(
        parse_tensor_metadata(tensors), SemanticConfig.from_source(sections["semantics"]),
        NumericPolicy.from_source(sections["numerics"]).dtype_bytes_overrides,
    )
    bundle = {"tensors": tensors, **sections}
    if "algorithm" in data:
        algorithm = data["algorithm"]
        allowed_algorithms = ("mla",) + tuple(ALGORITHMS)
        if algorithm is not None and (not isinstance(algorithm, str) or algorithm not in allowed_algorithms):
            raise MetadataError(f"workload.algorithm must be one of {allowed_algorithms} or null, got {algorithm!r}")
        bundle["algorithm"] = algorithm
        mappings["algorithm"] = ["algorithm"]
    bundle["_workload"] = {
        "schema": "mla_workload/v1", "source": source_path, "source_kind": "yaml" if source_path else "mapping",
        "name": data.get("name", kernel), "kernel": kernel, "shape": shape, "dimensions": dims,
        "requested_algorithm": data.get("algorithm"),
        "dtype": deepcopy(data["dtype"]), "tiling": tiling, "devices": devices,
        "seed": data.get("seed"), "tolerance": deepcopy(data.get("tolerance")),
        "declaration_mappings": mappings, "inferred": [],
        "notes": [
            "Shape lengths declare active and allocation extents; explicit semantic overrides select the active region.",
            "Storage dtype does not declare compute precision; tiling does not declare scheduled extents.",
            "Seed is recorded only; no tensor values or kernel execution are produced.",
        ],
    }
    return deepcopy(bundle)
