"""Metadata adapters: turn role-labelled tensor metadata into task parameters (theta).

Only the seven-input latent interface is implemented.  Its capabilities are
declared explicitly; anything outside them (packed sequences, quantized storage,
sharded tensors, zero-length axes, missing Q latent) raises ``CapabilityError``
rather than being "supported" by relaxing validation (spec 1.1, 1.2).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

from .dtypes import storage_bytes
from .errors import BindingError, CapabilityError, MetadataError, UnsupportedDtypeError
from .metadata import TensorMeta
from .semantics import SemanticConfig

# role -> canonical axis order
ROLE_AXES: dict[str, tuple[str, ...]] = {
    "q_latent": ("B", "Sq", "Rq"),
    "kv_latent": ("B", "Sk", "Rk"),
    "q_pe": ("B", "H", "Sq", "Dr"),
    "k_pe": ("B", "Sk", "Dr"),
    "w_q_nope": ("H", "Rq", "Dn"),
    "w_k_nope": ("H", "Rk", "Dn"),
    "w_v": ("H", "Rk", "Dv"),
}
ROLE_ALIASES: dict[str, str] = {
    "cq": "q_latent", "c_q": "q_latent", "q_latent": "q_latent", "kv_c": "kv_latent",
    "ck": "kv_latent", "c_k": "kv_latent", "kv_latent": "kv_latent", "c_kv": "kv_latent",
    "q_rope": "q_pe", "qr": "q_pe", "q_pe": "q_pe",
    "k_rope": "k_pe", "kr": "k_pe", "k_pe": "k_pe",
    "wq": "w_q_nope", "w_q": "w_q_nope", "w_q_nope": "w_q_nope",
    "wk": "w_k_nope", "w_k": "w_k_nope", "w_k_nope": "w_k_nope",
    "wv": "w_v", "w_v": "w_v",
}
# whitelisted primary bindings (spec 11.3); every other occurrence is a cross-check
PRIMARY_BINDINGS: dict[str, tuple[str, str]] = {
    "B": ("q_latent", "B"), "Sq": ("q_latent", "Sq"), "Rq": ("q_latent", "Rq"),
    "Sk": ("kv_latent", "Sk"), "Rk": ("kv_latent", "Rk"),
    "H": ("w_q_nope", "H"), "Dn": ("w_q_nope", "Dn"),
    "Dr": ("q_pe", "Dr"), "Dv": ("w_v", "Dv"),
}
DIM_TO_SYMBOL = {"B": "B", "H": "H", "Sq": "S_q", "Sk": "S_k", "Rq": "R_q", "Rk": "R_k",
                 "Dn": "D_n", "Dr": "D_r", "Dv": "D_v"}
ROLE_TO_BYTES_SYMBOL = {"q_latent": "s_Cq", "kv_latent": "s_Ck", "q_pe": "s_Qr", "k_pe": "s_Kr",
                        "w_q_nope": "s_Wq", "w_k_nope": "s_Wk", "w_v": "s_Wv"}


@dataclass
class AdapterCapabilities:
    axis_permutation: bool = True        # transposed physical order via axis names
    ragged_lengths: bool = True          # per-batch active lengths
    no_positional_branch: bool = True    # D_r = 0 (q_pe / k_pe absent or zero-width)
    kv_capacity_gt_active: bool = True   # cache capacity larger than consumed keys
    packed_sequences: bool = False
    quantized_storage: bool = False
    sharded_tensors: bool = False
    zero_length_axes: bool = False
    missing_q_latent: bool = False
    aliasing: bool = True                # alias_of metadata is recorded (dedup in ledgers)


@dataclass
class TaskParameters:
    """theta for the current invocation.  Extents are separated by meaning (spec 1.3)."""
    adapter: str
    dims: dict[str, int]                       # logical extents from shapes: B,H,Sq,Sk,Rq,Rk,Dn,Dr,Dv
    capacity: dict[str, int]                   # allocated extents (Sq, Sk) — may exceed active
    active: dict[str, Any]                     # active extents: Sq, Sk as int (uniform) or list (ragged)
    offset: int | None                         # Delta (uniform) ; None if unknown
    ragged: dict | None                        # {'Sq': [...], 'Sk': [...], 'Delta': [...]} when ragged
    dtypes: dict[str, str]                     # role -> dtype
    dtype_bytes: dict[str, int]                # symbol (s_Cq, ...) -> storage bytes
    provenance: dict[str, str]                 # dim -> source field
    cross_checks: list[str]                    # satisfied equality constraints
    tensors: dict[str, TensorMeta]
    aliases: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    dtype_overrides: dict | None = None

    def symbol_bindings(self) -> dict[str, Any]:
        """Bindings for theta symbols.  Sq / Sk bind to *active* extents (mathematical work); capacity is separate."""
        b = {DIM_TO_SYMBOL[k]: v for k, v in self.dims.items() if k not in ("Sq", "Sk")}
        if not self.ragged:
            b["S_q"] = self.active["Sq"]
            b["S_k"] = self.active["Sk"]
            if self.offset is not None:
                b["Delta"] = self.offset
        b.update(self.dtype_bytes)
        return b

    def capacity_bindings_for(self, role: str) -> dict[str, Any]:
        """Capacity bindings as they apply to ONE tensor (spec 1.3: allocation is per tensor).

        A role that declares no ``capacity_shape`` is allocated at its own shape extent; charging it another
        role's declared capacity would report bytes the metadata does not claim exist.
        """
        b = self.capacity_bindings()
        meta = self.tensors.get(role)
        axes = None if meta is None else (meta.axes or ROLE_AXES.get(role))
        if meta is None or axes is None:
            return b
        extents = meta.capacity_shape or meta.shape
        for i, key in enumerate(axes):
            sym = DIM_TO_SYMBOL.get(key)
            if sym is not None and i < len(extents):
                b[sym] = extents[i]
        return b

    def capacity_bindings(self) -> dict[str, Any]:
        """Bindings at allocation extents: S_q / S_k are the declared capacities (capacity_shape honoured)."""
        b = {DIM_TO_SYMBOL[k]: v for k, v in self.dims.items()}
        b["S_q"] = self.capacity["Sq"]
        b["S_k"] = self.capacity["Sk"]
        b.update(self.dtype_bytes)
        return b

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tensors"] = {k: v.to_dict() for k, v in self.tensors.items()}
        return d


class SevenInputLatentAdapter:
    name = "seven_input_latent"
    capabilities = AdapterCapabilities()

    def describe(self) -> dict:
        return {
            "adapter": self.name,
            "roles": {r: list(a) for r, a in ROLE_AXES.items()},
            "output": {"O": ["B", "H", "Sq", "Dv"], "LSE": ["B", "H", "Sq"]},
            "primary_bindings": {d: f"tensor.{r}.axis[{a}]" for d, (r, a) in PRIMARY_BINDINGS.items()},
            "capabilities": asdict(self.capabilities),
            "role_aliases": ROLE_ALIASES,
            "notes": [
                "Shapes are role/axis contracts; physical order may differ when 'axes' names are given.",
                "Sq/Sk from shapes are capacities; active lengths come from semantics.active_lengths.",
                "Projection of Q/K/V from latents is inside this entry scope unless projection_scope says otherwise.",
            ],
        }

    # ------------------------------------------------------------------
    def normalize_roles(self, tensors: dict[str, TensorMeta]) -> dict[str, TensorMeta]:
        out: dict[str, TensorMeta] = {}
        unknown = []
        for role, meta in tensors.items():
            canon = ROLE_ALIASES.get(role.lower())
            if canon is None:
                unknown.append(role)
                continue
            if canon in out:
                raise MetadataError(f"role {canon!r} supplied twice ({out[canon].role!r} and {role!r})")
            meta.role = canon if meta.role == role else meta.role
            out[canon] = meta
        if unknown:
            raise MetadataError(
                f"tensors {unknown} are not part of the seven-input latent interface; "
                "an arbitrary tensor list does not carry these semantics — use an explicit adapter")
        return out

    def extract(self, tensors: dict[str, TensorMeta], semantics: SemanticConfig,
                dtype_overrides: dict | None = None) -> TaskParameters:
        tensors = self.normalize_roles(dict(tensors))
        violations: list[str] = []
        notes: list[str] = []
        caps = self.capabilities

        # -- capability gates -------------------------------------------------
        for role, meta in tensors.items():
            if meta.quantization is not None and not caps.quantized_storage:
                raise CapabilityError(f"{role}: quantized storage metadata present; adapter {self.name} does not model quantization")
            if meta.sharding is not None and not caps.sharded_tensors:
                raise CapabilityError(f"{role}: sharded tensor metadata present; adapter {self.name} models single-device tensors only")
            sparse_keys = [k for k in meta.extra if "spars" in k.lower() or k.lower() in ("block_sparse", "nnz", "indices", "indptr")]
            if sparse_keys:
                raise CapabilityError(f"{role}: sparsity metadata {sparse_keys} present; adapter {self.name} models dense tensors only")
            other = sorted(k for k in meta.extra if k not in sparse_keys)
            if other:
                # a key the adapter does not read is not a declaration: say so, so a misspelled
                # 'capacty_shape' cannot pass for a declared capacity
                notes.append(f"{role}: metadata keys {other} are not read by adapter {self.name} and had no effect "
                             "(check for a misspelled field)")
        if semantics.segments is not None and not caps.packed_sequences:
            raise CapabilityError(f"packed-sequence segments declared; adapter {self.name} does not model packed sequences")

        required = ["q_latent", "kv_latent", "w_q_nope", "w_k_nope", "w_v"]
        missing = [r for r in required if r not in tensors]
        if "q_latent" in missing and not caps.missing_q_latent:
            raise CapabilityError("q_latent is missing; adapter does not model a missing Q latent")
        if missing:
            raise MetadataError(f"missing required roles: {missing}")
        pe_present = ("q_pe" in tensors, "k_pe" in tensors)
        if pe_present == (False, False):
            if not caps.no_positional_branch:
                raise CapabilityError("positional branch tensors absent")
            notes.append("q_pe and k_pe absent -> D_r = 0 (no positional branch scenario)")
        elif pe_present != (True, True):
            raise MetadataError("q_pe and k_pe must both be present or both absent")

        # -- rank checks -------------------------------------------------------
        for role, meta in tensors.items():
            axes = meta.axes or ROLE_AXES[role]
            if meta.axes is not None:
                if not caps.axis_permutation:
                    raise CapabilityError("axis permutation not supported")
                if sorted(meta.axes) != sorted(ROLE_AXES[role]):
                    violations.append(f"{role}: axis names {meta.axes} do not match role axes {ROLE_AXES[role]}")
                    continue
            if meta.rank != len(ROLE_AXES[role]):
                violations.append(f"{role}: rank {meta.rank} != {len(ROLE_AXES[role])} expected for axes {ROLE_AXES[role]}")
        if violations:
            raise BindingError(violations, "metadata does not match the seven-input latent contract")

        # -- collect every occurrence of every dimension -------------------------
        occurrences: dict[str, list[tuple[str, int]]] = {}
        for role, meta in tensors.items():
            axes = meta.axes or ROLE_AXES[role]
            for axis, extent in zip(axes, meta.shape):
                occurrences.setdefault(axis, []).append((f"tensor.{role}.axis[{axis}]", extent))
        dims: dict[str, int] = {}
        provenance: dict[str, str] = {}
        cross_checks: list[str] = []
        for dim, (prole, paxis) in PRIMARY_BINDINGS.items():
            occ = occurrences.get(dim, [])
            if not occ:
                if dim == "Dr":
                    dims["Dr"] = 0
                    provenance["Dr"] = "absent positional branch"
                    continue
                violations.append(f"dimension {dim} has no source tensor")
                continue
            primary = next(((src, ext) for src, ext in occ if src == f"tensor.{prole}.axis[{paxis}]"), occ[0])
            dims[dim] = primary[1]
            provenance[dim] = primary[0]
            for src, ext in occ:
                if ext != primary[1]:
                    violations.append(f"{dim}: {primary[0]}={primary[1]} but {src}={ext}")
                elif src != primary[0]:
                    cross_checks.append(f"{src} == {primary[0]} == {ext}")
        if violations:
            raise BindingError(violations, "contradictory shapes across the seven inputs")

        # -- zero-length handling -----------------------------------------------
        for dim in ("B", "H", "Sq", "Sk", "Rq", "Rk", "Dn", "Dv"):
            if dims.get(dim, 1) == 0 and not caps.zero_length_axes:
                raise CapabilityError(f"{dim} = 0: zero-length axes are outside the declared adapter capabilities")

        # -- dtypes ----------------------------------------------------------------
        dtypes = {role: meta.dtype for role, meta in tensors.items()}
        dtype_bytes: dict[str, int] = {}
        for role, meta in tensors.items():
            try:
                dtype_bytes[ROLE_TO_BYTES_SYMBOL[role]] = storage_bytes(meta.dtype, dtype_overrides)
            except UnsupportedDtypeError as e:
                if not caps.quantized_storage:
                    raise CapabilityError(str(e)) from e
        if pe_present == (False, False):
            notes.append("s_Qr / s_Kr have no source tensor (D_r = 0); positional-branch bytes are 0")
            dtype_bytes.setdefault("s_Qr", 0)
            dtype_bytes.setdefault("s_Kr", 0)
        # alias_of names a role, so it accepts the same spellings every other role reference does
        for role, meta in tensors.items():
            if meta.alias_of is not None and not isinstance(meta.alias_of, str):
                raise MetadataError(f"{role}: alias_of must name a role (string), got {meta.alias_of!r}")
        aliases = {role: ROLE_ALIASES.get(meta.alias_of.lower(), meta.alias_of)
                   for role, meta in tensors.items() if meta.alias_of}
        # an alias chain is one allocation: resolve every alias to its root, refusing cycles and self-aliases
        for role, target in list(aliases.items()):
            seen = [role]
            while target in aliases:
                if target in seen:
                    raise MetadataError(f"alias_of forms a cycle: {' -> '.join(seen + [target])}")
                seen.append(target)
                target = aliases[target]
            if target == role:
                raise MetadataError(f"{role}: alias_of names itself")
            aliases[role] = target
        for role, target in aliases.items():
            tensors[role].alias_of = target          # keep the ledger's grouping on canonical names
        dangling = sorted(f"{r} -> {a}" for r, a in aliases.items() if a not in tensors)
        if dangling:
            # an alias to a role that is not present would create a phantom allocation group
            raise MetadataError(f"alias_of names roles that were not supplied: {dangling}; known roles: {sorted(tensors)}")

        # -- capacity vs active lengths -------------------------------------------
        capacity = {"Sq": dims["Sq"], "Sk": dims["Sk"]}
        cap_sources: dict[str, list[tuple[str, int]]] = {"Sq": [], "Sk": []}
        for role, meta in tensors.items():
            if meta.capacity_shape:
                axes = meta.axes or ROLE_AXES[role]
                for key in ("Sq", "Sk"):
                    if key in axes:
                        cap_sources[key].append((role, meta.capacity_shape[axes.index(key)]))
        for key, srcs in cap_sources.items():
            vals = {v for _, v in srcs}
            if len(vals) > 1:
                violations.append(f"capacity_shape for {key} differs between roles: {srcs}")
            elif vals:
                capacity[key] = vals.pop()
        if violations:
            raise BindingError(violations, "capacity declarations contradict each other")
        active: dict[str, Any] = {"Sq": dims["Sq"], "Sk": dims["Sk"]}
        ragged = None
        al = {} if semantics.active_lengths is None else semantics.active_lengths
        if not isinstance(al, dict):
            raise BindingError([f"active_lengths must be a mapping with keys 'Sq' and/or 'Sk', got {type(al).__name__}"])
        unknown_al = sorted(set(al) - {"Sq", "Sk"})
        if unknown_al:
            raise BindingError([f"active_lengths has unknown keys {unknown_al}; expected 'Sq' and/or 'Sk'"],
                               "an active length that is not read is an undeclared extent, not a declared one")
        def _length(v, what):
            if isinstance(v, bool) or not isinstance(v, int):
                violations.append(f"{what}={v!r} must be a non-negative integer")
                return None
            if v < 0:
                violations.append(f"{what}={v} is negative")
                return None
            if v == 0 and not caps.zero_length_axes:
                raise CapabilityError(f"{what} = 0: zero active length is outside the declared adapter capabilities")
            return v

        for key in ("Sq", "Sk"):
            if key in al and al[key] is not None:
                val = al[key]
                if isinstance(val, (list, tuple)):
                    if not caps.ragged_lengths:
                        raise CapabilityError("ragged active lengths not supported")
                    if len(val) != dims["B"]:
                        violations.append(f"active_lengths.{key} has {len(val)} entries but B={dims['B']}")
                    checked = [_length(v, f"active_lengths.{key}[{i}]") for i, v in enumerate(val)]
                    for b_idx, v in enumerate(checked):
                        if v is not None and v > capacity[key]:
                            violations.append(f"active_lengths.{key}[{b_idx}]={v} exceeds capacity {capacity[key]}")
                    active[key] = [v for v in checked if v is not None]
                else:
                    v = _length(val, f"active_lengths.{key}")
                    if v is not None and v > capacity[key]:
                        violations.append(f"active_lengths.{key}={val} exceeds capacity {capacity[key]} from {provenance[key]}")
                    active[key] = v if v is not None else dims[key]
            else:
                if key == "Sk" and capacity["Sk"] != dims["Sk"]:
                    notes.append("kv capacity exceeds logical shape and no active Sk given: capacity-execution scenario")
        if violations:
            raise BindingError(violations, "active lengths contradict tensor capacities")

        offset, conflicts = semantics.offset()
        if conflicts:
            raise BindingError(conflicts, "position offset declared inconsistently")
        if offset is not None and (isinstance(offset, bool) or not isinstance(offset, int)):
            raise BindingError([f"position_offset={offset!r} must be an integer"])
        pbo = semantics.per_batch_offsets
        if pbo is not None:
            if isinstance(pbo, (str, bytes)) or not isinstance(pbo, (list, tuple)):
                raise BindingError([f"per_batch_offsets must be a list of B integers, got {pbo!r}"])
            if len(pbo) != dims["B"]:
                raise BindingError([f"per_batch_offsets has {len(pbo)} entries, B={dims['B']}"])
            if any(isinstance(x, bool) or not isinstance(x, int) for x in pbo):
                raise BindingError(["per_batch_offsets entries must be integers"])
            if offset is not None and any(int(x) != offset for x in pbo):
                raise BindingError([f"per_batch_offsets={pbo} conflicts with position_offset={offset}"], "offsets declared inconsistently")
            if not isinstance(active["Sq"], list) and not isinstance(active["Sk"], list):
                if len(set(pbo)) > 1:
                    active["Sq"] = [active["Sq"]] * dims["B"]
                    active["Sk"] = [active["Sk"]] * dims["B"]
                    notes.append("per_batch_offsets differ across batches with uniform lengths: analysed as a per-batch (ragged-offset) task")
                elif offset is None:
                    # one offset repeated for every batch is a uniform offset, not an unknown one
                    offset = int(pbo[0])
                    notes.append(f"per_batch_offsets are uniform ({offset}); analysed as a single declared offset")
        cache = semantics.cache or {}
        if cache.get("merged_weight_reuse") is not None:
            mwr = cache["merged_weight_reuse"]
            if isinstance(mwr, bool) or not isinstance(mwr, int) or mwr < 1:
                raise BindingError([f"cache.merged_weight_reuse={mwr!r} must be a positive integer"])
        if cache.get("cached_tokens") is not None:
            ct = cache["cached_tokens"]
            if isinstance(ct, bool) or not isinstance(ct, int) or ct < 0:
                raise BindingError([f"cache.cached_tokens={ct!r} must be a non-negative integer"])
        if cache.get("new_tokens") is not None:
            nt = cache["new_tokens"]
            if isinstance(nt, bool) or not isinstance(nt, int) or nt < 0:
                raise BindingError([f"cache.new_tokens={nt!r} must be a non-negative integer"])
            sk_vals = active["Sk"] if isinstance(active["Sk"], list) else [active["Sk"]]
            over = [v for v in sk_vals if nt > int(v)]
            if over:
                violations.append(f"cache.new_tokens={nt} exceeds the active S_k={over[0]}")
            ct = cache.get("cached_tokens")
            if ct is not None:
                # both describe the same projected-row count; they must agree rather than one silently winning
                implied = [int(v) - int(ct) for v in sk_vals]
                if any(nt != x for x in implied):
                    violations.append(
                        f"cache.new_tokens={nt} contradicts active S_k - cache.cached_tokens={implied[0] if len(set(implied)) == 1 else implied}; "
                        "declare only one of them, or make them agree")
        if violations:
            raise BindingError(violations, "cache declaration contradicts the task")
        if semantics.projection_scope == "none":
            raise CapabilityError(
                "projection_scope 'none' means Q and expanded K/V are received directly; that is a different entry scope "
                "than the seven-input latent interface (spec 1.2) — use an adapter for expanded inputs instead")
        if semantics.projection_scope == "new_tokens_only":
            cached = (semantics.cache or {}).get("cached_tokens")
            if cached is not None:
                sk_vals = active["Sk"] if isinstance(active["Sk"], list) else [active["Sk"]]
                bad = [v for v in sk_vals if int(cached) > int(v)]
                if bad:
                    violations.append(f"cache.cached_tokens={cached} exceeds active S_k={bad[0]}: projected-row counts would be negative")
                if int(cached) < 0:
                    violations.append("cache.cached_tokens must be non-negative")
        if violations:
            raise BindingError(violations, "cache declaration contradicts active lengths")
        if isinstance(active["Sq"], list) or isinstance(active["Sk"], list):
            B = dims["B"]
            sq = active["Sq"] if isinstance(active["Sq"], list) else [active["Sq"]] * B
            sk = active["Sk"] if isinstance(active["Sk"], list) else [active["Sk"]] * B
            if semantics.per_batch_offsets is not None:
                dl = [int(x) for x in semantics.per_batch_offsets]
            elif offset is not None:
                dl = [offset] * B
            elif semantics.mask == "none":
                dl = [0] * B          # offset irrelevant for the dense mask
            else:
                dl = None
            ragged = {"Sq": sq, "Sk": sk, "Delta": dl}
            if dl is None:
                notes.append("ragged lengths given but no offsets: visibility stays unknown until offsets are declared")
        if semantics.active_lengths is None and (semantics.mask is not None):
            notes.append("no active_lengths declared: shape extents are used as active lengths (declared, not inferred from cache)")

        return TaskParameters(
            adapter=self.name, dims=dims, capacity=capacity, active=active, offset=offset, ragged=ragged,
            dtypes=dtypes, dtype_bytes=dtype_bytes, provenance=provenance, cross_checks=cross_checks,
            tensors=tensors, aliases=aliases, notes=notes, dtype_overrides=dict(dtype_overrides) if dtype_overrides else None,
        )


ADAPTERS = {SevenInputLatentAdapter.name: SevenInputLatentAdapter}


def get_adapter(name: str):
    if name not in ADAPTERS:
        raise MetadataError(f"unknown adapter {name!r}; available: {sorted(ADAPTERS)}")
    return ADAPTERS[name]()
