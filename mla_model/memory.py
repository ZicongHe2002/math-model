"""Hierarchical byte model (spec 6, 7, 8): ledgers kept separate.

  interface bytes         logical sizes of the seven inputs / outputs (per-tensor dtype)
  HBM materialization     conditional intermediates (expanded K/V, Q^n, Q~, Z): write and read traffic
  transfer requests       sum_r m_r n_r per path (hbm_to_vmem, vmem_to_hbm, vmem_to_vreg, vreg_to_vmem)
  local objects           symbolic sizes of every tile-level object (7.1) with a storage-level assignment
  live storage            per-stage, per-level sums over alias-deduplicated live objects (7.3)
  layout coverage         LayoutBytes(m, n) = s L_s L_l ceil(m/L_s) ceil(n/L_l) with profile tiles (7.2)
  spill                   definitions only (peak / traffic / exposed time) — values unknown without evidence (8.1)
"""
from __future__ import annotations

from functools import lru_cache

import sympy as sp

from .adapters import TaskParameters
from .errors import MetadataError
from .algorithms import AlgorithmGraph, LocalObject, LEVELS, stage_scenario_at
from .symbolic import Sym, ceil, max_expr

_ROLE_EXPR = {
    "q_latent": Sym.B * Sym.S_q * Sym.R_q * Sym.s_Cq,
    "kv_latent": Sym.B * Sym.S_k * Sym.R_k * Sym.s_Ck,
    "q_pe": Sym.B * Sym.H * Sym.S_q * Sym.D_r * Sym.s_Qr,
    "k_pe": Sym.B * Sym.S_k * Sym.D_r * Sym.s_Kr,
    "w_q_nope": Sym.H * Sym.R_q * Sym.D_n * Sym.s_Wq,
    "w_k_nope": Sym.H * Sym.R_k * Sym.D_n * Sym.s_Wk,
    "w_v": Sym.H * Sym.R_k * Sym.D_v * Sym.s_Wv,
}


def interface_expressions() -> dict[str, sp.Basic]:
    """Symbolic logical bytes per role plus totals (each tensor with its own storage width)."""
    out = dict(_ROLE_EXPR)
    out["M_in"] = sp.Add(*_ROLE_EXPR.values())
    out["M_O"] = Sym.s_O * Sym.B * Sym.H * Sym.S_q * Sym.D_v
    out["M_LSE"] = Sym.s_LSE * Sym.B * Sym.H * Sym.S_q
    return out


def _stride_span_exceeds_shape(meta) -> bool:
    """Whether the declared strides / offset span more elements than the shape itself occupies.

    A contiguous stride tuple describes exactly prod(shape) elements, so declaring it truthfully must not make
    a previously bound allocation figure incomplete.  Only a layout that actually spans more is flagged.
    """
    if meta.strides is None and not meta.storage_offset:
        return False
    shape = tuple(meta.capacity_shape or meta.shape)
    if any(int(d) == 0 for d in shape):
        return False                      # no elements, no address range (the D_r = 0 positional tensors)
    if meta.strides is None:
        return bool(meta.storage_offset)
    if len(meta.strides) != len(shape):
        return True
    span = int(meta.storage_offset or 0) + 1 + sum(int(st) * (int(d) - 1) for st, d in zip(meta.strides, shape) if d)
    elements = 1
    for d in shape:
        elements *= int(d)
    return span > elements


def interface_ledger(task: TaskParameters | None):
    """Per-tensor logical and allocated bytes from the actual metadata (aliases deduplicated)."""
    if task is None:
        return None
    rows = []
    seen_storage = set()
    total_logical = 0
    total_alloc_unique = 0
    ov = task.dtype_overrides
    # an alias group occupies one allocation: charge it its largest member, so the total does not
    # depend on the order the roles happen to appear in the metadata
    group_alloc: dict[str, int] = {}
    group_owner: dict[str, str] = {}
    for role, meta in task.tensors.items():
        key = meta.alias_of or role
        alloc = meta.allocated_bytes(ov)
        if alloc > group_alloc.get(key, -1):
            group_alloc[key] = alloc
            group_owner[key] = role
    strided = []
    for role, meta in task.tensors.items():
        logical = meta.logical_bytes(ov)
        alloc = meta.allocated_bytes(ov)
        key = meta.alias_of or role
        charged = group_owner[key] == role
        # spec 6.1: "actual shape, stride, and aliasing affect the allocation ledger".  Aliasing is modelled;
        # a declared stride or storage offset is not, so it is recorded as an unmodelled input rather than
        # quietly ignored — the address range it implies can exceed prod(shape)*s.
        if _stride_span_exceeds_shape(meta):
            strided.append(role)
        rows.append({"role": role, "dtype": meta.dtype, "shape": list(meta.shape), "logical_bytes": logical,
                     "allocated_bytes": alloc, "alias_of": meta.alias_of, "counted_in_allocation": charged,
                     "alias_group": key, "strides": None if meta.strides is None else list(meta.strides),
                     "storage_offset": meta.storage_offset,
                     "allocation_from_strides_modelled": not _stride_span_exceeds_shape(meta)})
        total_logical += logical
    total_alloc_unique = sum(group_alloc.values())
    return {"tensors": rows, "logical_total": total_logical, "allocated_unique_total": total_alloc_unique,
            "roles_with_unmodelled_strides": strided}


def transfer_ledger(g: AlgorithmGraph) -> dict[str, dict]:
    """B_e = sum_r m_r n_r per path (spec 6.3) with the set of undeclared dependencies per path."""
    out: dict[str, dict] = {}
    for ev in g.transfers:
        row = out.setdefault(ev.path, {"total": sp.Integer(0), "undeclared": False, "events": [], "missing_fields": []})
        row["total"] = row["total"] + ev.total
        row["undeclared"] = row["undeclared"] or ev.undeclared
        row["events"].append(ev.id)
        if ev.undeclared:
            fields = [ev.conditional_on] if ev.conditional_on else []
            if "vreg" in ev.path:
                fields.append("register_schedule_evidence")      # a streaming event is always a scenario without evidence
            elif not fields:
                fields.append("transfer conditions")
            for field in fields:
                if field not in row["missing_fields"]:
                    row["missing_fields"].append(field)
    for row in out.values():
        row["total"] = sp.expand(row["total"])
    return out


def materialization_ledger(g: AlgorithmGraph):
    rows = []
    for m in g.materializations:
        rows.append({"id": m.id, "tensor": m.tensor, "bytes": m.bytes, "write_traffic": sp.expand(m.write_traffic),
                     "read_traffic": sp.expand(m.read_traffic), "traffic": sp.expand(m.traffic),
                     "conditional_on": m.conditional_on, "enabled": m.enabled})
    return rows


def active_locals(g: AlgorithmGraph, include_undeclared: bool = True, level: str | None = None) -> list[LocalObject]:
    """Objects that exist under the declared strategy (optionally one storage level); undeclared included when asked."""
    out = []
    for obj in g.locals:
        if obj.exists is False:
            continue
        if obj.exists is None and not include_undeclared:
            continue
        if level is not None and obj.level != level:
            continue
        out.append(obj)
    return out


def live_set(g: AlgorithmGraph, stage: str, include_undeclared: bool = True, level: str | None = None):
    """Sum of sizes of objects live at ``stage`` (one storage level or all), counting each alias group once.

    Returns (bytes_expr, contributing_object_ids, undeclared_field_names).
    """
    groups: dict[str, list[LocalObject]] = {}
    singles: list[LocalObject] = []
    undeclared: list[str] = []
    for obj in active_locals(g, include_undeclared, level):
        if stage not in obj.live_stages:
            continue
        if obj.exists is None and obj.conditional_on and obj.conditional_on not in undeclared:
            undeclared.append(obj.conditional_on)
        for scen in (obj.size_scenario, stage_scenario_at(obj, stage)):
            # an undeclared field that fixed this object's size, or its presence AT THIS CUT, shaped the set
            if scen and scen not in undeclared:
                undeclared.append(scen)
        if obj.alias_group:
            groups.setdefault(obj.alias_group, []).append(obj)
        else:
            singles.append(obj)
    total = sp.Integer(0)
    ids = []
    for obj in singles:
        total += obj.size
        ids.append(obj.id)
    all_members = {}
    for obj in active_locals(g, include_undeclared, level):
        if obj.alias_group:
            all_members.setdefault(obj.alias_group, []).append(obj)
    for grp, members in groups.items():
        # one merged allocation is sized by the largest member of the whole group, not by whichever members
        # happen to be live at this cut: an allocation does not shrink when one of its tenants goes out of use
        whole = all_members.get(grp, members)
        total += max_expr(tuple(m.size for m in whole))
        ids.append(f"alias[{grp}]:" + "|".join(m.id for m in members)
                   + ("" if len(whole) == len(members) else " (sized by the whole group: "
                      + "|".join(m.id for m in whole) + ")"))
    if not total.has(sp.Max):
        total = sp.expand(total)
    return total, ids, undeclared


def stage_live_sets(g: AlgorithmGraph, include_undeclared=True, level: str | None = None):
    return {st: live_set(g, st, include_undeclared, level) for st in g.stages}


def peak_live(g: AlgorithmGraph, include_undeclared=True, level: str | None = None, sets=None) -> sp.Basic:
    """M_{c,peak} = max_t sum_{a in A_c(t)} M_a over the source-level stage cuts for level c (or all levels).

    ``sets`` accepts an already computed ``stage_live_sets`` so a caller that also reports the
    per-stage live sets does not build them twice.  Structurally identical cuts are collapsed before
    the Max (which deduplicates anyway) to save its pairwise ordering decisions.
    """
    if sets is None:
        sets = stage_live_sets(g, include_undeclared, level)
    return max_expr(tuple(dict.fromkeys(v[0] for v in sets.values())))


LAYOUT_KINDS = ("vector_tiled", "compact", "replicated")


@lru_cache(maxsize=1)
def _known_local_ids_by_variant() -> dict:
    """Every local-object id each variant can produce under some declaration.

    A layout legal for one variant is legal to declare; an id this variant produces under another declaration
    (the positional branch with both branches live, the projection-pass tile with expanded K/V materialized)
    is "declared away" here, which the coverage rows word differently from an id another variant owns.
    """
    from .algorithms import ALGORITHMS, build_graph
    from .semantics import SemanticConfig
    from .implementation import ExecutionStrategy
    from .numerics import NumericPolicy
    sem, num = SemanticConfig(), NumericPolicy()
    by_variant = {}
    # the local-object set changes with the declarations that toggle objects into existence
    for a in ALGORITHMS:
        ids = set()
        for strat in (ExecutionStrategy(), ExecutionStrategy(materialize={"expanded_kv": True}),
                      ExecutionStrategy(materialize={"expanded_kv": False}), ExecutionStrategy(b_rq=1, b_rk=1),
                      ExecutionStrategy(scratch_bytes=1), ExecutionStrategy(both_qk_branches_live=True)):
            ids |= {o.id for o in build_graph(a, sem, strat, num).locals}
        by_variant[a] = frozenset(ids)
    return by_variant


KNOWN_LOCAL_IDS_BY_VARIANT = _known_local_ids_by_variant()
KNOWN_LOCAL_IDS = frozenset().union(*KNOWN_LOCAL_IDS_BY_VARIANT.values())
# "raw_bytes" is derived, never declared: it marks an object declared as a byte count rather than an element array


def layout_bytes(m, n, s, L_s=Sym.L_s, L_l=Sym.L_l):
    """LayoutBytes for a vector-tiled 2-D object: s L_s L_l ceil(m/L_s) ceil(n/L_l)  (spec 7.2)."""
    return s * L_s * L_l * ceil(m / L_s) * ceil(n / L_l)


def layout_coverage(g: AlgorithmGraph, layout_kinds: dict, tiles_by_symbol: dict):
    """Per-object layout coverage under declared layout kinds.

    layout_kinds: object id -> 'vector_tiled' | 'compact' | 'replicated'; undeclared -> vector_tiled scenario, flagged.
    tiles_by_symbol: bytes-symbol name -> (L_s, L_l) or None (unknown -> symbolic L_s, L_l)
    """
    rows = []
    unknown_kinds = sorted({k for k in layout_kinds.values() if k not in LAYOUT_KINDS})
    if unknown_kinds:
        raise MetadataError(f"unknown layout kind(s) {unknown_kinds}; expected one of {sorted(LAYOUT_KINDS)}")
    # a layout declared for an object this variant does not have would otherwise be dropped in silence, and the
    # caller would read the resulting vector-tiled scenario as their declared compact/replicated layout
    ids = {o.id for o in g.locals}
    stray = sorted(set(layout_kinds) - ids - KNOWN_LOCAL_IDS)
    if stray:
        raise MetadataError(f"layout declared for object id(s) {stray} that no variant contains; "
                            f"known ids in this ({g.variant}) graph: {sorted(ids)}")
    # an id another variant owns, or one this configuration declares away, is a legal declaration with
    # nothing to apply to; both are reported rather than dropped
    built = {o.id for o in active_locals(g)}
    inactive = sorted(set(layout_kinds) - built)
    # an id this variant produces under some declaration (present in the graph with exists=False, or absent
    # because the declaration removed it, e.g. both_qk_branches_live=False or D_r = 0) is declared away here
    this_variant_ids = ids | set(KNOWN_LOCAL_IDS_BY_VARIANT.get(g.variant, frozenset()))
    declared_away = sorted(set(layout_kinds) & (this_variant_ids - built))
    for oid in inactive:
        rows.append({"object": oid, "kind": layout_kinds[oid], "declared": True, "logical": None, "coverage": None,
                     "tile": None, "level": None, "not_in_this_variant": True,
                     "reason": ("this variant does not build the object under the declared configuration"
                                if oid in declared_away else "no variant builds this object here; another variant does")})
    for obj in active_locals(g):
        kind = layout_kinds.get(obj.id)
        declared = kind is not None
        if obj.is_raw_bytes:
            # spec 7.2's layout function maps an [m, n] element array through the profile tile; a declared raw
            # byte count is not such an array, so tiling it would invent bytes that the declaration rules out
            # no layout choice applies, so this is not an undeclared-layout scenario
            rows.append({"object": obj.id, "kind": "raw_bytes", "declared": True, "logical": obj.size,
                         "coverage": obj.size, "tile": None, "level": obj.level})
            continue
        kind = kind or "vector_tiled"        # scenario when undeclared; the metric carries the missing field
        m, n = obj.shape
        tile = tiles_by_symbol.get(obj.bytes_symbol.name)
        if kind == "compact":
            expr = obj.size
        elif kind == "replicated":
            expr = None  # replicated row states: depends on replication factor; unknown without evidence
        else:
            Ls, Ll = (sp.Integer(tile[0]), sp.Integer(tile[1])) if tile else (Sym.L_s, Sym.L_l)
            expr = layout_bytes(m, n, obj.bytes_symbol, Ls, Ll) * obj.buffers
        rows.append({"object": obj.id, "kind": kind, "declared": declared, "logical": obj.size, "coverage": expr, "tile": tile,
                     "level": obj.level})
    return rows


def spill_definitions():
    """Distinct spill / extra-movement quantities (spec 8); values remain unknown without compile/measurement evidence."""
    return {
        # spec 8.1: static instructions, dynamic bytes, allocation peak and exposed time are four different,
        # non-interchangeable quantities; all four are named here and none is inferred from another
        "N_spill_instructions": ("|{spill/fill instructions in the compiled program}|", "instructions",
                                 "static count of spill and fill instructions (not their dynamic execution count, not bytes)",
                                 ["compile_evidence"]),
        "M_spill_peak": ("max_t sum_{a in A_spill(t)} M_a", "byte", "peak spill backing allocation", ["compile_or_measurement_evidence"]),
        "B_spill_fill": ("sum_r bytes(r) * executions(r)", "byte", "dynamic spill/fill traffic", ["compile_or_measurement_evidence"]),
        "dT_spill": ("T_schedule(G_with_transfers) - T_schedule(G_comparison)", "s", "exposed latency vs a legal comparison graph",
                     ["compile_or_measurement_evidence", "declared comparison graph"]),
        "B_extra_opt": ("min_{p in Omega(theta, a, eta)} B_extra(p)", "byte",
                        "minimum extra movement over legal schedules/tilings/recomputations (spec 8.3); solvable only for a declared small graph, machine model and objective",
                        ["graph", "machine_model", "allowed_transformations", "objective"]),
        # spec 8.3: when speed is the objective these two formulations are the useful ones; they are named
        # here so a caller can see that the byte-minimising problem is not the same optimisation
        "T_opt": ("min_{p in Omega(theta, a, eta)} T(p)", "s",
                  "minimum time over the legal set (spec 8.3, speed objective); needs a solved schedule, not a bound",
                  ["graph", "machine_model", "allowed_transformations", "schedule_model"]),
        "B_extra_opt_under_time_budget": ("min_{p in Omega} B_extra(p) s.t. T(p) <= tau", "byte",
                                          "minimum extra movement subject to a declared time budget tau (spec 8.3)",
                                          ["graph", "machine_model", "allowed_transformations", "schedule_model", "time_budget_tau"]),
    }


def capacity_constraint(M_live, R):
    """max(0, M_live(t) - R): data at the cut that cannot stay in the level (not equal to spill traffic).

    Structural, so memoized: the same (live set, budget symbol) pairs recur on every analysis, and Max
    pays a pairwise ordering decision over the whole polynomial each time.
    """
    return max_expr((sp.Integer(0), M_live - R))
