"""Hardware lower bounds and calibrated predictions — different objects (spec 9)."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import sympy as sp

from .algorithms import AlgorithmGraph
from .hardware import HardwareProfile, Quantity
from .symbolic import Sym, substitute, numeric_value, free_symbol_names

# spec 9.2 maxes over every resource class u, so a class the graph counts work for gets its own bound even
# when no profile supplies its throughput: the bound is then unknown and the combination inherits that.
RESOURCE_SYMBOL = {"mxu": Sym.P_mxu, "exp": Sym.P_exp, "vpu": Sym.P_vpu, "reduce": Sym.P_red, "layout": Sym.P_layout}
RESOURCE_PROFILE_KEY = {"exp": "vpu.exp_throughput", "vpu": "vpu.elementwise_throughput", "reduce": "vpu.reduction_throughput",
                        "layout": "vpu.layout_throughput"}
# every path has its own bandwidth symbol and profile key; a missing direction is never borrowed from the other
PATH_SYMBOL = {"hbm_to_vmem": Sym.W_hbm, "vmem_to_hbm": Sym.W_hbm_w, "vmem_to_vreg": Sym.W_vmem, "vreg_to_vmem": Sym.W_vmem_w}
PATH_PROFILE_KEY = {"hbm_to_vmem": "hbm_to_vmem.bandwidth", "vmem_to_hbm": "vmem_to_hbm.bandwidth",
                    "vmem_to_vreg": "vmem_to_vreg.bandwidth", "vreg_to_vmem": "vreg_to_vmem.bandwidth"}


@dataclass
class ResourceBound:
    resource: str
    numerator: str
    denominator: str
    expression: str
    value: float | None
    unit: str = "s"
    scope: str | None = None
    status: str = "symbolic"
    missing: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def resource_lower_bounds(g: AlgorithmGraph, work_by_resource: dict[str, sp.Basic], paths: dict[str, dict],
                          bindings: dict, hardware: HardwareProfile | None, compute_dtype: str | None,
                          overlap_model: str | None, serial_groups=None, work_undeclared=None, vector_undeclared=None,
                          path_undeclared=None) -> dict:
    """T_LB = max(max_u W_u/P_u, max_e B_e/W_e, CP) under the declared overlap model.

    Each numerator is the executed-work expression of this graph scenario; each denominator is a
    profile upper bound with a scope.  Missing throughputs leave that resource's bound unknown and
    the combined bound partial.  With no declared overlap model both combination scenarios are returned.
    """
    bounds: list[ResourceBound] = []
    hw_bind = hardware.bindings(compute_dtype) if hardware else {}
    scopes = set()
    for res, W in work_by_resource.items():
        P = RESOURCE_SYMBOL.get(res)
        if P is None:
            continue
        expr = W / P
        sub = substitute(expr, {**bindings, **hw_bind})
        val = numeric_value(sub)
        q: Quantity | None = None
        if hardware:
            q = hardware.peak_matmul(compute_dtype) if res == "mxu" else hardware.compute.get(RESOURCE_PROFILE_KEY[res])
        notes = []
        missing = free_symbol_names(sub)
        if res == "mxu":
            notes.append("numerator: executed matmul FLOPs of this scenario (C_exec); denominator: matmul peak for the declared compute dtype")
            if compute_dtype is None:
                notes.append("compute dtype undeclared: no peak can be selected")
                missing = sorted(set(missing) | {"matmul_input_dtype"})
            missing = sorted(set(missing) | set(work_undeclared or []))
        else:
            missing = sorted(set(missing) | set(vector_undeclared or []))
        if q and q.scope:
            scopes.add(q.scope)
        if val is None:
            status = "unknown" if missing else "symbolic"
        else:
            status = "partial" if missing else "bound"
        bounds.append(ResourceBound(res, str(W), str(P), str(expr), val, scope=(q.scope if q else None), status=status,
                                    missing=missing, notes=notes))
    for path, row in paths.items():
        Wsym = PATH_SYMBOL.get(path)
        if Wsym is None:
            continue
        Bexpr = row["total"]
        expr = Bexpr / Wsym
        sub = substitute(expr, {**bindings, **hw_bind})
        val = numeric_value(sub)
        q = hardware.paths.get(PATH_PROFILE_KEY[path]) if hardware else None
        if q and q.scope:
            scopes.add(q.scope)
        # B_e/W_e is no firmer than B_e: the bound inherits everything the byte total rests on (the extents
        # the events were counted over, the materialization flags, the loop nest)
        missing = sorted(set(free_symbol_names(sub)) | set(path_undeclared or []))
        notes = ["conditional transfer requests of this scenario / path bandwidth upper bound; not an absolute limit for all algorithms"]
        if row.get("undeclared"):
            notes.append("path traffic depends on undeclared fields or on register-schedule evidence (scenario)")
            missing = sorted(set(missing) | set(row.get("missing_fields") or ["register_schedule_evidence" if "vreg" in path else "transfer conditions"]))
        if val is None:
            status = "unknown" if missing else "symbolic"
        else:
            status = "partial" if missing else "bound"
        bounds.append(ResourceBound(f"path:{path}", str(Bexpr), str(Wsym), str(expr), val, scope=(q.scope if q else None), status=status,
                                    missing=missing, notes=notes))
    bounds.append(ResourceBound("critical_path", "CP", "1", "CP", numeric_value(substitute(Sym.CP, bindings)), status="unknown",
                                missing=["CP"], notes=["dependency-graph path length; unknown without a declared schedule model"]))
    known = [b.value for b in bounds if b.value is not None]
    all_known = all(b.value is not None for b in bounds if b.resource != "critical_path")
    scope_note = []
    if len(scopes) > 1:
        scope_note.append(f"profile quantities have mixed scopes {sorted(scopes)}; a combined bound mixes scopes and must not be read as a single-device time")

    def combine(model: str, groups=None):
        if model == "full_overlap_max":
            return (max(known) if known else None), "max(" + ", ".join(b.expression for b in bounds) + ")"
        groups = groups or [[b.resource] for b in bounds if b.resource != "critical_path"]
        total = 0.0
        parts = []
        counted_any = False
        for grp in groups:
            vals = [b.value for b in bounds if b.resource in grp and b.value is not None]
            if not vals:
                # every term is non-negative, so omitting an unknown group keeps a valid lower bound;
                # the metric stays partial and names what is absent rather than reporting nothing at all
                parts.append("[" + ", ".join(grp) + ": unknown, omitted]")
                continue
            counted_any = True
            total += max(vals)
            absent = [r for r in grp if not any(b.resource == r and b.value is not None for b in bounds)]
            parts.append("max(" + ", ".join(r for r in grp if r not in absent) + ")"
                         + (f" [{', '.join(absent)}: unknown, omitted]" if absent else ""))
        # A resource the declared groups do not name is not thereby declared free: nothing says it is serial
        # with the groups, but T >= its own bound still holds, so the combination keeps it via a max.  Dropping
        # it would publish a sum that is smaller than a bound the model already knows (spec 9.2).  This runs
        # before any early return, so an unnamed resource still bounds a combination whose groups are unknown.
        named = {r for grp in groups for r in grp}
        loose = [b for b in bounds if b.resource not in named and b.resource != "critical_path" and b.value is not None]
        if not counted_any and not loose:
            return None, "unknown"
        if loose:
            total = max(total, max(b.value for b in loose))
            inner = " + ".join(parts) if parts else "no declared group has a value"
            return total, "max(" + inner + ", " + ", ".join(sorted(b.resource for b in loose)) + ")"
        return total, " + ".join(parts)

    # the combined bound inherits every unmet requirement of the terms it combines: a term that HAS a value
    # but was computed under an undeclared field makes the combination partial in exactly the same way.
    # CP is always among them: the combination omits the critical-path term of spec 9.2, so it is a lower
    # bound on the lower bound and never 'bound' until a schedule model supplies CP.
    combined_missing = sorted({f for b in bounds for f in b.missing})
    all_known = all_known and not combined_missing
    scenarios = {}
    if overlap_model is None:
        for model in ("full_overlap_max", "serial_stage_sum"):
            v, e = combine(model, serial_groups)
            scenarios[model] = {"value": v, "expression": e}
        combined, combined_expr = None, "scenario-dependent (overlap_model undeclared)"
        status = "partial" if known else "symbolic"
        overlap_note = ["overlap_model undeclared: both the fully-overlapped max and the fully-serial sum are reported as scenarios"]
        combined_missing = sorted(set(combined_missing) | {"overlap_model"})
    else:
        combined, combined_expr = combine(overlap_model, serial_groups)
        status = "bound" if (combined is not None and all_known) else ("partial" if known else "symbolic")
        overlap_note = []
    uncovered = []
    if serial_groups and overlap_model in (None, "serial_stage_sum"):
        named = {r for grp in serial_groups for r in grp}
        uncovered = sorted(b.resource for b in bounds if b.resource not in named and b.resource != "critical_path")
    # declared inputs with no consumer in this graph are named, never absorbed: a serial class with no bound row
    # and a path bandwidth with no modelled transfer events change nothing, and the report says so
    have = {b.resource for b in bounds}
    inert_classes = sorted({r for grp in (serial_groups or []) for r in grp if r not in have})
    unused_paths = sorted(f"{p} ({key})" for p, key in PATH_PROFILE_KEY.items()
                          if hardware is not None and key in hardware.paths and p not in paths)
    consumer_notes = (([f"serial_stage_groups names resource class(es) {inert_classes} for which this graph has no work "
                        f"and no transfer events: they contribute nothing to the combination"] if inert_classes else [])
                      + ([f"bandwidth declared for path(s) {unused_paths} has no consumer: this graph models no transfer "
                          f"events on that path, so the declaration changes nothing"] if unused_paths else []))
    return {"bounds": [b.to_dict() for b in bounds], "combined_expression": combined_expr, "resources_outside_declared_groups": uncovered,
            "combined_value_known_terms_only": combined, "status": status, "overlap_model": overlap_model or "undeclared",
            "combined_missing_fields": combined_missing, "scenarios": scenarios,
            "notes": ["T_actual >= T_LB only for the counted scenario; missing resources make T_LB a lower bound on the lower bound",
                      "serial_stage_sum groups are resource classes of the whole kernel (declared non-overlap between them), not per-pipeline-stage attribution"]
                     + ([f"serial_stage_groups is declared but inert under overlap_model=full_overlap_max, which maxes over every term"]
                        if (serial_groups and overlap_model == "full_overlap_max") else [])
                     + ([f"resources outside the declared serial groups are not declared serial with them, so they enter the "
                         f"combination through a max rather than the sum: {uncovered}"] if uncovered else [])
                     + consumer_notes + scope_note + overlap_note}


@dataclass
class NodePrediction:
    node: str
    work_expr: str                 # FLOP for a matmul node, element-ops for a vector resource class
    work_value: float | None
    work_unit: str = "FLOP"
    rate_unit: str = "FLOP/s"
    epsilon: tuple | None = None
    rate: float | None = None      # peak throughput in rate_unit
    startup_s: float | None = None
    t_low: float | None = None
    t_high: float | None = None
    bucket_source: str | None = None
    status: str = "uncovered"


from .hardware import device_matches as _device_matches   # one predicate for both report surfaces


_VECTOR_KIND_TO_BUCKET = {"exp": "exp", "reduce": "reduce", "layout": "layout", "vpu": "vector"}


def _graph_vector_work_by_resource(g: AlgorithmGraph, bindings: dict) -> dict:
    """Bound element-op count per resource class (None when it cannot be evaluated)."""
    return {res: numeric_value(substitute(expr, bindings)) for res, expr in g.resource_counts().items()}
_VECTOR_KIND_TO_RATE = RESOURCE_PROFILE_KEY   # exp / vpu / reduce / layout -> the profile key holding that throughput


def calibrated_prediction(g: AlgorithmGraph, bindings: dict, hardware: HardwareProfile | None, compute_dtype: str | None,
                          launch_overhead: float | None = None, model_error: float | None = None,
                          executed_extent_policy: str | None = None, work_undeclared=None,
                          class_dtypes: dict | None = None) -> dict:
    """t_v = F_v / (eps_v P_peak) + t_startup per matmul node; declared serial-sum scheduler; intervals from eps ranges.

    Only matmul nodes with an applicable calibration bucket are predicted; other nodes are listed as
    uncovered.  Missing startup / launch terms are reported as missing (never 0).  Without
    calibration the result is 'not_calibrated'.
    """
    if hardware is None or not hardware.calibration:
        return {"status": "not_calibrated", "applied": False, "predictions": [], "missing_fields": ["calibration"],
                "notes": ["no calibration buckets attached; only resource requirements and lower bounds are reported"]}
    # spec 9.1/9.3: efficiency evidence belongs to a device and a toolchain.  A bucket whose declared device
    # cannot be recognised as this profile's is still applied (dropping the caller's evidence would be its own
    # silent decision) but the mismatch is named, so the prediction never reads as verified-for-this-device.
    mismatched = [bk for bk in hardware.calibration if not _device_matches(bk.device, hardware.name)]
    tool_mismatch = [bk for bk in hardware.calibration
                     if bk.toolchain and hardware.toolchain and dict(bk.toolchain) != dict(hardware.toolchain)]
    tool_undeclared = [bk for bk in hardware.calibration if not bk.toolchain]
    usable_buckets = list(hardware.calibration)
    peak_q = hardware.peak_matmul(compute_dtype)
    peak = peak_q.value if (peak_q and peak_q.known()) else None
    preds: list[NodePrediction] = []
    covered_all = True
    any_applied = False
    total_lo = total_hi = 0.0
    missing: set[str] = set()
    for node in g.matmuls:
        F = substitute(node.flops, bindings)
        Fv = numeric_value(F)
        shape = node.tile_shape if node.tile_shape is not None else (node.M, node.N, node.K)
        M = numeric_value(substitute(shape[0], bindings))
        N = numeric_value(substitute(shape[1], bindings))
        K = numeric_value(substitute(shape[2], bindings))
        if Fv is not None and Fv == 0:
            # e.g. the positional-branch matmul at D_r = 0: nothing executes, so no startup is charged either
            preds.append(NodePrediction(node.id, str(node.flops), 0.0, "FLOP", "FLOP/s", None, peak, status="absent (no work)"))
            continue
        bucket = next((b for b in usable_buckets if b.node_kind == "matmul" and b.applies(compute_dtype, M, N, K)), None)
        if bucket is None or Fv is None or peak is None or bucket.epsilon_low is None or bucket.epsilon_high is None:
            preds.append(NodePrediction(node.id, str(node.flops), Fv, "FLOP", "FLOP/s", None, peak, status="uncovered"))
            covered_all = False
            # an uncovered node's time is absent from the interval: name the evidence that would cover it
            if Fv is None:
                missing.update(free_symbol_names(F))
            elif peak is None:
                missing.add("matmul_input_dtype" if compute_dtype is None else f"peak_flops[{compute_dtype}]")
            else:
                missing.add(f"calibration bucket[{node.id}]")
            continue
        any_applied = True
        startup = bucket.startup_s
        if startup is None:
            missing.add(f"calibration.startup_s[{node.id}]")
            startup_term = 0.0   # excluded, reported missing
        else:
            startup_term = startup
        t_hi = Fv / (bucket.epsilon_low * peak) + startup_term
        t_lo = Fv / (bucket.epsilon_high * peak) + startup_term
        preds.append(NodePrediction(node.id, str(node.flops), Fv, "FLOP", "FLOP/s", (bucket.epsilon_low, bucket.epsilon_high),
                                    peak, startup, t_lo, t_hi, bucket.source, "predicted"))
        total_lo += t_lo
        total_hi += t_hi
    # spec 9.3: vector, reduce, layout and memory use independent models — a declared bucket for one of those
    # node kinds is applied to its own resource class instead of being silently ignored.
    vector_work = _graph_vector_work_by_resource(g, bindings)
    if work_undeclared:
        # the node work these times divide is itself a scenario; the prediction cannot be firmer than its input
        missing.update(work_undeclared)
    for res, work in sorted(vector_work.items()):
        kind = _VECTOR_KIND_TO_BUCKET.get(res)
        # exp runs in exp_dtype, the reductions on the score dtype, the elementwise class on the state dtype:
        # a bucket is applicable to the dtype its class executes in, not to the matmul operand dtype
        class_dtype = (class_dtypes or {}).get(res, compute_dtype)
        bucket = next((bk for bk in usable_buckets if bk.node_kind == kind and bk.applies(class_dtype)), None) if kind else None
        if res == "layout" and bucket is not None and bucket.dtype is not None:
            # layout work spans objects of several dtypes, so the class has no single dtype to match; a bucket that
            # declares one is applied, and the unverified dtype match is named rather than read as verified
            missing.add(f"calibration bucket dtype match[layout] (bucket declared for {bucket.dtype}; layout work spans "
                        f"objects of several dtypes and is matched on kind only)")
        rate_q = hardware.compute.get(_VECTOR_KIND_TO_RATE[res]) if res in _VECTOR_KIND_TO_RATE else None
        rate = rate_q.value if (rate_q and rate_q.known()) else None
        if bucket is None or work is None or rate is None or bucket.epsilon_low is None or bucket.epsilon_high is None:
            preds.append(NodePrediction(f"vector:{res}", "W_resource[" + res + "]", work, "element-ops", "op/s", None, rate,
                                        bucket_source=None if bucket is None else bucket.source, status="uncovered"))
            covered_all = False
            if work is None:
                missing.add(f"W_resource[{res}]")
            elif rate is None:
                missing.add(f"{_VECTOR_KIND_TO_RATE.get(res, res)} (throughput for the {res} class)")
            else:
                missing.add(f"calibration bucket[{kind or res}]")
            continue
        any_applied = True
        st = bucket.startup_s
        if st is None:
            missing.add(f"calibration.startup_s[vector:{res}]")
            st = 0.0
        t_hi = work / (bucket.epsilon_low * rate) + st
        t_lo = work / (bucket.epsilon_high * rate) + st
        preds.append(NodePrediction(f"vector:{res}", f"W_resource[{res}]", work, "element-ops", "op/s",
                                    (bucket.epsilon_low, bucket.epsilon_high), rate, bucket.startup_s, t_lo, t_hi,
                                    bucket.source, "predicted"))
        total_lo += t_lo
        total_hi += t_hi
    if launch_overhead is None:
        missing.add("launch_overhead_s")
        launch = 0.0
    else:
        launch = launch_overhead
    # spec 9.3: T_hat = Schedule(...) + T_launch + eps_model.  The third term is never assumed to be zero.
    if model_error is None:
        missing.add("model_error_s (eps_model)")
        eps_model = 0.0
    else:
        eps_model = float(model_error)
    # a bucket kind with no consumer (hbm_transfer: transfer time is not part of the node-time interval) is
    # accepted by the profile; the report says so where it is declared instead of absorbing it silently
    consumed_kinds = {"matmul"} | set(_VECTOR_KIND_TO_BUCKET.values())
    unused_kinds = sorted({bk.node_kind for bk in usable_buckets} - consumed_kinds)
    unused_note = ([f"calibration bucket kind(s) {unused_kinds} have no consumer in this prediction: transfer time is not "
                    f"part of the predicted node-time interval (spec 9.3), so those buckets change nothing"] if unused_kinds else [])
    if not any_applied:
        return {"status": "not_calibrated", "applied": False, "predictions": [asdict(p) for p in preds],
                "missing_fields": sorted(missing) or ["applicable calibration bucket"],
                "notes": ["calibration attached but no bucket applies to this task's nodes (dtype/shape ranges); no prediction issued"]
                         + unused_note}
    status = "calibrated_partial" if (not covered_all or missing) else "calibrated_all_covered"
    scheduler = "serial_sum_of_predicted_nodes (declared, no overlap credit)"
    notes = unused_note + [
             "matmul nodes use matmul buckets; vector, reduce, layout and exp classes use their own buckets and "
             "throughputs (spec 9.3); a class with no applicable bucket is listed uncovered and its time is not included",
             "layout buckets are matched on kind only (layout work spans objects of several dtypes); a layout bucket that "
             "declares a dtype is applied with the unverified dtype match named in missing_fields",
             "attention nodes are matched to buckets by their executed per-visit tile shape (b_q, b_k, D); F_v is the FLOP-equivalent total over executed cells",
             "interval reflects the epsilon range of the matching buckets, not a validated prediction error on unseen shapes"
             + ("" if all(b.validated_on_unseen_shapes for b in usable_buckets) else " (buckets not validated on unseen shapes)")]
    if mismatched:
        devs = sorted({b.device for b in mismatched if b.device})
        notes.append(f"{len(mismatched)} bucket(s) do not identify this profile's device "
                     f"({devs or 'device undeclared'} vs profile {hardware.name!r}): the efficiency is applied as declared "
                     "but is not established to be evidence about this device")
        missing.add("calibration.device match")
    if tool_mismatch:
        notes.append(f"{len(tool_mismatch)} bucket(s) were measured under a different toolchain than the profile declares; "
                     "epsilon depends on the compiler, so the prediction is not established for this toolchain")
        missing.add("calibration.toolchain match")
    elif tool_undeclared and hardware.toolchain:
        missing.add("calibration.toolchain")
    # spec 9.3 writes epsilon_v(M, N, K, dtype, layout, eta): the bucket contract keys only on dtype and M/N/K
    notes.append("bucket applicability is keyed on dtype and M/N/K only; the layout and strategy arguments of the "
                 "specification's epsilon_v are not part of the bucket contract, so a bucket measured under one "
                 "layout/strategy is applied to another without a check")
    missing.add("calibration bucket applicability in layout and strategy")
    if executed_extent_policy == "padded_to_tile":
        notes.append("executed_extent_policy=padded_to_tile: the padding loss is already inside the node work F_v, so an "
                     "epsilon bucket measured on padded shapes would charge the same loss twice (spec 9.3)")
    if missing:
        notes.append(f"terms absent from the interval (not zero, unknown): {sorted(missing)}")
    return {"status": status, "applied": True, "scheduler": scheduler,
            "predictions": [asdict(p) for p in preds],
            "node_interval_s": [total_lo + launch + eps_model, total_hi + launch + eps_model],
            "matmul_interval_s": [total_lo + launch + eps_model, total_hi + launch + eps_model],   # kept: former name
            "terms": {"nodes_low": total_lo, "nodes_high": total_hi, "launch_overhead_s": launch_overhead, "model_error_s": model_error},
            "missing_fields": sorted(missing), "notes": notes}
