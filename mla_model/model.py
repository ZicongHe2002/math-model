"""MLAForwardModel / BoundModel: the reusable API (spec 11.2).

    model = MLAForwardModel()                        # stateless; no B/H/S/rank/tile inside
    symbolic = model.describe(adapter, algorithm)    # symbolic mode
    bound = model.bind(metadata, semantics, implementation, numerics, hardware, algorithm=...)
    report = bound.analyze()                         # bound / calibrated mode
    comparison = bound.compare(candidates)           # same-task strategy comparison

Every ``bind`` builds a fresh BoundModel; nothing from an earlier binding is retained.
Values computed under an undeclared field are reported with status 'partial' and the
field listed in ``missing_fields``; nothing is filled from examples and unknown is never 0.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

import sympy as sp

from . import memory as MEM
from . import performance as PERF
from . import pressure as PRESS
from . import visibility as VIS
from . import work as WORK
from .adapters import get_adapter, TaskParameters, ROLE_TO_BYTES_SYMBOL
from .algorithms import ALGORITHMS, AlgorithmGraph, build_graph, rows_q, rows_k, _projection_rows
from .errors import BindingError, MetadataError, TaskIdentityError
from .hardware import HardwareProfile
from .implementation import ExecutionStrategy
from .metadata import parse_tensor_metadata
from .numerics import NumericPolicy
from .report import ABSENT, ABSENT_JSON, Conflict, Metric, Report, _Absent, make_metric
from .semantics import SemanticConfig
from .symbolic import BlockStat, Constraint, Sym, evaluate, substitute, numeric_value, free_symbol_names, expr_str

# everything computed on the rectangle grid, and therefore on the scheduled extent
# the materialization *traffic* is read back per executed rectangle, so it moves with the grid; the
# materialized tensor's own size (bytes.mat.size.*) does not, and is excluded by the longer prefix test
GRID_FORMULA_PREFIXES = ("vis.C_rect", "vis.C_pad", "vis.waste", "vis.blocks.", "vis.closed.",
                         "work.exp.executed", "work.vector.", "work.resource.", "sched.", "bytes.path.",
                         "bytes.mat.mat_expanded_kv")     # read back per executed rectangle; the whole-tensor
                                                          # round trips of Q^n / Q~ / Z do not move with the grid
GRID_FORMULA_EXCLUDE = ("bytes.mat.size.",)
GRID_FORMULA_MARKERS = (".total.rect", ".graph.")


def _on_the_grid(formula_id: str) -> bool:
    if any(formula_id.startswith(p) for p in GRID_FORMULA_EXCLUDE):
        return False
    return (any(formula_id.startswith(p) for p in GRID_FORMULA_PREFIXES)
            or any(mark in formula_id for mark in GRID_FORMULA_MARKERS))

TASK_FIELD_NAMES = {"B", "H", "Sq", "Sk", "Rq", "Rk", "Dn", "Dr", "Dv", "S_q", "S_k", "R_q", "R_k", "D_n", "D_r", "D_v",
                    "mask", "position_offset", "scale_value", "scale_policy", "outputs", "active_lengths", "shape", "dtype",
                    "tensors", "tensor_metadata", "semantics", "numerics", "numeric_policy", "algorithm", "cache", "projection_scope"}
DEFAULT_SCALE_REL_TOL = 1e-9   # comparison tolerance for declared-vs-standard scale; echoed in the report, overridable


def _check_scheduled(task, strat: ExecutionStrategy, semantics=None):
    """active <= scheduled <= capacity (spec 1.3); ragged tasks with scheduled extents are not modeled.

    The active-length half of the test only applies to an axis whose active length was actually declared: on
    an undeclared axis ``task.active`` holds the capacity-execution scenario, and rejecting a declaration
    against an assumed value would refuse a legal task.
    """
    if task is None or not strat.scheduled_lengths:
        return
    if task.ragged:
        raise BindingError(["scheduled_lengths with ragged active lengths is not modeled"], "unsupported combination")
    declared_active = semantics.active_lengths or {} if semantics is not None else None
    violations = []
    for key in ("Sq", "Sk"):
        sched = strat.scheduled_lengths.get(key)
        if sched is None:
            continue
        active_declared = declared_active is None or declared_active.get(key) is not None
        if active_declared and sched < task.active[key]:
            violations.append(f"scheduled_lengths.{key}={sched} is smaller than the active length {task.active[key]}")
        if sched > task.capacity[key]:
            violations.append(f"scheduled_lengths.{key}={sched} exceeds the allocated capacity {task.capacity[key]}")
    if violations:
        raise BindingError(violations, "scheduled extents contradict active/capacity extents")


def _apply_scheduled_scenario(task, strat: ExecutionStrategy, sem):
    """On an axis whose active length is undeclared, a declared scheduled extent bounds the scenario.

    The capacity-execution scenario assumes the whole allocation is active; a kernel that iterates only up to a
    declared scheduled extent cannot process more, so the only scenario consistent with spec 1.3's
    active <= scheduled <= capacity is active = scheduled there.  The task's declared identity is unchanged:
    the fingerprint hashes declared active lengths only.
    """
    if task is None or task.ragged or not strat.scheduled_lengths:
        return task
    declared = sem.active_lengths or {} if sem is not None else {}
    new_active = dict(task.active)
    notes = list(task.notes)
    changed = False
    for key in ("Sq", "Sk"):
        sched = strat.scheduled_lengths.get(key)
        if sched is not None and declared.get(key) is None and new_active.get(key) != sched:
            new_active[key] = sched
            notes.append(f"active_lengths.{key} undeclared and scheduled_lengths.{key}={sched} declared: the capacity-execution "
                         f"scenario uses the scheduled extent as the active extent (a kernel cannot process rows it does not iterate)")
            changed = True
    if not changed:
        return task
    return replace(task, active=new_active, notes=notes)


class MLAForwardModel:
    """Family of MLA forward models; instantiate once, bind many times."""

    algorithms = ALGORITHMS

    def analyze_yaml(self, source, *, device="tpu-v6-e", algorithm="mla", mask=None):
        """One-call modelling from workload YAML, device and algorithm; see workflow.py."""
        from .workflow import analyze_yaml
        return analyze_yaml(source, device=device, algorithm=algorithm, mask=mask, model=self)

    def describe(self, adapter: str = "seven_input_latent", algorithm: str = "expanded") -> Report:
        """Symbolic mode: expressions, constraints and the dependency structure with nothing bound."""
        if algorithm not in ALGORITHMS:
            raise MetadataError(f"unknown algorithm {algorithm!r}; available {ALGORITHMS}")
        ad = get_adapter(adapter)
        bound = BoundModel(task=None, semantics=SemanticConfig(), strategy=ExecutionStrategy(name="symbolic"),
                           numerics=NumericPolicy(), hardware=None, algorithm=algorithm, adapter_name=adapter, mode="symbolic")
        rep = bound.analyze()
        rep.title = f"MLA forward — symbolic description ({algorithm})"
        rep.extras["adapter_contract"] = ad.describe()
        return rep

    def bind(self, tensor_metadata, semantic_config=None, implementation=None, numeric_policy=None,
             hardware_profile=None, *, algorithm: str | None = None, adapter: str = "seven_input_latent",
             dtype_overrides: dict | None = None) -> "BoundModel":
        """Bind the current invocation.

        ``algorithm`` is the explicitly specified path a of spec 0.1.  Spec 11.1 also allows it to be
        *preserved as unknown*: omitted, the report carries the algorithm-independent metrics once and every
        variant-dependent metric once per explicitly named variant scenario, with ``algorithm`` in its
        missing fields (spec 0.1: named scenarios, never a silently chosen default).
        """
        if algorithm is not None and algorithm not in ALGORITHMS:
            raise MetadataError(f"unknown algorithm {algorithm!r}; available {ALGORITHMS}")
        ad = get_adapter(adapter)
        tensors = parse_tensor_metadata(tensor_metadata)
        sem = SemanticConfig.from_source(semantic_config)
        strat = ExecutionStrategy.from_source(implementation)
        num = NumericPolicy.from_source(numeric_policy)
        hw = HardwareProfile.from_source(hardware_profile)
        overrides = dict(num.dtype_bytes_overrides or {})
        if dtype_overrides:
            # the keyword goes through the same validation and normalisation as the numeric-policy field
            overrides.update(NumericPolicy(dtype_bytes_overrides=dict(dtype_overrides)).dtype_bytes_overrides or {})
            num = replace(num, dtype_bytes_overrides=overrides)
        task = ad.extract(tensors, sem, overrides or None)
        _check_scheduled(task, strat, sem)
        task = _apply_scheduled_scenario(task, strat, sem)
        return BoundModel(task=task, semantics=sem, strategy=strat, numerics=num, hardware=hw,
                          algorithm=algorithm, adapter_name=adapter, mode="bound")


@dataclass
class BoundModel:
    task: TaskParameters | None
    semantics: SemanticConfig
    strategy: ExecutionStrategy
    numerics: NumericPolicy
    hardware: HardwareProfile | None
    algorithm: str
    adapter_name: str
    mode: str = "bound"
    _graph: AlgorithmGraph | None = field(default=None, repr=False)
    compile_evidence: dict | None = field(default=None, repr=False)
    # budgets declared by both the strategy and the hardware profile that disagree (filled while binding)
    _budget_disagreements: dict = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------ evidence attachment (agent-facing, spec 12.2)
    def attach_calibration(self, buckets) -> "BoundModel":
        """Return a new BoundModel whose hardware profile carries additional calibration buckets (validated on load)."""
        if self.hardware is None:
            raise MetadataError("attach_calibration needs a hardware profile to attach to")
        prof = HardwareProfile.from_source({**self.hardware.to_dict(), "calibration": [asdict_bucket(b) for b in self.hardware.calibration] + list(buckets)})
        return BoundModel(task=self.task, semantics=self.semantics, strategy=self.strategy, numerics=self.numerics, hardware=prof,
                          algorithm=self.algorithm, adapter_name=self.adapter_name, mode=self.mode, compile_evidence=self.compile_evidence)

    def attach_compile_evidence(self, evidence: dict) -> "BoundModel":
        """Return a new BoundModel carrying compile/measurement evidence.

        Recognised keys (all optional): compiled_flops, spill_peak_bytes, spill_fill_bytes, spill_exposed_s, source, toolchain.
        Values are reported with source_type 'compiled'/'measured'; nothing is inferred from them.
        """
        if not isinstance(evidence, dict):
            raise MetadataError("compile evidence must be a mapping")
        known = {"compiled_flops", "spill_peak_bytes", "spill_fill_bytes", "spill_exposed_s", "spill_instructions",
                 "spill_comparison_graph", "source", "toolchain",
                 # spec 12.1: when the compiler output lacks loop bodies or widths the result stays partial,
                 # so the caller can say what the dump did not contain instead of it reading as complete
                 "coverage", "omits"}
        unknown = sorted(set(evidence) - known)
        if unknown:
            raise MetadataError(f"unknown compile-evidence fields {unknown}; known: {sorted(known)}")
        return BoundModel(task=self.task, semantics=self.semantics, strategy=self.strategy, numerics=self.numerics, hardware=self.hardware,
                          algorithm=self.algorithm, adapter_name=self.adapter_name, mode=self.mode, compile_evidence=dict(evidence))

    # ------------------------------------------------------------------ bindings
    @property
    def ragged(self):
        return self.task.ragged if self.task else None

    def gamma(self) -> tuple[Any, list[Conflict], list[str]]:
        """Scale gamma from the declared policy; conflicts are returned, never resolved silently."""
        sem = self.semantics
        conflicts: list[Conflict] = []
        assumptions: list[str] = []
        standard = (Sym.D_n + Sym.D_r) ** sp.Rational(-1, 2)
        if sem.scale_policy == "standard":
            if sem.scale_value is not None and self.task is not None:
                std_val = numeric_value(substitute(standard, self.bindings(include_gamma=False)))
                tol = DEFAULT_SCALE_REL_TOL if sem.scale_match_rel_tol is None else float(sem.scale_match_rel_tol)
                tol_note = f"comparison tolerance rel_tol={tol}" + (" (default, override with scale_match_rel_tol)" if sem.scale_match_rel_tol is None else " (declared)")
                if std_val is not None and not math.isclose(float(sem.scale_value), float(std_val), rel_tol=tol, abs_tol=0.0):
                    conflicts.append(Conflict("scale", {"scale_policy": "standard", "standard_value": std_val,
                                                        "scale_value": sem.scale_value, "tolerance": tol_note},
                                              "declared scale_value differs from the standard policy value (D_n + D_r)^(-1/2); "
                                              "the declared value takes precedence (prompt 3) and the conflict is reported, neither source is overwritten"))
                    assumptions.append("conflict: declared scale_value used (precedence), standard value listed in the conflict record")
                    return sp.Float(float(sem.scale_value)), conflicts, assumptions
                assumptions.append(f"declared scale_value agrees with the standard value ({tol_note})")
            assumptions.append("scale policy 'standard': gamma = (D_n + D_r)^(-1/2)")
            return standard, conflicts, assumptions
        if sem.scale_policy == "declared":
            if sem.scale_value is None:
                conflicts.append(Conflict("scale", {"scale_policy": "declared", "scale_value": None},
                                          "scale_policy 'declared' without scale_value", severity="error"))
                return None, conflicts, assumptions
            assumptions.append("scale declared by the caller (model-specific corrections included by the caller)")
            return sp.Float(float(sem.scale_value)), conflicts, assumptions
        if sem.scale_value is not None:
            assumptions.append("scale_value given without a policy: treated as declared")
            return sp.Float(float(sem.scale_value)), conflicts, assumptions
        return None, conflicts, assumptions

    def bindings(self, include_gamma: bool = True) -> dict[str, Any]:
        b: dict[str, Any] = {}
        if self.task is not None:
            b.update(self.task.symbol_bindings())
        b.update(self.numerics.storage_bytes_bindings())
        b.update(self.strategy.bindings())
        if self.task is not None and not self.ragged:
            b.setdefault("S_q_sched", b.get("S_q"))      # scheduled extent undeclared -> equals the active extent (labeled scenario)
            b.setdefault("S_k_sched", b.get("S_k"))
        if self.hardware is not None:
            b.update(self.hardware.bindings(self.numerics.matmul_input_dtype))
        cache = self.semantics.cache or {}
        if cache.get("merged_weight_reuse") is not None:
            b["n_reuse"] = int(cache["merged_weight_reuse"])
        if cache.get("new_tokens") is not None:
            b["S_new"] = int(cache["new_tokens"])
        # declared budgets per storage level (never substituted for each other).  The strategy is the compiler's
        # scoped budget and the profile is the hardware capacity; spec 7.3 keeps them separate, so when both are
        # declared and disagree the strategy is used and the disagreement is recorded rather than hidden.
        for sym, field_name in (("R_vreg", "vreg_budget_bytes"), ("R_vmem", "vmem_budget_bytes")):
            declared = getattr(self.strategy, field_name)
            if declared is not None:
                from_profile = b.get(sym)
                if from_profile is not None and from_profile != declared:
                    self._budget_disagreements[sym] = {"strategy": declared, "hardware_profile": from_profile, "used": declared}
                b[sym] = declared
        if include_gamma:
            g, _, _ = self.gamma()
            if g is not None:
                v = numeric_value(substitute(g, b))
                if v is not None:
                    b["gamma"] = v
        return b

    def capacity_bindings(self) -> dict[str, Any]:
        b = dict(self.bindings())
        if self.task is not None:
            b.update(self.task.capacity_bindings())
        return b

    @property
    def graph(self) -> AlgorithmGraph:
        if self._graph is None:
            self._graph = build_graph(self.algorithm, self.semantics, self.strategy, self.numerics, self.ragged,
                                       dims=self.task.dims if self.task else None)
        return self._graph

    def with_strategy(self, strategy) -> "BoundModel":
        strat = ExecutionStrategy.from_source(strategy) if not isinstance(strategy, ExecutionStrategy) else strategy
        _check_scheduled(self.task, strat, self.semantics)
        task = _apply_scheduled_scenario(self.task, strat, self.semantics)
        return BoundModel(task=task, semantics=self.semantics, strategy=strat, numerics=self.numerics,
                          hardware=self.hardware, algorithm=self.algorithm, adapter_name=self.adapter_name, mode=self.mode,
                          compile_evidence=self.compile_evidence)

    def with_algorithm(self, algorithm: str | None) -> "BoundModel":
        if algorithm is not None and algorithm not in ALGORITHMS:
            raise MetadataError(f"unknown algorithm {algorithm!r}")
        return BoundModel(task=self.task, semantics=self.semantics, strategy=self.strategy, numerics=self.numerics,
                          hardware=self.hardware, algorithm=algorithm, adapter_name=self.adapter_name, mode=self.mode,
                          compile_evidence=self.compile_evidence)

    # ------------------------------------------------------------------ identity
    def fingerprint(self) -> str | None:
        """Normalized summary of adapter, dims/active lengths, mask/position, scale, output scope, numerical contract."""
        if self.task is None:
            return None
        sem = self.semantics
        g, _, _ = self.gamma()
        payload = {
            "adapter": self.adapter_name,
            "dims": self.task.dims,
            "active": {k: v for k, v in self.task.active.items() if (sem.active_lengths or {}).get(k) is not None},
            "active_declared_axes": sorted(k for k in ("Sq", "Sk") if (sem.active_lengths or {}).get(k) is not None),
            "offset": self.task.offset, "ragged": self.task.ragged,
            "mask": sem.mask, "scale": None if g is None else str(g),
            "outputs": None if sem.outputs is None else list(sem.outputs),
            "lse_convention": sem.lse_convention, "projection_scope": sem.projection_scope,
            "cache": None if sem.cache is None else {k: v for k, v in sem.cache.items() if k != "merged_weight_reuse"},
            "empty_row_policy": sem.empty_row_policy,
            "dtypes": self.task.dtypes,
            "numerics": {k: v for k, v in self.numerics.to_dict().items() if k not in ("notes",)},
        }
        text = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    # ------------------------------------------------------------------ analysis
    def analyze(self) -> Report:
        if self.algorithm is None:
            return self._analyze_algorithm_scenarios()
        alg = self.algorithm
        sem, strat, num, task = self.semantics, self.strategy, self.numerics, self.task
        b = self.bindings()
        cap_b = self.capacity_bindings()
        g = self.graph
        rep = Report(title=f"MLA forward analytical report ({alg}, strategy={strat.name})", mode=self.mode,
                     adapter=self.adapter_name, algorithm=alg, task_fingerprint=self.fingerprint())
        M: list[Metric] = rep.metrics
        gamma_expr, conflicts, gamma_assumptions = self.gamma()
        rep.conflicts.extend(conflicts)
        for cc in g.cast_conflicts:
            rep.conflicts.append(Conflict("cast_points", {"site": cc["site"], "declared_in_cast_points": cc["declared_in_cast_points"],
                                                          "dtypes": cc["dtypes"]}, cc["message"], severity="warning"))
        for sym, srcs in self._budget_disagreements.items():
            level = "VREG" if sym == "R_vreg" else "VMEM"
            rep.conflicts.append(Conflict(sym, srcs,
                                          f"{level} budget declared twice and differently: the execution strategy's scoped budget "
                                          f"({srcs['strategy']} bytes) and the hardware profile's capacity ({srcs['hardware_profile']} bytes) "
                                          "are separate quantities (spec 7.3); the strategy's scoped budget is used for feasibility",
                                          severity="warning"))
        ragged = self.ragged
        rq, rk = rows_q(ragged), rows_k(ragged)
        # spec 1.3 keeps shape and effective length apart per axis: a dictionary that declares only one of
        # them leaves the other axis on its shape extent, which is a scenario, not a declaration
        al = sem.active_lengths or {}
        # an axis present with a null value is not declared: use the same predicate the adapter uses when it
        # decides whether to fall back to the tensor extent, so the two can never disagree
        undeclared_axes = [k for k in ("Sq", "Sk") if al.get(k) is None]
        active_declared = sem.active_lengths is not None and not undeclared_axes
        # on an undeclared axis a declared scheduled extent bounds the scenario (see _apply_scheduled_scenario):
        # the assumption text names the extent that was actually used, per axis
        sched_bound_axes = [k for k in undeclared_axes
                            if task is not None and not task.ragged and (strat.scheduled_lengths or {}).get(k) is not None]
        per_axis = "; ".join(f"{k}: {'the declared scheduled extent' if k in sched_bound_axes else 'the tensor extent'}"
                             for k in undeclared_axes)
        if task is None:
            active_missing, active_assumption = [], []
        elif not undeclared_axes:
            active_missing, active_assumption = [], []
        elif not any(al.get(k) is not None for k in ("Sq", "Sk")):
            active_missing = ["active_lengths"]
            active_assumption = [f"capacity-execution scenario ({per_axis} used as the active length) because active_lengths is undeclared"]
        else:
            active_missing = [f"active_lengths.{k}" for k in undeclared_axes]
            active_assumption = [f"capacity-execution scenario on {', '.join(undeclared_axes)} ({per_axis} used as the active length) "
                                 f"because active_lengths declares only {sorted(k for k in al if al.get(k) is not None)}"]
        # a scheduled extent that was not declared is an assumption (spec 1.3), not a reading — per axis, like
        # the active lengths, since bindings() fills each axis separately from the active extent
        sl = strat.scheduled_lengths or {}
        sched_undeclared_axes = [k for k in ("Sq", "Sk") if sl.get(k) is None]
        if task is None or not sched_undeclared_axes or ragged:
            # a per-batch task has no declarable scheduled extent (not modelled): the assumption says so, and
            # naming an undeclarable field as missing would keep every grid metric partial forever
            sched_missing = []
        elif len(sched_undeclared_axes) == 2:
            sched_missing = ["scheduled_lengths"]
        else:
            sched_missing = [f"scheduled_lengths.{k}" for k in sched_undeclared_axes]
        sched_assumption = ([] if not sched_missing else
                            ["scheduled extent undeclared: assumed equal to the active extent (scenario)"]
                            + (["scheduled extents are not modeled for per-batch (ragged) tasks"] if ragged else []))
        exec_missing = list(g.undeclared["executed_work"]) + sched_missing
        vec_missing = list(g.undeclared["vector"])                     # fields affecting every vector count
        vec_kind_missing = g.undeclared_by_kind()                      # per-kind undeclared conditions
        vec_res_missing = g.undeclared_by_resource()
        xfer_missing = [f for f in g.undeclared["transfers"] if f != "register_schedule_evidence"]
        # Undeclared 'locals' fields are attached to the objects they shape (conditional_on, size_scenario,
        # stage_scenario).  Two of them change a set without any single member carrying them — score_alias_exp
        # and exp_alias_p_operand decide whether objects SHARE an allocation — so those keep a graph-level
        # fallback onto every pressure metric.
        ALIASING_FIELDS = ("score_alias_exp", "exp_alias_p_operand")
        aliasing_missing_all = [f for f in g.undeclared["locals"] if f in ALIASING_FIELDS]

        def aliasing_for(level):
            # the aliased objects (X, E, P) are vector temporaries: the flags shape the vreg and the combined
            # sets, never the vmem set, which must not be marked incomplete by them
            return aliasing_missing_all if level in (None, "vreg") else []
        # the documented dtype fallbacks are assumptions, so name them on the objects they size
        dtype_fallback_missing = []
        if num.projected_operand_dtype is None and num.matmul_input_dtype is not None:
            dtype_fallback_missing.append("projected_operand_dtype")
        if num.projection_accumulator_dtype is None and num.accumulator_dtype is not None:
            dtype_fallback_missing.append("projection_accumulator_dtype")
        FALLBACK_SYMBOLS = {"projected_operand_dtype": (Sym.s_Qn, Sym.s_Kn, Sym.s_V, Sym.s_Qt, Sym.s_Z),
                            "projection_accumulator_dtype": (Sym.s_Qacc,)}
        num_assumptions = num.assumptions()

        # ---- header ---------------------------------------------------------------
        rep.header["bindings"] = {k: v for k, v in b.items()}
        if task is not None:
            rep.header["dimension_provenance"] = task.provenance
            rep.header["cross_checks"] = task.cross_checks
            rep.header["capacity_extents"] = task.capacity
            rep.header["active_extents"] = {**task.active, "declared": active_declared,
                                            "declared_axes": sorted(k for k in al if al.get(k) is not None),
                                            "undeclared_axes": undeclared_axes}
            rep.header["dtypes"] = task.dtypes
            if task.notes:
                rep.warnings.extend(task.notes)
            if not active_declared:
                rep.warnings.append(f"active lengths undeclared for {undeclared_axes or ['Sq', 'Sk']}: {per_axis} used there "
                                    f"(capacity-execution scenario); dependent metrics carry {active_missing} in missing_fields")
        rep.header["semantics"] = sem.to_dict()
        rep.header["implementation"] = strat.to_dict()
        rep.header["numerics"] = num.to_dict()
        rep.header["numerics_assumptions"] = num_assumptions
        rep.header["hardware"] = None if self.hardware is None else {
            "name": self.hardware.name, "scope_default": self.hardware.scope_default,
            "provenance": self.hardware.provenance, "retrieved": self.hardware.retrieved,
            "toolchain": self.hardware.toolchain or None,
            # spec 9.1: each quantity carries its own value, unit, scope, source and confidence; the report
            # keeps them so a reader can see which figure is a datasheet number and which is measured
            "quantities": self.hardware.quantity_summary(),
            "calibration": self.hardware.calibration_summary()}
        if self.hardware is not None:
            rep.warnings.extend(self.hardware.scope_report())
            rep.warnings.extend(self.hardware.notes)
        rep.warnings.extend(g.scenario_notes)
        rep.warnings.extend(num_assumptions)
        for n in (sem.notes + num.notes + strat.notes):
            rep.warnings.append(n)

        def metric(name, fid, expr, bnd, **kw):
            """make_metric with the extent scenarios propagated to every metric that depends on them.

            Dependence is decided by formula id, not by the shape of the expression: for a ragged task the
            per-batch extents fold into integer constants, so an expression-shape test would silently miss
            exactly the metrics that depend on the undeclared lengths.
            """
            extra = list(kw.pop("extra_missing", []) or [])
            assumptions = list(kw.pop("assumptions", []) or [])
            if active_missing:
                # per axis: a quantity that involves only S_q does not depend on an undeclared S_k.  Dependence is
                # read from the unsubstituted expression (a grid statistic depends on both axes), never from the
                # folded value.
                syms = {str(x) for x in getattr(expr, "free_symbols", set())} if hasattr(expr, "free_symbols") else set()
                grid = _on_the_grid(fid) or any(str(f.func) in ("BlockStat", "CausalVisibleCount") for f in getattr(expr, "atoms", lambda *a: set())(sp.Function))
                for f in active_missing:
                    axis = f.rsplit(".", 1)[-1] if "." in f else None
                    if axis == "Sq" and not (grid or syms & {"S_q", "S_q_sched"}):
                        continue
                    if axis == "Sk" and not (grid or syms & {"S_k", "S_k_sched", "S_new"}):
                        continue
                    extra.append(f)
                if any(f in extra for f in active_missing):
                    assumptions += active_assumption
            if sched_missing and _on_the_grid(fid):
                extra += sched_missing
                assumptions += sched_assumption
            kw.setdefault("algorithm", alg)
            return make_metric(name, fid, expr, bnd, extra_missing=extra, assumptions=assumptions, **kw)

        # ---- task section -----------------------------------------------------------
        sec = "1. task and scope"
        for dim, sym in (("B", Sym.B), ("H", Sym.H), ("R_q", Sym.R_q), ("R_k", Sym.R_k), ("D_n", Sym.D_n), ("D_r", Sym.D_r), ("D_v", Sym.D_v)):
            M.append(make_metric(dim, f"task.dim.{dim}", sym, b, unit="count", scope="current invocation", section=sec, source_type="input",
                                 evidence=[task.provenance.get(dim.replace("_", ""), "")] if task else []))
        for key, sym, scope in (("S_q", Sym.S_q, "active query rows (mathematical work)"), ("S_k", Sym.S_k, "active keys consumed in this call")):
            # an extent metric depends on its own axis only: the other axis's declaration cannot shape it
            own_axis = key.replace("_", "")
            own_missing = [f for f in active_missing if f == "active_lengths" or f.endswith("." + own_axis)]
            own_assumption = list(active_assumption) if own_missing else []
            if ragged:
                m = Metric(f"{key}_active", f"task.active.{key}", f"per-batch list {key}_b", {}, list(ragged[own_axis]), "count",
                           scope + " (ragged: one entry per batch)", "partial" if own_missing else "bound", alg, "input", section=sec,
                           missing_fields=own_missing, assumptions=own_assumption)
            else:
                m = make_metric(f"{key}_active", f"task.active.{key}", sym, b, unit="count", scope=scope, section=sec, source_type="input",
                                extra_missing=own_missing, assumptions=own_assumption)
            M.append(m)
        sched = bool(strat.scheduled_lengths) and not ragged
        for key in ("S_q", "S_k"):
            axis = key.replace("_", "")
            declared = sl.get(axis)
            # falling back to the active extent inherits whatever that extent rests on, so both are named
            own_missing = [f for f in active_missing if f == "active_lengths" or f.endswith("." + axis)]
            fallback_missing = (([] if ragged else [f"scheduled_lengths.{axis}"]) + own_missing) if declared is None else []
            if ragged and declared is None:
                M.append(Metric(f"{key}_scheduled", f"task.scheduled.{key}", "not modelled for per-batch tasks", {}, None, "count",
                                "extent the kernel iterates over", "not_applicable", alg, "input", section=sec,
                                assumptions=["scheduled extents are not modelled for per-batch (ragged) tasks: each batch's grid "
                                             "spans its own active extent"], not_equivalent_to=["active extent", "allocated capacity"]))
                continue
            M.append(Metric(f"{key}_scheduled", f"task.scheduled.{key}", "declared scheduled extent (else = active)", {}, declared if declared is not None else b.get(key),
                            "count", "extent the kernel iterates over (execution rectangles, padding counts)", "bound" if declared is not None else "partial", alg, "input",
                            section=sec, missing_fields=fallback_missing,
                            assumptions=([] if declared is not None else
                                         ["scheduled extent undeclared: assumed equal to the active extent (scenario)"] + (list(active_assumption) if own_missing else [])),
                            not_equivalent_to=["active extent", "allocated capacity"]))
        if task is not None:
            M.append(Metric("S_q_capacity", "task.capacity.S_q", "allocated extent from tensor shape", {}, task.capacity["Sq"], "count",
                            "allocated query extent", "bound", alg, "input", section=sec, not_equivalent_to=["active rows", "scheduled rows"]))
            M.append(Metric("S_k_capacity", "task.capacity.S_k", "allocated extent from tensor shape", {}, task.capacity["Sk"], "count",
                            "allocated KV extent (not necessarily consumed)", "bound", alg, "input", section=sec,
                            not_equivalent_to=["active keys", "scheduled keys"]))
        if ragged and ragged.get("Delta"):
            M.append(Metric("Delta", "task.offset", "per-batch list Delta_b", {}, list(ragged["Delta"]), "positions", "key j visible to query i iff j <= i + Delta_b", "bound", alg, "input", section=sec))
        else:
            M.append(make_metric("Delta", "task.offset", Sym.Delta, b, unit="positions", scope="key j visible to query i iff j <= i + Delta", section=sec, source_type="input",
                                 extra_missing=([] if (b.get("Delta") is not None or sem.mask == "none") else ["position_offset"]),
                                 notes=["not applicable to the dense mask"] if sem.mask == "none" else [],
                                 status_override=("not_applicable" if sem.mask == "none" and b.get("Delta") is None else None)))
        M.append(Metric("mask", "task.mask", "declared mask kind", {}, sem.mask, "", "visibility semantics", "bound" if sem.mask else "unknown", alg, "input",
                        section=sec, missing_fields=[] if sem.mask else ["mask"], assumptions=["mask is declared, never inferred from shapes"]))
        if (strat.exp_alias_p_operand and num.exp_dtype and num.p_operand_dtype and num.exp_dtype != num.p_operand_dtype):
            rep.conflicts.append(Conflict("exp_alias_p_operand",
                                          {"exp_alias_p_operand": True, "exp_dtype": num.exp_dtype, "p_operand_dtype": num.p_operand_dtype},
                                          "the PV operand is declared to alias E, but the two carry different dtypes; one storage cannot hold both. "
                                          "Declare equal dtypes, or exp_alias_p_operand = False with an explicit cast",
                                          severity="error"))
        gm = make_metric("gamma", "task.scale", gamma_expr, b, unit="1", scope="score scale", section=sec, source_type="input",
                         assumptions=gamma_assumptions, extra_missing=[] if gamma_expr is not None else ["scale_policy/scale_value"])
        if any(c.field == "scale" for c in conflicts):
            gm.status = "conflict"
        M.append(gm)
        M.append(Metric("outputs", "task.outputs", "declared output scope", {}, None if sem.outputs is None else list(sem.outputs), "", "output scope",
                        "bound" if sem.outputs is not None else "unknown", alg, "input", section=sec, missing_fields=[] if sem.outputs is not None else ["outputs"]))
        wl = sem.wants_lse()
        M.append(Metric("lse_convention", "task.lse_convention", "declared", {}, sem.lse_convention, "", "LSE log base",
                        "bound" if sem.lse_convention else ("not_applicable" if wl is False else "unknown"), alg, "input",
                        section=sec, missing_fields=[] if (sem.lse_convention or wl is False) else ["lse_convention"],
                        assumptions=([] if sem.lse_convention else
                                     (["LSE is not among the declared outputs, so its log base is not part of this task"]
                                      if wl is False else
                                      ["LSE is requested but its log base is undeclared"]))))
        M.append(Metric("projection_scope", "task.projection_scope", "declared", {}, sem.projection_scope or "full (scenario: projection_scope undeclared)", "",
                        "which rows are projected in this call", "bound" if sem.projection_scope else "partial", alg, "input", section=sec,
                        missing_fields=[] if sem.projection_scope else ["projection_scope"]))
        if ragged:
            if ragged.get("Delta") and sem.mask == "causal":
                empty_expr = Sym.H * sp.Integer(sum(max(0, min(sq, -dl)) for sq, dl in zip(ragged["Sq"], ragged["Delta"])))
            elif sem.mask == "none":
                empty_expr = sp.Integer(0)
            else:
                empty_expr = None
        else:
            empty_expr = VIS.empty_rows_expr(sem.mask)
        em = metric("rows_without_visible_keys", "task.empty_rows", empty_expr, b, unit="rows", scope="(b,h,i) rows whose softmax is undefined", section=sec,
                    assumptions=["uniform lengths: B*H*max(0, min(S_q, -Delta))"], extra_missing=[] if sem.mask else ["mask"])
        if em.value not in (None, 0):
            if sem.empty_row_policy is None:
                rep.conflicts.append(Conflict("empty_row_policy", {"rows_without_visible_keys": em.value, "empty_row_policy": None},
                                              "rows with no visible key exist but no empty_row_policy is declared", severity="error"))
            elif sem.empty_row_policy == "error":
                rep.conflicts.append(Conflict("empty_row_policy", {"rows_without_visible_keys": em.value, "empty_row_policy": "error"},
                                              "policy 'error' with rows that have no visible key: the task is ill-defined as declared", severity="error"))
            else:
                em.assumptions.append("policy zero_output_neg_inf_lse: O = 0, LSE = -inf for these rows")
        M.append(em)

        # ---- visibility section -------------------------------------------------------
        sec = "2. visibility and rectangles"
        vis_ok = sem.mask is not None and (not ragged or ragged.get("Delta") or sem.mask == "none")
        C_valid = VIS.visible_count_expr(sem.mask, ragged if (ragged and vis_ok) else None) if vis_ok else None
        mv = metric("C_valid", "vis.C_valid", C_valid, b, unit="cells", scope="all (b,h,i,j) with mu=1; includes B and H", section=sec,
                    assumptions=["uniform unit-step positions; causal: j <= i + Delta"] if not ragged else ["explicit finite sum over batches"],
                    extra_missing=[] if sem.mask else ["mask"], not_equivalent_to=["exp instruction count", "executed cells"])
        M.append(mv)
        C_valid_val = mv.value
        policy_known = strat.rect_policy is not None
        mask_known = sem.mask is not None
        pol_note = None
        if vis_ok and policy_known:
            rag = ragged if ragged else None
            C_rect = VIS.rect_cells_expr(sem.mask, strat.rect_policy, strat.subdivide, rag, scheduled=sched)
            C_padc = VIS.padded_cells_expr(sem.mask, strat.rect_policy, strat.subdivide, rag, scheduled=sched)
            pol_note = f"rectangle selection policy '{strat.rect_policy}'" + (f" with sub-tiles {strat.subdivide}" if strat.subdivide else "") \
                + (" on the scheduled grid (padding rows/keys masked; fully-padding blocks skipped by skip_future)" if sched else "")
            M.append(metric("C_rect", "vis.C_rect", C_rect, b, unit="cells", scope="sum over selected rectangles of n_i*m_j (logical extents)", section=sec,
                            assumptions=[pol_note, "tail blocks use logical extents"],
                            not_equivalent_to=["C_valid"] + (["C_pad"] if strat.executed_extent_policy != "logical" else
                                                             ["C_pad under a padded execution extent"])))
            # spec 4.2: n_hat, m_hat come from the declared strategy or compilation evidence.  A declared
            # 'logical' extent policy means the executed extents ARE the logical ones, so C_pad = C_rect;
            # only an undeclared policy leaves the padded tile as a scenario.
            exec_pol_declared = strat.executed_extent_policy
            if exec_pol_declared == "logical":
                pad_expr, pad_scope, pad_note = (C_rect, "sum over selected rectangles of n_i*m_j: the declared "
                                                 "executed extents are the logical ones", "executed extents = logical extents (declared 'logical')")
            else:
                sub_note = " (p_q*p_k for the sub-blocks of a subdivided partial block)" if strat.subdivide else ""
                pad_expr, pad_scope = C_padc, f"sum over selected rectangles of b_q*b_k{sub_note} (extents padded to the tile)"
                pad_note = ("executed extents = full tile for every selected block (declared 'padded_to_tile')"
                            if exec_pol_declared else
                            "executed extents = full tile for every selected block (scenario: executed_extent_policy undeclared)")
            M.append(metric("C_pad", "vis.C_pad", pad_expr, b, unit="cells", scope=pad_scope, section=sec,
                            assumptions=[pol_note, pad_note],
                            extra_missing=[] if exec_pol_declared else ["executed_extent_policy"],
                            not_equivalent_to=["hardware array padding (must come from compilation evidence)"]))
            if not exec_pol_declared:
                # spec 4.2 asks for scenarios rather than an inferred choice
                M.append(metric("C_pad[scenario=logical]", "vis.C_pad.scenario.logical", C_rect, b, unit="cells",
                                scope="C_pad if the executed extents are the logical ones", section=sec,
                                assumptions=[pol_note], extra_missing=["executed_extent_policy"]))
                M.append(metric("C_pad[scenario=padded_to_tile]", "vis.C_pad.scenario.padded_to_tile", C_padc, b, unit="cells",
                                scope="C_pad if every selected block executes its full tile"
                                      + (" (sub-tile for subdivided partial blocks)" if strat.subdivide else ""), section=sec,
                                assumptions=[pol_note], extra_missing=["executed_extent_policy"]))
            if C_valid is not None:
                M.append(metric("rect_waste", "vis.waste", C_rect - C_valid, b, unit="cells", scope="C_rect - C_valid under the declared policy", section=sec))
            for stat, label in (("selected", "selected_blocks"), ("partial", "partial_blocks"), ("full", "fully_visible_blocks"),
                                ("future_selected", "future_blocks_selected"), ("padding_selected", "padding_blocks_selected"),
                                ("future_skipped", "future_blocks_skipped"),
                                ("padding_skipped", "padding_blocks_skipped"), ("masked_cells", "masked_cells")):
                e = VIS._aggregate(stat, sem.mask, strat.rect_policy, strat.subdivide, rag, scheduled=sched)
                M.append(metric(label, f"vis.blocks.{stat}", e, b, unit="blocks" if stat != "masked_cells" else "cells", scope="over all (b,h)", section=sec, assumptions=[pol_note]))
            exec_pol = strat.executed_extent_policy
            M.append(Metric("executed_extent_policy", "vis.exec_policy", "declared", {}, exec_pol, "", "n_hat, m_hat source", "bound" if exec_pol else "unknown", alg, "input",
                            section=sec, missing_fields=[] if exec_pol else ["executed_extent_policy"],
                            assumptions=["executed extents come from the declared strategy or compilation evidence, never from the hardware array size"]))
        else:
            for pol in ("full_scan", "skip_future"):
                if mask_known and not ragged:
                    e = VIS.rect_cells_expr(sem.mask, pol)
                    M.append(metric(f"C_rect[scenario={pol}]", f"vis.C_rect.scenario.{pol}", e, b, unit="cells", scope="scenario, policy not declared", section=sec,
                                    assumptions=[f"scenario policy {pol}"], extra_missing=["rect_policy"]))
            rep.unknowns.append("rect_policy not declared: C_rect/C_pad reported as named scenarios only" if mask_known else "mask not declared: visibility metrics unknown")
            if ragged and not ragged.get("Delta"):
                rep.unknowns.append("per_batch_offsets not declared for a ragged task: visibility stays unknown")
        if (not ragged and sem.mask == "causal" and b.get("S_q") is not None and b.get("S_q") == b.get("S_k") and b.get("Delta") == 0
                and strat.b_q and strat.b_k and strat.b_q == strat.b_k and b["S_q"] % strat.b_q == 0 and strat.rect_policy == "skip_future"):
            Sv, pv = b["S_q"], strat.b_q
            st = "partial" if active_missing else "bound"
            M.append(Metric("closed_form_rect_per_bh", "vis.closed.rect", "(S**2 + S*p)/2", {"S": Sv, "p": pv}, (Sv * Sv + Sv * pv) // 2, "cells",
                            "per (b,h); square causal, S divisible by p, skip future, diagonal in full", st, alg, "derived", section=sec, missing_fields=active_missing))
            M.append(Metric("closed_form_waste_per_bh", "vis.closed.waste", "S*(p-1)/2", {"S": Sv, "p": pv}, Sv * (pv - 1) // 2, "cells", "per (b,h)", st, alg, "derived",
                            section=sec, missing_fields=active_missing, notes=["explains geometric waste vs p; says nothing about latency monotonicity"]))

        # ---- work section --------------------------------------------------------------
        sec = "3. work (FLOPs and vector operations)"
        Cv = C_valid if C_valid is not None else Sym.C_valid
        NQ, NK, NV, scope_unknown, _frac = _projection_rows(sem, ragged)     # from the declared scope, independent of the variant graph
        scope_missing = ["projection_scope"] if scope_unknown else []
        scope_label = sem.projection_scope or "full (scenario: projection_scope undeclared)"
        cvm = [] if C_valid is not None else ["C_valid"]
        comps = WORK.useful_components(alg, Cv, NQ, NK, NV)
        for name, e in comps.items():
            M.append(metric(f"{name}[useful]", f"work.{alg}.{name}", e, b, unit="FLOP", scope=f"useful (C = C_valid), FMA=2, projection rows from scope '{scope_label}'", section=sec,
                            algorithm=alg, extra_missing=cvm + scope_missing))
        F_useful = WORK.useful_total(alg, Cv, NQ, NK, NV)
        M.append(metric("F_total[useful]", f"work.{alg}.total", F_useful, b, unit="FLOP",
                        scope=f"useful mathematical work under projection scope '{scope_label}'", section=sec, algorithm=alg,
                        assumptions=[f"N_Qproj={NQ}, N_Kproj={NK}, N_Vproj={NV}", "C already includes B and H", "FMA = 2; not an instruction count"]
                                    + (["absorbed_precomputed: amortized merge work listed separately as F_Wmerge_amortized"] if alg == "absorbed_precomputed" else []),
                        extra_missing=cvm + scope_missing, not_equivalent_to=["scheduled work", "compiled instruction count"]))
        closed = WORK.total_for(alg, Cv, rq, rk)
        closed_id = {"expanded": "F_E", "absorbed_two_step": "F_A", "absorbed_precomputed": "F_A_precomputed"}[alg]
        M.append(metric(f"{closed_id}[spec_closed_form]", f"work.{alg}.closed_form", closed, b, unit="FLOP",
                        scope="specification boxed formula: every Q row and every K/V row projected once (full projection, no cache)", section=sec, algorithm=alg,
                        assumptions=["equals F_total[useful] only when projection_scope = full" if alg != "absorbed_precomputed"
                                     else "precomputed variant: excludes the amortized merge work (listed separately); equals F_total[useful] under full projection"],
                        extra_missing=cvm))
        # spec 5.3: N^proj counts the rows *actually projected within this invocation*.  The absorbed paths
        # consume C_k directly as the K and V operand, so no K/V row is projected there ("when K/V are passed
        # in directly, they are 0 within this scope"); the output projection Z W^v is its own quantity.
        absorbed = alg != "expanded"
        NK_v, NV_v = (sp.Integer(0), sp.Integer(0)) if absorbed else (NK, NV)
        proj_note = ("absorbed path: C_k is consumed directly as both the K and the V operand, so no K/V row is "
                     "projected in this scope; the Z W^v output projection is reported separately"
                     if absorbed else "expanded path: every counted K/V row is projected in this scope")
        for name, expr, what in (("N_Qproj", NQ, "Q rows projected in this invocation"),
                                 ("N_Kproj", NK_v, "K rows projected in this invocation"),
                                 ("N_Vproj", NV_v, "V rows projected in this invocation")):
            ex = g.executed_proj_rows.get(name)
            if ex is not None and str(sp.expand(ex)) != str(sp.expand(expr)):
                # spec 5.2: a strategy that re-projects inside the KV loop, or skips tiles no block scans,
                # makes the executed count differ from the once-per-row scope count; both are reported
                M.append(metric(f"{name}[executed_graph]", f"work.projection.rows.{name}.executed", ex, b, unit="rows",
                                scope=f"{what} as the executed graph performs them, including the KV-loop multiplicity "
                                      "and excluding tiles no query block scans (spec 5.2)", section=sec, algorithm=alg,
                                assumptions=[proj_note], extra_missing=scope_missing + exec_missing,
                                not_equivalent_to=[f"{name}, the once-per-row count of the declared scope"]))
            M.append(metric(name, f"work.projection.rows.{name}", expr, b, unit="rows",
                            scope=f"{what} counted once per row, from the declared projection_scope and the variant's dataflow",
                            section=sec, algorithm=alg,
                            assumptions=[proj_note,
                                         "distinct rows, not row visits: a strategy that re-projects inside the KV loop "
                                         "raises the executed-graph work, not this count (spec 5.2)"],
                            extra_missing=scope_missing,
                            not_equivalent_to=["the tensor's row count", "rows read from HBM",
                                               "the executed-graph projected-row count under in-loop projection"]))
        # spec 5.3's F_proj is the Q/K/V projection of THIS variant's dataflow: the expanded path projects
        # Q^n and K^n/V, the absorbed two-step adds the absorb step Q^n (W^k)^T, and the precomputed variant
        # replaces both Q-side steps with the single merged projection C_q W~.
        if alg == "absorbed_precomputed":
            proj_expr = 2 * NQ * Sym.R_q * Sym.R_k
            proj_scope = "2 N_Qproj R_q R_k: the single merged Q projection C_q W~ this variant performs"
        elif alg == "absorbed_two_step":
            proj_expr = WORK.projection_general(NQ, NK_v, NV_v) + 2 * NQ * Sym.D_n * Sym.R_k
            proj_scope = ("2 N_Qproj R_q D_n + 2 N_Qproj D_n R_k: the Q projection and the absorb step; "
                          "N_Kproj = N_Vproj = 0 in this variant")
        else:
            proj_expr = WORK.projection_general(NQ, NK_v, NV_v)
            proj_scope = "2 N_Qproj R_q D_n + 2 N_Kproj R_k D_n + 2 N_Vproj R_k D_v with row counts from projection_scope"
        M.append(metric("F_proj[general]", "work.projection.general", proj_expr, b, unit="FLOP",
                        scope=proj_scope, section=sec, algorithm=alg,
                        assumptions=[f"N_Qproj={NQ}, N_Kproj={NK_v}, N_Vproj={NV_v}", proj_note,
                                     "useful-work counting: each row projected once. When the strategy projects inside the "
                                     "KV loop, the repeated work is in F_projection[executed_graph] (spec 5.2), not here."],
                        extra_missing=scope_missing,
                        not_equivalent_to=["F_projection[executed_graph], which carries the loop multiplicity"]))
        if vis_ok and policy_known:
            Cr = VIS.rect_cells_expr(sem.mask, strat.rect_policy, strat.subdivide, ragged if ragged else None, scheduled=sched)
            M.append(metric("F_total[rect]", f"work.{alg}.total.rect", WORK.useful_total(alg, Cr, NQ, NK, NV), b, unit="FLOP",
                            scope="rectangular-scheduled work (C = C_rect, no internal padding), same projection rows", section=sec, algorithm=alg,
                            assumptions=[pol_note or "declared rectangle policy",
                                         "projection counted once per row: the executed graph may differ where a tile is re-projected or skipped"],
                            not_equivalent_to=["useful work (C = C_valid)", "executed graph work", "compiled work"],
                            extra_missing=scope_missing))
        gw = WORK.graph_matmul_work(g)
        M.append(metric("F_total[executed_graph]", f"work.{alg}.graph.total", gw["total"], b, unit="FLOP",
                        scope="graph sum of 2MNK*multiplicity over executed cells C_exec (declared extent policy)", section=sec, algorithm=alg,
                        assumptions=g.assumptions, extra_missing=exec_missing))
        for cat, e in gw.items():
            if cat != "total":
                M.append(metric(f"F_{cat}[executed_graph]", f"work.{alg}.graph.{cat}", e, b, unit="FLOP", scope="graph category", section=sec, algorithm=alg, extra_missing=exec_missing))
        ev = self.compile_evidence or {}
        ev_src = [str(ev.get("source"))] if ev.get("source") else []
        # spec 12.1: when the compiler output omits loop bodies or widths the result must stay partial
        ev_omits = list(ev.get("omits") or [])
        ev_partial = bool(ev_omits) or (ev.get("coverage") not in (None, "complete"))
        ev_note = ([f"the compiler output is declared {ev.get('coverage') or 'partial'}"
                    + (f" and omits {ev_omits}" if ev_omits else "")
                    + ": complete dynamic behaviour cannot be inferred from it (spec 12.1)"] if ev_partial else [])
        have = ev.get("compiled_flops") is not None
        M.append(Metric("F_compiled", "work.compiled", "from compilation evidence only", {}, ev.get("compiled_flops"), "FLOP", "compiled instruction-level work",
                        ("partial" if ev_partial else "bound") if have else "unknown", alg, "compiled", section=sec, evidence=ev_src,
                        assumptions=ev_note,
                        missing_fields=([f"compile_evidence.{o}" for o in ev_omits] or (["compile_evidence.coverage"] if ev_partial else []))
                                       if have else ["compile_evidence"],
                        not_equivalent_to=["useful", "rect", "executed_graph"]))
        if ev_partial:
            rep.warnings.append(f"compile evidence declared incomplete (coverage={ev.get('coverage')!r}, omits={ev_omits}): "
                                "dependent metrics stay partial (spec 12.1)")
        # path difference: this variant vs expanded under the same scope rows and the same C
        other_total = WORK.useful_total("expanded", Cv, NQ, NK, NV)
        if alg == "expanded":
            diff_expr = WORK.useful_total("absorbed_two_step", Cv, NQ, NK, NV) - other_total
            diff_scope = "absorbed two-step minus expanded, same projection rows and same C (useful)"
        else:
            diff_expr = F_useful - other_total
            diff_scope = f"{alg} minus expanded, same projection rows and same C (useful)"
        M.append(metric("F_A_minus_F_E", "work.path_difference", diff_expr, b, unit="FLOP", scope=diff_scope, section=sec,
                        assumptions=["computation comparison only, not runtime", "reduces to 2BH Rk(Dn+Dv)(Sq-Sk) + 2C(2Rk-Dn-Dv) for full projection (two-step variant)"],
                        extra_missing=cvm + scope_missing))
        M.append(metric("F_A_minus_F_E[spec_closed_form]", "work.path_difference.closed_form", WORK.path_difference(Cv, rq, rk), b, unit="FLOP",
                        scope="2BH Rk (Dn+Dv)(Sq-Sk) + 2C(2Rk-Dn-Dv): full projection, no cache, two-step variant", section=sec,
                        assumptions=["sign decided by 2Rk vs Dn+Dv in square prefill"], extra_missing=cvm))
        # spec 3.2: for a precomputed W~ the merge work, the R_q x R_k weight storage and the effective reuse
        # count must be listed separately, and the merged cache is invalid after a weight update.
        if alg == "absorbed_precomputed":
            reuse_declared = (sem.cache or {}).get("merged_weight_reuse")
            M.append(metric("F_Wmerge[once]", "work.weight_merge.once", WORK.weight_merge_work(), b, unit="FLOP",
                            scope="the merge W~ = W^q (W^k)^T performed once, before any amortization", section=sec, algorithm=alg,
                            assumptions=["listed separately from the reuse count so it is readable even when the reuse "
                                         "count is undeclared (spec 3.2)",
                                         "the merged cache is invalid after a weight update: preprocessing is not free"],
                            not_equivalent_to=["the amortized share F_Wmerge_amortized[useful]"]))
            M.append(metric("M_Wmerged[stored]", "work.weight_merge.storage", Sym.H * Sym.R_q * Sym.R_k * Sym.s_Wq, b, unit="byte",
                            scope="storage of the precomputed merged weight W~ = W^q (W^k)^T over all heads (R_q x R_k per head)", section=sec,
                            assumptions=["listed separately from the merge work and from the reuse count (spec 3.2)",
                                         "invalid after a weight update: preprocessing is not free"],
                            not_equivalent_to=["W^q and W^k storage, which this replaces only if they are dropped"]))
            M.append(Metric("n_reuse[effective]", "work.weight_merge.reuse", "invocations sharing one merged weight",
                            {}, reuse_declared, "invocations",
                            "effective reuse count over which the merge work is amortized",
                            "bound" if reuse_declared is not None else "unknown", alg, "input", section=sec,
                            assumptions=["declared per task; the merged cache is invalidated by a weight update"],
                            missing_fields=[] if reuse_declared is not None else ["cache.merged_weight_reuse"],
                            not_equivalent_to=["a measured reuse rate"]))
        acc = self.numerics.acceptance
        M.append(Metric("equivalence[expanded vs absorbed]", "numerics.equivalence",
                        "matrix associativity: Q^n (W^k)^T C_k^T = Q^n (C_k W^k)^T", {},
                        "equal over the real numbers", "",
                        "the two paths express the same result in exact real arithmetic", "bound", alg, "definition", section=sec,
                        assumptions=["low-precision reassociation does not guarantee bitwise equality: the operand and "
                                     "accumulator dtypes, the accumulation order and the cast sites all differ between the paths"],
                        not_equivalent_to=["bitwise equality", "an error bound", "a measured deviation"]))
        M.append(Metric("acceptance[declared]", "numerics.acceptance", "caller-declared tolerances per test level", {},
                        dict(acc) if acc else None, "", "numeric acceptance declared with the task",
                        "bound" if acc else "unknown", alg, "input", section=sec,
                        assumptions=["declared only: this model runs no arithmetic, so no tolerance is evaluated here"],
                        missing_fields=[] if acc else ["acceptance"],
                        not_equivalent_to=["a measured error", "a proof that the variant meets it"]))
        vw = WORK.graph_vector_work(g)
        for kind, e in vw["by_kind"].items():
            M.append(metric(f"W_vec[{kind}]", f"work.vector.{kind}", e, b, unit="element-ops", scope="logical count before redundancy elimination; not instructions",
                            section=sec, algorithm=alg, extra_missing=vec_missing + exec_missing + vec_kind_missing.get(kind, []),
                            not_equivalent_to=["vector instruction count", "MXU FLOPs"]))
        for res, e in vw["by_resource"].items():
            M.append(metric(f"W_resource[{res}]", f"work.resource.{res}", e, b, unit="element-ops", scope="per resource class u (for W_u/P_u)", section=sec, algorithm=alg,
                            extra_missing=vec_missing + exec_missing + vec_res_missing.get(res, [])))
        M.append(metric("exp_count_executed", "work.exp.executed", g.exec_cells, b, unit="exp", scope="exp over executed cells (masked included)", section=sec,
                        not_equivalent_to=["C_valid"], extra_missing=exec_missing))
        rep.extras["vector_ops"] = [{"id": op.id, "stage": op.stage, "kind": op.kind, "resource": op.resource, "count": str(op.count),
                                     "conditional_on": op.conditional_on, "exists": op.exists, "description": op.description} for op in g.vector_ops]

        # ---- bytes: interface -------------------------------------------------------------
        sec = "4. bytes: interface and materialization"
        iface = MEM.interface_expressions()
        role_bytes = {}
        for role, e in iface.items():
            if role in ("M_in", "M_O", "M_LSE"):
                continue
            # spec 1.3/6.1: allocation is per tensor.  A role that declares no capacity_shape is charged its own
            # shape extent, not another role's declared capacity.
            role_b = cap_b if task is None else task.capacity_bindings_for(role)
            declares_cap = task is not None and getattr(task.tensors.get(role), "capacity_shape", None)
            if task is None:
                note = "no metadata bound: the allocation extent is symbolic"
            elif task.tensors.get(role) is None:
                note = "role absent (no positional branch, D_r = 0): no allocation"
            elif declares_cap:
                note = "allocation extent declared by this tensor's capacity_shape"
            else:
                note = "no capacity_shape declared for this tensor: its own shape is the allocation extent"
            m_role = make_metric(f"bytes[{role}]", f"bytes.logical.{role}", e, role_b, unit="byte",
                                 scope="logical size at this tensor's own allocation extent, per-tensor dtype", section=sec,
                                 source_type="input", assumptions=[note])
            role_bytes[role] = m_role.value
            M.append(m_role)
        # the sum is over the per-role values, each at that role's own allocation extent, so M_in cannot
        # disagree with the per-tensor rows it is the sum of
        # spec 11.4: the expression and bindings must reproduce the value.  Each role's term is substituted at
        # that role's own extents, so the printed expression evaluates to the same total as the per-role rows.
        m_in_expr = iface["M_in"]
        if task is not None:
            m_in_expr = sp.Add(*[substitute(iface[r], {k: v for k, v in task.capacity_bindings_for(r).items()
                                                        if k in ("S_q", "S_k", "R_q", "R_k", "D_n", "D_r", "D_v", "B", "H")})
                                 for r in iface if r not in ("M_in", "M_O", "M_LSE")])
        m_in = make_metric("M_in", "bytes.M_in", m_in_expr, cap_b, unit="byte",
                           scope="sum of the seven inputs, each at its OWN ALLOCATION extent (capacity_shape where declared)",
                           section=sec,
                           assumptions=["per-tensor storage widths s_x", "aliases not deduplicated in this symbolic sum (see ledger)",
                                        "equals the ledger's allocated_unique_total when no two roles alias"],
                           not_equivalent_to=["HBM traffic lower bound under caching/reuse",
                                              "M_in[metadata_ledger], which sums the LOGICAL shapes rather than the allocations"])
        if role_bytes and all(v is not None for v in role_bytes.values()):
            assert m_in.value is None or abs(m_in.value - sum(role_bytes.values())) < 1e-9, "M_in expression and rows disagree"
        M.append(m_in)
        M.append(metric("M_O", "bytes.M_O", Sym.s_O * rq * Sym.D_v, b, unit="byte", scope="output O at active extents (s_O * B H S_q * D_v; ragged: H sum_b Sq_b)", section=sec,
                        extra_missing=[] if b.get("s_O") else ["output_dtype"]))
        if wl is not False:
            M.append(metric("M_LSE", "bytes.M_LSE", Sym.s_LSE * rq, b, unit="byte", scope="LSE at active extents" + ("" if wl else " (conditional: outputs undeclared)"),
                            section=sec, source_type="conditional" if wl is None else "derived",
                            extra_missing=([] if b.get("s_LSE") else ["lse_dtype"]) + ([] if wl else ["outputs"])))
        ledger = MEM.interface_ledger(task)
        if ledger:
            rep.extras["interface_ledger"] = ledger
            M.append(Metric("M_in[metadata_ledger]", "bytes.ledger.logical_total", "sum of per-tensor prod(shape)*bytes(dtype)", {},
                            ledger["logical_total"], "byte",
                            "sum of the LOGICAL sizes prod(shape)*s from the actual metadata, not the allocations",
                            "bound", alg, "input", section=sec,
                            assumptions=["spec 6.1 M_logical: the declared shape, ignoring any larger capacity_shape"],
                            not_equivalent_to=["M_in, which sums each tensor at its allocation extent",
                                               "M_alloc_unique[metadata_ledger], the allocation total with aliases counted once"]))
            strided = ledger.get("roles_with_unmodelled_strides") or []
            M.append(Metric("M_alloc_unique[metadata_ledger]", "bytes.ledger.alloc_unique", "allocated bytes with aliases counted once", {},
                            ledger["allocated_unique_total"], "byte",
                            "capacity_shape and aliases honoured" + ("; declared strides/offsets not modelled" if strided else ""),
                            "partial" if strided else "bound", alg, "input", section=sec,
                            assumptions=([f"the strides/offset declared for {strided} span more elements than prod(shape), and that "
                                          "span is not folded into the allocation (spec 6.1: the logical element sum is not "
                                          "the union of address ranges); a contiguous declaration needs no such caveat"] if strided else []),
                            missing_fields=[f"{r}.allocated_bytes (declared strides span more than prod(shape))" for r in strided],
                            not_equivalent_to=["the union of address ranges"]))
            if strided:
                rep.warnings.append(f"strides/storage_offset declared for {strided} are recorded but not modelled in the "
                                    "allocation ledger; the logical element sum is not the union of address ranges (spec 6.1)")
        mat_rows = MEM.materialization_ledger(g)
        for row in mat_rows:
            m = metric(f"HBM_mat[{row['tensor']}]", f"bytes.mat.{row['id']}", row["traffic"], b, unit="byte", scope="write + read traffic if materialized (read under the declared scan policy)",
                       section=sec, source_type="conditional", assumptions=[f"conditional on {row['conditional_on']} = {row['enabled']}",
                                                                            f"write traffic {row['write_traffic']}; read traffic {row['read_traffic']}"],
                       extra_missing=[row["conditional_on"]] if row["enabled"] is None else [], not_equivalent_to=["interface bytes"])
            # spec 6.2 names the materialized tensor's own size (M^mat) as well as its traffic; they are
            # different quantities (a tensor read twice moves twice its size) and both are reported.
            sz = metric(f"M_mat[{row['tensor']}]", f"bytes.mat.size.{row['id']}", row["bytes"], b, unit="byte",
                        scope="logical size of the materialized intermediate itself (spec 6.2 M^mat)", section=sec, source_type="conditional",
                        assumptions=[f"conditional on {row['conditional_on']} = {row['enabled']}"],
                        extra_missing=[row["conditional_on"]] if row["enabled"] is None else [],
                        not_equivalent_to=["its HBM traffic", "interface bytes"])
            if row["enabled"] is False:
                for x in (m, sz):
                    x.status, x.value = "not_applicable", 0
                    x.missing_fields = []                                  # a tensor that does not exist depends on no extent
                    x.assumptions = [a for a in x.assumptions if "scenario" not in a]
                    x.coverage = "not applicable under the declared configuration"
                m.notes.append("materialization disabled by the declared strategy: no traffic")
                sz.notes.append("materialization disabled by the declared strategy: the tensor is not materialized")
            M.append(m)
            M.append(sz)
        # spec 6.2: "if materialization does not exist, it must not be counted" — a disabled row reports zero,
        # and the counterfactual keeps a name that cannot be read as reported traffic
        rep.extras["materialization_ledger"] = [
            {"id": r["id"], "tensor": r["tensor"], "conditional_on": r["conditional_on"], "enabled": r["enabled"],
             "bytes": expr_str(r["bytes"]), "write_traffic": expr_str(r["write_traffic"]), "read_traffic": expr_str(r["read_traffic"]),
             "bytes_value": 0 if r["enabled"] is False else evaluate(r["bytes"], b).value,
             "traffic_value": 0 if r["enabled"] is False else evaluate(r["traffic"], b).value,
             "bytes_value_if_materialized": evaluate(r["bytes"], b).value,
             "traffic_value_if_materialized": evaluate(r["traffic"], b).value} for r in mat_rows]

        # ---- bytes: transfers -----------------------------------------------------------------
        sec = "5. bytes: transfer requests per path"
        paths = MEM.transfer_ledger(g)
        for path in ("hbm_to_vmem", "vmem_to_hbm", "vmem_to_vreg", "vreg_to_vmem"):
            row = paths.get(path)
            if row is None:
                M.append(Metric(f"B[{path}]", f"bytes.path.{path}", "sum_r m_r n_r (no modeled events)", {}, None, "byte", "path traffic", "unknown", alg, "conditional",
                                section=sec, missing_fields=["register_schedule_evidence" if "vreg" in path else "transfer_events"],
                                notes=["VREG->VMEM stores/spill traffic cannot be derived from the source-level graph without compiler evidence"] if path == "vreg_to_vmem" else [],
                                not_equivalent_to=["spill traffic", "physical traffic with caching"]))
                continue
            if "vreg" in path:
                extra = ["register_schedule_evidence"] + [f for f in xfer_missing if f in ("rect_policy", "mask", "executed_extent_policy", "heads_per_program")]
            else:
                extra = list(xfer_missing)
            # the ledger already names each event-level condition (an undeclared materialization flag, a
            # tile-padded read); the path total inherits them so it never reads as fully declared
            extra += [f for f in row.get("missing_fields", []) if f not in extra]
            path_assumptions = ["scenario q_outer_kv_inner; weights re-read per program",
                                "no hardware cache effects are modeled: every event is counted at its declared multiplicity"]
            if "vreg" in path:
                path_assumptions.append("operand-streaming scenario without register-schedule evidence")
            split_events = [t.id for t in g.transfers if t.path == path and t.scenario and "cached" in (t.scenario or "")]
            scen_events = [t.id for t in g.transfers if t.path == path and t.undeclared and t.scenario]
            if scen_events:
                path_assumptions.append(
                    f"includes scenario events whose condition is undeclared ({', '.join(scen_events)}): "
                    "they are counted at their declared multiplicity and named in missing_fields")
            if split_events:
                path_assumptions.append(
                    f"includes a cached/new key split scenario ({', '.join(split_events)}): cached and newly projected keys are "
                    "assumed to be read in proportion to their counts")
            M.append(metric(f"B[{path}]", f"bytes.path.{path}", row["total"], b, unit="byte", scope="sum_r m_r n_r over the scenario's transfer events", section=sec,
                            source_type="conditional", assumptions=path_assumptions,
                            extra_missing=sorted(set(extra)), not_equivalent_to=["physical HBM traffic with caching", "spill traffic"]))
        rep.extras["transfer_events"] = [{"id": t.id, "path": t.path, "bytes_per_event": str(t.bytes_per_event), "executions": str(t.executions),
                                          "total": expr_str(sp.expand(t.total)), "value": evaluate(t.total, b).value, "description": t.description,
                                          "scenario": t.scenario, "undeclared": t.undeclared, "conditional_on": t.conditional_on} for t in g.transfers]

        # ---- local objects ----------------------------------------------------------------------
        sec = "6. local objects (symbolic sizes, spec 7.1)"
        for obj in g.locals:
            if obj.exists is False:
                continue
            obj_fallbacks = [f for f in dtype_fallback_missing if obj.bytes_symbol in FALLBACK_SYMBOLS[f]]
            nb = "" if str(obj.buffers) == "1" else f"; residency window of n_buf={obj.buffers} buffers (7.3 allocation, not the 7.1 logical size)"
            m = make_metric(f"local[{obj.id}]", f"local.{obj.id}.bytes", obj.size, b, unit="byte",
                            scope=f"one logical object{' x its buffer count' if nb else ''}; level={obj.level}; live in {list(obj.live_stages)}{nb}",
                            section=sec,
                            assumptions=[obj.description]
                                        + ([f"exists only if {obj.conditional_on} (undeclared)"] if obj.exists is None else [])
                                        + ([f"{obj.size_scenario} undeclared: the full rank window is shown as the scenario"] if obj.size_scenario else [])
                                        + ([f"{obj.stage_scenario} undeclared: the live-stage set printed in the scope is the scenario"]
                                           if obj.stage_scenario else []),
                            not_equivalent_to=["VREG allocation", "spill traffic"] + (["the 7.1 logical size of one buffer"] if nb else []),
                            notes=([f"alias group {obj.alias_group}"] if obj.alias_group else [])
                                  + ([f"logical size of one buffer: {obj.size / obj.buffers}"] if nb else []),
                            extra_missing=([obj.conditional_on] if (obj.exists is None and obj.conditional_on) else [])
                                          + obj_fallbacks + ([obj.size_scenario] if obj.size_scenario else [])
                                          + ([obj.stage_scenario] if obj.stage_scenario else []))
            M.append(m)
        tiles_by_symbol = {}
        if self.hardware:
            for sym_name, dt in self._dtype_by_symbol().items():
                t = self.hardware.layout_tile(dt)
                if t:
                    tiles_by_symbol[sym_name] = t
        cov = MEM.layout_coverage(g, strat.layout or {}, tiles_by_symbol)
        for row in cov:
            if row.get("not_in_this_variant"):
                # the coverage row says why the object is missing: declared away in this variant, or owned by another
                reason = row.get("reason") or "no variant builds this object here; another variant does"
                declared_away = "declared configuration" in reason
                M.append(Metric(f"layout[{row['object']}]", f"layout.{row['object']}",
                                "declared for an object this variant does not build", {}, None, "byte",
                                f"layout kind '{row['kind']}' declared, but the {alg} graph "
                                + ("does not build this object under the declared configuration" if declared_away
                                   else "has no object with this id"),
                                "not_applicable", alg, "input", section=sec,
                                assumptions=[reason + ": a legal declaration, recorded, and applied to nothing here"],
                                not_equivalent_to=["an unknown layout"]))
                continue
            if row["coverage"] is None:
                M.append(Metric(f"layout[{row['object']}]", f"layout.{row['object']}", "replicated layout: coverage depends on replication factor", {}, None, "byte",
                                "layout coverage", "unknown", alg, "conditional", section=sec, missing_fields=["replication_factor"]))
                continue
            M.append(make_metric(f"layout[{row['object']}]", f"layout.{row['object']}", row["coverage"], b, unit="byte",
                                 scope=f"layout coverage, kind={row['kind']}, level={row['level']}" + ("" if row["declared"] else " (scenario: kind undeclared)"), section=sec,
                                 source_type="conditional",
                                 assumptions=["LayoutBytes = s L_s L_l ceil(m/L_s) ceil(n/L_l) with profile tiles" if row["kind"] == "vector_tiled"
                                              else ("declared as a byte count, not an [m, n] element array: the 7.2 layout function does not apply"
                                                    if row["kind"] == "raw_bytes" else "compact layout: logical size")],
                                 extra_missing=[] if row["declared"] else [f"layout.{row['object']}"], not_equivalent_to=["register-file capacity"]))

        # ---- pressure ------------------------------------------------------------------------------
        sec = "7. stage pressure and envelope"
        for level in (None, "vreg", "vmem"):
            tag = "all" if level is None else level
            sets = MEM.stage_live_sets(g, level=level)
            aliasing_missing = aliasing_for(level)
            for st, (e, ids, undecl) in sets.items():
                M.append(make_metric(f"M_live[{tag}:{st}]", f"pressure.live.{tag}.{st}", e, b, unit="byte",
                                     scope=f"source-level live set at this stage cut, level={tag} (aliases deduplicated)", section=sec,
                                     assumptions=["objects: " + ", ".join(ids)] + ([f"includes objects whose existence depends on undeclared fields: {undecl}"] if undecl else []),
                                     extra_missing=list(undecl) + aliasing_missing,
                                     not_equivalent_to=["compiler register allocation", "VMEM allocation"]))
            peak_undecl = sorted({u for (_, _, und) in sets.values() for u in und})
            M.append(make_metric(f"M_peak[{tag}]", f"pressure.peak.{tag}", MEM.peak_live(g, level=level, sets=sets), b, unit="byte", scope=f"max over stage cuts, level={tag} (source level)",
                                 section=sec, extra_missing=list(peak_undecl) + aliasing_missing,
                                 not_equivalent_to=["actual spill", "OOM proof"]))
        env_rows = {}
        envs_by_level = {}
        for level in (None, "vreg", "vmem"):
            tag = "all" if level is None else level
            envs = envs_by_level[level] = PRESS.all_envelopes(g, level=level)
            aliasing_missing = aliasing_for(level)
            for st, env in envs.items():
                env_rows[f"{tag}:{st}"] = env.to_dict(b)
                M.append(make_metric(f"envelope[{tag}:{st}]", f"pressure.envelope.{tag}.{st}", env.expression, b, unit="byte",
                                     scope=f"a*b_q*b_k + b*b_q + c*b_k + d (+ remainder) for this stage, level={tag}", section=sec,
                                     assumptions=[f"a from {env.provenance['a']}", f"b from {env.provenance['b']}", f"c from {env.provenance['c']}", f"d from {env.provenance['d']}"]
                                                 + ([f"non-polynomial remainder from {env.provenance['remainder']}"] if env.remainder != 0 else []),
                                     extra_missing=list(env.undeclared) + aliasing_missing,
                                     not_equivalent_to=["measured spill", "spill probability"]))
        rep.extras["envelope_coefficients"] = env_rows

        # spec 10.3 "delay V load": report the V tile's lifetime overlap and whether prefetch is lost.
        # Without source/schedule evidence this is a candidate hypothesis, not a located bug.
        v_obj = next((o for o in g.locals if o.id == "v_tile" and o.exists is not False), None)
        v_missing = ["v_load"] if strat.v_load is None else []
        hyp = ("a candidate hypothesis under the declared lifetime model, not a located bug: no source or "
               "schedule evidence is attached")
        if v_obj is None:
            for name, fid, sc in (("V_lifetime_stages", "lifetime.v_tile.stages", "number of stage cuts the V tile is live at"),
                                  ("V_prefetch_overlapped", "lifetime.v_tile.prefetch", "V read/production overlapped with the score matmul")):
                M.append(Metric(name, fid, "|{s : v_tile live at s}|" if "stages" in fid else "v_tile live at the score stage",
                                {}, None, "stages" if "stages" in fid else "bool", sc, "not_applicable", alg, "derived", section=sec,
                                assumptions=["this variant has no separate V tile: C_k is both the K and the V operand"],
                                not_equivalent_to=["measured prefetch behaviour"]))
        else:
            live = list(v_obj.live_stages)
            M.append(Metric("V_lifetime_stages", "lifetime.v_tile.stages", "|{s : v_tile live at s}|", {}, len(live), "stages",
                            "number of stage cuts the V tile is live at, under the declared V-load policy",
                            "partial" if v_missing else "bound", alg, "derived", section=sec,
                            assumptions=[f"V live at {live}", hyp], missing_fields=v_missing,
                            not_equivalent_to=["measured V residency", "compiler scheduling"]))
            M.append(Metric("V_prefetch_overlapped", "lifetime.v_tile.prefetch", "v_tile live at the score stage",
                            {}, "score" in live, "bool",
                            "whether the V read/production is overlapped with the score matmul under the declared policy",
                            "partial" if v_missing else "bound", alg, "derived", section=sec,
                            assumptions=[("early load: the V transfer overlaps the score matmul" if "score" in live
                                          else "delayed load: the V transfer is exposed at the PV stage, prefetch is given up"), hyp],
                            missing_fields=v_missing, not_equivalent_to=["measured prefetch behaviour", "a compiler schedule"]))
        # spec 7.4 states the envelope and the b_k bound per stage s, so every stage gets its own bound and
        # the binding stage (the smallest bound) is named: reporting only the softmax stage would overstate
        # the feasible b_k whenever another stage retains more per-cell data.
        feas_rows = {}
        binding_rows = {}
        for level, Rsym, budget_field in (("vreg", Sym.R_vreg, "vreg_budget_bytes"), ("vmem", Sym.R_vmem, "vmem_budget_bytes")):
            R_val = b.get(Rsym.name)
            aliasing_missing = aliasing_for(level)
            stage_bounds = {}
            unevaluated: dict[str, list[str]] = {}
            over_budget: dict[str, float] = {}
            shaped: set[str] = set()
            for st in g.stages:
                env_s = envs_by_level[level][st]
                bound_expr, denom = env_s.feasible_bk(Rsym)
                # spec 7.4 states the bound only for a*b_q + c > 0; a stage with no b_k term does not constrain
                # b_k at all, and dividing by its zero denominator would publish sympy's complex infinity
                denom_v = numeric_value(substitute(denom, b))
                if denom_v is not None and denom_v <= 0:
                    # no b_k term here, but the stage can still exceed the budget on its own
                    exc = numeric_value(substitute(MEM.capacity_constraint(env_s.expression, Rsym), b))
                    if exc is not None and exc > 0:
                        over_budget[st] = exc
                    M.append(Metric(f"b_k_max[{level}:{st}]", f"pressure.feasible.{level}.{st}.b_k",
                                    "floor((R - b b_q - d)/(a b_q + c)), stated only for a b_q + c > 0", {}, None, "rows",
                                    f"the {st} stage at level {level} has no b_k term (a b_q + c = {denom_v}), so it places no bound on b_k",
                                    "not_applicable", alg, "derived", section=sec,
                                    assumptions=["this stage's live set does not grow with b_k; the binding bound comes from another stage"]
                                                + ([f"the absence of a b_k term rests on an object list shaped by {env_s.undeclared}"]
                                                   if env_s.undeclared else []),
                                    missing_fields=list(env_s.undeclared),
                                    not_equivalent_to=["an unbounded b_k: other stages still constrain it"]))
                else:
                    M.append(make_metric(f"b_k_max[{level}:{st}]", f"pressure.feasible.{level}.{st}.b_k", bound_expr, b, unit="rows",
                                         scope=f"floor((R - b b_q - d)/(a b_q + c)) for the {st} stage at level {level} under the declared budget {Rsym.name}", section=sec,
                                         assumptions=["valid only when a*b_q + c > 0", "capacity constraint under the declared source-level model, not a compiled spill/OOM criterion"],
                                         extra_missing=([] if R_val is not None else [budget_field]) + list(env_s.undeclared) + aliasing_missing))
                M.append(make_metric(f"excess_over_budget[{level}:{st}]", f"pressure.capacity_constraint.{level}.{st}", MEM.capacity_constraint(env_s.expression, Rsym), b, unit="byte",
                                     scope=f"max(0, M_live - R) at the {st} stage, level {level}: data that cannot stay at the level under the source-level model", section=sec,
                                     assumptions=["source-level logical live set; layout coverage not applied; not a compiled graph cut (spec 8.2 caveat)"],
                                     not_equivalent_to=["spill write traffic", "compiled live set"],
                                     extra_missing=([] if R_val is not None else [budget_field]) + list(env_s.undeclared) + aliasing_missing))
                if R_val is not None:
                    # spec 10.2: a constraint surface, not one point — the bound over a b_q sweep, with the
                    # declared b_q marked, so the caller sees the trade-off instead of a single best tile
                    sweep_limit = b.get("S_q") if not ragged else max(ragged["Sq"])     # [M8] the largest sequence in a ragged batch
                    rows = PRESS.feasible_region(env_s, b, Rsym, R_val, PRESS.bq_sweep(strat.b_q, limit=sweep_limit),
                                                 extra_undeclared=aliasing_missing)
                    feas_rows.setdefault(level, {})[st] = rows
                    declared_row = next((r for r in rows if r.get("declared_b_q")), None)
                    v = declared_row.get("b_k_max") if declared_row else None
                    if v is not None:
                        stage_bounds[st] = v
                        # fields that shaped an evaluated stage bound also shape the minimum over stages
                        shaped.update(env_s.undeclared)
                    elif not (declared_row and str(declared_row.get("status", "")).startswith("not_applicable")):
                        # a stage whose own bound could not be evaluated may be the tighter one
                        unevaluated[st] = sorted(set((declared_row.get("missing") if declared_row else None) or []) | set(env_s.undeclared))
            # the minimum is only a true minimum over the stages that could be evaluated
            miss_min = sorted({f for fs in unevaluated.values() for f in fs} | shaped | set(aliasing_missing)
                              | ({budget_field} if R_val is None else set()))
            if stage_bounds:
                st_min = min(stage_bounds, key=lambda k: stage_bounds[k])
                binding_rows[level] = {"binding_stage": st_min, "b_k_max": stage_bounds[st_min], "per_stage": stage_bounds,
                                       "stages_not_evaluated": sorted(unevaluated),
                                       "stages_over_budget_independently_of_b_k": sorted(over_budget)}
                M.append(Metric(f"b_k_max[{level}:binding_stage]", f"pressure.feasible.{level}.binding",
                                "min over stages s of floor((R - b_s b_q - d_s)/(a_s b_q + c_s))", {}, stage_bounds[st_min], "rows",
                                f"tightest per-stage capacity bound at level {level}; binding stage: {st_min}",
                                "partial" if (unevaluated or miss_min) else "bound", alg, "derived", section=sec,
                                assumptions=[f"per-stage bounds: {stage_bounds}"]
                                            + ([f"stages whose own bound is not evaluable, so a tighter one may exist: {sorted(unevaluated)}"] if unevaluated else [])
                                            + ([f"stages over the budget for every b_k (no b_k term of their own): {sorted(over_budget)}; "
                                                "no b_k makes the configuration feasible"] if over_budget else []),
                                missing_fields=miss_min, not_equivalent_to=["compiled spill/OOM criterion"]))
            else:
                # without a budget no stage was evaluated, but the fields that shape every stage's bound still
                # shape the minimum: the two surfaces must list the same blockers
                shaped_all = {f for st in g.stages for f in envs_by_level[level][st].undeclared} | set(aliasing_missing)
                blockers = sorted(set(miss_min) | shaped_all | ({budget_field} if R_val is None else set())
                                  | ({"b_q"} if strat.b_q is None else set()))
                M.append(Metric(f"b_k_max[{level}:binding_stage]", f"pressure.feasible.{level}.binding",
                                "min over stages s of floor((R - b_s b_q - d_s)/(a_s b_q + c_s))", {}, None, "rows",
                                f"tightest per-stage capacity bound at level {level}", "unknown", alg, "derived", section=sec,
                                missing_fields=blockers, not_equivalent_to=["compiled spill/OOM criterion"]))
        if feas_rows:
            rep.extras["feasible_region_by_stage"] = feas_rows
        if binding_rows:
            rep.extras["feasible_region_binding"] = binding_rows
        ev_map = {"M_spill_peak": "spill_peak_bytes", "B_spill_fill": "spill_fill_bytes", "dT_spill": "spill_exposed_s",
                  "N_spill_instructions": "spill_instructions"}
        for name, (expr_text, unit, scope, miss) in MEM.spill_definitions().items():
            val = ev.get(ev_map.get(name, "")) if ev else None
            # spec 8.1: the exposed-time counterfactual is meaningless without a legal, clearly defined
            # comparison graph, so a bare number stays partial with that requirement still named
            still_missing = [f for f in miss if f == "declared comparison graph" and not (ev or {}).get("spill_comparison_graph")]
            if ev_partial:
                # spec 12.1: an output declared to omit loop bodies or widths cannot establish complete
                # dynamic spill for the invocation, so a number read from it stays partial
                still_missing = sorted(set(still_missing) | {f"compile_evidence.{o}" for o in ev_omits}
                                       or {"compile_evidence.coverage"})
            status = "unknown" if val is None else ("partial" if still_missing else "bound")
            M.append(Metric(name, f"spill.{name}", expr_text, {}, val, unit, scope, status, alg,
                            "measured" if (name in ev_map and val is not None) else "definition", section=sec,
                            evidence=ev_src if val is not None else [],
                            assumptions=(["the comparison graph is not declared, so this difference is a restricted-model "
                                          "figure and must not be presented as an achievable speedup"]
                                         if (val is not None and "declared comparison graph" in miss and not (ev or {}).get("spill_comparison_graph")) else [])
                                        + (ev_note if val is not None else []),
                            missing_fields=(miss if val is None else still_missing),
                            not_equivalent_to=["each other", "pressure envelope"]))

        # ---- sensitivity -------------------------------------------------------------------------
        sec = "8. sensitivity and schedule counts"
        sens_caveat = ("a zero derivative is the absence of a *direct* dependence in the source-level logical size; "
                       "scheduling and buffer lifetime can still create an indirect effect (spec 10.1)")
        sens_exprs = {"score_bytes": Sym.b_q * Sym.b_k * Sym.s_X, "accumulator_bytes": Sym.b_q * g.accumulator_width * Sym.s_A}
        for name, ders in PRESS.sensitivities(sens_exprs).items():
            for var, d in ders.items():
                zero = (d == 0)
                M.append(make_metric(f"d({name})/d({var})", f"sens.{name}.{var}", d, b, unit="byte per row",
                                     scope="continuous relaxation of the source-level logical size", section=sec,
                                     assumptions=[sens_caveat] if zero else [],
                                     not_equivalent_to=(["an absence of any effect of " + var] if zero else [])
                                                      + ["a discrete tile change"],
                                     notes=["discrete tile changes must be compared by finite differences (compare())"]))
        # the program count and the visit count are defined by the loop nest, so an undeclared nest is named
        sched_field = ["schedule"] if strat.schedule is None else []
        M.append(metric("n_programs", "sched.programs", g.n_programs, b, unit="programs",
                        scope="(batch, head-group, q-block) programs under the declared loop nest", section=sec,
                        extra_missing=(["heads_per_program"] if strat.heads_per_program is None else []) + sched_field))
        M.append(metric("n_kv_block_visits", "sched.kv_visits", g.n_kv_visits, b, unit="visits", scope="selected (b, h, q-block, kv-block) rectangles", section=sec,
                        extra_missing=[f for f in ("rect_policy", "mask") if f in exec_missing] + sched_field))

        # ---- performance ----------------------------------------------------------------------------
        sec = "9. hardware lower bounds and calibration"
        # the mxu bound divides the executed graph work, so it inherits whatever that work rests on
        exec_missing = sorted(set(exec_missing) | set(active_missing))
        vec_missing = sorted(set(vec_missing) | set(active_missing))
        work_by_res = {"mxu": gw["total"]}
        work_by_res.update(vw["by_resource"])
        lb = PERF.resource_lower_bounds(g, work_by_res, paths, b, self.hardware, num.matmul_input_dtype, strat.overlap_model, strat.serial_stage_groups,
                                        work_undeclared=exec_missing, vector_undeclared=vec_missing + exec_missing,
                                        path_undeclared=sorted(set(xfer_missing) | set(active_missing) | set(sched_missing)))
        rep.extras["resource_lower_bounds"] = lb
        for row in lb["bounds"]:
            M.append(Metric(f"T_LB[{row['resource']}]", f"perf.lb.{row['resource']}", row["expression"], {}, row["value"], "s", f"scope={row['scope']}", row["status"], alg,
                            "derived", section=sec, missing_fields=row["missing"], notes=row["notes"], not_equivalent_to=["actual latency", "calibrated prediction"]))
        combined_missing = lb.get("combined_missing_fields", [])
        if strat.overlap_model is None:
            for name, sc in lb["scenarios"].items():
                M.append(Metric(f"T_LB[combined:scenario={name}]", f"perf.lb.combined.{name}", sc["expression"], {}, sc["value"], "s", f"overlap scenario {name}",
                                "partial" if sc["value"] is not None else "unknown", alg, "derived", section=sec,
                                missing_fields=sorted(set(combined_missing) | {"overlap_model"}), notes=lb["notes"],
                                not_equivalent_to=["calibrated prediction", "measured time"]))
        else:
            val = lb["combined_value_known_terms_only"]
            M.append(Metric("T_LB[combined]", "perf.lb.combined", lb["combined_expression"], {}, val, "s",
                            f"overlap model {lb['overlap_model']}", lb["status"] if val is not None else "unknown", alg, "derived",
                            section=sec, missing_fields=combined_missing, notes=lb["notes"],
                            not_equivalent_to=["calibrated prediction", "measured time"]))
        cal = PERF.calibrated_prediction(g, b, self.hardware, num.matmul_input_dtype, b.get("T_launch"), b.get("eps_model"),
                                         executed_extent_policy=strat.executed_extent_policy,
                                         work_undeclared=sorted(set(exec_missing) | set(vec_missing)),
                                         class_dtypes={"exp": num.exp_dtype, "reduce": num.score_dtype,
                                                       "vpu": num.state_dtype or num.score_dtype, "layout": None})
        rep.extras["calibrated_prediction"] = cal
        cal_missing = list(cal.get("missing_fields", []))
        cal_value = cal.get("node_interval_s")
        cal_status = "unknown" if cal_value is None else ("partial" if cal_missing else "bound")
        cal_scope = cal.get("scheduler") or "no applicable calibration: no prediction issued"
        cal_coverage = ("complete w.r.t. the node classes covered by the applicable buckets; every uncovered class is listed"
                        if cal_value is not None else "incomplete: no applicable calibration evidence attached")
        M.append(Metric("T_pred[node_interval]", "perf.calibrated.nodes",
                        "sum_v W_v/(eps_v P_v) + t_startup, + T_launch + eps_model", {},
                        cal_value, "s", cal_scope, cal_status, alg, "derived", section=sec,
                        notes=cal["notes"] + [f"calibration state: {cal['status']}",
                                              "derived: this graph's work divided by a measured efficiency interval; the measured "
                                              "inputs are the epsilon buckets, not this latency"], coverage=cal_coverage,
                        missing_fields=cal_missing or ([] if cal_value is not None else ["calibration"]),
                        not_equivalent_to=["T_LB", "hardware-peak lower bound"]))
        rep.mode = "calibrated" if cal.get("applied") else ("symbolic" if self.mode == "symbolic" else "bound")
        if self.hardware is not None and self.hardware.calibration and not cal.get("applied"):
            rep.warnings.append("calibration buckets attached but none applies to this task: mode stays 'bound'")

        # ---- dependency graph (spec 2: symbolic mode returns the dependency structure) -----------------------
        rep.extras["dependency_graph"] = {
            "metric_symbols": {m.metric: free_symbol_names(_expr_of(m)) for m in M if _expr_of(m) is not None},
            "matmul_nodes": [{"id": n.id, "stage": n.stage, "M": str(n.M), "N": str(n.N), "K": str(n.K), "multiplicity": str(n.multiplicity),
                              "tile_shape": None if n.tile_shape is None else [str(x) for x in n.tile_shape],
                              "tile_shape_value": None if n.tile_shape is None else [evaluate(x, b).value for x in n.tile_shape],
                              "counts_toward": n.counts_toward} for n in g.matmuls],
            "local_objects": [{"id": o.id, "size": str(o.size), "level": o.level, "live_stages": list(o.live_stages),
                               "alias_group": o.alias_group, "exists": o.exists, "conditional_on": o.conditional_on,
                               "size_scenario": o.size_scenario, "stage_scenario": o.stage_scenario} for o in g.locals],
            "transfer_events": [{"id": t.id, "path": t.path, "total": str(t.total), "undeclared": t.undeclared} for t in g.transfers],
            "stages": list(g.stages),
        }
        # ---- constraints ---------------------------------------------------------------------------------
        rep.constraints = [c.to_dict(b) for c in self.constraints()]
        for c in rep.constraints:
            if c["holds"] is False and c["severity"] == "error":
                rep.conflicts.append(Conflict(c["name"], {"relation": c["relation"]}, c["description"], severity="error"))
        # spec 10.3: quantities this variant provably lacks are emitted as explicit not_applicable metrics with no
        # value, so a comparison can read "disappeared" from the row itself instead of guessing from other rows
        present = {m.metric for m in M}
        for name, fid, reason in g.absent_metrics:
            if name in present:
                continue
            M.append(Metric(name, fid, "not formed in this variant", {}, None, "", "structural absence in this algorithm path",
                            "not_applicable", alg, "definition", section="6. local objects (symbolic sizes, spec 7.1)",
                            assumptions=[reason], not_equivalent_to=["an unknown value"]))
        # spec 11.4 / the implementation prompt: every metric states the algorithm path it belongs to.  Metrics
        # that hold for every variant (task dimensions, visibility, interface bytes) say so explicitly rather
        # than leaving the field blank, which would read as "not recorded".
        for m in M:
            if m.algorithm is None:
                m.algorithm = "all variants"
        # ---- unknowns ---------------------------------------------------------------------------------------
        missing = {}
        for m in M:
            for f in m.missing_fields:
                missing.setdefault(f, []).append(m.metric)
        for f, metrics in sorted(missing.items()):
            rep.unknowns.append(f"{f}: needed by {len(metrics)} metric(s), e.g. {metrics[:3]}")
        return rep

    def _analyze_algorithm_scenarios(self) -> Report:
        """Report with the algorithm preserved as unknown (spec 11.1), as explicitly named variant scenarios.

        Metrics that every variant agrees on (task, visibility, interface bytes) appear once; those that differ,
        or that only some variants have, appear once per variant as ``metric [algorithm=<a>]`` with the field
        named missing.  Nothing is chosen on the caller's behalf.
        """
        reps = {a: self.with_algorithm(a).analyze() for a in ALGORITHMS}
        base = reps[ALGORITHMS[0]]
        merged = Report(title=f"MLA forward analytical report (algorithm undeclared: {len(ALGORITHMS)} scenarios, "
                              f"strategy={self.strategy.name})", mode=base.mode, adapter=self.adapter_name,
                        algorithm=None, task_fingerprint=self.fingerprint())
        merged.header = dict(base.header)
        merged.header["algorithm"] = {"declared": False, "scenarios": list(ALGORITHMS)}
        names = []
        for a in ALGORITHMS:
            for m in reps[a].metrics:
                if m.metric not in names:
                    names.append(m.metric)
        for name in names:
            per = {a: reps[a].get(name) for a in ALGORITHMS}
            present = [a for a in ALGORITHMS if per[a] is not None]
            same = (len(present) == len(ALGORITHMS)
                    and len({(str(per[a].value), per[a].status, per[a].expression) for a in ALGORITHMS}) == 1)
            if same:
                # identical in every variant: one metric, labelled as holding for all of them (never blank,
                # which would read as "algorithm not recorded")
                merged.metrics.append(replace(per[present[0]], algorithm="all variants"))
                continue
            for a in present:
                src = per[a]
                merged.metrics.append(replace(
                    src, metric=f"{src.metric} [algorithm={a}]", algorithm=a,
                    status="partial" if src.status == "bound" else src.status,
                    missing_fields=sorted(set(src.missing_fields) | {"algorithm"}),
                    coverage="", assumptions=list(src.assumptions) + [f"scenario: algorithm = {a} (undeclared)"]))
        seen_conflicts = set()
        for a in ALGORITHMS:
            merged.warnings.extend(f"[{a}] {w}" for w in reps[a].warnings)
            for c in reps[a].conflicts:
                # a conflict every variant reports identically is one conflict, not three
                key = (c.field, c.severity, c.message, tuple(sorted((str(k), str(v)) for k, v in (c.sources or {}).items())))
                if key in seen_conflicts:
                    continue
                seen_conflicts.add(key)
                merged.conflicts.append(c)
            merged.extras[f"scenario:{a}"] = reps[a].extras
        merged.constraints = base.constraints
        merged.unknowns = sorted({u for a in ALGORITHMS for u in reps[a].unknowns})
        merged.warnings.insert(0, "algorithm undeclared: every variant-dependent metric is reported once per named "
                                  "variant scenario and carries 'algorithm' in missing_fields (spec 11.1)")
        return merged

    def _dtype_by_symbol(self) -> dict[str, str]:
        out = {}
        if self.task:
            for role, sym in ROLE_TO_BYTES_SYMBOL.items():
                if role in self.task.dtypes:
                    out[sym] = self.task.dtypes[role]
        dts, _ = self.numerics.resolved_dtypes()
        out.update(dts)
        return out

    def constraints(self) -> list[Constraint]:
        """Mathematical legality constraints.  Only constraints whose fields are declared are emitted."""
        s = self.strategy
        cs = [
            Constraint("q_tail", sp.Eq(sp.Mod(Sym.S_q, Sym.b_q), 0), "S_q divisible by b_q (False = a tail q-block exists; legal, affects padding)", severity="info"),
            Constraint("kv_tail", sp.Eq(sp.Mod(Sym.S_k, Sym.b_k), 0), "S_k divisible by b_k (False = a tail kv-block exists)", severity="info"),
            Constraint("Dr_zero_or_positive", Sym.D_r >= 0, "D_r = 0 is the explicit no-positional-branch scenario", severity="info"),
        ]
        if s.b_rq is not None:
            cs.append(Constraint("b_rq_le_R_q", Sym.b_rq <= Sym.R_q, "Q rank sub-tile must not exceed R_q", sources=["implementation.b_rq", "tensor.q_latent"]))
        if s.b_rk is not None:
            cs.append(Constraint("b_rk_le_R_k", Sym.b_rk <= Sym.R_k, "KV rank sub-tile must not exceed R_k", sources=["implementation.b_rk", "tensor.kv_latent"]))
        if s.heads_per_program is not None:
            cs.append(Constraint("h_pp_le_H", Sym.h_pp <= Sym.H, "heads per program must not exceed H", sources=["implementation.heads_per_program"]))
        if s.subdivide:
            cs.append(Constraint("subdivide_le_tile", sp.And(sp.Integer(s.subdivide[0]) <= Sym.b_q, sp.Integer(s.subdivide[1]) <= Sym.b_k),
                                 "sub-tiles must not exceed the tile", sources=["implementation.subdivide"]))
        if s.kv_buffers is not None:
            cs.append(Constraint("kv_buffers_ge_1", Sym.n_buf >= 1, "buffer count must be positive", sources=["implementation.kv_buffers"]))
        return cs

    # ------------------------------------------------------------------ comparison
    def compare(self, candidates, objectives=("B[hbm_to_vmem]", "W_resource[reduce]", "rect_waste",
                                              "M_live[vreg:softmax]", "M_live[vmem:softmax]", "n_programs")) -> dict:
        """Compare candidate strategies (and optionally an algorithm switch) for the *same* task (spec 10, 12.3).

        A dict candidate is a *patch* on the baseline strategy: only the fields it names change, so
        {"name": "bq8", "b_q": 8} is exactly the "reduce the Q subtile" study of spec 10.3 and every other
        declaration is held fixed.  (Without this, an omitted field would silently become undeclared and the
        required change-reports would read "unknown" instead of showing the change.)  Pass an
        ExecutionStrategy object to replace the strategy wholesale instead.  An optional 'algorithm' key
        switches the variant ("switch to absorbed"); any other key — task dims, semantics, numerics —
        raises TaskIdentityError.  Returns finite differences vs the baseline, legality / feasibility labels
        per candidate, the structural appear/disappear diff and MXU tile-shape changes, a Pareto set over the
        mathematically legal candidates, and a calibrated ranking only when calibration applied and the
        intervals separate.
        """
        if self.algorithm is None:
            raise TaskIdentityError(
                "compare() needs a declared algorithm: with the variant preserved as unknown the report holds one metric "
                "per named variant scenario, so there is nothing to difference against. Bind with an explicit algorithm "
                "(a candidate's 'algorithm' key still expresses the switch study), or compare each scenario separately.")
        base_rep = self.analyze()
        fp = self.fingerprint()
        rows = []
        watched = ["F_total[useful]", "F_total[executed_graph]", "F_A_minus_F_E", "C_rect", "C_pad", "rect_waste", "B[hbm_to_vmem]", "B[vmem_to_hbm]",
                   "HBM_mat[K^n,V]", "local[score_X]", "local[accumulator_A]", "local[v_tile]", "M_live[vreg:softmax]", "M_live[vmem:softmax]",
                   "M_peak[all]", "n_programs", "n_kv_block_visits", "V_lifetime_stages",
                   "W_resource[exp]", "W_resource[reduce]", "b_k_max[vreg:binding_stage]",
                   "T_LB[mxu]", "T_LB[path:hbm_to_vmem]", "T_LB[combined]",
                   "T_LB[combined:scenario=full_overlap_max]", "T_LB[combined:scenario=serial_stage_sum]", "T_pred[node_interval]"]
        if (isinstance(objectives, (str, bytes)) or not isinstance(objectives, (list, tuple))
                or not all(isinstance(o, str) for o in objectives)):
            raise MetadataError("objectives must be a list of metric names")
        # an objective the caller names is collected too, so naming one outside the watched set is not dropped
        watched = list(dict.fromkeys(watched + list(objectives)))
        strategy_fields = set(ExecutionStrategy.__dataclass_fields__)

        def rep_names(rep):
            return {m.metric for m in rep.metrics}

        def one(rep, k):
            # spec 10.3 asks for the *disappearance* of an intermediate on an algorithm switch, so "this variant
            # has no such quantity" and "the value is unknown" must read differently.  Absence is a property of
            # the row's own report: a not_applicable metric with no value (every variant emits those for the
            # quantities it provably lacks).  A name the report simply does not carry is unknown, never absent.
            m = rep.get(k)
            if m is None:
                return None
            return ABSENT if (m.status == "not_applicable" and m.value is None) else m.value

        def values(rep, _unused=None):
            return {k: one(rep, k) for k in watched}

        def statuses(rep):
            return {k: (rep.get(k).status if rep.get(k) is not None else None) for k in watched}

        def structure(rep):
            """Intermediates a variant creates: local objects and conditional HBM materializations."""
            dg = rep.extras.get("dependency_graph", {})
            objs = sorted(o["id"] for o in dg.get("local_objects", []) if o.get("exists") is not False)
            mats = sorted(r["id"] for r in rep.extras.get("materialization_ledger", []) if r.get("enabled") is not False)
            # the bound shape, so a tile change shows as a shape change (spec 10.3) instead of the same symbols
            tiles = {n["id"]: {"symbolic": n.get("tile_shape"), "value": n.get("tile_shape_value")}
                     for n in dg.get("matmul_nodes", []) if n.get("tile_shape")}
            return {"local_objects": objs, "materializations": mats, "mxu_tile_shapes": tiles}

        # First pass: build every candidate's report.  Values are read only afterwards, so whether a metric
        # counts as structurally absent cannot depend on the order the candidates happen to appear in.
        if isinstance(candidates, (str, bytes, Mapping)) or not isinstance(candidates, (list, tuple)):
            raise MetadataError("candidates must be a list of strategy patches (dicts) or ExecutionStrategy objects")
        built = []
        for i, cand in enumerate(candidates):
            algorithm = self.algorithm
            if isinstance(cand, ExecutionStrategy):
                strat = cand
            elif not isinstance(cand, Mapping):
                raise MetadataError(f"candidate {i} must be a dict of strategy fields or an ExecutionStrategy, got {type(cand).__name__}")
            else:
                cand = dict(cand)
                # an explicit null keeps the baseline's variant; only a named variant switches it
                algorithm = cand.pop("algorithm", None) or self.algorithm
                foreign = sorted(k for k in cand if k not in strategy_fields)
                if foreign:
                    raise TaskIdentityError(f"candidate {cand.get('name', i)} carries non-strategy fields {foreign}; task dims, semantics and numerics "
                                            "define the task and cannot be changed inside a same-task comparison (a shape change is a scaling study)")
                base_dict = self.strategy.to_dict()
                patch = {**base_dict, **cand}
                for mapping_field in ("materialize", "layout"):
                    # a shallow merge would drop the baseline's other keys, the very silent un-declaration the
                    # patch semantics exists to prevent
                    if isinstance(cand.get(mapping_field), Mapping):
                        patch[mapping_field] = {**(base_dict.get(mapping_field) or {}), **cand[mapping_field]}
                patch["name"] = cand.get("name", f"candidate_{i}")
                if not isinstance(patch["name"], str) or not patch["name"]:
                    raise MetadataError(f"candidate {i}: 'name' must be a non-empty string, got {patch['name']!r}")
                strat = ExecutionStrategy.from_source(patch, name=patch["name"])
            alt = self.with_strategy(strat).with_algorithm(algorithm)
            if alt.fingerprint() != fp:
                raise TaskIdentityError(f"candidate {strat.name} changed the task fingerprint")
            built.append((strat, algorithm, alt, alt.analyze()))

        names_seen = set(rep_names(base_rep))
        for _, _, _, rep in built:
            names_seen |= rep_names(rep)
        names = [self.strategy.name] + [st.name + ("" if a == self.algorithm else f" [{a}]") for st, a, _, _ in built]
        dup = sorted({n for n in names if names.count(n) > 1})
        if dup:
            raise MetadataError(f"candidate names must be unique (the derived lists are keyed by name); repeated: {dup}")
        # a watched name that no report carries is dropped from the table unless the caller asked for it
        watched = [k for k in watched if k in names_seen or k in objectives]
        base_vals = values(base_rep)
        base_struct = structure(base_rep)
        rows.append({"name": self.strategy.name, "role": "baseline", "algorithm": self.algorithm, "strategy": self.strategy.to_dict(),
                     "values": base_vals, "statuses": statuses(base_rep),
                     "delta_vs_baseline": {k: 0 if isinstance(v, (int, float)) and not isinstance(v, bool) else None
                                           for k, v in base_vals.items()},
                     "legality": self._legality(base_rep, self.strategy), "fingerprint": fp,
                     "structure": base_struct, "structure_delta": {"appear": [], "disappear": [], "mxu_tile_shape_changes": {}}})
        for strat, algorithm, alt, rep in built:
            vals = values(rep)
            st_c = structure(rep)
            shape_changes = {k: {"baseline": base_struct["mxu_tile_shapes"].get(k), "candidate": v}
                             for k, v in st_c["mxu_tile_shapes"].items() if base_struct["mxu_tile_shapes"].get(k) != v}
            shape_changes.update({k: {"baseline": v, "candidate": None} for k, v in base_struct["mxu_tile_shapes"].items()
                                  if k not in st_c["mxu_tile_shapes"]})
            appear = sorted((set(st_c["local_objects"]) | set(st_c["materializations"]))
                            - (set(base_struct["local_objects"]) | set(base_struct["materializations"])))
            disappear = sorted((set(base_struct["local_objects"]) | set(base_struct["materializations"]))
                               - (set(st_c["local_objects"]) | set(st_c["materializations"])))
            rows.append({"name": strat.name + ("" if algorithm == self.algorithm else f" [{algorithm}]"), "role": "candidate", "algorithm": algorithm,
                         "strategy": strat.to_dict(), "values": vals, "statuses": statuses(rep),
                         "delta_vs_baseline": PRESS.finite_difference(base_vals, vals),
                         "legality": alt._legality(rep, strat), "fingerprint": alt.fingerprint(),
                         "structure": st_c,
                         "structure_delta": {"appear": appear, "disappear": disappear, "mxu_tile_shape_changes": shape_changes}})
        same_alg = [r for r in rows if r["algorithm"] == self.algorithm]
        useful = {r["values"]["F_total[useful]"] for r in same_alg}
        invariant_ok = len(useful) == 1 and None not in useful
        # spec 10.2: the argmin ranges over K, so a mathematically illegal candidate is not a competitor.
        # Model-estimated infeasibility is uncertainty, not illegality, so those candidates stay in and are
        # annotated instead (10.2 forbids reading model uncertainty as a compilation verdict).
        eligible = [r for r in rows if r["legality"]["mathematical"] != "illegal"]
        excluded = [{"name": r["name"], "reason": "mathematically illegal", "violations": r["legality"]["violations"]}
                    for r in rows if r["legality"]["mathematical"] == "illegal"]
        infeasible_flagged = sorted(r["name"] for r in eligible
                                    if any(v.get("verdict") is False
                                           for v in (r["legality"].get("model_estimated_feasible_under_budget") or {}).values()))
        usable = [o for o in objectives if all(isinstance(r["values"].get(o), (int, float)) and not isinstance(r["values"].get(o), bool) for r in eligible)]
        unusable = [o for o in objectives if o not in usable]
        pareto = []
        for r in eligible:
            dominated = False
            for o_r in eligible:
                if o_r is r:
                    continue
                le = all(o_r["values"][o] <= r["values"][o] for o in usable)
                lt = any(o_r["values"][o] < r["values"][o] for o in usable)
                if usable and le and lt:
                    dominated = True
                    break
            if not dominated and usable:
                pareto.append(r["name"])
        intervals = [(r["name"], r["values"]["T_pred[node_interval]"]) for r in eligible if isinstance(r["values"].get("T_pred[node_interval]"), list)]
        ranking = None
        ranking_note = None
        if intervals:
            ranking = sorted(intervals, key=lambda x: x[1][1])
            # intervals separate only if each upper bound is below the next lower bound
            separated = all(ranking[i][1][1] < ranking[i + 1][1][0] for i in range(len(ranking) - 1))
            ranking_note = ("intervals separate: order is supported by the calibrated matmul model (vector/transfer time not covered)" if separated
                            else "intervals overlap: ranking inconclusive; treat as hypotheses requiring measurement")
        notes = []
        partial_obj = sorted({o for o in usable for r in eligible if r["statuses"].get(o) == "partial"})
        if partial_obj:
            notes.append("objectives whose values are scenario values (status partial) for some candidate, so the Pareto "
                         "filter compares scenarios, not complete figures: " + ", ".join(partial_obj))
        if unusable:
            notes.append("objectives left out of the Pareto filter because some candidate has no numeric value for them: "
                         + ", ".join(f"{o} (" + ("not a metric of these reports" if o not in names_seen
                                                 else "unknown or not applicable for some candidate") + ")" for o in unusable))
        if excluded:
            notes.append("excluded from the Pareto set and ranking as mathematically illegal (not in K): "
                         + ", ".join(f"{e['name']} ({', '.join(e['violations'])})" for e in excluded))
        if infeasible_flagged:
            notes.append("kept but model-estimated infeasible under the declared budget at some level "
                         "(model uncertainty, not a compilation verdict): " + ", ".join(infeasible_flagged))
        return _jsonable_comparison({
            "task_fingerprint": fp, "algorithm": self.algorithm, "rows": rows,
            "useful_work_invariant": invariant_ok, "pareto_objectives": usable, "pareto_set": pareto,
            "candidates_considered": [r["name"] for r in eligible], "excluded_candidates": excluded,
            "model_infeasible_candidates": infeasible_flagged,
            "calibrated_ranking": ranking, "ranking_note": ranking_note, "selection_notes": notes,
            "verdict": (ranking_note if intervals else "no calibration: Pareto set / hypotheses requiring validation, not a measured winner"),
        })

    def _legality(self, rep: Report, strat: ExecutionStrategy) -> dict:
        errs = [c for c in rep.constraints if c["holds"] is False and c["severity"] == "error"]
        unknown = [c["name"] for c in rep.constraints if c["holds"] is None and c["severity"] == "error"]
        # the feasibility verdict follows the tightest stage (spec 7.4 is stated per stage), and names it
        feas = rep.extras.get("feasible_region_binding", {})
        model_feasible = {}
        binding = {}
        for level, row in feas.items():
            if row.get("b_k_max") is not None and strat.b_k is not None:
                over = row.get("stages_over_budget_independently_of_b_k") or []
                unevaluated = row.get("stages_not_evaluated") or []
                empty = row["b_k_max"] < 1
                verdict = (strat.b_k <= row["b_k_max"]) and not over and not empty
                model_feasible[level] = {"verdict": verdict,
                                         # 'partial' is reserved for an incomplete evaluation; an over-budget stage or an
                                         # empty region is a definite (negative) verdict
                                         "status": "partial" if unevaluated else "bound",
                                         "stages_not_evaluated": unevaluated,
                                         "stages_over_budget_independently_of_b_k": over,
                                         "feasible_region_empty": empty}
                binding[level] = {"binding_stage": row.get("binding_stage"), "b_k_max": row["b_k_max"]}
        ev = self.compile_evidence or {}
        backend = (f"evidence attached from {ev.get('source') or 'an undeclared source'}"
                   f"{' (' + str(ev.get('toolchain')) + ')' if ev.get('toolchain') else ''}: the candidate compiled, "
                   "but this model did not verify it") if ev else "unknown (no compilation evidence)"
        return {"mathematical": "illegal" if errs else ("legal" if not unknown else "unknown"),
                "violations": [c["name"] for c in errs], "backend_support": backend,
                "model_estimated_feasible_under_budget": model_feasible or None,
                "binding_stage_per_level": binding or None}


def _jsonable_comparison(cmp: dict) -> dict:
    """Render the comparison with one canonical JSON encoding of the absence sentinel.

    Without this the dict carries a live object, so ``json.dumps`` fails and ``json.dumps(default=str)``
    disagrees with the report's own serialiser about how absence is spelled.
    """
    def conv(v):
        if isinstance(v, _Absent):
            return ABSENT_JSON
        if isinstance(v, dict):
            return {k: conv(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [conv(x) for x in v]
        return v
    return conv(cmp)


def _expr_of(m: Metric):
    """Best-effort sympy expression behind a metric (None for text-only metrics)."""
    return getattr(m, "_expr", None)


def asdict_bucket(b) -> dict:
    from dataclasses import asdict
    d = asdict(b)
    for r in ("m_range", "n_range", "k_range"):
        if d.get(r) is not None:
            d[r] = list(d[r])
    return d


def comparison_to_markdown(cmp: dict) -> str:
    lines = [f"# Strategy comparison (task {cmp['task_fingerprint']}, {cmp['algorithm']})", ""]
    lines.append(f"- verdict: {cmp['verdict']}")
    lines.append(f"- useful work invariant across candidates: {cmp['useful_work_invariant']}")
    lines.append(f"- Pareto objectives: {cmp['pareto_objectives']}  → Pareto set: {cmp['pareto_set']}")
    if cmp.get("calibrated_ranking"):
        lines.append(f"- calibrated ranking (predicted matmul interval, s): {cmp['calibrated_ranking']}  — {cmp.get('ranking_note')}")
    for note in cmp.get("selection_notes") or []:
        lines.append(f"- {note}")
    keys = list(cmp["rows"][0]["values"].keys())
    lines += ["", "| metric | " + " | ".join(r["name"] for r in cmp["rows"]) + " |", "|---|" + "---|" * len(cmp["rows"])]
    for k in keys:
        cells = []
        for r in cmp["rows"]:
            v = r["values"][k]
            d = r["delta_vs_baseline"].get(k)
            if isinstance(v, _Absent) or v == ABSENT_JSON:
                cell = "n/a"          # the variant has no such metric — not the same as an unknown value
            elif v is None:
                cell = "?"
            elif r["role"] == "baseline" or d is None or isinstance(d, list) or isinstance(d, str):
                cell = f"{v}"
            else:
                cell = f"{v} ({'+' if d >= 0 else ''}{d})"
            cells.append(cell)
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    lines += ["", "(`n/a` = this variant has no such metric; `?` = the metric exists but its value is unknown)"]
    lines += ["", "| candidate | mathematical legality | backend | model-feasible under declared budgets | binding stage |", "|---|---|---|---|---|"]
    for r in cmp["rows"]:
        L = r["legality"]
        lines.append(f"| {r['name']} | {L['mathematical']} {L['violations'] or ''} | {L['backend_support']} | "
                     f"{L['model_estimated_feasible_under_budget']} | {L.get('binding_stage_per_level')} |")
    struct = [r for r in cmp["rows"] if r.get("structure_delta") and (r["structure_delta"]["appear"] or r["structure_delta"]["disappear"]
                                                                      or r["structure_delta"]["mxu_tile_shape_changes"])]
    if struct:
        lines += ["", "| candidate | intermediates that appear | intermediates that disappear | MXU tile shape changes |", "|---|---|---|---|"]
        for r in struct:
            d = r["structure_delta"]
            shapes = "; ".join(f"{k}: {(v['baseline'] or {}).get('value')} → {(v['candidate'] or {}).get('value')}"
                               for k, v in d["mxu_tile_shape_changes"].items()) or "—"
            lines.append(f"| {r['name']} | {', '.join(d['appear']) or '—'} | {', '.join(d['disappear']) or '—'} | {shapes} |")
    return "\n".join(lines) + "\n"
