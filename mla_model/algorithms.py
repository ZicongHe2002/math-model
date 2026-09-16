"""Algorithm-variant graph builders (spec 3, 5, 6, 7).

Each builder produces a shape-aware ``AlgorithmGraph`` whose nodes carry symbolic
matrix shapes, execution multiplicities, local objects with stage liveness, storage
level and alias groups, transfer events per path, and conditional materializations.
Loop multiplicities stay symbolic (``B*ceil(H/h_pp)*BlockStat(...)`` etc.); nothing is
expanded to scalar instructions.

Variants:
  expanded                   Q^n = Cq Wq, K^n = Ck Wk, V = Ck Wv, X = Q^n K^n^T + Q^r K^r^T, O = P V
  absorbed_two_step          Q~ = (Cq Wq) Wk^T, X = Q~ Ck^T + Q^r K^r^T, Z = P Ck, O = Z Wv     (state width R_k)
  absorbed_precomputed       W~ = Wq Wk^T precomputed, Q~ = Cq W~, otherwise as absorbed_two_step

Undeclared strategy / numeric fields never pick a value silently: the affected nodes
carry ``exists=None`` / ``undeclared`` markers and the graph lists the missing field
names per metric domain (``g.undeclared``) so the report can mark metrics partial.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import sympy as sp

from .implementation import ExecutionStrategy
from .errors import MetadataError
from .numerics import CAST_SITES, CAST_SITE_DTYPES, NumericPolicy
from .semantics import SemanticConfig
from .symbolic import Sym, ceil, block_stat

STAGES = ("q_proj", "q_absorb", "kv_proj", "score", "softmax", "pv", "finalize")
ALGORITHMS = ("expanded", "absorbed_two_step", "absorbed_precomputed")
LEVELS = ("vmem", "vreg")
UNKNOWN_N_KEYVISITS = sp.Symbol("N_keyvisits", integer=True, nonnegative=True)
UNKNOWN_N_ROWVISITS = sp.Symbol("N_rowvisits", integer=True, nonnegative=True)
UNKNOWN_N_KVBLOCKS = sp.Symbol("N_kvblocks", integer=True, nonnegative=True)
UNKNOWN_C_MASKED = sp.Symbol("C_masked", integer=True, nonnegative=True)


@dataclass
class MatmulNode:
    id: str
    stage: str
    M: sp.Basic
    N: sp.Basic
    K: sp.Basic
    multiplicity: sp.Basic                 # number of executions of this logical matmul
    description: str
    operand_dtype_symbol: str | None = None
    acc_dtype_symbol: str | None = None
    counts_toward: str = "attention"       # 'projection' | 'attention' | 'output_projection' | 'weight_merge'
    conditional_on: str | None = None
    tile_shape: tuple | None = None        # executed per-visit (M, N, K) tile for calibration matching; None = whole-task shape applies

    @property
    def flops(self) -> sp.Basic:
        return 2 * self.M * self.N * self.K * self.multiplicity


@dataclass
class VectorOp:
    id: str
    stage: str
    kind: str                                # exp | log | cmp | add | mul | div | recip | cast | mask | broadcast | transpose
    count: sp.Basic                          # logical element operations
    resource: str                            # 'exp' | 'vpu' | 'reduce' | 'layout'
    description: str
    conditional_on: str | None = None
    exists: bool | None = True               # False -> excluded; None -> depends on an undeclared field (kept, flagged)


@dataclass
class LocalObject:
    id: str
    shape: tuple                             # symbolic (rows, cols) — cols may be 1 for row states
    bytes_symbol: sp.Symbol                  # s_* symbol (storage width)
    live_stages: tuple[str, ...]
    description: str
    alias_group: str | None = None           # objects in one group share storage; count once
    buffers: sp.Basic = sp.Integer(1)        # multiplicity for multi-buffered windows
    level: str = "vmem"                      # 'vmem' (residency windows, scratch) | 'vreg' (vector temporaries) — analysis-level assignment
    conditional_on: str | None = None
    exists: bool | None = True               # False -> excluded; None -> depends on undeclared strategy field
    size_scenario: str | None = None         # the object exists, but an undeclared field fixed its size
    stage_scenario: str | None = None        # the object exists, but an undeclared field fixed its live-stage set
    stage_scenario_alt: tuple[str, ...] = ()  # the live-stage set the other branch of that scenario would give
    is_raw_bytes: bool = False               # declared as a byte count, not an [m, n] element array (no layout function)

    @property
    def size(self) -> sp.Basic:
        rows, cols = self.shape
        return rows * cols * self.bytes_symbol * self.buffers


@dataclass
class TransferEvent:
    id: str
    path: str                                # hbm_to_vmem | vmem_to_hbm | vmem_to_vreg | vreg_to_vmem
    bytes_per_event: sp.Basic
    executions: sp.Basic
    description: str
    conditional_on: str | None = None
    scenario: str | None = None
    undeclared: bool = False                 # True -> depends on an undeclared field / no compiler evidence

    @property
    def total(self) -> sp.Basic:
        return self.bytes_per_event * self.executions


@dataclass
class Materialization:
    id: str
    tensor: str
    bytes: sp.Basic                          # logical size of the materialized tensor
    write_traffic: sp.Basic                  # bytes written to HBM
    read_traffic: sp.Basic                   # bytes read back from HBM under the declared scan policy
    conditional_on: str
    enabled: bool | None

    @property
    def traffic(self) -> sp.Basic:
        return self.write_traffic + self.read_traffic


@dataclass
class AlgorithmGraph:
    variant: str
    stages: tuple[str, ...]
    matmuls: list[MatmulNode] = field(default_factory=list)
    vector_ops: list[VectorOp] = field(default_factory=list)
    locals: list[LocalObject] = field(default_factory=list)
    transfers: list[TransferEvent] = field(default_factory=list)
    materializations: list[Materialization] = field(default_factory=list)
    accumulator_width: sp.Basic = Sym.D_v
    assumptions: list[str] = field(default_factory=list)
    scenario_notes: list[str] = field(default_factory=list)
    cast_conflicts: list[dict] = field(default_factory=list)   # declared cast sites vs the declared dtypes
    # undeclared fields per metric domain (report marks dependent metrics 'partial')
    undeclared: dict[str, list[str]] = field(default_factory=lambda: {"executed_work": [], "vector": [], "transfers": [], "locals": []})
    # loop structure symbols
    n_programs: sp.Basic = sp.Integer(1)
    n_kv_visits: sp.Basic = sp.Integer(1)    # total selected (b, h, q-block, kv-block) rectangles
    exec_cells: sp.Basic = sp.Integer(0)     # executed cells under the declared extent policy
    row_visits: sp.Basic = sp.Integer(0)
    key_visits: sp.Basic = sp.Integer(0)     # sum over selected rectangles of executed key columns (all b, one head)
    masked_cells: sp.Basic = sp.Integer(0)   # executed cells inside not-fully-visible rectangles
    # rows the executed graph actually projects (spec 5.2 loop multiplicity); None when it equals the scope count
    executed_proj_rows: dict = field(default_factory=dict)
    projected_operand_elements: sp.Basic = sp.Integer(0)  # elements written by the projection matmuls (cast site)
    # quantities this variant provably lacks (spec 10.3 "disappearance"): (metric name, formula id, reason).
    # analyze() emits them not_applicable with no value, so a comparison reads absence from the row itself.
    absent_metrics: list = field(default_factory=list)
    has_output_projection: bool = False       # a Z -> O projection exists (absorbed paths)

    def matmul_flops(self, counts_toward: str | None = None) -> sp.Basic:
        sel = [m.flops for m in self.matmuls if counts_toward is None or m.counts_toward == counts_toward]
        return sp.Add(*sel) if sel else sp.Integer(0)

    def vector_counts(self) -> dict[str, sp.Basic]:
        out: dict[str, sp.Basic] = {}
        for op in self.vector_ops:
            out.setdefault(op.kind, sp.Integer(0))          # a kind whose ops are all excluded reports 0 (declared absence)
            if op.exists is False:
                continue
            out[op.kind] = out[op.kind] + op.count
        return out

    def resource_counts(self) -> dict[str, sp.Basic]:
        out: dict[str, sp.Basic] = {}
        for op in self.vector_ops:
            out.setdefault(op.resource, sp.Integer(0))
            if op.exists is False:
                continue
            out[op.resource] = out[op.resource] + op.count
        return out

    def undeclared_vector_ops(self) -> list[str]:
        return sorted({op.conditional_on or op.id for op in self.vector_ops if op.exists is None})

    def undeclared_by_kind(self) -> dict[str, list[str]]:
        out: dict[str, set] = {}
        for op in self.vector_ops:
            if op.exists is None:
                out.setdefault(op.kind, set()).add(op.conditional_on or op.id)
        return {k: sorted(v) for k, v in out.items()}

    def undeclared_by_resource(self) -> dict[str, list[str]]:
        out: dict[str, set] = {}
        for op in self.vector_ops:
            if op.exists is None:
                out.setdefault(op.resource, set()).add(op.conditional_on or op.id)
        return {k: sorted(v) for k, v in out.items()}

    def mark_undeclared(self, domain: str, name: str):
        if name not in self.undeclared[domain]:
            self.undeclared[domain].append(name)


# ---------------------------------------------------------------------------
# helpers shared by the builders
# ---------------------------------------------------------------------------

def rows_q(ragged=None):
    """Total (batch, head, query-row) count: B*H*S_q, or H*sum_b Sq_b when ragged."""
    if ragged:
        return Sym.H * sp.Integer(sum(ragged["Sq"]))
    return Sym.B * Sym.H * Sym.S_q


def rows_k(ragged=None):
    """Total (batch, head, key-row) count: B*H*S_k, or H*sum_b Sk_b when ragged."""
    if ragged:
        return Sym.H * sp.Integer(sum(ragged["Sk"]))
    return Sym.B * Sym.H * Sym.S_k


def n_qblocks(ragged=None, scheduled: bool = False):
    """Total number of (batch, q-block) pairs: B*ceil(S_q/b_q) (scheduled extent when declared) or sum_b ceil(Sq_b/b_q)."""
    if ragged:
        return sp.Add(*[ceil(sp.Integer(sq) / Sym.b_q) for sq in ragged["Sq"]])
    return Sym.B * ceil((Sym.S_q_sched if scheduled else Sym.S_q) / Sym.b_q)


def head_groups():
    """Number of head groups (programs per (batch, q-block)) = ceil(H / h_pp)."""
    return ceil(Sym.H / Sym.h_pp)


def q_rows_b(ragged=None):
    """(batch, query-row) pairs actually present: B*S_q, or sum_b Sq_b when ragged."""
    if ragged:
        return sp.Integer(sum(ragged["Sq"]))
    return Sym.B * Sym.S_q


def _programs(ragged=None, scheduled: bool = False):
    """Number of (batch, head-group, q-block) programs under the q_outer_kv_inner nest."""
    return head_groups() * n_qblocks(ragged, scheduled)


def head_qblock_pairs(ragged=None, scheduled: bool = False):
    """(batch, head, q-block) triples: H * n_qblocks.

    Exact for per-head events even when h_pp does not divide H — programs * h_pp would charge
    ceil(H/h_pp)*h_pp heads instead of the H heads that exist.
    """
    return Sym.H * n_qblocks(ragged, scheduled)


class _Grid:
    """Batch-summed block statistics (per head) for the declared mask/policy/extents."""

    def __init__(self, sem: SemanticConfig, strat: ExecutionStrategy, ragged):
        self.mask = sem.mask
        self.policy = strat.rect_policy
        self.sub = strat.subdivide
        self.extent = strat.executed_extent_policy
        self.ragged = ragged
        self.scheduled = bool(strat.scheduled_lengths) and not ragged   # grid over scheduled extents, visibility over active
        self.known = self.mask is not None and self.policy is not None and (not ragged or ragged.get("Delta"))

    def per_b(self, stat: str):
        """sum over batches of the per-(b,h) statistic (one head)."""
        if not self.known:
            return None
        if self.ragged:
            return sp.Add(*[block_stat(stat, self.mask, sp.Integer(sq), sp.Integer(sk), Sym.b_q, Sym.b_k, sp.Integer(dl), self.policy, self.sub)
                            for sq, sk, dl in zip(self.ragged["Sq"], self.ragged["Sk"], self.ragged["Delta"])])
        delta = sp.Integer(0) if self.mask == "none" else Sym.Delta
        if self.scheduled:
            return Sym.B * block_stat(stat, self.mask, Sym.S_q_sched, Sym.S_k_sched, Sym.b_q, Sym.b_k, delta, self.policy, self.sub,
                                      Sq_active=Sym.S_q, Sk_active=Sym.S_k)
        return Sym.B * block_stat(stat, self.mask, Sym.S_q, Sym.S_k, Sym.b_q, Sym.b_k, delta, self.policy, self.sub)

    def executed(self, logical_stat: str, padded_stat: str, unknown_symbol):
        """Batch-summed executed statistic under the declared extent policy (unknown symbol when undeclared)."""
        if not self.known:
            return unknown_symbol
        if self.extent == "padded_to_tile":
            return self.per_b(padded_stat)
        if self.extent == "logical":
            return self.per_b(logical_stat)
        return unknown_symbol


def _softmax_vector_ops(g: AlgorithmGraph, cells, row_visits, masked_cells, D_A, num: NumericPolicy, sem: SemanticConfig, n_rows_total,
                        key_visits_for_transpose=sp.Integer(0)):
    """Logical vector-operation counts for the online softmax (spec 5.5 table)."""
    rec = num.recurrence
    g.vector_ops += [
        VectorOp("scale_scores", "score", "mul", cells, "vpu", "gamma * X on every executed cell (fused variants differ)"),
        VectorOp("mask_apply", "score", "mask", masked_cells, "vpu",
                 "additive/select mask on executed cells of rectangles that are not fully visible (fully visible blocks need no mask)",
                 conditional_on="mask application scenario: not-fully-visible rectangles only"),
        VectorOp("row_max_cmp", "softmax", "cmp", cells - row_visits, "reduce", "n*max(m-1,0) comparisons for rowmax over executed cells"),
        VectorOp("old_new_max_cmp", "softmax", "cmp", row_visits, "reduce", "max(m, rowmax) once per row per block visit"),
        VectorOp("rowmax_broadcast", "softmax", "broadcast", cells, "layout", "broadcast m' across the row before X - m' (may be free depending on layout)",
                 conditional_on="broadcast_materialized (undeclared backend behaviour)", exists=None),
        VectorOp("kv_operand_transpose", "score", "transpose", key_visits_for_transpose, "layout",
                 "re-layout / transpose of the K-side operand for the score matmul (free when the layout already matches)",
                 conditional_on="operand_transpose_materialized (undeclared backend behaviour)", exists=None),
        VectorOp("exp_E", "softmax", "exp", cells, "exp", "E = exp(X - m') on every executed cell (masked cells included)"),
        VectorOp("row_sum_add", "softmax", "add", cells - row_visits, "reduce", "n*max(m-1,0) additions for rowsum(E)"),
        VectorOp("rescale_alpha_exp", "softmax", "exp", row_visits, "exp", "alpha = exp(m - m') at most once per row per block visit"),
        VectorOp("acc_scale", "pv", "mul", row_visits * D_A, "vpu", "alpha * A (n*D_A per block visit)"),
        VectorOp("acc_add", "pv", "add", row_visits * D_A, "vpu", "A + E U (n*D_A per block visit; matmul epilogue may fuse)"),
        VectorOp("l_scale", "softmax", "mul", row_visits, "vpu", "alpha * l"),
        VectorOp("l_update", "softmax", "add", row_visits, "vpu", "l' = alpha*l + rowsum"),
    ]
    if rec == "normalized":
        g.vector_ops += [
            VectorOp("normalized_rescale", "pv", "mul", row_visits * D_A, "vpu",
                     "normalized recurrence: Y' = (alpha*l/l') Y + (E U)/l' — extra per-visit multiply on the accumulator"),
            VectorOp("normalized_recip", "softmax", "recip", row_visits, "vpu", "1/l' per row per visit"),
        ]
        g.assumptions.append("recurrence=normalized: accumulator rescaled to normalized form at every visit")
        if num.final_normalize is not None:
            note = ("final_normalize is declared but has no effect under a normalized recurrence: the accumulator is "
                    "already normalized at every visit, so there is no final division")
            g.assumptions.append(note)
            g.scenario_notes.append(note)          # reaches the report warnings, where a reader of W_vec[*] looks
    else:
        # spec 5.5 leaves the final row-wise normalization implementation-dependent: n*D_A divisions, OR n
        # reciprocals plus n*D_A multiplies.  They are alternatives, so exactly one is counted; undeclared,
        # the division form is the named scenario and the field is named missing (counting both would report
        # the cost of a backend that does the work twice).
        norm = num.final_normalize
        div_exists = norm != "reciprocal_multiply"          # divide is the scenario while the field is undeclared
        rm_exists = norm == "reciprocal_multiply"
        g.vector_ops += [
            VectorOp("final_normalize_div", "finalize", "div", n_rows_total * D_A, "vpu",
                     "Y = A / l once per row, as n*D_A divisions (form 'divide')",
                     conditional_on=None if norm else "final_normalize", exists=div_exists),
            VectorOp("final_normalize_recip", "finalize", "recip", n_rows_total, "vpu",
                     "1/l once per row (form 'reciprocal_multiply')",
                     conditional_on=None if norm else "final_normalize", exists=rm_exists),
            VectorOp("final_normalize_mul", "finalize", "mul", n_rows_total * D_A, "vpu",
                     "Y = A * (1/l), n*D_A multiplies (form 'reciprocal_multiply')",
                     conditional_on=None if norm else "final_normalize", exists=rm_exists),
        ]
        if norm is None:
            # the scenario op carries the undeclared field itself, so only div/recip/mul are marked, not every kind
            for op in g.vector_ops[-3:]:
                if op.id == "final_normalize_div":
                    op.exists = None
                    op.conditional_on = "final_normalize"
            g.scenario_notes.append("final_normalize not declared: the n*D_A division form is the scenario; the "
                                    "reciprocal-plus-multiply form of spec 5.5 is the alternative, not an addition")
        if rec is None:
            g.scenario_notes.append("recurrence not declared: unnormalized recurrence counts shown as the scenario; normalized is a separate scenario")
            g.mark_undeclared("vector", "recurrence")
    # spec 3.1/3.3: LSE = m + log(l) is produced per row when it is requested; asking for it must change the
    # vector-operation graph, not only the byte ledger
    wants = sem.wants_lse()
    g.vector_ops += [
        VectorOp("lse_log", "finalize", "log", n_rows_total, "vpu", "log(l) once per row for LSE = m + log(l)",
                 conditional_on=None if wants is not None else "outputs", exists=True if wants else (None if wants is None else False)),
        VectorOp("lse_add", "finalize", "add", n_rows_total, "vpu", "m + log(l) once per row",
                 conditional_on=None if wants is not None else "outputs", exists=True if wants else (None if wants is None else False)),
    ]
    if wants is None:
        g.mark_undeclared("vector", "outputs")
    if num.exp_base == "2":
        g.vector_ops.append(VectorOp("log2e_scale", "score", "mul", cells, "vpu",
                                     "exp2 path: logits scaled by log2(e) (may fuse with gamma)"))
        if sem.wants_lse() is not False:
            g.vector_ops.append(VectorOp("lse_restore", "finalize", "mul", n_rows_total, "vpu",
                                         "exp2 path: restore natural-log LSE = ln2 * (m2 + log2 l) per row",
                                         conditional_on=None if sem.wants_lse() else "outputs", exists=True if sem.wants_lse() else None))
    elif num.exp_base is None:
        g.scenario_notes.append("exp_base not declared: base-e op graph shown; exp2 adds a logit scale and an LSE restore")
        g.mark_undeclared("vector", "exp_base")
    _cast_vector_ops(g, cells, row_visits, n_rows_total, D_A, num)


# spec 3.3: a normalized accumulator, an unnormalized accumulator, exp2 and *different cast locations* must
# each generate a different vector-operation graph.  Every declared site therefore emits its own cast op with
# the element count of the tensor being cast; the sites are validated in NumericPolicy.
_CAST_SITE_OPS = dict(CAST_SITES)      # one table: the validator and the op graph cannot disagree


def _site_dtypes(g: AlgorithmGraph, num: NumericPolicy) -> dict:
    """(producer dtype, consumer dtype) per cast site for this variant.

    In the absorbed paths the final output is produced by the Z W^v projection, whose accumulator is the
    projection accumulator (falling back to the attention accumulator dtype), not the attention accumulator.
    """
    out = {}
    for site, (prod_f, cons_f) in CAST_SITE_DTYPES.items():
        a, b_ = getattr(num, prod_f), getattr(num, cons_f)
        if site == "accumulator_to_output" and g.has_output_projection:
            a = num.projection_accumulator_dtype or num.accumulator_dtype
        out[site] = (a, b_)
    return out


def _cast_vector_ops(g: AlgorithmGraph, cells, row_visits, n_rows_total, D_A, num: NumericPolicy):
    """Cast ops from the declared cast locations, or from dtype differences when none are declared."""
    counts = {
        "score_to_exp": cells,
        "exp_to_p": cells,
        "state_update": 2 * row_visits,                      # m and l
        "accumulator_to_output": n_rows_total * Sym.D_v,
        "projected_operand": g.projected_operand_elements,
        "z_to_output": n_rows_total * D_A,
    }
    assert set(counts) == set(_CAST_SITE_OPS), "cast site tables drifted apart"
    if num.cast_points is not None:
        for site, (stage, desc) in _CAST_SITE_OPS.items():
            declared = site in num.cast_points
            if site == "z_to_output" and not g.has_output_projection:
                if declared:
                    note = ("cast_points declares z_to_output, which this variant has no Z to cast: "
                            "the site is reported as absent, not silently dropped")
                    g.assumptions.append(note)
                    g.scenario_notes.append(note)          # reaches the report warnings, where a reader of W_vec[cast] looks
                g.vector_ops.append(VectorOp(f"cast[{site}]", stage, "cast", counts[site], "vpu", desc + " (not present in this variant)",
                                             conditional_on="z_to_output requires an output projection", exists=False))
                continue
            g.vector_ops.append(VectorOp(f"cast[{site}]", stage, "cast", counts[site], "vpu", desc,
                                         conditional_on=None, exists=declared))
        note = f"cast_points declared: the cast graph is exactly the declared sites {sorted(num.cast_points)}"
        g.assumptions.append(note)
        g.scenario_notes.append(note)
        # the dtype fields imply casts of their own at EVERY site; where they disagree with the declared site
        # list, both declarations are recorded rather than one quietly winning
        for site, (a, b_) in _site_dtypes(g, num).items():
            if site == "z_to_output" and not g.has_output_projection:
                continue
            if a is not None and b_ is not None and (a != b_) != (site in num.cast_points):
                g.cast_conflicts.append({
                    "site": site, "declared_in_cast_points": site in num.cast_points,
                    "dtypes": [a, b_],
                    "message": (f"cast_points {'declares' if site in num.cast_points else 'omits'} {site}, but the declared "
                                f"dtypes at that site are {a} and {b_}, which imply "
                                f"{'no cast' if a == b_ else 'a cast'}; the declared site list is used for the op graph")})
        return
    # undeclared: the two sites the dtype fields can imply, each flagged when its dtypes are undeclared
    def cast_exists(a, b):
        if a is None or b is None:
            return None
        return a != b
    site_dt = _site_dtypes(g, num)
    for site in ("exp_to_p", "accumulator_to_output"):
        a, b_ = site_dt[site]
        prod_f, cons_f = CAST_SITE_DTYPES[site]
        undeclared_fields = [f for f, v in ((prod_f, a), (cons_f, b_)) if v is None]
        # the condition is carried as the FIELD names it depends on, so missing_fields stays a list of fields;
        # exists=None only when one of them is undeclared, so only the cast kind is marked, not every kind
        g.vector_ops.append(VectorOp(f"cast[{site}]", _CAST_SITE_OPS[site][0], "cast", counts[site], "vpu",
                                     _CAST_SITE_OPS[site][1] + " (inferred from the declared dtypes)",
                                     conditional_on=", ".join(undeclared_fields) if undeclared_fields else "cast_points",
                                     exists=cast_exists(a, b_) if not undeclared_fields else None))
    # the site LIST itself is undeclared: attach that to the cast kind through a scenario op, not to every kind
    g.vector_ops.append(VectorOp("cast[sites_undeclared]", "softmax", "cast", sp.Integer(0), "vpu",
                                 "cast_points undeclared: the two dtype-implied sites above are the scenario; a declared list "
                                 "gives a different graph", conditional_on="cast_points", exists=None))
    g.scenario_notes.append("cast_points not declared: the cast graph is inferred from dtype differences at the two "
                            "modelled sites (exp_to_p, accumulator_to_output); a declared site list gives a different graph")


def _materialization_transfers(g, mats):
    """Spec 6.2: each write and each read of a materialized intermediate is counted on its path.

    Without these events a declared materialization would raise the traffic in the materialization ledger
    while leaving B[vmem_to_hbm] / B[hbm_to_vmem] unchanged, i.e. the two ledgers would disagree about the
    same bytes.  A materialization whose flag is undeclared contributes events marked undeclared, never zero.
    """
    for m in mats:
        if m.enabled is False:
            continue                     # declared off: spec 6.2 says it must not be counted
        note = ("additive materialization traffic: the producing computation and its operand reads are left as "
                "they are, so this is the round trip added on top, not a restructured dataflow "
                "(the expanded K/V materialization is modelled structurally instead)")
        g.transfers += [
            TransferEvent(f"{m.id}_write", "vmem_to_hbm", m.write_traffic, sp.Integer(1),
                          f"write the materialized {m.tensor} to HBM",
                          conditional_on=m.conditional_on, scenario=note, undeclared=(m.enabled is None)),
            TransferEvent(f"{m.id}_read", "hbm_to_vmem", m.read_traffic, sp.Integer(1),
                          f"read the materialized {m.tensor} back from HBM once per consuming pass",
                          conditional_on=m.conditional_on, scenario=note, undeclared=(m.enabled is None)),
        ]
        g.assumptions.append(f"{m.tensor} materialization counted additively: one whole-tensor write and one "
                             "whole-tensor read; the in-loop producer and its operand reads are unchanged")


V_DELAYED_STAGES = ("pv",)             # where the V tile lives when its load is delayed


def _v_tile_stages(g, strat: ExecutionStrategy, early_stages, produced_in_loop: bool):
    """(stages, scenario_field) for the V tile under the declared V-load policy (spec 10.3 "delay V load").

    ``v_load='delayed'`` holds V only for the PV stage, which shortens its lifetime overlap and gives up the
    prefetch that an early load overlaps with the score matmul.  Undeclared, the early load is the scenario
    and the field is named on every metric that the V tile's lifetime shapes.
    """
    if strat.v_load == "delayed":
        if produced_in_loop:
            # V is produced by a projection at the kv_proj stage here, so its *load* cannot be delayed without
            # moving that projection; the model does not model a moved projection, and says so rather than
            # publishing a graph whose operands are dead before their consumer
            raise MetadataError(
                "v_load='delayed' is not modelled when V is projected inside the KV loop (materialize.expanded_kv "
                "is false or undeclared): there is no separate V load to delay, and moving the V projection to the "
                "PV stage is a different dataflow this model does not build. Declare materialize.expanded_kv=true "
                "to stream a materialized V, or leave v_load undeclared.")
        g.assumptions.append("v_load=delayed: the V tile is loaded at the PV stage, so it is not live during "
                             "score/softmax; the prefetch that an early load overlaps with the score matmul is given up")
        return ("pv",), None
    if strat.v_load is None:
        g.mark_undeclared("locals", "v_load")
        g.scenario_notes.append("v_load not declared: the early V load (V live from its production/read until PV) "
                                "is shown as the scenario")
        return early_stages, "v_load"
    return early_stages, None


def stage_scenario_at(obj, stage: str) -> str | None:
    """The undeclared field that decided this object's presence AT ``stage``, or None if both branches agree.

    A scenario that does not change whether the object is live at a cut did not shape that cut, so naming it
    there would mark a fully determined set incomplete.
    """
    if not obj.stage_scenario:
        return None
    if obj.stage_scenario_alt and ((stage in obj.live_stages) == (stage in obj.stage_scenario_alt)):
        return None
    return obj.stage_scenario


def _q_read_rows(strat: ExecutionStrategy, ragged, scheduled):
    """(rows charged, whether the padded tail is included, the field that decides it).

    Spec 4.2 warns that a logical out-of-bounds is not an HBM read, and the same section says the executed
    extents come from the declared strategy.  So the declared ``executed_extent_policy`` decides: 'logical'
    charges the rows that exist, 'padded_to_tile' charges the whole tile.  Undeclared, the logical count is
    the scenario and the field is named, rather than both being added into one total.
    """
    logical = q_rows_b(ragged)
    padded = Sym.b_q * n_qblocks(ragged, scheduled)
    if strat.executed_extent_policy == "padded_to_tile":
        return padded, True, None
    return logical, False, (None if strat.executed_extent_policy else "executed_extent_policy")


def _declare_schedule(g, strat: ExecutionStrategy):
    """The loop nest shapes every transfer count and the program count, so an undeclared nest is a scenario.

    Only ``q_outer_kv_inner`` is modelled; leaving the field out used to produce the same numbers with nothing
    saying they rest on that choice.
    """
    if strat.schedule is None:
        g.mark_undeclared("transfers", "schedule")
        g.mark_undeclared("executed_work", "schedule")
        g.scenario_notes.append("schedule not declared: the q_outer_kv_inner loop nest is shown as the scenario; "
                                "transfer counts and the program count follow from it")


Q_NONRESIDENT_STAGES = ("score",)      # where a Q operand lives when it is re-read per KV block


def _q_operand_stages(g, strat: ExecutionStrategy, loop_carried):
    """(stages, scenario_field) over which a Q-side operand is held, given the declared Q residency.

    ``q_resident=False`` means Q is re-read for every KV block, so a Q operand is live only in the stage that
    consumes it.  It says nothing about the online-softmax state, which is loop-carried either way.  When the
    field is undeclared the returned scenario name is attached to the objects it shapes, so every live set,
    peak and envelope containing one of them names it as missing.
    """
    if strat.q_resident is False:
        g.assumptions.append("q_resident=False: Q operands are re-read per KV block, so they are live only in the "
                             "score stage; the loop-carried softmax state (m, l, A) is unaffected")
        return ("score",), None
    if strat.q_resident is None:
        g.mark_undeclared("locals", "q_resident")
        g.scenario_notes.append("q_resident not declared: Q operands held across the KV loop shown as the scenario")
        return loop_carried, "q_resident"
    return loop_carried, None


def _common_locals(g, strat: ExecutionStrategy, D_A, n_buf, state_stages, has_rope: bool | None = None,
                   q_operand_stages=None, q_stage_scenario=None):
    """Local objects shared by all variants: scores, E, P operand, row states, accumulator, Q^r/K^r tiles, scratch.

    ``state_stages`` are the stages over which the online-softmax state (m, l, A) is carried; they follow the
    KV loop and never depend on Q residency.  ``q_operand_stages`` are the stages over which a Q-side operand
    is held; a declared ``q_resident=False`` shortens those and only those (spec 7.3: the live set is per
    object, so re-reading Q per KV block cannot retire the loop-carried accumulator).
    """
    if q_operand_stages is None:
        q_operand_stages = state_stages
    alias_xe = None
    exp_exists: bool | None = True
    if strat.score_alias_exp is True:
        alias_xe = "score_or_E"
    elif strat.score_alias_exp is None:
        exp_exists = None            # undeclared: X and E both counted (worst case) but flagged
        g.mark_undeclared("locals", "score_alias_exp")
    if strat.exp_alias_p_operand is None:
        g.mark_undeclared("locals", "exp_alias_p_operand")
    p_exists = None if strat.exp_alias_p_operand is None else (not strat.exp_alias_p_operand)
    e_group = alias_xe
    p_group = None
    if strat.exp_alias_p_operand:
        p_group = alias_xe or "E_or_P"
        e_group = alias_xe or "E_or_P"
    g.locals += [
        LocalObject("score_X", (Sym.b_q, Sym.b_k), Sym.s_X, ("score", "softmax"), "score tile X = gamma(...)+M", alias_group=alias_xe, level="vreg"),
        LocalObject("exp_E", (Sym.b_q, Sym.b_k), Sym.s_E, ("softmax", "pv"), "E = exp(X - m')", alias_group=e_group, level="vreg",
                    conditional_on=None if strat.score_alias_exp is not None else "score_alias_exp", exists=exp_exists),
        LocalObject("p_operand", (Sym.b_q, Sym.b_k), Sym.s_P, ("pv",), "P/E operand in the matmul input dtype", alias_group=p_group, level="vreg",
                    conditional_on="exp_alias_p_operand", exists=p_exists),
        LocalObject("row_state_m", (Sym.b_q, 1), Sym.s_state, state_stages, "running row max m (loop-carried)", level="vreg"),
        LocalObject("row_state_l", (Sym.b_q, 1), Sym.s_state, state_stages, "running row sum l (loop-carried)", level="vreg"),
        LocalObject("row_alpha", (Sym.b_q, 1), Sym.s_state, ("softmax", "pv"), "alpha = exp(m - m')", level="vreg"),
        LocalObject("accumulator_A", (Sym.b_q, D_A), Sym.s_A, state_stages, f"attention accumulator A, width {D_A} (loop-carried)", level="vreg"),
    ]
    # the second QK partial product exists only if there IS a positional branch to combine
    if has_rope is False:
        g.locals.append(LocalObject("score_rope_branch", (Sym.b_q, Sym.b_k), Sym.s_X, ("score",),
                                    "second QK partial product (absent: D_r = 0, no positional branch)", level="vreg",
                                    conditional_on="D_r > 0", exists=False))
    else:
        if strat.both_qk_branches_live is None:
            g.mark_undeclared("locals", "both_qk_branches_live")
        if strat.both_qk_branches_live is not False:
            g.locals.append(LocalObject("score_rope_branch", (Sym.b_q, Sym.b_k), Sym.s_X, ("score",),
                                        "second QK partial product kept simultaneously (both_qk_branches_live)", level="vreg",
                                        conditional_on="both_qk_branches_live" if has_rope is not None else "both_qk_branches_live and D_r > 0",
                                        exists=(None if (strat.both_qk_branches_live is None or has_rope is None) else True)))
    g.locals += [
        LocalObject("q_pe_tile", (Sym.b_q, Sym.D_r), Sym.s_Qr, q_operand_stages, "Q^r operand held across the KV loop (shortened when q_resident=False)", level="vmem",
                    stage_scenario=q_stage_scenario, stage_scenario_alt=Q_NONRESIDENT_STAGES),
        LocalObject("k_pe_tile", (Sym.b_k, Sym.D_r), Sym.s_Kr, ("score",), "K^r tile for the current KV block", buffers=n_buf, level="vmem"),
        LocalObject("fixed_scratch", (sp.Integer(1), sp.Integer(1)), Sym.s_scratch, tuple(g.stages), "fixed scratch (d_s)", level="vmem", is_raw_bytes=True,
                    conditional_on="scratch_bytes", exists=(True if strat.scratch_bytes is not None else None)),
    ]
    if strat.scratch_bytes is None:
        g.mark_undeclared("locals", "scratch_bytes")
    if strat.kv_buffers is None:
        g.mark_undeclared("locals", "kv_buffers")


def _output_locals(g, sem: SemanticConfig):
    g.locals += [
        LocalObject("out_tile", (Sym.b_q, Sym.D_v), Sym.s_O, ("finalize",), "output tile O in output dtype", level="vmem"),
        LocalObject("lse_tile", (Sym.b_q, 1), Sym.s_LSE, ("finalize",), "LSE tile", level="vmem", conditional_on="outputs",
                    exists=(True if sem.wants_lse() else (None if sem.wants_lse() is None else False))),
    ]


def _output_transfers(g, sem: SemanticConfig, n_rows_total):
    """Output writes at logical row granularity (tail tiles write only their valid rows)."""
    g.transfers.append(TransferEvent("O_write", "vmem_to_hbm", Sym.D_v * Sym.s_O, n_rows_total, "write one output row per (b, h, active query row)"))
    wl = sem.wants_lse()
    if wl is not False:
        g.transfers.append(TransferEvent("LSE_write", "vmem_to_hbm", Sym.s_LSE, n_rows_total, "write one LSE value per (b, h, active query row)",
                                         conditional_on="outputs", undeclared=(wl is None)))
        if wl is None:
            g.mark_undeclared("transfers", "outputs")


def _operand_streaming(g, per_visit_bytes: list[tuple[str, sp.Basic]], per_program_bytes: list[tuple[str, sp.Basic]], kv_visits_total, per_head_events,
                       q_resident=None):
    """Scenario events on the VMEM->VREG path: operands streamed once per program / once per KV visit.

    No compiler evidence is available for register scheduling, so every event is marked undeclared
    (the report shows the path with status 'partial'); VREG->VMEM traffic (stores/spill) stays unknown.

    ``q_resident=False`` means the Q-side operands are re-read for every KV block, so their execution count
    is the visit count rather than one per (head, q-block).  Without that, the live sets would say the
    operands are not held while the byte ledger still charged them as if they were.
    """
    q_events = kv_visits_total if q_resident is False else per_head_events
    q_where = "once per (b, h, kv-block visit): q_resident=False" if q_resident is False else "once per (head, q-block)"
    for name, nbytes in per_program_bytes:
        g.transfers.append(TransferEvent(f"stream_{name}", "vmem_to_vreg", nbytes, q_events, f"stream {name} into registers {q_where}",
                                         scenario="operand streaming (no register-schedule evidence)"
                                                  + ("" if q_resident is not None else "; Q residency undeclared"),
                                         conditional_on="q_resident" if q_resident is None else None, undeclared=True))
    for name, nbytes in per_visit_bytes:
        g.transfers.append(TransferEvent(f"stream_{name}", "vmem_to_vreg", nbytes, kv_visits_total, f"stream {name} into registers once per (b, h, kv-block visit)",
                                         scenario="operand streaming (no register-schedule evidence)", undeclared=True))
    g.mark_undeclared("transfers", "register_schedule_evidence")


def _projection_rows(sem: SemanticConfig, ragged=None):
    """(N_Qproj, N_Kproj, N_Vproj, scope_undeclared, new_fraction) per spec 5.3 from the declared projection scope.

    new_fraction is the share of active keys that are newly projected (1 for full, (S_k - cached)/S_k for new_tokens_only).
    """
    scope = sem.projection_scope
    BHSq = rows_q(ragged)
    BHSk = rows_k(ragged)
    if scope in (None, "full"):
        return BHSq, BHSk, BHSk, (scope is None), sp.Integer(1)
    if scope == "new_tokens_only":
        cached = None if sem.cache is None else sem.cache.get("cached_tokens")
        if cached is None:
            new_rows = Sym.B * Sym.H * Sym.S_new
            frac = Sym.S_new / Sym.S_k
        elif ragged:
            new_keys = sum(max(0, sk - int(cached)) for sk in ragged["Sk"])
            new_rows = Sym.H * sp.Integer(new_keys)
            frac = sp.Rational(new_keys, max(1, sum(ragged["Sk"])))
        else:
            new_rows = Sym.B * Sym.H * sp.Max(0, Sym.S_k - sp.Integer(int(cached)))
            frac = sp.Max(0, Sym.S_k - sp.Integer(int(cached))) / Sym.S_k
        return BHSq, new_rows, new_rows, False, frac
    raise ValueError(f"projection scope {scope!r} is not representable by the seven-input latent graphs")


# ---------------------------------------------------------------------------
# expanded
# ---------------------------------------------------------------------------

def build_expanded(sem: SemanticConfig, strat: ExecutionStrategy, num: NumericPolicy, ragged=None, dims=None) -> AlgorithmGraph:
    has_rope = None if not dims or "Dr" not in dims else dims["Dr"] > 0
    g = AlgorithmGraph(variant="expanded", stages=("q_proj", "kv_proj", "score", "softmax", "pv", "finalize"), accumulator_width=Sym.D_v)
    grid = _Grid(sem, strat, ragged)
    cells_b = grid.executed("rect_cells", "padded_cells", None)
    g.exec_cells = Sym.H * cells_b if cells_b is not None else Sym.C_pad
    rv = grid.executed("row_visits", "row_visits_padded", None)
    g.row_visits = Sym.H * rv if rv is not None else UNKNOWN_N_ROWVISITS
    kv = grid.executed("key_visits", "key_visits_padded", None)
    key_visits_b = kv if kv is not None else UNKNOWN_N_KEYVISITS          # batch-summed, per head
    mc = grid.executed("masked_cells", "masked_cells_padded", None)
    g.masked_cells = Sym.H * mc if mc is not None else UNKNOWN_C_MASKED
    sel = grid.per_b("selected")
    g.n_kv_visits = Sym.H * sel if sel is not None else UNKNOWN_N_KVBLOCKS
    g.key_visits = key_visits_b
    g.n_programs = _programs(ragged, grid.scheduled)
    if strat.executed_extent_policy is None:
        g.mark_undeclared("executed_work", "executed_extent_policy")
        g.mark_undeclared("vector", "executed_extent_policy")
        g.mark_undeclared("transfers", "executed_extent_policy")
    if strat.rect_policy is None:
        for dom in ("executed_work", "vector", "transfers"):
            g.mark_undeclared(dom, "rect_policy")
    if sem.mask is None:
        for dom in ("executed_work", "vector", "transfers"):
            g.mark_undeclared(dom, "mask")
    n_buf = Sym.n_buf
    _declare_schedule(g, strat)
    NQ, NK, NV, scope_unknown, new_frac = _projection_rows(sem, ragged)
    if scope_unknown:
        g.scenario_notes.append("projection_scope not declared: full projection (every Q/K/V row once) shown as the scenario")
        g.mark_undeclared("executed_work", "projection_scope")
        g.mark_undeclared("transfers", "projection_scope")
    brq = Sym.b_rq if strat.b_rq is not None else Sym.R_q
    brk = Sym.b_rk if strat.b_rk is not None else Sym.R_k
    for field, declared in (("b_rq", strat.b_rq), ("b_rk", strat.b_rk)):
        if declared is None:
            g.mark_undeclared("locals", field)
    if strat.b_rq is None or strat.b_rk is None:
        g.scenario_notes.append("rank sub-tiles undeclared: the full R_q / R_k window is shown as the scenario")
    n_rows_total = rows_q(ragged)
    mat = strat.materialize or {}
    mat_kv = mat.get("expanded_kv")
    if mat_kv is None:
        g.mark_undeclared("transfers", "materialize.expanded_kv")
        g.mark_undeclared("locals", "materialize.expanded_kv")
    in_loop_projection = not mat_kv          # None/False: K/V projected per KV block inside the loop (scenario)
    loop_carried = ("kv_proj", "score", "softmax", "pv", "finalize") if in_loop_projection else ("score", "softmax", "pv", "finalize")

    # --- matmuls (whole-task multiplicities) ---
    q_operand_stages, q_stage_scenario = _q_operand_stages(g, strat, loop_carried)
    if strat.projection_in_kv_loop is None:
        g.mark_undeclared("executed_work", "projection_in_kv_loop")
        g.scenario_notes.append("projection_in_kv_loop not declared: Q projected once per q-block shown as the scenario")
    if strat.projection_in_kv_loop:
        # repeated per selected KV block visit: projected rows = sum over selected rectangles of n_i (follows the rect policy)
        q_rows = g.row_visits
        g.assumptions.append("projection_in_kv_loop=True: Q projection repeated for every selected KV block visit (rows = row visits)")
    else:
        q_rows = NQ
    # spec 5.2: when K/V are projected inside the KV loop (no materialized expanded cache), every q-block that
    # scans a KV tile re-projects it, so the projected-row count is the executed key-column count, not S_k once.
    if in_loop_projection:
        kv_proj_rows = Sym.H * key_visits_b * new_frac
        g.assumptions.append("K/V projected inside the KV loop: projected rows = executed key columns (each q-block re-projects the tile)")
        if new_frac != 1:
            # spec 5.3/6.3: which visits land on new tokens depends on where the cache boundary falls in the
            # block grid; splitting the visit total by the global new-token share is a scenario, not a count
            g.assumptions.append("cached/new split scenario: executed key-column visits are apportioned by the global "
                                 "new-token share (S_k - cached)/S_k, which is exact only when every q-block visits "
                                 "cached and new columns in that proportion")
            g.mark_undeclared("executed_work", "cache boundary position in the block grid")
            g.mark_undeclared("transfers", "cache boundary position in the block grid")
    else:
        kv_proj_rows = None
    K_rows = kv_proj_rows if kv_proj_rows is not None else NK
    V_rows = kv_proj_rows if kv_proj_rows is not None else NV
    g.executed_proj_rows = {"N_Qproj": q_rows, "N_Kproj": K_rows, "N_Vproj": V_rows}
    g.matmuls += [
        MatmulNode("Q_proj", "q_proj", q_rows, Sym.D_n, Sym.R_q, sp.Integer(1), "Q^n = C_q W^q  ([rows, R_q] x [R_q, D_n])", "s_Cq", "s_Qacc", "projection"),
        MatmulNode("K_proj", "kv_proj", K_rows, Sym.D_n, Sym.R_k, sp.Integer(1), "K^n = C_k W^k  ([rows, R_k] x [R_k, D_n])", "s_Ck", "s_Qacc", "projection"),
        MatmulNode("V_proj", "kv_proj", V_rows, Sym.D_v, Sym.R_k, sp.Integer(1), "V = C_k W^v  ([rows, R_k] x [R_k, D_v])", "s_Ck", "s_Qacc", "projection"),
        MatmulNode("QK_nope", "score", g.exec_cells, sp.Integer(1), Sym.D_n, sp.Integer(1), "Q^n K^n^T over executed cells: 2*C*D_n", "s_Qn", "s_X", "attention",
                   tile_shape=(Sym.b_q, Sym.b_k, Sym.D_n)),
        MatmulNode("QK_rope", "score", g.exec_cells, sp.Integer(1), Sym.D_r, sp.Integer(1), "Q^r K^r^T over executed cells: 2*C*D_r", "s_Qr", "s_X", "attention",
                   tile_shape=(Sym.b_q, Sym.b_k, Sym.D_r)),
        MatmulNode("PV", "pv", g.exec_cells, sp.Integer(1), Sym.D_v, sp.Integer(1), "P V over executed cells: 2*C*D_v", "s_P", "s_A", "attention",
                   tile_shape=(Sym.b_q, Sym.D_v, Sym.b_k)),
    ]
    g.projected_operand_elements = q_rows * Sym.D_n + K_rows * Sym.D_n + V_rows * Sym.D_v
    for name, fid in (("HBM_mat[Q~]", "bytes.mat.mat_q_tilde"), ("M_mat[Q~]", "bytes.mat.size.mat_q_tilde"),
                      ("HBM_mat[Z]", "bytes.mat.mat_z"), ("M_mat[Z]", "bytes.mat.size.mat_z"),
                      ("local[q_tilde]", "local.q_tilde.bytes"), ("local[z_tile]", "local.z_tile.bytes"),
                      ("F_output_projection[executed_graph]", "work.expanded.graph.output_projection")):
        g.absent_metrics.append((name, fid, "the expanded path forms no Q~ and no Z: it consumes expanded K^n and V"))
    for key in ("q_tilde", "z"):
        if mat.get(key) is not None:
            g.assumptions.append(f"materialize.{key} is declared but the expanded path has no {key} to materialize: no effect")
    _softmax_vector_ops(g, g.exec_cells, g.row_visits, g.masked_cells, Sym.D_v, num, sem, n_rows_total, Sym.H * key_visits_b * Sym.D_n)
    g.vector_ops.append(VectorOp("combine_qk_branches", "score", "add", sp.Piecewise((g.exec_cells, Sym.D_r > 0), (sp.Integer(0), True)), "vpu",
                                 "X = Q^n K^n^T + Q^r K^r^T (0 when D_r = 0; fused K-concat variants differ)", conditional_on="D_r > 0"))

    # --- local objects (spec 7.1) ---
    rq_scen = "b_rq" if strat.b_rq is None else None
    rk_scen = "b_rk" if strat.b_rk is None else None
    g.locals += [
        LocalObject("q_latent_window", (Sym.b_q, Sym.R_q), Sym.s_Cq, ("q_proj",), "Q latent full window b_q x R_q",
                    exists=(strat.b_rq is None), size_scenario=rq_scen),
        LocalObject("q_latent_rank_tile", (Sym.b_q, brq), Sym.s_Cq, ("q_proj",), "Q latent rank sub-tile b_q x b_rq", exists=(strat.b_rq is not None)),
        LocalObject("wq_rank_tile", (brq, Sym.D_n), Sym.s_Wq, ("q_proj",), "W^q rank sub-tile b_rq x D_n (per head)", size_scenario=rq_scen),
        LocalObject("q_proj_acc", (Sym.b_q, Sym.D_n), Sym.s_Qacc, ("q_proj",), "Q projection accumulator b_q x D_n", level="vreg"),
        LocalObject("q_nope_operand", (Sym.b_q, Sym.D_n), Sym.s_Qn, q_operand_stages, "Q^n operand held across the KV loop (shortened when q_resident=False)",
                    stage_scenario=q_stage_scenario, stage_scenario_alt=Q_NONRESIDENT_STAGES),
    ]
    if in_loop_projection:
        v_stages_inloop, v_scen_inloop = _v_tile_stages(g, strat, ("kv_proj", "score", "softmax", "pv"), True)
        g.locals += [
            LocalObject("kv_latent_rank_tile", (Sym.b_k, brk), Sym.s_Ck, ("kv_proj",), "KV latent (rank sub-)tile b_k x b_rk", buffers=n_buf,
                        conditional_on=None if mat_kv is False else "materialize.expanded_kv", exists=(True if mat_kv is False else None), size_scenario=rk_scen),
            LocalObject("wk_rank_tile", (brk, Sym.D_n), Sym.s_Wk, ("kv_proj",), "W^k rank sub-tile", conditional_on=None if mat_kv is False else "materialize.expanded_kv",
                        exists=(True if mat_kv is False else None), size_scenario=rk_scen),
            LocalObject("wv_rank_tile", (brk, Sym.D_v), Sym.s_Wv, ("kv_proj",), "W^v rank sub-tile", conditional_on=None if mat_kv is False else "materialize.expanded_kv",
                        exists=(True if mat_kv is False else None), size_scenario=rk_scen),
            LocalObject("k_nope_tile", (Sym.b_k, Sym.D_n), Sym.s_Kn, ("kv_proj", "score"), "projected K^n tile b_k x D_n"),
            LocalObject("v_tile", (Sym.b_k, Sym.D_v), Sym.s_V, v_stages_inloop, "projected V tile b_k x D_v (live until PV)",
                        stage_scenario=v_scen_inloop, stage_scenario_alt=V_DELAYED_STAGES),
        ]
    else:
        v_stages_mat, v_scen_mat = _v_tile_stages(g, strat, ("score", "softmax", "pv"), False)
        g.locals += [
            LocalObject("kv_latent_rank_tile", (Sym.b_k, brk), Sym.s_Ck, ("kv_proj",), "KV latent tile of the separate projection pass", buffers=n_buf, size_scenario=rk_scen),
            LocalObject("wk_rank_tile", (brk, Sym.D_n), Sym.s_Wk, ("kv_proj",), "W^k rank sub-tile (projection pass)", size_scenario=rk_scen),
            LocalObject("wv_rank_tile", (brk, Sym.D_v), Sym.s_Wv, ("kv_proj",), "W^v rank sub-tile (projection pass)", size_scenario=rk_scen),
            LocalObject("kv_proj_out_tile", (Sym.b_k, Sym.D_n + Sym.D_v), Sym.s_Kn, ("kv_proj",), "projected K^n|V tile written to HBM (projection pass)", level="vreg"),
            LocalObject("k_nope_tile", (Sym.b_k, Sym.D_n), Sym.s_Kn, ("score",), "materialized K^n tile streamed per KV block", buffers=n_buf),
            LocalObject("v_tile", (Sym.b_k, Sym.D_v), Sym.s_V, v_stages_mat, "materialized V tile streamed per KV block", buffers=n_buf,
                        stage_scenario=v_scen_mat, stage_scenario_alt=V_DELAYED_STAGES),
        ]
    _common_locals(g, strat, Sym.D_v, n_buf, loop_carried, has_rope, q_operand_stages=q_operand_stages,
                   q_stage_scenario=q_stage_scenario)
    _output_locals(g, sem)

    # --- transfer events (hbm -> vmem) under the q_outer_kv_inner scenario ---
    q_read_rows, q_read_padded, q_read_field = _q_read_rows(strat, ragged, grid.scheduled)
    kv_col_events = head_groups() * key_visits_b          # (head-group, b, executed key column) events
    kv_col_events_h = Sym.H * key_visits_b                # per-head key columns (head-dependent tensors)
    g.transfers += [
        TransferEvent("Cq_read", "hbm_to_vmem", Sym.R_q * Sym.s_Cq, head_groups() * q_read_rows,
                      f"Q latent row per (head group, {'scheduled tile row' if q_read_padded else 'query row'})",
                      conditional_on=q_read_field, undeclared=bool(q_read_field),
                      scenario=("whole tiles read, including the tail padding (declared padded_to_tile)" if q_read_padded
                                else "logical rows read (spec 4.2: a logical out-of-bounds is not an HBM read)")),
        TransferEvent("Qr_read", "hbm_to_vmem", Sym.D_r * Sym.s_Qr, Sym.H * q_read_rows,
                      f"Q^r row per (head, {'scheduled tile row' if q_read_padded else 'query row'})",
                      conditional_on=q_read_field, undeclared=bool(q_read_field),
                      scenario=("whole tiles read, including the tail padding (declared padded_to_tile)" if q_read_padded
                                else "logical rows read (spec 4.2)")),
        TransferEvent("Wq_read", "hbm_to_vmem", Sym.R_q * Sym.D_n * Sym.s_Wq, head_qblock_pairs(ragged, grid.scheduled), "W^q per (program, head) — no cross-program weight residency assumed",
                      scenario="weights re-read per program"),
        TransferEvent("Kr_read", "hbm_to_vmem", Sym.D_r * Sym.s_Kr, kv_col_events, "K^r per executed key column, once per (head-group, program-row)"),
    ]
    if in_loop_projection:
        g.transfers += [
            TransferEvent("Ck_read", "hbm_to_vmem", Sym.R_k * Sym.s_Ck, kv_col_events * new_frac,
                          "KV latent per executed key column, shared by h_pp heads (new tokens only when a cache is declared); "
                          "present because expanded K/V are not materialized",
                          conditional_on="materialize.expanded_kv" if mat_kv is None else None, undeclared=(mat_kv is None)),
            TransferEvent("Wk_read", "hbm_to_vmem", Sym.R_k * Sym.D_n * Sym.s_Wk, head_qblock_pairs(ragged, grid.scheduled), "W^k per (program, head)", scenario="weights re-read per program",
                          conditional_on="materialize.expanded_kv" if mat_kv is None else None, undeclared=(mat_kv is None)),
            TransferEvent("Wv_read", "hbm_to_vmem", Sym.R_k * Sym.D_v * Sym.s_Wv, head_qblock_pairs(ragged, grid.scheduled), "W^v per (program, head)", scenario="weights re-read per program",
                          conditional_on="materialize.expanded_kv" if mat_kv is None else None, undeclared=(mat_kv is None)),
        ]
        if sem.projection_scope == "new_tokens_only":
            g.transfers.append(TransferEvent("Kexp_cached_read", "hbm_to_vmem", (Sym.s_Kn * Sym.D_n + Sym.s_V * Sym.D_v), kv_col_events_h * (1 - new_frac),
                                             "expanded K^n|V of cached tokens per executed key column per head (proportional split scenario)",
                                             scenario="cached/new keys split proportionally to their counts"))
            g.assumptions.append("new_tokens_only: KV read traffic split between cached expanded K/V and new latent tokens in proportion to key counts")
    else:
        g.transfers += [
            TransferEvent("Ck_read_once", "hbm_to_vmem", Sym.R_k * Sym.s_Ck, NK / Sym.H, "KV latent read once per projected token (projection pass)"),
            TransferEvent("Wk_read_once", "hbm_to_vmem", Sym.R_k * Sym.D_n * Sym.s_Wk, Sym.H, "W^k once per head (projection pass)"),
            TransferEvent("Wv_read_once", "hbm_to_vmem", Sym.R_k * Sym.D_v * Sym.s_Wv, Sym.H, "W^v once per head (projection pass)"),
            TransferEvent("KV_mat_write", "vmem_to_hbm", (Sym.s_Kn * Sym.D_n + Sym.s_V * Sym.D_v), NK, "write materialized K^n|V once per projected (b, h, token)"),
            TransferEvent("KV_mat_read", "hbm_to_vmem", (Sym.s_Kn * Sym.D_n + Sym.s_V * Sym.D_v), kv_col_events_h,
                          "read materialized K^n|V per executed key column per head"),
        ]
        g.assumptions.append("materialize.expanded_kv=True: separate projection pass writes K^n|V to HBM; the attention loop streams expanded tiles")
    _output_transfers(g, sem, n_rows_total)
    _operand_streaming(g, [("K_tile", Sym.b_k * Sym.D_n * Sym.s_Kn), ("V_tile", Sym.b_k * Sym.D_v * Sym.s_V), ("Kr_tile", Sym.b_k * Sym.D_r * Sym.s_Kr)],
                       [("Qn_operand", Sym.b_q * Sym.D_n * Sym.s_Qn), ("Qr_operand", Sym.b_q * Sym.D_r * Sym.s_Qr)], g.n_kv_visits,
                       head_qblock_pairs(ragged, grid.scheduled), q_resident=strat.q_resident)
    if strat.heads_per_program is None:
        g.scenario_notes.append("heads_per_program (h_pp) undeclared: KV-latent/K^r sharing across heads stays symbolic")
        g.mark_undeclared("transfers", "heads_per_program")

    # --- HBM materialization ledger (spec 6.2) ---
    kv_bytes = rows_k(ragged) * (Sym.s_Kn * Sym.D_n + Sym.s_V * Sym.D_v)
    g.materializations += [
        Materialization("mat_expanded_kv", "K^n,V", kv_bytes, NK * (Sym.s_Kn * Sym.D_n + Sym.s_V * Sym.D_v),
                        kv_col_events_h * (Sym.s_Kn * Sym.D_n + Sym.s_V * Sym.D_v), "materialize.expanded_kv", mat_kv),
        Materialization("mat_q_nope", "Q^n", rows_q(ragged) * Sym.D_n * Sym.s_Qn, rows_q(ragged) * Sym.D_n * Sym.s_Qn, rows_q(ragged) * Sym.D_n * Sym.s_Qn,
                        "materialize.q_nope", mat.get("q_nope")),
    ]
    if mat.get("q_nope") is None:
        g.mark_undeclared("transfers", "materialize.q_nope")
    # K^n|V already has its own per-token write and per-visit read events in the projection-pass branch above,
    # so only the remaining intermediates are given generic events here
    _materialization_transfers(g, [m for m in g.materializations if m.id == "mat_q_nope"])
    g.assumptions += [
        "entry scope: seven latent inputs; Q^n, K^n, V projected inside this scope with the declared projection_scope",
        "outside the entry scope and therefore counted nowhere in this report: RMSNorm/LayerNorm, the RoPE "
        "rotation that produces Q^r/K^r, the latent down-projection that produces C_q/C_k, the output "
        "projection W^O after this attention, and the KV-cache update",
        "loop nest scenario q_outer_kv_inner: one program per (batch, head-group of h_pp heads, q-block) scanning selected KV blocks",
        "accumulator width D_A = D_v (expanded)",
        "storage levels are an analysis assignment (residency windows -> vmem, vector temporaries -> vreg), not compiler allocation",
    ]
    return g


# ---------------------------------------------------------------------------
# absorbed two-step / precomputed
# ---------------------------------------------------------------------------

def build_absorbed(sem: SemanticConfig, strat: ExecutionStrategy, num: NumericPolicy, ragged=None, precomputed_weights: bool = False, dims=None) -> AlgorithmGraph:
    has_rope = None if not dims or "Dr" not in dims else dims["Dr"] > 0
    variant = "absorbed_precomputed" if precomputed_weights else "absorbed_two_step"
    g = AlgorithmGraph(variant=variant, stages=("q_proj", "q_absorb", "score", "softmax", "pv", "finalize"), accumulator_width=Sym.R_k)
    grid = _Grid(sem, strat, ragged)
    cells_b = grid.executed("rect_cells", "padded_cells", None)
    g.exec_cells = Sym.H * cells_b if cells_b is not None else Sym.C_pad
    rv = grid.executed("row_visits", "row_visits_padded", None)
    g.row_visits = Sym.H * rv if rv is not None else UNKNOWN_N_ROWVISITS
    kv = grid.executed("key_visits", "key_visits_padded", None)
    key_visits_b = kv if kv is not None else UNKNOWN_N_KEYVISITS
    mc = grid.executed("masked_cells", "masked_cells_padded", None)
    g.masked_cells = Sym.H * mc if mc is not None else UNKNOWN_C_MASKED
    sel = grid.per_b("selected")
    g.n_kv_visits = Sym.H * sel if sel is not None else UNKNOWN_N_KVBLOCKS
    g.key_visits = key_visits_b
    g.n_programs = _programs(ragged, grid.scheduled)
    if strat.executed_extent_policy is None:
        for dom in ("executed_work", "vector", "transfers"):
            g.mark_undeclared(dom, "executed_extent_policy")
    if strat.rect_policy is None:
        for dom in ("executed_work", "vector", "transfers"):
            g.mark_undeclared(dom, "rect_policy")
    if sem.mask is None:
        for dom in ("executed_work", "vector", "transfers"):
            g.mark_undeclared(dom, "mask")
    n_buf = Sym.n_buf
    _declare_schedule(g, strat)
    NQ, _NK, _NV, scope_unknown, _frac = _projection_rows(sem, ragged)
    n_rows_total = rows_q(ragged)
    brq = Sym.b_rq if strat.b_rq is not None else Sym.R_q
    if strat.b_rq is None:
        g.mark_undeclared("locals", "b_rq")
        g.scenario_notes.append("rank sub-tile undeclared: the full R_q window is shown as the scenario")
    if strat.b_rk is not None:
        # the absorbed paths consume C_k whole as the K and V operand: there is no KV projection to sub-tile
        g.assumptions.append("b_rk is declared but has no effect in the absorbed path: C_k is consumed whole as the "
                             "K and V operand, so no KV projection rank sub-tile exists here")
    loop_carried = ("score", "softmax", "pv", "finalize")
    q_operand_stages, q_stage_scenario = _q_operand_stages(g, strat, loop_carried)
    if strat.v_load is not None:
        g.assumptions.append("v_load has no effect in the absorbed path: C_k is the K operand and the V operand, "
                             "so there is no separate V tile whose load can be delayed")

    if precomputed_weights:
        cache = sem.cache or {}
        if cache.get("merged_weight_reuse") is None:
            g.mark_undeclared("executed_work", "cache.merged_weight_reuse")
        g.matmuls += [
            MatmulNode("W_merge", "q_proj", Sym.H * Sym.R_q, Sym.R_k, Sym.D_n, 1 / Sym.n_reuse,
                       "W~ = W^q W^k^T ([H*R_q, D_n] x [D_n, R_k]); amortized over n_reuse invocations", "s_Wq", "s_Qacc", "weight_merge"),
            MatmulNode("Q_tilde", "q_absorb", NQ, Sym.R_k, Sym.R_q, sp.Integer(1), "Q~ = C_q W~ ([N_Qproj, R_q] x [R_q, R_k])", "s_Cq", "s_Qacc", "projection"),
        ]
        g.locals.append(LocalObject("w_merged", (Sym.R_q, Sym.R_k), Sym.s_Wq, ("q_absorb",), "merged weight W~ tile (per head) — invalid after weight update"))
        g.assumptions.append("precomputed W~ = W^q W^k^T; merge work amortized over n_reuse invocations (unknown unless declared)")
    else:
        g.matmuls += [
            MatmulNode("Q_proj", "q_proj", NQ, Sym.D_n, Sym.R_q, sp.Integer(1), "Q^n = C_q W^q", "s_Cq", "s_Qacc", "projection"),
            MatmulNode("Q_absorb", "q_absorb", NQ, Sym.R_k, Sym.D_n, sp.Integer(1), "Q~ = Q^n W^k^T ([N_Qproj, D_n] x [D_n, R_k])", "s_Qn", "s_Qacc", "projection"),
        ]
    g.matmuls += [
        MatmulNode("QC_latent", "score", g.exec_cells, sp.Integer(1), Sym.R_k, sp.Integer(1), "Q~ C_k^T over executed cells: 2*C*R_k", "s_Qt", "s_X", "attention",
                   tile_shape=(Sym.b_q, Sym.b_k, Sym.R_k)),
        MatmulNode("QK_rope", "score", g.exec_cells, sp.Integer(1), Sym.D_r, sp.Integer(1), "Q^r K^r^T over executed cells: 2*C*D_r", "s_Qr", "s_X", "attention",
                   tile_shape=(Sym.b_q, Sym.b_k, Sym.D_r)),
        MatmulNode("PC_latent", "pv", g.exec_cells, sp.Integer(1), Sym.R_k, sp.Integer(1), "Z = P C_k over executed cells: 2*C*R_k", "s_P", "s_A", "attention",
                   tile_shape=(Sym.b_q, Sym.R_k, Sym.b_k)),
        MatmulNode("Z_Wv", "finalize", n_rows_total, Sym.D_v, Sym.R_k, sp.Integer(1), "O = Z W^v ([B*H*S_q, R_k] x [R_k, D_v])", "s_Z", "s_Qacc", "output_projection"),
    ]
    g.has_output_projection = True
    g.executed_proj_rows = {"N_Qproj": NQ, "N_Kproj": sp.Integer(0), "N_Vproj": sp.Integer(0)}
    for name, fid in (("HBM_mat[K^n,V]", "bytes.mat.mat_expanded_kv"), ("M_mat[K^n,V]", "bytes.mat.size.mat_expanded_kv"),
                      ("HBM_mat[Q^n]", "bytes.mat.mat_q_nope"), ("M_mat[Q^n]", "bytes.mat.size.mat_q_nope"),
                      ("local[v_tile]", "local.v_tile.bytes"), ("local[k_nope_tile]", "local.k_nope_tile.bytes"),
                      ("local[q_nope_operand]", "local.q_nope_operand.bytes"),
                      ("V_lifetime_stages", "lifetime.v_tile.stages"), ("V_prefetch_overlapped", "lifetime.v_tile.prefetch")):
        g.absent_metrics.append((name, fid, "the absorbed path never expands K^n or V and never forms a Q^n operand: "
                                            "C_k is the K and the V operand"))
    mat_here = strat.materialize or {}
    for key in ("expanded_kv", "q_nope"):
        if mat_here.get(key) is not None:
            g.assumptions.append(f"materialize.{key} is declared but the absorbed path has no {key} to materialize: no effect")
    g.projected_operand_elements = NQ * Sym.R_k + (sp.Integer(0) if precomputed_weights else NQ * Sym.D_n)
    _softmax_vector_ops(g, g.exec_cells, g.row_visits, g.masked_cells, Sym.R_k, num, sem, n_rows_total, Sym.H * key_visits_b * Sym.R_k)
    g.vector_ops.append(VectorOp("combine_qk_branches", "score", "add", sp.Piecewise((g.exec_cells, Sym.D_r > 0), (sp.Integer(0), True)), "vpu",
                                 "X = Q~ C_k^T + Q^r K^r^T (0 when D_r = 0)", conditional_on="D_r > 0"))

    g.locals += [
        LocalObject("q_latent_window", (Sym.b_q, Sym.R_q), Sym.s_Cq, ("q_proj",) if not precomputed_weights else ("q_absorb",), "Q latent window",
                    exists=(strat.b_rq is None), size_scenario=("b_rq" if strat.b_rq is None else None)),
        LocalObject("q_latent_rank_tile", (Sym.b_q, brq), Sym.s_Cq, ("q_proj",) if not precomputed_weights else ("q_absorb",), "Q latent rank sub-tile", exists=(strat.b_rq is not None)),
        LocalObject("wq_rank_tile", (brq, Sym.D_n), Sym.s_Wq, ("q_proj",), "W^q rank sub-tile", exists=not precomputed_weights,
                    size_scenario=("b_rq" if strat.b_rq is None else None)),
        LocalObject("q_proj_acc", (Sym.b_q, Sym.D_n), Sym.s_Qacc, ("q_proj", "q_absorb"), "Q^n accumulator", level="vreg", exists=not precomputed_weights),
        LocalObject("wk_full", (Sym.R_k, Sym.D_n), Sym.s_Wk, ("q_absorb",), "W^k for the absorb step (per head)", exists=not precomputed_weights),
        LocalObject("q_tilde_acc", (Sym.b_q, Sym.R_k), Sym.s_Qacc, ("q_absorb",), "Q~ accumulator", level="vreg"),
        LocalObject("q_tilde", (Sym.b_q, Sym.R_k), Sym.s_Qt, q_operand_stages, "Q~ operand held across the KV loop (shortened when q_resident=False)",
                    stage_scenario=q_stage_scenario, stage_scenario_alt=Q_NONRESIDENT_STAGES),
        LocalObject("kv_latent_tile", (Sym.b_k, Sym.R_k), Sym.s_Ck, ("score", "softmax", "pv"),
                    "C_k tile used both as K and as V operand", buffers=n_buf),
        LocalObject("wv_full", (Sym.R_k, Sym.D_v), Sym.s_Wv, ("finalize",), "W^v for the output projection"),
        LocalObject("z_tile", (Sym.b_q, Sym.R_k), Sym.s_Z, ("finalize",), "normalized latent output Z", alias_group="acc_or_Z", level="vreg"),
    ]
    _common_locals(g, strat, Sym.R_k, n_buf, loop_carried, has_rope, q_operand_stages=q_operand_stages,
                   q_stage_scenario=q_stage_scenario)
    for obj in g.locals:
        if obj.id == "accumulator_A":
            obj.alias_group = "acc_or_Z"
    _output_locals(g, sem)

    q_read_rows, q_read_padded, q_read_field = _q_read_rows(strat, ragged, grid.scheduled)
    kv_col_events = head_groups() * key_visits_b
    g.transfers += [
        TransferEvent("Cq_read", "hbm_to_vmem", Sym.R_q * Sym.s_Cq, head_groups() * q_read_rows,
                      f"Q latent row per (head group, {'scheduled tile row' if q_read_padded else 'query row'})",
                      conditional_on=q_read_field, undeclared=bool(q_read_field),
                      scenario=("whole tiles read, including the tail padding (declared padded_to_tile)" if q_read_padded
                                else "logical rows read (spec 4.2: a logical out-of-bounds is not an HBM read)")),
        TransferEvent("Qr_read", "hbm_to_vmem", Sym.D_r * Sym.s_Qr, Sym.H * q_read_rows,
                      f"Q^r row per (head, {'scheduled tile row' if q_read_padded else 'query row'})",
                      conditional_on=q_read_field, undeclared=bool(q_read_field),
                      scenario=("whole tiles read, including the tail padding (declared padded_to_tile)" if q_read_padded
                                else "logical rows read (spec 4.2)")),
        TransferEvent("Ck_read", "hbm_to_vmem", Sym.R_k * Sym.s_Ck, kv_col_events, "KV latent per executed key column, shared by h_pp heads"),
        TransferEvent("Kr_read", "hbm_to_vmem", Sym.D_r * Sym.s_Kr, kv_col_events, "K^r per executed key column, shared by h_pp heads"),
        TransferEvent("Wv_read", "hbm_to_vmem", Sym.R_k * Sym.D_v * Sym.s_Wv, head_qblock_pairs(ragged, grid.scheduled), "W^v per (program, head)", scenario="weights re-read per program"),
    ]
    if precomputed_weights:
        g.transfers.append(TransferEvent("Wmerged_read", "hbm_to_vmem", Sym.R_q * Sym.R_k * Sym.s_Wq, head_qblock_pairs(ragged, grid.scheduled), "W~ per (program, head)", scenario="weights re-read per program"))
    else:
        g.transfers += [
            TransferEvent("Wq_read", "hbm_to_vmem", Sym.R_q * Sym.D_n * Sym.s_Wq, head_qblock_pairs(ragged, grid.scheduled), "W^q per (program, head)", scenario="weights re-read per program"),
            TransferEvent("Wk_read", "hbm_to_vmem", Sym.R_k * Sym.D_n * Sym.s_Wk, head_qblock_pairs(ragged, grid.scheduled), "W^k per (program, head)", scenario="weights re-read per program"),
        ]
    _output_transfers(g, sem, n_rows_total)
    _operand_streaming(g, [("Ck_tile", Sym.b_k * Sym.R_k * Sym.s_Ck), ("Kr_tile", Sym.b_k * Sym.D_r * Sym.s_Kr)],
                       [("Qt_operand", Sym.b_q * Sym.R_k * Sym.s_Qt), ("Qr_operand", Sym.b_q * Sym.D_r * Sym.s_Qr)], g.n_kv_visits,
                       head_qblock_pairs(ragged, grid.scheduled), q_resident=strat.q_resident)
    mat = strat.materialize or {}
    qt_bytes = rows_q(ragged) * Sym.R_k * Sym.s_Qt
    z_bytes = rows_q(ragged) * Sym.R_k * Sym.s_Z
    g.materializations += [
        Materialization("mat_q_tilde", "Q~", qt_bytes, qt_bytes, qt_bytes, "materialize.q_tilde", mat.get("q_tilde")),
        Materialization("mat_z", "Z", z_bytes, z_bytes, z_bytes, "materialize.z", mat.get("z")),
    ]
    for key in ("q_tilde", "z"):
        if mat.get(key) is None:
            g.mark_undeclared("transfers", f"materialize.{key}")
    _materialization_transfers(g, [m for m in g.materializations if m.id in ("mat_q_tilde", "mat_z")])
    g.assumptions += [
        "entry scope: seven latent inputs; Q~ formed inside this scope; no expanded K/V materialization",
        "outside the entry scope and therefore counted nowhere in this report: RMSNorm/LayerNorm, the RoPE "
        "rotation that produces Q^r/K^r, the latent down-projection that produces C_q/C_k, the output "
        "projection W^O after this attention, and the KV-cache update",
        "loop nest scenario q_outer_kv_inner",
        "accumulator width D_A = R_k (absorbed); Z = A / l still needs the W^v projection",
        "storage levels are an analysis assignment (residency windows -> vmem, vector temporaries -> vreg), not compiler allocation",
    ]
    if scope_unknown:
        g.scenario_notes.append("projection_scope not declared: full Q projection shown")
        g.mark_undeclared("executed_work", "projection_scope")
    if strat.heads_per_program is None:
        g.scenario_notes.append("heads_per_program (h_pp) undeclared: KV-latent sharing across heads stays symbolic")
        g.mark_undeclared("transfers", "heads_per_program")
    return g


def build_graph(algorithm: str, sem: SemanticConfig, strat: ExecutionStrategy, num: NumericPolicy, ragged=None, dims=None) -> AlgorithmGraph:
    if algorithm == "expanded":
        return build_expanded(sem, strat, num, ragged, dims=dims)
    if algorithm == "absorbed_two_step":
        return build_absorbed(sem, strat, num, ragged, precomputed_weights=False, dims=dims)
    if algorithm == "absorbed_precomputed":
        return build_absorbed(sem, strat, num, ragged, precomputed_weights=True, dims=dims)
    raise ValueError(f"unknown algorithm {algorithm!r}; available: {ALGORITHMS}")
