"""Symbolic layer.

The structural expression tree is sympy (an installed symbolic library, spec 2).
Expressions are always *constructed* programmatically; user strings are never parsed
with ``eval``/``sympify``.  This module provides:

* a registry of canonical symbols (task, tile, dtype-width, layout, hardware),
* custom aggregate functions (visible-cell counts, block statistics) that stay
  symbolic until every argument is an integer, then evaluate in closed form or by
  block enumeration,
* ``evaluate`` for partial binding: known symbols are substituted, unresolved
  symbols are reported as ``missing`` instead of being forced to 0,
* ``Constraint`` predicates that evaluate to True / False / None (unknown).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Iterable

import sympy as sp
from sympy.core.symbol import Str

# ---------------------------------------------------------------------------
# Symbol registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, sp.Symbol] = {}


def symbol(name: str, **assumptions) -> sp.Symbol:
    """Return the canonical Symbol for ``name`` (created on first use)."""
    if name in _REGISTRY:
        return _REGISTRY[name]
    s = sp.Symbol(name, **assumptions)
    _REGISTRY[name] = s
    return s


def _pos(name):
    return symbol(name, integer=True, positive=True)


def _nonneg(name):
    return symbol(name, integer=True, nonnegative=True)


def _real_pos(name):
    return symbol(name, real=True, positive=True)


class Sym:
    """Namespace of canonical symbols.  Names match the specification's symbol table."""

    # task dimensions (theta)
    B = _pos("B")
    H = _pos("H")
    S_q = _nonneg("S_q")
    S_k = _nonneg("S_k")
    R_q = _nonneg("R_q")
    R_k = _nonneg("R_k")
    D_n = _nonneg("D_n")
    D_r = _nonneg("D_r")
    D_v = _nonneg("D_v")
    Delta = symbol("Delta", integer=True)          # causal offset: key j visible to query i iff j <= i + Delta
    S_new = _nonneg("S_new")                       # newly projected KV tokens when a cache exists
    S_q_sched = _nonneg("S_q_sched")               # scheduled (executed) query extent, >= active, <= capacity
    S_k_sched = _nonneg("S_k_sched")               # scheduled (executed) key extent
    # generic projected-row counts (spec 5.3)
    N_Qproj = _nonneg("N_Qproj")
    N_Kproj = _nonneg("N_Kproj")
    N_Vproj = _nonneg("N_Vproj")
    # visibility aggregates
    C_valid = _nonneg("C_valid")
    C_rect = _nonneg("C_rect")
    C_pad = _nonneg("C_pad")
    # execution strategy (kappa)
    b_q = _pos("b_q")
    b_k = _pos("b_k")
    b_rq = _pos("b_rq")
    b_rk = _pos("b_rk")
    n_buf = _pos("n_buf")                          # buffer count for streamed input windows
    h_pp = _pos("h_pp")                            # heads per program (KV-latent sharing granularity)
    n_reuse = _pos("n_reuse")                      # reuse count of a precomputed merged weight
    # layout (lambda) and budgets
    L_s = _pos("L_s")                              # sublane extent of a vector tile
    L_l = _pos("L_l")                              # lane extent of a vector tile
    R_budget = _real_pos("R_budget")               # generic capacity budget (kept for backwards compatibility)
    R_vreg = _real_pos("R_vreg")                   # declared usable register-tile budget
    R_vmem = _real_pos("R_vmem")                   # declared VMEM scoped budget
    # numerics (nu)
    gamma = symbol("gamma", real=True)             # score scale
    # storage bytes per element (s_x)
    s_Cq = _pos("s_Cq"); s_Ck = _pos("s_Ck"); s_Qr = _pos("s_Qr"); s_Kr = _pos("s_Kr")
    s_Wq = _pos("s_Wq"); s_Wk = _pos("s_Wk"); s_Wv = _pos("s_Wv")
    s_O = _pos("s_O"); s_LSE = _pos("s_LSE")
    s_X = _pos("s_X")           # score
    s_E = _pos("s_E")           # exponential result
    s_P = _pos("s_P")           # P/E operand sent to the PV / PC matmul
    s_A = _pos("s_A")           # accumulator
    s_state = _pos("s_state")   # compact row state (m, l)
    s_Qacc = _pos("s_Qacc")     # Q projection accumulator
    s_Qn = _pos("s_Qn")         # projected Q^n operand
    s_Kn = _pos("s_Kn")         # projected K^n operand / materialized K
    s_V = _pos("s_V")           # projected V operand / materialized V
    s_Qt = _pos("s_Qt")         # absorbed Q-tilde
    s_Z = _pos("s_Z")           # absorbed latent output Z
    s_scratch = _nonneg("s_scratch")  # fixed scratch bytes (d_s)
    # hardware (eta)
    P_mxu = _real_pos("P_mxu")          # matmul throughput upper bound [FLOP/s] for the compute dtype
    P_exp = _real_pos("P_exp")          # exponential throughput [op/s]
    P_vpu = _real_pos("P_vpu")          # elementwise vector throughput [op/s]
    P_red = _real_pos("P_red")          # reduction throughput [op/s]
    P_layout = _real_pos("P_layout")    # layout / transpose / broadcast throughput [op/s]
    W_hbm = _real_pos("W_hbm")          # HBM->VMEM read bandwidth [B/s]
    W_hbm_w = _real_pos("W_hbm_w")      # VMEM->HBM write bandwidth [B/s]
    W_vmem = _real_pos("W_vmem")        # VMEM->VREG bandwidth [B/s]
    W_vmem_w = _real_pos("W_vmem_w")    # VREG->VMEM bandwidth [B/s]
    CP = _real_pos("CP")                # critical-path lower bound [s]
    T_launch = _nonneg("T_launch")


def all_symbols() -> dict[str, sp.Symbol]:
    return dict(_REGISTRY)


# ---------------------------------------------------------------------------
# Custom aggregate functions (stay symbolic until all arguments are integers)
# ---------------------------------------------------------------------------

def _all_int(args: Iterable) -> bool:
    return all(isinstance(a, sp.Integer) for a in args)


def causal_visible_count(Sq: int, Sk: int, delta: int) -> int:
    """sum_{i=0}^{Sq-1} max(0, min(Sk, i + delta + 1)) in O(1) integer arithmetic.

    Row i sees keys j with 0 <= j <= min(Sk-1, i+delta).  Rows split into three
    ranges: no visible key (i + delta + 1 <= 0), saturated (i + delta + 1 >= Sk),
    and a middle arithmetic progression.
    """
    Sq, Sk, delta = int(Sq), int(Sk), int(delta)
    if Sq <= 0 or Sk <= 0:
        return 0
    a = delta + 1
    # rows with i + a >= Sk contribute Sk each
    i_sat = max(0, Sk - a)                   # first saturated row index (clamped at 0)
    n_sat = max(0, Sq - i_sat)
    # middle rows: 0 < i + a < Sk  ->  i in [max(0, 1 - a), min(Sq - 1, Sk - a - 1)]
    i_lo = max(0, 1 - a)
    i_hi = min(Sq - 1, Sk - a - 1)
    middle = 0
    if i_lo <= i_hi:
        cnt = i_hi - i_lo + 1
        middle = cnt * a + (i_lo + i_hi) * cnt // 2
    return middle + n_sat * Sk


class CausalVisibleCount(sp.Function):
    """CausalVisibleCount(S_q, S_k, Delta) = sum_{i=0}^{S_q-1} max(0, min(S_k, i + Delta + 1)).

    Per (batch, head) count of visible (query, key) cells under unit-step causal
    semantics j <= i + Delta.  Spec 4.1.
    """
    nargs = 3

    @classmethod
    def eval(cls, Sq, Sk, Delta):
        if _all_int((Sq, Sk, Delta)):
            return sp.Integer(causal_visible_count(int(Sq), int(Sk), int(Delta)))
        return None


# Block statistics.  The numeric implementation lives in visibility.py; it is
# injected here to avoid a circular import.
_BLOCK_STAT_IMPL = None


def register_block_stat_impl(fn):
    global _BLOCK_STAT_IMPL
    _BLOCK_STAT_IMPL = fn


class BlockStat(sp.Function):
    """BlockStat(stat, mask, S_q_grid, S_k_grid, b_q, b_k, Delta, policy, p_q, p_k, S_q_active, S_k_active)

    Per (batch, head) statistic of the rectangle set E selected by ``policy`` on a
    (b_q x b_k) grid over the *scheduled* (S_q_grid x S_k_grid) extents, with visibility
    defined on the *active* extents (rows >= S_q_active or keys >= S_k_active are padding).
    ``stat`` is one of: rect_cells, padded_cells, selected, partial, full, future_selected,
    padding_selected, future_skipped, padding_skipped, row_visits, row_visits_padded, key_visits,
    key_visits_padded, masked_cells, masked_cells_padded, blocks_total.  ``p_q, p_k`` are sub-tile
    extents for the subdivision policy (0 when unused).  Spec 1.3 / 4.2.
    """
    nargs = 12

    @classmethod
    def eval(cls, stat, mask, Sq, Sk, bq, bk, Delta, policy, pq, pk, Sqa, Ska):
        if _BLOCK_STAT_IMPL is None:
            return None
        if _all_int((Sq, Sk, bq, bk, Delta, pq, pk, Sqa, Ska)) and isinstance(stat, Str) and isinstance(mask, Str) and isinstance(policy, Str):
            stats = _BLOCK_STAT_IMPL(str(mask), int(Sq), int(Sk), int(bq), int(bk), int(Delta), str(policy),
                                     (int(pq), int(pk)) if int(pq) > 0 else None, int(Sqa), int(Ska))
            key = str(stat)
            if key not in stats:
                raise ValueError(f"unknown block statistic {key!r}; known: {sorted(stats)}")
            return sp.Integer(stats[key])
        return None


def block_stat(stat: str, mask: str, Sq, Sk, bq, bk, Delta, policy: str, subdivide=None, Sq_active=None, Sk_active=None):
    """Grid extents default to the active extents (no separate scheduled extent declared)."""
    pq, pk = (subdivide if subdivide else (0, 0))
    Sqa = Sq if Sq_active is None else Sq_active
    Ska = Sk if Sk_active is None else Sk_active
    return BlockStat(Str(stat), Str(mask), Sq, Sk, bq, bk, Delta, Str(policy), sp.Integer(pq), sp.Integer(pk), Sqa, Ska)


# ---------------------------------------------------------------------------
# Helpers to build expressions
# ---------------------------------------------------------------------------

ceil = sp.ceiling
floor = sp.floor
Max = sp.Max
Min = sp.Min
Piecewise = sp.Piecewise


def to_sympy_number(v):
    if isinstance(v, bool):
        return sp.Integer(int(v))
    if isinstance(v, int):
        return sp.Integer(v)
    if isinstance(v, float):
        if v == float("inf"):
            return sp.oo
        if v == float("-inf"):
            return -sp.oo
        return sp.Float(v)
    if isinstance(v, sp.Basic):
        return v
    raise TypeError(f"cannot bind value of type {type(v).__name__}: {v!r}")


def substitute(expr: sp.Basic, bindings: dict[str, Any]) -> sp.Basic:
    """Replace symbols named in ``bindings`` (name -> value); leave everything else symbolic."""
    if not isinstance(expr, sp.Basic):
        expr = to_sympy_number(expr)
    mapping = {}
    for name, val in bindings.items():
        if val is None:
            continue
        s = _REGISTRY.get(name)
        if s is None or s not in expr.free_symbols:
            continue
        mapping[s] = to_sympy_number(val)
    if not mapping:
        return expr
    out = expr.xreplace(mapping)
    return out


def numeric_value(expr: sp.Basic):
    """Convert a fully bound sympy expression to int/float; None if still symbolic."""
    if expr.free_symbols or expr.has(sp.Function) and any(isinstance(a, sp.Function) and not a.is_number for a in expr.atoms(sp.Function)):
        return None
    try:
        expr = sp.nsimplify(expr) if expr.is_Rational else expr
    except Exception:  # pragma: no cover - defensive
        pass
    if expr is sp.oo:
        return float("inf")
    if expr is -sp.oo:
        return float("-inf")
    if expr.is_Integer:
        return int(expr)
    if expr.is_Rational:
        return float(expr)
    try:
        val = float(expr.evalf())
    except (TypeError, ValueError):
        return None
    if val.is_integer() and expr.is_integer:
        return int(val)
    return val


@dataclass
class Evaluation:
    expression: str
    bindings: dict[str, Any]
    residual: str | None
    value: Any
    missing: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.value is not None:
            return "bound"
        return "partial" if self.bindings else "symbolic"


def evaluate(expr, bindings: dict[str, Any]) -> Evaluation:
    """Partially evaluate ``expr`` under ``bindings`` (symbol-name -> value).

    Values that are None are treated as unknown.  Unresolved symbols are returned
    in ``missing``; the value is None (never 0) when anything is missing.
    """
    if not isinstance(expr, sp.Basic):
        expr = to_sympy_number(expr)
    # sorted, so the recorded bindings (and therefore the shipped report files) do not depend on the hash seed
    used = {s.name: bindings[s.name] for s in sorted(expr.free_symbols, key=lambda x: x.name)
            if s.name in bindings and bindings[s.name] is not None}
    sub = substitute(expr, used)
    value = numeric_value(sub)
    missing = sorted(s.name for s in sub.free_symbols)
    residual = None if value is not None else str(sub)
    return Evaluation(expression=str(expr), bindings=used, residual=residual, value=value, missing=missing)


def expr_str(expr) -> str:
    return str(expr) if isinstance(expr, sp.Basic) else str(to_sympy_number(expr))


def free_symbol_names(expr) -> list[str]:
    if not isinstance(expr, sp.Basic):
        return []
    return sorted(s.name for s in expr.free_symbols)


# ---------------------------------------------------------------------------
# Constraints / predicates
# ---------------------------------------------------------------------------

@dataclass
class Constraint:
    """A predicate over symbols with an explanation; evaluates to True/False/None(unknown)."""
    name: str
    relation: sp.Basic          # sympy relational or Boolean
    description: str = ""
    severity: str = "error"     # 'error' | 'warning' | 'info'
    sources: list[str] = field(default_factory=list)

    def evaluate(self, bindings: dict[str, Any]):
        sub = substitute(self.relation, bindings)
        if sub is sp.true or sub == True:  # noqa: E712 - sympy booleans
            return True
        if sub is sp.false or sub == False:  # noqa: E712
            return False
        return None

    def to_dict(self, bindings: dict[str, Any] | None = None) -> dict:
        result = None if bindings is None else self.evaluate(bindings)
        return {
            "name": self.name,
            "relation": str(self.relation),
            "description": self.description,
            "severity": self.severity,
            "sources": list(self.sources),
            "holds": result,
            "missing_fields": [] if result is not None else free_symbol_names(substitute(self.relation, bindings or {})),
        }


@lru_cache(maxsize=8192)
def poly_coefficients(expr: sp.Basic, x: sp.Symbol, y: sp.Symbol):
    """Split ``expr`` into a*x*y + b*x + c*y + d (+ remainder).

    Coefficients may still contain other symbols.  Terms that are not polynomial in
    (x, y) — e.g. ceil(x / L) from layout coverage — are returned in ``remainder``
    so the caller can label the envelope as approximate.

    The split is exact and additive: ``expand`` is distributive and each resulting term is
    classified on its own, so the decomposition of a sum equals the component-wise sum of the
    decompositions.  That is what lets ``stage_envelope`` attribute objects and build the total
    in a single pass.  No simplification happens here; callers that want the factored rendering
    of an aggregate coefficient call ``collect_coefficient`` on it.
    """
    expr = sp.expand(expr)
    a = b = c = d = sp.Integer(0)
    remainder = sp.Integer(0)
    terms = expr.args if isinstance(expr, sp.Add) else (expr,)
    for t in terms:
        if not t.has(x) and not t.has(y):
            d += t
            continue
        try:
            p = sp.Poly(t, x, y)
        except sp.PolynomialError:
            remainder += t
            continue
        if p.total_degree() > 2 or any(m not in ((1, 1), (1, 0), (0, 1), (0, 0)) for m in p.monoms()):
            remainder += t
            continue
        a += p.coeff_monomial(x * y)
        b += p.coeff_monomial(x)
        c += p.coeff_monomial(y)
        d += p.coeff_monomial(1)
    return a, b, c, d, remainder


@lru_cache(maxsize=8192)
def collect_coefficient(expr: sp.Basic) -> sp.Basic:
    """Canonical rendering of an aggregate envelope coefficient (common factors pulled out).

    These coefficients are sums of products of dimension and storage-width symbols, for which
    ``factor_terms`` reproduces exactly what ``simplify`` produced at a small fraction of its cost.
    """
    return sp.factor_terms(expr)


@lru_cache(maxsize=4096)
def max_expr(args: tuple) -> sp.Basic:
    """``Max`` over a tuple of expressions, memoized.

    ``Max`` pays a pairwise ordering decision over every pair of (large) polynomials.  Live-set and
    alias expressions depend only on the graph structure, not on the numeric bindings, so the same
    tuples recur on every analysis of a given algorithm variant.
    """
    if not args:
        return sp.Integer(0)
    return sp.Max(*args) if len(args) > 1 else args[0]


def clear_expression_caches() -> None:
    """Drop the structural memo tables (long-running processes, or tests asserting on cache state)."""
    poly_coefficients.cache_clear()
    collect_coefficient.cache_clear()
    max_expr.cache_clear()
