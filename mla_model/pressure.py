"""Stage pressure envelope, feasible region, sensitivity and discrete differences (spec 7.4, 10)."""
from __future__ import annotations

from dataclasses import dataclass, field

import sympy as sp

from .algorithms import AlgorithmGraph, stage_scenario_at
from .memory import active_locals
from .symbolic import Sym, poly_coefficients, collect_coefficient, max_expr, substitute, numeric_value


@dataclass
class Envelope:
    stage: str
    level: str | None
    a: sp.Basic
    b: sp.Basic
    c: sp.Basic
    d: sp.Basic
    remainder: sp.Basic
    provenance: dict[str, list[str]] = field(default_factory=dict)   # coefficient -> contributing object ids
    undeclared: list[str] = field(default_factory=list)              # undeclared fields the object list depends on

    @property
    def expression(self) -> sp.Basic:
        return self.a * Sym.b_q * Sym.b_k + self.b * Sym.b_q + self.c * Sym.b_k + self.d + self.remainder

    def feasible_bk(self, R):
        """b_k <= floor((R - b b_q - d) / (a b_q + c)) valid when a b_q + c > 0 (spec 7.4)."""
        denom = self.a * Sym.b_q + self.c
        return sp.floor((R - self.b * Sym.b_q - self.d) / denom), denom

    def to_dict(self, bindings: dict | None = None) -> dict:
        def show(e):
            if bindings:
                s = substitute(e, bindings)
                v = numeric_value(s)
                return {"expression": str(e), "value": v, "residual": None if v is not None else str(s)}
            return {"expression": str(e), "value": None, "residual": None}
        return {"stage": self.stage, "level": self.level, "a": show(self.a), "b": show(self.b), "c": show(self.c), "d": show(self.d),
                "remainder": show(self.remainder), "provenance": self.provenance, "undeclared": self.undeclared}


_COEFS = ("a", "b", "c", "d", "remainder")


def stage_envelope(g: AlgorithmGraph, stage: str, include_undeclared=True, level: str | None = None) -> Envelope:
    """Derive M_s(b_q, b_k) = a b_q b_k + b b_q + c b_k + d from the live object list of ``stage``.

    Exactly one coefficient extraction per live object: the split is additive, so the per-object
    coefficients both record the provenance and sum to the coefficients of the total.  (Extracting
    the total separately, as an earlier version did, re-ran the whole decomposition once per object.)

    Alias groups with identical shapes are counted once with Max over their widths; groups with
    different shapes stay as Max(...) and land in the labeled remainder (Max is not polynomial).
    """
    prov: dict[str, list[str]] = {k: [] for k in _COEFS}
    undeclared: list[str] = []
    groups: dict[str, list] = {}
    items: list[tuple[sp.Basic, str]] = []
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
            continue
        items.append((obj.size, obj.id))
    all_members: dict[str, list] = {}
    for obj in active_locals(g, include_undeclared, level):
        if obj.alias_group:
            all_members.setdefault(obj.alias_group, []).append(obj)
    for grp, members in groups.items():
        # the allocation is sized by the whole alias group, so a stage where only one tenant is live still
        # carries the group's size (matches live_set; otherwise the envelope and the live set disagree)
        members = all_members.get(grp, members)
        if len(members) == 1:
            items.append((members[0].size, members[0].id))
            continue
        shapes = {tuple(str(x) for x in m.shape) for m in members}
        if len(shapes) == 1:
            rows, cols = members[0].shape
            size = rows * cols * max_expr(tuple(m.bytes_symbol * m.buffers for m in members))
        else:
            size = max_expr(tuple(m.size for m in members))
        items.append((size, "alias[" + grp + "]:" + "|".join(m.id for m in members)))
    totals = [sp.Integer(0)] * 5
    for size, oid in items:
        parts = poly_coefficients(size, Sym.b_q, Sym.b_k)
        for i, key in enumerate(_COEFS):
            if parts[i] != 0:
                prov[key].append(oid)
            totals[i] = totals[i] + parts[i]
    a, b, c, d, rem = (collect_coefficient(t) for t in totals)
    return Envelope(stage, level, a, b, c, d, rem, prov, undeclared)


def all_envelopes(g: AlgorithmGraph, include_undeclared=True, level: str | None = None) -> dict[str, Envelope]:
    return {st: stage_envelope(g, st, include_undeclared, level) for st in g.stages}


def bq_sweep(declared_bq, limit=None, n=8):
    """A b_q axis for the constraint surface: powers of two around the declared b_q, plus the declared value.

    Spec 10.2 asks for a constraint surface rather than one best tile, so the bound is evaluated over a range
    of b_q instead of only at the declared point.  The sweep is a presentation choice, not a claim that these
    tiles are legal or supported; every row carries its own status.
    """
    vals = set()
    v = 1
    while v <= (limit or 512) and len(vals) < n:
        vals.add(v)
        v *= 2
    if declared_bq:
        vals.add(int(declared_bq))
    return sorted(vals)


from .report import SYMBOL_TO_FIELD as _SYMBOL_TO_FIELD, field_names as _fields_for   # one spelling on every surface


def feasible_region(env: Envelope, bindings: dict, R_symbol: sp.Symbol, R_value, bq_candidates, extra_undeclared=()):
    """Evaluate the b_k upper bound for each candidate b_q under budget R_value (declared model only).

    ``extra_undeclared`` carries the graph-level fields the metric of the same quantity also names (the
    aliasing flags), so the surface and the metric apply one status rule.
    """
    rows = []
    undeclared_all = sorted(set(env.undeclared) | set(extra_undeclared))
    for bq in bq_candidates:
        bnd = dict(bindings)
        bnd["b_q"] = int(bq)
        bnd[R_symbol.name] = R_value
        bound_expr, denom = env.feasible_bk(R_symbol)
        denom_v = numeric_value(substitute(denom, bnd))
        if denom_v is None:
            # unresolved symbols are a partial evaluation, as everywhere else in the report; 'unknown' is reserved
            # for a quantity that has no expression without evidence
            rows.append({"b_q": bq, "b_k_max": None, "status": "partial", "declared_b_q": bool(bindings.get("b_q") == bq),
                         "missing": sorted(set(_fields_for(substitute(denom, bnd).free_symbols)) | set(undeclared_all)),
                         "undeclared": undeclared_all})
            continue
        if denom_v <= 0:
            # no b_k term at this stage: b_k is unconstrained here, which is not the same as unevaluable
            rows.append({"b_q": bq, "b_k_max": None, "status": "not_applicable (a*b_q + c <= 0)",
                         "declared_b_q": bool(bindings.get("b_q") == bq)})
            continue
        sub = substitute(bound_expr, bnd)
        val = numeric_value(sub)
        rem_v = numeric_value(substitute(env.remainder, bnd))
        # a row is only 'bound' when it evaluated AND nothing undeclared shaped it — the same rule the
        # metric of the same quantity uses, so the two cannot disagree
        rows.append({"b_q": bq, "b_k_max": val, "status": "bound" if (val is not None and not undeclared_all) else "partial",
                     "declared_b_q": bool(bindings.get("b_q") == bq), "feasible": None if val is None else val >= 1,
                     "missing": sorted(set(_fields_for(sub.free_symbols)) | set(undeclared_all)) if (val is None or undeclared_all) else [],
                     "remainder_ignored": None if env.remainder == 0 else rem_v, "undeclared": undeclared_all})
    return rows


def sensitivities(exprs: dict[str, sp.Basic], wrt=(Sym.b_q, Sym.b_k)) -> dict[str, dict[str, sp.Basic]]:
    """Partial derivatives of continuous relaxations (ceil/floor/min/max terms are reported as non-differentiable)."""
    out = {}
    for name, e in exprs.items():
        out[name] = {}
        for s in wrt:
            if e.has(sp.ceiling) or e.has(sp.floor) or e.has(sp.Max) or e.has(sp.Min):
                out[name][s.name] = None
            else:
                out[name][s.name] = sp.simplify(sp.diff(e, s))
    return out


def _is_absent(v) -> bool:
    from .report import is_absent
    return is_absent(v)


def finite_difference(values_a: dict, values_b: dict) -> dict:
    """f(kappa') - f(kappa) for every metric present in both bindings (None when either side is unknown)."""
    out = {}
    for k in values_a:
        if k in values_b:
            va, vb = values_a[k], values_b[k]
            if _is_absent(va) or _is_absent(vb):
                # one side does not have the metric at all: a structural difference, not a numeric delta
                out[k] = va if (_is_absent(va) and _is_absent(vb)) else ("appears" if _is_absent(va) else "disappears")
            elif va is None or vb is None:
                out[k] = None
            elif isinstance(va, (list, tuple)) and isinstance(vb, (list, tuple)):
                out[k] = [y - x for x, y in zip(va, vb)] if len(va) == len(vb) else None   # interval endpoints
            elif isinstance(va, (int, float)) and isinstance(vb, (int, float)) and not isinstance(va, bool):
                out[k] = vb - va
            else:
                out[k] = None
    return out
