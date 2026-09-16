"""Work accounting: matmul FLOPs (useful / rectangular-scheduled / compiled) and vector-operation counts (spec 5)."""
from __future__ import annotations

import sympy as sp

from .algorithms import AlgorithmGraph
from .symbolic import Sym

B, H, Sq, Sk, Rq, Rk, Dn, Dr, Dv = Sym.B, Sym.H, Sym.S_q, Sym.S_k, Sym.R_q, Sym.R_k, Sym.D_n, Sym.D_r, Sym.D_v


def f_matmul(M, N, K):
    """F_matmul(M, N, K) = 2 M N K  (FMA = 2 convention; not instruction count)."""
    return 2 * M * N * K


# --- closed forms from the specification (projection boundary: every row once) ---

def expanded_components(C, rows_q=None, rows_k=None):
    rows_q = B * H * Sq if rows_q is None else rows_q
    rows_k = B * H * Sk if rows_k is None else rows_k
    return {
        "F_Q": f_matmul(rows_q, Dn, Rq),
        "F_K": f_matmul(rows_k, Dn, Rk),
        "F_V": f_matmul(rows_k, Dv, Rk),
        "F_QK_n": 2 * C * Dn,
        "F_QK_r": 2 * C * Dr,
        "F_PV": 2 * C * Dv,
    }


def expanded_total(C, rows_q=None, rows_k=None):
    """F_E = 2BH[Sq Rq Dn + Sk Rk (Dn + Dv)] + 2C(Dn + Dr + Dv)   (rows_q = BH Sq, rows_k = BH Sk; ragged sums allowed)."""
    rows_q = B * H * Sq if rows_q is None else rows_q
    rows_k = B * H * Sk if rows_k is None else rows_k
    return 2 * rows_q * Rq * Dn + 2 * rows_k * Rk * (Dn + Dv) + 2 * C * (Dn + Dr + Dv)


def absorbed_components(C, rows_q=None, rows_k=None):
    rows_q = B * H * Sq if rows_q is None else rows_q
    return {
        "F_Q": f_matmul(rows_q, Dn, Rq),
        "F_Qtilde": f_matmul(rows_q, Rk, Dn),
        "F_QC": 2 * C * Rk,
        "F_QK_r": 2 * C * Dr,
        "F_PC": 2 * C * Rk,
        "F_ZWv": f_matmul(rows_q, Dv, Rk),
    }


def absorbed_total(C, rows_q=None, rows_k=None):
    """F_A = 2BH[Sq Rq Dn + Sq Rk (Dn + Dv)] + 2C(2Rk + Dr)."""
    rows_q = B * H * Sq if rows_q is None else rows_q
    return 2 * rows_q * Rq * Dn + 2 * rows_q * Rk * (Dn + Dv) + 2 * C * (2 * Rk + Dr)


def absorbed_precomputed_total(C, rows_q=None, rows_k=None):
    """Q~ = Cq W~ directly: 2BH Sq Rq Rk + 2BH Sq Rk Dv + 2C(2Rk + Dr); merge work 2 H Rq Dn Rk / n_reuse listed separately."""
    rows_q = B * H * Sq if rows_q is None else rows_q
    return 2 * rows_q * Rq * Rk + 2 * rows_q * Rk * Dv + 2 * C * (2 * Rk + Dr)


def weight_merge_work():
    return f_matmul(H * Rq, Rk, Dn)


def path_difference(C, rows_q=None, rows_k=None):
    """F_A - F_E = 2BH Rk (Dn + Dv)(Sq - Sk) + 2C(2Rk - Dn - Dv)   (= 2 Rk (Dn+Dv)(rows_q - rows_k) + ...)."""
    rows_q = B * H * Sq if rows_q is None else rows_q
    rows_k = B * H * Sk if rows_k is None else rows_k
    return 2 * Rk * (Dn + Dv) * (rows_q - rows_k) + 2 * C * (2 * Rk - Dn - Dv)


def projection_general(NQ, NK, NV):
    """F_proj = 2 NQ Rq Dn + 2 NK Rk Dn + 2 NV Rk Dv  (spec 5.3)."""
    return 2 * NQ * Rq * Dn + 2 * NK * Rk * Dn + 2 * NV * Rk * Dv


def total_for(algorithm: str, C, rows_q=None, rows_k=None):
    if algorithm == "expanded":
        return expanded_total(C, rows_q, rows_k)
    if algorithm == "absorbed_two_step":
        return absorbed_total(C, rows_q, rows_k)
    if algorithm == "absorbed_precomputed":
        return absorbed_precomputed_total(C, rows_q, rows_k)
    raise ValueError(algorithm)


def components_for(algorithm: str, C, rows_q=None, rows_k=None):
    rq = B * H * Sq if rows_q is None else rows_q
    if algorithm == "expanded":
        return expanded_components(C, rows_q, rows_k)
    if algorithm == "absorbed_two_step":
        return absorbed_components(C, rows_q, rows_k)
    if algorithm == "absorbed_precomputed":
        comps = absorbed_components(C, rows_q, rows_k)
        comps.pop("F_Q")
        comps["F_Qtilde"] = f_matmul(rq, Rk, Rq)
        comps["F_Wmerge_amortized"] = weight_merge_work() / Sym.n_reuse
        return comps
    raise ValueError(algorithm)


# --- useful work under the declared projection scope (spec 5.3 generalisation) ---

def useful_total(algorithm: str, C, NQ, NK, NV):
    """Useful work with projected-row counts from the declared projection scope.

    expanded            : 2 NQ Rq Dn + 2 NK Rk Dn + 2 NV Rk Dv + 2C(Dn + Dr + Dv)
    absorbed_two_step   : 2 NQ Rq Dn + 2 NQ Dn Rk + 2C(2Rk + Dr) + 2 NQ Rk Dv
    absorbed_precomputed: 2 NQ Rq Rk + 2C(2Rk + Dr) + 2 NQ Rk Dv   (+ 2 H Rq Dn Rk / n_reuse merge share)
    With NQ = BH Sq and NK = NV = BH Sk these reduce to the boxed F_E / F_A of the specification.
    """
    if algorithm == "expanded":
        return 2 * NQ * Rq * Dn + 2 * NK * Rk * Dn + 2 * NV * Rk * Dv + 2 * C * (Dn + Dr + Dv)
    if algorithm == "absorbed_two_step":
        return 2 * NQ * Rq * Dn + 2 * NQ * Dn * Rk + 2 * C * (2 * Rk + Dr) + 2 * NQ * Rk * Dv
    if algorithm == "absorbed_precomputed":
        # merge work 2 H Rq Dn Rk / n_reuse is listed as a separate component (spec 3.2), not folded into the total
        return 2 * NQ * Rq * Rk + 2 * C * (2 * Rk + Dr) + 2 * NQ * Rk * Dv
    raise ValueError(algorithm)


def useful_components(algorithm: str, C, NQ, NK, NV):
    if algorithm == "expanded":
        return {"F_Q": f_matmul(NQ, Dn, Rq), "F_K": f_matmul(NK, Dn, Rk), "F_V": f_matmul(NV, Dv, Rk),
                "F_QK_n": 2 * C * Dn, "F_QK_r": 2 * C * Dr, "F_PV": 2 * C * Dv}
    if algorithm == "absorbed_two_step":
        return {"F_Q": f_matmul(NQ, Dn, Rq), "F_Qtilde": f_matmul(NQ, Rk, Dn), "F_QC": 2 * C * Rk, "F_QK_r": 2 * C * Dr,
                "F_PC": 2 * C * Rk, "F_ZWv": f_matmul(NQ, Dv, Rk)}
    if algorithm == "absorbed_precomputed":
        return {"F_Qtilde": f_matmul(NQ, Rk, Rq), "F_QC": 2 * C * Rk, "F_QK_r": 2 * C * Dr, "F_PC": 2 * C * Rk,
                "F_ZWv": f_matmul(NQ, Dv, Rk), "F_Wmerge_amortized": weight_merge_work() / Sym.n_reuse}
    raise ValueError(algorithm)


def path_difference_general(C, NQ, NK, NV):
    """F_A - F_E with explicit row counts: 2 NQ Rk (Dn + Dv) - 2 NK Rk Dn - 2 NV Rk Dv + 2C(2Rk - Dn - Dv)."""
    return useful_total("absorbed_two_step", C, NQ, NK, NV) - useful_total("expanded", C, NQ, NK, NV)


# --- graph-based accounting -------------------------------------------------

def graph_matmul_work(g: AlgorithmGraph) -> dict[str, sp.Basic]:
    """Sum of 2MNK * multiplicity per category from the algorithm graph (executed cells C_exec)."""
    out = {}
    for cat in ("projection", "attention", "output_projection", "weight_merge"):
        val = g.matmul_flops(cat)
        if val != 0:
            out[cat] = sp.expand(val)
    out["total"] = sp.expand(g.matmul_flops())
    return out


def graph_vector_work(g: AlgorithmGraph) -> dict[str, sp.Basic]:
    """Logical vector-operation counts grouped by kind and by resource class."""
    return {"by_kind": {k: sp.expand(v) for k, v in g.vector_counts().items()},
            "by_resource": {k: sp.expand(v) for k, v in g.resource_counts().items()}}
