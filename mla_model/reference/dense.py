"""Dense (fully materialized) references for the expanded and absorbed MLA paths.

Row-vector convention (spec 3): for one (b, h):
  expanded : Q^n = Cq Wq, K^n = Ck Wk, V = Ck Wv, X = gamma (Q^n K^n^T + Q^r K^r^T) + M, O = softmax(X) V
  absorbed : Q~ = (Cq Wq) Wk^T, X = gamma (Q~ Ck^T + Q^r K^r^T) + M, Z = softmax(X) Ck, O = Z Wv
All-masked rows follow ``empty_row_policy``: 'error' raises, 'zero_output_neg_inf_lse' gives O = 0, LSE = -inf.
"""
from __future__ import annotations

import numpy as np

from ..errors import ReferenceSemanticsError


def _check_inputs(Cq, Ck, Qr, Kr, Wq, Wk, Wv):
    B, Sq, Rq = Cq.shape
    B2, Sk, Rk = Ck.shape
    H, Rq2, Dn = Wq.shape
    H2, Rk2, Dn2 = Wk.shape
    H3, Rk3, Dv = Wv.shape
    if Qr is None or Kr is None:
        Dr = 0
        Qr = np.zeros((B, H, Sq, 0), dtype=Cq.dtype)
        Kr = np.zeros((B, Sk, 0), dtype=Cq.dtype)
    Bq, Hq, Sq2, Dr = Qr.shape
    Bk, Sk2, Dr2 = Kr.shape
    problems = []
    if not (B == B2 == Bq == Bk):
        problems.append("batch mismatch")
    if not (Sq == Sq2):
        problems.append("Sq mismatch")
    if not (Sk == Sk2):
        problems.append("Sk mismatch")
    if not (Rq == Rq2):
        problems.append("Rq mismatch")
    if not (Rk == Rk2 == Rk3):
        problems.append("Rk mismatch")
    if not (H == H2 == H3 == Hq):
        problems.append("H mismatch")
    if not (Dn == Dn2):
        problems.append("Dn mismatch")
    if Dr != Dr2:
        problems.append("Dr mismatch")
    if problems:
        raise ValueError("reference input shapes inconsistent: " + ", ".join(problems))
    return Qr, Kr, dict(B=B, H=H, Sq=Sq, Sk=Sk, Rq=Rq, Rk=Rk, Dn=Dn, Dr=Dr, Dv=Dv)


def softmax_rows(X, mask, empty_row_policy="error"):
    """Row softmax with -inf masking.  Returns (P, LSE) with natural-log LSE.

    X: [..., Sq, Sk] float64, mask: broadcastable bool.  Fully masked rows follow the policy.
    """
    X = np.asarray(X, dtype=np.float64)
    mask = np.broadcast_to(np.asarray(mask, dtype=bool), X.shape)
    Xm = np.where(mask, X, -np.inf)
    any_visible = mask.any(axis=-1)
    if not any_visible.all():
        if empty_row_policy == "error":
            raise ReferenceSemanticsError("row with no visible key; softmax normalization undefined under policy 'error'")
        if empty_row_policy != "zero_output_neg_inf_lse":
            raise ReferenceSemanticsError(f"unknown empty_row_policy {empty_row_policy!r}")
    m = np.where(any_visible, Xm.max(axis=-1, initial=-np.inf), 0.0)  # avoid -inf - (-inf)
    E = np.where(mask, np.exp(Xm - m[..., None]), 0.0)
    l = E.sum(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        P = np.where(any_visible[..., None], E / np.where(l > 0, l, 1.0)[..., None], 0.0)
        LSE = np.where(any_visible, m + np.log(np.where(l > 0, l, 1.0)), -np.inf)
    return P, LSE


def dense_attention_from_scores(X, mask, U, empty_row_policy="error"):
    """Given raw scores X (already scaled), visibility mask and value operand U -> (Y = P U, LSE)."""
    P, LSE = softmax_rows(X, mask, empty_row_policy)
    return P @ U, LSE


def dense_expanded(Cq, Ck, Qr, Kr, Wq, Wk, Wv, scale, mask, empty_row_policy="error", dtype=np.float64):
    """Expanded path.  Returns (O [B,H,Sq,Dv], LSE [B,H,Sq])."""
    Cq, Ck, Wq, Wk, Wv = (np.asarray(a, dtype=dtype) for a in (Cq, Ck, Wq, Wk, Wv))
    Qr, Kr, d = _check_inputs(Cq, Ck, None if Qr is None else np.asarray(Qr, dtype=dtype),
                              None if Kr is None else np.asarray(Kr, dtype=dtype), Wq, Wk, Wv)
    Qn = np.einsum("bsr,hrd->bhsd", Cq, Wq)          # [B,H,Sq,Dn]
    Kn = np.einsum("btr,hrd->bhtd", Ck, Wk)          # [B,H,Sk,Dn]
    V = np.einsum("btr,hrv->bhtv", Ck, Wv)           # [B,H,Sk,Dv]
    X = np.einsum("bhsd,bhtd->bhst", Qn, Kn)
    if d["Dr"] > 0:
        X = X + np.einsum("bhsr,btr->bhst", Qr, Kr)
    X = scale * X
    O, LSE = dense_attention_from_scores(X, mask, V, empty_row_policy)
    return O, LSE


def dense_absorbed(Cq, Ck, Qr, Kr, Wq, Wk, Wv, scale, mask, empty_row_policy="error", dtype=np.float64):
    """Absorbed two-step path.  Returns (O [B,H,Sq,Dv], LSE [B,H,Sq]); state width is R_k."""
    Cq, Ck, Wq, Wk, Wv = (np.asarray(a, dtype=dtype) for a in (Cq, Ck, Wq, Wk, Wv))
    Qr, Kr, d = _check_inputs(Cq, Ck, None if Qr is None else np.asarray(Qr, dtype=dtype),
                              None if Kr is None else np.asarray(Kr, dtype=dtype), Wq, Wk, Wv)
    Qn = np.einsum("bsr,hrd->bhsd", Cq, Wq)          # [B,H,Sq,Dn]
    Qt = np.einsum("bhsd,hkd->bhsk", Qn, Wk)         # Q~ = Q^n W^k^T  [B,H,Sq,Rk]
    X = np.einsum("bhsk,btk->bhst", Qt, Ck)
    if d["Dr"] > 0:
        X = X + np.einsum("bhsr,btr->bhst", Qr, Kr)
    X = scale * X
    Z, LSE = dense_attention_from_scores(X, mask, Ck[:, None, :, :], empty_row_policy)   # Z = P Ck  [B,H,Sq,Rk]
    O = np.einsum("bhsk,hkv->bhsv", Z, Wv)
    return O, LSE
