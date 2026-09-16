"""Online-softmax recurrence and partition merge (spec 3.3), numpy float64.

State per row: (m, l, A) with A of width D_A (D_v expanded / R_k absorbed).
Empty state: m = -inf, l = 0, A = 0 — handled by explicit branches so that
-inf - (-inf) never reaches exp.

Two recurrences are distinct code paths:
  unnormalized : A' = alpha A + E U,   Y = A / l at the end
  normalized   : Y' = (alpha l / l') Y + (E U) / l'   (Y kept normalized at every step)
exp_base '2' scales logits by log2(e), uses 2**(.) and restores natural-log LSE.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..errors import ReferenceSemanticsError

LOG2E = 1.0 / np.log(2.0)


@dataclass
class OnlineState:
    m: np.ndarray            # [rows]  running max (in the internal base)
    l: np.ndarray            # [rows]  running denominator
    A: np.ndarray            # [rows, D_A]  accumulator (unnormalized) or normalized output Y
    normalized: bool = False
    exp_base: str = "e"

    def copy(self):
        return OnlineState(self.m.copy(), self.l.copy(), self.A.copy(), self.normalized, self.exp_base)


def empty_state(rows: int, width: int, normalized=False, exp_base="e") -> OnlineState:
    return OnlineState(np.full(rows, -np.inf), np.zeros(rows), np.zeros((rows, width)), normalized, exp_base)


def _exp(x, base):
    return np.exp2(x) if base == "2" else np.exp(x)


def _internal(X, base):
    return X * LOG2E if base == "2" else X


def update_state(state: OnlineState, X_block, mask_block, U_block) -> OnlineState:
    """One block update.  X_block [rows, m] already scaled; mask_block bool; U_block [m, D_A]."""
    base = state.exp_base
    Xi = _internal(np.asarray(X_block, dtype=np.float64), base)
    mask_block = np.broadcast_to(np.asarray(mask_block, dtype=bool), Xi.shape)
    Xm = np.where(mask_block, Xi, -np.inf)
    has_new = mask_block.any(axis=-1)
    block_max = np.where(has_new, Xm.max(axis=-1, initial=-np.inf), -np.inf)
    m_new = np.maximum(state.m, block_max)
    out = state.copy()
    # rows where nothing (old or new) is visible stay empty
    live = np.isfinite(m_new)
    if not live.any():
        return out
    ml = m_new[live]
    old_live = np.isfinite(state.m[live])
    alpha = np.where(old_live, _exp(np.where(old_live, state.m[live] - ml, 0.0), base), 0.0)
    E = np.where(mask_block[live], _exp(Xm[live] - ml[:, None], base), 0.0)
    rowsum = E.sum(axis=-1)
    EU = E @ np.asarray(U_block, dtype=np.float64)
    l_new = alpha * state.l[live] + rowsum
    if state.normalized:
        # Y' = (alpha l / l') Y + (E U) / l'
        with np.errstate(divide="ignore", invalid="ignore"):
            w_old = np.where(l_new > 0, alpha * state.l[live] / l_new, 0.0)
            out.A[live] = w_old[:, None] * state.A[live] + EU / l_new[:, None]
    else:
        out.A[live] = alpha[:, None] * state.A[live] + EU
    out.l[live] = l_new
    out.m[live] = ml
    return out


def merge_states(s1: OnlineState, s2: OnlineState) -> OnlineState:
    """Merge two states from non-overlapping KV partitions (spec 3.3)."""
    if s1.normalized != s2.normalized or s1.exp_base != s2.exp_base:
        raise ReferenceSemanticsError("cannot merge states with different recurrence/base conventions")
    out = s1.copy()
    e1 = ~np.isfinite(s1.m)
    e2 = ~np.isfinite(s2.m)
    both = ~e1 & ~e2
    only2 = e1 & ~e2
    out.m[only2], out.l[only2], out.A[only2] = s2.m[only2], s2.l[only2], s2.A[only2]
    if both.any():
        m = np.maximum(s1.m[both], s2.m[both])
        a1 = _exp(s1.m[both] - m, s1.exp_base)
        a2 = _exp(s2.m[both] - m, s1.exp_base)
        l = a1 * s1.l[both] + a2 * s2.l[both]
        if s1.normalized:
            with np.errstate(divide="ignore", invalid="ignore"):
                out.A[both] = ((a1 * s1.l[both])[:, None] * s1.A[both] + (a2 * s2.l[both])[:, None] * s2.A[both]) / l[:, None]
        else:
            out.A[both] = a1[:, None] * s1.A[both] + a2[:, None] * s2.A[both]
        out.m[both], out.l[both] = m, l
    return out


def finalize_state(state: OnlineState, empty_row_policy="error"):
    """Return (Y, LSE_natural).  Y = A / l for unnormalized; Y = A for normalized."""
    empty = ~np.isfinite(state.m)
    if empty.any():
        if empty_row_policy == "error":
            raise ReferenceSemanticsError("row with no visible key reached finalize under policy 'error'")
        if empty_row_policy != "zero_output_neg_inf_lse":
            raise ReferenceSemanticsError(f"unknown empty_row_policy {empty_row_policy!r}")
    with np.errstate(divide="ignore", invalid="ignore"):
        if state.normalized:
            Y = np.where(empty[:, None], 0.0, state.A)
        else:
            Y = np.where(empty[:, None], 0.0, state.A / np.where(state.l > 0, state.l, 1.0)[:, None])
        if state.exp_base == "2":
            lse = (state.m + np.log2(np.where(state.l > 0, state.l, 1.0))) * np.log(2.0)
        else:
            lse = state.m + np.log(np.where(state.l > 0, state.l, 1.0))
    LSE = np.where(empty, -np.inf, lse)
    return Y, LSE


def convert_lse(lse_natural, convention: str = "natural_log"):
    """Convert a natural-log LSE to the declared output convention ('natural_log' or 'log2')."""
    if convention == "natural_log":
        return lse_natural
    if convention == "log2":
        return lse_natural / np.log(2.0)
    raise ReferenceSemanticsError(f"unknown lse_convention {convention!r}")


def blockwise_states(X, mask, U, kv_boundaries, normalized=False, exp_base="e") -> list[OnlineState]:
    """Independent states for KV partitions [kv_boundaries[i], kv_boundaries[i+1]) processed with the recurrence."""
    rows, width = X.shape[0], U.shape[1]
    states = []
    for lo, hi in zip(kv_boundaries[:-1], kv_boundaries[1:]):
        st = empty_state(rows, width, normalized, exp_base)
        if hi > lo:
            st = update_state(st, X[:, lo:hi], mask[:, lo:hi], U[lo:hi])
        states.append(st)
    return states


def online_attention(X, mask, U, block_k, normalized=False, exp_base="e", empty_row_policy="error"):
    """Sequential online softmax over KV blocks of size block_k for one (b, h): (Y, LSE)."""
    X = np.asarray(X, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)
    rows, Sk = X.shape
    st = empty_state(rows, U.shape[1], normalized, exp_base)
    for lo in range(0, Sk, block_k):
        hi = min(Sk, lo + block_k)
        st = update_state(st, X[:, lo:hi], mask[:, lo:hi], U[lo:hi])
    return finalize_state(st, empty_row_policy)
