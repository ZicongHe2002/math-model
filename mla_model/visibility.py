"""Visible positions, rectangular execution and padding (spec 4).

* ``causal_visible_count`` closed form (O(1)) and ``enumerate_visible`` brute force,
* block classification for arbitrary rectangles/offsets/tails,
* block statistics for the selection policies full_scan / skip_future /
  skip_future_subdivide, at logical or padded execution extents,
* symbolic expressions built from the custom sympy aggregates.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterator

import sympy as sp

from . import symbolic as S
from .symbolic import Sym, CausalVisibleCount, block_stat, causal_visible_count

STATS = ("rect_cells", "padded_cells", "selected", "partial", "full", "future_selected", "padding_selected",
         "future_skipped", "padding_skipped",
         "row_visits", "row_visits_padded", "key_visits", "key_visits_padded", "masked_cells", "masked_cells_padded",
         "blocks_total")


ENUMERATION_BLOCK_LIMIT = 1 << 20   # these helpers are test oracles, not an analysis path


def _guard_enumeration(Sq: int, Sk: int, bq: int, bk: int, who: str) -> None:
    """Refuse to walk a rectangle grid large enough to look like a hang.

    ``block_stats`` evaluates the same statistics in O(number of active query blocks), so a caller
    that reaches this limit wants that function, not the oracle.
    """
    if bq <= 0 or bk <= 0:
        return
    n = (-(-Sq // bq)) * (-(-Sk // bk))
    if n > ENUMERATION_BLOCK_LIMIT:
        raise ValueError(
            f"{who} is a brute-force oracle: the {Sq}x{Sk} grid with {bq}x{bk} tiles holds {n} rectangles, "
            f"above ENUMERATION_BLOCK_LIMIT={ENUMERATION_BLOCK_LIMIT}. Use block_stats / block_stats_arithmetic, "
            f"which compute the same statistics in O(number of active query blocks).")


def enumerate_visible(Sq: int, Sk: int, delta: int) -> int:
    """Brute-force count of visible cells (testing oracle for the closed form)."""
    return sum(1 for i in range(Sq) for j in range(Sk) if j <= i + delta)


def mask_matrix(Sq: int, Sk: int, mask: str, delta: int = 0):
    """Boolean visibility matrix mu[i, j] for uniform semantics (small shapes; references/tests)."""
    import numpy as np
    if mask == "none":
        return np.ones((Sq, Sk), dtype=bool)
    if mask == "causal":
        i = np.arange(Sq)[:, None]
        j = np.arange(Sk)[None, :]
        return j <= i + delta
    raise ValueError(f"unknown mask kind {mask!r}")


def classify_block(q0: int, n: int, k0: int, m: int, delta: int, mask: str = "causal", Sq_active: int | None = None, Sk_active: int | None = None) -> str:
    """Classify rectangle I=[q0,q0+n) x J=[k0,k0+m) under unit-step causal j <= i + delta (spec 4.2).

    With scheduled extents larger than the active ones, rows >= Sq_active / keys >= Sk_active are
    padding: a block entirely in the padding region is 'padding'; a block touching it is at most 'partial'.
    """
    if n <= 0 or m <= 0:
        return "empty"
    if (Sq_active is not None and q0 >= Sq_active) or (Sk_active is not None and k0 >= Sk_active):
        return "padding"
    touches_padding = (Sq_active is not None and q0 + n > Sq_active) or (Sk_active is not None and k0 + m > Sk_active)
    if mask == "none":
        return "partial" if touches_padding else "visible"
    if k0 > q0 + n - 1 + delta:
        return "future"
    if k0 + m - 1 <= q0 + delta and not touches_padding:
        return "visible"
    return "partial"


@dataclass(frozen=True)
class Block:
    q0: int
    n: int
    k0: int
    m: int
    kind: str
    selected: bool
    n_exec: int          # executed extents under the requested padding (logical when padded=False)
    m_exec: int
    from_subdivision: bool = False


def iter_blocks(mask: str, Sq: int, Sk: int, bq: int, bk: int, delta: int, policy: str,
                subdivide: tuple[int, int] | None = None, padded: bool = False,
                Sq_active: int | None = None, Sk_active: int | None = None) -> Iterator[Block]:
    """Yield every block of the (scheduled) grid (or sub-block for the subdivision policy) with its class and selection.

    Selection: full_scan executes everything (padding included); skip_future skips 'future' and
    'padding' blocks (a scheduler that knows the active lengths); subdivision refines partial blocks.
    """
    if bq <= 0 or bk <= 0:
        raise ValueError("tile extents must be positive")
    if policy not in ("full_scan", "skip_future", "skip_future_subdivide"):
        raise ValueError(f"unknown rect policy {policy!r}")
    if policy == "skip_future_subdivide" and not subdivide:
        raise ValueError("subdivision policy needs (p_q, p_k)")
    skip_kinds = ("future", "padding")
    for q0 in range(0, Sq, bq):
        n = min(bq, Sq - q0)
        for k0 in range(0, Sk, bk):
            m = min(bk, Sk - k0)
            kind = classify_block(q0, n, k0, m, delta, mask, Sq_active, Sk_active)
            if policy == "skip_future_subdivide" and kind == "partial":
                pq, pk = subdivide
                for sq0 in range(q0, q0 + n, pq):
                    sn = min(pq, q0 + n - sq0)
                    for sk0 in range(k0, k0 + m, pk):
                        sm = min(pk, k0 + m - sk0)
                        skind = classify_block(sq0, sn, sk0, sm, delta, mask, Sq_active, Sk_active)
                        sel = skind not in skip_kinds
                        yield Block(sq0, sn, sk0, sm, skind, sel, pq if padded else sn, pk if padded else sm, True)
                continue
            if policy == "full_scan":
                sel = True
            else:
                sel = kind not in skip_kinds
            yield Block(q0, n, k0, m, kind, sel, bq if padded else n, bk if padded else m)


def block_stats_enumerate(mask: str, Sq: int, Sk: int, bq: int, bk: int, delta: int, policy: str,
                          subdivide: tuple[int, int] | None = None, Sq_active: int | None = None, Sk_active: int | None = None) -> dict[str, int]:
    """Reference implementation by explicit block enumeration (O(#blocks)); used by tests as the oracle for ``block_stats``.

    rect_cells       = sum_{(i,j) in E} n_i m_j           (logical extents)
    padded_cells     = sum_{(i,j) in E} n_hat_i m_hat_j   (tile / sub-tile extents)
    selected         = |E|
    partial / full   = number of selected partially / fully visible blocks
    future_skipped   = number of fully-future blocks not in E
    future_selected  = number of fully-future blocks that ARE in E (full_scan executes them; they hold no valid cell)
    padding_selected = number of blocks entirely past an active extent that are in E
    row_visits       = sum_{(i,j) in E} n_i
    row_visits_padded= sum_{(i,j) in E} n_hat_i
    key_visits       = sum_{(i,j) in E} m_j            (executed key columns, logical extents)
    key_visits_padded= sum_{(i,j) in E} m_hat_j
    masked_cells     = sum over selected blocks that are NOT fully visible of n_i m_j (cells needing a mask)
    masked_cells_padded = same with padded extents
    blocks_total     = number of blocks in the grid (sub-blocks for the subdivision policy)
    """
    if subdivide is not None:
        subdivide = tuple(subdivide)
    _guard_enumeration(Sq, Sk, bq, bk, "block_stats_enumerate")
    out = {k: 0 for k in STATS}
    for blk in iter_blocks(mask, Sq, Sk, bq, bk, delta, policy, subdivide, padded=True, Sq_active=Sq_active, Sk_active=Sk_active):
        out["blocks_total"] += 1
        if not blk.selected:
            out["padding_skipped" if blk.kind == "padding" else "future_skipped"] += 1
            continue
        out["selected"] += 1
        out["rect_cells"] += blk.n * blk.m
        out["row_visits"] += blk.n
        out["key_visits"] += blk.m
        if blk.kind == "partial":
            out["partial"] += 1
        elif blk.kind == "visible":
            out["full"] += 1
        elif blk.kind == "padding":
            out["padding_selected"] += 1
        else:
            out["future_selected"] += 1
        if blk.kind != "visible":
            out["masked_cells"] += blk.n * blk.m
        pn, pm = blk.n_exec, blk.m_exec
        out["padded_cells"] += pn * pm
        out["row_visits_padded"] += pn
        out["key_visits_padded"] += pm
        if blk.kind != "visible":
            out["masked_cells_padded"] += pn * pm
    return out


def _floor_div(a: int, b: int) -> int:
    return a // b   # Python floors toward -inf, which is what the boundary formulas need


def _row_prefixes(q0: int, n: int, nKB_act: int, K_end: int, bk: int, delta: int, mask: str,
                  q_touch: bool = False, k_touch: bool = False) -> tuple[int, int]:
    """For the q-block [q0, q0+n): (n_sel, n_full) over the ``nKB_act`` k-blocks whose origin is active.

    Blocks are indexed j = 0..nKB_act-1 with k0 = j*bk.  Under unit-step causal semantics the selected
    (non-future) blocks form a prefix j < n_sel and the fully visible ones a prefix j < n_full.
    ``K_end`` is the logical end of the last active-origin k-block (S_k when the grid does not run past
    the active extent).  ``q_touch`` / ``k_touch`` say whether this row, or the tail k-block, straddles
    the active boundary; a block touching padding is at most partially visible.
    """
    if nKB_act <= 0:
        return 0, 0
    if mask == "none":
        n_sel = nKB_act
    else:
        lim = q0 + n - 1 + delta                 # block j is not future iff j*bk <= lim
        n_sel = 0 if lim < 0 else min(nKB_act, _floor_div(lim, bk) + 1)
    if q_touch:
        return n_sel, 0                          # every block of this row straddles the active boundary
    if mask == "none":
        return n_sel, (nKB_act - 1 if k_touch else nKB_act)
    lim_full = q0 + delta                        # block j is fully visible iff k0 + m - 1 <= lim_full
    # non-tail blocks (j < nKB_act - 1) have kmax = j*bk + bk - 1
    n_full = 0 if lim_full - bk + 1 < 0 else min(nKB_act - 1, _floor_div(lim_full - bk + 1, bk) + 1)
    if n_full == nKB_act - 1 and not k_touch and K_end - 1 <= lim_full:
        n_full = nKB_act                         # the tail block is fully visible too
    return n_sel, n_full


def block_stats_arithmetic(mask: str, Sq: int, Sk: int, bq: int, bk: int, delta: int, policy: str,
                           subdivide: tuple[int, int] | None = None, pq_exec: int | None = None, pk_exec: int | None = None,
                           Sq_active: int | None = None, Sk_active: int | None = None) -> dict[str, int]:
    """O(#active q-blocks) evaluation of the block statistics (definitions as in ``block_stats_enumerate``).

    The grid spans the scheduled extents (``Sq``, ``Sk``) while visibility uses the active extents
    (``Sq_active``, ``Sk_active``, defaulting to the scheduled ones).  A block whose origin lies at or
    beyond an active extent is padding, and those occupy the three quadrants outside the (iQ x jK) block
    of active-origin rows and columns, so they are counted in closed form with no per-block work.  Inside
    that quadrant the grid is an ordinary problem over (Q_end x K_end) except that the tail block on each
    axis may straddle the active boundary, which forces it out of 'visible' into 'partial'.

    Partial blocks under the subdivision policy are evaluated recursively on their own sub-grid with the
    offset and the active extents shifted.  ``pq_exec``/``pk_exec`` are the padded extents charged here.
    """
    out = {k: 0 for k in STATS}
    if Sq <= 0 or Sk <= 0 or bq <= 0 or bk <= 0:
        return out
    pq_exec = bq if pq_exec is None else pq_exec
    pk_exec = bk if pk_exec is None else pk_exec
    Sqa = Sq if Sq_active is None else max(0, min(Sq_active, Sq))
    Ska = Sk if Sk_active is None else max(0, min(Sk_active, Sk))

    nQB, nKB = -(-Sq // bq), -(-Sk // bk)
    iQ, jK = -(-Sqa // bq), -(-Ska // bk)        # blocks whose origin lies inside the active extent
    Q_end, K_end = min(iQ * bq, Sq), min(jK * bk, Sk)
    q_tail_touch, k_tail_touch = Q_end > Sqa, K_end > Ska
    full_scan = policy == "full_scan"

    for q0 in range(0, Q_end, bq):                # rows whose origin is inside the active extent
        n = min(bq, Sq - q0)                      # logical extent comes from the scheduled grid
        q_touch = (q0 + bq >= Q_end) and q_tail_touch
        n_sel, n_full = _row_prefixes(q0, n, jK, K_end, bk, delta, mask, q_touch, k_tail_touch)
        keys_full = min(n_full * bk, K_end)       # key columns covered by fully visible blocks
        n_partial = n_sel - n_full                # partial blocks lie inside the selected prefix
        if full_scan:
            n_vis, keys_vis = nKB, Sk             # full_scan executes everything, padding included
            out["future_selected"] += jK - n_sel   # fully-future blocks executed anyway (no valid cell in them)
            out["padding_selected"] += nKB - jK    # columns past the active key extent, executed anyway
        else:
            n_vis, keys_vis = n_sel, min(n_sel * bk, K_end)
            out["future_skipped"] += jK - n_sel
            out["padding_skipped"] += nKB - jK    # columns past the active key extent
        out["blocks_total"] += nKB
        out["full"] += n_full
        if policy == "skip_future_subdivide" and n_partial > 0:
            pq, pk = subdivide
            out["selected"] += n_full
            out["rect_cells"] += n * keys_full
            out["row_visits"] += n * n_full
            out["key_visits"] += keys_full
            out["padded_cells"] += pq_exec * pk_exec * n_full
            out["row_visits_padded"] += pq_exec * n_full
            out["key_visits_padded"] += pk_exec * n_full
            for j in range(n_full, n_sel):
                k0 = j * bk
                m = min(bk, Sk - k0)
                sub = block_stats_arithmetic(mask, n, m, pq, pk, delta + q0 - k0, "skip_future", None, pq, pk,
                                             Sq_active=min(n, max(0, Sqa - q0)),
                                             Sk_active=min(m, max(0, Ska - k0)))
                for key in ("selected", "rect_cells", "row_visits", "key_visits", "padded_cells", "row_visits_padded",
                            "key_visits_padded", "masked_cells", "masked_cells_padded", "partial", "full",
                            "future_selected", "padding_selected", "future_skipped", "padding_skipped"):
                    out[key] += sub[key]
                out["blocks_total"] += sub["blocks_total"] - 1      # the coarse block is replaced by its sub-blocks
            continue
        out["selected"] += n_vis
        out["partial"] += n_partial
        out["rect_cells"] += n * keys_vis
        out["row_visits"] += n * n_vis
        out["key_visits"] += keys_vis
        out["padded_cells"] += pq_exec * pk_exec * n_vis
        out["row_visits_padded"] += pq_exec * n_vis
        out["key_visits_padded"] += pk_exec * n_vis
        out["masked_cells"] += n * (keys_vis - keys_full)
        out["masked_cells_padded"] += pq_exec * pk_exec * (n_vis - n_full)

    pad_rows = nQB - iQ                           # rows entirely past the active query extent: closed form
    if pad_rows > 0:
        sum_n = Sq - iQ * bq                      # total logical row extent of those rows
        nb = pad_rows * nKB
        out["blocks_total"] += nb
        if full_scan:
            out["selected"] += nb
            out["padding_selected"] += nb         # every block of these rows sits past the active query extent
            out["rect_cells"] += sum_n * Sk
            out["row_visits"] += sum_n * nKB
            out["key_visits"] += pad_rows * Sk
            out["padded_cells"] += pq_exec * pk_exec * nb
            out["row_visits_padded"] += pq_exec * nb
            out["key_visits_padded"] += pk_exec * nb
            out["masked_cells"] += sum_n * Sk     # a padding block is never fully visible
            out["masked_cells_padded"] += pq_exec * pk_exec * nb
        else:
            out["padding_skipped"] += nb
    return out


@lru_cache(maxsize=4096)
def block_stats(mask: str, Sq: int, Sk: int, bq: int, bk: int, delta: int, policy: str,
                subdivide: tuple[int, int] | None = None, Sq_active: int | None = None, Sk_active: int | None = None) -> dict[str, int]:
    """Per-(batch, head) statistics of the selected rectangle set E (spec 4.2); see ``block_stats_enumerate`` for definitions.

    Arithmetic O(#q-blocks) evaluation (memoized).  ``block_stats_enumerate`` is the brute-force oracle.
    """
    if policy not in ("full_scan", "skip_future", "skip_future_subdivide"):
        raise ValueError(f"unknown rect policy {policy!r}")
    if mask not in ("causal", "none"):
        raise ValueError(f"unknown mask kind {mask!r}")
    if policy == "skip_future_subdivide" and not subdivide:
        raise ValueError("subdivision policy needs (p_q, p_k)")
    if subdivide is not None:
        subdivide = tuple(subdivide)
    return block_stats_arithmetic(mask, Sq, Sk, bq, bk, delta, policy, subdivide,
                                  Sq_active=Sq_active, Sk_active=Sk_active)


def _block_stats_impl(mask, Sq, Sk, bq, bk, delta, policy, subdivide, Sq_active=None, Sk_active=None):
    if Sq_active == Sq and Sk_active == Sk:
        Sq_active = Sk_active = None
    return dict(block_stats(mask, Sq, Sk, bq, bk, delta, policy, None if subdivide is None else tuple(subdivide), Sq_active, Sk_active))


S.register_block_stat_impl(_block_stats_impl)


def enumerate_rect_cells(mask: str, Sq: int, Sk: int, bq: int, bk: int, delta: int, policy: str,
                         subdivide=None, Sq_active=None, Sk_active=None) -> int:
    """Independent brute-force check of rect_cells: count cells (i, j) covered by a selected block."""
    _guard_enumeration(Sq, Sk, bq, bk, "enumerate_rect_cells")
    covered = 0
    for blk in iter_blocks(mask, Sq, Sk, bq, bk, delta, policy, subdivide, Sq_active=Sq_active, Sk_active=Sk_active):
        if blk.selected:
            covered += sum(1 for i in range(blk.q0, blk.q0 + blk.n) for j in range(blk.k0, blk.k0 + blk.m))
    return covered


# ---------------------------------------------------------------------------
# Symbolic builders
# ---------------------------------------------------------------------------

def visible_count_expr(mask: str | None, ragged: dict | None = None) -> sp.Basic | None:
    """C_valid expression.

    uniform causal : B*H*CausalVisibleCount(S_q, S_k, Delta)
    uniform dense  : B*H*S_q*S_k
    ragged         : H * sum_b CausalVisibleCount(Sq_b, Sk_b, Delta_b)   (explicit finite sum; B concrete)
    unknown mask   : None (caller reports missing field 'mask')
    """
    if mask is None:
        return None
    if ragged:
        terms = []
        for sq, sk, dl in zip(ragged["Sq"], ragged["Sk"], ragged["Delta"]):
            if mask == "none":
                terms.append(sp.Integer(sq) * sp.Integer(sk))
            else:
                terms.append(CausalVisibleCount(sp.Integer(sq), sp.Integer(sk), sp.Integer(dl)))
        return Sym.H * sp.Add(*terms) if terms else sp.Integer(0)
    if mask == "none":
        return Sym.B * Sym.H * Sym.S_q * Sym.S_k
    return Sym.B * Sym.H * CausalVisibleCount(Sym.S_q, Sym.S_k, Sym.Delta)


def empty_rows_expr(mask: str | None) -> sp.Basic | None:
    """Rows with no visible key, uniform case.

    causal: B*H*max(0, min(S_q, -Delta)) when S_k > 0, every row (B*H*S_q) when S_k = 0;  dense: B*H*S_q if S_k = 0 else 0.
    """
    if mask is None:
        return None
    if mask == "none":
        return Sym.B * Sym.H * sp.Piecewise((Sym.S_q, sp.Eq(Sym.S_k, 0)), (sp.Integer(0), True))
    return Sym.B * Sym.H * sp.Piecewise((Sym.S_q, sp.Eq(Sym.S_k, 0)), (sp.Max(0, sp.Min(Sym.S_q, -Sym.Delta)), True))


def empty_rows_count(Sq: int, Sk: int, delta: int, mask: str) -> int:
    if Sk <= 0:
        return Sq
    if mask == "none":
        return 0
    return max(0, min(Sq, -delta))


def block_stat_expr(stat: str, mask: str, policy: str, subdivide=None, Sq=None, Sk=None, bq=None, bk=None,
                    delta=None, scheduled: bool = False) -> sp.Basic:
    """Symbolic per-(b,h) block statistic; evaluates once all tile/dim arguments are integers.

    For the dense mask the offset is irrelevant and fixed to 0 so that it never blocks evaluation.
    With ``scheduled=True`` the grid spans S_q_sched x S_k_sched while visibility uses the active S_q, S_k.
    """
    if delta is None:
        delta = sp.Integer(0) if mask == "none" else Sym.Delta
    Sq_act = Sq if Sq is not None else Sym.S_q
    Sk_act = Sk if Sk is not None else Sym.S_k
    if scheduled and Sq is None and Sk is None:
        return block_stat(stat, mask, Sym.S_q_sched, Sym.S_k_sched, bq if bq is not None else Sym.b_q, bk if bk is not None else Sym.b_k,
                          delta, policy, subdivide, Sq_active=Sq_act, Sk_active=Sk_act)
    return block_stat(stat, mask, Sq_act, Sk_act, bq if bq is not None else Sym.b_q, bk if bk is not None else Sym.b_k,
                      delta, policy, subdivide)


def rect_cells_expr(mask, policy, subdivide=None, ragged=None, scheduled=False):
    """C_rect = B*H*rect_cells (uniform) or H*sum_b rect_cells_b (ragged)."""
    return _aggregate("rect_cells", mask, policy, subdivide, ragged, scheduled)


def padded_cells_expr(mask, policy, subdivide=None, ragged=None, scheduled=False):
    return _aggregate("padded_cells", mask, policy, subdivide, ragged, scheduled)


def _aggregate(stat, mask, policy, subdivide, ragged, scheduled=False):
    if ragged:
        terms = [block_stat_expr(stat, mask, policy, subdivide, Sq=sp.Integer(sq), Sk=sp.Integer(sk), delta=sp.Integer(dl))
                 for sq, sk, dl in zip(ragged["Sq"], ragged["Sk"], ragged["Delta"])]
        return Sym.H * sp.Add(*terms) if terms else sp.Integer(0)
    return Sym.B * Sym.H * block_stat_expr(stat, mask, policy, subdivide, scheduled=scheduled)


# closed forms of spec 4.3 (square causal, S divisible by p, skip future, diagonal blocks in full)
S_sym = S.symbol("S", integer=True, positive=True)
p_sym = S.symbol("p", integer=True, positive=True)
RECT_SQUARE_CLOSED = (S_sym**2 + S_sym * p_sym) / 2
WASTE_SQUARE_CLOSED = S_sym * (p_sym - 1) / 2
VALID_SQUARE_CLOSED = S_sym * (S_sym + 1) / 2
