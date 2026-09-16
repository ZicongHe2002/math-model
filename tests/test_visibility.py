"""Visibility / rectangle formulas versus explicit enumeration (spec 4, 13.1)."""
import itertools
import random

import pytest

from conftest import TEST_SEED
from mla_model import visibility as V
from mla_model.symbolic import CausalVisibleCount, Sym, evaluate, causal_visible_count


@pytest.mark.parametrize("Sq,Sk", [(s, k) for s in (0, 1, 2, 5, 9) for k in (0, 1, 3, 7, 12)])
def test_causal_closed_form_matches_enumeration(Sq, Sk):
    for delta in range(-(Sq + 3), Sk + 4):
        assert causal_visible_count(Sq, Sk, delta) == V.enumerate_visible(Sq, Sk, delta)


def test_causal_special_cases():
    for S in range(1, 20):
        assert causal_visible_count(S, S, 0) == S * (S + 1) // 2
    # decode: Sq = 1 with cache offset L-1 sees all L keys, not one token
    for L in (1, 5, 300):
        assert causal_visible_count(1, L, L - 1) == L
    # dense scenario through the symbolic builder
    e = V.visible_count_expr("none")
    assert evaluate(e, {"B": 2, "H": 3, "S_q": 4, "S_k": 5}).value == 120


def test_symbolic_count_stays_symbolic_until_bound():
    e = V.visible_count_expr("causal")
    ev = evaluate(e, {"B": 2, "H": 2})
    assert ev.value is None and set(ev.missing) == {"Delta", "S_k", "S_q"}
    assert "CausalVisibleCount" in ev.residual
    assert evaluate(e, {"B": 2, "H": 2, "S_q": 3, "S_k": 3, "Delta": 0}).value == 24


def test_block_classification():
    assert V.classify_block(0, 4, 8, 4, 0) == "future"        # keys 8..11 vs queries 0..3
    assert V.classify_block(8, 4, 0, 4, 0) == "visible"       # keys 0..3 vs queries 8..11
    assert V.classify_block(0, 4, 0, 4, 0) == "partial"
    assert V.classify_block(0, 4, 4, 4, 3) == "partial"       # offset makes key 4..6 visible to query 3
    assert V.classify_block(0, 4, 4, 4, 7) == "visible"       # key 7 visible to query 0 only when Delta >= 7
    assert V.classify_block(0, 0, 4, 4, 0) == "empty"
    assert V.classify_block(0, 4, 8, 4, 0, mask="none") == "visible"


def test_rect_cells_against_brute_force_enumeration():
    rng = random.Random(TEST_SEED)
    for _ in range(300):
        Sq, Sk = rng.randint(1, 14), rng.randint(1, 14)
        bq, bk = rng.randint(1, 6), rng.randint(1, 6)
        delta = rng.randint(-Sq - 1, Sk + 1)
        for pol in ("full_scan", "skip_future"):
            st = V.block_stats("causal", Sq, Sk, bq, bk, delta, pol)
            assert st["rect_cells"] == V.enumerate_rect_cells("causal", Sq, Sk, bq, bk, delta, pol)
            assert st["rect_cells"] >= causal_visible_count(Sq, Sk, delta)
            assert st["padded_cells"] == st["selected"] * bq * bk
            assert st["selected"] + st["future_skipped"] == st["blocks_total"]
        assert V.block_stats("causal", Sq, Sk, bq, bk, delta, "full_scan")["rect_cells"] == Sq * Sk
        sub = V.block_stats("causal", Sq, Sk, bq, bk, delta, "skip_future_subdivide", (1, 1))
        assert sub["rect_cells"] == causal_visible_count(Sq, Sk, delta)   # 1x1 subdivision recovers exact visibility
        pq, pk = rng.randint(1, bq), rng.randint(1, bk)
        sub2 = V.block_stats("causal", Sq, Sk, bq, bk, delta, "skip_future_subdivide", (pq, pk))
        assert sub2["rect_cells"] == V.enumerate_rect_cells("causal", Sq, Sk, bq, bk, delta, "skip_future_subdivide", (pq, pk))
        assert causal_visible_count(Sq, Sk, delta) <= sub2["rect_cells"] <= V.block_stats("causal", Sq, Sk, bq, bk, delta, "skip_future")["rect_cells"]


def test_dense_mask_all_policies_cover_everything():
    for Sq, Sk, bq, bk in itertools.product((1, 5, 8), (1, 6, 8), (1, 3, 4), (2, 3, 8)):
        for pol in ("full_scan", "skip_future"):
            assert V.block_stats("none", Sq, Sk, bq, bk, 0, pol)["rect_cells"] == Sq * Sk


@pytest.mark.parametrize("S,p", [(8, 2), (8, 4), (8, 8), (12, 3), (16, 4), (64, 16)])
def test_square_causal_closed_forms(S, p):
    st = V.block_stats("causal", S, S, p, p, 0, "skip_future")
    assert st["rect_cells"] == (S * S + S * p) // 2
    assert st["rect_cells"] - causal_visible_count(S, S, 0) == S * (p - 1) // 2
    assert st["partial"] == S // p and st["full"] == (S // p) * (S // p - 1) // 2


def test_block_stat_symbolic_binding_equals_numeric():
    e = V.rect_cells_expr("causal", "skip_future")
    ev = evaluate(e, {"B": 2, "H": 3, "S_q": 7, "S_k": 9, "b_q": 3, "b_k": 4, "Delta": 1})
    assert ev.value == 6 * V.block_stats("causal", 7, 9, 3, 4, 1, "skip_future")["rect_cells"]
    partial = evaluate(e, {"B": 2, "H": 3, "S_q": 7})
    assert partial.value is None and "b_q" in partial.missing


def test_ragged_visible_count_is_explicit_sum():
    ragged = {"Sq": [3, 5], "Sk": [4, 6], "Delta": [0, 2]}
    e = V.visible_count_expr("causal", ragged)
    ev = evaluate(e, {"H": 2})
    assert ev.value == 2 * (causal_visible_count(3, 4, 0) + causal_visible_count(5, 6, 2))


def test_arithmetic_block_stats_match_enumeration():
    rng = random.Random(TEST_SEED + 7)
    for _ in range(400):
        Sq, Sk = rng.randint(1, 20), rng.randint(1, 20)
        bq, bk = rng.randint(1, 7), rng.randint(1, 7)
        delta = rng.randint(-Sq - 2, Sk + 2)
        mask = rng.choice(["causal", "causal", "none"])
        for pol in ("full_scan", "skip_future"):
            assert V.block_stats_arithmetic(mask, Sq, Sk, bq, bk, delta, pol) == V.block_stats_enumerate(mask, Sq, Sk, bq, bk, delta, pol)
        pq, pk = rng.randint(1, bq), rng.randint(1, bk)
        assert V.block_stats_arithmetic(mask, Sq, Sk, bq, bk, delta, "skip_future_subdivide", (pq, pk)) == \
            V.block_stats_enumerate(mask, Sq, Sk, bq, bk, delta, "skip_future_subdivide", (pq, pk))


def test_scheduled_grid_wider_than_active():
    rng = random.Random(TEST_SEED + 8)
    for _ in range(150):
        Sqa, Ska = rng.randint(1, 9), rng.randint(1, 9)
        Sq, Sk = Sqa + rng.randint(0, 6), Ska + rng.randint(0, 6)
        bq, bk = rng.randint(1, 4), rng.randint(1, 4)
        delta = rng.randint(-3, 6)
        pol = rng.choice(["full_scan", "skip_future"])
        st = V.block_stats("causal", Sq, Sk, bq, bk, delta, pol, None, Sqa, Ska)
        assert st["rect_cells"] == V.enumerate_rect_cells("causal", Sq, Sk, bq, bk, delta, pol, None, Sqa, Ska)
        covered = set()
        for blk in V.iter_blocks("causal", Sq, Sk, bq, bk, delta, pol, None, Sq_active=Sqa, Sk_active=Ska):
            if blk.selected:
                covered |= {(i, j) for i in range(blk.q0, blk.q0 + blk.n) for j in range(blk.k0, blk.k0 + blk.m)}
        visible = {(i, j) for i in range(Sqa) for j in range(Ska) if j <= i + delta}
        assert visible <= covered                                     # every visible cell is executed
        if pol == "full_scan":
            assert st["rect_cells"] == Sq * Sk and st["padding_skipped"] == 0
        else:
            # cells of fully-padding blocks are never executed under skip_future
            assert all(not (blk.kind == "padding" and blk.selected) for blk in V.iter_blocks("causal", Sq, Sk, bq, bk, delta, pol, None, Sq_active=Sqa, Sk_active=Ska))
        # same active and grid extents reproduce the plain statistics
        assert V.block_stats("causal", Sqa, Ska, bq, bk, delta, pol, None, Sqa, Ska) == V.block_stats("causal", Sqa, Ska, bq, bk, delta, pol)


def test_scheduled_grid_statistics_match_enumeration():
    """A grid wider than the active extents: every statistic, against brute-force enumeration."""
    rng = random.Random(TEST_SEED + 11)
    for _ in range(400):
        mask = rng.choice(["causal", "none"])
        Sqa, Ska = rng.randint(0, 9), rng.randint(0, 9)
        Sq, Sk = max(1, Sqa + rng.randint(0, 6)), max(1, Ska + rng.randint(0, 6))
        bq, bk = rng.randint(1, 5), rng.randint(1, 5)
        delta = rng.randint(-8, 9)
        for policy in ("full_scan", "skip_future"):
            a = V.block_stats_arithmetic(mask, Sq, Sk, bq, bk, delta, policy, None, None, None, Sqa, Ska)
            e = V.block_stats_enumerate(mask, Sq, Sk, bq, bk, delta, policy, None, Sqa, Ska)
            assert a == e, (mask, Sq, Sk, bq, bk, delta, policy, Sqa, Ska,
                            {k: (a[k], e[k]) for k in a if a[k] != e[k]})
        pq, pk = rng.randint(1, bq), rng.randint(1, bk)
        a = V.block_stats_arithmetic(mask, Sq, Sk, bq, bk, delta, "skip_future_subdivide", (pq, pk), None, None, Sqa, Ska)
        e = V.block_stats_enumerate(mask, Sq, Sk, bq, bk, delta, "skip_future_subdivide", (pq, pk), Sqa, Ska)
        assert a == e, (mask, Sq, Sk, bq, bk, delta, (pq, pk), Sqa, Ska,
                        {k: (a[k], e[k]) for k in a if a[k] != e[k]})


def test_no_analysis_path_enumerates_rectangles():
    """A huge scheduled grid with tiny tiles must stay fast, and the oracle must refuse rather than hang."""
    import time
    t = time.time()
    st = V.block_stats("causal", 131072, 131072, 8, 8, 0, "skip_future", None, 100000, 120000)
    elapsed = time.time() - t
    assert st["rect_cells"] > 0 and elapsed < 2.0, elapsed          # 2.7e8 rectangles in the grid
    with pytest.raises(ValueError, match="ENUMERATION_BLOCK_LIMIT"):
        V.block_stats_enumerate("causal", 131072, 131072, 8, 8, 0, "skip_future")


# --------------------------------------------------------------------------------------------------
# Cross-checks between the package's own two implementations: the closed-form evaluator and the
# brute-force enumerator.  These are differential tests, not independent oracles — both sides are this
# package's code — so the fast path is never trusted on its own, but a shared misreading of the
# specification would pass.  tests/independent/test_oracle_visibility.py holds the independent side.
# --------------------------------------------------------------------------------------------------

def test_arithmetic_block_stats_match_enumeration_exhaustively():
    from mla_model.visibility import block_stats_arithmetic, block_stats_enumerate
    checked = 0
    for mask in ("causal", "none"):
        for Sq in range(0, 8):
            for Sk in range(0, 8):
                for bq in range(1, 5):
                    for bk in range(1, 5):
                        for delta in range(-5, 7):
                            for policy in ("full_scan", "skip_future"):
                                a = block_stats_arithmetic(mask, Sq, Sk, bq, bk, delta, policy, None)
                                e = block_stats_enumerate(mask, Sq, Sk, bq, bk, delta, policy, None)
                                assert a == e, (mask, Sq, Sk, bq, bk, delta, policy,
                                                {k: (a[k], e[k]) for k in a if a[k] != e[k]})
                                checked += 1
    assert checked >= 20000, checked


def test_arithmetic_block_stats_match_enumeration_for_subdivision():
    from mla_model.visibility import block_stats_arithmetic, block_stats_enumerate
    rng = random.Random(TEST_SEED + 7)
    for _ in range(1500):
        mask = rng.choice(["causal", "none"])
        Sq, Sk = rng.randint(0, 14), rng.randint(0, 14)
        bq, bk = rng.randint(1, 6), rng.randint(1, 6)
        pq, pk = rng.randint(1, bq), rng.randint(1, bk)
        delta = rng.randint(-8, 10)
        a = block_stats_arithmetic(mask, Sq, Sk, bq, bk, delta, "skip_future_subdivide", (pq, pk))
        e = block_stats_enumerate(mask, Sq, Sk, bq, bk, delta, "skip_future_subdivide", (pq, pk))
        assert a == e, (mask, Sq, Sk, bq, bk, delta, (pq, pk), {k: (a[k], e[k]) for k in a if a[k] != e[k]})


def test_scheduled_grid_covers_every_visible_cell():
    """A scheduled grid wider than the active extents must still execute every visible cell."""
    from mla_model.visibility import iter_blocks, block_stats
    rng = random.Random(TEST_SEED + 8)
    for _ in range(200):
        Sqa, Ska = rng.randint(1, 9), rng.randint(1, 9)
        Sq, Sk = Sqa + rng.randint(0, 6), Ska + rng.randint(0, 6)
        bq, bk = rng.randint(1, 4), rng.randint(1, 4)
        delta = rng.randint(-3, 6)
        for policy in ("full_scan", "skip_future"):
            covered = set()
            for b in iter_blocks("causal", Sq, Sk, bq, bk, delta, policy, Sq_active=Sqa, Sk_active=Ska):
                if b.selected:
                    covered |= {(i, j) for i in range(b.q0, b.q0 + b.n) for j in range(b.k0, b.k0 + b.m)}
            visible = {(i, j) for i in range(Sqa) for j in range(Ska) if j <= i + delta}
            assert visible <= covered, (Sq, Sk, Sqa, Ska, bq, bk, delta, policy)
            st = block_stats("causal", Sq, Sk, bq, bk, delta, policy, None, Sqa, Ska)
            assert st["rect_cells"] == sum(b.n * b.m for b in iter_blocks("causal", Sq, Sk, bq, bk, delta, policy,
                                                                          Sq_active=Sqa, Sk_active=Ska) if b.selected)
