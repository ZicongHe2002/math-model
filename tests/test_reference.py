"""Small-shape algebra: expanded vs absorbed, dense vs online vs merge (spec 3, 13.1).

Tolerances below are declared for THIS float64 test level only; they are not project defaults.
"""
import random

import numpy as np
import pytest

from conftest import TEST_SEED, random_dims
from mla_model.errors import ReferenceSemanticsError
from mla_model.reference import (blockwise_states, dense_absorbed, dense_expanded, finalize_state, merge_states,
                                 online_attention)
from mla_model.visibility import mask_matrix

F64_TOL = 1e-9  # declared for float64 references at these tiny magnitudes


def make_inputs(nprng, d):
    return dict(
        Cq=nprng.standard_normal((d["B"], d["Sq"], d["Rq"])), Ck=nprng.standard_normal((d["B"], d["Sk"], d["Rk"])),
        Qr=nprng.standard_normal((d["B"], d["H"], d["Sq"], d["Dr"])), Kr=nprng.standard_normal((d["B"], d["Sk"], d["Dr"])),
        Wq=nprng.standard_normal((d["H"], d["Rq"], d["Dn"])), Wk=nprng.standard_normal((d["H"], d["Rk"], d["Dn"])),
        Wv=nprng.standard_normal((d["H"], d["Rk"], d["Dv"])),
    )


def test_expanded_equals_absorbed_over_random_families():
    rng = random.Random(TEST_SEED)
    nprng = np.random.default_rng(TEST_SEED)
    for _ in range(40):
        d = random_dims(rng)
        delta = rng.randint(0, d["Sk"])          # ensure row 0 sees at least one key -> no empty rows
        mask = mask_matrix(d["Sq"], d["Sk"], "causal", delta)
        x = make_inputs(nprng, d)
        scale = (d["Dn"] + d["Dr"]) ** -0.5
        O1, L1 = dense_expanded(scale=scale, mask=mask, **x)
        O2, L2 = dense_absorbed(scale=scale, mask=mask, **x)
        assert O1.shape == (d["B"], d["H"], d["Sq"], d["Dv"]) and L1.shape == (d["B"], d["H"], d["Sq"])
        assert np.allclose(O1, O2, atol=F64_TOL, rtol=F64_TOL) and np.allclose(L1, L2, atol=F64_TOL, rtol=F64_TOL)


@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize("base", ["e", "2"])
def test_online_paths_match_dense(normalized, base):
    rng = random.Random(TEST_SEED + 1)
    nprng = np.random.default_rng(TEST_SEED + 1)
    for _ in range(25):
        d = random_dims(rng)
        delta = rng.randint(0, d["Sk"])
        mask = mask_matrix(d["Sq"], d["Sk"], "causal", delta)
        x = make_inputs(nprng, d)
        scale = 0.37
        O, L = dense_expanded(scale=scale, mask=mask, **x)
        b, h = rng.randrange(d["B"]), rng.randrange(d["H"])
        Qn = x["Cq"][b] @ x["Wq"][h]
        Kn = x["Ck"][b] @ x["Wk"][h]
        V = x["Ck"][b] @ x["Wv"][h]
        X = scale * (Qn @ Kn.T + (x["Qr"][b, h] @ x["Kr"][b].T if d["Dr"] else 0))
        bk = rng.randint(1, d["Sk"])
        Y, LSE = online_attention(X, mask, V, bk, normalized=normalized, exp_base=base)
        assert np.allclose(Y, O[b, h], atol=F64_TOL, rtol=F64_TOL)
        assert np.allclose(LSE, L[b, h], atol=F64_TOL, rtol=F64_TOL)   # natural-log LSE regardless of internal base


def test_partition_merge_with_empty_partitions():
    rng = random.Random(TEST_SEED + 2)
    nprng = np.random.default_rng(TEST_SEED + 2)
    for _ in range(25):
        d = random_dims(rng)
        delta = rng.randint(0, d["Sk"])
        mask = mask_matrix(d["Sq"], d["Sk"], "causal", delta)
        X = nprng.standard_normal((d["Sq"], d["Sk"]))
        U = nprng.standard_normal((d["Sk"], d["Dv"]))
        Yd, Ld = online_attention(X, mask, U, d["Sk"])   # single block == dense
        cuts = sorted(set([0, d["Sk"]] + [rng.randint(0, d["Sk"]) for _ in range(3)]))
        cuts = [0, 0] + cuts + [d["Sk"], d["Sk"]]                  # deliberately include empty partitions
        for normalized in (False, True):
            states = blockwise_states(X, mask, U, cuts, normalized=normalized)
            rng.shuffle(states)                                    # merge order must not matter
            merged = states[0]
            for s in states[1:]:
                merged = merge_states(merged, s)
            Y, L = finalize_state(merged)
            assert np.allclose(Y, Yd, atol=F64_TOL, rtol=F64_TOL) and np.allclose(L, Ld, atol=F64_TOL, rtol=F64_TOL)


def test_empty_row_policies():
    X = np.zeros((3, 4))
    mask = mask_matrix(3, 4, "causal", -2)      # rows 0 and 1 see nothing, row 2 sees key 0
    U = np.ones((4, 2))
    with pytest.raises(ReferenceSemanticsError):
        online_attention(X, mask, U, 2, empty_row_policy="error")
    Y, L = online_attention(X, mask, U, 2, empty_row_policy="zero_output_neg_inf_lse")
    assert np.all(Y[:2] == 0) and np.all(np.isneginf(L[:2])) and np.isfinite(L[2])
    assert not np.isnan(Y).any() and not np.isnan(L).any()
    d = dict(B=1, H=1, Sq=3, Sk=4, Rq=2, Rk=2, Dn=2, Dr=1, Dv=2)
    x = make_inputs(np.random.default_rng(0), d)
    O, LSE = dense_expanded(scale=1.0, mask=mask, empty_row_policy="zero_output_neg_inf_lse", **x)
    assert np.all(O[0, 0, :2] == 0) and np.all(np.isneginf(LSE[0, 0, :2]))
