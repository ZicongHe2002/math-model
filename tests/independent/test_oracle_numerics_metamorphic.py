"""INDEPENDENT ORACLE: (A) numerics — an explicit-loop float64 MLA forward written from spec 3.1/3.2, compared with
the package references; (B) metamorphic properties of the public API (prompt section 8 list).

Tolerances are declared here for the float64 level of THIS file only.  Seeds are illustrative.
"""
import math
import pathlib
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from mla_model import BindingError, MLAForwardModel, TaskIdentityError  # noqa: E402
from mla_model.errors import ReferenceSemanticsError  # noqa: E402
from mla_model.reference import (blockwise_states, dense_absorbed, dense_expanded, finalize_state, merge_states,  # noqa: E402
                                 online_attention)

SEED = 90210          # illustrative, test-local
ATOL = RTOL = 1e-9    # float64 level of this file


# ------------------------------------------------------------------ part A: own dense implementation ---------------
def own_mla_forward(Cq, Ck, Qr, Kr, Wq, Wk, Wv, scale, mask, delta):
    """Spec 3.1 with explicit loops; mask 'none' or 'causal' with offset.  Returns (O, LSE_natural).  Rows with no
    visible key are not allowed here (caller keeps delta >= 0)."""
    B, Sq, Rq = Cq.shape
    _, Sk, Rk = Ck.shape
    H, _, Dn = Wq.shape
    Dv = Wv.shape[2]
    Dr = 0 if Qr is None else Qr.shape[3]
    O = np.zeros((B, H, Sq, Dv))
    LSE = np.zeros((B, H, Sq))
    for b in range(B):
        for h in range(H):
            Qn = Cq[b] @ Wq[h]             # [Sq, Dn]
            Kn = Ck[b] @ Wk[h]             # [Sk, Dn]
            V = Ck[b] @ Wv[h]              # [Sk, Dv]
            for i in range(Sq):
                scores = []
                for j in range(Sk):
                    if mask == "causal" and j > i + delta:
                        continue
                    s = float(Qn[i] @ Kn[j])
                    if Dr:
                        s += float(Qr[b, h, i] @ Kr[b, j])
                    scores.append((j, scale * s))
                m = max(s for _, s in scores)
                weights = [(j, math.exp(s - m)) for j, s in scores]
                l = sum(w for _, w in weights)
                O[b, h, i] = sum(w * V[j] for j, w in weights) / l
                LSE[b, h, i] = m + math.log(l)
    return O, LSE


def rand_inputs(rng, d):
    Qr = rng.standard_normal((d["B"], d["H"], d["Sq"], d["Dr"])) if d["Dr"] else None
    Kr = rng.standard_normal((d["B"], d["Sk"], d["Dr"])) if d["Dr"] else None
    return dict(Cq=rng.standard_normal((d["B"], d["Sq"], d["Rq"])), Ck=rng.standard_normal((d["B"], d["Sk"], d["Rk"])), Qr=Qr, Kr=Kr,
                Wq=rng.standard_normal((d["H"], d["Rq"], d["Dn"])), Wk=rng.standard_normal((d["H"], d["Rk"], d["Dn"])),
                Wv=rng.standard_normal((d["H"], d["Rk"], d["Dv"])))


def mask_matrix(Sq, Sk, mask, delta):
    return np.array([[mask == "none" or j <= i + delta for j in range(Sk)] for i in range(Sq)], dtype=bool)


def test_references_against_own_loops():
    prng = random.Random(SEED)
    nprng = np.random.default_rng(SEED)
    for _ in range(30):
        d = dict(B=prng.randint(1, 3), H=prng.randint(1, 3), Sq=prng.randint(1, 6), Sk=prng.randint(1, 7), Rq=prng.randint(1, 4), Rk=prng.randint(1, 5),
                 Dn=prng.randint(1, 4), Dr=prng.choice([0, 1, 2]), Dv=prng.randint(1, 4))
        mask = prng.choice(["causal", "none"])
        delta = prng.randint(0, d["Sk"])
        x = rand_inputs(nprng, d)
        scale = 1.0 / math.sqrt(d["Dn"] + d["Dr"])
        O_own, L_own = own_mla_forward(**x, scale=scale, mask=mask, delta=delta)
        M = mask_matrix(d["Sq"], d["Sk"], mask, delta)
        O_e, L_e = dense_expanded(**x, scale=scale, mask=M)
        O_a, L_a = dense_absorbed(**x, scale=scale, mask=M)
        for O, L in ((O_e, L_e), (O_a, L_a)):
            assert np.allclose(O, O_own, atol=ATOL, rtol=RTOL) and np.allclose(L, L_own, atol=ATOL, rtol=RTOL)
        # online / merge for a random (b, h) with independent state pieces
        b, h = prng.randrange(d["B"]), prng.randrange(d["H"])
        Qn = x["Cq"][b] @ x["Wq"][h]; Kn = x["Ck"][b] @ x["Wk"][h]; V = x["Ck"][b] @ x["Wv"][h]
        X = scale * (Qn @ Kn.T + (x["Qr"][b, h] @ x["Kr"][b].T if d["Dr"] else 0))
        for normalized in (False, True):
            for base in ("e", "2"):
                Y, L = online_attention(X, M, V, prng.randint(1, d["Sk"]), normalized=normalized, exp_base=base)
                assert np.allclose(Y, O_own[b, h], atol=ATOL, rtol=RTOL) and np.allclose(L, L_own[b, h], atol=ATOL, rtol=RTOL)
                cuts = sorted({0, d["Sk"], prng.randint(0, d["Sk"]), prng.randint(0, d["Sk"])})
                cuts = [0, 0] + cuts + [d["Sk"]]          # empty partitions included
                states = blockwise_states(X, M, V, cuts, normalized=normalized, exp_base=base)
                prng.shuffle(states)
                acc = states[0]
                for s in states[1:]:
                    acc = merge_states(acc, s)
                Y2, L2 = finalize_state(acc)
                assert np.allclose(Y2, O_own[b, h], atol=ATOL, rtol=RTOL) and np.allclose(L2, L_own[b, h], atol=ATOL, rtol=RTOL)


def test_empty_row_conventions():
    X = np.zeros((3, 3)); U = np.ones((3, 2))
    M = mask_matrix(3, 3, "causal", -2)                 # rows 0, 1 see nothing
    with pytest.raises(ReferenceSemanticsError):
        online_attention(X, M, U, 2, empty_row_policy="error")
    Y, L = online_attention(X, M, U, 2, empty_row_policy="zero_output_neg_inf_lse")
    assert np.all(Y[:2] == 0) and np.all(np.isneginf(L[:2])) and not np.isnan(Y).any() and not np.isnan(L).any()


# ------------------------------------------------------------------ part B: metamorphic via the public API -----------
def meta(B, H, Sq, Sk, Rq, Rk, Dn, Dr, Dv, dtype="bfloat16"):
    return {"q_latent": {"shape": [B, Sq, Rq], "dtype": dtype}, "kv_latent": {"shape": [B, Sk, Rk], "dtype": dtype},
            "q_pe": {"shape": [B, H, Sq, Dr], "dtype": dtype}, "k_pe": {"shape": [B, Sk, Dr], "dtype": dtype},
            "w_q_nope": {"shape": [H, Rq, Dn], "dtype": dtype}, "w_k_nope": {"shape": [H, Rk, Dn], "dtype": dtype}, "w_v": {"shape": [H, Rk, Dv], "dtype": dtype}}


def sem(Sq, Sk, **kw):
    s = {"mask": "causal", "position_offset": 0, "scale_policy": "standard", "outputs": ["O", "LSE"], "lse_convention": "natural_log",
         "projection_scope": "full", "empty_row_policy": "error", "active_lengths": {"Sq": Sq, "Sk": Sk}}
    s.update(kw)
    return s


IMPL = {"name": "base", "b_q": 4, "b_k": 4, "rect_policy": "skip_future", "executed_extent_policy": "logical", "kv_buffers": 2, "heads_per_program": 1,
        "score_alias_exp": True, "exp_alias_p_operand": False, "both_qk_branches_live": False, "scratch_bytes": 0, "projection_in_kv_loop": False,
        "materialize": {"expanded_kv": False, "q_nope": False}, "overlap_model": "full_overlap_max"}
NUM = {"matmul_input_dtype": "bfloat16", "accumulator_dtype": "float32", "score_dtype": "float32", "exp_dtype": "float32", "p_operand_dtype": "bfloat16",
       "state_dtype": "float32", "output_dtype": "bfloat16", "lse_dtype": "float32", "exp_base": "e", "recurrence": "unnormalized",
       "projected_operand_dtype": "bfloat16", "projection_accumulator_dtype": "float32"}
HW = {"name": "synthetic-oracle-device", "scope_default": "per_chip", "provenance": "synthetic fixture",
      "compute": {"mxu": {"peak_flops": {"bfloat16": {"value": 1e14, "unit": "FLOP/s", "scope": "per_chip", "precision": "bfloat16"}}}},
      "paths": {"hbm_to_vmem": {"bandwidth": {"value": 1e12, "unit": "byte/s", "scope": "per_chip"}}}}


def analyze(m, s, impl=IMPL, num=NUM, hw=None, alg="expanded"):
    return MLAForwardModel().bind(m, s, impl, num, hw, algorithm=alg).analyze()


def test_rebinding_and_dimension_changes():
    model = MLAForwardModel()
    a = model.bind(meta(2, 3, 8, 8, 4, 6, 3, 2, 5), sem(8, 8), IMPL, NUM, algorithm="expanded")
    ra = a.analyze()
    rb = model.bind(meta(1, 2, 12, 16, 2, 3, 4, 0, 2), sem(12, 16), {**IMPL, "b_q": 3}, NUM, algorithm="absorbed_two_step").analyze()
    ra2 = a.analyze()
    for k in ("F_total[useful]", "C_valid", "M_in", "n_programs"):
        assert ra.value(k) == ra2.value(k) and ra.value(k) != rb.value(k)
    assert ra.header["bindings"]["D_r"] == 2 and rb.header["bindings"]["D_r"] == 0 and rb.header["bindings"]["b_q"] == 3
    # (2) changing S_q changes C_valid / F_Q / M_O but not bytes[w_k_nope] or F_K
    longer = analyze(meta(2, 3, 12, 8, 4, 6, 3, 2, 5), sem(12, 8))
    assert longer.value("C_valid") > ra.value("C_valid") and longer.value("F_Q[useful]") == ra.value("F_Q[useful]") * 12 // 8
    assert longer.value("M_O") == ra.value("M_O") * 12 // 8
    assert longer.value("F_K[useful]") == ra.value("F_K[useful]") and longer.value("bytes[w_k_nope]") == ra.value("bytes[w_k_nope]")


def test_tile_head_batch_hardware_dtype_properties():
    base = analyze(meta(2, 3, 9, 11, 4, 6, 3, 2, 5), sem(9, 11))
    # (3) tiles change execution counts, not the task
    tiled = analyze(meta(2, 3, 9, 11, 4, 6, 3, 2, 5), sem(9, 11), impl={**IMPL, "b_q": 2, "b_k": 5})
    assert tiled.task_fingerprint == base.task_fingerprint
    for k in ("C_valid", "F_total[useful]", "M_in"):
        assert tiled.value(k) == base.value(k)
    assert tiled.value("n_programs") != base.value("n_programs") or tiled.value("C_pad") != base.value("C_pad")
    # (4) heads and batches
    h2 = analyze(meta(2, 2, 8, 8, 4, 6, 3, 2, 5), sem(8, 8))
    h4 = analyze(meta(2, 4, 8, 8, 4, 6, 3, 2, 5), sem(8, 8))
    b4 = analyze(meta(4, 2, 8, 8, 4, 6, 3, 2, 5), sem(8, 8))
    assert h4.value("F_total[useful]") == 2 * h2.value("F_total[useful]") == b4.value("F_total[useful]")
    assert h4.value("bytes[q_pe]") == 2 * h2.value("bytes[q_pe]") and h4.value("bytes[w_v]") == 2 * h2.value("bytes[w_v]")
    assert h4.value("bytes[kv_latent]") == h2.value("bytes[kv_latent]") and b4.value("bytes[w_v]") == h2.value("bytes[w_v]")
    # (5) hardware changes no work/byte metric; changes bounds only (and the peak is selected by the declared compute dtype)
    hw = analyze(meta(2, 3, 9, 11, 4, 6, 3, 2, 5), sem(9, 11), hw=HW)
    for k in ("F_total[useful]", "F_total[executed_graph]", "C_rect", "B[hbm_to_vmem]", "M_live[vreg:softmax]", "M_in", "W_vec[exp]"):
        assert hw.value(k) == base.value(k)
    assert base.value("T_LB[mxu]") is None and hw.value("T_LB[mxu]") == pytest.approx(hw.value("F_total[executed_graph]") / 1e14)
    hw_f32 = analyze(meta(2, 3, 9, 11, 4, 6, 3, 2, 5), sem(9, 11), num={**NUM, "matmul_input_dtype": "float32", "projected_operand_dtype": "float32"}, hw=HW)
    assert hw_f32.value("T_LB[mxu]") is None                          # no float32 peak in the profile -> unknown, not the bf16 figure
    # (6) storage dtype changes bytes only
    f32 = analyze(meta(2, 3, 9, 11, 4, 6, 3, 2, 5, dtype="float32"), sem(9, 11))
    assert f32.value("bytes[q_latent]") == 2 * base.value("bytes[q_latent]")
    assert f32.header["numerics"] == base.header["numerics"]
    for s_ in ("s_X", "s_A", "s_E", "s_P", "s_state", "s_O"):
        assert f32.header["bindings"][s_] == base.header["bindings"][s_]
    assert f32.value("local[score_X]") == base.value("local[score_X]") and f32.value("F_total[useful]") == base.value("F_total[useful]")


def test_partial_analysis_errors_and_compare():
    # (7) missing tiles / hardware -> partial, F_total still bound
    rep = MLAForwardModel().bind(meta(2, 3, 8, 8, 4, 6, 3, 2, 5), sem(8, 8), None, {"accumulator_dtype": "float32"}, algorithm="expanded").analyze()
    assert rep.value("F_total[useful]") is not None and rep.value("C_valid") is not None
    for k in ("local[score_X]", "M_live[vreg:softmax]", "T_LB[mxu]"):
        assert rep[k].value is None and rep[k].missing_fields
    assert rep.get("C_rect") is None
    # (8) contradictory shapes
    bad = meta(2, 3, 8, 8, 4, 6, 3, 2, 5); bad["w_v"]["shape"] = [3, 7, 5]
    with pytest.raises(BindingError):
        MLAForwardModel().bind(bad, sem(8, 8), IMPL, NUM, algorithm="expanded")
    # (9)/(10) compare
    bound = MLAForwardModel().bind(meta(2, 3, 16, 16, 4, 6, 3, 2, 5), sem(16, 16), IMPL, NUM, algorithm="expanded")
    with pytest.raises(TaskIdentityError):
        bound.compare([{**IMPL, "name": "x", "Sq": 4}])
    with pytest.raises(TaskIdentityError):
        bound.compare([{**IMPL, "name": "y", "score_dtype": "bfloat16"}])
    cmp = bound.compare([{**IMPL, "name": "bq8", "b_q": 8}, {**IMPL, "name": "bk8", "b_k": 8}])
    assert cmp["useful_work_invariant"] and len({r["values"]["F_total[useful]"] for r in cmp["rows"]}) == 1
    for r in cmp["rows"]:
        assert r["legality"]["mathematical"] == "legal"
    # per-candidate legality uses the candidate's own tiles: a rank sub-tile larger than R_q is illegal for that candidate only
    cmp2 = bound.compare([{**IMPL, "name": "bad_rank", "b_rq": 99}])
    assert cmp2["rows"][0]["legality"]["mathematical"] == "legal" and cmp2["rows"][1]["legality"]["mathematical"] == "illegal"
