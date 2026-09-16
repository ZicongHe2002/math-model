"""INDEPENDENT ORACLE: FLOP, byte, local-object and vector-op formulas transcribed from spec sections 5, 6, 7.1
with plain Python integers.  Expected values never come from the package's helpers.  Seed is illustrative.
"""
import pathlib
import random
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from mla_model import MLAForwardModel  # noqa: E402

SEED = 777001  # illustrative, test-local
BYTES = {"float32": 4, "bfloat16": 2, "float16": 2, "float8_e4m3fn": 1, "float64": 8}


def meta(d, dtypes):
    B, H, Sq, Sk, Rq, Rk, Dn, Dr, Dv = (d[k] for k in ("B", "H", "Sq", "Sk", "Rq", "Rk", "Dn", "Dr", "Dv"))
    m = {"q_latent": {"shape": [B, Sq, Rq], "dtype": dtypes["q_latent"]}, "kv_latent": {"shape": [B, Sk, Rk], "dtype": dtypes["kv_latent"]},
         "w_q_nope": {"shape": [H, Rq, Dn], "dtype": dtypes["w_q_nope"]}, "w_k_nope": {"shape": [H, Rk, Dn], "dtype": dtypes["w_k_nope"]},
         "w_v": {"shape": [H, Rk, Dv], "dtype": dtypes["w_v"]}}
    if Dr > 0:
        m["q_pe"] = {"shape": [B, H, Sq, Dr], "dtype": dtypes["q_pe"]}
        m["k_pe"] = {"shape": [B, Sk, Dr], "dtype": dtypes["k_pe"]}
    return m


def random_case(rng):
    d = dict(B=rng.randint(1, 3), H=rng.randint(1, 4), Sq=rng.randint(1, 12), Sk=rng.randint(1, 14), Rq=rng.randint(1, 6), Rk=rng.randint(1, 7),
             Dn=rng.randint(1, 6), Dr=rng.choice([0, 1, 2, 4]), Dv=rng.randint(1, 7))
    roles = ["q_latent", "kv_latent", "q_pe", "k_pe", "w_q_nope", "w_k_nope", "w_v"]
    dtypes = {r: rng.choice(list(BYTES)) for r in roles}
    num = {"matmul_input_dtype": rng.choice(["bfloat16", "float32"]), "accumulator_dtype": "float32", "score_dtype": rng.choice(["float32", "bfloat16"]),
           "exp_dtype": rng.choice(["float32", "bfloat16"]), "p_operand_dtype": rng.choice(["bfloat16", "float16"]), "state_dtype": "float32",
           "output_dtype": rng.choice(["bfloat16", "float32"]), "lse_dtype": "float32", "exp_base": "e", "recurrence": "unnormalized",
           "projected_operand_dtype": rng.choice(["bfloat16", "float32"]), "projection_accumulator_dtype": "float32"}
    delta = rng.randint(-2, d["Sk"])
    strat = {"b_q": rng.randint(1, d["Sq"] + 1), "b_k": rng.randint(1, d["Sk"] + 1), "rect_policy": rng.choice(["full_scan", "skip_future"]),
             "executed_extent_policy": rng.choice(["logical", "padded_to_tile"]), "kv_buffers": rng.choice([1, 2]), "heads_per_program": 1,
             "score_alias_exp": rng.choice([True, False]), "exp_alias_p_operand": False, "both_qk_branches_live": False, "scratch_bytes": 0,
             "projection_in_kv_loop": False, "materialize": {"expanded_kv": False, "q_nope": False, "q_tilde": False, "z": False}, "overlap_model": "full_overlap_max"}
    return d, dtypes, num, delta, strat


# ------------------------------------------------------------ oracles (spec 4.1, 4.2, 5, 6.1, 7.1) ----------------------
def C_valid(d, delta):
    return d["B"] * d["H"] * sum(1 for i in range(d["Sq"]) for j in range(d["Sk"]) if j <= i + delta)


def executed(d, delta, strat):
    """(cells, row_visits, key_visits) summed over all (b,h) under the declared policies.

    key_visits counts executed key columns.  Specification 5.2 requires the loop multiplicity to
    increase the work when a projection runs inside the KV loop, so this — not S_k once — is the
    projected-row count of the executed graph while the expanded cache is not materialized.
    """
    bq, bk = strat["b_q"], strat["b_k"]
    cells = rows = keys = 0
    for q0 in range(0, d["Sq"], bq):
        n = min(bq, d["Sq"] - q0)
        for k0 in range(0, d["Sk"], bk):
            m = min(bk, d["Sk"] - k0)
            future = k0 > q0 + n - 1 + delta
            if strat["rect_policy"] == "skip_future" and future:
                continue
            if strat["executed_extent_policy"] == "padded_to_tile":
                n_e, m_e = bq, bk
            else:
                n_e, m_e = n, m
            cells += n_e * m_e
            rows += n_e
            keys += m_e
    bh = d["B"] * d["H"]
    return bh * cells, bh * rows, bh * keys


def F_E(d, C, NQ, NK, NV):
    return 2 * NQ * d["Rq"] * d["Dn"] + 2 * NK * d["Rk"] * d["Dn"] + 2 * NV * d["Rk"] * d["Dv"] + 2 * C * (d["Dn"] + d["Dr"] + d["Dv"])


def F_A(d, C, NQ):
    return 2 * NQ * d["Rq"] * d["Dn"] + 2 * NQ * d["Rk"] * (d["Dn"] + d["Dv"]) + 2 * C * (2 * d["Rk"] + d["Dr"])


def bind(d, dtypes, num, delta, strat, algorithm, sem_extra=None):
    sem = {"mask": "causal", "position_offset": delta, "scale_policy": "standard", "outputs": ["O", "LSE"], "lse_convention": "natural_log",
           "projection_scope": "full", "empty_row_policy": "zero_output_neg_inf_lse", "active_lengths": {"Sq": d["Sq"], "Sk": d["Sk"]}}
    sem.update(sem_extra or {})
    return MLAForwardModel().bind(meta(d, dtypes), sem, strat, num, algorithm=algorithm).analyze()


def test_flops_and_bytes_random_families():
    rng = random.Random(SEED)
    for _ in range(50):
        d, dtypes, num, delta, strat = random_case(rng)
        C = C_valid(d, delta)
        BH = d["B"] * d["H"]
        NQ, NK = BH * d["Sq"], BH * d["Sk"]
        for alg in ("expanded", "absorbed_two_step"):
            rep = bind(d, dtypes, num, delta, strat, alg)
            expected = F_E(d, C, NQ, NK, NK) if alg == "expanded" else F_A(d, C, NQ)
            assert rep.value("F_total[useful]") == expected
            assert rep.value("F_A_minus_F_E") == F_A(d, C, NQ) - F_E(d, C, NQ, NK, NK)
            closed = "F_E[spec_closed_form]" if alg == "expanded" else "F_A[spec_closed_form]"
            assert rep.value(closed) == expected
            assert rep.value("F_Q[useful]") == 2 * NQ * d["Rq"] * d["Dn"]
            if alg == "expanded":
                assert rep.value("F_K[useful]") == 2 * NK * d["Rk"] * d["Dn"] and rep.value("F_V[useful]") == 2 * NK * d["Rk"] * d["Dv"]
                assert rep.value("F_QK_n[useful]") == 2 * C * d["Dn"] and rep.value("F_QK_r[useful]") == 2 * C * d["Dr"] and rep.value("F_PV[useful]") == 2 * C * d["Dv"]
            else:
                assert rep.value("F_Qtilde[useful]") == 2 * NQ * d["Rk"] * d["Dn"] and rep.value("F_ZWv[useful]") == 2 * NQ * d["Rk"] * d["Dv"]
                assert rep.value("F_QC[useful]") == 2 * C * d["Rk"] and rep.value("F_PC[useful]") == 2 * C * d["Rk"]
            # executed graph: attention over executed cells, plus the projections the graph actually runs
            cells, rows, keys = executed(d, delta, strat)
            att = 2 * cells * (d["Dn"] + d["Dr"] + d["Dv"]) if alg == "expanded" else 2 * cells * (2 * d["Rk"] + d["Dr"])
            if alg == "expanded":
                # the expanded cache is not materialized here, so each KV tile is projected once per
                # query block that scans it: projected rows = executed key columns (spec 5.2)
                proj = 2 * NQ * d["Rq"] * d["Dn"] + 2 * keys * d["Rk"] * (d["Dn"] + d["Dv"])
            else:
                proj = 2 * NQ * d["Rq"] * d["Dn"] + 2 * NQ * d["Rk"] * (d["Dn"] + d["Dv"])
            assert rep.value("F_total[executed_graph]") == att + proj
            if alg == "expanded":
                # materializing the expanded cache projects every key row exactly once instead
                mat = bind(d, dtypes, num, delta, {**strat, "materialize": {"expanded_kv": True, "q_nope": False}}, alg)
                assert mat.value("F_total[executed_graph]") == att + 2 * NQ * d["Rq"] * d["Dn"] + 2 * NK * d["Rk"] * (d["Dn"] + d["Dv"])
            assert rep.value("exp_count_executed") == cells
            # interface bytes: per-tensor dtype
            s = {r: BYTES[t] for r, t in dtypes.items()}
            exp_in = (d["B"] * d["Sq"] * d["Rq"] * s["q_latent"] + d["B"] * d["Sk"] * d["Rk"] * s["kv_latent"] + d["H"] * d["Rq"] * d["Dn"] * s["w_q_nope"]
                      + d["H"] * d["Rk"] * d["Dn"] * s["w_k_nope"] + d["H"] * d["Rk"] * d["Dv"] * s["w_v"])
            if d["Dr"] > 0:
                exp_in += d["B"] * d["H"] * d["Sq"] * d["Dr"] * s["q_pe"] + d["B"] * d["Sk"] * d["Dr"] * s["k_pe"]
            assert rep.value("M_in") == exp_in == rep.value("M_in[metadata_ledger]")
            assert rep.value("bytes[kv_latent]") == d["B"] * d["Sk"] * d["Rk"] * s["kv_latent"]
            assert rep.value("M_O") == BH * d["Sq"] * d["Dv"] * BYTES[num["output_dtype"]]
            assert rep.value("M_LSE") == BH * d["Sq"] * 4
            # local objects (spec 7.1)
            bq, bk = strat["b_q"], strat["b_k"]
            assert rep.value("local[score_X]") == bq * bk * BYTES[num["score_dtype"]]
            assert rep.value("local[exp_E]") == bq * bk * BYTES[num["exp_dtype"]]
            D_A = d["Dv"] if alg == "expanded" else d["Rk"]
            assert rep.value("local[accumulator_A]") == bq * D_A * 4
            assert rep.value("local[row_state_m]") == bq * 4
            if alg == "expanded":
                assert rep.value("local[q_latent_window]") == bq * d["Rq"] * s["q_latent"]
                assert rep.value("local[v_tile]") == bk * d["Dv"] * BYTES[num["projected_operand_dtype"]]
            # vector ops for the unnormalized recurrence (spec 5.5), executed extents
            assert rep.value("W_vec[exp]") == cells + rows                      # E on every cell + alpha per row visit
            assert rep.value("W_vec[cmp]") == (cells - rows) + rows            # rowmax comparisons + old/new max
            assert rep.value("W_vec[div]") == BH * d["Sq"] * D_A               # final normalization once per row
            cast_e = cells if num["exp_dtype"] != num["p_operand_dtype"] else 0
            cast_o = BH * d["Sq"] * d["Dv"] if num["accumulator_dtype"] != num["output_dtype"] else 0
            assert rep.value("W_vec[cast]") == cast_e + cast_o


def test_projection_scope_rows():
    rng = random.Random(SEED + 3)
    for _ in range(15):
        d, dtypes, num, delta, strat = random_case(rng)
        d["Sk"] = max(d["Sk"], 4)
        cached = rng.randint(0, d["Sk"])
        delta = max(delta, cached)     # keep offsets sensible for a cache
        BH = d["B"] * d["H"]
        C = C_valid(d, delta)
        rep = bind(d, dtypes, num, delta, strat, "expanded", {"projection_scope": "new_tokens_only", "cache": {"kind": "expanded_kv", "cached_tokens": cached}})
        new_rows = BH * (d["Sk"] - cached)
        assert rep.value("F_proj[general]") == 2 * BH * d["Sq"] * d["Rq"] * d["Dn"] + 2 * new_rows * d["Rk"] * (d["Dn"] + d["Dv"])
        assert rep.value("F_total[useful]") == F_E(d, C, BH * d["Sq"], new_rows, new_rows)
        assert rep.value("F_E[spec_closed_form]") == F_E(d, C, BH * d["Sq"], BH * d["Sk"], BH * d["Sk"])
        assert rep.value("F_A_minus_F_E") == F_A(d, C, BH * d["Sq"]) - F_E(d, C, BH * d["Sq"], new_rows, new_rows)


def test_materialized_kv_traffic_and_precomputed_variant():
    d = dict(B=2, H=3, Sq=9, Sk=11, Rq=4, Rk=5, Dn=3, Dr=2, Dv=4)
    dtypes = {r: "bfloat16" for r in ("q_latent", "kv_latent", "q_pe", "k_pe", "w_q_nope", "w_k_nope", "w_v")}
    num = {"matmul_input_dtype": "bfloat16", "accumulator_dtype": "float32", "score_dtype": "float32", "exp_dtype": "float32", "p_operand_dtype": "bfloat16",
           "state_dtype": "float32", "output_dtype": "bfloat16", "lse_dtype": "float32", "exp_base": "e", "recurrence": "unnormalized",
           "projected_operand_dtype": "bfloat16", "projection_accumulator_dtype": "float32"}
    strat = {"b_q": 4, "b_k": 3, "rect_policy": "full_scan", "executed_extent_policy": "logical", "kv_buffers": 2, "heads_per_program": 1,
             "score_alias_exp": True, "exp_alias_p_operand": False, "both_qk_branches_live": False, "scratch_bytes": 0, "projection_in_kv_loop": False,
             "materialize": {"expanded_kv": True, "q_nope": False}, "overlap_model": "full_overlap_max"}
    rep = bind(d, dtypes, num, 3, strat, "expanded")
    BH = d["B"] * d["H"]
    per_key = 2 * d["Dn"] + 2 * d["Dv"]                       # K^n and V rows in bf16
    q_blocks = -(-d["Sq"] // strat["b_q"])
    # spec 6.2/6.3: write once, read every key column once per q-block under full scan
    assert rep.value("HBM_mat[K^n,V]") == BH * d["Sk"] * per_key + BH * q_blocks * d["Sk"] * per_key
    # precomputed absorbed: merge work is a separate component, not folded into the useful total
    strat_a = {**strat, "materialize": {"q_tilde": False, "z": False}}
    rep_a = MLAForwardModel().bind(meta(d, dtypes), {"mask": "causal", "position_offset": 3, "scale_policy": "standard", "outputs": ["O"], "projection_scope": "full",
                                                     "empty_row_policy": "error", "active_lengths": {"Sq": 9, "Sk": 11}, "cache": {"merged_weight_reuse": 10}},
                                   strat_a, num, algorithm="absorbed_precomputed").analyze()
    C = C_valid(d, 3)
    NQ = BH * d["Sq"]
    assert rep_a.value("F_total[useful]") == 2 * NQ * d["Rq"] * d["Rk"] + 2 * C * (2 * d["Rk"] + d["Dr"]) + 2 * NQ * d["Rk"] * d["Dv"]
    assert rep_a.value("F_Wmerge_amortized[useful]") == pytest.approx(2 * d["H"] * d["Rq"] * d["Dn"] * d["Rk"] / 10)
