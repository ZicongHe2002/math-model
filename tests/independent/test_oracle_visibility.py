"""INDEPENDENT ORACLE: visibility / rectangle counts from brute-force enumeration written from spec section 4.

Expected values come only from the enumeration functions in this file (never from mla_model.visibility
or mla_model.symbolic).  Random configurations use a test-local seed (illustrative, not a default).
"""
import pathlib
import random
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from mla_model import MLAForwardModel  # noqa: E402

SEED = 424242  # illustrative, test-local

NUM = {"matmul_input_dtype": "bfloat16", "accumulator_dtype": "float32", "score_dtype": "float32", "exp_dtype": "float32",
       "p_operand_dtype": "bfloat16", "state_dtype": "float32", "output_dtype": "bfloat16", "lse_dtype": "float32",
       "exp_base": "e", "recurrence": "unnormalized"}


def meta(B, H, Sq, Sk, Rq, Rk, Dn, Dr, Dv):
    return {"q_latent": {"shape": [B, Sq, Rq], "dtype": "bfloat16"}, "kv_latent": {"shape": [B, Sk, Rk], "dtype": "bfloat16"},
            "q_pe": {"shape": [B, H, Sq, Dr], "dtype": "bfloat16"}, "k_pe": {"shape": [B, Sk, Dr], "dtype": "bfloat16"},
            "w_q_nope": {"shape": [H, Rq, Dn], "dtype": "bfloat16"}, "w_k_nope": {"shape": [H, Rk, Dn], "dtype": "bfloat16"},
            "w_v": {"shape": [H, Rk, Dv], "dtype": "bfloat16"}}


# ---------------------------------------------------------------- oracle (spec 4.1 / 4.2) -----------------------------
def visible(i, j, mask, delta):
    return True if mask == "none" else j <= i + delta


def oracle_valid(Sq, Sk, mask, delta):
    return sum(1 for i in range(Sq) for j in range(Sk) if visible(i, j, mask, delta))


def oracle_blocks(Sq, Sk, bq, bk, mask, delta, policy, sub=None):
    """Return list of (q0, n, k0, m, kind) for the executed rectangle set E under the policy."""
    def kind_of(q0, n, k0, m):
        if mask == "none":
            return "visible"
        if k0 > q0 + n - 1 + delta:          # spec 4.2: fully future
            return "future"
        if k0 + m - 1 <= q0 + delta:         # spec 4.2: fully visible
            return "visible"
        return "partial"
    out = []
    for q0 in range(0, Sq, bq):
        n = min(bq, Sq - q0)
        for k0 in range(0, Sk, bk):
            m = min(bk, Sk - k0)
            kind = kind_of(q0, n, k0, m)
            if policy == "full_scan":
                out.append((q0, n, k0, m, kind, False))
            elif policy == "skip_future":
                if kind != "future":
                    out.append((q0, n, k0, m, kind, False))
            elif policy == "skip_future_subdivide":
                if kind == "future":
                    continue
                if kind != "partial":
                    out.append((q0, n, k0, m, kind, False))
                    continue
                pq, pk = sub
                for sq0 in range(q0, q0 + n, pq):
                    sn = min(pq, q0 + n - sq0)
                    for sk0 in range(k0, k0 + m, pk):
                        sm = min(pk, k0 + m - sk0)
                        sk = kind_of(sq0, sn, sk0, sm)
                        if sk != "future":
                            out.append((sq0, sn, sk0, sm, sk, True))
    return out


def oracle_stats(Sq, Sk, bq, bk, mask, delta, policy, sub=None):
    blocks = oracle_blocks(Sq, Sk, bq, bk, mask, delta, policy, sub)
    rect = sum(n * m for _, n, _, m, _, _ in blocks)
    padded = sum((sub[0] * sub[1] if is_sub else bq * bk) for _, n, _, m, _, is_sub in blocks)
    partial = sum(1 for b in blocks if b[4] == "partial")
    full = sum(1 for b in blocks if b[4] == "visible")
    all_blocks = -(-Sq // bq) * -(-Sk // bk)
    future_skipped = 0 if policy == "full_scan" else sum(
        1 for q0 in range(0, Sq, bq) for k0 in range(0, Sk, bk)
        if mask == "causal" and k0 > q0 + min(bq, Sq - q0) - 1 + delta)
    masked = sum(n * m for _, n, _, m, k, _ in blocks if k != "visible")
    return {"rect": rect, "padded": padded, "selected": len(blocks), "partial": partial, "full": full,
            "future_skipped": future_skipped, "masked": masked}


def analyze(B, H, Sq, Sk, Rq, Rk, Dn, Dr, Dv, mask, delta, bq, bk, policy, extent="logical", sub=None, active=None):
    sem = {"mask": mask, "scale_policy": "standard", "outputs": ["O"], "projection_scope": "full",
           "empty_row_policy": "zero_output_neg_inf_lse", "active_lengths": active or {"Sq": Sq, "Sk": Sk}}
    if mask == "causal":
        sem["position_offset"] = delta
    impl = {"b_q": bq, "b_k": bk, "rect_policy": policy, "executed_extent_policy": extent, "kv_buffers": 2, "heads_per_program": 1,
            "score_alias_exp": True, "exp_alias_p_operand": False, "both_qk_branches_live": False, "scratch_bytes": 0,
            "projection_in_kv_loop": False, "materialize": {"expanded_kv": False, "q_nope": False}, "overlap_model": "full_overlap_max"}
    if sub:
        impl["subdivide"] = list(sub)
    return MLAForwardModel().bind(meta(B, H, Sq, Sk, Rq, Rk, Dn, Dr, Dv), sem, impl, NUM, algorithm="expanded").analyze()


def test_visible_cells_random_families():
    rng = random.Random(SEED)
    for _ in range(120):
        B, H = rng.randint(1, 3), rng.randint(1, 4)
        Sq, Sk = rng.randint(1, 15), rng.randint(1, 15)
        mask = rng.choice(["causal", "causal", "none"])
        delta = rng.randint(-Sq - 2, Sk + 2)
        bq, bk = rng.randint(1, Sq + 1), rng.randint(1, Sk + 1)
        policy = rng.choice(["full_scan", "skip_future"])
        rep = analyze(B, H, Sq, Sk, 3, 4, 3, 2, 3, mask, delta, bq, bk, policy)
        assert rep.value("C_valid") == B * H * oracle_valid(Sq, Sk, mask, delta)
        st = oracle_stats(Sq, Sk, bq, bk, mask, delta, policy)
        assert rep.value("C_rect") == B * H * st["rect"]
        # spec 4.2: n_hat, m_hat come from the declared strategy, so a declared 'logical' extent policy
        # makes C_pad the logical area; the padded-tile area is the other declared scenario
        assert rep.value("C_pad") == B * H * st["rect"]
        rep_pad = analyze(B, H, Sq, Sk, 3, 4, 3, 2, 3, mask, delta, bq, bk, policy, extent="padded_to_tile")
        assert rep_pad.value("C_pad") == B * H * st["padded"]
        assert rep.value("rect_waste") == B * H * (st["rect"] - oracle_valid(Sq, Sk, mask, delta))
        assert rep.value("selected_blocks") == B * H * st["selected"]
        assert rep.value("partial_blocks") == B * H * st["partial"]
        assert rep.value("fully_visible_blocks") == B * H * st["full"]
        assert rep.value("future_blocks_skipped") == B * H * st["future_skipped"]
        assert rep.value("masked_cells") == B * H * st["masked"]
        empty = B * H * sum(1 for i in range(Sq) if not any(visible(i, j, mask, delta) for j in range(Sk)))
        assert rep.value("rows_without_visible_keys") == empty


def test_decode_and_special_cases():
    for L in (1, 7, 300):
        rep = analyze(2, 3, 1, L, 4, 4, 4, 2, 4, "causal", L - 1, 1, 16, "skip_future")
        assert rep.value("C_valid") == 2 * 3 * L          # single query sees all L keys, not one
    for S in (8, 16):
        rep = analyze(1, 1, S, S, 4, 4, 4, 2, 4, "causal", 0, 4, 4, "skip_future")
        assert rep.value("C_valid") == S * (S + 1) // 2
        assert rep.value("closed_form_rect_per_bh") == (S * S + S * 4) // 2
        assert rep.value("closed_form_waste_per_bh") == S * 3 // 2
        assert rep.value("C_rect") == rep.value("closed_form_rect_per_bh")


def test_subdivision_policy():
    rng = random.Random(SEED + 1)
    for _ in range(60):
        Sq, Sk = rng.randint(1, 12), rng.randint(1, 12)
        bq, bk = rng.randint(1, 5), rng.randint(1, 5)
        pq, pk = rng.randint(1, bq), rng.randint(1, bk)
        delta = rng.randint(-3, 5)
        rep = analyze(1, 2, Sq, Sk, 3, 3, 3, 1, 3, "causal", delta, bq, bk, "skip_future_subdivide", sub=(pq, pk))
        st = oracle_stats(Sq, Sk, bq, bk, "causal", delta, "skip_future_subdivide", (pq, pk))
        assert rep.value("C_rect") == 2 * st["rect"]
        assert rep.value("C_pad") == 2 * st["rect"]          # declared extent policy is 'logical' here
        rep_pad = analyze(1, 2, Sq, Sk, 3, 3, 3, 1, 3, "causal", delta, bq, bk, "skip_future_subdivide", sub=(pq, pk),
                          extent="padded_to_tile")
        assert rep_pad.value("C_pad") == 2 * st["padded"]
        assert rep.value("selected_blocks") == 2 * st["selected"]
        if (pq, pk) == (1, 1):
            assert rep.value("C_rect") == rep.value("C_valid")


def test_padded_extents_drive_executed_counts():
    rep_l = analyze(1, 1, 10, 13, 3, 3, 3, 1, 3, "causal", 2, 4, 5, "skip_future", extent="logical")
    rep_p = analyze(1, 1, 10, 13, 3, 3, 3, 1, 3, "causal", 2, 4, 5, "skip_future", extent="padded_to_tile")
    st = oracle_stats(10, 13, 4, 5, "causal", 2, "skip_future")
    assert rep_l.value("exp_count_executed") == st["rect"] and rep_p.value("exp_count_executed") == st["padded"]
    assert rep_l.value("C_valid") == rep_p.value("C_valid") == oracle_valid(10, 13, "causal", 2)


def test_ragged_batches():
    rng = random.Random(SEED + 2)
    for _ in range(25):
        B, H = rng.randint(2, 3), rng.randint(1, 3)
        cap_q, cap_k = rng.randint(4, 10), rng.randint(4, 12)
        sq = [rng.randint(1, cap_q) for _ in range(B)]
        sk = [rng.randint(1, cap_k) for _ in range(B)]
        dl = [rng.randint(-2, 6) for _ in range(B)]
        bq, bk = rng.randint(1, 4), rng.randint(1, 4)
        sem = {"mask": "causal", "scale_policy": "standard", "outputs": ["O"], "projection_scope": "full", "empty_row_policy": "zero_output_neg_inf_lse",
               "active_lengths": {"Sq": sq, "Sk": sk}, "per_batch_offsets": dl}
        impl = {"b_q": bq, "b_k": bk, "rect_policy": "skip_future", "executed_extent_policy": "logical", "kv_buffers": 2, "heads_per_program": 1,
                "score_alias_exp": True, "exp_alias_p_operand": False, "both_qk_branches_live": False, "scratch_bytes": 0, "projection_in_kv_loop": False,
                "materialize": {"expanded_kv": False, "q_nope": False}, "overlap_model": "full_overlap_max"}
        rep = MLAForwardModel().bind(meta(B, H, cap_q, cap_k, 3, 4, 3, 2, 3), sem, impl, NUM, algorithm="expanded").analyze()
        assert rep.value("C_valid") == H * sum(oracle_valid(a, b, "causal", d) for a, b, d in zip(sq, sk, dl))
        assert rep.value("C_rect") == H * sum(oracle_stats(a, b, bq, bk, "causal", d, "skip_future")["rect"] for a, b, d in zip(sq, sk, dl))
        assert rep.value("S_k_capacity") == cap_k and rep.value("S_q_active") == sq
