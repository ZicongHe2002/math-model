"""Shared fixtures.  All numeric fixtures here are ILLUSTRATIVE test inputs, not production defaults.

The RNG seed below is a test-local reproducibility choice for *this* test module only.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TEST_SEED = 20260908  # illustrative, test-local


def make_metadata(B, H, Sq, Sk, Rq, Rk, Dn, Dr, Dv, dtype="bfloat16", pe=True, dtypes=None):
    dtypes = dtypes or {}
    d = lambda role: dtypes.get(role, dtype)  # noqa: E731
    m = {
        "q_latent": {"shape": [B, Sq, Rq], "dtype": d("q_latent")},
        "kv_latent": {"shape": [B, Sk, Rk], "dtype": d("kv_latent")},
        "w_q_nope": {"shape": [H, Rq, Dn], "dtype": d("w_q_nope")},
        "w_k_nope": {"shape": [H, Rk, Dn], "dtype": d("w_k_nope")},
        "w_v": {"shape": [H, Rk, Dv], "dtype": d("w_v")},
    }
    if pe:
        m["q_pe"] = {"shape": [B, H, Sq, Dr], "dtype": d("q_pe")}
        m["k_pe"] = {"shape": [B, Sk, Dr], "dtype": d("k_pe")}
    return m


def random_dims(rng: random.Random, allow_dr0=True):
    """Independently generated valid small configuration (spec 13.1 coverage)."""
    return dict(
        B=rng.randint(1, 3), H=rng.randint(1, 4), Sq=rng.randint(1, 13), Sk=rng.randint(1, 17),
        Rq=rng.randint(1, 6), Rk=rng.randint(1, 7), Dn=rng.randint(1, 6),
        Dr=(rng.choice([0, 1, 2, 3, 5]) if allow_dr0 else rng.randint(1, 5)), Dv=rng.randint(1, 7),
    )


def random_semantics(rng: random.Random, dims):
    causal = rng.random() < 0.8
    if causal:
        offset = rng.randint(-dims["Sq"], dims["Sk"] + 1)
        sem = {"mask": "causal", "position_offset": offset}
    else:
        sem = {"mask": "none"}
    sem.update({"scale_policy": "standard", "outputs": ["O", "LSE"], "lse_convention": "natural_log",
                "projection_scope": "full", "empty_row_policy": "zero_output_neg_inf_lse"})
    return sem


def random_strategy(rng: random.Random, dims, name="cand"):
    return {"name": name, "b_q": rng.randint(1, dims["Sq"] + 2), "b_k": rng.randint(1, dims["Sk"] + 2),
            "rect_policy": rng.choice(["full_scan", "skip_future"]), "executed_extent_policy": rng.choice(["logical", "padded_to_tile"]),
            "kv_buffers": rng.choice([1, 2]), "heads_per_program": 1, "score_alias_exp": rng.choice([True, False]),
            "exp_alias_p_operand": rng.choice([True, False])}


NUMERICS = {"matmul_input_dtype": "bfloat16", "accumulator_dtype": "float32", "score_dtype": "float32", "exp_dtype": "float32",
            "p_operand_dtype": "bfloat16", "state_dtype": "float32", "output_dtype": "bfloat16", "lse_dtype": "float32",
            "exp_base": "e", "recurrence": "unnormalized"}


@pytest.fixture
def rng():
    return random.Random(TEST_SEED)


@pytest.fixture
def model():
    from mla_model import MLAForwardModel
    return MLAForwardModel()
