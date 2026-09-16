"""Small-shape numerical references (numpy float64 / optional mpmath).

These are *independent* implementations of the mathematics in spec 3 for testing
the algebra (expanded vs absorbed, dense vs online vs partition-merge).  They are
not kernels and make no claim about BF16 device behaviour.
"""
from .dense import dense_expanded, dense_absorbed, dense_attention_from_scores, softmax_rows
from .online import (
    OnlineState, empty_state, online_attention, merge_states, finalize_state, blockwise_states, convert_lse,
)

__all__ = [
    "dense_expanded", "dense_absorbed", "dense_attention_from_scores", "softmax_rows",
    "OnlineState", "empty_state", "online_attention", "merge_states", "finalize_state", "blockwise_states", "convert_lse",
]
