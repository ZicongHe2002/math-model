"""Metadata adapter: extraction, cross-checks, explicit failures (spec 1, 13.2)."""
import numpy as np
import pytest

from conftest import make_metadata
from mla_model import BindingError, CapabilityError, MetadataError, TensorMeta, parse_tensor_metadata
from mla_model.adapters import SevenInputLatentAdapter
from mla_model.semantics import SemanticConfig


def extract(meta, sem=None):
    return SevenInputLatentAdapter().extract(parse_tensor_metadata(meta), SemanticConfig.from_source(sem or {"mask": "causal", "position_offset": 0}))


def test_dims_and_provenance():
    t = extract(make_metadata(2, 3, 5, 7, 4, 6, 3, 2, 5))
    assert t.dims == {"B": 2, "Sq": 5, "Rq": 4, "Sk": 7, "Rk": 6, "H": 3, "Dn": 3, "Dr": 2, "Dv": 5}
    assert t.provenance["B"] == "tensor.q_latent.axis[B]" and t.provenance["Dv"] == "tensor.w_v.axis[Dv]"
    assert any("tensor.w_k_nope.axis[H]" in c for c in t.cross_checks)
    assert t.dtype_bytes["s_Cq"] == 2 and t.symbol_bindings()["S_k"] == 7 and t.symbol_bindings()["Delta"] == 0


def test_contradictory_shapes_list_every_violation():
    meta = make_metadata(2, 4, 16, 16, 8, 8, 8, 4, 8)
    meta["w_k_nope"]["shape"] = [5, 8, 8]      # H mismatch
    meta["k_pe"]["shape"] = [3, 16, 4]         # B mismatch
    meta["w_v"]["shape"] = [4, 9, 8]           # Rk mismatch
    with pytest.raises(BindingError) as ei:
        extract(meta)
    text = str(ei.value)
    assert "H:" in text and "B:" in text and "Rk:" in text
    assert len(ei.value.violations) >= 3


def test_unlabeled_list_and_unknown_roles_are_rejected():
    with pytest.raises(MetadataError):
        parse_tensor_metadata([{"shape": [1, 2, 3], "dtype": "bfloat16"}])
    meta = make_metadata(1, 1, 2, 2, 2, 2, 2, 1, 2)
    meta["mystery"] = {"shape": [3], "dtype": "float32"}
    with pytest.raises(MetadataError):
        extract(meta)
    meta = make_metadata(1, 1, 2, 2, 2, 2, 2, 1, 2)
    del meta["w_v"]
    with pytest.raises(MetadataError):
        extract(meta)


def test_axis_permutation_gives_same_task():
    canon = extract(make_metadata(2, 3, 5, 7, 4, 6, 3, 2, 5))
    meta = make_metadata(2, 3, 5, 7, 4, 6, 3, 2, 5)
    meta["q_latent"] = {"shape": [5, 2, 4], "dtype": "bfloat16", "axes": ["Sq", "B", "Rq"]}
    meta["kv_latent"] = {"shape": [2, 6, 7], "dtype": "bfloat16", "axes": ["B", "Rk", "Sk"]}
    perm = extract(meta)
    assert perm.dims == canon.dims
    # same-extent axes are not silently matched: wrong axis names fail
    meta["kv_latent"]["axes"] = ["B", "Sk", "Rk"]
    with pytest.raises(BindingError):
        extract(meta)


def test_capacity_vs_active_length():
    meta = make_metadata(1, 2, 1, 4096, 8, 16, 8, 4, 8)
    t = extract(meta, {"mask": "causal", "position_offset": 999, "active_lengths": {"Sk": 1000}})
    assert t.capacity["Sk"] == 4096 and t.active["Sk"] == 1000 and t.symbol_bindings()["S_k"] == 1000
    assert t.capacity_bindings()["S_k"] == 4096
    with pytest.raises(BindingError):
        extract(meta, {"mask": "causal", "position_offset": 0, "active_lengths": {"Sk": 5000}})


def test_ragged_lengths():
    meta = make_metadata(3, 2, 8, 8, 4, 4, 4, 2, 4)
    t = extract(meta, {"mask": "causal", "active_lengths": {"Sq": [8, 3, 1], "Sk": [8, 3, 5]}, "per_batch_offsets": [0, 0, 4]})
    assert t.ragged == {"Sq": [8, 3, 1], "Sk": [8, 3, 5], "Delta": [0, 0, 4]}
    with pytest.raises(BindingError):
        extract(meta, {"mask": "causal", "position_offset": 0, "active_lengths": {"Sq": [8, 3]}})


def test_no_positional_branch():
    t = extract(make_metadata(1, 2, 4, 4, 3, 3, 3, 0, 3, pe=False))
    assert t.dims["Dr"] == 0 and t.dtype_bytes["s_Qr"] == 0
    meta = make_metadata(1, 2, 4, 4, 3, 3, 3, 2, 3)
    del meta["k_pe"]
    with pytest.raises(MetadataError):
        extract(meta)


def test_capability_rejections():
    meta = make_metadata(1, 2, 4, 4, 3, 3, 3, 2, 3)
    meta["kv_latent"]["dtype"] = "int4"
    with pytest.raises(CapabilityError):
        extract(meta)
    meta = make_metadata(1, 2, 4, 4, 3, 3, 3, 2, 3)
    meta["kv_latent"]["quantization"] = {"scheme": "int8-per-channel"}
    with pytest.raises(CapabilityError):
        extract(meta)
    meta = make_metadata(1, 2, 4, 4, 3, 3, 3, 2, 3)
    meta["q_latent"]["sharding"] = {"mesh": "2x1"}
    with pytest.raises(CapabilityError):
        extract(meta)
    with pytest.raises(CapabilityError):
        extract(make_metadata(1, 2, 4, 4, 3, 3, 3, 2, 3), {"mask": "causal", "position_offset": 0, "segments": [[0, 2], [2, 4]]})
    with pytest.raises(CapabilityError):
        extract(make_metadata(1, 2, 0, 4, 3, 3, 3, 2, 3))


def test_unknown_dtype_needs_override():
    meta = make_metadata(1, 2, 4, 4, 3, 3, 3, 2, 3)
    meta["w_v"]["dtype"] = "mystery16"
    with pytest.raises(CapabilityError):
        extract(meta)
    t = SevenInputLatentAdapter().extract(parse_tensor_metadata(meta), SemanticConfig(mask="causal", position_offset=0), {"mystery16": 2})
    assert t.dtype_bytes["s_Wv"] == 2


def test_from_array_like_reads_only_metadata():
    class Lazy:
        shape = (2, 5, 4)
        dtype = "bfloat16"

        def __array__(self):  # pragma: no cover - would indicate a value read
            raise AssertionError("values must not be read")

    tm = TensorMeta.from_array_like("q_latent", Lazy())
    assert tm.shape == (2, 5, 4) and tm.dtype == "bfloat16"
    arr = np.zeros((2, 7, 6), dtype=np.float32)
    tm2 = TensorMeta.from_array_like("kv_latent", arr)
    assert tm2.dtype == "float32" and tm2.logical_bytes() == 2 * 7 * 6 * 4


def test_offset_conflict_is_explicit():
    meta = make_metadata(1, 1, 4, 4, 2, 2, 2, 1, 2)
    with pytest.raises(BindingError):
        extract(meta, {"mask": "causal", "position_offset": 3, "query_position_start": 10, "key_position_start": 0})
    t = extract(meta, {"mask": "causal", "query_position_start": 10, "key_position_start": 4})
    assert t.offset == 6
