"""API, binding, metamorphic properties, reports and CLI (spec 11, 13.2)."""
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import NUMERICS, ROOT, TEST_SEED, make_metadata, random_dims, random_semantics, random_strategy
from mla_model import BindingError, MLAForwardModel, TaskIdentityError, comparison_to_markdown
from mla_model.symbolic import causal_visible_count
from mla_model.visibility import block_stats

SEM = {"mask": "causal", "position_offset": 0, "scale_policy": "standard", "outputs": ["O", "LSE"], "lse_convention": "natural_log",
       "projection_scope": "full", "empty_row_policy": "error"}
IMPL = {"name": "base", "b_q": 4, "b_k": 4, "rect_policy": "skip_future", "executed_extent_policy": "logical", "kv_buffers": 2,
        "heads_per_program": 1, "score_alias_exp": True, "exp_alias_p_operand": False, "both_qk_branches_live": False,
        "materialize": {"expanded_kv": False, "q_nope": False}, "scratch_bytes": 0, "overlap_model": "full_overlap_max", "projection_in_kv_loop": False}


def independent_useful_flops(d, C, algorithm):
    """Direct transcription of the spec formulas with plain integers (not through the model)."""
    B, H, Sq, Sk, Rq, Rk, Dn, Dr, Dv = (d[k] for k in ("B", "H", "Sq", "Sk", "Rq", "Rk", "Dn", "Dr", "Dv"))
    if algorithm == "expanded":
        return 2 * B * H * (Sq * Rq * Dn + Sk * Rk * (Dn + Dv)) + 2 * C * (Dn + Dr + Dv)
    return 2 * B * H * (Sq * Rq * Dn + Sq * Rk * (Dn + Dv)) + 2 * C * (2 * Rk + Dr)


def independent_visible(d, sem):
    if sem["mask"] == "none":
        return d["B"] * d["H"] * d["Sq"] * d["Sk"]
    return d["B"] * d["H"] * sum(1 for i in range(d["Sq"]) for j in range(d["Sk"]) if j <= i + sem["position_offset"])


def test_describe_is_symbolic(model):
    rep = model.describe(algorithm="expanded")
    assert rep.mode == "symbolic"
    f = rep["F_total[useful]"]
    assert f.value is None and "S_q" in f.missing_fields and "2*" in f.expression
    assert rep["C_valid"].status == "unknown"   # mask undeclared
    assert "adapter_contract" in rep.extras
    rep2 = model.describe(algorithm="absorbed_two_step")
    assert "R_k" in rep2["local[accumulator_A]"].expression and "D_v" not in rep2["local[accumulator_A]"].expression


def test_bound_values_match_independent_formulas(model):
    rng = random.Random(TEST_SEED)
    for _ in range(30):
        d = random_dims(rng)
        sem = random_semantics(rng, d)
        strat = random_strategy(rng, d)
        for alg in ("expanded", "absorbed_two_step"):
            bound = model.bind(make_metadata(**d), sem, strat, NUMERICS, algorithm=alg)
            rep = bound.analyze()
            C = independent_visible(d, sem)
            assert rep.value("C_valid") == C
            assert rep.value("F_total[useful]") == independent_useful_flops(d, C, alg)
            # Executed-graph work is not ordered against rect work in general: in-loop projection re-projects a
            # KV tile for every q-block that scans it (more), while skipped KV tiles are never projected (less).
            # What is invariant is that the attention part follows the executed cells, and that projecting the
            # KV tiles exactly once reproduces the rect accounting when the extents are logical.
            assert rep.value("F_total[executed_graph]") is not None
            once = model.bind(make_metadata(**d), sem, {**strat, "materialize": {"expanded_kv": True, "q_nope": False}},
                              NUMERICS, algorithm=alg).analyze()
            if strat["executed_extent_policy"] == "logical" and alg == "expanded":
                assert once.value("F_total[executed_graph]") == rep.value("F_total[rect]")
            delta = sem.get("position_offset", 0)
            st = block_stats(sem["mask"], d["Sq"], d["Sk"], strat["b_q"], strat["b_k"], delta, strat["rect_policy"])
            assert rep.value("C_rect") == d["B"] * d["H"] * st["rect_cells"]
            # C_pad follows the declared executed-extent policy (spec 4.2): logical extents or padded tiles
            expected_pad = st["padded_cells"] if strat["executed_extent_policy"] == "padded_to_tile" else st["rect_cells"]
            assert rep.value("C_pad") == d["B"] * d["H"] * expected_pad
            # the selected rectangles are fully classified: visible + partial + future + padding = selected
            assert (rep.value("fully_visible_blocks") + rep.value("partial_blocks")
                    + rep.value("future_blocks_selected") + rep.value("padding_blocks_selected")) == rep.value("selected_blocks")
            # interface bytes: per tensor product * 2 bytes (all bf16)
            expected_in = 2 * (d["B"] * d["Sq"] * d["Rq"] + d["B"] * d["Sk"] * d["Rk"] + d["B"] * d["H"] * d["Sq"] * d["Dr"] + d["B"] * d["Sk"] * d["Dr"]
                               + d["H"] * d["Rq"] * d["Dn"] + d["H"] * d["Rk"] * (d["Dn"] + d["Dv"]))
            assert rep.value("M_in") == expected_in == rep.value("M_in[metadata_ledger]")
            assert rep.value("M_O") == 2 * d["B"] * d["H"] * d["Sq"] * d["Dv"]
            assert rep.value("exp_count_executed") == (rep.value("C_pad") if strat["executed_extent_policy"] == "padded_to_tile" else rep.value("C_rect"))


def test_path_difference_matches_two_analyses(model):
    rng = random.Random(TEST_SEED + 5)
    for _ in range(10):
        d = random_dims(rng)
        sem = random_semantics(rng, d)
        e = model.bind(make_metadata(**d), sem, IMPL, NUMERICS, algorithm="expanded").analyze()
        a = model.bind(make_metadata(**d), sem, IMPL, NUMERICS, algorithm="absorbed_two_step").analyze()
        assert e.value("F_A_minus_F_E") == a.value("F_total[useful]") - e.value("F_total[useful]")
        assert "R_k" in a["local[accumulator_A]"].expression and "D_v" in e["local[accumulator_A]"].expression


def test_rebinding_does_not_leak(model):
    a = model.bind(make_metadata(2, 3, 5, 7, 4, 6, 3, 2, 5), SEM, IMPL, NUMERICS, algorithm="expanded")
    ra = a.analyze()
    b = model.bind(make_metadata(1, 1, 9, 9, 2, 2, 2, 0, 2, pe=False), SEM, {"b_q": 3, "b_k": 3, "rect_policy": "full_scan"}, {}, algorithm="expanded")
    rb = b.analyze()
    ra2 = a.analyze()
    assert ra.value("F_total[useful]") == ra2.value("F_total[useful]") != rb.value("F_total[useful]")
    assert rb.header["bindings"]["D_r"] == 0 and ra.header["bindings"]["D_r"] == 2
    assert "s_X" not in rb.header["bindings"] and rb["local[score_X]"].value is None   # numerics undeclared for b
    assert ra.task_fingerprint != rb.task_fingerprint


def test_metamorphic_task_dimension_change(model):
    base = model.bind(make_metadata(2, 3, 8, 8, 4, 6, 3, 2, 5), SEM, IMPL, NUMERICS, algorithm="expanded").analyze()
    longer = model.bind(make_metadata(2, 3, 12, 8, 4, 6, 3, 2, 5), SEM, IMPL, NUMERICS, algorithm="expanded").analyze()
    assert longer.value("C_valid") > base.value("C_valid")
    assert longer.value("F_Q[useful]") == base.value("F_Q[useful]") * 12 // 8
    assert longer.value("F_K[useful]") == base.value("F_K[useful]")                   # KV projection independent of S_q
    assert longer.value("bytes[w_k_nope]") == base.value("bytes[w_k_nope]")           # weights unchanged
    assert longer.value("M_O") == base.value("M_O") * 12 // 8


def test_metamorphic_tile_change_keeps_task(model):
    a = model.bind(make_metadata(2, 3, 9, 11, 4, 6, 3, 2, 5), SEM, IMPL, NUMERICS, algorithm="expanded")
    b = a.with_strategy({**IMPL, "name": "other", "b_q": 2, "b_k": 5})
    ra, rb = a.analyze(), b.analyze()
    assert ra.task_fingerprint == rb.task_fingerprint
    assert ra.value("C_valid") == rb.value("C_valid") and ra.value("F_total[useful]") == rb.value("F_total[useful]")
    assert ra.value("M_in") == rb.value("M_in")
    assert ra.value("n_programs") != rb.value("n_programs")
    assert ra.value("C_rect") != rb.value("C_rect") or ra.value("C_pad") != rb.value("C_pad")


def test_metamorphic_head_scaling(model):
    h2 = model.bind(make_metadata(2, 2, 8, 8, 4, 6, 3, 2, 5), SEM, IMPL, NUMERICS, algorithm="expanded").analyze()
    h4 = model.bind(make_metadata(2, 4, 8, 8, 4, 6, 3, 2, 5), SEM, IMPL, NUMERICS, algorithm="expanded").analyze()
    assert h4.value("F_total[useful]") == 2 * h2.value("F_total[useful]")
    assert h4.value("C_valid") == 2 * h2.value("C_valid")
    assert h4.value("bytes[kv_latent]") == h2.value("bytes[kv_latent]")     # shared latent input storage
    assert h4.value("bytes[q_pe]") == 2 * h2.value("bytes[q_pe]")            # head-dependent input
    assert h4.value("bytes[w_v]") == 2 * h2.value("bytes[w_v]")
    b2 = model.bind(make_metadata(4, 2, 8, 8, 4, 6, 3, 2, 5), SEM, IMPL, NUMERICS, algorithm="expanded").analyze()
    assert b2.value("F_total[useful]") == 2 * h2.value("F_total[useful]") and b2.value("bytes[w_v]") == h2.value("bytes[w_v]")


def test_metamorphic_hardware_does_not_change_math(model):
    hw = json.loads((ROOT / "examples" / "hardware" / "synthetic_calibrated_profile.json").read_text())
    no_hw = model.bind(make_metadata(2, 3, 16, 16, 4, 6, 3, 2, 5), SEM, IMPL, NUMERICS, algorithm="expanded").analyze()
    with_hw = model.bind(make_metadata(2, 3, 16, 16, 4, 6, 3, 2, 5), SEM, IMPL, NUMERICS, hw, algorithm="expanded").analyze()
    for k in ("F_total[useful]", "F_total[executed_graph]", "C_rect", "B[hbm_to_vmem]", "M_live[vreg:softmax]", "M_in"):
        assert no_hw.value(k) == with_hw.value(k)
    assert no_hw.value("T_LB[mxu]") is None and with_hw.value("T_LB[mxu]") is not None
    assert with_hw.mode == "calibrated" and no_hw.mode == "bound"
    assert no_hw["T_pred[node_interval]"].status == "unknown" and "calibration" in no_hw["T_pred[node_interval]"].missing_fields
    hw2 = json.loads(json.dumps(hw))
    hw2["compute"]["mxu"]["peak_flops"]["bfloat16"]["value"] *= 2
    faster = model.bind(make_metadata(2, 3, 16, 16, 4, 6, 3, 2, 5), SEM, IMPL, NUMERICS, hw2, algorithm="expanded").analyze()
    assert faster.value("T_LB[mxu]") == pytest.approx(with_hw.value("T_LB[mxu]") / 2)
    assert faster.value("F_total[useful]") == with_hw.value("F_total[useful]")


def test_metamorphic_storage_dtype_does_not_alter_numerics(model):
    bf = model.bind(make_metadata(2, 3, 8, 8, 4, 6, 3, 2, 5, dtype="bfloat16"), SEM, IMPL, NUMERICS, algorithm="expanded").analyze()
    f32 = model.bind(make_metadata(2, 3, 8, 8, 4, 6, 3, 2, 5, dtype="float32"), SEM, IMPL, NUMERICS, algorithm="expanded").analyze()
    assert f32.value("bytes[q_latent]") == 2 * bf.value("bytes[q_latent]")
    assert f32.header["numerics"] == bf.header["numerics"]
    for s in ("s_X", "s_A", "s_E", "s_P", "s_state", "s_O"):
        assert f32.header["bindings"][s] == bf.header["bindings"][s]
    assert f32.value("local[score_X]") == bf.value("local[score_X]") and f32.value("local[accumulator_A]") == bf.value("local[accumulator_A]")
    assert f32.value("F_total[useful]") == bf.value("F_total[useful]")


def test_missing_tile_and_hardware_gives_partial_analysis(model):
    rep = model.bind(make_metadata(2, 3, 8, 8, 4, 6, 3, 2, 5), SEM, None, {"accumulator_dtype": "float32"}, algorithm="expanded").analyze()
    assert rep.value("F_total[useful]") is not None and rep.value("C_valid") is not None
    m = rep["local[score_X]"]
    assert m.value is None and set(m.missing_fields) == {"b_q", "b_k", "s_X"}
    assert sorted(m.expression.replace(" ", "").split("*")) == ["b_k", "b_q", "s_X"]     # b_q*b_k*s_X up to ordering
    assert rep.get("C_rect") is None and any("scenario" in mm.metric for mm in rep.metrics if mm.metric.startswith("C_rect"))
    # overlap model undeclared -> both combination scenarios are reported, neither bound
    assert rep.get("T_LB[combined]") is None
    scen = [mm for mm in rep.metrics if mm.metric.startswith("T_LB[combined:scenario=")]
    assert len(scen) == 2 and all(mm.value is None and "overlap_model" in mm.missing_fields for mm in scen)
    assert any("rect_policy" in u for u in rep.unknowns)
    # no unknown=0: a zero is either bound / not applicable, or a labeled scenario value carrying its missing fields
    for mm in rep.metrics:
        if mm.value == 0:
            assert mm.status in ("bound", "not_applicable") or (mm.status == "partial" and mm.missing_fields), mm.metric


def test_contradictory_shapes_fail_instead_of_falling_back(model):
    meta = make_metadata(2, 4, 16, 16, 8, 8, 8, 4, 8)
    meta["w_k_nope"]["shape"] = [5, 8, 8]
    with pytest.raises(BindingError):
        model.bind(meta, SEM, IMPL, NUMERICS, algorithm="expanded")


def test_scale_conflict_is_reported_not_overwritten(model):
    sem = {**SEM, "scale_value": 0.5}
    rep = model.bind(make_metadata(1, 2, 8, 8, 8, 8, 8, 4, 8), sem, IMPL, NUMERICS, algorithm="expanded").analyze()
    assert any(c.field == "scale" for c in rep.conflicts)
    # declared value takes precedence (prompt section 3) but the conflict stays visible and both sources are listed
    assert rep["gamma"].status == "conflict" and rep["gamma"].value == 0.5
    conflict = next(c for c in rep.conflicts if c.field == "scale")
    assert conflict.sources["scale_value"] == 0.5 and conflict.sources["standard_value"] == pytest.approx((8 + 4) ** -0.5)
    assert rep.header["bindings"]["gamma"] == 0.5
    declared = model.bind(make_metadata(1, 2, 8, 8, 8, 8, 8, 4, 8), {**SEM, "scale_policy": "declared", "scale_value": 0.5}, IMPL, NUMERICS, algorithm="expanded").analyze()
    assert declared.value("gamma") == 0.5 and not declared.conflicts
    std = model.bind(make_metadata(1, 2, 8, 8, 8, 8, 8, 4, 8), SEM, IMPL, NUMERICS, algorithm="expanded").analyze()
    assert std.value("gamma") == pytest.approx((8 + 4) ** -0.5)


def test_empty_rows_are_flagged(model):
    sem = {**SEM, "position_offset": -3}
    rep = model.bind(make_metadata(1, 2, 8, 8, 4, 4, 4, 2, 4), sem, IMPL, NUMERICS, algorithm="expanded").analyze()
    assert rep.value("rows_without_visible_keys") == 2 * 3
    assert any(c.field == "empty_row_policy" for c in rep.conflicts)
    ok = model.bind(make_metadata(1, 2, 8, 8, 4, 4, 4, 2, 4), {**sem, "empty_row_policy": "zero_output_neg_inf_lse"}, IMPL, NUMERICS, algorithm="expanded").analyze()
    assert not any(c.field == "empty_row_policy" for c in ok.conflicts)


def test_capacity_active_and_decode(model):
    meta = make_metadata(3, 8, 1, 4096, 24, 64, 32, 16, 32)
    sem = {**SEM, "position_offset": 1499, "active_lengths": {"Sk": 1500}}
    rep = model.bind(meta, sem, {**IMPL, "b_q": 1, "b_k": 256}, NUMERICS, algorithm="absorbed_two_step").analyze()
    assert rep.value("C_valid") == 3 * 8 * 1500
    assert rep.value("S_k_capacity") == 4096 and rep.value("S_k_active") == 1500
    assert rep.value("bytes[kv_latent]") == 3 * 4096 * 64 * 2               # allocation extent
    assert rep.value("F_total[useful]") == independent_useful_flops(dict(B=3, H=8, Sq=1, Sk=1500, Rq=24, Rk=64, Dn=32, Dr=16, Dv=32), 3 * 8 * 1500, "absorbed_two_step")


def test_projection_scope_new_tokens_only(model):
    d = dict(B=2, H=3, Sq=4, Sk=20, Rq=4, Rk=6, Dn=3, Dr=2, Dv=5)
    sem = {**SEM, "position_offset": 16, "projection_scope": "new_tokens_only", "cache": {"kind": "expanded_kv", "cached_tokens": 16}}
    rep = model.bind(make_metadata(**d), sem, IMPL, NUMERICS, algorithm="expanded").analyze()
    C_exec = d["B"] * d["H"] * block_stats("causal", d["Sq"], d["Sk"], IMPL["b_q"], IMPL["b_k"], 16, "skip_future")["rect_cells"]
    # K/V projection only for the 4 new tokens; attention over executed (rectangular, logical-extent) cells
    expected = 2 * d["B"] * d["H"] * (d["Sq"] * d["Rq"] * d["Dn"] + 4 * d["Rk"] * (d["Dn"] + d["Dv"])) + 2 * C_exec * (d["Dn"] + d["Dr"] + d["Dv"])
    assert rep.value("F_total[executed_graph]") == expected
    # the useful total honours the declared scope; the spec's boxed closed form (full projection) is kept separately
    C = independent_visible(d, sem)
    assert rep.value("F_total[useful]") == 2 * d["B"] * d["H"] * (d["Sq"] * d["Rq"] * d["Dn"] + 4 * d["Rk"] * (d["Dn"] + d["Dv"])) + 2 * C * (d["Dn"] + d["Dr"] + d["Dv"])
    assert rep.value("F_E[spec_closed_form]") == independent_useful_flops(d, C, "expanded")
    assert rep["F_E[spec_closed_form]"].value > rep.value("F_total[useful]")
    assert rep.value("F_proj[general]") == 2 * d["B"] * d["H"] * (d["Sq"] * d["Rq"] * d["Dn"] + 4 * d["Rk"] * (d["Dn"] + d["Dv"]))
    from mla_model import CapabilityError
    with pytest.raises(CapabilityError):
        model.bind(make_metadata(**d), {**sem, "projection_scope": "none", "cache": None}, IMPL, NUMERICS, algorithm="expanded")
    with pytest.raises(BindingError):   # cached tokens beyond the active keys would give negative projected rows
        model.bind(make_metadata(**d), {**sem, "cache": {"kind": "expanded_kv", "cached_tokens": 25}}, IMPL, NUMERICS, algorithm="expanded")


def test_ragged_batch(model):
    meta = make_metadata(3, 2, 8, 8, 4, 4, 4, 2, 4)
    sem = {**SEM, "position_offset": None, "active_lengths": {"Sq": [8, 3, 1], "Sk": [8, 3, 5]}, "per_batch_offsets": [0, 0, 4]}
    rep = model.bind(meta, sem, IMPL, NUMERICS, algorithm="expanded").analyze()
    expected_C = 2 * (causal_visible_count(8, 8, 0) + causal_visible_count(3, 3, 0) + causal_visible_count(1, 5, 4))
    assert rep.value("C_valid") == expected_C
    rows_q, rows_k = 2 * (8 + 3 + 1), 2 * (8 + 3 + 5)
    assert rep.value("F_total[useful]") == 2 * rows_q * 4 * 4 + 2 * rows_k * 4 * (4 + 4) + 2 * expected_C * (4 + 2 + 4)
    assert rep.value("C_rect") == 2 * sum(block_stats("causal", sq, sk, 4, 4, dl, "skip_future")["rect_cells"] for sq, sk, dl in zip([8, 3, 1], [8, 3, 5], [0, 0, 4]))


def test_compare_same_task_only(model):
    bound = model.bind(make_metadata(2, 3, 16, 16, 4, 6, 3, 2, 5), SEM, IMPL, NUMERICS, algorithm="expanded")
    cmp = bound.compare([{**IMPL, "name": "bq8", "b_q": 8}, {**IMPL, "name": "full", "rect_policy": "full_scan"}])
    assert cmp["useful_work_invariant"] and len(cmp["rows"]) == 3
    assert all(r["values"]["F_total[useful]"] == cmp["rows"][0]["values"]["F_total[useful]"] for r in cmp["rows"])
    assert cmp["rows"][2]["delta_vs_baseline"]["rect_waste"] > 0
    assert cmp["pareto_set"] and cmp["calibrated_ranking"] is None
    assert "not a measured winner" in cmp["verdict"]
    md = comparison_to_markdown(cmp)
    assert "| rect_waste |" in md
    with pytest.raises(TaskIdentityError):
        bound.compare([{"name": "cheat", "b_q": 8, "Sq": 4}])
    with pytest.raises(TaskIdentityError):
        bound.compare([{"name": "cheat2", "b_q": 8, "accumulator_dtype": "bfloat16"}])
    sw = bound.compare([{**IMPL, "name": "absorbed", "algorithm": "absorbed_two_step"}])
    assert sw["rows"][1]["algorithm"] == "absorbed_two_step" and sw["rows"][1]["values"]["F_A_minus_F_E"] is not None
    assert "R_k" not in str(sw["rows"][0]["values"]["local[accumulator_A]"])


def test_report_serialization(model):
    rep = model.bind(make_metadata(2, 3, 8, 8, 4, 6, 3, 2, 5), SEM, IMPL, NUMERICS, algorithm="expanded").analyze()
    data = json.loads(rep.to_json())
    assert data["mode"] == "bound" and any(m["metric"] == "C_valid" for m in data["metrics"])
    for m in data["metrics"]:
        for key in ("metric", "formula_id", "expression", "bindings", "value", "unit", "scope", "status", "assumptions", "evidence", "coverage", "missing_fields"):
            assert key in m
    md = rep.to_markdown()
    assert "## 2. visibility and rectangles" in md and "C_valid" in md and "Unresolved inputs" in md


def test_core_does_not_import_jax():
    code = "import sys, mla_model; from mla_model import MLAForwardModel; MLAForwardModel().describe(); print('jax' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_cli_roundtrip(tmp_path):
    cfg = ROOT / "examples" / "configs" / "01_square_causal_prefill_expanded.json"
    out = subprocess.run([sys.executable, "-m", "mla_model", "analyze", "--config", str(cfg), "--format", "json"], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    data = json.loads(out.stdout)
    assert data["algorithm"] == "expanded"
    out = subprocess.run([sys.executable, "-m", "mla_model", "compare", "--config", str(cfg), "--out", str(tmp_path / "cmp.md")], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0 and (tmp_path / "cmp.md").exists()
    out = subprocess.run([sys.executable, "-m", "mla_model", "describe", "--algorithm", "absorbed_two_step"], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0 and "symbolic" in out.stdout
    bad = ROOT / "examples" / "configs" / "90_contradictory_shapes_expected_error.json"
    out = subprocess.run([sys.executable, "-m", "mla_model", "analyze", "--config", str(bad)], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 2 and "contradictory" in out.stderr


def test_scheduled_extents_separate_from_active_and_capacity(model):
    meta = make_metadata(1, 2, 4, 4096, 4, 8, 4, 2, 4)
    sem = {**SEM, "position_offset": 1496, "active_lengths": {"Sq": 4, "Sk": 1500}}
    impl = {**IMPL, "b_q": 4, "b_k": 256, "executed_extent_policy": "padded_to_tile", "scheduled_lengths": {"Sq": 4, "Sk": 2048}}
    rep = model.bind(meta, sem, impl, NUMERICS, algorithm="expanded").analyze()
    assert rep.value("S_k_active") == 1500 and rep.value("S_k_scheduled") == 2048 and rep.value("S_k_capacity") == 4096
    assert rep.value("C_valid") == 2 * sum(min(1500, i + 1497) for i in range(4))          # visibility on active keys only
    st = block_stats("causal", 4, 2048, 4, 256, 1496, "skip_future", None, 4, 1500)
    assert rep.value("C_rect") == 2 * st["rect_cells"] and rep.value("padding_blocks_skipped") == 2 * st["padding_skipped"] == 4
    assert rep.value("masked_cells") == 2 * st["masked_cells"]
    plain = model.bind(meta, sem, {**impl, "scheduled_lengths": None}, NUMERICS, algorithm="expanded").analyze()
    assert plain.value("C_valid") == rep.value("C_valid") and plain["S_k_scheduled"].status == "partial"
    with pytest.raises(BindingError):
        model.bind(meta, sem, {**impl, "scheduled_lengths": {"Sk": 1000}}, NUMERICS, algorithm="expanded")      # below active
    with pytest.raises(BindingError):
        model.bind(meta, sem, {**impl, "scheduled_lengths": {"Sk": 8192}}, NUMERICS, algorithm="expanded")      # above capacity


def test_evidence_attachment_and_dependency_graph(model):
    bound = model.bind(make_metadata(1, 2, 8, 8, 4, 6, 3, 2, 5), SEM, IMPL, NUMERICS, algorithm="expanded")
    rep = bound.analyze()
    assert rep["F_compiled"].status == "unknown" and rep["M_spill_peak"].value is None
    ev = bound.attach_compile_evidence({"compiled_flops": 123456, "spill_peak_bytes": 4096, "source": "synthetic LLO summary (fixture)"}).analyze()
    assert ev.value("F_compiled") == 123456 and ev["F_compiled"].source_type == "compiled" and ev.value("M_spill_peak") == 4096
    assert ev.value("F_total[useful]") == rep.value("F_total[useful]")                     # evidence never alters the mathematics
    dg = rep.extras["dependency_graph"]
    assert "S_q" in dg["metric_symbols"]["C_valid"] and any(n["id"] == "QK_nope" for n in dg["matmul_nodes"])
    sym = model.describe(algorithm="absorbed_two_step")
    assert "R_k" in sym.extras["dependency_graph"]["metric_symbols"]["local[accumulator_A]"]


SPEC_METRIC_FIELDS = ("metric", "formula_id", "expression", "bindings", "value", "unit", "scope", "status",
                      "assumptions", "evidence", "coverage", "missing_fields", "not_equivalent_to", "source_type")


def _reports_for_invariants(model):
    """One report per declaration regime: fully declared, partially declared, and symbolic."""
    full = model.bind(make_metadata(2, 3, 16, 16, 4, 6, 3, 2, 5), SEM, IMPL, NUMERICS, algorithm="expanded").analyze()
    sparse = model.bind(make_metadata(2, 3, 16, 16, 4, 6, 3, 2, 5), {"mask": "causal", "position_offset": 0},
                        None, {"accumulator_dtype": "float32"}, algorithm="absorbed_two_step").analyze()
    symbolic = model.describe(algorithm="absorbed_precomputed")
    return {"fully declared": full, "partially declared": sparse, "symbolic": symbolic}


def test_report_metric_invariants(model):
    """Spec 2 / 11.4: every metric carries the full field set, and status never lies about completeness."""
    for label, rep in _reports_for_invariants(model).items():
        assert rep.metrics, label
        for m in rep.metrics:
            d = m.to_dict()
            missing_keys = [k for k in SPEC_METRIC_FIELDS if k not in d]
            assert not missing_keys, f"{label}/{m.metric}: metric fields absent: {missing_keys}"
            assert m.coverage, f"{label}/{m.metric}: coverage is empty"
            assert m.source_type in ("input", "derived", "conditional", "compiled", "measured", "definition"), m.metric
            if m.status == "bound":
                assert not m.missing_fields, f"{label}/{m.metric}: status 'bound' with missing fields {m.missing_fields}"
                assert m.value is not None, f"{label}/{m.metric}: status 'bound' with no value"
            if m.status == "partial":
                assert m.missing_fields, f"{label}/{m.metric}: status 'partial' names no missing field"
            if m.status == "unknown":
                assert m.value is None, f"{label}/{m.metric}: status 'unknown' carries a value"
            if m.value is None and m.status not in ("unknown", "not_calibrated", "not_applicable", "conflict"):
                assert m.missing_fields or m.residual, f"{label}/{m.metric}: no value, no missing field, no residual"


def test_no_unknown_becomes_zero(model):
    """Spec 2: missing values are never forced to 0; a zero is a declared absence or a labeled scenario."""
    for label, rep in _reports_for_invariants(model).items():
        for m in rep.metrics:
            if m.value == 0 and not isinstance(m.value, bool):
                assert m.status in ("bound", "not_applicable") or (m.status == "partial" and m.missing_fields), \
                    f"{label}/{m.metric}: zero with status {m.status} and no missing field"


def test_expressions_survive_binding(model):
    """Spec 11.4: a bound metric keeps the expression it was derived from."""
    rep = model.bind(make_metadata(2, 3, 16, 16, 4, 6, 3, 2, 5), SEM, IMPL, NUMERICS, algorithm="expanded").analyze()
    for name in ("C_valid", "F_total[useful]", "M_in", "local[score_X]", "envelope[vreg:softmax]", "B[hbm_to_vmem]"):
        m = rep[name]
        assert m.value is not None and m.expression and m.expression != str(m.value), name
        assert m.bindings, f"{name}: no bindings recorded for a bound metric"


# --------------------------------------------------------------------------------------------------
# Regressions for defects found by the independent verification sweep.
# --------------------------------------------------------------------------------------------------

def test_rope_branch_object_absent_without_a_positional_branch(model):
    """D_r = 0 means there is no second QK partial product to keep live."""
    impl = {**IMPL, "both_qk_branches_live": True}
    with_rope = model.bind(make_metadata(1, 1, 8, 8, 4, 4, 4, 2, 4), SEM, impl, NUMERICS, algorithm="expanded").analyze()
    without = model.bind(make_metadata(1, 1, 8, 8, 4, 4, 4, 0, 4, pe=False), SEM, impl, NUMERICS, algorithm="expanded").analyze()
    assert with_rope.get("local[score_rope_branch]") is not None
    assert without.get("local[score_rope_branch]") is None
    assert without.value("M_live[vreg:score]") < with_rope.value("M_live[vreg:score]")
    assert without.value("F_QK_r[useful]") == 0 and without.value("W_vec[add]") < with_rope.value("W_vec[add]")


def test_in_loop_kv_projection_scales_with_q_blocks(model):
    """Spec 5.2: a KV tile projected inside the loop is re-projected by every q-block that scans it."""
    meta = make_metadata(1, 1, 16, 16, 4, 4, 4, 2, 4)
    sem = {**SEM, "active_lengths": {"Sq": 16, "Sk": 16}}
    in_loop = model.bind(meta, sem, {**IMPL, "materialize": {"expanded_kv": False, "q_nope": False}}, NUMERICS, algorithm="expanded").analyze()
    materialized = model.bind(meta, sem, {**IMPL, "materialize": {"expanded_kv": True, "q_nope": False}}, NUMERICS, algorithm="expanded").analyze()
    assert in_loop.value("F_projection[executed_graph]") > materialized.value("F_projection[executed_graph]")
    # the mathematical once-per-row quantity is unaffected by where the projection runs
    assert in_loop.value("F_total[useful]") == materialized.value("F_total[useful]")
    assert in_loop.task_fingerprint == materialized.task_fingerprint


def test_uniform_per_batch_offsets_are_a_declared_offset(model):
    """One offset repeated for every batch is a uniform offset, not an unknown one."""
    sem_no_offset = {k: v for k, v in SEM.items() if k != "position_offset"}
    rep = model.bind(make_metadata(3, 1, 8, 8, 4, 4, 4, 2, 4), {**sem_no_offset, "per_batch_offsets": [5, 5, 5]}, IMPL, NUMERICS,
                     algorithm="expanded").analyze()
    direct = model.bind(make_metadata(3, 1, 8, 8, 4, 4, 4, 2, 4), {**SEM, "position_offset": 5}, IMPL, NUMERICS,
                        algorithm="expanded").analyze()
    assert rep.value("Delta") == 5 and rep["Delta"].status == "bound"
    assert rep.value("C_valid") == direct.value("C_valid") is not None


def test_cache_new_tokens_is_validated(model):
    meta = make_metadata(1, 2, 4, 20, 4, 6, 3, 2, 5)
    sem = {**SEM, "position_offset": 16, "active_lengths": {"Sq": 4, "Sk": 20}, "projection_scope": "new_tokens_only"}
    for bad in ({"new_tokens": -5}, {"new_tokens": 99}, {"new_tokens": 7, "cached_tokens": 16}):
        with pytest.raises(BindingError):
            model.bind(meta, {**sem, "cache": bad}, IMPL, NUMERICS, algorithm="expanded")
    ok = model.bind(meta, {**sem, "cache": {"new_tokens": 4, "cached_tokens": 16}}, IMPL, NUMERICS, algorithm="expanded").analyze()
    assert ok.value("F_K[useful]") == 2 * 1 * 2 * 4 * 6 * 3     # 2 * (B*H*new) * R_k * D_n


def test_scheduled_extent_is_a_declaration_not_an_assumption(model):
    """Rectangle-grid metrics say when the scheduled extent was assumed; mathematical work does not depend on it."""
    meta = make_metadata(1, 1, 8, 8, 4, 4, 4, 2, 4)
    sem = {**SEM, "active_lengths": {"Sq": 8, "Sk": 8}}
    undeclared = model.bind(meta, sem, IMPL, NUMERICS, algorithm="expanded").analyze()
    declared = model.bind(meta, sem, {**IMPL, "scheduled_lengths": {"Sq": 8, "Sk": 8}}, NUMERICS, algorithm="expanded").analyze()
    assert undeclared["C_rect"].status == "partial" and "scheduled_lengths" in undeclared["C_rect"].missing_fields
    assert declared["C_rect"].status == "bound" and declared.value("C_rect") == undeclared.value("C_rect")
    for k in ("C_valid", "F_total[useful]", "M_in"):
        assert undeclared[k].status == "bound" and "scheduled_lengths" not in undeclared[k].missing_fields


def test_every_status_is_in_the_declared_vocabulary(model):
    """Statuses outside the vocabulary cannot be interpreted by a consumer of the report."""
    import json as _json
    hw = _json.loads((ROOT / "examples" / "hardware" / "synthetic_calibrated_profile.json").read_text())
    allowed = {"symbolic", "partial", "bound", "unknown", "conflict", "not_applicable"}
    for label, rep in (("calibrated", model.bind(make_metadata(2, 4, 64, 64, 8, 16, 16, 8, 16), SEM, IMPL, NUMERICS, hw, algorithm="expanded").analyze()),
                       ("bound", model.bind(make_metadata(2, 4, 64, 64, 8, 16, 16, 8, 16), SEM, IMPL, NUMERICS, algorithm="expanded").analyze()),
                       ("symbolic", model.describe(algorithm="expanded"))):
        bad = [(m.metric, m.status) for m in rep.metrics if m.status not in allowed]
        assert not bad, f"{label}: {bad}"
