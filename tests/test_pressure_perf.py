"""Stage envelope, feasible region, lower bounds and calibration objects (spec 7.4, 9)."""
import json

import pytest

from conftest import NUMERICS, ROOT, make_metadata
from mla_model.symbolic import evaluate

SEM = {"mask": "causal", "position_offset": 0, "scale_policy": "standard", "outputs": ["O"], "projection_scope": "full", "empty_row_policy": "error"}
IMPL = {"name": "base", "b_q": 8, "b_k": 16, "rect_policy": "skip_future", "executed_extent_policy": "padded_to_tile", "kv_buffers": 2,
        "heads_per_program": 1, "score_alias_exp": True, "exp_alias_p_operand": False, "vreg_budget_bytes": 4096, "both_qk_branches_live": False,
        "materialize": {"expanded_kv": False, "q_nope": False}, "scratch_bytes": 0, "overlap_model": "full_overlap_max", "projection_in_kv_loop": False}
SEM["active_lengths"] = {"Sq": 32, "Sk": 32}


def test_envelope_matches_live_set_and_coefficients(model):
    bound = model.bind(make_metadata(1, 2, 32, 32, 4, 6, 8, 4, 8), SEM, IMPL, NUMERICS, algorithm="expanded")
    rep = bound.analyze()
    assert rep.value("envelope[vreg:softmax]") == rep.value("M_live[vreg:softmax]")
    coef = rep.extras["envelope_coefficients"]["vreg:softmax"]
    b = rep.header["bindings"]
    # vreg level at the softmax cut: a = Max(s_E, s_X) (aliased score/E); b = D_v*s_A + 3 s_state (row states, alpha, accumulator); c = 0 (V tile is a VMEM window)
    assert coef["a"]["value"] == max(b["s_E"], b["s_X"])
    assert coef["c"]["value"] == 0
    assert coef["b"]["value"] == b["D_v"] * b["s_A"] + 3 * b["s_state"]
    vm = rep.extras["envelope_coefficients"]["vmem:softmax"]
    assert vm["c"]["value"] == b["D_v"] * b["s_V"] and vm["b"]["value"] == b["D_n"] * b["s_Qn"] + b["D_r"] * b["s_Qr"]
    assert coef["remainder"]["value"] == 0
    assert "score_X" in "".join(coef["provenance"]["a"])
    val = coef["a"]["value"] * 8 * 16 + coef["b"]["value"] * 8 + coef["c"]["value"] * 16 + coef["d"]["value"]
    assert val == rep.value("M_live[vreg:softmax]")


def test_feasible_region_under_declared_budget(model):
    bound = model.bind(make_metadata(1, 2, 32, 32, 4, 6, 8, 4, 8), SEM, IMPL, NUMERICS, algorithm="expanded")
    rep = bound.analyze()
    R = IMPL["vreg_budget_bytes"]
    coef = rep.extras["envelope_coefficients"]["vreg:softmax"]
    a, bb, c, d = (coef[k]["value"] for k in "abcd")
    bk_max = rep.value("b_k_max[vreg:softmax]")
    assert bk_max == (R - bb * 8 - d) // (a * 8 + c)
    assert a * 8 * bk_max + bb * 8 + c * bk_max + d <= R < a * 8 * (bk_max + 1) + bb * 8 + c * (bk_max + 1) + d
    # spec 10.2 asks for a surface, so the region is a b_q sweep with the declared point marked
    region = rep.extras["feasible_region_by_stage"]["vreg"]["softmax"]
    declared = [r for r in region if r["declared_b_q"]]
    assert len(declared) == 1 and declared[0]["b_q"] == 8 and declared[0]["b_k_max"] == bk_max
    assert len(region) > 1 and [r["b_q"] for r in region] == sorted(r["b_q"] for r in region)
    # the bound falls as b_q grows: the surface, not one best tile
    evaluated = [r for r in region if r["b_k_max"] is not None]
    assert all(x["b_k_max"] >= y["b_k_max"] for x, y in zip(evaluated, evaluated[1:]))
    excess = rep.value("excess_over_budget[vreg:softmax]")
    assert excess == max(0, rep.value("M_live[vreg:softmax]") - R)
    # spec 7.4 is stated per stage: every stage has its own bound and the binding one is the smallest
    per_stage = {st: rep.value(f"b_k_max[vreg:{st}]") for st in rep.extras["dependency_graph"]["stages"]}
    evaluated = {st: v for st, v in per_stage.items() if v is not None}
    binding = rep.extras["feasible_region_binding"]["vreg"]
    assert binding["b_k_max"] == min(evaluated.values())
    assert binding["binding_stage"] in evaluated and evaluated[binding["binding_stage"]] == binding["b_k_max"]
    assert rep.value("b_k_max[vreg:binding_stage]") == binding["b_k_max"]
    # without a budget the bound stays symbolic with R_budget missing, at every stage and for the minimum
    rep2 = model.bind(make_metadata(1, 2, 32, 32, 4, 6, 8, 4, 8), SEM, {**IMPL, "vreg_budget_bytes": None}, NUMERICS, algorithm="expanded").analyze()
    for name in ("b_k_max[vreg:softmax]", "b_k_max[vreg:binding_stage]"):
        assert rep2[name].value is None and "vreg_budget_bytes" in rep2[name].missing_fields


def test_alias_and_buffer_effects(model):
    base = model.bind(make_metadata(1, 2, 32, 32, 4, 6, 8, 4, 8), SEM, IMPL, NUMERICS, algorithm="expanded").analyze()
    no_alias = model.bind(make_metadata(1, 2, 32, 32, 4, 6, 8, 4, 8), SEM, {**IMPL, "score_alias_exp": False}, NUMERICS, algorithm="expanded").analyze()
    b = base.header["bindings"]
    assert no_alias.value("M_live[vreg:softmax]") - base.value("M_live[vreg:softmax]") == 8 * 16 * min(b["s_X"], b["s_E"])
    single = model.bind(make_metadata(1, 2, 32, 32, 4, 6, 8, 4, 8), SEM, {**IMPL, "kv_buffers": 1}, NUMERICS, algorithm="expanded").analyze()
    assert single.value("M_live[vmem:score]") < base.value("M_live[vmem:score]")     # K^r tile double-buffered in base
    assert single.value("F_total[useful]") == base.value("F_total[useful]")


def test_sensitivity_and_spill_definitions(model):
    rep = model.bind(make_metadata(1, 2, 32, 32, 4, 6, 8, 4, 8), SEM, IMPL, NUMERICS, algorithm="expanded").analyze()
    b = rep.header["bindings"]
    assert rep.value("d(score_bytes)/d(b_q)") == b["b_k"] * b["s_X"]
    assert rep.value("d(score_bytes)/d(b_k)") == b["b_q"] * b["s_X"]
    assert rep.value("d(accumulator_bytes)/d(b_k)") == 0
    for name in ("M_spill_peak", "B_spill_fill", "dT_spill", "B_extra_opt"):
        assert rep[name].status == "unknown" and rep[name].value is None


def test_lower_bounds_and_calibration_objects(model):
    hw = json.loads((ROOT / "examples" / "hardware" / "synthetic_calibrated_profile.json").read_text())
    rep = model.bind(make_metadata(2, 4, 64, 64, 8, 16, 16, 8, 16), SEM, IMPL, NUMERICS, hw, algorithm="expanded").analyze()
    lb = rep.extras["resource_lower_bounds"]
    by = {r["resource"]: r for r in lb["bounds"]}
    P = hw["compute"]["mxu"]["peak_flops"]["bfloat16"]["value"]
    assert by["mxu"]["value"] == pytest.approx(rep.value("F_total[executed_graph]") / P)
    assert by["exp"]["value"] == pytest.approx(rep.value("W_resource[exp]") / hw["compute"]["vpu"]["exp_throughput"]["value"])
    assert by["path:hbm_to_vmem"]["value"] == pytest.approx(rep.value("B[hbm_to_vmem]") / hw["paths"]["hbm_to_vmem"]["bandwidth"]["value"])
    assert by["critical_path"]["value"] is None
    known = [r["value"] for r in lb["bounds"] if r["value"] is not None]
    assert rep.value("T_LB[combined]") == pytest.approx(max(known))
    pred = rep.extras["calibrated_prediction"]
    assert pred["status"].startswith("calibrated") and pred["node_interval_s"][0] < pred["node_interval_s"][1]
    assert rep["T_pred[node_interval]"].value is not None and "T_LB" in rep["T_pred[node_interval]"].not_equivalent_to
    # each path has its own bandwidth symbol; the write direction never borrows the read figure
    assert by["path:vmem_to_hbm"]["denominator"] == "W_hbm_w" and by["path:vmem_to_vreg"]["denominator"] == "W_vmem"
    # missing VPU throughputs -> partial bound, no zero fallback
    hw2 = json.loads(json.dumps(hw))
    hw2["compute"]["vpu"]["exp_throughput"]["value"] = None
    rep2 = model.bind(make_metadata(2, 4, 64, 64, 8, 16, 16, 8, 16), SEM, IMPL, NUMERICS, hw2, algorithm="expanded").analyze()
    assert rep2["T_LB[exp]"].value is None and rep2["T_LB[combined]"].status == "partial"


def test_layout_coverage_uses_profile_tiles(model):
    hw = json.loads((ROOT / "examples" / "hardware" / "synthetic_calibrated_profile.json").read_text())
    rep = model.bind(make_metadata(1, 1, 32, 32, 4, 6, 8, 4, 8), SEM, IMPL, NUMERICS, hw, algorithm="expanded").analyze()
    m = rep["layout[row_state_m]"]           # [b_q, 1] float32 -> 8x128 tile: 4*8*128*ceil(8/8)*1
    assert m.value == 4 * 8 * 128 * 1 and m.value > rep.value("local[row_state_m]")
    nohw = model.bind(make_metadata(1, 1, 32, 32, 4, 6, 8, 4, 8), SEM, IMPL, NUMERICS, algorithm="expanded").analyze()
    assert nohw["layout[row_state_m]"].value is None and {"L_l", "L_s"} <= set(nohw["layout[row_state_m]"].missing_fields)


def test_hardware_scope_mismatch_is_warned(model):
    hw = json.loads((ROOT / "examples" / "hardware" / "synthetic_calibrated_profile.json").read_text())
    hw["paths"]["hbm_to_vmem"]["bandwidth"]["scope"] = "per_tensorcore"
    rep = model.bind(make_metadata(1, 1, 32, 32, 4, 6, 8, 4, 8), SEM, IMPL, NUMERICS, hw, algorithm="expanded").analyze()
    assert any("mixed scopes" in w for w in rep.warnings)


def test_analysis_is_memoized_and_reproducible(model):
    """The envelope extraction is a pure function of the graph, so a second analysis reuses it exactly."""
    import json as _json
    from mla_model.symbolic import poly_coefficients, clear_expression_caches
    clear_expression_caches()
    bound = model.bind(make_metadata(1, 2, 32, 32, 4, 6, 8, 4, 8), SEM, IMPL, NUMERICS, algorithm="expanded")
    first = _json.dumps(bound.analyze().to_dict(), sort_keys=True, default=str)
    misses_after_first = poly_coefficients.cache_info().misses
    second = _json.dumps(bound.analyze().to_dict(), sort_keys=True, default=str)
    assert first == second, "a second analysis of the same binding produced a different report"
    info = poly_coefficients.cache_info()
    assert info.hits > 0 and info.misses == misses_after_first, "the second analysis re-derived coefficients"


def test_envelope_coefficients_sum_to_the_live_set(model):
    """Provenance and totals come from one extraction, so they must still agree with the live set."""
    rep = model.bind(make_metadata(1, 2, 32, 32, 4, 6, 8, 4, 8), SEM, IMPL, NUMERICS, algorithm="expanded").analyze()
    b = rep.header["bindings"]
    for level in ("vreg", "vmem", "all"):
        coef = rep.extras["envelope_coefficients"][f"{level}:softmax"]
        a, bb, c, d = (coef[k]["value"] for k in "abcd")
        assert coef["remainder"]["value"] == 0
        assert a * b["b_q"] * b["b_k"] + bb * b["b_q"] + c * b["b_k"] + d == rep.value(f"M_live[{level}:softmax]")
        contributors = {o for key in "abcd" for o in coef["provenance"][key]}
        assert contributors, f"{level}: no object attributed to any coefficient"
