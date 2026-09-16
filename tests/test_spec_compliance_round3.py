"""Regressions for the third spec-compliance sweep.

Each test names the specification clause it protects.  Most expected values are written out here from the
specification text rather than read back from the package, so a change of behaviour has to be argued for.

Some tests are deliberately *differential*: they assert that two surfaces of the same report agree (the
per-stage bounds against the binding minimum, the materialization ledger against the transfer paths, a
comparison row against the report it came from).  Those catch a surface drifting away from the model, not a
misreading of the specification, and they are marked in place.
"""
import pytest

from conftest import NUMERICS, make_metadata
from mla_model import BindingError, MLAForwardModel
from mla_model.errors import MetadataError
from mla_model.visibility import block_stats_arithmetic, block_stats_enumerate

SEM = {"mask": "causal", "position_offset": 0, "scale_policy": "standard", "outputs": ["O"],
       "projection_scope": "full", "empty_row_policy": "zero_output_neg_inf_lse",
       "active_lengths": {"Sq": 32, "Sk": 32}}
IMPL = {"name": "base", "b_q": 8, "b_k": 8, "rect_policy": "skip_future", "executed_extent_policy": "logical",
        "kv_buffers": 2, "heads_per_program": 1, "score_alias_exp": True, "exp_alias_p_operand": False,
        "both_qk_branches_live": False, "materialize": {"expanded_kv": True, "q_nope": False}, "scratch_bytes": 0,
        "overlap_model": "full_overlap_max", "projection_in_kv_loop": False, "vreg_budget_bytes": 65536,
        "vmem_budget_bytes": 1 << 22, "b_rq": 4, "b_rk": 4, "q_resident": True, "v_load": "early"}


@pytest.fixture
def model():
    return MLAForwardModel()


def md():
    return make_metadata(1, 2, 32, 32, 4, 6, 8, 4, 8)


def stages_of(rep, obj_id):
    for o in rep.extras["dependency_graph"]["local_objects"]:
        if o["id"] == obj_id:
            return list(o["live_stages"])
    return None


# --- spec 7.3: the live set is per object; Q residency cannot retire the loop-carried state -----------
@pytest.mark.parametrize("alg,q_operand", [("expanded", "q_nope_operand"), ("absorbed_two_step", "q_tilde")])
def test_q_residency_shortens_only_the_q_operands(model, alg, q_operand):
    loop = ["score", "softmax", "pv", "finalize"]
    res = model.bind(md(), SEM, {**IMPL, "q_resident": True}, NUMERICS, algorithm=alg).analyze()
    non = model.bind(md(), SEM, {**IMPL, "q_resident": False}, NUMERICS, algorithm=alg).analyze()
    for state in ("row_state_m", "row_state_l", "accumulator_A"):
        assert stages_of(res, state) == loop, state
        assert stages_of(non, state) == loop, f"{state} is loop-carried whatever Q residency says"
    assert stages_of(res, q_operand) == loop
    assert stages_of(non, q_operand) == ["score"]
    assert stages_of(non, "q_pe_tile") == ["score"]
    # dropping Q from the loop cannot lower the softmax live set below the state it must still hold
    assert non.value("M_live[vreg:softmax]") == res.value("M_live[vreg:softmax]")


def test_undeclared_q_residency_is_named_on_the_metrics_it_shapes(model):
    rep = model.bind(md(), SEM, {k: v for k, v in IMPL.items() if k != "q_resident"}, NUMERICS, algorithm="expanded").analyze()
    assert "q_resident" in rep["M_live[vmem:softmax]"].missing_fields
    assert "q_resident" in rep["envelope[vmem:softmax]"].missing_fields
    assert rep["M_live[vmem:softmax]"].status == "partial"


# --- spec 7.4: the envelope and the b_k bound are stated per stage ------------------------------------
def test_feasible_bound_is_per_stage_and_the_binding_stage_is_named(model):
    rep = model.bind(md(), SEM, IMPL, NUMERICS, algorithm="expanded").analyze()
    # a hand-derived value for one stage: spec 7.4's floor((R - b*b_q - d)/(a*b_q + c)) from that stage's
    # own envelope coefficients, so this is not only a minimum-equals-a-minimum check
    coef = rep.extras["envelope_coefficients"]["vreg:pv"]
    a, bb, c, d = (coef[k]["value"] for k in "abcd")
    b_q = IMPL["b_q"]
    assert rep.value("b_k_max[vreg:pv]") == (IMPL["vreg_budget_bytes"] - bb * b_q - d) // (a * b_q + c)
    binding = rep.extras["feasible_region_binding"]["vreg"]
    per_stage = binding["per_stage"]
    assert len(per_stage) > 1
    assert binding["b_k_max"] == min(per_stage.values())
    assert per_stage[binding["binding_stage"]] == binding["b_k_max"]
    for st, v in per_stage.items():
        assert rep.value(f"b_k_max[vreg:{st}]") == v
    assert rep.value("b_k_max[vreg:binding_stage]") == binding["b_k_max"]


def test_stage_without_a_bk_term_is_not_applicable_not_infinite(model):
    rep = model.bind(md(), SEM, IMPL, NUMERICS, algorithm="expanded").analyze()
    m = rep["b_k_max[vreg:q_proj]"]
    assert m.status == "not_applicable" and m.value is None
    assert not any("zoo" in str(x.value) for x in rep.metrics)


def test_a_strategy_budget_contradicting_the_profile_is_a_conflict(model):
    hw = {"name": "p", "scope_default": "per_chip",
          "memory": {"vreg": {"usable_capacity_bytes": {"value": 131072, "unit": "byte", "scope": "per_chip"}}}}
    rep = model.bind(md(), SEM, IMPL, NUMERICS, hw, algorithm="expanded").analyze()
    row = next(c for c in rep.conflicts if c.field == "R_vreg")
    assert row.sources == {"strategy": 65536, "hardware_profile": 131072.0, "used": 65536}


# --- spec 7.1 / 7.2 -----------------------------------------------------------------------------------
def test_multibuffered_object_says_so_and_scratch_has_no_element_layout(model):
    rep = model.bind(md(), SEM, {**IMPL, "scratch_bytes": 4096}, NUMERICS, algorithm="expanded").analyze()
    kpe = rep["local[k_pe_tile]"]
    assert "buffer" in kpe.scope and "n_buf" in kpe.scope
    assert any("logical size of one buffer" in n for n in kpe.notes)
    lay = rep["layout[fixed_scratch]"]
    assert lay.value == 4096 and "raw_bytes" in lay.scope


# --- spec 4.2: n_hat, m_hat come from the declared strategy; the selected set is fully classified ------
@pytest.mark.parametrize("policy", ["full_scan", "skip_future"])
def test_padded_area_follows_the_declared_extent_policy(model, policy):
    impl = {**IMPL, "rect_policy": policy, "b_q": 5, "b_k": 5}
    logical = model.bind(md(), SEM, {**impl, "executed_extent_policy": "logical"}, NUMERICS, algorithm="expanded").analyze()
    padded = model.bind(md(), SEM, {**impl, "executed_extent_policy": "padded_to_tile"}, NUMERICS, algorithm="expanded").analyze()
    assert logical.value("C_pad") == logical.value("C_rect")
    assert padded.value("C_pad") > padded.value("C_rect")     # 32 is not divisible by 5
    undeclared = model.bind(md(), SEM, {k: v for k, v in impl.items() if k != "executed_extent_policy"},
                            NUMERICS, algorithm="expanded").analyze()
    assert "executed_extent_policy" in undeclared["C_pad"].missing_fields
    assert undeclared.value("C_pad[scenario=logical]") == logical.value("C_pad")
    assert undeclared.value("C_pad[scenario=padded_to_tile]") == padded.value("C_pad")


def test_full_scan_names_the_future_rectangles_it_executes(model):
    rep = model.bind(md(), SEM, {**IMPL, "rect_policy": "full_scan"}, NUMERICS, algorithm="expanded").analyze()
    assert rep.value("future_blocks_selected") > 0            # full_scan runs them; they hold no valid cell
    assert rep.value("future_blocks_skipped") == 0
    total = (rep.value("fully_visible_blocks") + rep.value("partial_blocks")
             + rep.value("future_blocks_selected") + rep.value("padding_blocks_selected"))
    assert total == rep.value("selected_blocks")


def test_classification_matches_the_enumeration_oracle():
    for mask in ("causal", "none"):
        for policy in ("full_scan", "skip_future"):
            for Sq, Sk, bq, bk, delta, sqa, ska in [(9, 7, 4, 3, 0, 5, 4), (16, 16, 5, 5, 2, 16, 16),
                                                    (13, 11, 3, 4, -2, 8, 6), (8, 8, 8, 8, 0, 3, 3)]:
                a = block_stats_arithmetic(mask, Sq, Sk, bq, bk, delta, policy, None, None, None, sqa, ska)
                e = block_stats_enumerate(mask, Sq, Sk, bq, bk, delta, policy, None, sqa, ska)
                assert a == e
                assert a["selected"] == a["partial"] + a["full"] + a["future_selected"] + a["padding_selected"]


# --- spec 1.3: shape and effective length are per axis -------------------------------------------------
def test_one_declared_axis_leaves_the_other_a_scenario(model):
    rep = model.bind(md(), {**SEM, "active_lengths": {"Sk": 20}}, IMPL, NUMERICS, algorithm="expanded").analyze()
    m = rep["C_valid"]
    assert m.status == "partial" and m.missing_fields == ["active_lengths.Sq"]
    both = model.bind(md(), {**SEM, "active_lengths": {"Sq": 32, "Sk": 20}}, IMPL, NUMERICS, algorithm="expanded").analyze()
    assert both["C_valid"].status == "bound" and both.value("C_valid") == m.value


# --- spec 3.2 / 3.3 ------------------------------------------------------------------------------------
def test_declared_cast_locations_change_the_vector_graph(model):
    counts = {}
    for sites in ([], ["exp_to_p"], ["score_to_exp", "exp_to_p"], ["score_to_exp", "exp_to_p", "state_update"]):
        rep = model.bind(md(), SEM, IMPL, {**NUMERICS, "cast_points": sites}, algorithm="expanded").analyze()
        counts[tuple(sites)] = rep.value("W_vec[cast]")
    assert len(set(counts.values())) == len(counts)          # different locations, different graphs
    assert counts[()] == 0
    undeclared = model.bind(md(), SEM, IMPL, NUMERICS, algorithm="expanded").analyze()
    assert "cast_points" in undeclared["W_vec[cast]"].missing_fields
    with pytest.raises(MetadataError):
        model.bind(md(), SEM, IMPL, {**NUMERICS, "cast_points": ["nowhere"]}, algorithm="expanded")


def test_merged_weight_storage_and_reuse_are_listed_separately(model):
    sem = {**SEM, "cache": {"merged_weight_reuse": 8}}
    rep = model.bind(md(), sem, IMPL, NUMERICS, algorithm="absorbed_precomputed").analyze()
    b = rep.header["bindings"]
    assert rep.value("M_Wmerged[stored]") == b["H"] * b["R_q"] * b["R_k"] * b["s_Wq"]
    assert rep.value("n_reuse[effective]") == 8
    no_reuse = model.bind(md(), SEM, IMPL, NUMERICS, algorithm="absorbed_precomputed").analyze()
    assert no_reuse["n_reuse[effective]"].status == "unknown"
    assert rep.get("M_Wmerged[stored]") is not None and model.bind(
        md(), SEM, IMPL, NUMERICS, algorithm="expanded").analyze().get("M_Wmerged[stored]") is None


def test_equivalence_and_acceptance_are_reported(model):
    rep = model.bind(md(), SEM, IMPL, {**NUMERICS, "acceptance": {"rel_tol": 0.01}}, algorithm="absorbed_two_step").analyze()
    eq = rep["equivalence[expanded vs absorbed]"]
    assert "real" in str(eq.value) and "bitwise equality" in eq.not_equivalent_to
    assert rep.value("acceptance[declared]") == {"rel_tol": 0.01}
    assert model.bind(md(), SEM, IMPL, NUMERICS, algorithm="expanded").analyze()["acceptance[declared]"].status == "unknown"


# --- spec 5.3: N^proj counts rows actually projected in this invocation --------------------------------
def test_absorbed_projects_no_kv_rows(model):
    exp = model.bind(md(), SEM, IMPL, NUMERICS, algorithm="expanded").analyze()
    abs_ = model.bind(md(), SEM, IMPL, NUMERICS, algorithm="absorbed_two_step").analyze()
    b = exp.header["bindings"]
    assert exp.value("N_Kproj") == exp.value("N_Vproj") == b["B"] * b["H"] * b["S_k"]
    assert abs_.value("N_Kproj") == abs_.value("N_Vproj") == 0
    NQ = abs_.value("N_Qproj")
    # the expanded projection charges Q^n and K^n/V; the absorbed two-step charges Q^n and the absorb step
    assert exp.value("F_proj[general]") == (2 * NQ * b["R_q"] * b["D_n"]
                                            + 2 * exp.value("N_Kproj") * b["R_k"] * (b["D_n"] + b["D_v"]))
    assert abs_.value("F_proj[general]") == 2 * NQ * b["R_q"] * b["D_n"] + 2 * NQ * b["D_n"] * b["R_k"]
    pre = model.bind(md(), {**SEM, "cache": {"merged_weight_reuse": 4}}, IMPL, NUMERICS,
                     algorithm="absorbed_precomputed").analyze()
    # the precomputed variant replaces both Q-side steps with the single merged projection C_q W~
    assert pre.value("F_proj[general]") == 2 * pre.value("N_Qproj") * b["R_q"] * b["R_k"]


# --- spec 6.2: each write and each read of a materialized intermediate is counted on its path ----------
@pytest.mark.parametrize("alg,flags,off,key", [
    # only the flag under test moves between the two runs, so the byte delta is attributable to it
    ("expanded", {"expanded_kv": True, "q_nope": True}, {"expanded_kv": True, "q_nope": False}, "mat_q_nope"),
    ("absorbed_two_step", {"q_tilde": True, "z": True}, {"q_tilde": False, "z": False}, "mat_q_tilde")])
def test_materialization_moves_bytes_on_the_paths(model, alg, flags, off, key):
    on = model.bind(md(), SEM, {**IMPL, "materialize": flags}, NUMERICS, algorithm=alg).analyze()
    without = model.bind(md(), SEM, {**IMPL, "materialize": off}, NUMERICS, algorithm=alg).analyze()
    row = next(r for r in on.extras["materialization_ledger"] if r["id"] == key)
    assert row["enabled"] is True and row["bytes_value"] > 0
    assert on.value("B[vmem_to_hbm]") > without.value("B[vmem_to_hbm]")
    assert on.value("B[hbm_to_vmem]") > without.value("B[hbm_to_vmem]")
    ledger = sum(r["traffic_value"] for r in on.extras["materialization_ledger"] if r["enabled"])
    base = sum(r["traffic_value"] for r in without.extras["materialization_ledger"] if r["enabled"])
    delta_paths = ((on.value("B[vmem_to_hbm]") - without.value("B[vmem_to_hbm]"))
                   + (on.value("B[hbm_to_vmem]") - without.value("B[hbm_to_vmem]")))
    assert delta_paths == ledger - base
    assert on.value(f"M_mat[{row['tensor']}]") == row["bytes_value"]


# --- spec 8.1: four non-interchangeable spill quantities ----------------------------------------------
def test_four_spill_quantities_and_the_comparison_graph(model):
    bound = model.bind(md(), SEM, IMPL, NUMERICS, algorithm="expanded")
    rep = bound.analyze()
    for name in ("N_spill_instructions", "M_spill_peak", "B_spill_fill", "dT_spill"):
        assert rep[name].status == "unknown" and rep[name].value is None
    bare = bound.attach_compile_evidence({"spill_exposed_s": 1e-6, "source": "probe"}).analyze()
    assert bare["dT_spill"].status == "partial" and "declared comparison graph" in bare["dT_spill"].missing_fields
    full = bound.attach_compile_evidence({"spill_exposed_s": 1e-6, "spill_comparison_graph": "no double buffer",
                                          "source": "probe"}).analyze()
    assert full["dT_spill"].status == "bound"


# --- spec 9.2 / 9.3 ------------------------------------------------------------------------------------
def test_resources_outside_declared_serial_groups_stay_in_the_bound(model):
    hw = {"name": "p", "scope_default": "per_chip",
          "compute": {"mxu": {"peak_flops": {"bfloat16": {"value": 1e14, "unit": "FLOP/s", "scope": "per_chip"}}},
                      "vpu": {"exp_throughput": {"value": 2e11, "unit": "op/s", "scope": "per_chip"},
                              "elementwise_throughput": {"value": 1e12, "unit": "op/s", "scope": "per_chip"},
                              "reduction_throughput": {"value": 5e11, "unit": "op/s", "scope": "per_chip"}}},
          "paths": {"hbm_to_vmem": {"bandwidth": {"value": 8e11, "unit": "byte/s", "scope": "per_chip"}},
                    "vmem_to_hbm": {"bandwidth": {"value": 8e11, "unit": "byte/s", "scope": "per_chip"}}}}
    impl = {**IMPL, "overlap_model": "serial_stage_sum", "serial_stage_groups": [["mxu"]]}
    rep = model.bind(md(), SEM, impl, NUMERICS, hw, algorithm="expanded").analyze()
    lb = rep.extras["resource_lower_bounds"]
    assert set(lb["resources_outside_declared_groups"]) >= {"exp", "reduce", "vpu"}
    combined = rep.value("T_LB[combined]")     # a declared overlap model gives one combination, not scenarios
    for res in ("exp", "reduce", "vpu", "path:hbm_to_vmem"):
        v = rep.value(f"T_LB[{res}]")
        if v is not None:
            assert combined >= v, f"the combination dropped {res}"


def test_layout_class_has_its_own_bound_and_names_its_missing_throughput(model):
    rep = model.bind(md(), SEM, IMPL, NUMERICS, algorithm="expanded").analyze()
    m = rep["T_LB[layout]"]
    assert m.value is None and "P_layout" in m.missing_fields
    assert "P_layout" in rep["T_LB[combined]"].missing_fields


def test_model_error_term_is_missing_not_zero(model):
    hw = {"name": "p", "scope_default": "per_chip",
          "compute": {"mxu": {"peak_flops": {"bfloat16": {"value": 1e14, "unit": "FLOP/s", "scope": "per_chip"}}}},
          "calibration": [{"node_kind": "matmul", "dtype": "bfloat16", "epsilon_low": 0.4, "epsilon_high": 0.8,
                           "startup_s": 1e-7, "source": "probe"}]}
    rep = model.bind(md(), SEM, IMPL, NUMERICS, hw, algorithm="expanded").analyze()
    cal = rep.extras["calibrated_prediction"]
    assert any("eps_model" in f for f in cal["missing_fields"])
    assert cal["terms"]["model_error_s"] is None
    hw2 = {**hw, "compute": {**hw["compute"], "model_error_s": {"value": 1e-7, "unit": "s", "scope": "per_chip"}}}
    rep2 = model.bind(md(), SEM, IMPL, NUMERICS, hw2, algorithm="expanded").analyze()
    cal2 = rep2.extras["calibrated_prediction"]
    assert cal2["terms"]["model_error_s"] == 1e-7
    assert not any("eps_model" in f for f in cal2["missing_fields"])
    lo, hi = cal["node_interval_s"]
    lo2, hi2 = cal2["node_interval_s"]
    assert lo2 == pytest.approx(lo + 1e-7) and hi2 == pytest.approx(hi + 1e-7)


def test_non_matmul_buckets_are_applied(model):
    hw = {"name": "p", "scope_default": "per_chip",
          "compute": {"mxu": {"peak_flops": {"bfloat16": {"value": 1e14, "unit": "FLOP/s", "scope": "per_chip"}}},
                      "vpu": {"exp_throughput": {"value": 2e11, "unit": "op/s", "scope": "per_chip"}}},
          "calibration": [{"node_kind": "matmul", "dtype": "bfloat16", "epsilon_low": 0.4, "epsilon_high": 0.8, "source": "p"}]}
    base = model.bind(md(), SEM, IMPL, NUMERICS, hw, algorithm="expanded").analyze().extras["calibrated_prediction"]
    assert next(p for p in base["predictions"] if p["node"] == "vector:exp")["status"] == "uncovered"
    hw2 = {**hw, "calibration": hw["calibration"] + [{"node_kind": "exp", "epsilon_low": 0.5, "epsilon_high": 0.9, "source": "p"}]}
    with_exp = model.bind(md(), SEM, IMPL, NUMERICS, hw2, algorithm="expanded").analyze().extras["calibrated_prediction"]
    assert next(p for p in with_exp["predictions"] if p["node"] == "vector:exp")["status"] == "predicted"
    assert with_exp["node_interval_s"][1] > base["node_interval_s"][1]


# --- spec 10.2 / 10.3 ----------------------------------------------------------------------------------
def test_illegal_candidates_leave_the_argmin_and_are_reported(model):
    bound = model.bind(md(), SEM, IMPL, NUMERICS, algorithm="expanded")
    cmp_ = bound.compare([{"name": "ok", "b_q": 4}, {"name": "illegal", "b_rq": 999}])
    assert "illegal" not in cmp_["pareto_set"]
    assert [e["name"] for e in cmp_["excluded_candidates"]] == ["illegal"]
    assert any("b_rq_le_R_q" in n for n in cmp_["selection_notes"])
    assert "illegal" not in [r[0] for r in (cmp_["calibrated_ranking"] or [])]


def test_reduce_q_subtile_reports_everything_the_template_asks_for(model):
    bound = model.bind(md(), SEM, {**IMPL, "b_q": 16}, NUMERICS, algorithm="expanded")
    row = bound.compare([{"name": "smallQ", "b_q": 8}])["rows"][1]
    assert row["delta_vs_baseline"]["local[score_X]"] < 0        # M_X
    assert row["delta_vs_baseline"]["local[accumulator_A]"] < 0  # M_A
    assert row["delta_vs_baseline"]["B[hbm_to_vmem]"] > 0        # KV reads rise
    assert row["delta_vs_baseline"]["n_programs"] > 0            # more query programs
    shapes = row["structure_delta"]["mxu_tile_shape_changes"]
    assert shapes and all(v["baseline"]["value"][0] == 16 and v["candidate"]["value"][0] == 8 for v in shapes.values())


def test_delay_v_load_reports_lifetime_and_prefetch(model):
    bound = model.bind(md(), SEM, IMPL, NUMERICS, algorithm="expanded")
    early = bound.analyze()
    late = model.bind(md(), SEM, {**IMPL, "v_load": "delayed"}, NUMERICS, algorithm="expanded").analyze()
    assert early.value("V_lifetime_stages") > late.value("V_lifetime_stages")
    assert early.value("V_prefetch_overlapped") is True and late.value("V_prefetch_overlapped") is False
    assert any("candidate hypothesis" in a for a in late["V_prefetch_overlapped"].assumptions)
    row = bound.compare([{"name": "delayV", "v_load": "delayed"}])["rows"][1]
    assert row["delta_vs_baseline"]["V_lifetime_stages"] < 0
    undeclared = model.bind(md(), SEM, {k: v for k, v in IMPL.items() if k != "v_load"}, NUMERICS, algorithm="expanded").analyze()
    assert "v_load" in undeclared["V_lifetime_stages"].missing_fields


def test_switch_to_absorbed_reports_the_structural_change(model):
    bound = model.bind(md(), SEM, IMPL, NUMERICS, algorithm="expanded")
    cmp_ = bound.compare([{"name": "abs", "algorithm": "absorbed_two_step"}])
    row = cmp_["rows"][1]
    assert row["values"]["F_A_minus_F_E"] is not None
    assert row["values"]["local[accumulator_A]"] != cmp_["rows"][0]["values"]["local[accumulator_A]"]  # D_v vs R_k width
    assert row["delta_vs_baseline"]["HBM_mat[K^n,V]"] == "disappears"
    assert "n/a" in __import__("mla_model").comparison_to_markdown(cmp_)
    assert {"q_tilde", "kv_latent_tile"} <= set(row["structure_delta"]["appear"])
    assert "v_tile" in row["structure_delta"]["disappear"]


def test_feasible_region_is_a_surface(model):
    rep = model.bind(md(), SEM, IMPL, NUMERICS, algorithm="expanded").analyze()
    rows = rep.extras["feasible_region_by_stage"]["vreg"]["softmax"]
    assert len(rows) > 3 and sum(1 for r in rows if r["declared_b_q"]) == 1
    vals = [r["b_k_max"] for r in rows if r["b_k_max"] is not None]
    assert all(x >= y for x, y in zip(vals, vals[1:]))


# --- spec 11.1 / 11.2: the algorithm may be preserved as unknown ---------------------------------------
def test_recommended_api_call_runs_without_an_algorithm(model):
    bound = model.bind(tensor_metadata=md(), semantic_config=SEM, implementation=IMPL, numeric_policy=NUMERICS)
    rep = bound.analyze()
    assert rep.algorithm is None
    assert rep.value("C_valid") is not None                    # algorithm-independent, reported once
    for alg in ("expanded", "absorbed_two_step", "absorbed_precomputed"):
        m = rep[f"F_total[useful] [algorithm={alg}]"]
        assert m.status == "partial" and "algorithm" in m.missing_fields
        assert m.value == model.bind(md(), SEM, IMPL, NUMERICS, algorithm=alg).analyze().value("F_total[useful]")
    assert rep.get("F_total[useful]") is None                  # never a silently chosen default
    # a metric that every variant agrees on appears once, labelled as holding for all of them
    shared = rep["C_valid"]
    assert shared.algorithm == "all variants"
    # the report invariants hold in this mode too
    for m in rep.metrics:
        assert m.status in {"symbolic", "partial", "bound", "unknown", "conflict", "not_applicable"}
        assert m.coverage and m.algorithm and m.formula_id and m.expression
    rep.to_json(), rep.to_markdown()


# --- explicit failure instead of fallback (implementation prompt) --------------------------------------
@pytest.mark.parametrize("group,patch", [
    ("implementation", {"layout": ["a", "b"]}),
    ("implementation", {"materialize": [1, 2]}),
    ("implementation", {"subdivide": [4], "rect_policy": "skip_future_subdivide"}),
    ("implementation", {"subdivide": [4, 4, 4], "rect_policy": "skip_future_subdivide"}),
    ("implementation", {"subdivide": "ab", "rect_policy": "skip_future_subdivide"}),
    ("numerics", {"dtype_bytes_overrides": [1, 2]}),
    ("numerics", {"dtype_bytes_overrides": {"bfloat16": "two"}}),
    ("semantics", {"outputs": "O"}),
    ("semantics", {"position_offset": 1.5}),
    ("semantics", {"cache": {"kind": "latent_only"}}),
])
def test_malformed_declarations_raise_instead_of_falling_back(model, group, patch):
    cfg = {"semantics": dict(SEM), "implementation": dict(IMPL), "numerics": dict(NUMERICS)}
    cfg[group].update(patch)
    with pytest.raises((MetadataError, BindingError)):
        model.bind(md(), cfg["semantics"], cfg["implementation"], cfg["numerics"], algorithm="expanded").analyze()


def test_dangling_alias_and_stray_layout_id_are_rejected(model):
    t = md()
    t["w_v"] = {**t["w_v"], "alias_of": "ghost"}
    with pytest.raises(MetadataError):
        model.bind(t, SEM, IMPL, NUMERICS, algorithm="expanded")
    with pytest.raises(MetadataError):
        model.bind(md(), SEM, {**IMPL, "layout": {"no_such_object": "compact"}}, NUMERICS, algorithm="expanded").analyze()


def test_unread_tensor_metadata_key_is_reported(model):
    t = md()
    t["kv_latent"] = {**t["kv_latent"], "capacty_shape": [1, 64, 6]}      # misspelled on purpose
    rep = model.bind(t, SEM, IMPL, NUMERICS, algorithm="expanded").analyze()
    assert any("capacty_shape" in w and "not read" in w for w in rep.warnings)


def test_per_tensor_allocation_extent(model):
    t = md()
    t["kv_latent"] = {**t["kv_latent"], "capacity_shape": [1, 4096, 6]}
    rep = model.bind(t, {**SEM, "active_lengths": {"Sq": 32, "Sk": 32}}, IMPL, NUMERICS, algorithm="expanded").analyze()
    ledger = {r["role"]: r for r in rep.extras["interface_ledger"]["tensors"]}
    for role in ("kv_latent", "k_pe", "q_latent"):
        assert rep.value(f"bytes[{role}]") == ledger[role]["allocated_bytes"], role


def test_no_declared_field_is_accepted_and_then_ignored(model):
    """The eight fields this review round touched must each be visible in the report somewhere.

    This is not a completeness sweep over the whole configuration surface; it covers the fields the round
    changed, which are the ones at risk of having been wired up only halfway.
    """
    def snap(sem=None, impl=None, num=None):
        rep = model.bind(md(), {**SEM, **(sem or {})}, {**IMPL, **(impl or {})}, {**NUMERICS, **(num or {})},
                         algorithm="expanded").analyze()
        return [(m.metric, str(m.value), m.status, tuple(m.missing_fields), m.scope) for m in rep.metrics]

    cases = [
        ("impl", "q_resident", True, False), ("impl", "v_load", "early", "delayed"),
        ("impl", "schedule", None, "q_outer_kv_inner"), ("impl", "b_rk", 2, 4),
        ("impl", "executed_extent_policy", "logical", "padded_to_tile"),
        ("num", "cast_points", [], ["exp_to_p"]),
        ("num", "final_normalize", "divide", "reciprocal_multiply"),
        ("num", "acceptance", None, {"rel_tol": 0.01}),
    ]
    inert = []
    for group, field, a, b in cases:
        kw = {"sem": None, "impl": None, "num": None}
        # the final-normalisation choice only exists for the unnormalized recurrence
        extra_num = {"recurrence": "unnormalized"} if field == "final_normalize" else {}
        outs = []
        for v in (a, b):
            kw = {"sem": None, "impl": None, "num": dict(extra_num)}
            kw[group] = {**(kw[group] or {}), field: v}
            outs.append(snap(**kw))
        if outs[0] == outs[1]:
            inert.append(field)
    assert not inert, f"declared fields with no visible effect on the report: {inert}"
