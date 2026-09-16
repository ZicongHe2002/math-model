"""Regressions for the fifth review round (verification of the fourth round's fixes): performance.

Each test names the problem it protects against.  Expected values are written here from the specification or
an independent hand computation, never read back from the package; a test that compares two surfaces of one
report against each other says so (differential) in its docstring.
"""
import json

import pytest

from conftest import NUMERICS, make_metadata
from test_verify_round4 import SEM, IMPL, HW, md
from mla_model import MLAForwardModel
from mla_model.errors import MetadataError

# ---- illustrative test inputs (never package defaults) ------------------------------------------------------
# HW with every throughput and every path bandwidth declared: no term is unknown for want of a profile figure.
PERF_HW_FULL = json.loads(json.dumps(HW))
PERF_HW_FULL["compute"]["vpu"]["layout_throughput"] = {"value": 1e12, "unit": "op/s", "scope": "per_chip"}
PERF_HW_FULL["paths"]["vmem_to_vreg"] = {"bandwidth": {"value": 4e12, "unit": "byte/s", "scope": "per_chip"}}
PERF_HW_FULL["paths"]["vreg_to_vmem"] = {"bandwidth": {"value": 4e12, "unit": "byte/s", "scope": "per_chip"}}

PERF_EPS_LO, PERF_EPS_HI, PERF_STARTUP, PERF_PEAK = 0.5, 0.8, 1e-7, 1e14       # HW peak_flops[bfloat16] = 1e14
PERF_MATMUL_BUCKET = {"node_kind": "matmul", "dtype": "bfloat16", "epsilon_low": PERF_EPS_LO,
                      "epsilon_high": PERF_EPS_HI, "startup_s": PERF_STARTUP, "device": "p"}


def perf_hw(base=HW, calibration=(), **compute_extra):
    """Deep copy of a profile dict with calibration buckets and extra compute quantities attached."""
    h = json.loads(json.dumps(base))
    h["calibration"] = list(calibration)
    for key, val in compute_extra.items():
        h["compute"][key] = val
    return h


def perf_report(hw, impl=IMPL, num=NUMERICS, alg="expanded", sem=SEM, tensors=None):
    return MLAForwardModel().bind(tensors or md(), sem, impl, num, hw, algorithm=alg).analyze()


def perf_rows(rep):
    return {p["node"]: p for p in rep.extras["calibrated_prediction"]["predictions"]}


# ---- [0] PERF-1 ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("alg", ["expanded", "absorbed_two_step"])
@pytest.mark.parametrize("overlap", ["full_overlap_max", "serial_stage_sum"])
def test_perf_combined_bound_is_never_bound_while_a_term_is_partial(alg, overlap):
    """[0] PERF-1: with a COMPLETE profile every term has a value, yet path:vmem_to_vreg is only a scenario
    (register_schedule_evidence) and CP is never known, so T_LB[combined] must be 'partial' and its
    missing_fields the union of its terms' missing_fields.  Design decision as implemented: the always-missing
    CP keeps the combination partial (spec 9.2 includes the critical-path term the model cannot supply).
    Pre-fix the status was derived from 'every term has a value' and read 'bound' with undeclared fields listed."""
    rep = perf_report(PERF_HW_FULL, {**IMPL, "overlap_model": overlap}, alg=alg)
    lb = rep.extras["resource_lower_bounds"]
    m = rep["T_LB[combined]"]
    assert m.value is not None and m.value > 0
    assert m.status == "partial"
    assert "CP" in m.missing_fields and "register_schedule_evidence" in m.missing_fields
    for row in lb["bounds"]:                                   # union over every term
        assert set(row["missing"]) <= set(m.missing_fields), row["resource"]
        assert set(rep[f"T_LB[{row['resource']}]"].missing_fields) <= set(m.missing_fields)
    # a term that HAS a value but rests on an undeclared field is partial, and so is the combination
    assert rep["T_LB[path:vmem_to_vreg]"].status == "partial"
    assert rep["T_LB[path:vmem_to_vreg]"].value is not None


def test_perf_combined_bound_names_undeclared_active_lengths_and_stays_partial():
    """[0] PERF-1: when active_lengths are undeclared every work term is a scenario; the combination must
    name 'active_lengths' and must not read 'bound' even though every term evaluates to a number."""
    sem = {k: v for k, v in SEM.items() if k != "active_lengths"}
    rep = perf_report(PERF_HW_FULL, {**IMPL, "overlap_model": "full_overlap_max"}, sem=sem)
    m = rep["T_LB[combined]"]
    assert m.value is not None
    assert m.status == "partial"
    assert "active_lengths" in m.missing_fields and "CP" in m.missing_fields
    assert rep["T_LB[mxu]"].status == "partial" and "active_lengths" in rep["T_LB[mxu]"].missing_fields


def test_perf_combined_status_agrees_between_metric_and_extras():
    """[0] PERF-1 (differential): the metric T_LB[combined] and extras['resource_lower_bounds'] are two
    surfaces of one combination; status, value and missing set must agree."""
    rep = perf_report(PERF_HW_FULL, {**IMPL, "overlap_model": "serial_stage_sum"})
    lb = rep.extras["resource_lower_bounds"]
    m = rep["T_LB[combined]"]
    assert m.status == lb["status"] == "partial"
    assert m.value == lb["combined_value_known_terms_only"]
    assert sorted(m.missing_fields) == sorted(lb["combined_missing_fields"])


# ---- [1] PERF-2 ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("alg", ["expanded", "absorbed_two_step", "absorbed_precomputed"])
def test_perf_absent_rope_matmul_charges_no_startup(alg):
    """[1] PERF-2: at D_r = 0 the positional matmul executes nothing (F_v = 0), so the calibrated prediction
    must not charge its t_startup.  Hand computation (spec 4.2 / 9.3): B=1, H=2, Sq=Sk=32, b_q=b_k=8, causal
    offset 0, skip_future keeps the kv-blocks j <= i for q-block i -> 1+2+3+4 = 10 rectangles of 64 cells per
    (b, h) -> 1280 executed cells; F_rope(D_r=8) = 2 * 1280 * 8 = 20480 FLOP.  Going from D_r = 8 to D_r = 0
    therefore removes exactly F_rope/(eps*P) + t_startup from each end of the interval; pre-fix only the FLOP
    term disappeared (the startup of the absent node was still summed)."""
    hw = perf_hw(PERF_HW_FULL, [PERF_MATMUL_BUCKET],
                 launch_overhead_s={"value": 1e-6, "unit": "s", "scope": "per_chip"},
                 model_error_s={"value": 0.0, "unit": "s", "scope": "per_chip"})
    r0 = perf_report(hw, alg=alg, tensors=make_metadata(1, 2, 32, 32, 4, 6, 8, 0, 8, pe=False))
    r8 = perf_report(hw, alg=alg, tensors=make_metadata(1, 2, 32, 32, 4, 6, 8, 8, 8))
    lo0, hi0 = r0.extras["calibrated_prediction"]["node_interval_s"]
    lo8, hi8 = r8.extras["calibrated_prediction"]["node_interval_s"]
    f_rope = 2 * 1280 * 8
    assert lo8 - lo0 == pytest.approx(f_rope / (PERF_EPS_HI * PERF_PEAK) + PERF_STARTUP, rel=1e-9)
    assert hi8 - hi0 == pytest.approx(f_rope / (PERF_EPS_LO * PERF_PEAK) + PERF_STARTUP, rel=1e-9)
    rope = perf_rows(r0)["QK_rope"]
    assert rope["status"] != "predicted" and rope["work_value"] == 0
    assert rope["t_low"] is None and rope["t_high"] is None and rope["startup_s"] is None
    # every predicted matmul node still carries its startup; the absent one is the only one without
    predicted = [p for p in perf_rows(r0).values() if p["status"] == "predicted" and p["work_unit"] == "FLOP"]
    assert predicted and all(p["startup_s"] == PERF_STARTUP and p["work_value"] > 0 for p in predicted)
    assert perf_rows(r8)["QK_rope"]["status"] == "predicted"


# ---- [2] PERF-3, [7] PERF-8, [8] PERF-9 ------------------------------------------------------------------------
def _perf_patch_bucket(**fields):
    def patch(h):
        h["calibration"].append({**PERF_MATMUL_BUCKET, **fields})
    return patch


def _perf_patch_compute(key, quantity):
    def patch(h):
        h["compute"][key] = quantity
    return patch


def _perf_patch_layout(tile):
    def patch(h):
        h["layout"] = {"float32": tile}
    return patch


PERF_MALFORMED = {
    # [2] PERF-3: malformed but plausible calibration / quantity declarations
    "m_range_one_element": _perf_patch_bucket(m_range=[1]),
    "m_range_strings": _perf_patch_bucket(m_range=["a", "b"]),
    "m_range_three": _perf_patch_bucket(m_range=[1, 2, 3]),
    "epsilon_low_string": _perf_patch_bucket(epsilon_low="abc"),
    "startup_string": _perf_patch_bucket(startup_s="x"),
    "startup_bool": _perf_patch_bucket(startup_s=True),
    "validation_error_string": _perf_patch_bucket(validated_on_unseen_shapes=True, validation_error="e"),
    "bucket_toolchain_string": _perf_patch_bucket(toolchain="llvm"),
    "bucket_device_int": _perf_patch_bucket(device=5),
    "quantity_value_string": _perf_patch_compute("model_error_s", {"value": "e", "unit": "s"}),
    "throughput_value_string": _perf_patch_compute("vpu", {"exp_throughput": {"value": "fast", "unit": "op/s"}}),
    "quantity_value_bool": _perf_patch_compute("launch_overhead_s", {"value": True, "unit": "s"}),
    "layout_string_tile": _perf_patch_layout(["a", 128]),
    # [7] PERF-8: a negative time term is a sign error, not a latency
    "negative_launch_overhead": _perf_patch_compute("launch_overhead_s", {"value": -1e-3, "unit": "s"}),
    "negative_model_error": _perf_patch_compute("model_error_s", {"value": -1e-3, "unit": "s"}),
    # [8] PERF-9: a zero or non-integer layout tile extent yields NaN bytes
    "layout_zero_tile": _perf_patch_layout([0, 128]),
    "layout_negative_tile": _perf_patch_layout([8, -128]),
    "layout_float_tile": _perf_patch_layout([8.0, 128]),
}


@pytest.mark.parametrize("case", sorted(PERF_MALFORMED))
def test_perf_malformed_profile_declarations_fail_as_metadata_errors_at_bind(case):
    """[2] PERF-3, [7] PERF-8, [8] PERF-9: every malformed hardware / calibration declaration in the table is
    refused with MetadataError when the profile is loaded (inside bind), never as a raw TypeError/ValueError and
    never later inside analyze() after a successful bind.  Pre-fix several escaped raw, and m_range problems and
    the bucket toolchain type only blew up inside analyze()."""
    hw = perf_hw(HW)
    PERF_MALFORMED[case](hw)
    with pytest.raises(MetadataError):
        MLAForwardModel().bind(md(), SEM, IMPL, NUMERICS, hw, algorithm="expanded")


def test_perf_zero_time_terms_and_null_range_bounds_are_legal():
    """[7] PERF-8 / [2] PERF-3 boundary guard (passes pre-fix by design): the sign check added for PERF-8 is
    '>= 0' for unit 's' and the range validation added for PERF-3 admits null bounds, so a zero launch overhead,
    a zero model error and an open-ended m_range [null, null] must still be accepted; this protects the fixes
    from over-tightening.  With both time terms 0 and every matmul node covered the interval is, from spec 9.3
    and the formula inventory (spec 5.3 / 4.2), hand-derived for md() = B=1, H=2, Sq=Sk=32, Rq=4, Rk=6, Dn=8,
    Dr=4, Dv=8, projection_scope full, causal skip_future with 8x8 tiles:
      N_Q = N_K = N_V = BH S = 64 rows;  C_rect = 1280 executed cells (10 rectangles of 64 per (b, h), 2 heads)
      Q_proj  = 2 N_Q Rq Dn = 2*64*4*8 =  4096      QK_nope = 2 C Dn = 2*1280*8 = 20480
      K_proj  = 2 N_K Rk Dn = 2*64*6*8 =  6144      QK_rope = 2 C Dr = 2*1280*4 = 10240
      V_proj  = 2 N_V Rk Dv = 2*64*6*8 =  6144      PV      = 2 C Dv = 2*1280*8 = 20480
      sum F_v = 67584 FLOP over 6 predicted nodes
      low  = 67584 / (0.8 * 1e14) + 6 * 1e-7 = 8.448e-10 + 6e-7      high = 67584 / (0.5 * 1e14) + 6 * 1e-7
    The final line is differential: the metric T_pred[node_interval] must publish that same interval."""
    hw = perf_hw(PERF_HW_FULL, [{**PERF_MATMUL_BUCKET, "m_range": [None, None]}],
                 launch_overhead_s={"value": 0.0, "unit": "s", "scope": "per_chip"},
                 model_error_s={"value": 0.0, "unit": "s", "scope": "per_chip"})
    rep = perf_report(hw)
    cal = rep.extras["calibrated_prediction"]
    assert cal["applied"] is True
    assert cal["terms"]["launch_overhead_s"] == 0.0 and cal["terms"]["model_error_s"] == 0.0
    sum_f, n_nodes = 4096 + 6144 + 6144 + 20480 + 10240 + 20480, 6
    lo, hi = cal["node_interval_s"]
    assert lo == pytest.approx(sum_f / (PERF_EPS_HI * PERF_PEAK) + n_nodes * PERF_STARTUP, rel=1e-12)
    assert hi == pytest.approx(sum_f / (PERF_EPS_LO * PERF_PEAK) + n_nodes * PERF_STARTUP, rel=1e-12)
    rows = perf_rows(rep)
    assert [r for r in rows if rows[r]["status"] == "predicted"] == ["Q_proj", "K_proj", "V_proj", "QK_nope", "QK_rope", "PV"]
    assert rep["T_pred[node_interval]"].value == [lo, hi]                   # (differential) metric == extras interval


def test_perf_negative_time_term_never_reaches_a_prediction():
    """[7] PERF-8: the pre-fix behaviour was a negative node_interval_s reported as 'partial'; the profile is
    now refused, so no report with a negative interval can exist for such a profile."""
    hw = perf_hw(PERF_HW_FULL, [PERF_MATMUL_BUCKET], launch_overhead_s={"value": -1e-3, "unit": "s", "scope": "per_chip"})
    with pytest.raises(MetadataError, match="non-negative"):
        perf_report(hw)


# ---- [3] PERF-4 ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("groups", [
    [[["mxu"]]],                     # nested list entry (unhashable)
    [[{"a": 1}]],                    # dict entry (unhashable)
    [["mxu", 5]],                    # non-string entry
    [["mxu"], "exp"],                # a group that is not a list
    [["nope"]],                      # not a resource class
    [["mxu"], ["mxu"]],              # a class in two groups
], ids=["nested_list", "dict_entry", "int_entry", "bare_string_group", "unknown_class", "duplicate_class"])
def test_perf_serial_stage_groups_entries_are_validated_as_metadata_errors(groups):
    """[3] PERF-4: every malformed serial_stage_groups declaration is refused with MetadataError; pre-fix an
    unhashable entry (nested list / dict) escaped as a raw TypeError from the set comprehension."""
    with pytest.raises(MetadataError):
        MLAForwardModel().bind(md(), SEM, {**IMPL, "overlap_model": "serial_stage_sum", "serial_stage_groups": groups},
                               NUMERICS, HW, algorithm="expanded")


# ---- [4] PERF-5 ----------------------------------------------------------------------------------------------
def _perf_vector_bucket(kind, dtype):
    return {"node_kind": kind, "dtype": dtype, "epsilon_low": 0.5, "epsilon_high": 1.0, "startup_s": 0.0, "device": "p"}


def test_perf_exp_bucket_is_matched_on_the_exp_dtype_not_the_matmul_dtype():
    """[4] PERF-5: NUMERICS runs exp in float32 with bfloat16 matmul operands.  A bucket for the dtype the
    exp actually executes in (float32) must be applied; a bucket for a dtype in which no exp runs (bfloat16)
    must be uncovered and named.  Pre-fix this was the other way round."""
    right = perf_report(perf_hw(HW, [_perf_vector_bucket("exp", "float32")]))
    wrong = perf_report(perf_hw(HW, [_perf_vector_bucket("exp", "bfloat16")]))
    assert perf_rows(right)["vector:exp"]["status"] == "predicted"
    assert "calibration bucket[exp]" not in right.extras["calibrated_prediction"]["missing_fields"]
    assert perf_rows(wrong)["vector:exp"]["status"] == "uncovered"
    assert "calibration bucket[exp]" in wrong.extras["calibrated_prediction"]["missing_fields"]
    assert wrong.extras["calibrated_prediction"]["applied"] is False


def test_perf_dtype_agnostic_exp_bucket_predicts_the_same_as_the_exp_dtype_bucket():
    """[4] PERF-5 (differential): a bucket with dtype null applies to any dtype, so its prediction row must
    equal the row produced by the bucket declared for exp_dtype; the value itself is W_exp/(eps*P_exp) with
    eps_high = 1 so t_low = W_exp / P_exp and t_high = 2 t_low (eps_low = 0.5)."""
    typed = perf_rows(perf_report(perf_hw(HW, [_perf_vector_bucket("exp", "float32")])))["vector:exp"]
    agnostic = perf_rows(perf_report(perf_hw(HW, [_perf_vector_bucket("exp", None)])))["vector:exp"]
    assert typed["status"] == agnostic["status"] == "predicted"
    assert typed["t_low"] == agnostic["t_low"] and typed["t_high"] == agnostic["t_high"]
    assert typed["t_high"] == pytest.approx(2 * typed["t_low"], rel=1e-12)
    assert typed["t_low"] == pytest.approx(typed["work_value"] / 2e11, rel=1e-12)   # HW exp_throughput = 2e11


@pytest.mark.parametrize("kind,res,field,matching,other", [
    ("reduce", "vector:reduce", "score_dtype", "float32", "bfloat16"),
    ("vector", "vector:vpu", "state_dtype", "bfloat16", "float32"),
], ids=["reduce_on_score_dtype", "vpu_on_state_dtype"])
def test_perf_reduce_and_vpu_buckets_are_matched_on_their_own_class_dtype(kind, res, field, matching, other):
    """[4] PERF-5: reductions are matched on score_dtype and the elementwise class on state_dtype (design as
    implemented); a bucket declared for that dtype is predicted, one declared for another dtype is uncovered."""
    hw = perf_hw(HW, [_perf_vector_bucket(kind, matching)])
    assert perf_rows(perf_report(hw, num={**NUMERICS, field: matching}))[res]["status"] == "predicted"
    assert perf_rows(perf_report(hw, num={**NUMERICS, field: other}))[res]["status"] == "uncovered"


@pytest.mark.parametrize("bucket_dtype,score_dtype,expected", [
    ("float32", "float32", "predicted"),
    ("bfloat16", "bfloat16", "predicted"),
    ("float32", "bfloat16", "uncovered"),
    ("bfloat16", "float32", "uncovered"),
])
def test_perf_vpu_bucket_falls_back_to_score_dtype_when_state_dtype_is_undeclared(bucket_dtype, score_dtype, expected):
    """[4] PERF-5: the elementwise class is matched on state_dtype, and when state_dtype is undeclared on the
    score dtype it falls back to (design as implemented: `state_dtype or score_dtype`); it is never matched on
    the matmul operand dtype (bfloat16 here), which would make the ('float32', 'float32') bucket uncovered."""
    num = {k: v for k, v in NUMERICS.items() if k != "state_dtype"}
    hw = perf_hw(HW, [_perf_vector_bucket("vector", bucket_dtype)])
    assert perf_rows(perf_report(hw, num={**num, "score_dtype": score_dtype}))["vector:vpu"]["status"] == expected


# Not pinned (unfixed, reported): the fourth PERF-5 class, 'layout', is passed class dtype None, so a layout bucket
# declared for ANY dtype (int8 included) is applied as 'predicted' with no dtype-match caveat.  A test here would
# enshrine that defect; it is left to the code owner (add a layout-class dtype, or a note + missing_fields entry).


# ---- [5] PERF-6 ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("device,matches", [
    ("tpu v6e", False),                     # first word 'tpu' is in the name, but the string is not a substring
    ("v6e trillium", False),
    ("gpu-h100", False),
    ("tpu-v6e", True),                      # substring of the profile name
    ("TPU-V6E (illustrative) rev2", True),  # the profile name is a substring of the device (case-insensitive)
])
def test_perf_device_match_is_one_predicate_on_both_report_surfaces(device, matches):
    """[5] PERF-6 (differential): the profile note ('validity not established', surfaced in rep.warnings) and
    the prediction caveat ('calibration.device match' in T_pred missing_fields) must agree on whether a bucket's
    device identifies the profile, and the shared predicate is case-insensitive substring either way.  Pre-fix
    the loader used 'first word of the device is in the name' and said nothing for 'tpu v6e'."""
    hw = perf_hw(HW, [{**PERF_MATMUL_BUCKET, "device": device}])
    hw["name"] = "tpu-v6e (illustrative)"
    rep = perf_report(hw)
    load_note = any("validity not established" in w and repr(device) in w for w in rep.warnings)
    cal = rep.extras["calibrated_prediction"]
    pred_caveat = "calibration.device match" in cal["missing_fields"]
    assert cal["applied"] is True
    assert load_note == pred_caveat == (not matches)
    assert pred_caveat == ("calibration.device match" in rep["T_pred[node_interval]"].missing_fields)
    assert any("identify this profile's device" in n for n in cal["notes"]) == (not matches)


# ---- [6] PERF-7 ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("bucket_dtype", [None, "bfloat16"])
def test_perf_undeclared_matmul_dtype_is_named_by_both_surfaces(bucket_dtype):
    """[6] PERF-7: with matmul_input_dtype undeclared no peak can be selected.  The lower bound names
    'matmul_input_dtype'; the prediction must name the same field, not a non-existent bucket problem
    ('applicable calibration bucket') nor the spelling 'peak_flops[None]'."""
    num = {k: v for k, v in NUMERICS.items() if k != "matmul_input_dtype"}
    rep = perf_report(perf_hw(HW, [{**PERF_MATMUL_BUCKET, "dtype": bucket_dtype}]), num=num)
    lb_missing = rep["T_LB[mxu]"].missing_fields
    cal = rep.extras["calibrated_prediction"]
    pred = rep["T_pred[node_interval]"]
    assert rep["T_LB[mxu]"].status == "unknown" and "matmul_input_dtype" in lb_missing
    assert pred.status == "unknown" and pred.value is None
    assert "matmul_input_dtype" in cal["missing_fields"] and "matmul_input_dtype" in pred.missing_fields
    for surface in (cal["missing_fields"], pred.missing_fields):
        assert "applicable calibration bucket" not in surface
        assert not any("peak_flops[None]" in f for f in surface)
    assert all(p["status"] == "uncovered" for p in cal["predictions"] if p["work_unit"] == "FLOP")


# ---- [10] PERF-11 --------------------------------------------------------------------------------------------
def test_perf_mixed_group_expression_names_its_unknown_member_as_omitted():
    """[10] PERF-11: HW has no layout_throughput, so in the group [layout, mxu] only mxu has a value.  The
    printed combination must say the layout term was omitted rather than 'max(layout, mxu)' as if it took part;
    P_layout stays in missing_fields.  (differential) the number equals the full_overlap_max combination, since a
    single-member sum is that member and every other class enters through the max (spec 9.2)."""
    impl = {**IMPL, "overlap_model": "serial_stage_sum", "serial_stage_groups": [["layout", "mxu"]]}
    rep = perf_report(HW, impl)
    lb = rep.extras["resource_lower_bounds"]
    expr = lb["combined_expression"]
    assert rep["T_LB[layout]"].value is None and rep["T_LB[mxu]"].value is not None
    assert "max(layout, mxu)" not in expr and "max(mxu, layout)" not in expr
    assert "max(mxu)" in expr and "[layout: unknown, omitted]" in expr
    assert rep["T_LB[combined]"].expression == expr
    assert "P_layout" in rep["T_LB[combined]"].missing_fields
    full = perf_report(HW, {**IMPL, "overlap_model": "full_overlap_max"})
    assert rep.value("T_LB[combined]") == full.value("T_LB[combined]")
    # a group with NO known member keeps the whole-group spelling
    alone = perf_report(HW, {**IMPL, "overlap_model": "serial_stage_sum", "serial_stage_groups": [["layout"], ["mxu"]]})
    assert "[layout: unknown, omitted]" in alone.extras["resource_lower_bounds"]["combined_expression"]
