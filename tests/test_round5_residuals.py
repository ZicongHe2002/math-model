"""Regressions for the fifth review round (verification of the fourth round's fixes): residuals.

Each test names the problem it protects against.  Expected values are written here from the specification or
an independent hand computation, never read back from the package; a test that compares two surfaces of one
report against each other says so (differential) in its docstring.
"""
import json

import pytest

from conftest import NUMERICS, make_metadata
from test_verify_round4 import SEM, IMPL, HW, md
from mla_model import MLAForwardModel
from mla_model.errors import BindingError, MetadataError, MLAModelError

RESID_INTERNAL_SYMBOLS = {"n_buf", "s_scratch", "R_vmem", "R_vreg", "h_pp"}   # never a missing_fields label


def resid_bind(tensors=None, semantics=SEM, strategy=IMPL, numerics=NUMERICS, hardware=None):
    return MLAForwardModel().bind(tensors or md(), semantics, strategy, numerics, hardware, algorithm="expanded")


def resid_model_error(fn):
    """Return the exact model error class ``fn`` raises; a raw TypeError/ValueError propagates and fails the test."""
    with pytest.raises(MLAModelError) as ei:
        fn()
    assert ei.type in (MetadataError, BindingError), ei.type     # the two bind-time kinds; the caller picks the exact one
    return ei.type


def resid_hw(**extra):
    """HW plus extra top-level keys, with deep copies of the nested compute/paths tables so HW is never mutated."""
    hw = json.loads(json.dumps(HW))
    hw.update(extra)
    return hw


# --- R1 [32] C4 ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("candidates, objectives", [
    ([json.loads('{"name": 7, "b_q": 4}')], None),            # a JSON caller writing a number for the name
    ([{"name": "a", "b_q": 4}], 5),
    ([{"name": "a", "b_q": 4}], "F_total[useful]"),           # a bare string is not a list of metric names
])
def test_resid_compare_rejects_malformed_input_as_metadata_error(candidates, objectives):
    """R1 [32]: compare() validates its inputs and raises MetadataError, not a raw TypeError."""
    bound = resid_bind()
    kw = {} if objectives is None else {"objectives": objectives}
    assert resid_model_error(lambda: bound.compare(candidates, **kw)) is MetadataError


def test_resid_compare_accepts_a_list_of_objectives():
    """R1 [32]: the legal form (a list of metric names) is accepted."""
    assert isinstance(resid_bind().compare([{"name": "a", "b_q": 4}], objectives=["F_total[useful]"]), dict)


# --- R2 [26] P6 ----------------------------------------------------------------------------------------------
def test_resid_declared_cast_point_notes_reach_the_warnings():
    """R2 [26]: notes about declared cast sites are warnings, not only executed_graph assumptions."""
    rep = resid_bind(numerics={**NUMERICS, "cast_points": ["z_to_output", "exp_to_p"]}).analyze()
    absent = [w for w in rep.warnings if "cast_points declares z_to_output" in w and "has no Z to cast" in w]
    assert len(absent) == 1 and "reported as absent" in absent[0]
    exact = [w for w in rep.warnings if "cast graph is exactly the declared sites ['exp_to_p', 'z_to_output']" in w]
    assert len(exact) == 1
    # W_vec[cast] counts only exp_to_p, one cast per executed score cell.  B=1, H=2, Sq=Sk=32 causal (offset 0),
    # 8x8 tiles, skip_future, logical extents: q-block i (0..3) executes k-blocks 0..i -> 1+2+3+4 = 10 rectangles
    # of 64 cells = 640 cells per (b, h); times B*H = 2 -> 1280 (rectangle cells, not the 528 logical visible cells).
    assert rep["W_vec[cast]"].value == 1280


# --- R3 [60]/[62] P5/P7 ---------------------------------------------------------------------------------------
def test_resid_missing_fields_name_the_declaring_field_not_the_symbol():
    """R3 [60]/[62]: an undeclared kv_buffers is listed as 'kv_buffers' everywhere, never as 'n_buf' (and no other
    internal symbol leaks); the feasible-region rows agree with the metric (differential)."""
    rep = resid_bind(strategy={k: v for k, v in IMPL.items() if k != "kv_buffers"}).analyze()
    assert rep["b_k_max[vmem:score]"].missing_fields == ["kv_buffers"]
    leaks = {m.metric: m.missing_fields for m in rep.metrics if RESID_INTERNAL_SYMBOLS & set(m.missing_fields)}
    assert leaks == {}
    rows = rep.extras["feasible_region_by_stage"]["vmem"]["score"]
    assert rows and all(r["missing"] == rep["b_k_max[vmem:score]"].missing_fields for r in rows)


def test_resid_missing_budget_is_listed_once_by_its_field_name():
    """R3 [60]/[62]: an undeclared vmem budget appears once as 'vmem_budget_bytes', never as 'R_vmem'."""
    dropped = ("vmem_budget_bytes", "vreg_budget_bytes", "q_resident", "v_load")
    rep = resid_bind(strategy={k: v for k, v in IMPL.items() if k not in dropped}).analyze()
    missing = rep["b_k_max[vmem:pv]"].missing_fields
    assert missing.count("vmem_budget_bytes") == 1 and "R_vmem" not in missing


# --- R4 [55] P17 ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("value", ["abc", [1], True, float("nan")], ids=["str", "list", "bool", "nan"])
def test_resid_malformed_scale_value_fails_at_bind(value):
    """R4 [55]: a non-finite or non-numeric declared scale is a MetadataError at bind, not a raw error or a silent 1.0."""
    sem = {**SEM, "scale_policy": "declared", "scale_value": value}
    assert resid_model_error(lambda: resid_bind(semantics=sem)) is MetadataError


def test_resid_negative_scale_tolerance_fails_at_bind():
    """R4 [55]: a negative scale_match_rel_tol is rejected at bind."""
    assert resid_model_error(lambda: resid_bind(semantics={**SEM, "scale_match_rel_tol": -1})) is MetadataError


def test_resid_legal_declared_scale_is_reported():
    """R4 [55]: a legal declared scale binds and the gamma metric carries the declared value."""
    rep = resid_bind(semantics={**SEM, "scale_policy": "declared", "scale_value": 0.25}).analyze()
    assert rep["gamma"].value == 0.25


# --- R5 [49] P11 ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("value", [[], 0, ""], ids=["list", "zero", "str"])
def test_resid_falsy_malformed_byte_width_overrides_fail(value):
    """R5 [49]: a falsy non-mapping dtype_bytes_overrides is a MetadataError, not silently treated as 'none'."""
    assert resid_model_error(lambda: resid_bind(numerics={**NUMERICS, "dtype_bytes_overrides": value})) is MetadataError


def test_resid_empty_byte_width_overrides_equal_absent():
    """R5 [49]: an empty mapping is legal and yields the same metrics as omitting the key (differential)."""
    with_empty = resid_bind(numerics={**NUMERICS, "dtype_bytes_overrides": {}}).analyze()
    without = resid_bind().analyze()
    assert with_empty["M_in"].value == without["M_in"].value
    assert [m.to_dict() for m in with_empty.metrics] == [m.to_dict() for m in without.metrics]
    assert with_empty.warnings == without.warnings


# --- R6 [45] P7 -----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("tensors, strategy", [
    (None, {**IMPL, "both_qk_branches_live": False, "layout": {"score_rope_branch": "compact"}}),
    (make_metadata(1, 2, 32, 32, 4, 6, 8, 0, 8, pe=False), {**IMPL, "layout": {"score_rope_branch": "compact"}}),
], ids=["branches_not_live", "D_r_zero"])
def test_resid_layout_row_says_object_not_built_under_configuration(tensors, strategy):
    """R6 [45]: an object the variant knows but does not build here is worded as configuration-dependent."""
    m = resid_bind(tensors=tensors, strategy=strategy).analyze()["layout[score_rope_branch]"]
    assert m.status == "not_applicable"
    assert "does not build this object under the declared configuration" in m.scope
    assert m.assumptions[0].startswith("this variant does not build the object under the declared configuration")


def test_resid_layout_row_says_object_belongs_to_another_variant():
    """R6 [45]: an object id that only another variant builds is worded as 'no object with this id'."""
    m = resid_bind(strategy={**IMPL, "layout": {"q_tilde": "compact"}}).analyze()["layout[q_tilde]"]
    assert m.status == "not_applicable"
    assert "has no object with this id" in m.scope
    assert m.assumptions[0].startswith("no variant builds this object here; another variant does")


# --- R7 [39] P1 residual ----------------------------------------------------------------------------------------
def test_resid_active_length_assumption_names_the_extent_actually_used():
    """R7 [39]: when the scheduled extent stands in for an undeclared active length, the assumption and the
    warning say 'scheduled extent', not 'tensor extent'; without it they say 'tensor extent'."""
    tensors = make_metadata(3, 2, 1, 64, 4, 6, 8, 4, 8)
    sem = {**SEM, "active_lengths": {"Sq": 1}, "position_offset": 19}
    base = {k: v for k, v in IMPL.items() if k != "scheduled_lengths"}
    base.update(b_q=1, b_k=8)

    rep = resid_bind(tensors, sem, {**base, "scheduled_lengths": {"Sk": 16}}).analyze()
    m = rep["S_k_active"]
    assert m.value == 16                                         # the declared scheduled extent
    assert any("the declared scheduled extent" in a for a in m.assumptions)
    assert not any("the tensor extent" in a for a in m.assumptions)
    lines = [w for w in rep.warnings if w.startswith("active lengths undeclared for")]
    assert len(lines) == 1 and "Sk: the declared scheduled extent" in lines[0]

    ctrl = resid_bind(tensors, sem, base).analyze()
    assert ctrl["S_k_active"].value == 64                        # the tensor extent Sk
    assert any("the tensor extent" in a for a in ctrl["S_k_active"].assumptions)
    ctrl_lines = [w for w in ctrl.warnings if w.startswith("active lengths undeclared for")]
    assert len(ctrl_lines) == 1 and "Sk: the tensor extent" in ctrl_lines[0]


# --- R8 [4] PERF-5 layout class -----------------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", ["int8", None])
def test_resid_layout_bucket_with_dtype_is_applied_with_named_caveat(dtype):
    """R8 [4]: a layout bucket declaring a dtype is applied (calibrated_partial) with the unverified dtype match
    named once in missing_fields; a bucket with no dtype carries no such entry."""
    hw = resid_hw(calibration=[{"node_kind": "layout", "dtype": dtype, "epsilon_low": 0.5, "epsilon_high": 0.9,
                                "source": "t"}])
    hw["compute"]["vpu"]["layout_throughput"] = {"value": 1e12, "unit": "op/s", "scope": "per_chip"}
    cp = resid_bind(hardware=hw).analyze().extras["calibrated_prediction"]
    assert cp["status"] == "calibrated_partial"
    caveats = [f for f in cp["missing_fields"] if f.startswith("calibration bucket dtype match[layout]")]
    if dtype is None:
        assert caveats == []
    else:
        assert len(caveats) == 1 and "int8" in caveats[0]
    assert any("layout buckets are matched on kind only" in n for n in cp["notes"])


# --- R9 [2] PERF-3 ---------------------------------------------------------------------------------------------
RESID_BUCKET = {"node_kind": "matmul", "dtype": "bfloat16", "epsilon_low": 0.35, "epsilon_high": 0.7, "source": "t"}


@pytest.mark.parametrize("profile", [
    {**HW, "toolchain": "llvm"},
    {**HW, "calibration": [{**RESID_BUCKET, "m_range": 5}]},
    {**HW, "calibration": [{**RESID_BUCKET, "n_range": 5}]},
    {**HW, "calibration": [{**RESID_BUCKET, "k_range": "ab"}]},
], ids=["toolchain_str", "m_range_int", "n_range_int", "k_range_str"])
def test_resid_malformed_profile_pieces_fail_explicitly_at_bind(profile):
    """R9 [2]: a non-mapping toolchain or a non-pair shape range is a MetadataError at bind, not a raw TypeError."""
    assert resid_model_error(lambda: resid_bind(hardware=profile)) is MetadataError


def test_resid_unbounded_shape_range_is_legal():
    """R9 [2]: [null, null] is the legal 'any size' range and binds."""
    resid_bind(hardware={**HW, "calibration": [{**RESID_BUCKET, "m_range": [None, None]}]})


# --- R10 [11] PERF-12 ------------------------------------------------------------------------------------------
RESID_HBM_BUCKET = {"node_kind": "hbm_transfer", "epsilon_low": 0.5, "epsilon_high": 0.9, "source": "t"}
RESID_MATMUL_BUCKET = {**RESID_BUCKET, "startup_s": 2e-7}


@pytest.mark.parametrize("buckets, status", [
    ([RESID_HBM_BUCKET], "not_calibrated"),
    ([RESID_HBM_BUCKET, RESID_MATMUL_BUCKET], "calibrated_partial"),
], ids=["only_bucket", "with_matmul_bucket"])
def test_resid_declared_inputs_with_no_consumer_are_named(buckets, status):
    """R10 [11]: a serial group, a path bandwidth and a calibration bucket that nothing consumes are each named."""
    strategy = {**IMPL, "overlap_model": "serial_stage_sum", "serial_stage_groups": [["path:vreg_to_vmem"], ["mxu"]]}
    hw = resid_hw(calibration=buckets)
    hw["paths"]["vreg_to_vmem"] = {"bandwidth": {"value": 4e12, "unit": "byte/s", "scope": "per_chip"}}
    rep = resid_bind(strategy=strategy, hardware=hw).analyze()
    rlb = rep.extras["resource_lower_bounds"]["notes"]
    assert len([n for n in rlb if "path:vreg_to_vmem" in n and "no work and no transfer events" in n]) == 1
    assert len([n for n in rlb if "vreg_to_vmem" in n and "has no consumer" in n]) == 1
    cp = rep.extras["calibrated_prediction"]
    assert cp["status"] == status
    assert any("calibration bucket kind(s) ['hbm_transfer'] have no consumer" in n for n in cp["notes"])


def test_resid_no_consumer_notes_absent_without_the_declarations():
    """R10 [11]: control -- without the group, bandwidth and bucket none of the no-consumer notes appear."""
    rep = resid_bind(hardware=HW).analyze()
    assert not any("no work and no transfer events" in n or "has no consumer" in n
                   for n in rep.extras["resource_lower_bounds"]["notes"])
    assert not any("have no consumer" in n for n in rep.extras["calibrated_prediction"]["notes"])


# --- R11 [47] P9 task section ------------------------------------------------------------------------------------
def test_resid_task_extent_metrics_name_their_own_axis_only():
    """R11 [47]: with only Sq declared active and no scheduled lengths, each extent metric lists exactly the
    declarations of its own axis, and N_Qproj (which depends on S_q only) is bound."""
    rep = resid_bind(semantics={**SEM, "active_lengths": {"Sq": 6}},
                     strategy={k: v for k, v in IMPL.items() if k != "scheduled_lengths"}).analyze()
    assert rep["S_q_active"].status == "bound" and rep["S_q_active"].missing_fields == []
    assert rep["S_k_active"].status == "partial" and rep["S_k_active"].missing_fields == ["active_lengths.Sk"]
    assert rep["S_q_scheduled"].missing_fields == ["scheduled_lengths.Sq"]
    assert rep["S_k_scheduled"].missing_fields == ["scheduled_lengths.Sk", "active_lengths.Sk"]
    # N_Qproj counts Q-projection rows B*H*S_q = 1*2*6 = 12 (md(): B=1, H=2; declared S_q = 6).
    assert rep["N_Qproj"].status == "bound" and rep["N_Qproj"].missing_fields == []
    assert rep["N_Qproj"].value == 12
