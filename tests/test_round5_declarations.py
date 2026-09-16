"""Regressions for the fifth review round (verification of the fourth round's fixes): declarations.

Each test names the problem it protects against.  Expected values are written here from the specification or
an independent hand computation, never read back from the package; a test that compares two surfaces of one
report against each other says so (differential) in its docstring.
"""
import json

import pytest

from conftest import NUMERICS, make_metadata
from test_verify_round4 import SEM, IMPL, md
from mla_model import MLAForwardModel
from mla_model.errors import BindingError, MetadataError

DECL_MODEL = MLAForwardModel()
DECL_SEM_NO_ACTIVE = {k: v for k, v in SEM.items() if k != "active_lengths"}
DECL_IMPL_NO_SCHED = {k: v for k, v in IMPL.items() if k != "scheduled_lengths"}
DECL_VARIANTS = ("expanded", "absorbed_two_step", "absorbed_precomputed")
# the materialization flags each variant reads, all declared, so a path total has no undeclared event left
DECL_MAT_DECLARED = {"expanded": {"expanded_kv": True, "q_nope": False},
                     "absorbed_two_step": {"q_tilde": False, "z": False},
                     "absorbed_precomputed": {"q_tilde": False, "z": False}}


def decl_bind(tensors=None, sem=SEM, impl=IMPL, num=NUMERICS, algorithm="expanded", **kw):
    return DECL_MODEL.bind(tensors if tensors is not None else md(), sem, impl, num, algorithm=algorithm, **kw)


# ----------------------------------------------------------------------------------------- bytes ledgers
def test_decl_m_in_expression_and_bindings_reproduce_its_value():
    """[12] P1: M_in's retained expression substituted at its own bindings must equal its value (spec 11.4).

    Hand computation (md(): B=1 H=2 Sq=Sk=32 Rq=4 Rk=6 Dn=8 Dr=4 Dv=8, bf16 = 2 bytes), each role at its OWN
    allocation extent: q_latent cap [1,64,4] -> 64*4*2 = 512; kv_latent cap [1,96,6] -> 96*6*2 = 1152;
    q_pe (no capacity) 1*2*32*4*2 = 512; k_pe 1*32*4*2 = 256; w_q_nope 2*4*8*2 = 128; w_k_nope 2*6*8*2 = 192;
    w_v 2*6*8*2 = 192; total 2944.  The pre-fix expression evaluated q_pe/k_pe at the GLOBAL capacities
    (64 / 96) and gave 3968 next to a value of 2944.
    """
    import sympy as sp
    t = md()
    t["q_latent"] = {**t["q_latent"], "capacity_shape": [1, 64, 4]}
    t["kv_latent"] = {**t["kv_latent"], "capacity_shape": [1, 96, 6]}
    rep = decl_bind(t).analyze()
    m = rep["M_in"]
    assert m.status == "bound" and m.value == 2944
    at_bindings = sp.sympify(m.expression).subs({sp.Symbol(k): v for k, v in m.bindings.items()})
    assert int(at_bindings) == m.value
    per_role = sum(rep.value(f"bytes[{r}]") for r in ("q_latent", "kv_latent", "q_pe", "k_pe", "w_q_nope", "w_k_nope", "w_v"))
    assert per_role == 2944 == m.value
    mj = next(x for x in json.loads(rep.to_json())["metrics"] if x["metric"] == "M_in")      # JSON surface agrees
    assert mj["value"] == 2944 and int(sp.sympify(mj["expression"]).subs({sp.Symbol(k): v for k, v in mj["bindings"].items()})) == 2944


def test_decl_contiguous_strides_on_a_zero_width_tensor_stay_bound():
    """[13] P2: a zero-element tensor (D_r = 0) spans no address range, so its contiguous strides cannot make the
    allocation ledger incomplete.  Hand total: q_latent 32*4*2=256 + kv_latent 32*6*2=384 + q_pe 0 + k_pe 0
    + w_q_nope 128 + w_k_nope 192 + w_v 192 = 1152."""
    t = md(Dr=0)
    t["q_pe"] = {**t["q_pe"], "strides": [0, 0, 0, 1]}
    t["k_pe"] = {**t["k_pe"], "strides": [0, 0, 1]}
    rep = decl_bind(t).analyze()
    m = rep["M_alloc_unique[metadata_ledger]"]
    assert m.status == "bound" and m.value == 1152 and m.missing_fields == []
    assert rep.extras["interface_ledger"]["roles_with_unmodelled_strides"] == []
    assert not any("not modelled" in w and "strides" in w for w in rep.warnings)


def test_decl_whole_tensor_materialization_traffic_does_not_depend_on_the_grid():
    """[14] P3: Q~ / Q^n round trips are one write + one read of the whole tensor, so an undeclared scheduled
    extent cannot change them and must not be named as missing; the expanded K^n|V read IS per executed
    rectangle and keeps the dependence.  Hand values (md(), bf16): HBM_mat[Q~] = 2*B*H*R_k*S_q*s = 2*1*2*6*32*2
    = 1536; HBM_mat[Q^n] = 2*B*H*D_n*S_q*s = 2*1*2*8*32*2 = 2048."""
    rep = decl_bind(impl={**DECL_IMPL_NO_SCHED, "materialize": {"q_tilde": True, "z": True}}, algorithm="absorbed_two_step").analyze()
    assert rep["HBM_mat[Q~]"].status == "bound" and rep.value("HBM_mat[Q~]") == 1536
    assert "scheduled_lengths" not in rep["HBM_mat[Q~]"].missing_fields
    rep = decl_bind(impl={**DECL_IMPL_NO_SCHED, "materialize": {"expanded_kv": True, "q_nope": True}}).analyze()
    assert rep["HBM_mat[Q^n]"].status == "bound" and rep.value("HBM_mat[Q^n]") == 2048
    assert rep["HBM_mat[K^n,V]"].status == "partial" and "scheduled_lengths" in rep["HBM_mat[K^n,V]"].missing_fields


def test_decl_disabled_materialization_rows_name_no_missing_extent():
    """[15] P4: a materialization the strategy declares off reports 0 / not_applicable and cannot depend on the
    active or scheduled extents, so its missing_fields are empty even when both are undeclared."""
    rep = decl_bind(sem=DECL_SEM_NO_ACTIVE, impl={**DECL_IMPL_NO_SCHED, "materialize": {"expanded_kv": False, "q_nope": False}}).analyze()
    for name in ("HBM_mat[K^n,V]", "M_mat[K^n,V]", "HBM_mat[Q^n]", "M_mat[Q^n]"):
        m = rep[name]
        assert m.status == "not_applicable" and m.value == 0 and m.missing_fields == [], name
        assert not any("scenario" in a for a in m.assumptions), name


@pytest.mark.parametrize("patch", [
    {"strides": ["a", 1, 1]}, {"strides": [50.5, 5, 1]}, {"strides": [True, 1, 1]},
    {"capacity_shape": "abc"}, {"capacity_shape": [2, None, 5]}, {"capacity_shape": [1, 10.7, 4]}, {"capacity_shape": ["a", 8, 4]},
    {"storage_offset": "x"}, {"storage_offset": True}, {"storage_offset": -1},
])
def test_decl_malformed_stride_capacity_offset_metadata_is_a_metadata_error(patch):
    """[16] P5 / [55] P17 (metadata half): strides, capacity_shape and storage_offset entries are validated like
    shape entries — a malformed value is a MetadataError, never a raw ValueError/TypeError, and a non-integer
    is refused instead of being truncated to a smaller (or no) allocation."""
    t = md()
    t["q_latent"] = {**t["q_latent"], **patch}
    with pytest.raises(MetadataError):
        decl_bind(t)


def test_decl_path_total_names_only_declarable_fields_for_an_undeclared_materialization():
    """[17] P6: one undeclared flag appears once, under its field name; neither the condition sentence
    ('... is not True') nor the 'transfer conditions' placeholder is a field a caller could declare."""
    rep = decl_bind(impl={**IMPL, "materialize": {}}).analyze()
    m = rep["B[hbm_to_vmem]"]
    assert m.status == "partial"
    assert sorted(m.missing_fields) == ["materialize.expanded_kv", "materialize.q_nope"]
    assert all(" " not in f for f in m.missing_fields)


def test_decl_absent_positional_roles_are_described_as_absent():
    """[18] P7: with q_pe / k_pe absent (D_r = 0) there is no tensor whose 'own shape' could be the allocation
    extent; the row says the role is absent and reports 0 bytes."""
    rep = decl_bind(make_metadata(1, 2, 32, 32, 4, 6, 8, 4, 8, pe=False)).analyze()
    assert rep.value("D_r") == 0
    for role in ("q_pe", "k_pe"):
        m = rep[f"bytes[{role}]"]
        assert m.status == "bound" and m.value == 0
        assert any("absent" in a for a in m.assumptions), m.assumptions
        assert not any("own shape" in a for a in m.assumptions), m.assumptions


@pytest.mark.parametrize("alg", DECL_VARIANTS)
def test_decl_ragged_tasks_can_reach_a_bound_path_total(alg):
    """[19] P8 and [51] P13: a per-batch task has no declarable scheduled extent (bind() refuses one, the
    documented modelling limit), so no metric may name scheduled_lengths as missing, the fully declared path
    total reaches bound, and S_q/S_k_scheduled are not_applicable with the ragged note rather than a
    value-less 'partial'."""
    sem = {**SEM, "active_lengths": {"Sq": [8, 5, 3], "Sk": [12, 7, 4]}}
    rep = decl_bind(md(B=3), sem=sem, impl={**DECL_IMPL_NO_SCHED, "materialize": DECL_MAT_DECLARED[alg]}, algorithm=alg).analyze()
    assert rep["B[hbm_to_vmem]"].status == "bound"
    offenders = [m.metric for m in rep.metrics if any(f.startswith("scheduled_lengths") for f in m.missing_fields)]
    assert offenders == []
    for key in ("S_q_scheduled", "S_k_scheduled"):
        m = rep[key]
        assert m.status == "not_applicable" and m.value is None and m.missing_fields == []
        assert any("ragged" in a or "per-batch" in a for a in m.assumptions), m.assumptions
    assert rep["S_q_active"].status == "bound" and rep.value("S_q_active") == [8, 5, 3]
    with pytest.raises(BindingError):                                      # the limit itself is still explicit
        decl_bind(md(B=3), sem=sem, impl=IMPL, algorithm=alg)


# -------------------------------------------------------------------------------- extents and identity
@pytest.mark.parametrize("alg", DECL_VARIANTS)
def test_decl_scheduled_extent_bounds_the_scenario_on_an_undeclared_active_axis(alg):
    """[39] P1: with active_lengths.Sk undeclared and scheduled_lengths.Sk = 16 the capacity-execution scenario
    uses active = scheduled (spec 1.3: active <= scheduled <= capacity), names active_lengths.Sk as missing and
    the report can no longer print useful work above the executed rectangle work.
    Hand: B=3 H=2 Sq=1 Sk_active=16 Delta=19 causal -> row 0 sees keys j <= 19, i.e. all 16: C_valid = 6*16 = 96;
    b_q=1 b_k=8 skip_future on the 1 x 16 grid keeps both key blocks (min key 8 <= 0 + 19): C_rect = 6*16 = 96.
    Pre-fix: S_k_active = 64, C_valid = 120 > C_rect = 96, rect_waste = -24.
    Not asserted here: the S_k_active metric's own assumption sentence still reads 'the tensor extent is used'
    while its value is the scheduled 16 (residual wording defect, reported as unfixed); the test pins the value,
    the header, the warning and the JSON surface, and checks the value is NOT the tensor extent."""
    sem = {**SEM, "position_offset": 19, "active_lengths": {"Sq": 1}}
    impl = {**DECL_IMPL_NO_SCHED, "b_q": 1, "b_k": 8, "scheduled_lengths": {"Sk": 16}}
    rep = decl_bind(make_metadata(3, 2, 1, 64, 4, 6, 8, 4, 8), sem=sem, impl=impl, algorithm=alg).analyze()
    assert rep["S_k_active"].status == "partial" and rep.value("S_k_active") == 16
    assert rep.value("S_k_active") != 64                                    # not the tensor (capacity) extent
    assert rep["S_k_active"].missing_fields == ["active_lengths.Sk"]
    assert rep.value("S_k_scheduled") == 16 and rep.header["active_extents"]["Sk"] == 16
    assert rep.value("C_valid") == 96 and rep.value("C_rect") == 96
    assert rep.value("rect_waste") == 0 and rep.value("C_valid") <= rep.value("C_rect")
    assert any("scheduled_lengths.Sk=16" in w and "scheduled extent as the active extent" in w for w in rep.warnings)
    js = json.loads(rep.to_json())                                           # JSON surface agrees with the metric
    assert next(x for x in js["metrics"] if x["metric"] == "S_k_active")["value"] == 16
    assert next(x for x in js["metrics"] if x["metric"] == "C_valid")["value"] == 96


def test_decl_dense_mask_scenario_respects_the_scheduled_extents():
    """[39] P1 (dense case): active undeclared, scheduled {Sq:1, Sk:32} on a 3 x 64 allocation.
    Hand (md(Sq=3, Sk=64): B=1 H=2): C_valid = B*H*1*32 = 2*32 = 64 = C_rect (every cell of the 1 x 32 grid is
    visible; b_q=1 b_k=8 -> 1 x 4 blocks of 1*8 = 32 cells per (b,h)), waste 0.
    Pre-fix: C_valid = 384 (1*2*3*64, the tensor extents) against C_rect = 64 (1*2*1*32), waste -320."""
    sem = {**{k: v for k, v in DECL_SEM_NO_ACTIVE.items() if k != "position_offset"}, "mask": "none"}
    impl = {**IMPL, "b_q": 1, "b_k": 8, "scheduled_lengths": {"Sq": 1, "Sk": 32}}
    rep = decl_bind(md(Sq=3, Sk=64), sem=sem, impl=impl).analyze()
    assert rep.value("S_q_active") == 1 and rep.value("S_k_active") == 32
    assert rep.value("C_valid") == 64 and rep.value("C_rect") == 64 and rep.value("rect_waste") == 0


@pytest.mark.parametrize("alias", [["kv_latent"], 3, {"role": "kv_latent"}])
def test_decl_non_string_alias_of_is_a_metadata_error(alias):
    """[40] P2: alias_of names a role, so a non-string value is refused as MetadataError instead of escaping as
    an unhashable-type TypeError from the dangling-alias check."""
    t = md()
    t["k_pe"] = {**t["k_pe"], "alias_of": alias}
    with pytest.raises(MetadataError):
        decl_bind(t)


@pytest.mark.parametrize("qs,ks", [("abc", 0), (5, "x"), (1.5, 0), (True, 0)])
def test_decl_position_start_fields_are_type_checked(qs, ks):
    """[41] P3: query_position_start / key_position_start derive Delta like position_offset does, so a malformed
    value is a BindingError, and 1.5 or True is refused rather than silently coerced to 1."""
    sem = {**{k: v for k, v in SEM.items() if k != "position_offset"}, "query_position_start": qs, "key_position_start": ks}
    with pytest.raises(BindingError):
        decl_bind(sem=sem)


def test_decl_position_starts_derive_the_offset():
    """[41] P3 (positive companion): Delta = query_position_start - key_position_start = 5 - 2 = 3 (spec 1.4).
    Passes pre-fix: guards the fix's scope (the type check must not break well-formed integer starts), not the
    regression itself, which test_decl_position_start_fields_are_type_checked pins."""
    sem = {**{k: v for k, v in SEM.items() if k != "position_offset"}, "query_position_start": 5, "key_position_start": 2}
    assert decl_bind(sem=sem).analyze().value("Delta") == 3


@pytest.mark.parametrize("pbo", [5, "ab", 0.5])
def test_decl_scalar_per_batch_offsets_is_a_binding_error(pbo):
    """[42] P4: per_batch_offsets must be a list of B integers; a scalar is refused before len() is taken."""
    sem = {**{k: v for k, v in SEM.items() if k != "position_offset"}, "per_batch_offsets": pbo}
    with pytest.raises(BindingError):
        decl_bind(sem=sem)


@pytest.mark.parametrize("alg,key", [("expanded", "q_tilde"), ("expanded", "z"), ("absorbed_two_step", "expanded_kv"),
                                     ("absorbed_two_step", "q_nope"), ("absorbed_precomputed", "expanded_kv")])
def test_decl_materialize_flag_of_another_variant_leaves_a_trace(alg, key):
    """[43] P5: a materialize flag for an intermediate this variant does not build is a legal declaration, and
    the report says it had no effect (design as implemented: a graph assumption on the executed-graph work
    metric, visible in the JSON and markdown surfaces) instead of ignoring it silently."""
    mat = {**DECL_MAT_DECLARED[alg], key: True}
    rep = decl_bind(impl={**IMPL, "materialize": mat}, algorithm=alg).analyze()
    trace = [a for m in rep.metrics for a in m.assumptions if f"materialize.{key}" in a and "no effect" in a]
    assert trace, f"no trace of the ignored materialize.{key}"
    assert rep.header["implementation"]["materialize"][key] is True
    assert f"materialize.{key}" in rep.to_json() and "no effect" in rep.to_markdown()


@pytest.mark.parametrize("alg,mat,expect", [
    ("expanded", {"expanded_kv": True, "q_nope": False}, "bound"),
    ("expanded", {"expanded_kv": False, "q_nope": False}, "not_applicable"),
    ("expanded", {}, "not_applicable"),
    ("absorbed_two_step", {}, "not_applicable"),
    ("absorbed_precomputed", {}, "not_applicable"),
])
def test_decl_layout_for_an_object_that_exists_only_under_a_declaration_is_legal(alg, mat, expect):
    """[44] P6: kv_proj_out_tile exists only when the expanded variant materializes K/V; a layout for it is a
    legal declaration everywhere — applied where the object is built, not_applicable elsewhere, never a
    'no variant contains' MetadataError."""
    rep = decl_bind(impl={**IMPL, "materialize": mat, "layout": {"kv_proj_out_tile": "compact"}}, algorithm=alg).analyze()
    m = rep["layout[kv_proj_out_tile]"]
    assert m.status == expect
    assert (m.value is not None) == (expect == "bound")


def test_decl_layout_for_an_id_no_variant_has_is_still_refused():
    """[44] P6 (guard): widening the known-id set must not accept an id that no variant ever builds.
    Passes pre-fix: guards the fix's scope, not the regression, which
    test_decl_layout_for_an_object_that_exists_only_under_a_declaration_is_legal pins."""
    with pytest.raises(MetadataError):
        decl_bind(impl={**IMPL, "layout": {"no_such_object": "compact"}}).analyze()


@pytest.mark.parametrize("alg", ("expanded", "absorbed_two_step"))
@pytest.mark.parametrize("tensors,live", [(md(Dr=0), True), (md(Dr=0), False), (md(), False),
                                          (make_metadata(1, 2, 32, 32, 4, 6, 8, 4, 8, pe=False), True)])
def test_decl_layout_for_an_object_declared_away_is_reported_not_dropped(alg, tensors, live):
    """[45] P7: score_rope_branch declared away (both_qk_branches_live False, or D_r = 0 with either setting)
    yields the same not_applicable layout row that an id of another variant does; the declaration is never
    dropped from the report."""
    rep = decl_bind(tensors, impl={**IMPL, "both_qk_branches_live": live, "layout": {"score_rope_branch": "compact"}}, algorithm=alg).analyze()
    m = rep.get("layout[score_rope_branch]")
    assert m is not None and m.status == "not_applicable" and m.value is None
    assert "layout[score_rope_branch]" in rep.to_markdown()


def test_decl_layout_applies_when_the_object_is_built():
    """[45] P7 (positive companion): with D_r > 0 and both branches live the same declaration is applied (bound).
    Passes pre-fix: guards the fix's scope (a not_applicable row must not replace a legitimately applied one), not
    the regression, which test_decl_layout_for_an_object_declared_away_is_reported_not_dropped pins."""
    rep = decl_bind(impl={**IMPL, "both_qk_branches_live": True, "layout": {"score_rope_branch": "compact"}}).analyze()
    assert rep["layout[score_rope_branch]"].status == "bound"


def test_decl_bind_dtype_overrides_and_numerics_overrides_are_one_declaration():
    """[46] P8: bind(dtype_overrides=...) is normalised and validated through the numeric policy, so the alias
    spelling 'bf16' widens every bfloat16 symbol exactly like numerics.dtype_bytes_overrides.
    Hand: md() has 128+192+256+128+64+96+96 = 960 bf16 elements -> M_in = 960*3 = 2880 (differential between
    the two entry points as well)."""
    via_numerics = decl_bind(num={**NUMERICS, "dtype_bytes_overrides": {"bf16": 3}}).analyze()
    via_kwarg = decl_bind(dtype_overrides={"bf16": 3}).analyze()
    for rep in (via_numerics, via_kwarg):
        assert rep.value("M_in") == 2880
        assert rep.header["bindings"]["s_Cq"] == 3 and rep.header["bindings"]["s_P"] == 3
        assert rep.header["numerics"]["dtype_bytes_overrides"] == {"bfloat16": 3}
    assert via_numerics.value("M_in") == via_kwarg.value("M_in")
    for bad in ({"bf16": 0}, {"bf16": "3"}, {"bf16": 2.5}):
        with pytest.raises(MetadataError):
            decl_bind(dtype_overrides=bad)
    # a width only the override supplies is honoured from either entry point (no UnsupportedDtypeError)
    a = decl_bind(num={**NUMERICS, "score_dtype": "fp6"}, dtype_overrides={"fp6": 1}).analyze()
    b = decl_bind(num={**NUMERICS, "score_dtype": "fp6", "dtype_bytes_overrides": {"fp6": 1}}).analyze()
    assert a.header["bindings"]["s_X"] == 1 == b.header["bindings"]["s_X"]


@pytest.mark.parametrize("alg", DECL_VARIANTS + (None,))
def test_decl_missing_active_axis_is_attributed_per_axis(alg):
    """[47] P9: with active_lengths {Sq: 6} only, quantities that involve S_q alone are bound and quantities
    that involve S_k carry active_lengths.Sk.  Hand (md(), bf16): N_Qproj = B*H*S_q = 1*2*6 = 12;
    M_O = s_O*B*H*S_q*D_v = 2*1*2*6*8 = 192."""
    rep = decl_bind(sem={**SEM, "active_lengths": {"Sq": 6}}, algorithm=alg).analyze()
    assert rep["S_q_active"].status == "bound" and rep.value("S_q_active") == 6 and rep["S_q_active"].missing_fields == []
    assert rep["N_Qproj"].status == "bound" and rep.value("N_Qproj") == 12
    assert rep["M_O"].status == "bound" and rep.value("M_O") == 192
    assert rep["S_q_scheduled"].status == "bound" and rep.value("S_q_scheduled") == 32
    assert rep["S_k_active"].status == "partial" and rep["S_k_active"].missing_fields == ["active_lengths.Sk"]
    assert rep["C_valid"].status == "partial" and rep["C_valid"].missing_fields == ["active_lengths.Sk"]
    if alg == "expanded":
        assert rep["N_Kproj"].status == "partial" and rep["N_Kproj"].missing_fields == ["active_lengths.Sk"]


def test_decl_null_scheduled_axis_is_undeclared_like_a_null_active_axis():
    """[48] P10: scheduled_lengths {Sq: 32, Sk: null} is accepted and Sk counts as undeclared (the rule
    active_lengths follows); differential against omitting the key entirely."""
    explicit = decl_bind(impl={**IMPL, "scheduled_lengths": json.loads('{"Sq": 32, "Sk": null}')}).analyze()
    omitted = decl_bind(impl={**IMPL, "scheduled_lengths": {"Sq": 32}}).analyze()
    for rep in (explicit, omitted):
        assert rep["S_q_scheduled"].status == "bound" and rep.value("S_q_scheduled") == 32
        assert rep["S_k_scheduled"].status == "partial" and rep["S_k_scheduled"].missing_fields == ["scheduled_lengths.Sk"]
        assert rep.header["implementation"]["scheduled_lengths"] == {"Sq": 32}
    assert explicit.value("C_rect") == omitted.value("C_rect")
    both_null = decl_bind(impl={**IMPL, "scheduled_lengths": {"Sq": None, "Sk": None}}).analyze()
    assert both_null.header["implementation"]["scheduled_lengths"] is None
    assert both_null["S_q_scheduled"].missing_fields == ["scheduled_lengths.Sq"]


@pytest.mark.parametrize("al", [[], 0, False, "Sq"])
def test_decl_falsy_malformed_active_lengths_are_refused(al):
    """[49] P11 (active_lengths half): [] / 0 / False are malformed declarations, not 'undeclared'."""
    with pytest.raises(BindingError):
        decl_bind(sem={**SEM, "active_lengths": al})


@pytest.mark.parametrize("sl", [[], 0, False])
def test_decl_falsy_malformed_scheduled_lengths_are_refused(sl):
    """[49] P11 (scheduled_lengths half): the same rule for the strategy's per-axis declaration."""
    with pytest.raises(MetadataError):
        decl_bind(impl={**IMPL, "scheduled_lengths": sl})


def test_decl_fingerprint_and_header_agree_on_declared_axes():
    """[50] P12: identical tasks hash identically whether active_lengths is absent, {} or all-null, and a
    per-axis declaration hashes the same with or without a null sibling; differential against the header's
    declared_axes and the JSON task_fingerprint."""
    def run(al):
        sem = dict(DECL_SEM_NO_ACTIVE)
        if al is not None:
            sem["active_lengths"] = al
        bound = decl_bind(sem=sem, impl=DECL_IMPL_NO_SCHED)
        rep = bound.analyze()
        return bound.fingerprint(), rep.header["active_extents"], json.loads(rep.to_json())["task_fingerprint"]
    undeclared = [run(al) for al in (None, {}, {"Sq": None, "Sk": None})]
    assert len({fp for fp, _, _ in undeclared}) == 1
    assert all(h["declared"] is False and h["declared_axes"] == [] for _, h, _ in undeclared)
    partial = [run(al) for al in ({"Sq": 6}, {"Sq": 6, "Sk": None})]
    assert len({fp for fp, _, _ in partial}) == 1
    assert all(h["declared"] is False and h["declared_axes"] == ["Sq"] for _, h, _ in partial)
    full_fp, full_h, _ = run({"Sq": 32, "Sk": 32})
    assert full_h["declared"] is True and len({undeclared[0][0], partial[0][0], full_fp}) == 3
    assert all(fp == js for fp, _, js in undeclared + partial)


def test_decl_alias_chains_collapse_to_one_storage_group():
    """[52] P14: q_pe -> k_pe -> kv_latent is one allocation rooted at kv_latent, the same as the star
    q_pe -> kv_latent, k_pe -> kv_latent (differential); cycles and self-aliases are refused.
    Hand (md(), bf16): logical total 1920; the group {kv_latent 384, q_pe 512, k_pe 256} is charged its largest
    member 512, so allocated_unique_total = 256 + 512 + 128 + 192 + 192 = 1280."""
    def ledger(t):
        rep = decl_bind(t).analyze()
        L = rep.extras["interface_ledger"]
        return L, rep.value("M_alloc_unique[metadata_ledger]")
    chain = md(); chain["q_pe"] = {**chain["q_pe"], "alias_of": "k_pe"}; chain["k_pe"] = {**chain["k_pe"], "alias_of": "kv_latent"}
    star = md(); star["q_pe"] = {**star["q_pe"], "alias_of": "kv_latent"}; star["k_pe"] = {**star["k_pe"], "alias_of": "kv_latent"}
    Lc, mc = ledger(chain)
    Ls, ms = ledger(star)
    assert mc == 1280 == Lc["allocated_unique_total"] and Lc["logical_total"] == 1920
    assert ms == mc and Ls["allocated_unique_total"] == Lc["allocated_unique_total"]
    groups = {r["role"]: r["alias_group"] for r in Lc["tensors"]}
    assert groups["q_pe"] == "kv_latent" == groups["k_pe"]
    assert sum(r["counted_in_allocation"] for r in Lc["tensors"] if r["alias_group"] == "kv_latent") == 1
    cycle = md(); cycle["q_pe"] = {**cycle["q_pe"], "alias_of": "k_pe"}; cycle["k_pe"] = {**cycle["k_pe"], "alias_of": "q_pe"}
    with pytest.raises(MetadataError):
        decl_bind(cycle)
    selfy = md(); selfy["kv_latent"] = {**selfy["kv_latent"], "alias_of": "kv_latent"}
    with pytest.raises(MetadataError):
        decl_bind(selfy)


def test_decl_subdivide_without_its_policy_is_noted():
    """[53] P15: subdivide declared under rect_policy 'skip_future' has no effect and the report says so.
    Hand C_rect (md(), b_q=b_k=8, causal Delta=0, skip_future): 10 of the 16 blocks per (b,h) are kept,
    10*64*B*H = 640*2 = 1280, identical with and without the sub-tiles (differential)."""
    with_sub = decl_bind(impl={**IMPL, "rect_policy": "skip_future", "subdivide": [2, 2]}).analyze()
    without = decl_bind(impl={**IMPL, "rect_policy": "skip_future"}).analyze()
    assert with_sub.value("C_rect") == 1280 == without.value("C_rect")
    assert any("subdivide" in w and "no effect" in w for w in with_sub.warnings)
    assert not any("subdivide" in w for w in without.warnings)
