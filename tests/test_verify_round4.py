"""Regressions for the problems the fix-verification sweep found.

Each test names the property it protects.  Expected values are written here from the specification or from
an independent hand computation, not read back from the package.
"""
import json

import pytest

from conftest import NUMERICS, make_metadata
from mla_model import MLAForwardModel, is_absent
from mla_model.errors import BindingError, MetadataError, TaskIdentityError

SEM = {"mask": "causal", "position_offset": 0, "scale_policy": "standard", "outputs": ["O"],
       "projection_scope": "full", "empty_row_policy": "zero_output_neg_inf_lse",
       "active_lengths": {"Sq": 32, "Sk": 32}}
IMPL = {"name": "base", "b_q": 8, "b_k": 8, "rect_policy": "skip_future", "executed_extent_policy": "logical",
        "kv_buffers": 2, "heads_per_program": 1, "score_alias_exp": True, "exp_alias_p_operand": False,
        "both_qk_branches_live": False, "materialize": {"expanded_kv": True, "q_nope": False}, "scratch_bytes": 0,
        "overlap_model": "full_overlap_max", "projection_in_kv_loop": False, "schedule": "q_outer_kv_inner",
        "vreg_budget_bytes": 65536, "vmem_budget_bytes": 1 << 22, "b_rq": 4, "b_rk": 4,
        "q_resident": True, "v_load": "early", "scheduled_lengths": {"Sq": 32, "Sk": 32}}
HW = {"name": "p", "scope_default": "per_chip",
      "compute": {"mxu": {"peak_flops": {"bfloat16": {"value": 1e14, "unit": "FLOP/s", "scope": "per_chip"}}},
                  "vpu": {"exp_throughput": {"value": 2e11, "unit": "op/s", "scope": "per_chip"},
                          "elementwise_throughput": {"value": 1e12, "unit": "op/s", "scope": "per_chip"},
                          "reduction_throughput": {"value": 5e11, "unit": "op/s", "scope": "per_chip"}}},
      "paths": {"hbm_to_vmem": {"bandwidth": {"value": 8e11, "unit": "byte/s", "scope": "per_chip"}},
                "vmem_to_hbm": {"bandwidth": {"value": 8e11, "unit": "byte/s", "scope": "per_chip"}}}}


@pytest.fixture
def model():
    return MLAForwardModel()


def md(**kw):
    d = dict(B=1, H=2, Sq=32, Sk=32, Rq=4, Rk=6, Dn=8, Dr=4, Dv=8)
    d.update(kw)
    return make_metadata(*[d[k] for k in ("B", "H", "Sq", "Sk", "Rq", "Rk", "Dn", "Dr", "Dv")])


# --- spec 1.3: a null-valued axis is not a declared axis ----------------------------------------------
def test_a_null_active_length_is_not_a_declaration(model):
    """The JSON a caller writes for "I know Sq, not Sk" must not read as a declared full capacity."""
    explicit_null = model.bind(md(), {**SEM, "active_lengths": json.loads('{"Sq": 20, "Sk": null}')},
                               IMPL, NUMERICS, algorithm="expanded").analyze()
    omitted = model.bind(md(), {**SEM, "active_lengths": {"Sq": 20}}, IMPL, NUMERICS, algorithm="expanded").analyze()
    for rep in (explicit_null, omitted):
        m = rep["C_valid"]
        assert m.status == "partial" and m.missing_fields == ["active_lengths.Sk"]
        assert rep.header["active_extents"]["declared"] is False
        assert rep.header["active_extents"]["undeclared_axes"] == ["Sk"]
    assert explicit_null.value("C_valid") == omitted.value("C_valid")


def test_scheduled_extent_is_not_tested_against_an_assumed_active_length(model):
    """Rejecting a declaration on the strength of a value the report itself calls a scenario is wrong."""
    sem = {k: v for k, v in SEM.items() if k != "active_lengths"}      # active lengths undeclared
    rep = model.bind(md(), sem, {**IMPL, "scheduled_lengths": {"Sq": 8, "Sk": 8}}, NUMERICS,
                     algorithm="expanded").analyze()                    # 8 < the shape extent 32: legal
    assert rep.value("S_q_scheduled") == 8
    with pytest.raises(BindingError):                                   # still rejected against a DECLARED one
        model.bind(md(), SEM, {**IMPL, "scheduled_lengths": {"Sq": 8, "Sk": 8}}, NUMERICS, algorithm="expanded")


def test_scheduled_extents_are_per_axis(model):
    rep = model.bind(md(), SEM, {**IMPL, "scheduled_lengths": {"Sq": 32}}, NUMERICS, algorithm="expanded").analyze()
    assert rep["S_q_scheduled"].status == "bound" and rep.value("S_q_scheduled") == 32
    assert rep["S_k_scheduled"].status == "partial"
    assert "scheduled_lengths.Sk" in rep["S_k_scheduled"].missing_fields


# --- explicit failure over silent coercion -------------------------------------------------------------
@pytest.mark.parametrize("patch", [
    {"materialize": {"expanded_kv": 0}},                 # a truthy stand-in read as False by one consumer
    {"materialize": {"expandedkv": True}},               # a misspelled key
    {"subdivide": [3.5, 3], "rect_policy": "skip_future_subdivide"},           # a non-integer sub-tile
])
def test_strategy_declarations_are_validated(model, patch):
    with pytest.raises((MetadataError, BindingError)):
        model.bind(md(), SEM, {**IMPL, **patch}, NUMERICS, algorithm="expanded").analyze()


def test_a_sub_tile_larger_than_its_tile_is_reported_illegal(model):
    """A legality violation is a reported constraint, not an exception (spec 10.2 labels legality)."""
    rep = model.bind(md(), SEM, {**IMPL, "b_q": 2, "b_k": 2, "rect_policy": "skip_future_subdivide",
                                 "subdivide": [3, 3]}, NUMERICS, algorithm="expanded").analyze()
    broken = [c for c in rep.constraints if c["name"] == "subdivide_le_tile"]
    assert broken and broken[0]["holds"] is False and broken[0]["severity"] == "error"


def test_alias_target_accepts_the_same_spellings_as_every_other_role(model):
    from mla_model.adapters import ROLE_ALIASES
    spelling = next(k for k, v in ROLE_ALIASES.items() if v == "w_k_nope")
    t = md(Dn=8, Dv=8)
    t["w_v"] = {**t["w_v"], "alias_of": spelling}
    rep = model.bind(t, SEM, IMPL, NUMERICS, algorithm="expanded").analyze()
    groups = {r["role"]: r["alias_group"] for r in rep.extras["interface_ledger"]["tensors"]}
    assert groups["w_v"] == "w_k_nope"


# --- spec 6.1: allocation is per tensor ----------------------------------------------------------------
def test_every_capacity_axis_is_honoured_and_the_total_is_the_sum(model):
    t = md()
    t["kv_latent"] = {**t["kv_latent"], "capacity_shape": [1, 64, 7]}    # a capacity on the rank axis too
    rep = model.bind(t, SEM, IMPL, NUMERICS, algorithm="expanded").analyze()
    ledger = {r["role"]: r for r in rep.extras["interface_ledger"]["tensors"]}
    total = 0
    for role, row in ledger.items():
        assert rep.value(f"bytes[{role}]") == row["allocated_bytes"], role
        total += row["allocated_bytes"]
    assert rep.value("M_in") == total


def test_a_contiguous_stride_declaration_stays_bound(model):
    t = md()
    shape = t["q_latent"]["shape"]
    contiguous = [shape[1] * shape[2], shape[2], 1]
    t["q_latent"] = {**t["q_latent"], "strides": contiguous}
    assert model.bind(t, SEM, IMPL, NUMERICS, algorithm="expanded").analyze()["M_alloc_unique[metadata_ledger]"].status == "bound"
    t["q_latent"] = {**t["q_latent"], "strides": [shape[1] * shape[2] * 2, shape[2] * 2, 1]}   # padded rows
    assert model.bind(t, SEM, IMPL, NUMERICS, algorithm="expanded").analyze()["M_alloc_unique[metadata_ledger]"].status == "partial"


# --- spec 4.2: the declared extent policy decides the query-side read too ------------------------------
def test_query_side_reads_follow_the_declared_extent_policy(model):
    """A tail tile is read whole only if the strategy says so; the two are alternatives, not a sum."""
    tail = md(Sq=36)                                                     # 36 is not a multiple of b_q = 8
    sem = {**SEM, "active_lengths": {"Sq": 36, "Sk": 32}}
    impl = {**IMPL, "scheduled_lengths": {"Sq": 36, "Sk": 32}}
    logical = model.bind(tail, sem, {**impl, "executed_extent_policy": "logical"}, NUMERICS, algorithm="expanded").analyze()
    padded = model.bind(tail, sem, {**impl, "executed_extent_policy": "padded_to_tile"}, NUMERICS, algorithm="expanded").analyze()
    ev = lambda rep, i: next(e["value"] for e in rep.extras["transfer_events"] if e["id"] == i)
    b = logical.header["bindings"]
    groups = b["H"] // b["h_pp"]                                         # one read per (head group, row)
    assert ev(logical, "Cq_read") == groups * 36 * b["R_q"] * b["s_Cq"]  # the rows that exist
    assert ev(padded, "Cq_read") == groups * 40 * b["R_q"] * b["s_Cq"]   # 5 tiles of 8
    # a fully declared configuration leaves no permanent "tile-padded" caveat on the path
    assert logical["B[hbm_to_vmem]"].status == "bound"


# --- spec 5.5: the two normalisation forms are alternatives --------------------------------------------
def test_normalisation_forms_are_exclusive(model):
    num = {**NUMERICS, "recurrence": "unnormalized"}
    div = model.bind(md(), SEM, IMPL, {**num, "final_normalize": "divide"}, algorithm="expanded").analyze()
    rm = model.bind(md(), SEM, IMPL, {**num, "final_normalize": "reciprocal_multiply"}, algorithm="expanded").analyze()
    both = model.bind(md(), SEM, IMPL, num, algorithm="expanded").analyze()
    assert div.value("W_vec[recip]") == 0 and div.value("W_vec[div]") > 0
    assert rm.value("W_vec[div]") == 0 and rm.value("W_vec[recip]") > 0
    # undeclared: one named scenario, never the sum of both forms
    assert both.value("W_vec[div]") == div.value("W_vec[div]")
    assert both.value("W_vec[recip]") == div.value("W_vec[recip]")
    assert "final_normalize" in both["W_vec[div]"].missing_fields


def test_cast_sites_contradicting_the_declared_dtypes_are_a_conflict(model):
    num = {**NUMERICS, "exp_dtype": "float32", "p_operand_dtype": "bfloat16", "cast_points": []}
    rep = model.bind(md(), SEM, IMPL, num, algorithm="expanded").analyze()
    sites = {c.sources["site"] for c in rep.conflicts if c.field == "cast_points"}
    assert "exp_to_p" in sites
    agreed = {**num, "cast_points": ["exp_to_p"]}
    rep2 = model.bind(md(), SEM, IMPL, agreed, algorithm="expanded").analyze()
    assert "exp_to_p" not in {c.sources["site"] for c in rep2.conflicts if c.field == "cast_points"}


# --- spec 9.2: the combination is never smaller than one of its own terms -------------------------------
@pytest.mark.parametrize("groups", [[["mxu"]], [["layout"]], [["mxu"], ["exp", "reduce", "vpu"]], None])
def test_the_combined_bound_dominates_every_term(model, groups):
    impl = {**IMPL, "overlap_model": "serial_stage_sum"}
    if groups:
        impl["serial_stage_groups"] = groups
    rep = model.bind(md(), SEM, impl, NUMERICS, HW, algorithm="expanded").analyze()
    combined = rep.value("T_LB[combined]")
    terms = [rep.value(f"T_LB[{r}]") for r in ("mxu", "exp", "reduce", "vpu", "path:hbm_to_vmem", "path:vmem_to_hbm")]
    terms = [t for t in terms if t is not None]
    assert combined is not None and combined >= max(terms) - 1e-18


def test_the_combined_bound_inherits_the_caveats_of_its_terms(model):
    """A sum of partial terms cannot itself be complete."""
    rep = model.bind(md(), SEM, {**IMPL, "executed_extent_policy": None}, NUMERICS, HW, algorithm="expanded").analyze()
    m = rep["T_LB[combined]"]
    per_term = {f for r in ("mxu", "exp", "reduce", "vpu") for f in (rep[f"T_LB[{r}]"].missing_fields or [])}
    assert per_term and per_term <= set(m.missing_fields)
    assert m.status != "bound"


# --- spec 10: the comparison -----------------------------------------------------------------------------
def test_a_patch_keeps_the_baseline_mapping_declarations(model):
    base = {**IMPL, "materialize": {"expanded_kv": True, "q_nope": True}}
    row = model.bind(md(), SEM, base, NUMERICS, algorithm="expanded").compare(
        [{"name": "noQn", "materialize": {"q_nope": False}}])["rows"][1]
    assert row["strategy"]["materialize"] == {"expanded_kv": True, "q_nope": False}


def test_absence_comes_from_the_metric_not_from_a_missing_name(model):
    cmp_ = model.bind(md(), SEM, IMPL, NUMERICS, HW, algorithm="expanded").compare(
        [{"name": "abs", "algorithm": "absorbed_two_step"}], objectives=("B[hbm_to_vmem]", "no_such_metric"))
    base, cand = cmp_["rows"]
    assert is_absent(cand["values"]["HBM_mat[K^n,V]"])          # the variant has no expanded K/V to materialize
    assert is_absent(cand["values"]["V_lifetime_stages"])       # reported not_applicable by that variant
    assert not is_absent(base["values"]["V_lifetime_stages"])
    # a combined lower bound exists under a declared overlap model, and is watched under its own name
    assert base["values"]["T_LB[combined]"] is not None
    # an objective that cannot be used is named, not dropped in silence
    assert any("no_such_metric" in n for n in cmp_["selection_notes"])
    json.dumps(cmp_)                                            # no live sentinel escapes into the result


def test_compare_refuses_an_undeclared_algorithm(model):
    with pytest.raises(TaskIdentityError):
        model.bind(md(), SEM, IMPL, NUMERICS).compare([{"name": "x", "b_q": 4}])


def test_compile_evidence_survives_the_candidate_derivation(model):
    bound = model.bind(md(), SEM, IMPL, NUMERICS, algorithm="expanded").attach_compile_evidence(
        {"compiled_flops": 1234, "source": "toolchain dump"})
    for row in bound.compare([{"name": "bq4", "b_q": 4}])["rows"]:
        assert "toolchain dump" in row["legality"]["backend_support"]


# --- spec 7.3 / 7.4 --------------------------------------------------------------------------------------
def test_the_lifetime_scenario_is_named_only_where_it_bites(model):
    """An undeclared residency does not make the score-stage set incomplete: both branches agree there."""
    impl = {k: v for k, v in IMPL.items() if k != "q_resident"}
    rep = model.bind(md(), SEM, impl, {**NUMERICS, "cast_points": [], "final_normalize": "divide"},
                     algorithm="expanded").analyze()
    assert "q_resident" not in (rep["M_live[vmem:score]"].missing_fields or [])
    assert "q_resident" in rep["M_live[vmem:softmax]"].missing_fields
    assert "q_resident" in rep["local[q_nope_operand]"].missing_fields


def test_the_byte_ledger_and_the_live_sets_agree_about_residency(model):
    resident = model.bind(md(), SEM, {**IMPL, "q_resident": True}, NUMERICS, algorithm="expanded").analyze()
    rere = model.bind(md(), SEM, {**IMPL, "q_resident": False}, NUMERICS, algorithm="expanded").analyze()
    # re-reading Q per KV block shortens its lifetime AND moves more bytes; one without the other is incoherent
    assert rere.value("B[vmem_to_vreg]") > resident.value("B[vmem_to_vreg]")


def test_a_delayed_load_for_an_in_loop_projected_v_is_refused(model):
    with pytest.raises(MetadataError):
        model.bind(md(), SEM, {**IMPL, "v_load": "delayed", "materialize": {"expanded_kv": False}},
                   NUMERICS, algorithm="expanded").analyze()


def test_a_layout_for_another_variants_object_is_legal_but_inert(model):
    rep = model.bind(md(), SEM, {**IMPL, "layout": {"q_tilde": "compact"}}, NUMERICS, algorithm="expanded").analyze()
    assert rep["layout[q_tilde]"].status == "not_applicable"
    assert model.bind(md(), SEM, {**IMPL, "layout": {"q_tilde": "compact"}}, NUMERICS,
                      algorithm="absorbed_two_step").analyze()["layout[q_tilde]"].status == "bound"
    with pytest.raises(MetadataError):
        model.bind(md(), SEM, {**IMPL, "layout": {"not_an_object": "compact"}}, NUMERICS, algorithm="expanded").analyze()


def test_reports_are_reproducible_across_processes(model):
    """Binding order must not depend on the hash seed, or the shipped artifacts do not reproduce."""
    import subprocess, sys, os
    script = (
        "import sys; sys.path.insert(0, %r);\n"
        "from conftest import NUMERICS, make_metadata\n"
        "from mla_model import MLAForwardModel\n"
        "print(MLAForwardModel().bind(make_metadata(1,2,32,32,4,6,8,4,8), %r, %r, NUMERICS, algorithm='expanded')"
        ".analyze().to_json())" % (str(__import__('pathlib').Path(__file__).parent), SEM, IMPL))
    outs = []
    for seed in ("0", "1"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        outs.append(subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env,
                                   cwd=str(__import__('pathlib').Path(__file__).parent.parent)).stdout)
    assert outs[0] and outs[0] == outs[1]


def test_absence_does_not_depend_on_candidate_order(model):
    """Whether a metric reads as absent must not depend on which candidate happens to come first."""
    bound = model.bind(md(), SEM, IMPL, NUMERICS, HW, algorithm="expanded")
    switch = {"name": "abs", "algorithm": "absorbed_two_step"}
    tile = {"name": "bq4", "b_q": 4}
    first = bound.compare([switch, tile])
    second = bound.compare([tile, switch])
    by_name = lambda c: {r["name"]: r["values"] for r in c["rows"]}
    a, b_ = by_name(first), by_name(second)
    assert set(a) == set(b_)
    for name in a:
        assert a[name] == b_[name], name


def test_every_not_applicable_metric_explains_itself(model):
    for impl in (IMPL, {**IMPL, "materialize": {"expanded_kv": False}}, {**IMPL, "b_rq": None, "b_rk": None}):
        for alg in ("expanded", "absorbed_two_step", "absorbed_precomputed"):
            rep = model.bind(md(), SEM, impl, NUMERICS, HW, algorithm=alg).analyze()
            silent = [m.metric for m in rep.metrics if m.status == "not_applicable" and not (m.assumptions or m.notes)]
            assert not silent, f"{alg}: not_applicable without a reason: {silent}"


def test_the_two_input_totals_say_which_extent_they_use(model):
    t = md()
    t["kv_latent"] = {**t["kv_latent"], "capacity_shape": [1, 4096, 6]}
    rep = model.bind(t, SEM, IMPL, NUMERICS, algorithm="expanded").analyze()
    ledger = rep.extras["interface_ledger"]
    assert rep.value("M_in") == ledger["allocated_unique_total"]          # allocation extents
    assert rep.value("M_in[metadata_ledger]") == ledger["logical_total"]  # logical shapes
    assert rep.value("M_in") != rep.value("M_in[metadata_ledger]")        # they differ here, so both must say so
    assert any("M_in[metadata_ledger]" in x for x in rep["M_in"].not_equivalent_to)
    assert any("M_in," in x for x in rep["M_in[metadata_ledger]"].not_equivalent_to)


def test_a_resource_bound_is_no_firmer_than_the_quantity_it_divides(model):
    """B_e/W_e inherits whatever B_e rests on; otherwise dropping a declaration changes a 'bound' value."""
    declared = model.bind(md(), SEM, IMPL, NUMERICS, HW, algorithm="expanded").analyze()
    dropped = model.bind(md(), {k: v for k, v in SEM.items() if k != "active_lengths"}, IMPL, NUMERICS, HW,
                         algorithm="expanded").analyze()
    for name in ("T_LB[path:hbm_to_vmem]", "T_LB[mxu]"):
        a, b_ = declared[name], dropped[name]
        if a.value is not None and b_.value is not None and a.value != b_.value:
            assert b_.status != "bound" and b_.missing_fields, f"{name} changed value but still claims to be complete"
