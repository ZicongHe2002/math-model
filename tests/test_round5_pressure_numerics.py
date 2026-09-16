"""Regressions for the fifth review round (verification of the fourth round's fixes): pressure_numerics.

Each test names the problem it protects against.  Expected values are written here from the specification or
an independent hand computation, never read back from the package; a test that compares two surfaces of one
report against each other says so (differential) in its docstring.
"""
import pytest

from conftest import NUMERICS, make_metadata
from test_verify_round4 import SEM, IMPL, md
from mla_model import MLAForwardModel
from mla_model.errors import MetadataError

PRESSURE_BUDGETS = {"vreg_budget_bytes": 32768, "vmem_budget_bytes": 8192}     # illustrative test inputs


def PRESSURE_impl(**changes):
    """IMPL with overrides; a value of Ellipsis removes the field (an undeclared field, not a null one)."""
    d = {**IMPL, **changes}
    return {k: v for k, v in d.items() if v is not ...}


def PRESSURE_num(**changes):
    d = {**NUMERICS, **changes}
    return {k: v for k, v in d.items() if v is not ...}


def PRESSURE_bind(alg, impl=None, num=None, sem=None, tensors=None):
    return MLAForwardModel().bind(tensors or md(), sem or SEM, impl or IMPL, num or NUMERICS, algorithm=alg)


# ---------------------------------------------------------------------------------------------------------
# numerics: cast sites, final_normalize, vector-metric status
# ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [[["exp_to_p"]], [{"site": "exp_to_p"}], [1], ["exp_to_p", None]])
def test_pressure_nested_cast_points_is_a_metadata_error(bad):
    """[21] P1: a cast_points list whose element is not a site name must be refused with MetadataError from the
    validator, never escape as a raw TypeError from set().  The nested list and the dict are the unhashable shapes
    that escaped pre-fix; [1] and ['exp_to_p', None] are hashable controls that already raised MetadataError
    pre-fix through the unknown-site path, so every case additionally requires the entry-type message ('site
    names (strings)'), which distinguishes the new guard from the unknown-site message and from a raw TypeError."""
    with pytest.raises(MetadataError) as ei:
        PRESSURE_bind("expanded", num=PRESSURE_num(cast_points=bad))
    assert "site names (strings)" in str(ei.value), str(ei.value)
    assert "unknown cast_points" not in str(ei.value)


@pytest.mark.parametrize("alg, numerics, site", [
    # producer dtype != consumer dtype at a site the declared list OMITS -> contradiction at that site
    ("expanded", dict(score_dtype="float32", exp_dtype="bfloat16", cast_points=["exp_to_p", "accumulator_to_output"]), "score_to_exp"),
    ("expanded", dict(exp_dtype="float32", state_dtype="bfloat16", cast_points=["exp_to_p", "accumulator_to_output"]), "state_update"),
    ("expanded", dict(projection_accumulator_dtype="float32", projected_operand_dtype="bfloat16",
                      cast_points=["exp_to_p", "accumulator_to_output"]), "projected_operand"),
    ("absorbed_two_step", dict(accumulator_dtype="float32", projected_operand_dtype="bfloat16",
                               cast_points=["exp_to_p", "accumulator_to_output"]), "z_to_output"),
    # equal dtypes at a site the declared list INCLUDES -> the same contradiction the other way round
    ("expanded", dict(score_dtype="float32", exp_dtype="float32", cast_points=["score_to_exp", "exp_to_p", "accumulator_to_output"]), "score_to_exp"),
])
def test_pressure_cast_conflict_checked_at_every_site(alg, numerics, site):
    """[22] P2: the cast_points-vs-dtypes contradiction rule runs over all six CAST_SITES, not only exp_to_p and
    accumulator_to_output; each contradicting site produces a warning Conflict('cast_points') naming it."""
    rep = PRESSURE_bind(alg, num=PRESSURE_num(**numerics)).analyze()
    hits = [c for c in rep.conflicts if c.field == "cast_points" and site in c.message]
    assert hits, f"no Conflict('cast_points') for {site}: {[c.message for c in rep.conflicts]}"
    assert all(c.severity == "warning" for c in hits)


PRESSURE_ALL_SITES_DTYPES = dict(score_dtype="float32", exp_dtype="bfloat16", p_operand_dtype="bfloat16",
                                 state_dtype="float32", accumulator_dtype="float32", output_dtype="bfloat16",
                                 projection_accumulator_dtype="float32", projected_operand_dtype="bfloat16")
# hand-derived from CAST_SITE_DTYPES (spec: producer dtype != consumer dtype <=> a cast at that site):
#   score_to_exp          float32 -> bfloat16   cast      declared
#   exp_to_p              bfloat16 -> bfloat16  no cast   omitted
#   state_update          bfloat16 -> float32   cast      declared
#   accumulator_to_output float32 -> bfloat16   cast      declared
#   projected_operand     float32 -> bfloat16   cast      declared
#   z_to_output           not a site of the expanded variant (no Z), skipped by the rule
PRESSURE_ALL_SITES_AGREEING = ["score_to_exp", "state_update", "accumulator_to_output", "projected_operand"]


def test_pressure_no_cast_conflict_when_declarations_agree():
    """[22] P2 (control, paired with the positive cases above): a site list that agrees with the declared dtypes
    at ALL six sites, using every one of the four newly checked sites, raises no Conflict('cast_points') -- the
    rule over all sites produces no false positive.  Flipping a single newly checked site (dropping state_update,
    whose dtypes bfloat16 -> float32 imply a cast) yields exactly one conflict, naming that site only."""
    agree = PRESSURE_bind("expanded", num=PRESSURE_num(cast_points=PRESSURE_ALL_SITES_AGREEING,
                                                        **PRESSURE_ALL_SITES_DTYPES)).analyze()
    assert [c.message for c in agree.conflicts if c.field == "cast_points"] == []
    flipped = [s for s in PRESSURE_ALL_SITES_AGREEING if s != "state_update"]
    one = PRESSURE_bind("expanded", num=PRESSURE_num(cast_points=flipped, **PRESSURE_ALL_SITES_DTYPES)).analyze()
    hits = [c for c in one.conflicts if c.field == "cast_points"]
    assert len(hits) == 1 and "state_update" in hits[0].message, [c.message for c in hits]
    assert hits[0].severity == "warning"


def test_pressure_absorbed_output_cast_uses_the_projection_accumulator():
    """[27] P7: in the absorbed paths O = Z W^v is produced by the projection matmul, so the producer dtype of the
    accumulator_to_output site is projection_accumulator_dtype (bfloat16 here == output_dtype -> no cast), not the
    attention accumulator (float32).  Both the declared-list conflict check and the inferred op must agree."""
    dt = dict(accumulator_dtype="float32", projection_accumulator_dtype="bfloat16", output_dtype="bfloat16")
    declared = PRESSURE_bind("absorbed_two_step", num=PRESSURE_num(cast_points=["exp_to_p"], **dt)).analyze()
    assert [c for c in declared.conflicts if c.field == "cast_points" and "accumulator_to_output" in c.message] == []
    inferred = PRESSURE_bind("absorbed_two_step", num=PRESSURE_num(cast_points=..., **dt)).analyze()
    ops = {o["id"]: o for o in inferred.extras["vector_ops"]}
    assert ops["cast[accumulator_to_output]"]["exists"] is False
    # the expanded path has no output projection: its producer IS the attention accumulator (float32 != bfloat16)
    expanded = PRESSURE_bind("expanded", num=PRESSURE_num(cast_points=..., **dt)).analyze()
    assert {o["id"]: o for o in expanded.extras["vector_ops"]}["cast[accumulator_to_output]"]["exists"] is True


def test_pressure_single_consistent_final_normalize_warning():
    """[23] P3: with final_normalize undeclared the report carries exactly ONE warning about it, and that warning
    does not claim that both forms are counted: only the division form (n*D_A = B*H*S_q*D_v = 1*2*32*8 = 512
    divisions) is in the graph, the reciprocal form is excluded (0 reciprocals)."""
    rep = PRESSURE_bind("expanded", num=PRESSURE_num(final_normalize=...)).analyze()
    notes = [w for w in rep.warnings if "final_normalize" in w]
    assert len(notes) == 1, notes
    assert "both are counted" not in notes[0]
    assert rep.value("W_vec[div]") == 512
    assert rep.value("W_vec[recip]") == 0
    ops = {o["id"]: o for o in rep.extras["vector_ops"]}
    assert ops["final_normalize_recip"]["exists"] is False and ops["final_normalize_mul"]["exists"] is False


def test_pressure_inert_final_normalize_note_reaches_the_warnings():
    """[26] P6: under a normalized recurrence a declared final_normalize has no effect; the note saying so must
    reach rep.warnings (where a reader of W_vec[*] looks), not only F_total[executed_graph].assumptions.  The
    cast_points half of the problem (notes about a declared site list) is not covered here: see the report."""
    rep = PRESSURE_bind("expanded", num=PRESSURE_num(recurrence="normalized", final_normalize="divide")).analyze()
    notes = [w for w in rep.warnings if "final_normalize" in w and "no effect" in w]
    assert len(notes) == 1, rep.warnings
    assert rep.get("W_vec[div]") is None          # no final division exists under the normalized recurrence


@pytest.mark.parametrize("dropped", ["exp_dtype", "p_operand_dtype"])
def test_pressure_missing_fields_are_field_names_not_conditions(dropped):
    """[25] P5: with one dtype of the exp_to_p site undeclared, W_vec[cast] names the FIELD that is undeclared
    (plus the undeclared site list), never a condition string such as 'exp_dtype != p_operand_dtype'; the kinds
    whose counts do not depend on the dtypes (exp, mul) stay bound."""
    rep = PRESSURE_bind("expanded", num=PRESSURE_num(**{dropped: ..., "cast_points": ...})).analyze()
    assert rep["W_vec[cast]"].status == "partial"
    assert rep["W_vec[cast]"].missing_fields == sorted(["cast_points", dropped])
    for kind in ("exp", "mul", "cmp", "mask"):
        assert rep[f"W_vec[{kind}]"].status == "bound" and rep[f"W_vec[{kind}]"].missing_fields == [], kind
    for m in rep.metrics:
        assert all("!=" not in f for f in m.missing_fields), (m.metric, m.missing_fields)
    assert all("!=" not in u for u in rep.unknowns)


def test_pressure_vector_kind_status_is_per_kind():
    """[28] P8: an undeclared cast_points / final_normalize marks only the kinds whose counts depend on it.
    Hand count for W_vec[exp]: causal 32x32, offset 0, b_q=b_k=8, skip_future -> the 10 blocks (i, j) with j <= i,
    each 64 cells, over B*H = 2 -> 1280 exp(X - m') plus one alpha = exp(m - m') per row per visit:
    10 blocks * 8 rows * 2 = 160 -> 1440.  That count is exact, so its status must be 'bound' with nothing missing."""
    rep = PRESSURE_bind("expanded", num=PRESSURE_num(cast_points=..., final_normalize=...)).analyze()
    m = rep["W_vec[exp]"]
    assert (m.value, m.status, m.missing_fields) == (1440, "bound", [])
    for kind in ("cmp", "mask", "log", "add"):
        assert rep[f"W_vec[{kind}]"].status == "bound" and rep[f"W_vec[{kind}]"].missing_fields == [], kind
    assert rep["W_vec[cast]"].status == "partial" and rep["W_vec[cast]"].missing_fields == ["cast_points"]
    assert rep["W_vec[div]"].status == "partial" and rep["W_vec[div]"].missing_fields == ["final_normalize"]
    # declaring the two fields removes the scenario from exactly those kinds
    full = PRESSURE_bind("expanded", num=PRESSURE_num(cast_points=["exp_to_p", "accumulator_to_output"],
                                                       final_normalize="divide")).analyze()
    assert full["W_vec[cast]"].status == "bound" and full["W_vec[div]"].status == "bound"
    assert full.value("W_vec[exp]") == 1440


# ---------------------------------------------------------------------------------------------------------
# pressure: feasible region, legality, live sets
# ---------------------------------------------------------------------------------------------------------

def test_pressure_compare_flags_a_candidate_above_the_binding_bound():
    """[56] P1: compare's model-infeasible filter reads the per-level 'verdict' field, so a candidate whose b_k
    exceeds the binding b_k_max it reports at some level lands in model_infeasible_candidates and in the
    selection note; a candidate below every bound does not.  Differential against each row's own legality."""
    base = PRESSURE_impl(b_q=16, b_k=16, **PRESSURE_BUDGETS)
    bm = PRESSURE_bind("expanded", base)
    cmp = bm.compare([PRESSURE_impl(name="bk_big", b_q=16, b_k=128, **PRESSURE_BUDGETS),
                      PRESSURE_impl(name="bk_small", b_q=16, b_k=8, **PRESSURE_BUDGETS)])
    rows = {r["name"]: r for r in cmp["rows"]}
    big, small = rows["bk_big"]["legality"], rows["bk_small"]["legality"]
    assert any(128 > lv["b_k_max"] for lv in big["binding_stage_per_level"].values())
    assert any(v["verdict"] is False for v in big["model_estimated_feasible_under_budget"].values())
    assert all(v["verdict"] is True for v in small["model_estimated_feasible_under_budget"].values())
    assert "bk_big" in cmp["model_infeasible_candidates"] and "bk_small" not in cmp["model_infeasible_candidates"]
    assert any("bk_big" in n and "infeasible" in n for n in cmp["selection_notes"])
    assert "bk_big" in cmp["candidates_considered"]       # kept: model uncertainty is not illegality (spec 10.2)


def test_pressure_binding_metric_names_a_stage_over_budget_for_every_bk():
    """[65] P10 (and [56] for the over-budget case): a stage with no b_k term that exceeds the budget on its own
    (finalize here, live set > R for every b_k) is named in the binding metric's assumptions, agrees with the
    legality surface (differential), and its excess is max(0, M_live - R) per spec 7 with R = 500."""
    R = 500
    bm = PRESSURE_bind("absorbed_two_step", PRESSURE_impl(b_q=16, b_k=1, vmem_budget_bytes=R))
    rep = bm.analyze()
    excess = rep["excess_over_budget[vmem:finalize]"]
    assert excess.status == "bound" and excess.value > 0
    assert excess.value == rep.value("M_live[vmem:finalize]") - R
    assert rep["b_k_max[vmem:finalize]"].status == "not_applicable"      # no b_k term of its own
    binding_row = rep.extras["feasible_region_binding"]["vmem"]
    assert binding_row["stages_over_budget_independently_of_b_k"] == ["finalize"]
    m = rep["b_k_max[vmem:binding_stage]"]
    over_notes = [a for a in m.assumptions if "finalize" in a and "over the budget" in a]
    assert over_notes, m.assumptions
    assert "no b_k makes the configuration feasible" in over_notes[0]
    cmp = bm.compare([PRESSURE_impl(name="bk2", b_q=16, b_k=2, vmem_budget_bytes=R)])
    row = {r["name"]: r for r in cmp["rows"]}["bk2"]["legality"]["model_estimated_feasible_under_budget"]["vmem"]
    assert row["verdict"] is False and row["stages_over_budget_independently_of_b_k"] == ["finalize"]
    assert "bk2" in cmp["model_infeasible_candidates"]


@pytest.mark.parametrize("alg, flag, stage", [("expanded", "score_alias_exp", "score"),
                                              ("absorbed_two_step", "exp_alias_p_operand", "softmax")])
def test_pressure_feasible_rows_share_the_metric_status_rule(alg, flag, stage):
    """[57] P2: with an aliasing flag undeclared the graph-level fallback that makes b_k_max[vreg:<stage>]
    'partial' also reaches the feasible_region rows of the same quantity: same status, same missing field, same
    number on the declared-b_q row (differential between two surfaces of one report)."""
    rep = PRESSURE_bind(alg, PRESSURE_impl(**{flag: ...}, **PRESSURE_BUDGETS)).analyze()
    m = rep[f"b_k_max[vreg:{stage}]"]
    assert m.status == "partial" and m.missing_fields == [flag]
    rows = rep.extras["feasible_region_by_stage"]["vreg"][stage]
    declared = [r for r in rows if r["declared_b_q"]]
    assert len(declared) == 1
    assert declared[0]["b_k_max"] == m.value
    assert declared[0]["status"] == m.status and declared[0]["missing"] == m.missing_fields
    for r in rows:
        if r["b_k_max"] is not None:
            assert r["status"] == "partial" and flag in r["missing"] and flag in r["undeclared"], r


def test_pressure_vreg_aliasing_flag_does_not_downgrade_vmem_metrics():
    """[61] P6: score_alias_exp aliases vreg temporaries (X/E) only; the vmem live set, peak, envelope, b_k bound
    and excess are fully determined and stay 'bound' with nothing missing, while the vreg and combined metrics of
    the same report are 'partial' naming the flag.  The vmem value is the same whether the flag is True or False."""
    rep = PRESSURE_bind("expanded", PRESSURE_impl(score_alias_exp=..., **PRESSURE_BUDGETS)).analyze()
    for name in ("M_live[vmem:score]", "M_peak[vmem]", "envelope[vmem:score]", "b_k_max[vmem:score]",
                 "excess_over_budget[vmem:score]", "b_k_max[vmem:binding_stage]"):
        assert rep[name].status == "bound" and rep[name].missing_fields == [], name
    for name in ("M_live[vreg:score]", "M_live[all:score]", "b_k_max[vreg:binding_stage]"):
        assert rep[name].status == "partial" and rep[name].missing_fields == ["score_alias_exp"], name
    vmem_row = [r for r in rep.extras["feasible_region_by_stage"]["vmem"]["score"] if r["declared_b_q"]][0]
    assert vmem_row["status"] == "bound" and vmem_row["missing"] == [] and vmem_row["undeclared"] == []
    on = PRESSURE_bind("expanded", PRESSURE_impl(score_alias_exp=True, **PRESSURE_BUDGETS)).analyze()
    off = PRESSURE_bind("expanded", PRESSURE_impl(score_alias_exp=False, **PRESSURE_BUDGETS)).analyze()
    assert on.value("M_live[vmem:score]") == off.value("M_live[vmem:score]") == rep.value("M_live[vmem:score]")
    assert on.value("M_live[vreg:softmax]") < off.value("M_live[vreg:softmax]")     # the flag does act on vreg


@pytest.mark.parametrize("field", ["q_resident", "projection_in_kv_loop", "score_alias_exp", "exp_alias_p_operand",
                                   "both_qk_branches_live"])
@pytest.mark.parametrize("stand_in", [0, 1, "false", "true", "yes"])
def test_pressure_boolean_strategy_flags_are_validated(field, stand_in):
    """[58] P3: the three-valued residency / aliasing flags accept only true, false or absent; a stand-in such as
    0 or 'false' would silently take the opposite branch (q_resident=0 read as resident), so it is refused with
    MetadataError exactly as materialize.* already is."""
    with pytest.raises(MetadataError):
        PRESSURE_bind("expanded", PRESSURE_impl(**{field: stand_in}))
    assert PRESSURE_bind("expanded", PRESSURE_impl(**{field: ...})).analyze() is not None   # absent stays legal


def test_pressure_streaming_ledger_names_q_resident_when_undeclared():
    """[59] P4: the Q-operand streaming events depend on q_resident (once per (head, q-block) vs once per KV
    visit), so with the field undeclared they carry conditional_on='q_resident' and B[vmem_to_vreg] names it in
    missing_fields, matching the live-set metrics of the same report (differential).  The resident form is the
    scenario, so the undeclared value equals the q_resident=True value and is below the q_resident=False value."""
    rep = PRESSURE_bind("expanded", PRESSURE_impl(q_resident=...)).analyze()
    path = rep["B[vmem_to_vreg]"]
    assert path.status == "partial" and "q_resident" in path.missing_fields
    assert "q_resident" in rep["local[q_nope_operand]"].missing_fields
    q_events = [e for e in rep.extras["transfer_events"] if e["id"].startswith("stream_Q")]
    assert q_events and all(e["conditional_on"] == "q_resident" for e in q_events)
    resident = PRESSURE_bind("expanded", PRESSURE_impl(q_resident=True)).analyze()
    streamed = PRESSURE_bind("expanded", PRESSURE_impl(q_resident=False)).analyze()
    assert "q_resident" not in resident["B[vmem_to_vreg]"].missing_fields
    assert path.value == resident.value("B[vmem_to_vreg]") < streamed.value("B[vmem_to_vreg]")


def test_pressure_unevaluable_denominator_row_is_partial_and_names_the_field():
    """[60] P5: with kv_buffers undeclared the b_k denominator (a*b_q + c, which carries n_buf) cannot be evaluated;
    the feasible_region row uses 'partial' like the metric of the same quantity (FORMULA_INVENTORY: partial =
    unresolved symbols, unknown = no expression), names the declaring field kv_buffers, and the report's unknowns
    surface the field name."""
    rep = PRESSURE_bind("expanded", PRESSURE_impl(kv_buffers=..., vmem_budget_bytes=8192)).analyze()
    m = rep["b_k_max[vmem:score]"]
    assert m.value is None and m.status == "partial"
    row = [r for r in rep.extras["feasible_region_by_stage"]["vmem"]["score"] if r["declared_b_q"]][0]
    assert row["b_k_max"] is None and row["status"] == m.status == "partial"
    assert "kv_buffers" in row["missing"]
    assert any(u.startswith("kv_buffers:") for u in rep.unknowns), rep.unknowns


def test_pressure_binding_blockers_without_a_budget_union_the_stage_fields():
    """[62] P7: with no vmem budget the binding-stage metric is 'unknown' and its blockers are the union of the
    fields the per-stage bounds of the same report name (q_resident, v_load) plus the budget field, not the budget
    alone (differential across the per-stage metrics; symbols such as R_vmem are not fields and are excluded)."""
    rep = PRESSURE_bind("expanded", PRESSURE_impl(q_resident=..., v_load=..., vreg_budget_bytes=..., vmem_budget_bytes=...)).analyze()
    m = rep["b_k_max[vmem:binding_stage]"]
    assert m.value is None and m.status == "unknown"
    assert {"q_resident", "v_load", "vmem_budget_bytes"} <= set(m.missing_fields)
    per_stage = set()
    for st in ("score", "softmax", "pv", "finalize"):
        per_stage |= {f for f in rep[f"b_k_max[vmem:{st}]"].missing_fields if not f.startswith("R_")}
    assert per_stage <= set(m.missing_fields)
    assert {"q_resident", "v_load"} <= per_stage


def test_pressure_q_pe_tile_text_follows_the_declared_residency():
    """[63] P8: under q_resident=False the Q^r tile is live only in the score stage; its description must say the
    lifetime is shortened by that declaration, like the other Q operands, instead of asserting residency across
    the KV loop unconditionally."""
    rep = PRESSURE_bind("expanded", PRESSURE_impl(q_resident=False)).analyze()
    m = rep["local[q_pe_tile]"]
    assert "['score']" in m.scope
    text = " ".join(m.assumptions)
    assert "shortened when q_resident=False" in text
    assert "resident across KV loop" not in text


def test_pressure_ragged_b_q_sweep_is_capped_at_the_longest_sequence():
    """[64] P9: for a ragged task (active Sq = [48, 20, 7] in a capacity extent of 64) the b_q axis of the
    constraint surface is capped at the longest ACTIVE sequence, 48, exactly as a uniform S_q = 48 task is:
    powers of two <= 48 plus the declared b_q -> [1, 2, 4, 8, 16, 32] (16 is declared).  A cap at the capacity
    extent would add 64; the 512 fallback would add 64 and 128; neither may appear."""
    t = make_metadata(3, 4, 64, 96, 16, 32, 24, 8, 24)       # q capacity 64 > longest active 48
    ragged_sem = {k: v for k, v in SEM.items() if k != "position_offset"}
    ragged_sem.update({"active_lengths": {"Sq": [48, 20, 7], "Sk": [96, 20, 50]}, "per_batch_offsets": [48, 0, 43]})
    strat = PRESSURE_impl(b_q=16, b_k=16, scheduled_lengths=..., vreg_budget_bytes=32768, vmem_budget_bytes=1 << 20)
    bm = PRESSURE_bind("absorbed_two_step", strat, sem=ragged_sem, tensors=t)
    assert bm.bindings().get("S_q") is None                  # S_q is unbound for a ragged task
    axis = [r["b_q"] for r in bm.analyze().extras["feasible_region_by_stage"]["vreg"]["pv"]]
    assert axis == [1, 2, 4, 8, 16, 32]
    uniform_sem = {**{k: v for k, v in SEM.items() if k != "position_offset"},
                   "active_lengths": {"Sq": 48, "Sk": 96}, "position_offset": 48}
    uniform = PRESSURE_bind("absorbed_two_step", strat, sem=uniform_sem, tensors=t).analyze()
    assert [r["b_q"] for r in uniform.extras["feasible_region_by_stage"]["vreg"]["pv"]] == axis
