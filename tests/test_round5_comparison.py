"""Regressions for the fifth review round (verification of the fourth round's fixes): comparison.

Each test names the problem it protects against.  Expected values are written here from the specification or
an independent hand computation, never read back from the package; a test that compares two surfaces of one
report against each other says so (differential) in its docstring.
"""
import json

import pytest

from conftest import NUMERICS
from test_verify_round4 import SEM, IMPL, HW, md
from mla_model import MLAForwardModel, is_absent, ABSENT_JSON
from mla_model.errors import MetadataError, MLAModelError

COMPARISON_SCENARIO_NAMES = ("T_LB[combined:scenario=full_overlap_max]", "T_LB[combined:scenario=serial_stage_sum]")
COMPARISON_EXPANDED_ONLY = ("HBM_mat[K^n,V]", "local[v_tile]", "V_lifetime_stages")   # spec 10.3: gone on "switch to absorbed"


def COMPARISON_MD(cmp):
    from mla_model import comparison_to_markdown
    return comparison_to_markdown(cmp)


def COMPARISON_BIND(impl, algorithm="expanded", hw=HW):
    return MLAForwardModel().bind(md(), SEM, impl, NUMERICS, hw, algorithm=algorithm)


def COMPARISON_ROW(cmp, name):
    return next(r for r in cmp["rows"] if r["name"] == name)


def COMPARISON_NUMERIC(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


# --- [29] C1 -------------------------------------------------------------------------------------------
def test_comparison_undeclared_overlap_is_unknown_not_a_structural_change():
    """[29] C1: a metric the baseline emits only under scenario spellings (overlap_model undeclared) is UNKNOWN in
    the plain spelling, never 'absent'; no appears/disappears is reported when no algorithm switch happened.

    Differential in one place: the candidate's declared T_LB[combined] must equal the baseline's own
    full_overlap_max scenario figure (same task, same variant, same number by construction).  The value relation
    serial_stage_sum >= full_overlap_max comes from spec 9 (a sum of stage bounds is never below their max).
    """
    impl = {k: v for k, v in IMPL.items() if k not in ("overlap_model", "vreg_budget_bytes")}
    bound = COMPARISON_BIND(impl)
    cmp = bound.compare([{"name": "decl", "overlap_model": "full_overlap_max", "vreg_budget_bytes": 65536}])
    base, cand = COMPARISON_ROW(cmp, "base"), COMPARISON_ROW(cmp, "decl")
    # baseline: plain name unknown (not absent), scenario spellings numeric
    assert base["values"]["T_LB[combined]"] is None and not is_absent(base["values"]["T_LB[combined]"])
    assert base["statuses"]["T_LB[combined]"] is None
    full, serial = (base["values"][k] for k in COMPARISON_SCENARIO_NAMES)
    assert COMPARISON_NUMERIC(full) and COMPARISON_NUMERIC(serial) and serial >= full
    assert all(base["statuses"][k] == "partial" for k in COMPARISON_SCENARIO_NAMES)
    # candidate: plain name numeric and equal to the baseline's scenario, scenario spellings unknown (not absent)
    assert cand["values"]["T_LB[combined]"] == full
    for k in COMPARISON_SCENARIO_NAMES:
        assert cand["values"][k] is None and not is_absent(cand["values"][k])
    # the delta of a declaration difference is 'unknown', never a fictitious structural label
    for k in ("T_LB[combined]",) + COMPARISON_SCENARIO_NAMES:
        assert cand["delta_vs_baseline"][k] is None
    for r in cmp["rows"]:
        assert not any(is_absent(v) for v in r["values"].values())
        assert not any(d in ("appears", "disappears") for d in r["delta_vs_baseline"].values())
    text = COMPARISON_MD(cmp)
    assert "| T_LB[combined] | ? |" in text and "n/a" not in text.split("(`n/a`")[0]


def test_comparison_undeclared_rect_policy_surfaces_agree():
    """[29] C1: with rect_policy undeclared the baseline's C_rect/C_pad/rect_waste read as unknown ('?'), the
    selection note says 'unknown or not applicable', and no row says 'not present in this variant' (differential:
    values, delta, selection_notes and markdown of one comparison must tell one story).
    """
    impl = {k: v for k, v in IMPL.items() if k != "rect_policy"}
    bound = COMPARISON_BIND(impl)
    rep = bound.analyze()
    assert rep.get("C_rect") is None and rep.get("C_rect[scenario=skip_future]") is not None   # the precondition of the problem
    cmp = bound.compare([{"name": "sf", "rect_policy": "skip_future"}])
    base, cand = COMPARISON_ROW(cmp, "base"), COMPARISON_ROW(cmp, "sf")
    for k in ("C_rect", "C_pad", "rect_waste"):
        assert base["values"][k] is None and not is_absent(base["values"][k])
        assert COMPARISON_NUMERIC(cand["values"][k]) and cand["statuses"][k] == "bound"
        assert cand["delta_vs_baseline"][k] is None
    # rect_waste: the causal 32x32 task at b_q = b_k = 8 with skip_future keeps the 4 diagonal tiles + 6 strictly
    # lower ones per head = 10 tiles x 64 = 640 scores per (B,H); C_valid = 32*33/2 = 528; waste = 112 per head,
    # H = 2 heads -> 224
    assert cand["values"]["rect_waste"] == 224
    notes = " ".join(cmp["selection_notes"])
    assert "rect_waste (unknown or not applicable for some candidate)" in notes
    assert "not present in this variant" not in json.dumps(cmp)
    assert "| rect_waste | ? |" in COMPARISON_MD(cmp)


# --- [30] C2 -------------------------------------------------------------------------------------------
def test_comparison_absence_is_a_property_of_the_row_not_of_the_candidate_set():
    """[30] C2: an absorbed variant has no expanded K/V materialization whether or not an expanded candidate is
    in the list; its rows read ABSENT (n/a) in both comparisons and the markdown says 'n/a', never '?'.

    Differential: the row's values must be identical between the two comparisons (set independence), and its
    statuses must equal the row's own analyze() statuses.
    """
    bound = COMPARISON_BIND(IMPL, algorithm="absorbed_two_step")
    rep = bound.analyze()
    for k in COMPARISON_EXPANDED_ONLY:                       # the row's own report carries the explicit absence
        assert rep[k].status == "not_applicable" and rep[k].value is None
    alone = bound.compare([{"name": "bq4", "b_q": 4}])
    with_exp = bound.compare([{"name": "bq4", "b_q": 4}, {"name": "exp", "algorithm": "expanded"}])
    for cmp in (alone, with_exp):
        for name in ("base", "bq4"):
            row = COMPARISON_ROW(cmp, name)
            for k in COMPARISON_EXPANDED_ONLY:
                assert is_absent(row["values"][k]) and row["values"][k] == ABSENT_JSON
                assert row["statuses"][k] == "not_applicable"
    for name in ("base", "bq4"):
        assert COMPARISON_ROW(alone, name)["values"] == COMPARISON_ROW(with_exp, name)["values"]
    exp = COMPARISON_ROW(with_exp, "exp [expanded]")
    for k in COMPARISON_EXPANDED_ONLY:                       # the real switch shows the real structural label
        assert not is_absent(exp["values"][k]) and exp["delta_vs_baseline"][k] == "appears"
    # the expanded switch materialises K^n/V, so the row holds a positive byte count (the ledger itself is not the
    # subject of this test)
    assert COMPARISON_NUMERIC(exp["values"]["HBM_mat[K^n,V]"]) and exp["values"]["HBM_mat[K^n,V]"] > 0
    for cmp in (alone, with_exp):
        line = next(l for l in COMPARISON_MD(cmp).splitlines() if l.startswith("| HBM_mat[K^n,V] |"))
        assert line.startswith("| HBM_mat[K^n,V] | n/a | n/a |") and "?" not in line


# --- [31] C3 -------------------------------------------------------------------------------------------
def test_comparison_model_infeasible_candidates_are_flagged():
    """[31] C3: a row whose per-level feasibility verdict is False is listed in model_infeasible_candidates and the
    spec 10.2 'kept but model-estimated infeasible' note fires (pre-fix: truthiness of the per-level dicts).

    Hand argument for the baseline: at b_q = b_k = 8, Dv = 8 the pv stage holds an f32 accumulator of
    8*8*4 = 256 bytes and an f32 score tile of 8*8*4 = 256 bytes before the V tile, so a 512-byte vreg budget is
    already exhausted -> verdict False.  Differential: the flagged list must equal the rows with any False verdict.
    """
    bound = COMPARISON_BIND({**IMPL, "vreg_budget_bytes": 512})
    cmp = bound.compare([{"name": "bq4", "b_q": 4}])
    base = COMPARISON_ROW(cmp, "base")["legality"]["model_estimated_feasible_under_budget"]
    assert base["vreg"]["verdict"] is False
    assert base["vreg"]["status"] == "bound"                 # a complete evaluation, not an incomplete one
    flagged = cmp["model_infeasible_candidates"]
    assert "base" in flagged
    assert sorted(flagged) == sorted(r["name"] for r in cmp["rows"]
                                     if any(v["verdict"] is False for v in r["legality"]["model_estimated_feasible_under_budget"].values()))
    note = next(n for n in cmp["selection_notes"] if n.startswith("kept but model-estimated infeasible"))
    assert "base" in note and "not a compilation verdict" in note
    assert "base" in cmp["candidates_considered"]            # 10.2: uncertainty is not illegality, the row stays in K
    assert not cmp["excluded_candidates"]


# --- [32] C4 -------------------------------------------------------------------------------------------
@pytest.mark.parametrize("candidates, objectives", [
    ([None], None),                                          # a non-strategy element
    (["/nonexistent.json"], None),                           # a path string must not be opened
    ([b"bytes"], None),
    (5, None),                                               # not iterable
    ({"name": "x"}, None),                                   # a bare mapping is not a list of candidates
    ("abc", None),                                           # a string is not a list of candidates
    ([{"name": "c", "b_q": 4}], "rect_waste"),               # a str objective is not a sequence of names
    ([{"name": "c", "b_q": 4}], ["rect_waste", 3]),          # a non-str objective
])
def test_comparison_malformed_inputs_raise_metadata_error(candidates, objectives):
    """[32] C4: malformed candidates / objectives fail with MetadataError (the package contract), never with a
    raw AttributeError / FileNotFoundError / TypeError.
    """
    bound = COMPARISON_BIND(IMPL)
    kw = {} if objectives is None else {"objectives": objectives}
    with pytest.raises(MetadataError) as ei:
        bound.compare(candidates, **kw)
    assert isinstance(ei.value, MLAModelError)


def test_comparison_read_only_mapping_candidate_honours_its_algorithm_key():
    """[32] C4: a non-dict Mapping candidate goes through the patch path, so its 'algorithm' key switches the
    variant instead of being silently ignored (pre-fix: AttributeError on .name).
    """
    from types import MappingProxyType
    bound = COMPARISON_BIND(IMPL)
    cmp = bound.compare([MappingProxyType({"name": "mp", "b_q": 4, "algorithm": "absorbed_two_step"})])
    row = COMPARISON_ROW(cmp, "mp [absorbed_two_step]")
    assert row["algorithm"] == "absorbed_two_step" and row["strategy"]["b_q"] == 4
    assert row["strategy"]["b_k"] == IMPL["b_k"]                # a patch, the rest of the baseline is kept
    for k in COMPARISON_EXPANDED_ONLY:
        assert row["delta_vs_baseline"][k] == "disappears"      # spec 10.3: the switch study reports the disappearance


# --- [33] C5 -------------------------------------------------------------------------------------------
def test_comparison_candidate_with_algorithm_none_keeps_the_baseline_algorithm():
    """[33] C5: a candidate naming algorithm None keeps the baseline's variant (the implemented design decision):
    the row carries no variant suffix, its plain-named metrics are present, and F_total[useful] is invariant.

    F_total[useful] hand value: the useful FLOPs do not depend on the strategy, so both rows must hold one number
    and useful_work_invariant must be True; the number itself is checked as positive and equal across rows.
    """
    bound = COMPARISON_BIND(IMPL)
    cmp = bound.compare([{"name": "alg_none", "algorithm": None}, {"name": "plain"}])
    assert [r["name"] for r in cmp["rows"]] == ["base", "alg_none", "plain"]
    assert all(r["algorithm"] == "expanded" for r in cmp["rows"])
    useful = {r["values"]["F_total[useful]"] for r in cmp["rows"]}
    assert len(useful) == 1 and COMPARISON_NUMERIC(next(iter(useful))) and next(iter(useful)) > 0
    assert cmp["useful_work_invariant"] is True
    row = COMPARISON_ROW(cmp, "alg_none")
    for k in ("B[hbm_to_vmem]", "M_live[vreg:softmax]", "T_LB[combined]", "n_programs"):
        assert not is_absent(row["values"][k]) and COMPARISON_NUMERIC(row["values"][k])
        assert row["delta_vs_baseline"][k] == 0                 # an unchanged strategy differs in nothing
    assert "None" not in json.dumps(cmp["pareto_set"])


# --- [34] C6 -------------------------------------------------------------------------------------------
def test_comparison_names_no_report_carries_are_dropped_or_noted():
    """[34] C6: watched scenario spellings that no row's report carries are not rendered as '?' (which the legend
    reads as 'exists, value unknown'); a caller-named objective no report carries is kept and named in a selection
    note as 'not a metric of these reports'.
    """
    bound = COMPARISON_BIND(IMPL)                              # overlap_model declared in every row
    cmp = bound.compare([{"name": "bq4", "b_q": 4}])
    for r in cmp["rows"]:
        for k in COMPARISON_SCENARIO_NAMES:
            assert k not in r["values"] and k not in r["statuses"] and k not in r["delta_vs_baseline"]
        assert "T_LB[combined]" in r["values"] and COMPARISON_NUMERIC(r["values"]["T_LB[combined]"])
    assert "scenario=" not in COMPARISON_MD(cmp)
    # T_pred[node_interval] IS a metric of these reports (status unknown without calibration): '?' is right for it
    assert bound.analyze()["T_pred[node_interval]"].status == "unknown"
    assert all(r["values"]["T_pred[node_interval]"] is None for r in cmp["rows"])
    named = bound.compare([{"name": "bq4", "b_q": 4}], objectives=[COMPARISON_SCENARIO_NAMES[0], "B[hbm_to_vmem]"])
    assert all(COMPARISON_SCENARIO_NAMES[0] in r["values"] and r["values"][COMPARISON_SCENARIO_NAMES[0]] is None for r in named["rows"])
    assert named["pareto_objectives"] == ["B[hbm_to_vmem]"]
    note = next(n for n in named["selection_notes"] if n.startswith("objectives left out of the Pareto filter"))
    assert f"{COMPARISON_SCENARIO_NAMES[0]} (not a metric of these reports)" in note
    assert "unknown or not applicable" not in note


# --- [35] C7 -------------------------------------------------------------------------------------------
def test_comparison_both_absent_delta_uses_the_canonical_absence_spelling():
    """[35] C7: when both rows lack a metric the delta is the same canonical absence encoding as the values
    (is_absent recognises it); the comparison stays JSON-serialisable and holds no bare 'n/a'.
    """
    bound = COMPARISON_BIND(IMPL, algorithm="absorbed_two_step")
    cmp = bound.compare([{"name": "bq4", "b_q": 4}])
    row = COMPARISON_ROW(cmp, "bq4")
    for k in COMPARISON_EXPANDED_ONLY:
        assert is_absent(row["values"][k]) and is_absent(row["delta_vs_baseline"][k])
        assert row["delta_vs_baseline"][k] == ABSENT_JSON
    assert COMPARISON_ROW(cmp, "base")["delta_vs_baseline"]["HBM_mat[K^n,V]"] is None      # the baseline has no delta
    dumped = json.dumps(cmp)                                   # no live sentinel object survives
    assert '"n/a"' not in dumped and ABSENT_JSON in dumped
    assert not is_absent("n/a")


# --- [36] C8 -------------------------------------------------------------------------------------------
def test_comparison_rows_carry_the_status_behind_every_value():
    """[36] C8: a row exposes the status behind each value (differential against analyze()), and a Pareto
    objective that is a scenario value for some row is named in a selection note (spec 11.4 traceability).
    """
    impl = {k: v for k, v in IMPL.items() if k != "materialize"}
    bound = COMPARISON_BIND(impl)
    rep = bound.analyze()
    assert rep["HBM_mat[K^n,V]"].status == "partial" and "materialize.expanded_kv" in rep["HBM_mat[K^n,V]"].missing_fields
    assert rep["B[hbm_to_vmem]"].status == "partial"
    cmp = bound.compare([{"name": "bq4", "b_q": 4}])
    base = COMPARISON_ROW(cmp, "base")
    assert set(base["statuses"]) == set(base["values"])
    for k in base["values"]:
        m = rep.get(k)
        assert base["statuses"][k] == (None if m is None else m.status)
        if m is not None and not is_absent(base["values"][k]):
            assert base["values"][k] == m.value
    assert base["statuses"]["HBM_mat[K^n,V]"] == "partial" and base["statuses"]["B[hbm_to_vmem]"] == "partial"
    assert base["statuses"]["F_total[useful]"] == "bound"
    note = next(n for n in cmp["selection_notes"] if "status partial" in n)
    assert "B[hbm_to_vmem]" in note and "compares scenarios" in note
    assert "B[hbm_to_vmem]" in cmp["pareto_objectives"]       # kept, but with the caveat above


# --- [37] C9 -------------------------------------------------------------------------------------------
def test_comparison_over_budget_stage_is_a_definite_verdict_not_partial():
    """[37] C9: a stage over the budget independently of b_k gives verdict False with status 'bound' ('partial' is
    reserved for stages_not_evaluated), and a non-positive b_k_max is annotated as an empty feasible region.

    Hand argument at a 64-byte vreg budget, b_q = 8, Dn = 8: the q_proj stage's f32 output tile alone is
    8*8*4 = 256 bytes with no b_k term, so it is over budget for every b_k; at the softmax stage the f32 row
    statistics m, l (2*8*4 = 64 bytes) already fill the budget before any score column, so b_k_max < 1.
    """
    bound = COMPARISON_BIND({**IMPL, "vreg_budget_bytes": 64})
    cmp = bound.compare([{"name": "bq4", "b_q": 4}])
    L = COMPARISON_ROW(cmp, "base")["legality"]
    vreg = L["model_estimated_feasible_under_budget"]["vreg"]
    assert vreg["verdict"] is False and vreg["status"] == "bound"
    assert "q_proj" in vreg["stages_over_budget_independently_of_b_k"]
    assert vreg["stages_not_evaluated"] == []
    assert vreg["feasible_region_empty"] is True
    assert L["binding_stage_per_level"]["vreg"]["b_k_max"] < 1
    for r in cmp["rows"]:                                      # differential: the flag agrees with the raw bound
        for level, row in r["legality"]["model_estimated_feasible_under_budget"].items():
            assert row["feasible_region_empty"] == (r["legality"]["binding_stage_per_level"][level]["b_k_max"] < 1)
            assert row["status"] == ("partial" if row["stages_not_evaluated"] else "bound")
    assert set(cmp["model_infeasible_candidates"]) == {"base", "bq4"}


# --- [38] C10 ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("candidates, repeated", [
    ([{"name": "base", "b_q": 4}], "base"),                    # the baseline's own name
    ([{"name": "d", "b_q": 4}, {"name": "d", "b_q": 2}], "d"),   # two candidates sharing a name
])
def test_comparison_duplicate_row_names_are_refused(candidates, repeated):
    """[38] C10: the derived lists (pareto_set, candidates_considered, ...) are keyed by row name, so a duplicate
    name is refused with MetadataError (the implemented design decision) instead of yielding ['base', 'base'].
    """
    bound = COMPARISON_BIND(IMPL)
    with pytest.raises(MetadataError, match="unique") as ei:
        bound.compare(candidates)
    assert repeated in str(ei.value)


def test_comparison_variant_suffix_makes_a_reused_name_distinct():
    """[38] C10: the uniqueness check is keyed on the row name as rendered (strategy name plus variant suffix).
    Boundary in both directions: the same strategy name under a different algorithm is a different row
    ('d [absorbed_two_step]') and is accepted with every derived list de-duplicated; a third candidate that
    renders to the same suffixed name is refused with MetadataError naming 'd [absorbed_two_step]'.

    The acceptance half is a companion guard for the [38] uniqueness check (it also held pre-fix, when compare()
    had no check at all); the refusal half is the regression proper.
    """
    bound = COMPARISON_BIND(IMPL)
    two = [{"name": "d", "b_q": 4}, {"name": "d", "b_q": 4, "algorithm": "absorbed_two_step"}]
    cmp = bound.compare(two)
    names = [r["name"] for r in cmp["rows"]]
    assert names == ["base", "d", "d [absorbed_two_step]"] and len(set(names)) == 3
    assert len(set(cmp["candidates_considered"])) == len(cmp["candidates_considered"])
    assert len(set(cmp["pareto_set"])) == len(cmp["pareto_set"])
    # a different strategy (b_q 2) under the same name and the same variant renders to the same row name
    with pytest.raises(MetadataError, match="unique") as ei:
        bound.compare(two + [{"name": "d", "b_q": 2, "algorithm": "absorbed_two_step"}])
    assert "d [absorbed_two_step]" in str(ei.value)
