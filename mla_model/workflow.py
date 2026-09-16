"""One-call workload modelling, with explicitly named scenarios for missing declarations.

The YAML adapter supplies facts. This layer supplies *conditional scenarios*, never
claims about the source kernel. Each assumed field is recorded and dependent
metrics remain partial. Hardware evidence is loaded from a versioned local profile.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from .algorithms import ALGORITHMS
from .devices import load_device_profile
from .errors import MetadataError
from .numerics import CAST_SITE_DTYPES
from .report import Report, default_coverage
from .workload import load_workload

SCENARIO_PRESET = "bf16_f32_streaming_v1"
SUMMARY_METRICS = (
    "C_valid", "C_rect", "F_total[useful]", "F_total[executed_graph]", "M_in", "M_O", "M_LSE",
    "local[score_X]", "local[accumulator_A]", "M_peak[vreg]", "M_peak[vmem]",
    "B[hbm_to_vmem]", "T_LB[mxu]", "T_LB[path:hbm_to_vmem]", "T_pred[node_interval]",
)


def _fill(target, defaults, prefix, assumed):
    for key, value in defaults.items():
        if target.get(key) is None:
            target[key] = deepcopy(value)
            assumed[f"{prefix}.{key}"] = deepcopy(value)


def _scenario(bundle, mask, variant, mask_is_assumed):
    sem, impl, num = (deepcopy(bundle[k]) for k in ("semantics", "implementation", "numerics"))
    assumed = {}
    sem["mask"] = mask
    if mask_is_assumed:
        assumed["semantics.mask"] = mask
    if mask == "causal" and all(sem.get(k) is None for k in
                                ("position_offset", "query_position_start", "key_position_start", "per_batch_offsets")):
        _fill(sem, {"position_offset": 0}, "semantics", assumed)
    semantic_defaults = {"outputs": ["O", "LSE"], "projection_scope": "full",
                         "empty_row_policy": "zero_output_neg_inf_lse"}
    if sem.get("scale_value") is None:
        semantic_defaults["scale_policy"] = "standard"
    _fill(sem, semantic_defaults, "semantics", assumed)
    if "LSE" in sem["outputs"]:
        _fill(sem, {"lse_convention": "natural_log"}, "semantics", assumed)

    # This named numerical scenario is independent of the input storage dtype.
    # Explicit numeric-policy fields always win; input dtype alone never declares
    # accumulator precision, reassociation, casts or acceptance success.
    _fill(num, {"matmul_input_dtype": "bfloat16", "accumulator_dtype": "float32",
                "projection_accumulator_dtype": "float32", "projected_operand_dtype": "bfloat16",
                "score_dtype": "float32", "exp_dtype": "float32", "p_operand_dtype": "bfloat16",
                "state_dtype": "float32", "output_dtype": "bfloat16", "lse_dtype": "float32",
                "exp_base": "e", "recurrence": "unnormalized", "final_normalize": "divide"},
          "numerics", assumed)
    if num.get("cast_points") is None:
        casts = []
        for site, (src, dst) in CAST_SITE_DTYPES.items():
            if site == "z_to_output" and variant == "expanded":
                continue
            if site == "accumulator_to_output" and variant != "expanded":
                src = "projection_accumulator_dtype"
            if num[src] != num[dst]:
                casts.append(site)
        _fill(num, {"cast_points": casts}, "numerics", assumed)

    _fill(impl, {"name": SCENARIO_PRESET, "schedule": "q_outer_kv_inner",
                 "rect_policy": "skip_future" if mask == "causal" else "full_scan",
                 "executed_extent_policy": "logical", "kv_buffers": 1, "heads_per_program": 1,
                 "q_resident": True, "projection_in_kv_loop": False, "v_load": "early",
                 "score_alias_exp": False, "exp_alias_p_operand": False,
                 "both_qk_branches_live": False, "scratch_bytes": 0}, "implementation", assumed)
    # No layout, capacity budget, overlap, calibration or merge reuse is invented.
    mat = impl.get("materialize") or {}
    _fill(mat, {k: False for k in ("expanded_kv", "q_nope", "q_tilde", "z")},
          "implementation.materialize", assumed)
    impl["materialize"] = mat
    active = sem.get("active_lengths") or {}
    if active and all(isinstance(v, int) and not isinstance(v, bool) for v in active.values()):
        scheduled = impl.get("scheduled_lengths") or {}
        _fill(scheduled, active, "implementation.scheduled_lengths", assumed)
        impl["scheduled_lengths"] = scheduled
    return sem, impl, num, assumed


def _annotate_scenario(rep, assumed, variant_is_assumed):
    """Keep the core's values, but never promote scenario inputs to observed facts.

    The conservative section-level propagation also covers graph branches whose
    symbolic expressions do not retain the flags that selected them. Input-only
    byte accounting and dimensions do not depend on this execution scenario.
    """
    sem_fields = {k for k in assumed if k.startswith("semantics.")}
    exec_fields = {k for k in assumed if k.startswith("implementation.") and k != "implementation.name"}
    num_fields = {k for k in assumed if k.startswith("numerics.")}
    for m in rep.metrics:
        dependencies = set()
        sec = m.section.split(".", 1)[0]
        if sec == "1":
            key = {"gamma": "scale_policy", "rows_without_visible_keys": "mask"}.get(m.metric, m.metric)
            if f"semantics.{key}" in sem_fields:
                dependencies.add(f"semantics.{key}")
            if m.metric in ("Delta", "rows_without_visible_keys") and "semantics.position_offset" in sem_fields:
                dependencies.add("semantics.position_offset")
        elif sec == "2":
            dependencies |= {k for k in sem_fields if k.endswith((".mask", ".position_offset"))}
            if m.metric != "C_valid":
                dependencies |= {k for k in exec_fields if any(x in k for x in
                                    ("rect_policy", "executed_extent_policy", "scheduled_lengths"))}
        elif sec == "3":
            if m.source_type not in ("compiled", "definition", "input"):
                dependencies |= sem_fields
                if "executed" in m.metric or m.metric.startswith(("W_vec", "W_resource", "exp_count")):
                    dependencies |= exec_fields
                if m.metric.startswith(("W_vec", "W_resource")):
                    dependencies |= num_fields
                if variant_is_assumed:
                    dependencies.add("algorithm.variant")
        elif sec == "4":
            if m.metric in ("M_O", "M_LSE"):
                dependencies |= {k for k in sem_fields if k.endswith(".outputs")}
                dtype = "output_dtype" if m.metric == "M_O" else "lse_dtype"
                if f"numerics.{dtype}" in num_fields:
                    dependencies.add(f"numerics.{dtype}")
            elif m.metric.startswith(("HBM_mat", "M_mat")):
                dependencies |= sem_fields | exec_fields | num_fields
                if variant_is_assumed:
                    dependencies.add("algorithm.variant")
        elif sec in ("5", "6", "7", "8", "9") and m.source_type not in ("compiled", "measured", "definition"):
            dependencies |= exec_fields | num_fields
            if sec in ("5", "9") or m.metric.startswith("n_"):
                dependencies |= sem_fields
            if variant_is_assumed:
                dependencies.add("algorithm.variant")
        if not dependencies or m.status in ("not_applicable", "conflict"):
            continue
        m.missing_fields = sorted(set(m.missing_fields) | dependencies)
        if m.value is not None:
            m.status = "partial"
        m.coverage = default_coverage(m.status, m.missing_fields)
        m.assumptions.append(f"Conditional {SCENARIO_PRESET} scenario; undeclared fields and their values are recorded in header.workload_scenario")
    rep.header["workload_scenario"] = {"preset": SCENARIO_PRESET, "assumed_fields": deepcopy(assumed),
                                        "variant_is_scenario": variant_is_assumed}
    rep.warnings.insert(0, "This is a named analytical scenario, not a recovered kernel implementation. "
                           "Fields absent from the workload remain named missing on dependent metrics.")
    for k in sorted(assumed):
        rep.unknowns.append(f"{k}: workload omitted this field; scenario uses {assumed[k]!r}")
    if variant_is_assumed:
        rep.unknowns.append("algorithm.variant: 'mla' selects the family; each variant is an independent scenario")


@dataclass
class WorkloadReport:
    workload: dict
    device: dict
    algorithm: str
    reports: dict[str, Report]
    normalized_input: dict
    scenarios: list[dict] = field(default_factory=list)

    def to_dict(self):
        rows = []
        for row in self.scenarios:
            rep = self.reports[row["id"]]
            rows.append({**deepcopy(row), "metrics": {k: rep[k].to_dict() for k in SUMMARY_METRICS if rep.get(k)},
                         "conflicts": [c.to_dict() for c in rep.conflicts]})
        return {"workload": deepcopy(self.workload), "device": deepcopy(self.device), "algorithm": self.algorithm,
                "scenarios": rows, "reports": {k: rep.to_dict() for k, rep in self.reports.items()}}

    def to_json(self, indent=2):
        return json.dumps(self.to_dict(), indent=indent, default=str, allow_nan=False)

    def to_markdown(self):
        def value(rep, name, divisor=1):
            v = rep.value(name)
            return "unknown" if not isinstance(v, (int, float)) else f"{v / divisor:,.6g}"
        lines = ["# MLA workload modelling", "",
                 f"- Device: **{self.device['name']}**, one chip; hardware data retrieved {self.device.get('retrieved')}",
                 f"- Algorithm: **{self.algorithm}**; {len(self.reports)} explicit scenario(s)",
                 f"- Scenario preset: `{SCENARIO_PRESET}`; YAML declarations override each preset field.",
                 "- FLOPs use FMA = 2. Time figures are resource lower bounds, not measured latency.",
                 "- Missing numerical and execution declarations are assumptions, listed per scenario; dependent metrics are partial.", "",
                 "| Mask | Variant | Useful GFLOP | Executed GFLOP | HBM reads (GiB) | VREG live peak (MiB) | MXU lower bound (ms) | HBM-read lower bound (ms) |",
                 "|---|---|---:|---:|---:|---:|---:|---:|"]
        for row in self.scenarios:
            rep = self.reports[row["id"]]
            vals = [value(rep, n, d) for n, d in (("F_total[useful]", 1e9), ("F_total[executed_graph]", 1e9),
                    ("B[hbm_to_vmem]", 2**30), ("M_peak[vreg]", 2**20), ("T_LB[mxu]", .001),
                    ("T_LB[path:hbm_to_vmem]", .001))]
            lines.append(f"| {row['mask']} | {row['algorithm']} | " + " | ".join(vals) + " |")
        first = next(iter(self.reports.values()))
        lines += ["", f"Input allocation: **{value(first, 'M_in', 2**20)} MiB** (seven input tensors, before materialization).",
                  "", "Physical HBM/VMEM capacities are device facts; they are not compiler allocation budgets. "
                  "VREG feasibility, actual spill and measured latency require further evidence. "
                  "The precomputed variant keeps merge work amortization unknown until cache.merged_weight_reuse is declared.", "",
                  "## Hardware sources", ""]
        sources = sorted({q.get("source") for q in self.device.get("quantities", []) if q.get("source")})
        lines += [f"- {s}" for s in sources]
        for row in self.scenarios:
            lines += ["", f"## {row['id']}", "",
                      f"Full report: [{row['id']}.md]({row['id']}.md) · [{row['id']}.json]({row['id']}.json)", "",
                      "Assumed fields (add a semantics, numerics or implementation section to YAML to declare them):", "",
                      "```json", json.dumps(row["assumptions"], indent=2), "```"]
            conflicts = self.reports[row["id"]].conflicts
            if conflicts:
                lines += ["", "Conflicts:", ""] + [f"- {c.field}: {c.message}" for c in conflicts]
        return "\n".join(lines) + "\n"

    def write(self, out_dir):
        target = Path(out_dir)
        # Build every serialization before creating output files.
        content = {"summary.json": self.to_json(), "summary.md": self.to_markdown(),
                   "normalized_input.json": json.dumps(self.normalized_input, indent=2, default=str, allow_nan=False)}
        for name, rep in self.reports.items():
            content[f"{name}.json"] = rep.to_json()
            content[f"{name}.md"] = rep.to_markdown()
        target.mkdir(parents=True, exist_ok=True)
        for name, data in content.items():
            (target / name).write_text(data, encoding="utf-8")
        return {name: str((target / name).resolve()) for name in content}


def analyze_yaml(source, *, device="tpu-v6-e", algorithm="mla", mask=None, model=None):
    """Model one workload using only its YAML, a named device and an algorithm.

    ``mla`` requests all three variants. Missing mask requests both causal and
    unmasked scenarios. No production arrays, accelerator jobs or network calls
    are performed. Explicit YAML declarations take precedence over scenario fields.
    """
    from .model import MLAForwardModel
    bundle = load_workload(source)
    hardware = load_device_profile(device)
    declared_algorithm = bundle.get("algorithm")
    if not isinstance(algorithm, str) or algorithm not in ("mla", *ALGORITHMS):
        raise MetadataError(f"algorithm must be mla or one of {ALGORITHMS}, got {algorithm!r}")
    if declared_algorithm and declared_algorithm != "mla":
        if algorithm not in ("mla", declared_algorithm):
            raise MetadataError(f"algorithm {algorithm!r} conflicts with YAML algorithm {declared_algorithm!r}")
        algorithm = declared_algorithm
    variants = ALGORITHMS if algorithm == "mla" else (algorithm,)
    declared_mask = bundle["semantics"].get("mask")
    if mask is not None and mask not in ("causal", "none"):
        raise MetadataError("mask must be causal or none")
    if declared_mask is not None and mask is not None and declared_mask != mask:
        raise MetadataError(f"mask {mask!r} conflicts with YAML semantics.mask {declared_mask!r}")
    selected_mask = declared_mask if declared_mask is not None else mask
    masks = (selected_mask,) if selected_mask is not None else ("causal", "none")
    model = model or MLAForwardModel()
    reports, rows, resolved = {}, [], {}
    for mask_kind in masks:
        for variant in variants:
            sem, impl, num, assumed = _scenario(bundle, mask_kind, variant, selected_mask is None)
            bound = model.bind(deepcopy(bundle["tensors"]), sem, impl, num, hardware, algorithm=variant)
            rep = bound.analyze()
            _annotate_scenario(rep, assumed, algorithm == "mla")
            sid = f"{mask_kind}.{variant}"
            rep.title = f"MLA workload scenario: {sid} ({SCENARIO_PRESET})"
            reports[sid] = rep
            rows.append({"id": sid, "mask": mask_kind, "algorithm": variant, "assumptions": assumed,
                         "variant_is_scenario": algorithm == "mla"})
            resolved[sid] = {"semantics": sem, "implementation": impl, "numerics": num, "algorithm": variant,
                             "assumed_fields": assumed}
    device_data = {"name": hardware.name, "retrieved": hardware.retrieved, "provenance": hardware.provenance,
                   "quantities": hardware.quantity_summary(), "num_devices": 1}
    normalized = {**deepcopy(bundle), "hardware": hardware.to_dict(), "scenario_preset": SCENARIO_PRESET,
                  "resolved_scenarios": resolved}
    return WorkloadReport(deepcopy(bundle["_workload"]), device_data, algorithm, reports, normalized, rows)
