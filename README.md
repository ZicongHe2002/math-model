# MLA forward — parameter-driven mathematical model

A reusable, CPU-only analytical model of the MLA (multi-head latent attention) forward pass:

    M(task parameters, algorithm, execution strategy, numeric policy, hardware profile) -> stage-level report

The same code and the same model object serve any valid configuration; every number in a report is traceable to a
symbolic formula, its bindings, scope, assumptions and evidence.  Nothing is filled from example values, unknowns are
`null`, and conflicting declarations are reported instead of resolved silently.

* Specification: `MLA_Forward_Parametric_Mathematical_Model.v3.en.md` (+ `.docx`), prompt: `MLA_Parametric_Model_Implementation_Prompt.v3.en(1).md`
* Formula inventory & derivations: [docs/FORMULA_INVENTORY.md](docs/FORMULA_INVENTORY.md)
* Specification review (discrepancies, assumptions): [docs/SPEC_REVIEW.md](docs/SPEC_REVIEW.md)
* Architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) · Handoff & test results: [docs/HANDOFF.md](docs/HANDOFF.md)

## Requirements

Python ≥ 3.10 with `sympy`, `numpy`, and `PyYAML`; `pytest` for the tests. The workspace's
`/opt/anaconda3/bin/python3` provides these dependencies. The core needs no TPU or API key and does not import JAX.
Run from this directory, or install the package with `python3 -m pip install .`.

## Model a workload YAML in one command

Use an existing MLA workload YAML containing its shape, input storage dtype, and tiling declarations:

```bash
/opt/anaconda3/bin/python3 -m mla_model model --yaml mla_fwd.yaml --device tpu-v6-e --algorithm mla
```

Use `python3` instead when that interpreter has the dependencies installed. The command writes its reports to
`reports/mla_fwd/tpu-v6e/` by default (`reports/<YAML filename stem>/<canonical device name>/`).
Select a mask and output directory explicitly when needed:

```bash
python3 -m mla_model model --yaml mla_fwd.yaml --device tpu-v6-e --algorithm mla --mask causal --out-dir reports/example
```

`--mask` accepts `causal` or `none`. If the YAML does not declare a mask and no override is supplied, the report
contains both named mask scenarios. `--algorithm mla` expands the three paths `expanded`, `absorbed_two_step`, and
`absorbed_precomputed`; pass any one of those names to analyze only that path. The source YAML is read without edits.

YAML `dtype` describes input storage only. Missing numerical and execution policies are filled for analysis under
the named `bf16_f32_streaming_v1` scenario, with the supplied assumptions recorded in the report header and
`normalized_input`. Metrics that depend on those assumptions remain `partial` and name the missing declarations.
Explicit YAML policies take precedence over the scenario. Hardware specifications provide conditional resource
bounds; actual latency and compiled spill still require applicable measurement or compiler evidence.

The bundled `tpu-v6e` profile loads locally and records official source URLs and its retrieval date. It uses
[Google Cloud's v6e compute and bandwidth specifications](https://docs.cloud.google.com/tpu/docs/v6e),
[explicit GiB HBM capacities](https://docs.cloud.google.com/compute/docs/tpus/tpu-machines), and
[JAX's physical VMEM specifications](https://docs.jax.dev/en/latest/pallas/tpu/hardware.html).
Compiler allocation budgets and usable VREG capacity remain unknown; physical capacity does not supply those budgets.

The same entry point is available in Python:

```python
from mla_model import MLAForwardModel

workload = MLAForwardModel().analyze_yaml("mla_fwd.yaml", device="tpu-v6-e", algorithm="mla")
expanded = workload.reports["causal.expanded"]  # when the causal scenario is included
workload.write("reports/example")
markdown = workload.to_markdown()
json_text = workload.to_json()
```

The result is a `WorkloadReport` containing the individual scenario reports and normalized input provenance.

## Commands

Symbolic description (no data bound):

```bash
python3 -m mla_model describe --algorithm expanded --format md
```

Analyze one invocation from a bundle JSON (tensors + semantics + implementation + numerics + optional hardware):

```bash
python3 -m mla_model analyze --config examples/configs/01_square_causal_prefill_expanded.json --format md
```

Or with separate files:

```bash
python3 -m mla_model analyze --metadata tensors.json --semantics sem.json --implementation impl.json --numerics num.json --hardware examples/hardware/tpu_v6e_illustrative.json --algorithm absorbed_two_step --format json --out report.json
```

Compare candidate strategies for the same task:

```bash
python3 -m mla_model compare --config examples/configs/06_calibrated_synthetic_expanded.json --format md
```

Regenerate every example report (`examples/generated/`):

```bash
python3 examples/run_examples.py
```

Run the tests:

```bash
python3 -m pytest tests -q
```

## Python API

```python
from mla_model import MLAForwardModel

model = MLAForwardModel()                                   # no dimensions, tiles or device inside
symbolic = model.describe(adapter="seven_input_latent", algorithm="expanded")

bound = model.bind(
    tensor_metadata={"q_latent": {"shape": [B, Sq, Rq], "dtype": "bfloat16"}, "kv_latent": {...},
                     "q_pe": {...}, "k_pe": {...}, "w_q_nope": {...}, "w_k_nope": {...}, "w_v": {...}},
    semantic_config={"mask": "causal", "position_offset": 0, "scale_policy": "standard", "active_lengths": {"Sq": Sq, "Sk": Sk},
                     "outputs": ["O", "LSE"], "projection_scope": "full", "empty_row_policy": "error"},
    implementation={"b_q": 16, "b_k": 16, "rect_policy": "skip_future", "executed_extent_policy": "logical",
                    "kv_buffers": 2, "heads_per_program": 1, "score_alias_exp": True, "exp_alias_p_operand": False,
                    "both_qk_branches_live": False, "materialize": {"expanded_kv": False, "q_nope": False},
                    "scratch_bytes": 0, "projection_in_kv_loop": False, "overlap_model": "full_overlap_max"},
    numeric_policy={"matmul_input_dtype": "bfloat16", "accumulator_dtype": "float32", "score_dtype": "float32", ...},
    hardware_profile=None,                                  # or a profile dict / JSON path
    algorithm="expanded",                                   # REQUIRED: "expanded" | "absorbed_two_step" | "absorbed_precomputed"
)
report = bound.analyze()
report.value("C_valid"); report["F_total[useful]"].expression; report.to_markdown(); report.to_json()

# same-task comparison; a candidate may switch algorithm, but any non-strategy key raises TaskIdentityError
comparison = bound.compare([{"name": "bq32", "b_q": 32, "b_k": 16, "rect_policy": "skip_future", "executed_extent_policy": "logical"},
                            {"name": "absorbed", "algorithm": "absorbed_two_step", "b_q": 16, "b_k": 16,
                             "rect_policy": "skip_future", "executed_extent_policy": "logical"}])

# evidence enters here; the metrics it fills are 'unknown' until it does
bound.attach_calibration([{"node_kind": "matmul", "dtype": "bfloat16", "epsilon_low": 0.4, "epsilon_high": 0.7}])
bound.attach_compile_evidence({"compiled_flops": 1_234_567, "spill_fill_bytes": 4096, "source": "toolchain dump"})
```

Metric names carry their scope, so query them as printed in the report: work as `F_total[useful]`,
`F_total[rect]`, `F_total[executed_graph]`; projected rows as `N_Qproj`, `N_Kproj`, `N_Vproj`; pressure per
storage level and stage as `M_live[vreg:softmax]`, `envelope[vmem:score]`, `M_peak[all]`; the tile bound per
stage as `b_k_max[vreg:pv]` with `b_k_max[vreg:binding_stage]` naming the tightest one; transfers as
`B[hbm_to_vmem]`, `B[vmem_to_vreg]`; bounds as `T_LB[mxu]`, `T_LB[layout]`, `T_LB[path:hbm_to_vmem]`, and
`T_LB[combined]` only when an overlap model is declared (otherwise one metric per named scenario).

`algorithm` may also be omitted, from `bind` and from `python3 -m mla_model analyze` alike, which the
specification's reading flow allows. The report then carries the algorithm-independent metrics once and every
variant-dependent metric once per named variant scenario, each `partial` with `algorithm` among its missing
fields. `compare` needs a declared variant, since a merged scenario report has no single metric to
difference; a candidate's `algorithm` key still expresses the switch study.

Tensor metadata may also be any objects exposing `.shape` and `.dtype` (numpy arrays, `jax.ShapeDtypeStruct`, …);
only metadata is read.

The three extents of specification section 1.3 stay separate: allocation comes from tensor shapes and any
`capacity_shape`, the active extent from `semantics.active_lengths`, and the scheduled extent the kernel iterates
over from `implementation.scheduled_lengths`.  A key-value cache capacity is never taken to be the number of keys
consumed in this call.

Anything left undeclared (tiles, scheduled extents, aliasing flags, Q residency, the V-load policy, cast sites,
the final-normalisation form, materialization, layout kinds, budgets, overlap model, either active length,
output scope, projection scope, hardware, calibration) does not stop the analysis: the affected metrics are
reported with status `partial` and the missing field named, or as explicitly named scenarios.  Contradictory declarations raise
`BindingError` / `CapabilityError` or appear in the report's conflict list.

## Layout

```
mla_model/            package (core)            tests/               pytest suite (+ tests/independent/ oracles)
mla_model/reference/  small-shape numpy refs    examples/configs/    illustrative configurations (not defaults)
docs/                 inventory, review, arch   examples/hardware/   profile template, illustrative v6e, synthetic calibrated
                                                examples/generated/  reports produced by run_examples.py
```

All example dimensions, tiles, dtypes, seeds and tolerances are **illustrative fixtures**; none is a production default
or an acceptance workload.
