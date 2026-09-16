# Architecture

```
metadata / JSON / array-like objects (shape, dtype only)
        │  parse_tensor_metadata                    (metadata.py)
        ▼
SevenInputLatentAdapter.extract  ──►  TaskParameters θ        (adapters.py)
   role/axis mapping · cross-checks · capacity vs active extents · capability gates
        │
        ▼
MLAForwardModel.bind(θ, SemanticConfig, ExecutionStrategy, NumericPolicy, HardwareProfile, algorithm=…)
        │                                                  (model.py, semantics.py, implementation.py,
        ▼                                                   numerics.py, hardware.py)
BoundModel ── graph = build_graph(algorithm, …)             (algorithms.py)
        │        matmul nodes · vector ops · local objects · transfer events · materializations
        ▼
BoundModel.analyze()  ──►  Report (metrics, constraints, conflicts, unknowns, scenarios, extras)
        │      visibility.py  work.py  memory.py  pressure.py  performance.py      (report.py)
        ▼
JSON / Markdown   ·   BoundModel.compare(candidates) ──► finite differences, Pareto set, ranking
```

## Separation of concerns

| Layer | Module | Holds |
|---|---|---|
| Task and scope | `adapters.py`, `semantics.py` | dimensions, the three extents, offset, scale policy, output scope, projection scope, empty-row policy |
| Algorithm variant | `algorithms.py` | expanded, absorbed two-step and absorbed precomputed graph builders |
| Numerical semantics | `numerics.py` | dtypes of intermediates, exponential base, recurrence form, final-normalisation form, cast sites, acceptance |
| Tiling, residency, layout, schedule | `implementation.py` | tiles, scheduled extents, rectangle and extent policies, buffers, alias flags, Q residency, V-load policy, materialization, layout kinds, budgets, overlap model and its serial resource groups |
| Hardware and calibration | `hardware.py`, `performance.py` | scoped quantities, layout tiles, calibration buckets, bounds versus predictions |
| Symbolic core | `symbolic.py` | symbol registry, custom aggregates, partial evaluation, constraints, coefficient extraction |
| Reporting | `report.py` | metric and report containers, JSON, Markdown |

## Per-stage pressure and feasibility

The specification states the stage envelope and the feasible `b_k` range for a stage, so both are computed for
every stage of every storage level. The binding stage is the one whose bound is smallest; it is named, and the
comparison's feasibility verdict follows it rather than the softmax stage alone. A stage whose live set has no
`b_k` term places no bound and is reported `not_applicable`. The feasible region is evaluated over a sweep of
`b_q` values, with the declared one marked, so the output is a constraint surface rather than one tile.

## Three extents, kept apart

The specification insists that an allocation, the data actually live, and the region a kernel iterates over are
three different things, so the model carries all three. Allocation extents come from tensor shapes and any
`capacity_shape`; they drive interface bytes. Active extents come from `semantics.active_lengths` and drive
all mathematical work. Scheduled extents come from `implementation.scheduled_lengths` and drive the rectangle
grid, padding counts and execution multiplicities. Each is validated against the others, and leaving active or
scheduled extents undeclared marks every dependent metric `partial` rather than quietly promoting a shape.

## Two storage levels, never substituted

Local objects carry a storage level: streamed residency windows and scratch are `vmem`, vector temporaries are
`vreg`. Live sets, peaks, envelopes and feasible regions are reported per level and for the union. The budgets
are separate symbols, so declaring only one leaves the other level's feasibility unknown instead of borrowing a
figure. The level assignment is an analysis assumption, stated as such in every metric's scope, not a claim
about compiler allocation.

## Modes

* **Symbolic** — `describe(adapter, algorithm)`: no metadata; each metric carries its expression and missing fields.
* **Bound** — `bind(...).analyze()`: expressions evaluated where possible, residual expressions kept, conflicts visible.
* **Algorithm undeclared** — `bind(...)` with no `algorithm`: the specification's 11.1 flow allows the path to be
  preserved as unknown. Algorithm-independent metrics appear once; every variant-dependent metric appears once
  per named variant scenario, marked `partial` with `algorithm` among its missing fields.
* **Calibrated** — the same with a profile whose calibration buckets actually apply to this task's nodes; predicted
  intervals appear beside, never instead of, the resource lower bounds. If no bucket applies, the mode stays bound.

## Statelessness and task identity

`MLAForwardModel` holds no parameters. Each `bind` returns a fresh `BoundModel`; `with_strategy`, `with_algorithm`,
`attach_calibration` and `attach_compile_evidence` all derive new bound objects that share θ. `fingerprint()`
hashes the adapter, dimensions, active lengths, offsets, mask, scale, output scope, projection scope, cache state,
dtypes and numeric policy. `compare()` refuses any candidate that changes it, and refuses any candidate key that is
not an execution-strategy field, so a shape or semantics change cannot masquerade as a speedup. A candidate may
name a different `algorithm`, which is how "switch to absorbed" is expressed.

## Symbolic evaluation

* sympy is the expression tree. User strings are never passed to `eval` or `sympify`.
* `CausalVisibleCount` and `BlockStat` are `sympy.Function` subclasses whose `eval` fires only once every argument
  is an integer. The first uses a constant-time closed form; the second uses closed-form arithmetic over query
  blocks, with brute-force enumeration kept as the test oracle and as the implementation for a scheduled grid
  wider than the active extents.
* `evaluate(expr, bindings)` substitutes what is known, converts a fully bound result to a number, and otherwise
  returns the residual expression and the sorted names of the missing symbols. Unknown is `None`, never zero.
* A value computed under an undeclared field is a scenario value: status `partial`, with the field named in
  `missing_fields`, never `bound`.

## Optional integrations (not prerequisites)

* `hardware.from_runtime_jax()` lazily imports JAX to read what the runtime reports; it returns `None` off-TPU.
* `attach_compile_evidence` and `attach_calibration` are where compiled or measured evidence enters; the metrics
  they fill are otherwise `unknown` and carry source type `compiled` or `measured`.
* Low-level compiler output ingestion, a small-graph movement solver, and an agent are **not implemented**.
