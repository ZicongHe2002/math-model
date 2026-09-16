# Handoff: what is implemented, what was measured, and what is not claimed

Everything below was produced and run in this workspace on a CPU-only macOS host with `python3` 3.11
(`/opt/anaconda3/bin/python3`), `sympy` 1.12 and `numpy` 2.3.1.  No network access, no device jobs, no
package installs, no changes outside this directory.

## 1. Executable functionality (verified by running it)

| Capability | Where | Verified how |
|---|---|---|
| Symbolic mode: expressions, constraints and the dependency graph with nothing bound | `MLAForwardModel.describe(adapter, algorithm)` | `python3 -m mla_model describe`; tests assert `F_total[useful]` has value `None`, named missing fields and a printable expression |
| Bound mode: reads the current metadata/config, evaluates what is known, keeps the rest symbolic | `model.bind(...).analyze()` | 419 tests; 9 example configurations regenerate reports |
| Algorithm preserved as unknown: variant-dependent metrics as named variant scenarios | `model.bind(...)` with no `algorithm` | the specification's own 11.2 call runs verbatim; a test checks each scenario against the corresponding declared-variant report |
| Calibrated mode: latency intervals from matching calibration buckets, per resource class, kept separate from lower bounds | `HardwareProfile.calibration`, `perf.calibrated.nodes` | example 06 reports `mode = calibrated`; tests assert the interval and that the mode stays `bound` when no bucket applies |
| Seven-input latent adapter: role/axis mapping, cross-checks, capability gates | `mla_model/adapters.py` | 11 adapter tests plus the independent oracles |
| Visibility, rectangle classification (including the future and padding rectangles a full scan executes), padding, scheduled extents | `mla_model/visibility.py` | 45 visibility tests. Committed differential tests check the closed-form evaluator against the package's own brute-force enumerator on 49,152 exhaustive uniform configurations, 1,500 random subdivision cases and 1,200 scheduled-grid cases, with zero mismatches; a further out-of-suite sweep of 188,160 active/scheduled cases and 110,592 subdivision cases also matched and confirmed that the four classes partition the selected set. The independent side, written from the specification rather than from the package, lives in `tests/independent/test_oracle_visibility.py` |
| Work: useful / rectangular / executed-graph FLOPs and vector-operation counts | `mla_model/work.py`, `algorithms.py` | an independent oracle transcribes the spec section 5 formulas with plain integers over 50 random families |
| Hierarchical bytes: interface, materialization, four transfer paths, local objects, layout coverage | `mla_model/memory.py` | an independent oracle recomputes spec 6 and 7.1 by hand; materialization traffic is checked against a hand-derived figure |
| Per-level (VREG / VMEM) live sets, stage envelope with coefficient provenance, feasible region | `mla_model/pressure.py` | tests recompute `a·b_q·b_k + b·b_q + c·b_k + d` and the `b_k` bound by hand |
| Resource lower bounds and calibrated predictions as distinct objects | `mla_model/performance.py` | tests assert the matrix-unit bound equals executed work over peak, that a profile without vector throughputs leaves those bounds unknown, and that each transfer path uses its own bandwidth |
| Same-task candidate comparison: finite differences, Pareto set, per-candidate legality, algorithm switch | `BoundModel.compare(...)` | example 01 compares five candidates, one of them an absorbed switch; task-identity tests |
| Small-shape references: dense expanded and absorbed, online recurrences, partition merge | `mla_model/reference/` | an independent oracle compares them against an explicit-loop float64 implementation written from spec section 3 |
| Explicit failure instead of fallback | `mla_model/errors.py`, the adapter | error fixtures 90 and 91; tests for contradictory shapes, bad caches, bad offsets and capability gates |
| JSON and Markdown reports, CPU-only command line, no installation required | `report.py`, `cli.py` | command-line round-trip tests; every example is produced through `python3 -m mla_model` |

## 2. Test results actually obtained

```
python3 -m pytest tests -q
419 passed in 189s
```

Breakdown by file:

| File | Tests |
|---|---|
| `tests/test_round5_declarations.py` | 70 |
| `tests/test_round5_performance.py` | 51 |
| `tests/test_round5_pressure_numerics.py` | 51 |
| `tests/test_visibility.py` | 45 |
| `tests/test_spec_compliance_round3.py` | 43 |
| `tests/test_round5_residuals.py` | 32 |
| `tests/test_model.py` | 31 |
| `tests/test_verify_round4.py` | 31 |
| `tests/test_round5_comparison.py` | 21 |
| `tests/test_adapter.py` | 11 |
| `tests/test_pressure_perf.py` | 9 |
| `tests/test_reference.py` | 7 |
| `tests/independent/test_oracle_numerics_metamorphic.py` | 5 |
| `tests/independent/test_oracle_visibility.py` | 5 |
| `tests/test_docs_match_reality.py` | 4 |
| `tests/independent/test_oracle_work_bytes.py` | 3 |

The table is produced by `python3 -m pytest tests -q --collect-only`, not transcribed, and
`tests/test_docs_match_reality.py` re-derives every count quoted here so the document cannot drift from the
code without the suite failing.

The files under `tests/independent/` compute every expected value from their own code written from the
specification.  They never call the package's helper functions to produce an expected value.  Tests that
compare two of the package's own implementations against each other are differential, not independent, and
live in `tests/test_visibility.py` instead: both sides are this package, so they catch a coding slip in one
implementation but would pass a shared misreading of the specification.

`tests/test_spec_compliance_round3.py`, `tests/test_verify_round4.py` and the five `tests/test_round5_*.py` files are
the regressions for the third, fourth and fifth review rounds.  Most assert a value derived here from the specification; a few are deliberately
differential, checking that two surfaces of the report agree (a live set against its envelope coefficients, a
symbolic byte metric against the per-tensor ledger, a comparison row against the report it came from). Those
are named as such in the file.

Two tests in `tests/test_model.py` enforce the report contract itself across a fully declared, a partially
declared and a symbolic report: every metric carries the full specification field set including coverage and
source type; status `bound` implies a value and no missing fields; status `partial` names a missing field;
status `unknown` carries no value; no unknown is reported as zero; and a bound metric still shows the
expression and bindings it came from.  Writing those checks surfaced one real gap, since fixed: the combined
resource lower bound could come back without a value and without naming the throughputs it needed.

`python3 examples/run_examples.py` regenerates every report.  All 9 valid configurations analyse; the two
fixtures whose names end in `expected_error` raise explicit errors; the scale-conflict fixture reports one
conflict and still analyses.  Output lands in `examples/generated/` as JSON and Markdown per configuration,
plus `.compare.*` where candidates are declared, plus `SUMMARY.json`.

Report shape for example 01, as an indication of granularity:

| Quantity | Count |
|---|---|
| Metrics | 234 across 9 sections |
| Bound / partial / unknown / not applicable | 78 / 117 / 23 / 16 |
| Constraints | 5 |
| Named unresolved input fields | 58 |

A third of that configuration's metrics are fully bound and the rest name what they still need, which is the
point: this example leaves rank sub-tiles, layout kinds, budgets, residency, cast sites and hardware
undeclared, and every quantity that depends on one of them says so.  The partial share rose over the
third review round precisely because fields that used to be filled in quietly are now named.

The package is about 7,100 lines and the tests about 4,300.

Performance measured on this host. Visibility statistics are closed-form arithmetic over query blocks,
including the case of a scheduled grid wider than the active extents, so no analysis path walks rectangles:

| Case | `analyze()` |
|---|---|
| `B = 8`, `H = 128`, `S_q = S_k = 131072`, 512-element tiles | 0.56 s |
| the same at 8-element tiles (268 million rectangles) | 0.19 s |
| 131072 active inside a 262144 scheduled grid, tensors allocated at 262144, 8-element tiles | 0.19 s |

A single `analyze()` on a small task settles at about 0.14 s once the structural expression caches are warm;
they are keyed on sympy expressions, so a second analysis of the same binding reproduces the first report
exactly without re-deriving anything. The brute-force enumerator remains as the test oracle and refuses,
with a clear error, to walk a grid above a million rectangles.

## 3. Symbolic-only results (no numbers without declarations)

* Tile sizes, buffer counts, heads per program, layout kinds, scheduled extents, budgets, overlap model,
  alias flags and the in-loop projection flag stay symbolic or produce explicitly named scenarios when
  undeclared.  Dependent metrics then carry status `partial` and the missing field name.
* Hardware throughputs, bandwidths, register capacities, layout tiles and calibration efficiencies come from
  a profile.  A missing one leaves that bound `unknown`; it never borrows another figure.
* Peak spill allocation, spill traffic, exposed spill time and the minimum extra movement are definitions
  with status `unknown` until compile or measurement evidence is attached.  Compiled work is likewise unknown.
* The critical path is unknown without a declared schedule model.

## 4. Optional interfaces (present, not exercised against real systems)

* `mla_model.hardware.from_runtime_jax()` lazily imports JAX and reads what the runtime exposes.  On this CPU
  host it returns `None`, which is the only behaviour ever observed.  It has never run on a TPU.
* `BoundModel.attach_compile_evidence({...})` and `attach_calibration([...])` accept evidence and report it
  with source type `compiled` or `measured`.  They were exercised with synthetic fixtures only.  No real
  low-level compiler output has been ingested.
* Each metric carries the fields an agent would consume: expression, bindings, value, unit, scope, status,
  assumptions, evidence, coverage, missing fields, non-equivalences, source type and residual expression.
  No agent is implemented or connected.

## 5. Synthetic and illustrative fixtures

Every numeric fixture is labelled where it lives:

* `examples/configs/*.json` hold illustrative parameter sets, not defaults and not an acceptance workload.
  Configuration 04 deliberately leaves tiles, hardware and active lengths undeclared to demonstrate partial
  analysis; 90 and 91 are expected-error fixtures; 92 demonstrates a reported conflict.
* `examples/hardware/example_profile_template.json` has every value null.
* `examples/hardware/tpu_v6e_illustrative.json` holds public figures typed by hand without network access and
  marked unverified in the profile's own provenance field.  Vector throughputs and register capacities are null.
* `examples/hardware/synthetic_calibrated_profile.json` holds invented numbers whose only purpose is to
  exercise the calibrated path.  Its bucket declares that it was not validated on unseen shapes.
* Test seeds and tolerances are declared per file and marked test-local.

## 6. What is explicitly not claimed

* No production kernel, no target compilation.
* No TPU execution and no measured speedup.  Nothing here has run on a TPU.
* No real low-level output ingestion; the compile-evidence path has only seen synthetic dictionaries.
* No live agent or language-model integration.
* No minimum-spill proof.  The minimum extra movement states the optimisation problem and stays unknown.
* The v6e profile's numbers are unverified transcriptions, and the calibrated interval is a model output over
  a synthetic profile, not a prediction validated on unseen shapes.
* `mode = calibrated` means at least one bucket matched.  A bucket is applied per resource class — matmul,
  exp, vector, reduce, layout — against that class's own throughput, and every class with no applicable
  bucket is listed `uncovered` and left out of the interval, as are the transfer paths, which have no
  consumer for an `hbm_transfer` bucket yet.  The interval also carries the launch overhead and the
  model-error term of specification 9.3, each named missing when it is not declared.

## 7. Review performed on this deliverable

The implementation was reviewed by a fan-out of independent agents against the specification and the prompt,
across six lenses: specification sections 0 to 4, 5 to 8, and 9 to 14 plus the prompt, mathematical
re-derivation, a hidden-default audit, and software robustness.  That produced 100 raw findings.  Each was
then put to two adversarial refuters.  Ten survived both refuters and eleven survived one; the rest were
refuted.  All 21 surviving findings were fixed, along with several refuted-but-real issues noticed while
fixing them.

The most consequential fixes:

- **Absorbed-path work difference.** The metric had subtracted an expanded total computed with zero key and
  value projection, which flipped its sign in absorbed reports. It is now built from the projection rows the
  declared scope implies, independently of which variant's graph is in hand.
- **Materialized key/value read traffic** was over-counted by a factor of the batch size.
- **Conditional vector operations.** Casts, the mask and the positional branch were counted even when the
  declared dtypes or a zero positional width made them impossible.
- **A family of silent defaults** — active lengths, projection scope, output scope, overlap model, layout
  kind, alias flags, budget level and calibration startup terms — now produce `partial` status with named
  missing fields instead of a bound-looking number.

A second sweep then ran five independent lenses (mathematics, hidden defaults, specification compliance,
robustness, report integrity) over the patched code and returned 33 findings. All 33 are now closed. The
ones worth naming:

- **A scheduled grid wider than the active extents** fell back to walking every rectangle, which hangs at
  large sizes. The closed-form evaluator now covers that case: padding rectangles occupy three quadrants
  counted in constant time, and only the last active block on each axis can straddle the boundary.
- **Key/value tiles projected inside the loop** were charged once per row rather than once per query block
  that scans them, which specification 5.2 requires; tiles no block scans are now charged nothing.
- **A mistyped active-length key** was ignored while the report still called the extent declared. Unknown
  keys are now rejected.
- **A profile with a zero throughput** reached a division; non-positive rates are now rejected on load.
- **The alias-deduplicated allocation total** depended on the order roles appeared in the metadata; an alias
  group is now charged its largest member.
- Undeclared rank sub-tiles, projected-operand dtypes, layout kinds and overlap model all produced
  bound-looking numbers; each now marks the metrics it sizes as partial and names itself.

Closing those findings surfaced a performance defect of my own: the per-level envelope extraction called
sympy's `simplify` on every coefficient and re-derived the whole decomposition once per object. Removing
both, and memoizing the structural expressions, made `analyze()` about seven times faster with output
verified byte-identical across 30 reports spanning three algorithm variants, six strategies and twelve
random configurations.

A third sweep audited every numbered section of the specification in turn: eleven auditors, each finding
verified by two independent refuters and a completeness critic over the whole audit. 25 findings survived
verification and the critic added 6 more; all are closed, and `tests/test_spec_compliance_round3.py` holds a
regression for each. The ones that changed reported numbers rather than labels:

- **A declared non-resident Q retired the loop-carried softmax state**, so every pressure figure was
  understated whenever `q_resident=False`. Residency now shortens Q operands only.
- **The feasible tile bound was evaluated for the softmax stage alone.** It is now computed for every stage
  and the binding one is named. On the shipped calibrated example the binding stage is the PV stage, whose
  bound is 51 rows against the 77 the softmax stage alone reported.
- **The absorbed variants were charged a full key/value projection** they never perform.
- **Declaring an intermediate materialized moved no bytes** on any transfer path.
- **The padded rectangle area ignored the declared extent policy**, and a full scan left the future
  rectangles it executes unclassified.
- **Query-side reads were charged a full tile past the tail**, which specification 4.2 warns against.
- **A resource outside the declared serial groups was dropped from the lower bound**, producing a combined
  bound smaller than one of its own terms.
- **Declared cast locations changed nothing**, and calibration buckets for anything but matrix multiplication
  were ignored.
- **Declaring only one active length** let the other axis fall back to the tensor shape while the metric
  still claimed to be bound.

A fourth round then verified the third round's own changes: twelve independent verifiers, one per fix area
plus sweeps for report invariants, silent defaults, cross-ledger consistency and documentation accuracy. They
returned 91 problems in the work just done, three of them blocking, and all are closed, with regressions in
`tests/test_verify_round4.py` (31 tests; several cover more than one closely related problem). The blocking three were an axis declared with a null value counting as
declared, a resource combination that discarded every known bound when no declared group had one, and a
combined bound that inherited only its unknown terms' caveats and so published a sum of partial terms as
complete.

A fifth round put the fourth round's own changes to seven more independent verifiers, one per fix area, each
probing from the specification with its own scripts. They returned 66 problems, sixteen major, and all are
closed with regressions in `tests/test_round5_*.py` (225 tests: one file per area plus one for the eleven residuals the
regression-writing pass itself surfaced; several cover more than one closely related problem). The ones that changed numbers or statuses: the combined lower bound was
labelled `bound` while its terms were partial; a zero-width positional branch was charged a matmul start-up in
the calibrated prediction; vector-class calibration buckets were matched against the matmul dtype instead of
the dtype the class executes in; no comparison candidate was ever flagged model-infeasible because the filter
tested a dict's truthiness; absence in a comparison depended on which other candidates were listed; an
undeclared active axis beside a declared scheduled extent produced useful work larger than the executed
rectangle work; cast sites were checked against the dtypes at two of six sites; and non-boolean stand-ins
for the residency and aliasing flags were honoured as `true`. The rest were explicit-failure gaps (raw
Python exceptions on malformed input) and label disagreements between two surfaces of one report, listed in
`docs/SPEC_REVIEW.md`. Three whole-report sweeps of that round could not finish on this account's usage
limit; the equivalent out-of-suite scripts were re-run afterwards and were clean.

`tests/test_docs_match_reality.py` now re-derives every figure this document quotes, so these numbers cannot
go stale without the suite failing.

`docs/SPEC_REVIEW.md` records the discrepancies and assumptions, including the full list from both rounds.

## 8. Exact commands

```bash
cd /Users/ziconghe/Desktop/company-project/math-model
python3 -m pytest tests -q
python3 examples/run_examples.py
python3 -m mla_model describe --algorithm expanded --format md
python3 -m mla_model analyze --config examples/configs/01_square_causal_prefill_expanded.json --format md
python3 -m mla_model compare --config examples/configs/06_calibrated_synthetic_expanded.json --format md
```

Separate input files instead of a bundle:

```bash
python3 -m mla_model analyze --metadata t.json --semantics s.json --implementation i.json --numerics n.json --hardware examples/hardware/tpu_v6e_illustrative.json --algorithm absorbed_two_step --format json --out report.json
```

## 9. Suggested next steps (not done here)

1. Attach real compile evidence and compare compiled work with the executed-graph total. Projection counts,
   cast locations and alias groups are the fields most likely to need correction.
2. Calibrate on the target device: fill in vector throughputs and register capacities, add matrix-unit buckets
   whose shape ranges match real tile shapes, and record a validation error on unseen shapes before trusting
   any ranking.
3. Implement the specification's small-graph movement solver as a separate module reporting a lower bound, a
   feasible solution and the gap, keeping it outside the analytical core.
4. Add adapters for the entry scopes this one rejects: pre-expanded keys and values, packed sequences, and
   quantized or sharded storage, each with its own declared capability list.
5. Replace the proportional cached/new split of executed key-column visits with the exact per-block split
   against the cache boundary; today it is a labelled scenario and the boundary position is named missing.
6. Fold layout coverage into the capacity constraint, so `max(0, M_live - R)` can be evaluated on laid-out
   sizes as well as logical ones.
