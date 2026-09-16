# Build a parameter-driven mathematical model of MLA forward

Use the attached `MLA_Forward_Parametric_Mathematical_Model.v3.md` or its Word counterpart as the corrected specification. Read the Chinese specification directly; write implementation documentation in English.

## Critical correction to earlier instructions

The user wants a reusable mathematical model, NOT a kernel optimization project for one fixed workload. Previously supplied numerical dimensions, tiles, dtypes, seeds, tolerances, and benchmark settings were illustrative examples of possible fields. They are NOT the actual production parameters, required defaults, or project-wide acceptance workload.

This prompt and the v3 specification supersede previous instructions that freeze those example values. Do not carry those values into model constructors, fallback settings, hardware assumptions, acceptance gates, or the principal test oracle.

Parameters are read from the current invocation and may differ every time. “Random parameters” means the system must handle varying valid configurations; it does not mean you should generate random production inputs instead of reading the supplied inputs. Random generation is useful for testing the model, not for substituting for user metadata.

## 1. The actual deliverable

Construct a symbolic and executable analytical model:

    M(task parameters, algorithm, execution strategy,
      numeric policy, hardware profile) -> stage-level analytical report

The primary deliverables are mathematical expressions, their derivations and applicability conditions, parameter extraction, constraint validation, reusable evaluation, and explainable reports. Real kernel rewriting, TPU execution, LLO ingestion, and a live LLM agent are optional later integrations, not prerequisites for the mathematical core.

The same code must work for multiple valid configurations without source edits. The same model object must support multiple independent bindings without retaining stale values from an earlier call.

Deliver three capabilities:

1. Symbolic mode: return expressions and constraints even before dimensions or tiles are bound.
2. Bound mode: infer/read the current parameters and evaluate what is known, preserving unresolved expressions.
3. Calibrated mode: optionally use matching hardware/measurement evidence for latency intervals and candidate comparison. Clearly distinguish this from a hardware-peak lower bound.

## 2. Read before implementing

Read the entire v3 specification and inspect the authorized workspace and existing code. Extract its requirements and formula inventory before implementation, then build and test a working mathematical core rather than returning another plan.

Reuse existing general-purpose code only after inspection. Remove fixed-workload assumptions from the analytical architecture. Preserve earlier artifacts and user changes; do not destructively replace notebooks or unrelated files. A prior test count is not evidence that your implementation passes tests.

Record discrepancies in a specification review file. If a formula or source interpretation is uncertain, expose the assumption and continue independent functionality rather than inventing a value.

## 3. Dynamic input and binding

Implement a metadata adapter for the semantic seven-input latent interface described in the specification:

    Cq: [B, Sq, Rq]
    Ck: [B, Sk, Rk]
    Q_rope: [B, H, Sq, Dr]
    K_rope: [B, Sk, Dr]
    Wq: [H, Rq, Dn]
    Wk: [H, Rk, Dn]
    Wv: [H, Rk, Dv]

These are symbolic role/axis contracts, not fixed numerical shapes. Infer dimensions from the current metadata and cross-check every related axis. Allow explicit semantic axis mappings instead of requiring a physical transpose. Other interfaces need explicit adapters; do not pretend an arbitrary tensor list has the seven-input semantics.

Read dtypes per tensor. Keep logical extents, allocated capacity, active lengths, and executed/padded extents separate. A KV cache capacity is not necessarily the number of keys consumed in the current call. Support or explicitly reject ragged, packed, quantized, transposed, and sharded cases according to the adapter's declared capabilities.

Read causal semantics, absolute position offsets, scale, output scope, LSE convention, cache state, and numerical policy from configuration. Do not infer them solely from shape. A named standard scaling policy may compute 1/sqrt(Dn+Dr), but an actual declared scale takes precedence; report conflicts instead of silently overwriting them.

Read tile sizes, layouts, buffer counts, and scheduling information when provided. Otherwise leave them symbolic or enumerate explicitly named scenarios. Never fill missing fields with values from an old example.

Use metadata-only access for shape/dtype inspection. Do not allocate full-size random arrays, run arbitrary notebook cells, execute configuration strings, or transfer large tensor values merely to determine their shapes.

## 4. Symbolic mathematical architecture

Separate:

    Task and scope
    Algorithm variant
    Numerical semantics
    Tiling/residency/layout/schedule
    Hardware and calibration

Represent expressions structurally with an AST or an appropriate installed symbolic library. Support sums, products, ceil/floor, min/max, predicates, piecewise cases, and partially bound variables. Do not use unrestricted eval on user strings.

Algorithm-specific builders generate a shape-aware graph. Shared primitives compute matmul work, reductions, exponentials, casts, tensor bytes, transfer events, aliases, and liveness. Keep loop multiplicities symbolic when practical; do not expand a large attention problem into all scalar instructions.

Implement expanded MLA first, with absorbed two-step as a separate mathematical variant. Do not build a single formula that swaps an algorithm name while retaining the wrong accumulator width or projection costs.

Important expressions include:

    F_matmul(M, N, K) = 2*M*N*K

    C_valid = sum over all (b,h,i,j) of visibility(b,h,i,j)

    C_valid for uniform simple causal alignment
        = B*H*sum_i max(0, min(Sk, i + offset + 1))

    F_expanded
        = 2*B*H*(Sq*Rq*Dn + Sk*Rk*(Dn+Dv))
          + 2*C*(Dn+Dr+Dv)

    F_absorbed_two_step
        = 2*B*H*(Sq*Rq*Dn + Sq*Rk*(Dn+Dv))
          + 2*C*(2*Rk+Dr)

These total-work formulas assume the specific projection boundary and once-per-row computation stated in the document. Existing caches, repeated projection, or precomputed weights require their own event/row counts. C already includes batches and heads.

Retain separate useful, rectangular-scheduled, and compiled work metrics. Generate mask classifications for arbitrary supported rectangular tiles, offsets, and tails. Do not assume causal=True identifies the implementation's pruning policy. Do not equate valid cells with the number of actual exp instructions.

Implement small-shape independent dense, online, and partition-merge references. Test normalized and unnormalized recurrence as separately identified paths. Account for empty states safely, maintain LSE log-base semantics, and distinguish real-arithmetic equivalence from mixed-precision acceptance.

## 5. Memory, pressure, and spill modeling

Each tensor's logical size is:

    bytes(X) = product(shape(X)) * storage_bytes(dtype(X))

Return symbolic local expressions such as:

    score_bytes = bq * bk * score_element_bytes
    q_rank_tile_bytes = bq * brq * q_element_bytes
    expanded_accumulator_bytes = bq * Dv * accumulator_element_bytes
    absorbed_accumulator_bytes = bq * Rk * accumulator_element_bytes

Do not replace these expressions with historical numeric answers.

Maintain separate ledgers for interface bytes, HBM materialization, conditional transfer requests, VMEM resident/scratch allocations, layout coverage, live storage, spill backing storage, static instructions, dynamic traffic, and exposed latency.

Use transfer events and execution multiplicities. Express reuse/residency explicitly. Deduplicate aliases and do not sum objects that never coexist. A source-level tensor liveness estimate is not compiler register allocation.

Hardware layout functions consume profile parameters rather than hard-coded vector dimensions. A register-tile capacity is not the total register-file capacity. Missing usable-register capacity and uncalibrated throughputs remain unknown.

Implement the specification's conditional stage envelope when its coefficients can be derived:

    estimated_stage_memory = a*bq*bk + b*bq + c*bk + d

Expose its coefficient provenance and feasible parameter region under a declared capacity. Do not label an uncalibrated proxy a measured spill quantity, OOM proof, or spill probability.

Define minimum extra movement only for a specified graph, machine model, allowed transformations, and objective. Keep the unconstrained whole-kernel minimum spill unknown. A later small-graph solver may report a lower bound, feasible solution, and gap; do not make this solver a blocker for the analytical core.

## 6. Hardware and performance mathematics

Load hardware properties externally or through a capability-checked runtime adapter. A TPU v6e profile is one possible input; the algorithm formulas must not depend on a hard-coded device model. Track per-core/per-TensorCore/per-chip scope, precision, toolchain, provenance, and calibration validity.

Implement resource lower bounds and calibrated predictions as different objects. Do not use HBM bandwidth for VMEM/register spill transfers. Count exp, reduction, and layout operations using their corresponding resource models rather than the MXU peak.

Preserve dependencies and serial kernel boundaries. Overlap must follow a declared resource/scheduling model. Avoid counting the same transfer or padding inefficiency twice.

Provide parameter sensitivity, finite differences around discrete tile changes, and resource trade-offs. The expected result is a feasible region or evidence-backed candidate set for the current input, not a universal best tile.

Missing hardware must not block FLOPs and byte accounting. Missing calibration must not block symbolic constraints, but must block claims of precise actual latency or measured performance winners.

## 7. API, reports, and agent boundary

Expose a reusable API equivalent to:

    model.describe(adapter, algorithm)
    model.bind(metadata, semantics, implementation, numerics, hardware)
    bound.analyze()
    bound.compare(candidate_strategies)

Provide a CPU-only CLI and a package invocation that does not require installation. The core must not import JAX or require TPU or an LLM API key. Optional adapters are lazily loaded.

Each metric includes expression, variable bindings, value or null, units, scope, algorithm path, assumptions, evidence, coverage, and missing fields. Preserve expressions after numerical binding so the report remains explainable.

Provide JSON and Markdown output. Include unresolved and conflicting inputs visibly, not just successful numbers. Never use unknown=0 as a fallback.

The agent consumes the mathematical report and proposes experiments or requests missing evidence. Arithmetic, binding, legality checks, and report generation remain deterministic code. Do not spend the first implementation milestone on a general agent framework or automatic kernel rewriting.

Task identity is per invocation: parameters may change between tasks, but candidate speedup comparisons for one task must preserve that task's semantics, numerical acceptance, and output scope. A change in shape is a separate scaling experiment, not a same-task speedup.

## 8. Acceptance across parameter families

Use independently generated valid small configurations, not only the earlier example. Vary batch/head counts, query versus KV length, ranks, non-RoPE/value widths, position dimensions, offsets, dtypes, and tiles, including supported tails.

Compare visibility/block formulas with explicit enumeration. Verify expanded/absorbed algebra in high precision and online/merge outputs against an independent dense reference. Separate mathematical tests from native BF16/device tests.

Add metamorphic tests:

- Rebinding a model does not retain old parameters.
- Changing a task dimension updates every dependent expression.
- Changing a tile does not change the underlying valid attention task.
- Shared input storage and head-dependent computation scale differently where appropriate.
- Changing hardware does not change useful mathematical FLOPs.
- Changing storage dtype does not silently alter the numerical policy.
- Missing tile or device information produces valid partial analysis.
- Contradictory shapes fail explicitly instead of falling back to example values.

Any numeric fixtures must be labeled illustrative. Do not use historic seeds, tolerances, sample counts, or dimensions as mandatory production defaults. Derive correctness requirements from the current scope and documented policy.

## 9. Work and handoff

Use only the authorized workspace, preserve user changes, and avoid downloads, environment upgrades, external calls, or device jobs without authorization. These constraints must not prevent the local mathematical deliverable.

Implement in this order:

    symbolic formulas and dynamic binding
    -> validation and constraint system
    -> algorithm work and hierarchical memory accounting
    -> stage pressure/sensitivity reports
    -> hardware lower bounds and optional calibration
    -> optional source/LLO and agent interfaces

Do not stop at another plan or scaffolding. Deliver working model code, derivations/formula inventory, metadata adapters, tests, multi-configuration examples, generated reports, and exact commands. Clearly distinguish executable functionality, symbolic-only results, optional interfaces, synthetic fixtures, and integrations actually verified.

In the final handoff report test results you actually obtained. Do not claim a production kernel, target compilation, a live agent, a minimum-spill proof, or TPU speedup without evidence.

Begin by reading the corrected specification. Build the reusable parameter-driven mathematical model, not a workload-specific kernel optimization project.
