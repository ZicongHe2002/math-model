# Specification review: discrepancies, ambiguities, and assumptions made

Sources read in full: `MLA_Forward_Parametric_Mathematical_Model.v3.en.md`, its Word counterpart
`MLA_Forward_Parametric_Mathematical_Model.v3.en.docx`, and
`MLA_Parametric_Model_Implementation_Prompt.v3.en(1).md`.

## Document-level discrepancies

| # | Observation | Handling |
|---|---|---|
| D1 | The implementation prompt says to use `MLA_Forward_Parametric_Mathematical_Model.v3.zh-CN.md` and to "read the Chinese specification directly". Only the English `v3.en.md` and `v3.en.docx` exist in the workspace. | The English specification was used as the authoritative text. If a zh-CN version exists elsewhere, differences must be re-checked. |
| D2 | `.docx` vs `.md`: a text diff of the pandoc plain-text renderings shows only LaTeX-rendering differences (`\left\lbrack` vs `[`, `\leq` vs `\le`); no sentence differs. | Treated as identical. |
| D3 | The prompt's expression `C_valid for uniform simple causal alignment = B*H*sum_i max(0, min(Sk, i + offset + 1))` and spec 4.1 agree; both use `j <= i + Delta`. | Implemented as `CausalVisibleCount`. |
| D4 | Spec 5.2 boxed `F_E` and 5.4 `F_A` assume "no existing projection, each item computed once". The prompt repeats them. Spec 5.3 generalises with projected-row counts. | `F_total[useful]` uses the 5.3 row counts from the declared `projection_scope`; the boxed formulas are emitted separately as `F_E[spec_closed_form]` / `F_A[spec_closed_form]` and coincide when the scope is `full`. |
| D5 | Spec 5.4 derivative `F_A - F_E = 2BH R_k (D_n + D_v)(S_q - S_k) + 2C(2R_k - D_n - D_v)` — verified algebraically from `F_A` and `F_E` (see `docs/FORMULA_INVENTORY.md`). | Both the general-row and the closed form are reported. |
| D6 | Spec 5.5 table gives "rescale exponential alpha: at most n times; initial/empty blocks may be special-cased". | Counted as exactly `n` per block visit (upper bound), labeled. |
| D7 | Spec 7.4 says the coefficients `a_s..d_s` "should be derived from the object list and calibrated when necessary". No calibration procedure is specified. | Derivation implemented (polynomial coefficient extraction with provenance); calibration hooks are not implemented (documented as future work). |
| D8 | Spec 8.3 describes a solvable minimum-movement problem for small graphs (state search / ILP). The prompt says this solver must not block the core. | Not implemented. `M_spill_peak`, `B_spill_fill`, `dT_spill` are reported as unknown definitions; the unconstrained whole-kernel minimum spill stays unknown. |
| D9 | Spec 9.1 mentions `TpuInfo` as an optional runtime adapter. | `hardware.from_runtime_jax()` lazily imports JAX and returns `None` off-TPU; only exercised on a CPU host (returns `None`). Fields it might fill are limited to capacities it actually reports. |
| D10 | Spec 6.3 example `B_KV,read = BH ceil(S_q/b_q) S_k (s_K D_n + s_V D_v)` assumes *expanded* K/V inputs; the seven-input interface carries latents. | Without materialization the transfer ledger streams the latent `C_k` and `K^r` per executed key column; with `materialize.expanded_kv = True` a projection pass writes `K^n|V` once and the loop reads expanded tiles per executed key column per head (`H * key_visits * (...)`, which equals the spec's `BH ceil(S_q/b_q) S_k (...)` under `full_scan`). |
| D11 | Spec 11.4 metric fields: `metric, formula_id, expression, bindings, value, unit, scope, status, missing_fields, assumptions, not_equivalent_to`; spec 2 additionally lists `evidence, coverage`. | All fields present in `report.Metric`, plus `source_type` (input/derived/conditional/compiled/measured/definition), `residual`, `section`, `notes`. |
| D12 | Spec 3.1: all-masked rows require a declared convention. | `empty_row_policy` is mandatory when such rows exist (including `S_k = 0`): the report emits a conflict when rows without visible keys exist and the policy is missing or `error`. References implement both conventions. |
| D16 | Spec 1.2: an operator that already receives Q and expanded K/V "must select another entry scope; projection must not be counted again". | `projection_scope = none` is rejected by the seven-input adapter (`CapabilityError`) instead of being modeled with the wrong tensors; an expanded-input adapter would be a separate entry scope. |
| D17 | Spec 1.3: without effective lengths, output a "capacity-execution scenario" and the missing fields. | When `active_lengths` is undeclared the shape extents are used, every dependent metric is `partial` with `active_lengths` in `missing_fields`, and the header/warnings state the scenario. |
| D18 | Spec 7.3: VMEM and VREG are separate; spec 7.4 budgets are per level. | Local objects carry a level assignment; live sets, peaks, envelopes, feasibility and capacity constraints are reported per level (`vreg`, `vmem`, `all`) against `R_vreg` / `R_vmem` only. |
| D19 | Spec 9.2: "sum the lower bounds of stages known to be serial". | `serial_stage_sum` groups *resource classes* over whole-kernel totals (a serial-resource model); grouping by pipeline stage is not implemented and is stated in the metric notes. |
| D13 | The prompt lists `LSE convention` as a configuration field; spec 3.3 says exp2 must restore natural-log LSE. | `lse_convention` is a declared field (`natural_log`/`log2`); the references compute in the declared internal base and always restore natural-log LSE, and `reference.convert_lse(lse, convention)` converts to the declared output convention. The exp2 path also counts a restore operation in the vector graph. |
| D20 | Spec 1.3 names three extents (allocation, logical active, scheduled). | `implementation.scheduled_lengths` carries the third. The rectangle grid spans the scheduled extents while visibility stays on the active ones: blocks past the active region are `padding` (skipped by `skip_future`, counted as `padding_skipped`), straddling blocks are at most `partial`. Undeclared, the scheduled extent equals the active extent and is labelled a scenario. Validated as active ≤ scheduled ≤ capacity on axes whose active length is declared; on an undeclared axis a declared scheduled extent bounds the capacity-execution scenario instead (active = scheduled there, the axis stays named missing), so the scenario itself never violates the ordering. |
| D21 | Spec 6.3 expresses reuse through event counts, not a blanket factor. | Per-head events (weights, `Q^r`) are counted as `H * n_qblocks`, which is exact even when `heads_per_program` does not divide `H`; shared latent / `K^r` loads scale with `ceil(H / h_pp)`. |
| D14 | Spec 1.2: "Axes with the same dimension but different semantics must not automatically be treated as the same axis". | Axis matching is by declared axis *name* only; when `axes` is absent the canonical order is assumed for that role (a declared contract, not an inference from extents). |
| D15 | Spec 2 requires the evaluator to support "finite sums". | Finite sums appear as explicit sympy `Add` over concrete batches (ragged) and as the closed-form aggregate `CausalVisibleCount`; no symbolic `Sum` object is exposed because both use cases have concrete bounds when evaluated. |

## Adversarial review round (2026-09-08)

Six independent review lenses (spec §0–4, §5–8, §9–14 + prompt, mathematics, hidden-defaults audit, software robustness) produced 100 findings; each was then challenged by two refuters (spec-text and code-execution lens).  The findings judged valid were fixed in the same session; the main ones:

* `F_A − F_E` in absorbed reports used `N_K = N_V = 0` for the expanded side (graph-derived rows) → rows now come from the declared projection scope independently of the variant graph.
* Materialized K/V read traffic multiplied the whole-batch tensor by `B ceil(S_q/b_q)` (an extra factor `B`) → reads are now per executed key column per head.
* Undeclared fields (`active_lengths`, `projection_scope`, `outputs`, `score_alias_exp`, `exp_alias_p_operand`, `materialize.*`, layout kinds, `overlap_model`, `recurrence`, `exp_base`, `projection_in_kv_loop`, `scratch_bytes`, `kv_buffers`, `heads_per_program`, calibration startup / launch terms) silently selected a value while the metric said `bound` → every such value is now `partial` with the field in `missing_fields`; undeclared overlap reports both combination scenarios.
* Conditional vector ops (casts, QK-branch combine, mask) were counted even when the declared dtypes / `D_r` made them impossible → counted only when the condition holds; mask counted on not-fully-visible rectangles.
* `compare()` judged candidate feasibility with the baseline's `b_k`, ignored an `algorithm` key, and accepted numerics keys silently → fixed; candidates may switch algorithm, any other non-strategy key raises.
* Sub-block padded statistics under `skip_future_subdivide` used a grid heuristic → executed extents are carried on each block.
* `cached_tokens > S_k` produced negative projected rows; ragged batches without offsets crashed; zero active lengths bypassed the capability gate; `capacity_shape` conflicts were merged by `max`; `per_batch_offsets` silently overrode `position_offset` → all are explicit errors now.
* Fixed scratch never reached the envelope's `d`; loop-carried state was not live at the in-loop `kv_proj` cut; `lse_tile` was counted when LSE was not an output; VMEM→VREG / VREG→VMEM paths were never reported → fixed (the register paths are labelled scenarios / unknown).
* Storage widths of projected operands were inferred from `matmul_input_dtype` without saying so → explicit `projected_operand_dtype` / `projection_accumulator_dtype` fields; the documented fallback is recorded as an assumption in the report header.
* The CLI ignored `hardware_file`; calibration buckets were not validated; layout/throughput dtype keys were not normalized; `algorithm` defaulted to `expanded` → fixed (the silent default was removed). The third round then made `algorithm` *optional but never defaulted*: omitted, the variant is preserved as unknown and reported as named scenarios, as specification 11.1 allows.

Fixed after that round, in the same spirit: attention matmul nodes now carry their executed per-visit tile
shape `(b_q, b_k, D)` so calibration buckets match on real shapes while the FLOP total stays the executed-cell
total; `analyze()` emits a `dependency_graph` extra (metric → free symbols, plus matmul, local-object and
transfer-event node lists); the references gained `convert_lse`; the spec 5.5 items listed separately
(`broadcast`, `transpose`, the `alpha*l` multiply, and the exp2 LSE restore) are emitted, with the two that
depend on unobservable backend behaviour marked undeclared; sparsity metadata is rejected by the adapter;
`scheduled_lengths` implements the third extent (D20); per-head transfer counts no longer assume
`h_pp | H` (D21).

Findings deliberately **not** changed (with reasons): `serial_stage_sum` remains a resource-class model rather
than a per-pipeline-stage sum (see D19); the declared scale now takes precedence on a conflict with the standard
policy (the conflict is still reported with both values, per the prompt's instruction to report conflicts rather
than overwrite); the closed-form visibility evaluator does not cover a scheduled grid wider than the active
extents, which falls back to rectangle enumeration (correct, but linear in the rectangle count).

## Second adversarial round (five lenses over the patched code)

Five independent lenses — mathematics, hidden defaults, specification compliance, robustness, report
integrity — returned 33 findings; all are closed.  Beyond the items already listed above:

* The closed-form block-statistics evaluator now covers a scheduled grid wider than the active extents
  (previously it fell back to rectangle enumeration, which hangs at large sizes).  Padding rectangles form
  three quadrants counted in constant time; only the last active-origin block on each axis can straddle the
  boundary, so straddling is two scalars rather than a per-block predicate.  Verified against brute-force
  enumeration on 1.6 million scheduled configurations with zero mismatches out of suite; a 1,200-case subset
  is committed as a test.
* K/V tiles projected inside the KV loop are charged once per query block that scans them (spec 5.2 loop
  multiplicity); tiles no block scans are charged nothing.  Neither direction dominates the once-per-row
  accounting, so `F_total[rect]` and `F_total[executed_graph]` are not ordered in general — they agree when
  the expanded cache is materialized and the extents are logical, which is what the tests assert.
* `active_lengths` with an unknown key, a hardware rate of zero or less, and an unknown layout kind are all
  rejected instead of being partly ignored or reaching a division.
* An alias group is charged its largest member, so the allocation total no longer depends on metadata order.
* Undeclared rank sub-tiles (`b_rq`, `b_rk`), the documented projected-operand and projection-accumulator
  dtype fallbacks, and the undeclared layout kind now mark the metrics they size as `partial` and name
  themselves, instead of producing a bound-looking number.
* Statuses are confined to the declared vocabulary, every path bound names its own bandwidth, and the
  combined lower bound inherits the unmet requirements of the terms it combines.

Still true after this round: `serial_stage_sum` remains a resource-class model (D19); the brute-force
enumerator stays the oracle for the closed form and now refuses, with an explicit error, to walk a grid
above a million rectangles rather than appearing to hang.

## Third adversarial round (whole specification, section by section)

Eleven auditors covered every numbered section against the specification text, each verified by two
independent refuters (a spec-reading lens and a reproduction lens) plus a completeness critic over the whole
audit. 25 findings survived verification and 6 more came from the critic; all are closed. What changed:

* **Live sets are per object again.** A declared `q_resident=False` used to retire the loop-carried softmax
  state (`m`, `l`, `A`) along with the Q operands, understating every pressure figure. Residency now shortens
  Q operands only, in the absorbed variants as well as the expanded one, and an undeclared residency is named
  on every live set, peak, envelope and feasibility bound it shapes.
* **The b_k bound is per stage.** Specification 7.4 states the envelope and the feasible range for a stage
  `s`; only the softmax stage was evaluated. Every stage now has its own bound, the binding (smallest) one is
  named, a stage with no `b_k` term is `not_applicable` rather than a division by zero, and the comparison's
  feasibility verdict follows the binding stage. On the shipped calibrated example the binding stage is the
  PV stage at 51 rows, while the softmax stage alone would have reported 77.
* **Declared cast locations now change the vector graph** (specification 3.3). `cast_points` was accepted and
  never read; it is now a validated set of sites in the modelled dataflow, each emitting its own cast op.
* **The projected-row counts and the projection work follow the variant.** The absorbed paths consume `C_k`
  directly, so `N_Kproj = N_Vproj = 0` there; they used to be charged a full K/V projection. `F_proj[general]`
  became variant-dependent with them: the two-step path charges the Q projection plus the absorb step, and the
  precomputed path charges the single merged projection instead of both. Where the executed graph projects a
  row more than once, or skips a tile nothing scans, the executed count is reported beside the once-per-row
  one as `N_*proj[executed_graph]`. The forms are tabulated in `docs/FORMULA_INVENTORY.md`.
* **Materialized intermediates move bytes.** Declaring `Q^n`, `Q~` or `Z` materialized produced no transfer
  event; each write and each read is now counted on its path, and the materialized tensor's own size
  (`M_mat[...]`) is reported beside its traffic.
* **The padded rectangle area follows the declared extent policy**, both scenarios are emitted when it is
  undeclared, and the selected set is now fully classified: visible, partial, fully-future-executed and
  padding-executed sum to the selected count (checked against the package's brute-force enumerator over
  188,160 cases out of suite; a smaller sweep is committed in `tests/test_visibility.py`).
* **Query-side reads are counted per row, not per padded tile.** Specification 4.2 warns that logical
  out-of-bounds is not an HBM read; the tile-padded read is now a separate, labelled scenario event.
* **Resource coverage in the lower bound.** A resource outside the declared serial groups is no longer
  dropped from the combination; the layout class has its own bound and names its missing throughput; a group
  whose value is unknown is omitted from the sum with the omission printed, since the partial sum is still a
  valid lower bound.
* **Calibration.** Buckets for `exp`, `reduce`, `layout` and `vector` are applied to their own resource
  classes instead of being ignored; the model-error term of specification 9.3 is a declarable quantity and is
  named missing when absent; a bucket claiming validation on unseen shapes must carry the validation error; a
  bucket whose device or toolchain does not identify this profile is applied but the mismatch is named.
* **Four spill quantities**, not three: the static instruction count joins the allocation peak, the dynamic
  bytes and the exposed time, and the exposed time stays partial until its comparison graph is declared.
* **Per-axis effective lengths.** Declaring only `Sq` used to let `Sk` fall back to the shape extent while
  the metric still claimed to be bound. Each axis is now declared or a named scenario.
* **Per-tensor allocation.** A role that declares no `capacity_shape` is charged its own shape extent instead
  of another role's declared capacity, so the symbolic byte metrics agree with the per-tensor ledger.
* **The optimisation surface.** The Pareto set and the ranking now range over the mathematically legal
  candidates only, with exclusions printed; the default objectives include reduction counts; the feasible
  region is a `b_q` sweep rather than one point; and the three evidence templates of specification 10.3 are
  each produced in full, including a declared `v_load` policy with lifetime and prefetch metrics.
* **The algorithm may be preserved as unknown** (specification 11.1). The recommended API call of 11.2 now
  runs verbatim, and `python3 -m mla_model analyze` accepts the same: variant-dependent metrics appear once
  per named variant scenario with `algorithm` missing. `compare` still needs a declared variant, because a
  merged scenario report has no single metric to difference; it says so rather than producing empty rows.
* Smaller closures: unread tensor metadata keys are reported instead of absorbed; a layout declared for an
  id no variant owns is rejected, while an id another variant owns (or one this configuration declares away)
  is reported `not_applicable`; the cache kind is validated; `serial_stage_groups` entries must be
  resource classes; a stride or storage offset whose span exceeds `prod(shape)` is recorded as an unmodelled
  input while a contiguous declaration leaves the allocation figure bound; an alias group is sized by
  the whole group at every cut; LSE production and both final-normalisation scenarios are counted as vector
  work; the merge work is reported undivided beside the reuse count and the merged-weight storage; the
  excluded pipeline stages are named in every report; and every metric carries an algorithm path.

## Fourth round: verifying the third round's own fixes

The third round's changes were then put to twelve independent verifiers — one per fix area, plus sweeps for
report invariants, silent defaults, cross-ledger consistency and documentation accuracy. They returned 91
problems in the work just done, three of them blocking, and all are closed.
`tests/test_verify_round4.py` holds the regressions (several cover more than one closely related problem);
`tests/test_docs_match_reality.py` re-derives every
figure these documents quote, so a document cannot drift from the code without the suite failing.

The three blocking ones were mine, introduced by the third round:

* **An axis declared with a null value** counted as declared in one module and undeclared in the other, so
  the natural JSON for "I know the query length, not the key length" produced a full-capacity figure labelled
  `bound`. Both modules now use one predicate.
* **The resource combination discarded every known bound** whenever no declared serial group happened to
  produce a value: the guard sat in front of the code meant to keep the unnamed resources. The combination is
  now never smaller than its own largest term, which a test asserts over four grouping declarations.
* **The combined bound inherited only the unmet requirements of its unknown terms**, so a sum of eight
  partial terms was published as complete. It now inherits from every term that enters it.

Others worth naming: a candidate patch shallow-merged the mapping-valued strategy fields and silently
un-declared the rest; whether a metric read as "absent" depended on the order the candidates were listed in;
the comparison embedded a live sentinel object that plain JSON could not serialise; compile evidence did not
survive the derivation that builds a candidate, so only the baseline could ever carry a backend label; the
two mutually exclusive normalisation forms of specification 5.5 were added together rather than chosen
between; a lifetime scenario was named at stages where both branches agree; the query-side read charged the
logical rows and the padded tail at once, which came to the same total it was meant to replace; and a
per-path lower bound did not inherit what its byte total rested on, so dropping a declaration changed a value
that still called itself complete.

Still true after this round: `serial_stage_sum` remains a resource-class model (D19); the minimum-movement
solver of 8.3 is defined, not solved; low-level compiler output enters only through the evidence hooks; the
cached/new key split is apportioned by the global new-token share and says so; and `hbm_transfer` calibration
buckets are accepted by the profile but have no consumer, since transfer time is not in the predicted
interval.

## Fifth round: a final double check of the fourth round's fixes

The fourth round's own changes were then verified once more by seven independent verifiers, one per fix area
(lower bounds and calibration, byte ledgers and materialization, numerics and the vector graph, the
comparison, declarations and validation, per-stage pressure, residency), each working from the specification
and from probes it wrote itself. They returned 66 problems, sixteen of them major; all are closed and
`tests/test_round5_*.py` hold the regressions, one file per area plus one for the residuals described below.
Three whole-report sweeps (documentation,
invariants, cross-ledger consistency) could not complete on this account's spend limit; the equivalent
sweeps were re-run as the out-of-suite scripts described in `docs/HANDOFF.md` and were clean.

The ones that changed reported numbers or statuses:

* **The combined lower bound was labelled `bound` while its terms were partial**: the status was computed
  before the union of the terms' missing fields was formed. It now follows that union; because the critical
  path is never declared the combination is never `bound`, and the report says why.
* **A positional branch with `D_r = 0` was charged a full matmul start-up** in the calibrated prediction
  although the graph itself marks it absent. Zero-work nodes are now reported absent and charge nothing.
* **Vector-class calibration buckets were matched against the matmul operand dtype**, so a bucket declared
  for the dtype the exponentials actually execute in was dropped as inapplicable. Each class is now matched
  against its own dtype (`exp_dtype`, the score dtype, the state dtype).
* **No candidate was ever flagged model-infeasible**: the legality rows had become per-level dicts and the
  filter tested their truthiness. It now tests the verdict field; a `b_k` above the binding bound and a stage
  over budget independently of `b_k` both reach `model_infeasible_candidates`.
* **Absence in a comparison depended on which other candidates were listed**: a name some other row emitted
  was declared absent, so the same absorbed row read `?` alone and `n/a` beside an expanded candidate, and a
  scenario spelling of an undeclared strategy field read as a structural disappearance. Absence is now a
  property of the row: every variant emits a `not_applicable` metric for each quantity it structurally lacks
  and the comparison reads that.
* **An undeclared active axis beside a declared scheduled extent** kept the capacity as the active scenario
  while the grid iterated the scheduled extent, printing useful work larger than the executed rectangle work
  and a negative waste. The scheduled extent now bounds the scenario on such an axis (D20).
* **Declared cast sites were checked against the declared dtypes at two of the six sites.** All six are
  checked, and the absorbed output cast compares the projection accumulator, not the attention accumulator.
* **Non-boolean stand-ins for the residency and aliasing flags** (`0`, `"false"`, `"yes"`) fell into the
  resident / non-aliased branch and were reported `bound`. Every boolean-or-absent strategy field is now
  validated, as the materialization flags already were.
* **The `M_in` metric published an expression and bindings that evaluated to a different number than its
  value** when tensors carried their own capacities; it is now the sum of per-role terms each substituted at
  its own extent, and an assertion keeps expression and value equal.
* **Alias chains and cycles were not collapsed** into one storage group, over-counting shared allocation; a
  self-alias was accepted. Aliases now resolve to a root, cycles and self-aliases are rejected.
* **The vmem-level pressure metrics were downgraded by the vreg-only aliasing flags**; the fallback now
  applies to the register level only. The feasible-region rows now use the same status rule and the same
  fallback fields as the metric of the same number.

Writing the regressions for these found eleven residuals of the fixes themselves, all closed and pinned in
`tests/test_round5_residuals.py`: a candidate whose `name` is not a string and a non-list `objectives` argument
escaped `compare()` as raw exceptions; the notes about declared cast sites reached one FLOP metric's
assumptions but not the warnings; the pressure metrics still spelled a missing input by its internal symbol
(`n_buf`, `R_vmem`) while the feasible-region rows spelled the field; a non-numeric or boolean `scale_value`
reached the scale getter unchecked; an empty list or zero given as `dtype_bytes_overrides` read as "no
overrides"; a layout declared for an object this variant declares away (`both_qk_branches_live = false`,
`D_r = 0`) was worded as another variant's; the active-length assumption still said "the tensor extent" on an
axis where the scheduled extent had bounded the scenario; a layout calibration bucket declaring a dtype was
applied without the unverified dtype match being named; a scalar `m_range` and a string profile `toolchain`
escaped as raw exceptions; and the consumer-less declarations (`path:vreg_to_vmem` in a serial group, a
`vreg_to_vmem` bandwidth, an `hbm_transfer` bucket) were accepted without a word, which the lower-bound and
prediction notes now say.

The rest were labels, wording and explicit failure: malformed calibration ranges, strides, capacity shapes,
storage offsets, alias targets, per-batch offsets, position starts, cast-site lists, serial-group entries
and comparison inputs all escaped as raw Python exceptions and now raise `MetadataError`/`BindingError`;
negative time terms and zero layout tiles are rejected on load; a null scheduled axis counts as undeclared
like a null active axis; the fingerprint and the header use one per-axis declared predicate; a
disabled materialization clears the extent caveats it cannot depend on; the materialization-traffic rows are
flagged by the grid only where their formula reads the grid; ragged tasks report the scheduled extents
`not_applicable` instead of asking for a field they cannot declare; a materialization key or layout id of
another variant is reported `not_applicable` rather than silently ignored, and `_known_local_ids` is built
over the declaration alternatives so the projection-pass tile is recognised; the extent metrics of the task
section name their own axis only; `subdivide` without its policy is noted; the vector-graph notes reach the
warnings; the both-absent finite difference uses the canonical absence encoding; duplicate candidate names
are refused; a candidate naming `algorithm: null` keeps the baseline variant; the byte-width overrides given
to `bind()` go through the same validation as the numerics field; and the binding-stage metric names the
stages that are over budget for every `b_k`.

Still true after this round: `serial_stage_sum` remains a resource-class model (D19); the minimum-movement
solver of 8.3 is defined, not solved; low-level compiler output enters only through the evidence hooks; the
cached/new key split is apportioned by the global new-token share and says so; `hbm_transfer` calibration
buckets, the `path:vreg_to_vmem` serial class and the `vreg_to_vmem` bandwidth are accepted but have no
consumer in the modelled graph, and the report says so where they are declared.

## Interpretation choices exposed as assumptions in reports

* **Loop nest scenario** `q_outer_kv_inner`: one program per (batch, group of `h_pp` heads, q-block) scanning the selected KV blocks. Transfer counts and `n_programs` are labeled with this scenario; alternative nests are not modeled.
* **Weights re-read per program** (no cross-program residency assumed) — labeled scenario on `W*_read` events.
* **Output writes** counted once per active row (tail tiles write valid rows only).
* **Undeclared alias flags** (`score_alias_exp`, `exp_alias_p_operand`, `both_qk_branches_live`): the worst case (no aliasing, both branches live) is included and every dependent live-set / envelope / feasibility metric is `partial` with the flag in `missing_fields`.
* **Projection scope undeclared**: full projection shown; all work metrics `partial` with `projection_scope` missing.
* **Storage levels**: `vmem` for residency windows / weights / scratch / output tiles, `vreg` for vector temporaries and accumulators — an analysis assignment stated in every report.
* **Standard scale** is only used when `scale_policy = standard`; a bare `scale_value` is treated as declared. If both are present and inconsistent beyond `scale_match_rel_tol`, the declared value takes precedence and a conflict records both values and the tolerance used; neither source is overwritten silently.
* **Dense mask** ignores the offset (fixed to 0 in block statistics) so an undeclared offset never blocks a dense analysis.

## Items intentionally left unknown (never filled)

Tile sizes, buffer counts, heads per program, layout kinds, budgets, throughputs, bandwidths, register capacities, calibration efficiencies, critical path, compiled work, spill quantities, replication factors, and `n_reuse` for precomputed merged weights.

## Open questions for the specification owner

1. Whether the `zh-CN` specification differs from the English one in any formula.
2. Whether the proportional cached/new split of executed key-column visits is acceptable, or the exact
   per-block split against the cache boundary is required (currently a labelled scenario with the boundary
   position named missing).
3. Whether `heads_per_program` should be mandatory for the transfer ledger, or symbolic `h_pp` (current) is acceptable.
4. Whether the `vmem`/`vreg` level assignment of Q operands (currently `vmem`, streamed to registers per visit) matches the intended kernel structure; the assignment is a declared analysis assumption.
