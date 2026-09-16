# Formula inventory, derivations and applicability conditions

Every metric emitted by `mla_model` carries a `formula_id`.  This inventory lists the
formula behind each id family, its derivation, the conditions under which it applies,
and what it is **not** equivalent to.  Symbols follow the v3 specification:

| symbol | meaning | source |
|---|---|---|
| `B, H` | batch, heads | `q_latent.shape[0]`, `w_q_nope.shape[0]` (cross-checked against every other tensor) |
| `S_q, S_k` | **active** query / key extents (mathematical work) | `semantics.active_lengths`; undeclared leaves every dependent metric `partial` (capacity-execution scenario) |
| `S_q_sched, S_k_sched` | **scheduled** extents the kernel iterates over (spec 1.3 third extent) | `implementation.scheduled_lengths`; default equal to the active extents, labelled as a scenario |
| `R_q, R_k` | Q / KV latent ranks | `q_latent.shape[2]`, `kv_latent.shape[2]` |
| `D_n, D_r, D_v` | non-positional head dim, positional dim, value dim | `w_q_nope.shape[2]`, `q_pe.shape[3]`, `w_v.shape[2]` |
| `Delta` | causal offset: key `j` visible to query `i` iff `j <= i + Delta` | `semantics.position_offset` or `query_position_start - key_position_start` |
| `b_q, b_k, b_rq, b_rk` | Q / KV / rank sub-tiles | `implementation.*` (else symbolic) |
| `n_buf, h_pp` | buffer count of streamed windows, heads per program | `implementation.kv_buffers`, `implementation.heads_per_program` |
| `s_*` | storage bytes per element of each object | per-tensor dtype (inputs) / numeric policy (intermediates) |
| `gamma` | score scale | `scale_policy` (`standard` = `(D_n + D_r)^(-1/2)`, `declared` = `scale_value`) |
| `L_s, L_l` | vector-tile sublane / lane extents | hardware profile `layout[dtype]` |
| `R_vreg`, `R_vmem` | declared budgets per storage level (never substituted for each other) | `implementation.vreg_budget_bytes` / `vmem_budget_bytes`, or profile `vreg.usable_capacity_bytes` / `vmem.scoped_budget_bytes` |
| `P_u, W_e` | resource throughput / path bandwidth upper bounds | hardware profile with scope |

Storage widths of the projected intermediates come from `numerics.projected_operand_dtype` and
`numerics.projection_accumulator_dtype`.  When those are undeclared the model falls back to the matmul
input dtype and the attention accumulator dtype respectively and records the fallback as an assumption
printed in the report, rather than treating it as a declaration.

All expressions are sympy trees built programmatically (`mla_model/symbolic.py`).  The custom
aggregates `CausalVisibleCount(...)` and `BlockStat(...)` stay symbolic until every argument
is an integer, so a partially bound report still shows the structure.  Once bound they evaluate by
closed-form arithmetic over query blocks — never by iterating cells, and (except for a scheduled grid
wider than the active extents) never by iterating rectangles, so a task spanning hundreds of millions of
rectangles still evaluates in milliseconds.  `block_stats_enumerate` is the brute-force oracle the tests
check that arithmetic against.

---

## 1. Visibility (`vis.*`, spec 4)

### `vis.C_valid` — visible cells
`C_valid = sum_{b,h,i,j} mu(b,h,i,j)`  — includes `B` and `H`; never multiply by `BH` again.

* uniform causal: `C_valid = B*H*CausalVisibleCount(S_q, S_k, Delta)` with
  `CausalVisibleCount = sum_{i=0}^{S_q-1} max(0, min(S_k, i + Delta + 1))`.
  Closed form (implemented in `causal_visible_count`): with `a = Delta + 1`, rows split into
  saturated rows (`i >= S_k - a`, each contributing `S_k`), empty rows (`i + a <= 0`), and a middle
  arithmetic progression `sum (i + a)` over `i in [max(0, 1-a), min(S_q-1, S_k-a-1)]`.  Verified against
  brute-force enumeration for all `Delta` in `[-(S_q+3), S_k+3]` (tests) — including
  `Delta < -S_q` (no visible cell) and `Delta >= S_k - 1` (dense).
* uniform dense (`mask = none`): `B*H*S_q*S_k`.
* ragged: `H * sum_b CausalVisibleCount(Sq_b, Sk_b, Delta_b)` (explicit finite sum; `B` concrete).
* Special cases: square prefill `S(S+1)/2` per (b,h); decode `S_q = 1, Delta = L-1` → `L` (not 1).

Applicability: unit-step positions, one offset per batch, no packed segments (adapter rejects them).
Not equivalent to: executed cells, exp count.

### `vis.C_rect`, `vis.C_pad`, `vis.blocks.*` — rectangle set `E`
Block `I_i = [q_i, q_i+n_i)`, `J_j = [k_j, k_j+m_j)`; tails have `n_i = min(b_q, S_q - q_i)`.
Classification (spec 4.2): future iff `k_j > q_i + n_i - 1 + Delta`; visible iff `k_j + m_j - 1 <= q_i + Delta`; else partial.

With a declared scheduled grid the rectangle grid spans `S_q_sched x S_k_sched` while visibility is still
defined on the active extents: a rectangle beyond the active region is classified `padding` (skipped by
`skip_future`, counted by `padding_skipped`), and one straddling the boundary can be at most `partial`, so its
cells appear in `masked_cells`.  `C_valid` never changes with the scheduled grid.

Policies (declared, never inferred from `causal=True`):
* `full_scan`: `E` = all blocks.
* `skip_future`: `E` = all non-future blocks (partial blocks executed whole).
* `skip_future_subdivide (p_q, p_k)`: partial blocks re-tiled at `(p_q, p_k)`; future sub-blocks skipped.

`C_rect = B*H*sum_{E} n_i m_j` (logical extents); `C_pad = B*H*sum_{E} n̂_i m̂_j`, where spec 4.2 requires `n̂, m̂` to come from the declared strategy: under `padded_to_tile` they are `b_q, b_k` (sub-tile extents for sub-blocks) and under `logical` they are `n_i, m_j`, so `C_pad = C_rect`.  With the policy undeclared the padded-tile figure is shown as the scenario, the field is named missing, and both `C_pad[scenario=logical]` and `C_pad[scenario=padded_to_tile]` are emitted.
The selected set is fully classified: `fully_visible_blocks + partial_blocks + future_blocks_selected + padding_blocks_selected = selected_blocks` (`full_scan` executes future and padding blocks; `skip_future` counts them in `future_blocks_skipped` / `padding_blocks_skipped` instead).
Further per-(b,h) statistics used by the graph: `key_visits = sum_{E} m_j` (executed key columns — drives KV read traffic), `row_visits = sum_{E} n_i`,
`masked_cells = sum over selected blocks that are not fully visible of n_i m_j` (cells that need a mask), and their padded variants.
`executed_extent_policy` selects logical or padded statistics; undeclared → the executed quantities stay symbols and dependent metrics are `partial`.
Block statistics are memoized per `(mask, S_q, S_k, b_q, b_k, Delta, policy, subdivide)`; enumeration is O(#blocks).

### `vis.closed.*` — spec 4.3 closed forms
For `S_q = S_k = S`, `Delta = 0`, `b_q = b_k = p | S`, `skip_future`:
`C_rect/BH = (S^2 + S p)/2`, `(C_rect - C_valid)/BH = S(p-1)/2`.  Derivation: `N = S/p` diagonal + lower blocks
`= N(N+1)/2` blocks of area `p^2`.  Emitted only when the conditions hold.

---

## 2. Work (`work.*`, spec 5)

### Primitive `F_matmul(M, N, K) = 2 M N K` (FMA = 2; not instruction count, not compiled work).

### Projection rows (spec 5.3)
`F_proj = 2 N_Qproj R_q D_n + 2 N_Kproj R_k D_n + 2 N_Vproj R_k D_v` with
* `projection_scope = full`: `N_Q = BH S_q`, `N_K = N_V = BH S_k`
* `new_tokens_only`: `N_K = N_V = BH (S_k - cached_tokens)` (or `BH S_new` if unknown)
* `none`: `N_K = N_V = 0` (expanded K/V supplied; projection outside the scope)
* absorbed variants: `N_K = N_V = 0` in every scope — `C_k` is consumed directly as the K and the V operand, so no K/V row is projected in this invocation (spec 5.3: "when K/V are passed in directly, they are 0 within this scope"); the `Z W^v` output projection is its own quantity.
`N_Qproj`, `N_Kproj` and `N_Vproj` are reported as metrics in their own right, counted once per row.  When the
executed graph projects a row more than once (in-loop projection) or skips a tile no query block scans, the
executed figure is emitted beside it as `N_*proj[executed_graph]` (spec 5.2 loop multiplicity).
Ragged: `BH S_q → H sum_b Sq_b`.

`F_proj[general]` follows the same dataflow, so it is variant-dependent:

| variant | `F_proj[general]` | why |
|---|---|---|
| expanded | `2 N_Qproj R_q D_n + 2 N_Kproj R_k D_n + 2 N_Vproj R_k D_v` | Q^n, K^n and V are all projected |
| absorbed two-step | `2 N_Qproj R_q D_n + 2 N_Qproj D_n R_k` | the Q projection plus the absorb step `Q^n (W^k)^T`; C_k is the K and the V operand, so no K/V row is projected |
| absorbed precomputed | `2 N_Qproj R_q R_k` | both Q-side steps collapse into the merged projection `C_q W~`; the merge work itself is `F_Wmerge[once]`, amortized as `F_Wmerge_amortized[useful]` |

The output projection `Z W^v` of the absorbed paths is not part of `F_proj`; it is `F_output_projection[executed_graph]`.

### `work.expanded.total` — useful expanded work
`F_total[useful] = 2 N_Q R_q D_n + 2 N_K R_k D_n + 2 N_V R_k D_v + 2 C_valid (D_n + D_r + D_v)`
Components: `F_Q, F_K, F_V, F_QK_n = 2C D_n, F_QK_r = 2C D_r, F_PV = 2C D_v`.
With full projection this is the spec's boxed
`F_E = 2BH[S_q R_q D_n + S_k R_k (D_n + D_v)] + 2C(D_n + D_r + D_v)` (emitted as `F_E[spec_closed_form]`).

### `work.absorbed_two_step.total`
`Q~ = (C_q W^q) W^k^T` costs `2 N_Q R_q D_n + 2 N_Q D_n R_k`; scores `2C R_k`; rope `2C D_r`; `Z = P C_k` `2C R_k`; `O = Z W^v` `2 N_Q R_k D_v`:
`F_A = 2BH[S_q R_q D_n + S_q R_k (D_n + D_v)] + 2C(2R_k + D_r)`.  Accumulator width `R_k`, independent of `D_v`.

### `work.absorbed_precomputed.total`
`W~ = W^q W^k^T` merge `2 H R_q D_n R_k / n_reuse` (amortised; `n_reuse` unknown unless declared), then `2 N_Q R_q R_k + 2C(2R_k + D_r) + 2 N_Q R_k D_v`; storage `H R_q R_k s`.

### `work.path_difference`
`F_A - F_E = 2 R_k (D_n + D_v)(N_Q - N_K) + 2C(2R_k - D_n - D_v)` (general rows) →
`2BH R_k (D_n + D_v)(S_q - S_k) + 2C(2R_k - D_n - D_v)` (full projection, `spec_closed_form`).
Computation comparison only, not runtime.

### `work.<alg>.total.rect` / `work.<alg>.graph.total`
Same formulas with `C = C_rect` (rectangular scheduled) / `C = C_exec` (graph sum `2MNK*multiplicity` over executed cells under the declared extent policy).  The two agree only when each KV tile is projected exactly once: with in-loop projection a tile is re-projected by
every q-block that scans it, and a tile no q-block scans is never projected at all, so neither quantity dominates the
other in general.  The tests assert the equality under materialized expanded K/V with logical extents, where the
projection does run once per row.  `work.compiled` is always unknown without compile evidence.

### `work.vector.*` (spec 5.5), per executed rectangle `n x m`, summed via `C_exec`, row visits `V = sum n_i` and masked cells `C_mask`:

| kind | count | note |
|---|---|---|
| exp (E) | `C_exec` | masked cells included |
| cmp (rowmax) | `C_exec - V` | `n*max(m-1,0)` |
| cmp (old/new max) | `V` | |
| add (rowsum) | `C_exec - V` | |
| exp (alpha) | `V` | upper bound (first block may skip) |
| mul/add (acc scale, acc add) | `V*D_A` each | `D_A = D_v` or `R_k` |
| add (l update) | `V` | |
| div (final normalize) | `BH S_q D_A` | unnormalized recurrence, `final_normalize = divide` |
| recip + mul (final normalize) | `BH S_q`, `BH S_q D_A` | unnormalized recurrence, `final_normalize = reciprocal_multiply`; spec 5.5 leaves the choice to the implementation. Undeclared, the division form is the named scenario and `final_normalize` is listed missing; the reciprocal form is the alternative, not added |
| mul/recip (normalized) | `V*D_A`, `V` | normalized recurrence only |
| mul (scale), log2e scale | `C_exec` each | log2e only for `exp_base = 2` |
| mask | `C_mask` | only rectangles that are not fully visible need a mask |
| cast[`<site>`] | per site: `C_exec` (score_to_exp, exp_to_p), `2V` (state_update), `BH S_q D_v` (accumulator_to_output), projected elements (projected_operand), `BH S_q D_A` (z_to_output) | spec 3.3: different cast *locations* must give different graphs, so a declared `cast_points` list emits exactly those sites.  Undeclared, only the two dtype-implied sites are counted, `cast_points` is named missing, and an undeclared dtype keeps the op flagged |
| log + add (LSE) | `BH S_q` each | `LSE = m + log(l)`, once per row, when LSE is requested |
| mul (alpha·l) | `V` | |
| mul (LSE restore) | `BH S_q` | `exp_base = 2` and LSE requested |
| broadcast, transpose | `C_exec`, — | backend-dependent; reported as undeclared scenario ops (`W_vec[broadcast]`, resource `layout`) |
| combine QK branches | `Piecewise(C_exec if D_r > 0 else 0)` | |

These are logical element counts, not vector instructions; the backend mapping is a separate model.

---

## 3. Bytes (`bytes.*`, spec 6)

* `bytes.logical.<role>` = `prod(shape) * s_role` with the tensor's own dtype; `bytes.M_in` = sum over the seven roles (allocation extents); `bytes.ledger.*` from actual metadata with `capacity_shape` honoured and `alias_of` counted once.
* `bytes.M_O = s_O B H S_q D_v`, `bytes.M_LSE = s_LSE B H S_q` (active extents).
* `bytes.mat.*` conditional materialization: expanded K/V `BH S_k (s_K D_n + s_V D_v) * (writes + reads)`; `Q^n`, `Q~`, `Z` similarly; status `not_applicable` when disabled, `partial` when undeclared.
* `bytes.path.<e>` = `sum_r m_r n_r` over the scenario's transfer events (q_outer_kv_inner; weights re-read per program; the KV latent is read per executed key column once per (batch, head-group): `Ck_read`: `m = R_k s_Ck`, `n = ceil(H/h_pp) * sum_b key_visits_b`; outputs written once per active row).  With `materialize.expanded_kv = True` the events switch to a projection pass (latent + weights read once, `K^n|V` written once) followed by expanded-tile reads per executed key column per head; with `projection_scope = new_tokens_only` the KV reads are split between cached expanded tiles and new latent tokens in proportion to their key counts (labelled scenario).
* `bytes.path.vmem_to_vreg`: operand-streaming scenario (Q operands once per program, K/V/K^r tiles once per KV visit) — always `partial` (missing `register_schedule_evidence`); `bytes.path.vreg_to_vmem` is `unknown` (store/spill traffic needs compiler evidence).
* `bytes.mat.*`: `write_traffic + read_traffic`, where the read side uses executed key columns (`H * key_visits * (s_K D_n + s_V D_v)`), so the whole-batch tensor is never multiplied by `B` twice.

Not equivalent to physical HBM traffic with caching, nor to spill traffic.

---

## 4. Local objects, layout, live sets, envelope (`local.*`, `layout.*`, `pressure.*`, spec 7)

Object sizes follow the 7.1 table (`b_q R_q s_Cq`, `b_q b_rq s_Cq`, `b_rq D_n s_Wq`, `b_q D_n s_Qacc`, `b_k b_rk s_Ck`, `b_q b_k s_X`, `b_q b_k s_E`, `b_q b_k s_P`, `b_q s_state`, `b_q D_v s_A` / `b_q R_k s_A`, …) with a `buffers` multiplier (`n_buf`) for streamed windows and explicit alias groups (`score_alias_exp`, `exp_alias_p_operand`, `acc_or_Z`).

* `layout.<obj>`: `LayoutBytes = s L_s L_l ceil(m/L_s) ceil(n/L_l)` with `(L_s, L_l)` from the profile for the object's dtype (else symbolic); `compact` = logical size; `replicated` = unknown.
* Every local object carries a storage-level assignment (`vmem` for residency windows, weights, scratch, output tiles; `vreg` for score/E/P temporaries, row states, accumulators).  This is an analysis assumption (spec 7.3 wording), not compiler allocation, and is stated in the report.
* `pressure.live.<level>.<stage>` (`all`, `vreg`, `vmem`): sum over objects live at the stage cut, one term per alias group (`Max` of members).  Loop-carried state (A, m, l, Q operands) is live at the in-loop `kv_proj` cut.
* `pressure.peak.<level>`: `max_t` over stage cuts.
* `pressure.envelope.<level>.<stage>`: `a b_q b_k + b b_q + c b_k + d (+ remainder)` by polynomial coefficient extraction; provenance per coefficient; `d` receives the declared fixed scratch (`s_scratch`).
* `pressure.feasible.<level>.<stage>.b_k`: `floor((R_level - b b_q - d)/(a b_q + c))` for **every** stage, valid iff `a b_q + c > 0`; a stage with no `b_k` term is `not_applicable`, not a division by zero.  `pressure.feasible.<level>.binding` is the smallest of them and names the binding stage; it stays `partial` while any stage is unevaluable or any field it depends on is undeclared.  `extras.feasible_region_by_stage` evaluates each stage over a `b_q` sweep (the constraint surface of spec 10.2), with the declared `b_q` marked.
* `pressure.capacity_constraint.<level>.<stage>`: `max(0, M_live - R_level)` per stage on the source-level live set (layout coverage not applied; not a compiled cut).
* `spill.*`: the four non-interchangeable quantities of spec 8.1 — `N_spill_instructions` (static count), `M_spill_peak` (allocation peak), `B_spill_fill` (dynamic bytes), `dT_spill` (exposed time) — plus the optimisation definitions `B_extra_opt`, `T_opt` and `B_extra_opt_under_time_budget` (spec 8.3).  All are `unknown` until compile/measurement evidence (or a declared small graph, machine model and objective) is attached; `dT_spill` stays `partial` until its comparison graph is declared, and any quantity read from a compiler output declared incomplete stays `partial`.

---

### Status semantics
`bound` = every symbol bound and every field the value depends on declared; `partial` = a value computed under a labelled scenario (the undeclared field is listed in `missing_fields`) or with unresolved symbols; `symbolic` = nothing bound; `unknown` = no expression available without evidence; `conflict` = contradictory declarations; `not_applicable` = disabled by the declared strategy.  `coverage` states what the value is complete with respect to.

## 5. Sensitivity (`sens.*`, `sched.*`, spec 10)
`dM_X/db_q = s_X b_k`, `dM_X/db_k = s_X b_q`, `dM_A/db_q = D_A s_A`, `dM_A/db_k = 0` (continuous relaxation).
`n_programs = ceil(H/h_pp) * B ceil(S_q_sched/b_q)` (the scheduled extent, which equals the active extent when
undeclared; ragged: `sum_b ceil(Sq_b/b_q)`), `n_kv_block_visits = B H |E|`.  Per-head transfer events are counted as
`H * n_qblocks`, which stays exact when `h_pp` does not divide `H`.
`compare()` reports discrete differences `f(kappa') - f(kappa)`, per-candidate legality (with the candidate's own tiles), a Pareto set, and — only with applicable calibration — a ranking whose note says whether the predicted intervals actually separate.  A candidate may carry `algorithm` ("switch to absorbed"); any other non-strategy key raises `TaskIdentityError`.

---

## 6. Performance (`perf.*`, spec 9)
* `perf.lb.<u>`: `W_u / P_u` for every resource class the graph counts work for — `mxu` (executed graph FLOPs over the peak for the declared compute dtype), `exp`, `vpu`, `reduce` and `layout`, each with its own profile throughput; a class whose throughput the profile does not supply is `unknown` and names it, rather than being dropped from the maximum.  `perf.lb.path:<e>`: `B_e / W_e` (VMEM/VREG paths never use HBM bandwidth); `critical_path` unknown without a schedule model.
* `perf.lb.combined`: `max(...)` under `full_overlap_max`, or sum of per-group maxima under `serial_stage_sum`.  Groups name resource classes, i.e. a serial-resource model; per-pipeline-stage grouping is not implemented (D19), and an entry that is not a known resource class is rejected.  A resource the declared groups do not name is *not* dropped: the combination becomes `max(sum of the groups, the unnamed resources)`, and `extras.resource_lower_bounds.resources_outside_declared_groups` lists them.  A group whose own value is unknown is omitted from the sum and printed as omitted, since every term is non-negative.  When `overlap_model` is undeclared both combinations are reported as `partial` scenarios (`T_LB[combined:scenario=...]`).  Each path has its own bandwidth symbol (`W_hbm`, `W_hbm_w`, `W_vmem`, `W_vmem_w`); a missing direction stays unknown.
* `perf.calibrated.nodes` (metric `T_pred[node_interval]`, result key `node_interval_s`; `matmul_interval_s` is kept as an alias of the former name): `sum_v W_v/(eps_v P_v) + t_startup`, plus `T_launch` and the model-error term `eps_model` of spec 9.3, over every node with an applicable bucket.  A matmul node with no work (the positional branch at `D_r = 0`) is reported absent and charged no startup.  A vector-class bucket is matched against the dtype that class executes in (exp in `exp_dtype`, reductions on the score dtype, elementwise on the state dtype), not the matmul operand dtype.  Layout buckets are matched on kind only, because layout work spans objects of several dtypes; a layout bucket that declares a dtype is applied with the unverified dtype match named in `missing_fields`.  A bucket kind with no consumer (`hbm_transfer`: transfer time is not part of the node-time interval) is named in the prediction's notes rather than absorbed.  Matmul nodes use `matmul` buckets; the `exp`, `reduce`, `layout` and `vector` resource classes use buckets of their own kind against their own throughputs, so declared non-matmul evidence is applied rather than ignored.  The interval comes from `[eps_low, eps_high]`; every absent term (a bucket, a throughput, `startup_s`, `launch_overhead_s`, `model_error_s`) is listed as missing, never treated as zero.  It is a *different object* from the lower bound and absent (`not_calibrated`) when no bucket applies.  Attention nodes are matched by their executed per-visit tile shape `(b_q, b_k, D)`, so `m/n/k` ranges apply to them normally.  A bucket that claims `validated_on_unseen_shapes` must carry `validation_error`; a bucket whose `device` or `toolchain` does not identify this profile is still applied but the mismatch is named in `missing_fields`, and bucket applicability keys only on dtype and `M/N/K`, never on layout or strategy (also named).
