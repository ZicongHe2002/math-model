# MLA forward analytical report (expanded, strategy=bq16_bk16)

- mode: **bound**  |  adapter: `seven_input_latent`  |  algorithm: `expanded`
- task fingerprint: `55fd2c864359ee4e`

## Bindings and declarations

- **bindings**: `{"B": 2, "R_q": 16, "R_k": 32, "H": 4, "D_n": 24, "D_r": 8, "D_v": 24, "S_q": 64, "S_k": 64, "Delta": 0, "s_Cq": 2, "s_Ck": 2, "s_Wq": 2, "s_Wk": 2, "s_Wv": ...`
- **dimension_provenance**: `{"B": "tensor.q_latent.axis[B]", "Sq": "tensor.q_latent.axis[Sq]", "Rq": "tensor.q_latent.axis[Rq]", "Sk": "tensor.kv_latent.axis[Sk]", "Rk": "tensor.kv_late...`
- **cross_checks**: `["tensor.kv_latent.axis[B] == tensor.q_latent.axis[B] == 2", "tensor.q_pe.axis[B] == tensor.q_latent.axis[B] == 2", "tensor.k_pe.axis[B] == tensor.q_latent.a...`
- **capacity_extents**: `{"Sq": 64, "Sk": 64}`
- **active_extents**: `{"Sq": 64, "Sk": 64, "declared": true, "declared_axes": ["Sk", "Sq"], "undeclared_axes": []}`
- **dtypes**: `{"q_latent": "bfloat16", "kv_latent": "bfloat16", "w_q_nope": "bfloat16", "w_k_nope": "bfloat16", "w_v": "bfloat16", "q_pe": "bfloat16", "k_pe": "bfloat16"}`
- **semantics**: `{"mask": "causal", "position_offset": 0, "query_position_start": null, "key_position_start": null, "scale_policy": "standard", "scale_value": null, "scale_ma...`
- **implementation**: `{"name": "bq16_bk16", "b_q": 16, "b_k": 16, "b_rq": null, "b_rk": null, "scheduled_lengths": null, "rect_policy": "skip_future", "subdivide": null, "executed...`
- **numerics**: `{"matmul_input_dtype": "bfloat16", "accumulator_dtype": "float32", "score_dtype": "float32", "exp_dtype": "float32", "p_operand_dtype": "bfloat16", "state_dt...`
- **numerics_assumptions**: `["projected_operand_dtype undeclared: Q^n/K^n/V/Q~/Z operand widths (s_Qn, s_Kn, s_V, s_Qt, s_Z) assumed = matmul_input_dtype", "projection_accumulator_dtype...`
- **hardware**: `{"name": "tpu-v6e (ILLUSTRATIVE profile; public figures entered by hand, not verified in this session)", "scope_default": "per_chip", "provenance": "Public G...`

## Unresolved inputs

- CP: needed by 2 metric(s), e.g. ['T_LB[critical_path]', 'T_LB[combined]']
- L_l: needed by 12 metric(s), e.g. ['layout[q_latent_window]', 'layout[wq_rank_tile]', 'layout[q_nope_operand]']
- L_s: needed by 12 metric(s), e.g. ['layout[q_latent_window]', 'layout[wq_rank_tile]', 'layout[q_nope_operand]']
- P_exp: needed by 2 metric(s), e.g. ['T_LB[exp]', 'T_LB[combined]']
- P_layout: needed by 2 metric(s), e.g. ['T_LB[layout]', 'T_LB[combined]']
- P_red: needed by 2 metric(s), e.g. ['T_LB[reduce]', 'T_LB[combined]']
- P_vpu: needed by 2 metric(s), e.g. ['T_LB[vpu]', 'T_LB[combined]']
- W_hbm_w: needed by 2 metric(s), e.g. ['T_LB[path:vmem_to_hbm]', 'T_LB[combined]']
- W_vmem: needed by 2 metric(s), e.g. ['T_LB[path:vmem_to_vreg]', 'T_LB[combined]']
- acceptance: needed by 1 metric(s), e.g. ['acceptance[declared]']
- allowed_transformations: needed by 3 metric(s), e.g. ['B_extra_opt', 'T_opt', 'B_extra_opt_under_time_budget']
- b_rk: needed by 12 metric(s), e.g. ['local[kv_latent_rank_tile]', 'local[wk_rank_tile]', 'local[wv_rank_tile]']
- b_rq: needed by 11 metric(s), e.g. ['local[q_latent_window]', 'local[wq_rank_tile]', 'M_live[all:q_proj]']
- broadcast_materialized (undeclared backend behaviour): needed by 2 metric(s), e.g. ['W_vec[broadcast]', 'W_resource[layout]']
- calibration: needed by 1 metric(s), e.g. ['T_pred[node_interval]']
- cast_points: needed by 2 metric(s), e.g. ['W_vec[cast]', 'W_resource[vpu]']
- compile_evidence: needed by 2 metric(s), e.g. ['F_compiled', 'N_spill_instructions']
- compile_or_measurement_evidence: needed by 3 metric(s), e.g. ['M_spill_peak', 'B_spill_fill', 'dT_spill']
- declared comparison graph: needed by 1 metric(s), e.g. ['dT_spill']
- final_normalize: needed by 2 metric(s), e.g. ['W_vec[div]', 'W_resource[vpu]']
- graph: needed by 3 metric(s), e.g. ['B_extra_opt', 'T_opt', 'B_extra_opt_under_time_budget']
- layout.accumulator_A: needed by 1 metric(s), e.g. ['layout[accumulator_A]']
- layout.exp_E: needed by 1 metric(s), e.g. ['layout[exp_E]']
- layout.k_nope_tile: needed by 1 metric(s), e.g. ['layout[k_nope_tile]']
- layout.k_pe_tile: needed by 1 metric(s), e.g. ['layout[k_pe_tile]']
- layout.kv_latent_rank_tile: needed by 1 metric(s), e.g. ['layout[kv_latent_rank_tile]']
- layout.lse_tile: needed by 1 metric(s), e.g. ['layout[lse_tile]']
- layout.out_tile: needed by 1 metric(s), e.g. ['layout[out_tile]']
- layout.p_operand: needed by 1 metric(s), e.g. ['layout[p_operand]']
- layout.q_latent_window: needed by 1 metric(s), e.g. ['layout[q_latent_window]']
- layout.q_nope_operand: needed by 1 metric(s), e.g. ['layout[q_nope_operand]']
- layout.q_pe_tile: needed by 1 metric(s), e.g. ['layout[q_pe_tile]']
- layout.q_proj_acc: needed by 1 metric(s), e.g. ['layout[q_proj_acc]']
- layout.row_alpha: needed by 1 metric(s), e.g. ['layout[row_alpha]']
- layout.row_state_l: needed by 1 metric(s), e.g. ['layout[row_state_l]']
- layout.row_state_m: needed by 1 metric(s), e.g. ['layout[row_state_m]']
- layout.score_X: needed by 1 metric(s), e.g. ['layout[score_X]']
- layout.v_tile: needed by 1 metric(s), e.g. ['layout[v_tile]']
- layout.wk_rank_tile: needed by 1 metric(s), e.g. ['layout[wk_rank_tile]']
- layout.wq_rank_tile: needed by 1 metric(s), e.g. ['layout[wq_rank_tile]']
- layout.wv_rank_tile: needed by 1 metric(s), e.g. ['layout[wv_rank_tile]']
- machine_model: needed by 3 metric(s), e.g. ['B_extra_opt', 'T_opt', 'B_extra_opt_under_time_budget']
- objective: needed by 1 metric(s), e.g. ['B_extra_opt']
- operand_transpose_materialized (undeclared backend behaviour): needed by 2 metric(s), e.g. ['W_vec[transpose]', 'W_resource[layout]']
- projected_operand_dtype: needed by 3 metric(s), e.g. ['local[q_nope_operand]', 'local[k_nope_tile]', 'local[v_tile]']
- projection_accumulator_dtype: needed by 1 metric(s), e.g. ['local[q_proj_acc]']
- q_resident: needed by 32 metric(s), e.g. ['B[vmem_to_vreg]', 'local[q_nope_operand]', 'local[q_pe_tile]']
- register_schedule_evidence: needed by 4 metric(s), e.g. ['B[vmem_to_vreg]', 'B[vreg_to_vmem]', 'T_LB[path:vmem_to_vreg]']
- schedule: needed by 34 metric(s), e.g. ['N_Kproj[executed_graph]', 'N_Vproj[executed_graph]', 'F_total[executed_graph]']
- schedule_model: needed by 2 metric(s), e.g. ['T_opt', 'B_extra_opt_under_time_budget']
- scheduled_lengths: needed by 47 metric(s), e.g. ['C_rect', 'C_pad', 'rect_waste']
- scheduled_lengths.Sk: needed by 1 metric(s), e.g. ['S_k_scheduled']
- scheduled_lengths.Sq: needed by 1 metric(s), e.g. ['S_q_scheduled']
- time_budget_tau: needed by 1 metric(s), e.g. ['B_extra_opt_under_time_budget']
- v_load: needed by 24 metric(s), e.g. ['local[v_tile]', 'M_live[all:kv_proj]', 'M_live[all:score]']
- vmem_budget_bytes: needed by 11 metric(s), e.g. ['excess_over_budget[vmem:q_proj]', 'b_k_max[vmem:kv_proj]', 'excess_over_budget[vmem:kv_proj]']
- vreg_budget_bytes: needed by 10 metric(s), e.g. ['excess_over_budget[vreg:q_proj]', 'excess_over_budget[vreg:kv_proj]', 'b_k_max[vreg:score]']

## Warnings

- schedule not declared: the q_outer_kv_inner loop nest is shown as the scenario; transfer counts and the program count follow from it
- rank sub-tiles undeclared: the full R_q / R_k window is shown as the scenario
- q_resident not declared: Q operands held across the KV loop shown as the scenario
- final_normalize not declared: the n*D_A division form is the scenario; the reciprocal-plus-multiply form of spec 5.5 is the alternative, not an addition
- cast_points not declared: the cast graph is inferred from dtype differences at the two modelled sites (exp_to_p, accumulator_to_output); a declared site list gives a different graph
- v_load not declared: the early V load (V live from its production/read until PV) is shown as the scenario
- projected_operand_dtype undeclared: Q^n/K^n/V/Q~/Z operand widths (s_Qn, s_Kn, s_V, s_Qt, s_Z) assumed = matmul_input_dtype
- projection_accumulator_dtype undeclared: projection accumulator width (s_Qacc) assumed = accumulator_dtype

## 1. task and scope

| metric | formula id | expression | value | unit | status | missing | scope | not equivalent to |
|---|---|---|---|---|---|---|---|---|
| B | `task.dim.B` | `B` | 2 | count | bound |  | current invocation |  |
| H | `task.dim.H` | `H` | 4 | count | bound |  | current invocation |  |
| R_q | `task.dim.R_q` | `R_q` | 16 | count | bound |  | current invocation |  |
| R_k | `task.dim.R_k` | `R_k` | 32 | count | bound |  | current invocation |  |
| D_n | `task.dim.D_n` | `D_n` | 24 | count | bound |  | current invocation |  |
| D_r | `task.dim.D_r` | `D_r` | 8 | count | bound |  | current invocation |  |
| D_v | `task.dim.D_v` | `D_v` | 24 | count | bound |  | current invocation |  |
| S_q_active | `task.active.S_q` | `S_q` | 64 | count | bound |  | active query rows (mathematical work) |  |
| S_k_active | `task.active.S_k` | `S_k` | 64 | count | bound |  | active keys consumed in this call |  |
| S_q_scheduled | `task.scheduled.S_q` | `declared scheduled extent (else = active)` | 64 | count | partial | scheduled_lengths.Sq | extent the kernel iterates over (execution rectangles, padding counts) | active extent; allocated capacity |
| S_k_scheduled | `task.scheduled.S_k` | `declared scheduled extent (else = active)` | 64 | count | partial | scheduled_lengths.Sk | extent the kernel iterates over (execution rectangles, padding counts) | active extent; allocated capacity |
| S_q_capacity | `task.capacity.S_q` | `allocated extent from tensor shape` | 64 | count | bound |  | allocated query extent | active rows; scheduled rows |
| S_k_capacity | `task.capacity.S_k` | `allocated extent from tensor shape` | 64 | count | bound |  | allocated KV extent (not necessarily consumed) | active keys; scheduled keys |
| Delta | `task.offset` | `Delta` | 0 | positions | bound |  | key j visible to query i iff j <= i + Delta |  |
| mask | `task.mask` | `declared mask kind` | causal |  | bound |  | visibility semantics |  |
| gamma | `task.scale` | `1/sqrt(D_n + D_r)` | 0.176777 | 1 | bound |  | score scale |  |
| outputs | `task.outputs` | `declared output scope` | ['O', 'LSE'] |  | bound |  | output scope |  |
| lse_convention | `task.lse_convention` | `declared` | natural_log |  | bound |  | LSE log base |  |
| projection_scope | `task.projection_scope` | `declared` | full |  | bound |  | which rows are projected in this call |  |
| rows_without_visible_keys | `task.empty_rows` | `B*H*Piecewise((S_q, Eq(S_k, 0)), (Max(0, Min(-Delta, S_q)), True))` | 0 | rows | bound |  | (b,h,i) rows whose softmax is undefined |  |

Assumptions:
- `S_q_scheduled`: scheduled extent undeclared: assumed equal to the active extent (scenario)
- `S_k_scheduled`: scheduled extent undeclared: assumed equal to the active extent (scenario)
- `mask`: mask is declared, never inferred from shapes
- `gamma`: scale policy 'standard': gamma = (D_n + D_r)^(-1/2)
- `rows_without_visible_keys`: uniform lengths: B*H*max(0, min(S_q, -Delta))

<details><summary>Provenance (bindings, coverage, source, evidence)</summary>

| metric | source | coverage | bindings | evidence |
|---|---|---|---|---|
| B | input | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"B": 2}` | tensor.q_latent.axis[B] |
| H | input | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"H": 4}` | tensor.w_q_nope.axis[H] |
| R_q | input | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"R_q": 16}` | tensor.q_latent.axis[Rq] |
| R_k | input | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"R_k": 32}` | tensor.kv_latent.axis[Rk] |
| D_n | input | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"D_n": 24}` | tensor.w_q_nope.axis[Dn] |
| D_r | input | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"D_r": 8}` | tensor.q_pe.axis[Dr] |
| D_v | input | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"D_v": 24}` | tensor.w_v.axis[Dv] |
| S_q_active | input | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"S_q": 64}` |  |
| S_k_active | input | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"S_k": 64}` |  |
| S_q_scheduled | input | incomplete: depends on undeclared/unbound scheduled_lengths.Sq | `{}` |  |
| S_k_scheduled | input | incomplete: depends on undeclared/unbound scheduled_lengths.Sk | `{}` |  |
| S_q_capacity | input | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{}` |  |
| S_k_capacity | input | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{}` |  |
| Delta | input | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"Delta": 0}` |  |
| mask | input | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{}` |  |
| gamma | input | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"D_n": 24, "D_r": 8}` |  |
| outputs | input | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{}` |  |
| lse_convention | input | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{}` |  |
| projection_scope | input | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{}` |  |
| rows_without_visible_keys | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"B": 2, "Delta": 0, "H": 4, "S_k": 64, "S_q": 64}` |  |

</details>

## 2. visibility and rectangles

| metric | formula id | expression | value | unit | status | missing | scope | not equivalent to |
|---|---|---|---|---|---|---|---|---|
| C_valid | `vis.C_valid` | `B*H*CausalVisibleCount(S_q, S_k, Delta)` | 16,640 | cells | bound |  | all (b,h,i,j) with mu=1; includes B and H | exp instruction count; executed cells |
| C_rect | `vis.C_rect` | `B*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)` | 20,480 | cells | partial | scheduled_lengths | sum over selected rectangles of n_i*m_j (logical extents) | C_valid; C_pad under a padded execution extent |
| C_pad | `vis.C_pad` | `B*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)` | 20,480 | cells | partial | scheduled_lengths | sum over selected rectangles of n_i*m_j: the declared executed extents are the logical ones | hardware array padding (must come from compilation evidence) |
| rect_waste | `vis.waste` | `-B*H*CausalVisibleCount(S_q, S_k, Delta) + B*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)` | 3840 | cells | partial | scheduled_lengths | C_rect - C_valid under the declared policy |  |
| selected_blocks | `vis.blocks.selected` | `B*H*BlockStat(selected, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)` | 80 | blocks | partial | scheduled_lengths | over all (b,h) |  |
| partial_blocks | `vis.blocks.partial` | `B*H*BlockStat(partial, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)` | 32 | blocks | partial | scheduled_lengths | over all (b,h) |  |
| fully_visible_blocks | `vis.blocks.full` | `B*H*BlockStat(full, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)` | 48 | blocks | partial | scheduled_lengths | over all (b,h) |  |
| future_blocks_selected | `vis.blocks.future_selected` | `B*H*BlockStat(future_selected, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)` | 0 | blocks | partial | scheduled_lengths | over all (b,h) |  |
| padding_blocks_selected | `vis.blocks.padding_selected` | `B*H*BlockStat(padding_selected, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)` | 0 | blocks | partial | scheduled_lengths | over all (b,h) |  |
| future_blocks_skipped | `vis.blocks.future_skipped` | `B*H*BlockStat(future_skipped, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)` | 48 | blocks | partial | scheduled_lengths | over all (b,h) |  |
| padding_blocks_skipped | `vis.blocks.padding_skipped` | `B*H*BlockStat(padding_skipped, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)` | 0 | blocks | partial | scheduled_lengths | over all (b,h) |  |
| masked_cells | `vis.blocks.masked_cells` | `B*H*BlockStat(masked_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)` | 8192 | cells | partial | scheduled_lengths | over all (b,h) |  |
| executed_extent_policy | `vis.exec_policy` | `declared` | logical |  | bound |  | n_hat, m_hat source |  |
| closed_form_rect_per_bh | `vis.closed.rect` | `(S**2 + S*p)/2` | 2560 | cells | bound |  | per (b,h); square causal, S divisible by p, skip future, diagonal in full |  |
| closed_form_waste_per_bh | `vis.closed.waste` | `S*(p-1)/2` | 480 | cells | bound |  | per (b,h) |  |

Assumptions:
- `C_valid`: uniform unit-step positions; causal: j <= i + Delta
- `C_rect`: rectangle selection policy 'skip_future'
- `C_rect`: tail blocks use logical extents
- `C_rect`: scheduled extent undeclared: assumed equal to the active extent (scenario)
- `C_pad`: rectangle selection policy 'skip_future'
- `C_pad`: executed extents = logical extents (declared 'logical')
- `C_pad`: scheduled extent undeclared: assumed equal to the active extent (scenario)
- `rect_waste`: scheduled extent undeclared: assumed equal to the active extent (scenario)
- `selected_blocks`: rectangle selection policy 'skip_future'
- `selected_blocks`: scheduled extent undeclared: assumed equal to the active extent (scenario)
- `partial_blocks`: rectangle selection policy 'skip_future'
- `partial_blocks`: scheduled extent undeclared: assumed equal to the active extent (scenario)
- `fully_visible_blocks`: rectangle selection policy 'skip_future'
- `fully_visible_blocks`: scheduled extent undeclared: assumed equal to the active extent (scenario)
- `future_blocks_selected`: rectangle selection policy 'skip_future'
- `future_blocks_selected`: scheduled extent undeclared: assumed equal to the active extent (scenario)
- `padding_blocks_selected`: rectangle selection policy 'skip_future'
- `padding_blocks_selected`: scheduled extent undeclared: assumed equal to the active extent (scenario)
- `future_blocks_skipped`: rectangle selection policy 'skip_future'
- `future_blocks_skipped`: scheduled extent undeclared: assumed equal to the active extent (scenario)
- `padding_blocks_skipped`: rectangle selection policy 'skip_future'
- `padding_blocks_skipped`: scheduled extent undeclared: assumed equal to the active extent (scenario)
- `masked_cells`: rectangle selection policy 'skip_future'
- `masked_cells`: scheduled extent undeclared: assumed equal to the active extent (scenario)
- `executed_extent_policy`: executed extents come from the declared strategy or compilation evidence, never from the hardware array size

<details><summary>Provenance (bindings, coverage, source, evidence)</summary>

| metric | source | coverage | bindings | evidence |
|---|---|---|---|---|
| C_valid | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"B": 2, "Delta": 0, "H": 4, "S_k": 64, "S_q": 64}` |  |
| C_rect | derived | incomplete: depends on undeclared/unbound scheduled_lengths | `{"B": 2, "Delta": 0, "H": 4, "S_k": 64, "S_q": 64, "b_k": 16, "b_q": 16}` |  |
| C_pad | derived | incomplete: depends on undeclared/unbound scheduled_lengths | `{"B": 2, "Delta": 0, "H": 4, "S_k": 64, "S_q": 64, "b_k": 16, "b_q": 16}` |  |
| rect_waste | derived | incomplete: depends on undeclared/unbound scheduled_lengths | `{"B": 2, "Delta": 0, "H": 4, "S_k": 64, "S_q": 64, "b_k": 16, "b_q": 16}` |  |
| selected_blocks | derived | incomplete: depends on undeclared/unbound scheduled_lengths | `{"B": 2, "Delta": 0, "H": 4, "S_k": 64, "S_q": 64, "b_k": 16, "b_q": 16}` |  |
| partial_blocks | derived | incomplete: depends on undeclared/unbound scheduled_lengths | `{"B": 2, "Delta": 0, "H": 4, "S_k": 64, "S_q": 64, "b_k": 16, "b_q": 16}` |  |
| fully_visible_blocks | derived | incomplete: depends on undeclared/unbound scheduled_lengths | `{"B": 2, "Delta": 0, "H": 4, "S_k": 64, "S_q": 64, "b_k": 16, "b_q": 16}` |  |
| future_blocks_selected | derived | incomplete: depends on undeclared/unbound scheduled_lengths | `{"B": 2, "Delta": 0, "H": 4, "S_k": 64, "S_q": 64, "b_k": 16, "b_q": 16}` |  |
| padding_blocks_selected | derived | incomplete: depends on undeclared/unbound scheduled_lengths | `{"B": 2, "Delta": 0, "H": 4, "S_k": 64, "S_q": 64, "b_k": 16, "b_q": 16}` |  |
| future_blocks_skipped | derived | incomplete: depends on undeclared/unbound scheduled_lengths | `{"B": 2, "Delta": 0, "H": 4, "S_k": 64, "S_q": 64, "b_k": 16, "b_q": 16}` |  |
| padding_blocks_skipped | derived | incomplete: depends on undeclared/unbound scheduled_lengths | `{"B": 2, "Delta": 0, "H": 4, "S_k": 64, "S_q": 64, "b_k": 16, "b_q": 16}` |  |
| masked_cells | derived | incomplete: depends on undeclared/unbound scheduled_lengths | `{"B": 2, "Delta": 0, "H": 4, "S_k": 64, "S_q": 64, "b_k": 16, "b_q": 16}` |  |
| executed_extent_policy | input | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{}` |  |
| closed_form_rect_per_bh | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"S": 64, "p": 16}` |  |
| closed_form_waste_per_bh | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"S": 64, "p": 16}` |  |

</details>

## 3. work (FLOPs and vector operations)

| metric | formula id | expression | value | unit | status | missing | scope | not equivalent to |
|---|---|---|---|---|---|---|---|---|
| F_Q[useful] | `work.expanded.F_Q` | `2*B*D_n*H*R_q*S_q` | 393,216 | FLOP | bound |  | useful (C = C_valid), FMA=2, projection rows from scope 'full' |  |
| F_K[useful] | `work.expanded.F_K` | `2*B*D_n*H*R_k*S_k` | 786,432 | FLOP | bound |  | useful (C = C_valid), FMA=2, projection rows from scope 'full' |  |
| F_V[useful] | `work.expanded.F_V` | `2*B*D_v*H*R_k*S_k` | 786,432 | FLOP | bound |  | useful (C = C_valid), FMA=2, projection rows from scope 'full' |  |
| F_QK_n[useful] | `work.expanded.F_QK_n` | `2*B*D_n*H*CausalVisibleCount(S_q, S_k, Delta)` | 798,720 | FLOP | bound |  | useful (C = C_valid), FMA=2, projection rows from scope 'full' |  |
| F_QK_r[useful] | `work.expanded.F_QK_r` | `2*B*D_r*H*CausalVisibleCount(S_q, S_k, Delta)` | 266,240 | FLOP | bound |  | useful (C = C_valid), FMA=2, projection rows from scope 'full' |  |
| F_PV[useful] | `work.expanded.F_PV` | `2*B*D_v*H*CausalVisibleCount(S_q, S_k, Delta)` | 798,720 | FLOP | bound |  | useful (C = C_valid), FMA=2, projection rows from scope 'full' |  |
| F_total[useful] | `work.expanded.total` | `2*B*D_n*H*R_k*S_k + 2*B*D_n*H*R_q*S_q + 2*B*D_v*H*R_k*S_k + 2*B*H*(D_n + D_r + D_v)*CausalVisibleCount(S_q, S_k, Delta)` | 3,829,760 | FLOP | bound |  | useful mathematical work under projection scope 'full' | scheduled work; compiled instruction count |
| F_E[spec_closed_form] | `work.expanded.closed_form` | `2*B*D_n*H*R_q*S_q + 2*B*H*R_k*S_k*(D_n + D_v) + 2*B*H*(D_n + D_r + D_v)*CausalVisibleCount(S_q, S_k, Delta)` | 3,829,760 | FLOP | bound |  | specification boxed formula: every Q row and every K/V row projected once (full projection, no cache) |  |
| N_Qproj | `work.projection.rows.N_Qproj` | `B*H*S_q` | 512 | rows | bound |  | Q rows projected in this invocation counted once per row, from the declared projection_scope and the variant's dataflow | the tensor's row count; rows read from HBM; the executed-graph projected-row count under in-loop projection |
| N_Kproj[executed_graph] | `work.projection.rows.N_Kproj.executed` | `B*H*BlockStat(key_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)` | 1280 | rows | partial | schedule, scheduled_lengths | K rows projected in this invocation as the executed graph performs them, including the KV-loop multiplicity and excluding tiles no query block scans (spec 5.2) | N_Kproj, the once-per-row count of the declared scope |
| N_Kproj | `work.projection.rows.N_Kproj` | `B*H*S_k` | 512 | rows | bound |  | K rows projected in this invocation counted once per row, from the declared projection_scope and the variant's dataflow | the tensor's row count; rows read from HBM; the executed-graph projected-row count under in-loop projection |
| N_Vproj[executed_graph] | `work.projection.rows.N_Vproj.executed` | `B*H*BlockStat(key_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)` | 1280 | rows | partial | schedule, scheduled_lengths | V rows projected in this invocation as the executed graph performs them, including the KV-loop multiplicity and excluding tiles no query block scans (spec 5.2) | N_Vproj, the once-per-row count of the declared scope |
| N_Vproj | `work.projection.rows.N_Vproj` | `B*H*S_k` | 512 | rows | bound |  | V rows projected in this invocation counted once per row, from the declared projection_scope and the variant's dataflow | the tensor's row count; rows read from HBM; the executed-graph projected-row count under in-loop projection |
| F_proj[general] | `work.projection.general` | `2*B*D_n*H*R_k*S_k + 2*B*D_n*H*R_q*S_q + 2*B*D_v*H*R_k*S_k` | 1,966,080 | FLOP | bound |  | 2 N_Qproj R_q D_n + 2 N_Kproj R_k D_n + 2 N_Vproj R_k D_v with row counts from projection_scope | F_projection[executed_graph], which carries the loop multiplicity |
| F_total[rect] | `work.expanded.total.rect` | `2*B*D_n*H*R_k*S_k + 2*B*D_n*H*R_q*S_q + 2*B*D_v*H*R_k*S_k + 2*B*H*(D_n + D_r + D_v)*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)` | 4,259,840 | FLOP | partial | scheduled_lengths | rectangular-scheduled work (C = C_rect, no internal padding), same projection rows | useful work (C = C_valid); executed graph work; compiled work |
| F_total[executed_graph] | `work.expanded.graph.total` | `2*B*D_n*H*R_k*BlockStat(key_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + 2*B*D_n*H*R_q*S_q + 2*B*D_n*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + 2*B*D_r*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + 2*B*D_v*H*R_k*BlockStat(key_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + 2*B*D_v*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)` | 6,619,136 | FLOP | partial | schedule, scheduled_lengths | graph sum of 2MNK*multiplicity over executed cells C_exec (declared extent policy) |  |
| F_projection[executed_graph] | `work.expanded.graph.projection` | `2*B*D_n*H*R_k*BlockStat(key_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + 2*B*D_n*H*R_q*S_q + 2*B*D_v*H*R_k*BlockStat(key_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)` | 4,325,376 | FLOP | partial | schedule, scheduled_lengths | graph category |  |
| F_attention[executed_graph] | `work.expanded.graph.attention` | `2*B*D_n*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + 2*B*D_r*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + 2*B*D_v*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)` | 2,293,760 | FLOP | partial | schedule, scheduled_lengths | graph category |  |
| F_compiled | `work.compiled` | `from compilation evidence only` |  | FLOP | unknown | compile_evidence | compiled instruction-level work | useful; rect; executed_graph |
| F_A_minus_F_E | `work.path_difference` | `-2*B*D_n*H*R_k*S_k + 2*B*D_n*H*R_k*S_q - 2*B*D_v*H*R_k*S_k + 2*B*D_v*H*R_k*S_q + 2*B*H*(D_r + 2*R_k)*CausalVisibleCount(S_q, S_k, Delta) - 2*B*H*(D_n + D_r + D_v)*CausalVisibleCount(S_q, S_k, Delta)` | 532,480 | FLOP | bound |  | absorbed two-step minus expanded, same projection rows and same C (useful) |  |
| F_A_minus_F_E[spec_closed_form] | `work.path_difference.closed_form` | `2*B*H*(-D_n - D_v + 2*R_k)*CausalVisibleCount(S_q, S_k, Delta) + 2*R_k*(D_n + D_v)*(-B*H*S_k + B*H*S_q)` | 532,480 | FLOP | bound |  | 2BH Rk (Dn+Dv)(Sq-Sk) + 2C(2Rk-Dn-Dv): full projection, no cache, two-step variant |  |
| equivalence[expanded vs absorbed] | `numerics.equivalence` | `matrix associativity: Q^n (W^k)^T C_k^T = Q^n (C_k W^k)^T` | equal over the real numbers |  | bound |  | the two paths express the same result in exact real arithmetic | bitwise equality; an error bound; a measured deviation |
| acceptance[declared] | `numerics.acceptance` | `caller-declared tolerances per test level` |  |  | unknown | acceptance | numeric acceptance declared with the task | a measured error; a proof that the variant meets it |
| W_vec[mul] | `work.vector.mul` | `B*D_v*H*BlockStat(row_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + B*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + B*H*BlockStat(row_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)` | 52,480 | element-ops | partial | schedule, scheduled_lengths | logical count before redundancy elimination; not instructions | vector instruction count; MXU FLOPs |
| W_vec[mask] | `work.vector.mask` | `B*H*BlockStat(masked_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)` | 8192 | element-ops | partial | schedule, scheduled_lengths | logical count before redundancy elimination; not instructions | vector instruction count; MXU FLOPs |
| W_vec[cmp] | `work.vector.cmp` | `B*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)` | 20,480 | element-ops | partial | schedule, scheduled_lengths | logical count before redundancy elimination; not instructions | vector instruction count; MXU FLOPs |
| W_vec[broadcast] | `work.vector.broadcast` | `B*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)` | 20,480 | element-ops | partial | broadcast_materialized (undeclared backend behaviour), schedule, scheduled_lengths | logical count before redundancy elimination; not instructions | vector instruction count; MXU FLOPs |
| W_vec[transpose] | `work.vector.transpose` | `B*D_n*H*BlockStat(key_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)` | 30,720 | element-ops | partial | operand_transpose_materialized (undeclared backend behaviour), schedule, scheduled_lengths | logical count before redundancy elimination; not instructions | vector instruction count; MXU FLOPs |
| W_vec[exp] | `work.vector.exp` | `B*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + B*H*BlockStat(row_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)` | 21,760 | element-ops | partial | schedule, scheduled_lengths | logical count before redundancy elimination; not instructions | vector instruction count; MXU FLOPs |
| W_vec[add] | `work.vector.add` | `B*D_v*H*BlockStat(row_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + B*H*S_q + B*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + Piecewise((B*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k), D_r > 0), (0, True))` | 72,192 | element-ops | partial | schedule, scheduled_lengths | logical count before redundancy elimination; not instructions | vector instruction count; MXU FLOPs |
| W_vec[div] | `work.vector.div` | `B*D_v*H*S_q` | 12,288 | element-ops | partial | final_normalize, schedule, scheduled_lengths | logical count before redundancy elimination; not instructions | vector instruction count; MXU FLOPs |
| W_vec[recip] | `work.vector.recip` | `0` | 0 | element-ops | partial | schedule, scheduled_lengths | logical count before redundancy elimination; not instructions | vector instruction count; MXU FLOPs |
| W_vec[log] | `work.vector.log` | `B*H*S_q` | 512 | element-ops | partial | schedule, scheduled_lengths | logical count before redundancy elimination; not instructions | vector instruction count; MXU FLOPs |
| W_vec[cast] | `work.vector.cast` | `B*D_v*H*S_q + B*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)` | 32,768 | element-ops | partial | cast_points, schedule, scheduled_lengths | logical count before redundancy elimination; not instructions | vector instruction count; MXU FLOPs |
| W_resource[vpu] | `work.resource.vpu` | `2*B*D_v*H*S_q + 2*B*D_v*H*BlockStat(row_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + 2*B*H*S_q + B*H*BlockStat(masked_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + 2*B*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + 2*B*H*BlockStat(row_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + Piecewise((B*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k), D_r > 0), (0, True))` | 159,232 | element-ops | partial | cast_points, final_normalize, schedule, scheduled_lengths | per resource class u (for W_u/P_u) |  |
| W_resource[reduce] | `work.resource.reduce` | `2*B*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) - B*H*BlockStat(row_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)` | 39,680 | element-ops | partial | schedule, scheduled_lengths | per resource class u (for W_u/P_u) |  |
| W_resource[layout] | `work.resource.layout` | `B*D_n*H*BlockStat(key_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + B*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)` | 51,200 | element-ops | partial | broadcast_materialized (undeclared backend behaviour), operand_transpose_materialized (undeclared backend behaviour), schedule, scheduled_lengths | per resource class u (for W_u/P_u) |  |
| W_resource[exp] | `work.resource.exp` | `B*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + B*H*BlockStat(row_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)` | 21,760 | element-ops | partial | schedule, scheduled_lengths | per resource class u (for W_u/P_u) |  |
| exp_count_executed | `work.exp.executed` | `B*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)` | 20,480 | exp | partial | schedule, scheduled_lengths | exp over executed cells (masked included) | C_valid |

Assumptions:
- `F_total[useful]`: N_Qproj=B*H*S_q, N_Kproj=B*H*S_k, N_Vproj=B*H*S_k
- `F_total[useful]`: C already includes B and H
- `F_total[useful]`: FMA = 2; not an instruction count
- `F_E[spec_closed_form]`: equals F_total[useful] only when projection_scope = full
- `N_Qproj`: expanded path: every counted K/V row is projected in this scope
- `N_Qproj`: distinct rows, not row visits: a strategy that re-projects inside the KV loop raises the executed-graph work, not this count (spec 5.2)
- `N_Kproj[executed_graph]`: expanded path: every counted K/V row is projected in this scope
- `N_Kproj`: expanded path: every counted K/V row is projected in this scope
- `N_Kproj`: distinct rows, not row visits: a strategy that re-projects inside the KV loop raises the executed-graph work, not this count (spec 5.2)
- `N_Vproj[executed_graph]`: expanded path: every counted K/V row is projected in this scope
- `N_Vproj`: expanded path: every counted K/V row is projected in this scope
- `N_Vproj`: distinct rows, not row visits: a strategy that re-projects inside the KV loop raises the executed-graph work, not this count (spec 5.2)
- `F_proj[general]`: N_Qproj=B*H*S_q, N_Kproj=B*H*S_k, N_Vproj=B*H*S_k
- `F_proj[general]`: expanded path: every counted K/V row is projected in this scope
- `F_proj[general]`: useful-work counting: each row projected once. When the strategy projects inside the KV loop, the repeated work is in F_projection[executed_graph] (spec 5.2), not here.
- `F_total[rect]`: rectangle selection policy 'skip_future'
- `F_total[rect]`: projection counted once per row: the executed graph may differ where a tile is re-projected or skipped
- `F_total[rect]`: scheduled extent undeclared: assumed equal to the active extent (scenario)
- `F_total[executed_graph]`: K/V projected inside the KV loop: projected rows = executed key columns (each q-block re-projects the tile)
- `F_total[executed_graph]`: entry scope: seven latent inputs; Q^n, K^n, V projected inside this scope with the declared projection_scope
- `F_total[executed_graph]`: outside the entry scope and therefore counted nowhere in this report: RMSNorm/LayerNorm, the RoPE rotation that produces Q^r/K^r, the latent down-projection that produces C_q/C_k, the output projection W^O after this attention, and the KV-cache update
- `F_total[executed_graph]`: loop nest scenario q_outer_kv_inner: one program per (batch, head-group of h_pp heads, q-block) scanning selected KV blocks
- `F_total[executed_graph]`: accumulator width D_A = D_v (expanded)
- `F_total[executed_graph]`: storage levels are an analysis assignment (residency windows -> vmem, vector temporaries -> vreg), not compiler allocation
- `F_total[executed_graph]`: scheduled extent undeclared: assumed equal to the active extent (scenario)
- `F_projection[executed_graph]`: scheduled extent undeclared: assumed equal to the active extent (scenario)
- `F_attention[executed_graph]`: scheduled extent undeclared: assumed equal to the active extent (scenario)
- `F_A_minus_F_E`: computation comparison only, not runtime
- `F_A_minus_F_E`: reduces to 2BH Rk(Dn+Dv)(Sq-Sk) + 2C(2Rk-Dn-Dv) for full projection (two-step variant)
- `F_A_minus_F_E[spec_closed_form]`: sign decided by 2Rk vs Dn+Dv in square prefill
- `equivalence[expanded vs absorbed]`: low-precision reassociation does not guarantee bitwise equality: the operand and accumulator dtypes, the accumulation order and the cast sites all differ between the paths
- `acceptance[declared]`: declared only: this model runs no arithmetic, so no tolerance is evaluated here
- `W_vec[mul]`: scheduled extent undeclared: assumed equal to the active extent (scenario)
- `W_vec[mask]`: scheduled extent undeclared: assumed equal to the active extent (scenario)
- `W_vec[cmp]`: scheduled extent undeclared: assumed equal to the active extent (scenario)
- `W_vec[broadcast]`: scheduled extent undeclared: assumed equal to the active extent (scenario)
- `W_vec[transpose]`: scheduled extent undeclared: assumed equal to the active extent (scenario)
- `W_vec[exp]`: scheduled extent undeclared: assumed equal to the active extent (scenario)
- `W_vec[add]`: scheduled extent undeclared: assumed equal to the active extent (scenario)
- `W_vec[div]`: scheduled extent undeclared: assumed equal to the active extent (scenario)
- `W_vec[recip]`: scheduled extent undeclared: assumed equal to the active extent (scenario)
- `W_vec[log]`: scheduled extent undeclared: assumed equal to the active extent (scenario)
- `W_vec[cast]`: scheduled extent undeclared: assumed equal to the active extent (scenario)
- `W_resource[vpu]`: scheduled extent undeclared: assumed equal to the active extent (scenario)
- `W_resource[reduce]`: scheduled extent undeclared: assumed equal to the active extent (scenario)
- `W_resource[layout]`: scheduled extent undeclared: assumed equal to the active extent (scenario)
- `W_resource[exp]`: scheduled extent undeclared: assumed equal to the active extent (scenario)
- `exp_count_executed`: scheduled extent undeclared: assumed equal to the active extent (scenario)

<details><summary>Provenance (bindings, coverage, source, evidence)</summary>

| metric | source | coverage | bindings | evidence |
|---|---|---|---|---|
| F_Q[useful] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"B": 2, "D_n": 24, "H": 4, "R_q": 16, "S_q": 64}` |  |
| F_K[useful] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"B": 2, "D_n": 24, "H": 4, "R_k": 32, "S_k": 64}` |  |
| F_V[useful] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"B": 2, "D_v": 24, "H": 4, "R_k": 32, "S_k": 64}` |  |
| F_QK_n[useful] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"B": 2, "D_n": 24, "Delta": 0, "H": 4, "S_k": 64, "S_q": 64}` |  |
| F_QK_r[useful] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"B": 2, "D_r": 8, "Delta": 0, "H": 4, "S_k": 64, "S_q": 64}` |  |
| F_PV[useful] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"B": 2, "D_v": 24, "Delta": 0, "H": 4, "S_k": 64, "S_q": 64}` |  |
| F_total[useful] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"B": 2, "D_n": 24, "D_r": 8, "D_v": 24, "Delta": 0, "H": 4, "R_k": 32, "R_q": 16, "S_k...` |  |
| F_E[spec_closed_form] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"B": 2, "D_n": 24, "D_r": 8, "D_v": 24, "Delta": 0, "H": 4, "R_k": 32, "R_q": 16, "S_k...` |  |
| N_Qproj | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"B": 2, "H": 4, "S_q": 64}` |  |
| N_Kproj[executed_graph] | derived | incomplete: depends on undeclared/unbound schedule, scheduled_lengths | `{"B": 2, "Delta": 0, "H": 4, "S_k": 64, "S_q": 64, "b_k": 16, "b_q": 16}` |  |
| N_Kproj | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"B": 2, "H": 4, "S_k": 64}` |  |
| N_Vproj[executed_graph] | derived | incomplete: depends on undeclared/unbound schedule, scheduled_lengths | `{"B": 2, "Delta": 0, "H": 4, "S_k": 64, "S_q": 64, "b_k": 16, "b_q": 16}` |  |
| N_Vproj | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"B": 2, "H": 4, "S_k": 64}` |  |
| F_proj[general] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"B": 2, "D_n": 24, "D_v": 24, "H": 4, "R_k": 32, "R_q": 16, "S_k": 64, "S_q": 64}` |  |
| F_total[rect] | derived | incomplete: depends on undeclared/unbound scheduled_lengths | `{"B": 2, "D_n": 24, "D_r": 8, "D_v": 24, "Delta": 0, "H": 4, "R_k": 32, "R_q": 16, "S_k...` |  |
| F_total[executed_graph] | derived | incomplete: depends on undeclared/unbound schedule, scheduled_lengths | `{"B": 2, "D_n": 24, "D_r": 8, "D_v": 24, "Delta": 0, "H": 4, "R_k": 32, "R_q": 16, "S_k...` |  |
| F_projection[executed_graph] | derived | incomplete: depends on undeclared/unbound schedule, scheduled_lengths | `{"B": 2, "D_n": 24, "D_v": 24, "Delta": 0, "H": 4, "R_k": 32, "R_q": 16, "S_k": 64, "S_...` |  |
| F_attention[executed_graph] | derived | incomplete: depends on undeclared/unbound schedule, scheduled_lengths | `{"B": 2, "D_n": 24, "D_r": 8, "D_v": 24, "Delta": 0, "H": 4, "S_k": 64, "S_q": 64, "b_k...` |  |
| F_compiled | compiled | incomplete: depends on undeclared/unbound compile_evidence | `{}` |  |
| F_A_minus_F_E | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"B": 2, "D_n": 24, "D_r": 8, "D_v": 24, "Delta": 0, "H": 4, "R_k": 32, "S_k": 64, "S_q...` |  |
| F_A_minus_F_E[spec_closed_form] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"B": 2, "D_n": 24, "D_v": 24, "Delta": 0, "H": 4, "R_k": 32, "S_k": 64, "S_q": 64}` |  |
| equivalence[expanded vs absorbed] | definition | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{}` |  |
| acceptance[declared] | input | incomplete: depends on undeclared/unbound acceptance | `{}` |  |
| W_vec[mul] | derived | incomplete: depends on undeclared/unbound schedule, scheduled_lengths | `{"B": 2, "D_v": 24, "Delta": 0, "H": 4, "S_k": 64, "S_q": 64, "b_k": 16, "b_q": 16}` |  |
| W_vec[mask] | derived | incomplete: depends on undeclared/unbound schedule, scheduled_lengths | `{"B": 2, "Delta": 0, "H": 4, "S_k": 64, "S_q": 64, "b_k": 16, "b_q": 16}` |  |
| W_vec[cmp] | derived | incomplete: depends on undeclared/unbound schedule, scheduled_lengths | `{"B": 2, "Delta": 0, "H": 4, "S_k": 64, "S_q": 64, "b_k": 16, "b_q": 16}` |  |
| W_vec[broadcast] | derived | incomplete: depends on undeclared/unbound broadcast_materialized (undeclared backend behaviour), schedule, scheduled_lengths | `{"B": 2, "Delta": 0, "H": 4, "S_k": 64, "S_q": 64, "b_k": 16, "b_q": 16}` |  |
| W_vec[transpose] | derived | incomplete: depends on undeclared/unbound operand_transpose_materialized (undeclared backend behaviour), schedule, scheduled_lengths | `{"B": 2, "D_n": 24, "Delta": 0, "H": 4, "S_k": 64, "S_q": 64, "b_k": 16, "b_q": 16}` |  |
| W_vec[exp] | derived | incomplete: depends on undeclared/unbound schedule, scheduled_lengths | `{"B": 2, "Delta": 0, "H": 4, "S_k": 64, "S_q": 64, "b_k": 16, "b_q": 16}` |  |
| W_vec[add] | derived | incomplete: depends on undeclared/unbound schedule, scheduled_lengths | `{"B": 2, "D_r": 8, "D_v": 24, "Delta": 0, "H": 4, "S_k": 64, "S_q": 64, "b_k": 16, "b_q...` |  |
| W_vec[div] | derived | incomplete: depends on undeclared/unbound final_normalize, schedule, scheduled_lengths | `{"B": 2, "D_v": 24, "H": 4, "S_q": 64}` |  |
| W_vec[recip] | derived | incomplete: depends on undeclared/unbound schedule, scheduled_lengths | `{}` |  |
| W_vec[log] | derived | incomplete: depends on undeclared/unbound schedule, scheduled_lengths | `{"B": 2, "H": 4, "S_q": 64}` |  |
| W_vec[cast] | derived | incomplete: depends on undeclared/unbound cast_points, schedule, scheduled_lengths | `{"B": 2, "D_v": 24, "Delta": 0, "H": 4, "S_k": 64, "S_q": 64, "b_k": 16, "b_q": 16}` |  |
| W_resource[vpu] | derived | incomplete: depends on undeclared/unbound cast_points, final_normalize, schedule, scheduled_lengths | `{"B": 2, "D_r": 8, "D_v": 24, "Delta": 0, "H": 4, "S_k": 64, "S_q": 64, "b_k": 16, "b_q...` |  |
| W_resource[reduce] | derived | incomplete: depends on undeclared/unbound schedule, scheduled_lengths | `{"B": 2, "Delta": 0, "H": 4, "S_k": 64, "S_q": 64, "b_k": 16, "b_q": 16}` |  |
| W_resource[layout] | derived | incomplete: depends on undeclared/unbound broadcast_materialized (undeclared backend behaviour), operand_transpose_materialized (undeclared backend behaviour), schedule, scheduled_lengths | `{"B": 2, "D_n": 24, "Delta": 0, "H": 4, "S_k": 64, "S_q": 64, "b_k": 16, "b_q": 16}` |  |
| W_resource[exp] | derived | incomplete: depends on undeclared/unbound schedule, scheduled_lengths | `{"B": 2, "Delta": 0, "H": 4, "S_k": 64, "S_q": 64, "b_k": 16, "b_q": 16}` |  |
| exp_count_executed | derived | incomplete: depends on undeclared/unbound schedule, scheduled_lengths | `{"B": 2, "Delta": 0, "H": 4, "S_k": 64, "S_q": 64, "b_k": 16, "b_q": 16}` |  |

</details>

## 4. bytes: interface and materialization

| metric | formula id | expression | value | unit | status | missing | scope | not equivalent to |
|---|---|---|---|---|---|---|---|---|
| bytes[q_latent] | `bytes.logical.q_latent` | `B*R_q*S_q*s_Cq` | 4096 | byte | bound |  | logical size at this tensor's own allocation extent, per-tensor dtype |  |
| bytes[kv_latent] | `bytes.logical.kv_latent` | `B*R_k*S_k*s_Ck` | 8192 | byte | bound |  | logical size at this tensor's own allocation extent, per-tensor dtype |  |
| bytes[q_pe] | `bytes.logical.q_pe` | `B*D_r*H*S_q*s_Qr` | 8192 | byte | bound |  | logical size at this tensor's own allocation extent, per-tensor dtype |  |
| bytes[k_pe] | `bytes.logical.k_pe` | `B*D_r*S_k*s_Kr` | 2048 | byte | bound |  | logical size at this tensor's own allocation extent, per-tensor dtype |  |
| bytes[w_q_nope] | `bytes.logical.w_q_nope` | `D_n*H*R_q*s_Wq` | 3072 | byte | bound |  | logical size at this tensor's own allocation extent, per-tensor dtype |  |
| bytes[w_k_nope] | `bytes.logical.w_k_nope` | `D_n*H*R_k*s_Wk` | 6144 | byte | bound |  | logical size at this tensor's own allocation extent, per-tensor dtype |  |
| bytes[w_v] | `bytes.logical.w_v` | `D_v*H*R_k*s_Wv` | 6144 | byte | bound |  | logical size at this tensor's own allocation extent, per-tensor dtype |  |
| M_in | `bytes.M_in` | `4096*s_Ck + 2048*s_Cq + 1024*s_Kr + 4096*s_Qr + 3072*s_Wk + 1536*s_Wq + 3072*s_Wv` | 37,888 | byte | bound |  | sum of the seven inputs, each at its OWN ALLOCATION extent (capacity_shape where declared) | HBM traffic lower bound under caching/reuse; M_in[metadata_ledger], which sums the LOGICAL shapes rather than the allocations |
| M_O | `bytes.M_O` | `B*D_v*H*S_q*s_O` | 24,576 | byte | bound |  | output O at active extents (s_O * B H S_q * D_v; ragged: H sum_b Sq_b) |  |
| M_LSE | `bytes.M_LSE` | `B*H*S_q*s_LSE` | 2048 | byte | bound |  | LSE at active extents |  |
| M_in[metadata_ledger] | `bytes.ledger.logical_total` | `sum of per-tensor prod(shape)*bytes(dtype)` | 37,888 | byte | bound |  | sum of the LOGICAL sizes prod(shape)*s from the actual metadata, not the allocations | M_in, which sums each tensor at its allocation extent; M_alloc_unique[metadata_ledger], the allocation total with aliases counted once |
| M_alloc_unique[metadata_ledger] | `bytes.ledger.alloc_unique` | `allocated bytes with aliases counted once` | 37,888 | byte | bound |  | capacity_shape and aliases honoured | the union of address ranges |
| HBM_mat[K^n,V] | `bytes.mat.mat_expanded_kv` | `B*D_n*H*S_k*s_Kn + B*D_n*H*s_Kn*BlockStat(key_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + B*D_v*H*S_k*s_V + B*D_v*H*s_V*BlockStat(key_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)` | 0 | byte | not_applicable |  | write + read traffic if materialized (read under the declared scan policy) | interface bytes |
| M_mat[K^n,V] | `bytes.mat.size.mat_expanded_kv` | `B*H*S_k*(D_n*s_Kn + D_v*s_V)` | 0 | byte | not_applicable |  | logical size of the materialized intermediate itself (spec 6.2 M^mat) | its HBM traffic; interface bytes |
| HBM_mat[Q^n] | `bytes.mat.mat_q_nope` | `2*B*D_n*H*S_q*s_Qn` | 0 | byte | not_applicable |  | write + read traffic if materialized (read under the declared scan policy) | interface bytes |
| M_mat[Q^n] | `bytes.mat.size.mat_q_nope` | `B*D_n*H*S_q*s_Qn` | 0 | byte | not_applicable |  | logical size of the materialized intermediate itself (spec 6.2 M^mat) | its HBM traffic; interface bytes |

Assumptions:
- `bytes[q_latent]`: no capacity_shape declared for this tensor: its own shape is the allocation extent
- `bytes[kv_latent]`: no capacity_shape declared for this tensor: its own shape is the allocation extent
- `bytes[q_pe]`: no capacity_shape declared for this tensor: its own shape is the allocation extent
- `bytes[k_pe]`: no capacity_shape declared for this tensor: its own shape is the allocation extent
- `bytes[w_q_nope]`: no capacity_shape declared for this tensor: its own shape is the allocation extent
- `bytes[w_k_nope]`: no capacity_shape declared for this tensor: its own shape is the allocation extent
- `bytes[w_v]`: no capacity_shape declared for this tensor: its own shape is the allocation extent
- `M_in`: per-tensor storage widths s_x
- `M_in`: aliases not deduplicated in this symbolic sum (see ledger)
- `M_in`: equals the ledger's allocated_unique_total when no two roles alias
- `M_in[metadata_ledger]`: spec 6.1 M_logical: the declared shape, ignoring any larger capacity_shape
- `HBM_mat[K^n,V]`: conditional on materialize.expanded_kv = False
- `HBM_mat[K^n,V]`: write traffic B*D_n*H*S_k*s_Kn + B*D_v*H*S_k*s_V; read traffic B*D_n*H*s_Kn*BlockStat(key_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + B*D_v*H*s_V*BlockStat(key_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)
- `M_mat[K^n,V]`: conditional on materialize.expanded_kv = False
- `HBM_mat[Q^n]`: conditional on materialize.q_nope = False
- `HBM_mat[Q^n]`: write traffic B*D_n*H*S_q*s_Qn; read traffic B*D_n*H*S_q*s_Qn
- `M_mat[Q^n]`: conditional on materialize.q_nope = False

<details><summary>Provenance (bindings, coverage, source, evidence)</summary>

| metric | source | coverage | bindings | evidence |
|---|---|---|---|---|
| bytes[q_latent] | input | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"B": 2, "R_q": 16, "S_q": 64, "s_Cq": 2}` |  |
| bytes[kv_latent] | input | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"B": 2, "R_k": 32, "S_k": 64, "s_Ck": 2}` |  |
| bytes[q_pe] | input | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"B": 2, "D_r": 8, "H": 4, "S_q": 64, "s_Qr": 2}` |  |
| bytes[k_pe] | input | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"B": 2, "D_r": 8, "S_k": 64, "s_Kr": 2}` |  |
| bytes[w_q_nope] | input | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"D_n": 24, "H": 4, "R_q": 16, "s_Wq": 2}` |  |
| bytes[w_k_nope] | input | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"D_n": 24, "H": 4, "R_k": 32, "s_Wk": 2}` |  |
| bytes[w_v] | input | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"D_v": 24, "H": 4, "R_k": 32, "s_Wv": 2}` |  |
| M_in | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"s_Ck": 2, "s_Cq": 2, "s_Kr": 2, "s_Qr": 2, "s_Wk": 2, "s_Wq": 2, "s_Wv": 2}` |  |
| M_O | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"B": 2, "D_v": 24, "H": 4, "S_q": 64, "s_O": 2}` |  |
| M_LSE | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"B": 2, "H": 4, "S_q": 64, "s_LSE": 4}` |  |
| M_in[metadata_ledger] | input | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{}` |  |
| M_alloc_unique[metadata_ledger] | input | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{}` |  |
| HBM_mat[K^n,V] | conditional | not applicable under the declared configuration | `{"B": 2, "D_n": 24, "D_v": 24, "Delta": 0, "H": 4, "S_k": 64, "S_q": 64, "b_k": 16, "b_...` |  |
| M_mat[K^n,V] | conditional | not applicable under the declared configuration | `{"B": 2, "D_n": 24, "D_v": 24, "H": 4, "S_k": 64, "s_Kn": 2, "s_V": 2}` |  |
| HBM_mat[Q^n] | conditional | not applicable under the declared configuration | `{"B": 2, "D_n": 24, "H": 4, "S_q": 64, "s_Qn": 2}` |  |
| M_mat[Q^n] | conditional | not applicable under the declared configuration | `{"B": 2, "D_n": 24, "H": 4, "S_q": 64, "s_Qn": 2}` |  |

</details>

## 5. bytes: transfer requests per path

| metric | formula id | expression | value | unit | status | missing | scope | not equivalent to |
|---|---|---|---|---|---|---|---|---|
| B[hbm_to_vmem] | `bytes.path.hbm_to_vmem` | `B*D_n*H*R_k*s_Wk*ceiling(S_q/b_q) + B*D_n*H*R_q*s_Wq*ceiling(S_q/b_q) + B*D_r*H*S_q*s_Qr + B*D_r*s_Kr*ceiling(H/h_pp)*BlockStat(key_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + B*D_v*H*R_k*s_Wv*ceiling(S_q/b_q) + B*R_k*s_Ck*ceiling(H/h_pp)*BlockStat(key_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + B*R_q*S_q*s_Cq*ceiling(H/h_pp)` | 249,856 | byte | partial | schedule, scheduled_lengths | sum_r m_r n_r over the scenario's transfer events | physical HBM traffic with caching; spill traffic |
| B[vmem_to_hbm] | `bytes.path.vmem_to_hbm` | `B*D_v*H*S_q*s_O + B*H*S_q*s_LSE` | 26,624 | byte | partial | schedule, scheduled_lengths | sum_r m_r n_r over the scenario's transfer events | physical HBM traffic with caching; spill traffic |
| B[vmem_to_vreg] | `bytes.path.vmem_to_vreg` | `B*D_n*H*b_k*s_Kn*BlockStat(selected, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + B*D_n*H*b_q*s_Qn*ceiling(S_q/b_q) + B*D_r*H*b_k*s_Kr*BlockStat(selected, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + B*D_r*H*b_q*s_Qr*ceiling(S_q/b_q) + B*D_v*H*b_k*s_V*BlockStat(selected, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)` | 176,128 | byte | partial | q_resident, register_schedule_evidence, scheduled_lengths | sum_r m_r n_r over the scenario's transfer events | physical HBM traffic with caching; spill traffic |
| B[vreg_to_vmem] | `bytes.path.vreg_to_vmem` | `sum_r m_r n_r (no modeled events)` |  | byte | unknown | register_schedule_evidence | path traffic | spill traffic; physical traffic with caching |

Assumptions:
- `B[hbm_to_vmem]`: scenario q_outer_kv_inner; weights re-read per program
- `B[hbm_to_vmem]`: no hardware cache effects are modeled: every event is counted at its declared multiplicity
- `B[hbm_to_vmem]`: scheduled extent undeclared: assumed equal to the active extent (scenario)
- `B[vmem_to_hbm]`: scenario q_outer_kv_inner; weights re-read per program
- `B[vmem_to_hbm]`: no hardware cache effects are modeled: every event is counted at its declared multiplicity
- `B[vmem_to_hbm]`: scheduled extent undeclared: assumed equal to the active extent (scenario)
- `B[vmem_to_vreg]`: scenario q_outer_kv_inner; weights re-read per program
- `B[vmem_to_vreg]`: no hardware cache effects are modeled: every event is counted at its declared multiplicity
- `B[vmem_to_vreg]`: operand-streaming scenario without register-schedule evidence
- `B[vmem_to_vreg]`: includes scenario events whose condition is undeclared (stream_Qn_operand, stream_Qr_operand, stream_K_tile, stream_V_tile, stream_Kr_tile): they are counted at their declared multiplicity and named in missing_fields
- `B[vmem_to_vreg]`: scheduled extent undeclared: assumed equal to the active extent (scenario)

<details><summary>Provenance (bindings, coverage, source, evidence)</summary>

| metric | source | coverage | bindings | evidence |
|---|---|---|---|---|
| B[hbm_to_vmem] | conditional | incomplete: depends on undeclared/unbound schedule, scheduled_lengths | `{"B": 2, "D_n": 24, "D_r": 8, "D_v": 24, "Delta": 0, "H": 4, "R_k": 32, "R_q": 16, "S_k...` |  |
| B[vmem_to_hbm] | conditional | incomplete: depends on undeclared/unbound schedule, scheduled_lengths | `{"B": 2, "D_v": 24, "H": 4, "S_q": 64, "s_LSE": 4, "s_O": 2}` |  |
| B[vmem_to_vreg] | conditional | incomplete: depends on undeclared/unbound q_resident, register_schedule_evidence, scheduled_lengths | `{"B": 2, "D_n": 24, "D_r": 8, "D_v": 24, "Delta": 0, "H": 4, "S_k": 64, "S_q": 64, "b_k...` |  |
| B[vreg_to_vmem] | conditional | incomplete: depends on undeclared/unbound register_schedule_evidence | `{}` |  |

</details>

## 6. local objects (symbolic sizes, spec 7.1)

| metric | formula id | expression | value | unit | status | missing | scope | not equivalent to |
|---|---|---|---|---|---|---|---|---|
| local[q_latent_window] | `local.q_latent_window.bytes` | `R_q*b_q*s_Cq` | 512 | byte | partial | b_rq | one logical object; level=vmem; live in ['q_proj'] | VREG allocation; spill traffic |
| local[wq_rank_tile] | `local.wq_rank_tile.bytes` | `D_n*R_q*s_Wq` | 768 | byte | partial | b_rq | one logical object; level=vmem; live in ['q_proj'] | VREG allocation; spill traffic |
| local[q_proj_acc] | `local.q_proj_acc.bytes` | `D_n*b_q*s_Qacc` | 1536 | byte | partial | projection_accumulator_dtype | one logical object; level=vreg; live in ['q_proj'] | VREG allocation; spill traffic |
| local[q_nope_operand] | `local.q_nope_operand.bytes` | `D_n*b_q*s_Qn` | 768 | byte | partial | projected_operand_dtype, q_resident | one logical object; level=vmem; live in ['kv_proj', 'score', 'softmax', 'pv', 'finalize'] | VREG allocation; spill traffic |
| local[kv_latent_rank_tile] | `local.kv_latent_rank_tile.bytes` | `R_k*b_k*n_buf*s_Ck` | 2048 | byte | partial | b_rk | one logical object x its buffer count; level=vmem; live in ['kv_proj']; residency window of n_buf=n_buf buffers (7.3 allocation, not the 7.1 logical size) | VREG allocation; spill traffic; the 7.1 logical size of one buffer |
| local[wk_rank_tile] | `local.wk_rank_tile.bytes` | `D_n*R_k*s_Wk` | 1536 | byte | partial | b_rk | one logical object; level=vmem; live in ['kv_proj'] | VREG allocation; spill traffic |
| local[wv_rank_tile] | `local.wv_rank_tile.bytes` | `D_v*R_k*s_Wv` | 1536 | byte | partial | b_rk | one logical object; level=vmem; live in ['kv_proj'] | VREG allocation; spill traffic |
| local[k_nope_tile] | `local.k_nope_tile.bytes` | `D_n*b_k*s_Kn` | 768 | byte | partial | projected_operand_dtype | one logical object; level=vmem; live in ['kv_proj', 'score'] | VREG allocation; spill traffic |
| local[v_tile] | `local.v_tile.bytes` | `D_v*b_k*s_V` | 768 | byte | partial | projected_operand_dtype, v_load | one logical object; level=vmem; live in ['kv_proj', 'score', 'softmax', 'pv'] | VREG allocation; spill traffic |
| local[score_X] | `local.score_X.bytes` | `b_k*b_q*s_X` | 1024 | byte | bound |  | one logical object; level=vreg; live in ['score', 'softmax'] | VREG allocation; spill traffic |
| local[exp_E] | `local.exp_E.bytes` | `b_k*b_q*s_E` | 1024 | byte | bound |  | one logical object; level=vreg; live in ['softmax', 'pv'] | VREG allocation; spill traffic |
| local[p_operand] | `local.p_operand.bytes` | `b_k*b_q*s_P` | 512 | byte | bound |  | one logical object; level=vreg; live in ['pv'] | VREG allocation; spill traffic |
| local[row_state_m] | `local.row_state_m.bytes` | `b_q*s_state` | 64 | byte | bound |  | one logical object; level=vreg; live in ['kv_proj', 'score', 'softmax', 'pv', 'finalize'] | VREG allocation; spill traffic |
| local[row_state_l] | `local.row_state_l.bytes` | `b_q*s_state` | 64 | byte | bound |  | one logical object; level=vreg; live in ['kv_proj', 'score', 'softmax', 'pv', 'finalize'] | VREG allocation; spill traffic |
| local[row_alpha] | `local.row_alpha.bytes` | `b_q*s_state` | 64 | byte | bound |  | one logical object; level=vreg; live in ['softmax', 'pv'] | VREG allocation; spill traffic |
| local[accumulator_A] | `local.accumulator_A.bytes` | `D_v*b_q*s_A` | 1536 | byte | bound |  | one logical object; level=vreg; live in ['kv_proj', 'score', 'softmax', 'pv', 'finalize'] | VREG allocation; spill traffic |
| local[q_pe_tile] | `local.q_pe_tile.bytes` | `D_r*b_q*s_Qr` | 256 | byte | partial | q_resident | one logical object; level=vmem; live in ['kv_proj', 'score', 'softmax', 'pv', 'finalize'] | VREG allocation; spill traffic |
| local[k_pe_tile] | `local.k_pe_tile.bytes` | `D_r*b_k*n_buf*s_Kr` | 512 | byte | bound |  | one logical object x its buffer count; level=vmem; live in ['score']; residency window of n_buf=n_buf buffers (7.3 allocation, not the 7.1 logical size) | VREG allocation; spill traffic; the 7.1 logical size of one buffer |
| local[fixed_scratch] | `local.fixed_scratch.bytes` | `s_scratch` | 0 | byte | bound |  | one logical object; level=vmem; live in ['q_proj', 'kv_proj', 'score', 'softmax', 'pv', 'finalize'] | VREG allocation; spill traffic |
| local[out_tile] | `local.out_tile.bytes` | `D_v*b_q*s_O` | 768 | byte | bound |  | one logical object; level=vmem; live in ['finalize'] | VREG allocation; spill traffic |
| local[lse_tile] | `local.lse_tile.bytes` | `b_q*s_LSE` | 64 | byte | bound |  | one logical object; level=vmem; live in ['finalize'] | VREG allocation; spill traffic |
| layout[q_latent_window] | `layout.q_latent_window` | `L_l*L_s*s_Cq*ceiling(R_q/L_l)*ceiling(b_q/L_s) → 2*L_l*L_s*ceiling(16/L_l)*ceiling(16/L_s)` |  | byte | partial | L_l, L_s, layout.q_latent_window | layout coverage, kind=vector_tiled, level=vmem (scenario: kind undeclared) | register-file capacity |
| layout[wq_rank_tile] | `layout.wq_rank_tile` | `L_l*L_s*s_Wq*ceiling(D_n/L_l)*ceiling(R_q/L_s) → 2*L_l*L_s*ceiling(24/L_l)*ceiling(16/L_s)` |  | byte | partial | L_l, L_s, layout.wq_rank_tile | layout coverage, kind=vector_tiled, level=vmem (scenario: kind undeclared) | register-file capacity |
| layout[q_proj_acc] | `layout.q_proj_acc` | `1024*s_Qacc*ceiling(D_n/128)*ceiling(b_q/8)` | 8192 | byte | partial | layout.q_proj_acc | layout coverage, kind=vector_tiled, level=vreg (scenario: kind undeclared) | register-file capacity |
| layout[q_nope_operand] | `layout.q_nope_operand` | `L_l*L_s*s_Qn*ceiling(D_n/L_l)*ceiling(b_q/L_s) → 2*L_l*L_s*ceiling(24/L_l)*ceiling(16/L_s)` |  | byte | partial | L_l, L_s, layout.q_nope_operand | layout coverage, kind=vector_tiled, level=vmem (scenario: kind undeclared) | register-file capacity |
| layout[kv_latent_rank_tile] | `layout.kv_latent_rank_tile` | `L_l*L_s*n_buf*s_Ck*ceiling(R_k/L_l)*ceiling(b_k/L_s) → 4*L_l*L_s*ceiling(32/L_l)*ceiling(16/L_s)` |  | byte | partial | L_l, L_s, layout.kv_latent_rank_tile | layout coverage, kind=vector_tiled, level=vmem (scenario: kind undeclared) | register-file capacity |
| layout[wk_rank_tile] | `layout.wk_rank_tile` | `L_l*L_s*s_Wk*ceiling(D_n/L_l)*ceiling(R_k/L_s) → 2*L_l*L_s*ceiling(24/L_l)*ceiling(32/L_s)` |  | byte | partial | L_l, L_s, layout.wk_rank_tile | layout coverage, kind=vector_tiled, level=vmem (scenario: kind undeclared) | register-file capacity |
| layout[wv_rank_tile] | `layout.wv_rank_tile` | `L_l*L_s*s_Wv*ceiling(D_v/L_l)*ceiling(R_k/L_s) → 2*L_l*L_s*ceiling(24/L_l)*ceiling(32/L_s)` |  | byte | partial | L_l, L_s, layout.wv_rank_tile | layout coverage, kind=vector_tiled, level=vmem (scenario: kind undeclared) | register-file capacity |
| layout[k_nope_tile] | `layout.k_nope_tile` | `L_l*L_s*s_Kn*ceiling(D_n/L_l)*ceiling(b_k/L_s) → 2*L_l*L_s*ceiling(24/L_l)*ceiling(16/L_s)` |  | byte | partial | L_l, L_s, layout.k_nope_tile | layout coverage, kind=vector_tiled, level=vmem (scenario: kind undeclared) | register-file capacity |
| layout[v_tile] | `layout.v_tile` | `L_l*L_s*s_V*ceiling(D_v/L_l)*ceiling(b_k/L_s) → 2*L_l*L_s*ceiling(24/L_l)*ceiling(16/L_s)` |  | byte | partial | L_l, L_s, layout.v_tile | layout coverage, kind=vector_tiled, level=vmem (scenario: kind undeclared) | register-file capacity |
| layout[score_X] | `layout.score_X` | `1024*s_X*ceiling(b_k/128)*ceiling(b_q/8)` | 8192 | byte | partial | layout.score_X | layout coverage, kind=vector_tiled, level=vreg (scenario: kind undeclared) | register-file capacity |
| layout[exp_E] | `layout.exp_E` | `1024*s_E*ceiling(b_k/128)*ceiling(b_q/8)` | 8192 | byte | partial | layout.exp_E | layout coverage, kind=vector_tiled, level=vreg (scenario: kind undeclared) | register-file capacity |
| layout[p_operand] | `layout.p_operand` | `L_l*L_s*s_P*ceiling(b_k/L_l)*ceiling(b_q/L_s) → 2*L_l*L_s*ceiling(16/L_l)*ceiling(16/L_s)` |  | byte | partial | L_l, L_s, layout.p_operand | layout coverage, kind=vector_tiled, level=vreg (scenario: kind undeclared) | register-file capacity |
| layout[row_state_m] | `layout.row_state_m` | `1024*s_state*ceiling(b_q/8)` | 8192 | byte | partial | layout.row_state_m | layout coverage, kind=vector_tiled, level=vreg (scenario: kind undeclared) | register-file capacity |
| layout[row_state_l] | `layout.row_state_l` | `1024*s_state*ceiling(b_q/8)` | 8192 | byte | partial | layout.row_state_l | layout coverage, kind=vector_tiled, level=vreg (scenario: kind undeclared) | register-file capacity |
| layout[row_alpha] | `layout.row_alpha` | `1024*s_state*ceiling(b_q/8)` | 8192 | byte | partial | layout.row_alpha | layout coverage, kind=vector_tiled, level=vreg (scenario: kind undeclared) | register-file capacity |
| layout[accumulator_A] | `layout.accumulator_A` | `1024*s_A*ceiling(D_v/128)*ceiling(b_q/8)` | 8192 | byte | partial | layout.accumulator_A | layout coverage, kind=vector_tiled, level=vreg (scenario: kind undeclared) | register-file capacity |
| layout[q_pe_tile] | `layout.q_pe_tile` | `L_l*L_s*s_Qr*ceiling(D_r/L_l)*ceiling(b_q/L_s) → 2*L_l*L_s*ceiling(8/L_l)*ceiling(16/L_s)` |  | byte | partial | L_l, L_s, layout.q_pe_tile | layout coverage, kind=vector_tiled, level=vmem (scenario: kind undeclared) | register-file capacity |
| layout[k_pe_tile] | `layout.k_pe_tile` | `L_l*L_s*n_buf*s_Kr*ceiling(D_r/L_l)*ceiling(b_k/L_s) → 4*L_l*L_s*ceiling(8/L_l)*ceiling(16/L_s)` |  | byte | partial | L_l, L_s, layout.k_pe_tile | layout coverage, kind=vector_tiled, level=vmem (scenario: kind undeclared) | register-file capacity |
| layout[fixed_scratch] | `layout.fixed_scratch` | `s_scratch` | 0 | byte | bound |  | layout coverage, kind=raw_bytes, level=vmem | register-file capacity |
| layout[out_tile] | `layout.out_tile` | `L_l*L_s*s_O*ceiling(D_v/L_l)*ceiling(b_q/L_s) → 2*L_l*L_s*ceiling(24/L_l)*ceiling(16/L_s)` |  | byte | partial | L_l, L_s, layout.out_tile | layout coverage, kind=vector_tiled, level=vmem (scenario: kind undeclared) | register-file capacity |
| layout[lse_tile] | `layout.lse_tile` | `1024*s_LSE*ceiling(b_q/8)` | 8192 | byte | partial | layout.lse_tile | layout coverage, kind=vector_tiled, level=vmem (scenario: kind undeclared) | register-file capacity |
| HBM_mat[Q~] | `bytes.mat.mat_q_tilde` | `not formed in this variant` |  |  | not_applicable |  | structural absence in this algorithm path | an unknown value |
| M_mat[Q~] | `bytes.mat.size.mat_q_tilde` | `not formed in this variant` |  |  | not_applicable |  | structural absence in this algorithm path | an unknown value |
| HBM_mat[Z] | `bytes.mat.mat_z` | `not formed in this variant` |  |  | not_applicable |  | structural absence in this algorithm path | an unknown value |
| M_mat[Z] | `bytes.mat.size.mat_z` | `not formed in this variant` |  |  | not_applicable |  | structural absence in this algorithm path | an unknown value |
| local[q_tilde] | `local.q_tilde.bytes` | `not formed in this variant` |  |  | not_applicable |  | structural absence in this algorithm path | an unknown value |
| local[z_tile] | `local.z_tile.bytes` | `not formed in this variant` |  |  | not_applicable |  | structural absence in this algorithm path | an unknown value |
| F_output_projection[executed_graph] | `work.expanded.graph.output_projection` | `not formed in this variant` |  |  | not_applicable |  | structural absence in this algorithm path | an unknown value |

Assumptions:
- `local[q_latent_window]`: Q latent full window b_q x R_q
- `local[q_latent_window]`: b_rq undeclared: the full rank window is shown as the scenario
- `local[wq_rank_tile]`: W^q rank sub-tile b_rq x D_n (per head)
- `local[wq_rank_tile]`: b_rq undeclared: the full rank window is shown as the scenario
- `local[q_proj_acc]`: Q projection accumulator b_q x D_n
- `local[q_nope_operand]`: Q^n operand held across the KV loop (shortened when q_resident=False)
- `local[q_nope_operand]`: q_resident undeclared: the live-stage set printed in the scope is the scenario
- `local[kv_latent_rank_tile]`: KV latent (rank sub-)tile b_k x b_rk
- `local[kv_latent_rank_tile]`: b_rk undeclared: the full rank window is shown as the scenario
- `local[wk_rank_tile]`: W^k rank sub-tile
- `local[wk_rank_tile]`: b_rk undeclared: the full rank window is shown as the scenario
- `local[wv_rank_tile]`: W^v rank sub-tile
- `local[wv_rank_tile]`: b_rk undeclared: the full rank window is shown as the scenario
- `local[k_nope_tile]`: projected K^n tile b_k x D_n
- `local[v_tile]`: projected V tile b_k x D_v (live until PV)
- `local[v_tile]`: v_load undeclared: the live-stage set printed in the scope is the scenario
- `local[score_X]`: score tile X = gamma(...)+M
- `local[exp_E]`: E = exp(X - m')
- `local[p_operand]`: P/E operand in the matmul input dtype
- `local[row_state_m]`: running row max m (loop-carried)
- `local[row_state_l]`: running row sum l (loop-carried)
- `local[row_alpha]`: alpha = exp(m - m')
- `local[accumulator_A]`: attention accumulator A, width D_v (loop-carried)
- `local[q_pe_tile]`: Q^r operand held across the KV loop (shortened when q_resident=False)
- `local[q_pe_tile]`: q_resident undeclared: the live-stage set printed in the scope is the scenario
- `local[k_pe_tile]`: K^r tile for the current KV block
- `local[fixed_scratch]`: fixed scratch (d_s)
- `local[out_tile]`: output tile O in output dtype
- `local[lse_tile]`: LSE tile
- `layout[q_latent_window]`: LayoutBytes = s L_s L_l ceil(m/L_s) ceil(n/L_l) with profile tiles
- `layout[wq_rank_tile]`: LayoutBytes = s L_s L_l ceil(m/L_s) ceil(n/L_l) with profile tiles
- `layout[q_proj_acc]`: LayoutBytes = s L_s L_l ceil(m/L_s) ceil(n/L_l) with profile tiles
- `layout[q_nope_operand]`: LayoutBytes = s L_s L_l ceil(m/L_s) ceil(n/L_l) with profile tiles
- `layout[kv_latent_rank_tile]`: LayoutBytes = s L_s L_l ceil(m/L_s) ceil(n/L_l) with profile tiles
- `layout[wk_rank_tile]`: LayoutBytes = s L_s L_l ceil(m/L_s) ceil(n/L_l) with profile tiles
- `layout[wv_rank_tile]`: LayoutBytes = s L_s L_l ceil(m/L_s) ceil(n/L_l) with profile tiles
- `layout[k_nope_tile]`: LayoutBytes = s L_s L_l ceil(m/L_s) ceil(n/L_l) with profile tiles
- `layout[v_tile]`: LayoutBytes = s L_s L_l ceil(m/L_s) ceil(n/L_l) with profile tiles
- `layout[score_X]`: LayoutBytes = s L_s L_l ceil(m/L_s) ceil(n/L_l) with profile tiles
- `layout[exp_E]`: LayoutBytes = s L_s L_l ceil(m/L_s) ceil(n/L_l) with profile tiles
- `layout[p_operand]`: LayoutBytes = s L_s L_l ceil(m/L_s) ceil(n/L_l) with profile tiles
- `layout[row_state_m]`: LayoutBytes = s L_s L_l ceil(m/L_s) ceil(n/L_l) with profile tiles
- `layout[row_state_l]`: LayoutBytes = s L_s L_l ceil(m/L_s) ceil(n/L_l) with profile tiles
- `layout[row_alpha]`: LayoutBytes = s L_s L_l ceil(m/L_s) ceil(n/L_l) with profile tiles
- `layout[accumulator_A]`: LayoutBytes = s L_s L_l ceil(m/L_s) ceil(n/L_l) with profile tiles
- `layout[q_pe_tile]`: LayoutBytes = s L_s L_l ceil(m/L_s) ceil(n/L_l) with profile tiles
- `layout[k_pe_tile]`: LayoutBytes = s L_s L_l ceil(m/L_s) ceil(n/L_l) with profile tiles
- `layout[fixed_scratch]`: declared as a byte count, not an [m, n] element array: the 7.2 layout function does not apply
- `layout[out_tile]`: LayoutBytes = s L_s L_l ceil(m/L_s) ceil(n/L_l) with profile tiles
- `layout[lse_tile]`: LayoutBytes = s L_s L_l ceil(m/L_s) ceil(n/L_l) with profile tiles
- `HBM_mat[Q~]`: the expanded path forms no Q~ and no Z: it consumes expanded K^n and V
- `M_mat[Q~]`: the expanded path forms no Q~ and no Z: it consumes expanded K^n and V
- `HBM_mat[Z]`: the expanded path forms no Q~ and no Z: it consumes expanded K^n and V
- `M_mat[Z]`: the expanded path forms no Q~ and no Z: it consumes expanded K^n and V
- `local[q_tilde]`: the expanded path forms no Q~ and no Z: it consumes expanded K^n and V
- `local[z_tile]`: the expanded path forms no Q~ and no Z: it consumes expanded K^n and V
- `F_output_projection[executed_graph]`: the expanded path forms no Q~ and no Z: it consumes expanded K^n and V

<details><summary>Provenance (bindings, coverage, source, evidence)</summary>

| metric | source | coverage | bindings | evidence |
|---|---|---|---|---|
| local[q_latent_window] | derived | incomplete: depends on undeclared/unbound b_rq | `{"R_q": 16, "b_q": 16, "s_Cq": 2}` |  |
| local[wq_rank_tile] | derived | incomplete: depends on undeclared/unbound b_rq | `{"D_n": 24, "R_q": 16, "s_Wq": 2}` |  |
| local[q_proj_acc] | derived | incomplete: depends on undeclared/unbound projection_accumulator_dtype | `{"D_n": 24, "b_q": 16, "s_Qacc": 4}` |  |
| local[q_nope_operand] | derived | incomplete: depends on undeclared/unbound projected_operand_dtype, q_resident | `{"D_n": 24, "b_q": 16, "s_Qn": 2}` |  |
| local[kv_latent_rank_tile] | derived | incomplete: depends on undeclared/unbound b_rk | `{"R_k": 32, "b_k": 16, "n_buf": 2, "s_Ck": 2}` |  |
| local[wk_rank_tile] | derived | incomplete: depends on undeclared/unbound b_rk | `{"D_n": 24, "R_k": 32, "s_Wk": 2}` |  |
| local[wv_rank_tile] | derived | incomplete: depends on undeclared/unbound b_rk | `{"D_v": 24, "R_k": 32, "s_Wv": 2}` |  |
| local[k_nope_tile] | derived | incomplete: depends on undeclared/unbound projected_operand_dtype | `{"D_n": 24, "b_k": 16, "s_Kn": 2}` |  |
| local[v_tile] | derived | incomplete: depends on undeclared/unbound projected_operand_dtype, v_load | `{"D_v": 24, "b_k": 16, "s_V": 2}` |  |
| local[score_X] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"b_k": 16, "b_q": 16, "s_X": 4}` |  |
| local[exp_E] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"b_k": 16, "b_q": 16, "s_E": 4}` |  |
| local[p_operand] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"b_k": 16, "b_q": 16, "s_P": 2}` |  |
| local[row_state_m] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"b_q": 16, "s_state": 4}` |  |
| local[row_state_l] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"b_q": 16, "s_state": 4}` |  |
| local[row_alpha] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"b_q": 16, "s_state": 4}` |  |
| local[accumulator_A] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"D_v": 24, "b_q": 16, "s_A": 4}` |  |
| local[q_pe_tile] | derived | incomplete: depends on undeclared/unbound q_resident | `{"D_r": 8, "b_q": 16, "s_Qr": 2}` |  |
| local[k_pe_tile] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"D_r": 8, "b_k": 16, "n_buf": 2, "s_Kr": 2}` |  |
| local[fixed_scratch] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"s_scratch": 0}` |  |
| local[out_tile] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"D_v": 24, "b_q": 16, "s_O": 2}` |  |
| local[lse_tile] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"b_q": 16, "s_LSE": 4}` |  |
| layout[q_latent_window] | conditional | incomplete: depends on undeclared/unbound L_l, L_s, layout.q_latent_window | `{"R_q": 16, "b_q": 16, "s_Cq": 2}` |  |
| layout[wq_rank_tile] | conditional | incomplete: depends on undeclared/unbound L_l, L_s, layout.wq_rank_tile | `{"D_n": 24, "R_q": 16, "s_Wq": 2}` |  |
| layout[q_proj_acc] | conditional | incomplete: depends on undeclared/unbound layout.q_proj_acc | `{"D_n": 24, "b_q": 16, "s_Qacc": 4}` |  |
| layout[q_nope_operand] | conditional | incomplete: depends on undeclared/unbound L_l, L_s, layout.q_nope_operand | `{"D_n": 24, "b_q": 16, "s_Qn": 2}` |  |
| layout[kv_latent_rank_tile] | conditional | incomplete: depends on undeclared/unbound L_l, L_s, layout.kv_latent_rank_tile | `{"R_k": 32, "b_k": 16, "n_buf": 2, "s_Ck": 2}` |  |
| layout[wk_rank_tile] | conditional | incomplete: depends on undeclared/unbound L_l, L_s, layout.wk_rank_tile | `{"D_n": 24, "R_k": 32, "s_Wk": 2}` |  |
| layout[wv_rank_tile] | conditional | incomplete: depends on undeclared/unbound L_l, L_s, layout.wv_rank_tile | `{"D_v": 24, "R_k": 32, "s_Wv": 2}` |  |
| layout[k_nope_tile] | conditional | incomplete: depends on undeclared/unbound L_l, L_s, layout.k_nope_tile | `{"D_n": 24, "b_k": 16, "s_Kn": 2}` |  |
| layout[v_tile] | conditional | incomplete: depends on undeclared/unbound L_l, L_s, layout.v_tile | `{"D_v": 24, "b_k": 16, "s_V": 2}` |  |
| layout[score_X] | conditional | incomplete: depends on undeclared/unbound layout.score_X | `{"b_k": 16, "b_q": 16, "s_X": 4}` |  |
| layout[exp_E] | conditional | incomplete: depends on undeclared/unbound layout.exp_E | `{"b_k": 16, "b_q": 16, "s_E": 4}` |  |
| layout[p_operand] | conditional | incomplete: depends on undeclared/unbound L_l, L_s, layout.p_operand | `{"b_k": 16, "b_q": 16, "s_P": 2}` |  |
| layout[row_state_m] | conditional | incomplete: depends on undeclared/unbound layout.row_state_m | `{"b_q": 16, "s_state": 4}` |  |
| layout[row_state_l] | conditional | incomplete: depends on undeclared/unbound layout.row_state_l | `{"b_q": 16, "s_state": 4}` |  |
| layout[row_alpha] | conditional | incomplete: depends on undeclared/unbound layout.row_alpha | `{"b_q": 16, "s_state": 4}` |  |
| layout[accumulator_A] | conditional | incomplete: depends on undeclared/unbound layout.accumulator_A | `{"D_v": 24, "b_q": 16, "s_A": 4}` |  |
| layout[q_pe_tile] | conditional | incomplete: depends on undeclared/unbound L_l, L_s, layout.q_pe_tile | `{"D_r": 8, "b_q": 16, "s_Qr": 2}` |  |
| layout[k_pe_tile] | conditional | incomplete: depends on undeclared/unbound L_l, L_s, layout.k_pe_tile | `{"D_r": 8, "b_k": 16, "n_buf": 2, "s_Kr": 2}` |  |
| layout[fixed_scratch] | conditional | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"s_scratch": 0}` |  |
| layout[out_tile] | conditional | incomplete: depends on undeclared/unbound L_l, L_s, layout.out_tile | `{"D_v": 24, "b_q": 16, "s_O": 2}` |  |
| layout[lse_tile] | conditional | incomplete: depends on undeclared/unbound layout.lse_tile | `{"b_q": 16, "s_LSE": 4}` |  |
| HBM_mat[Q~] | definition | not applicable under the declared configuration | `{}` |  |
| M_mat[Q~] | definition | not applicable under the declared configuration | `{}` |  |
| HBM_mat[Z] | definition | not applicable under the declared configuration | `{}` |  |
| M_mat[Z] | definition | not applicable under the declared configuration | `{}` |  |
| local[q_tilde] | definition | not applicable under the declared configuration | `{}` |  |
| local[z_tile] | definition | not applicable under the declared configuration | `{}` |  |
| F_output_projection[executed_graph] | definition | not applicable under the declared configuration | `{}` |  |

</details>

## 7. stage pressure and envelope

| metric | formula id | expression | value | unit | status | missing | scope | not equivalent to |
|---|---|---|---|---|---|---|---|---|
| M_live[all:q_proj] | `pressure.live.all.q_proj` | `D_n*R_q*s_Wq + D_n*b_q*s_Qacc + R_q*b_q*s_Cq + s_scratch` | 2816 | byte | partial | b_rq | source-level live set at this stage cut, level=all (aliases deduplicated) | compiler register allocation; VMEM allocation |
| M_live[all:kv_proj] | `pressure.live.all.kv_proj` | `D_n*R_k*s_Wk + D_n*b_k*s_Kn + D_n*b_q*s_Qn + D_r*b_q*s_Qr + D_v*R_k*s_Wv + D_v*b_k*s_V + D_v*b_q*s_A + R_k*b_k*n_buf*s_Ck + 2*b_q*s_state + s_scratch` | 9344 | byte | partial | b_rk, q_resident, v_load | source-level live set at this stage cut, level=all (aliases deduplicated) | compiler register allocation; VMEM allocation |
| M_live[all:score] | `pressure.live.all.score` | `D_n*b_k*s_Kn + D_n*b_q*s_Qn + D_r*b_k*n_buf*s_Kr + D_r*b_q*s_Qr + D_v*b_k*s_V + D_v*b_q*s_A + 2*b_q*s_state + s_scratch + Max(b_k*b_q*s_E, b_k*b_q*s_X)` | 5760 | byte | partial | v_load | source-level live set at this stage cut, level=all (aliases deduplicated) | compiler register allocation; VMEM allocation |
| M_live[all:softmax] | `pressure.live.all.softmax` | `D_n*b_q*s_Qn + D_r*b_q*s_Qr + D_v*b_k*s_V + D_v*b_q*s_A + 3*b_q*s_state + s_scratch + Max(b_k*b_q*s_E, b_k*b_q*s_X)` | 4544 | byte | partial | q_resident, v_load | source-level live set at this stage cut, level=all (aliases deduplicated) | compiler register allocation; VMEM allocation |
| M_live[all:pv] | `pressure.live.all.pv` | `D_n*b_q*s_Qn + D_r*b_q*s_Qr + D_v*b_k*s_V + D_v*b_q*s_A + b_k*b_q*s_P + 3*b_q*s_state + s_scratch + Max(b_k*b_q*s_E, b_k*b_q*s_X)` | 5056 | byte | partial | q_resident | source-level live set at this stage cut, level=all (aliases deduplicated) | compiler register allocation; VMEM allocation |
| M_live[all:finalize] | `pressure.live.all.finalize` | `D_n*b_q*s_Qn + D_r*b_q*s_Qr + D_v*b_q*s_A + D_v*b_q*s_O + b_q*s_LSE + 2*b_q*s_state + s_scratch` | 3520 | byte | partial | q_resident | source-level live set at this stage cut, level=all (aliases deduplicated) | compiler register allocation; VMEM allocation |
| M_peak[all] | `pressure.peak.all` | `Max(D_n*R_q*s_Wq + D_n*b_q*s_Qacc + R_q*b_q*s_Cq + s_scratch, D_n*b_q*s_Qn + D_r*b_q*s_Qr + D_v*b_q*s_A + D_v*b_q*s_O + b_q*s_LSE + 2*b_q*s_state + s_scratch, D_n*b_q*s_Qn + D_r*b_q*s_Qr + D_v*b_k*s_V + D_v*b_q*s_A + b_k*b_q*s_P + 3*b_q*s_state + s_scratch + Max(b_k*b_q*s_E, b_k*b_q*s_X), D_n*b_k*s_Kn + D_n*b_q*s_Qn + D_r*b_k*n_buf*s_Kr + D_r*b_q*s_Qr + D_v*b_k*s_V + D_v*b_q*s_A + 2*b_q*s_state + s_scratch + Max(b_k*b_q*s_E, b_k*b_q*s_X), D_n*R_k*s_Wk + D_n*b_k*s_Kn + D_n*b_q*s_Qn + D_r*b_q*s_Qr + D_v*R_k*s_Wv + D_v*b_k*s_V + D_v*b_q*s_A + R_k*b_k*n_buf*s_Ck + 2*b_q*s_state + s_scratch)` | 9344 | byte | partial | b_rk, b_rq, q_resident, v_load | max over stage cuts, level=all (source level) | actual spill; OOM proof |
| M_live[vreg:q_proj] | `pressure.live.vreg.q_proj` | `D_n*b_q*s_Qacc` | 1536 | byte | bound |  | source-level live set at this stage cut, level=vreg (aliases deduplicated) | compiler register allocation; VMEM allocation |
| M_live[vreg:kv_proj] | `pressure.live.vreg.kv_proj` | `D_v*b_q*s_A + 2*b_q*s_state` | 1664 | byte | bound |  | source-level live set at this stage cut, level=vreg (aliases deduplicated) | compiler register allocation; VMEM allocation |
| M_live[vreg:score] | `pressure.live.vreg.score` | `D_v*b_q*s_A + 2*b_q*s_state + Max(b_k*b_q*s_E, b_k*b_q*s_X)` | 2688 | byte | bound |  | source-level live set at this stage cut, level=vreg (aliases deduplicated) | compiler register allocation; VMEM allocation |
| M_live[vreg:softmax] | `pressure.live.vreg.softmax` | `D_v*b_q*s_A + 3*b_q*s_state + Max(b_k*b_q*s_E, b_k*b_q*s_X)` | 2752 | byte | bound |  | source-level live set at this stage cut, level=vreg (aliases deduplicated) | compiler register allocation; VMEM allocation |
| M_live[vreg:pv] | `pressure.live.vreg.pv` | `D_v*b_q*s_A + b_k*b_q*s_P + 3*b_q*s_state + Max(b_k*b_q*s_E, b_k*b_q*s_X)` | 3264 | byte | bound |  | source-level live set at this stage cut, level=vreg (aliases deduplicated) | compiler register allocation; VMEM allocation |
| M_live[vreg:finalize] | `pressure.live.vreg.finalize` | `D_v*b_q*s_A + 2*b_q*s_state` | 1664 | byte | bound |  | source-level live set at this stage cut, level=vreg (aliases deduplicated) | compiler register allocation; VMEM allocation |
| M_peak[vreg] | `pressure.peak.vreg` | `Max(D_n*b_q*s_Qacc, D_v*b_q*s_A + b_k*b_q*s_P + 3*b_q*s_state + Max(b_k*b_q*s_E, b_k*b_q*s_X))` | 3264 | byte | bound |  | max over stage cuts, level=vreg (source level) | actual spill; OOM proof |
| M_live[vmem:q_proj] | `pressure.live.vmem.q_proj` | `D_n*R_q*s_Wq + R_q*b_q*s_Cq + s_scratch` | 1280 | byte | partial | b_rq | source-level live set at this stage cut, level=vmem (aliases deduplicated) | compiler register allocation; VMEM allocation |
| M_live[vmem:kv_proj] | `pressure.live.vmem.kv_proj` | `D_n*R_k*s_Wk + D_n*b_k*s_Kn + D_n*b_q*s_Qn + D_r*b_q*s_Qr + D_v*R_k*s_Wv + D_v*b_k*s_V + R_k*b_k*n_buf*s_Ck + s_scratch` | 7680 | byte | partial | b_rk, q_resident, v_load | source-level live set at this stage cut, level=vmem (aliases deduplicated) | compiler register allocation; VMEM allocation |
| M_live[vmem:score] | `pressure.live.vmem.score` | `D_n*b_k*s_Kn + D_n*b_q*s_Qn + D_r*b_k*n_buf*s_Kr + D_r*b_q*s_Qr + D_v*b_k*s_V + s_scratch` | 3072 | byte | partial | v_load | source-level live set at this stage cut, level=vmem (aliases deduplicated) | compiler register allocation; VMEM allocation |
| M_live[vmem:softmax] | `pressure.live.vmem.softmax` | `D_n*b_q*s_Qn + D_r*b_q*s_Qr + D_v*b_k*s_V + s_scratch` | 1792 | byte | partial | q_resident, v_load | source-level live set at this stage cut, level=vmem (aliases deduplicated) | compiler register allocation; VMEM allocation |
| M_live[vmem:pv] | `pressure.live.vmem.pv` | `D_n*b_q*s_Qn + D_r*b_q*s_Qr + D_v*b_k*s_V + s_scratch` | 1792 | byte | partial | q_resident | source-level live set at this stage cut, level=vmem (aliases deduplicated) | compiler register allocation; VMEM allocation |
| M_live[vmem:finalize] | `pressure.live.vmem.finalize` | `D_n*b_q*s_Qn + D_r*b_q*s_Qr + D_v*b_q*s_O + b_q*s_LSE + s_scratch` | 1856 | byte | partial | q_resident | source-level live set at this stage cut, level=vmem (aliases deduplicated) | compiler register allocation; VMEM allocation |
| M_peak[vmem] | `pressure.peak.vmem` | `Max(D_n*R_q*s_Wq + R_q*b_q*s_Cq + s_scratch, D_n*b_q*s_Qn + D_r*b_q*s_Qr + D_v*b_q*s_O + b_q*s_LSE + s_scratch, D_n*b_k*s_Kn + D_n*b_q*s_Qn + D_r*b_k*n_buf*s_Kr + D_r*b_q*s_Qr + D_v*b_k*s_V + s_scratch, D_n*R_k*s_Wk + D_n*b_k*s_Kn + D_n*b_q*s_Qn + D_r*b_q*s_Qr + D_v*R_k*s_Wv + D_v*b_k*s_V + R_k*b_k*n_buf*s_Ck + s_scratch)` | 7680 | byte | partial | b_rk, b_rq, q_resident, v_load | max over stage cuts, level=vmem (source level) | actual spill; OOM proof |
| envelope[all:q_proj] | `pressure.envelope.all.q_proj` | `D_n*R_q*s_Wq + b_q*(D_n*s_Qacc + R_q*s_Cq) + s_scratch` | 2816 | byte | partial | b_rq | a*b_q*b_k + b*b_q + c*b_k + d (+ remainder) for this stage, level=all | measured spill; spill probability |
| envelope[all:kv_proj] | `pressure.envelope.all.kv_proj` | `D_n*R_k*s_Wk + D_v*R_k*s_Wv + b_k*(D_n*s_Kn + D_v*s_V + R_k*n_buf*s_Ck) + b_q*(D_n*s_Qn + D_r*s_Qr + D_v*s_A + 2*s_state) + s_scratch` | 9344 | byte | partial | b_rk, q_resident, v_load | a*b_q*b_k + b*b_q + c*b_k + d (+ remainder) for this stage, level=all | measured spill; spill probability |
| envelope[all:score] | `pressure.envelope.all.score` | `b_k*b_q*Max(s_E, s_X) + b_k*(D_n*s_Kn + D_r*n_buf*s_Kr + D_v*s_V) + b_q*(D_n*s_Qn + D_r*s_Qr + D_v*s_A + 2*s_state) + s_scratch` | 5760 | byte | partial | v_load | a*b_q*b_k + b*b_q + c*b_k + d (+ remainder) for this stage, level=all | measured spill; spill probability |
| envelope[all:softmax] | `pressure.envelope.all.softmax` | `D_v*b_k*s_V + b_k*b_q*Max(s_E, s_X) + b_q*(D_n*s_Qn + D_r*s_Qr + D_v*s_A + 3*s_state) + s_scratch` | 4544 | byte | partial | q_resident, v_load | a*b_q*b_k + b*b_q + c*b_k + d (+ remainder) for this stage, level=all | measured spill; spill probability |
| envelope[all:pv] | `pressure.envelope.all.pv` | `D_v*b_k*s_V + b_k*b_q*(s_P + Max(s_E, s_X)) + b_q*(D_n*s_Qn + D_r*s_Qr + D_v*s_A + 3*s_state) + s_scratch` | 5056 | byte | partial | q_resident | a*b_q*b_k + b*b_q + c*b_k + d (+ remainder) for this stage, level=all | measured spill; spill probability |
| envelope[all:finalize] | `pressure.envelope.all.finalize` | `b_q*(D_n*s_Qn + D_r*s_Qr + D_v*s_A + D_v*s_O + s_LSE + 2*s_state) + s_scratch` | 3520 | byte | partial | q_resident | a*b_q*b_k + b*b_q + c*b_k + d (+ remainder) for this stage, level=all | measured spill; spill probability |
| envelope[vreg:q_proj] | `pressure.envelope.vreg.q_proj` | `D_n*b_q*s_Qacc` | 1536 | byte | bound |  | a*b_q*b_k + b*b_q + c*b_k + d (+ remainder) for this stage, level=vreg | measured spill; spill probability |
| envelope[vreg:kv_proj] | `pressure.envelope.vreg.kv_proj` | `b_q*(D_v*s_A + 2*s_state)` | 1664 | byte | bound |  | a*b_q*b_k + b*b_q + c*b_k + d (+ remainder) for this stage, level=vreg | measured spill; spill probability |
| envelope[vreg:score] | `pressure.envelope.vreg.score` | `b_k*b_q*Max(s_E, s_X) + b_q*(D_v*s_A + 2*s_state)` | 2688 | byte | bound |  | a*b_q*b_k + b*b_q + c*b_k + d (+ remainder) for this stage, level=vreg | measured spill; spill probability |
| envelope[vreg:softmax] | `pressure.envelope.vreg.softmax` | `b_k*b_q*Max(s_E, s_X) + b_q*(D_v*s_A + 3*s_state)` | 2752 | byte | bound |  | a*b_q*b_k + b*b_q + c*b_k + d (+ remainder) for this stage, level=vreg | measured spill; spill probability |
| envelope[vreg:pv] | `pressure.envelope.vreg.pv` | `b_k*b_q*(s_P + Max(s_E, s_X)) + b_q*(D_v*s_A + 3*s_state)` | 3264 | byte | bound |  | a*b_q*b_k + b*b_q + c*b_k + d (+ remainder) for this stage, level=vreg | measured spill; spill probability |
| envelope[vreg:finalize] | `pressure.envelope.vreg.finalize` | `b_q*(D_v*s_A + 2*s_state)` | 1664 | byte | bound |  | a*b_q*b_k + b*b_q + c*b_k + d (+ remainder) for this stage, level=vreg | measured spill; spill probability |
| envelope[vmem:q_proj] | `pressure.envelope.vmem.q_proj` | `D_n*R_q*s_Wq + R_q*b_q*s_Cq + s_scratch` | 1280 | byte | partial | b_rq | a*b_q*b_k + b*b_q + c*b_k + d (+ remainder) for this stage, level=vmem | measured spill; spill probability |
| envelope[vmem:kv_proj] | `pressure.envelope.vmem.kv_proj` | `D_n*R_k*s_Wk + D_v*R_k*s_Wv + b_k*(D_n*s_Kn + D_v*s_V + R_k*n_buf*s_Ck) + b_q*(D_n*s_Qn + D_r*s_Qr) + s_scratch` | 7680 | byte | partial | b_rk, q_resident, v_load | a*b_q*b_k + b*b_q + c*b_k + d (+ remainder) for this stage, level=vmem | measured spill; spill probability |
| envelope[vmem:score] | `pressure.envelope.vmem.score` | `b_k*(D_n*s_Kn + D_r*n_buf*s_Kr + D_v*s_V) + b_q*(D_n*s_Qn + D_r*s_Qr) + s_scratch` | 3072 | byte | partial | v_load | a*b_q*b_k + b*b_q + c*b_k + d (+ remainder) for this stage, level=vmem | measured spill; spill probability |
| envelope[vmem:softmax] | `pressure.envelope.vmem.softmax` | `D_v*b_k*s_V + b_q*(D_n*s_Qn + D_r*s_Qr) + s_scratch` | 1792 | byte | partial | q_resident, v_load | a*b_q*b_k + b*b_q + c*b_k + d (+ remainder) for this stage, level=vmem | measured spill; spill probability |
| envelope[vmem:pv] | `pressure.envelope.vmem.pv` | `D_v*b_k*s_V + b_q*(D_n*s_Qn + D_r*s_Qr) + s_scratch` | 1792 | byte | partial | q_resident | a*b_q*b_k + b*b_q + c*b_k + d (+ remainder) for this stage, level=vmem | measured spill; spill probability |
| envelope[vmem:finalize] | `pressure.envelope.vmem.finalize` | `b_q*(D_n*s_Qn + D_r*s_Qr + D_v*s_O + s_LSE) + s_scratch` | 1856 | byte | partial | q_resident | a*b_q*b_k + b*b_q + c*b_k + d (+ remainder) for this stage, level=vmem | measured spill; spill probability |
| V_lifetime_stages | `lifetime.v_tile.stages` | `\|{s : v_tile live at s}\|` | 4 | stages | partial | v_load | number of stage cuts the V tile is live at, under the declared V-load policy | measured V residency; compiler scheduling |
| V_prefetch_overlapped | `lifetime.v_tile.prefetch` | `v_tile live at the score stage` | True | bool | partial | v_load | whether the V read/production is overlapped with the score matmul under the declared policy | measured prefetch behaviour; a compiler schedule |
| b_k_max[vreg:q_proj] | `pressure.feasible.vreg.q_proj.b_k` | `floor((R - b b_q - d)/(a b_q + c)), stated only for a b_q + c > 0` |  | rows | not_applicable |  | the q_proj stage at level vreg has no b_k term (a b_q + c = 0), so it places no bound on b_k | an unbounded b_k: other stages still constrain it |
| excess_over_budget[vreg:q_proj] | `pressure.capacity_constraint.vreg.q_proj` | `Max(0, D_n*b_q*s_Qacc - R_vreg) → Max(0, 1536 - R_vreg)` |  | byte | partial | vreg_budget_bytes | max(0, M_live - R) at the q_proj stage, level vreg: data that cannot stay at the level under the source-level model | spill write traffic; compiled live set |
| b_k_max[vreg:kv_proj] | `pressure.feasible.vreg.kv_proj.b_k` | `floor((R - b b_q - d)/(a b_q + c)), stated only for a b_q + c > 0` |  | rows | not_applicable |  | the kv_proj stage at level vreg has no b_k term (a b_q + c = 0), so it places no bound on b_k | an unbounded b_k: other stages still constrain it |
| excess_over_budget[vreg:kv_proj] | `pressure.capacity_constraint.vreg.kv_proj` | `Max(0, -R_vreg + b_q*(D_v*s_A + 2*s_state)) → Max(0, 1664 - R_vreg)` |  | byte | partial | vreg_budget_bytes | max(0, M_live - R) at the kv_proj stage, level vreg: data that cannot stay at the level under the source-level model | spill write traffic; compiled live set |
| b_k_max[vreg:score] | `pressure.feasible.vreg.score.b_k` | `floor((R_vreg - b_q*(D_v*s_A + 2*s_state))/(b_q*Max(s_E, s_X))) → floor(R_vreg/64) - 26` |  | rows | partial | vreg_budget_bytes | floor((R - b b_q - d)/(a b_q + c)) for the score stage at level vreg under the declared budget R_vreg |  |
| excess_over_budget[vreg:score] | `pressure.capacity_constraint.vreg.score` | `Max(0, -R_vreg + b_k*b_q*Max(s_E, s_X) + b_q*(D_v*s_A + 2*s_state)) → Max(0, 2688 - R_vreg)` |  | byte | partial | vreg_budget_bytes | max(0, M_live - R) at the score stage, level vreg: data that cannot stay at the level under the source-level model | spill write traffic; compiled live set |
| b_k_max[vreg:softmax] | `pressure.feasible.vreg.softmax.b_k` | `floor((R_vreg - b_q*(D_v*s_A + 3*s_state))/(b_q*Max(s_E, s_X))) → floor(R_vreg/64) - 27` |  | rows | partial | vreg_budget_bytes | floor((R - b b_q - d)/(a b_q + c)) for the softmax stage at level vreg under the declared budget R_vreg |  |
| excess_over_budget[vreg:softmax] | `pressure.capacity_constraint.vreg.softmax` | `Max(0, -R_vreg + b_k*b_q*Max(s_E, s_X) + b_q*(D_v*s_A + 3*s_state)) → Max(0, 2752 - R_vreg)` |  | byte | partial | vreg_budget_bytes | max(0, M_live - R) at the softmax stage, level vreg: data that cannot stay at the level under the source-level model | spill write traffic; compiled live set |
| b_k_max[vreg:pv] | `pressure.feasible.vreg.pv.b_k` | `floor((R_vreg - b_q*(D_v*s_A + 3*s_state))/(b_q*(s_P + Max(s_E, s_X)))) → floor(R_vreg/96) - 18` |  | rows | partial | vreg_budget_bytes | floor((R - b b_q - d)/(a b_q + c)) for the pv stage at level vreg under the declared budget R_vreg |  |
| excess_over_budget[vreg:pv] | `pressure.capacity_constraint.vreg.pv` | `Max(0, -R_vreg + b_k*b_q*(s_P + Max(s_E, s_X)) + b_q*(D_v*s_A + 3*s_state)) → Max(0, 3264 - R_vreg)` |  | byte | partial | vreg_budget_bytes | max(0, M_live - R) at the pv stage, level vreg: data that cannot stay at the level under the source-level model | spill write traffic; compiled live set |
| b_k_max[vreg:finalize] | `pressure.feasible.vreg.finalize.b_k` | `floor((R - b b_q - d)/(a b_q + c)), stated only for a b_q + c > 0` |  | rows | not_applicable |  | the finalize stage at level vreg has no b_k term (a b_q + c = 0), so it places no bound on b_k | an unbounded b_k: other stages still constrain it |
| excess_over_budget[vreg:finalize] | `pressure.capacity_constraint.vreg.finalize` | `Max(0, -R_vreg + b_q*(D_v*s_A + 2*s_state)) → Max(0, 1664 - R_vreg)` |  | byte | partial | vreg_budget_bytes | max(0, M_live - R) at the finalize stage, level vreg: data that cannot stay at the level under the source-level model | spill write traffic; compiled live set |
| b_k_max[vreg:binding_stage] | `pressure.feasible.vreg.binding` | `min over stages s of floor((R - b_s b_q - d_s)/(a_s b_q + c_s))` |  | rows | unknown | vreg_budget_bytes | tightest per-stage capacity bound at level vreg | compiled spill/OOM criterion |
| b_k_max[vmem:q_proj] | `pressure.feasible.vmem.q_proj.b_k` | `floor((R - b b_q - d)/(a b_q + c)), stated only for a b_q + c > 0` |  | rows | not_applicable | b_rq | the q_proj stage at level vmem has no b_k term (a b_q + c = 0), so it places no bound on b_k | an unbounded b_k: other stages still constrain it |
| excess_over_budget[vmem:q_proj] | `pressure.capacity_constraint.vmem.q_proj` | `Max(0, D_n*R_q*s_Wq + R_q*b_q*s_Cq - R_vmem + s_scratch) → Max(0, 1280 - R_vmem)` |  | byte | partial | b_rq, vmem_budget_bytes | max(0, M_live - R) at the q_proj stage, level vmem: data that cannot stay at the level under the source-level model | spill write traffic; compiled live set |
| b_k_max[vmem:kv_proj] | `pressure.feasible.vmem.kv_proj.b_k` | `floor((-D_n*R_k*s_Wk - D_v*R_k*s_Wv + R_vmem - b_q*(D_n*s_Qn + D_r*s_Qr) - s_scratch)/(D_n*s_Kn + D_v*s_V + R_k*n_buf*s_Ck)) → floor(R_vmem/224 - 128/7)` |  | rows | partial | b_rk, q_resident, v_load, vmem_budget_bytes | floor((R - b b_q - d)/(a b_q + c)) for the kv_proj stage at level vmem under the declared budget R_vmem |  |
| excess_over_budget[vmem:kv_proj] | `pressure.capacity_constraint.vmem.kv_proj` | `Max(0, D_n*R_k*s_Wk + D_v*R_k*s_Wv - R_vmem + b_k*(D_n*s_Kn + D_v*s_V + R_k*n_buf*s_Ck) + b_q*(D_n*s_Qn + D_r*s_Qr) + s_scratch) → Max(0, 7680 - R_vmem)` |  | byte | partial | b_rk, q_resident, v_load, vmem_budget_bytes | max(0, M_live - R) at the kv_proj stage, level vmem: data that cannot stay at the level under the source-level model | spill write traffic; compiled live set |
| b_k_max[vmem:score] | `pressure.feasible.vmem.score.b_k` | `floor((R_vmem - b_q*(D_n*s_Qn + D_r*s_Qr) - s_scratch)/(D_n*s_Kn + D_r*n_buf*s_Kr + D_v*s_V)) → floor(R_vmem/128) - 8` |  | rows | partial | v_load, vmem_budget_bytes | floor((R - b b_q - d)/(a b_q + c)) for the score stage at level vmem under the declared budget R_vmem |  |
| excess_over_budget[vmem:score] | `pressure.capacity_constraint.vmem.score` | `Max(0, -R_vmem + b_k*(D_n*s_Kn + D_r*n_buf*s_Kr + D_v*s_V) + b_q*(D_n*s_Qn + D_r*s_Qr) + s_scratch) → Max(0, 3072 - R_vmem)` |  | byte | partial | v_load, vmem_budget_bytes | max(0, M_live - R) at the score stage, level vmem: data that cannot stay at the level under the source-level model | spill write traffic; compiled live set |
| b_k_max[vmem:softmax] | `pressure.feasible.vmem.softmax.b_k` | `floor((R_vmem - b_q*(D_n*s_Qn + D_r*s_Qr) - s_scratch)/(D_v*s_V)) → floor(R_vmem/48 - 64/3)` |  | rows | partial | q_resident, v_load, vmem_budget_bytes | floor((R - b b_q - d)/(a b_q + c)) for the softmax stage at level vmem under the declared budget R_vmem |  |
| excess_over_budget[vmem:softmax] | `pressure.capacity_constraint.vmem.softmax` | `Max(0, D_v*b_k*s_V - R_vmem + b_q*(D_n*s_Qn + D_r*s_Qr) + s_scratch) → Max(0, 1792 - R_vmem)` |  | byte | partial | q_resident, v_load, vmem_budget_bytes | max(0, M_live - R) at the softmax stage, level vmem: data that cannot stay at the level under the source-level model | spill write traffic; compiled live set |
| b_k_max[vmem:pv] | `pressure.feasible.vmem.pv.b_k` | `floor((R_vmem - b_q*(D_n*s_Qn + D_r*s_Qr) - s_scratch)/(D_v*s_V)) → floor(R_vmem/48 - 64/3)` |  | rows | partial | q_resident, vmem_budget_bytes | floor((R - b b_q - d)/(a b_q + c)) for the pv stage at level vmem under the declared budget R_vmem |  |
| excess_over_budget[vmem:pv] | `pressure.capacity_constraint.vmem.pv` | `Max(0, D_v*b_k*s_V - R_vmem + b_q*(D_n*s_Qn + D_r*s_Qr) + s_scratch) → Max(0, 1792 - R_vmem)` |  | byte | partial | q_resident, vmem_budget_bytes | max(0, M_live - R) at the pv stage, level vmem: data that cannot stay at the level under the source-level model | spill write traffic; compiled live set |
| b_k_max[vmem:finalize] | `pressure.feasible.vmem.finalize.b_k` | `floor((R - b b_q - d)/(a b_q + c)), stated only for a b_q + c > 0` |  | rows | not_applicable | q_resident | the finalize stage at level vmem has no b_k term (a b_q + c = 0), so it places no bound on b_k | an unbounded b_k: other stages still constrain it |
| excess_over_budget[vmem:finalize] | `pressure.capacity_constraint.vmem.finalize` | `Max(0, -R_vmem + b_q*(D_n*s_Qn + D_r*s_Qr + D_v*s_O + s_LSE) + s_scratch) → Max(0, 1856 - R_vmem)` |  | byte | partial | q_resident, vmem_budget_bytes | max(0, M_live - R) at the finalize stage, level vmem: data that cannot stay at the level under the source-level model | spill write traffic; compiled live set |
| b_k_max[vmem:binding_stage] | `pressure.feasible.vmem.binding` | `min over stages s of floor((R - b_s b_q - d_s)/(a_s b_q + c_s))` |  | rows | unknown | b_rk, b_rq, q_resident, v_load, vmem_budget_bytes | tightest per-stage capacity bound at level vmem | compiled spill/OOM criterion |
| N_spill_instructions | `spill.N_spill_instructions` | `\|{spill/fill instructions in the compiled program}\|` |  | instructions | unknown | compile_evidence | static count of spill and fill instructions (not their dynamic execution count, not bytes) | each other; pressure envelope |
| M_spill_peak | `spill.M_spill_peak` | `max_t sum_{a in A_spill(t)} M_a` |  | byte | unknown | compile_or_measurement_evidence | peak spill backing allocation | each other; pressure envelope |
| B_spill_fill | `spill.B_spill_fill` | `sum_r bytes(r) * executions(r)` |  | byte | unknown | compile_or_measurement_evidence | dynamic spill/fill traffic | each other; pressure envelope |
| dT_spill | `spill.dT_spill` | `T_schedule(G_with_transfers) - T_schedule(G_comparison)` |  | s | unknown | compile_or_measurement_evidence, declared comparison graph | exposed latency vs a legal comparison graph | each other; pressure envelope |
| B_extra_opt | `spill.B_extra_opt` | `min_{p in Omega(theta, a, eta)} B_extra(p)` |  | byte | unknown | graph, machine_model, allowed_transformations, objective | minimum extra movement over legal schedules/tilings/recomputations (spec 8.3); solvable only for a declared small graph, machine model and objective | each other; pressure envelope |
| T_opt | `spill.T_opt` | `min_{p in Omega(theta, a, eta)} T(p)` |  | s | unknown | graph, machine_model, allowed_transformations, schedule_model | minimum time over the legal set (spec 8.3, speed objective); needs a solved schedule, not a bound | each other; pressure envelope |
| B_extra_opt_under_time_budget | `spill.B_extra_opt_under_time_budget` | `min_{p in Omega} B_extra(p) s.t. T(p) <= tau` |  | byte | unknown | graph, machine_model, allowed_transformations, schedule_model, time_budget_tau | minimum extra movement subject to a declared time budget tau (spec 8.3) | each other; pressure envelope |

Assumptions:
- `M_live[all:q_proj]`: objects: q_latent_window, wq_rank_tile, q_proj_acc, fixed_scratch
- `M_live[all:q_proj]`: includes objects whose existence depends on undeclared fields: ['b_rq']
- `M_live[all:kv_proj]`: objects: q_nope_operand, kv_latent_rank_tile, wk_rank_tile, wv_rank_tile, k_nope_tile, v_tile, row_state_m, row_state_l, accumulator_A, q_pe_tile, fixed_scratch
- `M_live[all:kv_proj]`: includes objects whose existence depends on undeclared fields: ['q_resident', 'b_rk', 'v_load']
- `M_live[all:score]`: objects: q_nope_operand, k_nope_tile, v_tile, row_state_m, row_state_l, accumulator_A, q_pe_tile, k_pe_tile, fixed_scratch, alias[score_or_E]:score_X (sized by the whole group: score_X|exp_E)
- `M_live[all:score]`: includes objects whose existence depends on undeclared fields: ['v_load']
- `M_live[all:softmax]`: objects: q_nope_operand, v_tile, row_state_m, row_state_l, row_alpha, accumulator_A, q_pe_tile, fixed_scratch, alias[score_or_E]:score_X|exp_E
- `M_live[all:softmax]`: includes objects whose existence depends on undeclared fields: ['q_resident', 'v_load']
- `M_live[all:pv]`: objects: q_nope_operand, v_tile, p_operand, row_state_m, row_state_l, row_alpha, accumulator_A, q_pe_tile, fixed_scratch, alias[score_or_E]:exp_E (sized by the whole group: score_X|exp_E)
- `M_live[all:pv]`: includes objects whose existence depends on undeclared fields: ['q_resident']
- `M_live[all:finalize]`: objects: q_nope_operand, row_state_m, row_state_l, accumulator_A, q_pe_tile, fixed_scratch, out_tile, lse_tile
- `M_live[all:finalize]`: includes objects whose existence depends on undeclared fields: ['q_resident']
- `M_live[vreg:q_proj]`: objects: q_proj_acc
- `M_live[vreg:kv_proj]`: objects: row_state_m, row_state_l, accumulator_A
- `M_live[vreg:score]`: objects: row_state_m, row_state_l, accumulator_A, alias[score_or_E]:score_X (sized by the whole group: score_X|exp_E)
- `M_live[vreg:softmax]`: objects: row_state_m, row_state_l, row_alpha, accumulator_A, alias[score_or_E]:score_X|exp_E
- `M_live[vreg:pv]`: objects: p_operand, row_state_m, row_state_l, row_alpha, accumulator_A, alias[score_or_E]:exp_E (sized by the whole group: score_X|exp_E)
- `M_live[vreg:finalize]`: objects: row_state_m, row_state_l, accumulator_A
- `M_live[vmem:q_proj]`: objects: q_latent_window, wq_rank_tile, fixed_scratch
- `M_live[vmem:q_proj]`: includes objects whose existence depends on undeclared fields: ['b_rq']
- `M_live[vmem:kv_proj]`: objects: q_nope_operand, kv_latent_rank_tile, wk_rank_tile, wv_rank_tile, k_nope_tile, v_tile, q_pe_tile, fixed_scratch
- `M_live[vmem:kv_proj]`: includes objects whose existence depends on undeclared fields: ['q_resident', 'b_rk', 'v_load']
- `M_live[vmem:score]`: objects: q_nope_operand, k_nope_tile, v_tile, q_pe_tile, k_pe_tile, fixed_scratch
- `M_live[vmem:score]`: includes objects whose existence depends on undeclared fields: ['v_load']
- `M_live[vmem:softmax]`: objects: q_nope_operand, v_tile, q_pe_tile, fixed_scratch
- `M_live[vmem:softmax]`: includes objects whose existence depends on undeclared fields: ['q_resident', 'v_load']
- `M_live[vmem:pv]`: objects: q_nope_operand, v_tile, q_pe_tile, fixed_scratch
- `M_live[vmem:pv]`: includes objects whose existence depends on undeclared fields: ['q_resident']
- `M_live[vmem:finalize]`: objects: q_nope_operand, q_pe_tile, fixed_scratch, out_tile, lse_tile
- `M_live[vmem:finalize]`: includes objects whose existence depends on undeclared fields: ['q_resident']
- `envelope[all:q_proj]`: a from []
- `envelope[all:q_proj]`: b from ['q_latent_window', 'q_proj_acc']
- `envelope[all:q_proj]`: c from []
- `envelope[all:q_proj]`: d from ['wq_rank_tile', 'fixed_scratch']
- `envelope[all:kv_proj]`: a from []
- `envelope[all:kv_proj]`: b from ['q_nope_operand', 'row_state_m', 'row_state_l', 'accumulator_A', 'q_pe_tile']
- `envelope[all:kv_proj]`: c from ['kv_latent_rank_tile', 'k_nope_tile', 'v_tile']
- `envelope[all:kv_proj]`: d from ['wk_rank_tile', 'wv_rank_tile', 'fixed_scratch']
- `envelope[all:score]`: a from ['alias[score_or_E]:score_X|exp_E']
- `envelope[all:score]`: b from ['q_nope_operand', 'row_state_m', 'row_state_l', 'accumulator_A', 'q_pe_tile']
- `envelope[all:score]`: c from ['k_nope_tile', 'v_tile', 'k_pe_tile']
- `envelope[all:score]`: d from ['fixed_scratch']
- `envelope[all:softmax]`: a from ['alias[score_or_E]:score_X|exp_E']
- `envelope[all:softmax]`: b from ['q_nope_operand', 'row_state_m', 'row_state_l', 'row_alpha', 'accumulator_A', 'q_pe_tile']
- `envelope[all:softmax]`: c from ['v_tile']
- `envelope[all:softmax]`: d from ['fixed_scratch']
- `envelope[all:pv]`: a from ['p_operand', 'alias[score_or_E]:score_X|exp_E']
- `envelope[all:pv]`: b from ['q_nope_operand', 'row_state_m', 'row_state_l', 'row_alpha', 'accumulator_A', 'q_pe_tile']
- `envelope[all:pv]`: c from ['v_tile']
- `envelope[all:pv]`: d from ['fixed_scratch']
- `envelope[all:finalize]`: a from []
- `envelope[all:finalize]`: b from ['q_nope_operand', 'row_state_m', 'row_state_l', 'accumulator_A', 'q_pe_tile', 'out_tile', 'lse_tile']
- `envelope[all:finalize]`: c from []
- `envelope[all:finalize]`: d from ['fixed_scratch']
- `envelope[vreg:q_proj]`: a from []
- `envelope[vreg:q_proj]`: b from ['q_proj_acc']
- `envelope[vreg:q_proj]`: c from []
- `envelope[vreg:q_proj]`: d from []
- `envelope[vreg:kv_proj]`: a from []
- `envelope[vreg:kv_proj]`: b from ['row_state_m', 'row_state_l', 'accumulator_A']
- `envelope[vreg:kv_proj]`: c from []
- `envelope[vreg:kv_proj]`: d from []
- `envelope[vreg:score]`: a from ['alias[score_or_E]:score_X|exp_E']
- `envelope[vreg:score]`: b from ['row_state_m', 'row_state_l', 'accumulator_A']
- `envelope[vreg:score]`: c from []
- `envelope[vreg:score]`: d from []
- `envelope[vreg:softmax]`: a from ['alias[score_or_E]:score_X|exp_E']
- `envelope[vreg:softmax]`: b from ['row_state_m', 'row_state_l', 'row_alpha', 'accumulator_A']
- `envelope[vreg:softmax]`: c from []
- `envelope[vreg:softmax]`: d from []
- `envelope[vreg:pv]`: a from ['p_operand', 'alias[score_or_E]:score_X|exp_E']
- `envelope[vreg:pv]`: b from ['row_state_m', 'row_state_l', 'row_alpha', 'accumulator_A']
- `envelope[vreg:pv]`: c from []
- `envelope[vreg:pv]`: d from []
- `envelope[vreg:finalize]`: a from []
- `envelope[vreg:finalize]`: b from ['row_state_m', 'row_state_l', 'accumulator_A']
- `envelope[vreg:finalize]`: c from []
- `envelope[vreg:finalize]`: d from []
- `envelope[vmem:q_proj]`: a from []
- `envelope[vmem:q_proj]`: b from ['q_latent_window']
- `envelope[vmem:q_proj]`: c from []
- `envelope[vmem:q_proj]`: d from ['wq_rank_tile', 'fixed_scratch']
- `envelope[vmem:kv_proj]`: a from []
- `envelope[vmem:kv_proj]`: b from ['q_nope_operand', 'q_pe_tile']
- `envelope[vmem:kv_proj]`: c from ['kv_latent_rank_tile', 'k_nope_tile', 'v_tile']
- `envelope[vmem:kv_proj]`: d from ['wk_rank_tile', 'wv_rank_tile', 'fixed_scratch']
- `envelope[vmem:score]`: a from []
- `envelope[vmem:score]`: b from ['q_nope_operand', 'q_pe_tile']
- `envelope[vmem:score]`: c from ['k_nope_tile', 'v_tile', 'k_pe_tile']
- `envelope[vmem:score]`: d from ['fixed_scratch']
- `envelope[vmem:softmax]`: a from []
- `envelope[vmem:softmax]`: b from ['q_nope_operand', 'q_pe_tile']
- `envelope[vmem:softmax]`: c from ['v_tile']
- `envelope[vmem:softmax]`: d from ['fixed_scratch']
- `envelope[vmem:pv]`: a from []
- `envelope[vmem:pv]`: b from ['q_nope_operand', 'q_pe_tile']
- `envelope[vmem:pv]`: c from ['v_tile']
- `envelope[vmem:pv]`: d from ['fixed_scratch']
- `envelope[vmem:finalize]`: a from []
- `envelope[vmem:finalize]`: b from ['q_nope_operand', 'q_pe_tile', 'out_tile', 'lse_tile']
- `envelope[vmem:finalize]`: c from []
- `envelope[vmem:finalize]`: d from ['fixed_scratch']
- `V_lifetime_stages`: V live at ['kv_proj', 'score', 'softmax', 'pv']
- `V_lifetime_stages`: a candidate hypothesis under the declared lifetime model, not a located bug: no source or schedule evidence is attached
- `V_prefetch_overlapped`: early load: the V transfer overlaps the score matmul
- `V_prefetch_overlapped`: a candidate hypothesis under the declared lifetime model, not a located bug: no source or schedule evidence is attached
- `b_k_max[vreg:q_proj]`: this stage's live set does not grow with b_k; the binding bound comes from another stage
- `excess_over_budget[vreg:q_proj]`: source-level logical live set; layout coverage not applied; not a compiled graph cut (spec 8.2 caveat)
- `b_k_max[vreg:kv_proj]`: this stage's live set does not grow with b_k; the binding bound comes from another stage
- `excess_over_budget[vreg:kv_proj]`: source-level logical live set; layout coverage not applied; not a compiled graph cut (spec 8.2 caveat)
- `b_k_max[vreg:score]`: valid only when a*b_q + c > 0
- `b_k_max[vreg:score]`: capacity constraint under the declared source-level model, not a compiled spill/OOM criterion
- `excess_over_budget[vreg:score]`: source-level logical live set; layout coverage not applied; not a compiled graph cut (spec 8.2 caveat)
- `b_k_max[vreg:softmax]`: valid only when a*b_q + c > 0
- `b_k_max[vreg:softmax]`: capacity constraint under the declared source-level model, not a compiled spill/OOM criterion
- `excess_over_budget[vreg:softmax]`: source-level logical live set; layout coverage not applied; not a compiled graph cut (spec 8.2 caveat)
- `b_k_max[vreg:pv]`: valid only when a*b_q + c > 0
- `b_k_max[vreg:pv]`: capacity constraint under the declared source-level model, not a compiled spill/OOM criterion
- `excess_over_budget[vreg:pv]`: source-level logical live set; layout coverage not applied; not a compiled graph cut (spec 8.2 caveat)
- `b_k_max[vreg:finalize]`: this stage's live set does not grow with b_k; the binding bound comes from another stage
- `excess_over_budget[vreg:finalize]`: source-level logical live set; layout coverage not applied; not a compiled graph cut (spec 8.2 caveat)
- `b_k_max[vmem:q_proj]`: this stage's live set does not grow with b_k; the binding bound comes from another stage
- `b_k_max[vmem:q_proj]`: the absence of a b_k term rests on an object list shaped by ['b_rq']
- `excess_over_budget[vmem:q_proj]`: source-level logical live set; layout coverage not applied; not a compiled graph cut (spec 8.2 caveat)
- `b_k_max[vmem:kv_proj]`: valid only when a*b_q + c > 0
- `b_k_max[vmem:kv_proj]`: capacity constraint under the declared source-level model, not a compiled spill/OOM criterion
- `excess_over_budget[vmem:kv_proj]`: source-level logical live set; layout coverage not applied; not a compiled graph cut (spec 8.2 caveat)
- `b_k_max[vmem:score]`: valid only when a*b_q + c > 0
- `b_k_max[vmem:score]`: capacity constraint under the declared source-level model, not a compiled spill/OOM criterion
- `excess_over_budget[vmem:score]`: source-level logical live set; layout coverage not applied; not a compiled graph cut (spec 8.2 caveat)
- `b_k_max[vmem:softmax]`: valid only when a*b_q + c > 0
- `b_k_max[vmem:softmax]`: capacity constraint under the declared source-level model, not a compiled spill/OOM criterion
- `excess_over_budget[vmem:softmax]`: source-level logical live set; layout coverage not applied; not a compiled graph cut (spec 8.2 caveat)
- `b_k_max[vmem:pv]`: valid only when a*b_q + c > 0
- `b_k_max[vmem:pv]`: capacity constraint under the declared source-level model, not a compiled spill/OOM criterion
- `excess_over_budget[vmem:pv]`: source-level logical live set; layout coverage not applied; not a compiled graph cut (spec 8.2 caveat)
- `b_k_max[vmem:finalize]`: this stage's live set does not grow with b_k; the binding bound comes from another stage
- `b_k_max[vmem:finalize]`: the absence of a b_k term rests on an object list shaped by ['q_resident']
- `excess_over_budget[vmem:finalize]`: source-level logical live set; layout coverage not applied; not a compiled graph cut (spec 8.2 caveat)

<details><summary>Provenance (bindings, coverage, source, evidence)</summary>

| metric | source | coverage | bindings | evidence |
|---|---|---|---|---|
| M_live[all:q_proj] | derived | incomplete: depends on undeclared/unbound b_rq | `{"D_n": 24, "R_q": 16, "b_q": 16, "s_Cq": 2, "s_Qacc": 4, "s_Wq": 2, "s_scratch": 0}` |  |
| M_live[all:kv_proj] | derived | incomplete: depends on undeclared/unbound b_rk, q_resident, v_load | `{"D_n": 24, "D_r": 8, "D_v": 24, "R_k": 32, "b_k": 16, "b_q": 16, "n_buf": 2, "s_A": 4,...` |  |
| M_live[all:score] | derived | incomplete: depends on undeclared/unbound v_load | `{"D_n": 24, "D_r": 8, "D_v": 24, "b_k": 16, "b_q": 16, "n_buf": 2, "s_A": 4, "s_E": 4, ...` |  |
| M_live[all:softmax] | derived | incomplete: depends on undeclared/unbound q_resident, v_load | `{"D_n": 24, "D_r": 8, "D_v": 24, "b_k": 16, "b_q": 16, "s_A": 4, "s_E": 4, "s_Qn": 2, "...` |  |
| M_live[all:pv] | derived | incomplete: depends on undeclared/unbound q_resident | `{"D_n": 24, "D_r": 8, "D_v": 24, "b_k": 16, "b_q": 16, "s_A": 4, "s_E": 4, "s_P": 2, "s...` |  |
| M_live[all:finalize] | derived | incomplete: depends on undeclared/unbound q_resident | `{"D_n": 24, "D_r": 8, "D_v": 24, "b_q": 16, "s_A": 4, "s_LSE": 4, "s_O": 2, "s_Qn": 2, ...` |  |
| M_peak[all] | derived | incomplete: depends on undeclared/unbound b_rk, b_rq, q_resident, v_load | `{"D_n": 24, "D_r": 8, "D_v": 24, "R_k": 32, "R_q": 16, "b_k": 16, "b_q": 16, "n_buf": 2...` |  |
| M_live[vreg:q_proj] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"D_n": 24, "b_q": 16, "s_Qacc": 4}` |  |
| M_live[vreg:kv_proj] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"D_v": 24, "b_q": 16, "s_A": 4, "s_state": 4}` |  |
| M_live[vreg:score] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"D_v": 24, "b_k": 16, "b_q": 16, "s_A": 4, "s_E": 4, "s_X": 4, "s_state": 4}` |  |
| M_live[vreg:softmax] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"D_v": 24, "b_k": 16, "b_q": 16, "s_A": 4, "s_E": 4, "s_X": 4, "s_state": 4}` |  |
| M_live[vreg:pv] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"D_v": 24, "b_k": 16, "b_q": 16, "s_A": 4, "s_E": 4, "s_P": 2, "s_X": 4, "s_state": 4}` |  |
| M_live[vreg:finalize] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"D_v": 24, "b_q": 16, "s_A": 4, "s_state": 4}` |  |
| M_peak[vreg] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"D_n": 24, "D_v": 24, "b_k": 16, "b_q": 16, "s_A": 4, "s_E": 4, "s_P": 2, "s_Qacc": 4,...` |  |
| M_live[vmem:q_proj] | derived | incomplete: depends on undeclared/unbound b_rq | `{"D_n": 24, "R_q": 16, "b_q": 16, "s_Cq": 2, "s_Wq": 2, "s_scratch": 0}` |  |
| M_live[vmem:kv_proj] | derived | incomplete: depends on undeclared/unbound b_rk, q_resident, v_load | `{"D_n": 24, "D_r": 8, "D_v": 24, "R_k": 32, "b_k": 16, "b_q": 16, "n_buf": 2, "s_Ck": 2...` |  |
| M_live[vmem:score] | derived | incomplete: depends on undeclared/unbound v_load | `{"D_n": 24, "D_r": 8, "D_v": 24, "b_k": 16, "b_q": 16, "n_buf": 2, "s_Kn": 2, "s_Kr": 2...` |  |
| M_live[vmem:softmax] | derived | incomplete: depends on undeclared/unbound q_resident, v_load | `{"D_n": 24, "D_r": 8, "D_v": 24, "b_k": 16, "b_q": 16, "s_Qn": 2, "s_Qr": 2, "s_V": 2, ...` |  |
| M_live[vmem:pv] | derived | incomplete: depends on undeclared/unbound q_resident | `{"D_n": 24, "D_r": 8, "D_v": 24, "b_k": 16, "b_q": 16, "s_Qn": 2, "s_Qr": 2, "s_V": 2, ...` |  |
| M_live[vmem:finalize] | derived | incomplete: depends on undeclared/unbound q_resident | `{"D_n": 24, "D_r": 8, "D_v": 24, "b_q": 16, "s_LSE": 4, "s_O": 2, "s_Qn": 2, "s_Qr": 2,...` |  |
| M_peak[vmem] | derived | incomplete: depends on undeclared/unbound b_rk, b_rq, q_resident, v_load | `{"D_n": 24, "D_r": 8, "D_v": 24, "R_k": 32, "R_q": 16, "b_k": 16, "b_q": 16, "n_buf": 2...` |  |
| envelope[all:q_proj] | derived | incomplete: depends on undeclared/unbound b_rq | `{"D_n": 24, "R_q": 16, "b_q": 16, "s_Cq": 2, "s_Qacc": 4, "s_Wq": 2, "s_scratch": 0}` |  |
| envelope[all:kv_proj] | derived | incomplete: depends on undeclared/unbound b_rk, q_resident, v_load | `{"D_n": 24, "D_r": 8, "D_v": 24, "R_k": 32, "b_k": 16, "b_q": 16, "n_buf": 2, "s_A": 4,...` |  |
| envelope[all:score] | derived | incomplete: depends on undeclared/unbound v_load | `{"D_n": 24, "D_r": 8, "D_v": 24, "b_k": 16, "b_q": 16, "n_buf": 2, "s_A": 4, "s_E": 4, ...` |  |
| envelope[all:softmax] | derived | incomplete: depends on undeclared/unbound q_resident, v_load | `{"D_n": 24, "D_r": 8, "D_v": 24, "b_k": 16, "b_q": 16, "s_A": 4, "s_E": 4, "s_Qn": 2, "...` |  |
| envelope[all:pv] | derived | incomplete: depends on undeclared/unbound q_resident | `{"D_n": 24, "D_r": 8, "D_v": 24, "b_k": 16, "b_q": 16, "s_A": 4, "s_E": 4, "s_P": 2, "s...` |  |
| envelope[all:finalize] | derived | incomplete: depends on undeclared/unbound q_resident | `{"D_n": 24, "D_r": 8, "D_v": 24, "b_q": 16, "s_A": 4, "s_LSE": 4, "s_O": 2, "s_Qn": 2, ...` |  |
| envelope[vreg:q_proj] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"D_n": 24, "b_q": 16, "s_Qacc": 4}` |  |
| envelope[vreg:kv_proj] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"D_v": 24, "b_q": 16, "s_A": 4, "s_state": 4}` |  |
| envelope[vreg:score] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"D_v": 24, "b_k": 16, "b_q": 16, "s_A": 4, "s_E": 4, "s_X": 4, "s_state": 4}` |  |
| envelope[vreg:softmax] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"D_v": 24, "b_k": 16, "b_q": 16, "s_A": 4, "s_E": 4, "s_X": 4, "s_state": 4}` |  |
| envelope[vreg:pv] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"D_v": 24, "b_k": 16, "b_q": 16, "s_A": 4, "s_E": 4, "s_P": 2, "s_X": 4, "s_state": 4}` |  |
| envelope[vreg:finalize] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"D_v": 24, "b_q": 16, "s_A": 4, "s_state": 4}` |  |
| envelope[vmem:q_proj] | derived | incomplete: depends on undeclared/unbound b_rq | `{"D_n": 24, "R_q": 16, "b_q": 16, "s_Cq": 2, "s_Wq": 2, "s_scratch": 0}` |  |
| envelope[vmem:kv_proj] | derived | incomplete: depends on undeclared/unbound b_rk, q_resident, v_load | `{"D_n": 24, "D_r": 8, "D_v": 24, "R_k": 32, "b_k": 16, "b_q": 16, "n_buf": 2, "s_Ck": 2...` |  |
| envelope[vmem:score] | derived | incomplete: depends on undeclared/unbound v_load | `{"D_n": 24, "D_r": 8, "D_v": 24, "b_k": 16, "b_q": 16, "n_buf": 2, "s_Kn": 2, "s_Kr": 2...` |  |
| envelope[vmem:softmax] | derived | incomplete: depends on undeclared/unbound q_resident, v_load | `{"D_n": 24, "D_r": 8, "D_v": 24, "b_k": 16, "b_q": 16, "s_Qn": 2, "s_Qr": 2, "s_V": 2, ...` |  |
| envelope[vmem:pv] | derived | incomplete: depends on undeclared/unbound q_resident | `{"D_n": 24, "D_r": 8, "D_v": 24, "b_k": 16, "b_q": 16, "s_Qn": 2, "s_Qr": 2, "s_V": 2, ...` |  |
| envelope[vmem:finalize] | derived | incomplete: depends on undeclared/unbound q_resident | `{"D_n": 24, "D_r": 8, "D_v": 24, "b_q": 16, "s_LSE": 4, "s_O": 2, "s_Qn": 2, "s_Qr": 2,...` |  |
| V_lifetime_stages | derived | incomplete: depends on undeclared/unbound v_load | `{}` |  |
| V_prefetch_overlapped | derived | incomplete: depends on undeclared/unbound v_load | `{}` |  |
| b_k_max[vreg:q_proj] | derived | not applicable under the declared configuration | `{}` |  |
| excess_over_budget[vreg:q_proj] | derived | incomplete: depends on undeclared/unbound vreg_budget_bytes | `{"D_n": 24, "b_q": 16, "s_Qacc": 4}` |  |
| b_k_max[vreg:kv_proj] | derived | not applicable under the declared configuration | `{}` |  |
| excess_over_budget[vreg:kv_proj] | derived | incomplete: depends on undeclared/unbound vreg_budget_bytes | `{"D_v": 24, "b_q": 16, "s_A": 4, "s_state": 4}` |  |
| b_k_max[vreg:score] | derived | incomplete: depends on undeclared/unbound vreg_budget_bytes | `{"D_v": 24, "b_q": 16, "s_A": 4, "s_E": 4, "s_X": 4, "s_state": 4}` |  |
| excess_over_budget[vreg:score] | derived | incomplete: depends on undeclared/unbound vreg_budget_bytes | `{"D_v": 24, "b_k": 16, "b_q": 16, "s_A": 4, "s_E": 4, "s_X": 4, "s_state": 4}` |  |
| b_k_max[vreg:softmax] | derived | incomplete: depends on undeclared/unbound vreg_budget_bytes | `{"D_v": 24, "b_q": 16, "s_A": 4, "s_E": 4, "s_X": 4, "s_state": 4}` |  |
| excess_over_budget[vreg:softmax] | derived | incomplete: depends on undeclared/unbound vreg_budget_bytes | `{"D_v": 24, "b_k": 16, "b_q": 16, "s_A": 4, "s_E": 4, "s_X": 4, "s_state": 4}` |  |
| b_k_max[vreg:pv] | derived | incomplete: depends on undeclared/unbound vreg_budget_bytes | `{"D_v": 24, "b_q": 16, "s_A": 4, "s_E": 4, "s_P": 2, "s_X": 4, "s_state": 4}` |  |
| excess_over_budget[vreg:pv] | derived | incomplete: depends on undeclared/unbound vreg_budget_bytes | `{"D_v": 24, "b_k": 16, "b_q": 16, "s_A": 4, "s_E": 4, "s_P": 2, "s_X": 4, "s_state": 4}` |  |
| b_k_max[vreg:finalize] | derived | not applicable under the declared configuration | `{}` |  |
| excess_over_budget[vreg:finalize] | derived | incomplete: depends on undeclared/unbound vreg_budget_bytes | `{"D_v": 24, "b_q": 16, "s_A": 4, "s_state": 4}` |  |
| b_k_max[vreg:binding_stage] | derived | incomplete: depends on undeclared/unbound vreg_budget_bytes | `{}` |  |
| b_k_max[vmem:q_proj] | derived | not applicable under the declared configuration | `{}` |  |
| excess_over_budget[vmem:q_proj] | derived | incomplete: depends on undeclared/unbound b_rq, vmem_budget_bytes | `{"D_n": 24, "R_q": 16, "b_q": 16, "s_Cq": 2, "s_Wq": 2, "s_scratch": 0}` |  |
| b_k_max[vmem:kv_proj] | derived | incomplete: depends on undeclared/unbound b_rk, q_resident, v_load, vmem_budget_bytes | `{"D_n": 24, "D_r": 8, "D_v": 24, "R_k": 32, "b_q": 16, "n_buf": 2, "s_Ck": 2, "s_Kn": 2...` |  |
| excess_over_budget[vmem:kv_proj] | derived | incomplete: depends on undeclared/unbound b_rk, q_resident, v_load, vmem_budget_bytes | `{"D_n": 24, "D_r": 8, "D_v": 24, "R_k": 32, "b_k": 16, "b_q": 16, "n_buf": 2, "s_Ck": 2...` |  |
| b_k_max[vmem:score] | derived | incomplete: depends on undeclared/unbound v_load, vmem_budget_bytes | `{"D_n": 24, "D_r": 8, "D_v": 24, "b_q": 16, "n_buf": 2, "s_Kn": 2, "s_Kr": 2, "s_Qn": 2...` |  |
| excess_over_budget[vmem:score] | derived | incomplete: depends on undeclared/unbound v_load, vmem_budget_bytes | `{"D_n": 24, "D_r": 8, "D_v": 24, "b_k": 16, "b_q": 16, "n_buf": 2, "s_Kn": 2, "s_Kr": 2...` |  |
| b_k_max[vmem:softmax] | derived | incomplete: depends on undeclared/unbound q_resident, v_load, vmem_budget_bytes | `{"D_n": 24, "D_r": 8, "D_v": 24, "b_q": 16, "s_Qn": 2, "s_Qr": 2, "s_V": 2, "s_scratch"...` |  |
| excess_over_budget[vmem:softmax] | derived | incomplete: depends on undeclared/unbound q_resident, v_load, vmem_budget_bytes | `{"D_n": 24, "D_r": 8, "D_v": 24, "b_k": 16, "b_q": 16, "s_Qn": 2, "s_Qr": 2, "s_V": 2, ...` |  |
| b_k_max[vmem:pv] | derived | incomplete: depends on undeclared/unbound q_resident, vmem_budget_bytes | `{"D_n": 24, "D_r": 8, "D_v": 24, "b_q": 16, "s_Qn": 2, "s_Qr": 2, "s_V": 2, "s_scratch"...` |  |
| excess_over_budget[vmem:pv] | derived | incomplete: depends on undeclared/unbound q_resident, vmem_budget_bytes | `{"D_n": 24, "D_r": 8, "D_v": 24, "b_k": 16, "b_q": 16, "s_Qn": 2, "s_Qr": 2, "s_V": 2, ...` |  |
| b_k_max[vmem:finalize] | derived | not applicable under the declared configuration | `{}` |  |
| excess_over_budget[vmem:finalize] | derived | incomplete: depends on undeclared/unbound q_resident, vmem_budget_bytes | `{"D_n": 24, "D_r": 8, "D_v": 24, "b_q": 16, "s_LSE": 4, "s_O": 2, "s_Qn": 2, "s_Qr": 2,...` |  |
| b_k_max[vmem:binding_stage] | derived | incomplete: depends on undeclared/unbound b_rk, b_rq, q_resident, v_load, vmem_budget_bytes | `{}` |  |
| N_spill_instructions | definition | incomplete: depends on undeclared/unbound compile_evidence | `{}` |  |
| M_spill_peak | definition | incomplete: depends on undeclared/unbound compile_or_measurement_evidence | `{}` |  |
| B_spill_fill | definition | incomplete: depends on undeclared/unbound compile_or_measurement_evidence | `{}` |  |
| dT_spill | definition | incomplete: depends on undeclared/unbound compile_or_measurement_evidence, declared comparison graph | `{}` |  |
| B_extra_opt | definition | incomplete: depends on undeclared/unbound graph, machine_model, allowed_transformations, objective | `{}` |  |
| T_opt | definition | incomplete: depends on undeclared/unbound graph, machine_model, allowed_transformations, schedule_model | `{}` |  |
| B_extra_opt_under_time_budget | definition | incomplete: depends on undeclared/unbound graph, machine_model, allowed_transformations, schedule_model, time_budget_tau | `{}` |  |

</details>

## 8. sensitivity and schedule counts

| metric | formula id | expression | value | unit | status | missing | scope | not equivalent to |
|---|---|---|---|---|---|---|---|---|
| d(score_bytes)/d(b_q) | `sens.score_bytes.b_q` | `b_k*s_X` | 64 | byte per row | bound |  | continuous relaxation of the source-level logical size | a discrete tile change |
| d(score_bytes)/d(b_k) | `sens.score_bytes.b_k` | `b_q*s_X` | 64 | byte per row | bound |  | continuous relaxation of the source-level logical size | a discrete tile change |
| d(accumulator_bytes)/d(b_q) | `sens.accumulator_bytes.b_q` | `D_v*s_A` | 96 | byte per row | bound |  | continuous relaxation of the source-level logical size | a discrete tile change |
| d(accumulator_bytes)/d(b_k) | `sens.accumulator_bytes.b_k` | `0` | 0 | byte per row | bound |  | continuous relaxation of the source-level logical size | an absence of any effect of b_k; a discrete tile change |
| n_programs | `sched.programs` | `B*ceiling(H/h_pp)*ceiling(S_q/b_q)` | 32 | programs | partial | schedule, scheduled_lengths | (batch, head-group, q-block) programs under the declared loop nest |  |
| n_kv_block_visits | `sched.kv_visits` | `B*H*BlockStat(selected, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)` | 80 | visits | partial | schedule, scheduled_lengths | selected (b, h, q-block, kv-block) rectangles |  |

Assumptions:
- `d(accumulator_bytes)/d(b_k)`: a zero derivative is the absence of a *direct* dependence in the source-level logical size; scheduling and buffer lifetime can still create an indirect effect (spec 10.1)
- `n_programs`: scheduled extent undeclared: assumed equal to the active extent (scenario)
- `n_kv_block_visits`: scheduled extent undeclared: assumed equal to the active extent (scenario)

<details><summary>Provenance (bindings, coverage, source, evidence)</summary>

| metric | source | coverage | bindings | evidence |
|---|---|---|---|---|
| d(score_bytes)/d(b_q) | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"b_k": 16, "s_X": 4}` |  |
| d(score_bytes)/d(b_k) | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"b_q": 16, "s_X": 4}` |  |
| d(accumulator_bytes)/d(b_q) | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"D_v": 24, "s_A": 4}` |  |
| d(accumulator_bytes)/d(b_k) | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{}` |  |
| n_programs | derived | incomplete: depends on undeclared/unbound schedule, scheduled_lengths | `{"B": 2, "H": 4, "S_q": 64, "b_q": 16, "h_pp": 1}` |  |
| n_kv_block_visits | derived | incomplete: depends on undeclared/unbound schedule, scheduled_lengths | `{"B": 2, "Delta": 0, "H": 4, "S_k": 64, "S_q": 64, "b_k": 16, "b_q": 16}` |  |

</details>

## 9. hardware lower bounds and calibration

| metric | formula id | expression | value | unit | status | missing | scope | not equivalent to |
|---|---|---|---|---|---|---|---|---|
| T_LB[mxu] | `perf.lb.mxu` | `(2*B*D_n*H*R_k*BlockStat(key_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + 2*B*D_n*H*R_q*S_q + 2*B*D_n*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + 2*B*D_r*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + 2*B*D_v*H*R_k*BlockStat(key_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + 2*B*D_v*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k))/P_mxu` | 7.2104e-09 | s | partial | schedule, scheduled_lengths | scope=per_chip | actual latency; calibrated prediction |
| T_LB[vpu] | `perf.lb.vpu` | `(2*B*D_v*H*S_q + 2*B*D_v*H*BlockStat(row_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + 2*B*H*S_q + B*H*BlockStat(masked_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + 2*B*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + 2*B*H*BlockStat(row_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + Piecewise((B*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k), D_r > 0), (0, True)))/P_vpu` |  | s | unknown | P_vpu, schedule, scheduled_lengths | scope=per_chip | actual latency; calibrated prediction |
| T_LB[reduce] | `perf.lb.reduce` | `(2*B*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) - B*H*BlockStat(row_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k))/P_red` |  | s | unknown | P_red, schedule, scheduled_lengths | scope=per_chip | actual latency; calibrated prediction |
| T_LB[layout] | `perf.lb.layout` | `(B*D_n*H*BlockStat(key_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + B*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k))/P_layout` |  | s | unknown | P_layout, schedule, scheduled_lengths | scope=None | actual latency; calibrated prediction |
| T_LB[exp] | `perf.lb.exp` | `(B*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + B*H*BlockStat(row_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k))/P_exp` |  | s | unknown | P_exp, schedule, scheduled_lengths | scope=per_chip | actual latency; calibrated prediction |
| T_LB[path:hbm_to_vmem] | `perf.lb.path:hbm_to_vmem` | `(B*D_n*H*R_k*s_Wk*ceiling(S_q/b_q) + B*D_n*H*R_q*s_Wq*ceiling(S_q/b_q) + B*D_r*H*S_q*s_Qr + B*D_r*s_Kr*ceiling(H/h_pp)*BlockStat(key_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + B*D_v*H*R_k*s_Wv*ceiling(S_q/b_q) + B*R_k*s_Ck*ceiling(H/h_pp)*BlockStat(key_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + B*R_q*S_q*s_Cq*ceiling(H/h_pp))/W_hbm` | 1.5235e-07 | s | partial | schedule, scheduled_lengths | scope=per_chip | actual latency; calibrated prediction |
| T_LB[path:vmem_to_hbm] | `perf.lb.path:vmem_to_hbm` | `(B*D_v*H*S_q*s_O + B*H*S_q*s_LSE)/W_hbm_w` |  | s | unknown | W_hbm_w, schedule, scheduled_lengths | scope=None | actual latency; calibrated prediction |
| T_LB[path:vmem_to_vreg] | `perf.lb.path:vmem_to_vreg` | `(B*D_n*H*b_k*s_Kn*BlockStat(selected, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + B*D_n*H*b_q*s_Qn*ceiling(S_q/b_q) + B*D_r*H*b_k*s_Kr*BlockStat(selected, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + B*D_r*H*b_q*s_Qr*ceiling(S_q/b_q) + B*D_v*H*b_k*s_V*BlockStat(selected, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k))/W_vmem` |  | s | unknown | W_vmem, q_resident, register_schedule_evidence, schedule, scheduled_lengths | scope=per_tensorcore | actual latency; calibrated prediction |
| T_LB[critical_path] | `perf.lb.critical_path` | `CP` |  | s | unknown | CP | scope=None | actual latency; calibrated prediction |
| T_LB[combined] | `perf.lb.combined` | `max((2*B*D_n*H*R_k*BlockStat(key_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + 2*B*D_n*H*R_q*S_q + 2*B*D_n*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + 2*B*D_r*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + 2*B*D_v*H*R_k*BlockStat(key_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + 2*B*D_v*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k))/P_mxu, (2*B*D_v*H*S_q + 2*B*D_v*H*BlockStat(row_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + 2*B*H*S_q + B*H*BlockStat(masked_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + 2*B*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + 2*B*H*BlockStat(row_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + Piecewise((B*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k), D_r > 0), (0, True)))/P_vpu, (2*B*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) - B*H*BlockStat(row_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k))/P_red, (B*D_n*H*BlockStat(key_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + B*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k))/P_layout, (B*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + B*H*BlockStat(row_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k))/P_exp, (B*D_n*H*R_k*s_Wk*ceiling(S_q/b_q) + B*D_n*H*R_q*s_Wq*ceiling(S_q/b_q) + B*D_r*H*S_q*s_Qr + B*D_r*s_Kr*ceiling(H/h_pp)*BlockStat(key_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + B*D_v*H*R_k*s_Wv*ceiling(S_q/b_q) + B*R_k*s_Ck*ceiling(H/h_pp)*BlockStat(key_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + B*R_q*S_q*s_Cq*ceiling(H/h_pp))/W_hbm, (B*D_v*H*S_q*s_O + B*H*S_q*s_LSE)/W_hbm_w, (B*D_n*H*b_k*s_Kn*BlockStat(selected, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + B*D_n*H*b_q*s_Qn*ceiling(S_q/b_q) + B*D_r*H*b_k*s_Kr*BlockStat(selected, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + B*D_r*H*b_q*s_Qr*ceiling(S_q/b_q) + B*D_v*H*b_k*s_V*BlockStat(selected, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k))/W_vmem, CP)` | 1.5235e-07 | s | partial | CP, P_exp, P_layout, P_red, P_vpu, W_hbm_w, W_vmem, q_resident, register_schedule_evidence, schedule, scheduled_lengths | overlap model full_overlap_max | calibrated prediction; measured time |
| T_pred[node_interval] | `perf.calibrated.nodes` | `sum_v W_v/(eps_v P_v) + t_startup, + T_launch + eps_model` |  | s | unknown | calibration | no applicable calibration: no prediction issued | T_LB; hardware-peak lower bound |

<details><summary>Provenance (bindings, coverage, source, evidence)</summary>

| metric | source | coverage | bindings | evidence |
|---|---|---|---|---|
| T_LB[mxu] | derived | incomplete: depends on undeclared/unbound schedule, scheduled_lengths | `{}` |  |
| T_LB[vpu] | derived | incomplete: depends on undeclared/unbound P_vpu, schedule, scheduled_lengths | `{}` |  |
| T_LB[reduce] | derived | incomplete: depends on undeclared/unbound P_red, schedule, scheduled_lengths | `{}` |  |
| T_LB[layout] | derived | incomplete: depends on undeclared/unbound P_layout, schedule, scheduled_lengths | `{}` |  |
| T_LB[exp] | derived | incomplete: depends on undeclared/unbound P_exp, schedule, scheduled_lengths | `{}` |  |
| T_LB[path:hbm_to_vmem] | derived | incomplete: depends on undeclared/unbound schedule, scheduled_lengths | `{}` |  |
| T_LB[path:vmem_to_hbm] | derived | incomplete: depends on undeclared/unbound W_hbm_w, schedule, scheduled_lengths | `{}` |  |
| T_LB[path:vmem_to_vreg] | derived | incomplete: depends on undeclared/unbound W_vmem, q_resident, register_schedule_evidence, schedule, scheduled_lengths | `{}` |  |
| T_LB[critical_path] | derived | incomplete: depends on undeclared/unbound CP | `{}` |  |
| T_LB[combined] | derived | incomplete: depends on undeclared/unbound CP, P_exp, P_layout, P_red, P_vpu, W_hbm_w, W_vmem, q_resident, register_schedule_evidence, schedule, scheduled_lengths | `{}` |  |
| T_pred[node_interval] | derived | incomplete: no applicable calibration evidence attached | `{}` |  |

</details>

## Constraints

| constraint | relation | holds | severity | missing |
|---|---|---|---|---|
| q_tail | `Eq(Mod(S_q, b_q), 0)` | True | info |  |
| kv_tail | `Eq(Mod(S_k, b_k), 0)` | True | info |  |
| Dr_zero_or_positive | `True` | True | info |  |
| h_pp_le_H | `h_pp <= H` | True | error |  |
| kv_buffers_ge_1 | `True` | True | error |  |

## vector_ops

```json
[
  {
    "id": "scale_scores",
    "stage": "score",
    "kind": "mul",
    "resource": "vpu",
    "count": "B*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)",
    "conditional_on": null,
    "exists": true,
    "description": "gamma * X on every executed cell (fused variants differ)"
  },
  {
    "id": "mask_apply",
    "stage": "score",
    "kind": "mask",
    "resource": "vpu",
    "count": "B*H*BlockStat(masked_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)",
    "conditional_on": "mask application scenario: not-fully-visible rectangles only",
    "exists": true,
    "description": "additive/select mask on executed cells of rectangles that are not fully visible (fully visible blocks need no mask)"
  },
  {
    "id": "row_max_cmp",
    "stage": "softmax",
    "kind": "cmp",
    "resource": "reduce",
    "count": "B*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) - B*H*BlockStat(row_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)",
    "conditional_on": null,
    "exists": true,
    "description": "n*max(m-1,0) comparisons for rowmax over executed cells"
  },
  {
    "id": "old_new_max_cmp",
    "stage": "softmax",
    "kind": "cmp",
    "resource": "reduce",
    "count": "B*H*BlockStat(row_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)",
    "conditional_on": null,
    "exists": true,
    "description": "max(m, rowmax) once per row per block visit"
  },
  {
    "id": "rowmax_broadcast",
    "stage": "softmax",
    "kind": "broadcast",
    "resource": "layout",
    "count": "B*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)",
    "conditional_on": "broadcast_materialized (undeclared backend behaviour)",
    "exists": null,
    "description": "broadcast m' across the row before X - m' (may be free depending on layout)"
  },
  {
    "id": "kv_operand_transpose",
    "stage": "score",
    "kind": "transpose",
    "resource": "layout",
    "count": "B*D_n*H*BlockStat(key_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)",
    "conditional_on": "operand_transpose_materialized (undeclared backend behaviour)",
    "exists": null,
    "description": "re-layout / transpose of the K-side operand for the score matmul (free when the layout already matches)"
  },
  {
    "id": "exp_E",
    "stage": "softmax",
    "kind": "exp",
    "resource": "exp",
    "count": "B*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)",
    "conditional_on": null,
    "exists": true,
    "description": "E = exp(X - m') on every executed cell (masked cells included)"
  },
  {
    "id": "row_sum_add",
    "stage": "softmax",
    "kind": "add",
    "resource": "reduce",
    "count": "B*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) - B*H*BlockStat(row_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)",
    "conditional_on": null,
    "exists": true,
    "description": "n*max(m-1,0) additions for rowsum(E)"
  },
  {
    "id": "rescale_alpha_exp",
    "stage": "softmax",
    "kind": "exp",
    "resource": "exp",
    "count": "B*H*BlockStat(row_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)",
    "conditional_on": null,
    "exists": true,
    "description": "alpha = exp(m - m') at most once per row per block visit"
  },
  {
    "id": "acc_scale",
    "stage": "pv",
    "kind": "mul",
    "resource": "vpu",
    "count": "B*D_v*H*BlockStat(row_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)",
    "conditional_on": null,
    "exists": true,
    "description": "alpha * A (n*D_A per block visit)"
  },
  {
    "id": "acc_add",
    "stage": "pv",
    "kind": "add",
    "resource": "vpu",
    "count": "B*D_v*H*BlockStat(row_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)",
    "conditional_on": null,
    "exists": true,
    "description": "A + E U (n*D_A per block visit; matmul epilogue may fuse)"
  },
  {
    "id": "l_scale",
    "stage": "softmax",
    "kind": "mul",
    "resource": "vpu",
    "count": "B*H*BlockStat(row_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)",
    "conditional_on": null,
    "exists": true,
    "description": "alpha * l"
  },
  {
    "id": "l_update",
    "stage": "softmax",
    "kind": "add",
    "resource": "vpu",
    "count": "B*H*BlockStat(row_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)",
    "conditional_on": null,
    "exists": true,
    "description": "l' = alpha*l + rowsum"
  },
  {
    "id": "final_normalize_div",
    "stage": "finalize",
    "kind": "div",
    "resource": "vpu",
    "count": "B*D_v*H*S_q",
    "conditional_on": "final_normalize",
    "exists": null,
    "description": "Y = A / l once per row, as n*D_A divisions (form 'divide')"
  },
  {
    "id": "final_normalize_recip",
    "stage": "finalize",
    "kind": "recip",
    "resource": "vpu",
    "count": "B*H*S_q",
    "conditional_on": "final_normalize",
    "exists": false,
    "description": "1/l once per row (form 'reciprocal_multiply')"
  },
  {
    "id": "final_normalize_mul",
    "stage": "finalize",
    "kind": "mul",
    "resource": "vpu",
    "count": "B*D_v*H*S_q",
    "conditional_on": "final_normalize",
    "exists": false,
    "description": "Y = A * (1/l), n*D_A multiplies (form 'reciprocal_multiply')"
  },
  {
    "id": "lse_log",
    "stage": "finalize",
    "kind": "log",
    "resource": "vpu",
    "count": "B*H*S_q",
    "conditional_on": null,
    "exists": true,
    "description": "log(l) once per row for LSE = m + log(l)"
  },
  {
    "id": "lse_add",
    "stage": "finalize",
    "kind": "add",
    "resource": "vpu",
    "count": "B*H*S_q",
    "conditional_on": null,
    "exists": true,
    "description": "m + log(l) once per row"
  },
  {
    "id": "cast[exp_to_p]",
    "stage": "pv",
    "kind": "cast",
    "resource": "vpu",
    "count": "B*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)",
    "conditional_on": "cast_points",
    "exists": true,
    "description": "cast E to the PV / P C_k matmul operand dtype (inferred from the declared dtypes)"
  },
  {
    "id": "cast[accumulator_to_output]",
    "stage": "finalize",
    "kind": "cast",
    "resource": "vpu",
    "count": "B*D_v*H*S_q",
    "conditional_on": "cast_points",
    "exists": true,
    "description": "cast the final output rows to the output dtype (n*D_v) (inferred from the declared dtypes)"
  },
  {
    "id": "cast[sites_undeclared]",
    "stage": "softmax",
    "kind": "cast",
    "resource": "vpu",
    "count": "0",
    "conditional_on": "cast_points",
    "exists": null,
    "description": "cast_points undeclared: the two dtype-implied sites above are the scenario; a declared list gives a different graph"
  },
  {
    "id": "combine_qk_branches",
    "stage": "score",
    "kind": "add",
    "resource": "vpu",
    "count": "Piecewise((B*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k), D_r > 0), (0, True))",
    "conditional_on": "D_r > 0",
    "exists": true,
    "description": "X = Q^n K^n^T + Q^r K^r^T (0 when D_r = 0; fused K-concat variants differ)"
  }
]
```

## interface_ledger

```json
{
  "tensors": [
    {
      "role": "q_latent",
      "dtype": "bfloat16",
      "shape": [
        2,
        64,
        16
      ],
      "logical_bytes": 4096,
      "allocated_bytes": 4096,
      "alias_of": null,
      "counted_in_allocation": true,
      "alias_group": "q_latent",
      "strides": null,
      "storage_offset": 0,
      "allocation_from_strides_modelled": true
    },
    {
      "role": "kv_latent",
      "dtype": "bfloat16",
      "shape": [
        2,
        64,
        32
      ],
      "logical_bytes": 8192,
      "allocated_bytes": 8192,
      "alias_of": null,
      "counted_in_allocation": true,
      "alias_group": "kv_latent",
      "strides": null,
      "storage_offset": 0,
      "allocation_from_strides_modelled": true
    },
    {
      "role": "w_q_nope",
      "dtype": "bfloat16",
      "shape": [
        4,
        16,
        24
      ],
      "logical_bytes": 3072,
      "allocated_bytes": 3072,
      "alias_of": null,
      "counted_in_allocation": true,
      "alias_group": "w_q_nope",
      "strides": null,
      "storage_offset": 0,
      "allocation_from_strides_modelled": true
    },
    {
      "role": "w_k_nope",
      "dtype": "bfloat16",
      "shape": [
        4,
        32,
        24
      ],
      "logical_bytes": 6144,
      "allocated_bytes": 6144,
      "alias_of": null,
      "counted_in_allocation": true,
      "alias_group": "w_k_nope",
      "strides": null,
      "storage_offset": 0,
      "allocation_from_strides_modelled": true
    },
    {
      "role": "w_v",
      "dtype": "bfloat16",
      "shape": [
        4,
        32,
        24
      ],
      "logical_bytes": 6144,
      "allocated_bytes": 6144,
      "alias_of": null,
      "counted_in_allocation": true,
      "alias_group": "w_v",
      "strides": null,
      "storage_offset": 0,
      "allocation_from_strides_modelled": true
    },
    {
      "role": "q_pe",
      "dtype": "bfloat16",
      "shape": [
        2,
        4,
        64,
        8
      ],
      "logical_bytes": 8192,
      "allocated_bytes": 8192,
      "alias_of": null,
      "counted_in_allocation": true,
      "alias_group": "q_pe",
      "strides": null,
      "storage_offset": 0,
      "allocation_from_strides_modelled": true
    },
    {
      "role": "k_pe",
      "dtype": "bfloat16",
      "shape": [
        2,
        64,
        8
      ],
      "logical_bytes": 2048,
      "allocated_bytes": 2048,
      "alias_of": null,
      "counted_in_allocation": true,
      "alias_group": "k_pe",
      "strides": null,
      "storage_offset": 0,
      "allocation_from_strides_modelled": true
    }
  ],
  "logical_total": 37888,
  "allocated_unique_total": 37888,
  "roles_with_unmodelled_strides": []
}
```

## materialization_ledger

```json
[
  {
    "id": "mat_expanded_kv",
    "tensor": "K^n,V",
    "conditional_on": "materialize.expanded_kv",
    "enabled": false,
    "bytes": "B*H*S_k*(D_n*s_Kn + D_v*s_V)",
    "write_traffic": "B*D_n*H*S_k*s_Kn + B*D_v*H*S_k*s_V",
    "read_traffic": "B*D_n*H*s_Kn*BlockStat(key_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + B*D_v*H*s_V*BlockStat(key_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)",
    "bytes_value": 0,
    "traffic_value": 0,
    "bytes_value_if_materialized": 49152,
    "traffic_value_if_materialized": 172032
  },
  {
    "id": "mat_q_nope",
    "tensor": "Q^n",
    "conditional_on": "materialize.q_nope",
    "enabled": false,
    "bytes": "B*D_n*H*S_q*s_Qn",
    "write_traffic": "B*D_n*H*S_q*s_Qn",
    "read_traffic": "B*D_n*H*S_q*s_Qn",
    "bytes_value": 0,
    "traffic_value": 0,
    "bytes_value_if_materialized": 24576,
    "traffic_value_if_materialized": 49152
  }
]
```

## transfer_events

```json
[
  {
    "id": "Cq_read",
    "path": "hbm_to_vmem",
    "bytes_per_event": "R_q*s_Cq",
    "executions": "B*S_q*ceiling(H/h_pp)",
    "total": "B*R_q*S_q*s_Cq*ceiling(H/h_pp)",
    "value": 16384,
    "description": "Q latent row per (head group, query row)",
    "scenario": "logical rows read (spec 4.2: a logical out-of-bounds is not an HBM read)",
    "undeclared": false,
    "conditional_on": null
  },
  {
    "id": "Qr_read",
    "path": "hbm_to_vmem",
    "bytes_per_event": "D_r*s_Qr",
    "executions": "B*H*S_q",
    "total": "B*D_r*H*S_q*s_Qr",
    "value": 8192,
    "description": "Q^r row per (head, query row)",
    "scenario": "logical rows read (spec 4.2)",
    "undeclared": false,
    "conditional_on": null
  },
  {
    "id": "Wq_read",
    "path": "hbm_to_vmem",
    "bytes_per_event": "D_n*R_q*s_Wq",
    "executions": "B*H*ceiling(S_q/b_q)",
    "total": "B*D_n*H*R_q*s_Wq*ceiling(S_q/b_q)",
    "value": 24576,
    "description": "W^q per (program, head) \u2014 no cross-program weight residency assumed",
    "scenario": "weights re-read per program",
    "undeclared": false,
    "conditional_on": null
  },
  {
    "id": "Kr_read",
    "path": "hbm_to_vmem",
    "bytes_per_event": "D_r*s_Kr",
    "executions": "B*ceiling(H/h_pp)*BlockStat(key_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)",
    "total": "B*D_r*s_Kr*ceiling(H/h_pp)*BlockStat(key_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)",
    "value": 20480,
    "description": "K^r per executed key column, once per (head-group, program-row)",
    "scenario": null,
    "undeclared": false,
    "conditional_on": null
  },
  {
    "id": "Ck_read",
    "path": "hbm_to_vmem",
    "bytes_per_event": "R_k*s_Ck",
    "executions": "B*ceiling(H/h_pp)*BlockStat(key_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)",
    "total": "B*R_k*s_Ck*ceiling(H/h_pp)*BlockStat(key_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)",
    "value": 81920,
    "description": "KV latent per executed key column, shared by h_pp heads (new tokens only when a cache is declared); present because expanded K/V are not materialized",
    "scenario": null,
    "undeclared": false,
    "conditional_on": null
  },
  {
    "id": "Wk_read",
    "path": "hbm_to_vmem",
    "bytes_per_event": "D_n*R_k*s_Wk",
    "executions": "B*H*ceiling(S_q/b_q)",
    "total": "B*D_n*H*R_k*s_Wk*ceiling(S_q/b_q)",
    "value": 49152,
    "description": "W^k per (program, head)",
    "scenario": "weights re-read per program",
    "undeclared": false,
    "conditional_on": null
  },
  {
    "id": "Wv_read",
    "path": "hbm_to_vmem",
    "bytes_per_event": "D_v*R_k*s_Wv",
    "executions": "B*H*ceiling(S_q/b_q)",
    "total": "B*D_v*H*R_k*s_Wv*ceiling(S_q/b_q)",
    "value": 49152,
    "description": "W^v per (program, head)",
    "scenario": "weights re-read per program",
    "undeclared": false,
    "conditional_on": null
  },
  {
    "id": "O_write",
    "path": "vmem_to_hbm",
    "bytes_per_event": "D_v*s_O",
    "executions": "B*H*S_q",
    "total": "B*D_v*H*S_q*s_O",
    "value": 24576,
    "description": "write one output row per (b, h, active query row)",
    "scenario": null,
    "undeclared": false,
    "conditional_on": null
  },
  {
    "id": "LSE_write",
    "path": "vmem_to_hbm",
    "bytes_per_event": "s_LSE",
    "executions": "B*H*S_q",
    "total": "B*H*S_q*s_LSE",
    "value": 2048,
    "description": "write one LSE value per (b, h, active query row)",
    "scenario": null,
    "undeclared": false,
    "conditional_on": "outputs"
  },
  {
    "id": "stream_Qn_operand",
    "path": "vmem_to_vreg",
    "bytes_per_event": "D_n*b_q*s_Qn",
    "executions": "B*H*ceiling(S_q/b_q)",
    "total": "B*D_n*H*b_q*s_Qn*ceiling(S_q/b_q)",
    "value": 24576,
    "description": "stream Qn_operand into registers once per (head, q-block)",
    "scenario": "operand streaming (no register-schedule evidence); Q residency undeclared",
    "undeclared": true,
    "conditional_on": "q_resident"
  },
  {
    "id": "stream_Qr_operand",
    "path": "vmem_to_vreg",
    "bytes_per_event": "D_r*b_q*s_Qr",
    "executions": "B*H*ceiling(S_q/b_q)",
    "total": "B*D_r*H*b_q*s_Qr*ceiling(S_q/b_q)",
    "value": 8192,
    "description": "stream Qr_operand into registers once per (head, q-block)",
    "scenario": "operand streaming (no register-schedule evidence); Q residency undeclared",
    "undeclared": true,
    "conditional_on": "q_resident"
  },
  {
    "id": "stream_K_tile",
    "path": "vmem_to_vreg",
    "bytes_per_event": "D_n*b_k*s_Kn",
    "executions": "B*H*BlockStat(selected, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)",
    "total": "B*D_n*H*b_k*s_Kn*BlockStat(selected, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)",
    "value": 61440,
    "description": "stream K_tile into registers once per (b, h, kv-block visit)",
    "scenario": "operand streaming (no register-schedule evidence)",
    "undeclared": true,
    "conditional_on": null
  },
  {
    "id": "stream_V_tile",
    "path": "vmem_to_vreg",
    "bytes_per_event": "D_v*b_k*s_V",
    "executions": "B*H*BlockStat(selected, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)",
    "total": "B*D_v*H*b_k*s_V*BlockStat(selected, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)",
    "value": 61440,
    "description": "stream V_tile into registers once per (b, h, kv-block visit)",
    "scenario": "operand streaming (no register-schedule evidence)",
    "undeclared": true,
    "conditional_on": null
  },
  {
    "id": "stream_Kr_tile",
    "path": "vmem_to_vreg",
    "bytes_per_event": "D_r*b_k*s_Kr",
    "executions": "B*H*BlockStat(selected, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)",
    "total": "B*D_r*H*b_k*s_Kr*BlockStat(selected, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)",
    "value": 20480,
    "description": "stream Kr_tile into registers once per (b, h, kv-block visit)",
    "scenario": "operand streaming (no register-schedule evidence)",
    "undeclared": true,
    "conditional_on": null
  }
]
```

## envelope_coefficients

```json
{
  "all:q_proj": {
    "stage": "q_proj",
    "level": null,
    "a": {
      "expression": "0",
      "value": 0,
      "residual": null
    },
    "b": {
      "expression": "D_n*s_Qacc + R_q*s_Cq",
      "value": 128,
      "residual": null
    },
    "c": {
      "expression": "0",
      "value": 0,
      "residual": null
    },
    "d": {
      "expression": "D_n*R_q*s_Wq + s_scratch",
      "value": 768,
      "residual": null
    },
    "remainder": {
      "expression": "0",
      "value": 0,
      "residual": null
    },
    "provenance": {
      "a": [],
      "b": [
        "q_latent_window",
        "q_proj_acc"
      ],
      "c": [],
      "d": [
        "wq_rank_tile",
        "fixed_scratch"
      ],
      "remainder": []
    },
    "undeclared": [
      "b_rq"
    ]
  },
  "all:kv_proj": {
    "stage": "kv_proj",
    "level": null,
    "a": {
      "expression": "0",
      "value": 0,
      "residual": null
    },
    "b": {
      "expression": "D_n*s_Qn + D_r*s_Qr + D_v*s_A + 2*s_state",
      "value": 168,
      "residual": null
    },
    "c": {
      "expression": "D_n*s_Kn + D_v*s_V + R_k*n_buf*s_Ck",
      "value": 224,
      "residual": null
    },
    "d": {
      "expression": "D_n*R_k*s_Wk + D_v*R_k*s_Wv + s_scratch",
      "value": 3072,
      "residual": null
    },
    "remainder": {
      "expression": "0",
      "value": 0,
      "residual": null
    },
    "provenance": {
      "a": [],
      "b": [
        "q_nope_operand",
        "row_state_m",
        "row_state_l",
        "accumulator_A",
        "q_pe_tile"
      ],
      "c": [
        "kv_latent_rank_tile",
        "k_nope_tile",
        "v_tile"
      ],
      "d": [
        "wk_rank_tile",
        "wv_rank_tile",
        "fixed_scratch"
      ],
      "remainder": []
    },
    "undeclared": [
      "q_resident",
      "b_rk",
      "v_load"
    ]
  },
  "all:score": {
    "stage": "score",
    "level": null,
    "a": {
      "expression": "Max(s_E, s_X)",
      "value": 4,
      "residual": null
    },
    "b": {
      "expression": "D_n*s_Qn + D_r*s_Qr + D_v*s_A + 2*s_state",
      "value": 168,
      "residual": null
    },
    "c": {
      "expression": "D_n*s_Kn + D_r*n_buf*s_Kr + D_v*s_V",
      "value": 128,
      "residual": null
    },
    "d": {
      "expression": "s_scratch",
      "value": 0,
      "residual": null
    },
    "remainder": {
      "expression": "0",
      "value": 0,
      "residual": null
    },
    "provenance": {
      "a": [
        "alias[score_or_E]:score_X|exp_E"
      ],
      "b": [
        "q_nope_operand",
        "row_state_m",
        "row_state_l",
        "accumulator_A",
        "q_pe_tile"
      ],
      "c": [
        "k_nope_tile",
        "v_tile",
        "k_pe_tile"
      ],
      "d": [
        "fixed_scratch"
      ],
      "remainder": []
    },
    "undeclared": [
      "v_load"
    ]
  },
  "all:softmax": {
    "stage": "softmax",
    "level": null,
    "a": {
      "expression": "Max(s_E, s_X)",
      "value": 4,
      "residual": null
    },
    "b": {
      "expression": "D_n*s_Qn + D_r*s_Qr + D_v*s_A + 3*s_state",
      "value": 172,
      "residual": null
    },
    "c": {
      "expression": "D_v*s_V",
      "value": 48,
      "residual": null
    },
    "d": {
      "expression": "s_scratch",
      "value": 0,
      "residual": null
    },
    "remainder": {
      "expression": "0",
      "value": 0,
      "residual": null
    },
    "provenance": {
      "a": [
        "alias[score_or_E]:score_X|exp_E"
      ],
      "b": [
        "q_nope_operand",
        "row_state_m",
        "row_state_l",
        "row_alpha",
        "accumulator_A",
        "q_pe_tile"
      ],
      "c": [
        "v_tile"
      ],
      "d": [
        "fixed_scratch"
      ],
      "remainder": []
    },
    "undeclared": [
      "q_resident",
      "v_load"
    ]
  },
  "all:pv": {
    "stage": "pv",
    "level": null,
    "a": {
      "expression": "s_P + Max(s_E, s_X)",
      "value": 6,
      "residual": null
    },
    "b": {
      "expression": "D_n*s_Qn + D_r*s_Qr + D_v*s_A + 3*s_state",
      "value": 172,
      "residual": null
    },
    "c": {
      "expression": "D_v*s_V",
      "value": 48,
      "residual": null
    },
    "d": {
      "expression": "s_scratch",
      "value": 0,
      "residual": null
    },
    "remainder": {
      "expression": "0",
      "value": 0,
      "residual": null
    },
    "provenance": {
      "a": [
        "p_operand",
        "alias[score_or_E]:score_X|exp_E"
      ],
      "b": [
        "q_nope_operand",
        "row_state_m",
        "row_state_l",
        "row_alpha",
        "accumulator_A",
        "q_pe_tile"
      ],
      "c": [
        "v_tile"
      ],
      "d": [
        "fixed_scratch"
      ],
      "remainder": []
    },
    "undeclared": [
      "q_resident"
    ]
  },
  "all:finalize": {
    "stage": "finalize",
    "level": null,
    "a": {
      "expression": "0",
      "value": 0,
      "residual": null
    },
    "b": {
      "expression": "D_n*s_Qn + D_r*s_Qr + D_v*s_A + D_v*s_O + s_LSE + 2*s_state",
      "value": 220,
      "residual": null
    },
    "c": {
      "expression": "0",
      "value": 0,
      "residual": null
    },
    "d": {
      "expression": "s_scratch",
      "value": 0,
      "residual": null
    },
    "remainder": {
      "expression": "0",
      "value": 0,
      "residual": null
    },
    "provenance": {
      "a": [],
      "b": [
        "q_nope_operand",
        "row_state_m",
        "row_state_l",
        "accumulator_A",
        "q_pe_tile",
        "out_tile",
        "lse_tile"
      ],
      "c": [],
      "d": [
        "fixed_scratch"
      ],
      "remainder": []
    },
    "undeclared": [
      "q_resident"
    ]
  },
  "vreg:q_proj": {
    "stage": "q_proj",
    "level": "vreg",
    "a": {
      "expression": "0",
      "value": 0,
      "residual": null
    },
    "b": {
      "expression": "D_n*s_Qacc",
      "value": 96,
      "residual": null
    },
    "c": {
      "expression": "0",
      "value": 0,
      "residual": null
    },
    "d": {
      "expression": "0",
      "value": 0,
      "residual": null
    },
    "remainder": {
      "expression": "0",
      "value": 0,
      "residual": null
    },
    "provenance": {
      "a": [],
      "b": [
        "q_proj_acc"
      ],
      "c": [],
      "d": [],
      "remainder": []
    },
    "undeclared": []
  },
  "vreg:kv_proj": {
    "stage": "kv_proj",
    "level": "vreg",
    "a": {
      "expression": "0",
      "value": 0,
      "residual": null
    },
    "b": {
      "expression": "D_v*s_A + 2*s_state",
      "value": 104,
      "residual": null
    },
    "c": {
      "expression": "0",
      "value": 0,
      "residual": null
    },
    "d": {
      "expression": "0",
      "value": 0,
      "residual": null
    },
    "remainder": {
      "expression": "0",
      "value": 0,
      "residual": null
    },
    "provenance": {
      "a": [],
      "b": [
        "row_state_m",
        "row_state_l",
        "accumulator_A"
      ],
      "c": [],
      "d": [],
      "remainder": []
    },
    "undeclared": []
  },
  "vreg:score": {
    "stage": "score",
    "level": "vreg",
    "a": {
      "expression": "Max(s_E, s_X)",
      "value": 4,
      "residual": null
    },
    "b": {
      "expression": "D_v*s_A + 2*s_state",
      "value": 104,
      "residual": null
    },
    "c": {
      "expression": "0",
      "value": 0,
      "residual": null
    },
    "d": {
      "expression": "0",
      "value": 0,
      "residual": null
    },
    "remainder": {
      "expression": "0",
      "value": 0,
      "residual": null
    },
    "provenance": {
      "a": [
        "alias[score_or_E]:score_X|exp_E"
      ],
      "b": [
        "row_state_m",
        "row_state_l",
        "accumulator_A"
      ],
      "c": [],
      "d": [],
      "remainder": []
    },
    "undeclared": []
  },
  "vreg:softmax": {
    "stage": "softmax",
    "level": "vreg",
    "a": {
      "expression": "Max(s_E, s_X)",
      "value": 4,
      "residual": null
    },
    "b": {
      "expression": "D_v*s_A + 3*s_state",
      "value": 108,
      "residual": null
    },
    "c": {
      "expression": "0",
      "value": 0,
      "residual": null
    },
    "d": {
      "expression": "0",
      "value": 0,
      "residual": null
    },
    "remainder": {
      "expression": "0",
      "value": 0,
      "residual": null
    },
    "provenance": {
      "a": [
        "alias[score_or_E]:score_X|exp_E"
      ],
      "b": [
        "row_state_m",
        "row_state_l",
        "row_alpha",
        "accumulator_A"
      ],
      "c": [],
      "d": [],
      "remainder": []
    },
    "undeclared": []
  },
  "vreg:pv": {
    "stage": "pv",
    "level": "vreg",
    "a": {
      "expression": "s_P + Max(s_E, s_X)",
      "value": 6,
      "residual": null
    },
    "b": {
      "expression": "D_v*s_A + 3*s_state",
      "value": 108,
      "residual": null
    },
    "c": {
      "expression": "0",
      "value": 0,
      "residual": null
    },
    "d": {
      "expression": "0",
      "value": 0,
      "residual": null
    },
    "remainder": {
      "expression": "0",
      "value": 0,
      "residual": null
    },
    "provenance": {
      "a": [
        "p_operand",
        "alias[score_or_E]:score_X|exp_E"
      ],
      "b": [
        "row_state_m",
        "row_state_l",
        "row_alpha",
        "accumulator_A"
      ],
      "c": [],
      "d": [],
      "remainder": []
    },
    "undeclared": []
  },
  "vreg:finalize": {
    "stage": "finalize",
    "level": "vreg",
    "a": {
      "expression": "0",
      "value": 0,
      "residual": null
    },
    "b": {
      "expression": "D_v*s_A + 2*s_state",
      "value": 104,
      "residual": null
    },
    "c": {
      "expression": "0",
      "value": 0,
      "residual": null
    },
    "d": {
      "expression": "0",
      "value": 0,
      "residual": null
    },
    "remainder": {
      "expression": "0",
      "value": 0,
      "residual": null
    },
    "provenance": {
      "a": [],
      "b": [
        "row_state_m",
        "row_state_l",
        "accumulator_A"
      ],
      "c": [],
      "d": [],
      "remainder": []
    },
    "undeclared": []
  },
  "vmem:q_proj": {
    "stage": "q_proj",
    "level": "vmem",
    "a": {
      "expression": "0",
      "value": 0,
      "residual": null
    },
    "b": {
      "expression": "R_q*s_Cq",
      "value": 32,
      "residual": null
    },
    "c": {
      "expression": "0",
      "value": 0,
      "residual": null
    },
    "d": {
      "expression": "D_n*R_q*s_Wq + s_scratch",
      "value": 768,
      "residual": null
    },
    "remainder": {
      "expression": "0",
      "value": 0,
      "residual": null
    },
    "provenance": {
      "a": [],
      "b": [
        "q_latent_window"
      ],
      "c": [],
      "d": [
        "wq_rank_tile",
        "fixed_scratch"
      ],
      "remainder": []
    },
    "undeclared": [
      "b_rq"
    ]
  },
  "vmem:kv_proj": {
    "stage": "kv_proj",
    "level": "vmem",
    "a": {
      "expression": "0",
      "value": 0,
      "residual": null
    },
    "b": {
      "expression": "D_n*s_Qn + D_r*s_Qr",
      "value": 64,
      "residual": null
    },
    "c": {
      "expression": "D_n*s_Kn + D_v*s_V + R_k*n_buf*s_Ck",
      "value": 224,
      "residual": null
    },
    "d": {
      "expression": "D_n*R_k*s_Wk + D_v*R_k*s_Wv + s_scratch",
      "value": 3072,
      "residual": null
    },
    "remainder": {
      "expression": "0",
      "value": 0,
      "residual": null
    },
    "provenance": {
      "a": [],
      "b": [
        "q_nope_operand",
        "q_pe_tile"
      ],
      "c": [
        "kv_latent_rank_tile",
        "k_nope_tile",
        "v_tile"
      ],
      "d": [
        "wk_rank_tile",
        "wv_rank_tile",
        "fixed_scratch"
      ],
      "remainder": []
    },
    "undeclared": [
      "q_resident",
      "b_rk",
      "v_load"
    ]
  },
  "vmem:score": {
    "stage": "score",
    "level": "vmem",
    "a": {
      "expression": "0",
      "value": 0,
      "residual": null
    },
    "b": {
      "expression": "D_n*s_Qn + D_r*s_Qr",
      "value": 64,
      "residual": null
    },
    "c": {
      "expression": "D_n*s_Kn + D_r*n_buf*s_Kr + D_v*s_V",
      "value": 128,
      "residual": null
    },
    "d": {
      "expression": "s_scratch",
      "value": 0,
      "residual": null
    },
    "remainder": {
      "expression": "0",
      "value": 0,
      "residual": null
    },
    "provenance": {
      "a": [],
      "b": [
        "q_nope_operand",
        "q_pe_tile"
      ],
      "c": [
        "k_nope_tile",
        "v_tile",
        "k_pe_tile"
      ],
      "d": [
        "fixed_scratch"
      ],
      "remainder": []
    },
    "undeclared": [
      "v_load"
    ]
  },
  "vmem:softmax": {
    "stage": "softmax",
    "level": "vmem",
    "a": {
      "expression": "0",
      "value": 0,
      "residual": null
    },
    "b": {
      "expression": "D_n*s_Qn + D_r*s_Qr",
      "value": 64,
      "residual": null
    },
    "c": {
      "expression": "D_v*s_V",
      "value": 48,
      "residual": null
    },
    "d": {
      "expression": "s_scratch",
      "value": 0,
      "residual": null
    },
    "remainder": {
      "expression": "0",
      "value": 0,
      "residual": null
    },
    "provenance": {
      "a": [],
      "b": [
        "q_nope_operand",
        "q_pe_tile"
      ],
      "c": [
        "v_tile"
      ],
      "d": [
        "fixed_scratch"
      ],
      "remainder": []
    },
    "undeclared": [
      "q_resident",
      "v_load"
    ]
  },
  "vmem:pv": {
    "stage": "pv",
    "level": "vmem",
    "a": {
      "expression": "0",
      "value": 0,
      "residual": null
    },
    "b": {
      "expression": "D_n*s_Qn + D_r*s_Qr",
      "value": 64,
      "residual": null
    },
    "c": {
      "expression": "D_v*s_V",
      "value": 48,
      "residual": null
    },
    "d": {
      "expression": "s_scratch",
      "value": 0,
      "residual": null
    },
    "remainder": {
      "expression": "0",
      "value": 0,
      "residual": null
    },
    "provenance": {
      "a": [],
      "b": [
        "q_nope_operand",
        "q_pe_tile"
      ],
      "c": [
        "v_tile"
      ],
      "d": [
        "fixed_scratch"
      ],
      "remainder": []
    },
    "undeclared": [
      "q_resident"
    ]
  },
  "vmem:finalize": {
    "stage": "finalize",
    "level": "vmem",
    "a": {
      "expression": "0",
      "value": 0,
      "residual": null
    },
    "b": {
      "expression": "D_n*s_Qn + D_r*s_Qr + D_v*s_O + s_LSE",
      "value": 116,
      "residual": null
    },
    "c": {
      "expression": "0",
      "value": 0,
      "residual": null
    },
    "d": {
      "expression": "s_scratch",
      "value": 0,
      "residual": null
    },
    "remainder": {
      "expression": "0",
      "value": 0,
      "residual": null
    },
    "provenance": {
      "a": [],
      "b": [
        "q_nope_operand",
        "q_pe_tile",
        "out_tile",
        "lse_tile"
      ],
      "c": [],
      "d": [
        "fixed_scratch"
      ],
      "remainder": []
    },
    "undeclared": [
      "q_resident"
    ]
  }
}
```

## resource_lower_bounds

```json
{
  "bounds": [
    {
      "resource": "mxu",
      "numerator": "2*B*D_n*H*R_k*BlockStat(key_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + 2*B*D_n*H*R_q*S_q + 2*B*D_n*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + 2*B*D_r*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + 2*B*D_v*H*R_k*BlockStat(key_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + 2*B*D_v*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)",
      "denominator": "P_mxu",
      "expression": "(2*B*D_n*H*R_k*BlockStat(key_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + 2*B*D_n*H*R_q*S_q + 2*B*D_n*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + 2*B*D_r*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + 2*B*D_v*H*R_k*BlockStat(key_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + 2*B*D_v*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k))/P_mxu",
      "value": 7.2103877995642695e-09,
      "unit": "s",
      "scope": "per_chip",
      "status": "partial",
      "missing": [
        "schedule",
        "scheduled_lengths"
      ],
      "notes": [
        "numerator: executed matmul FLOPs of this scenario (C_exec); denominator: matmul peak for the declared compute dtype"
      ]
    },
    {
      "resource": "vpu",
      "numerator": "2*B*D_v*H*S_q + 2*B*D_v*H*BlockStat(row_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + 2*B*H*S_q + B*H*BlockStat(masked_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + 2*B*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + 2*B*H*BlockStat(row_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + Piecewise((B*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k), D_r > 0), (0, True))",
      "denominator": "P_vpu",
      "expression": "(2*B*D_v*H*S_q + 2*B*D_v*H*BlockStat(row_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + 2*B*H*S_q + B*H*BlockStat(masked_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + 2*B*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + 2*B*H*BlockStat(row_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + Piecewise((B*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k), D_r > 0), (0, True)))/P_vpu",
      "value": null,
      "unit": "s",
      "scope": "per_chip",
      "status": "unknown",
      "missing": [
        "P_vpu",
        "schedule",
        "scheduled_lengths"
      ],
      "notes": []
    },
    {
      "resource": "reduce",
      "numerator": "2*B*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) - B*H*BlockStat(row_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)",
      "denominator": "P_red",
      "expression": "(2*B*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) - B*H*BlockStat(row_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k))/P_red",
      "value": null,
      "unit": "s",
      "scope": "per_chip",
      "status": "unknown",
      "missing": [
        "P_red",
        "schedule",
        "scheduled_lengths"
      ],
      "notes": []
    },
    {
      "resource": "layout",
      "numerator": "B*D_n*H*BlockStat(key_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + B*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)",
      "denominator": "P_layout",
      "expression": "(B*D_n*H*BlockStat(key_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + B*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k))/P_layout",
      "value": null,
      "unit": "s",
      "scope": null,
      "status": "unknown",
      "missing": [
        "P_layout",
        "schedule",
        "scheduled_lengths"
      ],
      "notes": []
    },
    {
      "resource": "exp",
      "numerator": "B*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + B*H*BlockStat(row_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)",
      "denominator": "P_exp",
      "expression": "(B*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + B*H*BlockStat(row_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k))/P_exp",
      "value": null,
      "unit": "s",
      "scope": "per_chip",
      "status": "unknown",
      "missing": [
        "P_exp",
        "schedule",
        "scheduled_lengths"
      ],
      "notes": []
    },
    {
      "resource": "path:hbm_to_vmem",
      "numerator": "B*D_n*H*R_k*s_Wk*ceiling(S_q/b_q) + B*D_n*H*R_q*s_Wq*ceiling(S_q/b_q) + B*D_r*H*S_q*s_Qr + B*D_r*s_Kr*ceiling(H/h_pp)*BlockStat(key_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + B*D_v*H*R_k*s_Wv*ceiling(S_q/b_q) + B*R_k*s_Ck*ceiling(H/h_pp)*BlockStat(key_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + B*R_q*S_q*s_Cq*ceiling(H/h_pp)",
      "denominator": "W_hbm",
      "expression": "(B*D_n*H*R_k*s_Wk*ceiling(S_q/b_q) + B*D_n*H*R_q*s_Wq*ceiling(S_q/b_q) + B*D_r*H*S_q*s_Qr + B*D_r*s_Kr*ceiling(H/h_pp)*BlockStat(key_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + B*D_v*H*R_k*s_Wv*ceiling(S_q/b_q) + B*R_k*s_Ck*ceiling(H/h_pp)*BlockStat(key_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + B*R_q*S_q*s_Cq*ceiling(H/h_pp))/W_hbm",
      "value": 1.5235121951219514e-07,
      "unit": "s",
      "scope": "per_chip",
      "status": "partial",
      "missing": [
        "schedule",
        "scheduled_lengths"
      ],
      "notes": [
        "conditional transfer requests of this scenario / path bandwidth upper bound; not an absolute limit for all algorithms"
      ]
    },
    {
      "resource": "path:vmem_to_hbm",
      "numerator": "B*D_v*H*S_q*s_O + B*H*S_q*s_LSE",
      "denominator": "W_hbm_w",
      "expression": "(B*D_v*H*S_q*s_O + B*H*S_q*s_LSE)/W_hbm_w",
      "value": null,
      "unit": "s",
      "scope": null,
      "status": "unknown",
      "missing": [
        "W_hbm_w",
        "schedule",
        "scheduled_lengths"
      ],
      "notes": [
        "conditional transfer requests of this scenario / path bandwidth upper bound; not an absolute limit for all algorithms"
      ]
    },
    {
      "resource": "path:vmem_to_vreg",
      "numerator": "B*D_n*H*b_k*s_Kn*BlockStat(selected, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + B*D_n*H*b_q*s_Qn*ceiling(S_q/b_q) + B*D_r*H*b_k*s_Kr*BlockStat(selected, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + B*D_r*H*b_q*s_Qr*ceiling(S_q/b_q) + B*D_v*H*b_k*s_V*BlockStat(selected, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)",
      "denominator": "W_vmem",
      "expression": "(B*D_n*H*b_k*s_Kn*BlockStat(selected, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + B*D_n*H*b_q*s_Qn*ceiling(S_q/b_q) + B*D_r*H*b_k*s_Kr*BlockStat(selected, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + B*D_r*H*b_q*s_Qr*ceiling(S_q/b_q) + B*D_v*H*b_k*s_V*BlockStat(selected, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k))/W_vmem",
      "value": null,
      "unit": "s",
      "scope": "per_tensorcore",
      "status": "unknown",
      "missing": [
        "W_vmem",
        "q_resident",
        "register_schedule_evidence",
        "schedule",
        "scheduled_lengths"
      ],
      "notes": [
        "conditional transfer requests of this scenario / path bandwidth upper bound; not an absolute limit for all algorithms",
        "path traffic depends on undeclared fields or on register-schedule evidence (scenario)"
      ]
    },
    {
      "resource": "critical_path",
      "numerator": "CP",
      "denominator": "1",
      "expression": "CP",
      "value": null,
      "unit": "s",
      "scope": null,
      "status": "unknown",
      "missing": [
        "CP"
      ],
      "notes": [
        "dependency-graph path length; unknown without a declared schedule model"
      ]
    }
  ],
  "combined_expression": "max((2*B*D_n*H*R_k*BlockStat(key_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + 2*B*D_n*H*R_q*S_q + 2*B*D_n*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + 2*B*D_r*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + 2*B*D_v*H*R_k*BlockStat(key_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + 2*B*D_v*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k))/P_mxu, (2*B*D_v*H*S_q + 2*B*D_v*H*BlockStat(row_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + 2*B*H*S_q + B*H*BlockStat(masked_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + 2*B*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + 2*B*H*BlockStat(row_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + Piecewise((B*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k), D_r > 0), (0, True)))/P_vpu, (2*B*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) - B*H*BlockStat(row_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k))/P_red, (B*D_n*H*BlockStat(key_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + B*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k))/P_layout, (B*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + B*H*BlockStat(row_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k))/P_exp, (B*D_n*H*R_k*s_Wk*ceiling(S_q/b_q) + B*D_n*H*R_q*s_Wq*ceiling(S_q/b_q) + B*D_r*H*S_q*s_Qr + B*D_r*s_Kr*ceiling(H/h_pp)*BlockStat(key_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + B*D_v*H*R_k*s_Wv*ceiling(S_q/b_q) + B*R_k*s_Ck*ceiling(H/h_pp)*BlockStat(key_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + B*R_q*S_q*s_Cq*ceiling(H/h_pp))/W_hbm, (B*D_v*H*S_q*s_O + B*H*S_q*s_LSE)/W_hbm_w, (B*D_n*H*b_k*s_Kn*BlockStat(selected, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + B*D_n*H*b_q*s_Qn*ceiling(S_q/b_q) + B*D_r*H*b_k*s_Kr*BlockStat(selected, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k) + B*D_r*H*b_q*s_Qr*ceiling(S_q/b_q) + B*D_v*H*b_k*s_V*BlockStat(selected, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k))/W_vmem, CP)",
  "resources_outside_declared_groups": [],
  "combined_value_known_terms_only": 1.5235121951219514e-07,
  "status": "partial",
  "overlap_model": "full_overlap_max",
  "combined_missing_fields": [
    "CP",
    "P_exp",
    "P_layout",
    "P_red",
    "P_vpu",
    "W_hbm_w",
    "W_vmem",
    "q_resident",
    "register_schedule_evidence",
    "schedule",
    "scheduled_lengths"
  ],
  "scenarios": {},
  "notes": [
    "T_actual >= T_LB only for the counted scenario; missing resources make T_LB a lower bound on the lower bound",
    "serial_stage_sum groups are resource classes of the whole kernel (declared non-overlap between them), not per-pipeline-stage attribution",
    "profile quantities have mixed scopes ['per_chip', 'per_tensorcore']; a combined bound mixes scopes and must not be read as a single-device time"
  ]
}
```

## calibrated_prediction

```json
{
  "status": "not_calibrated",
  "applied": false,
  "predictions": [],
  "missing_fields": [
    "calibration"
  ],
  "notes": [
    "no calibration buckets attached; only resource requirements and lower bounds are reported"
  ]
}
```

## dependency_graph

```json
{
  "metric_symbols": {
    "B": [
      "B"
    ],
    "H": [
      "H"
    ],
    "R_q": [
      "R_q"
    ],
    "R_k": [
      "R_k"
    ],
    "D_n": [
      "D_n"
    ],
    "D_r": [
      "D_r"
    ],
    "D_v": [
      "D_v"
    ],
    "S_q_active": [
      "S_q"
    ],
    "S_k_active": [
      "S_k"
    ],
    "Delta": [
      "Delta"
    ],
    "gamma": [
      "D_n",
      "D_r"
    ],
    "rows_without_visible_keys": [
      "B",
      "Delta",
      "H",
      "S_k",
      "S_q"
    ],
    "C_valid": [
      "B",
      "Delta",
      "H",
      "S_k",
      "S_q"
    ],
    "C_rect": [
      "B",
      "Delta",
      "H",
      "S_k",
      "S_q",
      "b_k",
      "b_q"
    ],
    "C_pad": [
      "B",
      "Delta",
      "H",
      "S_k",
      "S_q",
      "b_k",
      "b_q"
    ],
    "rect_waste": [
      "B",
      "Delta",
      "H",
      "S_k",
      "S_q",
      "b_k",
      "b_q"
    ],
    "selected_blocks": [
      "B",
      "Delta",
      "H",
      "S_k",
      "S_q",
      "b_k",
      "b_q"
    ],
    "partial_blocks": [
      "B",
      "Delta",
      "H",
      "S_k",
      "S_q",
      "b_k",
      "b_q"
    ],
    "fully_visible_blocks": [
      "B",
      "Delta",
      "H",
      "S_k",
      "S_q",
      "b_k",
      "b_q"
    ],
    "future_blocks_selected": [
      "B",
      "Delta",
      "H",
      "S_k",
      "S_q",
      "b_k",
      "b_q"
    ],
    "padding_blocks_selected": [
      "B",
      "Delta",
      "H",
      "S_k",
      "S_q",
      "b_k",
      "b_q"
    ],
    "future_blocks_skipped": [
      "B",
      "Delta",
      "H",
      "S_k",
      "S_q",
      "b_k",
      "b_q"
    ],
    "padding_blocks_skipped": [
      "B",
      "Delta",
      "H",
      "S_k",
      "S_q",
      "b_k",
      "b_q"
    ],
    "masked_cells": [
      "B",
      "Delta",
      "H",
      "S_k",
      "S_q",
      "b_k",
      "b_q"
    ],
    "F_Q[useful]": [
      "B",
      "D_n",
      "H",
      "R_q",
      "S_q"
    ],
    "F_K[useful]": [
      "B",
      "D_n",
      "H",
      "R_k",
      "S_k"
    ],
    "F_V[useful]": [
      "B",
      "D_v",
      "H",
      "R_k",
      "S_k"
    ],
    "F_QK_n[useful]": [
      "B",
      "D_n",
      "Delta",
      "H",
      "S_k",
      "S_q"
    ],
    "F_QK_r[useful]": [
      "B",
      "D_r",
      "Delta",
      "H",
      "S_k",
      "S_q"
    ],
    "F_PV[useful]": [
      "B",
      "D_v",
      "Delta",
      "H",
      "S_k",
      "S_q"
    ],
    "F_total[useful]": [
      "B",
      "D_n",
      "D_r",
      "D_v",
      "Delta",
      "H",
      "R_k",
      "R_q",
      "S_k",
      "S_q"
    ],
    "F_E[spec_closed_form]": [
      "B",
      "D_n",
      "D_r",
      "D_v",
      "Delta",
      "H",
      "R_k",
      "R_q",
      "S_k",
      "S_q"
    ],
    "N_Qproj": [
      "B",
      "H",
      "S_q"
    ],
    "N_Kproj[executed_graph]": [
      "B",
      "Delta",
      "H",
      "S_k",
      "S_q",
      "b_k",
      "b_q"
    ],
    "N_Kproj": [
      "B",
      "H",
      "S_k"
    ],
    "N_Vproj[executed_graph]": [
      "B",
      "Delta",
      "H",
      "S_k",
      "S_q",
      "b_k",
      "b_q"
    ],
    "N_Vproj": [
      "B",
      "H",
      "S_k"
    ],
    "F_proj[general]": [
      "B",
      "D_n",
      "D_v",
      "H",
      "R_k",
      "R_q",
      "S_k",
      "S_q"
    ],
    "F_total[rect]": [
      "B",
      "D_n",
      "D_r",
      "D_v",
      "Delta",
      "H",
      "R_k",
      "R_q",
      "S_k",
      "S_q",
      "b_k",
      "b_q"
    ],
    "F_total[executed_graph]": [
      "B",
      "D_n",
      "D_r",
      "D_v",
      "Delta",
      "H",
      "R_k",
      "R_q",
      "S_k",
      "S_q",
      "b_k",
      "b_q"
    ],
    "F_projection[executed_graph]": [
      "B",
      "D_n",
      "D_v",
      "Delta",
      "H",
      "R_k",
      "R_q",
      "S_k",
      "S_q",
      "b_k",
      "b_q"
    ],
    "F_attention[executed_graph]": [
      "B",
      "D_n",
      "D_r",
      "D_v",
      "Delta",
      "H",
      "S_k",
      "S_q",
      "b_k",
      "b_q"
    ],
    "F_A_minus_F_E": [
      "B",
      "D_n",
      "D_r",
      "D_v",
      "Delta",
      "H",
      "R_k",
      "S_k",
      "S_q"
    ],
    "F_A_minus_F_E[spec_closed_form]": [
      "B",
      "D_n",
      "D_v",
      "Delta",
      "H",
      "R_k",
      "S_k",
      "S_q"
    ],
    "W_vec[mul]": [
      "B",
      "D_v",
      "Delta",
      "H",
      "S_k",
      "S_q",
      "b_k",
      "b_q"
    ],
    "W_vec[mask]": [
      "B",
      "Delta",
      "H",
      "S_k",
      "S_q",
      "b_k",
      "b_q"
    ],
    "W_vec[cmp]": [
      "B",
      "Delta",
      "H",
      "S_k",
      "S_q",
      "b_k",
      "b_q"
    ],
    "W_vec[broadcast]": [
      "B",
      "Delta",
      "H",
      "S_k",
      "S_q",
      "b_k",
      "b_q"
    ],
    "W_vec[transpose]": [
      "B",
      "D_n",
      "Delta",
      "H",
      "S_k",
      "S_q",
      "b_k",
      "b_q"
    ],
    "W_vec[exp]": [
      "B",
      "Delta",
      "H",
      "S_k",
      "S_q",
      "b_k",
      "b_q"
    ],
    "W_vec[add]": [
      "B",
      "D_r",
      "D_v",
      "Delta",
      "H",
      "S_k",
      "S_q",
      "b_k",
      "b_q"
    ],
    "W_vec[div]": [
      "B",
      "D_v",
      "H",
      "S_q"
    ],
    "W_vec[recip]": [],
    "W_vec[log]": [
      "B",
      "H",
      "S_q"
    ],
    "W_vec[cast]": [
      "B",
      "D_v",
      "Delta",
      "H",
      "S_k",
      "S_q",
      "b_k",
      "b_q"
    ],
    "W_resource[vpu]": [
      "B",
      "D_r",
      "D_v",
      "Delta",
      "H",
      "S_k",
      "S_q",
      "b_k",
      "b_q"
    ],
    "W_resource[reduce]": [
      "B",
      "Delta",
      "H",
      "S_k",
      "S_q",
      "b_k",
      "b_q"
    ],
    "W_resource[layout]": [
      "B",
      "D_n",
      "Delta",
      "H",
      "S_k",
      "S_q",
      "b_k",
      "b_q"
    ],
    "W_resource[exp]": [
      "B",
      "Delta",
      "H",
      "S_k",
      "S_q",
      "b_k",
      "b_q"
    ],
    "exp_count_executed": [
      "B",
      "Delta",
      "H",
      "S_k",
      "S_q",
      "b_k",
      "b_q"
    ],
    "bytes[q_latent]": [
      "B",
      "R_q",
      "S_q",
      "s_Cq"
    ],
    "bytes[kv_latent]": [
      "B",
      "R_k",
      "S_k",
      "s_Ck"
    ],
    "bytes[q_pe]": [
      "B",
      "D_r",
      "H",
      "S_q",
      "s_Qr"
    ],
    "bytes[k_pe]": [
      "B",
      "D_r",
      "S_k",
      "s_Kr"
    ],
    "bytes[w_q_nope]": [
      "D_n",
      "H",
      "R_q",
      "s_Wq"
    ],
    "bytes[w_k_nope]": [
      "D_n",
      "H",
      "R_k",
      "s_Wk"
    ],
    "bytes[w_v]": [
      "D_v",
      "H",
      "R_k",
      "s_Wv"
    ],
    "M_in": [
      "s_Ck",
      "s_Cq",
      "s_Kr",
      "s_Qr",
      "s_Wk",
      "s_Wq",
      "s_Wv"
    ],
    "M_O": [
      "B",
      "D_v",
      "H",
      "S_q",
      "s_O"
    ],
    "M_LSE": [
      "B",
      "H",
      "S_q",
      "s_LSE"
    ],
    "HBM_mat[K^n,V]": [
      "B",
      "D_n",
      "D_v",
      "Delta",
      "H",
      "S_k",
      "S_q",
      "b_k",
      "b_q",
      "s_Kn",
      "s_V"
    ],
    "M_mat[K^n,V]": [
      "B",
      "D_n",
      "D_v",
      "H",
      "S_k",
      "s_Kn",
      "s_V"
    ],
    "HBM_mat[Q^n]": [
      "B",
      "D_n",
      "H",
      "S_q",
      "s_Qn"
    ],
    "M_mat[Q^n]": [
      "B",
      "D_n",
      "H",
      "S_q",
      "s_Qn"
    ],
    "B[hbm_to_vmem]": [
      "B",
      "D_n",
      "D_r",
      "D_v",
      "Delta",
      "H",
      "R_k",
      "R_q",
      "S_k",
      "S_q",
      "b_k",
      "b_q",
      "h_pp",
      "s_Ck",
      "s_Cq",
      "s_Kr",
      "s_Qr",
      "s_Wk",
      "s_Wq",
      "s_Wv"
    ],
    "B[vmem_to_hbm]": [
      "B",
      "D_v",
      "H",
      "S_q",
      "s_LSE",
      "s_O"
    ],
    "B[vmem_to_vreg]": [
      "B",
      "D_n",
      "D_r",
      "D_v",
      "Delta",
      "H",
      "S_k",
      "S_q",
      "b_k",
      "b_q",
      "s_Kn",
      "s_Kr",
      "s_Qn",
      "s_Qr",
      "s_V"
    ],
    "local[q_latent_window]": [
      "R_q",
      "b_q",
      "s_Cq"
    ],
    "local[wq_rank_tile]": [
      "D_n",
      "R_q",
      "s_Wq"
    ],
    "local[q_proj_acc]": [
      "D_n",
      "b_q",
      "s_Qacc"
    ],
    "local[q_nope_operand]": [
      "D_n",
      "b_q",
      "s_Qn"
    ],
    "local[kv_latent_rank_tile]": [
      "R_k",
      "b_k",
      "n_buf",
      "s_Ck"
    ],
    "local[wk_rank_tile]": [
      "D_n",
      "R_k",
      "s_Wk"
    ],
    "local[wv_rank_tile]": [
      "D_v",
      "R_k",
      "s_Wv"
    ],
    "local[k_nope_tile]": [
      "D_n",
      "b_k",
      "s_Kn"
    ],
    "local[v_tile]": [
      "D_v",
      "b_k",
      "s_V"
    ],
    "local[score_X]": [
      "b_k",
      "b_q",
      "s_X"
    ],
    "local[exp_E]": [
      "b_k",
      "b_q",
      "s_E"
    ],
    "local[p_operand]": [
      "b_k",
      "b_q",
      "s_P"
    ],
    "local[row_state_m]": [
      "b_q",
      "s_state"
    ],
    "local[row_state_l]": [
      "b_q",
      "s_state"
    ],
    "local[row_alpha]": [
      "b_q",
      "s_state"
    ],
    "local[accumulator_A]": [
      "D_v",
      "b_q",
      "s_A"
    ],
    "local[q_pe_tile]": [
      "D_r",
      "b_q",
      "s_Qr"
    ],
    "local[k_pe_tile]": [
      "D_r",
      "b_k",
      "n_buf",
      "s_Kr"
    ],
    "local[fixed_scratch]": [
      "s_scratch"
    ],
    "local[out_tile]": [
      "D_v",
      "b_q",
      "s_O"
    ],
    "local[lse_tile]": [
      "b_q",
      "s_LSE"
    ],
    "layout[q_latent_window]": [
      "L_l",
      "L_s",
      "R_q",
      "b_q",
      "s_Cq"
    ],
    "layout[wq_rank_tile]": [
      "D_n",
      "L_l",
      "L_s",
      "R_q",
      "s_Wq"
    ],
    "layout[q_proj_acc]": [
      "D_n",
      "b_q",
      "s_Qacc"
    ],
    "layout[q_nope_operand]": [
      "D_n",
      "L_l",
      "L_s",
      "b_q",
      "s_Qn"
    ],
    "layout[kv_latent_rank_tile]": [
      "L_l",
      "L_s",
      "R_k",
      "b_k",
      "n_buf",
      "s_Ck"
    ],
    "layout[wk_rank_tile]": [
      "D_n",
      "L_l",
      "L_s",
      "R_k",
      "s_Wk"
    ],
    "layout[wv_rank_tile]": [
      "D_v",
      "L_l",
      "L_s",
      "R_k",
      "s_Wv"
    ],
    "layout[k_nope_tile]": [
      "D_n",
      "L_l",
      "L_s",
      "b_k",
      "s_Kn"
    ],
    "layout[v_tile]": [
      "D_v",
      "L_l",
      "L_s",
      "b_k",
      "s_V"
    ],
    "layout[score_X]": [
      "b_k",
      "b_q",
      "s_X"
    ],
    "layout[exp_E]": [
      "b_k",
      "b_q",
      "s_E"
    ],
    "layout[p_operand]": [
      "L_l",
      "L_s",
      "b_k",
      "b_q",
      "s_P"
    ],
    "layout[row_state_m]": [
      "b_q",
      "s_state"
    ],
    "layout[row_state_l]": [
      "b_q",
      "s_state"
    ],
    "layout[row_alpha]": [
      "b_q",
      "s_state"
    ],
    "layout[accumulator_A]": [
      "D_v",
      "b_q",
      "s_A"
    ],
    "layout[q_pe_tile]": [
      "D_r",
      "L_l",
      "L_s",
      "b_q",
      "s_Qr"
    ],
    "layout[k_pe_tile]": [
      "D_r",
      "L_l",
      "L_s",
      "b_k",
      "n_buf",
      "s_Kr"
    ],
    "layout[fixed_scratch]": [
      "s_scratch"
    ],
    "layout[out_tile]": [
      "D_v",
      "L_l",
      "L_s",
      "b_q",
      "s_O"
    ],
    "layout[lse_tile]": [
      "b_q",
      "s_LSE"
    ],
    "M_live[all:q_proj]": [
      "D_n",
      "R_q",
      "b_q",
      "s_Cq",
      "s_Qacc",
      "s_Wq",
      "s_scratch"
    ],
    "M_live[all:kv_proj]": [
      "D_n",
      "D_r",
      "D_v",
      "R_k",
      "b_k",
      "b_q",
      "n_buf",
      "s_A",
      "s_Ck",
      "s_Kn",
      "s_Qn",
      "s_Qr",
      "s_V",
      "s_Wk",
      "s_Wv",
      "s_scratch",
      "s_state"
    ],
    "M_live[all:score]": [
      "D_n",
      "D_r",
      "D_v",
      "b_k",
      "b_q",
      "n_buf",
      "s_A",
      "s_E",
      "s_Kn",
      "s_Kr",
      "s_Qn",
      "s_Qr",
      "s_V",
      "s_X",
      "s_scratch",
      "s_state"
    ],
    "M_live[all:softmax]": [
      "D_n",
      "D_r",
      "D_v",
      "b_k",
      "b_q",
      "s_A",
      "s_E",
      "s_Qn",
      "s_Qr",
      "s_V",
      "s_X",
      "s_scratch",
      "s_state"
    ],
    "M_live[all:pv]": [
      "D_n",
      "D_r",
      "D_v",
      "b_k",
      "b_q",
      "s_A",
      "s_E",
      "s_P",
      "s_Qn",
      "s_Qr",
      "s_V",
      "s_X",
      "s_scratch",
      "s_state"
    ],
    "M_live[all:finalize]": [
      "D_n",
      "D_r",
      "D_v",
      "b_q",
      "s_A",
      "s_LSE",
      "s_O",
      "s_Qn",
      "s_Qr",
      "s_scratch",
      "s_state"
    ],
    "M_peak[all]": [
      "D_n",
      "D_r",
      "D_v",
      "R_k",
      "R_q",
      "b_k",
      "b_q",
      "n_buf",
      "s_A",
      "s_Ck",
      "s_Cq",
      "s_E",
      "s_Kn",
      "s_Kr",
      "s_LSE",
      "s_O",
      "s_P",
      "s_Qacc",
      "s_Qn",
      "s_Qr",
      "s_V",
      "s_Wk",
      "s_Wq",
      "s_Wv",
      "s_X",
      "s_scratch",
      "s_state"
    ],
    "M_live[vreg:q_proj]": [
      "D_n",
      "b_q",
      "s_Qacc"
    ],
    "M_live[vreg:kv_proj]": [
      "D_v",
      "b_q",
      "s_A",
      "s_state"
    ],
    "M_live[vreg:score]": [
      "D_v",
      "b_k",
      "b_q",
      "s_A",
      "s_E",
      "s_X",
      "s_state"
    ],
    "M_live[vreg:softmax]": [
      "D_v",
      "b_k",
      "b_q",
      "s_A",
      "s_E",
      "s_X",
      "s_state"
    ],
    "M_live[vreg:pv]": [
      "D_v",
      "b_k",
      "b_q",
      "s_A",
      "s_E",
      "s_P",
      "s_X",
      "s_state"
    ],
    "M_live[vreg:finalize]": [
      "D_v",
      "b_q",
      "s_A",
      "s_state"
    ],
    "M_peak[vreg]": [
      "D_n",
      "D_v",
      "b_k",
      "b_q",
      "s_A",
      "s_E",
      "s_P",
      "s_Qacc",
      "s_X",
      "s_state"
    ],
    "M_live[vmem:q_proj]": [
      "D_n",
      "R_q",
      "b_q",
      "s_Cq",
      "s_Wq",
      "s_scratch"
    ],
    "M_live[vmem:kv_proj]": [
      "D_n",
      "D_r",
      "D_v",
      "R_k",
      "b_k",
      "b_q",
      "n_buf",
      "s_Ck",
      "s_Kn",
      "s_Qn",
      "s_Qr",
      "s_V",
      "s_Wk",
      "s_Wv",
      "s_scratch"
    ],
    "M_live[vmem:score]": [
      "D_n",
      "D_r",
      "D_v",
      "b_k",
      "b_q",
      "n_buf",
      "s_Kn",
      "s_Kr",
      "s_Qn",
      "s_Qr",
      "s_V",
      "s_scratch"
    ],
    "M_live[vmem:softmax]": [
      "D_n",
      "D_r",
      "D_v",
      "b_k",
      "b_q",
      "s_Qn",
      "s_Qr",
      "s_V",
      "s_scratch"
    ],
    "M_live[vmem:pv]": [
      "D_n",
      "D_r",
      "D_v",
      "b_k",
      "b_q",
      "s_Qn",
      "s_Qr",
      "s_V",
      "s_scratch"
    ],
    "M_live[vmem:finalize]": [
      "D_n",
      "D_r",
      "D_v",
      "b_q",
      "s_LSE",
      "s_O",
      "s_Qn",
      "s_Qr",
      "s_scratch"
    ],
    "M_peak[vmem]": [
      "D_n",
      "D_r",
      "D_v",
      "R_k",
      "R_q",
      "b_k",
      "b_q",
      "n_buf",
      "s_Ck",
      "s_Cq",
      "s_Kn",
      "s_Kr",
      "s_LSE",
      "s_O",
      "s_Qn",
      "s_Qr",
      "s_V",
      "s_Wk",
      "s_Wq",
      "s_Wv",
      "s_scratch"
    ],
    "envelope[all:q_proj]": [
      "D_n",
      "R_q",
      "b_q",
      "s_Cq",
      "s_Qacc",
      "s_Wq",
      "s_scratch"
    ],
    "envelope[all:kv_proj]": [
      "D_n",
      "D_r",
      "D_v",
      "R_k",
      "b_k",
      "b_q",
      "n_buf",
      "s_A",
      "s_Ck",
      "s_Kn",
      "s_Qn",
      "s_Qr",
      "s_V",
      "s_Wk",
      "s_Wv",
      "s_scratch",
      "s_state"
    ],
    "envelope[all:score]": [
      "D_n",
      "D_r",
      "D_v",
      "b_k",
      "b_q",
      "n_buf",
      "s_A",
      "s_E",
      "s_Kn",
      "s_Kr",
      "s_Qn",
      "s_Qr",
      "s_V",
      "s_X",
      "s_scratch",
      "s_state"
    ],
    "envelope[all:softmax]": [
      "D_n",
      "D_r",
      "D_v",
      "b_k",
      "b_q",
      "s_A",
      "s_E",
      "s_Qn",
      "s_Qr",
      "s_V",
      "s_X",
      "s_scratch",
      "s_state"
    ],
    "envelope[all:pv]": [
      "D_n",
      "D_r",
      "D_v",
      "b_k",
      "b_q",
      "s_A",
      "s_E",
      "s_P",
      "s_Qn",
      "s_Qr",
      "s_V",
      "s_X",
      "s_scratch",
      "s_state"
    ],
    "envelope[all:finalize]": [
      "D_n",
      "D_r",
      "D_v",
      "b_q",
      "s_A",
      "s_LSE",
      "s_O",
      "s_Qn",
      "s_Qr",
      "s_scratch",
      "s_state"
    ],
    "envelope[vreg:q_proj]": [
      "D_n",
      "b_q",
      "s_Qacc"
    ],
    "envelope[vreg:kv_proj]": [
      "D_v",
      "b_q",
      "s_A",
      "s_state"
    ],
    "envelope[vreg:score]": [
      "D_v",
      "b_k",
      "b_q",
      "s_A",
      "s_E",
      "s_X",
      "s_state"
    ],
    "envelope[vreg:softmax]": [
      "D_v",
      "b_k",
      "b_q",
      "s_A",
      "s_E",
      "s_X",
      "s_state"
    ],
    "envelope[vreg:pv]": [
      "D_v",
      "b_k",
      "b_q",
      "s_A",
      "s_E",
      "s_P",
      "s_X",
      "s_state"
    ],
    "envelope[vreg:finalize]": [
      "D_v",
      "b_q",
      "s_A",
      "s_state"
    ],
    "envelope[vmem:q_proj]": [
      "D_n",
      "R_q",
      "b_q",
      "s_Cq",
      "s_Wq",
      "s_scratch"
    ],
    "envelope[vmem:kv_proj]": [
      "D_n",
      "D_r",
      "D_v",
      "R_k",
      "b_k",
      "b_q",
      "n_buf",
      "s_Ck",
      "s_Kn",
      "s_Qn",
      "s_Qr",
      "s_V",
      "s_Wk",
      "s_Wv",
      "s_scratch"
    ],
    "envelope[vmem:score]": [
      "D_n",
      "D_r",
      "D_v",
      "b_k",
      "b_q",
      "n_buf",
      "s_Kn",
      "s_Kr",
      "s_Qn",
      "s_Qr",
      "s_V",
      "s_scratch"
    ],
    "envelope[vmem:softmax]": [
      "D_n",
      "D_r",
      "D_v",
      "b_k",
      "b_q",
      "s_Qn",
      "s_Qr",
      "s_V",
      "s_scratch"
    ],
    "envelope[vmem:pv]": [
      "D_n",
      "D_r",
      "D_v",
      "b_k",
      "b_q",
      "s_Qn",
      "s_Qr",
      "s_V",
      "s_scratch"
    ],
    "envelope[vmem:finalize]": [
      "D_n",
      "D_r",
      "D_v",
      "b_q",
      "s_LSE",
      "s_O",
      "s_Qn",
      "s_Qr",
      "s_scratch"
    ],
    "excess_over_budget[vreg:q_proj]": [
      "D_n",
      "R_vreg",
      "b_q",
      "s_Qacc"
    ],
    "excess_over_budget[vreg:kv_proj]": [
      "D_v",
      "R_vreg",
      "b_q",
      "s_A",
      "s_state"
    ],
    "b_k_max[vreg:score]": [
      "D_v",
      "R_vreg",
      "b_q",
      "s_A",
      "s_E",
      "s_X",
      "s_state"
    ],
    "excess_over_budget[vreg:score]": [
      "D_v",
      "R_vreg",
      "b_k",
      "b_q",
      "s_A",
      "s_E",
      "s_X",
      "s_state"
    ],
    "b_k_max[vreg:softmax]": [
      "D_v",
      "R_vreg",
      "b_q",
      "s_A",
      "s_E",
      "s_X",
      "s_state"
    ],
    "excess_over_budget[vreg:softmax]": [
      "D_v",
      "R_vreg",
      "b_k",
      "b_q",
      "s_A",
      "s_E",
      "s_X",
      "s_state"
    ],
    "b_k_max[vreg:pv]": [
      "D_v",
      "R_vreg",
      "b_q",
      "s_A",
      "s_E",
      "s_P",
      "s_X",
      "s_state"
    ],
    "excess_over_budget[vreg:pv]": [
      "D_v",
      "R_vreg",
      "b_k",
      "b_q",
      "s_A",
      "s_E",
      "s_P",
      "s_X",
      "s_state"
    ],
    "excess_over_budget[vreg:finalize]": [
      "D_v",
      "R_vreg",
      "b_q",
      "s_A",
      "s_state"
    ],
    "excess_over_budget[vmem:q_proj]": [
      "D_n",
      "R_q",
      "R_vmem",
      "b_q",
      "s_Cq",
      "s_Wq",
      "s_scratch"
    ],
    "b_k_max[vmem:kv_proj]": [
      "D_n",
      "D_r",
      "D_v",
      "R_k",
      "R_vmem",
      "b_q",
      "n_buf",
      "s_Ck",
      "s_Kn",
      "s_Qn",
      "s_Qr",
      "s_V",
      "s_Wk",
      "s_Wv",
      "s_scratch"
    ],
    "excess_over_budget[vmem:kv_proj]": [
      "D_n",
      "D_r",
      "D_v",
      "R_k",
      "R_vmem",
      "b_k",
      "b_q",
      "n_buf",
      "s_Ck",
      "s_Kn",
      "s_Qn",
      "s_Qr",
      "s_V",
      "s_Wk",
      "s_Wv",
      "s_scratch"
    ],
    "b_k_max[vmem:score]": [
      "D_n",
      "D_r",
      "D_v",
      "R_vmem",
      "b_q",
      "n_buf",
      "s_Kn",
      "s_Kr",
      "s_Qn",
      "s_Qr",
      "s_V",
      "s_scratch"
    ],
    "excess_over_budget[vmem:score]": [
      "D_n",
      "D_r",
      "D_v",
      "R_vmem",
      "b_k",
      "b_q",
      "n_buf",
      "s_Kn",
      "s_Kr",
      "s_Qn",
      "s_Qr",
      "s_V",
      "s_scratch"
    ],
    "b_k_max[vmem:softmax]": [
      "D_n",
      "D_r",
      "D_v",
      "R_vmem",
      "b_q",
      "s_Qn",
      "s_Qr",
      "s_V",
      "s_scratch"
    ],
    "excess_over_budget[vmem:softmax]": [
      "D_n",
      "D_r",
      "D_v",
      "R_vmem",
      "b_k",
      "b_q",
      "s_Qn",
      "s_Qr",
      "s_V",
      "s_scratch"
    ],
    "b_k_max[vmem:pv]": [
      "D_n",
      "D_r",
      "D_v",
      "R_vmem",
      "b_q",
      "s_Qn",
      "s_Qr",
      "s_V",
      "s_scratch"
    ],
    "excess_over_budget[vmem:pv]": [
      "D_n",
      "D_r",
      "D_v",
      "R_vmem",
      "b_k",
      "b_q",
      "s_Qn",
      "s_Qr",
      "s_V",
      "s_scratch"
    ],
    "excess_over_budget[vmem:finalize]": [
      "D_n",
      "D_r",
      "D_v",
      "R_vmem",
      "b_q",
      "s_LSE",
      "s_O",
      "s_Qn",
      "s_Qr",
      "s_scratch"
    ],
    "d(score_bytes)/d(b_q)": [
      "b_k",
      "s_X"
    ],
    "d(score_bytes)/d(b_k)": [
      "b_q",
      "s_X"
    ],
    "d(accumulator_bytes)/d(b_q)": [
      "D_v",
      "s_A"
    ],
    "d(accumulator_bytes)/d(b_k)": [],
    "n_programs": [
      "B",
      "H",
      "S_q",
      "b_q",
      "h_pp"
    ],
    "n_kv_block_visits": [
      "B",
      "Delta",
      "H",
      "S_k",
      "S_q",
      "b_k",
      "b_q"
    ]
  },
  "matmul_nodes": [
    {
      "id": "Q_proj",
      "stage": "q_proj",
      "M": "B*H*S_q",
      "N": "D_n",
      "K": "R_q",
      "multiplicity": "1",
      "tile_shape": null,
      "tile_shape_value": null,
      "counts_toward": "projection"
    },
    {
      "id": "K_proj",
      "stage": "kv_proj",
      "M": "B*H*BlockStat(key_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)",
      "N": "D_n",
      "K": "R_k",
      "multiplicity": "1",
      "tile_shape": null,
      "tile_shape_value": null,
      "counts_toward": "projection"
    },
    {
      "id": "V_proj",
      "stage": "kv_proj",
      "M": "B*H*BlockStat(key_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)",
      "N": "D_v",
      "K": "R_k",
      "multiplicity": "1",
      "tile_shape": null,
      "tile_shape_value": null,
      "counts_toward": "projection"
    },
    {
      "id": "QK_nope",
      "stage": "score",
      "M": "B*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)",
      "N": "1",
      "K": "D_n",
      "multiplicity": "1",
      "tile_shape": [
        "b_q",
        "b_k",
        "D_n"
      ],
      "tile_shape_value": [
        16,
        16,
        24
      ],
      "counts_toward": "attention"
    },
    {
      "id": "QK_rope",
      "stage": "score",
      "M": "B*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)",
      "N": "1",
      "K": "D_r",
      "multiplicity": "1",
      "tile_shape": [
        "b_q",
        "b_k",
        "D_r"
      ],
      "tile_shape_value": [
        16,
        16,
        8
      ],
      "counts_toward": "attention"
    },
    {
      "id": "PV",
      "stage": "pv",
      "M": "B*H*BlockStat(rect_cells, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)",
      "N": "1",
      "K": "D_v",
      "multiplicity": "1",
      "tile_shape": [
        "b_q",
        "D_v",
        "b_k"
      ],
      "tile_shape_value": [
        16,
        24,
        16
      ],
      "counts_toward": "attention"
    }
  ],
  "local_objects": [
    {
      "id": "q_latent_window",
      "size": "R_q*b_q*s_Cq",
      "level": "vmem",
      "live_stages": [
        "q_proj"
      ],
      "alias_group": null,
      "exists": true,
      "conditional_on": null,
      "size_scenario": "b_rq",
      "stage_scenario": null
    },
    {
      "id": "q_latent_rank_tile",
      "size": "R_q*b_q*s_Cq",
      "level": "vmem",
      "live_stages": [
        "q_proj"
      ],
      "alias_group": null,
      "exists": false,
      "conditional_on": null,
      "size_scenario": null,
      "stage_scenario": null
    },
    {
      "id": "wq_rank_tile",
      "size": "D_n*R_q*s_Wq",
      "level": "vmem",
      "live_stages": [
        "q_proj"
      ],
      "alias_group": null,
      "exists": true,
      "conditional_on": null,
      "size_scenario": "b_rq",
      "stage_scenario": null
    },
    {
      "id": "q_proj_acc",
      "size": "D_n*b_q*s_Qacc",
      "level": "vreg",
      "live_stages": [
        "q_proj"
      ],
      "alias_group": null,
      "exists": true,
      "conditional_on": null,
      "size_scenario": null,
      "stage_scenario": null
    },
    {
      "id": "q_nope_operand",
      "size": "D_n*b_q*s_Qn",
      "level": "vmem",
      "live_stages": [
        "kv_proj",
        "score",
        "softmax",
        "pv",
        "finalize"
      ],
      "alias_group": null,
      "exists": true,
      "conditional_on": null,
      "size_scenario": null,
      "stage_scenario": "q_resident"
    },
    {
      "id": "kv_latent_rank_tile",
      "size": "R_k*b_k*n_buf*s_Ck",
      "level": "vmem",
      "live_stages": [
        "kv_proj"
      ],
      "alias_group": null,
      "exists": true,
      "conditional_on": null,
      "size_scenario": "b_rk",
      "stage_scenario": null
    },
    {
      "id": "wk_rank_tile",
      "size": "D_n*R_k*s_Wk",
      "level": "vmem",
      "live_stages": [
        "kv_proj"
      ],
      "alias_group": null,
      "exists": true,
      "conditional_on": null,
      "size_scenario": "b_rk",
      "stage_scenario": null
    },
    {
      "id": "wv_rank_tile",
      "size": "D_v*R_k*s_Wv",
      "level": "vmem",
      "live_stages": [
        "kv_proj"
      ],
      "alias_group": null,
      "exists": true,
      "conditional_on": null,
      "size_scenario": "b_rk",
      "stage_scenario": null
    },
    {
      "id": "k_nope_tile",
      "size": "D_n*b_k*s_Kn",
      "level": "vmem",
      "live_stages": [
        "kv_proj",
        "score"
      ],
      "alias_group": null,
      "exists": true,
      "conditional_on": null,
      "size_scenario": null,
      "stage_scenario": null
    },
    {
      "id": "v_tile",
      "size": "D_v*b_k*s_V",
      "level": "vmem",
      "live_stages": [
        "kv_proj",
        "score",
        "softmax",
        "pv"
      ],
      "alias_group": null,
      "exists": true,
      "conditional_on": null,
      "size_scenario": null,
      "stage_scenario": "v_load"
    },
    {
      "id": "score_X",
      "size": "b_k*b_q*s_X",
      "level": "vreg",
      "live_stages": [
        "score",
        "softmax"
      ],
      "alias_group": "score_or_E",
      "exists": true,
      "conditional_on": null,
      "size_scenario": null,
      "stage_scenario": null
    },
    {
      "id": "exp_E",
      "size": "b_k*b_q*s_E",
      "level": "vreg",
      "live_stages": [
        "softmax",
        "pv"
      ],
      "alias_group": "score_or_E",
      "exists": true,
      "conditional_on": null,
      "size_scenario": null,
      "stage_scenario": null
    },
    {
      "id": "p_operand",
      "size": "b_k*b_q*s_P",
      "level": "vreg",
      "live_stages": [
        "pv"
      ],
      "alias_group": null,
      "exists": true,
      "conditional_on": "exp_alias_p_operand",
      "size_scenario": null,
      "stage_scenario": null
    },
    {
      "id": "row_state_m",
      "size": "b_q*s_state",
      "level": "vreg",
      "live_stages": [
        "kv_proj",
        "score",
        "softmax",
        "pv",
        "finalize"
      ],
      "alias_group": null,
      "exists": true,
      "conditional_on": null,
      "size_scenario": null,
      "stage_scenario": null
    },
    {
      "id": "row_state_l",
      "size": "b_q*s_state",
      "level": "vreg",
      "live_stages": [
        "kv_proj",
        "score",
        "softmax",
        "pv",
        "finalize"
      ],
      "alias_group": null,
      "exists": true,
      "conditional_on": null,
      "size_scenario": null,
      "stage_scenario": null
    },
    {
      "id": "row_alpha",
      "size": "b_q*s_state",
      "level": "vreg",
      "live_stages": [
        "softmax",
        "pv"
      ],
      "alias_group": null,
      "exists": true,
      "conditional_on": null,
      "size_scenario": null,
      "stage_scenario": null
    },
    {
      "id": "accumulator_A",
      "size": "D_v*b_q*s_A",
      "level": "vreg",
      "live_stages": [
        "kv_proj",
        "score",
        "softmax",
        "pv",
        "finalize"
      ],
      "alias_group": null,
      "exists": true,
      "conditional_on": null,
      "size_scenario": null,
      "stage_scenario": null
    },
    {
      "id": "q_pe_tile",
      "size": "D_r*b_q*s_Qr",
      "level": "vmem",
      "live_stages": [
        "kv_proj",
        "score",
        "softmax",
        "pv",
        "finalize"
      ],
      "alias_group": null,
      "exists": true,
      "conditional_on": null,
      "size_scenario": null,
      "stage_scenario": "q_resident"
    },
    {
      "id": "k_pe_tile",
      "size": "D_r*b_k*n_buf*s_Kr",
      "level": "vmem",
      "live_stages": [
        "score"
      ],
      "alias_group": null,
      "exists": true,
      "conditional_on": null,
      "size_scenario": null,
      "stage_scenario": null
    },
    {
      "id": "fixed_scratch",
      "size": "s_scratch",
      "level": "vmem",
      "live_stages": [
        "q_proj",
        "kv_proj",
        "score",
        "softmax",
        "pv",
        "finalize"
      ],
      "alias_group": null,
      "exists": true,
      "conditional_on": "scratch_bytes",
      "size_scenario": null,
      "stage_scenario": null
    },
    {
      "id": "out_tile",
      "size": "D_v*b_q*s_O",
      "level": "vmem",
      "live_stages": [
        "finalize"
      ],
      "alias_group": null,
      "exists": true,
      "conditional_on": null,
      "size_scenario": null,
      "stage_scenario": null
    },
    {
      "id": "lse_tile",
      "size": "b_q*s_LSE",
      "level": "vmem",
      "live_stages": [
        "finalize"
      ],
      "alias_group": null,
      "exists": true,
      "conditional_on": "outputs",
      "size_scenario": null,
      "stage_scenario": null
    }
  ],
  "transfer_events": [
    {
      "id": "Cq_read",
      "path": "hbm_to_vmem",
      "total": "B*R_q*S_q*s_Cq*ceiling(H/h_pp)",
      "undeclared": false
    },
    {
      "id": "Qr_read",
      "path": "hbm_to_vmem",
      "total": "B*D_r*H*S_q*s_Qr",
      "undeclared": false
    },
    {
      "id": "Wq_read",
      "path": "hbm_to_vmem",
      "total": "B*D_n*H*R_q*s_Wq*ceiling(S_q/b_q)",
      "undeclared": false
    },
    {
      "id": "Kr_read",
      "path": "hbm_to_vmem",
      "total": "B*D_r*s_Kr*ceiling(H/h_pp)*BlockStat(key_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)",
      "undeclared": false
    },
    {
      "id": "Ck_read",
      "path": "hbm_to_vmem",
      "total": "B*R_k*s_Ck*ceiling(H/h_pp)*BlockStat(key_visits, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)",
      "undeclared": false
    },
    {
      "id": "Wk_read",
      "path": "hbm_to_vmem",
      "total": "B*D_n*H*R_k*s_Wk*ceiling(S_q/b_q)",
      "undeclared": false
    },
    {
      "id": "Wv_read",
      "path": "hbm_to_vmem",
      "total": "B*D_v*H*R_k*s_Wv*ceiling(S_q/b_q)",
      "undeclared": false
    },
    {
      "id": "O_write",
      "path": "vmem_to_hbm",
      "total": "B*D_v*H*S_q*s_O",
      "undeclared": false
    },
    {
      "id": "LSE_write",
      "path": "vmem_to_hbm",
      "total": "B*H*S_q*s_LSE",
      "undeclared": false
    },
    {
      "id": "stream_Qn_operand",
      "path": "vmem_to_vreg",
      "total": "B*D_n*H*b_q*s_Qn*ceiling(S_q/b_q)",
      "undeclared": true
    },
    {
      "id": "stream_Qr_operand",
      "path": "vmem_to_vreg",
      "total": "B*D_r*H*b_q*s_Qr*ceiling(S_q/b_q)",
      "undeclared": true
    },
    {
      "id": "stream_K_tile",
      "path": "vmem_to_vreg",
      "total": "B*D_n*H*b_k*s_Kn*BlockStat(selected, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)",
      "undeclared": true
    },
    {
      "id": "stream_V_tile",
      "path": "vmem_to_vreg",
      "total": "B*D_v*H*b_k*s_V*BlockStat(selected, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)",
      "undeclared": true
    },
    {
      "id": "stream_Kr_tile",
      "path": "vmem_to_vreg",
      "total": "B*D_r*H*b_k*s_Kr*BlockStat(selected, causal, S_q, S_k, b_q, b_k, Delta, skip_future, 0, 0, S_q, S_k)",
      "undeclared": true
    }
  ],
  "stages": [
    "q_proj",
    "kv_proj",
    "score",
    "softmax",
    "pv",
    "finalize"
  ]
}
```
