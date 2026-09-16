# MLA forward analytical report (absorbed_two_step, strategy=bq16_bk32)

- mode: **bound**  |  adapter: `seven_input_latent`  |  algorithm: `absorbed_two_step`
- task fingerprint: `4c8a371c71a6b4a3`

## Bindings and declarations

- **bindings**: `{"B": 3, "R_q": 16, "R_k": 32, "H": 4, "D_n": 24, "D_r": 8, "D_v": 24, "s_Cq": 2, "s_Ck": 2, "s_Wq": 2, "s_Wk": 2, "s_Wv": 2, "s_Qr": 2, "s_Kr": 2, "s_X": 4,...`
- **dimension_provenance**: `{"B": "tensor.q_latent.axis[B]", "Sq": "tensor.q_latent.axis[Sq]", "Rq": "tensor.q_latent.axis[Rq]", "Sk": "tensor.kv_latent.axis[Sk]", "Rk": "tensor.kv_late...`
- **cross_checks**: `["tensor.kv_latent.axis[B] == tensor.q_latent.axis[B] == 3", "tensor.q_pe.axis[B] == tensor.q_latent.axis[B] == 3", "tensor.k_pe.axis[B] == tensor.q_latent.a...`
- **capacity_extents**: `{"Sq": 48, "Sk": 96}`
- **active_extents**: `{"Sq": [48, 20, 7], "Sk": [96, 20, 50], "declared": true, "declared_axes": ["Sk", "Sq"], "undeclared_axes": []}`
- **dtypes**: `{"q_latent": "bfloat16", "kv_latent": "bfloat16", "w_q_nope": "bfloat16", "w_k_nope": "bfloat16", "w_v": "bfloat16", "q_pe": "bfloat16", "k_pe": "bfloat16"}`
- **semantics**: `{"mask": "causal", "position_offset": null, "query_position_start": null, "key_position_start": null, "scale_policy": "standard", "scale_value": null, "scale...`
- **implementation**: `{"name": "bq16_bk32", "b_q": 16, "b_k": 32, "b_rq": null, "b_rk": null, "scheduled_lengths": null, "rect_policy": "skip_future", "subdivide": null, "executed...`
- **numerics**: `{"matmul_input_dtype": "bfloat16", "accumulator_dtype": "float32", "score_dtype": "float32", "exp_dtype": "float32", "p_operand_dtype": "bfloat16", "state_dt...`
- **numerics_assumptions**: `["projected_operand_dtype undeclared: Q^n/K^n/V/Q~/Z operand widths (s_Qn, s_Kn, s_V, s_Qt, s_Z) assumed = matmul_input_dtype", "projection_accumulator_dtype...`
- **hardware**: `null`

## Unresolved inputs

- CP: needed by 2 metric(s), e.g. ['T_LB[critical_path]', 'T_LB[combined]']
- L_l: needed by 20 metric(s), e.g. ['layout[q_latent_window]', 'layout[wq_rank_tile]', 'layout[q_proj_acc]']
- L_s: needed by 20 metric(s), e.g. ['layout[q_latent_window]', 'layout[wq_rank_tile]', 'layout[q_proj_acc]']
- P_exp: needed by 2 metric(s), e.g. ['T_LB[exp]', 'T_LB[combined]']
- P_layout: needed by 2 metric(s), e.g. ['T_LB[layout]', 'T_LB[combined]']
- P_mxu: needed by 2 metric(s), e.g. ['T_LB[mxu]', 'T_LB[combined]']
- P_red: needed by 2 metric(s), e.g. ['T_LB[reduce]', 'T_LB[combined]']
- P_vpu: needed by 2 metric(s), e.g. ['T_LB[vpu]', 'T_LB[combined]']
- W_hbm: needed by 2 metric(s), e.g. ['T_LB[path:hbm_to_vmem]', 'T_LB[combined]']
- W_hbm_w: needed by 2 metric(s), e.g. ['T_LB[path:vmem_to_hbm]', 'T_LB[combined]']
- W_vmem: needed by 2 metric(s), e.g. ['T_LB[path:vmem_to_vreg]', 'T_LB[combined]']
- acceptance: needed by 1 metric(s), e.g. ['acceptance[declared]']
- allowed_transformations: needed by 3 metric(s), e.g. ['B_extra_opt', 'T_opt', 'B_extra_opt_under_time_budget']
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
- layout.k_pe_tile: needed by 1 metric(s), e.g. ['layout[k_pe_tile]']
- layout.kv_latent_tile: needed by 1 metric(s), e.g. ['layout[kv_latent_tile]']
- layout.lse_tile: needed by 1 metric(s), e.g. ['layout[lse_tile]']
- layout.out_tile: needed by 1 metric(s), e.g. ['layout[out_tile]']
- layout.p_operand: needed by 1 metric(s), e.g. ['layout[p_operand]']
- layout.q_latent_window: needed by 1 metric(s), e.g. ['layout[q_latent_window]']
- layout.q_pe_tile: needed by 1 metric(s), e.g. ['layout[q_pe_tile]']
- layout.q_proj_acc: needed by 1 metric(s), e.g. ['layout[q_proj_acc]']
- layout.q_tilde: needed by 1 metric(s), e.g. ['layout[q_tilde]']
- layout.q_tilde_acc: needed by 1 metric(s), e.g. ['layout[q_tilde_acc]']
- layout.row_alpha: needed by 1 metric(s), e.g. ['layout[row_alpha]']
- layout.row_state_l: needed by 1 metric(s), e.g. ['layout[row_state_l]']
- layout.row_state_m: needed by 1 metric(s), e.g. ['layout[row_state_m]']
- layout.score_X: needed by 1 metric(s), e.g. ['layout[score_X]']
- layout.wk_full: needed by 1 metric(s), e.g. ['layout[wk_full]']
- layout.wq_rank_tile: needed by 1 metric(s), e.g. ['layout[wq_rank_tile]']
- layout.wv_full: needed by 1 metric(s), e.g. ['layout[wv_full]']
- layout.z_tile: needed by 1 metric(s), e.g. ['layout[z_tile]']
- machine_model: needed by 3 metric(s), e.g. ['B_extra_opt', 'T_opt', 'B_extra_opt_under_time_budget']
- objective: needed by 1 metric(s), e.g. ['B_extra_opt']
- operand_transpose_materialized (undeclared backend behaviour): needed by 2 metric(s), e.g. ['W_vec[transpose]', 'W_resource[layout]']
- projected_operand_dtype: needed by 2 metric(s), e.g. ['local[q_tilde]', 'local[z_tile]']
- projection_accumulator_dtype: needed by 2 metric(s), e.g. ['local[q_proj_acc]', 'local[q_tilde_acc]']
- q_resident: needed by 26 metric(s), e.g. ['B[vmem_to_vreg]', 'local[q_tilde]', 'local[q_pe_tile]']
- register_schedule_evidence: needed by 4 metric(s), e.g. ['B[vmem_to_vreg]', 'B[vreg_to_vmem]', 'T_LB[path:vmem_to_vreg]']
- schedule: needed by 33 metric(s), e.g. ['F_total[executed_graph]', 'F_projection[executed_graph]', 'F_attention[executed_graph]']
- schedule_model: needed by 2 metric(s), e.g. ['T_opt', 'B_extra_opt_under_time_budget']
- time_budget_tau: needed by 1 metric(s), e.g. ['B_extra_opt_under_time_budget']
- vmem_budget_bytes: needed by 10 metric(s), e.g. ['excess_over_budget[vmem:q_proj]', 'excess_over_budget[vmem:q_absorb]', 'b_k_max[vmem:score]']
- vreg_budget_bytes: needed by 10 metric(s), e.g. ['excess_over_budget[vreg:q_proj]', 'excess_over_budget[vreg:q_absorb]', 'b_k_max[vreg:score]']

## Warnings

- schedule not declared: the q_outer_kv_inner loop nest is shown as the scenario; transfer counts and the program count follow from it
- rank sub-tile undeclared: the full R_q window is shown as the scenario
- q_resident not declared: Q operands held across the KV loop shown as the scenario
- final_normalize not declared: the n*D_A division form is the scenario; the reciprocal-plus-multiply form of spec 5.5 is the alternative, not an addition
- cast_points not declared: the cast graph is inferred from dtype differences at the two modelled sites (exp_to_p, accumulator_to_output); a declared site list gives a different graph
- projected_operand_dtype undeclared: Q^n/K^n/V/Q~/Z operand widths (s_Qn, s_Kn, s_V, s_Qt, s_Z) assumed = matmul_input_dtype
- projection_accumulator_dtype undeclared: projection accumulator width (s_Qacc) assumed = accumulator_dtype

## 1. task and scope

| metric | formula id | expression | value | unit | status | missing | scope | not equivalent to |
|---|---|---|---|---|---|---|---|---|
| B | `task.dim.B` | `B` | 3 | count | bound |  | current invocation |  |
| H | `task.dim.H` | `H` | 4 | count | bound |  | current invocation |  |
| R_q | `task.dim.R_q` | `R_q` | 16 | count | bound |  | current invocation |  |
| R_k | `task.dim.R_k` | `R_k` | 32 | count | bound |  | current invocation |  |
| D_n | `task.dim.D_n` | `D_n` | 24 | count | bound |  | current invocation |  |
| D_r | `task.dim.D_r` | `D_r` | 8 | count | bound |  | current invocation |  |
| D_v | `task.dim.D_v` | `D_v` | 24 | count | bound |  | current invocation |  |
| S_q_active | `task.active.S_q` | `per-batch list S_q_b` | [48, 20, 7] | count | bound |  | active query rows (mathematical work) (ragged: one entry per batch) |  |
| S_k_active | `task.active.S_k` | `per-batch list S_k_b` | [96, 20, 50] | count | bound |  | active keys consumed in this call (ragged: one entry per batch) |  |
| S_q_scheduled | `task.scheduled.S_q` | `not modelled for per-batch tasks` |  | count | not_applicable |  | extent the kernel iterates over | active extent; allocated capacity |
| S_k_scheduled | `task.scheduled.S_k` | `not modelled for per-batch tasks` |  | count | not_applicable |  | extent the kernel iterates over | active extent; allocated capacity |
| S_q_capacity | `task.capacity.S_q` | `allocated extent from tensor shape` | 48 | count | bound |  | allocated query extent | active rows; scheduled rows |
| S_k_capacity | `task.capacity.S_k` | `allocated extent from tensor shape` | 96 | count | bound |  | allocated KV extent (not necessarily consumed) | active keys; scheduled keys |
| Delta | `task.offset` | `per-batch list Delta_b` | [48, 0, 43] | positions | bound |  | key j visible to query i iff j <= i + Delta_b |  |
| mask | `task.mask` | `declared mask kind` | causal |  | bound |  | visibility semantics |  |
| gamma | `task.scale` | `1/sqrt(D_n + D_r)` | 0.176777 | 1 | bound |  | score scale |  |
| outputs | `task.outputs` | `declared output scope` | ['O', 'LSE'] |  | bound |  | output scope |  |
| lse_convention | `task.lse_convention` | `declared` | natural_log |  | bound |  | LSE log base |  |
| projection_scope | `task.projection_scope` | `declared` | full |  | bound |  | which rows are projected in this call |  |
| rows_without_visible_keys | `task.empty_rows` | `0` | 0 | rows | bound |  | (b,h,i) rows whose softmax is undefined |  |

Assumptions:
- `S_q_scheduled`: scheduled extents are not modelled for per-batch (ragged) tasks: each batch's grid spans its own active extent
- `S_k_scheduled`: scheduled extents are not modelled for per-batch (ragged) tasks: each batch's grid spans its own active extent
- `mask`: mask is declared, never inferred from shapes
- `gamma`: scale policy 'standard': gamma = (D_n + D_r)^(-1/2)
- `rows_without_visible_keys`: uniform lengths: B*H*max(0, min(S_q, -Delta))

<details><summary>Provenance (bindings, coverage, source, evidence)</summary>

| metric | source | coverage | bindings | evidence |
|---|---|---|---|---|
| B | input | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"B": 3}` | tensor.q_latent.axis[B] |
| H | input | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"H": 4}` | tensor.w_q_nope.axis[H] |
| R_q | input | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"R_q": 16}` | tensor.q_latent.axis[Rq] |
| R_k | input | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"R_k": 32}` | tensor.kv_latent.axis[Rk] |
| D_n | input | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"D_n": 24}` | tensor.w_q_nope.axis[Dn] |
| D_r | input | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"D_r": 8}` | tensor.q_pe.axis[Dr] |
| D_v | input | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"D_v": 24}` | tensor.w_v.axis[Dv] |
| S_q_active | input | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{}` |  |
| S_k_active | input | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{}` |  |
| S_q_scheduled | input | not applicable under the declared configuration | `{}` |  |
| S_k_scheduled | input | not applicable under the declared configuration | `{}` |  |
| S_q_capacity | input | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{}` |  |
| S_k_capacity | input | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{}` |  |
| Delta | input | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{}` |  |
| mask | input | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{}` |  |
| gamma | input | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"D_n": 24, "D_r": 8}` |  |
| outputs | input | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{}` |  |
| lse_convention | input | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{}` |  |
| projection_scope | input | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{}` |  |
| rows_without_visible_keys | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{}` |  |

</details>

## 2. visibility and rectangles

| metric | formula id | expression | value | unit | status | missing | scope | not equivalent to |
|---|---|---|---|---|---|---|---|---|
| C_valid | `vis.C_valid` | `4019*H` | 16,076 | cells | bound |  | all (b,h,i,j) with mu=1; includes B and H | exp instruction count; executed cells |
| C_rect | `vis.C_rect` | `H*(BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96))` | 19,384 | cells | bound |  | sum over selected rectangles of n_i*m_j (logical extents) | C_valid; C_pad under a padded execution extent |
| C_pad | `vis.C_pad` | `H*(BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96))` | 19,384 | cells | bound |  | sum over selected rectangles of n_i*m_j: the declared executed extents are the logical ones | hardware array padding (must come from compilation evidence) |
| rect_waste | `vis.waste` | `-4019*H + H*(BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96))` | 3308 | cells | bound |  | C_rect - C_valid under the declared policy |  |
| selected_blocks | `vis.blocks.selected` | `H*(BlockStat(selected, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + BlockStat(selected, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + BlockStat(selected, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96))` | 48 | blocks | bound |  | over all (b,h) |  |
| partial_blocks | `vis.blocks.partial` | `H*(BlockStat(partial, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + BlockStat(partial, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + BlockStat(partial, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96))` | 24 | blocks | bound |  | over all (b,h) |  |
| fully_visible_blocks | `vis.blocks.full` | `H*(BlockStat(full, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + BlockStat(full, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + BlockStat(full, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96))` | 24 | blocks | bound |  | over all (b,h) |  |
| future_blocks_selected | `vis.blocks.future_selected` | `H*(BlockStat(future_selected, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + BlockStat(future_selected, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + BlockStat(future_selected, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96))` | 0 | blocks | bound |  | over all (b,h) |  |
| padding_blocks_selected | `vis.blocks.padding_selected` | `H*(BlockStat(padding_selected, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + BlockStat(padding_selected, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + BlockStat(padding_selected, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96))` | 0 | blocks | bound |  | over all (b,h) |  |
| future_blocks_skipped | `vis.blocks.future_skipped` | `H*(BlockStat(future_skipped, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + BlockStat(future_skipped, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + BlockStat(future_skipped, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96))` | 4 | blocks | bound |  | over all (b,h) |  |
| padding_blocks_skipped | `vis.blocks.padding_skipped` | `H*(BlockStat(padding_skipped, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + BlockStat(padding_skipped, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + BlockStat(padding_skipped, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96))` | 0 | blocks | bound |  | over all (b,h) |  |
| masked_cells | `vis.blocks.masked_cells` | `H*(BlockStat(masked_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + BlockStat(masked_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + BlockStat(masked_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96))` | 8248 | cells | bound |  | over all (b,h) |  |
| executed_extent_policy | `vis.exec_policy` | `declared` | logical |  | bound |  | n_hat, m_hat source |  |

Assumptions:
- `C_valid`: explicit finite sum over batches
- `C_rect`: rectangle selection policy 'skip_future'
- `C_rect`: tail blocks use logical extents
- `C_pad`: rectangle selection policy 'skip_future'
- `C_pad`: executed extents = logical extents (declared 'logical')
- `selected_blocks`: rectangle selection policy 'skip_future'
- `partial_blocks`: rectangle selection policy 'skip_future'
- `fully_visible_blocks`: rectangle selection policy 'skip_future'
- `future_blocks_selected`: rectangle selection policy 'skip_future'
- `padding_blocks_selected`: rectangle selection policy 'skip_future'
- `future_blocks_skipped`: rectangle selection policy 'skip_future'
- `padding_blocks_skipped`: rectangle selection policy 'skip_future'
- `masked_cells`: rectangle selection policy 'skip_future'
- `executed_extent_policy`: executed extents come from the declared strategy or compilation evidence, never from the hardware array size

<details><summary>Provenance (bindings, coverage, source, evidence)</summary>

| metric | source | coverage | bindings | evidence |
|---|---|---|---|---|
| C_valid | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"H": 4}` |  |
| C_rect | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"H": 4, "b_k": 32, "b_q": 16}` |  |
| C_pad | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"H": 4, "b_k": 32, "b_q": 16}` |  |
| rect_waste | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"H": 4, "b_k": 32, "b_q": 16}` |  |
| selected_blocks | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"H": 4, "b_k": 32, "b_q": 16}` |  |
| partial_blocks | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"H": 4, "b_k": 32, "b_q": 16}` |  |
| fully_visible_blocks | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"H": 4, "b_k": 32, "b_q": 16}` |  |
| future_blocks_selected | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"H": 4, "b_k": 32, "b_q": 16}` |  |
| padding_blocks_selected | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"H": 4, "b_k": 32, "b_q": 16}` |  |
| future_blocks_skipped | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"H": 4, "b_k": 32, "b_q": 16}` |  |
| padding_blocks_skipped | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"H": 4, "b_k": 32, "b_q": 16}` |  |
| masked_cells | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"H": 4, "b_k": 32, "b_q": 16}` |  |
| executed_extent_policy | input | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{}` |  |

</details>

## 3. work (FLOPs and vector operations)

| metric | formula id | expression | value | unit | status | missing | scope | not equivalent to |
|---|---|---|---|---|---|---|---|---|
| F_Q[useful] | `work.absorbed_two_step.F_Q` | `150*D_n*H*R_q` | 230,400 | FLOP | bound |  | useful (C = C_valid), FMA=2, projection rows from scope 'full' |  |
| F_Qtilde[useful] | `work.absorbed_two_step.F_Qtilde` | `150*D_n*H*R_k` | 460,800 | FLOP | bound |  | useful (C = C_valid), FMA=2, projection rows from scope 'full' |  |
| F_QC[useful] | `work.absorbed_two_step.F_QC` | `8038*H*R_k` | 1,028,864 | FLOP | bound |  | useful (C = C_valid), FMA=2, projection rows from scope 'full' |  |
| F_QK_r[useful] | `work.absorbed_two_step.F_QK_r` | `8038*D_r*H` | 257,216 | FLOP | bound |  | useful (C = C_valid), FMA=2, projection rows from scope 'full' |  |
| F_PC[useful] | `work.absorbed_two_step.F_PC` | `8038*H*R_k` | 1,028,864 | FLOP | bound |  | useful (C = C_valid), FMA=2, projection rows from scope 'full' |  |
| F_ZWv[useful] | `work.absorbed_two_step.F_ZWv` | `150*D_v*H*R_k` | 460,800 | FLOP | bound |  | useful (C = C_valid), FMA=2, projection rows from scope 'full' |  |
| F_total[useful] | `work.absorbed_two_step.total` | `150*D_n*H*R_k + 150*D_n*H*R_q + 150*D_v*H*R_k + 8038*H*(D_r + 2*R_k)` | 3,466,944 | FLOP | bound |  | useful mathematical work under projection scope 'full' | scheduled work; compiled instruction count |
| F_A[spec_closed_form] | `work.absorbed_two_step.closed_form` | `150*D_n*H*R_q + 150*H*R_k*(D_n + D_v) + 8038*H*(D_r + 2*R_k)` | 3,466,944 | FLOP | bound |  | specification boxed formula: every Q row and every K/V row projected once (full projection, no cache) |  |
| N_Qproj | `work.projection.rows.N_Qproj` | `75*H` | 300 | rows | bound |  | Q rows projected in this invocation counted once per row, from the declared projection_scope and the variant's dataflow | the tensor's row count; rows read from HBM; the executed-graph projected-row count under in-loop projection |
| N_Kproj | `work.projection.rows.N_Kproj` | `0` | 0 | rows | bound |  | K rows projected in this invocation counted once per row, from the declared projection_scope and the variant's dataflow | the tensor's row count; rows read from HBM; the executed-graph projected-row count under in-loop projection |
| N_Vproj | `work.projection.rows.N_Vproj` | `0` | 0 | rows | bound |  | V rows projected in this invocation counted once per row, from the declared projection_scope and the variant's dataflow | the tensor's row count; rows read from HBM; the executed-graph projected-row count under in-loop projection |
| F_proj[general] | `work.projection.general` | `150*D_n*H*R_k + 150*D_n*H*R_q` | 691,200 | FLOP | bound |  | 2 N_Qproj R_q D_n + 2 N_Qproj D_n R_k: the Q projection and the absorb step; N_Kproj = N_Vproj = 0 in this variant | F_projection[executed_graph], which carries the loop multiplicity |
| F_total[rect] | `work.absorbed_two_step.total.rect` | `150*D_n*H*R_k + 150*D_n*H*R_q + 150*D_v*H*R_k + 2*H*(D_r + 2*R_k)*(BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96))` | 3,943,296 | FLOP | bound |  | rectangular-scheduled work (C = C_rect, no internal padding), same projection rows | useful work (C = C_valid); executed graph work; compiled work |
| F_total[executed_graph] | `work.absorbed_two_step.graph.total` | `150*D_n*H*R_k + 150*D_n*H*R_q + 2*D_r*H*BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + 2*D_r*H*BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + 2*D_r*H*BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + 150*D_v*H*R_k + 4*H*R_k*BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + 4*H*R_k*BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + 4*H*R_k*BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96)` | 3,943,296 | FLOP | partial | schedule | graph sum of 2MNK*multiplicity over executed cells C_exec (declared extent policy) |  |
| F_projection[executed_graph] | `work.absorbed_two_step.graph.projection` | `150*D_n*H*R_k + 150*D_n*H*R_q` | 691,200 | FLOP | partial | schedule | graph category |  |
| F_attention[executed_graph] | `work.absorbed_two_step.graph.attention` | `2*D_r*H*BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + 2*D_r*H*BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + 2*D_r*H*BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + 4*H*R_k*BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + 4*H*R_k*BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + 4*H*R_k*BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96)` | 2,791,296 | FLOP | partial | schedule | graph category |  |
| F_output_projection[executed_graph] | `work.absorbed_two_step.graph.output_projection` | `150*D_v*H*R_k` | 460,800 | FLOP | partial | schedule | graph category |  |
| F_compiled | `work.compiled` | `from compilation evidence only` |  | FLOP | unknown | compile_evidence | compiled instruction-level work | useful; rect; executed_graph |
| F_A_minus_F_E | `work.path_difference` | `-182*D_n*H*R_k - 182*D_v*H*R_k + 8038*H*(D_r + 2*R_k) - 8038*H*(D_n + D_r + D_v)` | -603,776 | FLOP | bound |  | absorbed_two_step minus expanded, same projection rows and same C (useful) |  |
| F_A_minus_F_E[spec_closed_form] | `work.path_difference.closed_form` | `-182*H*R_k*(D_n + D_v) + 8038*H*(-D_n - D_v + 2*R_k)` | -603,776 | FLOP | bound |  | 2BH Rk (Dn+Dv)(Sq-Sk) + 2C(2Rk-Dn-Dv): full projection, no cache, two-step variant |  |
| equivalence[expanded vs absorbed] | `numerics.equivalence` | `matrix associativity: Q^n (W^k)^T C_k^T = Q^n (C_k W^k)^T` | equal over the real numbers |  | bound |  | the two paths express the same result in exact real arithmetic | bitwise equality; an error bound; a measured deviation |
| acceptance[declared] | `numerics.acceptance` | `caller-declared tolerances per test level` |  |  | unknown | acceptance | numeric acceptance declared with the task | a measured error; a proof that the variant meets it |
| W_vec[mul] | `work.vector.mul` | `H*R_k*BlockStat(row_visits, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + H*R_k*BlockStat(row_visits, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + H*R_k*BlockStat(row_visits, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + H*BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + H*BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + H*BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + H*BlockStat(row_visits, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + H*BlockStat(row_visits, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + H*BlockStat(row_visits, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96)` | 40,768 | element-ops | partial | schedule | logical count before redundancy elimination; not instructions | vector instruction count; MXU FLOPs |
| W_vec[mask] | `work.vector.mask` | `H*BlockStat(masked_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + H*BlockStat(masked_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + H*BlockStat(masked_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96)` | 8248 | element-ops | partial | schedule | logical count before redundancy elimination; not instructions | vector instruction count; MXU FLOPs |
| W_vec[cmp] | `work.vector.cmp` | `H*BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + H*BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + H*BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96)` | 19,384 | element-ops | partial | schedule | logical count before redundancy elimination; not instructions | vector instruction count; MXU FLOPs |
| W_vec[broadcast] | `work.vector.broadcast` | `H*BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + H*BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + H*BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96)` | 19,384 | element-ops | partial | broadcast_materialized (undeclared backend behaviour), schedule | logical count before redundancy elimination; not instructions | vector instruction count; MXU FLOPs |
| W_vec[transpose] | `work.vector.transpose` | `H*R_k*BlockStat(key_visits, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + H*R_k*BlockStat(key_visits, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + H*R_k*BlockStat(key_visits, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96)` | 44,288 | element-ops | partial | operand_transpose_materialized (undeclared backend behaviour), schedule | logical count before redundancy elimination; not instructions | vector instruction count; MXU FLOPs |
| W_vec[exp] | `work.vector.exp` | `H*BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + H*BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + H*BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + H*BlockStat(row_visits, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + H*BlockStat(row_visits, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + H*BlockStat(row_visits, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96)` | 20,032 | element-ops | partial | schedule | logical count before redundancy elimination; not instructions | vector instruction count; MXU FLOPs |
| W_vec[add] | `work.vector.add` | `H*R_k*BlockStat(row_visits, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + H*R_k*BlockStat(row_visits, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + H*R_k*BlockStat(row_visits, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + 75*H + H*BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + H*BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + H*BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + Piecewise((H*BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + H*BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + H*BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96), D_r > 0), (0, True))` | 59,804 | element-ops | partial | schedule | logical count before redundancy elimination; not instructions | vector instruction count; MXU FLOPs |
| W_vec[div] | `work.vector.div` | `75*H*R_k` | 9600 | element-ops | partial | final_normalize, schedule | logical count before redundancy elimination; not instructions | vector instruction count; MXU FLOPs |
| W_vec[recip] | `work.vector.recip` | `0` | 0 | element-ops | partial | schedule | logical count before redundancy elimination; not instructions | vector instruction count; MXU FLOPs |
| W_vec[log] | `work.vector.log` | `75*H` | 300 | element-ops | partial | schedule | logical count before redundancy elimination; not instructions | vector instruction count; MXU FLOPs |
| W_vec[cast] | `work.vector.cast` | `75*D_v*H + H*BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + H*BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + H*BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96)` | 26,584 | element-ops | partial | cast_points, schedule | logical count before redundancy elimination; not instructions | vector instruction count; MXU FLOPs |
| W_resource[vpu] | `work.resource.vpu` | `75*D_v*H + 75*H*R_k + 2*H*R_k*BlockStat(row_visits, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + 2*H*R_k*BlockStat(row_visits, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + 2*H*R_k*BlockStat(row_visits, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + 150*H + H*BlockStat(masked_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + H*BlockStat(masked_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + H*BlockStat(masked_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + 2*H*BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + 2*H*BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + 2*H*BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + 2*H*BlockStat(row_visits, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + 2*H*BlockStat(row_visits, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + 2*H*BlockStat(row_visits, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + Piecewise((H*BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + H*BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + H*BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96), D_r > 0), (0, True))` | 126,568 | element-ops | partial | cast_points, final_normalize, schedule | per resource class u (for W_u/P_u) |  |
| W_resource[reduce] | `work.resource.reduce` | `2*H*BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + 2*H*BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + 2*H*BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) - H*BlockStat(row_visits, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) - H*BlockStat(row_visits, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) - H*BlockStat(row_visits, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96)` | 38,120 | element-ops | partial | schedule | per resource class u (for W_u/P_u) |  |
| W_resource[layout] | `work.resource.layout` | `H*R_k*BlockStat(key_visits, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + H*R_k*BlockStat(key_visits, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + H*R_k*BlockStat(key_visits, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + H*BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + H*BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + H*BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96)` | 63,672 | element-ops | partial | broadcast_materialized (undeclared backend behaviour), operand_transpose_materialized (undeclared backend behaviour), schedule | per resource class u (for W_u/P_u) |  |
| W_resource[exp] | `work.resource.exp` | `H*BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + H*BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + H*BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + H*BlockStat(row_visits, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + H*BlockStat(row_visits, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + H*BlockStat(row_visits, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96)` | 20,032 | element-ops | partial | schedule | per resource class u (for W_u/P_u) |  |
| exp_count_executed | `work.exp.executed` | `H*(BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96))` | 19,384 | exp | partial | schedule | exp over executed cells (masked included) | C_valid |

Assumptions:
- `F_total[useful]`: N_Qproj=75*H, N_Kproj=166*H, N_Vproj=166*H
- `F_total[useful]`: C already includes B and H
- `F_total[useful]`: FMA = 2; not an instruction count
- `F_A[spec_closed_form]`: equals F_total[useful] only when projection_scope = full
- `N_Qproj`: absorbed path: C_k is consumed directly as both the K and the V operand, so no K/V row is projected in this scope; the Z W^v output projection is reported separately
- `N_Qproj`: distinct rows, not row visits: a strategy that re-projects inside the KV loop raises the executed-graph work, not this count (spec 5.2)
- `N_Kproj`: absorbed path: C_k is consumed directly as both the K and the V operand, so no K/V row is projected in this scope; the Z W^v output projection is reported separately
- `N_Kproj`: distinct rows, not row visits: a strategy that re-projects inside the KV loop raises the executed-graph work, not this count (spec 5.2)
- `N_Vproj`: absorbed path: C_k is consumed directly as both the K and the V operand, so no K/V row is projected in this scope; the Z W^v output projection is reported separately
- `N_Vproj`: distinct rows, not row visits: a strategy that re-projects inside the KV loop raises the executed-graph work, not this count (spec 5.2)
- `F_proj[general]`: N_Qproj=75*H, N_Kproj=0, N_Vproj=0
- `F_proj[general]`: absorbed path: C_k is consumed directly as both the K and the V operand, so no K/V row is projected in this scope; the Z W^v output projection is reported separately
- `F_proj[general]`: useful-work counting: each row projected once. When the strategy projects inside the KV loop, the repeated work is in F_projection[executed_graph] (spec 5.2), not here.
- `F_total[rect]`: rectangle selection policy 'skip_future'
- `F_total[rect]`: projection counted once per row: the executed graph may differ where a tile is re-projected or skipped
- `F_total[executed_graph]`: entry scope: seven latent inputs; Q~ formed inside this scope; no expanded K/V materialization
- `F_total[executed_graph]`: outside the entry scope and therefore counted nowhere in this report: RMSNorm/LayerNorm, the RoPE rotation that produces Q^r/K^r, the latent down-projection that produces C_q/C_k, the output projection W^O after this attention, and the KV-cache update
- `F_total[executed_graph]`: loop nest scenario q_outer_kv_inner
- `F_total[executed_graph]`: accumulator width D_A = R_k (absorbed); Z = A / l still needs the W^v projection
- `F_total[executed_graph]`: storage levels are an analysis assignment (residency windows -> vmem, vector temporaries -> vreg), not compiler allocation
- `F_A_minus_F_E`: computation comparison only, not runtime
- `F_A_minus_F_E`: reduces to 2BH Rk(Dn+Dv)(Sq-Sk) + 2C(2Rk-Dn-Dv) for full projection (two-step variant)
- `F_A_minus_F_E[spec_closed_form]`: sign decided by 2Rk vs Dn+Dv in square prefill
- `equivalence[expanded vs absorbed]`: low-precision reassociation does not guarantee bitwise equality: the operand and accumulator dtypes, the accumulation order and the cast sites all differ between the paths
- `acceptance[declared]`: declared only: this model runs no arithmetic, so no tolerance is evaluated here

<details><summary>Provenance (bindings, coverage, source, evidence)</summary>

| metric | source | coverage | bindings | evidence |
|---|---|---|---|---|
| F_Q[useful] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"D_n": 24, "H": 4, "R_q": 16}` |  |
| F_Qtilde[useful] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"D_n": 24, "H": 4, "R_k": 32}` |  |
| F_QC[useful] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"H": 4, "R_k": 32}` |  |
| F_QK_r[useful] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"D_r": 8, "H": 4}` |  |
| F_PC[useful] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"H": 4, "R_k": 32}` |  |
| F_ZWv[useful] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"D_v": 24, "H": 4, "R_k": 32}` |  |
| F_total[useful] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"D_n": 24, "D_r": 8, "D_v": 24, "H": 4, "R_k": 32, "R_q": 16}` |  |
| F_A[spec_closed_form] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"D_n": 24, "D_r": 8, "D_v": 24, "H": 4, "R_k": 32, "R_q": 16}` |  |
| N_Qproj | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"H": 4}` |  |
| N_Kproj | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{}` |  |
| N_Vproj | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{}` |  |
| F_proj[general] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"D_n": 24, "H": 4, "R_k": 32, "R_q": 16}` |  |
| F_total[rect] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"D_n": 24, "D_r": 8, "D_v": 24, "H": 4, "R_k": 32, "R_q": 16, "b_k": 32, "b_q": 16}` |  |
| F_total[executed_graph] | derived | incomplete: depends on undeclared/unbound schedule | `{"D_n": 24, "D_r": 8, "D_v": 24, "H": 4, "R_k": 32, "R_q": 16, "b_k": 32, "b_q": 16}` |  |
| F_projection[executed_graph] | derived | incomplete: depends on undeclared/unbound schedule | `{"D_n": 24, "H": 4, "R_k": 32, "R_q": 16}` |  |
| F_attention[executed_graph] | derived | incomplete: depends on undeclared/unbound schedule | `{"D_r": 8, "H": 4, "R_k": 32, "b_k": 32, "b_q": 16}` |  |
| F_output_projection[executed_graph] | derived | incomplete: depends on undeclared/unbound schedule | `{"D_v": 24, "H": 4, "R_k": 32}` |  |
| F_compiled | compiled | incomplete: depends on undeclared/unbound compile_evidence | `{}` |  |
| F_A_minus_F_E | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"D_n": 24, "D_r": 8, "D_v": 24, "H": 4, "R_k": 32}` |  |
| F_A_minus_F_E[spec_closed_form] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"D_n": 24, "D_v": 24, "H": 4, "R_k": 32}` |  |
| equivalence[expanded vs absorbed] | definition | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{}` |  |
| acceptance[declared] | input | incomplete: depends on undeclared/unbound acceptance | `{}` |  |
| W_vec[mul] | derived | incomplete: depends on undeclared/unbound schedule | `{"H": 4, "R_k": 32, "b_k": 32, "b_q": 16}` |  |
| W_vec[mask] | derived | incomplete: depends on undeclared/unbound schedule | `{"H": 4, "b_k": 32, "b_q": 16}` |  |
| W_vec[cmp] | derived | incomplete: depends on undeclared/unbound schedule | `{"H": 4, "b_k": 32, "b_q": 16}` |  |
| W_vec[broadcast] | derived | incomplete: depends on undeclared/unbound broadcast_materialized (undeclared backend behaviour), schedule | `{"H": 4, "b_k": 32, "b_q": 16}` |  |
| W_vec[transpose] | derived | incomplete: depends on undeclared/unbound operand_transpose_materialized (undeclared backend behaviour), schedule | `{"H": 4, "R_k": 32, "b_k": 32, "b_q": 16}` |  |
| W_vec[exp] | derived | incomplete: depends on undeclared/unbound schedule | `{"H": 4, "b_k": 32, "b_q": 16}` |  |
| W_vec[add] | derived | incomplete: depends on undeclared/unbound schedule | `{"D_r": 8, "H": 4, "R_k": 32, "b_k": 32, "b_q": 16}` |  |
| W_vec[div] | derived | incomplete: depends on undeclared/unbound final_normalize, schedule | `{"H": 4, "R_k": 32}` |  |
| W_vec[recip] | derived | incomplete: depends on undeclared/unbound schedule | `{}` |  |
| W_vec[log] | derived | incomplete: depends on undeclared/unbound schedule | `{"H": 4}` |  |
| W_vec[cast] | derived | incomplete: depends on undeclared/unbound cast_points, schedule | `{"D_v": 24, "H": 4, "b_k": 32, "b_q": 16}` |  |
| W_resource[vpu] | derived | incomplete: depends on undeclared/unbound cast_points, final_normalize, schedule | `{"D_r": 8, "D_v": 24, "H": 4, "R_k": 32, "b_k": 32, "b_q": 16}` |  |
| W_resource[reduce] | derived | incomplete: depends on undeclared/unbound schedule | `{"H": 4, "b_k": 32, "b_q": 16}` |  |
| W_resource[layout] | derived | incomplete: depends on undeclared/unbound broadcast_materialized (undeclared backend behaviour), operand_transpose_materialized (undeclared backend behaviour), schedule | `{"H": 4, "R_k": 32, "b_k": 32, "b_q": 16}` |  |
| W_resource[exp] | derived | incomplete: depends on undeclared/unbound schedule | `{"H": 4, "b_k": 32, "b_q": 16}` |  |
| exp_count_executed | derived | incomplete: depends on undeclared/unbound schedule | `{"H": 4, "b_k": 32, "b_q": 16}` |  |

</details>

## 4. bytes: interface and materialization

| metric | formula id | expression | value | unit | status | missing | scope | not equivalent to |
|---|---|---|---|---|---|---|---|---|
| bytes[q_latent] | `bytes.logical.q_latent` | `B*R_q*S_q*s_Cq` | 4608 | byte | bound |  | logical size at this tensor's own allocation extent, per-tensor dtype |  |
| bytes[kv_latent] | `bytes.logical.kv_latent` | `B*R_k*S_k*s_Ck` | 18,432 | byte | bound |  | logical size at this tensor's own allocation extent, per-tensor dtype |  |
| bytes[q_pe] | `bytes.logical.q_pe` | `B*D_r*H*S_q*s_Qr` | 9216 | byte | bound |  | logical size at this tensor's own allocation extent, per-tensor dtype |  |
| bytes[k_pe] | `bytes.logical.k_pe` | `B*D_r*S_k*s_Kr` | 4608 | byte | bound |  | logical size at this tensor's own allocation extent, per-tensor dtype |  |
| bytes[w_q_nope] | `bytes.logical.w_q_nope` | `D_n*H*R_q*s_Wq` | 3072 | byte | bound |  | logical size at this tensor's own allocation extent, per-tensor dtype |  |
| bytes[w_k_nope] | `bytes.logical.w_k_nope` | `D_n*H*R_k*s_Wk` | 6144 | byte | bound |  | logical size at this tensor's own allocation extent, per-tensor dtype |  |
| bytes[w_v] | `bytes.logical.w_v` | `D_v*H*R_k*s_Wv` | 6144 | byte | bound |  | logical size at this tensor's own allocation extent, per-tensor dtype |  |
| M_in | `bytes.M_in` | `9216*s_Ck + 2304*s_Cq + 2304*s_Kr + 4608*s_Qr + 3072*s_Wk + 1536*s_Wq + 3072*s_Wv` | 52,224 | byte | bound |  | sum of the seven inputs, each at its OWN ALLOCATION extent (capacity_shape where declared) | HBM traffic lower bound under caching/reuse; M_in[metadata_ledger], which sums the LOGICAL shapes rather than the allocations |
| M_O | `bytes.M_O` | `75*D_v*H*s_O` | 14,400 | byte | bound |  | output O at active extents (s_O * B H S_q * D_v; ragged: H sum_b Sq_b) |  |
| M_LSE | `bytes.M_LSE` | `75*H*s_LSE` | 1200 | byte | bound |  | LSE at active extents |  |
| M_in[metadata_ledger] | `bytes.ledger.logical_total` | `sum of per-tensor prod(shape)*bytes(dtype)` | 52,224 | byte | bound |  | sum of the LOGICAL sizes prod(shape)*s from the actual metadata, not the allocations | M_in, which sums each tensor at its allocation extent; M_alloc_unique[metadata_ledger], the allocation total with aliases counted once |
| M_alloc_unique[metadata_ledger] | `bytes.ledger.alloc_unique` | `allocated bytes with aliases counted once` | 52,224 | byte | bound |  | capacity_shape and aliases honoured | the union of address ranges |
| HBM_mat[Q~] | `bytes.mat.mat_q_tilde` | `150*H*R_k*s_Qt` | 0 | byte | not_applicable |  | write + read traffic if materialized (read under the declared scan policy) | interface bytes |
| M_mat[Q~] | `bytes.mat.size.mat_q_tilde` | `75*H*R_k*s_Qt` | 0 | byte | not_applicable |  | logical size of the materialized intermediate itself (spec 6.2 M^mat) | its HBM traffic; interface bytes |
| HBM_mat[Z] | `bytes.mat.mat_z` | `150*H*R_k*s_Z` | 0 | byte | not_applicable |  | write + read traffic if materialized (read under the declared scan policy) | interface bytes |
| M_mat[Z] | `bytes.mat.size.mat_z` | `75*H*R_k*s_Z` | 0 | byte | not_applicable |  | logical size of the materialized intermediate itself (spec 6.2 M^mat) | its HBM traffic; interface bytes |

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
- `HBM_mat[Q~]`: conditional on materialize.q_tilde = False
- `HBM_mat[Q~]`: write traffic 75*H*R_k*s_Qt; read traffic 75*H*R_k*s_Qt
- `M_mat[Q~]`: conditional on materialize.q_tilde = False
- `HBM_mat[Z]`: conditional on materialize.z = False
- `HBM_mat[Z]`: write traffic 75*H*R_k*s_Z; read traffic 75*H*R_k*s_Z
- `M_mat[Z]`: conditional on materialize.z = False

<details><summary>Provenance (bindings, coverage, source, evidence)</summary>

| metric | source | coverage | bindings | evidence |
|---|---|---|---|---|
| bytes[q_latent] | input | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"B": 3, "R_q": 16, "S_q": 48, "s_Cq": 2}` |  |
| bytes[kv_latent] | input | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"B": 3, "R_k": 32, "S_k": 96, "s_Ck": 2}` |  |
| bytes[q_pe] | input | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"B": 3, "D_r": 8, "H": 4, "S_q": 48, "s_Qr": 2}` |  |
| bytes[k_pe] | input | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"B": 3, "D_r": 8, "S_k": 96, "s_Kr": 2}` |  |
| bytes[w_q_nope] | input | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"D_n": 24, "H": 4, "R_q": 16, "s_Wq": 2}` |  |
| bytes[w_k_nope] | input | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"D_n": 24, "H": 4, "R_k": 32, "s_Wk": 2}` |  |
| bytes[w_v] | input | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"D_v": 24, "H": 4, "R_k": 32, "s_Wv": 2}` |  |
| M_in | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"s_Ck": 2, "s_Cq": 2, "s_Kr": 2, "s_Qr": 2, "s_Wk": 2, "s_Wq": 2, "s_Wv": 2}` |  |
| M_O | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"D_v": 24, "H": 4, "s_O": 2}` |  |
| M_LSE | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"H": 4, "s_LSE": 4}` |  |
| M_in[metadata_ledger] | input | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{}` |  |
| M_alloc_unique[metadata_ledger] | input | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{}` |  |
| HBM_mat[Q~] | conditional | not applicable under the declared configuration | `{"H": 4, "R_k": 32, "s_Qt": 2}` |  |
| M_mat[Q~] | conditional | not applicable under the declared configuration | `{"H": 4, "R_k": 32, "s_Qt": 2}` |  |
| HBM_mat[Z] | conditional | not applicable under the declared configuration | `{"H": 4, "R_k": 32, "s_Z": 2}` |  |
| M_mat[Z] | conditional | not applicable under the declared configuration | `{"H": 4, "R_k": 32, "s_Z": 2}` |  |

</details>

## 5. bytes: transfer requests per path

| metric | formula id | expression | value | unit | status | missing | scope | not equivalent to |
|---|---|---|---|---|---|---|---|---|
| B[hbm_to_vmem] | `bytes.path.hbm_to_vmem` | `D_n*H*R_k*s_Wk*ceiling(7/b_q) + D_n*H*R_k*s_Wk*ceiling(20/b_q) + D_n*H*R_k*s_Wk*ceiling(48/b_q) + D_n*H*R_q*s_Wq*ceiling(7/b_q) + D_n*H*R_q*s_Wq*ceiling(20/b_q) + D_n*H*R_q*s_Wq*ceiling(48/b_q) + 75*D_r*H*s_Qr + D_r*s_Kr*ceiling(H/h_pp)*BlockStat(key_visits, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + D_r*s_Kr*ceiling(H/h_pp)*BlockStat(key_visits, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + D_r*s_Kr*ceiling(H/h_pp)*BlockStat(key_visits, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + D_v*H*R_k*s_Wv*ceiling(7/b_q) + D_v*H*R_k*s_Wv*ceiling(20/b_q) + D_v*H*R_k*s_Wv*ceiling(48/b_q) + R_k*s_Ck*ceiling(H/h_pp)*BlockStat(key_visits, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + R_k*s_Ck*ceiling(H/h_pp)*BlockStat(key_visits, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + R_k*s_Ck*ceiling(H/h_pp)*BlockStat(key_visits, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + 75*R_q*s_Cq*ceiling(H/h_pp)` | 127,040 | byte | partial | schedule | sum_r m_r n_r over the scenario's transfer events | physical HBM traffic with caching; spill traffic |
| B[vmem_to_hbm] | `bytes.path.vmem_to_hbm` | `75*D_v*H*s_O + 75*H*s_LSE` | 15,600 | byte | partial | schedule | sum_r m_r n_r over the scenario's transfer events | physical HBM traffic with caching; spill traffic |
| B[vmem_to_vreg] | `bytes.path.vmem_to_vreg` | `D_r*H*b_k*s_Kr*BlockStat(selected, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + D_r*H*b_k*s_Kr*BlockStat(selected, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + D_r*H*b_k*s_Kr*BlockStat(selected, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + D_r*H*b_q*s_Qr*ceiling(7/b_q) + D_r*H*b_q*s_Qr*ceiling(20/b_q) + D_r*H*b_q*s_Qr*ceiling(48/b_q) + H*R_k*b_k*s_Ck*BlockStat(selected, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + H*R_k*b_k*s_Ck*BlockStat(selected, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + H*R_k*b_k*s_Ck*BlockStat(selected, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + H*R_k*b_q*s_Qt*ceiling(7/b_q) + H*R_k*b_q*s_Qt*ceiling(20/b_q) + H*R_k*b_q*s_Qt*ceiling(48/b_q)` | 153,600 | byte | partial | q_resident, register_schedule_evidence | sum_r m_r n_r over the scenario's transfer events | physical HBM traffic with caching; spill traffic |
| B[vreg_to_vmem] | `bytes.path.vreg_to_vmem` | `sum_r m_r n_r (no modeled events)` |  | byte | unknown | register_schedule_evidence | path traffic | spill traffic; physical traffic with caching |

Assumptions:
- `B[hbm_to_vmem]`: scenario q_outer_kv_inner; weights re-read per program
- `B[hbm_to_vmem]`: no hardware cache effects are modeled: every event is counted at its declared multiplicity
- `B[vmem_to_hbm]`: scenario q_outer_kv_inner; weights re-read per program
- `B[vmem_to_hbm]`: no hardware cache effects are modeled: every event is counted at its declared multiplicity
- `B[vmem_to_vreg]`: scenario q_outer_kv_inner; weights re-read per program
- `B[vmem_to_vreg]`: no hardware cache effects are modeled: every event is counted at its declared multiplicity
- `B[vmem_to_vreg]`: operand-streaming scenario without register-schedule evidence
- `B[vmem_to_vreg]`: includes scenario events whose condition is undeclared (stream_Qt_operand, stream_Qr_operand, stream_Ck_tile, stream_Kr_tile): they are counted at their declared multiplicity and named in missing_fields

<details><summary>Provenance (bindings, coverage, source, evidence)</summary>

| metric | source | coverage | bindings | evidence |
|---|---|---|---|---|
| B[hbm_to_vmem] | conditional | incomplete: depends on undeclared/unbound schedule | `{"D_n": 24, "D_r": 8, "D_v": 24, "H": 4, "R_k": 32, "R_q": 16, "b_k": 32, "b_q": 16, "h...` |  |
| B[vmem_to_hbm] | conditional | incomplete: depends on undeclared/unbound schedule | `{"D_v": 24, "H": 4, "s_LSE": 4, "s_O": 2}` |  |
| B[vmem_to_vreg] | conditional | incomplete: depends on undeclared/unbound q_resident, register_schedule_evidence | `{"D_r": 8, "H": 4, "R_k": 32, "b_k": 32, "b_q": 16, "s_Ck": 2, "s_Kr": 2, "s_Qr": 2, "s...` |  |
| B[vreg_to_vmem] | conditional | incomplete: depends on undeclared/unbound register_schedule_evidence | `{}` |  |

</details>

## 6. local objects (symbolic sizes, spec 7.1)

| metric | formula id | expression | value | unit | status | missing | scope | not equivalent to |
|---|---|---|---|---|---|---|---|---|
| local[q_latent_window] | `local.q_latent_window.bytes` | `R_q*b_q*s_Cq` | 512 | byte | partial | b_rq | one logical object; level=vmem; live in ['q_proj'] | VREG allocation; spill traffic |
| local[wq_rank_tile] | `local.wq_rank_tile.bytes` | `D_n*R_q*s_Wq` | 768 | byte | partial | b_rq | one logical object; level=vmem; live in ['q_proj'] | VREG allocation; spill traffic |
| local[q_proj_acc] | `local.q_proj_acc.bytes` | `D_n*b_q*s_Qacc` | 1536 | byte | partial | projection_accumulator_dtype | one logical object; level=vreg; live in ['q_proj', 'q_absorb'] | VREG allocation; spill traffic |
| local[wk_full] | `local.wk_full.bytes` | `D_n*R_k*s_Wk` | 1536 | byte | bound |  | one logical object; level=vmem; live in ['q_absorb'] | VREG allocation; spill traffic |
| local[q_tilde_acc] | `local.q_tilde_acc.bytes` | `R_k*b_q*s_Qacc` | 2048 | byte | partial | projection_accumulator_dtype | one logical object; level=vreg; live in ['q_absorb'] | VREG allocation; spill traffic |
| local[q_tilde] | `local.q_tilde.bytes` | `R_k*b_q*s_Qt` | 1024 | byte | partial | projected_operand_dtype, q_resident | one logical object; level=vmem; live in ['score', 'softmax', 'pv', 'finalize'] | VREG allocation; spill traffic |
| local[kv_latent_tile] | `local.kv_latent_tile.bytes` | `R_k*b_k*n_buf*s_Ck` | 4096 | byte | bound |  | one logical object x its buffer count; level=vmem; live in ['score', 'softmax', 'pv']; residency window of n_buf=n_buf buffers (7.3 allocation, not the 7.1 logical size) | VREG allocation; spill traffic; the 7.1 logical size of one buffer |
| local[wv_full] | `local.wv_full.bytes` | `D_v*R_k*s_Wv` | 1536 | byte | bound |  | one logical object; level=vmem; live in ['finalize'] | VREG allocation; spill traffic |
| local[z_tile] | `local.z_tile.bytes` | `R_k*b_q*s_Z` | 1024 | byte | partial | projected_operand_dtype | one logical object; level=vreg; live in ['finalize'] | VREG allocation; spill traffic |
| local[score_X] | `local.score_X.bytes` | `b_k*b_q*s_X` | 2048 | byte | bound |  | one logical object; level=vreg; live in ['score', 'softmax'] | VREG allocation; spill traffic |
| local[exp_E] | `local.exp_E.bytes` | `b_k*b_q*s_E` | 2048 | byte | bound |  | one logical object; level=vreg; live in ['softmax', 'pv'] | VREG allocation; spill traffic |
| local[p_operand] | `local.p_operand.bytes` | `b_k*b_q*s_P` | 1024 | byte | bound |  | one logical object; level=vreg; live in ['pv'] | VREG allocation; spill traffic |
| local[row_state_m] | `local.row_state_m.bytes` | `b_q*s_state` | 64 | byte | bound |  | one logical object; level=vreg; live in ['score', 'softmax', 'pv', 'finalize'] | VREG allocation; spill traffic |
| local[row_state_l] | `local.row_state_l.bytes` | `b_q*s_state` | 64 | byte | bound |  | one logical object; level=vreg; live in ['score', 'softmax', 'pv', 'finalize'] | VREG allocation; spill traffic |
| local[row_alpha] | `local.row_alpha.bytes` | `b_q*s_state` | 64 | byte | bound |  | one logical object; level=vreg; live in ['softmax', 'pv'] | VREG allocation; spill traffic |
| local[accumulator_A] | `local.accumulator_A.bytes` | `R_k*b_q*s_A` | 2048 | byte | bound |  | one logical object; level=vreg; live in ['score', 'softmax', 'pv', 'finalize'] | VREG allocation; spill traffic |
| local[q_pe_tile] | `local.q_pe_tile.bytes` | `D_r*b_q*s_Qr` | 256 | byte | partial | q_resident | one logical object; level=vmem; live in ['score', 'softmax', 'pv', 'finalize'] | VREG allocation; spill traffic |
| local[k_pe_tile] | `local.k_pe_tile.bytes` | `D_r*b_k*n_buf*s_Kr` | 1024 | byte | bound |  | one logical object x its buffer count; level=vmem; live in ['score']; residency window of n_buf=n_buf buffers (7.3 allocation, not the 7.1 logical size) | VREG allocation; spill traffic; the 7.1 logical size of one buffer |
| local[fixed_scratch] | `local.fixed_scratch.bytes` | `s_scratch` | 0 | byte | bound |  | one logical object; level=vmem; live in ['q_proj', 'q_absorb', 'score', 'softmax', 'pv', 'finalize'] | VREG allocation; spill traffic |
| local[out_tile] | `local.out_tile.bytes` | `D_v*b_q*s_O` | 768 | byte | bound |  | one logical object; level=vmem; live in ['finalize'] | VREG allocation; spill traffic |
| local[lse_tile] | `local.lse_tile.bytes` | `b_q*s_LSE` | 64 | byte | bound |  | one logical object; level=vmem; live in ['finalize'] | VREG allocation; spill traffic |
| layout[q_latent_window] | `layout.q_latent_window` | `L_l*L_s*s_Cq*ceiling(R_q/L_l)*ceiling(b_q/L_s) → 2*L_l*L_s*ceiling(16/L_l)*ceiling(16/L_s)` |  | byte | partial | L_l, L_s, layout.q_latent_window | layout coverage, kind=vector_tiled, level=vmem (scenario: kind undeclared) | register-file capacity |
| layout[wq_rank_tile] | `layout.wq_rank_tile` | `L_l*L_s*s_Wq*ceiling(D_n/L_l)*ceiling(R_q/L_s) → 2*L_l*L_s*ceiling(24/L_l)*ceiling(16/L_s)` |  | byte | partial | L_l, L_s, layout.wq_rank_tile | layout coverage, kind=vector_tiled, level=vmem (scenario: kind undeclared) | register-file capacity |
| layout[q_proj_acc] | `layout.q_proj_acc` | `L_l*L_s*s_Qacc*ceiling(D_n/L_l)*ceiling(b_q/L_s) → 4*L_l*L_s*ceiling(24/L_l)*ceiling(16/L_s)` |  | byte | partial | L_l, L_s, layout.q_proj_acc | layout coverage, kind=vector_tiled, level=vreg (scenario: kind undeclared) | register-file capacity |
| layout[wk_full] | `layout.wk_full` | `L_l*L_s*s_Wk*ceiling(D_n/L_l)*ceiling(R_k/L_s) → 2*L_l*L_s*ceiling(24/L_l)*ceiling(32/L_s)` |  | byte | partial | L_l, L_s, layout.wk_full | layout coverage, kind=vector_tiled, level=vmem (scenario: kind undeclared) | register-file capacity |
| layout[q_tilde_acc] | `layout.q_tilde_acc` | `L_l*L_s*s_Qacc*ceiling(R_k/L_l)*ceiling(b_q/L_s) → 4*L_l*L_s*ceiling(32/L_l)*ceiling(16/L_s)` |  | byte | partial | L_l, L_s, layout.q_tilde_acc | layout coverage, kind=vector_tiled, level=vreg (scenario: kind undeclared) | register-file capacity |
| layout[q_tilde] | `layout.q_tilde` | `L_l*L_s*s_Qt*ceiling(R_k/L_l)*ceiling(b_q/L_s) → 2*L_l*L_s*ceiling(32/L_l)*ceiling(16/L_s)` |  | byte | partial | L_l, L_s, layout.q_tilde | layout coverage, kind=vector_tiled, level=vmem (scenario: kind undeclared) | register-file capacity |
| layout[kv_latent_tile] | `layout.kv_latent_tile` | `L_l*L_s*n_buf*s_Ck*ceiling(R_k/L_l)*ceiling(b_k/L_s) → 4*L_l*L_s*ceiling(32/L_l)*ceiling(32/L_s)` |  | byte | partial | L_l, L_s, layout.kv_latent_tile | layout coverage, kind=vector_tiled, level=vmem (scenario: kind undeclared) | register-file capacity |
| layout[wv_full] | `layout.wv_full` | `L_l*L_s*s_Wv*ceiling(D_v/L_l)*ceiling(R_k/L_s) → 2*L_l*L_s*ceiling(24/L_l)*ceiling(32/L_s)` |  | byte | partial | L_l, L_s, layout.wv_full | layout coverage, kind=vector_tiled, level=vmem (scenario: kind undeclared) | register-file capacity |
| layout[z_tile] | `layout.z_tile` | `L_l*L_s*s_Z*ceiling(R_k/L_l)*ceiling(b_q/L_s) → 2*L_l*L_s*ceiling(32/L_l)*ceiling(16/L_s)` |  | byte | partial | L_l, L_s, layout.z_tile | layout coverage, kind=vector_tiled, level=vreg (scenario: kind undeclared) | register-file capacity |
| layout[score_X] | `layout.score_X` | `L_l*L_s*s_X*ceiling(b_k/L_l)*ceiling(b_q/L_s) → 4*L_l*L_s*ceiling(32/L_l)*ceiling(16/L_s)` |  | byte | partial | L_l, L_s, layout.score_X | layout coverage, kind=vector_tiled, level=vreg (scenario: kind undeclared) | register-file capacity |
| layout[exp_E] | `layout.exp_E` | `L_l*L_s*s_E*ceiling(b_k/L_l)*ceiling(b_q/L_s) → 4*L_l*L_s*ceiling(32/L_l)*ceiling(16/L_s)` |  | byte | partial | L_l, L_s, layout.exp_E | layout coverage, kind=vector_tiled, level=vreg (scenario: kind undeclared) | register-file capacity |
| layout[p_operand] | `layout.p_operand` | `L_l*L_s*s_P*ceiling(b_k/L_l)*ceiling(b_q/L_s) → 2*L_l*L_s*ceiling(32/L_l)*ceiling(16/L_s)` |  | byte | partial | L_l, L_s, layout.p_operand | layout coverage, kind=vector_tiled, level=vreg (scenario: kind undeclared) | register-file capacity |
| layout[row_state_m] | `layout.row_state_m` | `L_l*L_s*s_state*ceiling(1/L_l)*ceiling(b_q/L_s) → 4*L_l*L_s*ceiling(1/L_l)*ceiling(16/L_s)` |  | byte | partial | L_l, L_s, layout.row_state_m | layout coverage, kind=vector_tiled, level=vreg (scenario: kind undeclared) | register-file capacity |
| layout[row_state_l] | `layout.row_state_l` | `L_l*L_s*s_state*ceiling(1/L_l)*ceiling(b_q/L_s) → 4*L_l*L_s*ceiling(1/L_l)*ceiling(16/L_s)` |  | byte | partial | L_l, L_s, layout.row_state_l | layout coverage, kind=vector_tiled, level=vreg (scenario: kind undeclared) | register-file capacity |
| layout[row_alpha] | `layout.row_alpha` | `L_l*L_s*s_state*ceiling(1/L_l)*ceiling(b_q/L_s) → 4*L_l*L_s*ceiling(1/L_l)*ceiling(16/L_s)` |  | byte | partial | L_l, L_s, layout.row_alpha | layout coverage, kind=vector_tiled, level=vreg (scenario: kind undeclared) | register-file capacity |
| layout[accumulator_A] | `layout.accumulator_A` | `L_l*L_s*s_A*ceiling(R_k/L_l)*ceiling(b_q/L_s) → 4*L_l*L_s*ceiling(32/L_l)*ceiling(16/L_s)` |  | byte | partial | L_l, L_s, layout.accumulator_A | layout coverage, kind=vector_tiled, level=vreg (scenario: kind undeclared) | register-file capacity |
| layout[q_pe_tile] | `layout.q_pe_tile` | `L_l*L_s*s_Qr*ceiling(D_r/L_l)*ceiling(b_q/L_s) → 2*L_l*L_s*ceiling(8/L_l)*ceiling(16/L_s)` |  | byte | partial | L_l, L_s, layout.q_pe_tile | layout coverage, kind=vector_tiled, level=vmem (scenario: kind undeclared) | register-file capacity |
| layout[k_pe_tile] | `layout.k_pe_tile` | `L_l*L_s*n_buf*s_Kr*ceiling(D_r/L_l)*ceiling(b_k/L_s) → 4*L_l*L_s*ceiling(8/L_l)*ceiling(32/L_s)` |  | byte | partial | L_l, L_s, layout.k_pe_tile | layout coverage, kind=vector_tiled, level=vmem (scenario: kind undeclared) | register-file capacity |
| layout[fixed_scratch] | `layout.fixed_scratch` | `s_scratch` | 0 | byte | bound |  | layout coverage, kind=raw_bytes, level=vmem | register-file capacity |
| layout[out_tile] | `layout.out_tile` | `L_l*L_s*s_O*ceiling(D_v/L_l)*ceiling(b_q/L_s) → 2*L_l*L_s*ceiling(24/L_l)*ceiling(16/L_s)` |  | byte | partial | L_l, L_s, layout.out_tile | layout coverage, kind=vector_tiled, level=vmem (scenario: kind undeclared) | register-file capacity |
| layout[lse_tile] | `layout.lse_tile` | `L_l*L_s*s_LSE*ceiling(1/L_l)*ceiling(b_q/L_s) → 4*L_l*L_s*ceiling(1/L_l)*ceiling(16/L_s)` |  | byte | partial | L_l, L_s, layout.lse_tile | layout coverage, kind=vector_tiled, level=vmem (scenario: kind undeclared) | register-file capacity |
| HBM_mat[K^n,V] | `bytes.mat.mat_expanded_kv` | `not formed in this variant` |  |  | not_applicable |  | structural absence in this algorithm path | an unknown value |
| M_mat[K^n,V] | `bytes.mat.size.mat_expanded_kv` | `not formed in this variant` |  |  | not_applicable |  | structural absence in this algorithm path | an unknown value |
| HBM_mat[Q^n] | `bytes.mat.mat_q_nope` | `not formed in this variant` |  |  | not_applicable |  | structural absence in this algorithm path | an unknown value |
| M_mat[Q^n] | `bytes.mat.size.mat_q_nope` | `not formed in this variant` |  |  | not_applicable |  | structural absence in this algorithm path | an unknown value |
| local[v_tile] | `local.v_tile.bytes` | `not formed in this variant` |  |  | not_applicable |  | structural absence in this algorithm path | an unknown value |
| local[k_nope_tile] | `local.k_nope_tile.bytes` | `not formed in this variant` |  |  | not_applicable |  | structural absence in this algorithm path | an unknown value |
| local[q_nope_operand] | `local.q_nope_operand.bytes` | `not formed in this variant` |  |  | not_applicable |  | structural absence in this algorithm path | an unknown value |

Assumptions:
- `local[q_latent_window]`: Q latent window
- `local[q_latent_window]`: b_rq undeclared: the full rank window is shown as the scenario
- `local[wq_rank_tile]`: W^q rank sub-tile
- `local[wq_rank_tile]`: b_rq undeclared: the full rank window is shown as the scenario
- `local[q_proj_acc]`: Q^n accumulator
- `local[wk_full]`: W^k for the absorb step (per head)
- `local[q_tilde_acc]`: Q~ accumulator
- `local[q_tilde]`: Q~ operand held across the KV loop (shortened when q_resident=False)
- `local[q_tilde]`: q_resident undeclared: the live-stage set printed in the scope is the scenario
- `local[kv_latent_tile]`: C_k tile used both as K and as V operand
- `local[wv_full]`: W^v for the output projection
- `local[z_tile]`: normalized latent output Z
- `local[score_X]`: score tile X = gamma(...)+M
- `local[exp_E]`: E = exp(X - m')
- `local[p_operand]`: P/E operand in the matmul input dtype
- `local[row_state_m]`: running row max m (loop-carried)
- `local[row_state_l]`: running row sum l (loop-carried)
- `local[row_alpha]`: alpha = exp(m - m')
- `local[accumulator_A]`: attention accumulator A, width R_k (loop-carried)
- `local[q_pe_tile]`: Q^r operand held across the KV loop (shortened when q_resident=False)
- `local[q_pe_tile]`: q_resident undeclared: the live-stage set printed in the scope is the scenario
- `local[k_pe_tile]`: K^r tile for the current KV block
- `local[fixed_scratch]`: fixed scratch (d_s)
- `local[out_tile]`: output tile O in output dtype
- `local[lse_tile]`: LSE tile
- `layout[q_latent_window]`: LayoutBytes = s L_s L_l ceil(m/L_s) ceil(n/L_l) with profile tiles
- `layout[wq_rank_tile]`: LayoutBytes = s L_s L_l ceil(m/L_s) ceil(n/L_l) with profile tiles
- `layout[q_proj_acc]`: LayoutBytes = s L_s L_l ceil(m/L_s) ceil(n/L_l) with profile tiles
- `layout[wk_full]`: LayoutBytes = s L_s L_l ceil(m/L_s) ceil(n/L_l) with profile tiles
- `layout[q_tilde_acc]`: LayoutBytes = s L_s L_l ceil(m/L_s) ceil(n/L_l) with profile tiles
- `layout[q_tilde]`: LayoutBytes = s L_s L_l ceil(m/L_s) ceil(n/L_l) with profile tiles
- `layout[kv_latent_tile]`: LayoutBytes = s L_s L_l ceil(m/L_s) ceil(n/L_l) with profile tiles
- `layout[wv_full]`: LayoutBytes = s L_s L_l ceil(m/L_s) ceil(n/L_l) with profile tiles
- `layout[z_tile]`: LayoutBytes = s L_s L_l ceil(m/L_s) ceil(n/L_l) with profile tiles
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
- `HBM_mat[K^n,V]`: the absorbed path never expands K^n or V and never forms a Q^n operand: C_k is the K and the V operand
- `M_mat[K^n,V]`: the absorbed path never expands K^n or V and never forms a Q^n operand: C_k is the K and the V operand
- `HBM_mat[Q^n]`: the absorbed path never expands K^n or V and never forms a Q^n operand: C_k is the K and the V operand
- `M_mat[Q^n]`: the absorbed path never expands K^n or V and never forms a Q^n operand: C_k is the K and the V operand
- `local[v_tile]`: the absorbed path never expands K^n or V and never forms a Q^n operand: C_k is the K and the V operand
- `local[k_nope_tile]`: the absorbed path never expands K^n or V and never forms a Q^n operand: C_k is the K and the V operand
- `local[q_nope_operand]`: the absorbed path never expands K^n or V and never forms a Q^n operand: C_k is the K and the V operand

<details><summary>Provenance (bindings, coverage, source, evidence)</summary>

| metric | source | coverage | bindings | evidence |
|---|---|---|---|---|
| local[q_latent_window] | derived | incomplete: depends on undeclared/unbound b_rq | `{"R_q": 16, "b_q": 16, "s_Cq": 2}` |  |
| local[wq_rank_tile] | derived | incomplete: depends on undeclared/unbound b_rq | `{"D_n": 24, "R_q": 16, "s_Wq": 2}` |  |
| local[q_proj_acc] | derived | incomplete: depends on undeclared/unbound projection_accumulator_dtype | `{"D_n": 24, "b_q": 16, "s_Qacc": 4}` |  |
| local[wk_full] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"D_n": 24, "R_k": 32, "s_Wk": 2}` |  |
| local[q_tilde_acc] | derived | incomplete: depends on undeclared/unbound projection_accumulator_dtype | `{"R_k": 32, "b_q": 16, "s_Qacc": 4}` |  |
| local[q_tilde] | derived | incomplete: depends on undeclared/unbound projected_operand_dtype, q_resident | `{"R_k": 32, "b_q": 16, "s_Qt": 2}` |  |
| local[kv_latent_tile] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"R_k": 32, "b_k": 32, "n_buf": 2, "s_Ck": 2}` |  |
| local[wv_full] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"D_v": 24, "R_k": 32, "s_Wv": 2}` |  |
| local[z_tile] | derived | incomplete: depends on undeclared/unbound projected_operand_dtype | `{"R_k": 32, "b_q": 16, "s_Z": 2}` |  |
| local[score_X] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"b_k": 32, "b_q": 16, "s_X": 4}` |  |
| local[exp_E] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"b_k": 32, "b_q": 16, "s_E": 4}` |  |
| local[p_operand] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"b_k": 32, "b_q": 16, "s_P": 2}` |  |
| local[row_state_m] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"b_q": 16, "s_state": 4}` |  |
| local[row_state_l] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"b_q": 16, "s_state": 4}` |  |
| local[row_alpha] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"b_q": 16, "s_state": 4}` |  |
| local[accumulator_A] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"R_k": 32, "b_q": 16, "s_A": 4}` |  |
| local[q_pe_tile] | derived | incomplete: depends on undeclared/unbound q_resident | `{"D_r": 8, "b_q": 16, "s_Qr": 2}` |  |
| local[k_pe_tile] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"D_r": 8, "b_k": 32, "n_buf": 2, "s_Kr": 2}` |  |
| local[fixed_scratch] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"s_scratch": 0}` |  |
| local[out_tile] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"D_v": 24, "b_q": 16, "s_O": 2}` |  |
| local[lse_tile] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"b_q": 16, "s_LSE": 4}` |  |
| layout[q_latent_window] | conditional | incomplete: depends on undeclared/unbound L_l, L_s, layout.q_latent_window | `{"R_q": 16, "b_q": 16, "s_Cq": 2}` |  |
| layout[wq_rank_tile] | conditional | incomplete: depends on undeclared/unbound L_l, L_s, layout.wq_rank_tile | `{"D_n": 24, "R_q": 16, "s_Wq": 2}` |  |
| layout[q_proj_acc] | conditional | incomplete: depends on undeclared/unbound L_l, L_s, layout.q_proj_acc | `{"D_n": 24, "b_q": 16, "s_Qacc": 4}` |  |
| layout[wk_full] | conditional | incomplete: depends on undeclared/unbound L_l, L_s, layout.wk_full | `{"D_n": 24, "R_k": 32, "s_Wk": 2}` |  |
| layout[q_tilde_acc] | conditional | incomplete: depends on undeclared/unbound L_l, L_s, layout.q_tilde_acc | `{"R_k": 32, "b_q": 16, "s_Qacc": 4}` |  |
| layout[q_tilde] | conditional | incomplete: depends on undeclared/unbound L_l, L_s, layout.q_tilde | `{"R_k": 32, "b_q": 16, "s_Qt": 2}` |  |
| layout[kv_latent_tile] | conditional | incomplete: depends on undeclared/unbound L_l, L_s, layout.kv_latent_tile | `{"R_k": 32, "b_k": 32, "n_buf": 2, "s_Ck": 2}` |  |
| layout[wv_full] | conditional | incomplete: depends on undeclared/unbound L_l, L_s, layout.wv_full | `{"D_v": 24, "R_k": 32, "s_Wv": 2}` |  |
| layout[z_tile] | conditional | incomplete: depends on undeclared/unbound L_l, L_s, layout.z_tile | `{"R_k": 32, "b_q": 16, "s_Z": 2}` |  |
| layout[score_X] | conditional | incomplete: depends on undeclared/unbound L_l, L_s, layout.score_X | `{"b_k": 32, "b_q": 16, "s_X": 4}` |  |
| layout[exp_E] | conditional | incomplete: depends on undeclared/unbound L_l, L_s, layout.exp_E | `{"b_k": 32, "b_q": 16, "s_E": 4}` |  |
| layout[p_operand] | conditional | incomplete: depends on undeclared/unbound L_l, L_s, layout.p_operand | `{"b_k": 32, "b_q": 16, "s_P": 2}` |  |
| layout[row_state_m] | conditional | incomplete: depends on undeclared/unbound L_l, L_s, layout.row_state_m | `{"b_q": 16, "s_state": 4}` |  |
| layout[row_state_l] | conditional | incomplete: depends on undeclared/unbound L_l, L_s, layout.row_state_l | `{"b_q": 16, "s_state": 4}` |  |
| layout[row_alpha] | conditional | incomplete: depends on undeclared/unbound L_l, L_s, layout.row_alpha | `{"b_q": 16, "s_state": 4}` |  |
| layout[accumulator_A] | conditional | incomplete: depends on undeclared/unbound L_l, L_s, layout.accumulator_A | `{"R_k": 32, "b_q": 16, "s_A": 4}` |  |
| layout[q_pe_tile] | conditional | incomplete: depends on undeclared/unbound L_l, L_s, layout.q_pe_tile | `{"D_r": 8, "b_q": 16, "s_Qr": 2}` |  |
| layout[k_pe_tile] | conditional | incomplete: depends on undeclared/unbound L_l, L_s, layout.k_pe_tile | `{"D_r": 8, "b_k": 32, "n_buf": 2, "s_Kr": 2}` |  |
| layout[fixed_scratch] | conditional | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"s_scratch": 0}` |  |
| layout[out_tile] | conditional | incomplete: depends on undeclared/unbound L_l, L_s, layout.out_tile | `{"D_v": 24, "b_q": 16, "s_O": 2}` |  |
| layout[lse_tile] | conditional | incomplete: depends on undeclared/unbound L_l, L_s, layout.lse_tile | `{"b_q": 16, "s_LSE": 4}` |  |
| HBM_mat[K^n,V] | definition | not applicable under the declared configuration | `{}` |  |
| M_mat[K^n,V] | definition | not applicable under the declared configuration | `{}` |  |
| HBM_mat[Q^n] | definition | not applicable under the declared configuration | `{}` |  |
| M_mat[Q^n] | definition | not applicable under the declared configuration | `{}` |  |
| local[v_tile] | definition | not applicable under the declared configuration | `{}` |  |
| local[k_nope_tile] | definition | not applicable under the declared configuration | `{}` |  |
| local[q_nope_operand] | definition | not applicable under the declared configuration | `{}` |  |

</details>

## 7. stage pressure and envelope

| metric | formula id | expression | value | unit | status | missing | scope | not equivalent to |
|---|---|---|---|---|---|---|---|---|
| M_live[all:q_proj] | `pressure.live.all.q_proj` | `D_n*R_q*s_Wq + D_n*b_q*s_Qacc + R_q*b_q*s_Cq + s_scratch` | 2816 | byte | partial | b_rq | source-level live set at this stage cut, level=all (aliases deduplicated) | compiler register allocation; VMEM allocation |
| M_live[all:q_absorb] | `pressure.live.all.q_absorb` | `D_n*R_k*s_Wk + D_n*b_q*s_Qacc + R_k*b_q*s_Qacc + s_scratch` | 5120 | byte | bound |  | source-level live set at this stage cut, level=all (aliases deduplicated) | compiler register allocation; VMEM allocation |
| M_live[all:score] | `pressure.live.all.score` | `D_r*b_k*n_buf*s_Kr + D_r*b_q*s_Qr + R_k*b_k*n_buf*s_Ck + R_k*b_q*s_Qt + 2*b_q*s_state + s_scratch + Max(R_k*b_q*s_A, R_k*b_q*s_Z) + Max(b_k*b_q*s_E, b_k*b_q*s_X)` | 10,624 | byte | bound |  | source-level live set at this stage cut, level=all (aliases deduplicated) | compiler register allocation; VMEM allocation |
| M_live[all:softmax] | `pressure.live.all.softmax` | `D_r*b_q*s_Qr + R_k*b_k*n_buf*s_Ck + R_k*b_q*s_Qt + 3*b_q*s_state + s_scratch + Max(R_k*b_q*s_A, R_k*b_q*s_Z) + Max(b_k*b_q*s_E, b_k*b_q*s_X)` | 9664 | byte | partial | q_resident | source-level live set at this stage cut, level=all (aliases deduplicated) | compiler register allocation; VMEM allocation |
| M_live[all:pv] | `pressure.live.all.pv` | `D_r*b_q*s_Qr + R_k*b_k*n_buf*s_Ck + R_k*b_q*s_Qt + b_k*b_q*s_P + 3*b_q*s_state + s_scratch + Max(R_k*b_q*s_A, R_k*b_q*s_Z) + Max(b_k*b_q*s_E, b_k*b_q*s_X)` | 10,688 | byte | partial | q_resident | source-level live set at this stage cut, level=all (aliases deduplicated) | compiler register allocation; VMEM allocation |
| M_live[all:finalize] | `pressure.live.all.finalize` | `D_r*b_q*s_Qr + D_v*R_k*s_Wv + D_v*b_q*s_O + R_k*b_q*s_Qt + b_q*s_LSE + 2*b_q*s_state + s_scratch + Max(R_k*b_q*s_A, R_k*b_q*s_Z)` | 5824 | byte | partial | q_resident | source-level live set at this stage cut, level=all (aliases deduplicated) | compiler register allocation; VMEM allocation |
| M_peak[all] | `pressure.peak.all` | `Max(D_n*R_k*s_Wk + D_n*b_q*s_Qacc + R_k*b_q*s_Qacc + s_scratch, D_n*R_q*s_Wq + D_n*b_q*s_Qacc + R_q*b_q*s_Cq + s_scratch, D_r*b_q*s_Qr + D_v*R_k*s_Wv + D_v*b_q*s_O + R_k*b_q*s_Qt + b_q*s_LSE + 2*b_q*s_state + s_scratch + Max(R_k*b_q*s_A, R_k*b_q*s_Z), D_r*b_q*s_Qr + R_k*b_k*n_buf*s_Ck + R_k*b_q*s_Qt + b_k*b_q*s_P + 3*b_q*s_state + s_scratch + Max(R_k*b_q*s_A, R_k*b_q*s_Z) + Max(b_k*b_q*s_E, b_k*b_q*s_X), D_r*b_k*n_buf*s_Kr + D_r*b_q*s_Qr + R_k*b_k*n_buf*s_Ck + R_k*b_q*s_Qt + 2*b_q*s_state + s_scratch + Max(R_k*b_q*s_A, R_k*b_q*s_Z) + Max(b_k*b_q*s_E, b_k*b_q*s_X))` | 10,688 | byte | partial | b_rq, q_resident | max over stage cuts, level=all (source level) | actual spill; OOM proof |
| M_live[vreg:q_proj] | `pressure.live.vreg.q_proj` | `D_n*b_q*s_Qacc` | 1536 | byte | bound |  | source-level live set at this stage cut, level=vreg (aliases deduplicated) | compiler register allocation; VMEM allocation |
| M_live[vreg:q_absorb] | `pressure.live.vreg.q_absorb` | `D_n*b_q*s_Qacc + R_k*b_q*s_Qacc` | 3584 | byte | bound |  | source-level live set at this stage cut, level=vreg (aliases deduplicated) | compiler register allocation; VMEM allocation |
| M_live[vreg:score] | `pressure.live.vreg.score` | `2*b_q*s_state + Max(R_k*b_q*s_A, R_k*b_q*s_Z) + Max(b_k*b_q*s_E, b_k*b_q*s_X)` | 4224 | byte | bound |  | source-level live set at this stage cut, level=vreg (aliases deduplicated) | compiler register allocation; VMEM allocation |
| M_live[vreg:softmax] | `pressure.live.vreg.softmax` | `3*b_q*s_state + Max(R_k*b_q*s_A, R_k*b_q*s_Z) + Max(b_k*b_q*s_E, b_k*b_q*s_X)` | 4288 | byte | bound |  | source-level live set at this stage cut, level=vreg (aliases deduplicated) | compiler register allocation; VMEM allocation |
| M_live[vreg:pv] | `pressure.live.vreg.pv` | `b_k*b_q*s_P + 3*b_q*s_state + Max(R_k*b_q*s_A, R_k*b_q*s_Z) + Max(b_k*b_q*s_E, b_k*b_q*s_X)` | 5312 | byte | bound |  | source-level live set at this stage cut, level=vreg (aliases deduplicated) | compiler register allocation; VMEM allocation |
| M_live[vreg:finalize] | `pressure.live.vreg.finalize` | `2*b_q*s_state + Max(R_k*b_q*s_A, R_k*b_q*s_Z)` | 2176 | byte | bound |  | source-level live set at this stage cut, level=vreg (aliases deduplicated) | compiler register allocation; VMEM allocation |
| M_peak[vreg] | `pressure.peak.vreg` | `Max(D_n*b_q*s_Qacc + R_k*b_q*s_Qacc, b_k*b_q*s_P + 3*b_q*s_state + Max(R_k*b_q*s_A, R_k*b_q*s_Z) + Max(b_k*b_q*s_E, b_k*b_q*s_X))` | 5312 | byte | bound |  | max over stage cuts, level=vreg (source level) | actual spill; OOM proof |
| M_live[vmem:q_proj] | `pressure.live.vmem.q_proj` | `D_n*R_q*s_Wq + R_q*b_q*s_Cq + s_scratch` | 1280 | byte | partial | b_rq | source-level live set at this stage cut, level=vmem (aliases deduplicated) | compiler register allocation; VMEM allocation |
| M_live[vmem:q_absorb] | `pressure.live.vmem.q_absorb` | `D_n*R_k*s_Wk + s_scratch` | 1536 | byte | bound |  | source-level live set at this stage cut, level=vmem (aliases deduplicated) | compiler register allocation; VMEM allocation |
| M_live[vmem:score] | `pressure.live.vmem.score` | `D_r*b_k*n_buf*s_Kr + D_r*b_q*s_Qr + R_k*b_k*n_buf*s_Ck + R_k*b_q*s_Qt + s_scratch` | 6400 | byte | bound |  | source-level live set at this stage cut, level=vmem (aliases deduplicated) | compiler register allocation; VMEM allocation |
| M_live[vmem:softmax] | `pressure.live.vmem.softmax` | `D_r*b_q*s_Qr + R_k*b_k*n_buf*s_Ck + R_k*b_q*s_Qt + s_scratch` | 5376 | byte | partial | q_resident | source-level live set at this stage cut, level=vmem (aliases deduplicated) | compiler register allocation; VMEM allocation |
| M_live[vmem:pv] | `pressure.live.vmem.pv` | `D_r*b_q*s_Qr + R_k*b_k*n_buf*s_Ck + R_k*b_q*s_Qt + s_scratch` | 5376 | byte | partial | q_resident | source-level live set at this stage cut, level=vmem (aliases deduplicated) | compiler register allocation; VMEM allocation |
| M_live[vmem:finalize] | `pressure.live.vmem.finalize` | `D_r*b_q*s_Qr + D_v*R_k*s_Wv + D_v*b_q*s_O + R_k*b_q*s_Qt + b_q*s_LSE + s_scratch` | 3648 | byte | partial | q_resident | source-level live set at this stage cut, level=vmem (aliases deduplicated) | compiler register allocation; VMEM allocation |
| M_peak[vmem] | `pressure.peak.vmem` | `Max(D_n*R_k*s_Wk + s_scratch, D_n*R_q*s_Wq + R_q*b_q*s_Cq + s_scratch, D_r*b_k*n_buf*s_Kr + D_r*b_q*s_Qr + R_k*b_k*n_buf*s_Ck + R_k*b_q*s_Qt + s_scratch, D_r*b_q*s_Qr + D_v*R_k*s_Wv + D_v*b_q*s_O + R_k*b_q*s_Qt + b_q*s_LSE + s_scratch)` | 6400 | byte | partial | b_rq, q_resident | max over stage cuts, level=vmem (source level) | actual spill; OOM proof |
| envelope[all:q_proj] | `pressure.envelope.all.q_proj` | `D_n*R_q*s_Wq + b_q*(D_n*s_Qacc + R_q*s_Cq) + s_scratch` | 2816 | byte | partial | b_rq | a*b_q*b_k + b*b_q + c*b_k + d (+ remainder) for this stage, level=all | measured spill; spill probability |
| envelope[all:q_absorb] | `pressure.envelope.all.q_absorb` | `D_n*R_k*s_Wk + b_q*s_Qacc*(D_n + R_k) + s_scratch` | 5120 | byte | bound |  | a*b_q*b_k + b*b_q + c*b_k + d (+ remainder) for this stage, level=all | measured spill; spill probability |
| envelope[all:score] | `pressure.envelope.all.score` | `b_k*b_q*Max(s_E, s_X) + b_k*n_buf*(D_r*s_Kr + R_k*s_Ck) + b_q*(D_r*s_Qr + R_k*s_Qt + R_k*Max(s_A, s_Z) + 2*s_state) + s_scratch` | 10,624 | byte | bound |  | a*b_q*b_k + b*b_q + c*b_k + d (+ remainder) for this stage, level=all | measured spill; spill probability |
| envelope[all:softmax] | `pressure.envelope.all.softmax` | `R_k*b_k*n_buf*s_Ck + b_k*b_q*Max(s_E, s_X) + b_q*(D_r*s_Qr + R_k*s_Qt + R_k*Max(s_A, s_Z) + 3*s_state) + s_scratch` | 9664 | byte | partial | q_resident | a*b_q*b_k + b*b_q + c*b_k + d (+ remainder) for this stage, level=all | measured spill; spill probability |
| envelope[all:pv] | `pressure.envelope.all.pv` | `R_k*b_k*n_buf*s_Ck + b_k*b_q*(s_P + Max(s_E, s_X)) + b_q*(D_r*s_Qr + R_k*s_Qt + R_k*Max(s_A, s_Z) + 3*s_state) + s_scratch` | 10,688 | byte | partial | q_resident | a*b_q*b_k + b*b_q + c*b_k + d (+ remainder) for this stage, level=all | measured spill; spill probability |
| envelope[all:finalize] | `pressure.envelope.all.finalize` | `D_v*R_k*s_Wv + b_q*(D_r*s_Qr + D_v*s_O + R_k*s_Qt + R_k*Max(s_A, s_Z) + s_LSE + 2*s_state) + s_scratch` | 5824 | byte | partial | q_resident | a*b_q*b_k + b*b_q + c*b_k + d (+ remainder) for this stage, level=all | measured spill; spill probability |
| envelope[vreg:q_proj] | `pressure.envelope.vreg.q_proj` | `D_n*b_q*s_Qacc` | 1536 | byte | bound |  | a*b_q*b_k + b*b_q + c*b_k + d (+ remainder) for this stage, level=vreg | measured spill; spill probability |
| envelope[vreg:q_absorb] | `pressure.envelope.vreg.q_absorb` | `b_q*s_Qacc*(D_n + R_k)` | 3584 | byte | bound |  | a*b_q*b_k + b*b_q + c*b_k + d (+ remainder) for this stage, level=vreg | measured spill; spill probability |
| envelope[vreg:score] | `pressure.envelope.vreg.score` | `b_k*b_q*Max(s_E, s_X) + b_q*(R_k*Max(s_A, s_Z) + 2*s_state)` | 4224 | byte | bound |  | a*b_q*b_k + b*b_q + c*b_k + d (+ remainder) for this stage, level=vreg | measured spill; spill probability |
| envelope[vreg:softmax] | `pressure.envelope.vreg.softmax` | `b_k*b_q*Max(s_E, s_X) + b_q*(R_k*Max(s_A, s_Z) + 3*s_state)` | 4288 | byte | bound |  | a*b_q*b_k + b*b_q + c*b_k + d (+ remainder) for this stage, level=vreg | measured spill; spill probability |
| envelope[vreg:pv] | `pressure.envelope.vreg.pv` | `b_k*b_q*(s_P + Max(s_E, s_X)) + b_q*(R_k*Max(s_A, s_Z) + 3*s_state)` | 5312 | byte | bound |  | a*b_q*b_k + b*b_q + c*b_k + d (+ remainder) for this stage, level=vreg | measured spill; spill probability |
| envelope[vreg:finalize] | `pressure.envelope.vreg.finalize` | `b_q*(R_k*Max(s_A, s_Z) + 2*s_state)` | 2176 | byte | bound |  | a*b_q*b_k + b*b_q + c*b_k + d (+ remainder) for this stage, level=vreg | measured spill; spill probability |
| envelope[vmem:q_proj] | `pressure.envelope.vmem.q_proj` | `D_n*R_q*s_Wq + R_q*b_q*s_Cq + s_scratch` | 1280 | byte | partial | b_rq | a*b_q*b_k + b*b_q + c*b_k + d (+ remainder) for this stage, level=vmem | measured spill; spill probability |
| envelope[vmem:q_absorb] | `pressure.envelope.vmem.q_absorb` | `D_n*R_k*s_Wk + s_scratch` | 1536 | byte | bound |  | a*b_q*b_k + b*b_q + c*b_k + d (+ remainder) for this stage, level=vmem | measured spill; spill probability |
| envelope[vmem:score] | `pressure.envelope.vmem.score` | `b_k*n_buf*(D_r*s_Kr + R_k*s_Ck) + b_q*(D_r*s_Qr + R_k*s_Qt) + s_scratch` | 6400 | byte | bound |  | a*b_q*b_k + b*b_q + c*b_k + d (+ remainder) for this stage, level=vmem | measured spill; spill probability |
| envelope[vmem:softmax] | `pressure.envelope.vmem.softmax` | `R_k*b_k*n_buf*s_Ck + b_q*(D_r*s_Qr + R_k*s_Qt) + s_scratch` | 5376 | byte | partial | q_resident | a*b_q*b_k + b*b_q + c*b_k + d (+ remainder) for this stage, level=vmem | measured spill; spill probability |
| envelope[vmem:pv] | `pressure.envelope.vmem.pv` | `R_k*b_k*n_buf*s_Ck + b_q*(D_r*s_Qr + R_k*s_Qt) + s_scratch` | 5376 | byte | partial | q_resident | a*b_q*b_k + b*b_q + c*b_k + d (+ remainder) for this stage, level=vmem | measured spill; spill probability |
| envelope[vmem:finalize] | `pressure.envelope.vmem.finalize` | `D_v*R_k*s_Wv + b_q*(D_r*s_Qr + D_v*s_O + R_k*s_Qt + s_LSE) + s_scratch` | 3648 | byte | partial | q_resident | a*b_q*b_k + b*b_q + c*b_k + d (+ remainder) for this stage, level=vmem | measured spill; spill probability |
| V_lifetime_stages | `lifetime.v_tile.stages` | `\|{s : v_tile live at s}\|` |  | stages | not_applicable |  | number of stage cuts the V tile is live at | measured prefetch behaviour |
| V_prefetch_overlapped | `lifetime.v_tile.prefetch` | `v_tile live at the score stage` |  | bool | not_applicable |  | V read/production overlapped with the score matmul | measured prefetch behaviour |
| b_k_max[vreg:q_proj] | `pressure.feasible.vreg.q_proj.b_k` | `floor((R - b b_q - d)/(a b_q + c)), stated only for a b_q + c > 0` |  | rows | not_applicable |  | the q_proj stage at level vreg has no b_k term (a b_q + c = 0), so it places no bound on b_k | an unbounded b_k: other stages still constrain it |
| excess_over_budget[vreg:q_proj] | `pressure.capacity_constraint.vreg.q_proj` | `Max(0, D_n*b_q*s_Qacc - R_vreg) → Max(0, 1536 - R_vreg)` |  | byte | partial | vreg_budget_bytes | max(0, M_live - R) at the q_proj stage, level vreg: data that cannot stay at the level under the source-level model | spill write traffic; compiled live set |
| b_k_max[vreg:q_absorb] | `pressure.feasible.vreg.q_absorb.b_k` | `floor((R - b b_q - d)/(a b_q + c)), stated only for a b_q + c > 0` |  | rows | not_applicable |  | the q_absorb stage at level vreg has no b_k term (a b_q + c = 0), so it places no bound on b_k | an unbounded b_k: other stages still constrain it |
| excess_over_budget[vreg:q_absorb] | `pressure.capacity_constraint.vreg.q_absorb` | `Max(0, -R_vreg + b_q*s_Qacc*(D_n + R_k)) → Max(0, 3584 - R_vreg)` |  | byte | partial | vreg_budget_bytes | max(0, M_live - R) at the q_absorb stage, level vreg: data that cannot stay at the level under the source-level model | spill write traffic; compiled live set |
| b_k_max[vreg:score] | `pressure.feasible.vreg.score.b_k` | `floor((R_vreg - b_q*(R_k*Max(s_A, s_Z) + 2*s_state))/(b_q*Max(s_E, s_X))) → floor(R_vreg/64) - 34` |  | rows | partial | vreg_budget_bytes | floor((R - b b_q - d)/(a b_q + c)) for the score stage at level vreg under the declared budget R_vreg |  |
| excess_over_budget[vreg:score] | `pressure.capacity_constraint.vreg.score` | `Max(0, -R_vreg + b_k*b_q*Max(s_E, s_X) + b_q*(R_k*Max(s_A, s_Z) + 2*s_state)) → Max(0, 4224 - R_vreg)` |  | byte | partial | vreg_budget_bytes | max(0, M_live - R) at the score stage, level vreg: data that cannot stay at the level under the source-level model | spill write traffic; compiled live set |
| b_k_max[vreg:softmax] | `pressure.feasible.vreg.softmax.b_k` | `floor((R_vreg - b_q*(R_k*Max(s_A, s_Z) + 3*s_state))/(b_q*Max(s_E, s_X))) → floor(R_vreg/64) - 35` |  | rows | partial | vreg_budget_bytes | floor((R - b b_q - d)/(a b_q + c)) for the softmax stage at level vreg under the declared budget R_vreg |  |
| excess_over_budget[vreg:softmax] | `pressure.capacity_constraint.vreg.softmax` | `Max(0, -R_vreg + b_k*b_q*Max(s_E, s_X) + b_q*(R_k*Max(s_A, s_Z) + 3*s_state)) → Max(0, 4288 - R_vreg)` |  | byte | partial | vreg_budget_bytes | max(0, M_live - R) at the softmax stage, level vreg: data that cannot stay at the level under the source-level model | spill write traffic; compiled live set |
| b_k_max[vreg:pv] | `pressure.feasible.vreg.pv.b_k` | `floor((R_vreg - b_q*(R_k*Max(s_A, s_Z) + 3*s_state))/(b_q*(s_P + Max(s_E, s_X)))) → floor(R_vreg/96 - 70/3)` |  | rows | partial | vreg_budget_bytes | floor((R - b b_q - d)/(a b_q + c)) for the pv stage at level vreg under the declared budget R_vreg |  |
| excess_over_budget[vreg:pv] | `pressure.capacity_constraint.vreg.pv` | `Max(0, -R_vreg + b_k*b_q*(s_P + Max(s_E, s_X)) + b_q*(R_k*Max(s_A, s_Z) + 3*s_state)) → Max(0, 5312 - R_vreg)` |  | byte | partial | vreg_budget_bytes | max(0, M_live - R) at the pv stage, level vreg: data that cannot stay at the level under the source-level model | spill write traffic; compiled live set |
| b_k_max[vreg:finalize] | `pressure.feasible.vreg.finalize.b_k` | `floor((R - b b_q - d)/(a b_q + c)), stated only for a b_q + c > 0` |  | rows | not_applicable |  | the finalize stage at level vreg has no b_k term (a b_q + c = 0), so it places no bound on b_k | an unbounded b_k: other stages still constrain it |
| excess_over_budget[vreg:finalize] | `pressure.capacity_constraint.vreg.finalize` | `Max(0, -R_vreg + b_q*(R_k*Max(s_A, s_Z) + 2*s_state)) → Max(0, 2176 - R_vreg)` |  | byte | partial | vreg_budget_bytes | max(0, M_live - R) at the finalize stage, level vreg: data that cannot stay at the level under the source-level model | spill write traffic; compiled live set |
| b_k_max[vreg:binding_stage] | `pressure.feasible.vreg.binding` | `min over stages s of floor((R - b_s b_q - d_s)/(a_s b_q + c_s))` |  | rows | unknown | vreg_budget_bytes | tightest per-stage capacity bound at level vreg | compiled spill/OOM criterion |
| b_k_max[vmem:q_proj] | `pressure.feasible.vmem.q_proj.b_k` | `floor((R - b b_q - d)/(a b_q + c)), stated only for a b_q + c > 0` |  | rows | not_applicable | b_rq | the q_proj stage at level vmem has no b_k term (a b_q + c = 0), so it places no bound on b_k | an unbounded b_k: other stages still constrain it |
| excess_over_budget[vmem:q_proj] | `pressure.capacity_constraint.vmem.q_proj` | `Max(0, D_n*R_q*s_Wq + R_q*b_q*s_Cq - R_vmem + s_scratch) → Max(0, 1280 - R_vmem)` |  | byte | partial | b_rq, vmem_budget_bytes | max(0, M_live - R) at the q_proj stage, level vmem: data that cannot stay at the level under the source-level model | spill write traffic; compiled live set |
| b_k_max[vmem:q_absorb] | `pressure.feasible.vmem.q_absorb.b_k` | `floor((R - b b_q - d)/(a b_q + c)), stated only for a b_q + c > 0` |  | rows | not_applicable |  | the q_absorb stage at level vmem has no b_k term (a b_q + c = 0), so it places no bound on b_k | an unbounded b_k: other stages still constrain it |
| excess_over_budget[vmem:q_absorb] | `pressure.capacity_constraint.vmem.q_absorb` | `Max(0, D_n*R_k*s_Wk - R_vmem + s_scratch) → Max(0, 1536 - R_vmem)` |  | byte | partial | vmem_budget_bytes | max(0, M_live - R) at the q_absorb stage, level vmem: data that cannot stay at the level under the source-level model | spill write traffic; compiled live set |
| b_k_max[vmem:score] | `pressure.feasible.vmem.score.b_k` | `floor((R_vmem - b_q*(D_r*s_Qr + R_k*s_Qt) - s_scratch)/(n_buf*(D_r*s_Kr + R_k*s_Ck))) → floor(R_vmem/160) - 8` |  | rows | partial | vmem_budget_bytes | floor((R - b b_q - d)/(a b_q + c)) for the score stage at level vmem under the declared budget R_vmem |  |
| excess_over_budget[vmem:score] | `pressure.capacity_constraint.vmem.score` | `Max(0, -R_vmem + b_k*n_buf*(D_r*s_Kr + R_k*s_Ck) + b_q*(D_r*s_Qr + R_k*s_Qt) + s_scratch) → Max(0, 6400 - R_vmem)` |  | byte | partial | vmem_budget_bytes | max(0, M_live - R) at the score stage, level vmem: data that cannot stay at the level under the source-level model | spill write traffic; compiled live set |
| b_k_max[vmem:softmax] | `pressure.feasible.vmem.softmax.b_k` | `floor((R_vmem - b_q*(D_r*s_Qr + R_k*s_Qt) - s_scratch)/(R_k*n_buf*s_Ck)) → floor(R_vmem/128) - 10` |  | rows | partial | q_resident, vmem_budget_bytes | floor((R - b b_q - d)/(a b_q + c)) for the softmax stage at level vmem under the declared budget R_vmem |  |
| excess_over_budget[vmem:softmax] | `pressure.capacity_constraint.vmem.softmax` | `Max(0, R_k*b_k*n_buf*s_Ck - R_vmem + b_q*(D_r*s_Qr + R_k*s_Qt) + s_scratch) → Max(0, 5376 - R_vmem)` |  | byte | partial | q_resident, vmem_budget_bytes | max(0, M_live - R) at the softmax stage, level vmem: data that cannot stay at the level under the source-level model | spill write traffic; compiled live set |
| b_k_max[vmem:pv] | `pressure.feasible.vmem.pv.b_k` | `floor((R_vmem - b_q*(D_r*s_Qr + R_k*s_Qt) - s_scratch)/(R_k*n_buf*s_Ck)) → floor(R_vmem/128) - 10` |  | rows | partial | q_resident, vmem_budget_bytes | floor((R - b b_q - d)/(a b_q + c)) for the pv stage at level vmem under the declared budget R_vmem |  |
| excess_over_budget[vmem:pv] | `pressure.capacity_constraint.vmem.pv` | `Max(0, R_k*b_k*n_buf*s_Ck - R_vmem + b_q*(D_r*s_Qr + R_k*s_Qt) + s_scratch) → Max(0, 5376 - R_vmem)` |  | byte | partial | q_resident, vmem_budget_bytes | max(0, M_live - R) at the pv stage, level vmem: data that cannot stay at the level under the source-level model | spill write traffic; compiled live set |
| b_k_max[vmem:finalize] | `pressure.feasible.vmem.finalize.b_k` | `floor((R - b b_q - d)/(a b_q + c)), stated only for a b_q + c > 0` |  | rows | not_applicable | q_resident | the finalize stage at level vmem has no b_k term (a b_q + c = 0), so it places no bound on b_k | an unbounded b_k: other stages still constrain it |
| excess_over_budget[vmem:finalize] | `pressure.capacity_constraint.vmem.finalize` | `Max(0, D_v*R_k*s_Wv - R_vmem + b_q*(D_r*s_Qr + D_v*s_O + R_k*s_Qt + s_LSE) + s_scratch) → Max(0, 3648 - R_vmem)` |  | byte | partial | q_resident, vmem_budget_bytes | max(0, M_live - R) at the finalize stage, level vmem: data that cannot stay at the level under the source-level model | spill write traffic; compiled live set |
| b_k_max[vmem:binding_stage] | `pressure.feasible.vmem.binding` | `min over stages s of floor((R - b_s b_q - d_s)/(a_s b_q + c_s))` |  | rows | unknown | b_rq, q_resident, vmem_budget_bytes | tightest per-stage capacity bound at level vmem | compiled spill/OOM criterion |
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
- `M_live[all:q_absorb]`: objects: q_proj_acc, wk_full, q_tilde_acc, fixed_scratch
- `M_live[all:score]`: objects: q_tilde, kv_latent_tile, row_state_m, row_state_l, q_pe_tile, k_pe_tile, fixed_scratch, alias[score_or_E]:score_X (sized by the whole group: score_X|exp_E), alias[acc_or_Z]:accumulator_A (sized by the whole group: z_tile|accumulator_A)
- `M_live[all:softmax]`: objects: q_tilde, kv_latent_tile, row_state_m, row_state_l, row_alpha, q_pe_tile, fixed_scratch, alias[score_or_E]:score_X|exp_E, alias[acc_or_Z]:accumulator_A (sized by the whole group: z_tile|accumulator_A)
- `M_live[all:softmax]`: includes objects whose existence depends on undeclared fields: ['q_resident']
- `M_live[all:pv]`: objects: q_tilde, kv_latent_tile, p_operand, row_state_m, row_state_l, row_alpha, q_pe_tile, fixed_scratch, alias[score_or_E]:exp_E (sized by the whole group: score_X|exp_E), alias[acc_or_Z]:accumulator_A (sized by the whole group: z_tile|accumulator_A)
- `M_live[all:pv]`: includes objects whose existence depends on undeclared fields: ['q_resident']
- `M_live[all:finalize]`: objects: q_tilde, wv_full, row_state_m, row_state_l, q_pe_tile, fixed_scratch, out_tile, lse_tile, alias[acc_or_Z]:z_tile|accumulator_A
- `M_live[all:finalize]`: includes objects whose existence depends on undeclared fields: ['q_resident']
- `M_live[vreg:q_proj]`: objects: q_proj_acc
- `M_live[vreg:q_absorb]`: objects: q_proj_acc, q_tilde_acc
- `M_live[vreg:score]`: objects: row_state_m, row_state_l, alias[score_or_E]:score_X (sized by the whole group: score_X|exp_E), alias[acc_or_Z]:accumulator_A (sized by the whole group: z_tile|accumulator_A)
- `M_live[vreg:softmax]`: objects: row_state_m, row_state_l, row_alpha, alias[score_or_E]:score_X|exp_E, alias[acc_or_Z]:accumulator_A (sized by the whole group: z_tile|accumulator_A)
- `M_live[vreg:pv]`: objects: p_operand, row_state_m, row_state_l, row_alpha, alias[score_or_E]:exp_E (sized by the whole group: score_X|exp_E), alias[acc_or_Z]:accumulator_A (sized by the whole group: z_tile|accumulator_A)
- `M_live[vreg:finalize]`: objects: row_state_m, row_state_l, alias[acc_or_Z]:z_tile|accumulator_A
- `M_live[vmem:q_proj]`: objects: q_latent_window, wq_rank_tile, fixed_scratch
- `M_live[vmem:q_proj]`: includes objects whose existence depends on undeclared fields: ['b_rq']
- `M_live[vmem:q_absorb]`: objects: wk_full, fixed_scratch
- `M_live[vmem:score]`: objects: q_tilde, kv_latent_tile, q_pe_tile, k_pe_tile, fixed_scratch
- `M_live[vmem:softmax]`: objects: q_tilde, kv_latent_tile, q_pe_tile, fixed_scratch
- `M_live[vmem:softmax]`: includes objects whose existence depends on undeclared fields: ['q_resident']
- `M_live[vmem:pv]`: objects: q_tilde, kv_latent_tile, q_pe_tile, fixed_scratch
- `M_live[vmem:pv]`: includes objects whose existence depends on undeclared fields: ['q_resident']
- `M_live[vmem:finalize]`: objects: q_tilde, wv_full, q_pe_tile, fixed_scratch, out_tile, lse_tile
- `M_live[vmem:finalize]`: includes objects whose existence depends on undeclared fields: ['q_resident']
- `envelope[all:q_proj]`: a from []
- `envelope[all:q_proj]`: b from ['q_latent_window', 'q_proj_acc']
- `envelope[all:q_proj]`: c from []
- `envelope[all:q_proj]`: d from ['wq_rank_tile', 'fixed_scratch']
- `envelope[all:q_absorb]`: a from []
- `envelope[all:q_absorb]`: b from ['q_proj_acc', 'q_tilde_acc']
- `envelope[all:q_absorb]`: c from []
- `envelope[all:q_absorb]`: d from ['wk_full', 'fixed_scratch']
- `envelope[all:score]`: a from ['alias[score_or_E]:score_X|exp_E']
- `envelope[all:score]`: b from ['q_tilde', 'row_state_m', 'row_state_l', 'q_pe_tile', 'alias[acc_or_Z]:z_tile|accumulator_A']
- `envelope[all:score]`: c from ['kv_latent_tile', 'k_pe_tile']
- `envelope[all:score]`: d from ['fixed_scratch']
- `envelope[all:softmax]`: a from ['alias[score_or_E]:score_X|exp_E']
- `envelope[all:softmax]`: b from ['q_tilde', 'row_state_m', 'row_state_l', 'row_alpha', 'q_pe_tile', 'alias[acc_or_Z]:z_tile|accumulator_A']
- `envelope[all:softmax]`: c from ['kv_latent_tile']
- `envelope[all:softmax]`: d from ['fixed_scratch']
- `envelope[all:pv]`: a from ['p_operand', 'alias[score_or_E]:score_X|exp_E']
- `envelope[all:pv]`: b from ['q_tilde', 'row_state_m', 'row_state_l', 'row_alpha', 'q_pe_tile', 'alias[acc_or_Z]:z_tile|accumulator_A']
- `envelope[all:pv]`: c from ['kv_latent_tile']
- `envelope[all:pv]`: d from ['fixed_scratch']
- `envelope[all:finalize]`: a from []
- `envelope[all:finalize]`: b from ['q_tilde', 'row_state_m', 'row_state_l', 'q_pe_tile', 'out_tile', 'lse_tile', 'alias[acc_or_Z]:z_tile|accumulator_A']
- `envelope[all:finalize]`: c from []
- `envelope[all:finalize]`: d from ['wv_full', 'fixed_scratch']
- `envelope[vreg:q_proj]`: a from []
- `envelope[vreg:q_proj]`: b from ['q_proj_acc']
- `envelope[vreg:q_proj]`: c from []
- `envelope[vreg:q_proj]`: d from []
- `envelope[vreg:q_absorb]`: a from []
- `envelope[vreg:q_absorb]`: b from ['q_proj_acc', 'q_tilde_acc']
- `envelope[vreg:q_absorb]`: c from []
- `envelope[vreg:q_absorb]`: d from []
- `envelope[vreg:score]`: a from ['alias[score_or_E]:score_X|exp_E']
- `envelope[vreg:score]`: b from ['row_state_m', 'row_state_l', 'alias[acc_or_Z]:z_tile|accumulator_A']
- `envelope[vreg:score]`: c from []
- `envelope[vreg:score]`: d from []
- `envelope[vreg:softmax]`: a from ['alias[score_or_E]:score_X|exp_E']
- `envelope[vreg:softmax]`: b from ['row_state_m', 'row_state_l', 'row_alpha', 'alias[acc_or_Z]:z_tile|accumulator_A']
- `envelope[vreg:softmax]`: c from []
- `envelope[vreg:softmax]`: d from []
- `envelope[vreg:pv]`: a from ['p_operand', 'alias[score_or_E]:score_X|exp_E']
- `envelope[vreg:pv]`: b from ['row_state_m', 'row_state_l', 'row_alpha', 'alias[acc_or_Z]:z_tile|accumulator_A']
- `envelope[vreg:pv]`: c from []
- `envelope[vreg:pv]`: d from []
- `envelope[vreg:finalize]`: a from []
- `envelope[vreg:finalize]`: b from ['row_state_m', 'row_state_l', 'alias[acc_or_Z]:z_tile|accumulator_A']
- `envelope[vreg:finalize]`: c from []
- `envelope[vreg:finalize]`: d from []
- `envelope[vmem:q_proj]`: a from []
- `envelope[vmem:q_proj]`: b from ['q_latent_window']
- `envelope[vmem:q_proj]`: c from []
- `envelope[vmem:q_proj]`: d from ['wq_rank_tile', 'fixed_scratch']
- `envelope[vmem:q_absorb]`: a from []
- `envelope[vmem:q_absorb]`: b from []
- `envelope[vmem:q_absorb]`: c from []
- `envelope[vmem:q_absorb]`: d from ['wk_full', 'fixed_scratch']
- `envelope[vmem:score]`: a from []
- `envelope[vmem:score]`: b from ['q_tilde', 'q_pe_tile']
- `envelope[vmem:score]`: c from ['kv_latent_tile', 'k_pe_tile']
- `envelope[vmem:score]`: d from ['fixed_scratch']
- `envelope[vmem:softmax]`: a from []
- `envelope[vmem:softmax]`: b from ['q_tilde', 'q_pe_tile']
- `envelope[vmem:softmax]`: c from ['kv_latent_tile']
- `envelope[vmem:softmax]`: d from ['fixed_scratch']
- `envelope[vmem:pv]`: a from []
- `envelope[vmem:pv]`: b from ['q_tilde', 'q_pe_tile']
- `envelope[vmem:pv]`: c from ['kv_latent_tile']
- `envelope[vmem:pv]`: d from ['fixed_scratch']
- `envelope[vmem:finalize]`: a from []
- `envelope[vmem:finalize]`: b from ['q_tilde', 'q_pe_tile', 'out_tile', 'lse_tile']
- `envelope[vmem:finalize]`: c from []
- `envelope[vmem:finalize]`: d from ['wv_full', 'fixed_scratch']
- `V_lifetime_stages`: this variant has no separate V tile: C_k is both the K and the V operand
- `V_prefetch_overlapped`: this variant has no separate V tile: C_k is both the K and the V operand
- `b_k_max[vreg:q_proj]`: this stage's live set does not grow with b_k; the binding bound comes from another stage
- `excess_over_budget[vreg:q_proj]`: source-level logical live set; layout coverage not applied; not a compiled graph cut (spec 8.2 caveat)
- `b_k_max[vreg:q_absorb]`: this stage's live set does not grow with b_k; the binding bound comes from another stage
- `excess_over_budget[vreg:q_absorb]`: source-level logical live set; layout coverage not applied; not a compiled graph cut (spec 8.2 caveat)
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
- `b_k_max[vmem:q_absorb]`: this stage's live set does not grow with b_k; the binding bound comes from another stage
- `excess_over_budget[vmem:q_absorb]`: source-level logical live set; layout coverage not applied; not a compiled graph cut (spec 8.2 caveat)
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
| M_live[all:q_absorb] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"D_n": 24, "R_k": 32, "b_q": 16, "s_Qacc": 4, "s_Wk": 2, "s_scratch": 0}` |  |
| M_live[all:score] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"D_r": 8, "R_k": 32, "b_k": 32, "b_q": 16, "n_buf": 2, "s_A": 4, "s_Ck": 2, "s_E": 4, ...` |  |
| M_live[all:softmax] | derived | incomplete: depends on undeclared/unbound q_resident | `{"D_r": 8, "R_k": 32, "b_k": 32, "b_q": 16, "n_buf": 2, "s_A": 4, "s_Ck": 2, "s_E": 4, ...` |  |
| M_live[all:pv] | derived | incomplete: depends on undeclared/unbound q_resident | `{"D_r": 8, "R_k": 32, "b_k": 32, "b_q": 16, "n_buf": 2, "s_A": 4, "s_Ck": 2, "s_E": 4, ...` |  |
| M_live[all:finalize] | derived | incomplete: depends on undeclared/unbound q_resident | `{"D_r": 8, "D_v": 24, "R_k": 32, "b_q": 16, "s_A": 4, "s_LSE": 4, "s_O": 2, "s_Qr": 2, ...` |  |
| M_peak[all] | derived | incomplete: depends on undeclared/unbound b_rq, q_resident | `{"D_n": 24, "D_r": 8, "D_v": 24, "R_k": 32, "R_q": 16, "b_k": 32, "b_q": 16, "n_buf": 2...` |  |
| M_live[vreg:q_proj] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"D_n": 24, "b_q": 16, "s_Qacc": 4}` |  |
| M_live[vreg:q_absorb] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"D_n": 24, "R_k": 32, "b_q": 16, "s_Qacc": 4}` |  |
| M_live[vreg:score] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"R_k": 32, "b_k": 32, "b_q": 16, "s_A": 4, "s_E": 4, "s_X": 4, "s_Z": 2, "s_state": 4}` |  |
| M_live[vreg:softmax] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"R_k": 32, "b_k": 32, "b_q": 16, "s_A": 4, "s_E": 4, "s_X": 4, "s_Z": 2, "s_state": 4}` |  |
| M_live[vreg:pv] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"R_k": 32, "b_k": 32, "b_q": 16, "s_A": 4, "s_E": 4, "s_P": 2, "s_X": 4, "s_Z": 2, "s_...` |  |
| M_live[vreg:finalize] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"R_k": 32, "b_q": 16, "s_A": 4, "s_Z": 2, "s_state": 4}` |  |
| M_peak[vreg] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"D_n": 24, "R_k": 32, "b_k": 32, "b_q": 16, "s_A": 4, "s_E": 4, "s_P": 2, "s_Qacc": 4,...` |  |
| M_live[vmem:q_proj] | derived | incomplete: depends on undeclared/unbound b_rq | `{"D_n": 24, "R_q": 16, "b_q": 16, "s_Cq": 2, "s_Wq": 2, "s_scratch": 0}` |  |
| M_live[vmem:q_absorb] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"D_n": 24, "R_k": 32, "s_Wk": 2, "s_scratch": 0}` |  |
| M_live[vmem:score] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"D_r": 8, "R_k": 32, "b_k": 32, "b_q": 16, "n_buf": 2, "s_Ck": 2, "s_Kr": 2, "s_Qr": 2...` |  |
| M_live[vmem:softmax] | derived | incomplete: depends on undeclared/unbound q_resident | `{"D_r": 8, "R_k": 32, "b_k": 32, "b_q": 16, "n_buf": 2, "s_Ck": 2, "s_Qr": 2, "s_Qt": 2...` |  |
| M_live[vmem:pv] | derived | incomplete: depends on undeclared/unbound q_resident | `{"D_r": 8, "R_k": 32, "b_k": 32, "b_q": 16, "n_buf": 2, "s_Ck": 2, "s_Qr": 2, "s_Qt": 2...` |  |
| M_live[vmem:finalize] | derived | incomplete: depends on undeclared/unbound q_resident | `{"D_r": 8, "D_v": 24, "R_k": 32, "b_q": 16, "s_LSE": 4, "s_O": 2, "s_Qr": 2, "s_Qt": 2,...` |  |
| M_peak[vmem] | derived | incomplete: depends on undeclared/unbound b_rq, q_resident | `{"D_n": 24, "D_r": 8, "D_v": 24, "R_k": 32, "R_q": 16, "b_k": 32, "b_q": 16, "n_buf": 2...` |  |
| envelope[all:q_proj] | derived | incomplete: depends on undeclared/unbound b_rq | `{"D_n": 24, "R_q": 16, "b_q": 16, "s_Cq": 2, "s_Qacc": 4, "s_Wq": 2, "s_scratch": 0}` |  |
| envelope[all:q_absorb] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"D_n": 24, "R_k": 32, "b_q": 16, "s_Qacc": 4, "s_Wk": 2, "s_scratch": 0}` |  |
| envelope[all:score] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"D_r": 8, "R_k": 32, "b_k": 32, "b_q": 16, "n_buf": 2, "s_A": 4, "s_Ck": 2, "s_E": 4, ...` |  |
| envelope[all:softmax] | derived | incomplete: depends on undeclared/unbound q_resident | `{"D_r": 8, "R_k": 32, "b_k": 32, "b_q": 16, "n_buf": 2, "s_A": 4, "s_Ck": 2, "s_E": 4, ...` |  |
| envelope[all:pv] | derived | incomplete: depends on undeclared/unbound q_resident | `{"D_r": 8, "R_k": 32, "b_k": 32, "b_q": 16, "n_buf": 2, "s_A": 4, "s_Ck": 2, "s_E": 4, ...` |  |
| envelope[all:finalize] | derived | incomplete: depends on undeclared/unbound q_resident | `{"D_r": 8, "D_v": 24, "R_k": 32, "b_q": 16, "s_A": 4, "s_LSE": 4, "s_O": 2, "s_Qr": 2, ...` |  |
| envelope[vreg:q_proj] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"D_n": 24, "b_q": 16, "s_Qacc": 4}` |  |
| envelope[vreg:q_absorb] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"D_n": 24, "R_k": 32, "b_q": 16, "s_Qacc": 4}` |  |
| envelope[vreg:score] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"R_k": 32, "b_k": 32, "b_q": 16, "s_A": 4, "s_E": 4, "s_X": 4, "s_Z": 2, "s_state": 4}` |  |
| envelope[vreg:softmax] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"R_k": 32, "b_k": 32, "b_q": 16, "s_A": 4, "s_E": 4, "s_X": 4, "s_Z": 2, "s_state": 4}` |  |
| envelope[vreg:pv] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"R_k": 32, "b_k": 32, "b_q": 16, "s_A": 4, "s_E": 4, "s_P": 2, "s_X": 4, "s_Z": 2, "s_...` |  |
| envelope[vreg:finalize] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"R_k": 32, "b_q": 16, "s_A": 4, "s_Z": 2, "s_state": 4}` |  |
| envelope[vmem:q_proj] | derived | incomplete: depends on undeclared/unbound b_rq | `{"D_n": 24, "R_q": 16, "b_q": 16, "s_Cq": 2, "s_Wq": 2, "s_scratch": 0}` |  |
| envelope[vmem:q_absorb] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"D_n": 24, "R_k": 32, "s_Wk": 2, "s_scratch": 0}` |  |
| envelope[vmem:score] | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"D_r": 8, "R_k": 32, "b_k": 32, "b_q": 16, "n_buf": 2, "s_Ck": 2, "s_Kr": 2, "s_Qr": 2...` |  |
| envelope[vmem:softmax] | derived | incomplete: depends on undeclared/unbound q_resident | `{"D_r": 8, "R_k": 32, "b_k": 32, "b_q": 16, "n_buf": 2, "s_Ck": 2, "s_Qr": 2, "s_Qt": 2...` |  |
| envelope[vmem:pv] | derived | incomplete: depends on undeclared/unbound q_resident | `{"D_r": 8, "R_k": 32, "b_k": 32, "b_q": 16, "n_buf": 2, "s_Ck": 2, "s_Qr": 2, "s_Qt": 2...` |  |
| envelope[vmem:finalize] | derived | incomplete: depends on undeclared/unbound q_resident | `{"D_r": 8, "D_v": 24, "R_k": 32, "b_q": 16, "s_LSE": 4, "s_O": 2, "s_Qr": 2, "s_Qt": 2,...` |  |
| V_lifetime_stages | derived | not applicable under the declared configuration | `{}` |  |
| V_prefetch_overlapped | derived | not applicable under the declared configuration | `{}` |  |
| b_k_max[vreg:q_proj] | derived | not applicable under the declared configuration | `{}` |  |
| excess_over_budget[vreg:q_proj] | derived | incomplete: depends on undeclared/unbound vreg_budget_bytes | `{"D_n": 24, "b_q": 16, "s_Qacc": 4}` |  |
| b_k_max[vreg:q_absorb] | derived | not applicable under the declared configuration | `{}` |  |
| excess_over_budget[vreg:q_absorb] | derived | incomplete: depends on undeclared/unbound vreg_budget_bytes | `{"D_n": 24, "R_k": 32, "b_q": 16, "s_Qacc": 4}` |  |
| b_k_max[vreg:score] | derived | incomplete: depends on undeclared/unbound vreg_budget_bytes | `{"R_k": 32, "b_q": 16, "s_A": 4, "s_E": 4, "s_X": 4, "s_Z": 2, "s_state": 4}` |  |
| excess_over_budget[vreg:score] | derived | incomplete: depends on undeclared/unbound vreg_budget_bytes | `{"R_k": 32, "b_k": 32, "b_q": 16, "s_A": 4, "s_E": 4, "s_X": 4, "s_Z": 2, "s_state": 4}` |  |
| b_k_max[vreg:softmax] | derived | incomplete: depends on undeclared/unbound vreg_budget_bytes | `{"R_k": 32, "b_q": 16, "s_A": 4, "s_E": 4, "s_X": 4, "s_Z": 2, "s_state": 4}` |  |
| excess_over_budget[vreg:softmax] | derived | incomplete: depends on undeclared/unbound vreg_budget_bytes | `{"R_k": 32, "b_k": 32, "b_q": 16, "s_A": 4, "s_E": 4, "s_X": 4, "s_Z": 2, "s_state": 4}` |  |
| b_k_max[vreg:pv] | derived | incomplete: depends on undeclared/unbound vreg_budget_bytes | `{"R_k": 32, "b_q": 16, "s_A": 4, "s_E": 4, "s_P": 2, "s_X": 4, "s_Z": 2, "s_state": 4}` |  |
| excess_over_budget[vreg:pv] | derived | incomplete: depends on undeclared/unbound vreg_budget_bytes | `{"R_k": 32, "b_k": 32, "b_q": 16, "s_A": 4, "s_E": 4, "s_P": 2, "s_X": 4, "s_Z": 2, "s_...` |  |
| b_k_max[vreg:finalize] | derived | not applicable under the declared configuration | `{}` |  |
| excess_over_budget[vreg:finalize] | derived | incomplete: depends on undeclared/unbound vreg_budget_bytes | `{"R_k": 32, "b_q": 16, "s_A": 4, "s_Z": 2, "s_state": 4}` |  |
| b_k_max[vreg:binding_stage] | derived | incomplete: depends on undeclared/unbound vreg_budget_bytes | `{}` |  |
| b_k_max[vmem:q_proj] | derived | not applicable under the declared configuration | `{}` |  |
| excess_over_budget[vmem:q_proj] | derived | incomplete: depends on undeclared/unbound b_rq, vmem_budget_bytes | `{"D_n": 24, "R_q": 16, "b_q": 16, "s_Cq": 2, "s_Wq": 2, "s_scratch": 0}` |  |
| b_k_max[vmem:q_absorb] | derived | not applicable under the declared configuration | `{}` |  |
| excess_over_budget[vmem:q_absorb] | derived | incomplete: depends on undeclared/unbound vmem_budget_bytes | `{"D_n": 24, "R_k": 32, "s_Wk": 2, "s_scratch": 0}` |  |
| b_k_max[vmem:score] | derived | incomplete: depends on undeclared/unbound vmem_budget_bytes | `{"D_r": 8, "R_k": 32, "b_q": 16, "n_buf": 2, "s_Ck": 2, "s_Kr": 2, "s_Qr": 2, "s_Qt": 2...` |  |
| excess_over_budget[vmem:score] | derived | incomplete: depends on undeclared/unbound vmem_budget_bytes | `{"D_r": 8, "R_k": 32, "b_k": 32, "b_q": 16, "n_buf": 2, "s_Ck": 2, "s_Kr": 2, "s_Qr": 2...` |  |
| b_k_max[vmem:softmax] | derived | incomplete: depends on undeclared/unbound q_resident, vmem_budget_bytes | `{"D_r": 8, "R_k": 32, "b_q": 16, "n_buf": 2, "s_Ck": 2, "s_Qr": 2, "s_Qt": 2, "s_scratc...` |  |
| excess_over_budget[vmem:softmax] | derived | incomplete: depends on undeclared/unbound q_resident, vmem_budget_bytes | `{"D_r": 8, "R_k": 32, "b_k": 32, "b_q": 16, "n_buf": 2, "s_Ck": 2, "s_Qr": 2, "s_Qt": 2...` |  |
| b_k_max[vmem:pv] | derived | incomplete: depends on undeclared/unbound q_resident, vmem_budget_bytes | `{"D_r": 8, "R_k": 32, "b_q": 16, "n_buf": 2, "s_Ck": 2, "s_Qr": 2, "s_Qt": 2, "s_scratc...` |  |
| excess_over_budget[vmem:pv] | derived | incomplete: depends on undeclared/unbound q_resident, vmem_budget_bytes | `{"D_r": 8, "R_k": 32, "b_k": 32, "b_q": 16, "n_buf": 2, "s_Ck": 2, "s_Qr": 2, "s_Qt": 2...` |  |
| b_k_max[vmem:finalize] | derived | not applicable under the declared configuration | `{}` |  |
| excess_over_budget[vmem:finalize] | derived | incomplete: depends on undeclared/unbound q_resident, vmem_budget_bytes | `{"D_r": 8, "D_v": 24, "R_k": 32, "b_q": 16, "s_LSE": 4, "s_O": 2, "s_Qr": 2, "s_Qt": 2,...` |  |
| b_k_max[vmem:binding_stage] | derived | incomplete: depends on undeclared/unbound b_rq, q_resident, vmem_budget_bytes | `{}` |  |
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
| d(score_bytes)/d(b_q) | `sens.score_bytes.b_q` | `b_k*s_X` | 128 | byte per row | bound |  | continuous relaxation of the source-level logical size | a discrete tile change |
| d(score_bytes)/d(b_k) | `sens.score_bytes.b_k` | `b_q*s_X` | 64 | byte per row | bound |  | continuous relaxation of the source-level logical size | a discrete tile change |
| d(accumulator_bytes)/d(b_q) | `sens.accumulator_bytes.b_q` | `R_k*s_A` | 128 | byte per row | bound |  | continuous relaxation of the source-level logical size | a discrete tile change |
| d(accumulator_bytes)/d(b_k) | `sens.accumulator_bytes.b_k` | `0` | 0 | byte per row | bound |  | continuous relaxation of the source-level logical size | an absence of any effect of b_k; a discrete tile change |
| n_programs | `sched.programs` | `(ceiling(7/b_q) + ceiling(20/b_q) + ceiling(48/b_q))*ceiling(H/h_pp)` | 6 | programs | partial | schedule | (batch, head-group, q-block) programs under the declared loop nest |  |
| n_kv_block_visits | `sched.kv_visits` | `H*(BlockStat(selected, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + BlockStat(selected, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + BlockStat(selected, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96))` | 48 | visits | partial | schedule | selected (b, h, q-block, kv-block) rectangles |  |

Assumptions:
- `d(accumulator_bytes)/d(b_k)`: a zero derivative is the absence of a *direct* dependence in the source-level logical size; scheduling and buffer lifetime can still create an indirect effect (spec 10.1)

<details><summary>Provenance (bindings, coverage, source, evidence)</summary>

| metric | source | coverage | bindings | evidence |
|---|---|---|---|---|
| d(score_bytes)/d(b_q) | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"b_k": 32, "s_X": 4}` |  |
| d(score_bytes)/d(b_k) | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"b_q": 16, "s_X": 4}` |  |
| d(accumulator_bytes)/d(b_q) | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{"R_k": 32, "s_A": 4}` |  |
| d(accumulator_bytes)/d(b_k) | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{}` |  |
| n_programs | derived | incomplete: depends on undeclared/unbound schedule | `{"H": 4, "b_q": 16, "h_pp": 4}` |  |
| n_kv_block_visits | derived | incomplete: depends on undeclared/unbound schedule | `{"H": 4, "b_k": 32, "b_q": 16}` |  |

</details>

## 9. hardware lower bounds and calibration

| metric | formula id | expression | value | unit | status | missing | scope | not equivalent to |
|---|---|---|---|---|---|---|---|---|
| T_LB[mxu] | `perf.lb.mxu` | `(150*D_n*H*R_k + 150*D_n*H*R_q + 2*D_r*H*BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + 2*D_r*H*BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + 2*D_r*H*BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + 150*D_v*H*R_k + 4*H*R_k*BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + 4*H*R_k*BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + 4*H*R_k*BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96))/P_mxu` |  | s | unknown | P_mxu, schedule | scope=None | actual latency; calibrated prediction |
| T_LB[vpu] | `perf.lb.vpu` | `(75*D_v*H + 75*H*R_k + 2*H*R_k*BlockStat(row_visits, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + 2*H*R_k*BlockStat(row_visits, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + 2*H*R_k*BlockStat(row_visits, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + 150*H + H*BlockStat(masked_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + H*BlockStat(masked_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + H*BlockStat(masked_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + 2*H*BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + 2*H*BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + 2*H*BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + 2*H*BlockStat(row_visits, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + 2*H*BlockStat(row_visits, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + 2*H*BlockStat(row_visits, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + Piecewise((H*BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + H*BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + H*BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96), D_r > 0), (0, True)))/P_vpu` |  | s | unknown | P_vpu, schedule | scope=None | actual latency; calibrated prediction |
| T_LB[reduce] | `perf.lb.reduce` | `(2*H*BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + 2*H*BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + 2*H*BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) - H*BlockStat(row_visits, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) - H*BlockStat(row_visits, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) - H*BlockStat(row_visits, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96))/P_red` |  | s | unknown | P_red, schedule | scope=None | actual latency; calibrated prediction |
| T_LB[layout] | `perf.lb.layout` | `(H*R_k*BlockStat(key_visits, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + H*R_k*BlockStat(key_visits, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + H*R_k*BlockStat(key_visits, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + H*BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + H*BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + H*BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96))/P_layout` |  | s | unknown | P_layout, schedule | scope=None | actual latency; calibrated prediction |
| T_LB[exp] | `perf.lb.exp` | `(H*BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + H*BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + H*BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + H*BlockStat(row_visits, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + H*BlockStat(row_visits, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + H*BlockStat(row_visits, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96))/P_exp` |  | s | unknown | P_exp, schedule | scope=None | actual latency; calibrated prediction |
| T_LB[path:hbm_to_vmem] | `perf.lb.path:hbm_to_vmem` | `(D_n*H*R_k*s_Wk*ceiling(7/b_q) + D_n*H*R_k*s_Wk*ceiling(20/b_q) + D_n*H*R_k*s_Wk*ceiling(48/b_q) + D_n*H*R_q*s_Wq*ceiling(7/b_q) + D_n*H*R_q*s_Wq*ceiling(20/b_q) + D_n*H*R_q*s_Wq*ceiling(48/b_q) + 75*D_r*H*s_Qr + D_r*s_Kr*ceiling(H/h_pp)*BlockStat(key_visits, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + D_r*s_Kr*ceiling(H/h_pp)*BlockStat(key_visits, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + D_r*s_Kr*ceiling(H/h_pp)*BlockStat(key_visits, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + D_v*H*R_k*s_Wv*ceiling(7/b_q) + D_v*H*R_k*s_Wv*ceiling(20/b_q) + D_v*H*R_k*s_Wv*ceiling(48/b_q) + R_k*s_Ck*ceiling(H/h_pp)*BlockStat(key_visits, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + R_k*s_Ck*ceiling(H/h_pp)*BlockStat(key_visits, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + R_k*s_Ck*ceiling(H/h_pp)*BlockStat(key_visits, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + 75*R_q*s_Cq*ceiling(H/h_pp))/W_hbm` |  | s | unknown | W_hbm, schedule | scope=None | actual latency; calibrated prediction |
| T_LB[path:vmem_to_hbm] | `perf.lb.path:vmem_to_hbm` | `(75*D_v*H*s_O + 75*H*s_LSE)/W_hbm_w` |  | s | unknown | W_hbm_w, schedule | scope=None | actual latency; calibrated prediction |
| T_LB[path:vmem_to_vreg] | `perf.lb.path:vmem_to_vreg` | `(D_r*H*b_k*s_Kr*BlockStat(selected, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + D_r*H*b_k*s_Kr*BlockStat(selected, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + D_r*H*b_k*s_Kr*BlockStat(selected, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + D_r*H*b_q*s_Qr*ceiling(7/b_q) + D_r*H*b_q*s_Qr*ceiling(20/b_q) + D_r*H*b_q*s_Qr*ceiling(48/b_q) + H*R_k*b_k*s_Ck*BlockStat(selected, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + H*R_k*b_k*s_Ck*BlockStat(selected, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + H*R_k*b_k*s_Ck*BlockStat(selected, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + H*R_k*b_q*s_Qt*ceiling(7/b_q) + H*R_k*b_q*s_Qt*ceiling(20/b_q) + H*R_k*b_q*s_Qt*ceiling(48/b_q))/W_vmem` |  | s | unknown | W_vmem, q_resident, register_schedule_evidence, schedule | scope=None | actual latency; calibrated prediction |
| T_LB[critical_path] | `perf.lb.critical_path` | `CP` |  | s | unknown | CP | scope=None | actual latency; calibrated prediction |
| T_LB[combined] | `perf.lb.combined` | `max((150*D_n*H*R_k + 150*D_n*H*R_q + 2*D_r*H*BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + 2*D_r*H*BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + 2*D_r*H*BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + 150*D_v*H*R_k + 4*H*R_k*BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + 4*H*R_k*BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + 4*H*R_k*BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96))/P_mxu, (75*D_v*H + 75*H*R_k + 2*H*R_k*BlockStat(row_visits, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + 2*H*R_k*BlockStat(row_visits, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + 2*H*R_k*BlockStat(row_visits, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + 150*H + H*BlockStat(masked_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + H*BlockStat(masked_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + H*BlockStat(masked_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + 2*H*BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + 2*H*BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + 2*H*BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + 2*H*BlockStat(row_visits, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + 2*H*BlockStat(row_visits, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + 2*H*BlockStat(row_visits, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + Piecewise((H*BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + H*BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + H*BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96), D_r > 0), (0, True)))/P_vpu, (2*H*BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + 2*H*BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + 2*H*BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) - H*BlockStat(row_visits, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) - H*BlockStat(row_visits, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) - H*BlockStat(row_visits, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96))/P_red, (H*R_k*BlockStat(key_visits, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + H*R_k*BlockStat(key_visits, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + H*R_k*BlockStat(key_visits, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + H*BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + H*BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + H*BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96))/P_layout, (H*BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + H*BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + H*BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + H*BlockStat(row_visits, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + H*BlockStat(row_visits, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + H*BlockStat(row_visits, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96))/P_exp, (D_n*H*R_k*s_Wk*ceiling(7/b_q) + D_n*H*R_k*s_Wk*ceiling(20/b_q) + D_n*H*R_k*s_Wk*ceiling(48/b_q) + D_n*H*R_q*s_Wq*ceiling(7/b_q) + D_n*H*R_q*s_Wq*ceiling(20/b_q) + D_n*H*R_q*s_Wq*ceiling(48/b_q) + 75*D_r*H*s_Qr + D_r*s_Kr*ceiling(H/h_pp)*BlockStat(key_visits, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + D_r*s_Kr*ceiling(H/h_pp)*BlockStat(key_visits, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + D_r*s_Kr*ceiling(H/h_pp)*BlockStat(key_visits, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + D_v*H*R_k*s_Wv*ceiling(7/b_q) + D_v*H*R_k*s_Wv*ceiling(20/b_q) + D_v*H*R_k*s_Wv*ceiling(48/b_q) + R_k*s_Ck*ceiling(H/h_pp)*BlockStat(key_visits, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + R_k*s_Ck*ceiling(H/h_pp)*BlockStat(key_visits, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + R_k*s_Ck*ceiling(H/h_pp)*BlockStat(key_visits, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + 75*R_q*s_Cq*ceiling(H/h_pp))/W_hbm, (75*D_v*H*s_O + 75*H*s_LSE)/W_hbm_w, (D_r*H*b_k*s_Kr*BlockStat(selected, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + D_r*H*b_k*s_Kr*BlockStat(selected, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + D_r*H*b_k*s_Kr*BlockStat(selected, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + D_r*H*b_q*s_Qr*ceiling(7/b_q) + D_r*H*b_q*s_Qr*ceiling(20/b_q) + D_r*H*b_q*s_Qr*ceiling(48/b_q) + H*R_k*b_k*s_Ck*BlockStat(selected, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + H*R_k*b_k*s_Ck*BlockStat(selected, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + H*R_k*b_k*s_Ck*BlockStat(selected, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + H*R_k*b_q*s_Qt*ceiling(7/b_q) + H*R_k*b_q*s_Qt*ceiling(20/b_q) + H*R_k*b_q*s_Qt*ceiling(48/b_q))/W_vmem, CP)` |  | s | unknown | CP, P_exp, P_layout, P_mxu, P_red, P_vpu, W_hbm, W_hbm_w, W_vmem, q_resident, register_schedule_evidence, schedule | overlap model full_overlap_max | calibrated prediction; measured time |
| T_pred[node_interval] | `perf.calibrated.nodes` | `sum_v W_v/(eps_v P_v) + t_startup, + T_launch + eps_model` |  | s | unknown | calibration | no applicable calibration: no prediction issued | T_LB; hardware-peak lower bound |

<details><summary>Provenance (bindings, coverage, source, evidence)</summary>

| metric | source | coverage | bindings | evidence |
|---|---|---|---|---|
| T_LB[mxu] | derived | incomplete: depends on undeclared/unbound P_mxu, schedule | `{}` |  |
| T_LB[vpu] | derived | incomplete: depends on undeclared/unbound P_vpu, schedule | `{}` |  |
| T_LB[reduce] | derived | incomplete: depends on undeclared/unbound P_red, schedule | `{}` |  |
| T_LB[layout] | derived | incomplete: depends on undeclared/unbound P_layout, schedule | `{}` |  |
| T_LB[exp] | derived | incomplete: depends on undeclared/unbound P_exp, schedule | `{}` |  |
| T_LB[path:hbm_to_vmem] | derived | incomplete: depends on undeclared/unbound W_hbm, schedule | `{}` |  |
| T_LB[path:vmem_to_hbm] | derived | incomplete: depends on undeclared/unbound W_hbm_w, schedule | `{}` |  |
| T_LB[path:vmem_to_vreg] | derived | incomplete: depends on undeclared/unbound W_vmem, q_resident, register_schedule_evidence, schedule | `{}` |  |
| T_LB[critical_path] | derived | incomplete: depends on undeclared/unbound CP | `{}` |  |
| T_LB[combined] | derived | incomplete: depends on undeclared/unbound CP, P_exp, P_layout, P_mxu, P_red, P_vpu, W_hbm, W_hbm_w, W_vmem, q_resident, register_schedule_evidence, schedule | `{}` |  |
| T_pred[node_interval] | derived | incomplete: no applicable calibration evidence attached | `{}` |  |

</details>

## Constraints

| constraint | relation | holds | severity | missing |
|---|---|---|---|---|
| q_tail | `Eq(Mod(S_q, b_q), 0)` | None | info | S_q |
| kv_tail | `Eq(Mod(S_k, b_k), 0)` | None | info | S_k |
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
    "count": "H*(BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96))",
    "conditional_on": null,
    "exists": true,
    "description": "gamma * X on every executed cell (fused variants differ)"
  },
  {
    "id": "mask_apply",
    "stage": "score",
    "kind": "mask",
    "resource": "vpu",
    "count": "H*(BlockStat(masked_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + BlockStat(masked_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + BlockStat(masked_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96))",
    "conditional_on": "mask application scenario: not-fully-visible rectangles only",
    "exists": true,
    "description": "additive/select mask on executed cells of rectangles that are not fully visible (fully visible blocks need no mask)"
  },
  {
    "id": "row_max_cmp",
    "stage": "softmax",
    "kind": "cmp",
    "resource": "reduce",
    "count": "H*(BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96)) - H*(BlockStat(row_visits, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + BlockStat(row_visits, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + BlockStat(row_visits, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96))",
    "conditional_on": null,
    "exists": true,
    "description": "n*max(m-1,0) comparisons for rowmax over executed cells"
  },
  {
    "id": "old_new_max_cmp",
    "stage": "softmax",
    "kind": "cmp",
    "resource": "reduce",
    "count": "H*(BlockStat(row_visits, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + BlockStat(row_visits, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + BlockStat(row_visits, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96))",
    "conditional_on": null,
    "exists": true,
    "description": "max(m, rowmax) once per row per block visit"
  },
  {
    "id": "rowmax_broadcast",
    "stage": "softmax",
    "kind": "broadcast",
    "resource": "layout",
    "count": "H*(BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96))",
    "conditional_on": "broadcast_materialized (undeclared backend behaviour)",
    "exists": null,
    "description": "broadcast m' across the row before X - m' (may be free depending on layout)"
  },
  {
    "id": "kv_operand_transpose",
    "stage": "score",
    "kind": "transpose",
    "resource": "layout",
    "count": "H*R_k*(BlockStat(key_visits, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + BlockStat(key_visits, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + BlockStat(key_visits, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96))",
    "conditional_on": "operand_transpose_materialized (undeclared backend behaviour)",
    "exists": null,
    "description": "re-layout / transpose of the K-side operand for the score matmul (free when the layout already matches)"
  },
  {
    "id": "exp_E",
    "stage": "softmax",
    "kind": "exp",
    "resource": "exp",
    "count": "H*(BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96))",
    "conditional_on": null,
    "exists": true,
    "description": "E = exp(X - m') on every executed cell (masked cells included)"
  },
  {
    "id": "row_sum_add",
    "stage": "softmax",
    "kind": "add",
    "resource": "reduce",
    "count": "H*(BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96)) - H*(BlockStat(row_visits, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + BlockStat(row_visits, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + BlockStat(row_visits, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96))",
    "conditional_on": null,
    "exists": true,
    "description": "n*max(m-1,0) additions for rowsum(E)"
  },
  {
    "id": "rescale_alpha_exp",
    "stage": "softmax",
    "kind": "exp",
    "resource": "exp",
    "count": "H*(BlockStat(row_visits, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + BlockStat(row_visits, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + BlockStat(row_visits, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96))",
    "conditional_on": null,
    "exists": true,
    "description": "alpha = exp(m - m') at most once per row per block visit"
  },
  {
    "id": "acc_scale",
    "stage": "pv",
    "kind": "mul",
    "resource": "vpu",
    "count": "H*R_k*(BlockStat(row_visits, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + BlockStat(row_visits, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + BlockStat(row_visits, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96))",
    "conditional_on": null,
    "exists": true,
    "description": "alpha * A (n*D_A per block visit)"
  },
  {
    "id": "acc_add",
    "stage": "pv",
    "kind": "add",
    "resource": "vpu",
    "count": "H*R_k*(BlockStat(row_visits, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + BlockStat(row_visits, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + BlockStat(row_visits, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96))",
    "conditional_on": null,
    "exists": true,
    "description": "A + E U (n*D_A per block visit; matmul epilogue may fuse)"
  },
  {
    "id": "l_scale",
    "stage": "softmax",
    "kind": "mul",
    "resource": "vpu",
    "count": "H*(BlockStat(row_visits, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + BlockStat(row_visits, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + BlockStat(row_visits, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96))",
    "conditional_on": null,
    "exists": true,
    "description": "alpha * l"
  },
  {
    "id": "l_update",
    "stage": "softmax",
    "kind": "add",
    "resource": "vpu",
    "count": "H*(BlockStat(row_visits, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + BlockStat(row_visits, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + BlockStat(row_visits, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96))",
    "conditional_on": null,
    "exists": true,
    "description": "l' = alpha*l + rowsum"
  },
  {
    "id": "final_normalize_div",
    "stage": "finalize",
    "kind": "div",
    "resource": "vpu",
    "count": "75*H*R_k",
    "conditional_on": "final_normalize",
    "exists": null,
    "description": "Y = A / l once per row, as n*D_A divisions (form 'divide')"
  },
  {
    "id": "final_normalize_recip",
    "stage": "finalize",
    "kind": "recip",
    "resource": "vpu",
    "count": "75*H",
    "conditional_on": "final_normalize",
    "exists": false,
    "description": "1/l once per row (form 'reciprocal_multiply')"
  },
  {
    "id": "final_normalize_mul",
    "stage": "finalize",
    "kind": "mul",
    "resource": "vpu",
    "count": "75*H*R_k",
    "conditional_on": "final_normalize",
    "exists": false,
    "description": "Y = A * (1/l), n*D_A multiplies (form 'reciprocal_multiply')"
  },
  {
    "id": "lse_log",
    "stage": "finalize",
    "kind": "log",
    "resource": "vpu",
    "count": "75*H",
    "conditional_on": null,
    "exists": true,
    "description": "log(l) once per row for LSE = m + log(l)"
  },
  {
    "id": "lse_add",
    "stage": "finalize",
    "kind": "add",
    "resource": "vpu",
    "count": "75*H",
    "conditional_on": null,
    "exists": true,
    "description": "m + log(l) once per row"
  },
  {
    "id": "cast[exp_to_p]",
    "stage": "pv",
    "kind": "cast",
    "resource": "vpu",
    "count": "H*(BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96))",
    "conditional_on": "cast_points",
    "exists": true,
    "description": "cast E to the PV / P C_k matmul operand dtype (inferred from the declared dtypes)"
  },
  {
    "id": "cast[accumulator_to_output]",
    "stage": "finalize",
    "kind": "cast",
    "resource": "vpu",
    "count": "75*D_v*H",
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
    "count": "Piecewise((H*(BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96)), D_r > 0), (0, True))",
    "conditional_on": "D_r > 0",
    "exists": true,
    "description": "X = Q~ C_k^T + Q^r K^r^T (0 when D_r = 0)"
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
        3,
        48,
        16
      ],
      "logical_bytes": 4608,
      "allocated_bytes": 4608,
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
        3,
        96,
        32
      ],
      "logical_bytes": 18432,
      "allocated_bytes": 18432,
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
        3,
        4,
        48,
        8
      ],
      "logical_bytes": 9216,
      "allocated_bytes": 9216,
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
        3,
        96,
        8
      ],
      "logical_bytes": 4608,
      "allocated_bytes": 4608,
      "alias_of": null,
      "counted_in_allocation": true,
      "alias_group": "k_pe",
      "strides": null,
      "storage_offset": 0,
      "allocation_from_strides_modelled": true
    }
  ],
  "logical_total": 52224,
  "allocated_unique_total": 52224,
  "roles_with_unmodelled_strides": []
}
```

## materialization_ledger

```json
[
  {
    "id": "mat_q_tilde",
    "tensor": "Q~",
    "conditional_on": "materialize.q_tilde",
    "enabled": false,
    "bytes": "75*H*R_k*s_Qt",
    "write_traffic": "75*H*R_k*s_Qt",
    "read_traffic": "75*H*R_k*s_Qt",
    "bytes_value": 0,
    "traffic_value": 0,
    "bytes_value_if_materialized": 19200,
    "traffic_value_if_materialized": 38400
  },
  {
    "id": "mat_z",
    "tensor": "Z",
    "conditional_on": "materialize.z",
    "enabled": false,
    "bytes": "75*H*R_k*s_Z",
    "write_traffic": "75*H*R_k*s_Z",
    "read_traffic": "75*H*R_k*s_Z",
    "bytes_value": 0,
    "traffic_value": 0,
    "bytes_value_if_materialized": 19200,
    "traffic_value_if_materialized": 38400
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
    "executions": "75*ceiling(H/h_pp)",
    "total": "75*R_q*s_Cq*ceiling(H/h_pp)",
    "value": 2400,
    "description": "Q latent row per (head group, query row)",
    "scenario": "logical rows read (spec 4.2: a logical out-of-bounds is not an HBM read)",
    "undeclared": false,
    "conditional_on": null
  },
  {
    "id": "Qr_read",
    "path": "hbm_to_vmem",
    "bytes_per_event": "D_r*s_Qr",
    "executions": "75*H",
    "total": "75*D_r*H*s_Qr",
    "value": 4800,
    "description": "Q^r row per (head, query row)",
    "scenario": "logical rows read (spec 4.2)",
    "undeclared": false,
    "conditional_on": null
  },
  {
    "id": "Ck_read",
    "path": "hbm_to_vmem",
    "bytes_per_event": "R_k*s_Ck",
    "executions": "ceiling(H/h_pp)*(BlockStat(key_visits, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + BlockStat(key_visits, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + BlockStat(key_visits, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96))",
    "total": "R_k*s_Ck*ceiling(H/h_pp)*BlockStat(key_visits, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + R_k*s_Ck*ceiling(H/h_pp)*BlockStat(key_visits, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + R_k*s_Ck*ceiling(H/h_pp)*BlockStat(key_visits, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96)",
    "value": 22144,
    "description": "KV latent per executed key column, shared by h_pp heads",
    "scenario": null,
    "undeclared": false,
    "conditional_on": null
  },
  {
    "id": "Kr_read",
    "path": "hbm_to_vmem",
    "bytes_per_event": "D_r*s_Kr",
    "executions": "ceiling(H/h_pp)*(BlockStat(key_visits, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + BlockStat(key_visits, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + BlockStat(key_visits, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96))",
    "total": "D_r*s_Kr*ceiling(H/h_pp)*BlockStat(key_visits, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + D_r*s_Kr*ceiling(H/h_pp)*BlockStat(key_visits, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + D_r*s_Kr*ceiling(H/h_pp)*BlockStat(key_visits, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96)",
    "value": 5536,
    "description": "K^r per executed key column, shared by h_pp heads",
    "scenario": null,
    "undeclared": false,
    "conditional_on": null
  },
  {
    "id": "Wv_read",
    "path": "hbm_to_vmem",
    "bytes_per_event": "D_v*R_k*s_Wv",
    "executions": "H*(ceiling(7/b_q) + ceiling(20/b_q) + ceiling(48/b_q))",
    "total": "D_v*H*R_k*s_Wv*ceiling(7/b_q) + D_v*H*R_k*s_Wv*ceiling(20/b_q) + D_v*H*R_k*s_Wv*ceiling(48/b_q)",
    "value": 36864,
    "description": "W^v per (program, head)",
    "scenario": "weights re-read per program",
    "undeclared": false,
    "conditional_on": null
  },
  {
    "id": "Wq_read",
    "path": "hbm_to_vmem",
    "bytes_per_event": "D_n*R_q*s_Wq",
    "executions": "H*(ceiling(7/b_q) + ceiling(20/b_q) + ceiling(48/b_q))",
    "total": "D_n*H*R_q*s_Wq*ceiling(7/b_q) + D_n*H*R_q*s_Wq*ceiling(20/b_q) + D_n*H*R_q*s_Wq*ceiling(48/b_q)",
    "value": 18432,
    "description": "W^q per (program, head)",
    "scenario": "weights re-read per program",
    "undeclared": false,
    "conditional_on": null
  },
  {
    "id": "Wk_read",
    "path": "hbm_to_vmem",
    "bytes_per_event": "D_n*R_k*s_Wk",
    "executions": "H*(ceiling(7/b_q) + ceiling(20/b_q) + ceiling(48/b_q))",
    "total": "D_n*H*R_k*s_Wk*ceiling(7/b_q) + D_n*H*R_k*s_Wk*ceiling(20/b_q) + D_n*H*R_k*s_Wk*ceiling(48/b_q)",
    "value": 36864,
    "description": "W^k per (program, head)",
    "scenario": "weights re-read per program",
    "undeclared": false,
    "conditional_on": null
  },
  {
    "id": "O_write",
    "path": "vmem_to_hbm",
    "bytes_per_event": "D_v*s_O",
    "executions": "75*H",
    "total": "75*D_v*H*s_O",
    "value": 14400,
    "description": "write one output row per (b, h, active query row)",
    "scenario": null,
    "undeclared": false,
    "conditional_on": null
  },
  {
    "id": "LSE_write",
    "path": "vmem_to_hbm",
    "bytes_per_event": "s_LSE",
    "executions": "75*H",
    "total": "75*H*s_LSE",
    "value": 1200,
    "description": "write one LSE value per (b, h, active query row)",
    "scenario": null,
    "undeclared": false,
    "conditional_on": "outputs"
  },
  {
    "id": "stream_Qt_operand",
    "path": "vmem_to_vreg",
    "bytes_per_event": "R_k*b_q*s_Qt",
    "executions": "H*(ceiling(7/b_q) + ceiling(20/b_q) + ceiling(48/b_q))",
    "total": "H*R_k*b_q*s_Qt*ceiling(7/b_q) + H*R_k*b_q*s_Qt*ceiling(20/b_q) + H*R_k*b_q*s_Qt*ceiling(48/b_q)",
    "value": 24576,
    "description": "stream Qt_operand into registers once per (head, q-block)",
    "scenario": "operand streaming (no register-schedule evidence); Q residency undeclared",
    "undeclared": true,
    "conditional_on": "q_resident"
  },
  {
    "id": "stream_Qr_operand",
    "path": "vmem_to_vreg",
    "bytes_per_event": "D_r*b_q*s_Qr",
    "executions": "H*(ceiling(7/b_q) + ceiling(20/b_q) + ceiling(48/b_q))",
    "total": "D_r*H*b_q*s_Qr*ceiling(7/b_q) + D_r*H*b_q*s_Qr*ceiling(20/b_q) + D_r*H*b_q*s_Qr*ceiling(48/b_q)",
    "value": 6144,
    "description": "stream Qr_operand into registers once per (head, q-block)",
    "scenario": "operand streaming (no register-schedule evidence); Q residency undeclared",
    "undeclared": true,
    "conditional_on": "q_resident"
  },
  {
    "id": "stream_Ck_tile",
    "path": "vmem_to_vreg",
    "bytes_per_event": "R_k*b_k*s_Ck",
    "executions": "H*(BlockStat(selected, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + BlockStat(selected, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + BlockStat(selected, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96))",
    "total": "H*R_k*b_k*s_Ck*BlockStat(selected, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + H*R_k*b_k*s_Ck*BlockStat(selected, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + H*R_k*b_k*s_Ck*BlockStat(selected, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96)",
    "value": 98304,
    "description": "stream Ck_tile into registers once per (b, h, kv-block visit)",
    "scenario": "operand streaming (no register-schedule evidence)",
    "undeclared": true,
    "conditional_on": null
  },
  {
    "id": "stream_Kr_tile",
    "path": "vmem_to_vreg",
    "bytes_per_event": "D_r*b_k*s_Kr",
    "executions": "H*(BlockStat(selected, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + BlockStat(selected, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + BlockStat(selected, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96))",
    "total": "D_r*H*b_k*s_Kr*BlockStat(selected, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + D_r*H*b_k*s_Kr*BlockStat(selected, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + D_r*H*b_k*s_Kr*BlockStat(selected, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96)",
    "value": 24576,
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
  "all:q_absorb": {
    "stage": "q_absorb",
    "level": null,
    "a": {
      "expression": "0",
      "value": 0,
      "residual": null
    },
    "b": {
      "expression": "s_Qacc*(D_n + R_k)",
      "value": 224,
      "residual": null
    },
    "c": {
      "expression": "0",
      "value": 0,
      "residual": null
    },
    "d": {
      "expression": "D_n*R_k*s_Wk + s_scratch",
      "value": 1536,
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
        "q_proj_acc",
        "q_tilde_acc"
      ],
      "c": [],
      "d": [
        "wk_full",
        "fixed_scratch"
      ],
      "remainder": []
    },
    "undeclared": []
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
      "expression": "D_r*s_Qr + R_k*s_Qt + R_k*Max(s_A, s_Z) + 2*s_state",
      "value": 216,
      "residual": null
    },
    "c": {
      "expression": "n_buf*(D_r*s_Kr + R_k*s_Ck)",
      "value": 160,
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
        "q_tilde",
        "row_state_m",
        "row_state_l",
        "q_pe_tile",
        "alias[acc_or_Z]:z_tile|accumulator_A"
      ],
      "c": [
        "kv_latent_tile",
        "k_pe_tile"
      ],
      "d": [
        "fixed_scratch"
      ],
      "remainder": []
    },
    "undeclared": []
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
      "expression": "D_r*s_Qr + R_k*s_Qt + R_k*Max(s_A, s_Z) + 3*s_state",
      "value": 220,
      "residual": null
    },
    "c": {
      "expression": "R_k*n_buf*s_Ck",
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
        "q_tilde",
        "row_state_m",
        "row_state_l",
        "row_alpha",
        "q_pe_tile",
        "alias[acc_or_Z]:z_tile|accumulator_A"
      ],
      "c": [
        "kv_latent_tile"
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
  "all:pv": {
    "stage": "pv",
    "level": null,
    "a": {
      "expression": "s_P + Max(s_E, s_X)",
      "value": 6,
      "residual": null
    },
    "b": {
      "expression": "D_r*s_Qr + R_k*s_Qt + R_k*Max(s_A, s_Z) + 3*s_state",
      "value": 220,
      "residual": null
    },
    "c": {
      "expression": "R_k*n_buf*s_Ck",
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
        "p_operand",
        "alias[score_or_E]:score_X|exp_E"
      ],
      "b": [
        "q_tilde",
        "row_state_m",
        "row_state_l",
        "row_alpha",
        "q_pe_tile",
        "alias[acc_or_Z]:z_tile|accumulator_A"
      ],
      "c": [
        "kv_latent_tile"
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
      "expression": "D_r*s_Qr + D_v*s_O + R_k*s_Qt + R_k*Max(s_A, s_Z) + s_LSE + 2*s_state",
      "value": 268,
      "residual": null
    },
    "c": {
      "expression": "0",
      "value": 0,
      "residual": null
    },
    "d": {
      "expression": "D_v*R_k*s_Wv + s_scratch",
      "value": 1536,
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
        "q_tilde",
        "row_state_m",
        "row_state_l",
        "q_pe_tile",
        "out_tile",
        "lse_tile",
        "alias[acc_or_Z]:z_tile|accumulator_A"
      ],
      "c": [],
      "d": [
        "wv_full",
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
  "vreg:q_absorb": {
    "stage": "q_absorb",
    "level": "vreg",
    "a": {
      "expression": "0",
      "value": 0,
      "residual": null
    },
    "b": {
      "expression": "s_Qacc*(D_n + R_k)",
      "value": 224,
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
        "q_proj_acc",
        "q_tilde_acc"
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
      "expression": "R_k*Max(s_A, s_Z) + 2*s_state",
      "value": 136,
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
        "alias[acc_or_Z]:z_tile|accumulator_A"
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
      "expression": "R_k*Max(s_A, s_Z) + 3*s_state",
      "value": 140,
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
        "alias[acc_or_Z]:z_tile|accumulator_A"
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
      "expression": "R_k*Max(s_A, s_Z) + 3*s_state",
      "value": 140,
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
        "alias[acc_or_Z]:z_tile|accumulator_A"
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
      "expression": "R_k*Max(s_A, s_Z) + 2*s_state",
      "value": 136,
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
        "alias[acc_or_Z]:z_tile|accumulator_A"
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
  "vmem:q_absorb": {
    "stage": "q_absorb",
    "level": "vmem",
    "a": {
      "expression": "0",
      "value": 0,
      "residual": null
    },
    "b": {
      "expression": "0",
      "value": 0,
      "residual": null
    },
    "c": {
      "expression": "0",
      "value": 0,
      "residual": null
    },
    "d": {
      "expression": "D_n*R_k*s_Wk + s_scratch",
      "value": 1536,
      "residual": null
    },
    "remainder": {
      "expression": "0",
      "value": 0,
      "residual": null
    },
    "provenance": {
      "a": [],
      "b": [],
      "c": [],
      "d": [
        "wk_full",
        "fixed_scratch"
      ],
      "remainder": []
    },
    "undeclared": []
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
      "expression": "D_r*s_Qr + R_k*s_Qt",
      "value": 80,
      "residual": null
    },
    "c": {
      "expression": "n_buf*(D_r*s_Kr + R_k*s_Ck)",
      "value": 160,
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
        "q_tilde",
        "q_pe_tile"
      ],
      "c": [
        "kv_latent_tile",
        "k_pe_tile"
      ],
      "d": [
        "fixed_scratch"
      ],
      "remainder": []
    },
    "undeclared": []
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
      "expression": "D_r*s_Qr + R_k*s_Qt",
      "value": 80,
      "residual": null
    },
    "c": {
      "expression": "R_k*n_buf*s_Ck",
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
        "q_tilde",
        "q_pe_tile"
      ],
      "c": [
        "kv_latent_tile"
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
  "vmem:pv": {
    "stage": "pv",
    "level": "vmem",
    "a": {
      "expression": "0",
      "value": 0,
      "residual": null
    },
    "b": {
      "expression": "D_r*s_Qr + R_k*s_Qt",
      "value": 80,
      "residual": null
    },
    "c": {
      "expression": "R_k*n_buf*s_Ck",
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
        "q_tilde",
        "q_pe_tile"
      ],
      "c": [
        "kv_latent_tile"
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
      "expression": "D_r*s_Qr + D_v*s_O + R_k*s_Qt + s_LSE",
      "value": 132,
      "residual": null
    },
    "c": {
      "expression": "0",
      "value": 0,
      "residual": null
    },
    "d": {
      "expression": "D_v*R_k*s_Wv + s_scratch",
      "value": 1536,
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
        "q_tilde",
        "q_pe_tile",
        "out_tile",
        "lse_tile"
      ],
      "c": [],
      "d": [
        "wv_full",
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
      "numerator": "150*D_n*H*R_k + 150*D_n*H*R_q + 2*D_r*H*BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + 2*D_r*H*BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + 2*D_r*H*BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + 150*D_v*H*R_k + 4*H*R_k*BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + 4*H*R_k*BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + 4*H*R_k*BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96)",
      "denominator": "P_mxu",
      "expression": "(150*D_n*H*R_k + 150*D_n*H*R_q + 2*D_r*H*BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + 2*D_r*H*BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + 2*D_r*H*BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + 150*D_v*H*R_k + 4*H*R_k*BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + 4*H*R_k*BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + 4*H*R_k*BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96))/P_mxu",
      "value": null,
      "unit": "s",
      "scope": null,
      "status": "unknown",
      "missing": [
        "P_mxu",
        "schedule"
      ],
      "notes": [
        "numerator: executed matmul FLOPs of this scenario (C_exec); denominator: matmul peak for the declared compute dtype"
      ]
    },
    {
      "resource": "vpu",
      "numerator": "75*D_v*H + 75*H*R_k + 2*H*R_k*BlockStat(row_visits, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + 2*H*R_k*BlockStat(row_visits, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + 2*H*R_k*BlockStat(row_visits, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + 150*H + H*BlockStat(masked_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + H*BlockStat(masked_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + H*BlockStat(masked_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + 2*H*BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + 2*H*BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + 2*H*BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + 2*H*BlockStat(row_visits, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + 2*H*BlockStat(row_visits, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + 2*H*BlockStat(row_visits, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + Piecewise((H*BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + H*BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + H*BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96), D_r > 0), (0, True))",
      "denominator": "P_vpu",
      "expression": "(75*D_v*H + 75*H*R_k + 2*H*R_k*BlockStat(row_visits, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + 2*H*R_k*BlockStat(row_visits, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + 2*H*R_k*BlockStat(row_visits, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + 150*H + H*BlockStat(masked_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + H*BlockStat(masked_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + H*BlockStat(masked_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + 2*H*BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + 2*H*BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + 2*H*BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + 2*H*BlockStat(row_visits, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + 2*H*BlockStat(row_visits, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + 2*H*BlockStat(row_visits, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + Piecewise((H*BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + H*BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + H*BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96), D_r > 0), (0, True)))/P_vpu",
      "value": null,
      "unit": "s",
      "scope": null,
      "status": "unknown",
      "missing": [
        "P_vpu",
        "schedule"
      ],
      "notes": []
    },
    {
      "resource": "reduce",
      "numerator": "2*H*BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + 2*H*BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + 2*H*BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) - H*BlockStat(row_visits, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) - H*BlockStat(row_visits, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) - H*BlockStat(row_visits, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96)",
      "denominator": "P_red",
      "expression": "(2*H*BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + 2*H*BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + 2*H*BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) - H*BlockStat(row_visits, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) - H*BlockStat(row_visits, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) - H*BlockStat(row_visits, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96))/P_red",
      "value": null,
      "unit": "s",
      "scope": null,
      "status": "unknown",
      "missing": [
        "P_red",
        "schedule"
      ],
      "notes": []
    },
    {
      "resource": "layout",
      "numerator": "H*R_k*BlockStat(key_visits, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + H*R_k*BlockStat(key_visits, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + H*R_k*BlockStat(key_visits, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + H*BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + H*BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + H*BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96)",
      "denominator": "P_layout",
      "expression": "(H*R_k*BlockStat(key_visits, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + H*R_k*BlockStat(key_visits, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + H*R_k*BlockStat(key_visits, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + H*BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + H*BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + H*BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96))/P_layout",
      "value": null,
      "unit": "s",
      "scope": null,
      "status": "unknown",
      "missing": [
        "P_layout",
        "schedule"
      ],
      "notes": []
    },
    {
      "resource": "exp",
      "numerator": "H*BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + H*BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + H*BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + H*BlockStat(row_visits, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + H*BlockStat(row_visits, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + H*BlockStat(row_visits, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96)",
      "denominator": "P_exp",
      "expression": "(H*BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + H*BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + H*BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + H*BlockStat(row_visits, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + H*BlockStat(row_visits, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + H*BlockStat(row_visits, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96))/P_exp",
      "value": null,
      "unit": "s",
      "scope": null,
      "status": "unknown",
      "missing": [
        "P_exp",
        "schedule"
      ],
      "notes": []
    },
    {
      "resource": "path:hbm_to_vmem",
      "numerator": "D_n*H*R_k*s_Wk*ceiling(7/b_q) + D_n*H*R_k*s_Wk*ceiling(20/b_q) + D_n*H*R_k*s_Wk*ceiling(48/b_q) + D_n*H*R_q*s_Wq*ceiling(7/b_q) + D_n*H*R_q*s_Wq*ceiling(20/b_q) + D_n*H*R_q*s_Wq*ceiling(48/b_q) + 75*D_r*H*s_Qr + D_r*s_Kr*ceiling(H/h_pp)*BlockStat(key_visits, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + D_r*s_Kr*ceiling(H/h_pp)*BlockStat(key_visits, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + D_r*s_Kr*ceiling(H/h_pp)*BlockStat(key_visits, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + D_v*H*R_k*s_Wv*ceiling(7/b_q) + D_v*H*R_k*s_Wv*ceiling(20/b_q) + D_v*H*R_k*s_Wv*ceiling(48/b_q) + R_k*s_Ck*ceiling(H/h_pp)*BlockStat(key_visits, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + R_k*s_Ck*ceiling(H/h_pp)*BlockStat(key_visits, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + R_k*s_Ck*ceiling(H/h_pp)*BlockStat(key_visits, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + 75*R_q*s_Cq*ceiling(H/h_pp)",
      "denominator": "W_hbm",
      "expression": "(D_n*H*R_k*s_Wk*ceiling(7/b_q) + D_n*H*R_k*s_Wk*ceiling(20/b_q) + D_n*H*R_k*s_Wk*ceiling(48/b_q) + D_n*H*R_q*s_Wq*ceiling(7/b_q) + D_n*H*R_q*s_Wq*ceiling(20/b_q) + D_n*H*R_q*s_Wq*ceiling(48/b_q) + 75*D_r*H*s_Qr + D_r*s_Kr*ceiling(H/h_pp)*BlockStat(key_visits, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + D_r*s_Kr*ceiling(H/h_pp)*BlockStat(key_visits, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + D_r*s_Kr*ceiling(H/h_pp)*BlockStat(key_visits, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + D_v*H*R_k*s_Wv*ceiling(7/b_q) + D_v*H*R_k*s_Wv*ceiling(20/b_q) + D_v*H*R_k*s_Wv*ceiling(48/b_q) + R_k*s_Ck*ceiling(H/h_pp)*BlockStat(key_visits, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + R_k*s_Ck*ceiling(H/h_pp)*BlockStat(key_visits, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + R_k*s_Ck*ceiling(H/h_pp)*BlockStat(key_visits, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + 75*R_q*s_Cq*ceiling(H/h_pp))/W_hbm",
      "value": null,
      "unit": "s",
      "scope": null,
      "status": "unknown",
      "missing": [
        "W_hbm",
        "schedule"
      ],
      "notes": [
        "conditional transfer requests of this scenario / path bandwidth upper bound; not an absolute limit for all algorithms"
      ]
    },
    {
      "resource": "path:vmem_to_hbm",
      "numerator": "75*D_v*H*s_O + 75*H*s_LSE",
      "denominator": "W_hbm_w",
      "expression": "(75*D_v*H*s_O + 75*H*s_LSE)/W_hbm_w",
      "value": null,
      "unit": "s",
      "scope": null,
      "status": "unknown",
      "missing": [
        "W_hbm_w",
        "schedule"
      ],
      "notes": [
        "conditional transfer requests of this scenario / path bandwidth upper bound; not an absolute limit for all algorithms"
      ]
    },
    {
      "resource": "path:vmem_to_vreg",
      "numerator": "D_r*H*b_k*s_Kr*BlockStat(selected, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + D_r*H*b_k*s_Kr*BlockStat(selected, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + D_r*H*b_k*s_Kr*BlockStat(selected, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + D_r*H*b_q*s_Qr*ceiling(7/b_q) + D_r*H*b_q*s_Qr*ceiling(20/b_q) + D_r*H*b_q*s_Qr*ceiling(48/b_q) + H*R_k*b_k*s_Ck*BlockStat(selected, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + H*R_k*b_k*s_Ck*BlockStat(selected, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + H*R_k*b_k*s_Ck*BlockStat(selected, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + H*R_k*b_q*s_Qt*ceiling(7/b_q) + H*R_k*b_q*s_Qt*ceiling(20/b_q) + H*R_k*b_q*s_Qt*ceiling(48/b_q)",
      "denominator": "W_vmem",
      "expression": "(D_r*H*b_k*s_Kr*BlockStat(selected, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + D_r*H*b_k*s_Kr*BlockStat(selected, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + D_r*H*b_k*s_Kr*BlockStat(selected, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + D_r*H*b_q*s_Qr*ceiling(7/b_q) + D_r*H*b_q*s_Qr*ceiling(20/b_q) + D_r*H*b_q*s_Qr*ceiling(48/b_q) + H*R_k*b_k*s_Ck*BlockStat(selected, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + H*R_k*b_k*s_Ck*BlockStat(selected, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + H*R_k*b_k*s_Ck*BlockStat(selected, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + H*R_k*b_q*s_Qt*ceiling(7/b_q) + H*R_k*b_q*s_Qt*ceiling(20/b_q) + H*R_k*b_q*s_Qt*ceiling(48/b_q))/W_vmem",
      "value": null,
      "unit": "s",
      "scope": null,
      "status": "unknown",
      "missing": [
        "W_vmem",
        "q_resident",
        "register_schedule_evidence",
        "schedule"
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
  "combined_expression": "max((150*D_n*H*R_k + 150*D_n*H*R_q + 2*D_r*H*BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + 2*D_r*H*BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + 2*D_r*H*BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + 150*D_v*H*R_k + 4*H*R_k*BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + 4*H*R_k*BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + 4*H*R_k*BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96))/P_mxu, (75*D_v*H + 75*H*R_k + 2*H*R_k*BlockStat(row_visits, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + 2*H*R_k*BlockStat(row_visits, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + 2*H*R_k*BlockStat(row_visits, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + 150*H + H*BlockStat(masked_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + H*BlockStat(masked_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + H*BlockStat(masked_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + 2*H*BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + 2*H*BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + 2*H*BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + 2*H*BlockStat(row_visits, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + 2*H*BlockStat(row_visits, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + 2*H*BlockStat(row_visits, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + Piecewise((H*BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + H*BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + H*BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96), D_r > 0), (0, True)))/P_vpu, (2*H*BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + 2*H*BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + 2*H*BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) - H*BlockStat(row_visits, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) - H*BlockStat(row_visits, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) - H*BlockStat(row_visits, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96))/P_red, (H*R_k*BlockStat(key_visits, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + H*R_k*BlockStat(key_visits, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + H*R_k*BlockStat(key_visits, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + H*BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + H*BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + H*BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96))/P_layout, (H*BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + H*BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + H*BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + H*BlockStat(row_visits, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + H*BlockStat(row_visits, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + H*BlockStat(row_visits, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96))/P_exp, (D_n*H*R_k*s_Wk*ceiling(7/b_q) + D_n*H*R_k*s_Wk*ceiling(20/b_q) + D_n*H*R_k*s_Wk*ceiling(48/b_q) + D_n*H*R_q*s_Wq*ceiling(7/b_q) + D_n*H*R_q*s_Wq*ceiling(20/b_q) + D_n*H*R_q*s_Wq*ceiling(48/b_q) + 75*D_r*H*s_Qr + D_r*s_Kr*ceiling(H/h_pp)*BlockStat(key_visits, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + D_r*s_Kr*ceiling(H/h_pp)*BlockStat(key_visits, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + D_r*s_Kr*ceiling(H/h_pp)*BlockStat(key_visits, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + D_v*H*R_k*s_Wv*ceiling(7/b_q) + D_v*H*R_k*s_Wv*ceiling(20/b_q) + D_v*H*R_k*s_Wv*ceiling(48/b_q) + R_k*s_Ck*ceiling(H/h_pp)*BlockStat(key_visits, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + R_k*s_Ck*ceiling(H/h_pp)*BlockStat(key_visits, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + R_k*s_Ck*ceiling(H/h_pp)*BlockStat(key_visits, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + 75*R_q*s_Cq*ceiling(H/h_pp))/W_hbm, (75*D_v*H*s_O + 75*H*s_LSE)/W_hbm_w, (D_r*H*b_k*s_Kr*BlockStat(selected, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + D_r*H*b_k*s_Kr*BlockStat(selected, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + D_r*H*b_k*s_Kr*BlockStat(selected, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + D_r*H*b_q*s_Qr*ceiling(7/b_q) + D_r*H*b_q*s_Qr*ceiling(20/b_q) + D_r*H*b_q*s_Qr*ceiling(48/b_q) + H*R_k*b_k*s_Ck*BlockStat(selected, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + H*R_k*b_k*s_Ck*BlockStat(selected, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + H*R_k*b_k*s_Ck*BlockStat(selected, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96) + H*R_k*b_q*s_Qt*ceiling(7/b_q) + H*R_k*b_q*s_Qt*ceiling(20/b_q) + H*R_k*b_q*s_Qt*ceiling(48/b_q))/W_vmem, CP)",
  "resources_outside_declared_groups": [],
  "combined_value_known_terms_only": null,
  "status": "symbolic",
  "overlap_model": "full_overlap_max",
  "combined_missing_fields": [
    "CP",
    "P_exp",
    "P_layout",
    "P_mxu",
    "P_red",
    "P_vpu",
    "W_hbm",
    "W_hbm_w",
    "W_vmem",
    "q_resident",
    "register_schedule_evidence",
    "schedule"
  ],
  "scenarios": {},
  "notes": [
    "T_actual >= T_LB only for the counted scenario; missing resources make T_LB a lower bound on the lower bound",
    "serial_stage_sum groups are resource classes of the whole kernel (declared non-overlap between them), not per-pipeline-stage attribution"
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
    "gamma": [
      "D_n",
      "D_r"
    ],
    "rows_without_visible_keys": [],
    "C_valid": [
      "H"
    ],
    "C_rect": [
      "H",
      "b_k",
      "b_q"
    ],
    "C_pad": [
      "H",
      "b_k",
      "b_q"
    ],
    "rect_waste": [
      "H",
      "b_k",
      "b_q"
    ],
    "selected_blocks": [
      "H",
      "b_k",
      "b_q"
    ],
    "partial_blocks": [
      "H",
      "b_k",
      "b_q"
    ],
    "fully_visible_blocks": [
      "H",
      "b_k",
      "b_q"
    ],
    "future_blocks_selected": [
      "H",
      "b_k",
      "b_q"
    ],
    "padding_blocks_selected": [
      "H",
      "b_k",
      "b_q"
    ],
    "future_blocks_skipped": [
      "H",
      "b_k",
      "b_q"
    ],
    "padding_blocks_skipped": [
      "H",
      "b_k",
      "b_q"
    ],
    "masked_cells": [
      "H",
      "b_k",
      "b_q"
    ],
    "F_Q[useful]": [
      "D_n",
      "H",
      "R_q"
    ],
    "F_Qtilde[useful]": [
      "D_n",
      "H",
      "R_k"
    ],
    "F_QC[useful]": [
      "H",
      "R_k"
    ],
    "F_QK_r[useful]": [
      "D_r",
      "H"
    ],
    "F_PC[useful]": [
      "H",
      "R_k"
    ],
    "F_ZWv[useful]": [
      "D_v",
      "H",
      "R_k"
    ],
    "F_total[useful]": [
      "D_n",
      "D_r",
      "D_v",
      "H",
      "R_k",
      "R_q"
    ],
    "F_A[spec_closed_form]": [
      "D_n",
      "D_r",
      "D_v",
      "H",
      "R_k",
      "R_q"
    ],
    "N_Qproj": [
      "H"
    ],
    "N_Kproj": [],
    "N_Vproj": [],
    "F_proj[general]": [
      "D_n",
      "H",
      "R_k",
      "R_q"
    ],
    "F_total[rect]": [
      "D_n",
      "D_r",
      "D_v",
      "H",
      "R_k",
      "R_q",
      "b_k",
      "b_q"
    ],
    "F_total[executed_graph]": [
      "D_n",
      "D_r",
      "D_v",
      "H",
      "R_k",
      "R_q",
      "b_k",
      "b_q"
    ],
    "F_projection[executed_graph]": [
      "D_n",
      "H",
      "R_k",
      "R_q"
    ],
    "F_attention[executed_graph]": [
      "D_r",
      "H",
      "R_k",
      "b_k",
      "b_q"
    ],
    "F_output_projection[executed_graph]": [
      "D_v",
      "H",
      "R_k"
    ],
    "F_A_minus_F_E": [
      "D_n",
      "D_r",
      "D_v",
      "H",
      "R_k"
    ],
    "F_A_minus_F_E[spec_closed_form]": [
      "D_n",
      "D_v",
      "H",
      "R_k"
    ],
    "W_vec[mul]": [
      "H",
      "R_k",
      "b_k",
      "b_q"
    ],
    "W_vec[mask]": [
      "H",
      "b_k",
      "b_q"
    ],
    "W_vec[cmp]": [
      "H",
      "b_k",
      "b_q"
    ],
    "W_vec[broadcast]": [
      "H",
      "b_k",
      "b_q"
    ],
    "W_vec[transpose]": [
      "H",
      "R_k",
      "b_k",
      "b_q"
    ],
    "W_vec[exp]": [
      "H",
      "b_k",
      "b_q"
    ],
    "W_vec[add]": [
      "D_r",
      "H",
      "R_k",
      "b_k",
      "b_q"
    ],
    "W_vec[div]": [
      "H",
      "R_k"
    ],
    "W_vec[recip]": [],
    "W_vec[log]": [
      "H"
    ],
    "W_vec[cast]": [
      "D_v",
      "H",
      "b_k",
      "b_q"
    ],
    "W_resource[vpu]": [
      "D_r",
      "D_v",
      "H",
      "R_k",
      "b_k",
      "b_q"
    ],
    "W_resource[reduce]": [
      "H",
      "b_k",
      "b_q"
    ],
    "W_resource[layout]": [
      "H",
      "R_k",
      "b_k",
      "b_q"
    ],
    "W_resource[exp]": [
      "H",
      "b_k",
      "b_q"
    ],
    "exp_count_executed": [
      "H",
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
      "D_v",
      "H",
      "s_O"
    ],
    "M_LSE": [
      "H",
      "s_LSE"
    ],
    "HBM_mat[Q~]": [
      "H",
      "R_k",
      "s_Qt"
    ],
    "M_mat[Q~]": [
      "H",
      "R_k",
      "s_Qt"
    ],
    "HBM_mat[Z]": [
      "H",
      "R_k",
      "s_Z"
    ],
    "M_mat[Z]": [
      "H",
      "R_k",
      "s_Z"
    ],
    "B[hbm_to_vmem]": [
      "D_n",
      "D_r",
      "D_v",
      "H",
      "R_k",
      "R_q",
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
      "D_v",
      "H",
      "s_LSE",
      "s_O"
    ],
    "B[vmem_to_vreg]": [
      "D_r",
      "H",
      "R_k",
      "b_k",
      "b_q",
      "s_Ck",
      "s_Kr",
      "s_Qr",
      "s_Qt"
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
    "local[wk_full]": [
      "D_n",
      "R_k",
      "s_Wk"
    ],
    "local[q_tilde_acc]": [
      "R_k",
      "b_q",
      "s_Qacc"
    ],
    "local[q_tilde]": [
      "R_k",
      "b_q",
      "s_Qt"
    ],
    "local[kv_latent_tile]": [
      "R_k",
      "b_k",
      "n_buf",
      "s_Ck"
    ],
    "local[wv_full]": [
      "D_v",
      "R_k",
      "s_Wv"
    ],
    "local[z_tile]": [
      "R_k",
      "b_q",
      "s_Z"
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
      "R_k",
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
      "L_l",
      "L_s",
      "b_q",
      "s_Qacc"
    ],
    "layout[wk_full]": [
      "D_n",
      "L_l",
      "L_s",
      "R_k",
      "s_Wk"
    ],
    "layout[q_tilde_acc]": [
      "L_l",
      "L_s",
      "R_k",
      "b_q",
      "s_Qacc"
    ],
    "layout[q_tilde]": [
      "L_l",
      "L_s",
      "R_k",
      "b_q",
      "s_Qt"
    ],
    "layout[kv_latent_tile]": [
      "L_l",
      "L_s",
      "R_k",
      "b_k",
      "n_buf",
      "s_Ck"
    ],
    "layout[wv_full]": [
      "D_v",
      "L_l",
      "L_s",
      "R_k",
      "s_Wv"
    ],
    "layout[z_tile]": [
      "L_l",
      "L_s",
      "R_k",
      "b_q",
      "s_Z"
    ],
    "layout[score_X]": [
      "L_l",
      "L_s",
      "b_k",
      "b_q",
      "s_X"
    ],
    "layout[exp_E]": [
      "L_l",
      "L_s",
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
      "L_l",
      "L_s",
      "b_q",
      "s_state"
    ],
    "layout[row_state_l]": [
      "L_l",
      "L_s",
      "b_q",
      "s_state"
    ],
    "layout[row_alpha]": [
      "L_l",
      "L_s",
      "b_q",
      "s_state"
    ],
    "layout[accumulator_A]": [
      "L_l",
      "L_s",
      "R_k",
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
      "L_l",
      "L_s",
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
    "M_live[all:q_absorb]": [
      "D_n",
      "R_k",
      "b_q",
      "s_Qacc",
      "s_Wk",
      "s_scratch"
    ],
    "M_live[all:score]": [
      "D_r",
      "R_k",
      "b_k",
      "b_q",
      "n_buf",
      "s_A",
      "s_Ck",
      "s_E",
      "s_Kr",
      "s_Qr",
      "s_Qt",
      "s_X",
      "s_Z",
      "s_scratch",
      "s_state"
    ],
    "M_live[all:softmax]": [
      "D_r",
      "R_k",
      "b_k",
      "b_q",
      "n_buf",
      "s_A",
      "s_Ck",
      "s_E",
      "s_Qr",
      "s_Qt",
      "s_X",
      "s_Z",
      "s_scratch",
      "s_state"
    ],
    "M_live[all:pv]": [
      "D_r",
      "R_k",
      "b_k",
      "b_q",
      "n_buf",
      "s_A",
      "s_Ck",
      "s_E",
      "s_P",
      "s_Qr",
      "s_Qt",
      "s_X",
      "s_Z",
      "s_scratch",
      "s_state"
    ],
    "M_live[all:finalize]": [
      "D_r",
      "D_v",
      "R_k",
      "b_q",
      "s_A",
      "s_LSE",
      "s_O",
      "s_Qr",
      "s_Qt",
      "s_Wv",
      "s_Z",
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
      "s_Kr",
      "s_LSE",
      "s_O",
      "s_P",
      "s_Qacc",
      "s_Qr",
      "s_Qt",
      "s_Wk",
      "s_Wq",
      "s_Wv",
      "s_X",
      "s_Z",
      "s_scratch",
      "s_state"
    ],
    "M_live[vreg:q_proj]": [
      "D_n",
      "b_q",
      "s_Qacc"
    ],
    "M_live[vreg:q_absorb]": [
      "D_n",
      "R_k",
      "b_q",
      "s_Qacc"
    ],
    "M_live[vreg:score]": [
      "R_k",
      "b_k",
      "b_q",
      "s_A",
      "s_E",
      "s_X",
      "s_Z",
      "s_state"
    ],
    "M_live[vreg:softmax]": [
      "R_k",
      "b_k",
      "b_q",
      "s_A",
      "s_E",
      "s_X",
      "s_Z",
      "s_state"
    ],
    "M_live[vreg:pv]": [
      "R_k",
      "b_k",
      "b_q",
      "s_A",
      "s_E",
      "s_P",
      "s_X",
      "s_Z",
      "s_state"
    ],
    "M_live[vreg:finalize]": [
      "R_k",
      "b_q",
      "s_A",
      "s_Z",
      "s_state"
    ],
    "M_peak[vreg]": [
      "D_n",
      "R_k",
      "b_k",
      "b_q",
      "s_A",
      "s_E",
      "s_P",
      "s_Qacc",
      "s_X",
      "s_Z",
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
    "M_live[vmem:q_absorb]": [
      "D_n",
      "R_k",
      "s_Wk",
      "s_scratch"
    ],
    "M_live[vmem:score]": [
      "D_r",
      "R_k",
      "b_k",
      "b_q",
      "n_buf",
      "s_Ck",
      "s_Kr",
      "s_Qr",
      "s_Qt",
      "s_scratch"
    ],
    "M_live[vmem:softmax]": [
      "D_r",
      "R_k",
      "b_k",
      "b_q",
      "n_buf",
      "s_Ck",
      "s_Qr",
      "s_Qt",
      "s_scratch"
    ],
    "M_live[vmem:pv]": [
      "D_r",
      "R_k",
      "b_k",
      "b_q",
      "n_buf",
      "s_Ck",
      "s_Qr",
      "s_Qt",
      "s_scratch"
    ],
    "M_live[vmem:finalize]": [
      "D_r",
      "D_v",
      "R_k",
      "b_q",
      "s_LSE",
      "s_O",
      "s_Qr",
      "s_Qt",
      "s_Wv",
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
      "s_Kr",
      "s_LSE",
      "s_O",
      "s_Qr",
      "s_Qt",
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
    "envelope[all:q_absorb]": [
      "D_n",
      "R_k",
      "b_q",
      "s_Qacc",
      "s_Wk",
      "s_scratch"
    ],
    "envelope[all:score]": [
      "D_r",
      "R_k",
      "b_k",
      "b_q",
      "n_buf",
      "s_A",
      "s_Ck",
      "s_E",
      "s_Kr",
      "s_Qr",
      "s_Qt",
      "s_X",
      "s_Z",
      "s_scratch",
      "s_state"
    ],
    "envelope[all:softmax]": [
      "D_r",
      "R_k",
      "b_k",
      "b_q",
      "n_buf",
      "s_A",
      "s_Ck",
      "s_E",
      "s_Qr",
      "s_Qt",
      "s_X",
      "s_Z",
      "s_scratch",
      "s_state"
    ],
    "envelope[all:pv]": [
      "D_r",
      "R_k",
      "b_k",
      "b_q",
      "n_buf",
      "s_A",
      "s_Ck",
      "s_E",
      "s_P",
      "s_Qr",
      "s_Qt",
      "s_X",
      "s_Z",
      "s_scratch",
      "s_state"
    ],
    "envelope[all:finalize]": [
      "D_r",
      "D_v",
      "R_k",
      "b_q",
      "s_A",
      "s_LSE",
      "s_O",
      "s_Qr",
      "s_Qt",
      "s_Wv",
      "s_Z",
      "s_scratch",
      "s_state"
    ],
    "envelope[vreg:q_proj]": [
      "D_n",
      "b_q",
      "s_Qacc"
    ],
    "envelope[vreg:q_absorb]": [
      "D_n",
      "R_k",
      "b_q",
      "s_Qacc"
    ],
    "envelope[vreg:score]": [
      "R_k",
      "b_k",
      "b_q",
      "s_A",
      "s_E",
      "s_X",
      "s_Z",
      "s_state"
    ],
    "envelope[vreg:softmax]": [
      "R_k",
      "b_k",
      "b_q",
      "s_A",
      "s_E",
      "s_X",
      "s_Z",
      "s_state"
    ],
    "envelope[vreg:pv]": [
      "R_k",
      "b_k",
      "b_q",
      "s_A",
      "s_E",
      "s_P",
      "s_X",
      "s_Z",
      "s_state"
    ],
    "envelope[vreg:finalize]": [
      "R_k",
      "b_q",
      "s_A",
      "s_Z",
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
    "envelope[vmem:q_absorb]": [
      "D_n",
      "R_k",
      "s_Wk",
      "s_scratch"
    ],
    "envelope[vmem:score]": [
      "D_r",
      "R_k",
      "b_k",
      "b_q",
      "n_buf",
      "s_Ck",
      "s_Kr",
      "s_Qr",
      "s_Qt",
      "s_scratch"
    ],
    "envelope[vmem:softmax]": [
      "D_r",
      "R_k",
      "b_k",
      "b_q",
      "n_buf",
      "s_Ck",
      "s_Qr",
      "s_Qt",
      "s_scratch"
    ],
    "envelope[vmem:pv]": [
      "D_r",
      "R_k",
      "b_k",
      "b_q",
      "n_buf",
      "s_Ck",
      "s_Qr",
      "s_Qt",
      "s_scratch"
    ],
    "envelope[vmem:finalize]": [
      "D_r",
      "D_v",
      "R_k",
      "b_q",
      "s_LSE",
      "s_O",
      "s_Qr",
      "s_Qt",
      "s_Wv",
      "s_scratch"
    ],
    "excess_over_budget[vreg:q_proj]": [
      "D_n",
      "R_vreg",
      "b_q",
      "s_Qacc"
    ],
    "excess_over_budget[vreg:q_absorb]": [
      "D_n",
      "R_k",
      "R_vreg",
      "b_q",
      "s_Qacc"
    ],
    "b_k_max[vreg:score]": [
      "R_k",
      "R_vreg",
      "b_q",
      "s_A",
      "s_E",
      "s_X",
      "s_Z",
      "s_state"
    ],
    "excess_over_budget[vreg:score]": [
      "R_k",
      "R_vreg",
      "b_k",
      "b_q",
      "s_A",
      "s_E",
      "s_X",
      "s_Z",
      "s_state"
    ],
    "b_k_max[vreg:softmax]": [
      "R_k",
      "R_vreg",
      "b_q",
      "s_A",
      "s_E",
      "s_X",
      "s_Z",
      "s_state"
    ],
    "excess_over_budget[vreg:softmax]": [
      "R_k",
      "R_vreg",
      "b_k",
      "b_q",
      "s_A",
      "s_E",
      "s_X",
      "s_Z",
      "s_state"
    ],
    "b_k_max[vreg:pv]": [
      "R_k",
      "R_vreg",
      "b_q",
      "s_A",
      "s_E",
      "s_P",
      "s_X",
      "s_Z",
      "s_state"
    ],
    "excess_over_budget[vreg:pv]": [
      "R_k",
      "R_vreg",
      "b_k",
      "b_q",
      "s_A",
      "s_E",
      "s_P",
      "s_X",
      "s_Z",
      "s_state"
    ],
    "excess_over_budget[vreg:finalize]": [
      "R_k",
      "R_vreg",
      "b_q",
      "s_A",
      "s_Z",
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
    "excess_over_budget[vmem:q_absorb]": [
      "D_n",
      "R_k",
      "R_vmem",
      "s_Wk",
      "s_scratch"
    ],
    "b_k_max[vmem:score]": [
      "D_r",
      "R_k",
      "R_vmem",
      "b_q",
      "n_buf",
      "s_Ck",
      "s_Kr",
      "s_Qr",
      "s_Qt",
      "s_scratch"
    ],
    "excess_over_budget[vmem:score]": [
      "D_r",
      "R_k",
      "R_vmem",
      "b_k",
      "b_q",
      "n_buf",
      "s_Ck",
      "s_Kr",
      "s_Qr",
      "s_Qt",
      "s_scratch"
    ],
    "b_k_max[vmem:softmax]": [
      "D_r",
      "R_k",
      "R_vmem",
      "b_q",
      "n_buf",
      "s_Ck",
      "s_Qr",
      "s_Qt",
      "s_scratch"
    ],
    "excess_over_budget[vmem:softmax]": [
      "D_r",
      "R_k",
      "R_vmem",
      "b_k",
      "b_q",
      "n_buf",
      "s_Ck",
      "s_Qr",
      "s_Qt",
      "s_scratch"
    ],
    "b_k_max[vmem:pv]": [
      "D_r",
      "R_k",
      "R_vmem",
      "b_q",
      "n_buf",
      "s_Ck",
      "s_Qr",
      "s_Qt",
      "s_scratch"
    ],
    "excess_over_budget[vmem:pv]": [
      "D_r",
      "R_k",
      "R_vmem",
      "b_k",
      "b_q",
      "n_buf",
      "s_Ck",
      "s_Qr",
      "s_Qt",
      "s_scratch"
    ],
    "excess_over_budget[vmem:finalize]": [
      "D_r",
      "D_v",
      "R_k",
      "R_vmem",
      "b_q",
      "s_LSE",
      "s_O",
      "s_Qr",
      "s_Qt",
      "s_Wv",
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
      "R_k",
      "s_A"
    ],
    "d(accumulator_bytes)/d(b_k)": [],
    "n_programs": [
      "H",
      "b_q",
      "h_pp"
    ],
    "n_kv_block_visits": [
      "H",
      "b_k",
      "b_q"
    ]
  },
  "matmul_nodes": [
    {
      "id": "Q_proj",
      "stage": "q_proj",
      "M": "75*H",
      "N": "D_n",
      "K": "R_q",
      "multiplicity": "1",
      "tile_shape": null,
      "tile_shape_value": null,
      "counts_toward": "projection"
    },
    {
      "id": "Q_absorb",
      "stage": "q_absorb",
      "M": "75*H",
      "N": "R_k",
      "K": "D_n",
      "multiplicity": "1",
      "tile_shape": null,
      "tile_shape_value": null,
      "counts_toward": "projection"
    },
    {
      "id": "QC_latent",
      "stage": "score",
      "M": "H*(BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96))",
      "N": "1",
      "K": "R_k",
      "multiplicity": "1",
      "tile_shape": [
        "b_q",
        "b_k",
        "R_k"
      ],
      "tile_shape_value": [
        16,
        32,
        32
      ],
      "counts_toward": "attention"
    },
    {
      "id": "QK_rope",
      "stage": "score",
      "M": "H*(BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96))",
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
        32,
        8
      ],
      "counts_toward": "attention"
    },
    {
      "id": "PC_latent",
      "stage": "pv",
      "M": "H*(BlockStat(rect_cells, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + BlockStat(rect_cells, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + BlockStat(rect_cells, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96))",
      "N": "1",
      "K": "R_k",
      "multiplicity": "1",
      "tile_shape": [
        "b_q",
        "R_k",
        "b_k"
      ],
      "tile_shape_value": [
        16,
        32,
        32
      ],
      "counts_toward": "attention"
    },
    {
      "id": "Z_Wv",
      "stage": "finalize",
      "M": "75*H",
      "N": "D_v",
      "K": "R_k",
      "multiplicity": "1",
      "tile_shape": null,
      "tile_shape_value": null,
      "counts_toward": "output_projection"
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
        "q_proj",
        "q_absorb"
      ],
      "alias_group": null,
      "exists": true,
      "conditional_on": null,
      "size_scenario": null,
      "stage_scenario": null
    },
    {
      "id": "wk_full",
      "size": "D_n*R_k*s_Wk",
      "level": "vmem",
      "live_stages": [
        "q_absorb"
      ],
      "alias_group": null,
      "exists": true,
      "conditional_on": null,
      "size_scenario": null,
      "stage_scenario": null
    },
    {
      "id": "q_tilde_acc",
      "size": "R_k*b_q*s_Qacc",
      "level": "vreg",
      "live_stages": [
        "q_absorb"
      ],
      "alias_group": null,
      "exists": true,
      "conditional_on": null,
      "size_scenario": null,
      "stage_scenario": null
    },
    {
      "id": "q_tilde",
      "size": "R_k*b_q*s_Qt",
      "level": "vmem",
      "live_stages": [
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
      "id": "kv_latent_tile",
      "size": "R_k*b_k*n_buf*s_Ck",
      "level": "vmem",
      "live_stages": [
        "score",
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
      "id": "wv_full",
      "size": "D_v*R_k*s_Wv",
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
      "id": "z_tile",
      "size": "R_k*b_q*s_Z",
      "level": "vreg",
      "live_stages": [
        "finalize"
      ],
      "alias_group": "acc_or_Z",
      "exists": true,
      "conditional_on": null,
      "size_scenario": null,
      "stage_scenario": null
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
      "size": "R_k*b_q*s_A",
      "level": "vreg",
      "live_stages": [
        "score",
        "softmax",
        "pv",
        "finalize"
      ],
      "alias_group": "acc_or_Z",
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
        "q_absorb",
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
      "total": "75*R_q*s_Cq*ceiling(H/h_pp)",
      "undeclared": false
    },
    {
      "id": "Qr_read",
      "path": "hbm_to_vmem",
      "total": "75*D_r*H*s_Qr",
      "undeclared": false
    },
    {
      "id": "Ck_read",
      "path": "hbm_to_vmem",
      "total": "R_k*s_Ck*ceiling(H/h_pp)*(BlockStat(key_visits, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + BlockStat(key_visits, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + BlockStat(key_visits, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96))",
      "undeclared": false
    },
    {
      "id": "Kr_read",
      "path": "hbm_to_vmem",
      "total": "D_r*s_Kr*ceiling(H/h_pp)*(BlockStat(key_visits, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + BlockStat(key_visits, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + BlockStat(key_visits, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96))",
      "undeclared": false
    },
    {
      "id": "Wv_read",
      "path": "hbm_to_vmem",
      "total": "D_v*H*R_k*s_Wv*(ceiling(7/b_q) + ceiling(20/b_q) + ceiling(48/b_q))",
      "undeclared": false
    },
    {
      "id": "Wq_read",
      "path": "hbm_to_vmem",
      "total": "D_n*H*R_q*s_Wq*(ceiling(7/b_q) + ceiling(20/b_q) + ceiling(48/b_q))",
      "undeclared": false
    },
    {
      "id": "Wk_read",
      "path": "hbm_to_vmem",
      "total": "D_n*H*R_k*s_Wk*(ceiling(7/b_q) + ceiling(20/b_q) + ceiling(48/b_q))",
      "undeclared": false
    },
    {
      "id": "O_write",
      "path": "vmem_to_hbm",
      "total": "75*D_v*H*s_O",
      "undeclared": false
    },
    {
      "id": "LSE_write",
      "path": "vmem_to_hbm",
      "total": "75*H*s_LSE",
      "undeclared": false
    },
    {
      "id": "stream_Qt_operand",
      "path": "vmem_to_vreg",
      "total": "H*R_k*b_q*s_Qt*(ceiling(7/b_q) + ceiling(20/b_q) + ceiling(48/b_q))",
      "undeclared": true
    },
    {
      "id": "stream_Qr_operand",
      "path": "vmem_to_vreg",
      "total": "D_r*H*b_q*s_Qr*(ceiling(7/b_q) + ceiling(20/b_q) + ceiling(48/b_q))",
      "undeclared": true
    },
    {
      "id": "stream_Ck_tile",
      "path": "vmem_to_vreg",
      "total": "H*R_k*b_k*s_Ck*(BlockStat(selected, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + BlockStat(selected, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + BlockStat(selected, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96))",
      "undeclared": true
    },
    {
      "id": "stream_Kr_tile",
      "path": "vmem_to_vreg",
      "total": "D_r*H*b_k*s_Kr*(BlockStat(selected, causal, 7, 50, b_q, b_k, 43, skip_future, 0, 0, 7, 50) + BlockStat(selected, causal, 20, 20, b_q, b_k, 0, skip_future, 0, 0, 20, 20) + BlockStat(selected, causal, 48, 96, b_q, b_k, 48, skip_future, 0, 0, 48, 96))",
      "undeclared": true
    }
  ],
  "stages": [
    "q_proj",
    "q_absorb",
    "score",
    "softmax",
    "pv",
    "finalize"
  ]
}
```
