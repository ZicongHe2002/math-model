# MLA forward — symbolic description (absorbed_two_step)

- mode: **symbolic**  |  adapter: `seven_input_latent`  |  algorithm: `absorbed_two_step`

## Bindings and declarations

- **bindings**: `{}`
- **semantics**: `{"mask": null, "position_offset": null, "query_position_start": null, "key_position_start": null, "scale_policy": null, "scale_value": null, "scale_match_rel...`
- **implementation**: `{"name": "symbolic", "b_q": null, "b_k": null, "b_rq": null, "b_rk": null, "scheduled_lengths": null, "rect_policy": null, "subdivide": null, "executed_exten...`
- **numerics**: `{"matmul_input_dtype": null, "accumulator_dtype": null, "score_dtype": null, "exp_dtype": null, "p_operand_dtype": null, "state_dtype": null, "output_dtype":...`
- **numerics_assumptions**: `[]`
- **hardware**: `null`

## Unresolved inputs

- mask not declared: visibility metrics unknown
- B: needed by 40 metric(s), e.g. ['B', 'F_Q[useful]', 'F_Qtilde[useful]']
- CP: needed by 3 metric(s), e.g. ['T_LB[critical_path]', 'T_LB[combined:scenario=full_overlap_max]', 'T_LB[combined:scenario=serial_stage_sum]']
- C_masked: needed by 5 metric(s), e.g. ['W_vec[mask]', 'W_resource[vpu]', 'T_LB[vpu]']
- C_pad: needed by 20 metric(s), e.g. ['F_total[executed_graph]', 'F_attention[executed_graph]', 'W_vec[mul]']
- C_valid: needed by 10 metric(s), e.g. ['F_Q[useful]', 'F_Qtilde[useful]', 'F_QC[useful]']
- D_n: needed by 43 metric(s), e.g. ['D_n', 'F_Q[useful]', 'F_Qtilde[useful]']
- D_r: needed by 49 metric(s), e.g. ['D_r', 'F_QK_r[useful]', 'F_total[useful]']
- D_v: needed by 32 metric(s), e.g. ['D_v', 'F_ZWv[useful]', 'F_total[useful]']
- Delta: needed by 1 metric(s), e.g. ['Delta']
- H: needed by 43 metric(s), e.g. ['H', 'F_Q[useful]', 'F_Qtilde[useful]']
- L_l: needed by 21 metric(s), e.g. ['layout[q_latent_window]', 'layout[wq_rank_tile]', 'layout[q_proj_acc]']
- L_s: needed by 21 metric(s), e.g. ['layout[q_latent_window]', 'layout[wq_rank_tile]', 'layout[q_proj_acc]']
- N_keyvisits: needed by 7 metric(s), e.g. ['W_vec[transpose]', 'W_resource[layout]', 'B[hbm_to_vmem]']
- N_kvblocks: needed by 5 metric(s), e.g. ['B[vmem_to_vreg]', 'n_kv_block_visits', 'T_LB[path:vmem_to_vreg]']
- N_rowvisits: needed by 11 metric(s), e.g. ['W_vec[mul]', 'W_vec[exp]', 'W_vec[add]']
- P_exp: needed by 3 metric(s), e.g. ['T_LB[exp]', 'T_LB[combined:scenario=full_overlap_max]', 'T_LB[combined:scenario=serial_stage_sum]']
- P_layout: needed by 3 metric(s), e.g. ['T_LB[layout]', 'T_LB[combined:scenario=full_overlap_max]', 'T_LB[combined:scenario=serial_stage_sum]']
- P_mxu: needed by 3 metric(s), e.g. ['T_LB[mxu]', 'T_LB[combined:scenario=full_overlap_max]', 'T_LB[combined:scenario=serial_stage_sum]']
- P_red: needed by 3 metric(s), e.g. ['T_LB[reduce]', 'T_LB[combined:scenario=full_overlap_max]', 'T_LB[combined:scenario=serial_stage_sum]']
- P_vpu: needed by 3 metric(s), e.g. ['T_LB[vpu]', 'T_LB[combined:scenario=full_overlap_max]', 'T_LB[combined:scenario=serial_stage_sum]']
- R_k: needed by 103 metric(s), e.g. ['R_k', 'F_Qtilde[useful]', 'F_QC[useful]']
- R_q: needed by 26 metric(s), e.g. ['R_q', 'F_Q[useful]', 'F_total[useful]']
- S_k: needed by 6 metric(s), e.g. ['S_k_active', 'F_A_minus_F_E', 'F_A_minus_F_E[spec_closed_form]']
- S_q: needed by 38 metric(s), e.g. ['S_q_active', 'F_Q[useful]', 'F_Qtilde[useful]']
- W_hbm: needed by 3 metric(s), e.g. ['T_LB[path:hbm_to_vmem]', 'T_LB[combined:scenario=full_overlap_max]', 'T_LB[combined:scenario=serial_stage_sum]']
- W_hbm_w: needed by 3 metric(s), e.g. ['T_LB[path:vmem_to_hbm]', 'T_LB[combined:scenario=full_overlap_max]', 'T_LB[combined:scenario=serial_stage_sum]']
- W_vmem: needed by 3 metric(s), e.g. ['T_LB[path:vmem_to_vreg]', 'T_LB[combined:scenario=full_overlap_max]', 'T_LB[combined:scenario=serial_stage_sum]']
- acceptance: needed by 1 metric(s), e.g. ['acceptance[declared]']
- accumulator_dtype, output_dtype: needed by 2 metric(s), e.g. ['W_vec[cast]', 'W_resource[vpu]']
- allowed_transformations: needed by 3 metric(s), e.g. ['B_extra_opt', 'T_opt', 'B_extra_opt_under_time_budget']
- b_k: needed by 44 metric(s), e.g. ['B[vmem_to_vreg]', 'local[kv_latent_tile]', 'local[score_X]']
- b_q: needed by 96 metric(s), e.g. ['B[hbm_to_vmem]', 'B[vmem_to_vreg]', 'local[q_latent_window]']
- b_rq: needed by 11 metric(s), e.g. ['local[q_latent_window]', 'local[wq_rank_tile]', 'M_live[all:q_proj]']
- both_qk_branches_live and D_r > 0: needed by 10 metric(s), e.g. ['local[score_rope_branch]', 'M_live[all:score]', 'M_peak[all]']
- broadcast_materialized (undeclared backend behaviour): needed by 2 metric(s), e.g. ['W_vec[broadcast]', 'W_resource[layout]']
- calibration: needed by 1 metric(s), e.g. ['T_pred[node_interval]']
- cast_points: needed by 2 metric(s), e.g. ['W_vec[cast]', 'W_resource[vpu]']
- compile_evidence: needed by 2 metric(s), e.g. ['F_compiled', 'N_spill_instructions']
- compile_or_measurement_evidence: needed by 3 metric(s), e.g. ['M_spill_peak', 'B_spill_fill', 'dT_spill']
- declared comparison graph: needed by 1 metric(s), e.g. ['dT_spill']
- executed_extent_policy: needed by 33 metric(s), e.g. ['F_total[executed_graph]', 'F_projection[executed_graph]', 'F_attention[executed_graph]']
- exp_alias_p_operand: needed by 37 metric(s), e.g. ['local[p_operand]', 'M_live[all:q_proj]', 'M_live[all:q_absorb]']
- exp_base: needed by 21 metric(s), e.g. ['W_vec[mul]', 'W_vec[mask]', 'W_vec[cmp]']
- exp_dtype, p_operand_dtype: needed by 2 metric(s), e.g. ['W_vec[cast]', 'W_resource[vpu]']
- final_normalize: needed by 2 metric(s), e.g. ['W_vec[div]', 'W_resource[vpu]']
- graph: needed by 3 metric(s), e.g. ['B_extra_opt', 'T_opt', 'B_extra_opt_under_time_budget']
- h_pp: needed by 3 metric(s), e.g. ['T_LB[path:hbm_to_vmem]', 'T_LB[combined:scenario=full_overlap_max]', 'T_LB[combined:scenario=serial_stage_sum]']
- heads_per_program: needed by 9 metric(s), e.g. ['B[hbm_to_vmem]', 'B[vmem_to_hbm]', 'B[vmem_to_vreg]']
- kv_buffers: needed by 24 metric(s), e.g. ['local[kv_latent_tile]', 'local[k_pe_tile]', 'layout[kv_latent_tile]']
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
- layout.score_rope_branch: needed by 1 metric(s), e.g. ['layout[score_rope_branch]']
- layout.wk_full: needed by 1 metric(s), e.g. ['layout[wk_full]']
- layout.wq_rank_tile: needed by 1 metric(s), e.g. ['layout[wq_rank_tile]']
- layout.wv_full: needed by 1 metric(s), e.g. ['layout[wv_full]']
- layout.z_tile: needed by 1 metric(s), e.g. ['layout[z_tile]']
- lse_convention: needed by 1 metric(s), e.g. ['lse_convention']
- lse_dtype: needed by 1 metric(s), e.g. ['M_LSE']
- machine_model: needed by 3 metric(s), e.g. ['B_extra_opt', 'T_opt', 'B_extra_opt_under_time_budget']
- mask: needed by 37 metric(s), e.g. ['mask', 'rows_without_visible_keys', 'C_valid']
- materialize.q_tilde: needed by 9 metric(s), e.g. ['HBM_mat[Q~]', 'M_mat[Q~]', 'B[hbm_to_vmem]']
- materialize.z: needed by 9 metric(s), e.g. ['HBM_mat[Z]', 'M_mat[Z]', 'B[hbm_to_vmem]']
- matmul_input_dtype: needed by 3 metric(s), e.g. ['T_LB[mxu]', 'T_LB[combined:scenario=full_overlap_max]', 'T_LB[combined:scenario=serial_stage_sum]']
- objective: needed by 1 metric(s), e.g. ['B_extra_opt']
- operand_transpose_materialized (undeclared backend behaviour): needed by 2 metric(s), e.g. ['W_vec[transpose]', 'W_resource[layout]']
- output_dtype: needed by 1 metric(s), e.g. ['M_O']
- outputs: needed by 38 metric(s), e.g. ['outputs', 'W_vec[mul]', 'W_vec[mask]']
- overlap_model: needed by 2 metric(s), e.g. ['T_LB[combined:scenario=full_overlap_max]', 'T_LB[combined:scenario=serial_stage_sum]']
- position_offset: needed by 1 metric(s), e.g. ['Delta']
- projection_scope: needed by 40 metric(s), e.g. ['projection_scope', 'F_Q[useful]', 'F_Qtilde[useful]']
- q_resident: needed by 27 metric(s), e.g. ['B[vmem_to_vreg]', 'local[q_tilde]', 'local[q_pe_tile]']
- rect_policy: needed by 34 metric(s), e.g. ['F_total[executed_graph]', 'F_projection[executed_graph]', 'F_attention[executed_graph]']
- recurrence: needed by 21 metric(s), e.g. ['W_vec[mul]', 'W_vec[mask]', 'W_vec[cmp]']
- register_schedule_evidence: needed by 5 metric(s), e.g. ['B[vmem_to_vreg]', 'B[vreg_to_vmem]', 'T_LB[path:vmem_to_vreg]']
- s_A: needed by 28 metric(s), e.g. ['local[accumulator_A]', 'layout[accumulator_A]', 'M_live[all:score]']
- s_Ck: needed by 30 metric(s), e.g. ['bytes[kv_latent]', 'M_in', 'B[hbm_to_vmem]']
- s_Cq: needed by 15 metric(s), e.g. ['bytes[q_latent]', 'M_in', 'B[hbm_to_vmem]']
- s_E: needed by 16 metric(s), e.g. ['local[exp_E]', 'layout[exp_E]', 'M_live[all:softmax]']
- s_Kr: needed by 18 metric(s), e.g. ['bytes[k_pe]', 'M_in', 'B[hbm_to_vmem]']
- s_LSE: needed by 14 metric(s), e.g. ['M_LSE', 'B[vmem_to_hbm]', 'local[lse_tile]']
- s_O: needed by 14 metric(s), e.g. ['M_O', 'B[vmem_to_hbm]', 'local[out_tile]']
- s_P: needed by 10 metric(s), e.g. ['local[p_operand]', 'layout[p_operand]', 'M_live[all:pv]']
- s_Qacc: needed by 16 metric(s), e.g. ['local[q_proj_acc]', 'local[q_tilde_acc]', 'layout[q_proj_acc]']
- s_Qr: needed by 35 metric(s), e.g. ['bytes[q_pe]', 'M_in', 'B[hbm_to_vmem]']
- s_Qt: needed by 37 metric(s), e.g. ['HBM_mat[Q~]', 'M_mat[Q~]', 'B[hbm_to_vmem]']
- s_Wk: needed by 15 metric(s), e.g. ['bytes[w_k_nope]', 'M_in', 'B[hbm_to_vmem]']
- s_Wq: needed by 15 metric(s), e.g. ['bytes[w_q_nope]', 'M_in', 'B[hbm_to_vmem]']
- s_Wv: needed by 15 metric(s), e.g. ['bytes[w_v]', 'M_in', 'B[hbm_to_vmem]']
- s_X: needed by 20 metric(s), e.g. ['local[score_X]', 'local[score_rope_branch]', 'layout[score_X]']
- s_Z: needed by 35 metric(s), e.g. ['HBM_mat[Z]', 'M_mat[Z]', 'B[hbm_to_vmem]']
- s_state: needed by 31 metric(s), e.g. ['local[row_state_m]', 'local[row_state_l]', 'local[row_alpha]']
- scale_policy/scale_value: needed by 1 metric(s), e.g. ['gamma']
- schedule: needed by 34 metric(s), e.g. ['F_total[executed_graph]', 'F_projection[executed_graph]', 'F_attention[executed_graph]']
- schedule_model: needed by 2 metric(s), e.g. ['T_opt', 'B_extra_opt_under_time_budget']
- scheduled_lengths.Sk: needed by 1 metric(s), e.g. ['S_k_scheduled']
- scheduled_lengths.Sq: needed by 1 metric(s), e.g. ['S_q_scheduled']
- score_alias_exp: needed by 37 metric(s), e.g. ['local[exp_E]', 'M_live[all:q_proj]', 'M_live[all:q_absorb]']
- scratch_bytes: needed by 41 metric(s), e.g. ['local[fixed_scratch]', 'layout[fixed_scratch]', 'M_live[all:q_proj]']
- time_budget_tau: needed by 1 metric(s), e.g. ['B_extra_opt_under_time_budget']
- vmem_budget_bytes: needed by 10 metric(s), e.g. ['excess_over_budget[vmem:q_proj]', 'excess_over_budget[vmem:q_absorb]', 'b_k_max[vmem:score]']
- vreg_budget_bytes: needed by 10 metric(s), e.g. ['excess_over_budget[vreg:q_proj]', 'excess_over_budget[vreg:q_absorb]', 'b_k_max[vreg:score]']

## Warnings

- schedule not declared: the q_outer_kv_inner loop nest is shown as the scenario; transfer counts and the program count follow from it
- rank sub-tile undeclared: the full R_q window is shown as the scenario
- q_resident not declared: Q operands held across the KV loop shown as the scenario
- final_normalize not declared: the n*D_A division form is the scenario; the reciprocal-plus-multiply form of spec 5.5 is the alternative, not an addition
- recurrence not declared: unnormalized recurrence counts shown as the scenario; normalized is a separate scenario
- exp_base not declared: base-e op graph shown; exp2 adds a logit scale and an LSE restore
- cast_points not declared: the cast graph is inferred from dtype differences at the two modelled sites (exp_to_p, accumulator_to_output); a declared site list gives a different graph
- projection_scope not declared: full Q projection shown
- heads_per_program (h_pp) undeclared: KV-latent sharing across heads stays symbolic

## 1. task and scope

| metric | formula id | expression | value | unit | status | missing | scope | not equivalent to |
|---|---|---|---|---|---|---|---|---|
| B | `task.dim.B` | `B → B` |  | count | symbolic | B | current invocation |  |
| H | `task.dim.H` | `H → H` |  | count | symbolic | H | current invocation |  |
| R_q | `task.dim.R_q` | `R_q → R_q` |  | count | symbolic | R_q | current invocation |  |
| R_k | `task.dim.R_k` | `R_k → R_k` |  | count | symbolic | R_k | current invocation |  |
| D_n | `task.dim.D_n` | `D_n → D_n` |  | count | symbolic | D_n | current invocation |  |
| D_r | `task.dim.D_r` | `D_r → D_r` |  | count | symbolic | D_r | current invocation |  |
| D_v | `task.dim.D_v` | `D_v → D_v` |  | count | symbolic | D_v | current invocation |  |
| S_q_active | `task.active.S_q` | `S_q → S_q` |  | count | symbolic | S_q | active query rows (mathematical work) |  |
| S_k_active | `task.active.S_k` | `S_k → S_k` |  | count | symbolic | S_k | active keys consumed in this call |  |
| S_q_scheduled | `task.scheduled.S_q` | `declared scheduled extent (else = active)` |  | count | partial | scheduled_lengths.Sq | extent the kernel iterates over (execution rectangles, padding counts) | active extent; allocated capacity |
| S_k_scheduled | `task.scheduled.S_k` | `declared scheduled extent (else = active)` |  | count | partial | scheduled_lengths.Sk | extent the kernel iterates over (execution rectangles, padding counts) | active extent; allocated capacity |
| Delta | `task.offset` | `Delta → Delta` |  | positions | symbolic | Delta, position_offset | key j visible to query i iff j <= i + Delta |  |
| mask | `task.mask` | `declared mask kind` |  |  | unknown | mask | visibility semantics |  |
| gamma | `task.scale` | `(undefined: missing declaration)` |  | 1 | unknown | scale_policy/scale_value | score scale |  |
| outputs | `task.outputs` | `declared output scope` |  |  | unknown | outputs | output scope |  |
| lse_convention | `task.lse_convention` | `declared` |  |  | unknown | lse_convention | LSE log base |  |
| projection_scope | `task.projection_scope` | `declared` | full (scenario: projection_scope undeclared) |  | partial | projection_scope | which rows are projected in this call |  |
| rows_without_visible_keys | `task.empty_rows` | `(undefined: missing declaration)` |  | rows | unknown | mask | (b,h,i) rows whose softmax is undefined |  |

Assumptions:
- `S_q_scheduled`: scheduled extent undeclared: assumed equal to the active extent (scenario)
- `S_k_scheduled`: scheduled extent undeclared: assumed equal to the active extent (scenario)
- `mask`: mask is declared, never inferred from shapes
- `lse_convention`: LSE is requested but its log base is undeclared
- `rows_without_visible_keys`: uniform lengths: B*H*max(0, min(S_q, -Delta))

<details><summary>Provenance (bindings, coverage, source, evidence)</summary>

| metric | source | coverage | bindings | evidence |
|---|---|---|---|---|
| B | input | incomplete: depends on undeclared/unbound B | `{}` |  |
| H | input | incomplete: depends on undeclared/unbound H | `{}` |  |
| R_q | input | incomplete: depends on undeclared/unbound R_q | `{}` |  |
| R_k | input | incomplete: depends on undeclared/unbound R_k | `{}` |  |
| D_n | input | incomplete: depends on undeclared/unbound D_n | `{}` |  |
| D_r | input | incomplete: depends on undeclared/unbound D_r | `{}` |  |
| D_v | input | incomplete: depends on undeclared/unbound D_v | `{}` |  |
| S_q_active | input | incomplete: depends on undeclared/unbound S_q | `{}` |  |
| S_k_active | input | incomplete: depends on undeclared/unbound S_k | `{}` |  |
| S_q_scheduled | input | incomplete: depends on undeclared/unbound scheduled_lengths.Sq | `{}` |  |
| S_k_scheduled | input | incomplete: depends on undeclared/unbound scheduled_lengths.Sk | `{}` |  |
| Delta | input | incomplete: depends on undeclared/unbound Delta, position_offset | `{}` |  |
| mask | input | incomplete: depends on undeclared/unbound mask | `{}` |  |
| gamma | input | incomplete: depends on undeclared/unbound scale_policy/scale_value | `{}` |  |
| outputs | input | incomplete: depends on undeclared/unbound outputs | `{}` |  |
| lse_convention | input | incomplete: depends on undeclared/unbound lse_convention | `{}` |  |
| projection_scope | input | incomplete: depends on undeclared/unbound projection_scope | `{}` |  |
| rows_without_visible_keys | derived | incomplete: depends on undeclared/unbound mask | `{}` |  |

</details>

## 2. visibility and rectangles

| metric | formula id | expression | value | unit | status | missing | scope | not equivalent to |
|---|---|---|---|---|---|---|---|---|
| C_valid | `vis.C_valid` | `(undefined: missing declaration)` |  | cells | unknown | mask | all (b,h,i,j) with mu=1; includes B and H | exp instruction count; executed cells |

Assumptions:
- `C_valid`: uniform unit-step positions; causal: j <= i + Delta

<details><summary>Provenance (bindings, coverage, source, evidence)</summary>

| metric | source | coverage | bindings | evidence |
|---|---|---|---|---|
| C_valid | derived | incomplete: depends on undeclared/unbound mask | `{}` |  |

</details>

## 3. work (FLOPs and vector operations)

| metric | formula id | expression | value | unit | status | missing | scope | not equivalent to |
|---|---|---|---|---|---|---|---|---|
| F_Q[useful] | `work.absorbed_two_step.F_Q` | `2*B*D_n*H*R_q*S_q → 2*B*D_n*H*R_q*S_q` |  | FLOP | symbolic | B, C_valid, D_n, H, R_q, S_q, projection_scope | useful (C = C_valid), FMA=2, projection rows from scope 'full (scenario: projection_scope undeclared)' |  |
| F_Qtilde[useful] | `work.absorbed_two_step.F_Qtilde` | `2*B*D_n*H*R_k*S_q → 2*B*D_n*H*R_k*S_q` |  | FLOP | symbolic | B, C_valid, D_n, H, R_k, S_q, projection_scope | useful (C = C_valid), FMA=2, projection rows from scope 'full (scenario: projection_scope undeclared)' |  |
| F_QC[useful] | `work.absorbed_two_step.F_QC` | `2*C_valid*R_k → 2*C_valid*R_k` |  | FLOP | symbolic | C_valid, R_k, projection_scope | useful (C = C_valid), FMA=2, projection rows from scope 'full (scenario: projection_scope undeclared)' |  |
| F_QK_r[useful] | `work.absorbed_two_step.F_QK_r` | `2*C_valid*D_r → 2*C_valid*D_r` |  | FLOP | symbolic | C_valid, D_r, projection_scope | useful (C = C_valid), FMA=2, projection rows from scope 'full (scenario: projection_scope undeclared)' |  |
| F_PC[useful] | `work.absorbed_two_step.F_PC` | `2*C_valid*R_k → 2*C_valid*R_k` |  | FLOP | symbolic | C_valid, R_k, projection_scope | useful (C = C_valid), FMA=2, projection rows from scope 'full (scenario: projection_scope undeclared)' |  |
| F_ZWv[useful] | `work.absorbed_two_step.F_ZWv` | `2*B*D_v*H*R_k*S_q → 2*B*D_v*H*R_k*S_q` |  | FLOP | symbolic | B, C_valid, D_v, H, R_k, S_q, projection_scope | useful (C = C_valid), FMA=2, projection rows from scope 'full (scenario: projection_scope undeclared)' |  |
| F_total[useful] | `work.absorbed_two_step.total` | `2*B*D_n*H*R_k*S_q + 2*B*D_n*H*R_q*S_q + 2*B*D_v*H*R_k*S_q + 2*C_valid*(D_r + 2*R_k) → 2*B*D_n*H*R_k*S_q + 2*B*D_n*H*R_q*S_q + 2*B*D_v*H*R_k*S_q + 2*C_valid*(D_r + 2*R_k)` |  | FLOP | symbolic | B, C_valid, D_n, D_r, D_v, H, R_k, R_q, S_q, projection_scope | useful mathematical work under projection scope 'full (scenario: projection_scope undeclared)' | scheduled work; compiled instruction count |
| F_A[spec_closed_form] | `work.absorbed_two_step.closed_form` | `2*B*D_n*H*R_q*S_q + 2*B*H*R_k*S_q*(D_n + D_v) + 2*C_valid*(D_r + 2*R_k) → 2*B*D_n*H*R_q*S_q + 2*B*H*R_k*S_q*(D_n + D_v) + 2*C_valid*(D_r + 2*R_k)` |  | FLOP | symbolic | B, C_valid, D_n, D_r, D_v, H, R_k, R_q, S_q | specification boxed formula: every Q row and every K/V row projected once (full projection, no cache) |  |
| N_Qproj | `work.projection.rows.N_Qproj` | `B*H*S_q → B*H*S_q` |  | rows | symbolic | B, H, S_q, projection_scope | Q rows projected in this invocation counted once per row, from the declared projection_scope and the variant's dataflow | the tensor's row count; rows read from HBM; the executed-graph projected-row count under in-loop projection |
| N_Kproj | `work.projection.rows.N_Kproj` | `0` | 0 | rows | partial | projection_scope | K rows projected in this invocation counted once per row, from the declared projection_scope and the variant's dataflow | the tensor's row count; rows read from HBM; the executed-graph projected-row count under in-loop projection |
| N_Vproj | `work.projection.rows.N_Vproj` | `0` | 0 | rows | partial | projection_scope | V rows projected in this invocation counted once per row, from the declared projection_scope and the variant's dataflow | the tensor's row count; rows read from HBM; the executed-graph projected-row count under in-loop projection |
| F_proj[general] | `work.projection.general` | `2*B*D_n*H*R_k*S_q + 2*B*D_n*H*R_q*S_q → 2*B*D_n*H*R_k*S_q + 2*B*D_n*H*R_q*S_q` |  | FLOP | symbolic | B, D_n, H, R_k, R_q, S_q, projection_scope | 2 N_Qproj R_q D_n + 2 N_Qproj D_n R_k: the Q projection and the absorb step; N_Kproj = N_Vproj = 0 in this variant | F_projection[executed_graph], which carries the loop multiplicity |
| F_total[executed_graph] | `work.absorbed_two_step.graph.total` | `2*B*D_n*H*R_k*S_q + 2*B*D_n*H*R_q*S_q + 2*B*D_v*H*R_k*S_q + 2*C_pad*D_r + 4*C_pad*R_k → 2*B*D_n*H*R_k*S_q + 2*B*D_n*H*R_q*S_q + 2*B*D_v*H*R_k*S_q + 2*C_pad*D_r + 4*C_pad*R_k` |  | FLOP | symbolic | B, C_pad, D_n, D_r, D_v, H, R_k, R_q, S_q, executed_extent_policy, mask, projection_scope, rect_policy, schedule | graph sum of 2MNK*multiplicity over executed cells C_exec (declared extent policy) |  |
| F_projection[executed_graph] | `work.absorbed_two_step.graph.projection` | `2*B*D_n*H*R_k*S_q + 2*B*D_n*H*R_q*S_q → 2*B*D_n*H*R_k*S_q + 2*B*D_n*H*R_q*S_q` |  | FLOP | symbolic | B, D_n, H, R_k, R_q, S_q, executed_extent_policy, mask, projection_scope, rect_policy, schedule | graph category |  |
| F_attention[executed_graph] | `work.absorbed_two_step.graph.attention` | `2*C_pad*D_r + 4*C_pad*R_k → 2*C_pad*D_r + 4*C_pad*R_k` |  | FLOP | symbolic | C_pad, D_r, R_k, executed_extent_policy, mask, projection_scope, rect_policy, schedule | graph category |  |
| F_output_projection[executed_graph] | `work.absorbed_two_step.graph.output_projection` | `2*B*D_v*H*R_k*S_q → 2*B*D_v*H*R_k*S_q` |  | FLOP | symbolic | B, D_v, H, R_k, S_q, executed_extent_policy, mask, projection_scope, rect_policy, schedule | graph category |  |
| F_compiled | `work.compiled` | `from compilation evidence only` |  | FLOP | unknown | compile_evidence | compiled instruction-level work | useful; rect; executed_graph |
| F_A_minus_F_E | `work.path_difference` | `-2*B*D_n*H*R_k*S_k + 2*B*D_n*H*R_k*S_q - 2*B*D_v*H*R_k*S_k + 2*B*D_v*H*R_k*S_q + 2*C_valid*(D_r + 2*R_k) - 2*C_valid*(D_n + D_r + D_v) → -2*B*D_n*H*R_k*S_k + 2*B*D_n*H*R_k*S_q - 2*B*D_v*H*R_k*S_k + 2*B*D_v*H*R_k*S_q + 2*C_valid*(D_r + 2*R_k) - 2*C_valid*(D_n + D_r + D_v)` |  | FLOP | symbolic | B, C_valid, D_n, D_r, D_v, H, R_k, S_k, S_q, projection_scope | absorbed_two_step minus expanded, same projection rows and same C (useful) |  |
| F_A_minus_F_E[spec_closed_form] | `work.path_difference.closed_form` | `2*C_valid*(-D_n - D_v + 2*R_k) + 2*R_k*(D_n + D_v)*(-B*H*S_k + B*H*S_q) → 2*C_valid*(-D_n - D_v + 2*R_k) + 2*R_k*(D_n + D_v)*(-B*H*S_k + B*H*S_q)` |  | FLOP | symbolic | B, C_valid, D_n, D_v, H, R_k, S_k, S_q | 2BH Rk (Dn+Dv)(Sq-Sk) + 2C(2Rk-Dn-Dv): full projection, no cache, two-step variant |  |
| equivalence[expanded vs absorbed] | `numerics.equivalence` | `matrix associativity: Q^n (W^k)^T C_k^T = Q^n (C_k W^k)^T` | equal over the real numbers |  | bound |  | the two paths express the same result in exact real arithmetic | bitwise equality; an error bound; a measured deviation |
| acceptance[declared] | `numerics.acceptance` | `caller-declared tolerances per test level` |  |  | unknown | acceptance | numeric acceptance declared with the task | a measured error; a proof that the variant meets it |
| W_vec[mul] | `work.vector.mul` | `C_pad + N_rowvisits*R_k + N_rowvisits → C_pad + N_rowvisits*R_k + N_rowvisits` |  | element-ops | symbolic | C_pad, N_rowvisits, R_k, executed_extent_policy, exp_base, mask, outputs, projection_scope, rect_policy, recurrence, schedule | logical count before redundancy elimination; not instructions | vector instruction count; MXU FLOPs |
| W_vec[mask] | `work.vector.mask` | `C_masked → C_masked` |  | element-ops | symbolic | C_masked, executed_extent_policy, exp_base, mask, outputs, projection_scope, rect_policy, recurrence, schedule | logical count before redundancy elimination; not instructions | vector instruction count; MXU FLOPs |
| W_vec[cmp] | `work.vector.cmp` | `C_pad → C_pad` |  | element-ops | symbolic | C_pad, executed_extent_policy, exp_base, mask, outputs, projection_scope, rect_policy, recurrence, schedule | logical count before redundancy elimination; not instructions | vector instruction count; MXU FLOPs |
| W_vec[broadcast] | `work.vector.broadcast` | `C_pad → C_pad` |  | element-ops | symbolic | C_pad, broadcast_materialized (undeclared backend behaviour), executed_extent_policy, exp_base, mask, outputs, projection_scope, rect_policy, recurrence, schedule | logical count before redundancy elimination; not instructions | vector instruction count; MXU FLOPs |
| W_vec[transpose] | `work.vector.transpose` | `H*N_keyvisits*R_k → H*N_keyvisits*R_k` |  | element-ops | symbolic | H, N_keyvisits, R_k, executed_extent_policy, exp_base, mask, operand_transpose_materialized (undeclared backend behaviour), outputs, projection_scope, rect_policy, recurrence, schedule | logical count before redundancy elimination; not instructions | vector instruction count; MXU FLOPs |
| W_vec[exp] | `work.vector.exp` | `C_pad + N_rowvisits → C_pad + N_rowvisits` |  | element-ops | symbolic | C_pad, N_rowvisits, executed_extent_policy, exp_base, mask, outputs, projection_scope, rect_policy, recurrence, schedule | logical count before redundancy elimination; not instructions | vector instruction count; MXU FLOPs |
| W_vec[add] | `work.vector.add` | `B*H*S_q + C_pad + N_rowvisits*R_k + Piecewise((C_pad, D_r > 0), (0, True)) → B*H*S_q + C_pad + N_rowvisits*R_k + Piecewise((C_pad, D_r > 0), (0, True))` |  | element-ops | symbolic | B, C_pad, D_r, H, N_rowvisits, R_k, S_q, executed_extent_policy, exp_base, mask, outputs, projection_scope, rect_policy, recurrence, schedule | logical count before redundancy elimination; not instructions | vector instruction count; MXU FLOPs |
| W_vec[div] | `work.vector.div` | `B*H*R_k*S_q → B*H*R_k*S_q` |  | element-ops | symbolic | B, H, R_k, S_q, executed_extent_policy, exp_base, final_normalize, mask, outputs, projection_scope, rect_policy, recurrence, schedule | logical count before redundancy elimination; not instructions | vector instruction count; MXU FLOPs |
| W_vec[recip] | `work.vector.recip` | `0` | 0 | element-ops | partial | executed_extent_policy, exp_base, mask, outputs, projection_scope, rect_policy, recurrence, schedule | logical count before redundancy elimination; not instructions | vector instruction count; MXU FLOPs |
| W_vec[log] | `work.vector.log` | `B*H*S_q → B*H*S_q` |  | element-ops | symbolic | B, H, S_q, executed_extent_policy, exp_base, mask, outputs, projection_scope, rect_policy, recurrence, schedule | logical count before redundancy elimination; not instructions | vector instruction count; MXU FLOPs |
| W_vec[cast] | `work.vector.cast` | `B*D_v*H*S_q + C_pad → B*D_v*H*S_q + C_pad` |  | element-ops | symbolic | B, C_pad, D_v, H, S_q, accumulator_dtype, output_dtype, cast_points, executed_extent_policy, exp_base, exp_dtype, p_operand_dtype, mask, outputs, projection_scope, rect_policy, recurrence, schedule | logical count before redundancy elimination; not instructions | vector instruction count; MXU FLOPs |
| W_resource[vpu] | `work.resource.vpu` | `B*D_v*H*S_q + B*H*R_k*S_q + 2*B*H*S_q + C_masked + 2*C_pad + 2*N_rowvisits*R_k + 2*N_rowvisits + Piecewise((C_pad, D_r > 0), (0, True)) → B*D_v*H*S_q + B*H*R_k*S_q + 2*B*H*S_q + C_masked + 2*C_pad + 2*N_rowvisits*R_k + 2*N_rowvisits + Piecewise((C_pad, D_r > 0), (0, True))` |  | element-ops | symbolic | B, C_masked, C_pad, D_r, D_v, H, N_rowvisits, R_k, S_q, accumulator_dtype, output_dtype, cast_points, executed_extent_policy, exp_base, exp_dtype, p_operand_dtype, final_normalize, mask, outputs, projection_scope, rect_policy, recurrence, schedule | per resource class u (for W_u/P_u) |  |
| W_resource[reduce] | `work.resource.reduce` | `2*C_pad - N_rowvisits → 2*C_pad - N_rowvisits` |  | element-ops | symbolic | C_pad, N_rowvisits, executed_extent_policy, exp_base, mask, outputs, projection_scope, rect_policy, recurrence, schedule | per resource class u (for W_u/P_u) |  |
| W_resource[layout] | `work.resource.layout` | `C_pad + H*N_keyvisits*R_k → C_pad + H*N_keyvisits*R_k` |  | element-ops | symbolic | C_pad, H, N_keyvisits, R_k, broadcast_materialized (undeclared backend behaviour), executed_extent_policy, exp_base, mask, operand_transpose_materialized (undeclared backend behaviour), outputs, projection_scope, rect_policy, recurrence, schedule | per resource class u (for W_u/P_u) |  |
| W_resource[exp] | `work.resource.exp` | `C_pad + N_rowvisits → C_pad + N_rowvisits` |  | element-ops | symbolic | C_pad, N_rowvisits, executed_extent_policy, exp_base, mask, outputs, projection_scope, rect_policy, recurrence, schedule | per resource class u (for W_u/P_u) |  |
| exp_count_executed | `work.exp.executed` | `C_pad → C_pad` |  | exp | symbolic | C_pad, executed_extent_policy, mask, projection_scope, rect_policy, schedule | exp over executed cells (masked included) | C_valid |

Assumptions:
- `F_total[useful]`: N_Qproj=B*H*S_q, N_Kproj=B*H*S_k, N_Vproj=B*H*S_k
- `F_total[useful]`: C already includes B and H
- `F_total[useful]`: FMA = 2; not an instruction count
- `F_A[spec_closed_form]`: equals F_total[useful] only when projection_scope = full
- `N_Qproj`: absorbed path: C_k is consumed directly as both the K and the V operand, so no K/V row is projected in this scope; the Z W^v output projection is reported separately
- `N_Qproj`: distinct rows, not row visits: a strategy that re-projects inside the KV loop raises the executed-graph work, not this count (spec 5.2)
- `N_Kproj`: absorbed path: C_k is consumed directly as both the K and the V operand, so no K/V row is projected in this scope; the Z W^v output projection is reported separately
- `N_Kproj`: distinct rows, not row visits: a strategy that re-projects inside the KV loop raises the executed-graph work, not this count (spec 5.2)
- `N_Vproj`: absorbed path: C_k is consumed directly as both the K and the V operand, so no K/V row is projected in this scope; the Z W^v output projection is reported separately
- `N_Vproj`: distinct rows, not row visits: a strategy that re-projects inside the KV loop raises the executed-graph work, not this count (spec 5.2)
- `F_proj[general]`: N_Qproj=B*H*S_q, N_Kproj=0, N_Vproj=0
- `F_proj[general]`: absorbed path: C_k is consumed directly as both the K and the V operand, so no K/V row is projected in this scope; the Z W^v output projection is reported separately
- `F_proj[general]`: useful-work counting: each row projected once. When the strategy projects inside the KV loop, the repeated work is in F_projection[executed_graph] (spec 5.2), not here.
- `F_total[executed_graph]`: Q~ materialization counted additively: one whole-tensor write and one whole-tensor read; the in-loop producer and its operand reads are unchanged
- `F_total[executed_graph]`: Z materialization counted additively: one whole-tensor write and one whole-tensor read; the in-loop producer and its operand reads are unchanged
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
| F_Q[useful] | derived | incomplete: depends on undeclared/unbound B, C_valid, D_n, H, R_q, S_q, projection_scope | `{}` |  |
| F_Qtilde[useful] | derived | incomplete: depends on undeclared/unbound B, C_valid, D_n, H, R_k, S_q, projection_scope | `{}` |  |
| F_QC[useful] | derived | incomplete: depends on undeclared/unbound C_valid, R_k, projection_scope | `{}` |  |
| F_QK_r[useful] | derived | incomplete: depends on undeclared/unbound C_valid, D_r, projection_scope | `{}` |  |
| F_PC[useful] | derived | incomplete: depends on undeclared/unbound C_valid, R_k, projection_scope | `{}` |  |
| F_ZWv[useful] | derived | incomplete: depends on undeclared/unbound B, C_valid, D_v, H, R_k, S_q, projection_scope | `{}` |  |
| F_total[useful] | derived | incomplete: depends on undeclared/unbound B, C_valid, D_n, D_r, D_v, H, R_k, R_q, S_q, projection_scope | `{}` |  |
| F_A[spec_closed_form] | derived | incomplete: depends on undeclared/unbound B, C_valid, D_n, D_r, D_v, H, R_k, R_q, S_q | `{}` |  |
| N_Qproj | derived | incomplete: depends on undeclared/unbound B, H, S_q, projection_scope | `{}` |  |
| N_Kproj | derived | incomplete: depends on undeclared/unbound projection_scope | `{}` |  |
| N_Vproj | derived | incomplete: depends on undeclared/unbound projection_scope | `{}` |  |
| F_proj[general] | derived | incomplete: depends on undeclared/unbound B, D_n, H, R_k, R_q, S_q, projection_scope | `{}` |  |
| F_total[executed_graph] | derived | incomplete: depends on undeclared/unbound B, C_pad, D_n, D_r, D_v, H, R_k, R_q, S_q, executed_extent_policy, mask, projection_scope, rect_policy, schedule | `{}` |  |
| F_projection[executed_graph] | derived | incomplete: depends on undeclared/unbound B, D_n, H, R_k, R_q, S_q, executed_extent_policy, mask, projection_scope, rect_policy, schedule | `{}` |  |
| F_attention[executed_graph] | derived | incomplete: depends on undeclared/unbound C_pad, D_r, R_k, executed_extent_policy, mask, projection_scope, rect_policy, schedule | `{}` |  |
| F_output_projection[executed_graph] | derived | incomplete: depends on undeclared/unbound B, D_v, H, R_k, S_q, executed_extent_policy, mask, projection_scope, rect_policy, schedule | `{}` |  |
| F_compiled | compiled | incomplete: depends on undeclared/unbound compile_evidence | `{}` |  |
| F_A_minus_F_E | derived | incomplete: depends on undeclared/unbound B, C_valid, D_n, D_r, D_v, H, R_k, S_k, S_q, projection_scope | `{}` |  |
| F_A_minus_F_E[spec_closed_form] | derived | incomplete: depends on undeclared/unbound B, C_valid, D_n, D_v, H, R_k, S_k, S_q | `{}` |  |
| equivalence[expanded vs absorbed] | definition | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{}` |  |
| acceptance[declared] | input | incomplete: depends on undeclared/unbound acceptance | `{}` |  |
| W_vec[mul] | derived | incomplete: depends on undeclared/unbound C_pad, N_rowvisits, R_k, executed_extent_policy, exp_base, mask, outputs, projection_scope, rect_policy, recurrence, schedule | `{}` |  |
| W_vec[mask] | derived | incomplete: depends on undeclared/unbound C_masked, executed_extent_policy, exp_base, mask, outputs, projection_scope, rect_policy, recurrence, schedule | `{}` |  |
| W_vec[cmp] | derived | incomplete: depends on undeclared/unbound C_pad, executed_extent_policy, exp_base, mask, outputs, projection_scope, rect_policy, recurrence, schedule | `{}` |  |
| W_vec[broadcast] | derived | incomplete: depends on undeclared/unbound C_pad, broadcast_materialized (undeclared backend behaviour), executed_extent_policy, exp_base, mask, outputs, projection_scope, rect_policy, recurrence, schedule | `{}` |  |
| W_vec[transpose] | derived | incomplete: depends on undeclared/unbound H, N_keyvisits, R_k, executed_extent_policy, exp_base, mask, operand_transpose_materialized (undeclared backend behaviour), outputs, projection_scope, rect_policy, recurrence, schedule | `{}` |  |
| W_vec[exp] | derived | incomplete: depends on undeclared/unbound C_pad, N_rowvisits, executed_extent_policy, exp_base, mask, outputs, projection_scope, rect_policy, recurrence, schedule | `{}` |  |
| W_vec[add] | derived | incomplete: depends on undeclared/unbound B, C_pad, D_r, H, N_rowvisits, R_k, S_q, executed_extent_policy, exp_base, mask, outputs, projection_scope, rect_policy, recurrence, schedule | `{}` |  |
| W_vec[div] | derived | incomplete: depends on undeclared/unbound B, H, R_k, S_q, executed_extent_policy, exp_base, final_normalize, mask, outputs, projection_scope, rect_policy, recurrence, schedule | `{}` |  |
| W_vec[recip] | derived | incomplete: depends on undeclared/unbound executed_extent_policy, exp_base, mask, outputs, projection_scope, rect_policy, recurrence, schedule | `{}` |  |
| W_vec[log] | derived | incomplete: depends on undeclared/unbound B, H, S_q, executed_extent_policy, exp_base, mask, outputs, projection_scope, rect_policy, recurrence, schedule | `{}` |  |
| W_vec[cast] | derived | incomplete: depends on undeclared/unbound B, C_pad, D_v, H, S_q, accumulator_dtype, output_dtype, cast_points, executed_extent_policy, exp_base, exp_dtype, p_operand_dtype, mask, outputs, projection_scope, rect_policy, recurrence, schedule | `{}` |  |
| W_resource[vpu] | derived | incomplete: depends on undeclared/unbound B, C_masked, C_pad, D_r, D_v, H, N_rowvisits, R_k, S_q, accumulator_dtype, output_dtype, cast_points, executed_extent_policy, exp_base, exp_dtype, p_operand_dtype, final_normalize, mask, outputs, projection_scope, rect_policy, recurrence, schedule | `{}` |  |
| W_resource[reduce] | derived | incomplete: depends on undeclared/unbound C_pad, N_rowvisits, executed_extent_policy, exp_base, mask, outputs, projection_scope, rect_policy, recurrence, schedule | `{}` |  |
| W_resource[layout] | derived | incomplete: depends on undeclared/unbound C_pad, H, N_keyvisits, R_k, broadcast_materialized (undeclared backend behaviour), executed_extent_policy, exp_base, mask, operand_transpose_materialized (undeclared backend behaviour), outputs, projection_scope, rect_policy, recurrence, schedule | `{}` |  |
| W_resource[exp] | derived | incomplete: depends on undeclared/unbound C_pad, N_rowvisits, executed_extent_policy, exp_base, mask, outputs, projection_scope, rect_policy, recurrence, schedule | `{}` |  |
| exp_count_executed | derived | incomplete: depends on undeclared/unbound C_pad, executed_extent_policy, mask, projection_scope, rect_policy, schedule | `{}` |  |

</details>

## 4. bytes: interface and materialization

| metric | formula id | expression | value | unit | status | missing | scope | not equivalent to |
|---|---|---|---|---|---|---|---|---|
| bytes[q_latent] | `bytes.logical.q_latent` | `B*R_q*S_q*s_Cq → B*R_q*S_q*s_Cq` |  | byte | symbolic | B, R_q, S_q, s_Cq | logical size at this tensor's own allocation extent, per-tensor dtype |  |
| bytes[kv_latent] | `bytes.logical.kv_latent` | `B*R_k*S_k*s_Ck → B*R_k*S_k*s_Ck` |  | byte | symbolic | B, R_k, S_k, s_Ck | logical size at this tensor's own allocation extent, per-tensor dtype |  |
| bytes[q_pe] | `bytes.logical.q_pe` | `B*D_r*H*S_q*s_Qr → B*D_r*H*S_q*s_Qr` |  | byte | symbolic | B, D_r, H, S_q, s_Qr | logical size at this tensor's own allocation extent, per-tensor dtype |  |
| bytes[k_pe] | `bytes.logical.k_pe` | `B*D_r*S_k*s_Kr → B*D_r*S_k*s_Kr` |  | byte | symbolic | B, D_r, S_k, s_Kr | logical size at this tensor's own allocation extent, per-tensor dtype |  |
| bytes[w_q_nope] | `bytes.logical.w_q_nope` | `D_n*H*R_q*s_Wq → D_n*H*R_q*s_Wq` |  | byte | symbolic | D_n, H, R_q, s_Wq | logical size at this tensor's own allocation extent, per-tensor dtype |  |
| bytes[w_k_nope] | `bytes.logical.w_k_nope` | `D_n*H*R_k*s_Wk → D_n*H*R_k*s_Wk` |  | byte | symbolic | D_n, H, R_k, s_Wk | logical size at this tensor's own allocation extent, per-tensor dtype |  |
| bytes[w_v] | `bytes.logical.w_v` | `D_v*H*R_k*s_Wv → D_v*H*R_k*s_Wv` |  | byte | symbolic | D_v, H, R_k, s_Wv | logical size at this tensor's own allocation extent, per-tensor dtype |  |
| M_in | `bytes.M_in` | `B*D_r*H*S_q*s_Qr + B*D_r*S_k*s_Kr + B*R_k*S_k*s_Ck + B*R_q*S_q*s_Cq + D_n*H*R_k*s_Wk + D_n*H*R_q*s_Wq + D_v*H*R_k*s_Wv → B*D_r*H*S_q*s_Qr + B*D_r*S_k*s_Kr + B*R_k*S_k*s_Ck + B*R_q*S_q*s_Cq + D_n*H*R_k*s_Wk + D_n*H*R_q*s_Wq + D_v*H*R_k*s_Wv` |  | byte | symbolic | B, D_n, D_r, D_v, H, R_k, R_q, S_k, S_q, s_Ck, s_Cq, s_Kr, s_Qr, s_Wk, s_Wq, s_Wv | sum of the seven inputs, each at its OWN ALLOCATION extent (capacity_shape where declared) | HBM traffic lower bound under caching/reuse; M_in[metadata_ledger], which sums the LOGICAL shapes rather than the allocations |
| M_O | `bytes.M_O` | `B*D_v*H*S_q*s_O → B*D_v*H*S_q*s_O` |  | byte | symbolic | B, D_v, H, S_q, output_dtype, s_O | output O at active extents (s_O * B H S_q * D_v; ragged: H sum_b Sq_b) |  |
| M_LSE | `bytes.M_LSE` | `B*H*S_q*s_LSE → B*H*S_q*s_LSE` |  | byte | symbolic | B, H, S_q, lse_dtype, outputs, s_LSE | LSE at active extents (conditional: outputs undeclared) |  |
| HBM_mat[Q~] | `bytes.mat.mat_q_tilde` | `2*B*H*R_k*S_q*s_Qt → 2*B*H*R_k*S_q*s_Qt` |  | byte | symbolic | B, H, R_k, S_q, materialize.q_tilde, s_Qt | write + read traffic if materialized (read under the declared scan policy) | interface bytes |
| M_mat[Q~] | `bytes.mat.size.mat_q_tilde` | `B*H*R_k*S_q*s_Qt → B*H*R_k*S_q*s_Qt` |  | byte | symbolic | B, H, R_k, S_q, materialize.q_tilde, s_Qt | logical size of the materialized intermediate itself (spec 6.2 M^mat) | its HBM traffic; interface bytes |
| HBM_mat[Z] | `bytes.mat.mat_z` | `2*B*H*R_k*S_q*s_Z → 2*B*H*R_k*S_q*s_Z` |  | byte | symbolic | B, H, R_k, S_q, materialize.z, s_Z | write + read traffic if materialized (read under the declared scan policy) | interface bytes |
| M_mat[Z] | `bytes.mat.size.mat_z` | `B*H*R_k*S_q*s_Z → B*H*R_k*S_q*s_Z` |  | byte | symbolic | B, H, R_k, S_q, materialize.z, s_Z | logical size of the materialized intermediate itself (spec 6.2 M^mat) | its HBM traffic; interface bytes |

Assumptions:
- `bytes[q_latent]`: no metadata bound: the allocation extent is symbolic
- `bytes[kv_latent]`: no metadata bound: the allocation extent is symbolic
- `bytes[q_pe]`: no metadata bound: the allocation extent is symbolic
- `bytes[k_pe]`: no metadata bound: the allocation extent is symbolic
- `bytes[w_q_nope]`: no metadata bound: the allocation extent is symbolic
- `bytes[w_k_nope]`: no metadata bound: the allocation extent is symbolic
- `bytes[w_v]`: no metadata bound: the allocation extent is symbolic
- `M_in`: per-tensor storage widths s_x
- `M_in`: aliases not deduplicated in this symbolic sum (see ledger)
- `M_in`: equals the ledger's allocated_unique_total when no two roles alias
- `HBM_mat[Q~]`: conditional on materialize.q_tilde = None
- `HBM_mat[Q~]`: write traffic B*H*R_k*S_q*s_Qt; read traffic B*H*R_k*S_q*s_Qt
- `M_mat[Q~]`: conditional on materialize.q_tilde = None
- `HBM_mat[Z]`: conditional on materialize.z = None
- `HBM_mat[Z]`: write traffic B*H*R_k*S_q*s_Z; read traffic B*H*R_k*S_q*s_Z
- `M_mat[Z]`: conditional on materialize.z = None

<details><summary>Provenance (bindings, coverage, source, evidence)</summary>

| metric | source | coverage | bindings | evidence |
|---|---|---|---|---|
| bytes[q_latent] | input | incomplete: depends on undeclared/unbound B, R_q, S_q, s_Cq | `{}` |  |
| bytes[kv_latent] | input | incomplete: depends on undeclared/unbound B, R_k, S_k, s_Ck | `{}` |  |
| bytes[q_pe] | input | incomplete: depends on undeclared/unbound B, D_r, H, S_q, s_Qr | `{}` |  |
| bytes[k_pe] | input | incomplete: depends on undeclared/unbound B, D_r, S_k, s_Kr | `{}` |  |
| bytes[w_q_nope] | input | incomplete: depends on undeclared/unbound D_n, H, R_q, s_Wq | `{}` |  |
| bytes[w_k_nope] | input | incomplete: depends on undeclared/unbound D_n, H, R_k, s_Wk | `{}` |  |
| bytes[w_v] | input | incomplete: depends on undeclared/unbound D_v, H, R_k, s_Wv | `{}` |  |
| M_in | derived | incomplete: depends on undeclared/unbound B, D_n, D_r, D_v, H, R_k, R_q, S_k, S_q, s_Ck, s_Cq, s_Kr, s_Qr, s_Wk, s_Wq, s_Wv | `{}` |  |
| M_O | derived | incomplete: depends on undeclared/unbound B, D_v, H, S_q, output_dtype, s_O | `{}` |  |
| M_LSE | conditional | incomplete: depends on undeclared/unbound B, H, S_q, lse_dtype, outputs, s_LSE | `{}` |  |
| HBM_mat[Q~] | conditional | incomplete: depends on undeclared/unbound B, H, R_k, S_q, materialize.q_tilde, s_Qt | `{}` |  |
| M_mat[Q~] | conditional | incomplete: depends on undeclared/unbound B, H, R_k, S_q, materialize.q_tilde, s_Qt | `{}` |  |
| HBM_mat[Z] | conditional | incomplete: depends on undeclared/unbound B, H, R_k, S_q, materialize.z, s_Z | `{}` |  |
| M_mat[Z] | conditional | incomplete: depends on undeclared/unbound B, H, R_k, S_q, materialize.z, s_Z | `{}` |  |

</details>

## 5. bytes: transfer requests per path

| metric | formula id | expression | value | unit | status | missing | scope | not equivalent to |
|---|---|---|---|---|---|---|---|---|
| B[hbm_to_vmem] | `bytes.path.hbm_to_vmem` | `B*D_n*H*R_k*s_Wk*ceiling(S_q/b_q) + B*D_n*H*R_q*s_Wq*ceiling(S_q/b_q) + B*D_r*H*S_q*s_Qr + B*D_v*H*R_k*s_Wv*ceiling(S_q/b_q) + B*H*R_k*S_q*s_Qt + B*H*R_k*S_q*s_Z + B*R_q*S_q*s_Cq*ceiling(H/h_pp) + D_r*N_keyvisits*s_Kr*ceiling(H/h_pp) + N_keyvisits*R_k*s_Ck*ceiling(H/h_pp) → B*D_n*H*R_k*s_Wk*ceiling(S_q/b_q) + B*D_n*H*R_q*s_Wq*ceiling(S_q/b_q) + B*D_r*H*S_q*s_Qr + B*D_v*H*R_k*s_Wv*ceiling(S_q/b_q) + B*H*R_k*S_q*s_Qt + B*H*R_k*S_q*s_Z + B*R_q*S_q*s_Cq*ceiling(H/h_pp) + D_r*N_keyvisits*s_Kr*ceiling(H/h_pp) + N_keyvisits*R_k*s_Ck*ceiling(H/h_pp)` |  | byte | symbolic | B, D_n, D_r, D_v, H, N_keyvisits, R_k, R_q, S_q, b_q, executed_extent_policy, heads_per_program, mask, materialize.q_tilde, materialize.z, outputs, rect_policy, s_Ck, s_Cq, s_Kr, s_Qr, s_Qt, s_Wk, s_Wq, s_Wv, s_Z, schedule | sum_r m_r n_r over the scenario's transfer events | physical HBM traffic with caching; spill traffic |
| B[vmem_to_hbm] | `bytes.path.vmem_to_hbm` | `B*D_v*H*S_q*s_O + B*H*R_k*S_q*s_Qt + B*H*R_k*S_q*s_Z + B*H*S_q*s_LSE → B*D_v*H*S_q*s_O + B*H*R_k*S_q*s_Qt + B*H*R_k*S_q*s_Z + B*H*S_q*s_LSE` |  | byte | symbolic | B, D_v, H, R_k, S_q, executed_extent_policy, heads_per_program, mask, materialize.q_tilde, materialize.z, outputs, rect_policy, s_LSE, s_O, s_Qt, s_Z, schedule | sum_r m_r n_r over the scenario's transfer events | physical HBM traffic with caching; spill traffic |
| B[vmem_to_vreg] | `bytes.path.vmem_to_vreg` | `B*D_r*H*b_q*s_Qr*ceiling(S_q/b_q) + B*H*R_k*b_q*s_Qt*ceiling(S_q/b_q) + D_r*N_kvblocks*b_k*s_Kr + N_kvblocks*R_k*b_k*s_Ck → B*D_r*H*b_q*s_Qr*ceiling(S_q/b_q) + B*H*R_k*b_q*s_Qt*ceiling(S_q/b_q) + D_r*N_kvblocks*b_k*s_Kr + N_kvblocks*R_k*b_k*s_Ck` |  | byte | symbolic | B, D_r, H, N_kvblocks, R_k, S_q, b_k, b_q, executed_extent_policy, heads_per_program, mask, q_resident, rect_policy, register_schedule_evidence, s_Ck, s_Kr, s_Qr, s_Qt | sum_r m_r n_r over the scenario's transfer events | physical HBM traffic with caching; spill traffic |
| B[vreg_to_vmem] | `bytes.path.vreg_to_vmem` | `sum_r m_r n_r (no modeled events)` |  | byte | unknown | register_schedule_evidence | path traffic | spill traffic; physical traffic with caching |

Assumptions:
- `B[hbm_to_vmem]`: scenario q_outer_kv_inner; weights re-read per program
- `B[hbm_to_vmem]`: no hardware cache effects are modeled: every event is counted at its declared multiplicity
- `B[hbm_to_vmem]`: includes scenario events whose condition is undeclared (Cq_read, Qr_read, mat_q_tilde_read, mat_z_read): they are counted at their declared multiplicity and named in missing_fields
- `B[vmem_to_hbm]`: scenario q_outer_kv_inner; weights re-read per program
- `B[vmem_to_hbm]`: no hardware cache effects are modeled: every event is counted at its declared multiplicity
- `B[vmem_to_hbm]`: includes scenario events whose condition is undeclared (mat_q_tilde_write, mat_z_write): they are counted at their declared multiplicity and named in missing_fields
- `B[vmem_to_vreg]`: scenario q_outer_kv_inner; weights re-read per program
- `B[vmem_to_vreg]`: no hardware cache effects are modeled: every event is counted at its declared multiplicity
- `B[vmem_to_vreg]`: operand-streaming scenario without register-schedule evidence
- `B[vmem_to_vreg]`: includes scenario events whose condition is undeclared (stream_Qt_operand, stream_Qr_operand, stream_Ck_tile, stream_Kr_tile): they are counted at their declared multiplicity and named in missing_fields

<details><summary>Provenance (bindings, coverage, source, evidence)</summary>

| metric | source | coverage | bindings | evidence |
|---|---|---|---|---|
| B[hbm_to_vmem] | conditional | incomplete: depends on undeclared/unbound B, D_n, D_r, D_v, H, N_keyvisits, R_k, R_q, S_q, b_q, executed_extent_policy, heads_per_program, mask, materialize.q_tilde, materialize.z, outputs, rect_policy, s_Ck, s_Cq, s_Kr, s_Qr, s_Qt, s_Wk, s_Wq, s_Wv, s_Z, schedule | `{}` |  |
| B[vmem_to_hbm] | conditional | incomplete: depends on undeclared/unbound B, D_v, H, R_k, S_q, executed_extent_policy, heads_per_program, mask, materialize.q_tilde, materialize.z, outputs, rect_policy, s_LSE, s_O, s_Qt, s_Z, schedule | `{}` |  |
| B[vmem_to_vreg] | conditional | incomplete: depends on undeclared/unbound B, D_r, H, N_kvblocks, R_k, S_q, b_k, b_q, executed_extent_policy, heads_per_program, mask, q_resident, rect_policy, register_schedule_evidence, s_Ck, s_Kr, s_Qr, s_Qt | `{}` |  |
| B[vreg_to_vmem] | conditional | incomplete: depends on undeclared/unbound register_schedule_evidence | `{}` |  |

</details>

## 6. local objects (symbolic sizes, spec 7.1)

| metric | formula id | expression | value | unit | status | missing | scope | not equivalent to |
|---|---|---|---|---|---|---|---|---|
| local[q_latent_window] | `local.q_latent_window.bytes` | `R_q*b_q*s_Cq → R_q*b_q*s_Cq` |  | byte | symbolic | R_q, b_q, b_rq, s_Cq | one logical object; level=vmem; live in ['q_proj'] | VREG allocation; spill traffic |
| local[wq_rank_tile] | `local.wq_rank_tile.bytes` | `D_n*R_q*s_Wq → D_n*R_q*s_Wq` |  | byte | symbolic | D_n, R_q, b_rq, s_Wq | one logical object; level=vmem; live in ['q_proj'] | VREG allocation; spill traffic |
| local[q_proj_acc] | `local.q_proj_acc.bytes` | `D_n*b_q*s_Qacc → D_n*b_q*s_Qacc` |  | byte | symbolic | D_n, b_q, s_Qacc | one logical object; level=vreg; live in ['q_proj', 'q_absorb'] | VREG allocation; spill traffic |
| local[wk_full] | `local.wk_full.bytes` | `D_n*R_k*s_Wk → D_n*R_k*s_Wk` |  | byte | symbolic | D_n, R_k, s_Wk | one logical object; level=vmem; live in ['q_absorb'] | VREG allocation; spill traffic |
| local[q_tilde_acc] | `local.q_tilde_acc.bytes` | `R_k*b_q*s_Qacc → R_k*b_q*s_Qacc` |  | byte | symbolic | R_k, b_q, s_Qacc | one logical object; level=vreg; live in ['q_absorb'] | VREG allocation; spill traffic |
| local[q_tilde] | `local.q_tilde.bytes` | `R_k*b_q*s_Qt → R_k*b_q*s_Qt` |  | byte | symbolic | R_k, b_q, q_resident, s_Qt | one logical object; level=vmem; live in ['score', 'softmax', 'pv', 'finalize'] | VREG allocation; spill traffic |
| local[kv_latent_tile] | `local.kv_latent_tile.bytes` | `R_k*b_k*n_buf*s_Ck → R_k*b_k*n_buf*s_Ck` |  | byte | symbolic | R_k, b_k, kv_buffers, s_Ck | one logical object x its buffer count; level=vmem; live in ['score', 'softmax', 'pv']; residency window of n_buf=n_buf buffers (7.3 allocation, not the 7.1 logical size) | VREG allocation; spill traffic; the 7.1 logical size of one buffer |
| local[wv_full] | `local.wv_full.bytes` | `D_v*R_k*s_Wv → D_v*R_k*s_Wv` |  | byte | symbolic | D_v, R_k, s_Wv | one logical object; level=vmem; live in ['finalize'] | VREG allocation; spill traffic |
| local[z_tile] | `local.z_tile.bytes` | `R_k*b_q*s_Z → R_k*b_q*s_Z` |  | byte | symbolic | R_k, b_q, s_Z | one logical object; level=vreg; live in ['finalize'] | VREG allocation; spill traffic |
| local[score_X] | `local.score_X.bytes` | `b_k*b_q*s_X → b_k*b_q*s_X` |  | byte | symbolic | b_k, b_q, s_X | one logical object; level=vreg; live in ['score', 'softmax'] | VREG allocation; spill traffic |
| local[exp_E] | `local.exp_E.bytes` | `b_k*b_q*s_E → b_k*b_q*s_E` |  | byte | symbolic | b_k, b_q, s_E, score_alias_exp | one logical object; level=vreg; live in ['softmax', 'pv'] | VREG allocation; spill traffic |
| local[p_operand] | `local.p_operand.bytes` | `b_k*b_q*s_P → b_k*b_q*s_P` |  | byte | symbolic | b_k, b_q, exp_alias_p_operand, s_P | one logical object; level=vreg; live in ['pv'] | VREG allocation; spill traffic |
| local[row_state_m] | `local.row_state_m.bytes` | `b_q*s_state → b_q*s_state` |  | byte | symbolic | b_q, s_state | one logical object; level=vreg; live in ['score', 'softmax', 'pv', 'finalize'] | VREG allocation; spill traffic |
| local[row_state_l] | `local.row_state_l.bytes` | `b_q*s_state → b_q*s_state` |  | byte | symbolic | b_q, s_state | one logical object; level=vreg; live in ['score', 'softmax', 'pv', 'finalize'] | VREG allocation; spill traffic |
| local[row_alpha] | `local.row_alpha.bytes` | `b_q*s_state → b_q*s_state` |  | byte | symbolic | b_q, s_state | one logical object; level=vreg; live in ['softmax', 'pv'] | VREG allocation; spill traffic |
| local[accumulator_A] | `local.accumulator_A.bytes` | `R_k*b_q*s_A → R_k*b_q*s_A` |  | byte | symbolic | R_k, b_q, s_A | one logical object; level=vreg; live in ['score', 'softmax', 'pv', 'finalize'] | VREG allocation; spill traffic |
| local[score_rope_branch] | `local.score_rope_branch.bytes` | `b_k*b_q*s_X → b_k*b_q*s_X` |  | byte | symbolic | b_k, b_q, both_qk_branches_live and D_r > 0, s_X | one logical object; level=vreg; live in ['score'] | VREG allocation; spill traffic |
| local[q_pe_tile] | `local.q_pe_tile.bytes` | `D_r*b_q*s_Qr → D_r*b_q*s_Qr` |  | byte | symbolic | D_r, b_q, q_resident, s_Qr | one logical object; level=vmem; live in ['score', 'softmax', 'pv', 'finalize'] | VREG allocation; spill traffic |
| local[k_pe_tile] | `local.k_pe_tile.bytes` | `D_r*b_k*n_buf*s_Kr → D_r*b_k*n_buf*s_Kr` |  | byte | symbolic | D_r, b_k, kv_buffers, s_Kr | one logical object x its buffer count; level=vmem; live in ['score']; residency window of n_buf=n_buf buffers (7.3 allocation, not the 7.1 logical size) | VREG allocation; spill traffic; the 7.1 logical size of one buffer |
| local[fixed_scratch] | `local.fixed_scratch.bytes` | `s_scratch → s_scratch` |  | byte | symbolic | scratch_bytes | one logical object; level=vmem; live in ['q_proj', 'q_absorb', 'score', 'softmax', 'pv', 'finalize'] | VREG allocation; spill traffic |
| local[out_tile] | `local.out_tile.bytes` | `D_v*b_q*s_O → D_v*b_q*s_O` |  | byte | symbolic | D_v, b_q, s_O | one logical object; level=vmem; live in ['finalize'] | VREG allocation; spill traffic |
| local[lse_tile] | `local.lse_tile.bytes` | `b_q*s_LSE → b_q*s_LSE` |  | byte | symbolic | b_q, outputs, s_LSE | one logical object; level=vmem; live in ['finalize'] | VREG allocation; spill traffic |
| layout[q_latent_window] | `layout.q_latent_window` | `L_l*L_s*s_Cq*ceiling(R_q/L_l)*ceiling(b_q/L_s) → L_l*L_s*s_Cq*ceiling(R_q/L_l)*ceiling(b_q/L_s)` |  | byte | symbolic | L_l, L_s, R_q, b_q, layout.q_latent_window, s_Cq | layout coverage, kind=vector_tiled, level=vmem (scenario: kind undeclared) | register-file capacity |
| layout[wq_rank_tile] | `layout.wq_rank_tile` | `L_l*L_s*s_Wq*ceiling(D_n/L_l)*ceiling(R_q/L_s) → L_l*L_s*s_Wq*ceiling(D_n/L_l)*ceiling(R_q/L_s)` |  | byte | symbolic | D_n, L_l, L_s, R_q, layout.wq_rank_tile, s_Wq | layout coverage, kind=vector_tiled, level=vmem (scenario: kind undeclared) | register-file capacity |
| layout[q_proj_acc] | `layout.q_proj_acc` | `L_l*L_s*s_Qacc*ceiling(D_n/L_l)*ceiling(b_q/L_s) → L_l*L_s*s_Qacc*ceiling(D_n/L_l)*ceiling(b_q/L_s)` |  | byte | symbolic | D_n, L_l, L_s, b_q, layout.q_proj_acc, s_Qacc | layout coverage, kind=vector_tiled, level=vreg (scenario: kind undeclared) | register-file capacity |
| layout[wk_full] | `layout.wk_full` | `L_l*L_s*s_Wk*ceiling(D_n/L_l)*ceiling(R_k/L_s) → L_l*L_s*s_Wk*ceiling(D_n/L_l)*ceiling(R_k/L_s)` |  | byte | symbolic | D_n, L_l, L_s, R_k, layout.wk_full, s_Wk | layout coverage, kind=vector_tiled, level=vmem (scenario: kind undeclared) | register-file capacity |
| layout[q_tilde_acc] | `layout.q_tilde_acc` | `L_l*L_s*s_Qacc*ceiling(R_k/L_l)*ceiling(b_q/L_s) → L_l*L_s*s_Qacc*ceiling(R_k/L_l)*ceiling(b_q/L_s)` |  | byte | symbolic | L_l, L_s, R_k, b_q, layout.q_tilde_acc, s_Qacc | layout coverage, kind=vector_tiled, level=vreg (scenario: kind undeclared) | register-file capacity |
| layout[q_tilde] | `layout.q_tilde` | `L_l*L_s*s_Qt*ceiling(R_k/L_l)*ceiling(b_q/L_s) → L_l*L_s*s_Qt*ceiling(R_k/L_l)*ceiling(b_q/L_s)` |  | byte | symbolic | L_l, L_s, R_k, b_q, layout.q_tilde, s_Qt | layout coverage, kind=vector_tiled, level=vmem (scenario: kind undeclared) | register-file capacity |
| layout[kv_latent_tile] | `layout.kv_latent_tile` | `L_l*L_s*n_buf*s_Ck*ceiling(R_k/L_l)*ceiling(b_k/L_s) → L_l*L_s*n_buf*s_Ck*ceiling(R_k/L_l)*ceiling(b_k/L_s)` |  | byte | symbolic | L_l, L_s, R_k, b_k, kv_buffers, layout.kv_latent_tile, s_Ck | layout coverage, kind=vector_tiled, level=vmem (scenario: kind undeclared) | register-file capacity |
| layout[wv_full] | `layout.wv_full` | `L_l*L_s*s_Wv*ceiling(D_v/L_l)*ceiling(R_k/L_s) → L_l*L_s*s_Wv*ceiling(D_v/L_l)*ceiling(R_k/L_s)` |  | byte | symbolic | D_v, L_l, L_s, R_k, layout.wv_full, s_Wv | layout coverage, kind=vector_tiled, level=vmem (scenario: kind undeclared) | register-file capacity |
| layout[z_tile] | `layout.z_tile` | `L_l*L_s*s_Z*ceiling(R_k/L_l)*ceiling(b_q/L_s) → L_l*L_s*s_Z*ceiling(R_k/L_l)*ceiling(b_q/L_s)` |  | byte | symbolic | L_l, L_s, R_k, b_q, layout.z_tile, s_Z | layout coverage, kind=vector_tiled, level=vreg (scenario: kind undeclared) | register-file capacity |
| layout[score_X] | `layout.score_X` | `L_l*L_s*s_X*ceiling(b_k/L_l)*ceiling(b_q/L_s) → L_l*L_s*s_X*ceiling(b_k/L_l)*ceiling(b_q/L_s)` |  | byte | symbolic | L_l, L_s, b_k, b_q, layout.score_X, s_X | layout coverage, kind=vector_tiled, level=vreg (scenario: kind undeclared) | register-file capacity |
| layout[exp_E] | `layout.exp_E` | `L_l*L_s*s_E*ceiling(b_k/L_l)*ceiling(b_q/L_s) → L_l*L_s*s_E*ceiling(b_k/L_l)*ceiling(b_q/L_s)` |  | byte | symbolic | L_l, L_s, b_k, b_q, layout.exp_E, s_E | layout coverage, kind=vector_tiled, level=vreg (scenario: kind undeclared) | register-file capacity |
| layout[p_operand] | `layout.p_operand` | `L_l*L_s*s_P*ceiling(b_k/L_l)*ceiling(b_q/L_s) → L_l*L_s*s_P*ceiling(b_k/L_l)*ceiling(b_q/L_s)` |  | byte | symbolic | L_l, L_s, b_k, b_q, layout.p_operand, s_P | layout coverage, kind=vector_tiled, level=vreg (scenario: kind undeclared) | register-file capacity |
| layout[row_state_m] | `layout.row_state_m` | `L_l*L_s*s_state*ceiling(1/L_l)*ceiling(b_q/L_s) → L_l*L_s*s_state*ceiling(1/L_l)*ceiling(b_q/L_s)` |  | byte | symbolic | L_l, L_s, b_q, layout.row_state_m, s_state | layout coverage, kind=vector_tiled, level=vreg (scenario: kind undeclared) | register-file capacity |
| layout[row_state_l] | `layout.row_state_l` | `L_l*L_s*s_state*ceiling(1/L_l)*ceiling(b_q/L_s) → L_l*L_s*s_state*ceiling(1/L_l)*ceiling(b_q/L_s)` |  | byte | symbolic | L_l, L_s, b_q, layout.row_state_l, s_state | layout coverage, kind=vector_tiled, level=vreg (scenario: kind undeclared) | register-file capacity |
| layout[row_alpha] | `layout.row_alpha` | `L_l*L_s*s_state*ceiling(1/L_l)*ceiling(b_q/L_s) → L_l*L_s*s_state*ceiling(1/L_l)*ceiling(b_q/L_s)` |  | byte | symbolic | L_l, L_s, b_q, layout.row_alpha, s_state | layout coverage, kind=vector_tiled, level=vreg (scenario: kind undeclared) | register-file capacity |
| layout[accumulator_A] | `layout.accumulator_A` | `L_l*L_s*s_A*ceiling(R_k/L_l)*ceiling(b_q/L_s) → L_l*L_s*s_A*ceiling(R_k/L_l)*ceiling(b_q/L_s)` |  | byte | symbolic | L_l, L_s, R_k, b_q, layout.accumulator_A, s_A | layout coverage, kind=vector_tiled, level=vreg (scenario: kind undeclared) | register-file capacity |
| layout[score_rope_branch] | `layout.score_rope_branch` | `L_l*L_s*s_X*ceiling(b_k/L_l)*ceiling(b_q/L_s) → L_l*L_s*s_X*ceiling(b_k/L_l)*ceiling(b_q/L_s)` |  | byte | symbolic | L_l, L_s, b_k, b_q, layout.score_rope_branch, s_X | layout coverage, kind=vector_tiled, level=vreg (scenario: kind undeclared) | register-file capacity |
| layout[q_pe_tile] | `layout.q_pe_tile` | `L_l*L_s*s_Qr*ceiling(D_r/L_l)*ceiling(b_q/L_s) → L_l*L_s*s_Qr*ceiling(D_r/L_l)*ceiling(b_q/L_s)` |  | byte | symbolic | D_r, L_l, L_s, b_q, layout.q_pe_tile, s_Qr | layout coverage, kind=vector_tiled, level=vmem (scenario: kind undeclared) | register-file capacity |
| layout[k_pe_tile] | `layout.k_pe_tile` | `L_l*L_s*n_buf*s_Kr*ceiling(D_r/L_l)*ceiling(b_k/L_s) → L_l*L_s*n_buf*s_Kr*ceiling(D_r/L_l)*ceiling(b_k/L_s)` |  | byte | symbolic | D_r, L_l, L_s, b_k, kv_buffers, layout.k_pe_tile, s_Kr | layout coverage, kind=vector_tiled, level=vmem (scenario: kind undeclared) | register-file capacity |
| layout[fixed_scratch] | `layout.fixed_scratch` | `s_scratch → s_scratch` |  | byte | symbolic | scratch_bytes | layout coverage, kind=raw_bytes, level=vmem | register-file capacity |
| layout[out_tile] | `layout.out_tile` | `L_l*L_s*s_O*ceiling(D_v/L_l)*ceiling(b_q/L_s) → L_l*L_s*s_O*ceiling(D_v/L_l)*ceiling(b_q/L_s)` |  | byte | symbolic | D_v, L_l, L_s, b_q, layout.out_tile, s_O | layout coverage, kind=vector_tiled, level=vmem (scenario: kind undeclared) | register-file capacity |
| layout[lse_tile] | `layout.lse_tile` | `L_l*L_s*s_LSE*ceiling(1/L_l)*ceiling(b_q/L_s) → L_l*L_s*s_LSE*ceiling(1/L_l)*ceiling(b_q/L_s)` |  | byte | symbolic | L_l, L_s, b_q, layout.lse_tile, s_LSE | layout coverage, kind=vector_tiled, level=vmem (scenario: kind undeclared) | register-file capacity |
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
- `local[exp_E]`: exists only if score_alias_exp (undeclared)
- `local[p_operand]`: P/E operand in the matmul input dtype
- `local[p_operand]`: exists only if exp_alias_p_operand (undeclared)
- `local[row_state_m]`: running row max m (loop-carried)
- `local[row_state_l]`: running row sum l (loop-carried)
- `local[row_alpha]`: alpha = exp(m - m')
- `local[accumulator_A]`: attention accumulator A, width R_k (loop-carried)
- `local[score_rope_branch]`: second QK partial product kept simultaneously (both_qk_branches_live)
- `local[score_rope_branch]`: exists only if both_qk_branches_live and D_r > 0 (undeclared)
- `local[q_pe_tile]`: Q^r operand held across the KV loop (shortened when q_resident=False)
- `local[q_pe_tile]`: q_resident undeclared: the live-stage set printed in the scope is the scenario
- `local[k_pe_tile]`: K^r tile for the current KV block
- `local[fixed_scratch]`: fixed scratch (d_s)
- `local[fixed_scratch]`: exists only if scratch_bytes (undeclared)
- `local[out_tile]`: output tile O in output dtype
- `local[lse_tile]`: LSE tile
- `local[lse_tile]`: exists only if outputs (undeclared)
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
- `layout[score_rope_branch]`: LayoutBytes = s L_s L_l ceil(m/L_s) ceil(n/L_l) with profile tiles
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
| local[q_latent_window] | derived | incomplete: depends on undeclared/unbound R_q, b_q, b_rq, s_Cq | `{}` |  |
| local[wq_rank_tile] | derived | incomplete: depends on undeclared/unbound D_n, R_q, b_rq, s_Wq | `{}` |  |
| local[q_proj_acc] | derived | incomplete: depends on undeclared/unbound D_n, b_q, s_Qacc | `{}` |  |
| local[wk_full] | derived | incomplete: depends on undeclared/unbound D_n, R_k, s_Wk | `{}` |  |
| local[q_tilde_acc] | derived | incomplete: depends on undeclared/unbound R_k, b_q, s_Qacc | `{}` |  |
| local[q_tilde] | derived | incomplete: depends on undeclared/unbound R_k, b_q, q_resident, s_Qt | `{}` |  |
| local[kv_latent_tile] | derived | incomplete: depends on undeclared/unbound R_k, b_k, kv_buffers, s_Ck | `{}` |  |
| local[wv_full] | derived | incomplete: depends on undeclared/unbound D_v, R_k, s_Wv | `{}` |  |
| local[z_tile] | derived | incomplete: depends on undeclared/unbound R_k, b_q, s_Z | `{}` |  |
| local[score_X] | derived | incomplete: depends on undeclared/unbound b_k, b_q, s_X | `{}` |  |
| local[exp_E] | derived | incomplete: depends on undeclared/unbound b_k, b_q, s_E, score_alias_exp | `{}` |  |
| local[p_operand] | derived | incomplete: depends on undeclared/unbound b_k, b_q, exp_alias_p_operand, s_P | `{}` |  |
| local[row_state_m] | derived | incomplete: depends on undeclared/unbound b_q, s_state | `{}` |  |
| local[row_state_l] | derived | incomplete: depends on undeclared/unbound b_q, s_state | `{}` |  |
| local[row_alpha] | derived | incomplete: depends on undeclared/unbound b_q, s_state | `{}` |  |
| local[accumulator_A] | derived | incomplete: depends on undeclared/unbound R_k, b_q, s_A | `{}` |  |
| local[score_rope_branch] | derived | incomplete: depends on undeclared/unbound b_k, b_q, both_qk_branches_live and D_r > 0, s_X | `{}` |  |
| local[q_pe_tile] | derived | incomplete: depends on undeclared/unbound D_r, b_q, q_resident, s_Qr | `{}` |  |
| local[k_pe_tile] | derived | incomplete: depends on undeclared/unbound D_r, b_k, kv_buffers, s_Kr | `{}` |  |
| local[fixed_scratch] | derived | incomplete: depends on undeclared/unbound scratch_bytes | `{}` |  |
| local[out_tile] | derived | incomplete: depends on undeclared/unbound D_v, b_q, s_O | `{}` |  |
| local[lse_tile] | derived | incomplete: depends on undeclared/unbound b_q, outputs, s_LSE | `{}` |  |
| layout[q_latent_window] | conditional | incomplete: depends on undeclared/unbound L_l, L_s, R_q, b_q, layout.q_latent_window, s_Cq | `{}` |  |
| layout[wq_rank_tile] | conditional | incomplete: depends on undeclared/unbound D_n, L_l, L_s, R_q, layout.wq_rank_tile, s_Wq | `{}` |  |
| layout[q_proj_acc] | conditional | incomplete: depends on undeclared/unbound D_n, L_l, L_s, b_q, layout.q_proj_acc, s_Qacc | `{}` |  |
| layout[wk_full] | conditional | incomplete: depends on undeclared/unbound D_n, L_l, L_s, R_k, layout.wk_full, s_Wk | `{}` |  |
| layout[q_tilde_acc] | conditional | incomplete: depends on undeclared/unbound L_l, L_s, R_k, b_q, layout.q_tilde_acc, s_Qacc | `{}` |  |
| layout[q_tilde] | conditional | incomplete: depends on undeclared/unbound L_l, L_s, R_k, b_q, layout.q_tilde, s_Qt | `{}` |  |
| layout[kv_latent_tile] | conditional | incomplete: depends on undeclared/unbound L_l, L_s, R_k, b_k, kv_buffers, layout.kv_latent_tile, s_Ck | `{}` |  |
| layout[wv_full] | conditional | incomplete: depends on undeclared/unbound D_v, L_l, L_s, R_k, layout.wv_full, s_Wv | `{}` |  |
| layout[z_tile] | conditional | incomplete: depends on undeclared/unbound L_l, L_s, R_k, b_q, layout.z_tile, s_Z | `{}` |  |
| layout[score_X] | conditional | incomplete: depends on undeclared/unbound L_l, L_s, b_k, b_q, layout.score_X, s_X | `{}` |  |
| layout[exp_E] | conditional | incomplete: depends on undeclared/unbound L_l, L_s, b_k, b_q, layout.exp_E, s_E | `{}` |  |
| layout[p_operand] | conditional | incomplete: depends on undeclared/unbound L_l, L_s, b_k, b_q, layout.p_operand, s_P | `{}` |  |
| layout[row_state_m] | conditional | incomplete: depends on undeclared/unbound L_l, L_s, b_q, layout.row_state_m, s_state | `{}` |  |
| layout[row_state_l] | conditional | incomplete: depends on undeclared/unbound L_l, L_s, b_q, layout.row_state_l, s_state | `{}` |  |
| layout[row_alpha] | conditional | incomplete: depends on undeclared/unbound L_l, L_s, b_q, layout.row_alpha, s_state | `{}` |  |
| layout[accumulator_A] | conditional | incomplete: depends on undeclared/unbound L_l, L_s, R_k, b_q, layout.accumulator_A, s_A | `{}` |  |
| layout[score_rope_branch] | conditional | incomplete: depends on undeclared/unbound L_l, L_s, b_k, b_q, layout.score_rope_branch, s_X | `{}` |  |
| layout[q_pe_tile] | conditional | incomplete: depends on undeclared/unbound D_r, L_l, L_s, b_q, layout.q_pe_tile, s_Qr | `{}` |  |
| layout[k_pe_tile] | conditional | incomplete: depends on undeclared/unbound D_r, L_l, L_s, b_k, kv_buffers, layout.k_pe_tile, s_Kr | `{}` |  |
| layout[fixed_scratch] | conditional | incomplete: depends on undeclared/unbound scratch_bytes | `{}` |  |
| layout[out_tile] | conditional | incomplete: depends on undeclared/unbound D_v, L_l, L_s, b_q, layout.out_tile, s_O | `{}` |  |
| layout[lse_tile] | conditional | incomplete: depends on undeclared/unbound L_l, L_s, b_q, layout.lse_tile, s_LSE | `{}` |  |
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
| M_live[all:q_proj] | `pressure.live.all.q_proj` | `D_n*R_q*s_Wq + D_n*b_q*s_Qacc + R_q*b_q*s_Cq + s_scratch → D_n*R_q*s_Wq + D_n*b_q*s_Qacc + R_q*b_q*s_Cq + s_scratch` |  | byte | symbolic | D_n, R_q, b_q, b_rq, exp_alias_p_operand, s_Cq, s_Qacc, s_Wq, score_alias_exp, scratch_bytes | source-level live set at this stage cut, level=all (aliases deduplicated) | compiler register allocation; VMEM allocation |
| M_live[all:q_absorb] | `pressure.live.all.q_absorb` | `D_n*R_k*s_Wk + D_n*b_q*s_Qacc + R_k*b_q*s_Qacc + s_scratch → D_n*R_k*s_Wk + D_n*b_q*s_Qacc + R_k*b_q*s_Qacc + s_scratch` |  | byte | symbolic | D_n, R_k, b_q, exp_alias_p_operand, s_Qacc, s_Wk, score_alias_exp, scratch_bytes | source-level live set at this stage cut, level=all (aliases deduplicated) | compiler register allocation; VMEM allocation |
| M_live[all:score] | `pressure.live.all.score` | `D_r*b_k*n_buf*s_Kr + D_r*b_q*s_Qr + R_k*b_k*n_buf*s_Ck + R_k*b_q*s_Qt + 2*b_k*b_q*s_X + 2*b_q*s_state + s_scratch + Max(R_k*b_q*s_A, R_k*b_q*s_Z) → D_r*b_k*n_buf*s_Kr + D_r*b_q*s_Qr + R_k*b_k*n_buf*s_Ck + R_k*b_q*s_Qt + 2*b_k*b_q*s_X + 2*b_q*s_state + s_scratch + Max(R_k*b_q*s_A, R_k*b_q*s_Z)` |  | byte | symbolic | D_r, R_k, b_k, b_q, both_qk_branches_live and D_r > 0, exp_alias_p_operand, kv_buffers, s_A, s_Ck, s_Kr, s_Qr, s_Qt, s_X, s_Z, s_state, score_alias_exp, scratch_bytes | source-level live set at this stage cut, level=all (aliases deduplicated) | compiler register allocation; VMEM allocation |
| M_live[all:softmax] | `pressure.live.all.softmax` | `D_r*b_q*s_Qr + R_k*b_k*n_buf*s_Ck + R_k*b_q*s_Qt + b_k*b_q*s_E + b_k*b_q*s_X + 3*b_q*s_state + s_scratch + Max(R_k*b_q*s_A, R_k*b_q*s_Z) → D_r*b_q*s_Qr + R_k*b_k*n_buf*s_Ck + R_k*b_q*s_Qt + b_k*b_q*s_E + b_k*b_q*s_X + 3*b_q*s_state + s_scratch + Max(R_k*b_q*s_A, R_k*b_q*s_Z)` |  | byte | symbolic | D_r, R_k, b_k, b_q, exp_alias_p_operand, kv_buffers, q_resident, s_A, s_Ck, s_E, s_Qr, s_Qt, s_X, s_Z, s_state, score_alias_exp, scratch_bytes | source-level live set at this stage cut, level=all (aliases deduplicated) | compiler register allocation; VMEM allocation |
| M_live[all:pv] | `pressure.live.all.pv` | `D_r*b_q*s_Qr + R_k*b_k*n_buf*s_Ck + R_k*b_q*s_Qt + b_k*b_q*s_E + b_k*b_q*s_P + 3*b_q*s_state + s_scratch + Max(R_k*b_q*s_A, R_k*b_q*s_Z) → D_r*b_q*s_Qr + R_k*b_k*n_buf*s_Ck + R_k*b_q*s_Qt + b_k*b_q*s_E + b_k*b_q*s_P + 3*b_q*s_state + s_scratch + Max(R_k*b_q*s_A, R_k*b_q*s_Z)` |  | byte | symbolic | D_r, R_k, b_k, b_q, exp_alias_p_operand, kv_buffers, q_resident, s_A, s_Ck, s_E, s_P, s_Qr, s_Qt, s_Z, s_state, score_alias_exp, scratch_bytes | source-level live set at this stage cut, level=all (aliases deduplicated) | compiler register allocation; VMEM allocation |
| M_live[all:finalize] | `pressure.live.all.finalize` | `D_r*b_q*s_Qr + D_v*R_k*s_Wv + D_v*b_q*s_O + R_k*b_q*s_Qt + b_q*s_LSE + 2*b_q*s_state + s_scratch + Max(R_k*b_q*s_A, R_k*b_q*s_Z) → D_r*b_q*s_Qr + D_v*R_k*s_Wv + D_v*b_q*s_O + R_k*b_q*s_Qt + b_q*s_LSE + 2*b_q*s_state + s_scratch + Max(R_k*b_q*s_A, R_k*b_q*s_Z)` |  | byte | symbolic | D_r, D_v, R_k, b_q, exp_alias_p_operand, outputs, q_resident, s_A, s_LSE, s_O, s_Qr, s_Qt, s_Wv, s_Z, s_state, score_alias_exp, scratch_bytes | source-level live set at this stage cut, level=all (aliases deduplicated) | compiler register allocation; VMEM allocation |
| M_peak[all] | `pressure.peak.all` | `Max(D_n*R_k*s_Wk + D_n*b_q*s_Qacc + R_k*b_q*s_Qacc + s_scratch, D_n*R_q*s_Wq + D_n*b_q*s_Qacc + R_q*b_q*s_Cq + s_scratch, D_r*b_q*s_Qr + D_v*R_k*s_Wv + D_v*b_q*s_O + R_k*b_q*s_Qt + b_q*s_LSE + 2*b_q*s_state + s_scratch + Max(R_k*b_q*s_A, R_k*b_q*s_Z), D_r*b_q*s_Qr + R_k*b_k*n_buf*s_Ck + R_k*b_q*s_Qt + b_k*b_q*s_E + b_k*b_q*s_P + 3*b_q*s_state + s_scratch + Max(R_k*b_q*s_A, R_k*b_q*s_Z), D_r*b_q*s_Qr + R_k*b_k*n_buf*s_Ck + R_k*b_q*s_Qt + b_k*b_q*s_E + b_k*b_q*s_X + 3*b_q*s_state + s_scratch + Max(R_k*b_q*s_A, R_k*b_q*s_Z), D_r*b_k*n_buf*s_Kr + D_r*b_q*s_Qr + R_k*b_k*n_buf*s_Ck + R_k*b_q*s_Qt + 2*b_k*b_q*s_X + 2*b_q*s_state + s_scratch + Max(R_k*b_q*s_A, R_k*b_q*s_Z)) → Max(D_n*R_k*s_Wk + D_n*b_q*s_Qacc + R_k*b_q*s_Qacc + s_scratch, D_n*R_q*s_Wq + D_n*b_q*s_Qacc + R_q*b_q*s_Cq + s_scratch, D_r*b_q*s_Qr + D_v*R_k*s_Wv + D_v*b_q*s_O + R_k*b_q*s_Qt + b_q*s_LSE + 2*b_q*s_state + s_scratch + Max(R_k*b_q*s_A, R_k*b_q*s_Z), D_r*b_q*s_Qr + R_k*b_k*n_buf*s_Ck + R_k*b_q*s_Qt + b_k*b_q*s_E + b_k*b_q*s_P + 3*b_q*s_state + s_scratch + Max(R_k*b_q*s_A, R_k*b_q*s_Z), D_r*b_q*s_Qr + R_k*b_k*n_buf*s_Ck + R_k*b_q*s_Qt + b_k*b_q*s_E + b_k*b_q*s_X + 3*b_q*s_state + s_scratch + Max(R_k*b_q*s_A, R_k*b_q*s_Z), D_r*b_k*n_buf*s_Kr + D_r*b_q*s_Qr + R_k*b_k*n_buf*s_Ck + R_k*b_q*s_Qt + 2*b_k*b_q*s_X + 2*b_q*s_state + s_scratch + Max(R_k*b_q*s_A, R_k*b_q*s_Z))` |  | byte | symbolic | D_n, D_r, D_v, R_k, R_q, b_k, b_q, b_rq, both_qk_branches_live and D_r > 0, exp_alias_p_operand, kv_buffers, outputs, q_resident, s_A, s_Ck, s_Cq, s_E, s_Kr, s_LSE, s_O, s_P, s_Qacc, s_Qr, s_Qt, s_Wk, s_Wq, s_Wv, s_X, s_Z, s_state, score_alias_exp, scratch_bytes | max over stage cuts, level=all (source level) | actual spill; OOM proof |
| M_live[vreg:q_proj] | `pressure.live.vreg.q_proj` | `D_n*b_q*s_Qacc → D_n*b_q*s_Qacc` |  | byte | symbolic | D_n, b_q, exp_alias_p_operand, s_Qacc, score_alias_exp | source-level live set at this stage cut, level=vreg (aliases deduplicated) | compiler register allocation; VMEM allocation |
| M_live[vreg:q_absorb] | `pressure.live.vreg.q_absorb` | `D_n*b_q*s_Qacc + R_k*b_q*s_Qacc → D_n*b_q*s_Qacc + R_k*b_q*s_Qacc` |  | byte | symbolic | D_n, R_k, b_q, exp_alias_p_operand, s_Qacc, score_alias_exp | source-level live set at this stage cut, level=vreg (aliases deduplicated) | compiler register allocation; VMEM allocation |
| M_live[vreg:score] | `pressure.live.vreg.score` | `2*b_k*b_q*s_X + 2*b_q*s_state + Max(R_k*b_q*s_A, R_k*b_q*s_Z) → 2*b_k*b_q*s_X + 2*b_q*s_state + Max(R_k*b_q*s_A, R_k*b_q*s_Z)` |  | byte | symbolic | R_k, b_k, b_q, both_qk_branches_live and D_r > 0, exp_alias_p_operand, s_A, s_X, s_Z, s_state, score_alias_exp | source-level live set at this stage cut, level=vreg (aliases deduplicated) | compiler register allocation; VMEM allocation |
| M_live[vreg:softmax] | `pressure.live.vreg.softmax` | `b_k*b_q*s_E + b_k*b_q*s_X + 3*b_q*s_state + Max(R_k*b_q*s_A, R_k*b_q*s_Z) → b_k*b_q*s_E + b_k*b_q*s_X + 3*b_q*s_state + Max(R_k*b_q*s_A, R_k*b_q*s_Z)` |  | byte | symbolic | R_k, b_k, b_q, exp_alias_p_operand, s_A, s_E, s_X, s_Z, s_state, score_alias_exp | source-level live set at this stage cut, level=vreg (aliases deduplicated) | compiler register allocation; VMEM allocation |
| M_live[vreg:pv] | `pressure.live.vreg.pv` | `b_k*b_q*s_E + b_k*b_q*s_P + 3*b_q*s_state + Max(R_k*b_q*s_A, R_k*b_q*s_Z) → b_k*b_q*s_E + b_k*b_q*s_P + 3*b_q*s_state + Max(R_k*b_q*s_A, R_k*b_q*s_Z)` |  | byte | symbolic | R_k, b_k, b_q, exp_alias_p_operand, s_A, s_E, s_P, s_Z, s_state, score_alias_exp | source-level live set at this stage cut, level=vreg (aliases deduplicated) | compiler register allocation; VMEM allocation |
| M_live[vreg:finalize] | `pressure.live.vreg.finalize` | `2*b_q*s_state + Max(R_k*b_q*s_A, R_k*b_q*s_Z) → 2*b_q*s_state + Max(R_k*b_q*s_A, R_k*b_q*s_Z)` |  | byte | symbolic | R_k, b_q, exp_alias_p_operand, s_A, s_Z, s_state, score_alias_exp | source-level live set at this stage cut, level=vreg (aliases deduplicated) | compiler register allocation; VMEM allocation |
| M_peak[vreg] | `pressure.peak.vreg` | `Max(D_n*b_q*s_Qacc + R_k*b_q*s_Qacc, 2*b_k*b_q*s_X + 2*b_q*s_state + Max(R_k*b_q*s_A, R_k*b_q*s_Z), b_k*b_q*s_E + b_k*b_q*s_P + 3*b_q*s_state + Max(R_k*b_q*s_A, R_k*b_q*s_Z), b_k*b_q*s_E + b_k*b_q*s_X + 3*b_q*s_state + Max(R_k*b_q*s_A, R_k*b_q*s_Z)) → Max(D_n*b_q*s_Qacc + R_k*b_q*s_Qacc, 2*b_k*b_q*s_X + 2*b_q*s_state + Max(R_k*b_q*s_A, R_k*b_q*s_Z), b_k*b_q*s_E + b_k*b_q*s_P + 3*b_q*s_state + Max(R_k*b_q*s_A, R_k*b_q*s_Z), b_k*b_q*s_E + b_k*b_q*s_X + 3*b_q*s_state + Max(R_k*b_q*s_A, R_k*b_q*s_Z))` |  | byte | symbolic | D_n, R_k, b_k, b_q, both_qk_branches_live and D_r > 0, exp_alias_p_operand, s_A, s_E, s_P, s_Qacc, s_X, s_Z, s_state, score_alias_exp | max over stage cuts, level=vreg (source level) | actual spill; OOM proof |
| M_live[vmem:q_proj] | `pressure.live.vmem.q_proj` | `D_n*R_q*s_Wq + R_q*b_q*s_Cq + s_scratch → D_n*R_q*s_Wq + R_q*b_q*s_Cq + s_scratch` |  | byte | symbolic | D_n, R_q, b_q, b_rq, s_Cq, s_Wq, scratch_bytes | source-level live set at this stage cut, level=vmem (aliases deduplicated) | compiler register allocation; VMEM allocation |
| M_live[vmem:q_absorb] | `pressure.live.vmem.q_absorb` | `D_n*R_k*s_Wk + s_scratch → D_n*R_k*s_Wk + s_scratch` |  | byte | symbolic | D_n, R_k, s_Wk, scratch_bytes | source-level live set at this stage cut, level=vmem (aliases deduplicated) | compiler register allocation; VMEM allocation |
| M_live[vmem:score] | `pressure.live.vmem.score` | `D_r*b_k*n_buf*s_Kr + D_r*b_q*s_Qr + R_k*b_k*n_buf*s_Ck + R_k*b_q*s_Qt + s_scratch → D_r*b_k*n_buf*s_Kr + D_r*b_q*s_Qr + R_k*b_k*n_buf*s_Ck + R_k*b_q*s_Qt + s_scratch` |  | byte | symbolic | D_r, R_k, b_k, b_q, kv_buffers, s_Ck, s_Kr, s_Qr, s_Qt, scratch_bytes | source-level live set at this stage cut, level=vmem (aliases deduplicated) | compiler register allocation; VMEM allocation |
| M_live[vmem:softmax] | `pressure.live.vmem.softmax` | `D_r*b_q*s_Qr + R_k*b_k*n_buf*s_Ck + R_k*b_q*s_Qt + s_scratch → D_r*b_q*s_Qr + R_k*b_k*n_buf*s_Ck + R_k*b_q*s_Qt + s_scratch` |  | byte | symbolic | D_r, R_k, b_k, b_q, kv_buffers, q_resident, s_Ck, s_Qr, s_Qt, scratch_bytes | source-level live set at this stage cut, level=vmem (aliases deduplicated) | compiler register allocation; VMEM allocation |
| M_live[vmem:pv] | `pressure.live.vmem.pv` | `D_r*b_q*s_Qr + R_k*b_k*n_buf*s_Ck + R_k*b_q*s_Qt + s_scratch → D_r*b_q*s_Qr + R_k*b_k*n_buf*s_Ck + R_k*b_q*s_Qt + s_scratch` |  | byte | symbolic | D_r, R_k, b_k, b_q, kv_buffers, q_resident, s_Ck, s_Qr, s_Qt, scratch_bytes | source-level live set at this stage cut, level=vmem (aliases deduplicated) | compiler register allocation; VMEM allocation |
| M_live[vmem:finalize] | `pressure.live.vmem.finalize` | `D_r*b_q*s_Qr + D_v*R_k*s_Wv + D_v*b_q*s_O + R_k*b_q*s_Qt + b_q*s_LSE + s_scratch → D_r*b_q*s_Qr + D_v*R_k*s_Wv + D_v*b_q*s_O + R_k*b_q*s_Qt + b_q*s_LSE + s_scratch` |  | byte | symbolic | D_r, D_v, R_k, b_q, outputs, q_resident, s_LSE, s_O, s_Qr, s_Qt, s_Wv, scratch_bytes | source-level live set at this stage cut, level=vmem (aliases deduplicated) | compiler register allocation; VMEM allocation |
| M_peak[vmem] | `pressure.peak.vmem` | `Max(D_n*R_k*s_Wk + s_scratch, D_n*R_q*s_Wq + R_q*b_q*s_Cq + s_scratch, D_r*b_k*n_buf*s_Kr + D_r*b_q*s_Qr + R_k*b_k*n_buf*s_Ck + R_k*b_q*s_Qt + s_scratch, D_r*b_q*s_Qr + D_v*R_k*s_Wv + D_v*b_q*s_O + R_k*b_q*s_Qt + b_q*s_LSE + s_scratch) → Max(D_n*R_k*s_Wk + s_scratch, D_n*R_q*s_Wq + R_q*b_q*s_Cq + s_scratch, D_r*b_k*n_buf*s_Kr + D_r*b_q*s_Qr + R_k*b_k*n_buf*s_Ck + R_k*b_q*s_Qt + s_scratch, D_r*b_q*s_Qr + D_v*R_k*s_Wv + D_v*b_q*s_O + R_k*b_q*s_Qt + b_q*s_LSE + s_scratch)` |  | byte | symbolic | D_n, D_r, D_v, R_k, R_q, b_k, b_q, b_rq, kv_buffers, outputs, q_resident, s_Ck, s_Cq, s_Kr, s_LSE, s_O, s_Qr, s_Qt, s_Wk, s_Wq, s_Wv, scratch_bytes | max over stage cuts, level=vmem (source level) | actual spill; OOM proof |
| envelope[all:q_proj] | `pressure.envelope.all.q_proj` | `D_n*R_q*s_Wq + b_q*(D_n*s_Qacc + R_q*s_Cq) + s_scratch → D_n*R_q*s_Wq + b_q*(D_n*s_Qacc + R_q*s_Cq) + s_scratch` |  | byte | symbolic | D_n, R_q, b_q, b_rq, exp_alias_p_operand, s_Cq, s_Qacc, s_Wq, score_alias_exp, scratch_bytes | a*b_q*b_k + b*b_q + c*b_k + d (+ remainder) for this stage, level=all | measured spill; spill probability |
| envelope[all:q_absorb] | `pressure.envelope.all.q_absorb` | `D_n*R_k*s_Wk + b_q*s_Qacc*(D_n + R_k) + s_scratch → D_n*R_k*s_Wk + b_q*s_Qacc*(D_n + R_k) + s_scratch` |  | byte | symbolic | D_n, R_k, b_q, exp_alias_p_operand, s_Qacc, s_Wk, score_alias_exp, scratch_bytes | a*b_q*b_k + b*b_q + c*b_k + d (+ remainder) for this stage, level=all | measured spill; spill probability |
| envelope[all:score] | `pressure.envelope.all.score` | `2*b_k*b_q*s_X + b_k*n_buf*(D_r*s_Kr + R_k*s_Ck) + b_q*(D_r*s_Qr + R_k*s_Qt + R_k*Max(s_A, s_Z) + 2*s_state) + s_scratch → 2*b_k*b_q*s_X + b_k*n_buf*(D_r*s_Kr + R_k*s_Ck) + b_q*(D_r*s_Qr + R_k*s_Qt + R_k*Max(s_A, s_Z) + 2*s_state) + s_scratch` |  | byte | symbolic | D_r, R_k, b_k, b_q, both_qk_branches_live and D_r > 0, exp_alias_p_operand, kv_buffers, s_A, s_Ck, s_Kr, s_Qr, s_Qt, s_X, s_Z, s_state, score_alias_exp, scratch_bytes | a*b_q*b_k + b*b_q + c*b_k + d (+ remainder) for this stage, level=all | measured spill; spill probability |
| envelope[all:softmax] | `pressure.envelope.all.softmax` | `R_k*b_k*n_buf*s_Ck + b_k*b_q*(s_E + s_X) + b_q*(D_r*s_Qr + R_k*s_Qt + R_k*Max(s_A, s_Z) + 3*s_state) + s_scratch → R_k*b_k*n_buf*s_Ck + b_k*b_q*(s_E + s_X) + b_q*(D_r*s_Qr + R_k*s_Qt + R_k*Max(s_A, s_Z) + 3*s_state) + s_scratch` |  | byte | symbolic | D_r, R_k, b_k, b_q, exp_alias_p_operand, kv_buffers, q_resident, s_A, s_Ck, s_E, s_Qr, s_Qt, s_X, s_Z, s_state, score_alias_exp, scratch_bytes | a*b_q*b_k + b*b_q + c*b_k + d (+ remainder) for this stage, level=all | measured spill; spill probability |
| envelope[all:pv] | `pressure.envelope.all.pv` | `R_k*b_k*n_buf*s_Ck + b_k*b_q*(s_E + s_P) + b_q*(D_r*s_Qr + R_k*s_Qt + R_k*Max(s_A, s_Z) + 3*s_state) + s_scratch → R_k*b_k*n_buf*s_Ck + b_k*b_q*(s_E + s_P) + b_q*(D_r*s_Qr + R_k*s_Qt + R_k*Max(s_A, s_Z) + 3*s_state) + s_scratch` |  | byte | symbolic | D_r, R_k, b_k, b_q, exp_alias_p_operand, kv_buffers, q_resident, s_A, s_Ck, s_E, s_P, s_Qr, s_Qt, s_Z, s_state, score_alias_exp, scratch_bytes | a*b_q*b_k + b*b_q + c*b_k + d (+ remainder) for this stage, level=all | measured spill; spill probability |
| envelope[all:finalize] | `pressure.envelope.all.finalize` | `D_v*R_k*s_Wv + b_q*(D_r*s_Qr + D_v*s_O + R_k*s_Qt + R_k*Max(s_A, s_Z) + s_LSE + 2*s_state) + s_scratch → D_v*R_k*s_Wv + b_q*(D_r*s_Qr + D_v*s_O + R_k*s_Qt + R_k*Max(s_A, s_Z) + s_LSE + 2*s_state) + s_scratch` |  | byte | symbolic | D_r, D_v, R_k, b_q, exp_alias_p_operand, outputs, q_resident, s_A, s_LSE, s_O, s_Qr, s_Qt, s_Wv, s_Z, s_state, score_alias_exp, scratch_bytes | a*b_q*b_k + b*b_q + c*b_k + d (+ remainder) for this stage, level=all | measured spill; spill probability |
| envelope[vreg:q_proj] | `pressure.envelope.vreg.q_proj` | `D_n*b_q*s_Qacc → D_n*b_q*s_Qacc` |  | byte | symbolic | D_n, b_q, exp_alias_p_operand, s_Qacc, score_alias_exp | a*b_q*b_k + b*b_q + c*b_k + d (+ remainder) for this stage, level=vreg | measured spill; spill probability |
| envelope[vreg:q_absorb] | `pressure.envelope.vreg.q_absorb` | `b_q*s_Qacc*(D_n + R_k) → b_q*s_Qacc*(D_n + R_k)` |  | byte | symbolic | D_n, R_k, b_q, exp_alias_p_operand, s_Qacc, score_alias_exp | a*b_q*b_k + b*b_q + c*b_k + d (+ remainder) for this stage, level=vreg | measured spill; spill probability |
| envelope[vreg:score] | `pressure.envelope.vreg.score` | `2*b_k*b_q*s_X + b_q*(R_k*Max(s_A, s_Z) + 2*s_state) → 2*b_k*b_q*s_X + b_q*(R_k*Max(s_A, s_Z) + 2*s_state)` |  | byte | symbolic | R_k, b_k, b_q, both_qk_branches_live and D_r > 0, exp_alias_p_operand, s_A, s_X, s_Z, s_state, score_alias_exp | a*b_q*b_k + b*b_q + c*b_k + d (+ remainder) for this stage, level=vreg | measured spill; spill probability |
| envelope[vreg:softmax] | `pressure.envelope.vreg.softmax` | `b_k*b_q*(s_E + s_X) + b_q*(R_k*Max(s_A, s_Z) + 3*s_state) → b_k*b_q*(s_E + s_X) + b_q*(R_k*Max(s_A, s_Z) + 3*s_state)` |  | byte | symbolic | R_k, b_k, b_q, exp_alias_p_operand, s_A, s_E, s_X, s_Z, s_state, score_alias_exp | a*b_q*b_k + b*b_q + c*b_k + d (+ remainder) for this stage, level=vreg | measured spill; spill probability |
| envelope[vreg:pv] | `pressure.envelope.vreg.pv` | `b_k*b_q*(s_E + s_P) + b_q*(R_k*Max(s_A, s_Z) + 3*s_state) → b_k*b_q*(s_E + s_P) + b_q*(R_k*Max(s_A, s_Z) + 3*s_state)` |  | byte | symbolic | R_k, b_k, b_q, exp_alias_p_operand, s_A, s_E, s_P, s_Z, s_state, score_alias_exp | a*b_q*b_k + b*b_q + c*b_k + d (+ remainder) for this stage, level=vreg | measured spill; spill probability |
| envelope[vreg:finalize] | `pressure.envelope.vreg.finalize` | `b_q*(R_k*Max(s_A, s_Z) + 2*s_state) → b_q*(R_k*Max(s_A, s_Z) + 2*s_state)` |  | byte | symbolic | R_k, b_q, exp_alias_p_operand, s_A, s_Z, s_state, score_alias_exp | a*b_q*b_k + b*b_q + c*b_k + d (+ remainder) for this stage, level=vreg | measured spill; spill probability |
| envelope[vmem:q_proj] | `pressure.envelope.vmem.q_proj` | `D_n*R_q*s_Wq + R_q*b_q*s_Cq + s_scratch → D_n*R_q*s_Wq + R_q*b_q*s_Cq + s_scratch` |  | byte | symbolic | D_n, R_q, b_q, b_rq, s_Cq, s_Wq, scratch_bytes | a*b_q*b_k + b*b_q + c*b_k + d (+ remainder) for this stage, level=vmem | measured spill; spill probability |
| envelope[vmem:q_absorb] | `pressure.envelope.vmem.q_absorb` | `D_n*R_k*s_Wk + s_scratch → D_n*R_k*s_Wk + s_scratch` |  | byte | symbolic | D_n, R_k, s_Wk, scratch_bytes | a*b_q*b_k + b*b_q + c*b_k + d (+ remainder) for this stage, level=vmem | measured spill; spill probability |
| envelope[vmem:score] | `pressure.envelope.vmem.score` | `b_k*n_buf*(D_r*s_Kr + R_k*s_Ck) + b_q*(D_r*s_Qr + R_k*s_Qt) + s_scratch → b_k*n_buf*(D_r*s_Kr + R_k*s_Ck) + b_q*(D_r*s_Qr + R_k*s_Qt) + s_scratch` |  | byte | symbolic | D_r, R_k, b_k, b_q, kv_buffers, s_Ck, s_Kr, s_Qr, s_Qt, scratch_bytes | a*b_q*b_k + b*b_q + c*b_k + d (+ remainder) for this stage, level=vmem | measured spill; spill probability |
| envelope[vmem:softmax] | `pressure.envelope.vmem.softmax` | `R_k*b_k*n_buf*s_Ck + b_q*(D_r*s_Qr + R_k*s_Qt) + s_scratch → R_k*b_k*n_buf*s_Ck + b_q*(D_r*s_Qr + R_k*s_Qt) + s_scratch` |  | byte | symbolic | D_r, R_k, b_k, b_q, kv_buffers, q_resident, s_Ck, s_Qr, s_Qt, scratch_bytes | a*b_q*b_k + b*b_q + c*b_k + d (+ remainder) for this stage, level=vmem | measured spill; spill probability |
| envelope[vmem:pv] | `pressure.envelope.vmem.pv` | `R_k*b_k*n_buf*s_Ck + b_q*(D_r*s_Qr + R_k*s_Qt) + s_scratch → R_k*b_k*n_buf*s_Ck + b_q*(D_r*s_Qr + R_k*s_Qt) + s_scratch` |  | byte | symbolic | D_r, R_k, b_k, b_q, kv_buffers, q_resident, s_Ck, s_Qr, s_Qt, scratch_bytes | a*b_q*b_k + b*b_q + c*b_k + d (+ remainder) for this stage, level=vmem | measured spill; spill probability |
| envelope[vmem:finalize] | `pressure.envelope.vmem.finalize` | `D_v*R_k*s_Wv + b_q*(D_r*s_Qr + D_v*s_O + R_k*s_Qt + s_LSE) + s_scratch → D_v*R_k*s_Wv + b_q*(D_r*s_Qr + D_v*s_O + R_k*s_Qt + s_LSE) + s_scratch` |  | byte | symbolic | D_r, D_v, R_k, b_q, outputs, q_resident, s_LSE, s_O, s_Qr, s_Qt, s_Wv, scratch_bytes | a*b_q*b_k + b*b_q + c*b_k + d (+ remainder) for this stage, level=vmem | measured spill; spill probability |
| V_lifetime_stages | `lifetime.v_tile.stages` | `\|{s : v_tile live at s}\|` |  | stages | not_applicable |  | number of stage cuts the V tile is live at | measured prefetch behaviour |
| V_prefetch_overlapped | `lifetime.v_tile.prefetch` | `v_tile live at the score stage` |  | bool | not_applicable |  | V read/production overlapped with the score matmul | measured prefetch behaviour |
| b_k_max[vreg:q_proj] | `pressure.feasible.vreg.q_proj.b_k` | `floor((R - b b_q - d)/(a b_q + c)), stated only for a b_q + c > 0` |  | rows | not_applicable |  | the q_proj stage at level vreg has no b_k term (a b_q + c = 0), so it places no bound on b_k | an unbounded b_k: other stages still constrain it |
| excess_over_budget[vreg:q_proj] | `pressure.capacity_constraint.vreg.q_proj` | `Max(0, D_n*b_q*s_Qacc - R_vreg) → Max(0, D_n*b_q*s_Qacc - R_vreg)` |  | byte | symbolic | D_n, b_q, exp_alias_p_operand, s_Qacc, score_alias_exp, vreg_budget_bytes | max(0, M_live - R) at the q_proj stage, level vreg: data that cannot stay at the level under the source-level model | spill write traffic; compiled live set |
| b_k_max[vreg:q_absorb] | `pressure.feasible.vreg.q_absorb.b_k` | `floor((R - b b_q - d)/(a b_q + c)), stated only for a b_q + c > 0` |  | rows | not_applicable |  | the q_absorb stage at level vreg has no b_k term (a b_q + c = 0), so it places no bound on b_k | an unbounded b_k: other stages still constrain it |
| excess_over_budget[vreg:q_absorb] | `pressure.capacity_constraint.vreg.q_absorb` | `Max(0, -R_vreg + b_q*s_Qacc*(D_n + R_k)) → Max(0, -R_vreg + b_q*s_Qacc*(D_n + R_k))` |  | byte | symbolic | D_n, R_k, b_q, exp_alias_p_operand, s_Qacc, score_alias_exp, vreg_budget_bytes | max(0, M_live - R) at the q_absorb stage, level vreg: data that cannot stay at the level under the source-level model | spill write traffic; compiled live set |
| b_k_max[vreg:score] | `pressure.feasible.vreg.score.b_k` | `floor((R_vreg - b_q*(R_k*Max(s_A, s_Z) + 2*s_state))/(2*b_q*s_X)) → floor((R_vreg - b_q*(R_k*Max(s_A, s_Z) + 2*s_state))/(2*b_q*s_X))` |  | rows | symbolic | R_k, b_q, both_qk_branches_live and D_r > 0, exp_alias_p_operand, s_A, s_X, s_Z, s_state, score_alias_exp, vreg_budget_bytes | floor((R - b b_q - d)/(a b_q + c)) for the score stage at level vreg under the declared budget R_vreg |  |
| excess_over_budget[vreg:score] | `pressure.capacity_constraint.vreg.score` | `Max(0, -R_vreg + 2*b_k*b_q*s_X + b_q*(R_k*Max(s_A, s_Z) + 2*s_state)) → Max(0, -R_vreg + 2*b_k*b_q*s_X + b_q*(R_k*Max(s_A, s_Z) + 2*s_state))` |  | byte | symbolic | R_k, b_k, b_q, both_qk_branches_live and D_r > 0, exp_alias_p_operand, s_A, s_X, s_Z, s_state, score_alias_exp, vreg_budget_bytes | max(0, M_live - R) at the score stage, level vreg: data that cannot stay at the level under the source-level model | spill write traffic; compiled live set |
| b_k_max[vreg:softmax] | `pressure.feasible.vreg.softmax.b_k` | `floor((R_vreg - b_q*(R_k*Max(s_A, s_Z) + 3*s_state))/(b_q*(s_E + s_X))) → floor((R_vreg - b_q*(R_k*Max(s_A, s_Z) + 3*s_state))/(b_q*(s_E + s_X)))` |  | rows | symbolic | R_k, b_q, exp_alias_p_operand, s_A, s_E, s_X, s_Z, s_state, score_alias_exp, vreg_budget_bytes | floor((R - b b_q - d)/(a b_q + c)) for the softmax stage at level vreg under the declared budget R_vreg |  |
| excess_over_budget[vreg:softmax] | `pressure.capacity_constraint.vreg.softmax` | `Max(0, -R_vreg + b_k*b_q*(s_E + s_X) + b_q*(R_k*Max(s_A, s_Z) + 3*s_state)) → Max(0, -R_vreg + b_k*b_q*(s_E + s_X) + b_q*(R_k*Max(s_A, s_Z) + 3*s_state))` |  | byte | symbolic | R_k, b_k, b_q, exp_alias_p_operand, s_A, s_E, s_X, s_Z, s_state, score_alias_exp, vreg_budget_bytes | max(0, M_live - R) at the softmax stage, level vreg: data that cannot stay at the level under the source-level model | spill write traffic; compiled live set |
| b_k_max[vreg:pv] | `pressure.feasible.vreg.pv.b_k` | `floor((R_vreg - b_q*(R_k*Max(s_A, s_Z) + 3*s_state))/(b_q*(s_E + s_P))) → floor((R_vreg - b_q*(R_k*Max(s_A, s_Z) + 3*s_state))/(b_q*(s_E + s_P)))` |  | rows | symbolic | R_k, b_q, exp_alias_p_operand, s_A, s_E, s_P, s_Z, s_state, score_alias_exp, vreg_budget_bytes | floor((R - b b_q - d)/(a b_q + c)) for the pv stage at level vreg under the declared budget R_vreg |  |
| excess_over_budget[vreg:pv] | `pressure.capacity_constraint.vreg.pv` | `Max(0, -R_vreg + b_k*b_q*(s_E + s_P) + b_q*(R_k*Max(s_A, s_Z) + 3*s_state)) → Max(0, -R_vreg + b_k*b_q*(s_E + s_P) + b_q*(R_k*Max(s_A, s_Z) + 3*s_state))` |  | byte | symbolic | R_k, b_k, b_q, exp_alias_p_operand, s_A, s_E, s_P, s_Z, s_state, score_alias_exp, vreg_budget_bytes | max(0, M_live - R) at the pv stage, level vreg: data that cannot stay at the level under the source-level model | spill write traffic; compiled live set |
| b_k_max[vreg:finalize] | `pressure.feasible.vreg.finalize.b_k` | `floor((R - b b_q - d)/(a b_q + c)), stated only for a b_q + c > 0` |  | rows | not_applicable |  | the finalize stage at level vreg has no b_k term (a b_q + c = 0), so it places no bound on b_k | an unbounded b_k: other stages still constrain it |
| excess_over_budget[vreg:finalize] | `pressure.capacity_constraint.vreg.finalize` | `Max(0, -R_vreg + b_q*(R_k*Max(s_A, s_Z) + 2*s_state)) → Max(0, -R_vreg + b_q*(R_k*Max(s_A, s_Z) + 2*s_state))` |  | byte | symbolic | R_k, b_q, exp_alias_p_operand, s_A, s_Z, s_state, score_alias_exp, vreg_budget_bytes | max(0, M_live - R) at the finalize stage, level vreg: data that cannot stay at the level under the source-level model | spill write traffic; compiled live set |
| b_k_max[vreg:binding_stage] | `pressure.feasible.vreg.binding` | `min over stages s of floor((R - b_s b_q - d_s)/(a_s b_q + c_s))` |  | rows | unknown | b_q, both_qk_branches_live and D_r > 0, exp_alias_p_operand, score_alias_exp, vreg_budget_bytes | tightest per-stage capacity bound at level vreg | compiled spill/OOM criterion |
| b_k_max[vmem:q_proj] | `pressure.feasible.vmem.q_proj.b_k` | `floor((R - b b_q - d)/(a b_q + c)), stated only for a b_q + c > 0` |  | rows | not_applicable | b_rq, scratch_bytes | the q_proj stage at level vmem has no b_k term (a b_q + c = 0), so it places no bound on b_k | an unbounded b_k: other stages still constrain it |
| excess_over_budget[vmem:q_proj] | `pressure.capacity_constraint.vmem.q_proj` | `Max(0, D_n*R_q*s_Wq + R_q*b_q*s_Cq - R_vmem + s_scratch) → Max(0, D_n*R_q*s_Wq + R_q*b_q*s_Cq - R_vmem + s_scratch)` |  | byte | symbolic | D_n, R_q, b_q, b_rq, s_Cq, s_Wq, scratch_bytes, vmem_budget_bytes | max(0, M_live - R) at the q_proj stage, level vmem: data that cannot stay at the level under the source-level model | spill write traffic; compiled live set |
| b_k_max[vmem:q_absorb] | `pressure.feasible.vmem.q_absorb.b_k` | `floor((R - b b_q - d)/(a b_q + c)), stated only for a b_q + c > 0` |  | rows | not_applicable | scratch_bytes | the q_absorb stage at level vmem has no b_k term (a b_q + c = 0), so it places no bound on b_k | an unbounded b_k: other stages still constrain it |
| excess_over_budget[vmem:q_absorb] | `pressure.capacity_constraint.vmem.q_absorb` | `Max(0, D_n*R_k*s_Wk - R_vmem + s_scratch) → Max(0, D_n*R_k*s_Wk - R_vmem + s_scratch)` |  | byte | symbolic | D_n, R_k, s_Wk, scratch_bytes, vmem_budget_bytes | max(0, M_live - R) at the q_absorb stage, level vmem: data that cannot stay at the level under the source-level model | spill write traffic; compiled live set |
| b_k_max[vmem:score] | `pressure.feasible.vmem.score.b_k` | `floor((R_vmem - b_q*(D_r*s_Qr + R_k*s_Qt) - s_scratch)/(n_buf*(D_r*s_Kr + R_k*s_Ck))) → floor((R_vmem - b_q*(D_r*s_Qr + R_k*s_Qt) - s_scratch)/(n_buf*(D_r*s_Kr + R_k*s_Ck)))` |  | rows | symbolic | D_r, R_k, b_q, kv_buffers, s_Ck, s_Kr, s_Qr, s_Qt, scratch_bytes, vmem_budget_bytes | floor((R - b b_q - d)/(a b_q + c)) for the score stage at level vmem under the declared budget R_vmem |  |
| excess_over_budget[vmem:score] | `pressure.capacity_constraint.vmem.score` | `Max(0, -R_vmem + b_k*n_buf*(D_r*s_Kr + R_k*s_Ck) + b_q*(D_r*s_Qr + R_k*s_Qt) + s_scratch) → Max(0, -R_vmem + b_k*n_buf*(D_r*s_Kr + R_k*s_Ck) + b_q*(D_r*s_Qr + R_k*s_Qt) + s_scratch)` |  | byte | symbolic | D_r, R_k, b_k, b_q, kv_buffers, s_Ck, s_Kr, s_Qr, s_Qt, scratch_bytes, vmem_budget_bytes | max(0, M_live - R) at the score stage, level vmem: data that cannot stay at the level under the source-level model | spill write traffic; compiled live set |
| b_k_max[vmem:softmax] | `pressure.feasible.vmem.softmax.b_k` | `floor((R_vmem - b_q*(D_r*s_Qr + R_k*s_Qt) - s_scratch)/(R_k*n_buf*s_Ck)) → floor((R_vmem - b_q*(D_r*s_Qr + R_k*s_Qt) - s_scratch)/(R_k*n_buf*s_Ck))` |  | rows | symbolic | D_r, R_k, b_q, kv_buffers, q_resident, s_Ck, s_Qr, s_Qt, scratch_bytes, vmem_budget_bytes | floor((R - b b_q - d)/(a b_q + c)) for the softmax stage at level vmem under the declared budget R_vmem |  |
| excess_over_budget[vmem:softmax] | `pressure.capacity_constraint.vmem.softmax` | `Max(0, R_k*b_k*n_buf*s_Ck - R_vmem + b_q*(D_r*s_Qr + R_k*s_Qt) + s_scratch) → Max(0, R_k*b_k*n_buf*s_Ck - R_vmem + b_q*(D_r*s_Qr + R_k*s_Qt) + s_scratch)` |  | byte | symbolic | D_r, R_k, b_k, b_q, kv_buffers, q_resident, s_Ck, s_Qr, s_Qt, scratch_bytes, vmem_budget_bytes | max(0, M_live - R) at the softmax stage, level vmem: data that cannot stay at the level under the source-level model | spill write traffic; compiled live set |
| b_k_max[vmem:pv] | `pressure.feasible.vmem.pv.b_k` | `floor((R_vmem - b_q*(D_r*s_Qr + R_k*s_Qt) - s_scratch)/(R_k*n_buf*s_Ck)) → floor((R_vmem - b_q*(D_r*s_Qr + R_k*s_Qt) - s_scratch)/(R_k*n_buf*s_Ck))` |  | rows | symbolic | D_r, R_k, b_q, kv_buffers, q_resident, s_Ck, s_Qr, s_Qt, scratch_bytes, vmem_budget_bytes | floor((R - b b_q - d)/(a b_q + c)) for the pv stage at level vmem under the declared budget R_vmem |  |
| excess_over_budget[vmem:pv] | `pressure.capacity_constraint.vmem.pv` | `Max(0, R_k*b_k*n_buf*s_Ck - R_vmem + b_q*(D_r*s_Qr + R_k*s_Qt) + s_scratch) → Max(0, R_k*b_k*n_buf*s_Ck - R_vmem + b_q*(D_r*s_Qr + R_k*s_Qt) + s_scratch)` |  | byte | symbolic | D_r, R_k, b_k, b_q, kv_buffers, q_resident, s_Ck, s_Qr, s_Qt, scratch_bytes, vmem_budget_bytes | max(0, M_live - R) at the pv stage, level vmem: data that cannot stay at the level under the source-level model | spill write traffic; compiled live set |
| b_k_max[vmem:finalize] | `pressure.feasible.vmem.finalize.b_k` | `floor((R - b b_q - d)/(a b_q + c)), stated only for a b_q + c > 0` |  | rows | not_applicable | q_resident, scratch_bytes, outputs | the finalize stage at level vmem has no b_k term (a b_q + c = 0), so it places no bound on b_k | an unbounded b_k: other stages still constrain it |
| excess_over_budget[vmem:finalize] | `pressure.capacity_constraint.vmem.finalize` | `Max(0, D_v*R_k*s_Wv - R_vmem + b_q*(D_r*s_Qr + D_v*s_O + R_k*s_Qt + s_LSE) + s_scratch) → Max(0, D_v*R_k*s_Wv - R_vmem + b_q*(D_r*s_Qr + D_v*s_O + R_k*s_Qt + s_LSE) + s_scratch)` |  | byte | symbolic | D_r, D_v, R_k, b_q, outputs, q_resident, s_LSE, s_O, s_Qr, s_Qt, s_Wv, scratch_bytes, vmem_budget_bytes | max(0, M_live - R) at the finalize stage, level vmem: data that cannot stay at the level under the source-level model | spill write traffic; compiled live set |
| b_k_max[vmem:binding_stage] | `pressure.feasible.vmem.binding` | `min over stages s of floor((R - b_s b_q - d_s)/(a_s b_q + c_s))` |  | rows | unknown | b_q, b_rq, outputs, q_resident, scratch_bytes, vmem_budget_bytes | tightest per-stage capacity bound at level vmem | compiled spill/OOM criterion |
| N_spill_instructions | `spill.N_spill_instructions` | `\|{spill/fill instructions in the compiled program}\|` |  | instructions | unknown | compile_evidence | static count of spill and fill instructions (not their dynamic execution count, not bytes) | each other; pressure envelope |
| M_spill_peak | `spill.M_spill_peak` | `max_t sum_{a in A_spill(t)} M_a` |  | byte | unknown | compile_or_measurement_evidence | peak spill backing allocation | each other; pressure envelope |
| B_spill_fill | `spill.B_spill_fill` | `sum_r bytes(r) * executions(r)` |  | byte | unknown | compile_or_measurement_evidence | dynamic spill/fill traffic | each other; pressure envelope |
| dT_spill | `spill.dT_spill` | `T_schedule(G_with_transfers) - T_schedule(G_comparison)` |  | s | unknown | compile_or_measurement_evidence, declared comparison graph | exposed latency vs a legal comparison graph | each other; pressure envelope |
| B_extra_opt | `spill.B_extra_opt` | `min_{p in Omega(theta, a, eta)} B_extra(p)` |  | byte | unknown | graph, machine_model, allowed_transformations, objective | minimum extra movement over legal schedules/tilings/recomputations (spec 8.3); solvable only for a declared small graph, machine model and objective | each other; pressure envelope |
| T_opt | `spill.T_opt` | `min_{p in Omega(theta, a, eta)} T(p)` |  | s | unknown | graph, machine_model, allowed_transformations, schedule_model | minimum time over the legal set (spec 8.3, speed objective); needs a solved schedule, not a bound | each other; pressure envelope |
| B_extra_opt_under_time_budget | `spill.B_extra_opt_under_time_budget` | `min_{p in Omega} B_extra(p) s.t. T(p) <= tau` |  | byte | unknown | graph, machine_model, allowed_transformations, schedule_model, time_budget_tau | minimum extra movement subject to a declared time budget tau (spec 8.3) | each other; pressure envelope |

Assumptions:
- `M_live[all:q_proj]`: objects: q_latent_window, wq_rank_tile, q_proj_acc, fixed_scratch
- `M_live[all:q_proj]`: includes objects whose existence depends on undeclared fields: ['b_rq', 'scratch_bytes']
- `M_live[all:q_absorb]`: objects: q_proj_acc, wk_full, q_tilde_acc, fixed_scratch
- `M_live[all:q_absorb]`: includes objects whose existence depends on undeclared fields: ['scratch_bytes']
- `M_live[all:score]`: objects: q_tilde, kv_latent_tile, score_X, row_state_m, row_state_l, score_rope_branch, q_pe_tile, k_pe_tile, fixed_scratch, alias[acc_or_Z]:accumulator_A (sized by the whole group: z_tile|accumulator_A)
- `M_live[all:score]`: includes objects whose existence depends on undeclared fields: ['both_qk_branches_live and D_r > 0', 'scratch_bytes']
- `M_live[all:softmax]`: objects: q_tilde, kv_latent_tile, score_X, exp_E, row_state_m, row_state_l, row_alpha, q_pe_tile, fixed_scratch, alias[acc_or_Z]:accumulator_A (sized by the whole group: z_tile|accumulator_A)
- `M_live[all:softmax]`: includes objects whose existence depends on undeclared fields: ['q_resident', 'score_alias_exp', 'scratch_bytes']
- `M_live[all:pv]`: objects: q_tilde, kv_latent_tile, exp_E, p_operand, row_state_m, row_state_l, row_alpha, q_pe_tile, fixed_scratch, alias[acc_or_Z]:accumulator_A (sized by the whole group: z_tile|accumulator_A)
- `M_live[all:pv]`: includes objects whose existence depends on undeclared fields: ['q_resident', 'score_alias_exp', 'exp_alias_p_operand', 'scratch_bytes']
- `M_live[all:finalize]`: objects: q_tilde, wv_full, row_state_m, row_state_l, q_pe_tile, fixed_scratch, out_tile, lse_tile, alias[acc_or_Z]:z_tile|accumulator_A
- `M_live[all:finalize]`: includes objects whose existence depends on undeclared fields: ['q_resident', 'scratch_bytes', 'outputs']
- `M_live[vreg:q_proj]`: objects: q_proj_acc
- `M_live[vreg:q_absorb]`: objects: q_proj_acc, q_tilde_acc
- `M_live[vreg:score]`: objects: score_X, row_state_m, row_state_l, score_rope_branch, alias[acc_or_Z]:accumulator_A (sized by the whole group: z_tile|accumulator_A)
- `M_live[vreg:score]`: includes objects whose existence depends on undeclared fields: ['both_qk_branches_live and D_r > 0']
- `M_live[vreg:softmax]`: objects: score_X, exp_E, row_state_m, row_state_l, row_alpha, alias[acc_or_Z]:accumulator_A (sized by the whole group: z_tile|accumulator_A)
- `M_live[vreg:softmax]`: includes objects whose existence depends on undeclared fields: ['score_alias_exp']
- `M_live[vreg:pv]`: objects: exp_E, p_operand, row_state_m, row_state_l, row_alpha, alias[acc_or_Z]:accumulator_A (sized by the whole group: z_tile|accumulator_A)
- `M_live[vreg:pv]`: includes objects whose existence depends on undeclared fields: ['score_alias_exp', 'exp_alias_p_operand']
- `M_live[vreg:finalize]`: objects: row_state_m, row_state_l, alias[acc_or_Z]:z_tile|accumulator_A
- `M_live[vmem:q_proj]`: objects: q_latent_window, wq_rank_tile, fixed_scratch
- `M_live[vmem:q_proj]`: includes objects whose existence depends on undeclared fields: ['b_rq', 'scratch_bytes']
- `M_live[vmem:q_absorb]`: objects: wk_full, fixed_scratch
- `M_live[vmem:q_absorb]`: includes objects whose existence depends on undeclared fields: ['scratch_bytes']
- `M_live[vmem:score]`: objects: q_tilde, kv_latent_tile, q_pe_tile, k_pe_tile, fixed_scratch
- `M_live[vmem:score]`: includes objects whose existence depends on undeclared fields: ['scratch_bytes']
- `M_live[vmem:softmax]`: objects: q_tilde, kv_latent_tile, q_pe_tile, fixed_scratch
- `M_live[vmem:softmax]`: includes objects whose existence depends on undeclared fields: ['q_resident', 'scratch_bytes']
- `M_live[vmem:pv]`: objects: q_tilde, kv_latent_tile, q_pe_tile, fixed_scratch
- `M_live[vmem:pv]`: includes objects whose existence depends on undeclared fields: ['q_resident', 'scratch_bytes']
- `M_live[vmem:finalize]`: objects: q_tilde, wv_full, q_pe_tile, fixed_scratch, out_tile, lse_tile
- `M_live[vmem:finalize]`: includes objects whose existence depends on undeclared fields: ['q_resident', 'scratch_bytes', 'outputs']
- `envelope[all:q_proj]`: a from []
- `envelope[all:q_proj]`: b from ['q_latent_window', 'q_proj_acc']
- `envelope[all:q_proj]`: c from []
- `envelope[all:q_proj]`: d from ['wq_rank_tile', 'fixed_scratch']
- `envelope[all:q_absorb]`: a from []
- `envelope[all:q_absorb]`: b from ['q_proj_acc', 'q_tilde_acc']
- `envelope[all:q_absorb]`: c from []
- `envelope[all:q_absorb]`: d from ['wk_full', 'fixed_scratch']
- `envelope[all:score]`: a from ['score_X', 'score_rope_branch']
- `envelope[all:score]`: b from ['q_tilde', 'row_state_m', 'row_state_l', 'q_pe_tile', 'alias[acc_or_Z]:z_tile|accumulator_A']
- `envelope[all:score]`: c from ['kv_latent_tile', 'k_pe_tile']
- `envelope[all:score]`: d from ['fixed_scratch']
- `envelope[all:softmax]`: a from ['score_X', 'exp_E']
- `envelope[all:softmax]`: b from ['q_tilde', 'row_state_m', 'row_state_l', 'row_alpha', 'q_pe_tile', 'alias[acc_or_Z]:z_tile|accumulator_A']
- `envelope[all:softmax]`: c from ['kv_latent_tile']
- `envelope[all:softmax]`: d from ['fixed_scratch']
- `envelope[all:pv]`: a from ['exp_E', 'p_operand']
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
- `envelope[vreg:score]`: a from ['score_X', 'score_rope_branch']
- `envelope[vreg:score]`: b from ['row_state_m', 'row_state_l', 'alias[acc_or_Z]:z_tile|accumulator_A']
- `envelope[vreg:score]`: c from []
- `envelope[vreg:score]`: d from []
- `envelope[vreg:softmax]`: a from ['score_X', 'exp_E']
- `envelope[vreg:softmax]`: b from ['row_state_m', 'row_state_l', 'row_alpha', 'alias[acc_or_Z]:z_tile|accumulator_A']
- `envelope[vreg:softmax]`: c from []
- `envelope[vreg:softmax]`: d from []
- `envelope[vreg:pv]`: a from ['exp_E', 'p_operand']
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
- `b_k_max[vmem:q_proj]`: the absence of a b_k term rests on an object list shaped by ['b_rq', 'scratch_bytes']
- `excess_over_budget[vmem:q_proj]`: source-level logical live set; layout coverage not applied; not a compiled graph cut (spec 8.2 caveat)
- `b_k_max[vmem:q_absorb]`: this stage's live set does not grow with b_k; the binding bound comes from another stage
- `b_k_max[vmem:q_absorb]`: the absence of a b_k term rests on an object list shaped by ['scratch_bytes']
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
- `b_k_max[vmem:finalize]`: the absence of a b_k term rests on an object list shaped by ['q_resident', 'scratch_bytes', 'outputs']
- `excess_over_budget[vmem:finalize]`: source-level logical live set; layout coverage not applied; not a compiled graph cut (spec 8.2 caveat)

<details><summary>Provenance (bindings, coverage, source, evidence)</summary>

| metric | source | coverage | bindings | evidence |
|---|---|---|---|---|
| M_live[all:q_proj] | derived | incomplete: depends on undeclared/unbound D_n, R_q, b_q, b_rq, exp_alias_p_operand, s_Cq, s_Qacc, s_Wq, score_alias_exp, scratch_bytes | `{}` |  |
| M_live[all:q_absorb] | derived | incomplete: depends on undeclared/unbound D_n, R_k, b_q, exp_alias_p_operand, s_Qacc, s_Wk, score_alias_exp, scratch_bytes | `{}` |  |
| M_live[all:score] | derived | incomplete: depends on undeclared/unbound D_r, R_k, b_k, b_q, both_qk_branches_live and D_r > 0, exp_alias_p_operand, kv_buffers, s_A, s_Ck, s_Kr, s_Qr, s_Qt, s_X, s_Z, s_state, score_alias_exp, scratch_bytes | `{}` |  |
| M_live[all:softmax] | derived | incomplete: depends on undeclared/unbound D_r, R_k, b_k, b_q, exp_alias_p_operand, kv_buffers, q_resident, s_A, s_Ck, s_E, s_Qr, s_Qt, s_X, s_Z, s_state, score_alias_exp, scratch_bytes | `{}` |  |
| M_live[all:pv] | derived | incomplete: depends on undeclared/unbound D_r, R_k, b_k, b_q, exp_alias_p_operand, kv_buffers, q_resident, s_A, s_Ck, s_E, s_P, s_Qr, s_Qt, s_Z, s_state, score_alias_exp, scratch_bytes | `{}` |  |
| M_live[all:finalize] | derived | incomplete: depends on undeclared/unbound D_r, D_v, R_k, b_q, exp_alias_p_operand, outputs, q_resident, s_A, s_LSE, s_O, s_Qr, s_Qt, s_Wv, s_Z, s_state, score_alias_exp, scratch_bytes | `{}` |  |
| M_peak[all] | derived | incomplete: depends on undeclared/unbound D_n, D_r, D_v, R_k, R_q, b_k, b_q, b_rq, both_qk_branches_live and D_r > 0, exp_alias_p_operand, kv_buffers, outputs, q_resident, s_A, s_Ck, s_Cq, s_E, s_Kr, s_LSE, s_O, s_P, s_Qacc, s_Qr, s_Qt, s_Wk, s_Wq, s_Wv, s_X, s_Z, s_state, score_alias_exp, scratch_bytes | `{}` |  |
| M_live[vreg:q_proj] | derived | incomplete: depends on undeclared/unbound D_n, b_q, exp_alias_p_operand, s_Qacc, score_alias_exp | `{}` |  |
| M_live[vreg:q_absorb] | derived | incomplete: depends on undeclared/unbound D_n, R_k, b_q, exp_alias_p_operand, s_Qacc, score_alias_exp | `{}` |  |
| M_live[vreg:score] | derived | incomplete: depends on undeclared/unbound R_k, b_k, b_q, both_qk_branches_live and D_r > 0, exp_alias_p_operand, s_A, s_X, s_Z, s_state, score_alias_exp | `{}` |  |
| M_live[vreg:softmax] | derived | incomplete: depends on undeclared/unbound R_k, b_k, b_q, exp_alias_p_operand, s_A, s_E, s_X, s_Z, s_state, score_alias_exp | `{}` |  |
| M_live[vreg:pv] | derived | incomplete: depends on undeclared/unbound R_k, b_k, b_q, exp_alias_p_operand, s_A, s_E, s_P, s_Z, s_state, score_alias_exp | `{}` |  |
| M_live[vreg:finalize] | derived | incomplete: depends on undeclared/unbound R_k, b_q, exp_alias_p_operand, s_A, s_Z, s_state, score_alias_exp | `{}` |  |
| M_peak[vreg] | derived | incomplete: depends on undeclared/unbound D_n, R_k, b_k, b_q, both_qk_branches_live and D_r > 0, exp_alias_p_operand, s_A, s_E, s_P, s_Qacc, s_X, s_Z, s_state, score_alias_exp | `{}` |  |
| M_live[vmem:q_proj] | derived | incomplete: depends on undeclared/unbound D_n, R_q, b_q, b_rq, s_Cq, s_Wq, scratch_bytes | `{}` |  |
| M_live[vmem:q_absorb] | derived | incomplete: depends on undeclared/unbound D_n, R_k, s_Wk, scratch_bytes | `{}` |  |
| M_live[vmem:score] | derived | incomplete: depends on undeclared/unbound D_r, R_k, b_k, b_q, kv_buffers, s_Ck, s_Kr, s_Qr, s_Qt, scratch_bytes | `{}` |  |
| M_live[vmem:softmax] | derived | incomplete: depends on undeclared/unbound D_r, R_k, b_k, b_q, kv_buffers, q_resident, s_Ck, s_Qr, s_Qt, scratch_bytes | `{}` |  |
| M_live[vmem:pv] | derived | incomplete: depends on undeclared/unbound D_r, R_k, b_k, b_q, kv_buffers, q_resident, s_Ck, s_Qr, s_Qt, scratch_bytes | `{}` |  |
| M_live[vmem:finalize] | derived | incomplete: depends on undeclared/unbound D_r, D_v, R_k, b_q, outputs, q_resident, s_LSE, s_O, s_Qr, s_Qt, s_Wv, scratch_bytes | `{}` |  |
| M_peak[vmem] | derived | incomplete: depends on undeclared/unbound D_n, D_r, D_v, R_k, R_q, b_k, b_q, b_rq, kv_buffers, outputs, q_resident, s_Ck, s_Cq, s_Kr, s_LSE, s_O, s_Qr, s_Qt, s_Wk, s_Wq, s_Wv, scratch_bytes | `{}` |  |
| envelope[all:q_proj] | derived | incomplete: depends on undeclared/unbound D_n, R_q, b_q, b_rq, exp_alias_p_operand, s_Cq, s_Qacc, s_Wq, score_alias_exp, scratch_bytes | `{}` |  |
| envelope[all:q_absorb] | derived | incomplete: depends on undeclared/unbound D_n, R_k, b_q, exp_alias_p_operand, s_Qacc, s_Wk, score_alias_exp, scratch_bytes | `{}` |  |
| envelope[all:score] | derived | incomplete: depends on undeclared/unbound D_r, R_k, b_k, b_q, both_qk_branches_live and D_r > 0, exp_alias_p_operand, kv_buffers, s_A, s_Ck, s_Kr, s_Qr, s_Qt, s_X, s_Z, s_state, score_alias_exp, scratch_bytes | `{}` |  |
| envelope[all:softmax] | derived | incomplete: depends on undeclared/unbound D_r, R_k, b_k, b_q, exp_alias_p_operand, kv_buffers, q_resident, s_A, s_Ck, s_E, s_Qr, s_Qt, s_X, s_Z, s_state, score_alias_exp, scratch_bytes | `{}` |  |
| envelope[all:pv] | derived | incomplete: depends on undeclared/unbound D_r, R_k, b_k, b_q, exp_alias_p_operand, kv_buffers, q_resident, s_A, s_Ck, s_E, s_P, s_Qr, s_Qt, s_Z, s_state, score_alias_exp, scratch_bytes | `{}` |  |
| envelope[all:finalize] | derived | incomplete: depends on undeclared/unbound D_r, D_v, R_k, b_q, exp_alias_p_operand, outputs, q_resident, s_A, s_LSE, s_O, s_Qr, s_Qt, s_Wv, s_Z, s_state, score_alias_exp, scratch_bytes | `{}` |  |
| envelope[vreg:q_proj] | derived | incomplete: depends on undeclared/unbound D_n, b_q, exp_alias_p_operand, s_Qacc, score_alias_exp | `{}` |  |
| envelope[vreg:q_absorb] | derived | incomplete: depends on undeclared/unbound D_n, R_k, b_q, exp_alias_p_operand, s_Qacc, score_alias_exp | `{}` |  |
| envelope[vreg:score] | derived | incomplete: depends on undeclared/unbound R_k, b_k, b_q, both_qk_branches_live and D_r > 0, exp_alias_p_operand, s_A, s_X, s_Z, s_state, score_alias_exp | `{}` |  |
| envelope[vreg:softmax] | derived | incomplete: depends on undeclared/unbound R_k, b_k, b_q, exp_alias_p_operand, s_A, s_E, s_X, s_Z, s_state, score_alias_exp | `{}` |  |
| envelope[vreg:pv] | derived | incomplete: depends on undeclared/unbound R_k, b_k, b_q, exp_alias_p_operand, s_A, s_E, s_P, s_Z, s_state, score_alias_exp | `{}` |  |
| envelope[vreg:finalize] | derived | incomplete: depends on undeclared/unbound R_k, b_q, exp_alias_p_operand, s_A, s_Z, s_state, score_alias_exp | `{}` |  |
| envelope[vmem:q_proj] | derived | incomplete: depends on undeclared/unbound D_n, R_q, b_q, b_rq, s_Cq, s_Wq, scratch_bytes | `{}` |  |
| envelope[vmem:q_absorb] | derived | incomplete: depends on undeclared/unbound D_n, R_k, s_Wk, scratch_bytes | `{}` |  |
| envelope[vmem:score] | derived | incomplete: depends on undeclared/unbound D_r, R_k, b_k, b_q, kv_buffers, s_Ck, s_Kr, s_Qr, s_Qt, scratch_bytes | `{}` |  |
| envelope[vmem:softmax] | derived | incomplete: depends on undeclared/unbound D_r, R_k, b_k, b_q, kv_buffers, q_resident, s_Ck, s_Qr, s_Qt, scratch_bytes | `{}` |  |
| envelope[vmem:pv] | derived | incomplete: depends on undeclared/unbound D_r, R_k, b_k, b_q, kv_buffers, q_resident, s_Ck, s_Qr, s_Qt, scratch_bytes | `{}` |  |
| envelope[vmem:finalize] | derived | incomplete: depends on undeclared/unbound D_r, D_v, R_k, b_q, outputs, q_resident, s_LSE, s_O, s_Qr, s_Qt, s_Wv, scratch_bytes | `{}` |  |
| V_lifetime_stages | derived | not applicable under the declared configuration | `{}` |  |
| V_prefetch_overlapped | derived | not applicable under the declared configuration | `{}` |  |
| b_k_max[vreg:q_proj] | derived | not applicable under the declared configuration | `{}` |  |
| excess_over_budget[vreg:q_proj] | derived | incomplete: depends on undeclared/unbound D_n, b_q, exp_alias_p_operand, s_Qacc, score_alias_exp, vreg_budget_bytes | `{}` |  |
| b_k_max[vreg:q_absorb] | derived | not applicable under the declared configuration | `{}` |  |
| excess_over_budget[vreg:q_absorb] | derived | incomplete: depends on undeclared/unbound D_n, R_k, b_q, exp_alias_p_operand, s_Qacc, score_alias_exp, vreg_budget_bytes | `{}` |  |
| b_k_max[vreg:score] | derived | incomplete: depends on undeclared/unbound R_k, b_q, both_qk_branches_live and D_r > 0, exp_alias_p_operand, s_A, s_X, s_Z, s_state, score_alias_exp, vreg_budget_bytes | `{}` |  |
| excess_over_budget[vreg:score] | derived | incomplete: depends on undeclared/unbound R_k, b_k, b_q, both_qk_branches_live and D_r > 0, exp_alias_p_operand, s_A, s_X, s_Z, s_state, score_alias_exp, vreg_budget_bytes | `{}` |  |
| b_k_max[vreg:softmax] | derived | incomplete: depends on undeclared/unbound R_k, b_q, exp_alias_p_operand, s_A, s_E, s_X, s_Z, s_state, score_alias_exp, vreg_budget_bytes | `{}` |  |
| excess_over_budget[vreg:softmax] | derived | incomplete: depends on undeclared/unbound R_k, b_k, b_q, exp_alias_p_operand, s_A, s_E, s_X, s_Z, s_state, score_alias_exp, vreg_budget_bytes | `{}` |  |
| b_k_max[vreg:pv] | derived | incomplete: depends on undeclared/unbound R_k, b_q, exp_alias_p_operand, s_A, s_E, s_P, s_Z, s_state, score_alias_exp, vreg_budget_bytes | `{}` |  |
| excess_over_budget[vreg:pv] | derived | incomplete: depends on undeclared/unbound R_k, b_k, b_q, exp_alias_p_operand, s_A, s_E, s_P, s_Z, s_state, score_alias_exp, vreg_budget_bytes | `{}` |  |
| b_k_max[vreg:finalize] | derived | not applicable under the declared configuration | `{}` |  |
| excess_over_budget[vreg:finalize] | derived | incomplete: depends on undeclared/unbound R_k, b_q, exp_alias_p_operand, s_A, s_Z, s_state, score_alias_exp, vreg_budget_bytes | `{}` |  |
| b_k_max[vreg:binding_stage] | derived | incomplete: depends on undeclared/unbound b_q, both_qk_branches_live and D_r > 0, exp_alias_p_operand, score_alias_exp, vreg_budget_bytes | `{}` |  |
| b_k_max[vmem:q_proj] | derived | not applicable under the declared configuration | `{}` |  |
| excess_over_budget[vmem:q_proj] | derived | incomplete: depends on undeclared/unbound D_n, R_q, b_q, b_rq, s_Cq, s_Wq, scratch_bytes, vmem_budget_bytes | `{}` |  |
| b_k_max[vmem:q_absorb] | derived | not applicable under the declared configuration | `{}` |  |
| excess_over_budget[vmem:q_absorb] | derived | incomplete: depends on undeclared/unbound D_n, R_k, s_Wk, scratch_bytes, vmem_budget_bytes | `{}` |  |
| b_k_max[vmem:score] | derived | incomplete: depends on undeclared/unbound D_r, R_k, b_q, kv_buffers, s_Ck, s_Kr, s_Qr, s_Qt, scratch_bytes, vmem_budget_bytes | `{}` |  |
| excess_over_budget[vmem:score] | derived | incomplete: depends on undeclared/unbound D_r, R_k, b_k, b_q, kv_buffers, s_Ck, s_Kr, s_Qr, s_Qt, scratch_bytes, vmem_budget_bytes | `{}` |  |
| b_k_max[vmem:softmax] | derived | incomplete: depends on undeclared/unbound D_r, R_k, b_q, kv_buffers, q_resident, s_Ck, s_Qr, s_Qt, scratch_bytes, vmem_budget_bytes | `{}` |  |
| excess_over_budget[vmem:softmax] | derived | incomplete: depends on undeclared/unbound D_r, R_k, b_k, b_q, kv_buffers, q_resident, s_Ck, s_Qr, s_Qt, scratch_bytes, vmem_budget_bytes | `{}` |  |
| b_k_max[vmem:pv] | derived | incomplete: depends on undeclared/unbound D_r, R_k, b_q, kv_buffers, q_resident, s_Ck, s_Qr, s_Qt, scratch_bytes, vmem_budget_bytes | `{}` |  |
| excess_over_budget[vmem:pv] | derived | incomplete: depends on undeclared/unbound D_r, R_k, b_k, b_q, kv_buffers, q_resident, s_Ck, s_Qr, s_Qt, scratch_bytes, vmem_budget_bytes | `{}` |  |
| b_k_max[vmem:finalize] | derived | not applicable under the declared configuration | `{}` |  |
| excess_over_budget[vmem:finalize] | derived | incomplete: depends on undeclared/unbound D_r, D_v, R_k, b_q, outputs, q_resident, s_LSE, s_O, s_Qr, s_Qt, s_Wv, scratch_bytes, vmem_budget_bytes | `{}` |  |
| b_k_max[vmem:binding_stage] | derived | incomplete: depends on undeclared/unbound b_q, b_rq, outputs, q_resident, scratch_bytes, vmem_budget_bytes | `{}` |  |
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
| d(score_bytes)/d(b_q) | `sens.score_bytes.b_q` | `b_k*s_X → b_k*s_X` |  | byte per row | symbolic | b_k, s_X | continuous relaxation of the source-level logical size | a discrete tile change |
| d(score_bytes)/d(b_k) | `sens.score_bytes.b_k` | `b_q*s_X → b_q*s_X` |  | byte per row | symbolic | b_q, s_X | continuous relaxation of the source-level logical size | a discrete tile change |
| d(accumulator_bytes)/d(b_q) | `sens.accumulator_bytes.b_q` | `R_k*s_A → R_k*s_A` |  | byte per row | symbolic | R_k, s_A | continuous relaxation of the source-level logical size | a discrete tile change |
| d(accumulator_bytes)/d(b_k) | `sens.accumulator_bytes.b_k` | `0` | 0 | byte per row | bound |  | continuous relaxation of the source-level logical size | an absence of any effect of b_k; a discrete tile change |
| n_programs | `sched.programs` | `B*ceiling(H/h_pp)*ceiling(S_q/b_q) → B*ceiling(H/h_pp)*ceiling(S_q/b_q)` |  | programs | symbolic | B, H, S_q, b_q, heads_per_program, schedule | (batch, head-group, q-block) programs under the declared loop nest |  |
| n_kv_block_visits | `sched.kv_visits` | `N_kvblocks → N_kvblocks` |  | visits | symbolic | N_kvblocks, mask, rect_policy, schedule | selected (b, h, q-block, kv-block) rectangles |  |

Assumptions:
- `d(accumulator_bytes)/d(b_k)`: a zero derivative is the absence of a *direct* dependence in the source-level logical size; scheduling and buffer lifetime can still create an indirect effect (spec 10.1)

<details><summary>Provenance (bindings, coverage, source, evidence)</summary>

| metric | source | coverage | bindings | evidence |
|---|---|---|---|---|
| d(score_bytes)/d(b_q) | derived | incomplete: depends on undeclared/unbound b_k, s_X | `{}` |  |
| d(score_bytes)/d(b_k) | derived | incomplete: depends on undeclared/unbound b_q, s_X | `{}` |  |
| d(accumulator_bytes)/d(b_q) | derived | incomplete: depends on undeclared/unbound R_k, s_A | `{}` |  |
| d(accumulator_bytes)/d(b_k) | derived | complete w.r.t. the declared task/strategy/numerics scenario and this formula | `{}` |  |
| n_programs | derived | incomplete: depends on undeclared/unbound B, H, S_q, b_q, heads_per_program, schedule | `{}` |  |
| n_kv_block_visits | derived | incomplete: depends on undeclared/unbound N_kvblocks, mask, rect_policy, schedule | `{}` |  |

</details>

## 9. hardware lower bounds and calibration

| metric | formula id | expression | value | unit | status | missing | scope | not equivalent to |
|---|---|---|---|---|---|---|---|---|
| T_LB[mxu] | `perf.lb.mxu` | `(2*B*D_n*H*R_k*S_q + 2*B*D_n*H*R_q*S_q + 2*B*D_v*H*R_k*S_q + 2*C_pad*D_r + 4*C_pad*R_k)/P_mxu` |  | s | unknown | B, C_pad, D_n, D_r, D_v, H, P_mxu, R_k, R_q, S_q, executed_extent_policy, mask, matmul_input_dtype, projection_scope, rect_policy, schedule | scope=None | actual latency; calibrated prediction |
| T_LB[vpu] | `perf.lb.vpu` | `(B*D_v*H*S_q + B*H*R_k*S_q + 2*B*H*S_q + C_masked + 2*C_pad + 2*N_rowvisits*R_k + 2*N_rowvisits + Piecewise((C_pad, D_r > 0), (0, True)))/P_vpu` |  | s | unknown | B, C_masked, C_pad, D_r, D_v, H, N_rowvisits, P_vpu, R_k, S_q, executed_extent_policy, exp_base, mask, outputs, projection_scope, rect_policy, recurrence, schedule | scope=None | actual latency; calibrated prediction |
| T_LB[reduce] | `perf.lb.reduce` | `(2*C_pad - N_rowvisits)/P_red` |  | s | unknown | C_pad, N_rowvisits, P_red, executed_extent_policy, exp_base, mask, outputs, projection_scope, rect_policy, recurrence, schedule | scope=None | actual latency; calibrated prediction |
| T_LB[layout] | `perf.lb.layout` | `(C_pad + H*N_keyvisits*R_k)/P_layout` |  | s | unknown | C_pad, H, N_keyvisits, P_layout, R_k, executed_extent_policy, exp_base, mask, outputs, projection_scope, rect_policy, recurrence, schedule | scope=None | actual latency; calibrated prediction |
| T_LB[exp] | `perf.lb.exp` | `(C_pad + N_rowvisits)/P_exp` |  | s | unknown | C_pad, N_rowvisits, P_exp, executed_extent_policy, exp_base, mask, outputs, projection_scope, rect_policy, recurrence, schedule | scope=None | actual latency; calibrated prediction |
| T_LB[path:hbm_to_vmem] | `perf.lb.path:hbm_to_vmem` | `(B*D_n*H*R_k*s_Wk*ceiling(S_q/b_q) + B*D_n*H*R_q*s_Wq*ceiling(S_q/b_q) + B*D_r*H*S_q*s_Qr + B*D_v*H*R_k*s_Wv*ceiling(S_q/b_q) + B*H*R_k*S_q*s_Qt + B*H*R_k*S_q*s_Z + B*R_q*S_q*s_Cq*ceiling(H/h_pp) + D_r*N_keyvisits*s_Kr*ceiling(H/h_pp) + N_keyvisits*R_k*s_Ck*ceiling(H/h_pp))/W_hbm` |  | s | unknown | B, D_n, D_r, D_v, H, N_keyvisits, R_k, R_q, S_q, W_hbm, b_q, executed_extent_policy, h_pp, heads_per_program, mask, materialize.q_tilde, materialize.z, outputs, rect_policy, s_Ck, s_Cq, s_Kr, s_Qr, s_Qt, s_Wk, s_Wq, s_Wv, s_Z, schedule | scope=None | actual latency; calibrated prediction |
| T_LB[path:vmem_to_hbm] | `perf.lb.path:vmem_to_hbm` | `(B*D_v*H*S_q*s_O + B*H*R_k*S_q*s_Qt + B*H*R_k*S_q*s_Z + B*H*S_q*s_LSE)/W_hbm_w` |  | s | unknown | B, D_v, H, R_k, S_q, W_hbm_w, executed_extent_policy, heads_per_program, mask, materialize.q_tilde, materialize.z, outputs, rect_policy, s_LSE, s_O, s_Qt, s_Z, schedule | scope=None | actual latency; calibrated prediction |
| T_LB[path:vmem_to_vreg] | `perf.lb.path:vmem_to_vreg` | `(B*D_r*H*b_q*s_Qr*ceiling(S_q/b_q) + B*H*R_k*b_q*s_Qt*ceiling(S_q/b_q) + D_r*N_kvblocks*b_k*s_Kr + N_kvblocks*R_k*b_k*s_Ck)/W_vmem` |  | s | unknown | B, D_r, H, N_kvblocks, R_k, S_q, W_vmem, b_k, b_q, executed_extent_policy, heads_per_program, mask, materialize.q_tilde, materialize.z, outputs, q_resident, rect_policy, register_schedule_evidence, s_Ck, s_Kr, s_Qr, s_Qt, schedule | scope=None | actual latency; calibrated prediction |
| T_LB[critical_path] | `perf.lb.critical_path` | `CP` |  | s | unknown | CP | scope=None | actual latency; calibrated prediction |
| T_LB[combined:scenario=full_overlap_max] | `perf.lb.combined.full_overlap_max` | `max((2*B*D_n*H*R_k*S_q + 2*B*D_n*H*R_q*S_q + 2*B*D_v*H*R_k*S_q + 2*C_pad*D_r + 4*C_pad*R_k)/P_mxu, (B*D_v*H*S_q + B*H*R_k*S_q + 2*B*H*S_q + C_masked + 2*C_pad + 2*N_rowvisits*R_k + 2*N_rowvisits + Piecewise((C_pad, D_r > 0), (0, True)))/P_vpu, (2*C_pad - N_rowvisits)/P_red, (C_pad + H*N_keyvisits*R_k)/P_layout, (C_pad + N_rowvisits)/P_exp, (B*D_n*H*R_k*s_Wk*ceiling(S_q/b_q) + B*D_n*H*R_q*s_Wq*ceiling(S_q/b_q) + B*D_r*H*S_q*s_Qr + B*D_v*H*R_k*s_Wv*ceiling(S_q/b_q) + B*H*R_k*S_q*s_Qt + B*H*R_k*S_q*s_Z + B*R_q*S_q*s_Cq*ceiling(H/h_pp) + D_r*N_keyvisits*s_Kr*ceiling(H/h_pp) + N_keyvisits*R_k*s_Ck*ceiling(H/h_pp))/W_hbm, (B*D_v*H*S_q*s_O + B*H*R_k*S_q*s_Qt + B*H*R_k*S_q*s_Z + B*H*S_q*s_LSE)/W_hbm_w, (B*D_r*H*b_q*s_Qr*ceiling(S_q/b_q) + B*H*R_k*b_q*s_Qt*ceiling(S_q/b_q) + D_r*N_kvblocks*b_k*s_Kr + N_kvblocks*R_k*b_k*s_Ck)/W_vmem, CP)` |  | s | unknown | B, CP, C_masked, C_pad, D_n, D_r, D_v, H, N_keyvisits, N_kvblocks, N_rowvisits, P_exp, P_layout, P_mxu, P_red, P_vpu, R_k, R_q, S_q, W_hbm, W_hbm_w, W_vmem, b_k, b_q, executed_extent_policy, exp_base, h_pp, heads_per_program, mask, materialize.q_tilde, materialize.z, matmul_input_dtype, outputs, overlap_model, projection_scope, q_resident, rect_policy, recurrence, register_schedule_evidence, s_Ck, s_Cq, s_Kr, s_LSE, s_O, s_Qr, s_Qt, s_Wk, s_Wq, s_Wv, s_Z, schedule | overlap scenario full_overlap_max | calibrated prediction; measured time |
| T_LB[combined:scenario=serial_stage_sum] | `perf.lb.combined.serial_stage_sum` | `unknown` |  | s | unknown | B, CP, C_masked, C_pad, D_n, D_r, D_v, H, N_keyvisits, N_kvblocks, N_rowvisits, P_exp, P_layout, P_mxu, P_red, P_vpu, R_k, R_q, S_q, W_hbm, W_hbm_w, W_vmem, b_k, b_q, executed_extent_policy, exp_base, h_pp, heads_per_program, mask, materialize.q_tilde, materialize.z, matmul_input_dtype, outputs, overlap_model, projection_scope, q_resident, rect_policy, recurrence, register_schedule_evidence, s_Ck, s_Cq, s_Kr, s_LSE, s_O, s_Qr, s_Qt, s_Wk, s_Wq, s_Wv, s_Z, schedule | overlap scenario serial_stage_sum | calibrated prediction; measured time |
| T_pred[node_interval] | `perf.calibrated.nodes` | `sum_v W_v/(eps_v P_v) + t_startup, + T_launch + eps_model` |  | s | unknown | calibration | no applicable calibration: no prediction issued | T_LB; hardware-peak lower bound |

<details><summary>Provenance (bindings, coverage, source, evidence)</summary>

| metric | source | coverage | bindings | evidence |
|---|---|---|---|---|
| T_LB[mxu] | derived | incomplete: depends on undeclared/unbound B, C_pad, D_n, D_r, D_v, H, P_mxu, R_k, R_q, S_q, executed_extent_policy, mask, matmul_input_dtype, projection_scope, rect_policy, schedule | `{}` |  |
| T_LB[vpu] | derived | incomplete: depends on undeclared/unbound B, C_masked, C_pad, D_r, D_v, H, N_rowvisits, P_vpu, R_k, S_q, executed_extent_policy, exp_base, mask, outputs, projection_scope, rect_policy, recurrence, schedule | `{}` |  |
| T_LB[reduce] | derived | incomplete: depends on undeclared/unbound C_pad, N_rowvisits, P_red, executed_extent_policy, exp_base, mask, outputs, projection_scope, rect_policy, recurrence, schedule | `{}` |  |
| T_LB[layout] | derived | incomplete: depends on undeclared/unbound C_pad, H, N_keyvisits, P_layout, R_k, executed_extent_policy, exp_base, mask, outputs, projection_scope, rect_policy, recurrence, schedule | `{}` |  |
| T_LB[exp] | derived | incomplete: depends on undeclared/unbound C_pad, N_rowvisits, P_exp, executed_extent_policy, exp_base, mask, outputs, projection_scope, rect_policy, recurrence, schedule | `{}` |  |
| T_LB[path:hbm_to_vmem] | derived | incomplete: depends on undeclared/unbound B, D_n, D_r, D_v, H, N_keyvisits, R_k, R_q, S_q, W_hbm, b_q, executed_extent_policy, h_pp, heads_per_program, mask, materialize.q_tilde, materialize.z, outputs, rect_policy, s_Ck, s_Cq, s_Kr, s_Qr, s_Qt, s_Wk, s_Wq, s_Wv, s_Z, schedule | `{}` |  |
| T_LB[path:vmem_to_hbm] | derived | incomplete: depends on undeclared/unbound B, D_v, H, R_k, S_q, W_hbm_w, executed_extent_policy, heads_per_program, mask, materialize.q_tilde, materialize.z, outputs, rect_policy, s_LSE, s_O, s_Qt, s_Z, schedule | `{}` |  |
| T_LB[path:vmem_to_vreg] | derived | incomplete: depends on undeclared/unbound B, D_r, H, N_kvblocks, R_k, S_q, W_vmem, b_k, b_q, executed_extent_policy, heads_per_program, mask, materialize.q_tilde, materialize.z, outputs, q_resident, rect_policy, register_schedule_evidence, s_Ck, s_Kr, s_Qr, s_Qt, schedule | `{}` |  |
| T_LB[critical_path] | derived | incomplete: depends on undeclared/unbound CP | `{}` |  |
| T_LB[combined:scenario=full_overlap_max] | derived | incomplete: depends on undeclared/unbound B, CP, C_masked, C_pad, D_n, D_r, D_v, H, N_keyvisits, N_kvblocks, N_rowvisits, P_exp, P_layout, P_mxu, P_red, P_vpu, R_k, R_q, S_q, W_hbm, W_hbm_w, W_vmem, b_k, b_q, executed_extent_policy, exp_base, h_pp, heads_per_program, mask, materialize.q_tilde, materialize.z, matmul_input_dtype, outputs, overlap_model, projection_scope, q_resident, rect_policy, recurrence, register_schedule_evidence, s_Ck, s_Cq, s_Kr, s_LSE, s_O, s_Qr, s_Qt, s_Wk, s_Wq, s_Wv, s_Z, schedule | `{}` |  |
| T_LB[combined:scenario=serial_stage_sum] | derived | incomplete: depends on undeclared/unbound B, CP, C_masked, C_pad, D_n, D_r, D_v, H, N_keyvisits, N_kvblocks, N_rowvisits, P_exp, P_layout, P_mxu, P_red, P_vpu, R_k, R_q, S_q, W_hbm, W_hbm_w, W_vmem, b_k, b_q, executed_extent_policy, exp_base, h_pp, heads_per_program, mask, materialize.q_tilde, materialize.z, matmul_input_dtype, outputs, overlap_model, projection_scope, q_resident, rect_policy, recurrence, register_schedule_evidence, s_Ck, s_Cq, s_Kr, s_LSE, s_O, s_Qr, s_Qt, s_Wk, s_Wq, s_Wv, s_Z, schedule | `{}` |  |
| T_pred[node_interval] | derived | incomplete: no applicable calibration evidence attached | `{}` |  |

</details>

## Constraints

| constraint | relation | holds | severity | missing |
|---|---|---|---|---|
| q_tail | `Eq(Mod(S_q, b_q), 0)` | None | info | S_q, b_q |
| kv_tail | `Eq(Mod(S_k, b_k), 0)` | None | info | S_k, b_k |
| Dr_zero_or_positive | `True` | True | info |  |

## vector_ops

```json
[
  {
    "id": "scale_scores",
    "stage": "score",
    "kind": "mul",
    "resource": "vpu",
    "count": "C_pad",
    "conditional_on": null,
    "exists": true,
    "description": "gamma * X on every executed cell (fused variants differ)"
  },
  {
    "id": "mask_apply",
    "stage": "score",
    "kind": "mask",
    "resource": "vpu",
    "count": "C_masked",
    "conditional_on": "mask application scenario: not-fully-visible rectangles only",
    "exists": true,
    "description": "additive/select mask on executed cells of rectangles that are not fully visible (fully visible blocks need no mask)"
  },
  {
    "id": "row_max_cmp",
    "stage": "softmax",
    "kind": "cmp",
    "resource": "reduce",
    "count": "C_pad - N_rowvisits",
    "conditional_on": null,
    "exists": true,
    "description": "n*max(m-1,0) comparisons for rowmax over executed cells"
  },
  {
    "id": "old_new_max_cmp",
    "stage": "softmax",
    "kind": "cmp",
    "resource": "reduce",
    "count": "N_rowvisits",
    "conditional_on": null,
    "exists": true,
    "description": "max(m, rowmax) once per row per block visit"
  },
  {
    "id": "rowmax_broadcast",
    "stage": "softmax",
    "kind": "broadcast",
    "resource": "layout",
    "count": "C_pad",
    "conditional_on": "broadcast_materialized (undeclared backend behaviour)",
    "exists": null,
    "description": "broadcast m' across the row before X - m' (may be free depending on layout)"
  },
  {
    "id": "kv_operand_transpose",
    "stage": "score",
    "kind": "transpose",
    "resource": "layout",
    "count": "H*N_keyvisits*R_k",
    "conditional_on": "operand_transpose_materialized (undeclared backend behaviour)",
    "exists": null,
    "description": "re-layout / transpose of the K-side operand for the score matmul (free when the layout already matches)"
  },
  {
    "id": "exp_E",
    "stage": "softmax",
    "kind": "exp",
    "resource": "exp",
    "count": "C_pad",
    "conditional_on": null,
    "exists": true,
    "description": "E = exp(X - m') on every executed cell (masked cells included)"
  },
  {
    "id": "row_sum_add",
    "stage": "softmax",
    "kind": "add",
    "resource": "reduce",
    "count": "C_pad - N_rowvisits",
    "conditional_on": null,
    "exists": true,
    "description": "n*max(m-1,0) additions for rowsum(E)"
  },
  {
    "id": "rescale_alpha_exp",
    "stage": "softmax",
    "kind": "exp",
    "resource": "exp",
    "count": "N_rowvisits",
    "conditional_on": null,
    "exists": true,
    "description": "alpha = exp(m - m') at most once per row per block visit"
  },
  {
    "id": "acc_scale",
    "stage": "pv",
    "kind": "mul",
    "resource": "vpu",
    "count": "N_rowvisits*R_k",
    "conditional_on": null,
    "exists": true,
    "description": "alpha * A (n*D_A per block visit)"
  },
  {
    "id": "acc_add",
    "stage": "pv",
    "kind": "add",
    "resource": "vpu",
    "count": "N_rowvisits*R_k",
    "conditional_on": null,
    "exists": true,
    "description": "A + E U (n*D_A per block visit; matmul epilogue may fuse)"
  },
  {
    "id": "l_scale",
    "stage": "softmax",
    "kind": "mul",
    "resource": "vpu",
    "count": "N_rowvisits",
    "conditional_on": null,
    "exists": true,
    "description": "alpha * l"
  },
  {
    "id": "l_update",
    "stage": "softmax",
    "kind": "add",
    "resource": "vpu",
    "count": "N_rowvisits",
    "conditional_on": null,
    "exists": true,
    "description": "l' = alpha*l + rowsum"
  },
  {
    "id": "final_normalize_div",
    "stage": "finalize",
    "kind": "div",
    "resource": "vpu",
    "count": "B*H*R_k*S_q",
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
    "count": "B*H*R_k*S_q",
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
    "conditional_on": "outputs",
    "exists": null,
    "description": "log(l) once per row for LSE = m + log(l)"
  },
  {
    "id": "lse_add",
    "stage": "finalize",
    "kind": "add",
    "resource": "vpu",
    "count": "B*H*S_q",
    "conditional_on": "outputs",
    "exists": null,
    "description": "m + log(l) once per row"
  },
  {
    "id": "cast[exp_to_p]",
    "stage": "pv",
    "kind": "cast",
    "resource": "vpu",
    "count": "C_pad",
    "conditional_on": "exp_dtype, p_operand_dtype",
    "exists": null,
    "description": "cast E to the PV / P C_k matmul operand dtype (inferred from the declared dtypes)"
  },
  {
    "id": "cast[accumulator_to_output]",
    "stage": "finalize",
    "kind": "cast",
    "resource": "vpu",
    "count": "B*D_v*H*S_q",
    "conditional_on": "accumulator_dtype, output_dtype",
    "exists": null,
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
    "count": "Piecewise((C_pad, D_r > 0), (0, True))",
    "conditional_on": "D_r > 0",
    "exists": true,
    "description": "X = Q~ C_k^T + Q^r K^r^T (0 when D_r = 0)"
  }
]
```

## materialization_ledger

```json
[
  {
    "id": "mat_q_tilde",
    "tensor": "Q~",
    "conditional_on": "materialize.q_tilde",
    "enabled": null,
    "bytes": "B*H*R_k*S_q*s_Qt",
    "write_traffic": "B*H*R_k*S_q*s_Qt",
    "read_traffic": "B*H*R_k*S_q*s_Qt",
    "bytes_value": null,
    "traffic_value": null,
    "bytes_value_if_materialized": null,
    "traffic_value_if_materialized": null
  },
  {
    "id": "mat_z",
    "tensor": "Z",
    "conditional_on": "materialize.z",
    "enabled": null,
    "bytes": "B*H*R_k*S_q*s_Z",
    "write_traffic": "B*H*R_k*S_q*s_Z",
    "read_traffic": "B*H*R_k*S_q*s_Z",
    "bytes_value": null,
    "traffic_value": null,
    "bytes_value_if_materialized": null,
    "traffic_value_if_materialized": null
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
    "value": null,
    "description": "Q latent row per (head group, query row)",
    "scenario": "logical rows read (spec 4.2: a logical out-of-bounds is not an HBM read)",
    "undeclared": true,
    "conditional_on": "executed_extent_policy"
  },
  {
    "id": "Qr_read",
    "path": "hbm_to_vmem",
    "bytes_per_event": "D_r*s_Qr",
    "executions": "B*H*S_q",
    "total": "B*D_r*H*S_q*s_Qr",
    "value": null,
    "description": "Q^r row per (head, query row)",
    "scenario": "logical rows read (spec 4.2)",
    "undeclared": true,
    "conditional_on": "executed_extent_policy"
  },
  {
    "id": "Ck_read",
    "path": "hbm_to_vmem",
    "bytes_per_event": "R_k*s_Ck",
    "executions": "N_keyvisits*ceiling(H/h_pp)",
    "total": "N_keyvisits*R_k*s_Ck*ceiling(H/h_pp)",
    "value": null,
    "description": "KV latent per executed key column, shared by h_pp heads",
    "scenario": null,
    "undeclared": false,
    "conditional_on": null
  },
  {
    "id": "Kr_read",
    "path": "hbm_to_vmem",
    "bytes_per_event": "D_r*s_Kr",
    "executions": "N_keyvisits*ceiling(H/h_pp)",
    "total": "D_r*N_keyvisits*s_Kr*ceiling(H/h_pp)",
    "value": null,
    "description": "K^r per executed key column, shared by h_pp heads",
    "scenario": null,
    "undeclared": false,
    "conditional_on": null
  },
  {
    "id": "Wv_read",
    "path": "hbm_to_vmem",
    "bytes_per_event": "D_v*R_k*s_Wv",
    "executions": "B*H*ceiling(S_q/b_q)",
    "total": "B*D_v*H*R_k*s_Wv*ceiling(S_q/b_q)",
    "value": null,
    "description": "W^v per (program, head)",
    "scenario": "weights re-read per program",
    "undeclared": false,
    "conditional_on": null
  },
  {
    "id": "Wq_read",
    "path": "hbm_to_vmem",
    "bytes_per_event": "D_n*R_q*s_Wq",
    "executions": "B*H*ceiling(S_q/b_q)",
    "total": "B*D_n*H*R_q*s_Wq*ceiling(S_q/b_q)",
    "value": null,
    "description": "W^q per (program, head)",
    "scenario": "weights re-read per program",
    "undeclared": false,
    "conditional_on": null
  },
  {
    "id": "Wk_read",
    "path": "hbm_to_vmem",
    "bytes_per_event": "D_n*R_k*s_Wk",
    "executions": "B*H*ceiling(S_q/b_q)",
    "total": "B*D_n*H*R_k*s_Wk*ceiling(S_q/b_q)",
    "value": null,
    "description": "W^k per (program, head)",
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
    "value": null,
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
    "value": null,
    "description": "write one LSE value per (b, h, active query row)",
    "scenario": null,
    "undeclared": true,
    "conditional_on": "outputs"
  },
  {
    "id": "stream_Qt_operand",
    "path": "vmem_to_vreg",
    "bytes_per_event": "R_k*b_q*s_Qt",
    "executions": "B*H*ceiling(S_q/b_q)",
    "total": "B*H*R_k*b_q*s_Qt*ceiling(S_q/b_q)",
    "value": null,
    "description": "stream Qt_operand into registers once per (head, q-block)",
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
    "value": null,
    "description": "stream Qr_operand into registers once per (head, q-block)",
    "scenario": "operand streaming (no register-schedule evidence); Q residency undeclared",
    "undeclared": true,
    "conditional_on": "q_resident"
  },
  {
    "id": "stream_Ck_tile",
    "path": "vmem_to_vreg",
    "bytes_per_event": "R_k*b_k*s_Ck",
    "executions": "N_kvblocks",
    "total": "N_kvblocks*R_k*b_k*s_Ck",
    "value": null,
    "description": "stream Ck_tile into registers once per (b, h, kv-block visit)",
    "scenario": "operand streaming (no register-schedule evidence)",
    "undeclared": true,
    "conditional_on": null
  },
  {
    "id": "stream_Kr_tile",
    "path": "vmem_to_vreg",
    "bytes_per_event": "D_r*b_k*s_Kr",
    "executions": "N_kvblocks",
    "total": "D_r*N_kvblocks*b_k*s_Kr",
    "value": null,
    "description": "stream Kr_tile into registers once per (b, h, kv-block visit)",
    "scenario": "operand streaming (no register-schedule evidence)",
    "undeclared": true,
    "conditional_on": null
  },
  {
    "id": "mat_q_tilde_write",
    "path": "vmem_to_hbm",
    "bytes_per_event": "B*H*R_k*S_q*s_Qt",
    "executions": "1",
    "total": "B*H*R_k*S_q*s_Qt",
    "value": null,
    "description": "write the materialized Q~ to HBM",
    "scenario": "additive materialization traffic: the producing computation and its operand reads are left as they are, so this is the round trip added on top, not a restructured dataflow (the expanded K/V materialization is modelled structurally instead)",
    "undeclared": true,
    "conditional_on": "materialize.q_tilde"
  },
  {
    "id": "mat_q_tilde_read",
    "path": "hbm_to_vmem",
    "bytes_per_event": "B*H*R_k*S_q*s_Qt",
    "executions": "1",
    "total": "B*H*R_k*S_q*s_Qt",
    "value": null,
    "description": "read the materialized Q~ back from HBM once per consuming pass",
    "scenario": "additive materialization traffic: the producing computation and its operand reads are left as they are, so this is the round trip added on top, not a restructured dataflow (the expanded K/V materialization is modelled structurally instead)",
    "undeclared": true,
    "conditional_on": "materialize.q_tilde"
  },
  {
    "id": "mat_z_write",
    "path": "vmem_to_hbm",
    "bytes_per_event": "B*H*R_k*S_q*s_Z",
    "executions": "1",
    "total": "B*H*R_k*S_q*s_Z",
    "value": null,
    "description": "write the materialized Z to HBM",
    "scenario": "additive materialization traffic: the producing computation and its operand reads are left as they are, so this is the round trip added on top, not a restructured dataflow (the expanded K/V materialization is modelled structurally instead)",
    "undeclared": true,
    "conditional_on": "materialize.z"
  },
  {
    "id": "mat_z_read",
    "path": "hbm_to_vmem",
    "bytes_per_event": "B*H*R_k*S_q*s_Z",
    "executions": "1",
    "total": "B*H*R_k*S_q*s_Z",
    "value": null,
    "description": "read the materialized Z back from HBM once per consuming pass",
    "scenario": "additive materialization traffic: the producing computation and its operand reads are left as they are, so this is the round trip added on top, not a restructured dataflow (the expanded K/V materialization is modelled structurally instead)",
    "undeclared": true,
    "conditional_on": "materialize.z"
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
      "value": null,
      "residual": null
    },
    "b": {
      "expression": "D_n*s_Qacc + R_q*s_Cq",
      "value": null,
      "residual": null
    },
    "c": {
      "expression": "0",
      "value": null,
      "residual": null
    },
    "d": {
      "expression": "D_n*R_q*s_Wq + s_scratch",
      "value": null,
      "residual": null
    },
    "remainder": {
      "expression": "0",
      "value": null,
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
      "b_rq",
      "scratch_bytes"
    ]
  },
  "all:q_absorb": {
    "stage": "q_absorb",
    "level": null,
    "a": {
      "expression": "0",
      "value": null,
      "residual": null
    },
    "b": {
      "expression": "s_Qacc*(D_n + R_k)",
      "value": null,
      "residual": null
    },
    "c": {
      "expression": "0",
      "value": null,
      "residual": null
    },
    "d": {
      "expression": "D_n*R_k*s_Wk + s_scratch",
      "value": null,
      "residual": null
    },
    "remainder": {
      "expression": "0",
      "value": null,
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
    "undeclared": [
      "scratch_bytes"
    ]
  },
  "all:score": {
    "stage": "score",
    "level": null,
    "a": {
      "expression": "2*s_X",
      "value": null,
      "residual": null
    },
    "b": {
      "expression": "D_r*s_Qr + R_k*s_Qt + R_k*Max(s_A, s_Z) + 2*s_state",
      "value": null,
      "residual": null
    },
    "c": {
      "expression": "n_buf*(D_r*s_Kr + R_k*s_Ck)",
      "value": null,
      "residual": null
    },
    "d": {
      "expression": "s_scratch",
      "value": null,
      "residual": null
    },
    "remainder": {
      "expression": "0",
      "value": null,
      "residual": null
    },
    "provenance": {
      "a": [
        "score_X",
        "score_rope_branch"
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
    "undeclared": [
      "both_qk_branches_live and D_r > 0",
      "scratch_bytes"
    ]
  },
  "all:softmax": {
    "stage": "softmax",
    "level": null,
    "a": {
      "expression": "s_E + s_X",
      "value": null,
      "residual": null
    },
    "b": {
      "expression": "D_r*s_Qr + R_k*s_Qt + R_k*Max(s_A, s_Z) + 3*s_state",
      "value": null,
      "residual": null
    },
    "c": {
      "expression": "R_k*n_buf*s_Ck",
      "value": null,
      "residual": null
    },
    "d": {
      "expression": "s_scratch",
      "value": null,
      "residual": null
    },
    "remainder": {
      "expression": "0",
      "value": null,
      "residual": null
    },
    "provenance": {
      "a": [
        "score_X",
        "exp_E"
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
      "q_resident",
      "score_alias_exp",
      "scratch_bytes"
    ]
  },
  "all:pv": {
    "stage": "pv",
    "level": null,
    "a": {
      "expression": "s_E + s_P",
      "value": null,
      "residual": null
    },
    "b": {
      "expression": "D_r*s_Qr + R_k*s_Qt + R_k*Max(s_A, s_Z) + 3*s_state",
      "value": null,
      "residual": null
    },
    "c": {
      "expression": "R_k*n_buf*s_Ck",
      "value": null,
      "residual": null
    },
    "d": {
      "expression": "s_scratch",
      "value": null,
      "residual": null
    },
    "remainder": {
      "expression": "0",
      "value": null,
      "residual": null
    },
    "provenance": {
      "a": [
        "exp_E",
        "p_operand"
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
      "q_resident",
      "score_alias_exp",
      "exp_alias_p_operand",
      "scratch_bytes"
    ]
  },
  "all:finalize": {
    "stage": "finalize",
    "level": null,
    "a": {
      "expression": "0",
      "value": null,
      "residual": null
    },
    "b": {
      "expression": "D_r*s_Qr + D_v*s_O + R_k*s_Qt + R_k*Max(s_A, s_Z) + s_LSE + 2*s_state",
      "value": null,
      "residual": null
    },
    "c": {
      "expression": "0",
      "value": null,
      "residual": null
    },
    "d": {
      "expression": "D_v*R_k*s_Wv + s_scratch",
      "value": null,
      "residual": null
    },
    "remainder": {
      "expression": "0",
      "value": null,
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
      "q_resident",
      "scratch_bytes",
      "outputs"
    ]
  },
  "vreg:q_proj": {
    "stage": "q_proj",
    "level": "vreg",
    "a": {
      "expression": "0",
      "value": null,
      "residual": null
    },
    "b": {
      "expression": "D_n*s_Qacc",
      "value": null,
      "residual": null
    },
    "c": {
      "expression": "0",
      "value": null,
      "residual": null
    },
    "d": {
      "expression": "0",
      "value": null,
      "residual": null
    },
    "remainder": {
      "expression": "0",
      "value": null,
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
      "value": null,
      "residual": null
    },
    "b": {
      "expression": "s_Qacc*(D_n + R_k)",
      "value": null,
      "residual": null
    },
    "c": {
      "expression": "0",
      "value": null,
      "residual": null
    },
    "d": {
      "expression": "0",
      "value": null,
      "residual": null
    },
    "remainder": {
      "expression": "0",
      "value": null,
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
      "expression": "2*s_X",
      "value": null,
      "residual": null
    },
    "b": {
      "expression": "R_k*Max(s_A, s_Z) + 2*s_state",
      "value": null,
      "residual": null
    },
    "c": {
      "expression": "0",
      "value": null,
      "residual": null
    },
    "d": {
      "expression": "0",
      "value": null,
      "residual": null
    },
    "remainder": {
      "expression": "0",
      "value": null,
      "residual": null
    },
    "provenance": {
      "a": [
        "score_X",
        "score_rope_branch"
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
    "undeclared": [
      "both_qk_branches_live and D_r > 0"
    ]
  },
  "vreg:softmax": {
    "stage": "softmax",
    "level": "vreg",
    "a": {
      "expression": "s_E + s_X",
      "value": null,
      "residual": null
    },
    "b": {
      "expression": "R_k*Max(s_A, s_Z) + 3*s_state",
      "value": null,
      "residual": null
    },
    "c": {
      "expression": "0",
      "value": null,
      "residual": null
    },
    "d": {
      "expression": "0",
      "value": null,
      "residual": null
    },
    "remainder": {
      "expression": "0",
      "value": null,
      "residual": null
    },
    "provenance": {
      "a": [
        "score_X",
        "exp_E"
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
    "undeclared": [
      "score_alias_exp"
    ]
  },
  "vreg:pv": {
    "stage": "pv",
    "level": "vreg",
    "a": {
      "expression": "s_E + s_P",
      "value": null,
      "residual": null
    },
    "b": {
      "expression": "R_k*Max(s_A, s_Z) + 3*s_state",
      "value": null,
      "residual": null
    },
    "c": {
      "expression": "0",
      "value": null,
      "residual": null
    },
    "d": {
      "expression": "0",
      "value": null,
      "residual": null
    },
    "remainder": {
      "expression": "0",
      "value": null,
      "residual": null
    },
    "provenance": {
      "a": [
        "exp_E",
        "p_operand"
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
    "undeclared": [
      "score_alias_exp",
      "exp_alias_p_operand"
    ]
  },
  "vreg:finalize": {
    "stage": "finalize",
    "level": "vreg",
    "a": {
      "expression": "0",
      "value": null,
      "residual": null
    },
    "b": {
      "expression": "R_k*Max(s_A, s_Z) + 2*s_state",
      "value": null,
      "residual": null
    },
    "c": {
      "expression": "0",
      "value": null,
      "residual": null
    },
    "d": {
      "expression": "0",
      "value": null,
      "residual": null
    },
    "remainder": {
      "expression": "0",
      "value": null,
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
      "value": null,
      "residual": null
    },
    "b": {
      "expression": "R_q*s_Cq",
      "value": null,
      "residual": null
    },
    "c": {
      "expression": "0",
      "value": null,
      "residual": null
    },
    "d": {
      "expression": "D_n*R_q*s_Wq + s_scratch",
      "value": null,
      "residual": null
    },
    "remainder": {
      "expression": "0",
      "value": null,
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
      "b_rq",
      "scratch_bytes"
    ]
  },
  "vmem:q_absorb": {
    "stage": "q_absorb",
    "level": "vmem",
    "a": {
      "expression": "0",
      "value": null,
      "residual": null
    },
    "b": {
      "expression": "0",
      "value": null,
      "residual": null
    },
    "c": {
      "expression": "0",
      "value": null,
      "residual": null
    },
    "d": {
      "expression": "D_n*R_k*s_Wk + s_scratch",
      "value": null,
      "residual": null
    },
    "remainder": {
      "expression": "0",
      "value": null,
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
    "undeclared": [
      "scratch_bytes"
    ]
  },
  "vmem:score": {
    "stage": "score",
    "level": "vmem",
    "a": {
      "expression": "0",
      "value": null,
      "residual": null
    },
    "b": {
      "expression": "D_r*s_Qr + R_k*s_Qt",
      "value": null,
      "residual": null
    },
    "c": {
      "expression": "n_buf*(D_r*s_Kr + R_k*s_Ck)",
      "value": null,
      "residual": null
    },
    "d": {
      "expression": "s_scratch",
      "value": null,
      "residual": null
    },
    "remainder": {
      "expression": "0",
      "value": null,
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
    "undeclared": [
      "scratch_bytes"
    ]
  },
  "vmem:softmax": {
    "stage": "softmax",
    "level": "vmem",
    "a": {
      "expression": "0",
      "value": null,
      "residual": null
    },
    "b": {
      "expression": "D_r*s_Qr + R_k*s_Qt",
      "value": null,
      "residual": null
    },
    "c": {
      "expression": "R_k*n_buf*s_Ck",
      "value": null,
      "residual": null
    },
    "d": {
      "expression": "s_scratch",
      "value": null,
      "residual": null
    },
    "remainder": {
      "expression": "0",
      "value": null,
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
      "q_resident",
      "scratch_bytes"
    ]
  },
  "vmem:pv": {
    "stage": "pv",
    "level": "vmem",
    "a": {
      "expression": "0",
      "value": null,
      "residual": null
    },
    "b": {
      "expression": "D_r*s_Qr + R_k*s_Qt",
      "value": null,
      "residual": null
    },
    "c": {
      "expression": "R_k*n_buf*s_Ck",
      "value": null,
      "residual": null
    },
    "d": {
      "expression": "s_scratch",
      "value": null,
      "residual": null
    },
    "remainder": {
      "expression": "0",
      "value": null,
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
      "q_resident",
      "scratch_bytes"
    ]
  },
  "vmem:finalize": {
    "stage": "finalize",
    "level": "vmem",
    "a": {
      "expression": "0",
      "value": null,
      "residual": null
    },
    "b": {
      "expression": "D_r*s_Qr + D_v*s_O + R_k*s_Qt + s_LSE",
      "value": null,
      "residual": null
    },
    "c": {
      "expression": "0",
      "value": null,
      "residual": null
    },
    "d": {
      "expression": "D_v*R_k*s_Wv + s_scratch",
      "value": null,
      "residual": null
    },
    "remainder": {
      "expression": "0",
      "value": null,
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
      "q_resident",
      "scratch_bytes",
      "outputs"
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
      "numerator": "2*B*D_n*H*R_k*S_q + 2*B*D_n*H*R_q*S_q + 2*B*D_v*H*R_k*S_q + 2*C_pad*D_r + 4*C_pad*R_k",
      "denominator": "P_mxu",
      "expression": "(2*B*D_n*H*R_k*S_q + 2*B*D_n*H*R_q*S_q + 2*B*D_v*H*R_k*S_q + 2*C_pad*D_r + 4*C_pad*R_k)/P_mxu",
      "value": null,
      "unit": "s",
      "scope": null,
      "status": "unknown",
      "missing": [
        "B",
        "C_pad",
        "D_n",
        "D_r",
        "D_v",
        "H",
        "P_mxu",
        "R_k",
        "R_q",
        "S_q",
        "executed_extent_policy",
        "mask",
        "matmul_input_dtype",
        "projection_scope",
        "rect_policy",
        "schedule"
      ],
      "notes": [
        "numerator: executed matmul FLOPs of this scenario (C_exec); denominator: matmul peak for the declared compute dtype",
        "compute dtype undeclared: no peak can be selected"
      ]
    },
    {
      "resource": "vpu",
      "numerator": "B*D_v*H*S_q + B*H*R_k*S_q + 2*B*H*S_q + C_masked + 2*C_pad + 2*N_rowvisits*R_k + 2*N_rowvisits + Piecewise((C_pad, D_r > 0), (0, True))",
      "denominator": "P_vpu",
      "expression": "(B*D_v*H*S_q + B*H*R_k*S_q + 2*B*H*S_q + C_masked + 2*C_pad + 2*N_rowvisits*R_k + 2*N_rowvisits + Piecewise((C_pad, D_r > 0), (0, True)))/P_vpu",
      "value": null,
      "unit": "s",
      "scope": null,
      "status": "unknown",
      "missing": [
        "B",
        "C_masked",
        "C_pad",
        "D_r",
        "D_v",
        "H",
        "N_rowvisits",
        "P_vpu",
        "R_k",
        "S_q",
        "executed_extent_policy",
        "exp_base",
        "mask",
        "outputs",
        "projection_scope",
        "rect_policy",
        "recurrence",
        "schedule"
      ],
      "notes": []
    },
    {
      "resource": "reduce",
      "numerator": "2*C_pad - N_rowvisits",
      "denominator": "P_red",
      "expression": "(2*C_pad - N_rowvisits)/P_red",
      "value": null,
      "unit": "s",
      "scope": null,
      "status": "unknown",
      "missing": [
        "C_pad",
        "N_rowvisits",
        "P_red",
        "executed_extent_policy",
        "exp_base",
        "mask",
        "outputs",
        "projection_scope",
        "rect_policy",
        "recurrence",
        "schedule"
      ],
      "notes": []
    },
    {
      "resource": "layout",
      "numerator": "C_pad + H*N_keyvisits*R_k",
      "denominator": "P_layout",
      "expression": "(C_pad + H*N_keyvisits*R_k)/P_layout",
      "value": null,
      "unit": "s",
      "scope": null,
      "status": "unknown",
      "missing": [
        "C_pad",
        "H",
        "N_keyvisits",
        "P_layout",
        "R_k",
        "executed_extent_policy",
        "exp_base",
        "mask",
        "outputs",
        "projection_scope",
        "rect_policy",
        "recurrence",
        "schedule"
      ],
      "notes": []
    },
    {
      "resource": "exp",
      "numerator": "C_pad + N_rowvisits",
      "denominator": "P_exp",
      "expression": "(C_pad + N_rowvisits)/P_exp",
      "value": null,
      "unit": "s",
      "scope": null,
      "status": "unknown",
      "missing": [
        "C_pad",
        "N_rowvisits",
        "P_exp",
        "executed_extent_policy",
        "exp_base",
        "mask",
        "outputs",
        "projection_scope",
        "rect_policy",
        "recurrence",
        "schedule"
      ],
      "notes": []
    },
    {
      "resource": "path:hbm_to_vmem",
      "numerator": "B*D_n*H*R_k*s_Wk*ceiling(S_q/b_q) + B*D_n*H*R_q*s_Wq*ceiling(S_q/b_q) + B*D_r*H*S_q*s_Qr + B*D_v*H*R_k*s_Wv*ceiling(S_q/b_q) + B*H*R_k*S_q*s_Qt + B*H*R_k*S_q*s_Z + B*R_q*S_q*s_Cq*ceiling(H/h_pp) + D_r*N_keyvisits*s_Kr*ceiling(H/h_pp) + N_keyvisits*R_k*s_Ck*ceiling(H/h_pp)",
      "denominator": "W_hbm",
      "expression": "(B*D_n*H*R_k*s_Wk*ceiling(S_q/b_q) + B*D_n*H*R_q*s_Wq*ceiling(S_q/b_q) + B*D_r*H*S_q*s_Qr + B*D_v*H*R_k*s_Wv*ceiling(S_q/b_q) + B*H*R_k*S_q*s_Qt + B*H*R_k*S_q*s_Z + B*R_q*S_q*s_Cq*ceiling(H/h_pp) + D_r*N_keyvisits*s_Kr*ceiling(H/h_pp) + N_keyvisits*R_k*s_Ck*ceiling(H/h_pp))/W_hbm",
      "value": null,
      "unit": "s",
      "scope": null,
      "status": "unknown",
      "missing": [
        "B",
        "D_n",
        "D_r",
        "D_v",
        "H",
        "N_keyvisits",
        "R_k",
        "R_q",
        "S_q",
        "W_hbm",
        "b_q",
        "executed_extent_policy",
        "h_pp",
        "heads_per_program",
        "mask",
        "materialize.q_tilde",
        "materialize.z",
        "outputs",
        "rect_policy",
        "s_Ck",
        "s_Cq",
        "s_Kr",
        "s_Qr",
        "s_Qt",
        "s_Wk",
        "s_Wq",
        "s_Wv",
        "s_Z",
        "schedule"
      ],
      "notes": [
        "conditional transfer requests of this scenario / path bandwidth upper bound; not an absolute limit for all algorithms",
        "path traffic depends on undeclared fields or on register-schedule evidence (scenario)"
      ]
    },
    {
      "resource": "path:vmem_to_hbm",
      "numerator": "B*D_v*H*S_q*s_O + B*H*R_k*S_q*s_Qt + B*H*R_k*S_q*s_Z + B*H*S_q*s_LSE",
      "denominator": "W_hbm_w",
      "expression": "(B*D_v*H*S_q*s_O + B*H*R_k*S_q*s_Qt + B*H*R_k*S_q*s_Z + B*H*S_q*s_LSE)/W_hbm_w",
      "value": null,
      "unit": "s",
      "scope": null,
      "status": "unknown",
      "missing": [
        "B",
        "D_v",
        "H",
        "R_k",
        "S_q",
        "W_hbm_w",
        "executed_extent_policy",
        "heads_per_program",
        "mask",
        "materialize.q_tilde",
        "materialize.z",
        "outputs",
        "rect_policy",
        "s_LSE",
        "s_O",
        "s_Qt",
        "s_Z",
        "schedule"
      ],
      "notes": [
        "conditional transfer requests of this scenario / path bandwidth upper bound; not an absolute limit for all algorithms",
        "path traffic depends on undeclared fields or on register-schedule evidence (scenario)"
      ]
    },
    {
      "resource": "path:vmem_to_vreg",
      "numerator": "B*D_r*H*b_q*s_Qr*ceiling(S_q/b_q) + B*H*R_k*b_q*s_Qt*ceiling(S_q/b_q) + D_r*N_kvblocks*b_k*s_Kr + N_kvblocks*R_k*b_k*s_Ck",
      "denominator": "W_vmem",
      "expression": "(B*D_r*H*b_q*s_Qr*ceiling(S_q/b_q) + B*H*R_k*b_q*s_Qt*ceiling(S_q/b_q) + D_r*N_kvblocks*b_k*s_Kr + N_kvblocks*R_k*b_k*s_Ck)/W_vmem",
      "value": null,
      "unit": "s",
      "scope": null,
      "status": "unknown",
      "missing": [
        "B",
        "D_r",
        "H",
        "N_kvblocks",
        "R_k",
        "S_q",
        "W_vmem",
        "b_k",
        "b_q",
        "executed_extent_policy",
        "heads_per_program",
        "mask",
        "materialize.q_tilde",
        "materialize.z",
        "outputs",
        "q_resident",
        "rect_policy",
        "register_schedule_evidence",
        "s_Ck",
        "s_Kr",
        "s_Qr",
        "s_Qt",
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
  "combined_expression": "scenario-dependent (overlap_model undeclared)",
  "resources_outside_declared_groups": [],
  "combined_value_known_terms_only": null,
  "status": "symbolic",
  "overlap_model": "undeclared",
  "combined_missing_fields": [
    "B",
    "CP",
    "C_masked",
    "C_pad",
    "D_n",
    "D_r",
    "D_v",
    "H",
    "N_keyvisits",
    "N_kvblocks",
    "N_rowvisits",
    "P_exp",
    "P_layout",
    "P_mxu",
    "P_red",
    "P_vpu",
    "R_k",
    "R_q",
    "S_q",
    "W_hbm",
    "W_hbm_w",
    "W_vmem",
    "b_k",
    "b_q",
    "executed_extent_policy",
    "exp_base",
    "h_pp",
    "heads_per_program",
    "mask",
    "materialize.q_tilde",
    "materialize.z",
    "matmul_input_dtype",
    "outputs",
    "overlap_model",
    "projection_scope",
    "q_resident",
    "rect_policy",
    "recurrence",
    "register_schedule_evidence",
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
    "s_Z",
    "schedule"
  ],
  "scenarios": {
    "full_overlap_max": {
      "value": null,
      "expression": "max((2*B*D_n*H*R_k*S_q + 2*B*D_n*H*R_q*S_q + 2*B*D_v*H*R_k*S_q + 2*C_pad*D_r + 4*C_pad*R_k)/P_mxu, (B*D_v*H*S_q + B*H*R_k*S_q + 2*B*H*S_q + C_masked + 2*C_pad + 2*N_rowvisits*R_k + 2*N_rowvisits + Piecewise((C_pad, D_r > 0), (0, True)))/P_vpu, (2*C_pad - N_rowvisits)/P_red, (C_pad + H*N_keyvisits*R_k)/P_layout, (C_pad + N_rowvisits)/P_exp, (B*D_n*H*R_k*s_Wk*ceiling(S_q/b_q) + B*D_n*H*R_q*s_Wq*ceiling(S_q/b_q) + B*D_r*H*S_q*s_Qr + B*D_v*H*R_k*s_Wv*ceiling(S_q/b_q) + B*H*R_k*S_q*s_Qt + B*H*R_k*S_q*s_Z + B*R_q*S_q*s_Cq*ceiling(H/h_pp) + D_r*N_keyvisits*s_Kr*ceiling(H/h_pp) + N_keyvisits*R_k*s_Ck*ceiling(H/h_pp))/W_hbm, (B*D_v*H*S_q*s_O + B*H*R_k*S_q*s_Qt + B*H*R_k*S_q*s_Z + B*H*S_q*s_LSE)/W_hbm_w, (B*D_r*H*b_q*s_Qr*ceiling(S_q/b_q) + B*H*R_k*b_q*s_Qt*ceiling(S_q/b_q) + D_r*N_kvblocks*b_k*s_Kr + N_kvblocks*R_k*b_k*s_Ck)/W_vmem, CP)"
    },
    "serial_stage_sum": {
      "value": null,
      "expression": "unknown"
    }
  },
  "notes": [
    "T_actual >= T_LB only for the counted scenario; missing resources make T_LB a lower bound on the lower bound",
    "serial_stage_sum groups are resource classes of the whole kernel (declared non-overlap between them), not per-pipeline-stage attribution",
    "overlap_model undeclared: both the fully-overlapped max and the fully-serial sum are reported as scenarios"
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
    "F_Q[useful]": [
      "B",
      "D_n",
      "H",
      "R_q",
      "S_q"
    ],
    "F_Qtilde[useful]": [
      "B",
      "D_n",
      "H",
      "R_k",
      "S_q"
    ],
    "F_QC[useful]": [
      "C_valid",
      "R_k"
    ],
    "F_QK_r[useful]": [
      "C_valid",
      "D_r"
    ],
    "F_PC[useful]": [
      "C_valid",
      "R_k"
    ],
    "F_ZWv[useful]": [
      "B",
      "D_v",
      "H",
      "R_k",
      "S_q"
    ],
    "F_total[useful]": [
      "B",
      "C_valid",
      "D_n",
      "D_r",
      "D_v",
      "H",
      "R_k",
      "R_q",
      "S_q"
    ],
    "F_A[spec_closed_form]": [
      "B",
      "C_valid",
      "D_n",
      "D_r",
      "D_v",
      "H",
      "R_k",
      "R_q",
      "S_q"
    ],
    "N_Qproj": [
      "B",
      "H",
      "S_q"
    ],
    "N_Kproj": [],
    "N_Vproj": [],
    "F_proj[general]": [
      "B",
      "D_n",
      "H",
      "R_k",
      "R_q",
      "S_q"
    ],
    "F_total[executed_graph]": [
      "B",
      "C_pad",
      "D_n",
      "D_r",
      "D_v",
      "H",
      "R_k",
      "R_q",
      "S_q"
    ],
    "F_projection[executed_graph]": [
      "B",
      "D_n",
      "H",
      "R_k",
      "R_q",
      "S_q"
    ],
    "F_attention[executed_graph]": [
      "C_pad",
      "D_r",
      "R_k"
    ],
    "F_output_projection[executed_graph]": [
      "B",
      "D_v",
      "H",
      "R_k",
      "S_q"
    ],
    "F_A_minus_F_E": [
      "B",
      "C_valid",
      "D_n",
      "D_r",
      "D_v",
      "H",
      "R_k",
      "S_k",
      "S_q"
    ],
    "F_A_minus_F_E[spec_closed_form]": [
      "B",
      "C_valid",
      "D_n",
      "D_v",
      "H",
      "R_k",
      "S_k",
      "S_q"
    ],
    "W_vec[mul]": [
      "C_pad",
      "N_rowvisits",
      "R_k"
    ],
    "W_vec[mask]": [
      "C_masked"
    ],
    "W_vec[cmp]": [
      "C_pad"
    ],
    "W_vec[broadcast]": [
      "C_pad"
    ],
    "W_vec[transpose]": [
      "H",
      "N_keyvisits",
      "R_k"
    ],
    "W_vec[exp]": [
      "C_pad",
      "N_rowvisits"
    ],
    "W_vec[add]": [
      "B",
      "C_pad",
      "D_r",
      "H",
      "N_rowvisits",
      "R_k",
      "S_q"
    ],
    "W_vec[div]": [
      "B",
      "H",
      "R_k",
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
      "C_pad",
      "D_v",
      "H",
      "S_q"
    ],
    "W_resource[vpu]": [
      "B",
      "C_masked",
      "C_pad",
      "D_r",
      "D_v",
      "H",
      "N_rowvisits",
      "R_k",
      "S_q"
    ],
    "W_resource[reduce]": [
      "C_pad",
      "N_rowvisits"
    ],
    "W_resource[layout]": [
      "C_pad",
      "H",
      "N_keyvisits",
      "R_k"
    ],
    "W_resource[exp]": [
      "C_pad",
      "N_rowvisits"
    ],
    "exp_count_executed": [
      "C_pad"
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
      "B",
      "D_n",
      "D_r",
      "D_v",
      "H",
      "R_k",
      "R_q",
      "S_k",
      "S_q",
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
    "HBM_mat[Q~]": [
      "B",
      "H",
      "R_k",
      "S_q",
      "s_Qt"
    ],
    "M_mat[Q~]": [
      "B",
      "H",
      "R_k",
      "S_q",
      "s_Qt"
    ],
    "HBM_mat[Z]": [
      "B",
      "H",
      "R_k",
      "S_q",
      "s_Z"
    ],
    "M_mat[Z]": [
      "B",
      "H",
      "R_k",
      "S_q",
      "s_Z"
    ],
    "B[hbm_to_vmem]": [
      "B",
      "D_n",
      "D_r",
      "D_v",
      "H",
      "N_keyvisits",
      "R_k",
      "R_q",
      "S_q",
      "b_q",
      "h_pp",
      "s_Ck",
      "s_Cq",
      "s_Kr",
      "s_Qr",
      "s_Qt",
      "s_Wk",
      "s_Wq",
      "s_Wv",
      "s_Z"
    ],
    "B[vmem_to_hbm]": [
      "B",
      "D_v",
      "H",
      "R_k",
      "S_q",
      "s_LSE",
      "s_O",
      "s_Qt",
      "s_Z"
    ],
    "B[vmem_to_vreg]": [
      "B",
      "D_r",
      "H",
      "N_kvblocks",
      "R_k",
      "S_q",
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
    "local[score_rope_branch]": [
      "b_k",
      "b_q",
      "s_X"
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
    "layout[score_rope_branch]": [
      "L_l",
      "L_s",
      "b_k",
      "b_q",
      "s_X"
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
      "B",
      "H",
      "S_q",
      "b_q",
      "h_pp"
    ],
    "n_kv_block_visits": [
      "N_kvblocks"
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
      "id": "Q_absorb",
      "stage": "q_absorb",
      "M": "B*H*S_q",
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
      "M": "C_pad",
      "N": "1",
      "K": "R_k",
      "multiplicity": "1",
      "tile_shape": [
        "b_q",
        "b_k",
        "R_k"
      ],
      "tile_shape_value": [
        null,
        null,
        null
      ],
      "counts_toward": "attention"
    },
    {
      "id": "QK_rope",
      "stage": "score",
      "M": "C_pad",
      "N": "1",
      "K": "D_r",
      "multiplicity": "1",
      "tile_shape": [
        "b_q",
        "b_k",
        "D_r"
      ],
      "tile_shape_value": [
        null,
        null,
        null
      ],
      "counts_toward": "attention"
    },
    {
      "id": "PC_latent",
      "stage": "pv",
      "M": "C_pad",
      "N": "1",
      "K": "R_k",
      "multiplicity": "1",
      "tile_shape": [
        "b_q",
        "R_k",
        "b_k"
      ],
      "tile_shape_value": [
        null,
        null,
        null
      ],
      "counts_toward": "attention"
    },
    {
      "id": "Z_Wv",
      "stage": "finalize",
      "M": "B*H*S_q",
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
      "alias_group": null,
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
      "alias_group": null,
      "exists": null,
      "conditional_on": "score_alias_exp",
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
      "exists": null,
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
      "id": "score_rope_branch",
      "size": "b_k*b_q*s_X",
      "level": "vreg",
      "live_stages": [
        "score"
      ],
      "alias_group": null,
      "exists": null,
      "conditional_on": "both_qk_branches_live and D_r > 0",
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
      "exists": null,
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
      "exists": null,
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
      "undeclared": true
    },
    {
      "id": "Qr_read",
      "path": "hbm_to_vmem",
      "total": "B*D_r*H*S_q*s_Qr",
      "undeclared": true
    },
    {
      "id": "Ck_read",
      "path": "hbm_to_vmem",
      "total": "N_keyvisits*R_k*s_Ck*ceiling(H/h_pp)",
      "undeclared": false
    },
    {
      "id": "Kr_read",
      "path": "hbm_to_vmem",
      "total": "D_r*N_keyvisits*s_Kr*ceiling(H/h_pp)",
      "undeclared": false
    },
    {
      "id": "Wv_read",
      "path": "hbm_to_vmem",
      "total": "B*D_v*H*R_k*s_Wv*ceiling(S_q/b_q)",
      "undeclared": false
    },
    {
      "id": "Wq_read",
      "path": "hbm_to_vmem",
      "total": "B*D_n*H*R_q*s_Wq*ceiling(S_q/b_q)",
      "undeclared": false
    },
    {
      "id": "Wk_read",
      "path": "hbm_to_vmem",
      "total": "B*D_n*H*R_k*s_Wk*ceiling(S_q/b_q)",
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
      "undeclared": true
    },
    {
      "id": "stream_Qt_operand",
      "path": "vmem_to_vreg",
      "total": "B*H*R_k*b_q*s_Qt*ceiling(S_q/b_q)",
      "undeclared": true
    },
    {
      "id": "stream_Qr_operand",
      "path": "vmem_to_vreg",
      "total": "B*D_r*H*b_q*s_Qr*ceiling(S_q/b_q)",
      "undeclared": true
    },
    {
      "id": "stream_Ck_tile",
      "path": "vmem_to_vreg",
      "total": "N_kvblocks*R_k*b_k*s_Ck",
      "undeclared": true
    },
    {
      "id": "stream_Kr_tile",
      "path": "vmem_to_vreg",
      "total": "D_r*N_kvblocks*b_k*s_Kr",
      "undeclared": true
    },
    {
      "id": "mat_q_tilde_write",
      "path": "vmem_to_hbm",
      "total": "B*H*R_k*S_q*s_Qt",
      "undeclared": true
    },
    {
      "id": "mat_q_tilde_read",
      "path": "hbm_to_vmem",
      "total": "B*H*R_k*S_q*s_Qt",
      "undeclared": true
    },
    {
      "id": "mat_z_write",
      "path": "vmem_to_hbm",
      "total": "B*H*R_k*S_q*s_Z",
      "undeclared": true
    },
    {
      "id": "mat_z_read",
      "path": "hbm_to_vmem",
      "total": "B*H*R_k*S_q*s_Z",
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

## adapter_contract

```json
{
  "adapter": "seven_input_latent",
  "roles": {
    "q_latent": [
      "B",
      "Sq",
      "Rq"
    ],
    "kv_latent": [
      "B",
      "Sk",
      "Rk"
    ],
    "q_pe": [
      "B",
      "H",
      "Sq",
      "Dr"
    ],
    "k_pe": [
      "B",
      "Sk",
      "Dr"
    ],
    "w_q_nope": [
      "H",
      "Rq",
      "Dn"
    ],
    "w_k_nope": [
      "H",
      "Rk",
      "Dn"
    ],
    "w_v": [
      "H",
      "Rk",
      "Dv"
    ]
  },
  "output": {
    "O": [
      "B",
      "H",
      "Sq",
      "Dv"
    ],
    "LSE": [
      "B",
      "H",
      "Sq"
    ]
  },
  "primary_bindings": {
    "B": "tensor.q_latent.axis[B]",
    "Sq": "tensor.q_latent.axis[Sq]",
    "Rq": "tensor.q_latent.axis[Rq]",
    "Sk": "tensor.kv_latent.axis[Sk]",
    "Rk": "tensor.kv_latent.axis[Rk]",
    "H": "tensor.w_q_nope.axis[H]",
    "Dn": "tensor.w_q_nope.axis[Dn]",
    "Dr": "tensor.q_pe.axis[Dr]",
    "Dv": "tensor.w_v.axis[Dv]"
  },
  "capabilities": {
    "axis_permutation": true,
    "ragged_lengths": true,
    "no_positional_branch": true,
    "kv_capacity_gt_active": true,
    "packed_sequences": false,
    "quantized_storage": false,
    "sharded_tensors": false,
    "zero_length_axes": false,
    "missing_q_latent": false,
    "aliasing": true
  },
  "role_aliases": {
    "cq": "q_latent",
    "c_q": "q_latent",
    "q_latent": "q_latent",
    "kv_c": "kv_latent",
    "ck": "kv_latent",
    "c_k": "kv_latent",
    "kv_latent": "kv_latent",
    "c_kv": "kv_latent",
    "q_rope": "q_pe",
    "qr": "q_pe",
    "q_pe": "q_pe",
    "k_rope": "k_pe",
    "kr": "k_pe",
    "k_pe": "k_pe",
    "wq": "w_q_nope",
    "w_q": "w_q_nope",
    "w_q_nope": "w_q_nope",
    "wk": "w_k_nope",
    "w_k": "w_k_nope",
    "w_k_nope": "w_k_nope",
    "wv": "w_v",
    "w_v": "w_v"
  },
  "notes": [
    "Shapes are role/axis contracts; physical order may differ when 'axes' names are given.",
    "Sq/Sk from shapes are capacities; active lengths come from semantics.active_lengths.",
    "Projection of Q/K/V from latents is inside this entry scope unless projection_scope says otherwise."
  ]
}
```
