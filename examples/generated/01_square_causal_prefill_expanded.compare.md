# Strategy comparison (task 55fd2c864359ee4e, expanded)

- verdict: no calibration: Pareto set / hypotheses requiring validation, not a measured winner
- useful work invariant across candidates: True
- Pareto objectives: ['B[hbm_to_vmem]', 'W_resource[reduce]', 'rect_waste', 'M_live[vreg:softmax]', 'M_live[vmem:softmax]', 'n_programs']  → Pareto set: ['bq32_bk16', 'bq16_bk16_h4']
- objectives whose values are scenario values (status partial) for some candidate, so the Pareto filter compares scenarios, not complete figures: B[hbm_to_vmem], M_live[vmem:softmax], W_resource[reduce], n_programs, rect_waste

| metric | bq16_bk16 | bq32_bk16 | bq16_bk32 | bq16_bk16_fullscan | bq16_bk16_h4 | switch_to_absorbed [absorbed_two_step] |
|---|---|---|---|---|---|---|
| F_total[useful] | 3829760 | 3829760 (+0) | 3829760 (+0) | 3829760 (+0) | 3829760 (+0) | 4362240 (+532480) |
| F_total[executed_graph] | 6619136 | 5505024 (-1114112) | 7864320 (+1245184) | 10354688 (+3735552) | 6619136 (+0) | 4915200 (-1703936) |
| F_A_minus_F_E | 532480 | 532480 (+0) | 532480 (+0) | 532480 (+0) | 532480 (+0) | 532480 (+0) |
| C_rect | 20480 | 24576 (+4096) | 24576 (+4096) | 32768 (+12288) | 20480 (+0) | 20480 (+0) |
| C_pad | 20480 | 24576 (+4096) | 24576 (+4096) | 32768 (+12288) | 20480 (+0) | 20480 (+0) |
| rect_waste | 3840 | 7936 (+4096) | 7936 (+4096) | 16128 (+12288) | 3840 (+0) | 3840 (+0) |
| B[hbm_to_vmem] | 249856 | 147456 (-102400) | 270336 (+20480) | 311296 (+61440) | 160768 (-89088) | 249856 (+0) |
| B[vmem_to_hbm] | 26624 | 26624 (+0) | 26624 (+0) | 26624 (+0) | 26624 (+0) | 26624 (+0) |
| HBM_mat[K^n,V] | 0 | 0 (+0) | 0 (+0) | 0 (+0) | 0 (+0) | n/a |
| local[score_X] | 1024 | 2048 (+1024) | 2048 (+1024) | 1024 (+0) | 1024 (+0) | 1024 (+0) |
| local[accumulator_A] | 1536 | 3072 (+1536) | 1536 (+0) | 1536 (+0) | 1536 (+0) | 2048 (+512) |
| local[v_tile] | 768 | 768 (+0) | 1536 (+768) | 768 (+0) | 768 (+0) | n/a |
| M_live[vreg:softmax] | 2752 | 5504 (+2752) | 3776 (+1024) | 2752 (+0) | 2752 (+0) | 3264 (+512) |
| M_live[vmem:softmax] | 1792 | 2816 (+1024) | 2560 (+768) | 1792 (+0) | 1792 (+0) | 3328 (+1536) |
| M_peak[all] | 9344 | 12032 (+2688) | 12928 (+3584) | 9344 (+0) | 9344 (+0) | 7104 (-2240) |
| n_programs | 32 | 16 (-16) | 32 (+0) | 32 (+0) | 8 (-24) | 32 (+0) |
| n_kv_block_visits | 80 | 48 (-32) | 48 (-32) | 128 (+48) | 80 (+0) | 80 (+0) |
| V_lifetime_stages | 4 | 4 (+0) | 4 (+0) | 4 (+0) | 4 (+0) | n/a |
| W_resource[exp] | 21760 | 26112 (+4352) | 25344 (+3584) | 34816 (+13056) | 21760 (+0) | 21760 (+0) |
| W_resource[reduce] | 39680 | 47616 (+7936) | 48384 (+8704) | 63488 (+23808) | 39680 (+0) | 39680 (+0) |
| b_k_max[vreg:binding_stage] | ? | ? | ? | ? | ? | ? |
| T_LB[mxu] | ? | ? | ? | ? | ? | ? |
| T_LB[path:hbm_to_vmem] | ? | ? | ? | ? | ? | ? |
| T_LB[combined] | ? | ? | ? | ? | ? | ? |
| T_pred[node_interval] | ? | ? | ? | ? | ? | ? |

(`n/a` = this variant has no such metric; `?` = the metric exists but its value is unknown)

| candidate | mathematical legality | backend | model-feasible under declared budgets | binding stage |
|---|---|---|---|---|
| bq16_bk16 | legal  | unknown (no compilation evidence) | None | None |
| bq32_bk16 | legal  | unknown (no compilation evidence) | None | None |
| bq16_bk32 | legal  | unknown (no compilation evidence) | None | None |
| bq16_bk16_fullscan | legal  | unknown (no compilation evidence) | None | None |
| bq16_bk16_h4 | legal  | unknown (no compilation evidence) | None | None |
| switch_to_absorbed [absorbed_two_step] | legal  | unknown (no compilation evidence) | None | None |

| candidate | intermediates that appear | intermediates that disappear | MXU tile shape changes |
|---|---|---|---|
| bq32_bk16 | — | — | QK_nope: [16, 16, 24] → [32, 16, 24]; QK_rope: [16, 16, 8] → [32, 16, 8]; PV: [16, 24, 16] → [32, 24, 16] |
| bq16_bk32 | — | — | QK_nope: [16, 16, 24] → [16, 32, 24]; QK_rope: [16, 16, 8] → [16, 32, 8]; PV: [16, 24, 16] → [16, 24, 32] |
| switch_to_absorbed [absorbed_two_step] | kv_latent_tile, q_tilde, q_tilde_acc, wk_full, wv_full, z_tile | k_nope_tile, kv_latent_rank_tile, q_nope_operand, v_tile, wk_rank_tile, wv_rank_tile | QC_latent: None → [16, 16, 32]; PC_latent: None → [16, 32, 16]; QK_nope: [16, 16, 24] → None; PV: [16, 24, 16] → None |
