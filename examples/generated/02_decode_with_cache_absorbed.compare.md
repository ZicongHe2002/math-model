# Strategy comparison (task 8de4242090fd5e82, absorbed_two_step)

- verdict: no calibration: Pareto set / hypotheses requiring validation, not a measured winner
- useful work invariant across candidates: True
- Pareto objectives: ['B[hbm_to_vmem]', 'W_resource[reduce]', 'rect_waste', 'M_live[vreg:softmax]', 'M_live[vmem:softmax]', 'n_programs']  → Pareto set: ['decode_bk256']
- objectives whose values are scenario values (status partial) for some candidate, so the Pareto filter compares scenarios, not complete figures: B[hbm_to_vmem], M_live[vmem:softmax], W_resource[reduce], n_programs

| metric | decode_bk256 | decode_bk512 | decode_bk256_h1 |
|---|---|---|---|
| F_total[useful] | 10601472 | 10601472 (+0) | 10601472 (+0) |
| F_total[executed_graph] | 10850304 | 10850304 (+0) | 10850304 (+0) |
| F_A_minus_F_E | -290107392 | -290107392 (+0) | -290107392 (+0) |
| C_rect | 36864 | 36864 (+0) | 36864 (+0) |
| C_pad | 36864 | 36864 (+0) | 36864 (+0) |
| rect_waste | 864 | 864 (+0) | 864 (+0) |
| B[hbm_to_vmem] | 971664 | 971664 (+0) | 6133632 (+5161968) |
| B[vmem_to_hbm] | 1536 | 1536 (+0) | 1536 (+0) |
| HBM_mat[K^n,V] | n/a | n/a | n/a |
| local[score_X] | 1024 | 2048 (+1024) | 1024 (+0) |
| local[accumulator_A] | 256 | 256 (+0) | 256 (+0) |
| local[v_tile] | n/a | n/a | n/a |
| M_live[vreg:softmax] | 1292 | 2316 (+1024) | 1292 (+0) |
| M_live[vmem:softmax] | 65696 | 131232 (+65536) | 65696 (+0) |
| M_peak[all] | 83368 | 166312 (+82944) | 83368 (+0) |
| n_programs | 3 | 3 (+0) | 24 (+21) |
| n_kv_block_visits | 144 | 72 (-72) | 144 (+0) |
| V_lifetime_stages | n/a | n/a | n/a |
| W_resource[exp] | 37008 | 36936 (-72) | 37008 (+0) |
| W_resource[reduce] | 73584 | 73656 (+72) | 73584 (+0) |
| b_k_max[vreg:binding_stage] | ? | ? | ? |
| T_LB[mxu] | ? | ? | ? |
| T_LB[path:hbm_to_vmem] | ? | ? | ? |
| T_LB[combined] | ? | ? | ? |
| T_pred[node_interval] | ? | ? | ? |

(`n/a` = this variant has no such metric; `?` = the metric exists but its value is unknown)

| candidate | mathematical legality | backend | model-feasible under declared budgets | binding stage |
|---|---|---|---|---|
| decode_bk256 | legal  | unknown (no compilation evidence) | None | None |
| decode_bk512 | legal  | unknown (no compilation evidence) | None | None |
| decode_bk256_h1 | legal  | unknown (no compilation evidence) | None | None |

| candidate | intermediates that appear | intermediates that disappear | MXU tile shape changes |
|---|---|---|---|
| decode_bk512 | — | — | QC_latent: [1, 256, 64] → [1, 512, 64]; QK_rope: [1, 256, 16] → [1, 512, 16]; PC_latent: [1, 64, 256] → [1, 64, 512] |
