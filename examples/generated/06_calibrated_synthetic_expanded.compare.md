# Strategy comparison (task f257b3dc6072875a, expanded)

- verdict: intervals overlap: ranking inconclusive; treat as hypotheses requiring measurement
- useful work invariant across candidates: True
- Pareto objectives: ['B[hbm_to_vmem]', 'W_resource[reduce]', 'rect_waste', 'M_live[vreg:softmax]', 'M_live[vmem:softmax]', 'n_programs']  → Pareto set: ['bq64_bk128', 'bq128_bk64']
- calibrated ranking (predicted matmul interval, s): [['bq128_bk128', [1.8543237485714286e-05, 3.088647497142858e-05]], ['bq128_bk64', [1.8543237485714286e-05, 3.088647497142858e-05]], ['bq64_bk128', [2.2138355199999998e-05, 3.80767104e-05]]]  — intervals overlap: ranking inconclusive; treat as hypotheses requiring measurement
- objectives whose values are scenario values (status partial) for some candidate, so the Pareto filter compares scenarios, not complete figures: B[hbm_to_vmem], M_live[vmem:softmax], W_resource[reduce], n_programs, rect_waste
- kept but model-estimated infeasible under the declared budget at some level (model uncertainty, not a compilation verdict): bq128_bk128, bq128_bk64

| metric | bq128_bk128 | bq64_bk128 | bq128_bk64 |
|---|---|---|---|
| F_total[useful] | 596508672 | 596508672 (+0) | 596508672 (+0) |
| F_total[executed_graph] | 864026624 | 1115684864 (+251658240) | 864026624 (+0) |
| F_A_minus_F_E | 134479872 | 134479872 (+0) | 134479872 (+0) |
| C_rect | 2621440 | 2621440 (+0) | 2621440 (+0) |
| C_pad | 2621440 | 2621440 (+0) | 2621440 (+0) |
| rect_waste | 520192 | 520192 (+0) | 520192 (+0) |
| B[hbm_to_vmem] | 5046272 | 9306112 (+4259840) | 5046272 (+0) |
| B[vmem_to_hbm] | 819200 | 819200 (+0) | 819200 (+0) |
| HBM_mat[K^n,V] | 0 | 0 (+0) | 0 (+0) |
| local[score_X] | 65536 | 32768 (-32768) | 32768 (-32768) |
| local[accumulator_A] | 24576 | 12288 (-12288) | 24576 (+0) |
| local[v_tile] | 12288 | 12288 (+0) | 6144 (-6144) |
| M_live[vreg:softmax] | 91648 | 45824 (-45824) | 58880 (-32768) |
| M_live[vmem:softmax] | 28672 | 20480 (-8192) | 22528 (-6144) |
| M_peak[all] | 153088 | 90624 (-62464) | 97792 (-55296) |
| n_programs | 64 | 128 (+64) | 64 (+0) |
| n_kv_block_visits | 160 | 320 (+160) | 320 (+160) |
| V_lifetime_stages | 4 | 4 (+0) | 4 (+0) |
| W_resource[exp] | 2641920 | 2641920 (+0) | 2662400 (+20480) |
| W_resource[reduce] | 5222400 | 5222400 (+0) | 5201920 (-20480) |
| b_k_max[vreg:binding_stage] | 51 | 136 (+85) | 51 (+0) |
| T_LB[mxu] | 8.64026624e-06 | 1.115684864e-05 (+2.5165823999999994e-06) | 8.64026624e-06 (+0.0) |
| T_LB[path:hbm_to_vmem] | 6.307839999999999e-06 | 1.1632639999999999e-05 (+5.324799999999999e-06) | 6.307839999999999e-06 (+0.0) |
| T_LB[combined] | 1.32096e-05 | 1.32096e-05 (+0.0) | 1.3729792e-05 (+5.201920000000003e-07) |
| T_pred[node_interval] | [1.8543237485714286e-05, 3.088647497142858e-05] | [2.2138355199999998e-05, 3.80767104e-05] | [1.8543237485714286e-05, 3.088647497142858e-05] |

(`n/a` = this variant has no such metric; `?` = the metric exists but its value is unknown)

| candidate | mathematical legality | backend | model-feasible under declared budgets | binding stage |
|---|---|---|---|---|
| bq128_bk128 | legal  | unknown (no compilation evidence) | {'vreg': {'verdict': False, 'status': 'bound', 'stages_not_evaluated': [], 'stages_over_budget_independently_of_b_k': [], 'feasible_region_empty': False}} | {'vreg': {'binding_stage': 'pv', 'b_k_max': 51}} |
| bq64_bk128 | legal  | unknown (no compilation evidence) | {'vreg': {'verdict': True, 'status': 'bound', 'stages_not_evaluated': [], 'stages_over_budget_independently_of_b_k': [], 'feasible_region_empty': False}} | {'vreg': {'binding_stage': 'pv', 'b_k_max': 136}} |
| bq128_bk64 | legal  | unknown (no compilation evidence) | {'vreg': {'verdict': False, 'status': 'bound', 'stages_not_evaluated': [], 'stages_over_budget_independently_of_b_k': [], 'feasible_region_empty': False}} | {'vreg': {'binding_stage': 'pv', 'b_k_max': 51}} |

| candidate | intermediates that appear | intermediates that disappear | MXU tile shape changes |
|---|---|---|---|
| bq64_bk128 | — | — | QK_nope: [128, 128, 48] → [64, 128, 48]; QK_rope: [128, 128, 16] → [64, 128, 16]; PV: [128, 48, 128] → [64, 48, 128] |
| bq128_bk64 | — | — | QK_nope: [128, 128, 48] → [128, 64, 48]; QK_rope: [128, 128, 16] → [128, 64, 16]; PV: [128, 48, 128] → [128, 48, 64] |
