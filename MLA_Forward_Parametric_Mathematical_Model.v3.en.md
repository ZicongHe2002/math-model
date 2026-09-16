---
title: "MLA Forward: General Parametric Mathematical Model"
subtitle: "Dynamic Parameter Reading → Symbolic Derivation → Stage-Wise Resource Model → Hardware Bounds and Optimization Directions"
lang: en
---

# 0. Requirement Corrections and Scope of This Version

**This project is to construct a family of models, not to optimize a fixed kernel for one set of known numbers.** The dimensions, dtypes, tiles, and test settings previously provided by the user are only examples of parameters that may occur. They are not production defaults, nor are they immutable baselines for the entire project. This version supersedes the fixed-task requirements in the old documents and English prompt that were built around that set of numbers.

The goal is to read the tensor metadata, configuration, algorithm path, and hardware information of the current invocation, and generate the computation amount, storage amount at each level, conditional register pressure, theoretical performance bounds, and candidate directions for this invocation. When different parameters are read on the next invocation, the same set of symbolic formulas is still used, without modifying the analyzer source code.

“Parameters are variable” means that shape, length, dtype, algorithm, and tile may differ across invocations; it does not require these parameters to follow any random distribution. “Dynamic reading” also does not mean guaranteeing that one TPU executable can accept all runtime shapes; parameter binding in this model and backend compilation strategy are different issues.

This version delivers mathematical and software interface specifications. It does not claim that a complete analyzer has already been implemented, that an Agent has been integrated, that real LLO has been obtained, or that TPU speedup has been measured. The model equations in this document are recommended engineering modeling methods; sources for external algorithm and tool facts are provided at the end.

## 0.1 One-Sentence Definition

$$
\mathcal M:(\theta,a,\kappa,\nu,\eta)\longmapsto\mathcal R.
$$

Here, $\theta$ is the task parameter set for this invocation, $a$ is the explicitly specified algorithm path, $\kappa$ is the execution strategy, $\nu$ is the numerical semantics, and $\eta$ is the hardware and calibration information. The output $\mathcal R$ contains symbolic expressions, bound numerical values, applicability conditions, uncertainty, and optimization rationale.

When only part of the parameters are known, output unbound expressions or multiple explicitly named scenarios; do not secretly use old example values merely to obtain a number.

## 0.2 Differences Between the Original Requirement and This Version

| Item | Requirement in This Version |
|---|---|
| Parameters | Read from actual input or supplied by the caller; examples belong only to separate examples |
| Mathematical object | A family of MLA forward models that can be instantiated repeatedly |
| Primary deliverable | Symbolic formulas, derivations, constraints, sensitivity, executable evaluation contract |
| Device | External hardware profile; v6e is one target configuration and is not embedded in the algorithm formulas |
| Agent | Consumes model reports, supplements missing evidence, and proposes experiments; does not guess numbers |
| Testing | Property tests across parameter families and randomized small-shape validation, not merely recomputing one answer |
| Optimization comparison | Different invocations may differ; candidate comparisons within one invocation preserve that task's semantics |

# 1. Parameter Space and Dynamic Reading

## 1.1 Symbol Table: Do Not Set Example Values as Defaults

| Category | Symbol | Meaning and Source |
|---|---|---|
| Task dimensions | $B,H,S_q,S_k$ | batch, head, query/KV lengths; from shape and effective-length metadata |
| MLA dimensions | $R_q,R_k,D_n,D_r,D_v$ | Q/KV latent dimensions, non-positional/positional branches, output head dimension |
| Visibility | $\mu_{b,h,i,j}$ | Whether visible; determined by semantics such as causal, offset, padding, and segment |
| Numerics | $s_x,\gamma,\nu$ | Storage bytes of each tensor, score scaling, compute/accumulation/conversion rules |
| Tiling | $b_q,b_k,b_{rq},b_{rk}$ | Q, KV, and projection-rank subtiles; if not provided, keep as variables or search variables |
| Execution | $\rho,\lambda,\sigma$ | Residency/buffering, layout, instruction and loop scheduling |
| Hardware | $P_u,W_e,C_m$ | Unit throughput, path bandwidth, storage capacity; device scope must be specified |

Dimensions are usually positive integers; $D_r=0$ may be treated as an explicitly supported no-positional-branch scenario. Zero length, missing Q latent, quantization, sparsity, multi-device sharding, and similar cases may use the corresponding model only when the adapter explicitly supports them; support must not be faked merely by relaxing shape validation.

## 1.2 The Seven Inputs Are Only an Adapter Protocol

First implement an adapter for the following semantic roles, with all variable values determined by the current input:

| Role | Normalized shape |
|---|---|
| $C_q$ / q_latent | $[B,S_q,R_q]$ |
| $C_k$ / kv_latent | $[B,S_k,R_k]$ |
| $Q^r$ / q_pe | $[B,H,S_q,D_r]$ |
| $K^r$ / k_pe | $[B,S_k,D_r]$ |
| $W^q$ | $[H,R_q,D_n]$ |
| $W^k$ | $[H,R_k,D_n]$ |
| $W^v$ | $[H,R_k,D_v]$ |
| Output $O$ | $[B,H,S_q,D_v]$ |

For example, recover $B,S_q,R_q$ from the shape of $C_q$, recover $H,R_q,D_n$ from the shape of $W^q$, and then use the other inputs for equality-constraint validation. Axes with the same dimension but different semantics must not automatically be treated as the same axis; transpose, fused weights, and differences in axis names must go through a declared role/axis mapping.

This is only a normalized representation and does not require the user to reorder the actual data. stride, storage offset, physical layout, and alias metadata are stored separately. If the actual operator already receives Q and expanded K/V, another entry scope must be selected, and projection must not be counted again.

Metadata mode should not read the values of large tensors. JAX's ShapeDtypeStruct is one possible carrier for describing shape/dtype, but the analysis core does not depend on JAX.[S5]

## 1.3 Shape, Effective Length, and Execution Length Are Not the Same Thing

A cache shape may represent maximum capacity rather than the effective KV length for the current invocation. Preserve separately: allocation extent, logical active extent, and scheduled extent. The first is used for static allocation bytes, the second for mathematically effective work, and the third for execution rectangles and padding counts.

For different lengths within a batch, record $S_{q,b},S_{k,b}$ and position mappings; packed sequences also require segment-matching relationships. Actual causal work cannot be inferred from tensor capacity alone. When effective length is unavailable, output a “capacity-execution scenario” and the missing fields rather than declaring on your own that the full capacity is active.

## 1.4 Fields That Cannot Be Uniquely Recovered from Shape

causal/non-causal behavior, position alignment, scaling, softmax exponential base, existing cache, whether projection is inside the measurement boundary, dtype conversion location, algorithm, tile, double buffering, and layout are not inevitable conclusions from shape. When external configuration conflicts with metadata, return a constraint conflict and list both sources; do not silently overwrite one with the other.

Standard scaling may be used as an explicitly named strategy: $\gamma=(D_n+D_r)^{-1/2}$. If the current model configuration supplies another valid scaling, use the actual semantics rather than imposing standard scaling on all inputs. Official MLA implementations may also add model-specific corrections on top of standard scaling.[S2]

# 2. Three Usage Modes of the Mathematical Model

**Symbolic mode:** when no actual numerical values are available, return $F_Q=2BHS_qR_qD_n$, storage expressions, tile constraints, and the dependency graph.

**Bound mode:** after the current metadata binds $\theta$ and the algorithm and some execution parameters are bound, evaluate expressions that can be evaluated; unbound items remain as variables/conditions.

**Calibrated mode:** read throughput data matching the device, compiler, and implementation characteristics to generate latency intervals and candidate comparisons; actual LLO is optional refinement evidence, not a prerequisite for constructing the first two modes.

Each result includes at least formula_id, expression, bindings, value, unit, scope, assumptions, evidence, coverage, and missing_fields. Source types distinguish input, derived, conditional, compiled, and measured; “complete” must state which declared model it is complete with respect to.

It is recommended to store mathematical expressions as a structured AST rather than allowing the program to execute arbitrary string eval. The evaluator supports integers, addition and multiplication, finite sums, ceil/floor, min/max, piecewise conditions, and symbolic functions; missing values are not forcibly converted to 0.

# 3. General Mathematical Definition of MLA Forward

The following definitions cover the current seven-input protocol; they do not automatically add the down projection that generates latents, RMSNorm, RoPE rotation generation, the full Transformer's $W^O$, or cache updates. When these stages are needed, explicitly add nodes through the scope.[S1]

When batch/head subscripts are omitted, matrices use the row-vector convention.

## 3.1 Expanded Path

$$
Q^n=C_qW^q,\qquad K^n=C_kW^k,\qquad V=C_kW^v.
$$

$$
X=\gamma\bigl(Q^n(K^n)^T+Q^r(K^r)^T\bigr)+M.
$$

$$
M_{b,h,i,j}=\begin{cases}0,&\mu_{b,h,i,j}=1,\\-\infty,&\mu_{b,h,i,j}=0.\end{cases}
$$

$$
P_{i,j}=\frac{\exp(X_{i,j}-m_i)}{\ell_i},\quad
m_i=\max_j X_{i,j},\quad \ell_i=\sum_j\exp(X_{i,j}-m_i).
$$

$$
O=PV,\qquad LSE_i=m_i+\log\ell_i.
$$

If an entire row is invisible, softmax itself does not have the usual normalization definition; the adapter must declare whether to raise an error or adopt an extended convention such as $O=0,LSE=-\infty$. Do not allow subtraction on an all-$-\infty$ row to produce NaN and then rely on accidental behavior.

## 3.2 Explicitly Named Absorbed Two-Step Path

$$
Q^n=C_qW^q,\qquad \widetilde Q=Q^n(W^k)^T.
$$

$$
X=\gamma\bigl(\widetilde Q C_k^T+Q^r(K^r)^T\bigr)+M,
\qquad Z=PC_k,\qquad O=ZW^v.
$$

Its attention-state width is $R_k$, not $D_v$. The scale and target output are still determined by the current task. By matrix associativity, the two paths express the same result over the real numbers; low-precision reassociation does not guarantee bitwise equality. The official code provides different execution paths for expanded and latent consumption, but each variant in this tool must be declared and accounted for independently.[S2]

If $W^q(W^k)^T$ is precomputed, it is another variant: the weight-merging work, $R_q\times R_k$ weight storage, and effective reuse count must be listed separately. The merged cache becomes invalid after a weight update, so preprocessing cannot be assumed to be always free.

## 3.3 Online Softmax State and Merge

For the already processed portion, maintain $m,\ell,A$, where the width $D_A$ of $A$ is $D_v$ in expanded and $R_k$ in the absorbed path above. Denote the value operand for the current block uniformly by $U_J$.

$$
m'=\max(m,\operatorname{rowmax}(X_J)),\quad
\alpha=\exp(m-m'),\quad E_J=\exp(X_J-m').
$$

$$
\ell'=\alpha\ell+\operatorname{rowsum}(E_J),\qquad
A'=\alpha A+E_JU_J.
$$

Finally $Y=A/\ell$; in expanded, $Y$ is $O$, while in absorbed, $Y$ is $Z$, which still needs projection. This kind of online normalization allows full score/P materialization in HBM to be avoided.[S3]

If two non-overlapping KV portions independently produce $(m_1,\ell_1,A_1)$ and $(m_2,\ell_2,A_2)$:

$$
m=\max(m_1,m_2),\quad a_i=\exp(m_i-m),
\quad \ell=a_1\ell_1+a_2\ell_2,\quad A=a_1A_1+a_2A_2.
$$

An empty portion uses an explicit branch: its weight is 0 and its state does not contribute; the implementation must avoid $-\infty-(-\infty)$ before evaluating the exponential. Splitting rows along Q does not require cross-Q softmax merge; splitting columns along KV requires the state merge above or an equivalent recurrence.

If a normalized output $Y_{old}=A/\ell$ is maintained, then for a non-empty new state it may also be written as $Y^{\prime}=(\alpha\ell/\ell^{\prime})Y_{old}+(E_JU_J)/\ell^{\prime}$. A normalized accumulator, an unnormalized accumulator, exp2, and different cast locations must generate different vector operation graphs. When exp2 is used, convert the logit base, and restore the final natural-log LSE according to the actual internal scale; do not merely replace the exp function name.[S4]

# 4. Visible Positions, Rectangular Execution, and Padding

## 4.1 Most General Effective-Position Count

$$
C_{\mathrm{valid}}=\sum_b\sum_h\sum_i\sum_j\mu_{b,h,i,j}.
$$

$C$ already includes batch and head, so $BH$ must not be multiplied again in the attention formulas. The $H$ factor may be factored out only when the same batch/head pairs share identical visibility.

For uniform lengths, unit-step positions, and $j\le i+\Delta$:

$$
C_{\mathrm{valid}}=BH\sum_{i=0}^{S_q-1}\max\{0,\min(S_k,i+\Delta+1)\}.
$$

Square causal prefill is the special case $S_q=S_k=S,\Delta=0$: $BH S(S+1)/2$; non-causal dense is $BHS_qS_k$. The position of a single-query decode depends on the actual cache offset; it must not be incorrectly counted as only one visible token merely because $S_q=1$.

## 4.2 Classification of Arbitrary Q/KV Rectangles

Let $I_i$ contain integer positions satisfying $q_i\le q<q_i+n_i$, and $J_j$ contain integer positions satisfying $k_j\le k<k_j+m_j$, as logically effective blocks. Under unit-step causal semantics:

- Fully future: $k_j>q_i+n_i-1+\Delta$.
- Fully visible: $k_j+m_j-1\le q_i+\Delta$.
- Otherwise partially visible; zero-length blocks are skipped separately.

Define the set of rectangles actually selected by the current strategy as $\mathcal E$. Its logical rectangular area and area according to backend execution extents are respectively:

$$
C_{\mathrm{rect}}=\sum_{b,h}\sum_{(i,j)\in\mathcal E}n_i m_j,
\qquad
C_{\mathrm{pad}}=\sum_{b,h}\sum_{(i,j)\in\mathcal E}\widehat n_i\widehat m_j.
$$

$\widehat n_i,\widehat m_j$ must come from the declared execution strategy or compilation evidence; the hardware array size must not be mechanically applied as padding in every dimension. Logical out-of-bounds does not mean the hardware actually reads out-of-bounds data from HBM.

Multiple scenarios such as full-scan, skip-future, and diagonal-subdivision may be returned instead of inferring that the original implementation already uses some pruning strategy. A causal flag alone cannot determine the actual number of execution rectangles.

## 4.3 Closed-Form Special Case of Causal Waste

For the scenario in which $S$ is divisible by a square subtile $p$, all fully future blocks are skipped, and diagonal blocks execute in full:

$$
C_{\mathrm{rect}}/BH=\frac{S^2+Sp}{2},\qquad
(C_{\mathrm{rect}}-C_{\mathrm{valid}})/BH=\frac{S(p-1)}2.
$$

This formula explains how geometric waste changes with $p$; it does not guarantee that latency is monotonic in $p$. Finer subdivision changes the numbers of reductions and updates. Such relationships are what the mathematical model needs to output, rather than an empirical ranking of one fixed set of tiles.

# 5. Stage-Wise FLOPs and Vector-Operation Model

## 5.1 General Primitive

For a conventional matrix multiplication with logical shapes $[M,K]\times[K,N]$, use the convention FMA=2:

$$
F_{\mathrm{matmul}}=2MNK.
$$

This is a commonly used workload-counting convention, not the exact number of scalar additions and not the number of instructions already executed by the hardware. The algorithm graph generates matrix shapes, and the shared primitive evaluator performs the counting. If actual execution includes padding, repeated decomposition, or recomputation, store scheduled/compiled work separately rather than overwriting the effective mathematical amount.

## 5.2 Expanded: Scenario with Full Reprojection from Latent

$$
F_Q=2BHS_qR_qD_n,
\quad F_K=2BHS_kR_kD_n,
\quad F_V=2BHS_kR_kD_v.
$$

$$
F_{QK,n}=2CD_n,\quad F_{QK,r}=2CD_r,\quad F_{PV}=2CD_v.
$$

$$
\boxed{F_E=2BH\left[S_qR_qD_n+S_kR_k(D_n+D_v)\right]
+2C(D_n+D_r+D_v).}
$$

Choosing $C_{\mathrm{valid}}$ yields conventional useful work; choosing $C_{\mathrm{rect}}$ yields work under that rectangular strategy without internal padding. Projection counts must also be specified; if projection is repeatedly executed inside the KV loop, the graph's loop multiplicity must increase the work, rather than still claiming that each token is projected only once.

## 5.3 Cache, Entry Scope, and Projection Counting

Define $N_Q^{proj},N_K^{proj},N_V^{proj}$ as the numbers of batch-head-token rows actually projected within this invocation. The general formula is:

$$
F_{proj}=2N_Q^{proj}R_qD_n
+2N_K^{proj}R_kD_n+2N_V^{proj}R_kD_v.
$$

In a full-projection scenario, $N_Q^{proj}=BHS_q$ and $N_K^{proj}=N_V^{proj}=BHS_k$. When an expanded cache already exists and only new tokens are projected, the K/V counts come from the new-token set; when K/V are passed in directly, they are 0 within this scope. Cache validity and lifetime must come from configuration and must not be assumed by the formula itself.

This allows prefill, decode, cache build, and cache consume to share the same primitive without incorrectly charging the entire historical KV again at every step, or conversely treating recomputation as free.

## 5.4 Independent Formula for Absorbed Two-Step

Under the same scope of “no existing projection, each item computed once”:

$$
F_A=2BH\left[S_qR_qD_n+S_qR_k(D_n+D_v)\right]
+2C(2R_k+D_r).
$$

Its accumulator width is defined independently from the PV contraction path and must not inherit expanded's $D_v$ as the latent width.

Subtracting further gives an interpretable path-selection condition:

$$
F_A-F_E=2BH R_k(D_n+D_v)(S_q-S_k)
+2C(2R_k-D_n-D_v).
$$

In square prefill, the first term is 0. Therefore, when this projection boundary and $C$ are the same, the relationship between the attention dimensions $2R_k$ and $D_n+D_v$ determines the sign of the conventional computation difference. Non-square tasks must also consider the first term; when cache or precomputation exists, substitute the corresponding projected-row counts again rather than applying this closed-form conclusion directly.

This is a computation comparison, not a runtime comparison. Reduced HBM traffic may offset more computation, and the reverse may also increase accumulator pressure.

## 5.5 Softmax Cannot Be Collapsed into Fixed MXU FLOPs

For example, for an actually executed $n\times m$ rectangle and an unnormalized recurrence, the logical operation graph before redundant-operation elimination can count the following separately:

| Operation | Reference logical count |
|---|---|
| Elementwise exponential $E$ | $nm$ times |
| row max comparisons | $n\max(m-1,0)$ times |
| row sum additions | $n\max(m-1,0)$ times |
| old/new maximum comparison | $n$ times |
| rescale exponential $\alpha$ | At most $n$ times; initial/empty blocks may be special-cased |
| Scale old accumulator and add new result | Each $nD_A$ times |
| Final row-wise normalization | Depends on actual implementation: division or reciprocal plus multiplication |

score combine, scale, mask, cast, broadcast, and transpose are listed separately. Exponentials may still be executed at masked positions, so the number of actual exp operations cannot be replaced directly by the number of valid cells. Logical counts of reduction comparisons/additions are also not actual vector-instruction counts; a backend mapping must be modeled separately.

Return $W_u$ for each operation type $u$, and later use the corresponding throughput $P_u$. Do not treat one exp as an equivalent cost to one BF16 matrix FLOP. Different online-softmax formulations change the non-matrix work.[S4]

# 6. Hierarchical Byte Model: Input, Materialization, Requests, and Residency

## 6.1 Logical Size of an Arbitrary Tensor

$$
M_{logical}(X)=s_X\prod_{d\in shape(X)}d.
$$

The dtype byte count is read from the current description; packed/quantized representations require additional models for packing, scale, zero point, and so on, and must not pretend to be physical allocations by multiplying the element count by fractional bytes. Actual shape, stride, and aliasing affect the allocation ledger; the sum of logical elements is not the union of address ranges.

The seven-input scenario with a uniform input storage dtype can be abbreviated as:

$$
M_{in}=s\left[BS_qR_q+BS_kR_k+BHS_qD_r+BS_kD_r
+HR_qD_n+HR_k(D_n+D_v)\right].
$$

$$
M_O=s_OBHS_qD_v,\qquad M_{LSE}=s_{LSE}BHS_q.
$$

An actual implementation should accumulate tensor by tensor and must not assume that all dtypes are the same. Reading the interface once and writing it once is only a conditional logical ledger, not an unconditional HBM lower bound under all cache/reuse conditions.

## 6.2 Whether Intermediate Tensors Are Materialized Is Strategy-Dependent

If expanded K/V are fully materialized in HBM:

$$
M_{KV}^{mat}=BHS_k(s_KD_n+s_VD_v).
$$

Each write and each read is counted separately on the corresponding path; if materialization does not exist, it must not be counted. If Q is also materialized, add Q reads/writes separately. latent absorbed can avoid these K/V tensors, but it must account for its own $\widetilde Q$ and $Z$.

Ordinary separated attention and tiled online attention may have the same principal matrix work but different HBM intermediate materialization; this is exactly the motivation for I/O-aware modeling.[S3]

## 6.3 General Formula for Path Traffic

For storage path $e$, let the bytes of each actual transfer event $r$ be $m_r$ and its execution count be $n_r$:

$$
B_e=\sum_{r\in transfers(e)}m_rn_r.
$$

At minimum, $e$ distinguishes HBM→VMEM, VMEM→HBM, VMEM→VREG, and VREG→VMEM. Sharing, residency, caching, materialization, and repeated loading are represented through events and execution counts rather than multiplying the total by an unexplained reuse factor.

A simple scenario example: if each query block independently scans all KV, there is no cross-query-block KV reuse, full scan is used, and input K/V are already expanded, then:

$$
B_{KV,read}=BH\operatorname{ceil}(S_q/b_q)
S_k(s_KD_n+s_VD_v).
$$

When only visible blocks are scanned, use instead:

$$
B_{KV,read}=\sum_{b,h}\sum_{(i,j)\in\mathcal E}
 m_j(s_KD_n+s_VD_v).
$$

This shows that for the same tensors, the read amount changes when $b_q$ and the residency strategy change; however, a real cache may change physical traffic, so both formulas must retain scenario labels. Shared $K^r$ may be accessed repeatedly across different heads/programs; mathematical sharing does not mean it is read only once.

# 7. Local Tensor, Layout, and Lifetime Model

## 7.1 No Longer Treat 3 MiB or 4 MiB as Constants

Let the current Q/KV computation subtiles be $b_q,b_k$, and the projection-rank subtiles be $b_{rq},b_{rk}$:

| Object | General expression |
|---|---|
| Q latent full window | $b_qR_qs_{Cq}$ |
| Q latent rank subtile | $b_qb_{rq}s_{Cq}$ |
| Q projection-weight rank subtile | $b_{rq}D_ns_{Wq}$ |
| Q projection accumulator | $b_qD_ns_{Qacc}$ |
| KV latent rank subtile | $b_kb_{rk}s_{Ck}$ |
| FP32/other-dtype score | $b_qb_ks_X$ |
| Exponential result E | $b_qb_ks_E$ |
| P/E sent to PV | $b_qb_ks_{Poperand}$ |
| Single compact row state | $b_qs_{state}$ |
| expanded accumulator | $b_qD_vs_A$ |
| absorbed accumulator | $b_qR_ks_A$ |

These expressions describe logical sizes. They are not all simultaneously live, and they are certainly not actual spill. Whether an object exists, whether score is overwritten by E, and whether the two QK branches are kept simultaneously all depend on the algorithm graph and alias relationships.

## 7.2 Layout Function and Hardware Profile

Define uniformly:

$$
M_{layout}(X)=\operatorname{LayoutBytes}(shape_X,dtype_X,\lambda_X,\eta).
$$

For an explicitly selected ordinary FP32 vector-tiling scenario, if each vector block contains $L_s\times L_l$ elements:

$$
M_{layout}(m,n)=s_f L_sL_l
\operatorname{ceil}(m/L_s)\operatorname{ceil}(n/L_l).
$$

$L_s,L_l$ come from the device and layout description and are not hard-coded in the general mathematical library. The Pallas documentation gives 8×128 as a typical 32-bit vector block on v6; this is only a hardware-specific reference, not a uniform rule for all dtypes and layouts and not the capacity of the entire register file.[S6]

For $[b_q,1]$, the coverage size under the layout above can be much larger than the effective scalar bytes. If the compiler uses a compact or replicated layout, calculate them separately rather than claiming that all row states are inflated in the same way.

## 7.3 Storage Objects and Live Sets

For storage category $c$ and time/graph cut $t$, merge aliases into a unique storage-allocation set $\mathcal A_c(t)$:

$$
M_{c,peak}=\max_t\sum_{a\in\mathcal A_c(t)}M_a.
$$

Lifetime is determined by produce, last-use, loop-carried state, and branch live-out. The peak computed from source-level tensors is only an assumption about the source graph; the real compiler may produce in tiles, fuse, recompute, or use other storage, so the analysis level must be stated.

VMEM and VREG are separate. Input residency windows, explicit scratch, and spill backing allocations may all occupy VMEM; vector temporaries occupy VREG. Hardware nominal capacity and compiler scoped allocation budget are also separate.

## 7.4 Interpretable Stage-Pressure Envelope

After fixing an algorithm/layout/lifetime assumption, the estimate for a stage can be written as:

$$
\widehat M_s(b_q,b_k)=a_s b_qb_k+b_s b_q+c_s b_k+d_s.
$$

$a_s$ comes from score/E/P values simultaneously retained in that stage and their dtypes; $b_s$ comes from Q, row state, and accumulator; $c_s$ comes from K/V; $d_s$ comes from fixed scratch. They should be derived from the object list and calibrated when necessary, rather than replaced by an arbitrary “risk threshold.”

If the available budget $R_s$ for that level is known and $a_sb_q+c_s>0$, the feasible range in that scenario is:

$$
b_k\le\operatorname{floor}\left(\frac{R_s-b_sb_q-d_s}{a_sb_q+c_s}\right).
$$

This is a capacity constraint under the declared model, not a necessary and sufficient condition for actual compiled spill/OOM. When the budget is unknown, preserve $R_s$ symbolically; when coefficients are unknown, provide multiple scenarios rather than generating pseudo-precise probabilities. ceil means round up, and floor means round down.

# 8. Mathematical Definition of Spill and Lower Bounds Under Limited Scope

## 8.1 Three Types of Quantities Must Have Different Names

$$
M_{spill,peak}=\max_t\sum_{a\in\mathcal A_{spill}(t)}M_a.
$$

$$
B_{spill/fill}=\sum_r bytes(r)\,executions(r).
$$

$$
\Delta T_{spill}=T_{schedule}(G_{with\ transfers})-
T_{schedule}(G_{comparison}).
$$

The comparison in the third expression must be legal and clearly defined. Simply deleting necessary transfers yields an unexecutable graph, so such a counterfactual can only be used as restricted-model analysis and must not be presented as actually achievable speedup.

The same spill space may be written and read repeatedly; static instructions, dynamic bytes, allocation peak, and exposed time are not interchangeable. Replacing automatic spill with explicit scratch may reduce labels while preserving the same movement.

## 8.2 Capacity Constraint on a Fixed Low-Level Graph Cut

If graph $G$, schedule, layout, and available VREG budget $R$ are known, recomputation is prohibited, and all other storage has been included in the model, then:

$$
\max(0,M_{live}(t)-R)
$$

constrains the amount of data at that cut that cannot remain in those registers. It is not automatically equal to newly added spill-write traffic: some data already has VMEM copies, and other graphs/schedules may eliminate that congested cut. Replacing the true low-level live set with the logical size of whole tensors destroys the interpretation of this constraint.

## 8.3 A Minimum-Movement Problem That Can Actually Be Solved

Let $\Omega(\theta,a,\eta)$ be the set of legal schedules/tilings/recomputations that satisfy the semantics, numerical constraints, and machine constraints of this invocation:

$$
B_{extra}^{opt}=\min_{p\in\Omega}B_{extra}(p).
$$

If the primary objective is speed, more useful formulations are:

$$
\min_{p\in\Omega}T(p),\qquad
\text{or}\quad \min_{p\in\Omega}B_{extra}(p)\ \text{s.t.}\ T(p)\le\tau.
$$

Small graphs can be solved by state search or integer programming, with the state containing register residency, VMEM copies, operation-completion state, and permitted recomputation. Output a proven lower bound, feasible solution, gap, and applicable parameter region. An ideal arrangement that has not been proven feasible cannot be called a globally optimal kernel.

Research on attention I/O complexity provides theoretical tools for well-defined multi-level/two-level storage problems, but these lower bounds cannot be converted directly into SPILL-label byte counts for a particular TPU compiler; mixed dtypes, MLA projections, and cache require the model to be delimited again.[S7]

With parameters alone, pressure expressions, capacity constraints for specified layouts, and conditional traffic can still be given; an unconfirmed minimum spill remains a symbolic optimization problem or unknown rather than causing the entire analysis to fail.

# 9. Hardware Parameterization and Latency Model

## 9.1 The Device Is Not a Hard-Coded Model Branch

Define a hardware profile: compute-unit types and counts, peak/effective throughput for each dtype, storage capacities, path bandwidths, layout parameters, startup costs, and the source of each item. Each numerical value must be labeled per-core, per-TensorCore, per-chip, or per-device, and values with inconsistent scopes must not be divided by one another.

The runtime environment may read some hardware fields, while other fields come from versioned device documentation or calibration. JAX TpuInfo is one optional adapter entry point, and its documentation explicitly distinguishes per-TensorCore information; it must not be assumed to expose all bandwidth, register capacity, or effective scheduling limits.[S8]

v6e can provide one profile, but the formulas do not embed any fixed TFLOP/s, VMEM capacity, or register count. Calibration results from the same hardware generation but different compilers also cannot be reused unconditionally.

## 9.2 Per-Resource Lower Bounds

For resource $u$, let the work be $W_u$ and the available throughput upper bound be $P_u$; for path $e$, let the transfer bytes be $B_e$ and the bandwidth upper bound be $W_e$; let the path-length lower bound of the true dependency graph be $CP$:

$$
T_{LB}=\max\left(\max_u W_u/P_u,\ \max_e B_e/W_e,\ CP\right),\qquad T_{actual}\ge T_{LB}.
$$

This expression requires each numerator to correspond to the same legal implementation/counting scope, and each denominator to truly be the upper bound of the corresponding resource. Dividing conditional HBM requests by peak bandwidth produces only a reference lower bound for that scenario and cannot be called an absolute limit for all algorithms.

If there are serial boundaries between stages that cannot be crossed, preserve the boundaries in the dependency graph or sum the lower bounds of stages known to be serial. Do not assume by default that all stages of the entire pipeline fully overlap. Roofline provides a compute/bandwidth-bound viewpoint, not an instruction-by-instruction scheduler.[S9]

## 9.3 Calibrated Prediction Model

For a matmul node $v$:

$$
\widehat t_v=\frac{F_v}{\epsilon_v(M,N,K,dtype,layout,\eta)P_{peak}}
+t_{startup,v}.
$$

$\epsilon_v$ is read from a matching bucket or fitted model. vector, reduce, layout, and memory use independent models; do not both count the same padding/low-utilization loss inside $F_v$ and then append the penalty again.

The complete time is produced by a resource-constrained scheduler:

$$
\widehat T=\operatorname{Schedule}(G,\{\widehat t_v\},\eta)
+T_{launch}+\varepsilon_{model}.
$$

If transfer timing is already included in the graph, do not add the same spill cost again. Without calibration, report only resource requirements, partial lower bounds, and assumed scenarios, and do not output precise actual latency; after calibration, report prediction intervals and validation error on new shapes rather than fitting only one input set.

# 10. Optimization Inferences the Model Should Proactively Provide

## 10.1 Parameter Sensitivity and Discrete Differences

The continuous relaxation of local score is $M_X=s_Xb_qb_k$, so:

$$
\frac{\partial M_X}{\partial b_q}=s_Xb_k,\qquad
\frac{\partial M_X}{\partial b_k}=s_Xb_q.
$$

Scaling both tile dimensions by a factor of $r$ scales logical score size by $r^2$; scaling only one dimension gives linear growth. A source-level logical accumulator grows linearly with $b_q$ and has no direct dependence on $b_k$, although scheduling and buffer lifetime may create indirect effects.

ceil, padding, and block classification contain discontinuities, so real candidate comparison should use the discrete difference $f(\kappa')-f(\kappa)$ rather than treating the derivative approximation as an exact optimality condition. The model simultaneously reports unchanged mathematical work, changes in rectangular waste, changes in traffic, and changes in peak pressure.

## 10.2 Output a Constraint Surface, Not a Fixed Best Tile

For example, return “under the declared softmax-live scenario, $a_sb_qb_k+b_sb_q+c_sb_k+d_s\le R_s$” and the feasible region after binding its parameters, then filter candidates using read traffic and reduction counts.

The objective for selecting candidates is:

$$
\kappa^{opt}(\theta,a,\eta)=\arg\min_{\kappa\in\mathcal K(\theta,a,\eta)}
\widehat T(\theta,a,\kappa,\eta).
$$

$\mathcal K$ varies with the current input and hardware. Mathematical legality, backend support, and model-estimated feasibility are labeled separately; model uncertainty cannot directly be judged as compilation OOM. Without calibration, return a Pareto set or hypotheses requiring validation rather than publishing an unsupported unique fastest configuration.

## 10.3 Example Template for Optimization Evidence

“Reduce the Q subtile” should report the changes in $M_X$ and $M_A$, whether the number of KV reads increases, and how the number of query programs and MXU shape change, rather than reporting only a decrease in spill risk.

“Delay V load” should report how much the lifetime overlap of V is reduced and whether prefetch is lost; without source/schedule evidence this is only a candidate hypothesis, not a located bug.

“Switch to absorbed” should report the independent $F_A-F_E$, latent-accumulator width, disappearance of expanded K/V materialization, and appearance of new intermediates; it must not claim to be faster solely because the cache is smaller.

# 11. Input and Report Interfaces: Read Instances, Do Not Fill in Fixed Template Numbers

## 11.1 Reading Flow

```text
Current tensor metadata / JSON / explicit function parameters
    → semantic-role mapping and axis normalization
    → shape / dtype / active-length consistency validation
    → current task parameter binding θ
    → algorithm / implementation / numeric binding or preservation as unknown
    → device profile and available calibration
    → construct symbolic graph and expressions
    → partial or complete evaluation
    → explainable report
```

Do not guess the parameters of the current invocation from filenames, seeds, old logs, or previous chats. Do not execute code when parsing user JSON; when reading a notebook, perform only static inspection or use an explicit adapter function, and do not automatically run arbitrary cells to obtain shapes.

## 11.2 Recommended API

```python
model = MLAForwardModel()  # not bound to one set of B/H/S/rank/tile

symbolic = model.describe(adapter="seven_input_latent", algorithm="expanded")

bound = model.bind(
    tensor_metadata=current_metadata,
    semantic_config=current_semantics,
    implementation=current_strategy,    # some fields may be missing
    numeric_policy=current_numerics,
    hardware_profile=current_device,
)

report = bound.analyze()
comparison = bound.compare(candidates=current_candidate_strategies)
```

These are interfaces to be implemented, not existing code that has already been run. The core object supports multiple binds and must not leak the parameters from the first binding into later invocations. Without a tile, it can still output global work expressions, input bytes, and symbolic local-tensor expressions; without a device, algorithm accounting is unaffected.

## 11.3 Parameter-Mapping Template

```yaml
schema: mla-parametric-v3
adapter: seven_input_latent
bindings:
  B: tensor.q_latent.shape[0]
  Sq: tensor.q_latent.shape[1]
  Rq: tensor.q_latent.shape[2]
  Sk: tensor.kv_latent.shape[1]
  Rk: tensor.kv_latent.shape[2]
  H: tensor.w_q_nope.shape[0]
  Dn: tensor.w_q_nope.shape[2]
  Dr: tensor.q_pe.shape[3]
  Dv: tensor.w_v.shape[2]
semantics: from_call_config
numerics: from_call_config
implementation: from_manifest_or_symbolic
hardware: from_runtime_or_profile
```

This is a field-parsing rule, not an arbitrary expression language that can be executed directly. The adapter uses whitelisted fields and indices, and then performs cross-constraint checks across all tensors. ragged active length is read separately; the shape mapping here cannot override it.

## 11.4 Single-Metric Format

```json
{
  "metric": "score_logical_bytes",
  "formula_id": "local.score.bytes",
  "expression": "b_q * b_k * s_score",
  "bindings": {},
  "value": null,
  "unit": "byte",
  "scope": "one logical score tile",
  "status": "symbolic",
  "missing_fields": ["b_q", "b_k", "s_score"],
  "assumptions": ["this algorithm materializes one logical score tile"],
  "not_equivalent_to": ["VREG allocation", "spill traffic"]
}
```

After binding actual parameters, retain the original expression and update bindings/value. Every conclusion can be traced back to a formula, source field, and algorithm scenario. Do not provide only the final number in the report while discarding its scope.

# 12. Compilation and Agent: Used for Refinement, Not as a Replacement for the Mathematical Model

## 12.1 The Role of Real LLO

The algorithm and parameters produce an explainable computation graph, but not the unique LLO of the original program. The actual source code, static parameters, layout, and target compiler determine a particular compilation result; abstract shape/dtype inputs can enter the compilation flow, but this does not mean that the program or target environment is unnecessary.[S10]

After integration, use the actual graph to correct projection counts, casts, aliases, loop bodies, and stage mappings. When LLO lacks loop bodies or widths, preserve the result as partial; the OpenXLA documentation explicitly warns that some lite protos omit internal loops, so complete dynamic spill for an invocation cannot be inferred from the few visible lines.[S11]

Parameter mode always remains independently usable; LLO alignment and microbenchmarks are later ways to reduce prediction uncertainty. The inability to obtain complete LLO must not be used as a reason not to construct the mathematical model.

## 12.2 Agent Input and Output

The Agent receives the current parameter bindings, equations for each stage, known/unknown fields, resource constraints, and residuals; it outputs verifiable hypotheses, candidate strategies, and required evidence. All formula evaluation and constraint checking are handled by deterministic code.

Recommended tools include only reading/analysis/comparison/optional validation: `describe_model`, `bind_metadata`, `analyze`, `compare_strategies`, `attach_compile_evidence`, `attach_calibration`. Automatic kernel writing and large-scale experiments are not prerequisite deliverables for this mathematical modeling effort.

## 12.3 Task Identity Is Not a Fixed Parameter Set

Define the task fingerprint for the current invocation as a normalized summary of the adapter, current dimensions/effective lengths, mask/position, scale, output scope, and numerical contract. Different invocations may have different fingerprints; baseline/candidate comparisons within the same invocation must use the same task fingerprint.

Therefore, “the project supports any legal parameters” and “the task cannot be changed to manufacture speedup” are not contradictory. rank/shape/scale are readable inputs, not optimization knobs that may be changed arbitrarily in every round. Performance analysis across shapes should be labeled a scaling study rather than claimed as same-task speedup.

# 13. Validation and Acceptance Across Parameter Families

## 13.1 Mathematical Validation

Randomly generate multiple legal small shapes covering $B>1$, different head counts, $S_q\ne S_k$, $D_n\ne D_v$, multiple ranks, $D_r=0$, causal offset, and non-divisible tiles. Explicitly enumerate visible positions and rectangle classifications, and compare them with the values obtained by binding the symbolic formulas.

Independently validate expanded/absorbed real-number algebra, dense/online softmax, different KV partitions, and merge using small-size high-precision computation. Do not use the same function both to generate expected values and to produce the result under test. Numerical precision and tolerance are explicitly specified for each test level; do not inherit old example seeds/tolerances as project-wide defaults.

## 13.2 Metamorphic Property Tests

When $B$ or $H$ is changed to another legal value, verify that work that should scale linearly changes accordingly while nonlinear/shared storage changes separately. Changing $S_q$ must not accidentally change KV weight size; changing $b_q$ must not change the conventional effective mathematical task, although it may change rectangular execution and read scenarios.

Changing the storage bytes of a dtype should change logical tensor bytes; this must not automatically scale softmax precision or matrix peak throughput by the same factor. Changes in hardware information must not change mathematical FLOPs, affecting only legality, layout proxies, or the time model.

For missing tiles, missing hardware, conflicting dimensions, and unsupported quantized/sharded inputs, validate partial output, explicit errors, and unknown states, avoiding fallback from abnormal inputs to historical examples.

## 13.3 Report and Model Quality

Acceptance focuses on the same model consuming multiple parameter sets without code modification, with every result carrying its expression and source. A fixed example may only be one of many regression fixtures and must not become the sole test oracle.

When measurement data becomes available later, validate on unseen shapes/algorithms/strategies and report prediction error, candidate ranking, and uncertainty. Do not fit on the test set and then report in-training accuracy as generalization performance; an uncalibrated pressure proxy must also not be named spill probability.

# 14. Implementation Priority and Replacement Notes

First priority: symbolic model, parameter reading, constraint validation, general formulas, hierarchical bytes, stage-pressure expressions, and cross-parameter testing. Second priority: hardware profile, sensitivity, conditional candidate comparison. Third priority: compilation/LLO/calibration and Agent-consumption interfaces.

It is not required to implement a new MLA kernel first, not required to connect a real LLM first, and not required to run on TPU first. Mathematical correctness, parameter coverage, and interpretability of the modeling engine are the core completion criteria for this version.

When replacing old materials, use this version of the document and the new English prompt in the same directory. The old B/H/S/rank/tile/seed/performance-sampling numbers may only be moved to an example directory labeled illustrative; they must not remain as the default task, required workload, or project-wide acceptance conditions.

This version is a design specification; the mathematical self-checks in the attachments test only equations and small-size algebra and do not prove that a BF16 target kernel, TPU performance model, or compiler integration has been completed.

# References and Evidence Boundaries

Retrieval date: 2026-09-08. The following are original papers, official code, or official documentation; online APIs may change, and the actual version should be recorded during implementation. The parametric model structure, pressure envelope, and software interfaces in this document are recommended designs and are not claimed to be existing official products.

[S1] DeepSeek-AI. DeepSeek-V2, mathematical background for MLA and decoupled RoPE. https://arxiv.org/html/2405.04434v5

[S2] DeepSeek-AI. DeepSeek-V3 inference/model.py, expanded and absorbed branches, model-specific scale. https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py

[S3] Dao et al. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. https://arxiv.org/abs/2205.14135

[S4] Dao. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. https://arxiv.org/abs/2307.08691

[S5] JAX. ShapeDtypeStruct API. https://docs.jax.dev/en/latest/_autosummary/jax.ShapeDtypeStruct.html

[S6] JAX. Writing TPU kernels with Pallas. https://docs.jax.dev/en/latest/pallas/tpu/details.html

[S7] Saha and Ye. The I/O Complexity of Attention, or How Optimal is Flash Attention? https://arxiv.org/abs/2402.07443

[S8] JAX. TpuInfo; TPU Hardware Reference. https://docs.jax.dev/en/latest/_autosummary/jax.experimental.pallas.tpu.TpuInfo.html ; https://docs.jax.dev/en/latest/pallas/tpu/hardware.html

[S9] OpenXLA. Roofline Model Tool. https://openxla.org/xprof/roofline_model

[S10] JAX. Ahead-of-time lowering and compilation. https://docs.jax.dev/en/latest/aot.html

[S11] OpenXLA. Custom Call Profiling. https://openxla.org/xprof/custom_call_profiling
