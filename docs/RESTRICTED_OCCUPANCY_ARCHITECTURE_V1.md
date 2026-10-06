# Restricted occupancy architecture v1

Non-executable prospective specification for M0-occupancy-development-v1.
Architecture freeze identity: SHA256
7193c6c7f789ceda45b447e75a199657d21f6279ea84be5544034a2c9ce4b186.
Route identity: SHA256
a47df0e39e0f69ce0ca137e371cb31c84de5fcd15aa767ccb66b2dca92624329.
These identify the frozen specification, not a model implementation or empirical result.
No Gate 0D, collection or comparative launch authority follows from this document.

## Causal information and timing

Each example is exactly the prospective three-observation prefix, followed by one forecast. At each observation t, both conditions receive the same three 32x32 binary masks M_i, observed-ever flags p_i, current-visibility flags v_i, twelve normalized boundary values B_ij^H/B_ij^V for the six ordered i!=j pairs, and the three-component executed command e_t leading into this observation. e_0 is zero. p_i is one only after that opaque token has been observed; v_i is one iff its current mask is nonempty. A remembered but currently invisible token has p=1, v=0 and a zero mask. Define H as neighboring cells (row,col) and (row,col+1), and V as (row,col) and (row+1,col). For i!=j, count a neighbor if its two nonzero observed masks carry i and j in either order; store that shared count symmetrically in both ordered channels B_ij and B_ji. H/V denote neighbor-axis orientation (H detects a vertical interface), not boundary ownership or a directed occluder relation. Divide each count by32*31=992, including zero counts. Six ordered pairs each carry H,V, exactly12 values; diagonal is implicitly zero. If either token is never observed or currently invisible, both counts are zero. Disjoint validated masks ensure no two distinct arrival tokens share their first occupied cell; malformed overlap/tie fails closed. Boundary ownership is UNKNOWN; there is no inferred hidden edge. Observed optical mask adjacency is the sole boundary source.

The announced three-component command a is used only at the final readout; it is not executed into recurrent state. Prefix observations and executed commands arrive in causal order. All conditions take exactly one update per observation, with no target observation, elapsed-time shortcut, skipped invisible updates, old-mask cache, prefix replay or future inventory. Elapsed prefix position is determined by the update schedule, with no additional learned feature or persistent counter required for prediction. Three steps are the sole resource-accounting example length; longer-prefix use is outside this freeze. Commands enter as their exact common three numeric command components, converted to float32 without trainable scaling, clipping, command-class embedding or conversion to world pose. This note supplies no new metric modality.

Opaque handles are equality/indexing bookkeeping only. Numeric/string token encodings, simulator roles/IDs, seed/family/generation records, hidden catalogue/extent, pose, depth and world coordinates are denied to both learners. Masks are optical image-plane observations. Missing, malformed, extra/future or privileged channels must fail closed before learner access. This freezes an oracle-association comparison, not learned identity recovery.

## Streaming six-permutation policy

Reserve three empty base slots before the first observation. As tokens first become observed, assign them causally to unused base slots; simultaneous arrivals are ordered by their first occupied raster cell in row-major order, never token spelling or role. Do not preassign hidden identities. Instantiate all six fixed bijections of the three base slots from frame zero, including currently empty slots. Record first-seen observation index and observed handle only in bookkeeping. First-seen index and base/permutation slot index are not numeric learner features. A later observed arrival fills its already reserved base slot in each corresponding stream. Initialize every recurrent float to zero. Never change a stream's bijection or restart/replay it.

Both training and evaluation maintain all six streams, run identical shared weights, remap each stream's three probabilities back to base-token order, and average probabilities (not logits). Train the common proposal loss on that averaged forecast; no stream is a new data example or independent replicate. Remapping an arbitrary tie/order change only permutes the six streams, so the averaged deterministic prediction is exactly slot-equivariant in mathematical arithmetic; freeze numerical checks at elementwise abs(x-y)<=1e-6+1e-5*abs(y) for finite float32 values. For state checks, match corresponding permutation streams under a causal input relabeling: inverse-remap candidate node states, compare global states directly, and compare dense states directly after the corresponding stream reindexing. For forecast checks, compare inverse-remapped averaged probabilities at the same tolerance. No averaging of states is required. A violation fails the permutation check; the tolerance is not tuned after data exposure. Dense recurrence need not be internally equivariant.

Encode each current base-token mask once per observation using the shared encoder, and share its transformed feature among the six permuted streams. This is identical in every learned condition. Encoder sharing across streams does not cache history; gradients from all six uses accumulate into the same encoder. Encode each observed-ever token, including its zero mask when currently invisible; never-observed slots receive constant zero features without an encoder call. Run all three recurrent slots and all six streams without recurrent/edge sparsity shortcuts; p-gating prevents empty reserved slots from carrying hidden inventory. A comparator joint state has no token-specific substate: reserved-slot input/edges are zero, but its shared64-float state is allowed to update. Forecast channels for unobserved slots are masked out; later targets cannot create inventory.

## Exact paper architecture

All trainable arrays and state are float32. There is no dropout, normalization layer, positional/slot embedding, residual feedforward block, auxiliary reconstruction head, pretrained encoder, or unlisted trainable tensor. Standard dense affine operations, ReLU, tanh, sigmoid, addition and multiplication suffice. All listed biases are trainable; initial biases are zero. Every listed weight lies on an actual input-to-forecast path. Widths were chosen prospectively to meet the 100000-parameter budget; large readout widths are effective nonlinear computation, not evidence of optimality and not dead-parameter padding.

Notation: D_(d,h)(x)=Wx+b, parameter count h(d+1). The mask encoder shared over slots and streams is x_i=p_i*tanh(D_(32,16)(ReLU(D_(1024,32)(flatten(M_i))))), yielding exactly sixteen features. Its architecture and parameter initialization are identical between candidate and comparator; weights are separately trainable during each fit.

Use a reset-after GRU(d,h) with three input matrices, three recurrent matrices and separate input/recurrent bias vectors: r=sigmoid(W_r x+b_ir+U_r h+b_hr), z=sigmoid(W_z x+b_iz+U_z h+b_hz), n=tanh(W_n x+b_in+r*(U_n h+b_hn)), h_new=(1-z)*n+z*h. Products are elementwise. Its parameter count is 3h(d+h+2). This exact convention fixes biases as well as recurrence.

Candidate, per stream: three shared-weight token states h_i in R^16 and one global g in R^16, hence exactly64 recurrent floats. At t, use previous h_i/g and current x_i:

- Sender feature s_j=p_j*tanh(D_(32,16)([h_j_old,x_j])); calculate once for each of three senders.
- w_ij=(B_ij^H+B_ij^V)/2, m_i=(p_i/2)*sum_(j!=i) w_ij*s_j. These are learned sender messages weighted by measured current optical boundary evidence. There are no self edges or hidden edges; no boundary implies a zero message. All six directed pair products are executed.
- h_i_new=p_i*GRU(53,16)([x_i,m_i,g_old,p_i,v_i,e_t],h_i_old). This input is 16+16+16+2+3=53. Visibility-zero remembered tokens continue updating; never-observed slots stay exactly zero.
- Let H=(h_1_new+h_2_new+h_3_new)/3, P=sum_i p_i/3, V=sum_i v_i/3, C_H=sum_(i!=j) B_ij^H/6, C_V=sum_(i!=j) B_ij^V/6. g_new=GRU(23,16)([H,e_t,P,V,C_H,C_V],g_old). This input is16+3+2+2=23. Global recurrence executes once at every observation even with no current visible token; its pooled flags/boundaries then reflect only observed evidence. There is no future-catalogue gating. Never-observed token states are p-gated to zero; observed invisible states update without reset. No extra memory is retained.
- At final readout, c=tanh(D_(1060,16)(ReLU(D_(35,1060)([H,g_new,a])))). For each i, q_i=sigmoid(D_(154,1)(ReLU(D_(34,154)([h_i_new,c,p_i,v_i])))). This is a shared token readout with a learned pooled context, not a mask-to-output bypass. Output p_i*q_i, with unobserved channels excluded by the common inventory rule. Recompute H from state at readout rather than retaining an extra learned state.

Comparator, per stream: one joint z in R^64. Concatenate [x_1,x_2,x_3,p_1,v_1,p_2,v_2,p_3,v_3,B_12^H,B_12^V,B_13^H,B_13^V,B_21^H,B_21^V,B_23^H,B_23^V,B_31^H,B_31^V,B_32^H,B_32^V,e_t], dimension48+6+12+3=69, and update z_new=GRU(69,64)(input,z_old). Joint readout is sigmoid(D_(529,3)(ReLU(D_(73,529)([z_new,a,p_1,v_1,p_2,v_2,p_3,v_3])))), dimension64+3+6=73; p-gate each output as above. It has trainable joint dense mixing and joint decoding. It receives every candidate observation, not only pooled boundary values.

The final heads receive compressed recurrent states, latest public bookkeeping and announced action. Neither has a direct current-mask/encoded-feature/boundary shortcut to the forecast. Boundaries affect candidate token messages and global recurrence, and comparator joint recurrence. This is a comparison of the entire specified relational architecture against the entire specified dense architecture: recurrence factorization, sharing, mixing and decoding differ together. Equal budgets do not identify a message-specific effect.

## Exact used trainable parameter arithmetic

| Component | Formula | Candidate | Comparator |
|---|---|---:|---:|
| Shared mask encoder |32*(1024+1)+16*(32+1)|33328|33328|
| Shared sender transform |16*(32+1)|528|0|
| Shared token GRU |3*16*(53+16+2)|3408|0|
| Global GRU |3*16*(23+16+2)|1968|0|
| Pooled readout context |1060*(35+1)+16*(1060+1)|55136|0|
| Shared token head |154*(34+1)+(154+1)|5545|0|
| Joint GRU |3*64*(69+64+2)|0|25920|
| Joint head |529*(73+1)+3*(529+1)|0|40736|
| Total ||99913|99984|

Sharing means parameters are counted once, not once per token or permutation. Candidate is0.087% below100000, comparator0.016% below100000; both meet +/-1%, differing by71 used parameters. No disconnected parameters count. Later authorized source checks must verify every tensor receives a valid gradient under a lawful nondegenerate synthetic check; accidental disconnected/dead code is a failure, not padding credit.

## Prospective operation accounting

Count a scalar multiply or add/subtract/divide as one operation; dense dot products use the conventional2*d*h proxy plus h bias additions (this deliberately does not subtract the final accumulation add). Each ReLU comparison, sigmoid or tanh is one named nonlinear operation in this proxy. A sigmoid/tanh proxy unit is not a claim about machine instructions, CPU latency or equal transcendental costs. Report named nonlinear calls separately. Concatenation/index remapping/zero assignment/memory copies are byte work rather than FLOPs and must be reported in memory/time accounting. No residual biases or activations are omitted: this architecture contains no residual blocks; the GRU interpolation arithmetic is included.

D_(d,h) cost=2dh+h. ReLU/tanh/sigmoid cost=h. GRU(d,h) cost=6h(d+h)+17h: six bias additions per hidden coordinate, three input/recurrent combination additions, one reset multiply, four final interpolation operations, three nonlinear calls. All slots/edges/streams run at fixed shape.

| Work | Arithmetic | Forward proxy operations |
|---|---|---:|
| Encoder per token |(2*1024*32+32+32)+(2*32*16+16+16)+16 p-gate|66672|
| Encoder per observation |n_t*66672; n_t=sum_i p_i|66672*n_t|
| Candidate shared sender transforms |3*(2*32*16+16+16)|3168|
| Candidate message weighting/aggregation |6*2 weight ops +48 sender p-gates +96 weighted products +48 sums +96 output scaling/gating|300|
| Candidate three token GRUs |3*(6*16*(53+16)+17*16)|20688|
| Candidate token-state p-gates |3*16|48|
| Candidate global summaries |48 H ops +6 P/V ops +12 boundary-summary ops|66|
| Candidate global GRU |6*16*(23+16)+17*16|4016|
| Candidate update, each stream |sum above|28286|
| Candidate final context |(2*35*1060+1060+1060)+(2*1060*16+16+16)|110272|
| Candidate three token heads |3*((2*34*154+154+154)+(2*154+1+1))|33270|
| Candidate final pool/output gate |48+3|51|
| Candidate readout, each stream |110272+33270+51|143593|
| Comparator update, each stream |6*64*(69+64)+17*64|52160|
| Comparator readout, each stream |(2*73*529+529+529)+(2*529*3+3+3)+3 output gates|81475|
| Six-output average, both |3*(5 additions+1 division)|18|

For exactly three observations and one six-stream averaged forecast, let N=sum_(t=0..2) n_t be the total number of observed-ever mask encodings,0<=N<=9. N is shared across all conditions and is determined only by causal observed inventory:

- Candidate F_C(N)=66672*N+6*(3*28286+143593)+18=66672*N+1370724 operations.
- Comparator F_D(N)=66672*N+6*(3*52160+81475)+18=66672*N+1427748 operations.
- At N=9, F_C=1970772 and F_D=2027796, a2.8121% comparator-denominator difference; max/min=1.028935.
- At N=0 (conservative arithmetic endpoint, not an admitted episode), F_C=1370724 and F_D=1427748, difference3.9940%; max/min=1.041601. Every possible shared causal inventory N lies between these endpoints and passes both10% conventions. There is no need to learn hidden inventory to budget compute.

These are full example costs with encoder and all six streams, not one-state or encoder-excluded costs. Matching is specific to the frozen three-observation example; different lengths need renewed accounting before any broader claim.

Named nonlinear operations: encoder48*N; candidate six streams*(3*(48 sender tanh+144 token-GRU+48 global-GRU)+1060 context ReLU+16 context tanh+462 token-head ReLU+3 sigmoid)=13566, hence13998 at N=9. Comparator six streams*(3*192 GRU+529 ReLU+3 sigmoid)=6648, hence7080 at N=9. Nonlinear composition differs substantially; treating nonlinear calls as unit proxy operations is an explicit matching-convention limitation. Actual CPU time is separate evidence.

Prospective training planning uses3*F(N) for forward plus two forward-equivalent reverse costs, excluding optimizer/loss and memory traffic: at N=9 candidate5912316 and comparator6083388, the same2.8121% difference. This is an estimate, not an exact automatic-differentiation backward count. Common Brier/stratum loss and six-output gradient averaging have equal shape; optimizer arithmetic differs negligibly with71 parameters but must be included in later reports. Exact actual forward/backward operator accounting, activation buffers, optimizer state and wall time require the later authorized source package. Do not broaden this to a training-compute claim based only on3F. If reviewed actual per-example forward/training accounting violates the frozen10% allowance, stop as fairness-INCONCLUSIVE or seek a new prospective reviewed specification before fresh evaluation exposure. No outcome-based widths, library/kernel or sparsity policy changes.
## State, buffers and diagnostics

Each stream retains exactly64 recurrent float32 values for both architectures: candidate3*16+16, comparator64. Six streams require384 floats,1536 logical bytes, even if an episode has fewer observed tokens. This excludes expressly charged metadata and temporary buffers: observed-token equality map (at most3 handles with causal base assignment), six constant bijections, p/v flags, current twelve boundary scalars, current executed and announced commands, 3*1024 current-mask values, shared3*16 encoder outputs, and per-stream message/readout activations. For example, candidate1060-wide context is a transient activation, never extra recurrent memory. Release preceding current masks/features between observations; readout has no historical raster store. Training backprop retains/recomputes lawful prefix activations only for gradient calculation, not predictor readout; report their memory separately. Six streams may execute sequentially with shared temporary workspace. Parameter bytes, gradients, optimizer/checkpoint state, trainer batches, prediction buffers and process overhead all count toward actual peak process RAM/disk;1536 bytes is not a total-memory claim. Charge CPU work for all six streams despite shared encoding.

Candidate action-zero diagnostic: identical99913 parameters,384 recurrent floats, operations/order/readout, but replace both executed and announced commands by zero before access. Candidate memory-reset diagnostic: identical architecture/budget, reset all recurrent floats to zero before every observation, still execute all updates and six readouts; final readout sees only the last update. These train from scratch with the same proposal examples/order/loss/updates. Neither diagnostic isolates messages or supplies an additional charter baseline. Cheap controls retain their separate proposal role and accounting.

## Freeze and lawful handoff

Independent review must reconcile this exact specification with the corrected v2 proposal, its common command normalization and permission contract, loss/training policy, endpoint uncertainty rule, and approved source identity. Freeze dimensions, equations, initialization policy, operation convention, three-step/six-stream schedule and budget checks BEFORE fresh data collection/exposure. Proposal seed allocation remains later; no seed was created here. Freeze initialization now: every dense/input/recurrent weight is independently uniform[-1/sqrt(fan_in),+1/sqrt(fan_in)] with fan_in equal to that matrix input width (recurrent matrices use hidden width); every bias is zero. Derive each uniform draw from the first64 bits, little-endian, of SHA256 over a canonically length-prefixed domain `M0-occupancy-development-v1/init-v1`, later authorized initialization seed, logical component/tensor name and row-major element index. Canonical bytes are the concatenation of UTF-8 domain and component/tensor name each preceded by its unsigned64 little-endian byte length, then unsigned64 little-endian initialization seed and element index; tensor names are encoder/layer1/W, encoder/layer2/W, candidate/sender/W, candidate/token-GRU/{Wr,Wz,Wn,Ur,Uz,Un}, candidate/global-GRU/{Wr,Wz,Wn,Ur,Uz,Un}, candidate/context/{layer1,layer2}/W, candidate/token-head/{layer1,layer2}/W, comparator/joint-GRU/{Wr,Wz,Wn,Ur,Uz,Un}, comparator/joint-head/{layer1,layer2}/W. Bias arrays are separately named but fixed zero, with no draws; u=(integer+0.5)/2^64 in float64, map to the interval, round once to float32. Encoder namespace is identical across conditions; candidate component namespaces are identical across its diagnostics; these initializations are also paired across both training budgets. Other model components have explicit architecture-specific names. No draws use geometry, labels, opaque handles, splits or evaluation. This pins a distribution/deterministic initializer without allocating a seed or executing it.

Training numerics are also prospective: float32 weights, activations, recurrent states, gradient accumulation and Adam first/second moments, CPU dense operations, no mixed precision, quantization, dropout, weight decay, gradient clipping or AMSGrad. Three-step full backpropagation through each example; inverse-remapped six-stream probabilities average before the common loss, with all six gradients accumulated and no stop-gradient encoder sharing. Average each example's nonempty absent/visible stratum Brier means equally, then average16 example losses; missing strata provide no zero-risk credit. Take one Adam update with lr0.001, beta1=0.9, beta2=0.999, epsilon=1e-8: m=beta1*m+(1-beta1)*grad; v=beta2*v+(1-beta2)*grad^2; bias-correct at update index k starting1; theta=theta-lr*mhat/(sqrt(vhat)+epsilon). Moments start zero. Reset recurrence at each prefix/example and checkpoint only after update1000; batch/order/geometry schedule and all24 fit policy remain v2's. Use one CPU thread and deterministic operations in later source. Explicitly order cross-stream probability/gradient accumulation, token pooling, message aggregation and example/stratum/batch reductions by the declared slot/pair/stream order and row-major element order; do not leave these reductions to unordered scheduling. Dense dot-product internal summation remains the selected vendor BLAS/runtime implementation, not a requirement for new handwritten linear algebra. Record CPU, OS, runtime/framework and BLAS vendor/version, thread settings and deterministic-operation settings in provenance. Repeated runs on that recorded environment must satisfy the pinned numerical checks; cross-host bit identity is not promised. These details do not change the operation proxy. Reject unsupported nondeterministic operations or provenance gaps as failed evidence; reject nonfinite input, loss, gradient, state or probability as failed evidence, never silently clip or replace. Adam cost, gradient/activation storage and actual backward operations must be measured/reported by later authorized work, not treated as extra recurrent memory.



## Memory-reset computational clarification

The later memory-reset diagnostic algebraically multiplies recurrent state by zero
before each observation without detaching the previous graph. This removes memory
information while preserving the declared three-step full BPTT computational policy.
Charge the reset multiplication work in its actual cost report. Report diagnostic
costs explicitly; the fixed10% actual compute match is the relational-versus-dense
primary fairness requirement. This is a prospective specification, not model code.
