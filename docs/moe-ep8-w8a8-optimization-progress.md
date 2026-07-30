# Flash MoE EP8 W8A8 Optimization Progress

## Scope and measurement contract

This note records W8A8-only experiments for the standalone DeepSeek-V4 Flash
MoE path. The optimization target is useful, repeatable progress toward the
AscendC implementation; 470 us remains an aspirational target rather than a
reason to accept unsafe or unfair changes.

The stable end-to-end comparison uses:

- EP8 on physical devices `0,2,4,6,8,10,12,14`;
- 16 local experts per rank and balanced routing;
- two warmup iterations followed by eight measured iterations;
- one barrier before the standalone MoE timing region;
- the slowest rank in each iteration;
- unchanged W8A8 weights, activations, routing, and correctness criteria.

At commit `2c7392af841531f43db69cded522631c7bfaaa1f`, the eight stable
samples were:

```text
503.3, 502.0, 492.2, 497.1, 495.4, 499.8, 491.5, 493.7 us
```

The median was 496.3 us and the mean was 496.9 us. This is the production
baseline for later end-to-end experiments.

## Corrected fused W13 component probe

The probe in
`models/deepseek/v4-flash/cann_grouped_matmul_swiglu_quant_w8a8_probe.cpp`
evaluates a CANN-style W8A8 fused component:

```text
W13 grouped matmul -> clipped SwiGLU -> row quantization
```

The original custom operator appeared to leave 14 experts unwritten. A
topology diagnostic showed the actual failure more precisely: only the first
32 output rows were complete, across all output columns, while rows 32 through
255 retained their sentinel values. All 256 output scales were written.

The cause was an incorrect storage-shape rewrite in the custom operator API.
It always applied the INT4 NZ physical shape
`[E, N/64, K/16, 16, 8]` even when the weight dtype was INT8. The fusion
tiling code consequently reconstructed `N=512` instead of `N=4096` and emitted
only 32 complete output rows.

The tracked patch
`models/deepseek/v4-flash/patches/vllm_ascend_w8a8_weight_nz_storage_shape.patch`
uses the correct dtype-dependent physical shape:

```text
INT8: [E, N/32, K/16, 16, 32]
INT4: [E, N/64, K/16, 16, 8]
```

After applying the patch in an isolated custom-op installation, the exact
`M=256, K=4096, N=4096, E=16` W8A8 probe passed all 256 by 2048 output
elements and all 256 output scales. The eight measured samples were:

```text
208.9, 213.9, 203.9, 212.4, 204.2, 213.3, 205.8, 213.6 us
```

The median was 210.6 us and the mean was 209.5 us.

For context, the separate public CANN components at the same padded row count
measured 195.5 us for W13 and about 31.4 us for activation and quantization.
The corrected fused result therefore demonstrates approximately 16.3 us of
component-level headroom relative to those separate stages.

This is not an EP8 end-to-end result. It excludes dispatch, W2, combine, the
PyPTO task graph, and integration overhead. It establishes that W13 fusion is
numerically valid and locally useful in W8A8.

The corrected run was submitted as
`task_20260730_022814_48498219542`. Its archived log is:

```text
artifacts/moe-w8a8-fused-shapefix-20260730/
  moe_w8a8_fused_shapefix_m256_20260730.log
```

## Native PyPTO W13 task merge

A separate experiment attempted to merge the existing gate and up matmuls
from two two-block tasks into one four-block dynamic task. It deliberately
kept the current W8A8 tensors and downstream pipeline unchanged, so it tested
task-level co-scheduling rather than a new operator ABI.

The first version failed during compilation because the merged Mat buffer used
1,064,960 bytes, above the 524,288-byte limit. Reducing each branch to one
pipeline stage still required 1,056,768 bytes. The compiler retained both
branch-side matmul buffers instead of reusing them.

No NPU job was submitted for either version. Both commits were reverted and
the stable implementation was restored at
`a093d3da9de32a1087a114e7612c59013b30f215`.

This closes the straightforward dynamic-branch merge. A future native fusion
must share storage inside one purpose-built kernel or use a supported fused
operator; wrapping two existing matmul bodies in one task is not sufficient.

## Early-resolve expert pipeline

The stable implementation already enabled early resolve for W2, the W2
epilogue, and the surrounding communication tasks. Commit `3853319` extends
it to gate, up, activation, and quantization. The change does not alter tensor
shapes, arithmetic, routing, block counts, or W8A8 storage. It allows completed
producer tiles to enter the downstream expert stages before all sibling tasks
have resolved.

The first version of this experiment at commit `d97a132` passed correctness
and measured:

```text
490.7, 490.8, 497.0, 495.9, 489.8, 491.8, 487.0, 490.3 us
```

Its median was 490.75 us, 5.55 us below the historical 496.3 us baseline.
That change was originally reverted only because it missed a former 486 us
promotion threshold, not because of a dependency or correctness failure.

The restored candidate was tested again after a long period of shared-machine
activity. Correctness passed, but its samples included one isolated
interruption:

```text
501.0, 504.8, 492.6, 500.2, 493.7, 502.0, 500.3, 622.0 us
```

The median was 500.6 us. The other seven samples have median 500.2 us and mean
499.2 us. Because this did not reproduce the historical absolute latency, a
paired stable-baseline control was run under the same current conditions:

```text
503.1, 502.9, 505.1, 505.3, 499.9, 505.6, 498.4, 501.2 us
```

The paired baseline median was 503.0 us and mean was 502.7 us. The candidate
therefore improved the primary median metric by 2.4 us. Its
`host_union_mean_us` was also lower, 3901 us versus 3965 us for the paired
baseline. The isolated 622.0 us sample makes the candidate's unfiltered mean
unrepresentative, so both the raw samples and the robust comparison are
retained here.

The two independent comparisons show a small 2.4-5.55 us median benefit. The
annotations are retained as a low-risk scheduling improvement, but they do not
materially close the gap to the AscendC result.

The tasks and logs are:

```text
candidate task: task_20260730_024656_16188983557
baseline task:  task_20260730_035153_192839263

artifacts/moe-ep8-early-resolve-restored-20260730/
  moe_ep8_early_resolve_restored_20260730.log
  moe_ep8_baseline_control_20260730.log
```

## Per-expert task-graph interleaving

The original rank-5 level-2 swimlane was reprocessed using the device-side
task records rather than the wider AICPU residency records. Relative to the
first device task, its main routed-stage envelopes were:

```text
dispatch_gather:     90.46 -> 119.68 us  (kernel work 8.22 us)
gate/up combined:   115.04 -> 373.72 us
activation:         208.14 -> 389.86 us
W2:                 275.30 -> 454.70 us
combine starts:     458.54 us
```

The trace also shows that a ready expert can wait behind other Cube work
before W2 starts. Per-expert W2-ready-to-start waits ranged from about 18 us
on the final critical experts to 118 us on earlier experts. Expert 14 ended
the critical W2 tail at 454.70 us after waiting 20.22 us between activation
completion and W2 start.

The generated baseline graph submitted gate/up work for all 16 experts before
submitting any activation, quantization, or W2 work. Commit `1203b19` changes
only graph submission order by keeping each expert's unchanged
gate/up-to-W2 chain adjacent inside the existing parallel expert loop.
Arithmetic, W8A8 storage, tile sizes, task counts, and dependencies are
unchanged.

Compile-only validation passed. Generated orchestration inspection confirmed
the intended per-expert task order. Exact-even-card EP8 correctness also
passed. The candidate's critical-rank samples were:

```text
494.0, 556.8, 493.8, 499.9, 499.2, 555.9, 495.3, 495.1 us
```

The median was 497.2 us and `host_union_mean_us=4422`. Because the result had
two local 556 us tail samples and the expected improvement was small, a paired
restored control was submitted. Its first run suffered multi-millisecond
whole-rank interruptions and is retained only as machine-noise evidence. The
second control also had two obvious whole-rank interruption rounds, but its
critical-rank median remained 502.0 us:

```text
502.5, 499.1, 501.5, 772.6, 4054.0, 498.5, 499.6, 3430.4 us
```

Removing only the two whole-rank multi-millisecond rounds gives a six-sample
control median of 500.55 us, still 3.3 us above the candidate's unfiltered
497.2 us median. The full medians differ by 4.8 us. This supports a small
3-5 us scheduling gain, but the noisy controls do not justify claiming a
cleaner or larger improvement.

The graph-order change is retained as a low-risk scheduling improvement at
commit `b4e35fa`, subject to later confirmation in a quieter server window.
The tasks and archived logs are:

```text
candidate:           task_20260730_052459_40562572012
interrupted control: task_20260730_052939_8510110660
control rerun:       task_20260730_053056_13123114754

artifacts/moe-ep8-interleaved-expert-pipeline-20260730/
```

A level-3 swimlane was then captured at the retained commit. Using actual
kernel intervals (`receive + local_setup` through `kernel-duration`) rather
than task dispatch residency, the earliest routed W2 kernel moved from
267.26 us in the original trace to 232.60 us. The final three critical
experts' quant-to-W2 waits also fell from roughly 4.2-11.8 us to 0.7-3.4 us.
This confirms that interleaving changes the intended queueing mechanism.

The same profiled invocation also exposed the tradeoff: early W2 work competed
with unfinished W13 work. Its W13 kernel envelope extended to about 302 us
versus about 245 us in the original trace, and the profiled total was
555.24 us. Profiled absolute times are not substituted for the unprofiled
benchmark, but this explains why the observed end-to-end gain is small and
potentially noise-sensitive.

The next scheduling experiment, if pursued, should use expert waves rather
than fully interleaving every expert: submit W13 for a small group, then that
group's downstream chain. This keeps some W2 queue reduction while limiting
W2/W13 bandwidth contention. The confirming trace task was:

```text
task_20260730_053452_2770822610
artifacts/moe-ep8-interleaved-expert-pipeline-20260730/trace-rank7/
```

### Grouped expert waves

Two bounded follow-up experiments tested whether grouping experts could keep
the W2 queueing benefit while avoiding full W2/W13 contention. The generated
orchestration for wave size eight was explicitly verified as:

```text
8 experts W13 -> 8 experts downstream -> 8 experts W13 -> 8 experts downstream
```

Wave size eight passed exact correctness and produced a clean critical-rank
series:

```text
506.0, 499.4, 495.2, 496.0, 499.9, 497.0, 497.7, 496.9 us
```

Its median was 497.3 us and `host_union_mean_us=4698`. This removed the two
556 us critical-rank tails seen in the fully interleaved run, but did not
improve its 497.2 us median.

Wave size four also passed exact correctness. Its critical-rank samples were:

```text
504.0, 2433.3, 496.3, 501.3, 497.2, 497.0, 499.7, 696.2 us
```

The full median was 500.5 us. Removing only the two clearly interrupted
rounds gives a six-sample median of about 498.5 us, still no better than full
interleaving or wave size eight.

Because neither bounded wave size improved median latency and the wave
implementation required substantially more code than the four-line graph
ordering change, both wave candidates were reverted. The retained source is
again exactly the full per-expert interleaving implementation. The tasks and
logs are:

```text
wave 8: task_20260730_054534_5754332013
wave 4: task_20260730_054802_67978615673

artifacts/moe-ep8-expert-wave8-20260730/
artifacts/moe-ep8-expert-wave4-20260730/
```

### Routed W2 block fan-out

The interleaved trace contains 64 routed W2 AICore blocks: four blocks for
each of 16 experts. Commit `26cae83` tested whether reducing scheduler and
propagation pressure could help by changing only:

```text
W2_INNER: 4 -> 8
W2 blocks per expert: 4 -> 2
total W2 blocks: 64 -> 32
```

The arithmetic, W8A8 storage, output tiles, task count, and dependencies were
unchanged. Compile-only passed, and generated orchestration confirmed
`set_block_num(2)` for `exp_w2_mm`.

Exact EP8 correctness passed, but critical-rank latency regressed clearly:

```text
539.9, 538.2, 529.6, 524.4, 529.4, 509.0, 518.6, 528.2 us
```

The median was 528.8 us, the mean was 527.2 us, and
`host_union_mean_us=3943`. The clean, repeatable device regression shows that
64 blocks are not redundant scheduler fan-out: reducing to 32 blocks loses
the Cube concurrency needed to hide W2 weight movement. The candidate was
reverted in commit `f68fef6`, and lower W2 fan-out is closed.

```text
task_20260730_055538_89912515562
artifacts/moe-ep8-w2-fanout2-20260730/
```

## Exact fused-W13 tiling and source-level extern probe

The corrected custom operator was instrumented in an isolated source tree to
dump the complete fusion tiling for the same `M=256, K=N=4096, E=16` W8A8
shape. The host-selected topology is 24 Cube blocks and 48 Vector blocks. Its
main Cube tile is:

```text
singleCoreM=256, singleCoreN=256, singleCoreK=4096
baseM=128, baseN=256, baseK=128
depthA1=8, depthB1=8, stepKa=4, stepKb=4
dbL0A=2, dbL0B=2, dbL0C=1
shareL1Size=98304, shareL0CSize=131072
```

The remaining fusion fields are:

```text
groupNum=16, ubFactorDimx=4, ubFactorDimy=2048
actRight=0, groupListType=0, isSingleTensor=1, swigluLimit=10
```

All recorded layout fields are zero, and the matmul batch fields are one. The
diagnostic task was `task_20260730_041256_101367718490`. Its timing includes
per-call host logging and is not performance evidence.

A source-level `pl.jit.extern` probe then reproduced the fixed tiling and raw
single-tensor ABI. The first compile attempt reached the CCE wrapper and failed
only on eight typed-pointer to `GM_ADDR` conversions. Commit `0fe4ce7` fixed
those conversions; the wrapper then compiled in 1.16 seconds and entered
device runtime.

The device did not make forward progress. After 60 seconds the runtime
reported:

```text
sched_error_code=100
sub_class=S1:running-stalled
CCU instruction address check error on both Cube and Vector
```

Source comparison found that the official entry also declares
`KERNEL_TYPE_MIX_AIC_1_2`. Commit `37ea2ad` added that exact declaration so
the kernel-side and PyPTO launch topology both requested one Cube and two
Vector lanes. A second single-card probe produced the same CCU exception at
the same instruction locations.

The two runtime tasks were:

```text
task_20260730_042204_13647575525
task_20260730_042620_151990924491
```

Their logs are preserved under:

```text
artifacts/moe-w8a8-fused-extern-probe-20260730/
```

This closes direct reuse of the upstream synchronization-heavy source through
the current persistent `pl.jit.extern` executor. The result does not disprove
W13 fusion itself: the official custom-op path is correct and has a 16.3 us
component advantage. It establishes that its CrossCore Cube/Vector protocol
cannot be transplanted into the current executor by copying tiling and launch
metadata alone. No EP8 task was submitted for this invalid integration.

## Native activation and row-quant fusion attempt

A narrower native candidate kept gate/up and W2 unchanged, then tried to keep
complete SwiGLU rows task-local through row amax and INT8 quantization. This
would remove the FP32 `h_tile_fp32` GM round trip without importing the
upstream CrossCore protocol.

The candidate was stopped at compile time. Four-row tiles first exposed the
required 32-byte local-row alignment. Eight-row tiles satisfied that rule and
kept the local FP32 buffer to 64 KiB, but PTOAS rejected the move from the
computed `8 x 128` tile into the corresponding subview:

```text
pto.tmov expects A2/A3 non-mat tmov to use matching src/dst shapes
```

Pinning the computed tile's valid shape to the same constant `8 x 128` did not
change the generated mismatch. The production implementation was restored in
commit `c28e700`. No NPU task was submitted. This records a current DSL/codegen
boundary rather than evidence against activation/quant fusion as an algorithm.

## Activation Vector tile width

Because `exp_gate_up_act` accounts for 64% of routed AIV work, commit
`c64128e` tested a smaller tiling-only candidate:

```text
ACT_INTER_TILE: 128 -> 256
ACT_GATE_INNER: 4 -> 2
```

The number of activation blocks, arithmetic, tensor boundaries, and W8A8 data
were unchanged. The candidate passed compile-only validation and exact EP8
correctness. Its last-arriving-rank samples were:

```text
502.0, 497.4, 497.1, 497.9, 499.9, 499.2, 501.1, 496.8 us
```

The median was 498.55 us, the mean was 498.93 us, and
`host_union_mean_us=3969`. This is only about 2 us below the recent
early-resolve run and remains within observed run-to-run noise. It did not
justify another eight-card paired control, so the baseline 128-column tile was
restored in commit `e6e05d6`.

The task and archived log are:

```text
task_20260730_044451_23694672647
artifacts/moe-ep8-act-tile256-20260730/moe_ep8_act_tile256_20260730.log
```

## Quantization row fan-out

PMU showed that `exp_h_q` accounts for 18.5% of routed AIV work but emits only
one block per expert. Commit `52a28a4` split each independent 16-row quant task
into two aligned 8-row blocks. It preserved the amax, scale, cast sequence,
W8A8 data, and output layout.

The candidate compiled and passed exact EP8 correctness, but doubling the
quant task count made the critical-rank tail unstable:

```text
666.4, 721.1, 495.1, 497.8, 760.0, 497.4, 604.7, 575.6 us
```

The median was 590.2 us and `host_union_mean_us=4133`. This is a clear
regression rather than a small noisy movement. The original single quant task
was restored in commit `c170046`; no confirmation run is needed.

The task and archived log are:

```text
task_20260730_045123_26323554515
artifacts/moe-ep8-quant-rows8-20260730/moe_ep8_quant_rows8_20260730.log
```

## W2 matmul and epilogue fusion

Commit `f145dce` fused each routed W2 output tile with its FP32 dequantization,
route-weight multiplication, per-channel scale, and BF16 store. Static PTOAS
inspection confirmed that the separate routed `exp_w2_act` task disappeared
and the old full-width INT32 GM store/reload was replaced by a direct
L0C-to-Vector conversion followed by the final BF16 store. The W2 weight-side
`PH-MR-001` warning remained unchanged, so the experiment isolated the
intermediate and task-boundary effect.

Correctness passed, but the comparable last-arriving-rank samples regressed:

```text
519.1, 513.7, 512.2, 507.6, 512.7, 509.3, 506.2, 505.7 us
```

The median was 510.8 us, the mean was 510.8 us, and
`host_union_mean_us=4050`. Although the GM traffic reduction was real, the
mixed task coupled the Vector epilogue to each weight-bound W2 Cube tile and
removed the original overlap between the separate stages. The original
overlapped W2 epilogue was restored in commit `d2bb1bc`.

The task, generated PTO, and archived log are:

```text
task_20260730_050708_332166013200
artifacts/moe-ep8-w2-epilogue-fusion-20260730/
```

## Plain transposed W8 storage

Commit `6b3e7e6` extended the exact-correctness INT8 matmul probe with a
single-variable storage comparison:

- control: weight stored as `[N, K]`, consumed with `b_trans=True`;
- candidate: weight stored as plain `[K, N]`, consumed with `b_trans=False`.

This is a host-side dense transpose only. It does not claim to implement
FRACTAL_NZ or another device-native blocked layout. One single-card task ran
gate-shaped and W2-shaped controls and candidates in the same process. All
four cases passed exact correctness. The medians were:

```text
gate control:       62.8 us
gate transposed:    68.5 us  (+9.1%)
W2 control:         38.6 us
W2 transposed:      42.7 us  (+10.6%)
```

Plain `[K, N]` storage is therefore decisively worse for both projection
directions. The current `b_trans=True` path is retained. No full EP8 run is
needed because the component-level mechanism regressed in both shapes.

The task and archived log are:

```text
task_20260730_051314_35826651611
artifacts/moe-w8-transposed-storage-20260730/moe_w8_transposed_storage_20260730.log
```

## Integration boundary

PyPTO `pl.jit.extern` can compile AscendC source into the persistent executor,
but it does not accept an already-built custom operator binary. A production
integration would therefore need:

1. a source-level persistent-kernel wrapper;
2. a complete and reproducible `TCubeTiling` configuration;
3. tensor and workspace ABI compatibility with the existing MoE buffers;
4. synchronization that does not introduce a nested kernel launch;
5. a synchronization implementation compatible with the persistent executor;
6. end-to-end correctness and the standard EP8 timing contract above.

The corrected standalone probe supplies component evidence and the exact
tiling/ABI. The failed source-level probe shows synchronization compatibility,
not missing tiling data, is now the blocker.

## Current conclusions

- The dominant bottleneck remains W8A8 weight movement; see
  [Flash MoE EP8 PMU Analysis Baseline](moe-ep8-pmu-analysis.md).
- CANN-style W13, SwiGLU, and quantization fusion is valid and shows measurable
  component-level benefit after fixing the INT8 NZ storage shape.
- A direct native PyPTO gate/up task merge exceeds the available Mat buffer and
  is not a viable implementation path in its current form.
- Directly embedding the upstream fused source compiles, but its CrossCore
  protocol stalls in the persistent executor and is closed for now.
- Native full-row activation/quant storage currently hits an A2/A3 tile-move
  shape restriction, while a simple 256-column activation tile is noise-level.
- Splitting row quantization into two blocks per expert increases scheduler
  tail variance and is rejected.
- Direct W2/epilogue fusion removes the intended GM intermediate but also
  removes useful Cube/Vector overlap, producing a clear latency regression.
- A plain dense `[K, N]` host-side W8 transpose regresses both gate and W2
  component timings; it is not a substitute for true blocked NZ storage.
- Interleaving each expert's unchanged task chain reduces W2 queueing and
  shows a small 3-5 us paired median benefit, although the control window was
  noisy and needs later confirmation.
- Grouping experts into waves of eight or four does not improve the fully
  interleaved median; both larger rewrites are rejected and reverted.
- Halving routed W2 blocks from 64 to 32 regresses median latency to 528.8 us;
  the current W2 Cube fan-out is required and is retained.
- End-to-end gains must be reported separately from component timings.
- Smaller scheduling improvements remain worthwhile when they preserve the
  stable correctness and measurement contract.
