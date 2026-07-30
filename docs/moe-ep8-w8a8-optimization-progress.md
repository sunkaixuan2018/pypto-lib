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
- End-to-end gains must be reported separately from component timings.
- Smaller scheduling improvements remain worthwhile when they preserve the
  stable correctness and measurement contract.
