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

## Integration boundary

PyPTO `pl.jit.extern` can compile AscendC source into the persistent executor,
but it does not accept an already-built custom operator binary. A production
integration would therefore need:

1. a source-level persistent-kernel wrapper;
2. a complete and reproducible `TCubeTiling` configuration;
3. tensor and workspace ABI compatibility with the existing MoE buffers;
4. synchronization that does not introduce a nested kernel launch;
5. end-to-end correctness and the standard EP8 timing contract above.

The corrected standalone probe supplies component evidence, but it does not
yet supply all of those integration guarantees. The next integration step
should only proceed after the exact tiling and ABI can be reproduced without
guessing.

## Current conclusions

- The dominant bottleneck remains W8A8 weight movement; see
  [Flash MoE EP8 PMU Analysis Baseline](moe-ep8-pmu-analysis.md).
- CANN-style W13, SwiGLU, and quantization fusion is valid and shows measurable
  component-level benefit after fixing the INT8 NZ storage shape.
- A direct native PyPTO gate/up task merge exceeds the available Mat buffer and
  is not a viable implementation path in its current form.
- End-to-end gains must be reported separately from component timings.
- Smaller scheduling improvements remain worthwhile when they preserve the
  stable correctness and measurement contract.
