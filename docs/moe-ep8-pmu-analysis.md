# Flash MoE EP8 PMU Analysis Baseline

## Executive conclusion

The routed expert is dominated by moving the INT8 projection inputs, especially
the weights, rather than by Cube arithmetic.

- `exp_gate_mm`, `exp_up_mm`, and `exp_w2_mm` report MTE2 busy for 82-84% of
  their task cycles, while Cube is busy for only about 10%.
- Each of the three matmul stages performs almost the same modeled input
  traffic per rank: 128 MiB of weights and about 8 MiB of repeatedly loaded
  activation tiles.
- The compiler reports that the pipelined right operand, which is the weight
  operand, can keep only one of two requested buffers. The stages therefore
  share storage and serialize instead of fully overlapping the next weight
  load.
- Increasing the gate/up block count from 32 to 64 did not improve the
  unprofiled EP8 latency. That result is consistent with the PMU data: more
  blocks do not remove the weight-movement bottleneck.

The next optimization should target right-operand buffering and load/compute
overlap. Gate/up fusion is not the first choice because it leaves almost all
weight traffic unchanged.

## Capture configuration

| Item | Value |
|---|---|
| Commit | `2c7392af841531f43db69cded522631c7bfaaa1f` |
| Platform | `a2a3` |
| EP | 8 |
| Devices | `0,2,4,6,8,10,12,14` |
| Experts per rank | 16 |
| Routing | Balanced |
| PMU event type | 2, pipeline utilization |
| Model launches | One correctness dispatch |
| Correctness | PASS |

The PMU option was injected into `runtime_cfg` without changing the model or
kernel source. The MoE entry still ran as `__main__`, so the standalone
pre-dispatch barrier remained enabled.

Raw files are under:

```text
build_output/_jit_l3_moe_20260728_233523/
  dfx_outputs/rank{0..7}/d0/pmu.csv
  next_levels/moe_test/kernel_config.py
  report/perf_hints.log
```

Every rank produced 377 PMU records. The six routed-expert functions map to
function IDs 22 through 27:

| Function ID | Kernel | Core type | PMU records per rank |
|---:|---|---|---:|
| 22 | `exp_gate_mm` | AIC | 32 |
| 23 | `exp_up_mm` | AIC | 32 |
| 24 | `exp_gate_up_act` | AIV | 64 |
| 25 | `exp_h_q` | AIV | 16 |
| 26 | `exp_w2_mm` | AIC | 64 |
| 27 | `exp_w2_act` | AIV | 16 |

## How to read the counters

The percentages below are cycle-weighted across all records from all eight
ranks:

```text
sum(pipe_busy_cycles) / sum(pmu_total_cycles)
```

The range in parentheses is the minimum and maximum aggregate percentage
across the eight ranks. Different hardware pipes overlap, so the percentages
must not be added together.

PMU event type 2 does not include FIXPIPE cycles or L2-cache hit/miss counters.
It can identify the dominant execution pipe, but it cannot distinguish a weight
served from L2 from one served from HBM. "Weight-movement bound" below means the
GM-to-L1 MTE2 path is dominant and the tensor-size accounting shows that weights
make up most of that traffic.

## Aggregated PMU results

| Kernel | Median task cycles | MTE2 | Cube | Vector | Scalar | MTE1 | MTE3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `exp_gate_mm` | 131,971 | 83.8% (83.3-84.2) | 10.3% (10.0-10.4) | 0% | 3.1% | 29.5% | 0% |
| `exp_up_mm` | 129,069 | 83.7% (83.5-84.1) | 10.4% (10.2-10.6) | 0% | 3.3% | 30.0% | 0% |
| `exp_gate_up_act` | 12,744 | 32.8% (29.9-34.3) | 0% | 43.0% (41.0-44.7) | 33.7% | 0% | 12.1% |
| `exp_h_q` | 14,740 | 47.7% (43.9-52.2) | 0% | 51.5% (48.4-53.9) | 19.8% | 0% | 9.3% |
| `exp_w2_mm` | 65,297 | 82.0% (81.6-82.5) | 10.3% (10.1-10.6) | 0% | 4.1% | 29.7% | 0% |
| `exp_w2_act` | 14,181 | 60.6% (56.9-62.9) | 0% | 52.8% (50.1-57.2) | 25.4% | 0% | 28.2% |

The rank ranges are narrow for the three AIC matmuls. Their bottleneck
classification is therefore stable and is not caused by one slow card.

## Stage-by-stage interpretation

### Gate and up matmuls

Gate and up are almost identical:

- MTE2 is busy for about 84% of task cycles.
- Cube is busy for only about 10%.
- Their mean aggregate work is 4.26 million and 4.20 million core-cycles per
  rank respectively.
- Each rank emits 32 PMU records, but only 19-24 AIC cores receive records in a
  given capture.

The prior `MM_GATE_INNER=2` experiment increased each stage to 64 blocks and
made core coverage more regular, but representative latency changed only from
501.0 us to 499.1 us. The PMU result explains why: the work is limited by
input movement, so adding more blocks does not make the Cube pipe busier.

### W2 matmul

W2 has twice as many PMU records as gate or up, and its median record is about
half as long. Its total AIC work is nevertheless almost the same:

| AIC stage | Mean aggregate core-cycles per rank | Share of routed AIC work |
|---|---:|---:|
| Gate | 4,259,199 | 33.5% |
| Up | 4,198,937 | 33.0% |
| W2 | 4,253,035 | 33.5% |

W2 uses all 24 AIC cores in every rank, but its MTE2/Cube split remains
82%/10%. Better core coverage therefore does not remove the same
weight-movement limit.

### Gate/up activation

`exp_gate_up_act` is the largest AIV stage by aggregate work:

| AIV stage | Mean aggregate core-cycles per rank | Share of routed AIV work |
|---|---:|---:|
| Gate/up activation | 817,095 | 64.0% |
| Quantization | 235,859 | 18.5% |
| W2 dequant/scale/store | 224,177 | 17.6% |

Its largest single pipe is Vector at 43%, followed by Scalar at 34% and MTE2
at 33%. This is a mixed compute stage rather than a pure memory stage. The
SwiGLU `exp`/`recip`, casts, scale application, and bounds handling are
plausible secondary targets, but this stage should be optimized after the
larger AIC load bottleneck.

### Quantization and W2 activation

- `exp_h_q` is balanced between Vector (51.5%) and MTE2 (47.7%). Improving only
  its arithmetic or only its loads is unlikely to produce a large isolated
  win.
- `exp_w2_act` is more movement-heavy: MTE2 is 60.6%, Vector is 52.8%, and
  MTE3 is 28.2%. It is a valid later target for reducing intermediate reads or
  stores, but it accounts for only 17.6% of routed AIV core work.
- Both stages emit only 16 records per rank and cover 6-12 AIV cores in this
  capture. More fan-out may reduce their local tail, but their total work is
  much smaller than the matmul path.

## Why MTE2 mostly represents weights

For one balanced EP8 rank, every one of the 16 local experts receives three
valid rows and is processed by one padded 16-row tile.

For each routed matmul:

| Input | Modeled bytes per rank |
|---|---:|
| INT8 projection weights | 16 experts x 2048 x 4096 = 128 MiB |
| Repeated INT8 activation tiles | About 8 MiB |
| Combined | About 136 MiB |

The same accounting applies to gate, up, and W2. Weight data is therefore
about 94% of their modeled GM input bytes. Across the three projections, the
routed expert moves about 384 MiB of weights before considering scales and
intermediates.

This also explains why the three AIC stages have almost equal aggregate
core-cycle totals despite different block counts.

## Compiler evidence

The same build emitted `PH-MR-001` for the right operand at the gate, up, and
W2 matmul locations in `expert_routed.py`.

For each location, software pipelining requested depth 2, but only one of the
two 32 KiB right-operand buffers fit once co-resident buffers were considered.
Pipeline stages then share storage and serialize.

This warning aligns with the PMU result:

1. The right operand is the transposed INT8 weight tile.
2. MTE2 is active for more than 82% of cycles.
3. Cube is active for only about 10%.
4. The requested double-buffer overlap is not fully realized.

The build did not emit a cache-line-granularity hint for these routed-expert
locations. The immediate problem is buffer overlap, not a trailing dimension
smaller than the 512-byte transfer floor.

## Optimization implications

### Priority 1: make right-operand pipelining real

The highest-value experiment is a small tile sweep that tries to make both
right-operand pipeline stages fit while retaining efficient transfers.

Candidate axes are:

- right/weight N fragment size;
- K fragment size;
- the number of co-live N fragments in one block;
- pipeline depth, used as a control to measure the cost of the currently
  serialized depth-2 request.

Every candidate must be checked against two constraints:

- INT8 K fragments should stay at or above the 512-byte cache-line floor;
- increasing N must not overflow the Mat/L1 budget.

Success evidence should include all three:

1. the `PH-MR-001` warning disappears or changes as predicted;
2. AIC aggregate/task cycles fall, not just block count or core coverage;
3. the unprofiled 2-warmup + 8-round EP8 median improves outside noise.

### Priority 2: reduce repeated activation loads only if tiling permits

The current loop shape reloads the 16-row activation tile for each output
fragment. Hoisting or reusing that tile could reduce part of the approximately
8 MiB activation traffic per projection.

This is smaller than the 128 MiB weight stream and may require more live
accumulators, so it should be attempted only when the buffer report shows room.

### Priority 3: optimize gate/up activation

If the AIC load path cannot be improved, `exp_gate_up_act` is the next
well-supported target. It contains 64% of routed AIV work and is led by Vector
and Scalar activity. Useful experiments include reducing conversion passes or
restructuring the SwiGLU expression, with numerical validation after every
change.

### Deprioritized: gate/up fusion as a standalone change

A simple gate/up fusion leaves 256 MiB of W1/W3 weight traffic unchanged. Under
the current tiling it can save at most the duplicated activation stream between
the two projections, about 8 MiB or roughly 3% of their modeled input bytes.
It may also increase right-buffer pressure.

Fusion may become useful after a tiling change creates buffer headroom, but the
PMU data does not support it as the first experiment.

## Recommended next experiment

Run a focused two- or three-candidate tile sweep around the gate/up and W2
right-operand buffers. For each candidate:

1. compile and inspect `perf_hints.log`;
2. capture one PMU correctness dispatch;
3. compare aggregate AIC cycles and MTE2/Cube ratios;
4. only benchmark candidates that improve the PMU evidence;
5. benchmark with the established 2 warmup plus 8 measured EP8 launches.

The first candidate should change only one tile/buffering axis. This keeps the
result attributable and avoids repeating the no-gain block-count experiment.

