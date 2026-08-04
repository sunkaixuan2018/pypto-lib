# MoE EP8 Gate-AIV Hash-Route Fusion Experiment

## Objective

Remove the standalone hash-route task by running hash lookup, score gather,
normalization, and route output writes inside the existing gate AIV task. The
fixed validation shape is EP8, 128 routed experts, 8 tokens, and top-k 6.

The stable standalone AIV route already reduces the route kernel body from
about 5.04 us to 1.82 us, but its extra task preparation consumes nearly all
of that gain. Its paired full-EP8 median is 495.95 us versus 496.45 us for the
native control.

## Implemented Variants

All component benchmarks used two warmup rounds and eight measured rounds on
myserver.

| Variant | Task | Result |
| --- | --- | --- |
| Eight-participant AIV-only soft barrier | `task_20260803_181211_393207329019` | Route outputs failed; each iteration was about 503 ms |
| Sixteen-participant soft barrier for both physical AIV lanes | `task_20260803_182039_17350028958` | Route outputs failed; median 520.03 ms |
| Explicit AIV0-only soft barrier and route writer | `task_20260803_182318_41379517828` | All outputs passed; median 251.65 ms |
| Twenty-four-block full-occupancy hard MIX barrier | `task_20260803_183407_109960111891` | Benchmark hung, device was force-recovered after about 58 s, and route outputs failed |

No full EP8 benchmark was submitted because every fused component variant was
already unusable.

## Root Cause

The gate group is lowered to one AIC kernel plus two physical AIV lanes:

```text
MixedKernels = {gate_aic, gate_aiv, gate_aiv}
```

The no-split vector lowering keeps the real gate post-processing on AIV0 and
replays synchronization operations on AIV1. This initially caused both lanes
to enter the same logical barrier.

After explicitly excluding AIV1, the barrier still timed out. PTO ISA's A2/A3
software `SYNCALL` implementation indexes its GM slots with the hardware
no-argument `get_block_idx()`. Under the persistent executor, that value
identifies the resident worker and is not the runtime logical SPMD block index
passed through the task arguments. A partial eight-block gate task therefore
does not populate the contiguous slots that `SYNCALL` polls.

Changing `used_cores` cannot repair this indexing mismatch. Expanding the task
to full occupancy and using the hardware MIX barrier also proved incompatible
with this gate task's persistent mixed execution.

## Decision

Do not enable gate-AIV route fusion with the current PyPTO runtime and PTO ISA
integration. The required cross-block publication cannot be expressed safely
inside this partial-occupancy mixed task.

The production gate structure is restored in commit `20fd263`. The standalone
compact route AIV remains available behind `PYPTO_ROUTE_HASH_IMPL=aiv`, while
the native route remains the default.

The fusion can be revisited only after one of these runtime capabilities is
available:

1. a soft barrier that accepts the runtime logical block index;
2. a supported single-AIV mixed launch for this gate task; or
3. direct task chaining inside one persistent submission without a
   cross-block in-kernel barrier.
