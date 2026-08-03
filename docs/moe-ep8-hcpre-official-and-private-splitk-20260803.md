# MoE EP8 HcPre Official Component and PyPTO Split-K Report

Date: 2026-08-03

## Outcome

The official vLLM-Ascend HcPre was built successfully as an isolated custom
operator on myserver with CANN 9.0.0. Its fixed decode-shape probe passed
correctness and measured a 37.0 us median.

The useful part of the official design was then reproduced in PyPTO:

- split the HcPre projection K dimension into 16 private partitions;
- let each Cube task write its own partial result without atomic add;
- remove the shared zero-seed task;
- reduce the private partials once in a short Vector task;
- keep the existing RMS path, because splitting RMS introduced more scheduling
  and atomic overhead than it removed.

The retained PyPTO candidate passes full balanced-routing EP8 correctness. It
improves standalone PyPTO HcPre from 36.0 us to 33.3 us and gives a provisional
1.6-3.6 us full-MoE reduction versus the clean 497.5 us control.

## Environment and Workload

- Host: `myserver`
- CANN: `/usr/local/Ascend/cann-9.0.0`
- SoC target: `ascend910_93`
- Official source: vLLM-Ascend commit
  `21382607d15a835728d9ea493ab101e7209799b5`
- PyPTO candidate source: commit
  `a7ace249ed001bad5b21a74e8bd428ee5def6f72`
- HcPre decode shape:
  - input: `[8, 4, 4096]`
  - flattened K: `16384`
  - projection weight: `[24, 16384]`
  - Sinkhorn iterations: `20`
- Full MoE:
  - EP8 on devices `0,2,4,6,8,10,12,14`
  - balanced routing
  - unchanged 256-entry routing workload
  - 2 warmups and 8 measured rounds

## Official HcPre Component

The official HcPre-only package was built with:

```text
bash build.sh --pkg --ops=hc_pre --soc=ascend910_93 -j8
```

It was installed in an isolated user path rather than the system CANN tree:

```text
/data/sunkaixuan/sunkaixuan_subdir/all_libs/
  vllm-ascend-hcpre-custom-20260803/vendors/custom_transformer
```

Both ACLNN symbols are present:

```text
aclnnHcPreGetWorkspaceSize
aclnnHcPre
```

The analytical nonzero probe exercises RMS, projection, sigmoid, Sinkhorn, and
stream mixing. It passed all outputs exactly within the official 5e-2
tolerance.

```text
warmup: 553.0, 34.9 us
measured: 37.2, 38.6, 36.0, 36.7, 38.3, 36.7, 37.1, 37.0 us
median: 37.0 us
mean: 37.2 us
workspace: 218103808 bytes
correctness: PASS
```

The first 553 us warmup includes operator initialization. The second warmup is
already at steady state.

## PyPTO Component Results

| Implementation | Median | Mean | Result |
|---|---:|---:|---|
| Retained PyPTO, linear split 4 + atomic | 36.0 us | 40.9 us | PASS |
| Official custom HcPre | 37.0 us | 37.2 us | PASS |
| Linear split 16 + atomic, RMS split 16 | 44.3 us | 48.9 us | PASS, reject |
| Private partial split 8 | 39.5 us | 41.1 us | PASS, reject |
| Private partial split 16 | **33.3 us** | **34.9 us** | PASS, retain |

The retained component samples were:

```text
39.2, 32.2, 41.1, 31.7, 36.9, 33.7, 33.0, 31.9 us
```

The official custom operator and PyPTO timings have slightly different runtime
boundaries. The official probe uses ACL device events around `aclnnHcPre`;
PyPTO reports resident-kernel `effective_us`. They are useful component
references but should not be treated as a perfect framework-to-framework
benchmark.

## Full EP8 Results

| Case | Critical-rank median | `host_union_mean_us` | Interpretation |
|---|---:|---:|---|
| Linear split 16 + atomic | 496.5 us | 3992 | No benefit |
| Linear split 16 + RMS split 16 | 498.0 us | 4097 | Reject |
| Retained clean control | 497.5 us | 3726 | Paired control |
| Private partial split 16, first run | **493.9 us** | 3874 | Clean |
| Private partial split 16, second run | 497.4 us reported | 4756 | Two interrupted rounds |

The second private-partial run contains two obvious interrupted last-rank
samples, 898.9 us and 1435.7 us. Excluding only those two whole-run
interruptions leaves:

```text
493.8, 497.0, 492.5, 499.6, 497.9, 494.7 us
filtered median: 495.85 us
```

Therefore the defensible end-to-end claim is a provisional 1.6-3.6 us
improvement, roughly 2-3 us, rather than claiming the best 493.9 us result
alone.

## Why the Successful Version Helps

The rejected 16-way atomic version creates more concurrent Cube work, but all
tasks contend on the same output and require a seed task first. Splitting RMS
adds another shared atomic accumulator and a finalize task. These extra
dependencies increase full-graph scheduling pressure.

The retained version follows the official two-part principle more closely:

```text
16 independent Cube split-K tasks
        |
        v
private partial buffer [16, 16, 32]
        |
        v
one Vector reduction
        |
        v
existing gate / Sinkhorn / mix_x stages
```

Each 1024-wide split contains four 256-wide pipelined matmul chunks. No Cube
task waits for a seed, no two Cube tasks atomically update the same result, and
the final reduction moves only 32 KiB of FP32 partial data.

## Scope Limits

The following official details were not copied:

- BF16 input staging shared by Cube and Vector;
- fine-grained AIC/AIV CrossCore handshakes inside one mixed kernel;
- the exact 22-part, 768-wide official K split;
- direct ACLNN invocation inside the PyPTO persistent MoE executor.

The current PyPTO HcPre receives FP32 input, so the official BF16 staging path
does not map directly. Direct ACLNN integration would also require a BF16
conversion and an additional framework/operator boundary, while the measured
official component is not faster than the retained PyPTO control. The
private-partial implementation captures the useful split-K/reduction behavior
without those costs.

## Artifacts

All scripts, raw queue logs, and checksums are stored under:

```text
artifacts/hcpre-official-component-20260803/
```

The same raw logs are retained on myserver at:

```text
/data/sunkaixuan/skx_log_output/hcpre_official_component_20260803/
```
