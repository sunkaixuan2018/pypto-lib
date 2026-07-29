# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exact-shape EP8 probe using ProcessGroupHCCL and the fused W4A8 MoE op."""

from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

import numpy
import scipy
import torch
import torch.distributed as dist
import torch_npu
from torch.distributed.distributed_c10d import _get_default_group

torch_npu.npu.config.allow_internal_format = True

WORLD_SIZE = 8
TOKENS = 8
HIDDEN = 4096
INTERMEDIATE = 2048
TOP_K = 6
LOCAL_EXPERTS = 16
GLOBAL_EXPERTS = WORLD_SIZE * LOCAL_EXPERTS
W13_N = 2 * INTERMEDIATE
MAX_OUTPUT_SIZE = 512
WARMUP = 2
MEASURED = 8
PACKED_ONE = 0x11111111
UNIT_SCALE_BITS = 0x000000003F800000


def _npu_tensor(shape: tuple[int, ...], value: int | float, dtype: torch.dtype) -> torch.Tensor:
    return torch.full(shape, value, dtype=dtype, device="npu")


def _nz_weight(rows: int, packed_columns: int, value: int) -> torch.Tensor:
    nd = _npu_tensor((rows, packed_columns), value, torch.int32)
    nz = torch_npu.npu_format_cast(nd, 29)
    if torch_npu.get_npu_format(nz) != 29:
        raise RuntimeError("expected FRACTAL_NZ format 29")
    return nz


def _comm_name(rank: int) -> str:
    process_group = _get_default_group()
    backend = process_group._get_backend(torch.device("npu"))
    return backend.get_hccl_comm_name(rank)


def _build_inputs(rank: int) -> dict[str, object]:
    x = _npu_tensor((TOKENS, HIDDEN), 1.0, torch.bfloat16)
    weight1 = [
        _nz_weight(HIDDEN, W13_N // 8, PACKED_ONE)
        for _ in range(LOCAL_EXPERTS)
    ]
    weight2 = [
        _nz_weight(INTERMEDIATE, HIDDEN // 8, 0)
        for _ in range(LOCAL_EXPERTS)
    ]
    scale1 = [
        _npu_tensor((W13_N,), UNIT_SCALE_BITS, torch.int64)
        for _ in range(LOCAL_EXPERTS)
    ]
    scale2 = [
        _npu_tensor((HIDDEN,), UNIT_SCALE_BITS, torch.int64)
        for _ in range(LOCAL_EXPERTS)
    ]
    bias1 = [
        _npu_tensor((W13_N,), float(HIDDEN), torch.float32)
        for _ in range(LOCAL_EXPERTS)
    ]
    bias2 = [
        _npu_tensor((HIDDEN,), 0.0, torch.float32)
        for _ in range(LOCAL_EXPERTS)
    ]

    route_base = rank * TOKENS * TOP_K
    expert_id = torch.tensor(
        [
            [(route_base + token * TOP_K + top) % GLOBAL_EXPERTS for top in range(TOP_K)]
            for token in range(TOKENS)
        ],
        dtype=torch.int32,
        device="npu",
    )
    probs = _npu_tensor((TOKENS, TOP_K), 1.0 / TOP_K, torch.float32)
    active_mask = _npu_tensor((TOKENS,), True, torch.bool)
    output = _npu_tensor((TOKENS, HIDDEN), 3.0, torch.bfloat16)
    expert_token_nums = _npu_tensor((1, LOCAL_EXPERTS), -1, torch.int32)
    return {
        "x": x,
        "weight1": weight1,
        "weight2": weight2,
        "expert_idx": expert_id,
        "scale1": scale1,
        "scale2": scale2,
        "bias1": bias1,
        "bias2": bias2,
        "probs": probs,
        "x_active_mask": active_mask,
        "out": output,
        "expert_token_nums": expert_token_nums,
    }


def _invoke(inputs: dict[str, object], group: str) -> None:
    torch.ops._C_ascend.dispatch_ffn_combine(
        **inputs,
        group=group,
        max_output_size=MAX_OUTPUT_SIZE,
        swiglu_limit=10.0,
    )


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != WORLD_SIZE:
        raise RuntimeError(f"expected world size {WORLD_SIZE}, got {world_size}")

    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl", init_method="env://")
    extension = Path(os.environ["W4A8_TORCH_EXTENSION"]).resolve()
    torch.ops.load_library(str(extension))
    group = _comm_name(rank)

    free_before, total_hbm = torch.npu.mem_get_info()
    inputs = _build_inputs(rank)
    torch.npu.synchronize()
    free_after, _ = torch.npu.mem_get_info()
    print(
        f"rank={rank} local_rank={local_rank} group={group} "
        f"python={sys.version.split()[0]} numpy={numpy.__version__} "
        f"scipy={scipy.__version__} "
        f"torch={torch.__version__} torch_npu={torch_npu.__version__} "
        f"internal_format_requested=True "
        f"extension={extension} free_before={free_before} "
        f"free_after={free_after} total_hbm={total_hbm}",
        flush=True,
    )

    measured_us: list[float] = []
    host_intervals: list[tuple[int, int]] = []
    for run in range(WARMUP + MEASURED):
        dist.barrier()
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        host_start = time.monotonic_ns()
        start.record()
        _invoke(inputs, group)
        end.record()
        end.synchronize()
        host_end = time.monotonic_ns()
        elapsed_us = start.elapsed_time(end) * 1000.0
        if run >= WARMUP:
            measured_us.append(elapsed_us)
            host_intervals.append((host_start, host_end))

    output = inputs["out"]
    expert_token_nums = inputs["expert_token_nums"]
    assert isinstance(output, torch.Tensor)
    assert isinstance(expert_token_nums, torch.Tensor)
    bad_output = int(torch.count_nonzero(output).item())
    counts = expert_token_nums.cpu().reshape(-1).tolist()
    local_ok = bad_output == 0 and counts == [3] * LOCAL_EXPERTS
    ok = torch.tensor([int(local_ok)], dtype=torch.int32, device="npu")
    dist.all_reduce(ok, op=dist.ReduceOp.MIN)
    print(
        f"rank={rank} bad_output={bad_output} expert_counts={counts} "
        f"measured_us={','.join(f'{value:.1f}' for value in measured_us)}",
        flush=True,
    )

    timing = torch.tensor(measured_us, dtype=torch.float64, device="npu")
    intervals = torch.tensor(host_intervals, dtype=torch.float64, device="npu")
    all_timing = [torch.empty_like(timing) for _ in range(WORLD_SIZE)]
    all_intervals = [torch.empty_like(intervals) for _ in range(WORLD_SIZE)]
    dist.all_gather(all_timing, timing)
    dist.all_gather(all_intervals, intervals)

    if rank == 0:
        timing_matrix = torch.stack(all_timing).cpu()
        interval_matrix = torch.stack(all_intervals).cpu()
        last_arriving = timing_matrix.max(dim=0).values.tolist()
        host_union = (
            interval_matrix[:, :, 1].max(dim=0).values
            - interval_matrix[:, :, 0].min(dim=0).values
        ).tolist()
        for sample, values in enumerate(timing_matrix.T.tolist()):
            print(
                f"sample[{sample}]_rank_us="
                + ",".join(f"{value:.1f}" for value in values)
                + f" last_arriving_us={last_arriving[sample]:.1f} "
                + f"host_union_us={host_union[sample] / 1000.0:.1f}",
                flush=True,
            )
        print(
            f"correctness={'PASS' if int(ok.item()) == 1 else 'FAIL'} "
            f"last_arriving_median_us={statistics.median(last_arriving):.1f} "
            f"last_arriving_mean_us={statistics.mean(last_arriving):.1f}",
            flush=True,
        )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
