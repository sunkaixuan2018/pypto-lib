# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Probe whether per-expert distributed waits can starve their push producers.

The ``separate`` mode models the proposed MoE dispatch experiment: one SPMD
task pushes all expert payloads, while a second SPMD task waits and gathers one
expert per block.  The ``fused`` mode puts push, notify, wait, and gather in the
same SPMD block.  Both modes use the EP8 decode dimensions that matter for
scheduling: 16 local experts, 8 ranks, and one 4096-byte activation row.

Run on the even-card EP8 set used by the MoE benchmark:

    python moe_dispatch_wait_probe.py -p a2a3 -d 0,2,4,6,8,10,12,14 \
        --mode separate

Use an external timeout for device runs.  A timeout or a missing producer block
is itself a failed liveness result; this probe must not be used in production.
"""

import argparse

import pypto.language as pl
import pypto.language.distributed as pld


N_RANKS = 8
N_LOCAL = 16
D = 4096


@pl.jit
def separate_wait_probe(
    inp: pl.Tensor[[N_LOCAL, D], pl.INT8],
    out: pl.Out[pl.Tensor[[N_LOCAL, N_RANKS, D], pl.INT8]],
    recv_x: pld.DistributedTensor[[N_LOCAL * N_RANKS, D], pl.INT8],
    data_arrived: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    """Run push and per-expert wait/gather as two independent SPMD tasks."""
    out_flat = pl.reshape(out, [N_LOCAL * N_RANKS, D])

    with pl.spmd(N_LOCAL, name_hint="probe_push", allow_early_resolve=True):
        expert = pl.tile.get_block_idx()
        row = expert * N_RANKS + my_rank
        for dst in pl.range(N_RANKS):
            pld.tensor.put(
                dst=recv_x,
                peer=dst,
                src=inp,
                dst_offsets=[row, 0],
                src_offsets=[expert, 0],
                shape=[1, D],
            )
            if dst != my_rank:
                pld.system.notify(
                    target=data_arrived,
                    peer=dst,
                    offsets=[my_rank, expert],
                    value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )

    # This is intentionally a separate task with 16 potential waiters.  Its
    # liveness tells us whether the corresponding MoE variant is safe to try.
    with pl.spmd(N_LOCAL, name_hint="probe_wait_gather", allow_early_resolve=True):
        expert = pl.tile.get_block_idx()
        for src in pl.range(N_RANKS):
            if src != my_rank:
                pld.system.wait(
                    signal=data_arrived,
                    offsets=[src, expert],
                    expected=1,
                    cmp=pld.WaitCmp.Ge,
                )
            row = expert * N_RANKS + src
            out_flat[row : row + 1, :] = recv_x[row : row + 1, :]


@pl.jit
def fused_wait_probe(
    inp: pl.Tensor[[N_LOCAL, D], pl.INT8],
    out: pl.Out[pl.Tensor[[N_LOCAL, N_RANKS, D], pl.INT8]],
    recv_x: pld.DistributedTensor[[N_LOCAL * N_RANKS, D], pl.INT8],
    data_arrived: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    """Run push, notify, per-expert wait, and gather in one SPMD task."""
    out_flat = pl.reshape(out, [N_LOCAL * N_RANKS, D])

    with pl.spmd(N_LOCAL, name_hint="probe_push_wait_gather", allow_early_resolve=True):
        expert = pl.tile.get_block_idx()
        row = expert * N_RANKS + my_rank
        for dst in pl.range(N_RANKS):
            pld.tensor.put(
                dst=recv_x,
                peer=dst,
                src=inp,
                dst_offsets=[row, 0],
                src_offsets=[expert, 0],
                shape=[1, D],
            )
            if dst != my_rank:
                pld.system.notify(
                    target=data_arrived,
                    peer=dst,
                    offsets=[my_rank, expert],
                    value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )

        for src in pl.range(N_RANKS):
            if src != my_rank:
                pld.system.wait(
                    signal=data_arrived,
                    offsets=[src, expert],
                    expected=1,
                    cmp=pld.WaitCmp.Ge,
                )
            src_row = expert * N_RANKS + src
            out_flat[src_row : src_row + 1, :] = recv_x[src_row : src_row + 1, :]


@pl.jit.host
def l3_separate_wait_probe(
    inputs: pl.Tensor[[N_RANKS, N_LOCAL, D], pl.INT8],
    outputs: pl.Out[pl.Tensor[[N_RANKS, N_LOCAL, N_RANKS, D], pl.INT8]],
):
    """Launch the separate-task probe on all ranks."""
    recv_x_buf = pld.alloc_window_buffer([N_LOCAL * N_RANKS, D], dtype=pl.INT8)
    data_arrived_buf = pld.alloc_window_buffer([N_RANKS, N_LOCAL], dtype=pl.INT32)

    for rank in pl.range(pld.world_size()):
        recv_x = pld.window(recv_x_buf, [N_LOCAL * N_RANKS, D], dtype=pl.INT8)
        data_arrived = pld.window(data_arrived_buf, [N_RANKS, N_LOCAL], dtype=pl.INT32)
        separate_wait_probe(
            inputs[rank],
            outputs[rank],
            recv_x,
            data_arrived,
            rank,
            device=rank,
        )


@pl.jit.host
def l3_fused_wait_probe(
    inputs: pl.Tensor[[N_RANKS, N_LOCAL, D], pl.INT8],
    outputs: pl.Out[pl.Tensor[[N_RANKS, N_LOCAL, N_RANKS, D], pl.INT8]],
):
    """Launch the single-task probe on all ranks."""
    recv_x_buf = pld.alloc_window_buffer([N_LOCAL * N_RANKS, D], dtype=pl.INT8)
    data_arrived_buf = pld.alloc_window_buffer([N_RANKS, N_LOCAL], dtype=pl.INT32)

    for rank in pl.range(pld.world_size()):
        recv_x = pld.window(recv_x_buf, [N_LOCAL * N_RANKS, D], dtype=pl.INT8)
        data_arrived = pld.window(data_arrived_buf, [N_RANKS, N_LOCAL], dtype=pl.INT32)
        fused_wait_probe(
            inputs[rank],
            outputs[rank],
            recv_x,
            data_arrived,
            rank,
            device=rank,
        )


def build_tensor_specs():
    """Create rank- and expert-distinct rows for visibility validation."""
    import torch

    from golden import TensorSpec

    def init_inputs():
        rank = torch.arange(N_RANKS, dtype=torch.int16).reshape(N_RANKS, 1, 1)
        expert = torch.arange(N_LOCAL, dtype=torch.int16).reshape(1, N_LOCAL, 1)
        column = torch.arange(D, dtype=torch.int16).reshape(1, 1, D)
        return ((rank * 31 + expert * 7 + column) % 127).to(torch.int8)

    return [
        TensorSpec(
            "inputs",
            [N_RANKS, N_LOCAL, D],
            torch.int8,
            init_value=init_inputs,
        ),
        TensorSpec(
            "outputs",
            [N_RANKS, N_LOCAL, N_RANKS, D],
            torch.int8,
            is_output=True,
        ),
    ]


def golden_probe(tensors):
    """Every destination gathers the same source-major expert rows."""
    gathered = tensors["inputs"].permute(1, 0, 2)
    tensors["outputs"][:] = gathered.unsqueeze(0).expand_as(tensors["outputs"])


if __name__ == "__main__":
    from golden import run_jit
    from pypto.ir.distributed_compiled_program import DistributedConfig

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--platform",
        type=str,
        default="a2a3",
        choices=["a2a3", "a2a3sim"],
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="0,2,4,6,8,10,12,14",
        help=f"comma-separated device ids (need exactly {N_RANKS})",
    )
    parser.add_argument(
        "--mode",
        choices=["separate", "fused"],
        default="separate",
    )
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()

    device_ids = [int(device) for device in args.device.split(",")]
    assert len(device_ids) == N_RANKS, (
        f"need exactly {N_RANKS} devices, got {device_ids}"
    )
    host_fn = (
        l3_separate_wait_probe
        if args.mode == "separate"
        else l3_fused_wait_probe
    )

    result = run_jit(
        fn=host_fn,
        specs=build_tensor_specs(),
        golden_fn=golden_probe,
        compile_only=args.compile_only,
        compile_cfg=dict(
            distributed_config=DistributedConfig(
                device_ids=device_ids,
                num_sub_workers=0,
            ),
        ),
        runtime_cfg=dict(platform=args.platform),
        rtol=0.0,
        atol=0.0,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
