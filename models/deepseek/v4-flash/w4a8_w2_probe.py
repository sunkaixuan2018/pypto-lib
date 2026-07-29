# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Standalone W2-shaped W4A8 feasibility probe for the EP8 decode MoE.

This deliberately does not change the production MoE path.  It compares:

* the current INT8 x INT8 W2 shape and tiling; and
* Catlass W4A8, where packed INT4 weights are expanded by AIV into a temporary
  INT8 workspace before the Cube matmul.

Both variants compute an exact INT32 result for M=16, K=2048, N=4096.  Keeping
the probe separate makes the first device run a strict compile/correctness and
isolated-performance gate before we take on MoE scheduling changes.
"""

import os
from pathlib import Path

import pypto.language as pl


M = 16
K = 2048
N = 4096
BLOCK_DIM = 24

L1_K = 512
L1_N = 256
STAGES = 2
WORKSPACE_BYTES = STAGES * L1_K * L1_N * BLOCK_DIM
MARKER_ROWS = BLOCK_DIM * 3
MARKER_COLUMNS = 16
MARKER_BYTES = MARKER_ROWS * MARKER_COLUMNS * 4
TILECAST_BYTES = 16 * 4 * L1_K * L1_N
TILECAST_OUTPUT_BYTES = MARKER_BYTES + TILECAST_BYTES
AIC_MMAD_K = 1024
AIC_MMAD_WORKSPACE_BYTES = BLOCK_DIM * STAGES * L1_K * L1_N
AIC_MMAD_OUTPUT_BYTES = MARKER_BYTES + M * N * 4

_REPO_ROOT = Path(__file__).resolve().parents[3]
_KERNEL_DIR = Path(__file__).parent / "kernels" / "w4a8_w2"
_ENTRY = _KERNEL_DIR / "entry.cpp"
_TOPOLOGY_ENTRY = _KERNEL_DIR / "topology_entry.cpp"
_CROSSCORE_ENTRY = _KERNEL_DIR / "crosscore_entry.cpp"
_RESOURCE_ENTRY = _KERNEL_DIR / "resource_entry.cpp"
_BLOCKMMAD_ENTRY = _KERNEL_DIR / "blockmmad_entry.cpp"
_PINGPONG_ENTRY = _KERNEL_DIR / "pingpong_entry.cpp"
_TILECAST_ENTRY = _KERNEL_DIR / "tilecast_entry.cpp"
_AIC_MMAD_ENTRY = _KERNEL_DIR / "aic_mmad_entry.cpp"
_CATLASS_INCLUDE = _REPO_ROOT / "third_party" / "catlass" / "include"


def _cann_include_dirs() -> tuple[Path, ...]:
    cann_root = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/latest"))
    devkit = cann_root / "aarch64-linux"
    candidates = (
        devkit / "include",
        devkit / "asc",
        devkit / "asc" / "include",
        devkit / "asc" / "include" / "adv_api",
        devkit / "asc" / "include" / "basic_api",
        devkit / "asc" / "include" / "c_api",
        devkit / "asc" / "include" / "interface",
        devkit / "asc" / "include" / "simt_api",
        devkit / "asc" / "include" / "utils",
        devkit / "tikcpp" / "tikcfw",
        devkit / "tikcpp" / "tikcfw" / "interface",
        devkit / "tikcpp" / "tikcfw" / "impl",
    )
    return tuple(path for path in candidates if path.is_dir())


_EXTERN_INCLUDE_DIRS = (_CATLASS_INCLUDE,) + _cann_include_dirs()


@pl.jit.extern(
    core_type="mixed",
    aic_source=_ENTRY,
    aiv_source=_ENTRY,
    include_dirs=_EXTERN_INCLUDE_DIRS,
    dual_aiv_dispatch=True,
)
def w4a8_w2_cce(
    out: pl.Out[pl.Tensor],
    activation: pl.Tensor,
    packed_weight_kn: pl.Tensor,
    workspace: pl.InOut[pl.Tensor],
) -> pl.Tensor: ...


@pl.jit.extern(
    core_type="mixed",
    aic_source=_TOPOLOGY_ENTRY,
    aiv_source=_TOPOLOGY_ENTRY,
    include_dirs=_EXTERN_INCLUDE_DIRS,
    dual_aiv_dispatch=True,
)
def mixed_topology_cce(topology: pl.Out[pl.Tensor]) -> pl.Tensor: ...


@pl.jit.extern(
    core_type="mixed",
    aic_source=_CROSSCORE_ENTRY,
    aiv_source=_CROSSCORE_ENTRY,
    include_dirs=_EXTERN_INCLUDE_DIRS,
    dual_aiv_dispatch=True,
)
def crosscore_handshake_cce(handshake: pl.Out[pl.Tensor]) -> pl.Tensor: ...


@pl.jit.extern(
    core_type="mixed",
    aic_source=_RESOURCE_ENTRY,
    aiv_source=_RESOURCE_ENTRY,
    include_dirs=_EXTERN_INCLUDE_DIRS,
    dual_aiv_dispatch=True,
)
def resource_lifecycle_cce(lifecycle: pl.Out[pl.Tensor]) -> pl.Tensor: ...


@pl.jit.extern(
    core_type="mixed",
    aic_source=_BLOCKMMAD_ENTRY,
    aiv_source=_BLOCKMMAD_ENTRY,
    include_dirs=_EXTERN_INCLUDE_DIRS,
    dual_aiv_dispatch=True,
)
def blockmmad_lifecycle_cce(lifecycle: pl.Out[pl.Tensor]) -> pl.Tensor: ...


@pl.jit.extern(
    core_type="mixed",
    aic_source=_PINGPONG_ENTRY,
    aiv_source=_PINGPONG_ENTRY,
    include_dirs=_EXTERN_INCLUDE_DIRS,
    dual_aiv_dispatch=True,
)
def pingpong_state_cce(state: pl.Out[pl.Tensor]) -> pl.Tensor: ...


@pl.jit.extern(
    core_type="mixed",
    aic_source=_TILECAST_ENTRY,
    aiv_source=_TILECAST_ENTRY,
    include_dirs=_EXTERN_INCLUDE_DIRS,
    dual_aiv_dispatch=True,
)
def tilecast_data_cce(
    output: pl.Out[pl.Tensor],
    packed_weight_kn: pl.Tensor,
) -> pl.Tensor: ...


@pl.jit.extern(
    core_type="mixed",
    aic_source=_AIC_MMAD_ENTRY,
    aiv_source=_AIC_MMAD_ENTRY,
    include_dirs=_EXTERN_INCLUDE_DIRS,
    dual_aiv_dispatch=True,
)
def aic_workspace_mmad_cce(
    output: pl.Out[pl.Tensor],
    activation: pl.Tensor,
    workspace: pl.Tensor,
) -> pl.Tensor: ...


@pl.jit
def aic_workspace_mmad_test(
    activation: pl.Tensor[[M, AIC_MMAD_K], pl.INT8],
    workspace: pl.Tensor[[AIC_MMAD_WORKSPACE_BYTES], pl.INT8],
    output: pl.Out[pl.Tensor[[AIC_MMAD_OUTPUT_BYTES], pl.INT8]],
):
    # The two prefilled workspace stages cover K=1024 exactly.  AIV only
    # produces the stock readiness flags; AIC executes the real BlockMmad.
    with pl.spmd(BLOCK_DIM, name_hint="aic_workspace_mmad", sync_start=True):
        output = aic_workspace_mmad_cce(output, activation, workspace)
    return output


@pl.jit
def tilecast_data_test(
    packed_weight_kn: pl.Tensor[[K, N // 2], pl.INT8],
    output: pl.Out[pl.Tensor[[TILECAST_OUTPUT_BYTES], pl.INT8]],
):
    # Exercise the real packed-INT4 GM -> UB -> INT8 -> GM path with both AIV
    # lanes.  Each active block converts four independent [512, 256] tiles.
    with pl.spmd(BLOCK_DIM, name_hint="tilecast_data", sync_start=True):
        output = tilecast_data_cce(output, packed_weight_kn)
    return output


@pl.jit
def pingpong_state_test(
    state: pl.Out[pl.Tensor[[BLOCK_DIM * 3, 16], pl.INT32]],
):
    # Reproduce the exact two-stage, four-generation W4 CrossCore protocol
    # without data movement or MMAD.
    with pl.spmd(BLOCK_DIM, name_hint="pingpong_state", sync_start=True):
        state = pingpong_state_cce(state)
    return state


@pl.jit
def blockmmad_lifecycle_test(
    lifecycle: pl.Out[pl.Tensor[[BLOCK_DIM * 3, 16], pl.INT32]],
):
    # Construct and destroy the exact W4 BlockMmad type without entering its
    # prologue or matmul body.
    with pl.spmd(BLOCK_DIM, name_hint="blockmmad_lifecycle", sync_start=True):
        lifecycle = blockmmad_lifecycle_cce(lifecycle)
    return lifecycle


@pl.jit
def resource_lifecycle_test(
    lifecycle: pl.Out[pl.Tensor[[BLOCK_DIM * 3, 16], pl.INT32]],
):
    # Add only Catlass Resource/TPipe lifetime around the already validated
    # single-stage CrossCore handshake.
    with pl.spmd(BLOCK_DIM, name_hint="resource_lifecycle", sync_start=True):
        lifecycle = resource_lifecycle_cce(lifecycle)
    return lifecycle


@pl.jit
def crosscore_handshake_test(
    handshake: pl.Out[pl.Tensor[[BLOCK_DIM * 3, 16], pl.INT32]],
):
    # Mirror one Catlass copy-ready/prologue-ready exchange without Resource,
    # TPipe, data movement, INT4 conversion, or Cube work.
    with pl.spmd(BLOCK_DIM, name_hint="crosscore_handshake", sync_start=True):
        handshake = crosscore_handshake_cce(handshake)
    return handshake


@pl.jit
def mixed_topology_test(
    topology: pl.Out[pl.Tensor[[BLOCK_DIM * 3, 8], pl.INT32]],
):
    # One row per AIC and per AIV lane.  The CCE probe records both PyPTO's
    # logical task identity and CANN's native topology view.
    with pl.spmd(BLOCK_DIM, name_hint="mixed_topology", sync_start=True):
        topology = mixed_topology_cce(topology)
    return topology


@pl.jit
def w4a8_w2_test(
    activation: pl.Tensor[[M, K], pl.INT8],
    packed_weight_kn: pl.Tensor[[K, N // 2], pl.INT8],
    workspace: pl.Tensor[[WORKSPACE_BYTES], pl.INT8],
    out: pl.Out[pl.Tensor[[M, N], pl.INT32]],
):
    # Catlass uses the hardware full-die block/sub-block ids.  A full-occupancy,
    # co-started task therefore matches its 24 AIC + 48 AIV scheduling contract.
    with pl.spmd(BLOCK_DIM, name_hint="w4a8_w2", sync_start=True):
        out = w4a8_w2_cce(out, activation, packed_weight_kn, workspace)
    return out


@pl.jit
def int8_w2_test(
    activation: pl.Tensor[[M, K], pl.INT8],
    weight_nk: pl.Tensor[[N, K], pl.INT8],
    out: pl.Out[pl.Tensor[[M, N], pl.INT32]],
):
    # Match the current production W2 split: 4 blocks, 4 x 256 output channels
    # per block, K=512 ping-pong pipeline.
    with pl.spmd(4, name_hint="int8_w2"):
        block_idx = pl.tile.get_block_idx()
        n_base = block_idx * 4 * 256
        for inner in pl.range(4):
            n0 = n_base + inner * 256
            acc = pl.create_tensor([M, 256], dtype=pl.INT32)
            for k0 in pl.pipeline(0, K, 512, stage=2):
                a_k = activation[:, k0 : k0 + 512]
                b_k = weight_nk[n0 : n0 + 256, k0 : k0 + 512]
                if k0 == 0:
                    acc = pl.matmul(a_k, b_k, b_trans=True, out_dtype=pl.INT32)
                else:
                    acc = pl.matmul_acc(acc, a_k, b_k, b_trans=True)
            out[:, n0 : n0 + 256] = acc
    return out


def _pack_int4_kn(weight_kn):
    """Pack adjacent N values as low/high signed nibbles."""
    import torch

    unsigned = torch.bitwise_and(weight_kn.to(torch.int16), 0xF)
    packed = unsigned[:, 0::2] | (unsigned[:, 1::2] << 4)
    return packed.to(torch.uint8).view(torch.int8).contiguous()


def _unpack_int4_kn(packed):
    """Inverse of :func:`_pack_int4_kn`, used by the independent golden."""
    import torch

    raw = packed.view(torch.uint8).to(torch.int16)
    low = raw & 0xF
    high = (raw >> 4) & 0xF
    low = torch.where(low < 8, low, low - 16)
    high = torch.where(high < 8, high, high - 16)
    weight = torch.empty(K, N, dtype=torch.int8)
    weight[:, 0::2] = low.to(torch.int8)
    weight[:, 1::2] = high.to(torch.int8)
    return weight


def _golden_matmul(activation, weight_kn):
    # Every exact accumulation is below 2**24 for this probe, so float32 matmul
    # is an exact and much faster CPU reference than a Python integer loop.
    import torch

    return (activation.float() @ weight_kn.float()).to(torch.int32)


def build_w4_specs():
    import torch
    from golden import TensorSpec

    torch.manual_seed(20260729)
    activation = torch.randint(-127, 128, (M, K), dtype=torch.int8)
    weight_kn = torch.randint(-8, 8, (K, N), dtype=torch.int8)
    packed = _pack_int4_kn(weight_kn)

    return [
        TensorSpec("activation", [M, K], torch.int8, init_value=lambda: activation),
        TensorSpec("packed_weight_kn", [K, N // 2], torch.int8, init_value=lambda: packed),
        TensorSpec("workspace", [WORKSPACE_BYTES], torch.int8, init_value=lambda: torch.zeros(WORKSPACE_BYTES, dtype=torch.int8)),
        TensorSpec("out", [M, N], torch.int32, is_output=True),
    ]


def build_int8_specs():
    import torch
    from golden import TensorSpec

    torch.manual_seed(20260729)
    activation = torch.randint(-127, 128, (M, K), dtype=torch.int8)
    # Use the same numeric W4 range so only storage/prologue/scheduling differ.
    weight_nk = torch.randint(-8, 8, (N, K), dtype=torch.int8)

    return [
        TensorSpec("activation", [M, K], torch.int8, init_value=lambda: activation),
        TensorSpec("weight_nk", [N, K], torch.int8, init_value=lambda: weight_nk),
        TensorSpec("out", [M, N], torch.int32, is_output=True),
    ]


def build_topology_specs():
    import torch
    from golden import TensorSpec

    return [
        TensorSpec(
            "topology",
            [BLOCK_DIM * 3, 8],
            torch.int32,
            is_output=True,
        ),
    ]


def build_crosscore_specs():
    import torch
    from golden import TensorSpec

    return [
        TensorSpec(
            "handshake",
            [BLOCK_DIM * 3, 16],
            torch.int32,
            is_output=True,
        ),
    ]


def build_resource_specs():
    import torch
    from golden import TensorSpec

    return [
        TensorSpec(
            "lifecycle",
            [BLOCK_DIM * 3, 16],
            torch.int32,
            is_output=True,
        ),
    ]


def build_blockmmad_specs():
    import torch
    from golden import TensorSpec

    return [
        TensorSpec(
            "blockmmad_lifecycle",
            [BLOCK_DIM * 3, 16],
            torch.int32,
            is_output=True,
        ),
    ]


def build_pingpong_specs():
    import torch
    from golden import TensorSpec

    return [
        TensorSpec(
            "pingpong_state",
            [BLOCK_DIM * 3, 16],
            torch.int32,
            is_output=True,
        ),
    ]


def _tilecast_weight():
    import torch

    k = torch.arange(K, dtype=torch.int16)[:, None]
    n = torch.arange(N, dtype=torch.int16)[None, :]
    return (((3 * k + 5 * n + 7) % 16) - 8).to(torch.int8)


def build_tilecast_specs():
    import torch
    from golden import TensorSpec

    packed = _pack_int4_kn(_tilecast_weight())
    return [
        TensorSpec(
            "packed_weight_kn",
            [K, N // 2],
            torch.int8,
            init_value=lambda: packed,
        ),
        TensorSpec(
            "tilecast_output",
            [TILECAST_OUTPUT_BYTES],
            torch.int8,
            is_output=True,
        ),
    ]


def _aic_mmad_inputs():
    import torch

    m = torch.arange(M, dtype=torch.int16)[:, None]
    k = torch.arange(AIC_MMAD_K, dtype=torch.int16)[None, :]
    activation = (((3 * m + 5 * k + 1) % 7) - 3).to(torch.int8)

    k = torch.arange(AIC_MMAD_K, dtype=torch.int16)[:, None]
    n = torch.arange(N, dtype=torch.int16)[None, :]
    weight = (((7 * k + 3 * n + 2) % 9) - 4).to(torch.int8)
    return activation, weight


def build_aic_mmad_specs():
    import torch
    from golden import TensorSpec

    activation, weight = _aic_mmad_inputs()
    workspace = torch.zeros(
        BLOCK_DIM, STAGES, L1_K, L1_N, dtype=torch.int8
    )
    for block_idx in range(16):
        n0 = block_idx * L1_N
        workspace[block_idx, 0] = weight[:L1_K, n0 : n0 + L1_N]
        workspace[block_idx, 1] = weight[L1_K:, n0 : n0 + L1_N]
    workspace = workspace.flatten().contiguous()

    return [
        TensorSpec(
            "activation",
            [M, AIC_MMAD_K],
            torch.int8,
            init_value=lambda: activation,
        ),
        TensorSpec(
            "aic_mmad_workspace",
            [AIC_MMAD_WORKSPACE_BYTES],
            torch.int8,
            init_value=lambda: workspace,
        ),
        TensorSpec(
            "aic_mmad_output",
            [AIC_MMAD_OUTPUT_BYTES],
            torch.int8,
            is_output=True,
        ),
    ]


def golden_aic_mmad(tensors):
    import torch

    output = tensors["aic_mmad_output"]
    output.zero_()
    marker = output[:MARKER_BYTES].view(torch.int32).reshape(
        MARKER_ROWS, MARKER_COLUMNS
    )
    for block_idx in range(BLOCK_DIM):
        active = block_idx < 16
        marker[block_idx, :11] = torch.tensor(
            [1, block_idx, -1, 1, 1, 1, int(active), 1, 0, 1, 1],
            dtype=torch.int32,
        )
        for lane in range(2):
            row = BLOCK_DIM + block_idx * 2 + lane
            rounds = [1, 1] if active else [-1, -1]
            marker[row, :11] = torch.tensor(
                [
                    2,
                    block_idx,
                    lane,
                    1,
                    1,
                    *rounds,
                    1,
                    1,
                    int(active),
                    1,
                ],
                dtype=torch.int32,
            )

    workspace = tensors["aic_mmad_workspace"].reshape(
        BLOCK_DIM, STAGES, L1_K, L1_N
    )
    weight = torch.empty(AIC_MMAD_K, N, dtype=torch.int8)
    for block_idx in range(16):
        n0 = block_idx * L1_N
        weight[:L1_K, n0 : n0 + L1_N] = workspace[block_idx, 0]
        weight[L1_K:, n0 : n0 + L1_N] = workspace[block_idx, 1]
    matrix_out = output[MARKER_BYTES:].view(torch.int32).reshape(M, N)
    matrix_out[:] = _golden_matmul(tensors["activation"], weight)


def golden_tilecast(tensors):
    import torch

    output = tensors["tilecast_output"]
    output.zero_()

    marker = output[:MARKER_BYTES].view(torch.int32).reshape(
        MARKER_ROWS, MARKER_COLUMNS
    )
    for block_idx in range(BLOCK_DIM):
        marker[block_idx, :15] = torch.tensor(
            [1, block_idx, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 1],
            dtype=torch.int32,
        )
        for lane in range(2):
            row = BLOCK_DIM + block_idx * 2 + lane
            active = block_idx < 16
            tile_done = [1, 1, 1, 1] if active else [-1, -1, -1, -1]
            marker[row, :15] = torch.tensor(
                [
                    2,
                    block_idx,
                    lane,
                    1,
                    1,
                    *tile_done,
                    1,
                    1,
                    1,
                    0,
                    int(active),
                    1,
                ],
                dtype=torch.int32,
            )

    weight = _unpack_int4_kn(tensors["packed_weight_kn"])
    converted = output[MARKER_BYTES:].reshape(16, 4, L1_K, L1_N)
    for block_idx in range(16):
        n0 = block_idx * L1_N
        for tile_idx in range(4):
            k0 = tile_idx * L1_K
            converted[block_idx, tile_idx] = weight[
                k0 : k0 + L1_K, n0 : n0 + L1_N
            ]


def golden_pingpong(tensors):
    import torch

    # engine, block, lane, before, ctor, rounds 0..3, dtor-enter/done,
    # resource-dtor, base-mod-64, active
    expected = torch.zeros(BLOCK_DIM * 3, 16, dtype=torch.int32)
    for block_idx in range(BLOCK_DIM):
        active = block_idx < 16
        rounds = [1, 1, 1, 1] if active else [-1, -1, -1, -1]
        expected[block_idx, :14] = torch.tensor(
            [
                1,
                block_idx,
                -1,
                1,
                1,
                *rounds,
                1,
                1,
                1,
                0,
                int(active),
            ],
            dtype=torch.int32,
        )
        for lane in range(2):
            row = BLOCK_DIM + block_idx * 2 + lane
            expected[row, :14] = torch.tensor(
                [
                    2,
                    block_idx,
                    lane,
                    1,
                    1,
                    *rounds,
                    1,
                    1,
                    1,
                    0,
                    int(active),
                ],
                dtype=torch.int32,
            )
    tensors["pingpong_state"][:] = expected


def golden_blockmmad(tensors):
    import torch

    # engine, block, lane, before, resource-ready, BlockMmad ctor/dtor,
    # Resource dtor, base-mod-64
    expected = torch.zeros(BLOCK_DIM * 3, 16, dtype=torch.int32)
    for block_idx in range(BLOCK_DIM):
        expected[block_idx, :9] = torch.tensor(
            [1, block_idx, -1, 1, 1, 1, 1, 1, 0], dtype=torch.int32
        )
        for lane in range(2):
            row = BLOCK_DIM + block_idx * 2 + lane
            expected[row, :9] = torch.tensor(
                [2, block_idx, lane, 1, 1, 1, 1, 1, 0], dtype=torch.int32
            )
    tensors["blockmmad_lifecycle"][:] = expected


def golden_resource(tensors):
    import torch

    # engine, logical block, lane, before, resource-ready, handshake-done,
    # resource-destroyed, base-mod-64
    expected = torch.zeros(BLOCK_DIM * 3, 16, dtype=torch.int32)
    for block_idx in range(BLOCK_DIM):
        expected[block_idx, :8] = torch.tensor(
            [1, block_idx, -1, 1, 1, 1, 1, 0], dtype=torch.int32
        )
        for lane in range(2):
            row = BLOCK_DIM + block_idx * 2 + lane
            expected[row, :8] = torch.tensor(
                [2, block_idx, lane, 1, 1, 1, 1, 0], dtype=torch.int32
            )
    tensors["lifecycle"][:] = expected


def golden_crosscore(tensors):
    import torch

    expected = torch.zeros(BLOCK_DIM * 3, 16, dtype=torch.int32)
    for block_idx in range(BLOCK_DIM):
        expected[block_idx, :5] = torch.tensor(
            [1, block_idx, -1, 1, 1], dtype=torch.int32
        )
        for lane in range(2):
            row = BLOCK_DIM + block_idx * 2 + lane
            expected[row, :5] = torch.tensor(
                [2, block_idx, lane, 1, 1], dtype=torch.int32
            )
    tensors["handshake"][:] = expected


def golden_topology(tensors):
    import torch

    expected = torch.zeros(BLOCK_DIM * 3, 8, dtype=torch.int32)
    for block_idx in range(BLOCK_DIM):
        expected[block_idx, :4] = torch.tensor(
            [1, block_idx, BLOCK_DIM, -1], dtype=torch.int32
        )
        for lane in range(2):
            row = BLOCK_DIM + block_idx * 2 + lane
            expected[row, :4] = torch.tensor(
                [2, block_idx, BLOCK_DIM, lane], dtype=torch.int32
            )
    tensors["topology"][:] = expected


def compare_topology(actual, expected, **_):
    import json
    import torch

    # Preserve the raw native CANN topology in the task log even when its
    # numbering differs from PyPTO's logical persistent-task identity.
    print("TOPOLOGY_COLUMNS=engine,logical_idx,logical_num,logical_lane,"
          "native_idx,native_num,native_lane,native_lanes")
    print("TOPOLOGY_RAW=" + json.dumps(actual.cpu().tolist(), separators=(",", ":")))

    logical_actual = actual[:, :4].cpu()
    logical_expected = expected[:, :4].cpu()
    if torch.equal(logical_actual, logical_expected):
        return True, ""
    mismatches = torch.nonzero(logical_actual != logical_expected)
    details = []
    for row, col in mismatches[:16].tolist():
        details.append(
            f"[{row},{col}] actual={logical_actual[row, col].item()} "
            f"expected={logical_expected[row, col].item()}"
        )
    return False, "logical MIX topology mismatch:\n" + "\n".join(details)


def golden_w4(tensors):
    tensors["out"][:] = _golden_matmul(
        tensors["activation"], _unpack_int4_kn(tensors["packed_weight_kn"])
    )


def golden_int8(tensors):
    tensors["out"][:] = _golden_matmul(
        tensors["activation"], tensors["weight_nk"].T.contiguous()
    )


if __name__ == "__main__":
    import argparse
    from golden import run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=(
            "w4a8",
            "int8",
            "topology",
            "crosscore",
            "resource",
            "blockmmad",
            "pingpong",
            "tilecast",
            "aicmmad",
        ),
        required=True,
    )
    parser.add_argument("-p", "--platform", default="a2a3", choices=("a2a3", "a2a3sim"))
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-l2-swimlane", type=int, nargs="?", const=1, default=0, choices=(0, 1, 2))
    parser.add_argument("--dump-passes", action="store_true")
    args = parser.parse_args()

    is_w4 = args.variant == "w4a8"
    is_topology = args.variant == "topology"
    is_crosscore = args.variant == "crosscore"
    is_resource = args.variant == "resource"
    is_blockmmad = args.variant == "blockmmad"
    is_pingpong = args.variant == "pingpong"
    is_tilecast = args.variant == "tilecast"
    is_aicmmad = args.variant == "aicmmad"
    result = run_jit(
        fn=(
            aic_workspace_mmad_test
            if is_aicmmad
            else tilecast_data_test
            if is_tilecast
            else pingpong_state_test
            if is_pingpong
            else blockmmad_lifecycle_test
            if is_blockmmad
            else resource_lifecycle_test
            if is_resource
            else crosscore_handshake_test
            if is_crosscore
            else mixed_topology_test
            if is_topology
            else w4a8_w2_test
            if is_w4
            else int8_w2_test
        ),
        specs=(
            build_aic_mmad_specs()
            if is_aicmmad
            else build_tilecast_specs()
            if is_tilecast
            else build_pingpong_specs()
            if is_pingpong
            else build_blockmmad_specs()
            if is_blockmmad
            else build_resource_specs()
            if is_resource
            else build_crosscore_specs()
            if is_crosscore
            else build_topology_specs()
            if is_topology
            else build_w4_specs()
            if is_w4
            else build_int8_specs()
        ),
        golden_fn=(
            golden_aic_mmad
            if is_aicmmad
            else golden_tilecast
            if is_tilecast
            else golden_pingpong
            if is_pingpong
            else golden_blockmmad
            if is_blockmmad
            else golden_resource
            if is_resource
            else golden_crosscore
            if is_crosscore
            else golden_topology
            if is_topology
            else golden_w4
            if is_w4
            else golden_int8
        ),
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=0.0,
        atol=0.0,
        compare_fn={"topology": compare_topology} if is_topology else None,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
