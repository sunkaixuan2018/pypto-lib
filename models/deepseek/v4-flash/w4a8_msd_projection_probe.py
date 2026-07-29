# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the LICENSE.
# -----------------------------------------------------------------------------------------------------------
"""Compile/correctness probe for direct packed-W4 MSD expert projections.

For a signed INT8 activation x, define signed INT4 planes:
    hi = ((x + 128) // 16) - 8
    lo = ((x + 128) % 16) - 8
so x = 16*hi + lo + 8 exactly.

The two planes are stacked into logical M=32 and multiplied by the same packed
signed-INT4 weights in one I4xI4 matmul. This reads each W4 tile once. A small
vector epilogue reconstructs:
    out = 16*acc_hi + acc_lo + 8*sum(weight).
"""

import os
from pathlib import Path

import pypto.language as pl


M = 16
PLANES_M = 2 * M
N_TILE = 256
INNER = 4

GATE_K = 4096
GATE_N = 2048
GATE_BLOCKS = 2

W2_K = 2048
W2_N = 4096
W2_BLOCKS = 4

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
_KERNEL_DIR = _HERE / "kernels" / "w4a8_msd_projection"
_CATLASS_INCLUDE = _REPO_ROOT / "third_party" / "catlass" / "include"


def _cann_include_dirs() -> tuple[Path, ...]:
    cann_root = Path(
        os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/latest")
    )
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


_EXTERN_INCLUDE_DIRS = (_CATLASS_INCLUDE, _KERNEL_DIR) + _cann_include_dirs()
_GATE_ENTRY = _KERNEL_DIR / "gate_entry.cpp"
_W2_ENTRY = _KERNEL_DIR / "w2_entry.cpp"
_SPLIT_GATE_ENTRY = _KERNEL_DIR / "split_gate_entry.cpp"
_SPLIT_W2_ENTRY = _KERNEL_DIR / "split_w2_entry.cpp"


@pl.jit.extern(
    core_type="mixed",
    aic_source=_SPLIT_GATE_ENTRY,
    aiv_source=_SPLIT_GATE_ENTRY,
    include_dirs=_EXTERN_INCLUDE_DIRS,
    dual_aiv_dispatch=True,
)
def gate_split_msd_cce(
    packed_planes: pl.Out[pl.Tensor],
    activation: pl.Tensor,
) -> pl.Tensor: ...


@pl.jit.extern(
    core_type="mixed",
    aic_source=_SPLIT_W2_ENTRY,
    aiv_source=_SPLIT_W2_ENTRY,
    include_dirs=_EXTERN_INCLUDE_DIRS,
    dual_aiv_dispatch=True,
)
def w2_split_msd_cce(
    packed_planes: pl.Out[pl.Tensor],
    activation: pl.Tensor,
) -> pl.Tensor: ...


@pl.jit.extern(
    core_type="mixed",
    aic_source=_GATE_ENTRY,
    aiv_source=_GATE_ENTRY,
    include_dirs=_EXTERN_INCLUDE_DIRS,
    dual_aiv_dispatch=True,
)
def gate_i4_msd_cce(
    acc_planes: pl.Out[pl.Tensor],
    packed_planes: pl.Tensor,
    packed_weight: pl.Tensor,
) -> pl.Tensor: ...


@pl.jit.extern(
    core_type="mixed",
    aic_source=_W2_ENTRY,
    aiv_source=_W2_ENTRY,
    include_dirs=_EXTERN_INCLUDE_DIRS,
    dual_aiv_dispatch=True,
)
def w2_i4_msd_cce(
    acc_planes: pl.Out[pl.Tensor],
    packed_planes: pl.Tensor,
    packed_weight: pl.Tensor,
) -> pl.Tensor: ...


@pl.jit
def gate_msd_packed(
    activation: pl.Tensor[[M, GATE_K], pl.INT8],
    packed_weight: pl.Tensor[[GATE_N, GATE_K // 2], pl.INT8],
    sum_weight: pl.Tensor[[GATE_N], pl.INT32],
    out: pl.Out[pl.Tensor[[M, GATE_N], pl.INT32]],
):
    packed_planes = pl.create_tensor(
        [PLANES_M, GATE_K // 2], dtype=pl.INT8
    )
    with pl.spmd(GATE_BLOCKS, name_hint="gate_split_msd"):
        packed_planes = gate_split_msd_cce(
            packed_planes, activation
        )
    acc_planes = pl.create_tensor([PLANES_M, GATE_N], dtype=pl.INT32)
    with pl.spmd(GATE_BLOCKS, name_hint="gate_i4_msd"):
        acc_planes = gate_i4_msd_cce(
            acc_planes, packed_planes, packed_weight
        )
    with pl.spmd(GATE_BLOCKS, name_hint="gate_msd_reconstruct"):
        block_idx = pl.tile.get_block_idx()
        n_base = block_idx * INNER * N_TILE
        for inner in pl.range(INNER):
            n0 = n_base + inner * N_TILE
            acc_hi = acc_planes[0:M, n0:n0 + N_TILE]
            acc_lo = acc_planes[M:PLANES_M, n0:n0 + N_TILE]
            combined = pl.add(pl.mul(acc_hi, 16), acc_lo)
            assist = pl.mul(
                pl.reshape(sum_weight[n0:n0 + N_TILE], [1, N_TILE]),
                8,
            )
            out[:, n0:n0 + N_TILE] = pl.col_expand_add(
                combined, assist
            )
    return out


@pl.jit
def w2_msd_packed(
    activation: pl.Tensor[[M, W2_K], pl.INT8],
    packed_weight: pl.Tensor[[W2_N, W2_K // 2], pl.INT8],
    sum_weight: pl.Tensor[[W2_N], pl.INT32],
    out: pl.Out[pl.Tensor[[M, W2_N], pl.INT32]],
):
    packed_planes = pl.create_tensor(
        [PLANES_M, W2_K // 2], dtype=pl.INT8
    )
    with pl.spmd(W2_BLOCKS, name_hint="w2_split_msd"):
        packed_planes = w2_split_msd_cce(
            packed_planes, activation
        )
    acc_planes = pl.create_tensor([PLANES_M, W2_N], dtype=pl.INT32)
    with pl.spmd(W2_BLOCKS, name_hint="w2_i4_msd"):
        acc_planes = w2_i4_msd_cce(
            acc_planes, packed_planes, packed_weight
        )
    with pl.spmd(W2_BLOCKS, name_hint="w2_msd_reconstruct"):
        block_idx = pl.tile.get_block_idx()
        n_base = block_idx * INNER * N_TILE
        for inner in pl.range(INNER):
            n0 = n_base + inner * N_TILE
            acc_hi = acc_planes[0:M, n0:n0 + N_TILE]
            acc_lo = acc_planes[M:PLANES_M, n0:n0 + N_TILE]
            combined = pl.add(pl.mul(acc_hi, 16), acc_lo)
            assist = pl.mul(
                pl.reshape(sum_weight[n0:n0 + N_TILE], [1, N_TILE]),
                8,
            )
            out[:, n0:n0 + N_TILE] = pl.col_expand_add(
                combined, assist
            )
    return out


def _pack_signed_int4(values):
    import torch

    unsigned = torch.bitwise_and(values.to(torch.int16), 0xF)
    packed = unsigned[..., 0::2] | (unsigned[..., 1::2] << 4)
    return packed.to(torch.uint8).view(torch.int8).contiguous()


def build_specs(k: int, n: int):
    import torch
    from golden import TensorSpec

    row = torch.arange(M, dtype=torch.int16)[:, None]
    col = torch.arange(k, dtype=torch.int16)[None, :]
    activation = (((11 * row + 17 * col + 3) % 256) - 128).to(
        torch.int8
    )
    out_col = torch.arange(n, dtype=torch.int16)[:, None]
    in_col = torch.arange(k, dtype=torch.int16)[None, :]
    weight = (((7 * out_col + 3 * in_col + 2) % 16) - 8).to(
        torch.int8
    )
    packed_weight = _pack_signed_int4(weight)
    sum_weight = weight.to(torch.int32).sum(dim=1)
    golden = torch.matmul(
        activation.to(torch.int32), weight.to(torch.int32).t()
    )

    return [
        TensorSpec(
            "activation", [M, k], torch.int8,
            init_value=lambda: activation,
        ),
        TensorSpec(
            "packed_weight", [n, k // 2], torch.int8,
            init_value=lambda: packed_weight,
        ),
        TensorSpec(
            "sum_weight", [n], torch.int32,
            init_value=lambda: sum_weight,
        ),
        TensorSpec("out", [M, n], torch.int32, is_output=True),
    ], golden


def make_golden(expected):
    def golden_projection(tensors):
        tensors["out"][:] = expected

    return golden_projection


if __name__ == "__main__":
    import argparse

    from golden import run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shape", required=True, choices=("gate", "w2")
    )
    parser.add_argument(
        "-p", "--platform", default="a2a3", choices=("a2a3", "a2a3sim")
    )
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument(
        "--enable-pmu", type=int, nargs="?", const=2, default=0
    )
    args = parser.parse_args()

    if args.shape == "gate":
        fn, k, n = gate_msd_packed, GATE_K, GATE_N
    else:
        fn, k, n = w2_msd_packed, W2_K, W2_N
    specs, expected = build_specs(k, n)

    result = run_jit(
        fn=fn,
        specs=specs,
        golden_fn=make_golden(expected),
        compile_only=args.compile_only,
        compile_cfg=dict(),
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_pmu=args.enable_pmu,
        ),
        rtol=0.0,
        atol=0.0,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
