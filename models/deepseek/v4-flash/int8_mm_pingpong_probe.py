# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compare production-shaped PyPTO INT8 matmul with explicit Catlass L1 ping-pong.

Both backends use the exact expert-routed task topology:
* gate/up: M=16, K=4096, N=2048, 2 blocks, 4x256 N tiles per block;
* W2:      M=16, K=2048, N=4096, 4 blocks, 4x256 N tiles per block.

Only the K-loop implementation changes. The Catlass candidate keeps two
independent GM->L1 stages and has no prologue or GM workspace.
"""

import os
from pathlib import Path

import pypto.language as pl


M = 16
N_TILE = 256
INNER = 4
K_TILE = 512

GATE_K = 4096
GATE_N = 2048
GATE_BLOCKS = 2

W2_K = 2048
W2_N = 4096
W2_BLOCKS = 4

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
_KERNEL_DIR = _HERE / "kernels" / "int8_mm_pingpong"
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


@pl.jit.extern(
    core_type="mixed",
    aic_source=_GATE_ENTRY,
    aiv_source=_GATE_ENTRY,
    include_dirs=_EXTERN_INCLUDE_DIRS,
    dual_aiv_dispatch=True,
)
def gate_pingpong_cce(
    out: pl.Out[pl.Tensor],
    activation: pl.Tensor,
    weight_nk: pl.Tensor,
) -> pl.Tensor: ...


@pl.jit.extern(
    core_type="mixed",
    aic_source=_W2_ENTRY,
    aiv_source=_W2_ENTRY,
    include_dirs=_EXTERN_INCLUDE_DIRS,
    dual_aiv_dispatch=True,
)
def w2_pingpong_cce(
    out: pl.Out[pl.Tensor],
    activation: pl.Tensor,
    weight_nk: pl.Tensor,
) -> pl.Tensor: ...


@pl.jit
def gate_control(
    activation: pl.Tensor[[M, GATE_K], pl.INT8],
    weight_nk: pl.Tensor[[GATE_N, GATE_K], pl.INT8],
    out: pl.Out[pl.Tensor[[M, GATE_N], pl.INT32]],
):
    with pl.spmd(GATE_BLOCKS, name_hint="gate_control"):
        block_idx = pl.tile.get_block_idx()
        n_base = block_idx * INNER * N_TILE
        for inner in pl.range(INNER):
            n0 = n_base + inner * N_TILE
            acc = pl.create_tensor([M, N_TILE], dtype=pl.INT32)
            for k0 in pl.pipeline(0, GATE_K, K_TILE, stage=2):
                a_k = activation[:, k0:k0 + K_TILE]
                b_k = weight_nk[n0:n0 + N_TILE, k0:k0 + K_TILE]
                if k0 == 0:
                    acc = pl.matmul(
                        a_k, b_k, b_trans=True, out_dtype=pl.INT32
                    )
                else:
                    acc = pl.matmul_acc(
                        acc, a_k, b_k, b_trans=True
                    )
            out[:, n0:n0 + N_TILE] = acc
    return out


@pl.jit
def gate_pingpong(
    activation: pl.Tensor[[M, GATE_K], pl.INT8],
    weight_nk: pl.Tensor[[GATE_N, GATE_K], pl.INT8],
    out: pl.Out[pl.Tensor[[M, GATE_N], pl.INT32]],
):
    with pl.spmd(GATE_BLOCKS, name_hint="gate_pingpong"):
        out = gate_pingpong_cce(out, activation, weight_nk)
    return out


@pl.jit
def w2_control(
    activation: pl.Tensor[[M, W2_K], pl.INT8],
    weight_nk: pl.Tensor[[W2_N, W2_K], pl.INT8],
    out: pl.Out[pl.Tensor[[M, W2_N], pl.INT32]],
):
    with pl.spmd(W2_BLOCKS, name_hint="w2_control"):
        block_idx = pl.tile.get_block_idx()
        n_base = block_idx * INNER * N_TILE
        for inner in pl.range(INNER):
            n0 = n_base + inner * N_TILE
            acc = pl.create_tensor([M, N_TILE], dtype=pl.INT32)
            for k0 in pl.pipeline(0, W2_K, K_TILE, stage=2):
                a_k = activation[:, k0:k0 + K_TILE]
                b_k = weight_nk[n0:n0 + N_TILE, k0:k0 + K_TILE]
                if k0 == 0:
                    acc = pl.matmul(
                        a_k, b_k, b_trans=True, out_dtype=pl.INT32
                    )
                else:
                    acc = pl.matmul_acc(
                        acc, a_k, b_k, b_trans=True
                    )
            out[:, n0:n0 + N_TILE] = acc
    return out


@pl.jit
def w2_pingpong(
    activation: pl.Tensor[[M, W2_K], pl.INT8],
    weight_nk: pl.Tensor[[W2_N, W2_K], pl.INT8],
    out: pl.Out[pl.Tensor[[M, W2_N], pl.INT32]],
):
    with pl.spmd(W2_BLOCKS, name_hint="w2_pingpong"):
        out = w2_pingpong_cce(out, activation, weight_nk)
    return out


def build_specs(k: int, n: int):
    import torch
    from golden import TensorSpec

    row = torch.arange(M, dtype=torch.int16)[:, None]
    col = torch.arange(k, dtype=torch.int16)[None, :]
    activation = (((3 * row + 5 * col + 1) % 7) - 3).to(torch.int8)

    out_col = torch.arange(n, dtype=torch.int16)[:, None]
    in_col = torch.arange(k, dtype=torch.int16)[None, :]
    weight = (((7 * out_col + 3 * in_col + 2) % 9) - 4).to(torch.int8)

    return [
        TensorSpec(
            "activation", [M, k], torch.int8,
            init_value=lambda: activation,
        ),
        TensorSpec(
            "weight_nk", [n, k], torch.int8,
            init_value=lambda: weight,
        ),
        TensorSpec("out", [M, n], torch.int32, is_output=True),
    ]


def golden_projection(tensors):
    import torch

    tensors["out"][:] = torch.matmul(
        tensors["activation"].to(torch.int32),
        tensors["weight_nk"].to(torch.int32).t(),
    )


if __name__ == "__main__":
    import argparse

    from golden import run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shape", required=True, choices=("gate", "w2")
    )
    parser.add_argument(
        "--backend", required=True, choices=("control", "pingpong")
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
        fn = gate_control if args.backend == "control" else gate_pingpong
        k, n = GATE_K, GATE_N
    else:
        fn = w2_control if args.backend == "control" else w2_pingpong
        k, n = W2_K, W2_N

    result = run_jit(
        fn=fn,
        specs=build_specs(k, n),
        golden_fn=golden_projection,
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
