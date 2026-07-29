# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""AIC-only probe for native INT4 x INT4 with FRACTAL_NZ weights.

The two signed-INT4 activation planes are packed offline. The kernel directly
consumes the packed RowMajor activation and packed zN weight and writes the two
INT32 accumulator groups. There is no runtime split, reconstruction, AIV
kernel, INT4-to-INT8 cast, or INT8 weight workspace.
"""

import os
from pathlib import Path

import pypto.language as pl


M = 32
K = 4096
N = 2048
BLOCKS = 2

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
_KERNEL_DIR = _HERE / "kernels" / "int4_nz_projection_probe"
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


_EXTERN_INCLUDE_DIRS = (
    _CATLASS_INCLUDE,
    _KERNEL_DIR,
) + _cann_include_dirs()
_GATE_ENTRY = _KERNEL_DIR / "gate_entry.cpp"


@pl.jit.extern(
    source=_GATE_ENTRY,
    core_type="aic",
    include_dirs=_EXTERN_INCLUDE_DIRS,
)
def gate_i4_nz_cce(
    acc_planes: pl.Out[pl.Tensor],
    packed_planes: pl.Tensor,
    packed_weight_z_n: pl.Tensor,
) -> pl.Tensor: ...


@pl.jit
def gate_i4_nz(
    packed_planes: pl.Tensor[[M, K // 2], pl.INT8],
    packed_weight_z_n: pl.Tensor[[N, K // 2], pl.INT8],
    acc_planes: pl.Out[pl.Tensor[[M, N], pl.INT32]],
):
    with pl.spmd(BLOCKS, name_hint="gate_i4_nz"):
        acc_planes = gate_i4_nz_cce(
            acc_planes,
            packed_planes,
            packed_weight_z_n,
        )
    return acc_planes


def _pack_signed_int4(values):
    import torch

    unsigned = torch.bitwise_and(values.to(torch.int16), 0xF)
    packed = unsigned[..., 0::2] | (unsigned[..., 1::2] << 4)
    return packed.to(torch.uint8).view(torch.int8).contiguous()


def _pack_weight_z_n(weight):
    """Pack logical [K,N] signed INT4 weight into Catlass zN bytes."""
    packed = _pack_signed_int4(weight)
    packed = packed.reshape(K // 16, 16, N // 64, 32)
    packed = packed.permute(2, 0, 1, 3).contiguous()
    return packed.reshape(N, K // 2)


def build_specs():
    import torch
    from golden import TensorSpec

    row = torch.arange(M, dtype=torch.int16)[:, None]
    col = torch.arange(K, dtype=torch.int16)[None, :]
    planes = (((11 * row + 5 * col + 1) % 16) - 8).to(torch.int8)

    in_col = torch.arange(K, dtype=torch.int16)[:, None]
    out_col = torch.arange(N, dtype=torch.int16)[None, :]
    weight = (((3 * in_col + 7 * out_col + 2) % 16) - 8).to(
        torch.int8
    )

    packed_planes = _pack_signed_int4(planes)
    packed_weight_z_n = _pack_weight_z_n(weight)
    expected = torch.matmul(
        planes.to(torch.int32),
        weight.to(torch.int32),
    )

    return [
        TensorSpec(
            "packed_planes",
            [M, K // 2],
            torch.int8,
            init_value=lambda: packed_planes,
        ),
        TensorSpec(
            "packed_weight_z_n",
            [N, K // 2],
            torch.int8,
            init_value=lambda: packed_weight_z_n,
        ),
        TensorSpec(
            "acc_planes",
            [M, N],
            torch.int32,
            is_output=True,
        ),
    ], expected


def make_golden(expected):
    def golden_projection(tensors):
        tensors["acc_planes"][:] = expected

    return golden_projection


if __name__ == "__main__":
    import argparse

    from golden import run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--platform",
        default="a2a3",
        choices=("a2a3", "a2a3sim"),
    )
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument(
        "--enable-pmu",
        type=int,
        nargs="?",
        const=2,
        default=0,
    )
    args = parser.parse_args()

    specs, expected = build_specs()
    result = run_jit(
        fn=gate_i4_nz,
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
