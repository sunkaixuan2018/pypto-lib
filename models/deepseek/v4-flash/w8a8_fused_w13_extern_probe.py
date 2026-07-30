# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Validate the corrected CANN W8A8 fused W13 path inside PyPTO extern."""

import math
import os
from pathlib import Path

import pypto.language as pl


M = 256
K = 4096
N = 4096
OUT_N = N // 2
EXPERTS = 16
CUBE_BLOCKS = 24

_HERE = Path(__file__).resolve().parent
_KERNEL_DIR = _HERE / "kernels" / "w8a8_fused_w13"
_ENTRY = _KERNEL_DIR / "entry.cpp"
_VLLM_ASCEND = Path(
    os.environ.get(
        "VLLM_ASCEND_SOURCE",
        "/data/sunkaixuan/sunkaixuan_subdir/all_libs/"
        "vllm-ascend-dsv4-gmm-20260730",
    )
)
_FUSION_KERNEL_DIR = (
    _VLLM_ASCEND
    / "csrc"
    / "gmm"
    / "grouped_matmul_swiglu_quant_v2"
    / "op_kernel"
)


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
    (_KERNEL_DIR, _FUSION_KERNEL_DIR) + _cann_include_dirs()
)


@pl.jit.extern(
    core_type="mixed",
    aic_source=_ENTRY,
    aiv_source=_ENTRY,
    include_dirs=_EXTERN_INCLUDE_DIRS,
    dual_aiv_dispatch=True,
)
def fused_w13_cce(
    out: pl.Out[pl.Tensor],
    out_scale: pl.InOut[pl.Tensor],
    workspace: pl.InOut[pl.Tensor],
    x: pl.Tensor,
    x_scale: pl.Tensor,
    group_list: pl.Tensor,
    weight_nz: pl.Tensor,
    weight_scale: pl.Tensor,
) -> pl.Tensor: ...


@pl.jit
def fused_w13_probe(
    x: pl.Tensor[[M, K], pl.INT8],
    x_scale: pl.Tensor[[M], pl.FP32],
    group_list: pl.Tensor[[EXPERTS], pl.INT64],
    weight_nz: pl.Tensor[[EXPERTS, N, K], pl.INT8],
    weight_scale: pl.Tensor[[EXPERTS, N], pl.FP32],
    out: pl.Out[pl.Tensor[[M, OUT_N], pl.INT8]],
    out_scale: pl.InOut[pl.Tensor[[M], pl.FP32]],
    workspace: pl.InOut[pl.Tensor[[M, N], pl.INT32]],
):
    with pl.spmd(
        CUBE_BLOCKS,
        name_hint="w8a8_fused_w13_extern",
        sync_start=True,
    ):
        out = fused_w13_cce(
            out,
            out_scale,
            workspace,
            x,
            x_scale,
            group_list,
            weight_nz,
            weight_scale,
        )
    return out


def build_specs():
    import torch
    from golden import TensorSpec

    group_list = (
        torch.arange(1, EXPERTS + 1, dtype=torch.int64)
        * (M // EXPERTS)
    )
    return [
        TensorSpec(
            "x", [M, K], torch.int8,
            init_value=lambda: torch.ones(M, K, dtype=torch.int8),
        ),
        TensorSpec(
            "x_scale", [M], torch.float32,
            init_value=lambda: torch.ones(M, dtype=torch.float32),
        ),
        TensorSpec(
            "group_list", [EXPERTS], torch.int64,
            init_value=lambda: group_list,
        ),
        TensorSpec(
            "weight_nz", [EXPERTS, N, K], torch.int8,
            init_value=lambda: torch.ones(
                EXPERTS, N, K, dtype=torch.int8
            ),
        ),
        TensorSpec(
            "weight_scale", [EXPERTS, N], torch.float32,
            init_value=lambda: torch.ones(
                EXPERTS, N, dtype=torch.float32
            ),
        ),
        TensorSpec("out", [M, OUT_N], torch.int8, is_output=True),
        TensorSpec(
            "out_scale", [M], torch.float32, is_output=True,
        ),
        TensorSpec(
            "workspace", [M, N], torch.int32,
            init_value=lambda: torch.zeros(M, N, dtype=torch.int32),
        ),
    ]


def golden(tensors):
    tensors["out"].fill_(127)
    gate = 10.0
    up = 10.0
    activation = gate / (1.0 + math.exp(-gate)) * up
    tensors["out_scale"].fill_(activation / 127.0)


if __name__ == "__main__":
    import argparse

    from golden import run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--platform", default="a2a3", choices=("a2a3",)
    )
    parser.add_argument("-d", "--device", type=int, default=0)
    args = parser.parse_args()

    result = run_jit(
        fn=fused_w13_probe,
        specs=build_specs(),
        golden_fn=golden,
        compile_cfg=dict(),
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
        ),
        rtol=0.002,
        atol=0.0,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
