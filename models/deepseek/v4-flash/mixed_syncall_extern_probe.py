# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Check whole-core SyncAll liveness in a PyPTO mixed extern task."""

from pathlib import Path

import pypto.language as pl


CUBE_BLOCKS = 24
VECTOR_LANES = CUBE_BLOCKS * 2
OUTPUTS = CUBE_BLOCKS + VECTOR_LANES

_HERE = Path(__file__).resolve().parent
_ENTRY = _HERE / "kernels" / "mixed_syncall_probe" / "entry.cpp"


@pl.jit.extern(
    core_type="mixed",
    aic_source=_ENTRY,
    aiv_source=_ENTRY,
    dual_aiv_dispatch=True,
)
def mixed_syncall_cce(out: pl.InOut[pl.Tensor]) -> pl.Tensor: ...


@pl.jit
def mixed_syncall_probe(
    out: pl.InOut[pl.Tensor[[OUTPUTS], pl.INT32]],
):
    with pl.spmd(
        CUBE_BLOCKS,
        name_hint="mixed_syncall_extern",
        sync_start=True,
    ):
        out = mixed_syncall_cce(out)
    return out


def build_specs():
    import torch
    from golden import TensorSpec

    return [
        TensorSpec(
            "out",
            [OUTPUTS],
            torch.int32,
            init_value=lambda: torch.full(
                [OUTPUTS], -1, dtype=torch.int32
            ),
            is_output=True,
        ),
    ]


def golden(tensors):
    import torch

    cube = torch.arange(
        1000, 1000 + CUBE_BLOCKS, dtype=torch.int32
    )
    vector = torch.arange(
        2000, 2000 + VECTOR_LANES, dtype=torch.int32
    )
    tensors["out"].copy_(torch.cat([cube, vector]))


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
        fn=mixed_syncall_probe,
        specs=build_specs(),
        golden_fn=golden,
        compile_cfg=dict(),
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
        ),
        rtol=0.0,
        atol=0.0,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
