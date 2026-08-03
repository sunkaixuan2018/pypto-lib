# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Fixed-shape EP8 hash routing through a compact AIV extern kernel."""

import os
from pathlib import Path

import pypto.language as pl


TOKENS = 8
EXPERTS = 128
TOPK = 6
SCORE_PAD = 256

_HERE = Path(__file__).resolve().parent
_ENTRY = _HERE / "kernels" / "route_hash_aiv" / "entry.cpp"


def _cann_include_dirs() -> tuple[Path, ...]:
    cann_root = Path(
        os.environ.get(
            "ASCEND_HOME_PATH",
            "/usr/local/Ascend/cann-9.0.0",
        )
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


@pl.jit.extern(
    core_type="aiv",
    source=_ENTRY,
    include_dirs=_cann_include_dirs(),
)
def route_hash_aiv_cce(
    weights: pl.InOut[pl.Tensor],
    indices: pl.InOut[pl.Tensor],
    route_scores: pl.Tensor,
    input_ids: pl.Tensor,
    tid2eid: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
) -> tuple[
    pl.Tensor[[TOKENS, TOPK], pl.FP32],
    pl.Tensor[[TOKENS, TOPK], pl.INT32],
]:
    return weights, indices
