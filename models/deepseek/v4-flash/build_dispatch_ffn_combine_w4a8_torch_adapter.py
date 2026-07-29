# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Build the thin Torch registration shim for the fused W4A8 MoE probe."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch_npu
from torch.utils.cpp_extension import load


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-ascend-source", type=Path, required=True)
    parser.add_argument("--custom-op-root", type=Path, required=True)
    parser.add_argument("--cann-root", type=Path, default=Path("/usr/local/Ascend/cann-9.0.0"))
    parser.add_argument("--build-directory", type=Path, required=True)
    args = parser.parse_args()

    source_dir = args.vllm_ascend_source.resolve()
    cann_root = args.cann_root.resolve()
    custom_vendor = (args.custom_op_root / "vendors" / "custom_transformer").resolve()
    torch_npu_root = Path(torch_npu.__file__).resolve().parent
    build_directory = args.build_directory.resolve()
    build_directory.mkdir(parents=True, exist_ok=True)

    local_dir = Path(__file__).resolve().parent
    sources = [
        local_dir / "dispatch_ffn_combine_w4a8_torch_adapter.cpp",
        source_dir / "csrc" / "aclnn_torch_adapter" / "NPUBridge.cpp",
        source_dir / "csrc" / "aclnn_torch_adapter" / "NPUStorageImpl.cpp",
    ]
    include_paths = [
        source_dir / "csrc",
        cann_root / "aarch64-linux" / "include",
        custom_vendor / "op_api" / "include" / "aclnnop",
        torch_npu_root / "include",
    ]
    library_paths = [
        torch_npu_root / "lib",
        cann_root / "aarch64-linux" / "lib64",
        custom_vendor / "op_api" / "lib",
    ]
    rpaths = [f"-Wl,-rpath,{path}" for path in library_paths]

    load(
        name="w4a8_dispatch_torch",
        sources=[str(path) for path in sources],
        extra_include_paths=[str(path) for path in include_paths],
        extra_cflags=["-O2"],
        extra_ldflags=[
            *(f"-L{path}" for path in library_paths),
            *rpaths,
            "-ltorch_npu",
            "-lascendcl",
            "-lcust_opapi",
            "-lopapi",
            "-lnnopbase",
        ],
        build_directory=str(build_directory),
        verbose=True,
        with_cuda=False,
        is_python_module=False,
    )
    libraries = sorted(build_directory.glob("w4a8_dispatch_torch*.so"))
    if len(libraries) != 1:
        raise RuntimeError(f"expected one built adapter, found {libraries}")
    print(libraries[0])


if __name__ == "__main__":
    main()
