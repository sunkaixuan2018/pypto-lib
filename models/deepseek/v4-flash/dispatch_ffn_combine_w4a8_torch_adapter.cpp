/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0.
 * Please refer to the License for details. You may not use this file except in
 * compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
 * KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
 * NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the
 * License.
 */

/**
 * Thin Torch registration shim for vLLM-Ascend's fused W4A8 MoE operator.
 *
 * The operator implementation and ACLNN adapter remain in the pinned
 * vLLM-Ascend source tree. This translation unit registers only the one
 * operation needed by the standalone EP8 probe.
 */

#include <torch/extension.h>
#include <torch/library.h>

#include "aclnn_torch_adapter/op_api_common.h"
#include "mc2/dispatch_ffn_combine/dispatch_ffn_combine_torch_adpt.h"

thread_local char g_hashBuf[kHashBufSize];
thread_local int g_hashOffset = 0;

TORCH_LIBRARY(_C_ascend, ops)
{
    ops.def(
        "dispatch_ffn_combine(Tensor x, Tensor[] weight1, Tensor[] weight2, "
        "Tensor expert_idx, Tensor[] scale1, Tensor[] scale2, Tensor[] bias1, "
        "Tensor[] bias2, Tensor probs, str group, int max_output_size, "
        "Tensor! out, Tensor! expert_token_nums, Tensor? x_active_mask=None, "
        "float swiglu_limit=1000000.0) -> "
        "(Tensor out, Tensor expert_token_nums)");
    ops.impl(
        "dispatch_ffn_combine", torch::kPrivateUse1,
        &vllm_ascend::dispatch_ffn_combine);
}
