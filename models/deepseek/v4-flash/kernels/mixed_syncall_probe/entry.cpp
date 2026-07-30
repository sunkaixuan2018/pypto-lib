/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the LICENSE.
 * -----------------------------------------------------------------------------------------------------------
 */

#include <cstdint>

#include "tensor.h"

#ifdef __CPU_SIM

#ifndef __gm__
#define __gm__
#endif
#ifndef __aicore__
#define __aicore__ [aicore]
#endif
extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    (void)args;
}

#else

#include "intrinsic.h"
#include "kernel_operator.h"

namespace {

constexpr uint32_t kCubeBlocks = 24;

template <typename T>
static __aicore__ __attribute__((always_inline)) __gm__ T *
tensor_data(__gm__ int64_t *args, int32_t index) {
    __gm__ ::Tensor *tensor =
        reinterpret_cast<__gm__ ::Tensor *>(args[index]);
    return reinterpret_cast<__gm__ T *>(tensor->buffer.addr) +
           tensor->start_offset;
}

}  // namespace

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    KERNEL_TASK_TYPE(3, KERNEL_TYPE_MIX_AIC_1_2);

    __gm__ int32_t *out = tensor_data<int32_t>(args, 0);
    const uint32_t block = static_cast<uint32_t>(get_block_idx(args));

#ifdef __DAV_C220_CUBE__
    const uint32_t out_index = block;
    const int32_t value = static_cast<int32_t>(1000 + block);
#elif defined(__DAV_C220_VEC__)
    const uint32_t lane = static_cast<uint32_t>(get_sub_block_id(args));
    const uint32_t out_index = kCubeBlocks + block * 2 + lane;
    const int32_t value = static_cast<int32_t>(2000 + block * 2 + lane);
#else
#error "mixed_syncall_probe requires a C220 Cube or Vector target"
#endif

    AscendC::SyncAll<false>();
    out[out_index] = value;
}

#endif
