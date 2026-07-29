/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
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
extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) { (void)args; }

#else

#include "intrinsic.h"
#include "kernel_operator.h"

namespace {

constexpr uint32_t kBlockDim = 24;
constexpr uint32_t kColumns = 8;

template <typename T>
static __aicore__ __attribute__((always_inline)) __gm__ T *
tensor_data(__gm__ int64_t *args, int32_t index) {
    __gm__ ::Tensor *tensor =
        reinterpret_cast<__gm__ ::Tensor *>(args[index]);
    return reinterpret_cast<__gm__ T *>(tensor->buffer.addr) +
           tensor->start_offset;
}

static __aicore__ __attribute__((always_inline)) void
publish_row(__gm__ int32_t *row, int32_t engine, int32_t logical_idx,
            int32_t logical_num, int32_t logical_lane) {
    row[0] = engine;
    row[1] = logical_idx;
    row[2] = logical_num;
    row[3] = logical_lane;
    row[4] = static_cast<int32_t>(AscendC::GetBlockIdx());
    row[5] = static_cast<int32_t>(AscendC::GetBlockNum());
    row[6] = static_cast<int32_t>(AscendC::GetSubBlockIdx());
    row[7] = static_cast<int32_t>(AscendC::GetSubBlockNum());
    dcci(row, SINGLE_CACHE_LINE, CACHELINE_OUT);
    dsb(DSB_DDR);
}

}  // namespace

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ int32_t *topology = tensor_data<int32_t>(args, 0);
    const uint32_t logical_idx =
        static_cast<uint32_t>(get_block_idx(args));
    const uint32_t logical_num =
        static_cast<uint32_t>(get_block_num(args));

#ifdef __DAV_C220_CUBE__
    const uint32_t row = logical_idx;
    publish_row(topology + row * kColumns, 1,
                static_cast<int32_t>(logical_idx),
                static_cast<int32_t>(logical_num), -1);
#elif defined(__DAV_C220_VEC__)
    const uint32_t logical_lane =
        static_cast<uint32_t>(get_sub_block_id(args));
    const uint32_t row =
        kBlockDim + logical_idx * 2 + logical_lane;
    publish_row(topology + row * kColumns, 2,
                static_cast<int32_t>(logical_idx),
                static_cast<int32_t>(logical_num),
                static_cast<int32_t>(logical_lane));
#endif
}

#endif
