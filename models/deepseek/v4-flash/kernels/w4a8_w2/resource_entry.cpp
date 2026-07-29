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
extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) { (void)args; }

#else

#ifndef CATLASS_ARCH
#define CATLASS_ARCH 2201
#endif

#include "intrinsic.h"
#include "kernel_operator.h"

#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"

namespace {

constexpr uint32_t kBlockDim = 24;
constexpr uint32_t kColumns = 16;
constexpr uint16_t kCopyReady = 0;
constexpr uint16_t kPrologueReady = 1;

template <typename T>
static __aicore__ __attribute__((always_inline)) __gm__ T *
tensor_data(__gm__ int64_t *args, int32_t index) {
    __gm__ ::Tensor *tensor =
        reinterpret_cast<__gm__ ::Tensor *>(args[index]);
    return reinterpret_cast<__gm__ T *>(tensor->buffer.addr) +
           tensor->start_offset;
}

static __aicore__ __attribute__((always_inline)) void
flush_row(__gm__ int32_t *row) {
    dcci(row, SINGLE_CACHE_LINE, CACHELINE_OUT);
    dsb(DSB_DDR);
}

static __aicore__ __attribute__((always_inline)) __gm__ int32_t *
initialize_row(__gm__ int32_t *lifecycle, uint32_t block_idx,
               int32_t engine, int32_t lane) {
    const uint32_t row_idx = engine == 1
        ? block_idx
        : kBlockDim + block_idx * 2 + static_cast<uint32_t>(lane);
    __gm__ int32_t *row = lifecycle + row_idx * kColumns;
    for (uint32_t column = 0; column < kColumns; ++column) {
        row[column] = 0;
    }
    row[0] = engine;
    row[1] = static_cast<int32_t>(block_idx);
    row[2] = lane;
    row[3] = 1;
    row[7] = static_cast<int32_t>(
        reinterpret_cast<uint64_t>(lifecycle) & 63U);
    flush_row(row);
    return row;
}

}  // namespace

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ int32_t *lifecycle = tensor_data<int32_t>(args, 0);
    const uint32_t block_idx =
        static_cast<uint32_t>(get_block_idx(args));

#ifdef __DAV_C220_CUBE__
    __gm__ int32_t *row = initialize_row(
        lifecycle, block_idx, 1, -1);
    {
        Catlass::Arch::Resource<Catlass::Arch::AtlasA2> resource;
        row[4] = 1;
        flush_row(row);

        AscendC::CrossCoreSetFlag<0x02, PIPE_MTE2>(kCopyReady);
        AscendC::CrossCoreWaitFlag(kPrologueReady);
        row[5] = 1;
        flush_row(row);
    }
    row[6] = 1;
    flush_row(row);
#elif defined(__DAV_C220_VEC__)
    const uint32_t lane =
        static_cast<uint32_t>(get_sub_block_id(args));
    __gm__ int32_t *row = initialize_row(
        lifecycle, block_idx, 2, static_cast<int32_t>(lane));
    {
        Catlass::Arch::Resource<Catlass::Arch::AtlasA2> resource;
        row[4] = 1;
        flush_row(row);

        AscendC::CrossCoreWaitFlag(kCopyReady);
        AscendC::CrossCoreSetFlag<0x02, PIPE_MTE3>(kPrologueReady);
        row[5] = 1;
        flush_row(row);
    }
    row[6] = 1;
    flush_row(row);
#endif
}

#endif
