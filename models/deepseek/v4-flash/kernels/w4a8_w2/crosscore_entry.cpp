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
// CACHELINE_OUT writes a complete 64-byte A3 data-cache line.  Give every
// physical engine its own line so one core cannot write back a stale copy of
// another core's before/after fields.
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

}  // namespace

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ int32_t *handshake = tensor_data<int32_t>(args, 0);
    const uint32_t block_idx =
        static_cast<uint32_t>(get_block_idx(args));

#ifdef __DAV_C220_CUBE__
    __gm__ int32_t *row = handshake + block_idx * kColumns;
    row[0] = 1;
    row[1] = static_cast<int32_t>(block_idx);
    row[2] = -1;
    row[3] = 1;
    row[4] = 0;
    row[5] = static_cast<int32_t>(
        reinterpret_cast<uint64_t>(handshake) & 63U);
    for (uint32_t column = 6; column < kColumns; ++column) {
        row[column] = 0;
    }
    flush_row(row);

    // Stock Catlass broadcasts one copy-ready flag to both AIV lanes, then
    // consumes the prologue-ready response once.
    AscendC::CrossCoreSetFlag<0x02, PIPE_MTE2>(kCopyReady);
    AscendC::CrossCoreWaitFlag(kPrologueReady);

    row[4] = 1;
    flush_row(row);
#elif defined(__DAV_C220_VEC__)
    const uint32_t lane =
        static_cast<uint32_t>(get_sub_block_id(args));
    const uint32_t row_idx = kBlockDim + block_idx * 2 + lane;
    __gm__ int32_t *row = handshake + row_idx * kColumns;
    row[0] = 2;
    row[1] = static_cast<int32_t>(block_idx);
    row[2] = static_cast<int32_t>(lane);
    row[3] = 1;
    row[4] = 0;
    row[5] = static_cast<int32_t>(
        reinterpret_cast<uint64_t>(handshake) & 63U);
    for (uint32_t column = 6; column < kColumns; ++column) {
        row[column] = 0;
    }
    flush_row(row);

    AscendC::CrossCoreWaitFlag(kCopyReady);
    AscendC::CrossCoreSetFlag<0x02, PIPE_MTE3>(kPrologueReady);

    row[4] = 1;
    flush_row(row);
#endif
}

#endif
