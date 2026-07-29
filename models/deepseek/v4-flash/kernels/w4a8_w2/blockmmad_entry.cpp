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
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/layout/layout.hpp"

namespace {

constexpr uint32_t kBlockDim = 24;
constexpr uint32_t kColumns = 16;

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
    row[8] = static_cast<int32_t>(
        reinterpret_cast<uint64_t>(lifecycle) & 63U);
    flush_row(row);
    return row;
}

using namespace Catlass;

using LayoutA = layout::RowMajor;
using LayoutPrologueB = layout::RowMajor;
using LayoutB = layout::RowMajor;
using LayoutC = layout::RowMajor;

using ElementA = int8_t;
using ElementPrologueB = AscendC::int4b_t;
using ElementB = int8_t;
using ElementC = int32_t;
using ArchTag = Arch::AtlasA2;

constexpr bool kEnableUnitFlag = false;
using DispatchPolicy =
    Gemm::MmadAtlasA2PingPongWithPrologue<kEnableUnitFlag>;
using L1TileShape = GemmShape<128, 256, 512>;
using L0TileShape = GemmShape<128, 256, 128>;

using PrologueSrcType =
    Gemm::GemmType<ElementPrologueB, LayoutPrologueB>;
using PrologueDstType = Gemm::GemmType<ElementB, LayoutB>;
using AType = Gemm::GemmType<ElementA, LayoutA>;
using BType = PrologueDstType;
using CType = Gemm::GemmType<ElementC, LayoutC>;

using PrologueA = void;
constexpr uint32_t kComputeLen = 24 * 1024;
using PrologueB = Gemm::Tile::TileCastInt4ToInt8<
    ArchTag, PrologueSrcType, PrologueDstType, kComputeLen>;
using TileCopy = Gemm::Tile::TileCopyWithPrologue<
    ArchTag, AType, BType, CType, PrologueA, PrologueB>;
using BlockMmad = Gemm::Block::BlockMmad<
    DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType, void,
    TileCopy>;

}  // namespace

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ int32_t *lifecycle = tensor_data<int32_t>(args, 0);
    const uint32_t block_idx =
        static_cast<uint32_t>(get_block_idx(args));
#ifdef __DAV_C220_CUBE__
    constexpr int32_t kEngine = 1;
    constexpr int32_t kLane = -1;
#elif defined(__DAV_C220_VEC__)
    constexpr int32_t kEngine = 2;
    const int32_t kLane =
        static_cast<int32_t>(get_sub_block_id(args));
#endif
    __gm__ int32_t *row =
        initialize_row(lifecycle, block_idx, kEngine, kLane);
    {
        Arch::Resource<ArchTag> resource;
        row[4] = 1;
        flush_row(row);
        {
            BlockMmad::Params params{};
            BlockMmad block_mmad(resource, params);
            row[5] = 1;
            flush_row(row);
        }
        row[6] = 1;
        flush_row(row);
    }
    row[7] = 1;
    flush_row(row);
}

#endif
