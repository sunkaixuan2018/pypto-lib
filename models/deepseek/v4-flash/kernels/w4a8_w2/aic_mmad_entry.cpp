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

// Bridge Catlass's topology queries to PyPTO's logical MIX task identity.
namespace AscendC {

[[block_local]] static uint32_t pypto_catlass_block_idx;
[[block_local]] static uint32_t pypto_catlass_block_num;
[[block_local]] static uint32_t pypto_catlass_sub_block_idx;
[[block_local]] static uint32_t pypto_catlass_sub_block_num;

static __aicore__ __attribute__((always_inline)) uint32_t
PyptoGetBlockIdx() {
    return pypto_catlass_block_idx;
}

static __aicore__ __attribute__((always_inline)) uint32_t
PyptoGetBlockNum() {
    return pypto_catlass_block_num;
}

static __aicore__ __attribute__((always_inline)) uint32_t
PyptoGetSubBlockIdx() {
    return pypto_catlass_sub_block_idx;
}

static __aicore__ __attribute__((always_inline)) uint32_t
PyptoGetSubBlockNum() {
    return pypto_catlass_sub_block_num;
}

}  // namespace AscendC

#define GetBlockIdx PyptoGetBlockIdx
#define GetBlockNum PyptoGetBlockNum
#define GetSubBlockIdx PyptoGetSubBlockIdx
#define GetSubBlockNum PyptoGetSubBlockNum

#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/w4a8_matmul.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/layout/layout.hpp"

#undef GetSubBlockNum
#undef GetSubBlockIdx
#undef GetBlockNum
#undef GetBlockIdx

namespace {

constexpr uint32_t kM = 16;
constexpr uint32_t kK = 1024;
constexpr uint32_t kN = 4096;
constexpr uint32_t kBlockDim = 24;
constexpr uint32_t kActiveBlocks = 16;
constexpr uint32_t kColumns = 16;
constexpr uint32_t kMarkerBytes =
    kBlockDim * 3 * kColumns * sizeof(int32_t);
constexpr uint16_t kCopyFlags[2] = {0, 2};
constexpr uint16_t kPrologueFlags[2] = {1, 3};

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
initialize_row(__gm__ int8_t *output, uint32_t block_idx,
               int32_t engine, int32_t lane) {
    const uint32_t row_idx = engine == 1
        ? block_idx
        : kBlockDim + block_idx * 2 + static_cast<uint32_t>(lane);
    __gm__ int32_t *marker = reinterpret_cast<__gm__ int32_t *>(output);
    __gm__ int32_t *row = marker + row_idx * kColumns;
    for (uint32_t column = 0; column < kColumns; ++column) {
        row[column] = 0;
    }
    row[0] = engine;
    row[1] = static_cast<int32_t>(block_idx);
    row[2] = lane;
    row[3] = 1;
    flush_row(row);
    return row;
}

static __aicore__ __attribute__((always_inline)) void
set_catlass_runtime_topology(__gm__ int64_t *args) {
    const uint32_t logical_block =
        static_cast<uint32_t>(get_block_idx(args));
    AscendC::pypto_catlass_block_num =
        static_cast<uint32_t>(get_block_num(args));
#ifdef __DAV_C220_CUBE__
    AscendC::pypto_catlass_block_idx = logical_block;
    AscendC::pypto_catlass_sub_block_idx = 0;
    AscendC::pypto_catlass_sub_block_num = 1;
#elif defined(__DAV_C220_VEC__)
    const uint32_t logical_lane =
        static_cast<uint32_t>(get_sub_block_id(args));
    AscendC::pypto_catlass_block_idx =
        logical_block * 2 + logical_lane;
    AscendC::pypto_catlass_sub_block_idx = logical_lane;
    AscendC::pypto_catlass_sub_block_num = 2;
#endif
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
using BlockScheduler = Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
using MatmulKernel =
    Gemm::Kernel::W4A8Matmul<BlockMmad, void, BlockScheduler>;

}  // namespace

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    // Signature order: 0=output, 1=activation, 2=prefilled workspace.
    __gm__ int8_t *output = tensor_data<int8_t>(args, 0);
    const uint32_t block_idx =
        static_cast<uint32_t>(get_block_idx(args));
    const bool active = block_idx < kActiveBlocks;
    set_catlass_runtime_topology(args);

#ifdef __DAV_C220_CUBE__
    __gm__ int32_t *row = initialize_row(output, block_idx, 1, -1);
    row[4] = 1;
    row[6] = active ? 1 : 0;
    row[8] = static_cast<int32_t>(
        reinterpret_cast<uint64_t>(output) & 63U);
    flush_row(row);
    {
        typename MatmulKernel::Params params{
            GemmCoord{kM, kN, kK},
            reinterpret_cast<GM_ADDR>(tensor_data<int8_t>(args, 1)),
            LayoutA{kM, kK},
            reinterpret_cast<GM_ADDR>(tensor_data<int8_t>(args, 2)),
            LayoutPrologueB{kK, kN, kN},
            reinterpret_cast<GM_ADDR>(output + kMarkerBytes),
            LayoutC{kM, kN},
            {},
            reinterpret_cast<GM_ADDR>(tensor_data<int8_t>(args, 2)),
        };
        MatmulKernel kernel;
        kernel.template operator()<AscendC::AIC>(params);
        row[5] = 1;
        flush_row(row);
    }
    row[7] = 1;
    row[9] = 1;
    row[10] = 1;
    flush_row(row);
#elif defined(__DAV_C220_VEC__)
    const uint32_t lane =
        static_cast<uint32_t>(get_sub_block_id(args));
    __gm__ int32_t *row = initialize_row(
        output, block_idx, 2, static_cast<int32_t>(lane));
    row[9] = active ? 1 : 0;
    if (!active) {
        row[5] = -1;
        row[6] = -1;
    }
    flush_row(row);

    {
        Arch::Resource<ArchTag> resource;
        {
            BlockMmad::Params params{};
            BlockMmad block_mmad(resource, params);
            row[4] = 1;
            flush_row(row);

            if (active) {
                for (uint32_t stage = 0; stage < 2; ++stage) {
                    AscendC::CrossCoreWaitFlag(kCopyFlags[stage]);
                    AscendC::CrossCoreSetFlag<0x02, PIPE_MTE3>(
                        kPrologueFlags[stage]);
                    row[5 + stage] = 1;
                    flush_row(row);
                }
            }
            row[7] = 1;
            flush_row(row);
        }
        row[8] = 1;
        flush_row(row);
    }
    row[10] = 1;
    flush_row(row);
#endif
}

#endif
