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

// PyPTO's MIX dispatcher supplies the logical AIV lane in its task payload;
// CANN's native GetSubBlockIdx/Num report zero in this runtime.  Redirect only
// the Catlass include so TileCast splits its 512 source rows across both lanes.
namespace AscendC {

[[block_local]] static uint32_t pypto_tilecast_lane;

static __aicore__ __attribute__((always_inline)) uint32_t
PyptoTileCastLane() {
    return pypto_tilecast_lane;
}

static __aicore__ __attribute__((always_inline)) uint32_t
PyptoTileCastLaneCount() {
    return 2;
}

}  // namespace AscendC

#define GetSubBlockIdx PyptoTileCastLane
#define GetSubBlockNum PyptoTileCastLaneCount

#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/tile/cast_int4_to_int8.hpp"
#include "catlass/layout/layout.hpp"

#undef GetSubBlockNum
#undef GetSubBlockIdx

namespace {

constexpr uint32_t kK = 2048;
constexpr uint32_t kN = 4096;
constexpr uint32_t kBlockDim = 24;
constexpr uint32_t kActiveBlocks = 16;
constexpr uint32_t kKTile = 512;
constexpr uint32_t kNTile = 256;
constexpr uint32_t kTileCount = 4;
constexpr uint32_t kColumns = 16;
constexpr uint32_t kMarkerBytes = kBlockDim * 3 * kColumns * sizeof(int32_t);
constexpr uint32_t kComputeLen = 24 * 1024;

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
    row[12] = static_cast<int32_t>(
        reinterpret_cast<uint64_t>(output) & 63U);
    flush_row(row);
    return row;
}

using namespace Catlass;

using ArchTag = Arch::AtlasA2;
using LayoutSrc = layout::RowMajor;
using LayoutDst = layout::RowMajor;
using SrcType = Gemm::GemmType<AscendC::int4b_t, LayoutSrc>;
using DstType = Gemm::GemmType<int8_t, LayoutDst>;
using TileCast = Gemm::Tile::TileCastInt4ToInt8<
    ArchTag, SrcType, DstType, kComputeLen>;

}  // namespace

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    // Signature order: 0=output, 1=packed_weight_kn.
    __gm__ int8_t *output = tensor_data<int8_t>(args, 0);
    const uint32_t block_idx =
        static_cast<uint32_t>(get_block_idx(args));

#ifdef __DAV_C220_CUBE__
    __gm__ int32_t *row = initialize_row(output, block_idx, 1, -1);
    for (uint32_t column = 4; column <= 11; ++column) {
        row[column] = -1;
    }
    row[14] = 1;
    flush_row(row);
#elif defined(__DAV_C220_VEC__)
    const uint32_t lane =
        static_cast<uint32_t>(get_sub_block_id(args));
    AscendC::pypto_tilecast_lane = lane;

    __gm__ int32_t *row = initialize_row(
        output, block_idx, 2, static_cast<int32_t>(lane));
    const bool active = block_idx < kActiveBlocks;
    row[13] = active ? 1 : 0;
    if (!active) {
        for (uint32_t tile = 0; tile < kTileCount; ++tile) {
            row[5 + tile] = -1;
        }
    }
    flush_row(row);

    {
        Arch::Resource<ArchTag> resource;
        {
            typename TileCast::Params params{};
            TileCast tilecast(resource, params);
            row[4] = 1;
            flush_row(row);

            if (active) {
                AscendC::GlobalTensor<AscendC::int4b_t> packed_weight;
                packed_weight.SetGlobalBuffer(
                    reinterpret_cast<__gm__ AscendC::int4b_t *>(
                        tensor_data<int8_t>(args, 1)));
                const LayoutSrc full_layout{kK, kN, kN};
                const LayoutDst tile_layout{kKTile, kNTile};
                const MatrixCoord tile_shape{kKTile, kNTile};

                __gm__ int8_t *converted = output + kMarkerBytes;
                for (uint32_t tile = 0; tile < kTileCount; ++tile) {
                    const MatrixCoord src_offset{
                        tile * kKTile, block_idx * kNTile};
                    auto src = packed_weight[
                        full_layout.GetOffset(src_offset)];
                    auto src_layout =
                        full_layout.GetTileLayout(tile_shape);

                    const uint64_t dst_offset =
                        (static_cast<uint64_t>(block_idx) * kTileCount + tile) *
                        kKTile * kNTile;
                    AscendC::GlobalTensor<int8_t> dst;
                    dst.SetGlobalBuffer(converted + dst_offset);

                    tilecast(dst, tile_layout, src, src_layout);
                    row[5 + tile] = 1;
                    flush_row(row);
                }
            }
            row[9] = 1;
            flush_row(row);
        }
        row[10] = 1;
        flush_row(row);
    }
    row[11] = 1;
    row[14] = 1;
    flush_row(row);
#endif
}

#endif
