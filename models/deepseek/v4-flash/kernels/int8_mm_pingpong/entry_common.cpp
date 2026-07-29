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

#if !defined(PYPTO_PROJECTION_K) || !defined(PYPTO_PROJECTION_N) || \
    !defined(PYPTO_PROJECTION_BLOCKS)
#error "Projection shape macros must be defined by the entry wrapper"
#endif

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

#ifndef CATLASS_ARCH
#define CATLASS_ARCH 2201
#endif

#include "intrinsic.h"
#include "kernel_operator.h"

#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"

namespace {

constexpr uint32_t kM = 16;
constexpr uint32_t kK = PYPTO_PROJECTION_K;
constexpr uint32_t kN = PYPTO_PROJECTION_N;
constexpr uint32_t kBlocks = PYPTO_PROJECTION_BLOCKS;
constexpr uint32_t kInner = 4;
constexpr uint32_t kNTile = 256;
constexpr uint32_t kKTile = 512;

static_assert(kN == kBlocks * kInner * kNTile);
static_assert(kK % kKTile == 0);

template <typename T>
static __aicore__ __attribute__((always_inline)) __gm__ T *
tensor_data(__gm__ int64_t *args, int32_t index) {
    __gm__ ::Tensor *tensor =
        reinterpret_cast<__gm__ ::Tensor *>(args[index]);
    return reinterpret_cast<__gm__ T *>(tensor->buffer.addr) +
           tensor->start_offset;
}

using namespace Catlass;

using ArchTag = Arch::AtlasA2;
using DispatchPolicy = Gemm::MmadAtlasA2Pingpong<false>;
using L1TileShape = GemmShape<128, kNTile, kKTile>;
using L0TileShape = GemmShape<128, kNTile, 128>;
using AType = Gemm::GemmType<int8_t, layout::RowMajor>;
// Physical input is [N, K] row-major, which is the same byte layout as
// logical [K, N] column-major. This matches expert_routed.py exactly.
using BType = Gemm::GemmType<int8_t, layout::ColumnMajor>;
using CType = Gemm::GemmType<int32_t, layout::RowMajor>;
using BlockMmad = Gemm::Block::BlockMmad<
    DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;

}  // namespace

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
#ifdef __DAV_C220_CUBE__
    // Signature order: 0=out[M,N], 1=activation[M,K], 2=weight[N,K].
    const uint32_t block_idx =
        static_cast<uint32_t>(get_block_idx(args));
    if (block_idx >= kBlocks) {
        return;
    }

    AscendC::GlobalTensor<int32_t> output;
    output.SetGlobalBuffer(tensor_data<int32_t>(args, 0));
    AscendC::GlobalTensor<int8_t> activation;
    activation.SetGlobalBuffer(tensor_data<int8_t>(args, 1));
    AscendC::GlobalTensor<int8_t> weight;
    weight.SetGlobalBuffer(tensor_data<int8_t>(args, 2));

    const layout::RowMajor layout_a{kM, kK};
    const layout::ColumnMajor layout_b{kK, kNTile};
    const layout::RowMajor layout_c{kM, kN, kN};
    const GemmCoord task_shape{kM, kNTile, kK};
    const auto task_c_layout =
        layout_c.GetTileLayout(MatrixCoord{kM, kNTile});

    Arch::Resource<ArchTag> resource;
    BlockMmad block_mmad(resource);
    for (uint32_t inner = 0; inner < kInner; ++inner) {
        const uint32_t n0 =
            (block_idx * kInner + inner) * kNTile;
        auto task_b = weight[static_cast<uint64_t>(n0) * kK];
        auto task_c = output[n0];
        block_mmad(
            activation, layout_a,
            task_b, layout_b,
            task_c, task_c_layout,
            task_shape);
    }
#else
    (void)args;
#endif
}

#endif
