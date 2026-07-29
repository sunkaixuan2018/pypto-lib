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

#ifndef CATLASS_ARCH
#define CATLASS_ARCH 2201
#endif

#include "catlass/arch/arch.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/w4a8_matmul.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/layout/layout.hpp"

namespace {

constexpr uint32_t kM = 16;
constexpr uint32_t kK = 2048;
constexpr uint32_t kN = 4096;

template <typename T>
static __aicore__ __attribute__((always_inline)) GM_ADDR
tensor_data(__gm__ int64_t *args, int32_t index) {
    __gm__ ::Tensor *tensor = reinterpret_cast<__gm__ ::Tensor *>(args[index]);
    __gm__ T *data =
        reinterpret_cast<__gm__ T *>(tensor->buffer.addr) + tensor->start_offset;
    return reinterpret_cast<GM_ADDR>(data);
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
using DispatchPolicy = Gemm::MmadAtlasA2PingPongWithPrologue<kEnableUnitFlag>;
using L1TileShape = GemmShape<128, 256, 512>;
using L0TileShape = GemmShape<128, 256, 128>;

using PrologueSrcType = Gemm::GemmType<ElementPrologueB, LayoutPrologueB>;
using PrologueDstType = Gemm::GemmType<ElementB, LayoutB>;
using AType = Gemm::GemmType<ElementA, LayoutA>;
using BType = PrologueDstType;
using CType = Gemm::GemmType<ElementC, LayoutC>;

using PrologueA = void;
constexpr uint32_t kComputeLen = 24 * 1024;
using PrologueB =
    Gemm::Tile::TileCastInt4ToInt8<ArchTag, PrologueSrcType, PrologueDstType, kComputeLen>;
using TileCopy =
    Gemm::Tile::TileCopyWithPrologue<ArchTag, AType, BType, CType, PrologueA, PrologueB>;
using BlockMmad = Gemm::Block::BlockMmad<
    DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType, void, TileCopy>;
using BlockScheduler = Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
using MatmulKernel = Gemm::Kernel::W4A8Matmul<BlockMmad, void, BlockScheduler>;

}  // namespace

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    // PyPTO packs tensors in signature order:
    //   0=out, 1=activation, 2=packed_weight_kn, 3=workspace.
    typename MatmulKernel::Params params{
        GemmCoord{kM, kN, kK},
        tensor_data<int8_t>(args, 1), LayoutA{kM, kK},
        tensor_data<int8_t>(args, 2), LayoutPrologueB{kK, kN, kN},
        tensor_data<int32_t>(args, 0), LayoutC{kM, kN},
        {},
        tensor_data<int8_t>(args, 3),
    };

    MatmulKernel kernel;
#ifdef __DAV_C220_CUBE__
    kernel.template operator()<AscendC::AIC>(params);
#elif defined(__DAV_C220_VEC__)
    kernel.template operator()<AscendC::AIV>(params);
#endif
}

#endif
