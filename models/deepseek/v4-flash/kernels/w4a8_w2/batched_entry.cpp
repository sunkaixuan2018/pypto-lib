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
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/layout/layout.hpp"

#undef GetSubBlockNum
#undef GetSubBlockIdx
#undef GetBlockNum
#undef GetBlockIdx

namespace {

constexpr uint32_t kMaxExperts = 16;
constexpr uint32_t kM = 16;
constexpr uint32_t kK = 2048;
constexpr uint32_t kN = 4096;
constexpr uint32_t kNTile = 256;
constexpr uint32_t kNTasks = kN / kNTile;
constexpr uint32_t kStages = 2;
constexpr uint32_t kKTile = 512;

template <typename T>
static __aicore__ __attribute__((always_inline)) __gm__ T *
tensor_data(__gm__ int64_t *args, int32_t index) {
    __gm__ ::Tensor *tensor =
        reinterpret_cast<__gm__ ::Tensor *>(args[index]);
    return reinterpret_cast<__gm__ T *>(tensor->buffer.addr) +
           tensor->start_offset;
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
using L1TileShape = GemmShape<128, kNTile, kKTile>;
using L0TileShape = GemmShape<128, kNTile, 128>;
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
    // Signature order:
    //   0=out[E,M,N], 1=activation[E,M,K],
    //   2=packed_weight[E,K,N/2], 3=workspace[block,stage,Ktile,Ntile],
    //   4=active_experts[1].
    const uint32_t block_idx =
        static_cast<uint32_t>(get_block_idx(args));
    const uint32_t block_num =
        static_cast<uint32_t>(get_block_num(args));
    const uint32_t requested_experts =
        static_cast<uint32_t>(*tensor_data<int32_t>(args, 4));
    const uint32_t active_experts =
        requested_experts < kMaxExperts ? requested_experts : kMaxExperts;
    const uint32_t total_tasks = active_experts * kNTasks;
    set_catlass_runtime_topology(args);

    Arch::Resource<ArchTag> resource;
    BlockMmad::Params params{};
    BlockMmad block_mmad(resource, params);

    const LayoutA layout_a{kM, kK};
    const LayoutPrologueB layout_packed{kK, kN, kN};
    const LayoutB layout_workspace{kKTile, kNTile};
    const LayoutC layout_c{kM, kN, kN};
    const GemmCoord task_shape{kM, kNTile, kK};

    const uint64_t workspace_offset =
        static_cast<uint64_t>(block_idx) * kStages * kKTile * kNTile;

#ifdef __DAV_C220_CUBE__
    AscendC::GlobalTensor<ElementA> activation;
    activation.SetGlobalBuffer(tensor_data<ElementA>(args, 1));
    AscendC::GlobalTensor<ElementB> workspace;
    workspace.SetGlobalBuffer(
        tensor_data<ElementB>(args, 3) + workspace_offset);
    AscendC::GlobalTensor<ElementC> out;
    out.SetGlobalBuffer(tensor_data<ElementC>(args, 0));

    for (uint32_t task = block_idx; task < total_tasks;
         task += block_num) {
        const uint32_t expert = task / kNTasks;
        const uint32_t n_task = task - expert * kNTasks;
        const uint32_t n0 = n_task * kNTile;

        auto task_a = activation[
            static_cast<uint64_t>(expert) * kM * kK];
        auto task_c = out[
            static_cast<uint64_t>(expert) * kM * kN + n0];
        auto task_c_layout =
            layout_c.GetTileLayout(MatrixCoord{kM, kNTile});
        block_mmad(
            task_a, layout_a,
            workspace, layout_workspace,
            task_c, task_c_layout,
            task_shape);
    }
#elif defined(__DAV_C220_VEC__)
    AscendC::GlobalTensor<ElementPrologueB> packed_weight;
    packed_weight.SetGlobalBuffer(
        reinterpret_cast<__gm__ ElementPrologueB *>(
            tensor_data<int8_t>(args, 2)));
    AscendC::GlobalTensor<ElementB> workspace;
    workspace.SetGlobalBuffer(
        tensor_data<ElementB>(args, 3) + workspace_offset);

    for (uint32_t task = block_idx; task < total_tasks;
         task += block_num) {
        const uint32_t expert = task / kNTasks;
        const uint32_t n_task = task - expert * kNTasks;
        const uint32_t n0 = n_task * kNTile;
        const uint64_t packed_offset =
            static_cast<uint64_t>(expert) * kK * kN +
            layout_packed.GetOffset(MatrixCoord{0, n0});
        auto task_packed = packed_weight[packed_offset];
        auto task_packed_layout =
            layout_packed.GetTileLayout(MatrixCoord{kK, kNTile});
        block_mmad.Prologue(
            task_packed, task_packed_layout,
            workspace, layout_workspace,
            task_shape);
    }
#endif
}

#endif
