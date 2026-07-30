/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0.
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

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"

// This is the C220 device-side layout generated from
// GMMSwigluQuantV2TilingFusionData. The values below are pinned to the
// production-shaped M=256, K=N=4096, E=16 W8A8 probe.
#pragma pack(push, 8)
struct GMMSwigluQuantV2TilingFusionData {
    int64_t cubeBlockDim;
    int64_t vectorBlockDim;
    int64_t groupNum;
    int64_t K;
    int64_t N;
    int64_t M;
    int64_t ubFactorDimx;
    int64_t ubFactorDimy;
    int64_t actRight;
    int64_t groupListType;
    int8_t isSingleTensor;
    float swigluLimit;
    TCubeTiling matmulTiling;
};
#pragma pack(pop)

template <typename T>
static __aicore__ __attribute__((always_inline)) __gm__ T *
pypto_raw_tensor_addr(int32_t index, GM_ADDR address) {
    (void)index;
    return reinterpret_cast<__gm__ T *>(address);
}

// The source operator receives a tensor-list descriptor even for its
// single-tensor path. PyPTO extern passes a raw GM pointer. The single-tensor
// branch only requests element zero, after which it uses direct expert
// offsets, so replacing the lookup is sufficient for this fixed ABI.
#define GetTensorAddr pypto_raw_tensor_addr
#include "grouped_matmul_swiglu_quant_spilit_fusion.h"
#undef GetTensorAddr

namespace {

template <typename T>
static __aicore__ __attribute__((always_inline)) __gm__ T *
tensor_data(__gm__ int64_t *args, int32_t index) {
    __gm__ ::Tensor *tensor =
        reinterpret_cast<__gm__ ::Tensor *>(args[index]);
    return reinterpret_cast<__gm__ T *>(tensor->buffer.addr) +
           tensor->start_offset;
}

static __aicore__ __attribute__((always_inline)) void
init_tiling(GMMSwigluQuantV2TilingFusionData &tiling) {
    tiling.cubeBlockDim = 24;
    tiling.vectorBlockDim = 48;
    tiling.groupNum = 16;
    tiling.K = 4096;
    tiling.N = 4096;
    tiling.M = 256;
    tiling.ubFactorDimx = 4;
    tiling.ubFactorDimy = 2048;
    tiling.actRight = 0;
    tiling.groupListType = 0;
    tiling.isSingleTensor = 1;
    tiling.swigluLimit = 10.0F;

    TCubeTiling &mm = tiling.matmulTiling;
    mm.usedCoreNum = 24;
    mm.M = 256;
    mm.N = 4096;
    mm.Ka = 4096;
    mm.Kb = 4096;
    mm.singleCoreM = 256;
    mm.singleCoreN = 256;
    mm.singleCoreK = 4096;
    mm.baseM = 128;
    mm.baseN = 256;
    mm.baseK = 128;
    mm.depthA1 = 8;
    mm.depthB1 = 8;
    mm.stepM = 1;
    mm.stepN = 1;
    mm.isBias = 0;
    mm.transLength = 0;
    mm.iterateOrder = 0;
    mm.shareMode = 0;
    mm.shareL1Size = 98304;
    mm.shareL0CSize = 131072;
    mm.shareUbSize = 0;
    mm.batchM = 1;
    mm.batchN = 1;
    mm.singleBatchM = 1;
    mm.singleBatchN = 1;
    mm.stepKa = 4;
    mm.stepKb = 4;
    mm.depthAL1CacheUB = 0;
    mm.depthBL1CacheUB = 0;
    mm.dbL0A = 2;
    mm.dbL0B = 2;
    mm.dbL0C = 1;
    mm.ALayoutInfoB = 0;
    mm.ALayoutInfoS = 0;
    mm.ALayoutInfoN = 0;
    mm.ALayoutInfoG = 0;
    mm.ALayoutInfoD = 0;
    mm.BLayoutInfoB = 0;
    mm.BLayoutInfoS = 0;
    mm.BLayoutInfoN = 0;
    mm.BLayoutInfoG = 0;
    mm.BLayoutInfoD = 0;
    mm.CLayoutInfoB = 0;
    mm.CLayoutInfoS1 = 0;
    mm.CLayoutInfoN = 0;
    mm.CLayoutInfoG = 0;
    mm.CLayoutInfoS2 = 0;
    mm.BatchNum = 0;
    mm.mxTypePara = 0;
}

}  // namespace

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    KERNEL_TASK_TYPE(3, KERNEL_TYPE_MIX_AIC_1_2);

    // Tensor ABI: out, out_scale, workspace, x, x_scale, group_list,
    // weight_nz, weight_scale.
    GM_ADDR out = reinterpret_cast<GM_ADDR>(tensor_data<int8_t>(args, 0));
    GM_ADDR out_scale = reinterpret_cast<GM_ADDR>(tensor_data<float>(args, 1));
    GM_ADDR workspace = reinterpret_cast<GM_ADDR>(tensor_data<int32_t>(args, 2));
    GM_ADDR x = reinterpret_cast<GM_ADDR>(tensor_data<int8_t>(args, 3));
    GM_ADDR x_scale = reinterpret_cast<GM_ADDR>(tensor_data<float>(args, 4));
    GM_ADDR group_list = reinterpret_cast<GM_ADDR>(tensor_data<int64_t>(args, 5));
    GM_ADDR weight_nz = reinterpret_cast<GM_ADDR>(tensor_data<int8_t>(args, 6));
    GM_ADDR weight_scale = reinterpret_cast<GM_ADDR>(tensor_data<float>(args, 7));

    GMMSwigluQuantV2TilingFusionData tiling{};
    init_tiling(tiling);

    AscendC::TPipe pipe;
    GroupedMatmulDequantSwigluQuant::
        GroupedMatmulDequantSwigluQuantFusion op(
            &pipe, &tiling, &tiling.matmulTiling);
#ifdef __DAV_C220_CUBE__
    op.mm.SetSubBlockIdx(0);
    op.mm.Init(&tiling.matmulTiling, &pipe);
#endif
    op.Init(
        x, weight_nz, weight_scale, x_scale,
        nullptr, group_list, out, out_scale, workspace);
    op.Process();
}

#endif
