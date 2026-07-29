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

#if !defined(PYPTO_PROJECTION_K) || !defined(PYPTO_PROJECTION_BLOCKS)
#error "Projection shape macros must be defined by the split entry wrapper"
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

namespace {

constexpr uint32_t kM = 16;
constexpr uint32_t kK = PYPTO_PROJECTION_K;
constexpr uint32_t kBlocks = PYPTO_PROJECTION_BLOCKS;
constexpr uint32_t kLanes = 2 * kBlocks;
constexpr uint32_t kPackedRow = kK / 2;
constexpr uint32_t kPlaneBytes = kM * kPackedRow;
constexpr uint32_t kBytesPerLane = kPlaneBytes / kLanes;
constexpr uint32_t kChunkBytes = 256;
constexpr event_t kCopyEvent = EVENT_ID2;

static_assert(kK % 2 == 0);
static_assert(kPlaneBytes % kLanes == 0);
static_assert(kBytesPerLane % kChunkBytes == 0);

template <typename T>
static __aicore__ __attribute__((always_inline)) __gm__ T *
tensor_data(__gm__ int64_t *args, int32_t index) {
    __gm__ ::Tensor *tensor =
        reinterpret_cast<__gm__ ::Tensor *>(args[index]);
    return reinterpret_cast<__gm__ T *>(tensor->buffer.addr) +
           tensor->start_offset;
}

static __aicore__ __attribute__((always_inline)) uint8_t
signed_nibble(int8_t value) {
    return static_cast<uint8_t>(value) & 0x0FU;
}

}  // namespace

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
#ifdef __DAV_C220_VEC__
    // Signature order:
    //   0=packed_planes[2M,K/2], 1=activation[M,K].
    const uint32_t block_idx =
        static_cast<uint32_t>(get_block_idx(args));
    const uint32_t lane =
        static_cast<uint32_t>(get_sub_block_id(args));
    const uint32_t logical_lane = block_idx * 2 + lane;
    if (logical_lane >= kLanes) {
        return;
    }

    AscendC::GlobalTensor<int8_t> activation;
    activation.SetGlobalBuffer(tensor_data<int8_t>(args, 1));
    AscendC::GlobalTensor<int8_t> packed_planes;
    packed_planes.SetGlobalBuffer(tensor_data<int8_t>(args, 0));

    Catlass::Arch::Resource<Catlass::Arch::AtlasA2> resource;
    auto input_ub =
        resource.ubBuf.template GetBufferByByte<int8_t>(0);
    auto high_ub =
        resource.ubBuf.template GetBufferByByte<int8_t>(2 * kChunkBytes);
    auto low_ub =
        resource.ubBuf.template GetBufferByByte<int8_t>(3 * kChunkBytes);

    const AscendC::DataCopyExtParams input_copy(
        1, 2 * kChunkBytes, 0, 0, 0);
    const AscendC::DataCopyPadExtParams<int8_t> input_pad(
        false, 0, 0, 0);
    const AscendC::DataCopyExtParams output_copy(
        1, kChunkBytes, 0, 0, 0);

    const uint32_t lane_base = logical_lane * kBytesPerLane;
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(kCopyEvent);
    for (uint32_t offset = 0; offset < kBytesPerLane;
         offset += kChunkBytes) {
        const uint32_t packed_offset = lane_base + offset;
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(kCopyEvent);
        AscendC::DataCopyPad(
            input_ub, activation[2 * packed_offset],
            input_copy, input_pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(kCopyEvent);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(kCopyEvent);

        for (uint32_t i = 0; i < kChunkBytes; ++i) {
            const int16_t x0 =
                static_cast<int16_t>(input_ub.GetValue(2 * i));
            const int16_t x1 =
                static_cast<int16_t>(input_ub.GetValue(2 * i + 1));
            const uint16_t u0 = static_cast<uint16_t>(x0 + 128);
            const uint16_t u1 = static_cast<uint16_t>(x1 + 128);
            const int8_t hi0 =
                static_cast<int8_t>((u0 >> 4) - 8);
            const int8_t hi1 =
                static_cast<int8_t>((u1 >> 4) - 8);
            const int8_t lo0 =
                static_cast<int8_t>((u0 & 0x0F) - 8);
            const int8_t lo1 =
                static_cast<int8_t>((u1 & 0x0F) - 8);
            high_ub.SetValue(
                i,
                static_cast<int8_t>(
                    signed_nibble(hi0) |
                    (signed_nibble(hi1) << 4)));
            low_ub.SetValue(
                i,
                static_cast<int8_t>(
                    signed_nibble(lo0) |
                    (signed_nibble(lo1) << 4)));
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kCopyEvent);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kCopyEvent);
        AscendC::DataCopyPad(
            packed_planes[packed_offset], high_ub, output_copy);
        AscendC::DataCopyPad(
            packed_planes[kPlaneBytes + packed_offset],
            low_ub, output_copy);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(kCopyEvent);
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(kCopyEvent);
#else
    (void)args;
#endif
}

#endif
