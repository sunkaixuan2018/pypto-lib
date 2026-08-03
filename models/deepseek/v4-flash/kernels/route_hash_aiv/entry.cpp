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
extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    (void)args;
}

#else

#include "intrinsic.h"
#include "kernel_operator.h"

namespace {

constexpr uint32_t kTokens = 8;
constexpr uint32_t kExperts = 128;
constexpr uint32_t kScoreStride = 256;
constexpr uint32_t kTopK = 6;
constexpr uint32_t kAlignedTopK = 8;
constexpr float kRouteScale = 1.5F;

template <typename T>
static __aicore__ __attribute__((always_inline)) __gm__ T *
tensor_data(__gm__ int64_t *args, int32_t index) {
    __gm__ ::Tensor *tensor =
        reinterpret_cast<__gm__ ::Tensor *>(args[index]);
    return reinterpret_cast<__gm__ T *>(tensor->buffer.addr) +
           tensor->start_offset;
}

}  // namespace

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
#ifdef __DAV_C220_VEC__
    const uint32_t row = static_cast<uint32_t>(get_block_idx(args));
    int32_t active_tokens = static_cast<int32_t>(args[5]);
    if (active_tokens < 0) {
        active_tokens = 0;
    } else if (active_tokens > static_cast<int32_t>(kTokens)) {
        active_tokens = static_cast<int32_t>(kTokens);
    }
    if (row >= static_cast<uint32_t>(active_tokens)) {
        return;
    }

    AscendC::GlobalTensor<float> weights_gm;
    weights_gm.SetGlobalBuffer(tensor_data<float>(args, 0));
    AscendC::GlobalTensor<int32_t> indices_gm;
    indices_gm.SetGlobalBuffer(tensor_data<int32_t>(args, 1));
    AscendC::GlobalTensor<float> scores_gm;
    scores_gm.SetGlobalBuffer(tensor_data<float>(args, 2));
    AscendC::GlobalTensor<int64_t> input_ids_gm;
    input_ids_gm.SetGlobalBuffer(tensor_data<int64_t>(args, 3));
    AscendC::GlobalTensor<int32_t> tid2eid_gm;
    tid2eid_gm.SetGlobalBuffer(tensor_data<int32_t>(args, 4));

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> input_id_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> expert_id_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> score_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> weight_out_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> index_out_buf;
    pipe.InitBuffer(input_id_buf, 32);
    pipe.InitBuffer(expert_id_buf, kAlignedTopK * sizeof(int32_t));
    pipe.InitBuffer(score_buf, kExperts * sizeof(float));
    pipe.InitBuffer(weight_out_buf, kAlignedTopK * sizeof(float));
    pipe.InitBuffer(index_out_buf, kAlignedTopK * sizeof(int32_t));

    AscendC::LocalTensor<int64_t> input_id =
        input_id_buf.Get<int64_t>();
    AscendC::DataCopyExtParams input_id_copy{
        1, static_cast<uint32_t>(sizeof(int64_t)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<int64_t> input_id_pad{
        false, 0, 0, 0};
    AscendC::DataCopyPad(
        input_id, input_ids_gm[row], input_id_copy, input_id_pad);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    const int64_t token_id = input_id.GetValue(0);

    AscendC::LocalTensor<int32_t> expert_ids =
        expert_id_buf.Get<int32_t>();
    AscendC::DataCopyExtParams expert_id_copy{
        1, static_cast<uint32_t>(kTopK * sizeof(int32_t)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<int32_t> expert_id_pad{
        false, 0, 0, 0};
    AscendC::DataCopyPad(
        expert_ids,
        tid2eid_gm[static_cast<uint64_t>(token_id) * kTopK],
        expert_id_copy,
        expert_id_pad);

    AscendC::LocalTensor<float> scores = score_buf.Get<float>();
    AscendC::DataCopy(
        scores,
        scores_gm[static_cast<uint64_t>(row) * kScoreStride],
        kExperts);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID1);

    AscendC::LocalTensor<float> weight_out =
        weight_out_buf.Get<float>();
    AscendC::LocalTensor<int32_t> index_out =
        index_out_buf.Get<int32_t>();
    float score_sum = 0.0F;
    for (uint32_t k = 0; k < kTopK; ++k) {
        const int32_t expert = expert_ids.GetValue(k);
        const float score = scores.GetValue(static_cast<uint32_t>(expert));
        index_out.SetValue(k, expert);
        weight_out.SetValue(k, score);
        score_sum += score;
    }
    const float weight_scale = kRouteScale / score_sum;
    for (uint32_t k = 0; k < kTopK; ++k) {
        weight_out.SetValue(
            k, weight_out.GetValue(k) * weight_scale);
    }

    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID2);
    AscendC::DataCopyExtParams output_copy{
        1, static_cast<uint32_t>(kTopK * sizeof(float)), 0, 0, 0};
    AscendC::DataCopyPad(
        weights_gm[static_cast<uint64_t>(row) * kTopK],
        weight_out,
        output_copy);
    AscendC::DataCopyPad(
        indices_gm[static_cast<uint64_t>(row) * kTopK],
        index_out,
        output_copy);
#else
    (void)args;
#endif
}

#endif
