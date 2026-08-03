/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0.
 * Please refer to the License for details. You may not use this file except in
 * compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
 * KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
 * NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the
 * License.
 */

/**
 * Standalone official HcPre probe for the DeepSeek-V4 Flash decode shape.
 *
 * The nonzero analytical case exercises the RMS, projection, sigmoid,
 * Sinkhorn, and stream-mixing paths. The timed region contains only aclnnHcPre.
 */

#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include "aclnn_hc_pre.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

namespace {

constexpr int64_t kTokens = 8;
constexpr int64_t kHcMult = 4;
constexpr int64_t kHidden = 4096;
constexpr int64_t kHcDim = kHcMult * kHidden;
constexpr int64_t kMixHc = (2 + kHcMult) * kHcMult;
constexpr int64_t kSinkhornIters = 20;
constexpr double kHcEps = 1e-6;
constexpr double kNormEps = 1e-6;
constexpr int kWarmup = 2;
constexpr int kMeasured = 8;
constexpr float kTolerance = 5e-2F;

#define CHECK_ACL(expr)                                                                  \
    do {                                                                                 \
        const auto status = (expr);                                                      \
        if (status != ACL_SUCCESS) {                                                     \
            std::cerr << "ACL failure " << status << " at " << __FILE__ << ':'          \
                      << __LINE__ << ": " #expr << '\n';                                 \
            const char *recent = aclGetRecentErrMsg();                                   \
            if (recent != nullptr) {                                                     \
                std::cerr << recent << '\n';                                             \
            }                                                                            \
            std::exit(1);                                                                \
        }                                                                                \
    } while (false)

void *DeviceAlloc(size_t bytes)
{
    void *ptr = nullptr;
    CHECK_ACL(aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemset(ptr, bytes, 0, bytes));
    return ptr;
}

aclTensor *CreateTensor(
    void *data, const std::vector<int64_t> &dims, aclDataType dtype)
{
    std::vector<int64_t> strides(dims.size(), 1);
    for (int64_t i = static_cast<int64_t>(dims.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] =
            strides[static_cast<size_t>(i + 1)] * dims[static_cast<size_t>(i + 1)];
    }
    return aclCreateTensor(
        dims.data(), dims.size(), dtype, strides.data(), 0, ACL_FORMAT_ND,
        dims.data(), dims.size(), data);
}

float BFloat16ToFloat(uint16_t value)
{
    const uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

float Median(std::vector<float> values)
{
    std::sort(values.begin(), values.end());
    return (values[3] + values[4]) * 0.5F;
}

float MaxAbsError(const std::vector<float> &values, float expected)
{
    float result = 0.0F;
    for (float value : values) {
        if (!std::isfinite(value)) {
            return std::numeric_limits<float>::infinity();
        }
        result = std::max(result, std::abs(value - expected));
    }
    return result;
}

} // namespace

int main(int argc, char **argv)
{
    const int device = argc > 1 ? std::atoi(argv[1]) : 0;

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(device));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    const size_t xElements =
        static_cast<size_t>(kTokens * kHcMult * kHidden);
    const size_t hcFnElements = static_cast<size_t>(kMixHc * kHcDim);
    const size_t yElements = static_cast<size_t>(kTokens * kHidden);
    const size_t postElements = static_cast<size_t>(kTokens * kHcMult);
    const size_t combElements =
        static_cast<size_t>(kTokens * kHcMult * kHcMult);

    const size_t xBytes = xElements * sizeof(uint16_t);
    const size_t hcFnBytes = hcFnElements * sizeof(float);
    const size_t hcScaleBytes = 3 * sizeof(float);
    const size_t hcBaseBytes = static_cast<size_t>(kMixHc) * sizeof(float);
    const size_t yBytes = yElements * sizeof(uint16_t);
    const size_t postBytes = postElements * sizeof(float);
    const size_t combBytes = combElements * sizeof(float);

    std::vector<uint16_t> xHost(xElements, 0x3F80);
    std::vector<float> hcFnHost(
        hcFnElements, 1.0F / static_cast<float>(kHcDim));
    std::vector<float> hcScaleHost(3, 1.0F);
    std::vector<float> hcBaseHost(static_cast<size_t>(kMixHc), 0.0F);

    void *xDevice = DeviceAlloc(xBytes);
    void *hcFnDevice = DeviceAlloc(hcFnBytes);
    void *hcScaleDevice = DeviceAlloc(hcScaleBytes);
    void *hcBaseDevice = DeviceAlloc(hcBaseBytes);
    void *yDevice = DeviceAlloc(yBytes);
    void *postDevice = DeviceAlloc(postBytes);
    void *combDevice = DeviceAlloc(combBytes);

    CHECK_ACL(aclrtMemcpy(
        xDevice, xBytes, xHost.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(
        hcFnDevice, hcFnBytes, hcFnHost.data(), hcFnBytes,
        ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(
        hcScaleDevice, hcScaleBytes, hcScaleHost.data(), hcScaleBytes,
        ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(
        hcBaseDevice, hcBaseBytes, hcBaseHost.data(), hcBaseBytes,
        ACL_MEMCPY_HOST_TO_DEVICE));

    aclTensor *x = CreateTensor(
        xDevice, {kTokens, kHcMult, kHidden}, ACL_BF16);
    aclTensor *hcFn =
        CreateTensor(hcFnDevice, {kMixHc, kHcDim}, ACL_FLOAT);
    aclTensor *hcScale = CreateTensor(hcScaleDevice, {3}, ACL_FLOAT);
    aclTensor *hcBase = CreateTensor(hcBaseDevice, {kMixHc}, ACL_FLOAT);
    aclTensor *y =
        CreateTensor(yDevice, {kTokens, kHidden}, ACL_BF16);
    aclTensor *post =
        CreateTensor(postDevice, {kTokens, kHcMult}, ACL_FLOAT);
    aclTensor *comb = CreateTensor(
        combDevice, {kTokens, kHcMult, kHcMult}, ACL_FLOAT);
    if (x == nullptr || hcFn == nullptr || hcScale == nullptr ||
        hcBase == nullptr || y == nullptr || post == nullptr ||
        comb == nullptr) {
        std::cerr << "Failed to create one or more ACL tensors\n";
        return 1;
    }

    void *workspace = nullptr;
    uint64_t workspaceBytes = 0;
    std::vector<float> measuredUs;
    measuredUs.reserve(kMeasured);
    for (int run = 0; run < kWarmup + kMeasured; ++run) {
        uint64_t thisWorkspaceBytes = 0;
        aclOpExecutor *executor = nullptr;
        CHECK_ACL(aclnnHcPreGetWorkspaceSize(
            x, hcFn, hcScale, hcBase, kHcMult, kSinkhornIters, kHcEps,
            kNormEps, y, post, comb, &thisWorkspaceBytes, &executor));
        if (run == 0) {
            workspaceBytes = thisWorkspaceBytes;
            if (workspaceBytes > 0) {
                workspace = DeviceAlloc(workspaceBytes);
            }
        } else if (thisWorkspaceBytes != workspaceBytes) {
            std::cerr << "Workspace size changed between runs\n";
            return 1;
        }

        CHECK_ACL(aclrtMemset(yDevice, yBytes, 0xFF, yBytes));
        CHECK_ACL(aclrtMemset(postDevice, postBytes, 0xFF, postBytes));
        CHECK_ACL(aclrtMemset(combDevice, combBytes, 0xFF, combBytes));
        aclrtEvent start = nullptr;
        aclrtEvent end = nullptr;
        CHECK_ACL(aclrtCreateEvent(&start));
        CHECK_ACL(aclrtCreateEvent(&end));
        CHECK_ACL(aclrtRecordEvent(start, stream));
        CHECK_ACL(aclnnHcPre(workspace, workspaceBytes, executor, stream));
        CHECK_ACL(aclrtRecordEvent(end, stream));
        CHECK_ACL(aclrtSynchronizeEvent(end));
        float elapsedMs = 0.0F;
        CHECK_ACL(aclrtEventElapsedTime(&elapsedMs, start, end));
        CHECK_ACL(aclrtDestroyEvent(start));
        CHECK_ACL(aclrtDestroyEvent(end));

        const float elapsedUs = elapsedMs * 1000.0F;
        std::cout << (run < kWarmup ? "warmup" : "sample") << '['
                  << (run < kWarmup ? run : run - kWarmup) << "]_us="
                  << std::fixed << std::setprecision(1) << elapsedUs << '\n';
        if (run >= kWarmup) {
            measuredUs.push_back(elapsedUs);
        }
    }

    std::vector<uint16_t> yHost(yElements);
    std::vector<float> postHost(postElements);
    std::vector<float> combHost(combElements);
    CHECK_ACL(aclrtMemcpy(
        yHost.data(), yBytes, yDevice, yBytes, ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(
        postHost.data(), postBytes, postDevice, postBytes,
        ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(
        combHost.data(), combBytes, combDevice, combBytes,
        ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<float> yFloat(yElements);
    std::transform(
        yHost.begin(), yHost.end(), yFloat.begin(), BFloat16ToFloat);
    const float mix =
        1.0F / std::sqrt(1.0F + static_cast<float>(kNormEps));
    const float sigmoid = 1.0F / (1.0F + std::exp(-mix));
    const float expectedY =
        static_cast<float>(kHcMult) *
        (sigmoid + static_cast<float>(kHcEps));
    const float expectedPost = 2.0F * sigmoid;
    const float expectedComb = 1.0F / static_cast<float>(kHcMult);
    const float yMaxError = MaxAbsError(yFloat, expectedY);
    const float postMaxError = MaxAbsError(postHost, expectedPost);
    const float combMaxError = MaxAbsError(combHost, expectedComb);
    const bool correctness =
        yMaxError <= kTolerance && postMaxError <= kTolerance &&
        combMaxError <= kTolerance;
    const float mean =
        std::accumulate(measuredUs.begin(), measuredUs.end(), 0.0F) /
        measuredUs.size();

    std::cout << "shape=[" << kTokens << ',' << kHcMult << ',' << kHidden
              << "] hc_fn=[" << kMixHc << ',' << kHcDim << "]\n";
    std::cout << "workspace_bytes=" << workspaceBytes << '\n';
    std::cout << "correctness=" << (correctness ? "PASS" : "FAIL")
              << " y_max_abs_error=" << yMaxError
              << " post_max_abs_error=" << postMaxError
              << " comb_max_abs_error=" << combMaxError << '\n';
    std::cout << "expected_y=" << expectedY
              << " actual_y0=" << yFloat.front()
              << " expected_post=" << expectedPost
              << " actual_post0=" << postHost.front()
              << " expected_comb=" << expectedComb
              << " actual_comb0=" << combHost.front() << '\n';
    std::cout << "measured_median_us=" << Median(measuredUs)
              << " measured_mean_us=" << mean << '\n';

    if (workspace != nullptr) {
        CHECK_ACL(aclrtFree(workspace));
    }
    for (aclTensor *tensor : {comb, post, y, hcBase, hcScale, hcFn, x}) {
        CHECK_ACL(aclDestroyTensor(tensor));
    }
    for (void *ptr :
         {combDevice, postDevice, yDevice, hcBaseDevice, hcScaleDevice,
          hcFnDevice, xDevice}) {
        CHECK_ACL(aclrtFree(ptr));
    }
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(device));
    CHECK_ACL(aclFinalize());
    return correctness ? 0 : 1;
}
