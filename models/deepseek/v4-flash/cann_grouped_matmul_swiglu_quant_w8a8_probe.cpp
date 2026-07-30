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
 * Standalone W8A8 fused W13 + clipped SwiGLU + quant probe.
 *
 * The probe keeps the Flash MoE shape of 16 local experts, K=4096, and
 * concatenated gate/up N=4096. It compares fixed M=768 with M=256. Constant
 * nonzero inputs provide an exact closed-form output check while preserving
 * the production-sized INT8 FRACTAL_NZ weight traffic.
 */

#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include "aclnn_grouped_matmul_swiglu_quant_weight_nz_v2.h"

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

constexpr int64_t kExperts = 16;
constexpr int64_t kK = 4096;
constexpr int64_t kN = 4096;
constexpr int64_t kOutputN = kN / 2;
constexpr double kSwiGluLimit = 10.0;
constexpr int kWarmup = 2;
constexpr int kMeasured = 8;

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

void *DeviceAlloc(size_t bytes, uint8_t value = 0)
{
    void *ptr = nullptr;
    CHECK_ACL(aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemset(ptr, bytes, value, bytes));
    return ptr;
}

aclTensor *CreateTensor(
    void *data, const std::vector<int64_t> &dims, aclDataType dtype, aclFormat format,
    const std::vector<int64_t> &storageDims = {})
{
    std::vector<int64_t> strides(dims.size(), 1);
    for (int64_t i = static_cast<int64_t>(dims.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] =
            strides[static_cast<size_t>(i + 1)] * dims[static_cast<size_t>(i + 1)];
    }
    const std::vector<int64_t> &physicalDims = storageDims.empty() ? dims : storageDims;
    return aclCreateTensor(
        dims.data(), dims.size(), dtype, strides.data(), 0, format, physicalDims.data(),
        physicalDims.size(), data);
}

float Median(std::vector<float> values)
{
    std::sort(values.begin(), values.end());
    return (values[3] + values[4]) * 0.5F;
}

} // namespace

int main(int argc, char **argv)
{
    const int device = argc > 1 ? std::atoi(argv[1]) : 0;
    const int64_t m = argc > 2 ? std::atoll(argv[2]) : 768;
    if ((m != 768 && m != 256) || m % kExperts != 0) {
        std::cerr << "M must be 768 or 256 and divisible by 16\n";
        return 2;
    }
    const int64_t rowsPerExpert = m / kExperts;

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(device));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    const size_t xBytes = static_cast<size_t>(m * kK);
    const size_t weightBytes = static_cast<size_t>(kExperts * kK * kN);
    const size_t weightScaleBytes =
        static_cast<size_t>(kExperts * kN) * sizeof(float);
    const size_t xScaleBytes = static_cast<size_t>(m) * sizeof(float);
    const size_t groupListBytes = static_cast<size_t>(kExperts) * sizeof(int64_t);
    const size_t outputBytes = static_cast<size_t>(m * kOutputN);
    const size_t outputScaleBytes = static_cast<size_t>(m) * sizeof(float);

    void *xDevice = DeviceAlloc(xBytes, 0x01);
    // Constant +1 values are invariant under ND-to-NZ reordering.
    void *weightDevice = DeviceAlloc(weightBytes, 0x01);
    void *weightScaleDevice = DeviceAlloc(weightScaleBytes);
    void *xScaleDevice = DeviceAlloc(xScaleBytes);
    void *groupListDevice = DeviceAlloc(groupListBytes);
    void *outputDevice = DeviceAlloc(outputBytes, 0x5A);
    void *outputScaleDevice = DeviceAlloc(outputScaleBytes);

    std::vector<float> weightScaleHost(
        static_cast<size_t>(kExperts * kN), 1.0F);
    std::vector<float> xScaleHost(static_cast<size_t>(m), 1.0F);
    std::vector<int64_t> groupListHost(static_cast<size_t>(kExperts));
    for (int64_t expert = 0; expert < kExperts; ++expert) {
        groupListHost[static_cast<size_t>(expert)] =
            (expert + 1) * rowsPerExpert;
    }
    CHECK_ACL(aclrtMemcpy(
        weightScaleDevice, weightScaleBytes, weightScaleHost.data(), weightScaleBytes,
        ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(
        xScaleDevice, xScaleBytes, xScaleHost.data(), xScaleBytes,
        ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(
        groupListDevice, groupListBytes, groupListHost.data(), groupListBytes,
        ACL_MEMCPY_HOST_TO_DEVICE));

    aclTensor *x = CreateTensor(xDevice, {m, kK}, ACL_INT8, ACL_FORMAT_ND);
    aclTensor *weight = CreateTensor(
        weightDevice, {kExperts, kK, kN}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ,
        {kExperts, kN / 32, kK / 16, 16, 32});
    aclTensor *weightScale = CreateTensor(
        weightScaleDevice, {kExperts, kN}, ACL_FLOAT, ACL_FORMAT_ND);
    aclTensor *xScale =
        CreateTensor(xScaleDevice, {m}, ACL_FLOAT, ACL_FORMAT_ND);
    aclTensor *groupList =
        CreateTensor(groupListDevice, {kExperts}, ACL_INT64, ACL_FORMAT_ND);
    aclTensor *output =
        CreateTensor(outputDevice, {m, kOutputN}, ACL_INT8, ACL_FORMAT_ND);
    aclTensor *outputScale =
        CreateTensor(outputScaleDevice, {m}, ACL_FLOAT, ACL_FORMAT_ND);

    const aclTensor *weightItems[] = {weight};
    const aclTensor *weightScaleItems[] = {weightScale};
    aclTensorList *weights = aclCreateTensorList(weightItems, 1);
    aclTensorList *weightScales = aclCreateTensorList(weightScaleItems, 1);
    const int64_t tuningValues[] = {rowsPerExpert};
    aclIntArray *tuning = aclCreateIntArray(tuningValues, 1);
    if (x == nullptr || weight == nullptr || weightScale == nullptr ||
        xScale == nullptr || groupList == nullptr || output == nullptr ||
        outputScale == nullptr || weights == nullptr || weightScales == nullptr ||
        tuning == nullptr) {
        std::cerr << "Failed to create one or more ACL metadata objects\n";
        return 1;
    }

    void *workspace = nullptr;
    uint64_t workspaceBytes = 0;
    std::vector<float> measuredUs;
    measuredUs.reserve(kMeasured);
    const std::vector<float> scaleSentinel(
        static_cast<size_t>(m), std::numeric_limits<float>::quiet_NaN());
    for (int run = 0; run < kWarmup + kMeasured; ++run) {
        uint64_t thisWorkspaceBytes = 0;
        aclOpExecutor *executor = nullptr;
        CHECK_ACL(aclnnGroupedMatmulSwigluQuantWeightNzV2GetWorkspaceSize(
            x, weights, weightScales,
            /*weightAssistMatrix=*/nullptr,
            /*bias=*/nullptr, xScale,
            /*smoothScale=*/nullptr, groupList,
            /*dequantMode=per-channel=*/0,
            /*dequantDtype=FP32=*/0,
            /*quantMode=symmetric INT8=*/0,
            /*groupListType=cumulative=*/0, tuning, kSwiGluLimit, output,
            outputScale, &thisWorkspaceBytes, &executor));
        if (run == 0) {
            workspaceBytes = thisWorkspaceBytes;
            if (workspaceBytes > 0) {
                workspace = DeviceAlloc(workspaceBytes);
            }
        } else if (thisWorkspaceBytes != workspaceBytes) {
            std::cerr << "Workspace size changed between runs\n";
            return 1;
        }

        CHECK_ACL(aclrtMemset(outputDevice, outputBytes, 0x5A, outputBytes));
        CHECK_ACL(aclrtMemcpy(
            outputScaleDevice, outputScaleBytes, scaleSentinel.data(),
            outputScaleBytes, ACL_MEMCPY_HOST_TO_DEVICE));
        aclrtEvent start = nullptr;
        aclrtEvent end = nullptr;
        CHECK_ACL(aclrtCreateEvent(&start));
        CHECK_ACL(aclrtCreateEvent(&end));
        CHECK_ACL(aclrtRecordEvent(start, stream));
        CHECK_ACL(aclnnGroupedMatmulSwigluQuantWeightNzV2(
            workspace, workspaceBytes, executor, stream));
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

    std::vector<int8_t> outputHost(outputBytes);
    std::vector<float> outputScaleHost(static_cast<size_t>(m));
    CHECK_ACL(aclrtMemcpy(
        outputHost.data(), outputBytes, outputDevice, outputBytes,
        ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(
        outputScaleHost.data(), outputScaleBytes, outputScaleDevice, outputScaleBytes,
        ACL_MEMCPY_DEVICE_TO_HOST));

    const float projection = static_cast<float>(kK);
    const float gate = std::min(projection, static_cast<float>(kSwiGluLimit));
    const float up = std::max(
        -static_cast<float>(kSwiGluLimit),
        std::min(projection, static_cast<float>(kSwiGluLimit)));
    const float expectedActivation = gate / (1.0F + std::exp(-gate)) * up;
    const float expectedScale = expectedActivation / 127.0F;
    const size_t badQuant = static_cast<size_t>(
        std::count_if(outputHost.begin(), outputHost.end(), [](int8_t value) {
            return value != 127;
        }));
    const size_t badScale = static_cast<size_t>(std::count_if(
        outputScaleHost.begin(), outputScaleHost.end(), [expectedScale](float value) {
            return !std::isfinite(value) ||
                   std::abs(value - expectedScale) > expectedScale * 0.002F;
        }));
    const float mean =
        std::accumulate(measuredUs.begin(), measuredUs.end(), 0.0F) /
        measuredUs.size();

    std::cout << "M=" << m << " K=" << kK << " N=" << kN
              << " experts=" << kExperts
              << " rows_per_expert=" << rowsPerExpert << '\n';
    std::cout << "weight_bytes=" << weightBytes
              << " workspace_bytes=" << workspaceBytes << '\n';
    std::cout << "correctness="
              << (badQuant == 0 && badScale == 0 ? "PASS" : "FAIL")
              << " bad_quant_count=" << badQuant
              << " bad_scale_count=" << badScale
              << " expected_scale=" << expectedScale
              << " actual_scale0=" << outputScaleHost.front() << '\n';
    std::cout << "measured_median_us=" << Median(measuredUs)
              << " measured_mean_us=" << mean << '\n';

    if (workspace != nullptr) {
        CHECK_ACL(aclrtFree(workspace));
    }
    CHECK_ACL(aclDestroyIntArray(tuning));
    CHECK_ACL(aclDestroyTensorList(weightScales));
    CHECK_ACL(aclDestroyTensorList(weights));
    for (aclTensor *tensor :
         {outputScale, output, groupList, xScale, weightScale, weight, x}) {
        CHECK_ACL(aclDestroyTensor(tensor));
    }
    for (void *ptr :
         {outputScaleDevice, outputDevice, groupListDevice, xScaleDevice,
          weightScaleDevice, weightDevice, xDevice}) {
        CHECK_ACL(aclrtFree(ptr));
    }
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(device));
    CHECK_ACL(aclFinalize());
    return badQuant == 0 && badScale == 0 ? 0 : 1;
}
