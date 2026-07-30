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
 * Standalone W8A8 FRACTAL_NZ GroupedMatmul probe for routed MoE projections.
 *
 * The probe compares AscendC's fixed-capacity M=768 shape with a smaller
 * M=256 shape. Each shape has 16 equally sized local expert groups. Weights
 * are converted from ND to FRACTAL_NZ once, outside the timed region.
 */

#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_grouped_matmul_weight_nz.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {

constexpr int64_t kExperts = 16;
constexpr int64_t kW13K = 4096;
constexpr int64_t kW13N = 4096;
constexpr int64_t kW2K = 2048;
constexpr int64_t kW2N = 4096;
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

struct ProjectionShape {
    std::string name;
    int64_t m;
    int64_t k;
    int64_t n;
};

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

ProjectionShape ParseShape(int argc, char **argv)
{
    const std::string stage = argc > 2 ? argv[2] : "w13";
    const int64_t m = argc > 3 ? std::atoll(argv[3]) : 768;
    if (stage != "w13" && stage != "w2") {
        std::cerr << "stage must be w13 or w2\n";
        std::exit(2);
    }
    if (m != 768 && m != 256) {
        std::cerr << "M must be 768 or 256\n";
        std::exit(2);
    }
    if (m % kExperts != 0) {
        std::cerr << "M must be divisible by the expert count\n";
        std::exit(2);
    }
    return {
        stage,
        m,
        stage == "w13" ? kW13K : kW2K,
        stage == "w13" ? kW13N : kW2N,
    };
}

} // namespace

int main(int argc, char **argv)
{
    const int device = argc > 1 ? std::atoi(argv[1]) : 0;
    const ProjectionShape shape = ParseShape(argc, argv);
    const int64_t rowsPerExpert = shape.m / kExperts;

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(device));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    const size_t xBytes = static_cast<size_t>(shape.m * shape.k);
    const size_t weightBytes =
        static_cast<size_t>(kExperts * shape.k * shape.n);
    const size_t outputBytes =
        static_cast<size_t>(shape.m * shape.n) * sizeof(int32_t);
    const size_t groupListBytes = static_cast<size_t>(kExperts) * sizeof(int64_t);

    void *xDevice = DeviceAlloc(xBytes, 0x01);
    // A constant tensor has identical values before and after ND-to-NZ
    // reordering, so initialize the physical NZ allocation directly.
    void *weightNzDevice = DeviceAlloc(weightBytes, 0x01);
    void *outputDevice = DeviceAlloc(outputBytes, 0x5A);
    void *groupListDevice = DeviceAlloc(groupListBytes);

    std::vector<int64_t> groupListHost(static_cast<size_t>(kExperts));
    for (int64_t expert = 0; expert < kExperts; ++expert) {
        groupListHost[static_cast<size_t>(expert)] = (expert + 1) * rowsPerExpert;
    }
    CHECK_ACL(aclrtMemcpy(
        groupListDevice, groupListBytes, groupListHost.data(), groupListBytes,
        ACL_MEMCPY_HOST_TO_DEVICE));

    aclTensor *x =
        CreateTensor(xDevice, {shape.m, shape.k}, ACL_INT8, ACL_FORMAT_ND);
    aclTensor *weightNz = CreateTensor(
        weightNzDevice, {kExperts, shape.k, shape.n}, ACL_INT8,
        ACL_FORMAT_FRACTAL_NZ,
        {kExperts, shape.n / 16, shape.k / 32, 16, 32});
    aclTensor *output = CreateTensor(
        outputDevice, {shape.m, shape.n}, ACL_INT32, ACL_FORMAT_ND);
    aclTensor *groupList = CreateTensor(
        groupListDevice, {kExperts}, ACL_INT64, ACL_FORMAT_ND);
    if (x == nullptr || weightNz == nullptr || output == nullptr ||
        groupList == nullptr) {
        std::cerr << "Failed to create input ACL tensors\n";
        return 1;
    }

    const aclTensor *xItems[] = {x};
    const aclTensor *weightItems[] = {weightNz};
    const aclTensor *outputItems[] = {output};
    aclTensorList *xList = aclCreateTensorList(xItems, 1);
    aclTensorList *weightList = aclCreateTensorList(weightItems, 1);
    aclTensorList *outputList = aclCreateTensorList(outputItems, 1);
    const int64_t tuningValues[] = {rowsPerExpert};
    aclIntArray *tuning = aclCreateIntArray(tuningValues, 1);
    if (xList == nullptr || weightList == nullptr || outputList == nullptr ||
        tuning == nullptr) {
        std::cerr << "Failed to create GroupedMatmul metadata\n";
        return 1;
    }

    void *workspace = nullptr;
    uint64_t workspaceBytes = 0;
    std::vector<float> measuredUs;
    measuredUs.reserve(kMeasured);
    for (int run = 0; run < kWarmup + kMeasured; ++run) {
        uint64_t thisWorkspaceBytes = 0;
        aclOpExecutor *executor = nullptr;
        CHECK_ACL(aclnnGroupedMatmulWeightNzGetWorkspaceSize(
            xList, weightList, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            groupList, nullptr, nullptr, nullptr,
            /*splitItem=single input, single output=*/3,
            /*groupType=M axis=*/0,
            /*groupListType=cumulative=*/0,
            /*actType=none=*/0, tuning,
            /*quantGroupSize=*/0, outputList, nullptr, nullptr, &thisWorkspaceBytes,
            &executor));
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
        aclrtEvent start = nullptr;
        aclrtEvent end = nullptr;
        CHECK_ACL(aclrtCreateEvent(&start));
        CHECK_ACL(aclrtCreateEvent(&end));
        CHECK_ACL(aclrtRecordEvent(start, stream));
        CHECK_ACL(aclnnGroupedMatmulWeightNz(
            workspace, workspaceBytes, executor, stream));
        CHECK_ACL(aclrtRecordEvent(end, stream));
        CHECK_ACL(aclrtSynchronizeEvent(end));
        float elapsedMs = 0.0F;
        CHECK_ACL(aclrtEventElapsedTime(&elapsedMs, start, end));
        CHECK_ACL(aclrtDestroyEvent(start));
        CHECK_ACL(aclrtDestroyEvent(end));

        const float elapsedUs = elapsedMs * 1000.0F;
        std::cout << (run < kWarmup ? "warmup" : "sample") << '['
                  << (run < kWarmup ? run : run - kWarmup) << "]_us=" << std::fixed
                  << std::setprecision(1) << elapsedUs << '\n';
        if (run >= kWarmup) {
            measuredUs.push_back(elapsedUs);
        }
    }

    std::vector<int32_t> outputHost(static_cast<size_t>(shape.m * shape.n));
    CHECK_ACL(aclrtMemcpy(
        outputHost.data(), outputBytes, outputDevice, outputBytes,
        ACL_MEMCPY_DEVICE_TO_HOST));
    const int32_t expected = static_cast<int32_t>(shape.k);
    const size_t badOutput = static_cast<size_t>(
        std::count_if(outputHost.begin(), outputHost.end(), [expected](int32_t value) {
            return value != expected;
        }));
    const float mean =
        std::accumulate(measuredUs.begin(), measuredUs.end(), 0.0F) /
        measuredUs.size();

    std::cout << "stage=" << shape.name << " M=" << shape.m << " K=" << shape.k
              << " N=" << shape.n << " experts=" << kExperts
              << " rows_per_expert=" << rowsPerExpert << '\n';
    std::cout << "weight_nd_bytes=" << weightBytes
              << " weight_nz_bytes=" << weightBytes
              << " weight_nz_format=" << ACL_FORMAT_FRACTAL_NZ
              << " workspace_bytes=" << workspaceBytes << '\n';
    std::cout << "correctness=" << (badOutput == 0 ? "PASS" : "FAIL")
              << " bad_output_count=" << badOutput << " expected=" << expected
              << " actual0=" << outputHost.front() << '\n';
    std::cout << "measured_median_us=" << Median(measuredUs)
              << " measured_mean_us=" << mean << '\n';

    if (workspace != nullptr) {
        CHECK_ACL(aclrtFree(workspace));
    }
    CHECK_ACL(aclDestroyIntArray(tuning));
    CHECK_ACL(aclDestroyTensorList(outputList));
    CHECK_ACL(aclDestroyTensorList(weightList));
    CHECK_ACL(aclDestroyTensorList(xList));
    CHECK_ACL(aclDestroyTensor(weightNz));
    CHECK_ACL(aclDestroyTensor(groupList));
    CHECK_ACL(aclDestroyTensor(output));
    CHECK_ACL(aclDestroyTensor(x));
    CHECK_ACL(aclrtFree(weightNzDevice));
    CHECK_ACL(aclrtFree(groupListDevice));
    CHECK_ACL(aclrtFree(outputDevice));
    CHECK_ACL(aclrtFree(xDevice));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(device));
    CHECK_ACL(aclFinalize());
    return badOutput == 0 ? 0 : 1;
}
