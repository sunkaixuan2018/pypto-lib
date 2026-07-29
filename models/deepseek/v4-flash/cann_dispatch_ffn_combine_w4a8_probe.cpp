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
 * Standalone EP8 probe for vLLM-Ascend's fused W4A8 MoE operator.
 *
 * The probe uses the Flash decode shape and balanced routing: eight input
 * tokens per rank, top-k six, 16 local experts, hidden size 4096, and
 * intermediate size 2048. W1 is nonzero while W2 is zero, so every output
 * must be written as zero after dispatch, both grouped matmuls, and combine.
 */

#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <hccl/hccl_comm.h>

#include "aclnn_dispatch_ffn_combine_w4_a8.h"

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int64_t kWorldSize = 8;
constexpr int64_t kTokens = 8;
constexpr int64_t kHidden = 4096;
constexpr int64_t kIntermediate = 2048;
constexpr int64_t kTopK = 6;
constexpr int64_t kLocalExperts = 16;
constexpr int64_t kGlobalExperts = kWorldSize * kLocalExperts;
constexpr int64_t kW13N = 2 * kIntermediate;
constexpr int64_t kMaxOutputSize = 512;
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

#define CHECK_HCCL(expr)                                                                 \
    do {                                                                                 \
        const auto status = (expr);                                                      \
        if (status != HCCL_SUCCESS) {                                                    \
            std::cerr << "HCCL failure " << status << " at " << __FILE__ << ':'         \
                      << __LINE__ << ": " #expr << '\n';                                 \
            std::exit(1);                                                                \
        }                                                                                \
    } while (false)

class ThreadBarrier {
public:
    explicit ThreadBarrier(size_t participants) : participants_(participants) {}

    void Wait()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        const size_t generation = generation_;
        if (++arrived_ == participants_) {
            arrived_ = 0;
            ++generation_;
            condition_.notify_all();
            return;
        }
        condition_.wait(lock, [this, generation] { return generation_ != generation; });
    }

private:
    const size_t participants_;
    size_t arrived_ = 0;
    size_t generation_ = 0;
    std::mutex mutex_;
    std::condition_variable condition_;
};

struct RankResult {
    uint64_t workspaceBytes = 0;
    size_t freeBeforeWeights = 0;
    size_t freeAfterWeights = 0;
    size_t totalHbm = 0;
    std::vector<float> measuredUs;
    size_t badOutput = 0;
    std::vector<int32_t> expertTokenNums;
};

std::mutex gLogMutex;

void *DeviceAlloc(size_t bytes, uint8_t value = 0)
{
    void *ptr = nullptr;
    CHECK_ACL(aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemset(ptr, bytes, value, bytes));
    return ptr;
}

aclTensor *CreateTensor(
    void *data, const std::vector<int64_t> &dims, aclDataType dtype, aclFormat format)
{
    std::vector<int64_t> strides(dims.size(), 1);
    for (int64_t i = static_cast<int64_t>(dims.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] =
            strides[static_cast<size_t>(i + 1)] * dims[static_cast<size_t>(i + 1)];
    }
    return aclCreateTensor(
        dims.data(), dims.size(), dtype, strides.data(), 0, format, dims.data(), dims.size(),
        data);
}

std::vector<int32_t> ParseDevices(const std::string &text)
{
    std::vector<int32_t> devices;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (!item.empty()) {
            devices.push_back(std::stoi(item));
        }
    }
    return devices;
}

float Median(std::vector<float> values)
{
    std::sort(values.begin(), values.end());
    return (values[3] + values[4]) * 0.5F;
}

void RunRank(
    int rank, int device, HcclComm comm, bool queryOnly, ThreadBarrier &barrier,
    RankResult &result)
{
    CHECK_ACL(aclrtSetDevice(device));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));
    if (queryOnly) {
        CHECK_ACL(aclrtGetMemInfo(
            ACL_HBM_MEM, &result.freeBeforeWeights, &result.totalHbm));
    }

    char commName[COMM_NAME_MAX_LENGTH] = {};
    CHECK_HCCL(HcclGetCommName(comm, commName));

    const size_t xBytes = static_cast<size_t>(kTokens * kHidden) * sizeof(uint16_t);
    const size_t w1Bytes =
        static_cast<size_t>(kLocalExperts * kHidden * kW13N) / 2;
    const size_t w2Bytes =
        static_cast<size_t>(kLocalExperts * kIntermediate * kHidden) / 2;
    const size_t scale1Bytes =
        static_cast<size_t>(kLocalExperts * kW13N) * sizeof(uint64_t);
    const size_t scale2Bytes =
        static_cast<size_t>(kLocalExperts * kHidden) * sizeof(uint64_t);
    const size_t bias1Bytes =
        static_cast<size_t>(kLocalExperts * kW13N) * sizeof(float);
    const size_t bias2Bytes =
        static_cast<size_t>(kLocalExperts * kHidden) * sizeof(float);
    const size_t expertIdBytes =
        static_cast<size_t>(kTokens * kTopK) * sizeof(int32_t);
    const size_t probsBytes =
        static_cast<size_t>(kTokens * kTopK) * sizeof(float);
    const size_t activeMaskBytes = static_cast<size_t>(kTokens) * sizeof(bool);
    const size_t outputBytes =
        static_cast<size_t>(kTokens * kHidden) * sizeof(uint16_t);
    const size_t expertTokenBytes =
        static_cast<size_t>(kLocalExperts) * sizeof(int32_t);

    void *xDevice = DeviceAlloc(xBytes);
    void *w1Device = DeviceAlloc(w1Bytes, 0x11);
    void *w2Device = DeviceAlloc(w2Bytes);
    void *scale1Device = DeviceAlloc(scale1Bytes);
    void *scale2Device = DeviceAlloc(scale2Bytes);
    void *bias1Device = DeviceAlloc(bias1Bytes);
    void *bias2Device = DeviceAlloc(bias2Bytes);
    void *expertIdDevice = DeviceAlloc(expertIdBytes);
    void *probsDevice = DeviceAlloc(probsBytes);
    void *activeMaskDevice = DeviceAlloc(activeMaskBytes);
    void *outputDevice = DeviceAlloc(outputBytes, 0x40);
    void *expertTokenDevice = DeviceAlloc(expertTokenBytes);

    std::vector<uint16_t> xHost(static_cast<size_t>(kTokens * kHidden), 0x3F80);
    std::vector<uint64_t> scale1Host(
        static_cast<size_t>(kLocalExperts * kW13N), 0x000000003F800000ULL);
    std::vector<uint64_t> scale2Host(
        static_cast<size_t>(kLocalExperts * kHidden), 0x000000003F800000ULL);
    std::vector<float> bias1Host(
        static_cast<size_t>(kLocalExperts * kW13N),
        8.0F * static_cast<float>(kHidden));
    std::vector<int32_t> expertIdHost(static_cast<size_t>(kTokens * kTopK));
    std::vector<float> probsHost(
        static_cast<size_t>(kTokens * kTopK), 1.0F / static_cast<float>(kTopK));
    for (int64_t token = 0; token < kTokens; ++token) {
        for (int64_t top = 0; top < kTopK; ++top) {
            const int64_t route = rank * kTokens * kTopK + token * kTopK + top;
            expertIdHost[static_cast<size_t>(token * kTopK + top)] =
                static_cast<int32_t>(route % kGlobalExperts);
        }
    }

    CHECK_ACL(aclrtMemcpy(
        xDevice, xBytes, xHost.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(
        scale1Device, scale1Bytes, scale1Host.data(), scale1Bytes,
        ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(
        scale2Device, scale2Bytes, scale2Host.data(), scale2Bytes,
        ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(
        bias1Device, bias1Bytes, bias1Host.data(), bias1Bytes,
        ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(
        expertIdDevice, expertIdBytes, expertIdHost.data(), expertIdBytes,
        ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(
        probsDevice, probsBytes, probsHost.data(), probsBytes,
        ACL_MEMCPY_HOST_TO_DEVICE));
    std::vector<uint8_t> activeMaskBytesHost(static_cast<size_t>(kTokens), 1);
    CHECK_ACL(aclrtMemcpy(
        activeMaskDevice, activeMaskBytes, activeMaskBytesHost.data(), activeMaskBytes,
        ACL_MEMCPY_HOST_TO_DEVICE));

    aclTensor *x = CreateTensor(xDevice, {kTokens, kHidden}, ACL_BF16, ACL_FORMAT_ND);
    aclTensor *expertId =
        CreateTensor(expertIdDevice, {kTokens, kTopK}, ACL_INT32, ACL_FORMAT_ND);
    aclTensor *probs =
        CreateTensor(probsDevice, {kTokens, kTopK}, ACL_FLOAT, ACL_FORMAT_ND);
    aclTensor *activeMask =
        CreateTensor(activeMaskDevice, {kTokens}, ACL_BOOL, ACL_FORMAT_ND);
    aclTensor *output =
        CreateTensor(outputDevice, {kTokens, kHidden}, ACL_BF16, ACL_FORMAT_ND);
    aclTensor *expertTokenNums =
        CreateTensor(expertTokenDevice, {1, kLocalExperts}, ACL_INT32, ACL_FORMAT_ND);

    std::vector<aclTensor *> ownedTensors = {
        x, expertId, probs, activeMask, output, expertTokenNums};
    std::vector<const aclTensor *> w1Items;
    std::vector<const aclTensor *> w2Items;
    std::vector<const aclTensor *> scale1Items;
    std::vector<const aclTensor *> scale2Items;
    std::vector<const aclTensor *> bias1Items;
    std::vector<const aclTensor *> bias2Items;
    const size_t w1BytesPerExpert = static_cast<size_t>(kHidden * kW13N) / 2;
    const size_t w2BytesPerExpert = static_cast<size_t>(kIntermediate * kHidden) / 2;
    const size_t scale1BytesPerExpert = static_cast<size_t>(kW13N) * sizeof(uint64_t);
    const size_t scale2BytesPerExpert = static_cast<size_t>(kHidden) * sizeof(uint64_t);
    const size_t bias1BytesPerExpert = static_cast<size_t>(kW13N) * sizeof(float);
    const size_t bias2BytesPerExpert = static_cast<size_t>(kHidden) * sizeof(float);

    for (int64_t expert = 0; expert < kLocalExperts; ++expert) {
        aclTensor *w1 = CreateTensor(
            static_cast<uint8_t *>(w1Device) + expert * w1BytesPerExpert,
            {kHidden, kW13N / 8}, ACL_INT32, ACL_FORMAT_FRACTAL_NZ);
        aclTensor *w2 = CreateTensor(
            static_cast<uint8_t *>(w2Device) + expert * w2BytesPerExpert,
            {kIntermediate, kHidden / 8}, ACL_INT32, ACL_FORMAT_FRACTAL_NZ);
        aclTensor *scale1 = CreateTensor(
            static_cast<uint8_t *>(scale1Device) + expert * scale1BytesPerExpert,
            {kW13N}, ACL_INT64, ACL_FORMAT_ND);
        aclTensor *scale2 = CreateTensor(
            static_cast<uint8_t *>(scale2Device) + expert * scale2BytesPerExpert,
            {kHidden}, ACL_INT64, ACL_FORMAT_ND);
        aclTensor *bias1 = CreateTensor(
            static_cast<uint8_t *>(bias1Device) + expert * bias1BytesPerExpert,
            {kW13N}, ACL_FLOAT, ACL_FORMAT_ND);
        aclTensor *bias2 = CreateTensor(
            static_cast<uint8_t *>(bias2Device) + expert * bias2BytesPerExpert,
            {kHidden}, ACL_FLOAT, ACL_FORMAT_ND);
        ownedTensors.insert(
            ownedTensors.end(), {w1, w2, scale1, scale2, bias1, bias2});
        w1Items.push_back(w1);
        w2Items.push_back(w2);
        scale1Items.push_back(scale1);
        scale2Items.push_back(scale2);
        bias1Items.push_back(bias1);
        bias2Items.push_back(bias2);
    }

    aclTensorList *w1List = aclCreateTensorList(w1Items.data(), w1Items.size());
    aclTensorList *w2List = aclCreateTensorList(w2Items.data(), w2Items.size());
    aclTensorList *scale1List =
        aclCreateTensorList(scale1Items.data(), scale1Items.size());
    aclTensorList *scale2List =
        aclCreateTensorList(scale2Items.data(), scale2Items.size());
    aclTensorList *bias1List =
        aclCreateTensorList(bias1Items.data(), bias1Items.size());
    aclTensorList *bias2List =
        aclCreateTensorList(bias2Items.data(), bias2Items.size());

    if (std::find(ownedTensors.begin(), ownedTensors.end(), nullptr) !=
            ownedTensors.end() ||
        w1List == nullptr || w2List == nullptr || scale1List == nullptr ||
        scale2List == nullptr || bias1List == nullptr || bias2List == nullptr) {
        std::cerr << "rank " << rank << " failed to create ACL metadata\n";
        std::exit(1);
    }

    void *workspace = nullptr;
    uint64_t workspaceBytes = 0;
    result.measuredUs.reserve(kMeasured);
    for (int run = 0; run < kWarmup + kMeasured; ++run) {
        uint64_t thisWorkspaceBytes = 0;
        aclOpExecutor *executor = nullptr;
        CHECK_ACL(aclnnDispatchFFNCombineW4A8GetWorkspaceSize(
            x, w1List, w2List, expertId, scale1List, scale2List, bias1List, bias2List,
            probs, activeMask, commName, kMaxOutputSize, 10.0, output, expertTokenNums,
            &thisWorkspaceBytes, &executor));
        if (run == 0) {
            workspaceBytes = thisWorkspaceBytes;
            result.workspaceBytes = workspaceBytes;
            if (queryOnly) {
                CHECK_ACL(aclrtGetMemInfo(
                    ACL_HBM_MEM, &result.freeAfterWeights, &result.totalHbm));
                {
                    std::lock_guard<std::mutex> lock(gLogMutex);
                    std::cout << "query_rank=" << rank << " device=" << device
                              << " comm=" << commName
                              << " workspace_bytes=" << workspaceBytes
                              << " free_before_weights=" << result.freeBeforeWeights
                              << " free_after_weights=" << result.freeAfterWeights
                              << " total_hbm=" << result.totalHbm << std::endl;
                }
                barrier.Wait();
                break;
            }
            if (workspaceBytes > 0) {
                CHECK_ACL(
                    aclrtMalloc(&workspace, workspaceBytes, ACL_MEM_MALLOC_HUGE_FIRST));
            }
        } else if (thisWorkspaceBytes != workspaceBytes) {
            std::cerr << "rank " << rank << " workspace size changed\n";
            std::exit(1);
        }

        CHECK_ACL(aclrtMemset(outputDevice, outputBytes, 0x40, outputBytes));
        CHECK_HCCL(HcclBarrier(comm, stream));
        CHECK_ACL(aclrtSynchronizeStream(stream));
        barrier.Wait();

        aclrtEvent start = nullptr;
        aclrtEvent end = nullptr;
        CHECK_ACL(aclrtCreateEvent(&start));
        CHECK_ACL(aclrtCreateEvent(&end));
        CHECK_ACL(aclrtRecordEvent(start, stream));
        CHECK_ACL(aclnnDispatchFFNCombineW4A8(
            workspace, workspaceBytes, executor, stream));
        CHECK_ACL(aclrtRecordEvent(end, stream));
        CHECK_ACL(aclrtSynchronizeEvent(end));
        float elapsedMs = 0.0F;
        CHECK_ACL(aclrtEventElapsedTime(&elapsedMs, start, end));
        CHECK_ACL(aclrtDestroyEvent(start));
        CHECK_ACL(aclrtDestroyEvent(end));
        if (run >= kWarmup) {
            result.measuredUs.push_back(elapsedMs * 1000.0F);
        }
        barrier.Wait();
    }

    if (!queryOnly) {
        std::vector<uint16_t> outputHost(static_cast<size_t>(kTokens * kHidden));
        result.expertTokenNums.resize(static_cast<size_t>(kLocalExperts));
        CHECK_ACL(aclrtMemcpy(
            outputHost.data(), outputBytes, outputDevice, outputBytes,
            ACL_MEMCPY_DEVICE_TO_HOST));
        CHECK_ACL(aclrtMemcpy(
            result.expertTokenNums.data(), expertTokenBytes, expertTokenDevice,
            expertTokenBytes, ACL_MEMCPY_DEVICE_TO_HOST));
        result.badOutput = static_cast<size_t>(
            std::count_if(outputHost.begin(), outputHost.end(), [](uint16_t value) {
                return value != 0;
            }));
    }

    if (workspace != nullptr) {
        CHECK_ACL(aclrtFree(workspace));
    }
    CHECK_ACL(aclDestroyTensorList(bias2List));
    CHECK_ACL(aclDestroyTensorList(bias1List));
    CHECK_ACL(aclDestroyTensorList(scale2List));
    CHECK_ACL(aclDestroyTensorList(scale1List));
    CHECK_ACL(aclDestroyTensorList(w2List));
    CHECK_ACL(aclDestroyTensorList(w1List));
    for (aclTensor *tensor : ownedTensors) {
        CHECK_ACL(aclDestroyTensor(tensor));
    }
    CHECK_ACL(aclrtFree(expertTokenDevice));
    CHECK_ACL(aclrtFree(outputDevice));
    CHECK_ACL(aclrtFree(activeMaskDevice));
    CHECK_ACL(aclrtFree(probsDevice));
    CHECK_ACL(aclrtFree(expertIdDevice));
    CHECK_ACL(aclrtFree(bias2Device));
    CHECK_ACL(aclrtFree(bias1Device));
    CHECK_ACL(aclrtFree(scale2Device));
    CHECK_ACL(aclrtFree(scale1Device));
    CHECK_ACL(aclrtFree(w2Device));
    CHECK_ACL(aclrtFree(w1Device));
    CHECK_ACL(aclrtFree(xDevice));
    CHECK_ACL(aclrtDestroyStream(stream));
}

} // namespace

int main(int argc, char **argv)
{
    const std::string deviceText = argc > 1 ? argv[1] : "0,2,4,6,8,10,12,14";
    const char *queryOnlyEnv = std::getenv("W4A8_QUERY_ONLY");
    const bool queryOnly =
        queryOnlyEnv != nullptr && std::string(queryOnlyEnv) != "0";
    std::vector<int32_t> devices = ParseDevices(deviceText);
    if (devices.size() != kWorldSize) {
        std::cerr << "expected " << kWorldSize << " devices, got " << devices.size()
                  << " from " << deviceText << '\n';
        return 2;
    }

    CHECK_ACL(aclInit(nullptr));
    for (int32_t device : devices) {
        CHECK_ACL(aclrtSetDevice(device));
    }
    std::vector<HcclComm> comms(static_cast<size_t>(kWorldSize));
    CHECK_HCCL(HcclCommInitAll(
        static_cast<uint32_t>(kWorldSize), devices.data(), comms.data()));

    ThreadBarrier barrier(static_cast<size_t>(kWorldSize));
    std::vector<RankResult> results(static_cast<size_t>(kWorldSize));
    std::vector<std::thread> workers;
    for (int rank = 0; rank < kWorldSize; ++rank) {
        workers.emplace_back(
            RunRank, rank, devices[static_cast<size_t>(rank)],
            comms[static_cast<size_t>(rank)], queryOnly, std::ref(barrier),
            std::ref(results[static_cast<size_t>(rank)]));
    }
    for (std::thread &worker : workers) {
        worker.join();
    }

    if (queryOnly) {
        std::cout << "query_only=PASS ranks=" << kWorldSize << '\n';
        for (HcclComm comm : comms) {
            CHECK_HCCL(HcclCommDestroy(comm));
        }
        CHECK_ACL(aclFinalize());
        return 0;
    }

    bool correctness = true;
    for (int rank = 0; rank < kWorldSize; ++rank) {
        const RankResult &result = results[static_cast<size_t>(rank)];
        const bool countsCorrect =
            std::all_of(
                result.expertTokenNums.begin(), result.expertTokenNums.end(),
                [](int32_t count) { return count == 3; });
        correctness = correctness && result.badOutput == 0 && countsCorrect;
        std::cout << "rank=" << rank << " device="
                  << devices[static_cast<size_t>(rank)]
                  << " workspace_bytes=" << result.workspaceBytes
                  << " bad_output=" << result.badOutput << " expert_counts=";
        for (size_t i = 0; i < result.expertTokenNums.size(); ++i) {
            std::cout << (i == 0 ? "" : ",") << result.expertTokenNums[i];
        }
        std::cout << '\n';
    }

    std::vector<float> lastArriving;
    for (int sample = 0; sample < kMeasured; ++sample) {
        std::cout << "sample[" << sample << "]_rank_us=";
        float maximum = 0.0F;
        for (int rank = 0; rank < kWorldSize; ++rank) {
            const float value =
                results[static_cast<size_t>(rank)].measuredUs[static_cast<size_t>(sample)];
            maximum = std::max(maximum, value);
            std::cout << (rank == 0 ? "" : ",") << std::fixed << std::setprecision(1)
                      << value;
        }
        lastArriving.push_back(maximum);
        std::cout << " last_arriving_us=" << maximum << '\n';
    }

    const float mean =
        std::accumulate(lastArriving.begin(), lastArriving.end(), 0.0F) /
        lastArriving.size();
    std::cout << "correctness=" << (correctness ? "PASS" : "FAIL") << '\n';
    std::cout << "last_arriving_median_us=" << Median(lastArriving)
              << " last_arriving_mean_us=" << mean << '\n';

    for (HcclComm comm : comms) {
        CHECK_HCCL(HcclCommDestroy(comm));
    }
    CHECK_ACL(aclFinalize());
    return correctness ? 0 : 1;
}
