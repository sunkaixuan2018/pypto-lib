/**
 * Standalone host-side CANN probe for the EP8 routed-expert W13 stage.
 *
 * The production shape is 16 local experts, 16 routed rows per expert,
 * hidden size 4096, and concatenated gate/up width 4096.  The probe uses
 * packed INT4 FRACTAL_NZ weights and measures the official CANN fused
 * GroupedMatmul + SwiGLU + per-token INT8 quantization operator.
 *
 * Inputs and weights are deliberately zero for the first feasibility gate:
 * this gives an exact (all-zero) output check while retaining the real tensor
 * shapes, storage widths, tiling, and device-side memory traffic.
 */

#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_grouped_matmul_swiglu_quant_weight_nz_v2.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

namespace {

constexpr int64_t kExperts = 16;
constexpr int64_t kRowsPerExpert = 16;
constexpr int64_t kM = kExperts * kRowsPerExpert;
constexpr int64_t kK = 4096;
constexpr int64_t kW13N = 4096;
constexpr int64_t kOutputN = kW13N / 2;
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

size_t PackedInt4Bytes(int64_t elements)
{
    return static_cast<size_t>((elements + 1) / 2);
}

void *DeviceAlloc(size_t bytes)
{
    void *ptr = nullptr;
    CHECK_ACL(aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemset(ptr, bytes, 0, bytes));
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
        dims.data(), dims.size(), dtype, strides.data(), 0, format, dims.data(), dims.size(), data);
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
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(device));

    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    const size_t xBytes = static_cast<size_t>(kM * kK);
    const size_t weightBytes = PackedInt4Bytes(kExperts * kK * kW13N);
    const size_t weightScaleBytes =
        static_cast<size_t>(kExperts * kW13N * 2) * sizeof(float);
    const size_t assistBytes = static_cast<size_t>(kExperts * kW13N) * sizeof(float);
    const size_t xScaleBytes = static_cast<size_t>(kM);
    const size_t groupListBytes = static_cast<size_t>(kExperts) * sizeof(int64_t);
    const size_t outputBytes = static_cast<size_t>(kM * kOutputN);
    const size_t outputScaleBytes = static_cast<size_t>(kM) * sizeof(float);

    void *xDevice = DeviceAlloc(xBytes);
    void *weightDevice = DeviceAlloc(weightBytes);
    void *weightScaleDevice = DeviceAlloc(weightScaleBytes);
    void *assistDevice = DeviceAlloc(assistBytes);
    void *xScaleDevice = DeviceAlloc(xScaleBytes);
    void *groupListDevice = DeviceAlloc(groupListBytes);
    void *outputDevice = DeviceAlloc(outputBytes);
    void *outputScaleDevice = DeviceAlloc(outputScaleBytes);

    std::vector<float> weightScaleHost(
        static_cast<size_t>(kExperts * kW13N * 2), 1.0F);
    std::vector<uint8_t> xScaleHost(static_cast<size_t>(kM), 127U);
    std::vector<int64_t> groupListHost(static_cast<size_t>(kExperts), kRowsPerExpert);
    CHECK_ACL(aclrtMemcpy(
        weightScaleDevice, weightScaleBytes, weightScaleHost.data(), weightScaleBytes,
        ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(
        xScaleDevice, xScaleBytes, xScaleHost.data(), xScaleBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(
        groupListDevice, groupListBytes, groupListHost.data(), groupListBytes,
        ACL_MEMCPY_HOST_TO_DEVICE));

    aclTensor *x = CreateTensor(xDevice, {kM, kK}, ACL_INT8, ACL_FORMAT_ND);
    aclTensor *weight =
        CreateTensor(weightDevice, {kExperts, kK, kW13N}, ACL_INT4, ACL_FORMAT_FRACTAL_NZ);
    aclTensor *weightScale = CreateTensor(
        weightScaleDevice, {kExperts, kW13N, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    aclTensor *assist =
        CreateTensor(assistDevice, {kExperts, kW13N}, ACL_FLOAT, ACL_FORMAT_ND);
    aclTensor *xScale =
        CreateTensor(xScaleDevice, {kM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    aclTensor *groupList =
        CreateTensor(groupListDevice, {kExperts}, ACL_INT64, ACL_FORMAT_ND);
    aclTensor *output =
        CreateTensor(outputDevice, {kM, kOutputN}, ACL_INT8, ACL_FORMAT_ND);
    aclTensor *outputScale =
        CreateTensor(outputScaleDevice, {kM}, ACL_FLOAT, ACL_FORMAT_ND);

    const aclTensor *weightItems[] = {weight};
    const aclTensor *weightScaleItems[] = {weightScale};
    const aclTensor *assistItems[] = {assist};
    aclTensorList *weights = aclCreateTensorList(weightItems, 1);
    aclTensorList *weightScales = aclCreateTensorList(weightScaleItems, 1);
    aclTensorList *assists = aclCreateTensorList(assistItems, 1);
    const int64_t tuningValues[] = {kRowsPerExpert};
    aclIntArray *tuning = aclCreateIntArray(tuningValues, 1);

    if (x == nullptr || weight == nullptr || weightScale == nullptr || assist == nullptr ||
        xScale == nullptr || groupList == nullptr || output == nullptr ||
        outputScale == nullptr || weights == nullptr || weightScales == nullptr ||
        assists == nullptr || tuning == nullptr) {
        std::cerr << "Failed to create one or more ACL metadata objects\n";
        return 1;
    }

    void *workspace = nullptr;
    uint64_t workspaceBytes = 0;
    std::vector<float> measuredUs;
    measuredUs.reserve(kMeasured);

    for (int run = 0; run < kWarmup + kMeasured; ++run) {
        uint64_t thisWorkspaceBytes = 0;
        aclOpExecutor *executor = nullptr;
        CHECK_ACL(aclnnGroupedMatmulSwigluQuantWeightNzV2GetWorkspaceSize(
            x, weights, weightScales, assists, nullptr, xScale, nullptr, groupList,
            /*dequantMode=*/0, /*dequantDtype=*/0, /*quantMode=*/0,
            /*groupListType=count=*/1, tuning, output, outputScale, &thisWorkspaceBytes,
            &executor));
        if (run == 0) {
            workspaceBytes = thisWorkspaceBytes;
            if (workspaceBytes > 0) {
                CHECK_ACL(
                    aclrtMalloc(&workspace, workspaceBytes, ACL_MEM_MALLOC_HUGE_FIRST));
            }
            std::cout << "workspace_bytes=" << workspaceBytes << '\n';
        } else if (thisWorkspaceBytes != workspaceBytes) {
            std::cerr << "Workspace size changed between runs\n";
            return 1;
        }

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
                  << (run < kWarmup ? run : run - kWarmup) << "]_us=" << std::fixed
                  << std::setprecision(1) << elapsedUs << '\n';
        if (run >= kWarmup) {
            measuredUs.push_back(elapsedUs);
        }
    }

    std::vector<int8_t> outputHost(outputBytes);
    CHECK_ACL(aclrtMemcpy(
        outputHost.data(), outputBytes, outputDevice, outputBytes,
        ACL_MEMCPY_DEVICE_TO_HOST));
    const size_t nonzero = static_cast<size_t>(
        std::count_if(outputHost.begin(), outputHost.end(), [](int8_t value) {
            return value != 0;
        }));
    if (nonzero != 0) {
        std::cerr << "correctness=FAIL nonzero_output_count=" << nonzero << '\n';
        return 1;
    }

    const float mean =
        std::accumulate(measuredUs.begin(), measuredUs.end(), 0.0F) / measuredUs.size();
    std::cout << "correctness=PASS all_zero_output\n";
    std::cout << "measured_median_us=" << Median(measuredUs)
              << " measured_mean_us=" << mean << '\n';

    if (workspace != nullptr) {
        CHECK_ACL(aclrtFree(workspace));
    }
    CHECK_ACL(aclDestroyIntArray(tuning));
    CHECK_ACL(aclDestroyTensorList(assists));
    CHECK_ACL(aclDestroyTensorList(weightScales));
    CHECK_ACL(aclDestroyTensorList(weights));
    CHECK_ACL(aclDestroyTensor(outputScale));
    CHECK_ACL(aclDestroyTensor(output));
    CHECK_ACL(aclDestroyTensor(groupList));
    CHECK_ACL(aclDestroyTensor(xScale));
    CHECK_ACL(aclDestroyTensor(assist));
    CHECK_ACL(aclDestroyTensor(weightScale));
    CHECK_ACL(aclDestroyTensor(weight));
    CHECK_ACL(aclDestroyTensor(x));

    for (void *ptr : {xDevice, weightDevice, weightScaleDevice, assistDevice, xScaleDevice,
                      groupListDevice, outputDevice, outputScaleDevice}) {
        CHECK_ACL(aclrtFree(ptr));
    }
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(device));
    CHECK_ACL(aclFinalize());
    return 0;
}
