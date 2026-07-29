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
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
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
    const bool nonzero = argc > 2 && std::string(argv[2]) == "nonzero";
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(device));

    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    const size_t xBytes = static_cast<size_t>(kM * kK);
    const size_t weightBytes = PackedInt4Bytes(kExperts * kK * kW13N);
    const size_t weightScaleBytes =
        static_cast<size_t>(kExperts * kW13N) * sizeof(uint64_t);
    const size_t assistBytes = static_cast<size_t>(kExperts * kW13N) * sizeof(float);
    const size_t xScaleBytes = static_cast<size_t>(kM) * sizeof(float);
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
    if (nonzero) {
        // Every activation is +1 and every signed INT4 weight nibble is +1.
        // NZ reordering does not change a constant tensor, so this exercises
        // the packed-nibble, W13, SwiGLU, row-amax, and INT8-rounding path with
        // a closed-form reference while retaining the production-sized input.
        CHECK_ACL(aclrtMemset(xDevice, xBytes, 0x01, xBytes));
        CHECK_ACL(aclrtMemset(weightDevice, weightBytes, 0x11, weightBytes));
    }

    // Fixpipe dequant scale format stores FP32 bits in the low word and keeps
    // the high word zero.  It is not a pair of independent FP32 values.
    constexpr uint64_t kFixpipeUnitScale = 0x000000003F800000ULL;
    std::vector<uint64_t> weightScaleHost(
        static_cast<size_t>(kExperts * kW13N), kFixpipeUnitScale);
    // The A8W4 MSD path rewrites x as hi*16 + ((x & 0xf) - 8).  Restore
    // the subtracted eight through the official per-channel assist matrix.
    const float assistValue = nonzero ? 8.0F * static_cast<float>(kK) : 0.0F;
    std::vector<float> assistHost(static_cast<size_t>(kExperts * kW13N), assistValue);
    std::vector<float> xScaleHost(static_cast<size_t>(kM), 1.0F);
    std::vector<int64_t> groupListHost(static_cast<size_t>(kExperts), kRowsPerExpert);
    CHECK_ACL(aclrtMemcpy(
        weightScaleDevice, weightScaleBytes, weightScaleHost.data(), weightScaleBytes,
        ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(
        xScaleDevice, xScaleBytes, xScaleHost.data(), xScaleBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(
        assistDevice, assistBytes, assistHost.data(), assistBytes,
        ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(
        groupListDevice, groupListBytes, groupListHost.data(), groupListBytes,
        ACL_MEMCPY_HOST_TO_DEVICE));

    aclTensor *x = CreateTensor(xDevice, {kM, kK}, ACL_INT8, ACL_FORMAT_ND);
    aclTensor *xScale =
        CreateTensor(xScaleDevice, {kM}, ACL_FLOAT, ACL_FORMAT_ND);
    aclTensor *groupList =
        CreateTensor(groupListDevice, {kExperts}, ACL_INT64, ACL_FORMAT_ND);
    aclTensor *output =
        CreateTensor(outputDevice, {kM, kOutputN}, ACL_INT8, ACL_FORMAT_ND);
    aclTensor *outputScale =
        CreateTensor(outputScaleDevice, {kM}, ACL_FLOAT, ACL_FORMAT_ND);

    std::vector<aclTensor *> weightTensors;
    std::vector<aclTensor *> weightScaleTensors;
    std::vector<aclTensor *> assistTensors;
    std::vector<const aclTensor *> weightItems;
    std::vector<const aclTensor *> weightScaleItems;
    std::vector<const aclTensor *> assistItems;
    const size_t weightBytesPerExpert = PackedInt4Bytes(kK * kW13N);
    const size_t weightScaleBytesPerExpert =
        static_cast<size_t>(kW13N) * sizeof(uint64_t);
    const size_t assistBytesPerExpert = static_cast<size_t>(kW13N) * sizeof(float);
    for (int64_t expert = 0; expert < kExperts; ++expert) {
        aclTensor *weight = CreateTensor(
            static_cast<uint8_t *>(weightDevice) + expert * weightBytesPerExpert,
            {kK, kW13N}, ACL_INT4, ACL_FORMAT_FRACTAL_NZ,
            {kK / 64, kW13N / 16, 16, 64});
        aclTensor *weightScale = CreateTensor(
            static_cast<uint8_t *>(weightScaleDevice) + expert * weightScaleBytesPerExpert,
            {kW13N}, ACL_UINT64, ACL_FORMAT_ND);
        aclTensor *assist = CreateTensor(
            static_cast<uint8_t *>(assistDevice) + expert * assistBytesPerExpert, {kW13N},
            ACL_FLOAT, ACL_FORMAT_ND);
        weightTensors.push_back(weight);
        weightScaleTensors.push_back(weightScale);
        assistTensors.push_back(assist);
        weightItems.push_back(weight);
        weightScaleItems.push_back(weightScale);
        assistItems.push_back(assist);
    }
    aclTensorList *weights = aclCreateTensorList(weightItems.data(), weightItems.size());
    aclTensorList *weightScales =
        aclCreateTensorList(weightScaleItems.data(), weightScaleItems.size());
    aclTensorList *assists = aclCreateTensorList(assistItems.data(), assistItems.size());
    const int64_t tuningValues[] = {kRowsPerExpert};
    aclIntArray *tuning = aclCreateIntArray(tuningValues, 1);

    const auto hasNull = [](const std::vector<aclTensor *> &tensors) {
        return std::find(tensors.begin(), tensors.end(), nullptr) != tensors.end();
    };
    if (x == nullptr || xScale == nullptr || groupList == nullptr || output == nullptr ||
        outputScale == nullptr || hasNull(weightTensors) || hasNull(weightScaleTensors) ||
        hasNull(assistTensors) || weights == nullptr || weightScales == nullptr ||
        assists == nullptr || tuning == nullptr) {
        std::cerr << "Failed to create one or more ACL metadata objects\n";
        return 1;
    }

    void *workspace = nullptr;
    uint64_t workspaceBytes = 0;
    std::vector<float> measuredUs;
    measuredUs.reserve(kMeasured);
    const std::vector<float> scaleSentinel(
        static_cast<size_t>(kM), std::numeric_limits<float>::quiet_NaN());

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
        if (nonzero) {
            CHECK_ACL(aclrtMemset(outputDevice, outputBytes, 0x5A, outputBytes));
            CHECK_ACL(aclrtMemcpy(
                outputScaleDevice, outputScaleBytes, scaleSentinel.data(), outputScaleBytes,
                ACL_MEMCPY_HOST_TO_DEVICE));
        }
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
    std::vector<float> outputScaleHost(static_cast<size_t>(kM));
    CHECK_ACL(aclrtMemcpy(
        outputHost.data(), outputBytes, outputDevice, outputBytes,
        ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(
        outputScaleHost.data(), outputScaleBytes, outputScaleDevice, outputScaleBytes,
        ACL_MEMCPY_DEVICE_TO_HOST));
    if (nonzero) {
        const size_t badQuant = static_cast<size_t>(
            std::count_if(outputHost.begin(), outputHost.end(), [](int8_t value) {
                return value != 127;
            }));
        const float projection = static_cast<float>(kK);
        const float expectedActivation =
            projection * projection / (1.0F + std::exp(-projection));
        const float expectedScale = expectedActivation / 127.0F;
        const size_t badScale = static_cast<size_t>(std::count_if(
            outputScaleHost.begin(), outputScaleHost.end(), [expectedScale](float value) {
                return !std::isfinite(value) ||
                       std::abs(value - expectedScale) > expectedScale * 0.002F;
            }));
        if (badQuant != 0 || badScale != 0) {
            std::cerr << "correctness=FAIL bad_quant_count=" << badQuant
                      << " bad_scale_count=" << badScale
                      << " expected_scale=" << expectedScale
                      << " actual_scale0=" << outputScaleHost.front() << '\n';
            for (int64_t row = 0; row < kM; ++row) {
                const int8_t *rowBegin =
                    outputHost.data() + static_cast<size_t>(row * kOutputN);
                const int8_t *rowEnd = rowBegin + kOutputN;
                const size_t rowBadQuant = static_cast<size_t>(
                    std::count_if(rowBegin, rowEnd, [](int8_t value) {
                        return value != 127;
                    }));
                const float scale = outputScaleHost[static_cast<size_t>(row)];
                const bool rowBadScale =
                    !std::isfinite(scale) ||
                    std::abs(scale - expectedScale) > expectedScale * 0.002F;
                if (rowBadQuant == 0 && !rowBadScale) {
                    continue;
                }
                uint32_t scaleBits = 0;
                std::memcpy(&scaleBits, &scale, sizeof(scaleBits));
                const size_t sentinelCount = static_cast<size_t>(
                    std::count(rowBegin, rowEnd, static_cast<int8_t>(0x5A)));
                const size_t zeroCount =
                    static_cast<size_t>(std::count(rowBegin, rowEnd, static_cast<int8_t>(0)));
                const auto [minIt, maxIt] = std::minmax_element(rowBegin, rowEnd);
                std::cerr << "bad_row=" << row << " expert=" << row / kRowsPerExpert
                          << " local_row=" << row % kRowsPerExpert
                          << " scale=" << scale << " scale_bits=0x" << std::hex
                          << scaleBits << std::dec << " q_bad=" << rowBadQuant
                          << " q_sentinel=" << sentinelCount << " q_zero=" << zeroCount
                          << " q_min=" << static_cast<int>(*minIt)
                          << " q_max=" << static_cast<int>(*maxIt) << " q_first8=";
                for (int i = 0; i < 8; ++i) {
                    std::cerr << (i == 0 ? "" : ",") << static_cast<int>(rowBegin[i]);
                }
                std::cerr << '\n';
            }
            return 1;
        }
        std::cout << "correctness=PASS constant_nonzero_int4_nz"
                  << " output_scale0=" << outputScaleHost.front() << '\n';
    } else {
        const size_t nonzeroOutput = static_cast<size_t>(
            std::count_if(outputHost.begin(), outputHost.end(), [](int8_t value) {
                return value != 0;
            }));
        if (nonzeroOutput != 0) {
            std::cerr << "correctness=FAIL nonzero_output_count=" << nonzeroOutput << '\n';
            return 1;
        }
        std::cout << "correctness=PASS all_zero_output\n";
    }

    const float mean =
        std::accumulate(measuredUs.begin(), measuredUs.end(), 0.0F) / measuredUs.size();
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
    for (aclTensor *tensor : assistTensors) {
        CHECK_ACL(aclDestroyTensor(tensor));
    }
    for (aclTensor *tensor : weightScaleTensors) {
        CHECK_ACL(aclDestroyTensor(tensor));
    }
    for (aclTensor *tensor : weightTensors) {
        CHECK_ACL(aclDestroyTensor(tensor));
    }
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
