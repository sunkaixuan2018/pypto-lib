#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the pushed probe commit}"
PROBE_BIN="${PROBE_BIN:-/tmp/cann_gmm_swiglu_quant_w8a8_probe}"
CANN_ROOT="${CANN_ROOT:-/usr/local/Ascend/cann-9.0.0}"
CUSTOM_OP_ROOT="${CUSTOM_OP_ROOT:-/data/sunkaixuan/sunkaixuan_subdir/all_libs/vllm-ascend-dsv4-gmm-custom-20260730}"
VLLM_ASCEND_SOURCE="${VLLM_ASCEND_SOURCE:-/data/sunkaixuan/sunkaixuan_subdir/all_libs/vllm-ascend-dsv4-gmm-20260730}"

cd "$REPO"
if [[ "$(git rev-parse HEAD)" != "$EXPECTED_COMMIT" ]]; then
    echo "Unexpected commit $(git rev-parse HEAD); expected $EXPECTED_COMMIT" >&2
    exit 2
fi

set +u
source "$CANN_ROOT/set_env.sh"
set -u

CUSTOM_VENDOR_ROOT="$CUSTOM_OP_ROOT/vendors/custom_transformer"
export ASCEND_CUSTOM_OPP_PATH="$CUSTOM_VENDOR_ROOT${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
export LD_LIBRARY_PATH="$CUSTOM_VENDOR_ROOT/op_api/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

g++ -std=c++17 -O2 \
    models/deepseek/v4-flash/cann_grouped_matmul_swiglu_quant_w8a8_probe.cpp \
    -I"$CANN_ROOT/aarch64-linux/include" \
    -I"$VLLM_ASCEND_SOURCE/csrc/gmm/grouped_matmul_swiglu_quant_v2/op_host/op_api" \
    -L"$CANN_ROOT/aarch64-linux/lib64" \
    -L"$CUSTOM_VENDOR_ROOT/op_api/lib" \
    -Wl,-rpath,"$CANN_ROOT/aarch64-linux/lib64" \
    -Wl,-rpath,"$CUSTOM_VENDOR_ROOT/op_api/lib" \
    -lascendcl -lcust_opapi -lopapi -lnnopbase \
    -o "$PROBE_BIN"

echo "commit=$(git rev-parse HEAD)"
echo "device=${TASK_DEVICE:-0}"
status=0
for m in 768 256; do
    echo "case=M=$m"
    if ! "$PROBE_BIN" "${TASK_DEVICE:-0}" "$m"; then
        status=1
    fi
done
exit "$status"
