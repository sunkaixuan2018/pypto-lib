#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the pushed probe commit}"
CANN_ROOT="${CANN_ROOT:-/usr/local/Ascend/cann-9.0.0}"
CUSTOM_OP_ROOT="${CUSTOM_OP_ROOT:?set CUSTOM_OP_ROOT to the installed custom-op package}"
VLLM_ASCEND_SOURCE="${VLLM_ASCEND_SOURCE:?set VLLM_ASCEND_SOURCE to the vllm-ascend checkout}"
PROBE_BIN="${PROBE_BIN:-/tmp/cann_dispatch_ffn_combine_w4a8_probe}"

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

g++ -std=c++17 -O2 -pthread \
    models/deepseek/v4-flash/cann_dispatch_ffn_combine_w4a8_probe.cpp \
    -I"$CANN_ROOT/aarch64-linux/include" \
    -I"$VLLM_ASCEND_SOURCE/csrc/mc2/dispatch_ffn_combine_w4_a8/op_host/op_api" \
    -L"$CANN_ROOT/aarch64-linux/lib64" \
    -L"$CUSTOM_VENDOR_ROOT/op_api/lib" \
    -Wl,-rpath,"$CANN_ROOT/aarch64-linux/lib64" \
    -Wl,-rpath,"$CUSTOM_VENDOR_ROOT/op_api/lib" \
    -lascendcl -lhccl -lcust_opapi -lopapi -lnnopbase \
    -o "$PROBE_BIN"

echo "commit=$(git rev-parse HEAD)"
echo "devices=${TASK_DEVICE:-0,2,4,6,8,10,12,14}"
exec "$PROBE_BIN" "${TASK_DEVICE:-0,2,4,6,8,10,12,14}"
