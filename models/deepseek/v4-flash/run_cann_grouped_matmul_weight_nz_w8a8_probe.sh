#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the pushed probe commit}"
PROBE_BIN="${PROBE_BIN:-/tmp/cann_gmm_weight_nz_w8a8_probe}"
CANN_ROOT="${CANN_ROOT:-/usr/local/Ascend/cann-9.0.0}"

cd "$REPO"
if [[ "$(git rev-parse HEAD)" != "$EXPECTED_COMMIT" ]]; then
    echo "Unexpected commit $(git rev-parse HEAD); expected $EXPECTED_COMMIT" >&2
    exit 2
fi

set +u
source "$CANN_ROOT/set_env.sh"
set -u

g++ -std=c++17 -O2 \
    models/deepseek/v4-flash/cann_grouped_matmul_weight_nz_w8a8_probe.cpp \
    -I"$CANN_ROOT/aarch64-linux/include" \
    -L"$CANN_ROOT/aarch64-linux/lib64" \
    -Wl,-rpath,"$CANN_ROOT/aarch64-linux/lib64" \
    -lascendcl -lopapi -lnnopbase \
    -o "$PROBE_BIN"

echo "commit=$(git rev-parse HEAD)"
echo "device=${TASK_DEVICE:-0}"
for stage in w13 w2; do
    for m in 768 256; do
        echo "case=$stage,M=$m"
        "$PROBE_BIN" "${TASK_DEVICE:-0}" "$stage" "$m"
    done
done
