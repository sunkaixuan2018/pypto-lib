#!/usr/bin/env bash
set -euo pipefail

REPO=/data/sunkaixuan/sunkaixuan_subdir/pypto-lib
EXPECTED_COMMIT="${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the pushed probe commit}"
PROBE_BIN=/data/sunkaixuan/codex_sh/cann_gmm_swiglu_quant_probe_20260729
CANN_ROOT=/usr/local/Ascend/cann-9.0.0
PROBE_MODE="${PROBE_MODE:-zero}"

cd "$REPO"
if [[ "$(git rev-parse HEAD)" != "$EXPECTED_COMMIT" ]]; then
    echo "Unexpected commit $(git rev-parse HEAD); expected $EXPECTED_COMMIT" >&2
    exit 2
fi

set +u
source "$CANN_ROOT/set_env.sh"
set -u

g++ -std=c++17 -O2 \
    models/deepseek/v4-flash/cann_grouped_matmul_swiglu_quant_probe.cpp \
    -I"$CANN_ROOT/aarch64-linux/include" \
    -L"$CANN_ROOT/aarch64-linux/lib64" \
    -Wl,-rpath,"$CANN_ROOT/aarch64-linux/lib64" \
    -lascendcl -lopapi -lnnopbase \
    -o "$PROBE_BIN"

echo "commit=$(git rev-parse HEAD)"
echo "device=${TASK_DEVICE:-0}"
echo "mode=$PROBE_MODE"
exec "$PROBE_BIN" "${TASK_DEVICE:-0}" "$PROBE_MODE"
