#!/usr/bin/env bash
set -euo pipefail

REPO=/data/sunkaixuan/sunkaixuan_subdir/pypto-lib
PYPTO_ROOT=/data/sunkaixuan/sunkaixuan_subdir/all_pyptos/pypto-moe-ep8-20260729
EXPECTED_COMMIT=a5174270fb2e65f71694180dc66a805d76c4d412
EXPECTED_DEVICES=0,2,4,6,8,10,12,14

if [[ "${TASK_DEVICE:-}" != "$EXPECTED_DEVICES" ]]; then
    echo "Unexpected TASK_DEVICE=${TASK_DEVICE:-unset}; expected $EXPECTED_DEVICES" >&2
    exit 2
fi

cd "$REPO"
if [[ "$(git rev-parse HEAD)" != "$EXPECTED_COMMIT" ]]; then
    echo "Unexpected commit $(git rev-parse HEAD); expected $EXPECTED_COMMIT" >&2
    exit 2
fi

set +u
source /usr/local/Ascend/cann-9.0.0/set_env.sh
set -u
export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO"
export PTOAS_ROOT=/usr/local/ptoas/0.48
export PTO_ISA_ROOT="$PYPTO_ROOT/runtime/build/pto-isa"
export SIMPLER_PTO_ISA_COMMIT=83d01313d9bfc247c4b7c8bcf969d1019f0d106f
export PATH="$PTOAS_ROOT/bin:$PATH"
export PYPTO_LOG_LEVEL=error
export PYPTO_WARNING_LEVEL=none
export PYPTO_RUNTIME_LOG=error
export PYPTO_BENCH=1
export PYPTO_BENCH_ROUNDS=8
export PYPTO_BENCH_WARMUP=2
export PYPTO_BENCH_RAW=1
export PTO2_RING_DEP_POOL=16384
export PTO2_RING_TASK_WINDOW=16384
export PTO2_RING_HEAP=1073741824
export DSV4_HC_PRE_IMPL=separate

echo "source_commit=$EXPECTED_COMMIT"
echo "devices=$TASK_DEVICE"
echo "hc_pre_linear_ok=16"
echo "bench_rounds=$PYPTO_BENCH_ROUNDS warmup=$PYPTO_BENCH_WARMUP raw=$PYPTO_BENCH_RAW"
exec "$REPO/.venv/bin/python" \
    models/deepseek/v4-flash/moe.py \
    -p a2a3 \
    --ep 8 \
    -d "$TASK_DEVICE" \
    --balanced-routing
