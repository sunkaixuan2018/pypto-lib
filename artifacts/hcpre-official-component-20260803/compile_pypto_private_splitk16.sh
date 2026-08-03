#!/usr/bin/env bash
set -euo pipefail

REPO=/data/sunkaixuan/sunkaixuan_subdir/pypto-lib
PYPTO_ROOT=/data/sunkaixuan/sunkaixuan_subdir/all_pyptos/pypto-moe-ep8-20260729
EXPECTED_COMMIT=6cb467edfd58f454fce12e9ab7ee65231a8319c6

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
export PTO2_RING_DEP_POOL=16384
export PTO2_RING_TASK_WINDOW=16384
export PTO2_RING_HEAP=1073741824
export DSV4_HC_PRE_IMPL=separate

exec "$REPO/.venv/bin/python" \
    models/deepseek/v4-flash/moe.py \
    -p a2a3 \
    --ep 8 \
    -d 0,2,4,6,8,10,12,14 \
    --balanced-routing \
    --compile-only
