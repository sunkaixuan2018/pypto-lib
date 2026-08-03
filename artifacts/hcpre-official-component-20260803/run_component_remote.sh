#!/usr/bin/env bash
set -euo pipefail

CANN_ROOT=/usr/local/Ascend/cann-9.0.0
CUSTOM_OP_ROOT=/data/sunkaixuan/sunkaixuan_subdir/all_libs/vllm-ascend-hcpre-custom-20260803
PROBE_BIN=/data/sunkaixuan/skx_log_output/cann_hc_pre_probe_39184fa

if [[ -z "${TASK_DEVICE:-}" ]]; then
    echo "TASK_DEVICE was not assigned by task-submit" >&2
    exit 2
fi

set +u
source "$CANN_ROOT/set_env.sh"
source "$CUSTOM_OP_ROOT/vendors/custom_transformer/bin/set_env.bash"
set -u

echo "device=$TASK_DEVICE"
echo "probe=$PROBE_BIN"
"$PROBE_BIN" "$TASK_DEVICE"
