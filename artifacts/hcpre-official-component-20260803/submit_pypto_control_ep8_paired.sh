#!/usr/bin/env bash
set -euo pipefail

/usr/local/bin/task-submit \
    --device 0,2,4,6,8,10,12,14 \
    --max-time 180 \
    --run "bash /data/sunkaixuan/codex_sh/run_hcpre_control_ep8_paired_20260803.sh"
