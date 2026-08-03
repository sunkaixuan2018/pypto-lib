#!/usr/bin/env bash
set -euo pipefail

/usr/local/bin/task-submit \
    --device auto \
    --max-time 120 \
    --run "bash /data/sunkaixuan/codex_sh/run_hcpre_splitk16_rms16_component_20260803.sh"
