#!/usr/bin/env bash
set -euo pipefail

OUT=/data/sunkaixuan/skx_log_output/hcpre_official_component_20260803
mkdir -p "$OUT"

save_log() {
    local task_id=$1
    local name=$2
    /usr/local/bin/task-submit --log "$task_id" >"$OUT/$name.log"
}

save_log task_20260803_064633_16691082755 official_component
save_log task_20260803_064954_17693384878 pypto_linear_split16_ep8
save_log task_20260803_065410_188316014671 pypto_linear16_rms16_ep8
save_log task_20260803_065605_194474631344 pypto_linear16_rms16_component
save_log task_20260803_065737_198686422071 pypto_retained_component
save_log task_20260803_070642_22628377650 pypto_private_split16_component
save_log task_20260803_070723_228024427645 pypto_private_split16_ep8_first
save_log task_20260803_070908_233183330361 pypto_retained_ep8_interrupted
save_log task_20260803_071022_236779727861 pypto_retained_ep8_clean
save_log task_20260803_071131_23999389208 pypto_private_split16_ep8_second
save_log task_20260803_071344_24701053335 pypto_private_split8_component

sha256sum "$OUT"/*.log >"$OUT/SHA256SUMS"
