#!/usr/bin/env bash
set -euo pipefail

# 单配置构建入口。所有被测参数均在编译期固定，避免运行时分支污染 VF。
root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
case_id="${SIMT_CASE:-1}"
threads="${SIMT_THREADS:-32}"
data_size="${DATA_SIZE:-256}"
vf_calls="${VF_CALLS:-1}"
build_dir="${root_dir}/build/single_case${case_id}_m${threads}_n${data_size}_vf${vf_calls}"

cmake -S "${root_dir}" -B "${build_dir}" \
    -DCMAKE_ASC_RUN_MODE=npu \
    -DCMAKE_ASC_ARCHITECTURES=dav-3510 \
    -DSIMT_CASE="${case_id}" \
    -DSIMT_THREADS="${threads}" \
    -DDATA_SIZE="${data_size}" \
    -DVF_CALLS="${vf_calls}"
cmake --build "${build_dir}" -j

if [[ "${1:-}" == "--build-only" ]]; then
    exit 0
fi

exec "${build_dir}/simtvf_cycle_count" "$@"
