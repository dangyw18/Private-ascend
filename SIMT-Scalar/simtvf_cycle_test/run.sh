#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${root_dir}/build"
threads="${SIMT_THREADS:-32}"

cmake -S "${root_dir}" -B "${build_dir}" \
  -DCMAKE_ASC_RUN_MODE=npu \
  -DCMAKE_ASC_ARCHITECTURES=dav-3510 \
  -DSIMT_THREADS="${threads}"
cmake --build "${build_dir}" -j

if [[ "${1:-}" == "--build-only" ]]; then
  exit 0
fi

exec "${build_dir}/simtvf_cycle_count" "$@"
