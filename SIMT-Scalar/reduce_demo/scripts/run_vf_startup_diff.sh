#!/usr/bin/env bash
set -euo pipefail

# 用同一份源码分别编译 1-call 和 2-call 两个版本，再根据
# T(N) = F + N*R 计算单次 VF 增量运行时间 R 与固定开销 F。
demo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${demo_dir}"
python3 scripts/gen_data.py

run_variant() {
    local vf_calls="$1"
    local build_dir="${demo_dir}/build/vf${vf_calls}"

    cmake -S "${demo_dir}" -B "${build_dir}" \
        -DCMAKE_ASC_RUN_MODE=npu \
        -DCMAKE_ASC_ARCHITECTURES=dav-3510 \
        -DREDUCE_VF_CALLS="${vf_calls}" >&2
    cmake --build "${build_dir}" >&2

    local output
    output="$("${build_dir}/demo")"
    printf '%s\n' "${output}" >&2
    printf '%s\n' "${output}" | awk '/ reduce loop:/ {print $3; exit}'
}

one_call_cycles="$(run_variant 1)"
two_call_cycles="$(run_variant 2)"

if [[ ! "${one_call_cycles}" =~ ^[0-9]+$ || ! "${two_call_cycles}" =~ ^[0-9]+$ ]]; then
    printf '无法从输出中解析 reduce loop cycles。\n' >&2
    exit 1
fi

incremental_cycles=$((two_call_cycles - one_call_cycles))
fixed_cycles=$((2 * one_call_cycles - two_call_cycles))

printf '1-call T1              : %d cycles\n' "${one_call_cycles}"
printf '2-call T2              : %d cycles\n' "${two_call_cycles}"
printf 'increment R = T2-T1    : %d cycles\n' "${incremental_cycles}"
printf 'fixed F = 2*T1-T2      : %d cycles\n' "${fixed_cycles}"
