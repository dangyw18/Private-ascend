#!/usr/bin/env bash
set -euo pipefail

# 默认矩阵：2 cases * 4 个 M * 2 个 N = 16 组结果。
# 每组分别编译 VF_CALLS=1/2，因此实际生成并运行 32 个二进制。
root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
device_id="${DEVICE_ID:-0}"
samples="${SAMPLES:-21}"
read -r -a case_values <<< "${CASE_VALUES:-1 2}"
read -r -a thread_values <<< "${THREAD_VALUES:-32 64 128 256}"
read -r -a data_values <<< "${DATA_VALUES:-256 1024}"

if [[ ! "${device_id}" =~ ^[0-9]+$ || ! "${samples}" =~ ^[1-9][0-9]*$ ]]; then
    printf 'DEVICE_ID 必须是非负整数，SAMPLES 必须是正整数。\n' >&2
    exit 1
fi

result_dir="${root_dir}/results"
summary_file="${result_dir}/simtvf_startup_diff_summary.csv"
mkdir -p "${result_dir}"
printf 'case,M,N,T1,T2,increment_R,fixed_F\n' > "${summary_file}"

median_cycles() {
    sort -n | awk '
        { values[NR] = $1 }
        END {
            if (NR == 0) {
                exit 1
            }
            if (NR % 2 == 1) {
                print values[(NR + 1) / 2]
            } else {
                print int((values[NR / 2] + values[NR / 2 + 1]) / 2)
            }
        }
    '
}

run_variant() {
    local case_id="$1"
    local threads="$2"
    local data_size="$3"
    local vf_calls="$4"
    local tag="case${case_id}_m${threads}_n${data_size}_vf${vf_calls}"
    local build_dir="${root_dir}/build/matrix/${tag}"
    local raw_file="${result_dir}/${tag}.csv"

    printf '编译并运行 %s\n' "${tag}" >&2
    cmake -S "${root_dir}" -B "${build_dir}" \
        -DCMAKE_ASC_RUN_MODE=npu \
        -DCMAKE_ASC_ARCHITECTURES=dav-3510 \
        -DSIMT_CASE="${case_id}" \
        -DSIMT_THREADS="${threads}" \
        -DDATA_SIZE="${data_size}" \
        -DVF_CALLS="${vf_calls}" >&2
    cmake --build "${build_dir}" -j >&2

    local output
    output="$("${build_dir}/simtvf_cycle_count" "${device_id}" "${samples}")"
    printf '%s\n' "${output}" > "${raw_file}"

    local valid_samples
    valid_samples="$(printf '%s\n' "${output}" | awk -F, '
        $1 ~ /^[0-9]+$/ && $6 ~ /^[0-9]+$/ { count++ }
        END { print count + 0 }
    ')"
    if [[ "${valid_samples}" -ne "${samples}" ]]; then
        printf '%s 期望 %s 个样本，实际只解析到 %s 个，原始结果：%s\n' \
            "${tag}" "${samples}" "${valid_samples}" "${raw_file}" >&2
        exit 1
    fi

    if printf '%s\n' "${output}" | awk -F, '
        $1 ~ /^[0-9]+$/ && $7 != "1" { bad = 1 }
        END { exit bad ? 1 : 0 }
    '; then
        :
    else
        printf '%s 存在结果错误或非法 cycle，原始结果：%s\n' \
            "${tag}" "${raw_file}" >&2
        exit 1
    fi

    printf '%s\n' "${output}" | awk -F, '
        $1 ~ /^[0-9]+$/ { print $6 }
    ' | median_cycles
}

for case_id in "${case_values[@]}"; do
    for threads in "${thread_values[@]}"; do
        for data_size in "${data_values[@]}"; do
            t1="$(run_variant "${case_id}" "${threads}" "${data_size}" 1)"
            t2="$(run_variant "${case_id}" "${threads}" "${data_size}" 2)"

            if [[ ! "${t1}" =~ ^[0-9]+$ || ! "${t2}" =~ ^[0-9]+$ ]]; then
                printf '无法解析 case=%s M=%s N=%s 的 T1/T2。\n' \
                    "${case_id}" "${threads}" "${data_size}" >&2
                exit 1
            fi

            incremental=$((t2 - t1))
            fixed=$((2 * t1 - t2))

            printf '\ncase %s, M=%s, N=%s\n' "${case_id}" "${threads}" "${data_size}"
            printf '1-call T1              : %d cycles\n' "${t1}"
            printf '2-call T2              : %d cycles\n' "${t2}"
            printf 'increment R = T2-T1    : %d cycles\n' "${incremental}"
            printf 'fixed F = 2*T1-T2      : %d cycles\n' "${fixed}"

            printf '%s,%s,%s,%s,%s,%s,%s\n' \
                "${case_id}" "${threads}" "${data_size}" "${t1}" "${t2}" \
                "${incremental}" "${fixed}" >> "${summary_file}"
        done
    done
done

printf '\n汇总（每个 T 为 %s 次采样的中位数）\n' "${samples}"
awk -F, '
    NR == 1 {
        printf "%-6s %-6s %-8s %-10s %-10s %-12s %-12s\n", \
               $1, $2, $3, $4, $5, $6, $7
        next
    }
    {
        printf "%-6s %-6s %-8s %-10s %-10s %-12s %-12s\n", \
               $1, $2, $3, $4, $5, $6, $7
    }
' "${summary_file}"
printf '\nCSV 汇总：%s\n' "${summary_file}"
