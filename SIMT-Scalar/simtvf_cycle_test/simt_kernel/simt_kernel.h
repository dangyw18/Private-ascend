/* !
 * \file simt_kernel.h
 * \brief 被测 SIMT kernel，只包含 SIMTVF 入口计时与最小可调 workload。
 *
 * Host 启动、Vector 外层计时及结果分析统一放在 cycle_count/cycle_count.asc，
 * 避免把被测代码和测试框架混在同一个文件中。
 */
#pragma once

#include <cstdint>

#include "simt_api/common_functions.h"
#include "simt_api/device_functions.h"
#include "utils/debug/asc_time.h"

#ifndef SIMT_THREADS
#define SIMT_THREADS 32  // 默认启动一个 32-thread warp
#endif

// Ascend 950 SIMT VF 支持的线程数范围；不同线程数通过编译参数重新生成 kernel。
static_assert(SIMT_THREADS > 0 && SIMT_THREADS <= 2048,
              "SIMT_THREADS must be in [1, 2048]");

__simt_vf__ __launch_bounds__(SIMT_THREADS) inline void SimtKernel(
    __gm__ uint32_t* input, __gm__ uint32_t* output,
    __gm__ uint64_t* cycles, uint64_t outer_t0, uint32_t workload_iters)
{
    const uint32_t tid = static_cast<uint32_t>(threadIdx.x);

    // SIMT 内部能观察到的最早位置。所有线程大致同时进入，只让 thread 0 写时间戳，
    // 避免多个线程竞争同一个 cycles[0] 地址。
    if (tid == 0) {
        // 这里不能照抄外层 get_sys_cnt()：它属于 scalar 上下文，在 SIMT VF 中
        // 会编译失败或触发 CCEC 后端崩溃。clock() 是 beta.3 公开的 SIMT 时间戳接口。
        cycles[0] = __asc_simt_vf::clock() - outer_t0;
    }

    // 确保所有线程都在入口时间戳之后才开始 workload，避免计算体污染 cycles[0]。
    asc_syncthreads();

    // 可调计算体。workload_iters=0 时只保留一次 GM load 和一次 GM store；
    // 增加 workload_iters 只应明显增加 total_cycles，不应明显改变入口开销。
    uint32_t value = input[tid];
    for (uint32_t i = 0; i < workload_iters; ++i) {
        value = value * 1664525U + 1013904223U + tid;
    }
    output[tid] = value;
}
