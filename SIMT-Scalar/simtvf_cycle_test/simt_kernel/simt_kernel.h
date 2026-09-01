/* !
 * \file simt_kernel.h
 * \brief 两个用于 SIMTVF 固定开销差分实验的 AscendC SIMT kernel。
 *
 * M（SIMT_THREADS）控制启动线程数，N（DATA_SIZE）控制处理元素数。
 * 每个线程使用 thread-stride loop 处理 [0, N) 中的元素，对应：
 *
 *   with T.SimtVF(threads=M):
 *       for i in T.Parallel(N):
 *           ...
 */
#pragma once

#include <cstdint>

#include "simt_api/common_functions.h"
#include "simt_api/device_functions.h"

#ifndef SIMT_THREADS
#define SIMT_THREADS 32
#endif

#ifndef DATA_SIZE
#define DATA_SIZE 256
#endif

static_assert(SIMT_THREADS > 0 && SIMT_THREADS <= 2048,
              "SIMT_THREADS must be in [1, 2048]");
static_assert(DATA_SIZE > 0, "DATA_SIZE must be positive");

// Case 1：短指令体，每个元素执行 2 次 GM load、1 次浮点加法和 1 次 GM store。
__simt_vf__ __launch_bounds__(SIMT_THREADS) inline void SimtCase1(
    __gm__ float* a, __gm__ float* b, __gm__ float* c)
{
    const uint32_t index = static_cast<uint32_t>(threadIdx.x);
    const uint32_t stride = static_cast<uint32_t>(blockDim.x);

    for (uint32_t i = index; i < DATA_SIZE; i += stride) {
        a[i] = b[i] + c[i];
    }
}

// Case 2：长依赖链。访存地址、线程组织和循环范围与 Case 1 相同，
// 只增加 2 次乘法和 3 次加法，用于观察 VF 指令长度是否改变固定开销。
__simt_vf__ __launch_bounds__(SIMT_THREADS) inline void SimtCase2(
    __gm__ float* a, __gm__ float* b, __gm__ float* c)
{
    const uint32_t index = static_cast<uint32_t>(threadIdx.x);
    const uint32_t stride = static_cast<uint32_t>(blockDim.x);

    for (uint32_t i = index; i < DATA_SIZE; i += stride) {
        const float b_value = b[i];
        const float c_value = c[i];
        float value = b_value + c_value;
        value = value * value;
        value = b_value + value;
        value = value * value;
        value = c_value + value;
        a[i] = value;
    }
}
