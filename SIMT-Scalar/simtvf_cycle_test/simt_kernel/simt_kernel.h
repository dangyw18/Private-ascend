/*
 * The SIMT code under test.
 *
 * This file deliberately contains no host code and no result analysis.  Its
 * interface is one SIMTVF function called by cycle_count/cycle_count.asc.
 */
#pragma once

#include <cstdint>

#include "simt_api/common_functions.h"
#include "simt_api/device_functions.h"

#ifndef SIMT_THREADS
#define SIMT_THREADS 32
#endif

static_assert(SIMT_THREADS > 0 && SIMT_THREADS <= 2048,
              "SIMT_THREADS must be in [1, 2048]");

__simt_vf__ __launch_bounds__(SIMT_THREADS) inline void SimtKernel(
    __gm__ uint32_t* input, __gm__ uint32_t* output,
    __gm__ int64_t* cycles, int64_t outer_t0, uint32_t workload_iters)
{
    const uint32_t tid = static_cast<uint32_t>(threadIdx.x);

    // First observable point inside SIMTVF.  Only thread 0 records it.
    if (tid == 0) {
        // SIMT VF has a separate execution namespace.  Qualify the CCEC
        // scalar builtin explicitly instead of relying on unqualified lookup.
        cycles[0] = __cce_scalar::get_sys_cnt() - outer_t0;
    }

    // Keep the workload strictly after the entry timestamp.
    asc_syncthreads();

    uint32_t value = input[tid];
    for (uint32_t i = 0; i < workload_iters; ++i) {
        value = value * 1664525U + 1013904223U + tid;
    }
    output[tid] = value;
}
