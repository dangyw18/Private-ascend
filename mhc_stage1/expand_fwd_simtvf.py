"""Manually scoped SimtVF version of MHC expand forward."""

import tilelang
from tilelang import language as T


@tilelang.jit
def expand_to_mhc_fwd_simtvf(
    hidden: int,
    mhc_mult: int,
    threads: int = 128,
) -> tilelang.JITKernel:
    num_tokens = T.dynamic("num_tokens")
    blk_n = 32
    blk_h = 128

    @T.prim_func
    def kernel(
        x: T.Tensor[(num_tokens, hidden), T.bfloat16],
        out: T.Tensor[(num_tokens, mhc_mult, hidden), T.bfloat16],
    ) -> None:
        with T.Kernel(
            T.ceildiv(num_tokens, blk_n), T.ceildiv(hidden, blk_h)
        ) as (pid_i, pid_j):
            if num_tokens > 0:
                with T.SimtVF(threads=threads):
                    x_local = T.alloc_fragment((blk_n, blk_h), T.bfloat16)

                    # Preserve the source program's direct GM -> fragment copy.
                    T.copy(x[pid_i * blk_n, pid_j * blk_h], x_local)

                    for m in T.serial(mhc_mult):
                        for ti, tj in T.Parallel(blk_n, blk_h):
                            i = pid_i * blk_n + ti
                            j = pid_j * blk_h + tj
                            if i < num_tokens and j < hidden:
                                out[i, m, j] = x_local[ti, tj]

    return kernel
