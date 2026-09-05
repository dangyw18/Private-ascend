"""Manually scoped SimtVF version of MHC norm-weight merge forward."""

import tilelang
from tilelang import language as T


@tilelang.jit
def mhc_fn_normw_merge_fwd_simtvf(
    m: int,
    n: int,
    dtype: T.dtype = T.float32,
    threads: int = 128,
) -> tilelang.JITKernel:
    n_blk = 256

    @T.prim_func
    def kernel(
        fn: T.Tensor[(m, n), dtype],
        normw: T.Tensor[(n,), dtype],
        out_fn: T.Tensor[(m, n), dtype],
    ) -> None:
        _ = dtype
        with T.Kernel(m, T.ceildiv(n, n_blk)) as (pid_m, pid_n):
            with T.SimtVF(threads=threads):
                for i1_n in T.Parallel(n_blk):
                    i_n = pid_n * n_blk + i1_n
                    if i_n < n:
                        out_fn[pid_m, i_n] = fn[pid_m, i_n] * normw[i_n]

    return kernel
