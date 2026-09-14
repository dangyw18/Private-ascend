import tilelang
import torch
from tilelang import language as T


def gpu_mhc_fn_normw_merge_bwd(m: int, n: int, dtype: T.dtype = T.float32) -> tilelang.JITKernel:
    n_blk = 256

    @T.prim_func
    def gpu_mhc_fn_normw_merge_bwd_(
        fn: T.Tensor[(m, n), dtype],
        normw: T.Tensor[n, dtype],
        out_fn_grad: T.Tensor[(m, n), dtype],
        fn_grad: T.Tensor[(m, n), dtype],
        normw_grad: T.Tensor[n, dtype],
    ) -> None:
        _ = dtype
        with T.Kernel(T.ceildiv(n, n_blk)) as pid_n:
            normw_frag = T.alloc_fragment(n_blk, dtype)
            T.copy(normw[pid_n * n_blk], normw_frag)

            normw_grad_frag = T.alloc_fragment(n_blk, dtype)
            T.clear(normw_grad_frag)

            for i_m in T.serial(m):
                for i1_n in T.Parallel(n_blk):
                    i_n = pid_n * n_blk + i1_n
                    if i_n < n:
                        fn_grad[i_m, i_n] += out_fn_grad[i_m, i_n] * normw_frag[i1_n]
                        normw_grad_frag[i1_n] += out_fn_grad[i_m, i_n] * fn[i_m, i_n]

            for i1_n in T.Parallel(n_blk):
                normw_grad[pid_n * n_blk + i1_n] += normw_grad_frag[i1_n]

    return gpu_mhc_fn_normw_merge_bwd_

def mhc_fn_normw_merge_bwd(m: int, n: int, dtype: T.dtype = T.float32, n_thr: int = 128) -> tilelang.JITKernel:
    n_blk = 256

    @T.prim_func
    def mhc_fn_normw_merge_bwd_(
        fn: T.Tensor[(m, n), dtype],
        normw: T.Tensor[n, dtype],
        out_fn_grad: T.Tensor[(m, n), dtype],
        fn_grad: T.Tensor[(m, n), dtype],
        normw_grad: T.Tensor[n, dtype],
    ) -> None:
        _ = dtype
        with T.Kernel(T.ceildiv(n, n_blk)) as pid_n:
            normw_shared = T.alloc_shared(n_blk, dtype)
            T.copy(normw[pid_n * n_blk], normw_shared)

            normw_grad_shared = T.alloc_shared(n_blk, dtype)
            with T.SimtVF(threads=n_thr):
                normw_grad_frag = T.alloc_fragment(n_blk, dtype)
                T.clear(normw_grad_frag)
                T.copy(normw_grad_frag, normw_grad_shared)

            for i_m in T.serial(m):
                with T.SimtVF(threads=n_thr):
                    for i1_n in T.Parallel(n_blk):
                        i_n = pid_n * n_blk + i1_n
                        if i_n < n:
                            fn_grad[i_m, i_n] += out_fn_grad[i_m, i_n] * normw_shared[i1_n]
                        normw_grad_shared[i1_n] += out_fn_grad[i_m, i_n] * fn[i_m, i_n]

            for i1_n in T.Parallel(n_blk):
                normw_grad[pid_n * n_blk + i1_n] += normw_grad_shared[i1_n]

    return mhc_fn_normw_merge_bwd_



def ref_program(
    fn: torch.Tensor,
    normw: torch.Tensor,
) -> torch.Tensor:
    return fn * normw


if __name__ == "__main__":
    torch.manual_seed(42)
    m, n = 24, 7168
    # fn = torch.randn((m, n), device="npu", dtype=torch.float32)
    # normw = torch.randn((n,), device="npu", dtype=torch.float32)

    program = gpu_mhc_fn_normw_merge_bwd(m, n)
    kernel = tilelang.compile(program, target="ascend", out_idx=-1)
    # actual = kernel(fn, normw)
    # expected = ref_program(fn, normw)
    # torch.npu.synchronize()
    # torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    # print("PASS: mhc_fn_normw_merge_bwd")
