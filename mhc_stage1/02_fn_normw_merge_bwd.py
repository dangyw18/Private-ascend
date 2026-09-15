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
    out_fn_grad: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return out_fn_grad * normw, (out_fn_grad * fn).sum(dim=0)


if __name__ == "__main__":
    torch.manual_seed(42)
    m, n = 128, 512
    fn = torch.randn((m, n), device="npu", dtype=torch.float32)
    normw = torch.randn((n,), device="npu", dtype=torch.float32)
    out_fn_grad = torch.randn((m, n), device="npu", dtype=torch.float32)

    program = mhc_fn_normw_merge_bwd(m, n)
    kernel = tilelang.compile(program, target="ascend", out_idx=[-2, -1])
    fn_grad, normw_grad = kernel(fn, normw, out_fn_grad)
    torch.npu.synchronize()
    expected_fn_grad, expected_normw_grad = ref_program(fn, normw, out_fn_grad)
    torch.testing.assert_close(fn_grad, expected_fn_grad, rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(normw_grad, expected_normw_grad, rtol=1e-5, atol=2e-5)
    print("PASS: mhc_fn_normw_merge_bwd")

    gpu_program = gpu_mhc_fn_normw_merge_bwd(m, n)
    gpu_kernel = tilelang.compile(gpu_program, target="ascend", out_idx=[-2, -1])
    # gpu_actual = gpu_kernel(fn, normw, out_fn_grad)
    # gpu_expected = ref_program(fn, normw, out_fn_grad)
    # torch.npu.synchronize()
    # torch.testing.assert_close(gpu_actual[0], gpu_expected[0], rtol=1e-5, atol=2e-5)
    # torch.testing.assert_close(gpu_actual[1], gpu_expected[1], rtol=1e-5, atol=2e-5)
    # print("PASS: gpu_mhc_fn_normw_merge_bwd")

