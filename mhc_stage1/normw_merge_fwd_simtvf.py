import tilelang
import torch
from tilelang import language as T


def mhc_fn_normw_merge_fwd_simtvf(
    m: int,
    n: int,
    dtype: T.dtype = T.float32,
    threads: int = 128,
):
    n_blk = 256
    num_n_blocks = (n + n_blk - 1) // n_blk

    @T.prim_func
    def main(
        fn: T.Tensor[(m, n), dtype],
        normw: T.Tensor[(n,), dtype],
        out_fn: T.Tensor[(m, n), dtype],
    ) -> None:
        _ = dtype
        with T.Kernel(m * num_n_blocks) as pid:
            pid_m = pid // num_n_blocks
            pid_n = pid % num_n_blocks
            with T.SimtVF(threads=threads):
                for i1_n in T.Parallel(n_blk):
                    i_n = pid_n * n_blk + i1_n
                    if i_n < n:
                        out_fn[pid_m, i_n] = fn[pid_m, i_n] * normw[i_n]

    return main


def ref_program(
    fn: torch.Tensor,
    normw: torch.Tensor,
) -> torch.Tensor:
    return fn * normw


if __name__ == "__main__":
    torch.manual_seed(42)
    m, n = 24, 7168
    fn = torch.randn((m, n), device="npu", dtype=torch.float32)
    normw = torch.randn((n,), device="npu", dtype=torch.float32)

    program = mhc_fn_normw_merge_fwd_simtvf(m, n)
    kernel = tilelang.compile(program, target="ascend", out_idx=-1)
    actual = kernel(fn, normw)
    expected = ref_program(fn, normw)
    torch.npu.synchronize()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    print("PASS: mhc_fn_normw_merge_fwd")
