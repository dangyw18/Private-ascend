import tilelang
import torch
from tilelang import language as T

def expand_to_mhc_bwd_tl(hidden: int, mhc_mult: int) -> tilelang.JITKernel:
    n = T.dynamic('num_tokens')
    h = hidden
    mhc = mhc_mult

    blk_n = 32
    blk_h = 128

    @T.prim_func
    def expand_to_mhc_bwd_kernel(
        o_grad: T.Tensor[(n, mhc, h), T.bfloat16],
        x_grad: T.Tensor[(n, h), T.bfloat16],
    ) -> None:
        with T.Kernel(T.ceildiv(n, blk_n), T.ceildiv(h, blk_h)) as (pid_i, pid_j):
            if n > 0:
                xgl = T.alloc_fragment((blk_n, blk_h), T.float32)
                T.fill(xgl, 0)
                for m in T.serial(mhc):
                    for ti, tj in T.Parallel(blk_n, blk_h):
                        i = pid_i * blk_n + ti
                        j = pid_j * blk_h + tj
                        if i < n and j < h:
                            xgl[ti, tj] += o_grad[i, m, j]
                T.copy(xgl, x_grad[pid_i * blk_n, pid_j * blk_h])

    return expand_to_mhc_bwd_kernel




def ref_program(o_grad: torch.Tensor) -> torch.Tensor:
    return o_grad.float().sum(dim=1).to(torch.bfloat16)


if __name__ == "__main__":
    torch.manual_seed(42)
    num_tokens, hidden, mhc_mult = 128, 256, 4
    o_grad = torch.randn((num_tokens, mhc_mult, hidden), device="npu", dtype=torch.bfloat16)

    program = expand_to_mhc_bwd_tl(hidden, mhc_mult)
    kernel = tilelang.compile(program, target="ascend", out_idx=-1)
    x_grad = kernel(o_grad)
    torch.npu.synchronize()
    expected = ref_program(o_grad)
    torch.testing.assert_close(x_grad, expected, rtol=1e-5, atol=2e-5)
    print("PASS: expand_to_mhc_bwd")