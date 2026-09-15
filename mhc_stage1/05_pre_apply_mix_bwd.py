import math

import tilelang
import torch
from tilelang import language as T


def _mhc_pre_apply_mix_bwd(
    mhc_mult: int,
    hidden: int,
    n_thr: int = 128,
    h_blk: int = 1024,
) -> tilelang.JITKernel:
    n = T.dynamic('n')
    h = hidden
    mhc = mhc_mult

    h_blk = math.gcd(h_blk, hidden)

    @T.prim_func
    def _mhc_pre_apply_mix_bwd_kernel(
        o_grad: T.Tensor[(n, h), T.bfloat16],
        x: T.Tensor[(n, mhc, h), T.bfloat16],
        mix: T.Tensor[(n, mhc), T.float32],
        x_grad: T.Tensor[(n, mhc, h), T.bfloat16],
        mix_grad: T.Tensor[(n, mhc), T.float32],
    ) -> None:
        with T.Kernel(n, threads=n_thr) as pid_n:
            mixl = T.alloc_fragment(mhc, T.float32)
            T.copy(mix[pid_n, 0], mixl, disable_tma=True)

            mgl = T.alloc_reducer(mhc, T.float32, replication='all')
            T.fill(mgl, 0)

            for i0_h in T.Pipelined(h // h_blk, num_stages=2):
                ogs = T.alloc_shared(h_blk, T.bfloat16)
                ogl = T.alloc_fragment(h_blk, T.float32)
                T.copy(o_grad[pid_n, i0_h * h_blk], ogs, disable_tma=True)
                T.copy(ogs, ogl, disable_tma=True)

                xs = T.alloc_shared((mhc, h_blk), T.bfloat16)
                xl = T.alloc_fragment((mhc, h_blk), T.float32)
                T.copy(x[pid_n, 0, i0_h * h_blk], xs, disable_tma=True)
                T.copy(xs, xl, disable_tma=True)

                xgs = T.alloc_shared((mhc, h_blk), T.bfloat16)
                xgl = T.alloc_fragment((mhc, h_blk), T.float32)
                T.copy(x_grad[pid_n, 0, i0_h * h_blk], xgs, disable_tma=True)
                T.copy(xgs, xgl, disable_tma=True)

                for i_mhc, i1_h in T.Parallel(mhc, h_blk):
                    mgl[i_mhc] += ogl[i1_h] * xl[i_mhc, i1_h]
                    xgl[i_mhc, i1_h] += mixl[i_mhc] * ogl[i1_h]

                T.copy(xgl, x_grad[pid_n, 0, i0_h * h_blk], disable_tma=True)

            T.finalize_reducer(mgl)
            T.copy(mgl, mix_grad[pid_n, 0], disable_tma=True)

    return _mhc_pre_apply_mix_bwd_kernel

def ref_program(
    o_grad: torch.Tensor,
    x: torch.Tensor,
    mix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_grad = (mix.unsqueeze(-1) * o_grad.float().unsqueeze(1)).to(torch.bfloat16)
    mix_grad = torch.einsum('bh,bmh->bm', o_grad.float(), x.float())
    return x_grad, mix_grad

if __name__ == "__main__":
    torch.manual_seed(42)
    num_tokens, mhc_mult, hidden = 128, 4, 256
    o_grad = torch.randn((num_tokens, hidden), device="npu", dtype=torch.bfloat16)
    x = torch.randn((num_tokens, mhc_mult, hidden), device="npu", dtype=torch.bfloat16)
    mix = torch.randn((num_tokens, mhc_mult), device="npu", dtype=torch.float32)

    program = _mhc_pre_apply_mix_bwd(mhc_mult, hidden)
    kernel = tilelang.compile(program, target="ascend", out_idx=[-2, -1])
    x_grad, mix_grad = kernel(o_grad, x, mix)
    torch.npu.synchronize()
    expected_x_grad, expected_mix_grad = ref_program(o_grad, x, mix)
    torch.testing.assert_close(x_grad, expected_x_grad, rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(mix_grad, expected_mix_grad, rtol=1e-5, atol=2e-5)
    print("PASS: mhc_pre_apply_mix_bwd")