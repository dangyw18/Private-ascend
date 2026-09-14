import math
import torch
import tilelang
from tilelang import language as T


def _mhc_pre_apply_mix_fwd_choice1(
    mhc_mult: int,
    hidden: int,
    n_thr: int = 128,
    h_blk: int = 1024,
):
    n = T.dynamic('n')
    h = hidden
    mhc = mhc_mult
    h_blk = math.gcd(h_blk, hidden)

    @T.prim_func
    def main(
        x: T.Tensor[(n, mhc, h), T.bfloat16],
        mix: T.Tensor[(n, mhc), T.float32],
        o: T.Tensor[(n, h), T.bfloat16],
    ) -> None:
        with T.Kernel(n) as pid_n:
            mix_shared = T.alloc_shared(mhc, T.float32)
            T.copy(mix[pid_n, 0], mix_shared)

            for i0_h in T.Pipelined(h // h_blk, num_stages=2):
                xs = T.alloc_shared((mhc, h_blk), T.bfloat16)
                T.copy(x[pid_n, 0, i0_h * h_blk], xs)

                os = T.alloc_shared(h_blk, T.float32)
                with T.SimtVF(threads=n_thr):
                    ol = T.alloc_fragment(h_blk, T.float32)
                    T.copy(os, ol)
                    T.clear(ol)
                    T.copy(ol, os)

                for i_mhc in T.serial(mhc):
                    with T.SimtVF(threads=n_thr):
                        xl = T.alloc_fragment((mhc, h_blk), T.float32)
                        ol = T.alloc_fragment(h_blk, T.float32)
                        mixl = T.alloc_fragment(mhc, T.float32)
                        T.copy(os, ol)
                        T.copy(xs, xl)
                        T.copy(mix_shared, mixl)
                        for i1_h in T.Parallel(h_blk):
                            ol[i1_h] += mixl[i_mhc] * xl[i_mhc, i1_h]
                        T.copy(ol, os)

                with T.SimtVF(threads=n_thr):
                    T.copy(os, o[pid_n, i0_h * h_blk])

    return main


def _mhc_pre_apply_mix_fwd_choice2(
    mhc_mult: int,
    hidden: int,
    n_thr: int = 128,
    h_blk: int = 1024,
):
    n = T.dynamic('n')
    h = hidden
    mhc = mhc_mult
    h_blk = math.gcd(h_blk, hidden)

    @T.prim_func
    def main(
        x: T.Tensor[(n, mhc, h), T.bfloat16],
        mix: T.Tensor[(n, mhc), T.float32],
        o: T.Tensor[(n, h), T.bfloat16],
    ) -> None:
        with T.Kernel(n) as pid_n:
            mix_shared = T.alloc_shared(mhc, T.float32)
            T.copy(mix[pid_n, 0], mix_shared)

            for i0_h in T.Pipelined(h // h_blk, num_stages=2):
                xs = T.alloc_shared((mhc, h_blk), T.bfloat16)
                T.copy(x[pid_n, 0, i0_h * h_blk], xs)

                os = T.alloc_shared(h_blk, T.float32)
                with T.SimtVF(threads=n_thr):
                    T.clear(os)

                for i_mhc in T.serial(mhc):
                    with T.SimtVF(threads=n_thr):
                        xl = T.alloc_fragment((mhc, h_blk), T.float32)
                        mixl = T.alloc_fragment(mhc, T.float32)
                        T.copy(os, ol)
                        T.copy(xs, xl)
                        T.copy(mix_shared, mixl)
                        for i1_h in T.Parallel(h_blk):
                            os[i1_h] += mixl[i_mhc] * xl[i_mhc, i1_h]

                with T.SimtVF(threads=n_thr):
                    T.copy(os, o[pid_n, i0_h * h_blk])

    return main

def gpu_mhc_pre_apply_mix_fwd(
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
    def gpu_mhc_pre_apply_mix_fwd_kernel(
        x: T.Tensor[(n, mhc, h), T.bfloat16],
        mix: T.Tensor[(n, mhc), T.float32],
        o: T.Tensor[(n, h), T.bfloat16],
    ) -> None:
        with T.Kernel(n, threads=n_thr) as pid_n:
            mixl = T.alloc_fragment(mhc, T.float32)
            T.copy(mix[pid_n, 0], mixl)

            for i0_h in T.Pipelined(h // h_blk, num_stages=2):
                xs = T.alloc_shared((mhc, h_blk), T.bfloat16)
                xl = T.alloc_fragment((mhc, h_blk), T.float32)
                T.copy(x[pid_n, 0, i0_h * h_blk], xs, disable_tma=True)
                T.copy(xs, xl, disable_tma=True)

                os = T.alloc_shared(h_blk, T.bfloat16)
                ol = T.alloc_fragment(h_blk, T.float32)
                T.clear(ol)

                for i_mhc in T.serial(mhc):
                    for i1_h in T.Parallel(h_blk):
                        ol[i1_h] += mixl[i_mhc] * xl[i_mhc, i1_h]

                T.copy(ol, os, disable_tma=True)
                T.copy(os, o[pid_n, i0_h * h_blk], disable_tma=True)

    return gpu_mhc_pre_apply_mix_fwd_kernel

def ref_program(
    x: torch.Tensor,          # (n, mhc, h) bfloat16
    mix: torch.Tensor,        # (n, mhc) float32
) -> torch.Tensor:
    # 先将 x 转为 float32 计算，再转回 bfloat16 以匹配内核输出
    return (mix.unsqueeze(-1).float() * x.float()).sum(dim=1).to(torch.bfloat16)

if __name__ == "__main__":
    torch.manual_seed(42)

    # 参数
    num_tokens = 128
    mhc_mult = 4
    hidden = 1280
    n_thr = 128
    h_blk = 1024   

    # x = torch.randn((num_tokens, mhc_mult, hidden), device="npu", dtype=torch.bfloat16)
    # mix = torch.randn((num_tokens, mhc_mult), device="npu", dtype=torch.float32)

    # program = _mhc_pre_apply_mix_fwd(mhc_mult, hidden, n_thr, h_blk)
    # kernel = tilelang.compile(
    #     program,
    #     target="ascend",
    #     out_idx=-1,
    # )

    # actual = kernel(x, mix)
    # reference = ref_program(x, mix)

    # torch.npu.synchronize()
    # print("actual:", actual)
    # print("reference:", reference)
    # torch.testing.assert_close(actual, reference, rtol=1e-2, atol=1e-2)
    
    # print("PASS: _mhc_pre_apply_mix_fwd (bfloat16 test)")

    gpu_program = gpu_mhc_pre_apply_mix_fwd(mhc_mult, hidden, n_thr, h_blk)
    gpu_kernel = tilelang.compile(
        gpu_program,
        target="ascend",
        out_idx=-1,
    )