import tilelang
import torch
from tilelang import language as T


def expand_to_mhc_fwd_simtvf(
    hidden: int,
    mhc_mult: int,
    threads: int = 128,
):
    n = T.dynamic('num_tokens')
    h = hidden
    mhc = mhc_mult

    blk_n = 32
    blk_h = 128
    num_hidden_blocks = (h + blk_h - 1) // blk_h

    @T.prim_func
    def main(
        x: T.Tensor[(n, h), T.bfloat16],
        o: T.Tensor[(n, mhc, h), T.bfloat16],
    ) -> None:
        with T.Kernel(T.ceildiv(n, blk_n) * num_hidden_blocks) as pid:
            pid_i = pid // num_hidden_blocks
            pid_j = pid % num_hidden_blocks
            if n > 0:
                xl = T.alloc_shared((blk_n, blk_h), T.bfloat16)
                T.copy(x[pid_i * blk_n, pid_j * blk_h], xl)
                with T.SimtVF(threads=threads):
                    for m in T.serial(mhc):
                        for ti, tj in T.Parallel(blk_n, blk_h):
                            i = pid_i * blk_n + ti
                            j = pid_j * blk_h + tj
                            if i < n and j < h:
                                o[i, m, j] = xl[ti, tj]

    return main


def ref_program(x: torch.Tensor, mhc_mult: int) -> torch.Tensor:
    return x.unsqueeze(-2).expand(
        *x.shape[:-1], mhc_mult, x.shape[-1]
    ).contiguous()


if __name__ == "__main__":
    torch.manual_seed(42)
    hidden = 1280
    mhc_mult = 4
    x = torch.randn((128, hidden), device="npu", dtype=torch.bfloat16)

    program = expand_to_mhc_fwd_simtvf(hidden, mhc_mult)
    kernel = tilelang.compile(program, target="ascend", out_idx=-1)
    actual = kernel(x)
    expected = ref_program(x, mhc_mult)
    torch.npu.synchronize()
    torch.testing.assert_close(actual, expected)
    print("PASS: expand_to_mhc_fwd")
