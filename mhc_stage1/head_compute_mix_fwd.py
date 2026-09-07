import tilelang
import torch
from tilelang import language as T


def _mhc_head_compute_mix_fwd(
    mhc_mult: int,
    mhc_pre_eps: float,
    token_block_size: int,
    threads: int = 128,
):
    num_tokens = T.dynamic('num_tokens')

    @T.prim_func
    def main(
        # Input
        input_mix: T.Tensor[(num_tokens, mhc_mult), T.float32],
        mhc_scale: T.Tensor[(1,), T.float32],
        mhc_base: T.Tensor[(mhc_mult,), T.float32],
        # Output
        output_mix: T.Tensor[(num_tokens, mhc_mult), T.float32],
    ) -> None:
        with T.Kernel(T.ceildiv(num_tokens, token_block_size)) as pid:
            with T.SimtVF(threads=threads):
                for i1, j in T.Parallel(token_block_size, mhc_mult):
                    i = pid * token_block_size + i1
                    if i < num_tokens:
                        output_mix[i, j] = T.sigmoid(input_mix[i, j] * mhc_scale[0] + mhc_base[j]) + mhc_pre_eps

    return main


def ref_program(
    input_mix: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    mhc_pre_eps: float = 1e-6,
) -> torch.Tensor:
    return torch.sigmoid(input_mix * mhc_scale[0] + mhc_base) + mhc_pre_eps


if __name__ == "__main__":
    torch.manual_seed(42)
    num_tokens, mhc_mult = 128, 4
    mhc_pre_eps = 1e-6
    token_block_size = 32
    input_mix = torch.randn((num_tokens, mhc_mult), device="npu", dtype=torch.float32)
    mhc_scale = torch.randn((1,), device="npu", dtype=torch.float32)
    mhc_base = torch.randn((mhc_mult,), device="npu", dtype=torch.float32)

    program = _mhc_head_compute_mix_fwd(mhc_mult, mhc_pre_eps, token_block_size)
    kernel = tilelang.compile(program, target="ascend", out_idx=-1)
    actual = kernel(input_mix, mhc_scale, mhc_base)
    expected = ref_program(input_mix, mhc_scale, mhc_base, mhc_pre_eps)
    torch.npu.synchronize()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-5)
    print("PASS: mhc_head_compute_mix_fwd")
