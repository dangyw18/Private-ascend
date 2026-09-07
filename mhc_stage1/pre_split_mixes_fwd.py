import tilelang
import tilelang.language as T
import torch


THREADS = 128
TOKEN_BLOCK_SIZE = 32


def pre_split_mixes_fwd_program(
    num_tokens: int,
    mhc_mult: int,
    mhc_post_mult_value: float,
    mhc_pre_eps: float,
    token_block_size: int = TOKEN_BLOCK_SIZE,
    threads: int = THREADS,
):
    mhc_mult2 = mhc_mult * mhc_mult
    mhc_mult3 = mhc_mult * 2 + mhc_mult2

    @T.prim_func
    def main(
        input_mixes: T.Buffer((num_tokens, mhc_mult3), "float32"),
        mhc_scale: T.Buffer((3,), "float32"),
        mhc_base: T.Buffer((mhc_mult3,), "float32"),
        pre_layer_mix: T.Buffer((num_tokens, mhc_mult), "float32"),
        post_layer_mix: T.Buffer((num_tokens, mhc_mult), "float32"),
        comb_res_mix: T.Buffer((num_tokens, mhc_mult2), "float32"),
    ):
        with T.Kernel(num_tokens // token_block_size) as pid:
            input_mixes_shared = T.alloc_shared(
                (token_block_size, mhc_mult3), T.float32
            )
            pre_layer_mix_shared = T.alloc_shared(
                (token_block_size, mhc_mult), T.float32
            )
            post_layer_mix_shared = T.alloc_shared(
                (token_block_size, mhc_mult), T.float32
            )
            comb_res_mix_shared = T.alloc_shared(
                (token_block_size, mhc_mult2), T.float32
            )

            # MTE/DMA copies remain outside SimtVF.
            T.copy(input_mixes[pid * token_block_size, 0], input_mixes_shared)

            with T.SimtVF(threads=threads):
                input_mixes_frag = T.alloc_fragment(
                    (token_block_size, mhc_mult3), T.float32
                )
                pre_layer_mix_frag = T.alloc_fragment(
                    (token_block_size, mhc_mult), T.float32
                )
                post_layer_mix_frag = T.alloc_fragment(
                    (token_block_size, mhc_mult), T.float32
                )
                comb_res_mix_frag = T.alloc_fragment(
                    (token_block_size, mhc_mult2), T.float32
                )

                # Fragment normal copies belong to the same VF as their users.
                T.copy(input_mixes_shared, input_mixes_frag)

                for i, j in T.Parallel(token_block_size, mhc_mult):
                    pre_layer_mix_frag[i, j] = (
                        T.sigmoid(
                            input_mixes_frag[i, j] * mhc_scale[0] + mhc_base[j]
                        )
                        + mhc_pre_eps
                    )
                for i, j in T.Parallel(token_block_size, mhc_mult):
                    post_layer_mix_frag[i, j] = (
                        T.sigmoid(
                            input_mixes_frag[i, j + mhc_mult] * mhc_scale[1]
                            + mhc_base[j + mhc_mult]
                        )
                        * mhc_post_mult_value
                    )
                for i, j in T.Parallel(token_block_size, mhc_mult2):
                    comb_res_mix_frag[i, j] = (
                        input_mixes_frag[i, j + mhc_mult * 2] * mhc_scale[2]
                        + mhc_base[j + mhc_mult * 2]
                    )

                T.copy(pre_layer_mix_frag, pre_layer_mix_shared)
                T.copy(post_layer_mix_frag, post_layer_mix_shared)
                T.copy(comb_res_mix_frag, comb_res_mix_shared)

            T.copy(pre_layer_mix_shared, pre_layer_mix[pid * token_block_size, 0])
            T.copy(
                post_layer_mix_shared, post_layer_mix[pid * token_block_size, 0]
            )
            T.copy(comb_res_mix_shared, comb_res_mix[pid * token_block_size, 0])

    return main


def ref_program(
    input_mixes: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    mhc_mult: int = 4,
    mhc_post_mult_value: float = 2.0,
    mhc_pre_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mhc_mult2 = mhc_mult * mhc_mult
    pre = (
        torch.sigmoid(
            input_mixes[..., :mhc_mult] * mhc_scale[0]
            + mhc_base[:mhc_mult]
        )
        + mhc_pre_eps
    )
    post = (
        torch.sigmoid(
            input_mixes[..., mhc_mult : 2 * mhc_mult] * mhc_scale[1]
            + mhc_base[mhc_mult : 2 * mhc_mult]
        )
        * mhc_post_mult_value
    )
    comb = (
        input_mixes[..., 2 * mhc_mult : 2 * mhc_mult + mhc_mult2]
        * mhc_scale[2]
        + mhc_base[2 * mhc_mult : 2 * mhc_mult + mhc_mult2]
    )
    return pre, post, comb


if __name__ == "__main__":
    torch.manual_seed(42)
    num_tokens = 128
    mhc_mult = 4
    mhc_post_mult_value = 2.0
    mhc_pre_eps = 1e-6
    mhc_mult3 = mhc_mult * 2 + mhc_mult * mhc_mult
    input_mixes = torch.randn(
        (num_tokens, mhc_mult3), device="npu", dtype=torch.float32
    )
    mhc_scale = torch.randn((3,), device="npu", dtype=torch.float32)
    mhc_base = torch.randn((mhc_mult3,), device="npu", dtype=torch.float32)

    program = pre_split_mixes_fwd_program(
        num_tokens, mhc_mult, mhc_post_mult_value, mhc_pre_eps
    )
    kernel = tilelang.compile(program, target="ascend", out_idx=[3, 4, 5])
    actual = kernel(input_mixes, mhc_scale, mhc_base)
    expected = ref_program(
        input_mixes,
        mhc_scale,
        mhc_base,
        mhc_mult,
        mhc_post_mult_value,
        mhc_pre_eps,
    )
    torch.npu.synchronize()
    for actual_part, expected_part in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            actual_part, expected_part, rtol=1e-5, atol=2e-5
        )
    print("PASS: mhc_pre_split_mixes_fwd")
