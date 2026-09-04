"""Stage-1 Ascend SimtVF rewrite of MHC pre-split-mixes forward."""

from __future__ import annotations

from functools import lru_cache

import tilelang
import tilelang.language as T
import torch


THREADS = 128
TOKEN_BLOCK_SIZE = 32
_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}


def pre_split_mixes_fwd_program(
    num_tokens: int,
    mhc_mult: int,
    mhc_post_mult_value: float,
    mhc_pre_eps: float,
    token_block_size: int = TOKEN_BLOCK_SIZE,
    threads: int = THREADS,
):
    """Build the GM->UB | SimtVF | UB->GM Stage-1 program."""
    assert num_tokens > 0
    assert mhc_mult > 0
    assert token_block_size > 0
    assert num_tokens % token_block_size == 0
    assert threads > 0
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
            input_ub = T.alloc_shared(
                (token_block_size, mhc_mult3), T.float32
            )
            pre_layer_mix_ub = T.alloc_shared(
                (token_block_size, mhc_mult), T.float32
            )
            post_layer_mix_ub = T.alloc_shared(
                (token_block_size, mhc_mult), T.float32
            )
            comb_res_mix_ub = T.alloc_shared(
                (token_block_size, mhc_mult2), T.float32
            )

            # MTE/DMA copies remain outside SimtVF.
            T.copy(input_mixes[pid * token_block_size, 0], input_ub)

            with T.SimtVF(threads=threads):
                input_frag = T.alloc_fragment(
                    (token_block_size, mhc_mult3), T.float32
                )
                pre_frag = T.alloc_fragment(
                    (token_block_size, mhc_mult), T.float32
                )
                post_frag = T.alloc_fragment(
                    (token_block_size, mhc_mult), T.float32
                )
                comb_frag = T.alloc_fragment(
                    (token_block_size, mhc_mult2), T.float32
                )

                # Fragment normal copies belong to the same VF as their users.
                T.copy(input_ub, input_frag)

                for i, j in T.Parallel(token_block_size, mhc_mult):
                    pre_frag[i, j] = (
                        T.sigmoid(
                            input_frag[i, j] * mhc_scale[0] + mhc_base[j]
                        )
                        + mhc_pre_eps
                    )
                for i, j in T.Parallel(token_block_size, mhc_mult):
                    post_frag[i, j] = (
                        T.sigmoid(
                            input_frag[i, j + mhc_mult] * mhc_scale[1]
                            + mhc_base[j + mhc_mult]
                        )
                        * mhc_post_mult_value
                    )
                for i, j in T.Parallel(token_block_size, mhc_mult2):
                    comb_frag[i, j] = (
                        input_frag[i, j + mhc_mult * 2] * mhc_scale[2]
                        + mhc_base[j + mhc_mult * 2]
                    )

                T.copy(pre_frag, pre_layer_mix_ub)
                T.copy(post_frag, post_layer_mix_ub)
                T.copy(comb_frag, comb_res_mix_ub)

            T.copy(pre_layer_mix_ub, pre_layer_mix[pid * token_block_size, 0])
            T.copy(
                post_layer_mix_ub, post_layer_mix[pid * token_block_size, 0]
            )
            T.copy(comb_res_mix_ub, comb_res_mix[pid * token_block_size, 0])

    return main


@lru_cache(maxsize=None)
def _compile_pre_split_mixes_fwd(
    num_tokens: int,
    mhc_mult: int,
    mhc_post_mult_value: float,
    mhc_pre_eps: float,
    token_block_size: int,
    threads: int,
):
    return tilelang.compile(
        pre_split_mixes_fwd_program(
            num_tokens,
            mhc_mult,
            mhc_post_mult_value,
            mhc_pre_eps,
            token_block_size,
            threads,
        ),
        target="ascend",
        out_idx=[3, 4, 5],
        pass_configs=_PASS_CONFIGS,
    )


def mhc_pre_split_mixes_fwd(
    input_mixes: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    mhc_mult: int = 4,
    mhc_post_mult_value: float = 2.0,
    mhc_pre_eps: float = 1e-6,
    token_block_size: int = TOKEN_BLOCK_SIZE,
    threads: int = THREADS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compile and invoke the kernel on an input shaped ``(..., M3)``."""
    assert input_mixes.dtype == torch.float32
    assert input_mixes.ndim >= 2
    assert input_mixes.is_contiguous()
    mhc_mult2 = mhc_mult * mhc_mult
    mhc_mult3 = mhc_mult * 2 + mhc_mult2
    assert input_mixes.shape[-1] == mhc_mult3
    assert mhc_scale.shape == (3,)
    assert mhc_scale.dtype == torch.float32
    assert mhc_base.shape == (mhc_mult3,)
    assert mhc_base.dtype == torch.float32

    leading_shape = input_mixes.shape[:-1]
    flat_input = input_mixes.reshape(-1, mhc_mult3)
    assert flat_input.shape[0] % token_block_size == 0
    kernel = _compile_pre_split_mixes_fwd(
        flat_input.shape[0],
        mhc_mult,
        mhc_post_mult_value,
        mhc_pre_eps,
        token_block_size,
        threads,
    )
    pre, post, comb = kernel(flat_input, mhc_scale, mhc_base)
    return (
        pre.reshape(*leading_shape, mhc_mult, 1),
        post.reshape(*leading_shape, mhc_mult, 1),
        comb.reshape(*leading_shape, mhc_mult, mhc_mult),
    )


def mhc_pre_split_mixes_fwd_ref(
    input_mixes: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    mhc_mult: int = 4,
    mhc_post_mult_value: float = 2.0,
    mhc_pre_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Torch reference for the three forward outputs."""
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
    return (
        pre.unsqueeze(-1),
        post.unsqueeze(-1),
        comb.reshape(*input_mixes.shape[:-1], mhc_mult, mhc_mult),
    )


def _npu_device() -> torch.device:
    try:
        __import__("torch_npu")
    except ImportError as exc:
        raise RuntimeError("Running this example requires torch_npu.") from exc
    return torch.device("npu")


def main() -> None:
    torch.manual_seed(42)
    device = _npu_device()
    mhc_mult = 4
    mhc_mult3 = mhc_mult * 2 + mhc_mult * mhc_mult
    input_mixes = torch.randn(
        (1, 128, mhc_mult3), device=device, dtype=torch.float32
    )
    mhc_scale = torch.randn((3,), device=device, dtype=torch.float32)
    mhc_base = torch.randn((mhc_mult3,), device=device, dtype=torch.float32)

    actual = mhc_pre_split_mixes_fwd(
        input_mixes, mhc_scale, mhc_base, mhc_mult
    )
    expected = mhc_pre_split_mixes_fwd_ref(
        input_mixes, mhc_scale, mhc_base, mhc_mult
    )
    torch.npu.synchronize()
    for actual_part, expected_part in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            actual_part, expected_part, rtol=1e-5, atol=2e-5
        )
    print("PASS: mhc_pre_split_mixes_fwd")


if __name__ == "__main__":
    main()
