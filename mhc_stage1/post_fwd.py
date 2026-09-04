"""Stage-1 Ascend SimtVF rewrite of MHC post forward."""

from __future__ import annotations

from functools import lru_cache
import math

import tilelang
import tilelang.language as T
import torch


THREADS = 128
HIDDEN_BLOCK_SIZE = 256
_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}


def post_fwd_program(
    num_tokens: int,
    mhc_mult: int,
    hidden_size: int,
    hidden_block_size: int = HIDDEN_BLOCK_SIZE,
    threads: int = THREADS,
):
    """Build the MTE/serial-tile/SimtVF form used by the Ascend example."""
    assert num_tokens > 0
    assert mhc_mult > 0
    assert hidden_size > 0
    assert hidden_block_size > 0
    assert threads > 0
    hidden_block_size = math.gcd(hidden_size, hidden_block_size)

    @T.prim_func
    def main(
        comb_res_mix: T.Buffer((num_tokens, mhc_mult, mhc_mult), "float32"),
        residual: T.Buffer(
            (num_tokens, mhc_mult, hidden_size), "bfloat16"
        ),
        post_layer_mix: T.Buffer((num_tokens, mhc_mult), "float32"),
        x: T.Buffer((num_tokens, hidden_size), "bfloat16"),
        output: T.Buffer(
            (num_tokens, mhc_mult, hidden_size), "bfloat16"
        ),
    ):
        with T.Kernel(num_tokens) as token:
            output_ub = T.alloc_shared(
                (mhc_mult, hidden_block_size), T.bfloat16
            )
            residual_ub = T.alloc_shared(
                (mhc_mult, hidden_block_size), T.bfloat16
            )
            x_ub = T.alloc_shared((hidden_block_size,), T.bfloat16)
            comb_ub = T.alloc_shared((mhc_mult, mhc_mult), T.float32)
            post_mix_ub = T.alloc_shared((mhc_mult,), T.float32)

            T.copy(comb_res_mix[token, 0, 0], comb_ub)
            T.copy(post_layer_mix[token, 0], post_mix_ub)

            # Preserve the serial hidden tiling and all GM<->UB copies.
            for hidden_tile in T.serial(hidden_size // hidden_block_size):
                hidden_offset = hidden_tile * hidden_block_size
                T.copy(
                    residual[token, 0, hidden_offset], residual_ub
                )
                T.copy(x[token, hidden_offset], x_ub)

                with T.SimtVF(threads=threads):
                    output_frag = T.alloc_fragment(
                        (mhc_mult, hidden_block_size), T.float32
                    )
                    residual_frag = T.alloc_fragment(
                        (mhc_mult, hidden_block_size), T.float32
                    )
                    x_frag = T.alloc_fragment(
                        (hidden_block_size,), T.float32
                    )

                    # These dtype-changing UB<->fragment copies are normal
                    # thread copies and therefore remain inside SimtVF.
                    T.copy(residual_ub, residual_frag)
                    T.copy(x_ub, x_frag)
                    for output_mix, hidden_idx in T.Parallel(
                        mhc_mult, hidden_block_size
                    ):
                        output_frag[output_mix, hidden_idx] = (
                            post_mix_ub[output_mix] * x_frag[hidden_idx]
                        )
                        for input_mix in T.serial(mhc_mult):
                            output_frag[output_mix, hidden_idx] += (
                                comb_ub[input_mix, output_mix]
                                * residual_frag[input_mix, hidden_idx]
                            )
                    T.copy(output_frag, output_ub)

                T.copy(output_ub, output[token, 0, hidden_offset])

    return main


@lru_cache(maxsize=None)
def _compile_post_fwd(
    num_tokens: int,
    mhc_mult: int,
    hidden_size: int,
    hidden_block_size: int,
    threads: int,
):
    return tilelang.compile(
        post_fwd_program(
            num_tokens,
            mhc_mult,
            hidden_size,
            hidden_block_size,
            threads,
        ),
        target="ascend",
        out_idx=-1,
        pass_configs=_PASS_CONFIGS,
    )


def mhc_post_fwd(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    hidden_block_size: int = HIDDEN_BLOCK_SIZE,
    threads: int = THREADS,
) -> torch.Tensor:
    """Compile and invoke MHC post using the original public tensor shapes."""
    assert residual.dtype == torch.bfloat16
    assert residual.ndim == 4
    num_sequences, num_tokens, mhc_mult, hidden_size = residual.shape
    assert x.shape == (num_sequences, num_tokens, hidden_size)
    assert x.dtype == torch.bfloat16
    assert post_layer_mix.shape == (
        num_sequences,
        num_tokens,
        mhc_mult,
        1,
    )
    assert post_layer_mix.dtype == torch.float32
    assert comb_res_mix.shape == (
        num_sequences,
        num_tokens,
        mhc_mult,
        mhc_mult,
    )
    assert comb_res_mix.dtype == torch.float32
    assert x.is_contiguous()
    assert residual.is_contiguous()
    assert post_layer_mix.is_contiguous()
    assert comb_res_mix.is_contiguous()

    flat_tokens = num_sequences * num_tokens
    kernel = _compile_post_fwd(
        flat_tokens,
        mhc_mult,
        hidden_size,
        hidden_block_size,
        threads,
    )
    output = kernel(
        comb_res_mix.reshape(flat_tokens, mhc_mult, mhc_mult),
        residual.reshape(flat_tokens, mhc_mult, hidden_size),
        post_layer_mix.reshape(flat_tokens, mhc_mult),
        x.reshape(flat_tokens, hidden_size),
    )
    return output.reshape_as(residual)


def mhc_post_fwd_ref(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    """Torch reference: scaled residual base plus transposed mix matmul."""
    num_sequences, num_tokens, mhc_mult, hidden_size = residual.shape
    flat_tokens = num_sequences * num_tokens
    mixed_residual = torch.bmm(
        comb_res_mix.reshape(flat_tokens, mhc_mult, mhc_mult).transpose(1, 2),
        residual.float().reshape(flat_tokens, mhc_mult, hidden_size),
    ).reshape_as(residual)
    return (
        x.float().unsqueeze(-2) * post_layer_mix + mixed_residual
    ).bfloat16()


def _npu_device() -> torch.device:
    try:
        __import__("torch_npu")
    except ImportError as exc:
        raise RuntimeError("Running this example requires torch_npu.") from exc
    return torch.device("npu")


def main() -> None:
    torch.manual_seed(42)
    device = _npu_device()
    num_sequences, num_tokens, mhc_mult, hidden_size = 1, 8, 4, 1280
    x = torch.randn(
        (num_sequences, num_tokens, hidden_size),
        device=device,
        dtype=torch.bfloat16,
    )
    residual = torch.randn(
        (num_sequences, num_tokens, mhc_mult, hidden_size),
        device=device,
        dtype=torch.bfloat16,
    )
    post_layer_mix = torch.randn(
        (num_sequences, num_tokens, mhc_mult, 1),
        device=device,
        dtype=torch.float32,
    )
    comb_res_mix = torch.randn(
        (num_sequences, num_tokens, mhc_mult, mhc_mult),
        device=device,
        dtype=torch.float32,
    )

    actual = mhc_post_fwd(x, residual, post_layer_mix, comb_res_mix)
    expected = mhc_post_fwd_ref(x, residual, post_layer_mix, comb_res_mix)
    torch.npu.synchronize()
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    print("PASS: mhc_post_fwd")


if __name__ == "__main__":
    main()
