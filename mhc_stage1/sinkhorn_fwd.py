"""Stage-1 Ascend SimtVF rewrite of MHC Sinkhorn forward."""

from __future__ import annotations

from functools import lru_cache

import tilelang
import tilelang.language as T
import torch


THREADS = 256
TOKEN_BLOCK_SIZE = 1
_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}


def sinkhorn_fwd_program(
    num_tokens: int,
    hidden_size: int,
    repeat: int,
    eps: float,
    token_block_size: int = TOKEN_BLOCK_SIZE,
    threads: int = THREADS,
):
    """Build the verified GM->UB | SimtVF | UB->GM Sinkhorn pattern."""
    assert num_tokens > 0
    assert hidden_size > 0
    assert repeat >= 1
    assert token_block_size > 0
    assert num_tokens % token_block_size == 0
    assert threads > 0

    @T.prim_func
    def main(
        comb_res_mix: T.Buffer(
            (num_tokens, hidden_size, hidden_size), "float32"
        ),
        output: T.Buffer((num_tokens, hidden_size, hidden_size), "float32"),
    ):
        with T.Kernel(num_tokens // token_block_size) as pid:
            comb_ub = T.alloc_shared(
                (token_block_size, hidden_size, hidden_size), T.float32
            )

            # Keep the GM transfer on MTE/DMA, outside SimtVF.
            T.copy(
                comb_res_mix[pid * token_block_size, 0, 0], comb_ub
            )

            with T.SimtVF(threads=threads):
                comb_frag = T.alloc_fragment(
                    (token_block_size, hidden_size, hidden_size), T.float32
                )
                row_sum = T.alloc_fragment(
                    (token_block_size, hidden_size), T.float32
                )
                col_sum = T.alloc_fragment(
                    (token_block_size, hidden_size), T.float32
                )
                row_max = T.alloc_fragment(
                    (token_block_size, hidden_size), T.float32
                )

                # Explicit Parallel copies match the known-good mhc_case form.
                for i, j, k in T.Parallel(
                    token_block_size, hidden_size, hidden_size
                ):
                    comb_frag[i, j, k] = comb_ub[i, j, k]

                T.reduce_max(comb_frag, row_max, dim=2)
                for i, j, k in T.Parallel(
                    token_block_size, hidden_size, hidden_size
                ):
                    comb_frag[i, j, k] = T.exp(
                        comb_frag[i, j, k] - row_max[i, j]
                    )

                T.reduce_sum(comb_frag, row_sum, dim=2)
                for i, j, k in T.Parallel(
                    token_block_size, hidden_size, hidden_size
                ):
                    comb_frag[i, j, k] = (
                        comb_frag[i, j, k] / row_sum[i, j] + eps
                    )

                T.reduce_sum(comb_frag, col_sum, dim=1)
                for i, j, k in T.Parallel(
                    token_block_size, hidden_size, hidden_size
                ):
                    comb_frag[i, j, k] = comb_frag[i, j, k] / (
                        col_sum[i, k] + eps
                    )

                for _ in T.serial(repeat - 1):
                    T.reduce_sum(comb_frag, row_sum, dim=2)
                    for i, j, k in T.Parallel(
                        token_block_size, hidden_size, hidden_size
                    ):
                        comb_frag[i, j, k] = comb_frag[i, j, k] / (
                            row_sum[i, j] + eps
                        )

                    T.reduce_sum(comb_frag, col_sum, dim=1)
                    for i, j, k in T.Parallel(
                        token_block_size, hidden_size, hidden_size
                    ):
                        comb_frag[i, j, k] = comb_frag[i, j, k] / (
                            col_sum[i, k] + eps
                        )

                for i, j, k in T.Parallel(
                    token_block_size, hidden_size, hidden_size
                ):
                    comb_ub[i, j, k] = comb_frag[i, j, k]

            T.copy(comb_ub, output[pid * token_block_size, 0, 0])

    return main


@lru_cache(maxsize=None)
def _compile_sinkhorn_fwd(
    num_tokens: int,
    hidden_size: int,
    repeat: int,
    eps: float,
    token_block_size: int,
    threads: int,
):
    return tilelang.compile(
        sinkhorn_fwd_program(
            num_tokens,
            hidden_size,
            repeat,
            eps,
            token_block_size,
            threads,
        ),
        target="ascend",
        out_idx=-1,
        pass_configs=_PASS_CONFIGS,
    )


def mhc_sinkhorn_fwd(
    comb_res_mix: torch.Tensor,
    repeat: int = 3,
    eps: float = 1e-6,
    token_block_size: int = TOKEN_BLOCK_SIZE,
    threads: int = THREADS,
) -> torch.Tensor:
    """Compile and invoke Sinkhorn on an input shaped ``(..., H, H)``."""
    assert comb_res_mix.dtype == torch.float32
    assert comb_res_mix.ndim >= 3
    assert comb_res_mix.is_contiguous()
    hidden_size = comb_res_mix.shape[-1]
    assert comb_res_mix.shape[-2:] == (hidden_size, hidden_size)
    leading_shape = comb_res_mix.shape[:-2]
    flat_input = comb_res_mix.reshape(-1, hidden_size, hidden_size)
    assert flat_input.shape[0] % token_block_size == 0

    kernel = _compile_sinkhorn_fwd(
        flat_input.shape[0],
        hidden_size,
        repeat,
        eps,
        token_block_size,
        threads,
    )
    output = kernel(flat_input)
    return output.reshape(*leading_shape, hidden_size, hidden_size)


def mhc_sinkhorn_fwd_ref(
    comb_res_mix: torch.Tensor,
    repeat: int = 3,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Torch reference with the same normalization order as the kernel."""
    output = torch.softmax(comb_res_mix, dim=-1) + eps
    output = output / (output.sum(dim=-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        output = output / (output.sum(dim=-1, keepdim=True) + eps)
        output = output / (output.sum(dim=-2, keepdim=True) + eps)
    return output


def _npu_device() -> torch.device:
    try:
        __import__("torch_npu")
    except ImportError as exc:
        raise RuntimeError("Running this example requires torch_npu.") from exc
    return torch.device("npu")


def main() -> None:
    torch.manual_seed(42)
    device = _npu_device()
    comb_res_mix = torch.randn(
        (1, 64, 4, 4), device=device, dtype=torch.float32
    )

    actual = mhc_sinkhorn_fwd(comb_res_mix)
    expected = mhc_sinkhorn_fwd_ref(comb_res_mix)
    torch.npu.synchronize()
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    print("PASS: mhc_sinkhorn_fwd")


if __name__ == "__main__":
    main()
