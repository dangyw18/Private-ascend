"""Stage-1 Ascend SimtVF rewrite of MHC head-compute-mix forward."""

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


def head_compute_mix_fwd_program(
    num_tokens: int,
    mhc_mult: int,
    mhc_pre_eps: float,
    token_block_size: int = TOKEN_BLOCK_SIZE,
    threads: int = THREADS,
):
    """Build the explicitly scoped Ascend program.

    This is the Stage-1 direct-global case: the only ``T.Parallel`` nest and
    its scalar expression are enclosed by one SimtVF.
    """
    assert num_tokens > 0
    assert mhc_mult > 0
    assert token_block_size > 0
    assert threads > 0

    @T.prim_func
    def main(
        input_mix: T.Buffer((num_tokens, mhc_mult), "float32"),
        mhc_scale: T.Buffer((1,), "float32"),
        mhc_base: T.Buffer((mhc_mult,), "float32"),
        output_mix: T.Buffer((num_tokens, mhc_mult), "float32"),
    ):
        with T.Kernel(T.ceildiv(num_tokens, token_block_size)) as pid:
            with T.SimtVF(threads=threads):
                for local_token, mix_idx in T.Parallel(
                    token_block_size, mhc_mult
                ):
                    token = pid * token_block_size + local_token
                    if token < num_tokens:
                        output_mix[token, mix_idx] = (
                            T.sigmoid(
                                input_mix[token, mix_idx] * mhc_scale[0]
                                + mhc_base[mix_idx]
                            )
                            + mhc_pre_eps
                        )

    return main


@lru_cache(maxsize=None)
def _compile_head_compute_mix_fwd(
    num_tokens: int,
    mhc_mult: int,
    mhc_pre_eps: float,
    token_block_size: int,
    threads: int,
):
    return tilelang.compile(
        head_compute_mix_fwd_program(
            num_tokens,
            mhc_mult,
            mhc_pre_eps,
            token_block_size,
            threads,
        ),
        target="ascend",
        out_idx=-1,
        pass_configs=_PASS_CONFIGS,
    )


def mhc_head_compute_mix_fwd(
    input_mix: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    mhc_pre_eps: float = 1e-6,
    token_block_size: int = TOKEN_BLOCK_SIZE,
    threads: int = THREADS,
) -> torch.Tensor:
    """Compile and invoke the Ascend kernel on a tensor shaped ``(..., M)``."""
    assert input_mix.dtype == torch.float32
    assert input_mix.ndim >= 2
    assert input_mix.is_contiguous()
    mhc_mult = input_mix.shape[-1]
    assert mhc_scale.shape == (1,)
    assert mhc_scale.dtype == torch.float32
    assert mhc_base.shape == (mhc_mult,)
    assert mhc_base.dtype == torch.float32

    flat_input = input_mix.reshape(-1, mhc_mult)
    kernel = _compile_head_compute_mix_fwd(
        flat_input.shape[0],
        mhc_mult,
        mhc_pre_eps,
        token_block_size,
        threads,
    )
    output = kernel(flat_input, mhc_scale, mhc_base)
    return output.reshape_as(input_mix)


def mhc_head_compute_mix_fwd_ref(
    input_mix: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    mhc_pre_eps: float = 1e-6,
) -> torch.Tensor:
    """Torch reference for the forward operator."""
    return torch.sigmoid(input_mix * mhc_scale[0] + mhc_base) + mhc_pre_eps


def _npu_device() -> torch.device:
    try:
        __import__("torch_npu")
    except ImportError as exc:
        raise RuntimeError("Running this example requires torch_npu.") from exc
    return torch.device("npu")


def main() -> None:
    torch.manual_seed(42)
    device = _npu_device()
    input_mix = torch.randn((1, 128, 4), device=device, dtype=torch.float32)
    mhc_scale = torch.randn((1,), device=device, dtype=torch.float32)
    mhc_base = torch.randn((4,), device=device, dtype=torch.float32)

    actual = mhc_head_compute_mix_fwd(input_mix, mhc_scale, mhc_base)
    expected = mhc_head_compute_mix_fwd_ref(input_mix, mhc_scale, mhc_base)
    torch.npu.synchronize()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-5)
    print("PASS: mhc_head_compute_mix_fwd")


if __name__ == "__main__":
    main()
