"""MHC head-compute-mix forward — Ascend port of tile_kernels/mhc/head_compute_mix_kernel.py.

The kernel body is kept verbatim from the GPU original (same buffer and loop
names). The only Ascend change (pattern from examples/ascend/test_mhc_copy.py):

- The single ``T.Parallel`` nest is wrapped in ``T.SimtVF`` (light workload,
  direct GM access, no UB staging needed).
- The host wrapper compiles with ``target="ascend"``.
"""

from __future__ import annotations

from functools import lru_cache

import tilelang
import torch
from tilelang import language as T

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _mhc_head_compute_mix_fwd(
    mhc_mult: int,
    mhc_pre_eps: float,
    token_block_size: int,
    threads: int = 128,
) -> tilelang.JITKernel:
    num_tokens = T.dynamic('num_tokens')

    @T.prim_func
    def mhc_head_compute_mix_fwd_kernel(
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

    return mhc_head_compute_mix_fwd_kernel


@lru_cache(maxsize=None)
def _compile_mhc_head_compute_mix_fwd(
    mhc_mult: int,
    mhc_pre_eps: float,
    token_block_size: int,
    threads: int,
):
    prim = _mhc_head_compute_mix_fwd.get_tir(mhc_mult, mhc_pre_eps, token_block_size, threads)
    return tilelang.compile(prim, target="ascend", pass_configs=_PASS_CONFIGS)


def mhc_head_compute_mix_fwd(
    input_mix: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    mhc_pre_eps: float = 1e-6,
    token_block_size: int = 32,
    threads: int = 128,
) -> torch.Tensor:
    """Host wrapper mirroring tile_kernels/modeling/mhc/ops/head_compute_mix.py."""
    assert input_mix.ndim == 3
    shape = input_mix.shape
    mhc_mult = input_mix.shape[-1]

    input_mix = input_mix.contiguous().view(-1, mhc_mult)
    output_mix = torch.empty_like(input_mix)

    fwd_kernel = _compile_mhc_head_compute_mix_fwd(mhc_mult, mhc_pre_eps, token_block_size, threads)
    fwd_kernel(input_mix, mhc_scale, mhc_base, output_mix)
    return output_mix.view(shape)


def mhc_head_compute_mix_fwd_ref(
    input_mix: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    mhc_pre_eps: float = 1e-6,
) -> torch.Tensor:
    """Torch reference with the same formula as the kernel."""
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
