"""MHC post forward — Ascend port of tile_kernels/mhc/post_kernel.py.

The kernel body is kept verbatim from the GPU original (same buffer and loop
names). The only Ascend changes:

- ``threads=n_thr`` moves from ``T.Kernel`` into ``T.SimtVF``.
- ``T.pdl_sync`` / ``disable_tma`` / ``T.Pipelined`` are CUDA-only; the hidden
  tiling uses ``T.serial`` and the per-tile compute region is wrapped in
  ``T.SimtVF``.
- ``a`` / ``c`` are staged through shared (UB) buffers instead of direct
  GM -> fragment loads.
- The host wrapper compiles with ``target="ascend"``.
"""

from __future__ import annotations

import math
from functools import lru_cache

import tilelang
import torch
from tilelang import language as T

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _mhc_post_fwd(mhc: int, hidden: int, n_thr: int = 128, h_blk: int = 1024) -> tilelang.JITKernel:
    n = T.dynamic('num_tokens')
    h = hidden

    h_blk = math.gcd(hidden, h_blk)

    @T.prim_func
    def _mhc_post_fwd_kernel(
        a: T.Tensor[(n, mhc, mhc), T.float32],
        b: T.Tensor[(n, mhc, h), T.bfloat16],
        c: T.Tensor[(n, mhc), T.float32],
        d: T.Tensor[(n, h), T.bfloat16],
        x: T.Tensor[(n, mhc, h), T.bfloat16],
    ) -> None:
        with T.Kernel(n) as pid_n:
            x_shared = T.alloc_shared((mhc, h_blk), T.bfloat16)
            b_shared = T.alloc_shared((mhc, h_blk), T.bfloat16)
            d_shared = T.alloc_shared(h_blk, T.bfloat16)

            # Ascend: stage a / c through UB instead of GM -> fragment loads.
            a_shared = T.alloc_shared((mhc, mhc), T.float32)
            c_shared = T.alloc_shared(mhc, T.float32)
            T.copy(a[pid_n, 0, 0], a_shared)
            T.copy(c[pid_n, 0], c_shared)

            # Ascend: T.Pipelined is CUDA-only; serial hidden tiling instead.
            for i0_h in T.serial(T.ceildiv(h, h_blk)):
                T.copy(b[pid_n, 0, i0_h * h_blk], b_shared)
                T.copy(d[pid_n, i0_h * h_blk], d_shared)

                with T.SimtVF(threads=n_thr):
                    x_local = T.alloc_fragment((mhc, h_blk), T.float32)
                    b_local = T.alloc_fragment((mhc, h_blk), T.float32)
                    d_local = T.alloc_fragment(h_blk, T.float32)

                    a_local = T.alloc_fragment((mhc, mhc), T.float32)
                    c_local = T.alloc_fragment(mhc, T.float32)
                    for i_mhci, i_mhco in T.Parallel(mhc, mhc):
                        a_local[i_mhci, i_mhco] = a_shared[i_mhci, i_mhco]
                    for i_mhc in T.Parallel(mhc):
                        c_local[i_mhc] = c_shared[i_mhc]

                    T.copy(b_shared, b_local)
                    T.copy(d_shared, d_local)
                    for i_mhco, i1_h in T.Parallel(mhc, h_blk):
                        x_local[i_mhco, i1_h] = c_local[i_mhco] * d_local[i1_h]
                        for i_mhci in T.serial(mhc):
                            x_local[i_mhco, i1_h] += a_local[i_mhci, i_mhco] * b_local[i_mhci, i1_h]
                    T.copy(x_local, x_shared)

                T.copy(x_shared, x[pid_n, 0, i0_h * h_blk])

    return _mhc_post_fwd_kernel


@lru_cache(maxsize=None)
def _compile_mhc_post_fwd(mhc: int, hidden: int, n_thr: int, h_blk: int):
    prim = _mhc_post_fwd.get_tir(mhc, hidden, n_thr, h_blk)
    return tilelang.compile(prim, target="ascend", pass_configs=_PASS_CONFIGS)


def mhc_post_fwd(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    out: torch.Tensor | None = None,
    n_thr: int = 128,
    h_blk: int = 1024,
) -> torch.Tensor:
    """Host wrapper mirroring tile_kernels/mhc/post_kernel.py."""
    num_seqs, num_tokens, mhc, hidden = residual.shape

    assert x.dtype == torch.bfloat16, f'{x.dtype=}'
    assert residual.dtype == torch.bfloat16, f'{residual.dtype=}'
    assert post_layer_mix.dtype == torch.float32, f'{post_layer_mix.dtype=}'
    assert comb_res_mix.dtype == torch.float32, f'{comb_res_mix.dtype=}'
    assert x.shape == (num_seqs, num_tokens, hidden), f'{x.shape=}'
    assert post_layer_mix.shape == (num_seqs, num_tokens, mhc, 1), f'{post_layer_mix.shape=}'
    assert comb_res_mix.shape == (num_seqs, num_tokens, mhc, mhc), f'{comb_res_mix.shape=}'

    residual = residual.contiguous()
    assert x.is_contiguous()
    assert post_layer_mix.is_contiguous()
    assert comb_res_mix.is_contiguous()

    if out is None:
        out = torch.empty_like(residual)
    kernel = _compile_mhc_post_fwd(mhc, hidden, n_thr, h_blk)
    kernel(
        comb_res_mix.flatten(0, 1),
        residual.flatten(0, 1),
        post_layer_mix.flatten(0, 1).squeeze(-1),
        x.flatten(0, 1),
        out.flatten(0, 1),
    )
    return out


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
