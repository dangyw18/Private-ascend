"""MHC Sinkhorn forward — Ascend port of tile_kernels/mhc/sinkhorn_kernel.py.

The kernel body is kept verbatim from the GPU original. The only Ascend
changes (pattern from examples/ascend/test_mhc_copy.py and mhc_case.txt):

- GM <-> fragment copies are staged through a shared (UB) buffer outside the
  SimtVF scope so they stay on MTE.
- The fragment compute region is wrapped in ``T.SimtVF``.
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
def _mhc_sinkhorn_fwd(
    hidden_size: int,
    token_block_size: int,
    repeat: int,
    eps: float,
    threads: int = 256,
) -> tilelang.JITKernel:
    num_tokens = T.dynamic('num_tokens')

    @T.prim_func
    def mhc_sinkhorn_kernel(
        comb_res_mix: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32],
        comb_res_mix_out: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32],
    ) -> None:
        with T.Kernel(T.ceildiv(num_tokens, token_block_size)) as pid_x:
            # Ascend: stage GM -> UB outside SimtVF (MTE copy).
            comb_ub = T.alloc_shared((token_block_size, hidden_size, hidden_size), T.float32)
            T.copy(comb_res_mix[pid_x * token_block_size, 0, 0], comb_ub)

            with T.SimtVF(threads=threads):
                comb_frag = T.alloc_fragment((token_block_size, hidden_size, hidden_size), T.float32)
                row_sum = T.alloc_fragment((token_block_size, hidden_size), T.float32)
                col_sum = T.alloc_fragment((token_block_size, hidden_size), T.float32)

                for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
                    comb_frag[i, j, k] = comb_ub[i, j, k]

                # comb = comb.softmax(-1) + eps
                row_max = T.alloc_fragment((token_block_size, hidden_size), T.float32)
                T.reduce_max(comb_frag, row_max, dim=2)
                for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
                    comb_frag[i, j, k] = T.exp(comb_frag[i, j, k] - row_max[i, j])
                T.reduce_sum(comb_frag, row_sum, dim=2)
                for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
                    comb_frag[i, j, k] = comb_frag[i, j, k] / row_sum[i, j] + eps

                # comb = comb / (comb.sum(-2) + eps)
                T.reduce_sum(comb_frag, col_sum, dim=1)
                for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
                    comb_frag[i, j, k] = comb_frag[i, j, k] / (col_sum[i, k] + eps)

                for _ in T.serial(repeat - 1):
                    # comb = comb / (comb.sum(-1) + eps)
                    T.reduce_sum(comb_frag, row_sum, dim=2)
                    for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
                        comb_frag[i, j, k] = comb_frag[i, j, k] / (row_sum[i, j] + eps)

                    # comb = comb / (comb.sum(-2) + eps)
                    T.reduce_sum(comb_frag, col_sum, dim=1)
                    for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
                        comb_frag[i, j, k] = comb_frag[i, j, k] / (col_sum[i, k] + eps)

                for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
                    comb_ub[i, j, k] = comb_frag[i, j, k]

            # Ascend: stage UB -> GM outside SimtVF (MTE copy).
            T.copy(comb_ub, comb_res_mix_out[pid_x * token_block_size, 0, 0])

    return mhc_sinkhorn_kernel


@lru_cache(maxsize=None)
def _compile_mhc_sinkhorn_fwd(
    hidden_size: int,
    token_block_size: int,
    repeat: int,
    eps: float,
    threads: int,
):
    prim = _mhc_sinkhorn_fwd.get_tir(hidden_size, token_block_size, repeat, eps, threads)
    return tilelang.compile(prim, target="ascend", pass_configs=_PASS_CONFIGS)


def mhc_sinkhorn_fwd(
    x: torch.Tensor,
    repeat: int = 10,
    eps: float = 1e-6,
    token_block_size: int = 1,
    threads: int = 256,
) -> torch.Tensor:
    """Host wrapper mirroring tile_kernels/modeling/mhc/ops/sinkhorn.py."""
    shape = x.shape
    x = x.contiguous().view(-1, *x.shape[-2:])
    hidden_size = x.shape[1]
    output = torch.empty_like(x)
    fwd_kernel = _compile_mhc_sinkhorn_fwd(hidden_size, token_block_size, repeat, eps, threads)
    fwd_kernel(x, output)
    return output.view(shape)


def mhc_sinkhorn_fwd_ref(
    x: torch.Tensor,
    repeat: int = 10,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Torch reference with the same normalization order as the kernel."""
    output = torch.softmax(x, dim=-1) + eps
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
    comb_res_mix = torch.randn((1, 64, 4, 4), device=device, dtype=torch.float32)

    actual = mhc_sinkhorn_fwd(comb_res_mix)
    expected = mhc_sinkhorn_fwd_ref(comb_res_mix)
    torch.npu.synchronize()
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    print("PASS: mhc_sinkhorn_fwd")


if __name__ == "__main__":
    main()
