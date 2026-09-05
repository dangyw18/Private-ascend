"""Conservative Ascend SimtVF port of MHC pre-big-fuse forward."""

from __future__ import annotations

import math
from functools import lru_cache

import tilelang
import torch
from tilelang import language as T


SIMTVF_THREADS = 128
_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _mhc_pre_big_fuse(
    hidden_size: int,
    rms_eps: float,
    mhc_pre_eps: float,
    mhc_sinkhorn_eps: float,
    mhc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 16,
    mhc_mult: int = 4,
):
    num_tokens = T.dynamic('num_tokens')
    mhc_mult3 = mhc_mult * (2 + mhc_mult)
    hidden_block = math.gcd(512, hidden_size)

    @T.prim_func
    def mhc_pre_big_fuse(
        gemm_out_mul: T.Tensor[(n_splits, num_tokens, mhc_mult3), T.float32],
        gemm_out_sqrsum: T.Tensor[(n_splits, num_tokens), T.float32],
        mhc_scale: T.Tensor[(3,), T.float32],
        mhc_base: T.Tensor[(mhc_mult3,), T.float32],
        residual: T.Tensor[(num_tokens, mhc_mult, hidden_size), T.bfloat16],
        # outputs
        post_mix: T.Tensor[(num_tokens, mhc_mult), T.float32],
        comb_mix: T.Tensor[(num_tokens, mhc_mult * mhc_mult), T.float32],
        layer_input: T.Tensor[(num_tokens, hidden_size), T.bfloat16],
    ) -> None:
        with T.Kernel(num_tokens) as pid:
            ##################################################################
            # _mhc_pre_norm_fn_fwd_norm
            # Values shared by separate SimtVF regions use UB storage.
            mixes_shared = T.alloc_shared(mhc_mult3, T.float32)
            rms = T.alloc_shared(1, T.float32)
            mixes = T.alloc_shared(mhc_mult3, T.float32)
            rms[0] = 0
            for i_split in T.serial(n_splits):
                rms[0] += gemm_out_sqrsum[i_split, pid]
            rms[0] = T.rsqrt(rms[0] / (mhc_mult * hidden_size) + rms_eps)

            with T.SimtVF(threads=SIMTVF_THREADS):
                for j in T.Parallel(mhc_mult3):
                    mixes[j] = 0
                    for i_split in T.serial(n_splits):
                        mixes[j] += gemm_out_mul[i_split, pid, j]
                    mixes[j] *= rms[0]
                T.copy(mixes, mixes_shared)

            ##################################################################
            # _mhc_pre_split_mixes_fwd (post & comb)
            cm_shared = T.alloc_shared((mhc_mult, mhc_mult), T.float32)
            with T.SimtVF(threads=SIMTVF_THREADS):
                cm = T.alloc_fragment((mhc_mult, mhc_mult), T.float32)
                for j in T.Parallel(mhc_mult):
                    post_mix[pid, j] = T.sigmoid(mixes_shared[j + mhc_mult] * mhc_scale[1] + mhc_base[j + mhc_mult]) * mhc_post_mult_value
                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    cm[j, k] = mixes_shared[j * mhc_mult + k + mhc_mult * 2] * mhc_scale[2] + mhc_base[j * mhc_mult + k + mhc_mult * 2]

                ##################################################################
                # _mhc_sinkhorn_fwd
                row_sum = T.alloc_fragment(mhc_mult, T.float32)
                col_sum = T.alloc_fragment(mhc_mult, T.float32)

                # comb = comb.softmax(-1) + eps
                row_max = T.alloc_fragment(mhc_mult, T.float32)
                T.reduce_max(cm, row_max, dim=1)
                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    cm[j, k] = T.exp(cm[j, k] - row_max[j])
                T.reduce_sum(cm, row_sum, dim=1)
                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    cm[j, k] = cm[j, k] / row_sum[j] + mhc_sinkhorn_eps
                
                # comb = comb / (comb.sum(-2) + eps)
                T.reduce_sum(cm, col_sum, dim=0)
                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    cm[j, k] = cm[j, k] / (col_sum[k] + mhc_sinkhorn_eps)
                T.copy(cm, cm_shared)

            for _ in T.serial(sinkhorn_repeat - 1):
                # comb = comb / (comb.sum(-1) + eps)
                with T.SimtVF(threads=SIMTVF_THREADS):
                    cm = T.alloc_fragment((mhc_mult, mhc_mult), T.float32)
                    row_sum = T.alloc_fragment(mhc_mult, T.float32)
                    col_sum = T.alloc_fragment(mhc_mult, T.float32)
                    T.reduce_sum(cm, row_sum, dim=1)
                    T.copy(cm_shared, cm)
                    for j, k in T.Parallel(mhc_mult, mhc_mult):
                        cm[j, k] = cm[j, k] / (row_sum[j] + mhc_sinkhorn_eps)
                    # comb = comb / (comb.sum(-2) + eps)
                    T.reduce_sum(cm, col_sum, dim=0)
                    for j, k in T.Parallel(mhc_mult, mhc_mult):
                        cm[j, k] = cm[j, k] / (col_sum[k] + mhc_sinkhorn_eps)
                    T.copy(cm, cm_shared)

            # save comb_mix to global memory
            with T.SimtVF(threads=SIMTVF_THREADS):
                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    comb_mix[pid, j * mhc_mult + k] = cm[j, k]

            ##################################################################
            # _mhc_pre_split_mixes_fwd (pre)
            pre_mix_shared = T.alloc_shared(mhc_mult, T.float32)
            with T.SimtVF(threads=SIMTVF_THREADS):
                for j in T.Parallel(mhc_mult):
                    pre_mix_shared[j] = (
                        T.sigmoid(
                            mixes_shared[j] * mhc_scale[0] + mhc_base[j],
                        )
                        + mhc_pre_eps
                    )

            ###################################################################
            # _mhc_pre_apply_mix_fwd
            for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=2):
                xs = T.alloc_shared((mhc_mult, hidden_block), T.bfloat16)
                T.copy(residual[pid, 0, i0_h * hidden_block], xs, disable_tma=True)
                with T.SimtVF(threads=SIMTVF_THREADS):
                    xl = T.alloc_fragment((mhc_mult, hidden_block), T.float32)
                    T.copy(xs, xl, disable_tma=True)

                    ol = T.alloc_fragment(hidden_block, T.float32)
                    T.clear(ol)

                    for i_mhc in T.serial(mhc_mult):
                        pre = pre_mix_shared[i_mhc]
                        for i1_h in T.Parallel(hidden_block):
                            ol[i1_h] += pre * xl[i_mhc, i1_h]

                    T.copy(ol, layer_input[pid, i0_h * hidden_block], disable_tma=True)

    return mhc_pre_big_fuse


@lru_cache(maxsize=None)
def _compile_mhc_pre_big_fuse(
    hidden_size: int,
    rms_eps: float,
    mhc_pre_eps: float,
    mhc_sinkhorn_eps: float,
    mhc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int,
    mhc_mult: int,
):
    prim = _mhc_pre_big_fuse.get_tir(
        hidden_size,
        rms_eps,
        mhc_pre_eps,
        mhc_sinkhorn_eps,
        mhc_post_mult_value,
        sinkhorn_repeat,
        n_splits,
        mhc_mult,
    )
    return tilelang.compile(
        prim,
        target="ascend",
        pass_configs=_PASS_CONFIGS,
    )


def mhc_pre_big_fuse(
    gemm_out_mul: torch.Tensor,
    gemm_out_sqrsum: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    residual: torch.Tensor,
    rms_eps: float = 1e-6,
    mhc_pre_eps: float = 1e-6,
    mhc_sinkhorn_eps: float = 1e-6,
    mhc_post_mult_value: float = 2.0,
    sinkhorn_repeat: int = 10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compile and invoke the fused post-GEMM kernel on Ascend."""
    assert gemm_out_mul.dtype == torch.float32
    assert gemm_out_sqrsum.dtype == torch.float32
    assert mhc_scale.dtype == torch.float32
    assert mhc_base.dtype == torch.float32
    assert residual.dtype == torch.bfloat16
    assert gemm_out_mul.ndim == 3
    assert gemm_out_sqrsum.ndim == 2
    assert residual.ndim == 3
    assert mhc_scale.shape == (3,)
    assert gemm_out_mul.is_contiguous()
    assert gemm_out_sqrsum.is_contiguous()
    assert mhc_scale.is_contiguous()
    assert mhc_base.is_contiguous()
    assert residual.is_contiguous()

    n_splits, num_tokens, mhc_mult3 = gemm_out_mul.shape
    mhc_mult = residual.shape[1]
    hidden_size = residual.shape[2]
    assert mhc_mult3 == mhc_mult * (2 + mhc_mult)
    assert gemm_out_sqrsum.shape == (n_splits, num_tokens)
    assert mhc_base.shape == (mhc_mult3,)
    assert residual.shape[0] == num_tokens

    post_mix = torch.empty(
        (num_tokens, mhc_mult), device=residual.device, dtype=torch.float32
    )
    comb_mix = torch.empty(
        (num_tokens, mhc_mult * mhc_mult),
        device=residual.device,
        dtype=torch.float32,
    )
    layer_input = torch.empty(
        (num_tokens, hidden_size), device=residual.device, dtype=torch.bfloat16
    )
    kernel = _compile_mhc_pre_big_fuse(
        hidden_size,
        rms_eps,
        mhc_pre_eps,
        mhc_sinkhorn_eps,
        mhc_post_mult_value,
        sinkhorn_repeat,
        n_splits,
        mhc_mult,
    )
    kernel(
        gemm_out_mul,
        gemm_out_sqrsum,
        mhc_scale,
        mhc_base,
        residual,
        post_mix,
        comb_mix,
        layer_input,
    )
    return post_mix, comb_mix, layer_input


def mhc_pre_big_fuse_ref(
    gemm_out_mul: torch.Tensor,
    gemm_out_sqrsum: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    residual: torch.Tensor,
    rms_eps: float = 1e-6,
    mhc_pre_eps: float = 1e-6,
    mhc_sinkhorn_eps: float = 1e-6,
    mhc_post_mult_value: float = 2.0,
    sinkhorn_repeat: int = 10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Torch reference matching the fused kernel's three raw outputs."""
    mhc_mult = residual.shape[1]
    hidden_size = residual.shape[2]
    rms = torch.rsqrt(
        gemm_out_sqrsum.sum(dim=0) / (mhc_mult * hidden_size) + rms_eps
    )
    mixes = gemm_out_mul.sum(dim=0) * rms.unsqueeze(-1)

    post_mix = (
        torch.sigmoid(
            mixes[:, mhc_mult : 2 * mhc_mult] * mhc_scale[1]
            + mhc_base[mhc_mult : 2 * mhc_mult]
        )
        * mhc_post_mult_value
    )
    cm = (
        mixes[:, 2 * mhc_mult :] * mhc_scale[2]
        + mhc_base[2 * mhc_mult :]
    ).reshape(-1, mhc_mult, mhc_mult)

    row_max = cm.amax(dim=2, keepdim=True)
    cm = torch.exp(cm - row_max)
    cm = cm / cm.sum(dim=2, keepdim=True) + mhc_sinkhorn_eps
    cm = cm / (cm.sum(dim=1, keepdim=True) + mhc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        cm = cm / (cm.sum(dim=2, keepdim=True) + mhc_sinkhorn_eps)
        cm = cm / (cm.sum(dim=1, keepdim=True) + mhc_sinkhorn_eps)

    pre_mix = (
        torch.sigmoid(
            mixes[:, :mhc_mult] * mhc_scale[0]
            + mhc_base[:mhc_mult]
        )
        + mhc_pre_eps
    )
    layer_input = (
        residual.float() * pre_mix.unsqueeze(-1)
    ).sum(dim=1).to(torch.bfloat16)
    return post_mix, cm.reshape(cm.shape[0], -1), layer_input


def _npu_device() -> torch.device:
    try:
        __import__("torch_npu")
    except ImportError as exc:
        raise RuntimeError("Running this example requires torch_npu.") from exc
    return torch.device("npu")


def main() -> None:
    torch.manual_seed(42)
    device = _npu_device()
    n_splits = 1
    num_tokens = 32
    mhc_mult = 4
    hidden_size = 128
    mhc_mult3 = mhc_mult * (2 + mhc_mult)

    gemm_out_mul = torch.randn(
        (n_splits, num_tokens, mhc_mult3),
        device=device,
        dtype=torch.float32,
    )
    gemm_out_sqrsum = torch.rand(
        (n_splits, num_tokens), device=device, dtype=torch.float32
    ) * (mhc_mult * hidden_size)
    mhc_scale = torch.randn((3,), device=device, dtype=torch.float32)
    mhc_base = torch.randn((mhc_mult3,), device=device, dtype=torch.float32)
    residual = torch.randn(
        (num_tokens, mhc_mult, hidden_size),
        device=device,
        dtype=torch.bfloat16,
    )

    actual = mhc_pre_big_fuse(
        gemm_out_mul,
        gemm_out_sqrsum,
        mhc_scale,
        mhc_base,
        residual,
    )
    expected = mhc_pre_big_fuse_ref(
        gemm_out_mul,
        gemm_out_sqrsum,
        mhc_scale,
        mhc_base,
        residual,
    )
    torch.npu.synchronize()
    torch.testing.assert_close(actual[0], expected[0], rtol=1e-4, atol=2e-5)
    torch.testing.assert_close(actual[1], expected[1], rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(actual[2], expected[2], rtol=2e-2, atol=2e-2)
    print("PASS: mhc_pre_big_fuse")


if __name__ == "__main__":
    main()
