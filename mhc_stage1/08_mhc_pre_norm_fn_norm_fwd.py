import tilelang
import torch
from tilelang import language as T


def mhc_pre_norm_fn_fwd_norm(
    mhc_mult3: int,
    n_rms_group: int,
    rms_group_size: int,
    rms_eps: float,
    n_splits: int,
) -> tilelang.JITKernel:
    num_tokens = T.dynamic('num_tokens')
    n_thr = 32

    @T.prim_func
    def mhc_pre_norm_fn_fwd_norm_kernel(
        out_mul_splitted: T.Tensor[(n_splits, num_tokens, n_rms_group, mhc_mult3), T.float32],
        sqrsum_splitted: T.Tensor[(n_splits, num_tokens, n_rms_group), T.float32],
        out_mul: T.Tensor[(num_tokens, n_rms_group, mhc_mult3), T.float32],
        sqrsum: T.Tensor[(num_tokens, n_rms_group), T.float32],
        out: T.Tensor[(num_tokens, mhc_mult3), T.float32],
    ) -> None:
        with T.Kernel(num_tokens, threads=n_thr) as pid:
            rms = T.alloc_local(1, T.float32)
            out_l = T.alloc_fragment(mhc_mult3, T.float32)
            out_l0 = T.alloc_fragment(mhc_mult3, T.float32)
            T.clear(out_l)
            for k in T.serial(n_rms_group):
                rms[0] = 0
                for i_split in T.serial(n_splits):
                    rms[0] += sqrsum_splitted[i_split, pid, k]
                if T.get_thread_binding() == 0:
                    sqrsum[pid, k] = rms[0]
                rms[0] = T.rsqrt(rms[0] / rms_group_size + rms_eps)
                for j in T.Parallel(mhc_mult3):
                    out_l0[j] = 0
                    for i_split in T.serial(n_splits):
                        out_l0[j] += out_mul_splitted[i_split, pid, k, j]
                    out_l[j] += out_l0[j] * rms[0]
                T.copy(out_l0, out_mul[pid, k, :])
            T.copy(out_l[:], out[pid, :])

    return mhc_pre_norm_fn_fwd_norm_kernel

def gpu_mhc_pre_norm_fn_fwd_norm(
    mhc_mult3: int,
    n_rms_group: int,
    rms_group_size: int,
    rms_eps: float,
    n_splits: int,
) -> tilelang.JITKernel:
    num_tokens = T.dynamic('num_tokens')
    n_thr = 32

    @T.prim_func
    def gpu_mhc_pre_norm_fn_fwd_norm_kernel(
        out_mul_splitted: T.Tensor[(n_splits, num_tokens, n_rms_group, mhc_mult3), T.float32],
        sqrsum_splitted: T.Tensor[(n_splits, num_tokens, n_rms_group), T.float32],
        out_mul: T.Tensor[(num_tokens, n_rms_group, mhc_mult3), T.float32],
        sqrsum: T.Tensor[(num_tokens, n_rms_group), T.float32],
        out: T.Tensor[(num_tokens, mhc_mult3), T.float32],
    ) -> None:
        with T.Kernel(num_tokens, threads=n_thr) as pid:
            rms = T.alloc_fragment(1, T.float32)
            out_l = T.alloc_fragment(mhc_mult3, T.float32)
            out_l0 = T.alloc_fragment(mhc_mult3, T.float32)
            T.clear(out_l)
            for k in T.serial(n_rms_group):
                rms[0] = 0
                for i_split in T.serial(n_splits):
                    rms[0] += sqrsum_splitted[i_split, pid, k]
                if T.get_thread_binding() == 0:
                    sqrsum[pid, k] = rms[0]
                rms[0] = T.rsqrt(rms[0] / rms_group_size + rms_eps)
                for j in T.Parallel(mhc_mult3):
                    out_l0[j] = 0
                    for i_split in T.serial(n_splits):
                        out_l0[j] += out_mul_splitted[i_split, pid, k, j]
                    out_l[j] += out_l0[j] * rms[0]
                T.copy(out_l0, out_mul[pid, k, :])
            T.copy(out_l[:], out[pid, :])

    return gpu_mhc_pre_norm_fn_fwd_norm_kernel


def ref_program(
    out_mul_splitted: torch.Tensor,
    sqrsum_splitted: torch.Tensor,
    rms_group_size: int,
    rms_eps: float,
):
    out_mul = out_mul_splitted.sum(dim=0)
    sqrsum = sqrsum_splitted.sum(dim=0)
    rms = torch.rsqrt(sqrsum / rms_group_size + rms_eps)
    out = (out_mul * rms.unsqueeze(-1)).sum(dim=1)
    return out_mul, sqrsum, out


if __name__ == "__main__":
    torch.manual_seed(42)

    num_tokens = 128
    mhc_mult3 = 4
    n_rms_group = 8
    rms_group_size = 896
    rms_eps = 1e-6
    n_splits = 4

    out_mul_splitted = torch.randn((n_splits, num_tokens, n_rms_group, mhc_mult3), device="npu", dtype=torch.float32)
    # sqrsum 来自平方和，用正随机数保证 rsqrt 有定义（负值会得到 NaN）
    sqrsum_splitted = torch.rand((n_splits, num_tokens, n_rms_group), device="npu", dtype=torch.float32)

    program = mhc_pre_norm_fn_fwd_norm(mhc_mult3, n_rms_group, rms_group_size, rms_eps, n_splits)
    kernel = tilelang.compile(program, target="ascend", out_idx=[2, 3, 4])
    actual_out_mul, actual_sqrsum, actual_out = kernel(out_mul_splitted, sqrsum_splitted)
    expected_out_mul, expected_sqrsum, expected_out = ref_program(
        out_mul_splitted, sqrsum_splitted, rms_group_size, rms_eps
    )
    torch.npu.synchronize()
    torch.testing.assert_close(actual_out_mul, expected_out_mul, rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(actual_sqrsum, expected_sqrsum, rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(actual_out, expected_out, rtol=1e-5, atol=2e-5)
    print("PASS: mhc_pre_norm_fn_fwd_norm")