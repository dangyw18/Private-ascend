import tilelang
import torch
from tilelang import language as T


def _mhc_pre_norm_fn_bwd_norm(
    mhc_mult3: int,
    n_rms_group: int,
    rms_group_size: int,
    rms_eps: float,
) -> tilelang.JITKernel:
    num_tokens = T.dynamic('num_tokens')
    n_thr = 32

    @T.prim_func
    def _mhc_pre_norm_fn_bwd_norm_kernel(
        # Gradient of output
        out_grad: T.Tensor[(num_tokens, mhc_mult3), T.float32],
        # Saved inputs
        out_mul: T.Tensor[(num_tokens, n_rms_group, mhc_mult3), T.float32],
        sqrsum: T.Tensor[(num_tokens, n_rms_group), T.float32],
        # Computed gradient of inputs
        out_mul_grad: T.Tensor[(num_tokens, n_rms_group, mhc_mult3), T.float32],
        sqrsum_grad: T.Tensor[(num_tokens, n_rms_group), T.float32],
    ) -> None:
        with T.Kernel(num_tokens, n_rms_group, threads=n_thr) as (pid_i, pid_k):
            sqrsum_frag = T.alloc_fragment(1, T.float32)
            sqrsum_frag[0] = sqrsum[pid_i, pid_k]
            rms_frag = T.alloc_fragment(1, T.float32)
            rms_frag[0] = T.rsqrt(sqrsum_frag[0] / rms_group_size + rms_eps)

            rms_grad_frag = T.alloc_reducer(1, T.float32, replication='all')
            T.clear(rms_grad_frag)
            for j in T.Parallel(mhc_mult3):
                out_mul_grad[pid_i, pid_k, j] = out_grad[pid_i, j] * rms_frag[0]
                rms_grad_frag[0] += out_grad[pid_i, j] * out_mul[pid_i, pid_k, j]
            T.finalize_reducer(rms_grad_frag)

            for kk in T.Parallel(1):
                sqrsum_grad[pid_i, pid_k + kk] = rms_grad_frag[kk] * rms_frag[kk] / (sqrsum_frag[kk] + rms_eps * rms_group_size) / -2

    return _mhc_pre_norm_fn_bwd_norm_kernel

def ref_program(
    out_grad: torch.Tensor,
    out_mul: torch.Tensor,
    sqrsum: torch.Tensor,
    rms_group_size: int,
    rms_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    rms = torch.rsqrt(sqrsum / rms_group_size + rms_eps)
    out_mul_grad = out_grad.unsqueeze(1) * rms.unsqueeze(-1)
    rms_grad = (out_grad.unsqueeze(1) * out_mul).sum(dim=-1)
    sqrsum_grad = rms_grad * rms / (sqrsum + rms_eps * rms_group_size) / -2
    return out_mul_grad, sqrsum_grad

if __name__ == "__main__":
    torch.manual_seed(42)
    num_tokens, mhc_mult3 = 128, 12
    n_rms_group, rms_group_size = 4, 3
    rms_eps = 1e-6
    out_grad = torch.randn((num_tokens, mhc_mult3), device="npu", dtype=torch.float32)
    out_mul = torch.randn((num_tokens, n_rms_group, mhc_mult3), device="npu", dtype=torch.float32)
    sqrsum = torch.rand((num_tokens, n_rms_group), device="npu", dtype=torch.float32) + 0.1

    program = _mhc_pre_norm_fn_bwd_norm(mhc_mult3, n_rms_group, rms_group_size, rms_eps)
    kernel = tilelang.compile(program, target="ascend", out_idx=[-2, -1])
    out_mul_grad, sqrsum_grad = kernel(out_grad, out_mul, sqrsum)
    torch.npu.synchronize()
    expected_out_mul_grad, expected_sqrsum_grad = ref_program(out_grad, out_mul, sqrsum, rms_group_size, rms_eps)
    torch.testing.assert_close(out_mul_grad, expected_out_mul_grad, rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(sqrsum_grad, expected_sqrsum_grad, rtol=1e-5, atol=2e-5)
    print("PASS: mhc_pre_norm_fn_bwd_norm")