import tilelang
import torch
from tilelang import language as T


def gpu_mhc_pre_norm_fn_fwd_mul(
    mhc_mult3: int,
    n_rms_group: int,
    rms_group_size: int,
    token_block: int = 32,
    hidden_block: int = 256,
) -> tilelang.JITKernel:
    assert mhc_mult3 <= 32
    num_tokens = T.dynamic('num_tokens')
    assert rms_group_size % hidden_block == 0

    @T.prim_func
    def gpu_mhc_pre_norm_fn_fwd_mul_kernel(
        x: T.Tensor[(num_tokens, n_rms_group * rms_group_size), T.bfloat16],
        fn: T.Tensor[(mhc_mult3, n_rms_group * rms_group_size), T.float32],
        out: T.Tensor[(num_tokens, n_rms_group, mhc_mult3), T.float32],
        sqrsum: T.Tensor[(num_tokens, n_rms_group), T.float32],
    ) -> None:
        _ = mhc_mult3
        with T.Kernel(T.ceildiv(num_tokens, token_block), n_rms_group) as (pid_x, pid_y):
            out_frag = T.alloc_fragment((token_block, 32), T.float32)
            sqrsum_part = T.alloc_fragment((token_block, 4), T.float32)
            T.clear(out_frag)
            T.clear(sqrsum_part)
            for pz in T.Pipelined(rms_group_size // hidden_block, num_stages=2):
                x_smem_16 = T.alloc_shared((token_block, hidden_block), T.bfloat16)
                fn_smem = T.alloc_shared((32, hidden_block), T.float32)

                T.annotate_layout({x_smem_16: tilelang.layout.make_swizzled_layout(x_smem_16)})

                T.copy(x[pid_x * token_block, pid_y * rms_group_size + pz * hidden_block], x_smem_16)
                T.copy(fn[0, pid_y * rms_group_size + pz * hidden_block], fn_smem)

                x_frag_16 = T.alloc_fragment((token_block, hidden_block), T.bfloat16)
                T.copy(x_smem_16, x_frag_16)
                x_frag = T.alloc_fragment((token_block, hidden_block), T.float32)
                T.copy(x_frag_16, x_frag)

                for jj in T.serial(hidden_block // 4):
                    for i, j in T.Parallel(token_block, 4):
                        sqrsum_part[i, j] += x_frag[i, jj * 4 + j] * x_frag[i, jj * 4 + j]

                T.gemm(
                    x_frag,
                    fn_smem,
                    out_frag,
                    transpose_A=False,
                    transpose_B=True,
                    clear_accum=False,
                )
            sqrsum_l = T.alloc_fragment(token_block, T.float32)
            T.reduce_sum(sqrsum_part, sqrsum_l)
            for i in T.Parallel(token_block):
                sqrsum[pid_x * token_block + i, pid_y] = sqrsum_l[i]
            for i, j in T.Parallel(token_block, 32):
                if j < 24:
                    out[pid_x * token_block + i, pid_y, j] = out_frag[i, j]

    return gpu_mhc_pre_norm_fn_fwd_mul_kernel

def ref_program(
    x: torch.Tensor,
    fn: torch.Tensor,
    n_rms_group: int,
    rms_group_size: int,
):
    num_tokens = x.shape[0]
    xr = x.to(torch.float32).reshape(num_tokens, n_rms_group, rms_group_size)
    fnr = fn.to(torch.float32).reshape(-1, n_rms_group, rms_group_size)
    out = torch.einsum("tgh,jgh->tgj", xr, fnr)
    sqrsum = (xr * xr).sum(dim=-1)
    return out, sqrsum


if __name__ == "__main__":
    torch.manual_seed(42)
    num_tokens = 128
    mhc_mult3, n_rms_group, rms_group_size = 16, 256, 256
    token_block, hidden_block = 32, 256

    x = torch.randn((num_tokens, n_rms_group * rms_group_size), device="npu", dtype=torch.bfloat16)
    fn = torch.randn((mhc_mult3, n_rms_group * rms_group_size), device="npu", dtype=torch.float32)

    program = gpu_mhc_pre_norm_fn_fwd_mul(mhc_mult3, n_rms_group, rms_group_size, token_block, hidden_block)
    kernel = tilelang.compile(program, target="ascend", out_idx=[2, 3])
    actual_out, actual_sqrsum = kernel(x, fn)
    expected_out, expected_sqrsum = ref_program(x, fn, n_rms_group, rms_group_size)
    torch.npu.synchronize()
    torch.testing.assert_close(actual_out, expected_out, rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(actual_sqrsum, expected_sqrsum, rtol=1e-5, atol=2e-5)
    print("PASS: mhc_pre_norm_fn_fwd_mul")