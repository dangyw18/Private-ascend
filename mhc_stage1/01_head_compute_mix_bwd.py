import tilelang
import torch
from tilelang import language as T

def mhc_head_compute_mix_bwd(
    mhc_mult: int,
    token_block_size: int,
    num_sms: int,
) -> tilelang.JITKernel:
    num_tokens = T.dynamic('num_tokens')

    @T.prim_func
    def mhc_head_compute_mix_bwd_kernel(
        # Gradient of output
        output_mix_grad: T.Tensor[(num_tokens, mhc_mult), T.float32],
        # Cached activation
        input_mix: T.Tensor[(num_tokens, mhc_mult), T.float32],
        mhc_scale: T.Tensor[(1,), T.float32],
        mhc_base: T.Tensor[(mhc_mult,), T.float32],
        # Gradient of input
        input_mix_grad: T.Tensor[(num_tokens, mhc_mult), T.float32],
        mhc_scale_grad_partial: T.Tensor[(num_sms, 1), T.float32],
        mhc_base_grad_partial: T.Tensor[(num_sms, mhc_mult), T.float32],
    ) -> None:
        with T.Kernel(num_sms) as pid:
            mhc_scale_grad_reducer = T.alloc_reducer(1, T.float32, replication='all')
            mhc_base_grad_reducer = T.alloc_reducer(mhc_mult, T.float32, replication='all')
            T.fill(mhc_scale_grad_reducer, 0)
            T.fill(mhc_base_grad_reducer, 0)
            for t in T.Persistent(
                [T.ceildiv(num_tokens, token_block_size)],
                num_sms,
                pid,
                group_size=1,
            ):
                grad_frag = T.alloc_fragment((token_block_size, mhc_mult), T.float32)
                input_recompute_frag = T.alloc_fragment((token_block_size, mhc_mult), T.float32)
                for i1, j in T.Parallel(token_block_size, mhc_mult):
                    i = t * token_block_size + i1
                    if i < num_tokens:
                        input_recompute_frag[i1, j] = T.sigmoid(
                            input_mix[i, j] * mhc_scale[0] + mhc_base[j],
                        )
                        grad_frag[i1, j] = input_recompute_frag[i1, j] * (1 - input_recompute_frag[i1, j]) * output_mix_grad[i, j]
                        input_mix_grad[i, j] = grad_frag[i1, j] * mhc_scale[0]
                        mhc_scale_grad_reducer[0] += grad_frag[i1, j] * input_mix[i, j]
                        mhc_base_grad_reducer[j] += grad_frag[i1, j]
            T.finalize_reducer(mhc_scale_grad_reducer)
            T.finalize_reducer(mhc_base_grad_reducer)
            T.copy(mhc_scale_grad_reducer, mhc_scale_grad_partial[pid, :])
            T.copy(mhc_base_grad_reducer, mhc_base_grad_partial[pid, :])

    return mhc_head_compute_mix_bwd_kernel

def gpu_mhc_head_compute_mix_bwd(
    mhc_mult: int,
    token_block_size: int,
    num_sms: int,
) -> tilelang.JITKernel:
    num_tokens = T.dynamic('num_tokens')

    @T.prim_func
    def gpu_mhc_head_compute_mix_bwd_kernel(
        # Gradient of output
        output_mix_grad: T.Tensor[(num_tokens, mhc_mult), T.float32],
        # Cached activation
        input_mix: T.Tensor[(num_tokens, mhc_mult), T.float32],
        mhc_scale: T.Tensor[(1,), T.float32],
        mhc_base: T.Tensor[(mhc_mult,), T.float32],
        # Gradient of input
        input_mix_grad: T.Tensor[(num_tokens, mhc_mult), T.float32],
        mhc_scale_grad_partial: T.Tensor[(num_sms, 1), T.float32],
        mhc_base_grad_partial: T.Tensor[(num_sms, mhc_mult), T.float32],
    ) -> None:
        with T.Kernel(num_sms) as pid:
            mhc_scale_grad_reducer = T.alloc_reducer(1, T.float32, replication='all')
            mhc_base_grad_reducer = T.alloc_reducer(mhc_mult, T.float32, replication='all')
            T.fill(mhc_scale_grad_reducer, 0)
            T.fill(mhc_base_grad_reducer, 0)
            for t in T.Persistent(
                [T.ceildiv(num_tokens, token_block_size)],
                num_sms,
                pid,
                group_size=1,
            ):
                grad_frag = T.alloc_fragment((token_block_size, mhc_mult), T.float32)
                input_recompute_frag = T.alloc_fragment((token_block_size, mhc_mult), T.float32)
                for i1, j in T.Parallel(token_block_size, mhc_mult):
                    i = t * token_block_size + i1
                    if i < num_tokens:
                        input_recompute_frag[i1, j] = T.sigmoid(
                            input_mix[i, j] * mhc_scale[0] + mhc_base[j],
                        )
                        grad_frag[i1, j] = input_recompute_frag[i1, j] * (1 - input_recompute_frag[i1, j]) * output_mix_grad[i, j]
                        input_mix_grad[i, j] = grad_frag[i1, j] * mhc_scale[0]
                        mhc_scale_grad_reducer[0] += grad_frag[i1, j] * input_mix[i, j]
                        mhc_base_grad_reducer[j] += grad_frag[i1, j]
            T.finalize_reducer(mhc_scale_grad_reducer)
            T.finalize_reducer(mhc_base_grad_reducer)
            T.copy(mhc_scale_grad_reducer, mhc_scale_grad_partial[pid, :])
            T.copy(mhc_base_grad_reducer, mhc_base_grad_partial[pid, :])

    return gpu_mhc_head_compute_mix_bwd_kernel

def ref_program(
    output_mix_grad: torch.Tensor,
    input_mix: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sigmoid = torch.sigmoid(input_mix * mhc_scale[0] + mhc_base)
    grad = sigmoid * (1 - sigmoid) * output_mix_grad
    return grad * mhc_scale[0], (grad * input_mix).sum(dim=0), grad.sum(dim=0)


if __name__ == "__main__":
    torch.manual_seed(42)
    mhc_mult = 4
    num_tokens = 128
    token_block_size = 32
    num_sms = 1
    output_mix_grad = torch.randn((num_tokens, mhc_mult), device="npu", dtype=torch.float32)
    input_mix = torch.randn((num_tokens, mhc_mult), device="npu", dtype=torch.float32)
    mhc_scale = torch.randn((1,), device="npu", dtype=torch.float32)
    mhc_base = torch.randn((mhc_mult,), device="npu", dtype=torch.float32)

    program = mhc_head_compute_mix_bwd(mhc_mult, token_block_size, num_sms)
    kernel = tilelang.compile(program, target="ascend", out_idx=[-3, -2, -1])
    input_mix_grad, mhc_scale_grad_partial, mhc_base_grad_partial = kernel(
        output_mix_grad, input_mix, mhc_scale, mhc_base
    )
    torch.npu.synchronize()
    expected_input_mix_grad, expected_scale_grad, expected_base_grad = ref_program(
        output_mix_grad, input_mix, mhc_scale, mhc_base
    )
    torch.testing.assert_close(input_mix_grad, expected_input_mix_grad, rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(mhc_scale_grad_partial.sum(dim=0), expected_scale_grad, rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(mhc_base_grad_partial.sum(dim=0), expected_base_grad, rtol=1e-5, atol=2e-5)
    print("PASS: mhc_head_compute_mix_bwd")

    gpu_program = gpu_mhc_head_compute_mix_bwd(mhc_mult, token_block_size, num_sms)
    gpu_kernel = tilelang.compile(gpu_program, target="ascend", out_idx=[-3, -2, -1])
    # gpu_actual = gpu_kernel(output_mix_grad, input_mix, mhc_scale, mhc_base)
    # gpu_expected = ref_program(output_mix_grad, input_mix, mhc_scale, mhc_base)
    # torch.npu.synchronize()
    # torch.testing.assert_close(gpu_actual[0], gpu_expected[0], rtol=1e-5, atol=2e-5)
    # torch.testing.assert_close(gpu_actual[1].sum(dim=0), gpu_expected[1], rtol=1e-5, atol=2e-5)
    # torch.testing.assert_close(gpu_actual[2].sum(dim=0), gpu_expected[2], rtol=1e-5, atol=2e-5)
    # print("PASS: gpu_mhc_head_compute_mix_bwd")