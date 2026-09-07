import tilelang
import torch
from tilelang import language as T


def _mhc_sinkhorn_fwd(
    hidden_size: int,
    token_block_size: int,
    repeat: int,
    eps: float,
    threads: int = 256,
):
    num_tokens = T.dynamic('num_tokens')

    @T.prim_func
    def main(
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

    return main


def ref_program(
    x: torch.Tensor,
    repeat: int = 10,
    eps: float = 1e-6,
) -> torch.Tensor:
    output = torch.softmax(x, dim=-1) + eps
    output = output / (output.sum(dim=-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        output = output / (output.sum(dim=-1, keepdim=True) + eps)
        output = output / (output.sum(dim=-2, keepdim=True) + eps)
    return output


if __name__ == "__main__":
    torch.manual_seed(42)
    num_tokens, hidden_size = 64, 4
    token_block_size = 1
    repeat = 10
    eps = 1e-6
    comb_res_mix = torch.randn(
        (num_tokens, hidden_size, hidden_size), device="npu", dtype=torch.float32
    )

    program = _mhc_sinkhorn_fwd(hidden_size, token_block_size, repeat, eps)
    kernel = tilelang.compile(program, target="ascend", out_idx=-1)
    actual = kernel(comb_res_mix)
    expected = ref_program(comb_res_mix, repeat, eps)
    torch.npu.synchronize()
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    print("PASS: mhc_sinkhorn_fwd")
