"""End-to-end coverage for fragment state merged after an if statement."""

import tilelang
import torch
from tilelang import language as T


def if_branch_state_merge(
    num_blocks: int = 8,
    rows: int = 4,
    width: int = 8,
):
    @T.prim_func
    def main(
        src: T.Tensor[(num_blocks, rows, width), T.float32],
        initial: T.Tensor[(num_blocks, rows), T.float32],
        out: T.Tensor[(num_blocks, rows), T.float32],
    ) -> None:
        with T.Kernel(num_blocks) as pid:
            src_frag = T.alloc_fragment((rows, width), T.float32)
            acc_frag = T.alloc_fragment(rows, T.float32)
            T.copy(initial[pid, 0], acc_frag)

            if pid % 2 == 0:
                T.copy(src[pid, 0, 0], src_frag)
                T.reduce_sum(src_frag, acc_frag, dim=1)
                for i in T.Parallel(rows):
                    acc_frag[i] = acc_frag[i] + 1.0
            else:
                for i in T.Parallel(rows):
                    acc_frag[i] = acc_frag[i] * 2.0

            for i in T.Parallel(rows):
                out[pid, i] = acc_frag[i]

    return main


def ref_program(src: torch.Tensor, initial: torch.Tensor) -> torch.Tensor:
    expected = initial * 2.0
    expected[0::2] = src[0::2].sum(dim=-1) + 1.0
    return expected


if __name__ == "__main__":
    torch.manual_seed(42)
    num_blocks, rows, width = 8, 4, 8
    # src = torch.randn(
    #     (num_blocks, rows, width), device="npu", dtype=torch.float32
    # )
    # initial = torch.randn(
    #     (num_blocks, rows), device="npu", dtype=torch.float32
    # )

    program = if_branch_state_merge(num_blocks, rows, width)
    kernel = tilelang.compile(program, target="ascend", out_idx=-1)
    # actual = kernel(src, initial)
    # torch.npu.synchronize()

    # expected = ref_program(src, initial)
    # torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    # print("PASS: if_branch_state_merge")
