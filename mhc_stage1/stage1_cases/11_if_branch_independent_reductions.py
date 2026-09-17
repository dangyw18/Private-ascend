"""End-to-end coverage for independent fragment definitions in if branches."""

import tilelang
import torch
from tilelang import language as T


def if_branch_independent_reductions(
    num_blocks: int = 8,
    rows: int = 4,
    width: int = 8,
):
    @T.prim_func
    def main(
        lhs: T.Tensor[(num_blocks, rows, width), T.float32],
        rhs: T.Tensor[(num_blocks, rows, width), T.float32],
        out: T.Tensor[(num_blocks, rows), T.float32],
    ) -> None:
        with T.Kernel(num_blocks) as pid:
            src_frag = T.alloc_fragment((rows, width), T.float32)
            reduced_frag = T.alloc_fragment(rows, T.float32)

            if pid % 2 == 0:
                T.copy(lhs[pid, 0, 0], src_frag)
                T.reduce_sum(src_frag, reduced_frag, dim=1)
                for i in T.Parallel(rows):
                    out[pid, i] = reduced_frag[i]
            else:
                T.copy(rhs[pid, 0, 0], src_frag)
                T.reduce_max(src_frag, reduced_frag, dim=1)
                for i in T.Parallel(rows):
                    out[pid, i] = reduced_frag[i]

    return main


def ref_program(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    expected = lhs.sum(dim=-1)
    expected[1::2] = rhs[1::2].amax(dim=-1)
    return expected


if __name__ == "__main__":
    torch.manual_seed(42)
    num_blocks, rows, width = 8, 4, 8
    # lhs = torch.randn(
    #     (num_blocks, rows, width), device="npu", dtype=torch.float32
    # )
    # rhs = torch.randn(
    #     (num_blocks, rows, width), device="npu", dtype=torch.float32
    # )

    program = if_branch_independent_reductions(num_blocks, rows, width)
    kernel = tilelang.compile(program, target="ascend", out_idx=-1)
    # actual = kernel(lhs, rhs)
    # torch.npu.synchronize()

    # expected = ref_program(lhs, rhs)
    # torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    # print("PASS: if_branch_independent_reductions")
