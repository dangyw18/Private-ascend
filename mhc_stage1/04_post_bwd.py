import math
import tilelang
import torch
from tilelang import language as T

def _mhc_post_bwd(mhc: int, hidden: int, n_thr: int = 128, h_blk: int = 256) -> tilelang.JITKernel:
    assert mhc == 4
    n = T.dynamic('num_tokens')
    h = hidden

    h_blk = math.gcd(hidden, h_blk)

    @T.prim_func
    def _mhc_post_bwd_kernel(
        dx: T.Tensor[(n, 4, h), T.bfloat16],
        a: T.Tensor[(n, 4, 4), T.float32],
        b: T.Tensor[(n, 4, h), T.bfloat16],
        c: T.Tensor[(n, 4), T.float32],
        d: T.Tensor[(n, h), T.bfloat16],
        da: T.Tensor[(n, 4, 4), T.float32],
        db: T.Tensor[(n, 4, h), T.bfloat16],
        dc: T.Tensor[(n, 4), T.float32],
        dd: T.Tensor[(n, h), T.bfloat16],
    ) -> None:
        with T.Kernel(n, threads=n_thr) as pid_n:
            dx_shared = T.alloc_shared((4, h_blk), T.bfloat16)
            b_shared = T.alloc_shared((4, h_blk), T.bfloat16)
            db_shared = T.alloc_shared((4, h_blk), T.bfloat16)
            d_shared = T.alloc_shared(h_blk, T.bfloat16)
            dd_shared = T.alloc_shared(h_blk, T.bfloat16)

            dx_local = T.alloc_fragment((4, h_blk), T.float32)
            b_local = T.alloc_fragment((4, h_blk), T.float32)
            db_local = T.alloc_fragment((4, h_blk), T.float32)
            d_local = T.alloc_fragment(h_blk, T.float32)
            dd_local = T.alloc_fragment(h_blk, T.float32)

            a_local = T.alloc_fragment((4, 4), T.float32)
            c_local = T.alloc_fragment(4, T.float32)
            T.copy(a[pid_n, 0, 0], a_local)
            T.copy(c[pid_n, 0], c_local)

            da_reducer = T.alloc_reducer((4, 4), T.float32, replication='all')
            dc_reducer = T.alloc_reducer(4, T.float32, replication='all')
            T.clear(da_reducer)
            T.clear(dc_reducer)

            for i0_h in T.Pipelined(T.ceildiv(h, h_blk), num_stages=3):
                T.copy(dx[pid_n, 0, i0_h * h_blk], dx_shared, disable_tma=True)
                T.copy(b[pid_n, 0, i0_h * h_blk], b_shared, disable_tma=True)
                T.copy(d[pid_n, i0_h * h_blk], d_shared, disable_tma=True)

                T.copy(dx_shared, dx_local)
                T.copy(b_shared, b_local)
                T.copy(d_shared, d_local)

                # da and db
                T.clear(db_local)
                for i_mhci in T.serial(4):
                    for i_mhco in T.serial(4):
                        for i1_h in T.Parallel(h_blk):
                            db_local[i_mhci, i1_h] += a_local[i_mhci, i_mhco] * dx_local[i_mhco, i1_h]
                            da_reducer[i_mhci, i_mhco] += b_local[i_mhci, i1_h] * dx_local[i_mhco, i1_h]

                # dc and dd
                T.clear(dd_local)
                for i_mhc in T.serial(4):
                    for i1_h in T.Parallel(h_blk):
                        dc_reducer[i_mhc] += d_local[i1_h] * dx_local[i_mhc, i1_h]
                        dd_local[i1_h] += c_local[i_mhc] * dx_local[i_mhc, i1_h]

                T.copy(db_local, db_shared)
                T.copy(dd_local, dd_shared)

                T.copy(db_shared, db[pid_n, 0, i0_h * h_blk], disable_tma=True)
                T.copy(dd_shared, dd[pid_n, i0_h * h_blk], disable_tma=True)

            T.finalize_reducer(da_reducer)
            T.finalize_reducer(dc_reducer)
            T.copy(da_reducer, da[pid_n, 0, 0])
            T.copy(dc_reducer, dc[pid_n, 0])

    return _mhc_post_bwd_kernel

def mhc_post_bwd(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    d_o: torch.Tensor,
    fuse_grad_acc: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = d_o.shape[0] * d_o.shape[1]
    mhc = d_o.shape[2]
    h = d_o.shape[3]

    bwd_kernel = _mhc_post_bwd(mhc, h)
    (
        d_comb_res_mix,
        d_residual,
        d_post_layer_mix,
        d_x,
    ) = bwd_kernel(
        d_o.contiguous().view(n, mhc, h),
        comb_res_mix.view(n, mhc, mhc),
        residual.view(n, mhc, h),
        post_layer_mix.view(n, mhc),
        x.view(n, h),
    )
    assert isinstance(d_x, torch.Tensor)
    assert isinstance(d_post_layer_mix, torch.Tensor)
    assert isinstance(d_comb_res_mix, torch.Tensor)
    assert isinstance(d_residual, torch.Tensor)
    if fuse_grad_acc:
        residual.untyped_storage().grad_from_mhc_post = d_residual

    return (
        d_x.view_as(x),
        d_residual.view_as(residual),
        d_post_layer_mix.view_as(post_layer_mix),
        d_comb_res_mix.view_as(comb_res_mix),
    )

def ref_program(
    dx: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dx = dx.float()
    da = torch.einsum('bih,boh->bio', b.float(), dx)
    db = torch.einsum('bio,boh->bih', a, dx).to(torch.bfloat16)
    dc = torch.einsum('bh,bih->bi', d.float(), dx)
    dd = torch.einsum('bi,bih->bh', c, dx).to(torch.bfloat16)
    return da, db, dc, dd

if __name__ == "__main__":
    torch.manual_seed(42)
    num_tokens, mhc, hidden = 128, 4, 256
    dx = torch.randn((num_tokens, mhc, hidden), device="npu", dtype=torch.bfloat16)
    a = torch.randn((num_tokens, mhc, mhc), device="npu", dtype=torch.float32)
    b = torch.randn((num_tokens, mhc, hidden), device="npu", dtype=torch.bfloat16)
    c = torch.randn((num_tokens, mhc), device="npu", dtype=torch.float32)
    d = torch.randn((num_tokens, hidden), device="npu", dtype=torch.bfloat16)

    program = _mhc_post_bwd(mhc, hidden)
    kernel = tilelang.compile(program, target="ascend", out_idx=[-4, -3, -2, -1])
    da, db, dc, dd = kernel(dx, a, b, c, d)
    torch.npu.synchronize()
    expected_da, expected_db, expected_dc, expected_dd = ref_program(dx, a, b, c, d)
    torch.testing.assert_close(da, expected_da, rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(db, expected_db, rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(dc, expected_dc, rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(dd, expected_dd, rtol=1e-5, atol=2e-5)
    print("PASS: mhc_post_bwd")
