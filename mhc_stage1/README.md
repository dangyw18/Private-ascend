# MHC Stage-1 SimtVF kernels

This directory contains explicit Ascend rewrites of the Stage-1 MHC forward
kernels:

- `expand_fwd_simtvf.py`: MHC expansion with a one-dimensional Ascend grid.
- `normw_merge_fwd_simtvf.py`: norm-weight merge with a one-dimensional grid.
- `head_compute_mix_fwd.py`: one direct-global SimtVF region.
- `pre_split_mixes_fwd.py`: GM-to-UB copies, one fused compute SimtVF, then
  UB-to-GM copies.
- `sinkhorn_fwd.py`: the known-good `mhc_case.txt` Sinkhorn structure.
- `post_fwd.py`: serial hidden tiling with MTE copies outside, and one SimtVF
  compute region per tile.
- `pre_big_fuse_fwd_simtvf.py`: conservative port of the fused post-GEMM path.

Every operator file follows `example.py`: a TileLang program builder containing
one `@T.prim_func main`, one `ref_program`, and a direct correctness smoke test.
Run on a machine with the Ascend TileLang build and `torch_npu`, for example:

```bash
python mhc_stage1/expand_fwd_simtvf.py
python mhc_stage1/normw_merge_fwd_simtvf.py
python mhc_stage1/head_compute_mix_fwd.py
python mhc_stage1/pre_split_mixes_fwd.py
python mhc_stage1/sinkhorn_fwd.py
python mhc_stage1/post_fwd.py
python mhc_stage1/pre_big_fuse_fwd_simtvf.py
```

The pre-split and Sinkhorn smoke configurations require the flattened token
count to be divisible by `token_block_size`; their defaults satisfy this.

Use `test.py` to run any command through `msprof op` repeatedly and print the
mean profiler task duration:

```bash
python test.py --repeat 5 -- python sinkhorn_fwd.py
python test.py --repeat 5 -- python post_fwd.py
```

Options for `test.py` must appear before the target command. Use `--dry-run` to
inspect the generated `msprof op` command without executing it.
