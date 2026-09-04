# MHC Stage-1 SimtVF kernels

This directory contains explicit Ascend rewrites of the four forward kernels
selected by the Stage-1 auto-SimtVF design:

- `head_compute_mix_fwd.py`: one direct-global SimtVF region.
- `pre_split_mixes_fwd.py`: GM-to-UB copies, one fused compute SimtVF, then
  UB-to-GM copies.
- `sinkhorn_fwd.py`: the known-good `mhc_case.txt` Sinkhorn structure.
- `post_fwd.py`: serial hidden tiling with MTE copies outside, and one SimtVF
  compute region per tile.

Every file includes the TileLang program builder, a callable operator wrapper,
a Torch reference, and a `main()` correctness smoke test. Run on a machine with
the Ascend TileLang build and `torch_npu`, for example:

```bash
python mhc_stage1/head_compute_mix_fwd.py
python mhc_stage1/pre_split_mixes_fwd.py
python mhc_stage1/sinkhorn_fwd.py
python mhc_stage1/post_fwd.py
```

The pre-split and Sinkhorn smoke configurations require the flattened token
count to be divisible by `token_block_size`; their defaults satisfy this.
