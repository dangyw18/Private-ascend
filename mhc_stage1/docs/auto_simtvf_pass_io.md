# AutoSimtVF Pass 目标输入输出（Case 01 / 03 / 04）

- **Pass 位置**：09 After Simplify 与 10 After UnrollLoopSkipVF 之间
- **输入**：`*_tvm.log` 中 09 After Simplify 的 TVMIR（未划 SIMT VF，后续 VFChecker 报错）
- **输出**：`*_simtvf_ascend_tvm.log` 中 09 After Simplify 的 TVMIR（手动 DSL 加 simtvf 的结果）

## Case 01: head_compute_mix_fwd

### 输入

```python
@I.ir_module
class Module:
    @T.prim_func
    def gpu_mhc_head_compute_mix_fwd_kernel(input_mix_handle: T.handle, mhc_scale_handle: T.handle, mhc_base_handle: T.handle, output_mix_handle: T.handle):
        T.func_attr({"target": T.target({"host": {"keys": ["cpu"], "kind": "c", "tag": ""}, "keys": ["ascend"], "kind": "ascend", "tag": ""})})
        num_tokens = T.int32()
        input_mix = T.match_buffer(input_mix_handle, (num_tokens, 4), strides=(4, 1))
        mhc_scale = T.match_buffer(mhc_scale_handle, (1,), strides=(1,))
        mhc_base = T.match_buffer(mhc_base_handle, (4,), strides=(1,))
        output_mix = T.match_buffer(output_mix_handle, (num_tokens, 4), strides=(4, 1))
        # with T.sblock("root"):
        T.attr(T.bool(True), "tl.assume", "Buffer shape should be greater than or equal to 0: shape `num_tokens` from buffer `input_mix`, `output_mix`")
        bx = T.launch_thread("blockIdx.x", (num_tokens + 31) // 32)
        with T.sblock("tilelang_root"):
            T.reads()
            T.writes()
            T.sblock_attr({"tilelang.is_npu_kernel_frame": True})
            for i1 in T.parallel(32):
                for j in T.parallel(4):
                    if bx * 32 + i1 < num_tokens:
                        output_mix[bx * 32 + i1, j] = T.sigmoid(input_mix[bx * 32 + i1, j] * mhc_scale[0] + mhc_base[j]) + T.float32(9.9999999999999995e-07)
```

### 输出

```python
@I.ir_module
class Module:
    @T.prim_func
    def gpu_mhc_head_compute_mix_fwd_kernel(input_mix_handle: T.handle, mhc_scale_handle: T.handle, mhc_base_handle: T.handle, output_mix_handle: T.handle):
        T.func_attr({"target": T.target({"host": {"keys": ["cpu"], "kind": "c", "tag": ""}, "keys": ["ascend"], "kind": "ascend", "tag": ""})})
        num_tokens = T.int32()
        input_mix = T.match_buffer(input_mix_handle, (num_tokens, 4), strides=(4, 1))
        mhc_scale = T.match_buffer(mhc_scale_handle, (1,), strides=(1,))
        mhc_base = T.match_buffer(mhc_base_handle, (4,), strides=(1,))
        output_mix = T.match_buffer(output_mix_handle, (num_tokens, 4), strides=(4, 1))
        # with T.sblock("root"):
        T.attr(T.bool(True), "tl.assume", "Buffer shape should be greater than or equal to 0: shape `num_tokens` from buffer `input_mix`, `output_mix`")
        bx = T.launch_thread("blockIdx.x", (num_tokens + 31) // 32)
        with T.sblock("tilelang_root"):
            T.reads()
            T.writes()
            T.sblock_attr({"tilelang.is_npu_kernel_frame": True})
            with T.sblock("SIMT_VF", no_realize=True):
                T.reads()
                T.writes()
                T.sblock_attr({"tl.vf_source_index": T.int64(0)})
                simtvf_tx = T.launch_thread("threadIdx.x", 128)
                simtvf_ty = T.launch_thread("threadIdx.y", 1)
                simtvf_tz = T.launch_thread("threadIdx.z", 1)
                T.attr("simtvf", "tl.simtvf_scope", 1)
                for i1 in T.parallel(32):
                    for j in T.parallel(4):
                        if bx * 32 + i1 < num_tokens:
                            output_mix[bx * 32 + i1, j] = T.sigmoid(input_mix[bx * 32 + i1, j] * mhc_scale[0] + mhc_base[j]) + T.float32(9.9999999999999995e-07)
```

## Case 03: expand_to_mhc_fwd

### 输入

```python
@I.ir_module
class Module:
    @T.prim_func
    def gpu_expand_to_mhc_fwd_kernel(x_handle: T.handle, o_handle: T.handle):
        T.func_attr({"target": T.target({"host": {"keys": ["cpu"], "kind": "c", "tag": ""}, "keys": ["ascend"], "kind": "ascend", "tag": ""})})
        num_tokens = T.int32()
        x = T.match_buffer(x_handle, (num_tokens, 1280), "bfloat16", strides=(1280, 1))
        o = T.match_buffer(o_handle, (num_tokens, 4, 1280), "bfloat16", strides=(5120, 1280, 1))
        # with T.sblock("root"):
        T.attr(T.bool(True), "tl.assume", "Buffer shape should be greater than or equal to 0: shape `num_tokens` from buffer `x`, `o`")
        bx = T.launch_thread("blockIdx.x", (num_tokens + 31) // 32 * 10)
        with T.sblock("tilelang_root"):
            T.reads()
            T.writes()
            T.sblock_attr({"tilelang.is_npu_kernel_frame": True})
            xl = T.sblock_alloc_buffer((32, 128), "bfloat16", scope="local.fragment")
            if 0 < num_tokens:
                T.copy(T.region(x[bx // 10 * 32, bx % 10 * 128], 1, 32, 128), T.region(xl[0, 0], 2, 32, 128))
                for m in range(4):
                    for ti in T.parallel(32):
                        for tj in T.parallel(128):
                            if bx // 10 * 32 + ti < num_tokens:
                                o[bx // 10 * 32 + ti, m, bx % 10 * 128 + tj] = xl[ti, tj]
```

### 输出

```python
@I.ir_module
class Module:
    @T.prim_func
    def gpu_expand_to_mhc_fwd_kernel(x_handle: T.handle, o_handle: T.handle):
        T.func_attr({"target": T.target({"host": {"keys": ["cpu"], "kind": "c", "tag": ""}, "keys": ["ascend"], "kind": "ascend", "tag": ""})})
        num_tokens = T.int32()
        x = T.match_buffer(x_handle, (num_tokens, 1280), "bfloat16", strides=(1280, 1))
        o = T.match_buffer(o_handle, (num_tokens, 4, 1280), "bfloat16", strides=(5120, 1280, 1))
        # with T.sblock("root"):
        T.attr(T.bool(True), "tl.assume", "Buffer shape should be greater than or equal to 0: shape `num_tokens` from buffer `x`, `o`")
        bx = T.launch_thread("blockIdx.x", (num_tokens + 31) // 32 * 10)
        with T.sblock("tilelang_root"):
            T.reads()
            T.writes()
            T.sblock_attr({"tilelang.is_npu_kernel_frame": True})
            xl = T.sblock_alloc_buffer((32, 128), "bfloat16", scope="shared.dyn")
            if 0 < num_tokens:
                T.copy(T.region(x[bx // 10 * 32, bx % 10 * 128], 1, 32, 128), T.region(xl[0, 0], 2, 32, 128))
                for m in range(4):
                    with T.sblock("SIMT_VF", no_realize=True):
                        T.reads()
                        T.writes()
                        T.sblock_attr({"tl.vf_source_index": T.int64(0)})
                        simtvf_tx = T.launch_thread("threadIdx.x", 128)
                        simtvf_ty = T.launch_thread("threadIdx.y", 1)
                        simtvf_tz = T.launch_thread("threadIdx.z", 1)
                        T.attr("simtvf", "tl.simtvf_scope", 1)
                        for ti in T.parallel(32):
                            for tj in T.parallel(128):
                                if bx // 10 * 32 + ti < num_tokens:
                                    o[bx // 10 * 32 + ti, m, bx % 10 * 128 + tj] = xl[ti, tj]
```

## Case 04: mhc_post_fwd

### 输入

```python
@I.ir_module
class Module:
    @T.prim_func
    def gpu_mhc_post_fwd_kernel(a_handle: T.handle, b_handle: T.handle, c_handle: T.handle, d_handle: T.handle, x_handle: T.handle):
        T.func_attr({"target": T.target({"host": {"keys": ["cpu"], "kind": "c", "tag": ""}, "keys": ["ascend"], "kind": "ascend", "tag": ""})})
        num_tokens = T.int32()
        a = T.match_buffer(a_handle, (num_tokens, 4, 4), strides=(16, 4, 1))
        b = T.match_buffer(b_handle, (num_tokens, 4, 1280), "bfloat16", strides=(5120, 1280, 1))
        c = T.match_buffer(c_handle, (num_tokens, 4), strides=(4, 1))
        d = T.match_buffer(d_handle, (num_tokens, 1280), "bfloat16", strides=(1280, 1))
        x = T.match_buffer(x_handle, (num_tokens, 4, 1280), "bfloat16", strides=(5120, 1280, 1))
        # with T.sblock("root"):
        T.attr(T.bool(True), "tl.assume", "Buffer shape should be greater than or equal to 0: shape `num_tokens` from buffer `a`, `b`, `c`, `d`, `x`")
        bx = T.launch_thread("blockIdx.x", num_tokens)
        with T.sblock("tilelang_root"):
            T.reads()
            T.writes()
            T.sblock_attr({"tilelang.is_npu_kernel_frame": True})
            x_shared = T.sblock_alloc_buffer((4, 256), "bfloat16", scope="shared.dyn")
            b_shared = T.sblock_alloc_buffer((4, 256), "bfloat16", scope="shared.dyn")
            d_shared = T.sblock_alloc_buffer((256,), "bfloat16", scope="shared.dyn")
            x_local = T.sblock_alloc_buffer((4, 256), scope="local.fragment")
            b_local = T.sblock_alloc_buffer((4, 256), scope="local.fragment")
            d_local = T.sblock_alloc_buffer((256,), scope="local.fragment")
            a_local = T.sblock_alloc_buffer((4, 4), scope="local.fragment")
            c_local = T.sblock_alloc_buffer((4,), scope="local.fragment")
            T.copy(T.region(a[bx, 0, 0], 1, 1, 4, 4), T.region(a_local[0, 0], 2, 4, 4))
            T.copy(T.region(c[bx, 0], 1, 1, 4), T.region(c_local[0], 2, 4))
            for i0_h in T.serial(5, annotations={"num_stages": 2}):
                T.copy(T.region(b[bx, 0, i0_h * 256], 1, 1, 4, 256), T.region(b_shared[0, 0], 2, 4, 256), disable_tma=T.bool(True))
                T.copy(T.region(d[bx, i0_h * 256], 1, 1, 256), T.region(d_shared[0], 2, 256), disable_tma=T.bool(True))
                T.copy(T.region(b_shared[0, 0], 1, 4, 256), T.region(b_local[0, 0], 2, 4, 256))
                T.copy(T.region(d_shared[0], 1, 256), T.region(d_local[0], 2, 256))
                for i_mhco in T.parallel(4):
                    for i1_h in T.parallel(256):
                        x_local[i_mhco, i1_h] = c_local[i_mhco] * d_local[i1_h]
                        for i_mhci in range(4):
                            x_local[i_mhco, i1_h] = x_local[i_mhco, i1_h] + a_local[i_mhci, i_mhco] * b_local[i_mhci, i1_h]
                T.copy(T.region(x_local[0, 0], 1, 4, 256), T.region(x_shared[0, 0], 2, 4, 256))
                T.copy(T.region(x_shared[0, 0], 1, 4, 256), T.region(x[bx, 0, i0_h * 256], 2, 1, 4, 256), disable_tma=T.bool(True))
```

### 输出

```python
@I.ir_module
class Module:
    @T.prim_func
    def gpu_mhc_post_fwd_kernel(a_handle: T.handle, b_handle: T.handle, c_handle: T.handle, d_handle: T.handle, x_handle: T.handle):
        T.func_attr({"target": T.target({"host": {"keys": ["cpu"], "kind": "c", "tag": ""}, "keys": ["ascend"], "kind": "ascend", "tag": ""})})
        num_tokens = T.int32()
        a = T.match_buffer(a_handle, (num_tokens, 4, 4), strides=(16, 4, 1))
        b = T.match_buffer(b_handle, (num_tokens, 4, 1280), "bfloat16", strides=(5120, 1280, 1))
        c = T.match_buffer(c_handle, (num_tokens, 4), strides=(4, 1))
        d = T.match_buffer(d_handle, (num_tokens, 1280), "bfloat16", strides=(1280, 1))
        x = T.match_buffer(x_handle, (num_tokens, 4, 1280), "bfloat16", strides=(5120, 1280, 1))
        # with T.sblock("root"):
        T.attr(T.bool(True), "tl.assume", "Buffer shape should be greater than or equal to 0: shape `num_tokens` from buffer `a`, `b`, `c`, `d`, `x`")
        bx = T.launch_thread("blockIdx.x", num_tokens)
        with T.sblock("tilelang_root"):
            T.reads()
            T.writes()
            T.sblock_attr({"tilelang.is_npu_kernel_frame": True})
            x_shared = T.sblock_alloc_buffer((4, 256), "bfloat16", scope="shared.dyn")
            b_shared = T.sblock_alloc_buffer((4, 256), "bfloat16", scope="shared.dyn")
            d_shared = T.sblock_alloc_buffer((256,), "bfloat16", scope="shared.dyn")
            a_shared = T.sblock_alloc_buffer((4, 4), scope="shared.dyn")
            c_shared = T.sblock_alloc_buffer((4,), scope="shared.dyn")
            T.copy(T.region(a[bx, 0, 0], 1, 1, 4, 4), T.region(a_shared[0, 0], 2, 4, 4))
            T.copy(T.region(c[bx, 0], 1, 1, 4), T.region(c_shared[0], 2, 4))
            for i0_h in T.serial(5, annotations={"num_stages": 2}):
                T.copy(T.region(b[bx, 0, i0_h * 256], 1, 1, 4, 256), T.region(b_shared[0, 0], 2, 4, 256))
                T.copy(T.region(d[bx, i0_h * 256], 1, 1, 256), T.region(d_shared[0], 2, 256))
                with T.sblock("SIMT_VF", no_realize=True):
                    T.reads()
                    T.writes()
                    T.sblock_attr({"tl.vf_source_index": T.int64(0)})
                    x_local = T.sblock_alloc_buffer((4, 256), scope="local.fragment")
                    b_local = T.sblock_alloc_buffer((4, 256), scope="local.fragment")
                    d_local = T.sblock_alloc_buffer((256,), scope="local.fragment")
                    a_local = T.sblock_alloc_buffer((4, 4), scope="local.fragment")
                    c_local = T.sblock_alloc_buffer((4,), scope="local.fragment")
                    simtvf_tx = T.launch_thread("threadIdx.x", 128)
                    simtvf_ty = T.launch_thread("threadIdx.y", 1)
                    simtvf_tz = T.launch_thread("threadIdx.z", 1)
                    T.attr("simtvf", "tl.simtvf_scope", 1)
                    T.copy(T.region(a_shared[0, 0], 1, 4, 4), T.region(a_local[0, 0], 2, 4, 4))
                    T.copy(T.region(c_shared[0], 1, 4), T.region(c_local[0], 2, 4))
                    T.copy(T.region(b_shared[0, 0], 1, 4, 256), T.region(b_local[0, 0], 2, 4, 256))
                    T.copy(T.region(d_shared[0], 1, 256), T.region(d_local[0], 2, 256))
                    for i_mhco in T.parallel(4):
                        for i1_h in T.parallel(256):
                            x_local[i_mhco, i1_h] = c_local[i_mhco] * d_local[i1_h]
                            for i_mhci in range(4):
                                x_local[i_mhco, i1_h] = x_local[i_mhco, i1_h] + a_local[i_mhci, i_mhco] * b_local[i_mhci, i1_h]
                    T.copy(T.region(x_local[0, 0], 1, 4, 256), T.region(x_shared[0, 0], 2, 4, 256))
                T.copy(T.region(x_shared[0, 0], 1, 4, 256), T.region(x[bx, 0, i0_h * 256], 2, 1, 4, 256))
```
