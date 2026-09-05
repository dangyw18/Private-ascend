# Auto-SimtVF Pass 第一版设计

## 1. 需求背景

### 1.1 目标

TileLang CUDA DSL 中的 `T.Parallel` 依赖 GPU thread block 执行。Ascend 后端需要在不修改原算子 DSL 数据路径的前提下，自动识别适合 SIMT 的计算 region，并插入 `T.SimtVF`：

- 原程序是 `GM → fragment`，则 copy 与 fragment consumer/producer 一起进入 SimtVF，仍保持 `GM → fragment`。
- 原程序是 `GM → shared → fragment`，则保持 `GM → UB → fragment`：GM↔shared copy 在 VF 外，shared↔fragment copy和计算在 VF 内。
- 含 `T.gemm` 的程序要求在进入本 Pass 前已经显式给出 CV 之间的 buffer 和 copy；第一版不自动推导 CV 数据交接。
- 第一版优先保证语义等价和 CUDA DSL 迁移便利，不要求极致性能。

### 1.2 最基本的输入与输出

输入：直接 GM↔fragment，多段并行计算尚未圈域。

```python
with T.Kernel(grid) as bx:
    frag = T.alloc_fragment((N,), dtype)
    T.copy(A[bx * N], frag)

    for i in T.Parallel(N):
        frag[i] = frag[i] + 1

    scale = 2.0

    for i in T.Parallel(N):
        frag[i] = frag[i] * scale

    T.copy(frag, B[bx * N])
```

输出：保持 copy 和 storage 写法，只增加 SimtVF scope。

```python
with T.Kernel(grid) as bx:
    with T.SimtVF(threads=T):
        frag = T.alloc_fragment((N,), dtype)
        T.copy(A[bx * N], frag)

        for i in T.Parallel(N):
            frag[i] = frag[i] + 1

        scale = 2.0

        for i in T.Parallel(N):
            frag[i] = frag[i] * scale

        T.copy(frag, B[bx * N])
```

第一版 Pass 的接口建议保持简单：

```text
InferSimtVF(PrimFunc, Target, AutoSimtVFConfig) -> PrimFunc
```

复杂的 def-use、storage scope、collective 和融合判断隐藏在模块实现中。

## 2. 规则定义

### 2.1 Region seed

以下操作产生 SIMT region seed：

1. `T.Parallel`；
2. 被该 `T.Parallel` 直接或间接依赖的受支持 `T.reduce_*`；
3. 读写同一 `local.fragment` 的 copy、初始化和纯计算。

没有 seed 的纯标量代码第一版保持 Scalar。

### 2.2 依赖闭包

从 seed 沿 def-use 向前和向后扩张，吸收：

- `T.alloc_fragment`、`T.clear`；
- fragment 的 producer/consumer；
- 无副作用的标量表达式和 `let`；
- 只控制本 region 的 `T.serial`；
- 支持的 `T.reduce_*`；
- 与 fragment 相连的 `T.copy`。

下列操作默认停止扩张：

- GM↔shared 的 DMA copy；
- `T.gemm` 及已经显式给出的 CV 交接；
- `T.async_copy`、`T.ptx_wait_group` 等 CUDA 专用操作；
- pipe/barrier/sync；
- 已有 `T.SimtVF`、`T.SimdVF`；
- 未知副作用调用或无法证明安全的控制流。

### 2.3 保持原生 storage/copy 路径

| 原 DSL 数据路径 | 自动划分结果 |
|---|---|
| `GM → fragment → compute → GM` | 整条链进入同一个 SimtVF，不插入 UB |
| `GM → shared → fragment → compute → shared → GM` | GM↔shared 保留在 VF 外；shared↔fragment 与 compute 进入 VF |
| `GM/shared → T.gemm → shared/GM` | CV copy 必须由前置 Pass 或 DSL 显式给出；Auto-SimtVF 不跨越 GEMM |
| 直接在 `T.Parallel` 中访问 GM | Parallel body 进入 SimtVF，保持直接 GM load/store |

核心不变量：`local.fragment` 的定义、所有权相关访问和最后使用必须位于同一 SimtVF；shared/UB 可以作为 Scalar、SIMT 和其他执行域之间的交接存储。

### 2.4 Scalar 语句处理

| 标量语句 | 第一版规则 |
|---|---|
| 仅由常量和 region 内值构成的纯表达式 | 吸收进 SimtVF |
| 位于 `T.Parallel` body 内的 lane-varying 标量 | 必须随 Parallel 进入 SimtVF |
| `T.reduce_*` 产生、后续 Parallel 消费的值 | collective 与 consumer 放在同一 SimtVF |
| 控制 DMA/GEMM/sync 的标量 | 留在 VF 外 |
| `T.get_thread_binding()` 或线程角色分支 | 第一版仅在已有明确线程语义时接受，否则拒绝自动划分 |
| 有外部副作用或无法分类 | 保持 Scalar，并停止 region 扩张 |

### 2.5 合法融合规则

相邻 seed region 满足以下条件时合成一个 SimtVF：

1. 位于同一控制流 block，或整个 `T.serial` body 可一起纳入；
2. threads/layout 兼容；
3. 中间不存在 GM↔shared DMA、GEMM、sync 或未知副作用；
4. fragment def-use 闭合；
5. reduction/collective 布局受后端支持；
6. local/stack、UB 和线程资源不超过目标限制。

第一版采用“合法且资源允许则融合”的规则，不先实现 cycle-accurate Cost Model。实现中预留：

```text
estimate(region, Scalar | SIMT, context) -> EstimatedCycles
```

后续再用它比较 Scalar、SIMT 和 SIMD 候选，以及判断：

```text
gain(A, B) = C(A) + C(B) + C_handoff - C(fuse(A, B))
```

### 2.6 线程数

第一版按以下优先级选择：

1. 使用 `AutoSimtVFConfig.threads`；
2. 复用原 `T.Kernel(..., threads=n_thr)` 作为 hint；
3. 使用目标后端默认值。

最终物理线程数若会被 layout inference 调整，应将调整后的值记录到 IR，供后续 Cost Model 使用。

## 3. Case 分类及输入输出

### 3.1 MHC forward kernel 难度分类

| 难度 | Kernel | Case | 第一版策略/状态 |
|---|---|---|---|
| 简单 | `_mhc_head_compute_mix_fwd` | A | 单个直接 GM map，Parallel body 圈入 SimtVF；已手工改写 |
| 简单 | `_mhc_fn_normw_merge_fwd` | A | 单个直接 GM map，Parallel body 圈入 SimtVF；已手工改写 |
| 简单—中等 | `_mhc_pre_split_mixes_fwd` | B | GM↔fragment 与三个 Parallel 整体圈域；已手工改写 |
| 中等 | `_mhc_sinkhorn_fwd` | C | GM↔fragment、多个 reduction/Parallel/serial 整体圈域；已手工改写 |
| 中等 | `expand_to_mhc_fwd_tl` | D | GM→fragment 与 `serial→Parallel` 整体圈域；已手工改写 |
| 中高 | `_mhc_pre_apply_mix_fwd` | E | 保留 GM→shared；shared↔fragment、clear、serial/Parallel、fragment→shared 圈域；待实现 |
| 中高 | `_mhc_post_fwd` | E | 与 pre-apply 类似，适合作为 pipeline 集成 case；待实现 |
| 高 | `_mhc_pre_norm_fn_fwd_mul` | F | 含 GEMM、reduction 和 Parallel；要求 CV copy 已显式化；后续实现 |
| 高 | `_mhc_pre_norm_fn_fwd_norm` | G | 含 `T.get_thread_binding()`、per-thread scalar 和 copy；后续实现 |
| 很高 | `_mhc_pre_big_fuse` | G | 线程角色分支、reduction、pipeline 与多类数据流；非第一版目标 |
| 很高 | `_mhc_multilayer_recompute_kernel` | H | 含动态指针、`T.async_copy`、PTX wait；需 CUDA intrinsic Adapter，非第一版目标 |

### 3.2 Case A：直接 GM map

输入：

```python
for i in T.Parallel(N):
    C[i] = f(A[i])
```

输出：

```python
with T.SimtVF(threads=T):
    for i in T.Parallel(N):
        C[i] = f(A[i])
```

### 3.3 Case B：直接 GM↔fragment，多段 Parallel

输入：

```python
frag = T.alloc_fragment((N,), dtype)
T.copy(A, frag)
for i in T.Parallel(N):
    frag[i] = f(frag[i])
for i in T.Parallel(N):
    frag[i] = g(frag[i])
T.copy(frag, B)
```

输出：

```python
with T.SimtVF(threads=T):
    frag = T.alloc_fragment((N,), dtype)
    T.copy(A, frag)
    for i in T.Parallel(N):
        frag[i] = f(frag[i])
    for i in T.Parallel(N):
        frag[i] = g(frag[i])
    T.copy(frag, B)
```

### 3.4 Case C：Parallel 与 reduction/collective

输入：

```python
T.copy(A, frag)
T.reduce_sum(frag, reduced, dim=axis)
for i in T.Parallel(N):
    frag[i] = frag[i] / reduced[index(i)]
T.copy(frag, B)
```

输出：copy、reduction 和 consumer 必须处于同一 SimtVF，由现有 lowering 生成 collective 和同步。

```python
with T.SimtVF(threads=T):
    T.copy(A, frag)
    T.reduce_sum(frag, reduced, dim=axis)
    for i in T.Parallel(N):
        frag[i] = frag[i] / reduced[index(i)]
    T.copy(frag, B)
```

### 3.5 Case D：`serial → Parallel` 嵌套

输入：

```python
T.copy(A, frag)
for m in T.serial(M):
    for i in T.Parallel(N):
        B[m, i] = frag[i]
```

输出：外层 serial 与内部 Parallel 整体进入 SimtVF，避免 fragment 跨 VF。

```python
with T.SimtVF(threads=T):
    T.copy(A, frag)
    for m in T.serial(M):
        for i in T.Parallel(N):
            B[m, i] = frag[i]
```

### 3.6 Case E：GM→shared→fragment pipeline

输入：

```python
T.copy(A, a_shared)
T.copy(a_shared, a_frag)
for i in T.Parallel(N):
    out_frag[i] = f(a_frag[i])
T.copy(out_frag, out_shared)
T.copy(out_shared, B)
```

输出：保持原 storage 路径，只圈 shared↔fragment 与计算。

```python
T.copy(A, a_shared)
with T.SimtVF(threads=T):
    T.copy(a_shared, a_frag)
    for i in T.Parallel(N):
        out_frag[i] = f(a_frag[i])
    T.copy(out_frag, out_shared)
T.copy(out_shared, B)
```

### 3.7 Case F：含 GEMM/CV 交接

输入前提：CV 之间的 shared/UB buffer、copy 和同步已经显式存在。

```python
T.copy(A, a_shared)
T.gemm(a_shared, b_shared, accum)
T.copy(accum, v_shared)       # 前置输入已经显式给出
for i in T.Parallel(N):
    out[i] = epilogue(v_shared[i])
```

输出：第一版不跨 GEMM，只对其后的合法 Parallel region 圈 SimtVF。

```python
T.copy(A, a_shared)
T.gemm(a_shared, b_shared, accum)
T.copy(accum, v_shared)
with T.SimtVF(threads=T):
    for i in T.Parallel(N):
        out[i] = epilogue(v_shared[i])
```

### 3.8 Case G/H：线程角色或 CUDA 专用操作

包含 `T.get_thread_binding()` 角色分工、动态指针、`T.async_copy` 或 PTX wait 的程序，第一版不自动圈域。Pass 应保留原程序并给出明确的 unsupported reason，不允许静默生成可能错误的 SimtVF。
