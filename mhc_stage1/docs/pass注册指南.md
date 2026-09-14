# tilelang-xy Pass 注册指南（以新增 AutoSimtVF 为例）

## 1. Pass 注册的三层结构

一个 pass 从实现到接入 pipeline 共三层，缺一不可：

```
C++ 实现(.cc)  →  Python 包装函数  →  Pipeline 调用
     ▲                  ▲                 ▲
 src/**/*.cc    tilelang/**/__init__.py   tilelang/ascend/pipeline.py
```

- **C++ 实现**：写 pass 逻辑 + `TVM_FFI_STATIC_INIT_BLOCK` 注册全局函数 `tl.transform.<Name>`。
- **Python 包装**：`def <Name>(): return _ffi_api.<Name>()`，把 C++ 全局函数暴露成 Python 可调用对象。
- **Pipeline 调用**：`mod = xxx_transform.<Name>()(mod)`。

`_ffi_api` 是自动生成的：`tilelang/transform/_ffi_api.py` 里 `tvm_ffi.init_ffi_api("tl.transform", __name__)`，会按名字动态映射到 C++ 注册的 `tl.transform.*`。**无需修改此文件**。

## 2. 以 VerifyReducerEpoch 为例的完整链路

### (1) C++ 实现：`src/transform/verify_reducer_epoch.cc`

文件尾部两个关键点：

```cpp
tvm::transform::Pass VerifyReducerEpoch() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    ReducerEpochVerifier verifier;    // 实际逻辑类
    verifier.Run(f);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.VerifyReducerEpoch", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("tl.transform.VerifyReducerEpoch", VerifyReducerEpoch);
}
```

要点：
- `CreatePrimFuncPass(pass_func, opt_level, "pass_name", {})`：作用于 `PrimFunc` 的 pass（最常见）。作用于整个 `IRModule` 用 `CreateModulePass`。
- `pass_name` 是内部标识串（`tl.VerifyReducerEpoch`），与注册名无关。
- **注册名统一是 `tl.transform.<Name>`**，无论源码在 `src/transform/` 还是 `src/ascend/transform/`（`UnrollLoopSkipVF`、`InsertNd2Nz` 都注册成 `tl.transform.*`）。

### (2) Python 包装：`tilelang/transform/__init__.py`

```python
def VerifyReducerEpoch():
    """校验 reducer v2 epoch 的生命周期与访问规则。"""
    return _ffi_api.VerifyReducerEpoch()  # type: ignore
```

通用 pass 放这里；**Ascend 专用 pass 放 `tilelang/ascend/transform/__init__.py`**（该文件顶部 `from tilelang.transform import _ffi_api`，复用同一 FFI）。

### (3) Pipeline 调用：`tilelang/ascend/pipeline.py`

```python
mod = tilelang.transform.VerifyReducerEpoch()(mod)
print("---------------------12 After VerifyReducerEpoch---------------------\n", mod)
```

## 3. 新增 AutoSimtVF 的文件清单（放在 UnrollLoopSkipVF 前）

需 **新增 1 个文件、修改 2 个文件**：

| 动作 | 路径 | 内容 |
|---|---|---|
| 新增 | `tilelang-xy/src/ascend/transform/auto_simt_vf.cc` | C++ 实现 + 注册 |
| 修改 | `tilelang-xy/tilelang/ascend/transform/__init__.py` | 加 `AutoSimtVF()` 包装函数 |
| 修改 | `tilelang-xy/tilelang/ascend/pipeline.py` | 在 UnrollLoopSkipVF(14) 前插入调用 |

**无需修改**：`CMakeLists.txt`（源文件靠 `file(GLOB src/ascend/transform/*.cc)` 自动收集）、`_ffi_api.py`、`tilelang/transform/__init__.py`。

### (1) 新增 `src/ascend/transform/auto_simt_vf.cc`

骨架如下（参考 `unroll_loop_skip_vf.cc` 的结构）：

```cpp
#include <tvm/tirx/transform.h>       // 若用 tirx 内部 IR
// 依赖的 tilelang 头文件按需引入

namespace tvm {
namespace tl {
namespace transform {

using namespace tir::transform;       // 或 tirx::transform

Pass AutoSimtVF() {
  auto pass_func = [](PrimFunc func, const IRModule &m, PassContext) {
    // 1. 遍历/分类 func->body 中的语句（StmtClassifier）
    // 2. 划分 region、融合相邻 region
    // 3. 物化 fragment / 改写直读
    // 4. 生成 SIMT_VF block（emit SIMT_VF）
    return func;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.AutoSimtVF", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("tl.transform.AutoSimtVF", AutoSimtVF);
}

}  // namespace transform
}  // namespace tl
}  // namespace tvm
```

### (2) 修改 `tilelang/ascend/transform/__init__.py`

在 `UnrollLoopSkipVF()` 附近追加：

```python
def AutoSimtVF():
    """将根块内的 Parallel 计算区段包裹进 SIMT_VF block。

    放在 UnrollLoopSkipVF 之前，保证 VF 边界先就位，再让下游 unroll
    跳过 VF 内部的 serial 循环。
    """
    return _ffi_api.AutoSimtVF()  # type: ignore
```

### (3) 修改 `tilelang/ascend/pipeline.py`

在 `VerifyBufferInit`(13) 之后、`UnrollLoopSkipVF`(14) 之前插入：

```python
    mod = ascend_transform.AutoSimtVF()(mod)
    print("---------------------13.5 After AutoSimtVF---------------------\n", mod)
```

## 4. 关键注意事项

1. **GLOB 需重新配置 cmake**：新 `.cc` 文件放进 `src/ascend/transform/` 后，`file(GLOB)` 不会自动感知新文件，需重跑 `cmake` 配置（或删除 build 缓存目录）再编译。
2. **注册名固定 `tl.transform.<Name>`**，不要写成 `tl.ascend.transform.*`。
3. **包名决定导入方式**：Ascend pass 走 `from . import transform as ascend_transform`，所以包装函数必须放进 `tilelang/ascend/transform/__init__.py`；通用 pass 才放进 `tilelang/transform/__init__.py`。
4. **pass 作用粒度**：默认 `CreatePrimFuncPass`（逐函数应用）。若需跨函数/整模块视角（如全局 buffer 依赖），改用 `CreateModulePass`。
5. **`opt_level=0`**：本仓库现有 pass 均用 0，保持一致即可。
6. **打印编号**：在每个 pass 后 `print` 一行 `XX After <Name>`，便于 log 对齐（AutoSimtVF 已用 13.5 占位；若要连续编号可整体顺延后续编号）。