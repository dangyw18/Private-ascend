# reduce_demo

使用 **ccec 编译器内建指令**（`vdup`/`vlds`/`vcadd`/`vadd`/`vsts`）实现向量归约求和（ReduceSum），并借助 ccec 内建 `get_sys_cnt()` 统计 kernel 各阶段（GM→UB 搬运、reduce 计算、UB→GM 搬出、总耗时）的硬件 cycle 数。

## 文件结构

- `reduce_demo.asc`：kernel 主体与 host 侧入口（ACL 初始化、数据搬运、kernel 启动、结果校验）
- `data_utils.h`：输入/输出文件读写辅助工具（`ReadFile` / `WriteFile`），与 asc-devkit 一致
- `CMakeLists.txt`：ASC CMake 构建脚本，目标架构 `dav-3510`，运行模式 `npu`
- `scripts/gen_data.py`：数据生成脚本，生成 `input/input_x.bin`（4096 个 float32）与 `output/golden.bin`（8 个 float32，golden[0] = sum）

## 构建与运行

```bash
# 1. 生成测试数据（4096 个 float，golden 取和）
cd SIMT-Scalar/reduce_demo
python3 scripts/gen_data.py

# 2. 构建
cmake -B build -DCMAKE_ASC_RUN_MODE=npu -DCMAKE_ASC_ARCHITECTURES=dav-3510
cmake --build build

# 3. 运行（需在带 Ascend 设备的服务器上执行，host 侧 device_id = 2）
./build/demo
```

预期输出（服务器实测）：

```
Kernel cycle stats
 GM->UB copy: ... cycles
 reduce loop: ... cycles
 UB->GM copy: ... cycles
 kernel total: ... cycles
Output sum: <sum>
Golden sum: <sum>
test pass!
```

**注意**：本样例 host 侧 `kernel_reduce()` 中 `device_id = 2`，运行时请确认设备号可用（`npu-smi info` 查看），必要时按环境修改。

## 复现 1 次/2 次 VF 差分测试

README 表格中“运行 cycles / 启动 vf 时间”的估算采用如下线性假设：

```text
T1 = F + R
T2 = F + 2R

单次 VF 增量运行时间 R = T2 - T1
固定开销 F = T1 - R = 2*T1 - T2
```

其中，`T1` 和 `T2` 必须使用完全相同的 VF、参数、计时边界和同步方式，
唯一差别是计时区间内分别连续调用 1 次和 2 次 VF。本例选择完整的
`reduce_sum_vf_cce3` 进行该对照；两次调用会覆盖写入同一个输出，但数值结果相同，
仍可通过 golden 校验。

可以手动编译、运行两个版本：

```bash
cd SIMT-Scalar/reduce_demo
python3 scripts/gen_data.py

cmake -S . -B build/vf1 \
  -DCMAKE_ASC_RUN_MODE=npu \
  -DCMAKE_ASC_ARCHITECTURES=dav-3510 \
  -DREDUCE_VF_CALLS=1
cmake --build build/vf1
./build/vf1/demo

cmake -S . -B build/vf2 \
  -DCMAKE_ASC_RUN_MODE=npu \
  -DCMAKE_ASC_ARCHITECTURES=dav-3510 \
  -DREDUCE_VF_CALLS=2
cmake --build build/vf2
./build/vf2/demo
```

分别记录两次输出中的 `reduce loop`：

```text
T1 = 1-call 的 reduce loop cycles
T2 = 2-call 的 reduce loop cycles
R  = T2 - T1
F  = 2*T1 - T2
```

也可以直接运行 `bash scripts/run_vf_startup_diff.sh`，脚本会编译两个版本、
运行并打印上述差分结果。

需要注意：这里测量的是 **SIMD `__simd_vf__`**，不是 SIMT `__simt_vf__`。
而且 `F` 是线性拟合得到的固定项，除 VF 启停/排空外还可能含有同步及计时边界开销，
因此更严谨的名称是“固定开销估算”，不能直接证明它等于纯 VF 启动时间。

## 实现逻辑

### 归约算法（5 种，cce 指令实现）

| 算法 | 思路 | 实测 cycles（服务器） |
| --- | --- | --- |
| cce1 | 循环展开 4 向量：`vadd` 两两合并（4→2→1）后 `vcadd` 单向量归约，`vadd` 累加 | 309 |
| cce2 | 每向量独立 `vcadd` 归约后 `vadd` 累加；`plt_b32(POST_UPDATE)` 按剩余元素生成部分掩码 | 410 |
| cce3 | 每轮部分和 `ONEPT_B32` 单点写回 `y_local[i]`，`mem_bar(VST_VLD)` 后整向量二次归约 | 339 |
| cce4 | 同 cce3，但循环每次展开 2 向量（双发），提高 PIPE_V 并行度 | 338 |
| cce5 | 将 cce3 拆分为 `cce5_1`（部分和写回）与 `cce5_2`（二次归约）两个函数 | 333 |

kernel 实际调用 `cce5_1`（两次，部分和写回）与 `cce5_2`（两次，二次归约）；cce1~cce4 作为算法对比保留。

### kernel 主流程

```
t0 ── GM->UB 搬运（PIPE_MTE2）── t1 ── reduce 计算（PIPE_V）── t3 ── UB->GM 搬出（PIPE_MTE3）── t4
cycles[0] = t1-t0  GM->UB 搬运
cycles[1] = t3-t2  reduce 计算
cycles[2] = t4-t0  kernel 总耗时
cycles[3] = t4-t3  UB->GM 搬出
```

关键约束（3510 后端实测）：

1. **vector 指令与标量指令分离**：所有 vector 指令（`vlds`/`vadd`/`vcadd`/`vsts` 等）必须放在独立的 `__simd_vf__` 函数中；`get_sys_cnt()` 等标量操作留在 `__vector__` kernel 主体（标量上下文）。二者不可混写，否则后端报 `Do not know how to split the result of this operator!`。
2. **`get_sys_cnt()` 必须 noinline 包装**：它是 ccec 编译器内建，Release（-O2/-O3）下多次调用会被 CSE/常量折叠合并，导致各时间戳相等、cycle 差值为 0。
3. **计时前必须同步到 PIPE_S**：`get_sys_cnt()` 走 PIPE_S 标量流水线，与被测的 PIPE_MTE2/PIPE_V/PIPE_MTE3 异步。每次读计数器前必须插入对应流水线到 PIPE_S 的同步（`asc_sync_notify(from, PIPE_S, ...)` + `asc_sync_wait(from, PIPE_S, ...)`），否则 PIPE_S 可能越过下游同步提前执行，cycle 差值偏小甚至为 0。

## ccec OP 使用说明

以下指令均为 ccec 编译器内建（bisheng/clang 15.0.5，`__clang_cce_vector_intrinsics.h` / `__clang_cce_aicore_functions.h`），在 device 代码中直接可用，无需额外包含头文件。dav-3510 上 `vector_float` 对应 64 个 float 通道（v64）。

### get_sys_cnt

读取当前系统 cycle 计数器（只读，64bit，复位为 0，随系统时钟递增）。

```c
__aicore__ int64_t get_sys_cnt();
```

- **流水线**：PIPE_S
- **用法**：`__attribute__((noinline))` 包装后调用，避免 CSE/常量折叠
- **约束**：PIPE_S 与其他流水线异步，计时前需将目标流水线同步到 PIPE_S
- **换算**：Ascend 950PR/950DT 为 1 GHz（`time(us) = cycles / 1000`）；Atlas A3/A2 系列为 50 MHz（`time(us) = cycles / 50`）

### pset_b32

按 pattern 生成 32bit 谓词掩码寄存器。

```c
template <class T> CCE_INTRINSIC vector_bool pset_b32(T dist);
```

- `PAT_ALL`：所有通道有效（全 1 掩码）
- `PAT_VL1`：仅第 0 通道有效（配合 `ONEPT_B32` 单点存储）
- 示例：`vmask = pset_b32(PAT_ALL);`

### plt_b32

按剩余元素数生成部分掩码（predicate load tail），`POST_UPDATE` 模式自动扣减剩余计数。

```c
template <class T> CCE_INTRINSIC vector_bool plt_b32(uint32_t &scalar, T post);
```

- 仅接受 `POST_UPDATE`：生成掩码后 `scalar` 自动递减，用于循环尾部非整向量处理
- 示例：`vmask = plt_b32(remain, POST_UPDATE);`

### vdup

标量广播（broadcast）：将标量复制到向量寄存器各通道；`MODE_ZEROING` 下非掩码通道清零。

```c
template <class T> CCE_INTRINSIC void vdup(vector_float &dst, float src, vector_bool mask, T mode);
```

- `mode`：`MODE_ZEROING` / `MODE_MERGING`
- 示例：`vdup(acc_reg, 0.0f, vmask, MODE_ZEROING);`（清零累加寄存器）

### vlds

向量加载：从 UB 加载一个向量到寄存器。

```c
template <class T> CCE_INTRINSIC void vlds(vector_float &dst, __ubuf__ float *base, int32_t offset, T dist);
```

- `offset`：以**元素**为单位（内部自动乘 `sizeof(float)`）
- `dist`：`NORM`（常规整向量加载）
- 示例：`vlds(src_reg, x_local + i * one_rep_size, 0, NORM);`

### vadd

向量逐通道加法。

```c
template <class T> CCE_INTRINSIC void vadd(vector_float &dst, vector_float src0, vector_float src1, vector_bool mask, T mode = MODE_UNKNOWN);
```

- `mode`：`MODE_ZEROING` / `MODE_MERGING` / 缺省
- 示例：`vadd(mid_reg1, src_reg1, src_reg2, vmask);`

### vcadd

向量归约（cross add）：将向量内全部元素交叉相加，归约结果写入最低通道（lane 0）。dav-3510 上一次 `vcadd` 即完成整向量归约（对应 C API `asc_reduce_sum`，见 `kernel_reg_compute_vec_reduce_impl.h` 的 `ReduceSumImpl`），`MODE_ZEROING` 下其余通道清零。

```c
template <class T> CCE_INTRINSIC void vcadd(vector_float &dst, vector_float src, vector_bool mask, T mode);
```

- `mode`：`MODE_ZEROING` / `MODE_MERGING`
- 本样例用法：
  - cce5_1：每向量 `vcadd` 得部分和（lane 0），`ONEPT_B32` 单点写回 `y_local[i]`
  - cce5_2：对部分和向量再次 `vcadd` 得最终和
- 示例：`vcadd(dst_reg, src_reg, vmask_all, MODE_ZEROING);`

### vsts

向量存储：将寄存器写回 UB。

```c
template <class T> CCE_INTRINSIC void vsts(vector_float data, __ubuf__ float *base, int32_t offset, T dist, vector_bool mask);
```

- `offset`：以**元素**为单位
- `dist`：
  - `NORM_B32`：常规 32B 对齐整向量存储
  - `ONEPT_B32`：单点存储（仅第 0 通道 1 个元素，需配合 `PAT_VL1` 掩码）
- 示例：
  - `vsts(acc_reg, y_local, 0, NORM_B32, vmask);`
  - `vsts(dst_reg, y_local + i, 0, ONEPT_B32, vmask_one);`

### mem_bar

存储/加载屏障，保证同一 PIPE_V 内的内存访问顺序。

```c
template <class T> CCE_INTRINSIC void mem_bar(T mem_type);
```

- 支持类型：`VST_VLD`（store→load 屏障）、`VLD_VST`、`VST_VST`、`VV_ALL` 等
- 示例：`mem_bar(VST_VLD);` —— 确保前面所有 `vsts` 写回 UB 后再 `vlds`（cce3/cce4 二次归约前使用）

## host 侧 C API（asc_*）

| 接口 | 功能 |
| --- | --- |
| `asc_init()` | kernel 入口初始化（`__vector__` 函数首行调用） |
| `asc_copy_gm2ub_align(dst, src, count, len, ...)` | GM→UB 对齐搬运（PIPE_MTE2） |
| `asc_copy_ub2gm_align(dst, src, count, len, ...)` | UB→GM 对齐搬运（PIPE_MTE3） |
| `asc_sync_notify(from, to, event_id)` | 通知目标流水线事件完成 |
| `asc_sync_wait(from, to, event_id)` | 等待源流水线事件完成 |
| `asc_sync()` | 全流水线同步（kernel 收尾） |
| `asc_get_vf_len()` | 返回向量寄存器长度（字节），`/ sizeof(float)` 得到单向量元素数 |

## 产品支持情况

| 产品 | 支持 |
| --- | --- |
| Ascend 950PR / Ascend 950DT | 支持 |
