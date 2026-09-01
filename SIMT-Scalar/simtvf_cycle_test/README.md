# SIMTVF 冷启动 cycle 测试

本例沿用 `cycle_count_demo` 的系统 cycle 计数和 `PIPE_V -> PIPE_S` 同步思路：
Vector 侧用 `get_sys_cnt()`，SIMT 侧用公开的 `clock()` 接口，只测一个问题：
**当前 Vector kernel 中第一次 SIMTVF 调用，到 SIMT thread 0 执行入口时间戳的 cycle 数。**

## 文件结构

```text
simtvf_cycle_test/
├── simt_kernel/
│   └── simt_kernel.h       # 只放被测 SIMT kernel
├── cycle_count/
│   └── cycle_count.asc     # cycle 插桩、外层 kernel 和 host 启动
├── CMakeLists.txt
└── run.sh
```

`simt_kernel.h` 的唯一接口是 `SimtKernel(...)`。修改被测计算时只改这个文件；计时逻辑保持不变。

## 计时逻辑

```text
__vector__ kernel                        __simt_vf__

t0 = get_sys_cnt_safe()              // SIMD/Vector 侧
        |
        +--- asc_vf_call --------------> thread 0:
                                          entry = clock()  // SIMT 侧
                                          cycles[0] = entry - t0
                                          workload
        <--- PIPE_V -> PIPE_S wait ------+
t1 = get_sys_cnt_safe()              // SIMD/Vector 侧
cycles[1] = t1 - t0
```

输出：

| 字段 | 含义 |
| --- | --- |
| `cold_entry_cycles` | `entry-t0`，第一次 SIMTVF 调度到入口时间戳的 cycle |
| `total_cycles` | `t1-t0`，包括启动、计算、同步和退出，只作为对照 |

如果只照抄 `cycle_count_demo` 的外层 `t1-t0`，得到的是完整 SIMTVF 区间，不能单独称为冷启动。
因此本例只比它多一个 SIMT 入口时间戳。

两种读法不要混用：

- 要完全复刻 `cycle_count_demo` 的计时方式，只分析 `total_cycles`；它只由外层两次
  `get_sys_cnt()` 相减得到，不依赖 SIMT 内部的 `clock()`。
- 要分析“从发起调用到 thread 0 进入 SIMT”的区间，分析 `cold_entry_cycles`；
  该区间必须取得 SIMT 内部入口时间戳，所以需要 `clock()`。
- `total_cycles - cold_entry_cycles` 是进入 SIMT 后到外层同步完成的剩余区间，
  包含 workload、退出和同步，不能直接称为单独的销毁开销。

## 第一步：只跑一个最小用例

默认配置是 32 threads、0 次循环计算、重复 30 次：

```bash
cd /home/dangy/workspace/Private-ascend/SIMT-Scalar/simtvf_cycle_test
./run.sh --build-only
./build/simtvf_cycle_count 0 0 30 | tee result_t32_k0.csv
```

程序参数依次为：

```text
simtvf_cycle_count [device_id] [workload_iters] [samples]
```

先检查：

1. 所有 `correct` 均为 `1`。
2. `cold_entry_cycles > 0`。
3. `total_cycles >= cold_entry_cycles`。
4. 30 次结果是否集中在一个窄区间。

## 后续才做固定性对照

第一步合法后，再保持同一个二进制、只改变运行时计算量：

```bash
./build/simtvf_cycle_count 0 64 30  | tee result_t32_k64.csv
./build/simtvf_cycle_count 0 256 30 | tee result_t32_k256.csv
```

改变 threads 需要重新编译，例如：

```bash
SIMT_THREADS=64 ./run.sh --build-only
./build/simtvf_cycle_count 0 0 30 | tee result_t64_k0.csv
```

计算量增加时，`total_cycles` 应增加，而 `cold_entry_cycles` 应基本不变；否则入口计时受到计算体污染。
不同 `SIMT_THREADS` 下比较 `cold_entry_cycles`，才能判断冷启动是否与线程数无关。

这里的“冷启动”指 **单个 Vector kernel 中第一次 SIMTVF 调用**，不是设备上电或休眠唤醒。

## 计时接口合法性

不要在 `__simt_vf__` 中调用 scalar 内建函数 `get_sys_cnt()`，即使显式写成
`__cce_scalar::get_sys_cnt()`，也会在 CCEC 的 SIMT 后端阶段失败。

本例兼容服务器使用的 CANN 9.1.0-beta.3：

- Vector 侧用原有的 `get_sys_cnt()`；
- SIMT 侧 `__asc_simt_vf::clock()` 的实现读取 SIMT `CLOCK64`；
- 官方将两者都定义为从程序开始累计的 Cycle Count，因此可作时间戳差；
- SIMT `clock()` 仅在 Ascend 950PR/950DT 上受支持。

CANN 9.1.0-beta.3 尚未提供 `__asc_aicore::clock()`；该接口在正式版中才加入，
且其实现本身就是把 `get_sys_cnt()` 转为 `uint64_t`，因此这里直接使用
`get_sys_cnt()` 与正式版语义一致。
