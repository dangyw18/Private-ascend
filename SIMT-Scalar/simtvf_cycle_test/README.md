# SIMTVF 1-call/2-call cycle 差分测试

本测试完全使用 AscendC/CCEC 编译，不再在 SIMT 内部读取 `clock()`。它参考
`reduce_demo` 的方法，只在外层 `__vector__` kernel 使用 `get_sys_cnt()`，分别测量
一个 SIMTVF 调用和两个连续相同 SIMTVF 调用。

## 差分模型

对同一个 Case、同一个线程数 `M` 和数据量 `N`，分别编译两个二进制：

```text
T1 = F + R
T2 = F + 2R

increment R = T2 - T1
fixed F     = 2*T1 - T2
```

`R` 是连续增加一次相同 SIMTVF 调用带来的增量时间；`F` 是模型
`T(k)=F+k*R` 的固定项。`F` 可能包含首次 SIMD→SIMT 切换、最终排空、同步和计时边界，
所以在没有进一步基线前，应称为“固定开销估算”，不直接等同于纯启动时间。

这个模型本身也能检验假设：若 `T2` 接近 `2*T1`、`F` 接近 0，说明每次
`asc_vf_call` 很可能都承担近似完整的启动/退出成本，此时 1-call/2-call 差分无法
把单次启动从运行体中拆开。

## 两个 Case

两个 Case 使用相同的 GM 地址、线程组织和 thread-stride loop，只改变元素内的指令链：

```text
Case 1:
with T.SimtVF(threads=M):
    for i in T.Parallel(N):
        a[i] = b[i] + c[i]

Case 2:
with T.SimtVF(threads=M):
    for i in T.Parallel(N):
        a[i] = b[i] + c[i]
        a[i] = a[i] * a[i]
        a[i] = b[i] + a[i]
        a[i] = a[i] * a[i]
        a[i] = c[i] + a[i]
```

AscendC 实现位于 `simt_kernel/simt_kernel.h`。Case 2 使用一个局部标量保存 `a[i]`
的中间值，语义与上面的伪代码相同，同时避免不必要的 GM 中间写回。

## 默认矩阵

默认总共输出 16 组差分结果：

```text
Case: 1, 2
M:    32, 64, 128, 256
N:    256, 1024
```

- 固定 `N` 比较不同 `M`：观察启动线程数的影响；
- 固定 `M` 比较不同 `N`：观察数据量/每线程循环次数的影响；
- 固定 `M,N` 比较 Case 1/2：观察指令长度的影响。

每组需要编译 `VF_CALLS=1` 和 `VF_CALLS=2` 两个版本，因此默认共编译、运行
32 个二进制。每个版本默认采样 21 次，使用中位数计算 `T1/T2`。

## 一键运行 16 组矩阵

```bash
cd /home/d00957057/Private-ascend/SIMT-Scalar/simtvf_cycle_test
bash scripts/run_vf_startup_diff.sh
```

每组输出格式一致：

```text
case 1, M=32, N=256
1-call T1              : ... cycles
2-call T2              : ... cycles
increment R = T2-T1    : ... cycles
fixed F = 2*T1-T2      : ... cycles
```

脚本最后打印 16 组汇总表，并保存：

```text
results/simtvf_startup_diff_summary.csv
results/case*_m*_n*_vf*.csv
```

可以通过环境变量缩小或扩展矩阵：

```bash
SAMPLES=31 \
CASE_VALUES="1 2" \
THREAD_VALUES="32 64 128 256" \
DATA_VALUES="256 1024" \
DEVICE_ID=0 \
bash scripts/run_vf_startup_diff.sh
```

## 单个配置构建

例如只编译 Case 2、`M=64`、`N=1024`、2-call：

```bash
SIMT_CASE=2 SIMT_THREADS=64 DATA_SIZE=1024 VF_CALLS=2 \
bash run.sh --build-only
```

直接运行并采样 21 次：

```bash
SIMT_CASE=2 SIMT_THREADS=64 DATA_SIZE=1024 VF_CALLS=2 \
bash run.sh 0 21
```

单个二进制输出原始 CSV：

```text
sample,case,threads,data_size,vf_calls,total_cycles,correct
```

## 计时边界

```text
t0 = get_sys_cnt_safe()                 // PIPE_S
asc_vf_call<SimtCaseX>(...)             // 1 次或连续 2 次，PIPE_V
PIPE_V -> PIPE_S 同步
t1 = get_sys_cnt_safe()                 // PIPE_S
T  = t1 - t0
```

`get_sys_cnt()` 只出现在 Vector 外层，并使用 `noinline` 包装；SIMT kernel 内没有
`clock()` 或其他计时逻辑，因此被测指令体只包含用户给出的计算、必要的循环索引和 GM 访存。

## 文件结构

```text
simtvf_cycle_test/
├── simt_kernel/simt_kernel.h
├── cycle_count/cycle_count.asc
├── scripts/run_vf_startup_diff.sh
├── CMakeLists.txt
├── run.sh
└── README.md
```
