# Private-ascend

Ascend 性能测试与实验代码。

## 目录

- [`SIMT-Scalar/simtvf_cycle_test`](SIMT-Scalar/simtvf_cycle_test)：直接使用 AscendC/CCEC 和
  `get_sys_cnt()` 测量 kernel 内第一次 SIMTVF 调用的入口 cycle。
