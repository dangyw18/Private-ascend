# TVM 数据流与指令依赖 DAG 调研

> 目标：为 TileLang/Ascend 自动划分 `T.SimtVF` scope 找到可复用的 TVM 官方分析设施，避免从零实现数据流分析。
> 调研基于本地 `tilelang-xy/3rdparty/tvm`（`c858aab`）及 Apache TVM 官方源码/文档，日期：2026-09-09。

## 结论先行

TVM **有通用的 TIR Block 级依赖分析，也有接近指令级的访问收集器，但没有一个可直接套用的“任意 TIR 语句精确依赖 DAG pass”**。

当前需求最适合复用三层能力：

1. 用 `StmtExprVisitor` 遍历、切分语句；用 `GetSBlockReadWriteRegion` 自动推导每个语句单元的 `BufferRegion`。
2. 复用 `DepKind`、`StmtSRef`/`SBlockScope` 的 RAW/WAW/WAR 模型，但不要直接把 `SBlockScope` 当精确 legality oracle。
3. 复用 `arith::Analyzer`/`IntSet`，或直接复用 TileLang 已有的 `RegionsMayConflict`，对同一底层 storage 的 region 做保守相交判断。

DAG 本身只需采用标准的“按程序顺序扫描 + 邻接表 + DFS/Kahn 拓扑”算法；不建议引入 Relax、GraphExecutor 或旧 Ansor `ComputeDAG`。

## 官方设施与适用性

| 设施 | TVM 中的用途 | 对指令 DAG 的复用结论 |
|---|---|---|
| `GetSBlockReadWriteRegion` | 从 SBlock body 推导读写区域，opaque 访问同时记为读和写 | **直接复用**为访问摘要前端 |
| `SBlockDependenceInfo` / `SBlockScope` | ScheduleState 的 Block 层 RAW/WAW/WAR 图 | **复用模型与接口；算法需薄封装/增强** |
| `InjectSoftwarePipeline::BuildDependencyGraph` | 校验软件流水的 producer→consumer 顺序 | 仅可参考；只有 RAW，非通用图 |
| `StorageAccessVisitor` / `ThreadSyncPlanner` | 记录语句访问、线程轴、storage scope并规划同步 | 数据结构很接近需求，但属内部实现且冲突判定较粗 |
| `StmtExprVisitor` / `PostOrderVisit` | 通用 TIR AST 遍历 | **直接复用** |
| `UndefinedVars` / `SideEffect` | 变量 use-def、调用副作用分类 | **直接复用**做标量依赖和保守边界 |
| `arith::Analyzer` / `IntSet` | 符号化简、范围估计、可证明比较 | **直接复用**做 region 快速排斥与保守判断 |
| Structural Equal/Hash | IR 等价判断、缓存键 | 只用于摘要缓存；**不能作为程序点/node identity** |
| Relax `DataflowBlock`/use-def | 纯函数式 binding 数据流与算子融合 | 层级过高，不处理 TIR buffer side effect |
| Relay dependency graph（历史） | Relay 表达式依赖与 post-DFS | 仅能参考邻接表/DFS写法，不能复用到 TIR |
| TE read graph / Ansor `ComputeDAG`（历史） | Tensor producer DAG / 调度 workload | 只有 producer 输入边，无 WAR/WAW、别名和控制依赖 |
| GraphExecutor graph | 运行时按 JSON 调用已编译算子 | 发生在 lowering 之后，不包含 TIR 语句访问 |

## 1. TIR Block 读写区域与依赖图

### 1.1 访问区域自动推导

官方 `GetSBlockReadWriteRegion(block, buffer_var_map)` 会遍历 Block body，收集 `BufferLoad`/`BufferStore`，将循环变量按定义域放松成 `IntSet`；无法识别的 opaque buffer 访问按“全区域、既读又写”处理。这正适合给一个普通语句外包 synthetic SBlock 后生成访问摘要。

- 官方 API：[s_tir analysis](https://tvm.apache.org/docs/reference/api/doxygen/s__tir_2analysis_8h.html)
- 官方实现：[sblock_access_region_detector.cc](https://github.com/apache/tvm/blob/main/src/s_tir/analysis/sblock_access_region_detector.cc)
- 软件流水也采用“语句 → 临时 SBlock → 自动推导 reads/writes”的方式：[inject_software_pipeline.cc](https://github.com/apache/tvm/blob/main/src/s_tir/transform/inject_software_pipeline.cc)

局限：结果通常是各维 `IntSet` 的包围区间，非凸集合会被扩大；未知 intrinsic 必须保守记为 opaque，或由 TileLang 补充其 read/write 语义。

### 1.2 `SBlockDependenceInfo` / `SBlockScope`

`SBlockDependenceInfo(PrimFunc/IRModule)` 可在 Python 侧直接构造，通过 `get_sref`、`get_sblock_scope`、`get_deps_by_src/dst` 查询依赖。`DepKind` 定义了 RAW、WAW、WAR、Opaque；当前 `SBlockScope` 构造器实际生成前三种，opaque 访问由 read/write region 收集器保守折叠为读和写。

- 官方结构：[sblock_scope.h](https://tvm.apache.org/docs/reference/api/doxygen/sblock__scope_8h.html)
- 依赖信息对象：[sblock_dependence_info.h](https://tvm.apache.org/docs/reference/api/doxygen/sblock__dependence__info_8h_source.html)
- 官方实现：[sblock_scope.cc](https://github.com/apache/tvm/blob/main/src/s_tir/sblock_scope.cc)

但源码有三个关键限制：

1. 它按 `Buffer` 对象身份分组，不检查两个 `BufferRegion` 是否真的相交；同 Buffer 的不相交分片会产生假依赖。
2. 两个不同 Buffer view 即使共享同一 `buffer->data`，也可能漏掉别名依赖；指令级分析应以底层 storage Var 为第一层 key。
3. 它保留所有历史 readers/writers，再给当前 Block 连边；单 Buffer 高频访问时最坏可生成 `O(N²)` 条边。

另需注意：当前 Python `DepKind.WAR` 文档仍写着 “Not supported ... for now”，而本地 C++ 构造器确实生成 WAR；构造器虽定义 `kOpaque`，实际也未单独生成该类边。这个接口存在版本/文档不一致，不能单独作为完整精确的 legality oracle。

`ScheduleState` 内部维护同一套 SRef/Scope 信息，`GetProducers/GetConsumers` 主要从 RAW/WAW 边派生查询结果；它是调度状态缓存，不是另一套更精确的依赖算法：[schedule analysis](https://github.com/apache/tvm/blob/main/src/s_tir/schedule/analysis/analysis.cc)。

因此，`SBlockScope` 很适合快速做一个保守原型，也适合作为结果对照测试；生产 pass 应复用其概念，自己封装 storage alias、region conflict 和 barrier 语义。

## 2. 官方最接近“指令 DAG”的实现

### 2.1 `InjectSoftwarePipeline::BuildDependencyGraph`

该内部函数按原顺序扫描 Block：维护每个 `buffer->data` 的历史 writers，遇到 read 就连接所有 writer→reader。复杂度为 `O(A + E)`，其中 `A` 是访问数、`E` 是实际生成的边数。

优点是以底层 data Var 识别 view alias，并已有 synthetic SBlock 模式；不足是只生成 RAW，不生成 WAR/WAW，也不比较 region，所以它只是软件流水校验器，不是通用指令 DAG pass。同一文件另有 `MayConflict` 逐维执行 `IntSet::Intersect(...).IsNothing()`，可直接参考为进入 Analyzer/Z3 前的低成本 region 排斥快路径。

官方源码：[BuildDependencyGraph](https://github.com/apache/tvm/blob/main/src/s_tir/transform/inject_software_pipeline.cc#L1040)

### 2.2 `StorageAccessVisitor` / `ThreadSyncPlanner`

`StorageAccessVisitor::AccessEntry` 已包含 buffer、read/write/sync 类型、每维 `IntSet`、线程轴、storage scope；它还会递归汇总 loop/if。这一结构比 BlockScope 更接近低层指令分析。

- 访问结构：[storage_access.h](https://github.com/apache/tvm/blob/main/src/s_tir/transform/storage_access.h)
- 同步扫描：[thread_storage_sync.cc](https://github.com/apache/tvm/blob/main/src/s_tir/transform/thread_storage_sync.cc)

`ThreadSyncPlanner` 用顺序模拟维护“尚未同步的 reads/writes”，并额外旋转一遍 loop 来发现 loop-carried conflict。不过其冲突规则主要是 buffer identity 与“相同线程索引”的特判，源码仍有 `TODO: more standard set based testing`。可复用访问记录和作用域汇总思想，不应直接照搬为精确 region dependence。

## 3. 可直接复用的基础算法

### AST 遍历

`tirx::StmtExprVisitor` 同时访问 Stmt 与 PrimExpr；`PostOrderVisit` 保证对象节点只访问一次。建议写一个轻量 visitor，收集每个平坦 `SeqStmt` 的直接子语句，遇到 serial loop、if/while、allocate、barrier、已有 VF scope 时建立嵌套边界。

- 官方文档：[StmtExprVisitor](https://tvm.apache.org/docs/reference/api/doxygen/classtvm_1_1tirx_1_1StmtExprVisitor.html)
- 遍历接口：[stmt_functor.h](https://tvm.apache.org/docs/reference/api/doxygen/stmt__functor_8h.html)

标量侧可复用 `tirx::UndefinedVars` 收集表达式依赖，并用 `tirx::SideEffect` 区分 pure/read-state/update-state/opaque 调用。按当前保守策略，普通标量语句和 `T.serial` 不进入 SimtVF；这些分析主要用于建立顺序边和给出明确的停止原因，而不是主动把标量吸收到 scope。

### 符号区间与可证明判断

`arith::Analyzer` 聚合常量范围、模分析、`IntSet`、化简和约束证明；适合执行如下保守规则：

1. storage Var 不同：无冲突；
2. 常量区间或同一结构表达式明显不相交：无冲突；
3. 用 `Analyzer::CanProve(a_max < b_min || b_max < a_min)` 尝试证明任一维分离；
4. 证明失败：按有冲突处理，必要时才进入 Z3。

官方 Python API：[tvm.arith](https://tvm.apache.org/docs/reference/api/python/arith.html)

### 结构等价与哈希

Structural Equal/Hash 做 alpha-aware 的 IR 内容等价，适合缓存 `(region A, region B, constraint context)` 的比较结果或识别重复表达式；程序中两个结构相同的 store 仍是两个不同程序点，因此 DAG node map 必须使用对象身份或显式递增 node id。

官方说明：[Structural Equality and Hashing](https://tvm.apache.org/ffi/concepts/structural_eq_hash.html)

## 4. 不建议直接复用的图

Relax `DataflowBlock` 明确表示纯、无副作用、无控制流的 binding 区域；`DataflowBlockUseDef`、`FunctionUseDef` 和 `TopologicalSort` 对变量依赖很好用，但看不到 TIR BufferStore、别名、WAR/WAW 或跨迭代依赖。

- [Relax DataflowBlock/use-def API](https://tvm.apache.org/docs/reference/api/doxygen/relax_2analysis_8h_source.html)
- [Relax TopologicalSort 实现](https://github.com/apache/tvm/blob/main/src/relax/transform/topological_sort.cc)
- [融合用 IndexedForwardGraph 与 post-dominator](https://github.com/apache/tvm/blob/main/src/relax/analysis/graph_partitioner.h)

历史 Relay `DependencyGraph` 用 `MixedModeVisitor` 构造表达式父子边和 `post_dfs_order`，只适用于 Relay Expr：[v0.10 dependency_graph.cc](https://github.com/apache/tvm/blob/v0.10.0/src/relay/analysis/dependency_graph.cc)。

当前 TE 仍有 `CreateReadGraph` 与 `PostDFSOrder`，后者是标准 `O(V+E)` DFS 拓扑序；它只处理 `Operation::InputTensors()`：[te/operation/graph.cc](https://github.com/apache/tvm/blob/main/src/te/operation/graph.cc)。旧 Ansor `ComputeDAG` 是 TE workload/schedule 状态，不是 lowered instruction graph。

GraphExecutor 消费 JSON operator graph 并调用 PackedFunc，已经丢失 kernel 内语句信息：[Graph Executor（v0.13 官方接口）](https://tvm.apache.org/docs/v0.13.0/reference/api/doxygen/graph__executor_8h.html)。

## 5. 面向自动 SimtVF 划分的建议实现

### 推荐管线

1. **建立分析单元**：只在同一平坦 `SeqStmt` 中建图；`T.serial`、分支、barrier、已有 VF 和无法安全调整的 lifetime 边界不跨越，但可递归分析其内部。`local.fragment` allocation 则作为必须与其所有访问共同闭合的 ownership 约束处理。
2. **收集访问**：对普通语句复用 synthetic SBlock + `GetSBlockReadWriteRegion`；对 TileLang tile op、copy/reduce 和特殊寄存器补一层 intrinsic 语义表。
3. **统一资源键**：Buffer 访问以 `buffer->data` 为 key；标量 Let/Var、barrier、特殊硬件状态可建模为 synthetic resource。
4. **生成边**：首版若按整个 storage 保守串行化，可维护单个 last-writer 和 readers-since-write，并且不能再用 region 不相交删除边。若要利用 partial/disjoint region，则必须维护 region-aware reader/writer frontier：只连接可能相交项，且仅当新 write 被证明覆盖旧 region 时才删除旧 frontier；opaque 节点视为全资源读写/fence。
5. **精化冲突**：先用廉价规则过滤，再调用 `Analyzer`；无法证明不相交就保留边。第一版把 serial 当边界；以后若允许跨迭代，再单独计算 loop-carried dependence/distance。
6. **图算法**：邻接表保存 `src/dst/kind/resource/region/distance`；用 Kahn 或 TVM `PostDFSOrder` 同型实现做 `O(V+E)` 拓扑，调试阶段加 SCC/cycle diagnostic。以 `T.Parallel` 为 seed 的依赖闭包直接用 BFS/DFS，复杂度同为 `O(V+E)`。
7. **scope 划分**：DAG 只提供约束。`with T.SimtVF` 还要求 AST 上的连续区间，因此应取闭包节点的词法最小/最大位置、补齐中间节点后重新做 legality check；VF 能力分类、线程映射、跨线程 RAW 的同步和 fragment 生命周期仍由独立 checker 决定。

### 与当前 TileLang 代码的结合

本地 TileLang 已有：

- `/home/dangy/workspace/tilelang-xy/src/ascend/transform/auto_schedule/memory_detector.h`：注明 adapted from TVM `BlockReadWriteDetector`；
- `/home/dangy/workspace/tilelang-xy/src/ascend/transform/auto_schedule/dependency_analysis.cc`：`RegionsMayConflict` 已按 storage alias、region、约束上下文和跨迭代距离做 Analyzer/Z3 判定；
- `AnalyzeDependencies` 已覆盖 RAW、WAW、WAR，但目前对同 storage 的节点/region 组合做嵌套枚举，热点路径最坏接近二次并触发多次 Z3。

所以最小成本方案不是重写分析器，而是：

1. 保留现有 `MemoryAccessDetector` 对 TileLang intrinsic 的扩展；逐步把普通 TIR 访问委托给官方 `GetSBlockReadWriteRegion`。
2. 保留现有 `RegionsMayConflict` 作为精准慢路径，前置 storage bucket、常量区间和 StructuralHash memo 快路径，避免对无关节点全局两两调用 Z3。
3. 将 `AnalyzeDependencies` 的图输出统一为 `DepKind + adjacency list`。无 region 精度要求的第一版使用 whole-storage 线性 frontier；精准版沿用现有全 pair 判交，或实现 region-aware frontier 后再做 region/Z3 精化，不能将“单个 last-writer”和“跳过不相交 region”直接混用。

### 复杂度预期

| 阶段 | 建议复杂度 | 说明 |
|---|---:|---|
| AST/访问收集 | `O(IR size)` | visitor 单遍扫描 |
| 整 storage 保守 DAG | `O(A + E)` | 线性 frontier；允许假依赖但保持词法顺序 |
| 拓扑排序 | `O(V + E)` | DFS 或 Kahn |
| partial-region 精化 | 最坏 `O(Σ A_b² × P)` | `A_b` 为同 storage 访问数；先分桶、区间过滤和缓存，Z3仅作慢路径 |

最终建议：**以 TVM SBlock access-region 分析为前端，以现有 TileLang `RegionsMayConflict` 为精化器，以标准线性扫描/拓扑算法为图骨架**。`SBlockScope` 用于概念复用和回归对照，而不是直接决定 SimtVF 合法性。
