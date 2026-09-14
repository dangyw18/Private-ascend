# Ascend AutoSimtVF

本上下文定义从 TileLang TIR 自动规划 Ascend SimtVF region 时使用的统一语言，供设计、实现、诊断和测试共同引用。

## Language

**Execution domain**:
statement 所属的执行域，例如 MainScalar、SIMT、MTE、CV 或已有 VF。
_Avoid_: engine type（仅描述硬件实现时使用）

**Statement role**:
Stage 1 对当前 lexical scope 中 statement unit 的唯一行为分类：SIMT_SEED、SIMT_FUSIBLE、REGION_BOUNDARY 或 UNSUPPORTED。
_Avoid_: placement kind、execution type

**SIMT seed**:
必须建立 SimtVF 的最小初始单元；多个 seed 可在 region 成形阶段融合。
_Avoid_: SIMT root、anchor、required-VF anchor

**Statement unit**:
在一个 lexical scope 中参与 role 分类和 region 规划的最小语法单元；最外层 Parallel 连同其完整子树只构成一个 unit。
_Avoid_: 对 Parallel 子树中的每个 AST 节点重复分类

**Region boundary**:
SimtVF 不得跨越的位置；hard boundary 在后续阶段也不可融合，policy boundary 仅表示 Stage 1 的确定性选择。
_Avoid_: region cut、MUST_OUTSIDE

**MustCoLocate**:
要求两个 statement 或值版本位于同一 VF 的约束。
_Avoid_: dependency（普通数据依赖不必然要求共址）

**Fragment escape**:
fragment version 的 def/use 跨越 R3 已确定的 region partition，必须交由 fragment legalization 处理的状态。
_Avoid_: legalizable edge、fragment dependency

**Fragment version**:
由一次 fragment 定义及其到下一次覆盖前的 uses 构成的逻辑值版本。
_Avoid_: fragment buffer（指物理 buffer 时除外）

**Fragment legalization**:
把跨 region fragment 值变换为合法显式状态传递的语义过程。
_Avoid_: fragment optimization、register reuse
