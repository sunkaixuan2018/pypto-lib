# MoE EP8：DeepSeek V4 W4 融合 GMM 验证

## 结论

本轮验证的上游 `GroupedMatmulSwigluQuantV2` 在 myserver 当前
CANN 9.0.0 / A3 环境中无法正确处理生产形状，因此关闭并暂停该方向。

错误路径测得的约 `82-96 us` 不能作为性能数据，也不能用于端到端收益估算。
在当前环境中不再继续猜测 ABI、修改 group list 或扫描 tuning 参数。

## 为什么测试它

上游 vLLM-Ascend 在 2026-05-18 的 DeepSeek V4 算子提交中新增了
`GroupedMatmulSwigluQuantV2`。它在 A3 上明确包含 A8W4 路径，可将：

1. W13 GroupedMatmul
2. 反量化
3. SwiGLU
4. per-token INT8 再量化

放进一个算子。这个机制能同时减少真实 W4 权重搬运和中间张量写回，
与现有 PMU 指出的权重侧 MTE2 瓶颈一致。

验证使用上游 release 分支提交：

- vLLM-Ascend：`b22c2b44a8ed76a836f83479a50f42bd2e8e67c1`
- 算子最初引入提交：`b4f01009860b0b00e243e1eb95d0107b332c1bad`
- 本仓最终探针提交：`ec057ffc47c57db8d5fa3dad0ceeaa3b36655cc5`

## 测试形状

- `M=48`
- `E=16`
- 每专家 3 行
- `K=4096`
- W13 输出宽度 `N=4096`
- 输出 `48 x 2048`
- INT8 activation
- packed INT4 FRACTAL_NZ weight
- 2 次 warmup + 8 次测量
- 非零常量输入，使用闭式结果检查全部输出和 scale

## 三次运行

### 1. multi-tensor + count group list

测量值：

`96.4, 94.1, 93.8, 83.1, 83.2, 81.5, 93.3, 82.0 us`

中位数 `88.25 us`，但输出大面积未写，correctness FAIL。

### 2. single-tensor + count group list

改为上游 MoE 使用的单个 `[E,K,N]` tensor-list item。

测量值：

`83.9, 83.6, 82.1, 81.6, 80.8, 82.7, 68.8, 81.1 us`

中位数 `81.85 us`。只有前 6 行正确写回；其余
`42 x 2048 = 86,016` 个输出保持 sentinel，correctness FAIL。

### 3. single-tensor + cumulative group list

group list 改为 `[3,6,...,48]`，`groupListType=0`。

测量值：

`85.5, 82.8, 87.2, 84.1, 87.0, 86.8, 82.5, 84.7 us`

中位数 `85.1 us`。仍只有前 6 行正确写回，其余 86,016 个输出保持
sentinel，correctness FAIL。

三次运行的 workspace 均为 `83,882,496` bytes。

## 根因判断

两个独立审阅者都同意：

- 这不是普通数值误差，而是算子没有完整遍历 16 个专家。
- multi/single tensor-list 与 count/cumulative group-list 均不能改变
  “只写前两个专家”的边界，因此不值得继续扫这些接口参数。
- 最可能是上游算子与当前 CANN 9.0.0 的兼容性问题，或上游小 M
  A8W4 路径自身未覆盖该形状。

上游该 release 的 README 要求 CANN 9.0.1，而 myserver 当前为
CANN 9.0.0。这使得当前结果不能用于证明上游算子本身有缺陷，但足以证明
它不能在现有环境中作为可用优化。

Claude Code 审阅正常完成。OpenCode-DeepSeek 只调用一次并返回 HTTP 402，
随后按规则使用源码审阅兜底；没有重试。

## 重开条件

只有满足以下任一前置条件，才重开一次验证：

1. 存在隔离的 CANN 9.0.1 环境，并严格复用该提交的官方 Torch wrapper；
2. 上游提供 `E=16、每专家3行、A8W4 NZ` 的已知正确单测或修复。

重开后的硬门槛：

- 48/48 行全部写回；
- 无 sentinel；
- `bad_quant_count=0`；
- `bad_scale_count=0`；
- 48 个 scale 均为 finite；
- 正确性通过后，融合 W13 阶段中位数必须明显低于约 `45.6 us`，
  才允许进入 EP8 端到端集成。

## 证据

- `artifacts/moe-dsv4-gmm-20260730/vllm_dsv4_gmm_multitensor_20260730.log`
- `artifacts/moe-dsv4-gmm-20260730/vllm_dsv4_gmm_singletensor_20260730.log`
- `artifacts/moe-dsv4-gmm-20260730/vllm_dsv4_gmm_cumulative_20260730.log`
- 远端构建日志：
  `/data/sunkaixuan/skx_log_output/vllm_dsv4_gmm_build_20260730.log`
