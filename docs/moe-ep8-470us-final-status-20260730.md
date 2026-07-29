# MoE EP8 470 us 优化阶段结论

## 当前结论

本轮没有达到 `470 us`。

- 稳定基线：`496.3 us`
- 目标：`<=470 us`
- 当前差距：`26.3 us`
- 当前状态：`pause_recommended`
- 当前分支：`skx/optimize-moe`
- 当前提交：`830966a72165ef047d67231431526dedb1a6f9b5`

当前提交已经恢复稳定的生产 MoE 实现，只保留独立性能探针和分析材料。
本轮不再提交新的 NPU 任务。

## 最重要的性能证据

PMU 表明三次 routed expert 矩阵乘是主要瓶颈：

- MTE2 busy 约为 `82%–84%`
- Cube busy 约为 `10%`
- 每卡每次 MoE 读取约 `384 MiB` INT8 专家权重
- 权重约占三次投影输入流量的 `94%`

因此，剩余的通信、shared expert、`hc_post` 和小型 Vector kernel
都没有足够的独占时间覆盖 `26.3 us`。

泳道图进一步给出：

- dispatch gather 的可消除上限约 `8.22 us`
- per-expert combine 提前启动的理想收益上限约 `3.68 us`
- shared expert 已经被 routed expert 隐藏
- `hc_post` 和末端清理合计只有约十微秒

## 本阶段新增验证

### 1. W2 matmul 与 epilogue 融合

生成物仍通过显式 GM slot 传递 INT32 accumulator，并未消除 GM 往返。
候选在编译结构门禁被拒绝，没有提交 NPU。

### 2. routed activation 与 quant 融合

两个实现都通过 correctness，但均明显退化：

- 8 行映射：median `524.1 us`
- 2 行映射：median `518.5 us`

该方向已经关闭。

### 3. 通信关键路径

基于 rank5 的真实 pid4 kernel 区间重新计算后：

- gather 最大 kernel 为 `8.22 us`
- combine 静态拆分的理想收益只有 `3.68 us`

两项都不足以填补 `26.3 us`，通信方向关闭。

### 4. 原生 INT4/NZ AIC 探针

干净探针使用：

- host 预打包 signed-INT4 activation planes
- Catlass 官方 `layout::zN` packed weight
- AIC-only
- 无 AIV、TileCast、INT8 weight workspace

PTOAS 0.48 结果：

- exact INT32 correctness：PASS
- 8 个测量值：`60.5, 32.0, 32.2, 32.8, 33.4, 32.7, 32.8, 33.6 us`
- median：`32.8 us`
- 相比 INT8 control `56.5 us`，降低 `41.9%`
- MTE2 aggregate：`179489 -> 87948`，降低 `51.0%`

它证明真正的 packed INT4/NZ 可以减少权重搬运，但 median 比预注册的
`<=32 us` 门禁慢 `0.8 us`。

唯一允许的 FullLoadA 修复结果：

- exact correctness：PASS
- median：`39.3 us`
- MTE2 仅再降低 `3.4%`
- total cycles 增加 `10.6%`

因此 native INT4/NZ policy、tile、block 变体正式关闭，没有集成到 MoE。

### 5. PTOAS 0.50 外部能力检查

myserver 同时安装了 PTOAS 0.48 和 0.50，因此额外用同一个干净探针验证
0.50 是否属于新的底层能力。

结果：

- exact correctness：PASS
- 8 个测量值：`35.3, 44.4, 37.0, 47.4, 38.7, 36.2, 33.3, 33.0 us`
- median：`36.6 us`
- PMU total 只降低 `2.1%`
- PMU MTE2 只降低 `1.6%`

它比 PTOAS 0.48 更慢，未达到预注册的 `<=22.8 us` 重开门槛。
PTOAS 0.50 路线关闭。

## 为什么当前暂停

当前足够大的瓶颈仍然是三次投影的权重读取，但能直接改变这条路径的
仓内方案已经完成验证或被实现能力阻塞：

- tile、stage、block、fanout 调整
- INT8 ping-pong
- grouped gate/up
- activation/quant 和 W2 epilogue 融合
- Catlass packed W4
- CANN GroupedMatmul
- vLLM-Ascend W4 full fusion / MC2
- native INT4/NZ
- PTOAS 0.50

剩余小项的独占收益均小于约 `10 us`，没有可信方案可以单独覆盖
`26.3 us`。继续提交微调任务只会重复已有负证据。

## 重新启动优化的条件

出现新的 CANN、PTOAS、PyPTO 或 Catlass 底层能力后，先运行同一个
production-shaped 独立投影探针。

同时满足以下条件才重新进入 MoE 集成：

1. exact correctness PASS
2. 2 次 warmup + 8 次测量
3. median `<=22.8 us`
4. 相比 PTOAS 0.48 的 `32.8 us` 稳定改善至少 `10 us`
5. 权重 MTE2 指标有明确、可重复的实质变化

在这些条件出现前，稳定对照仍使用 `496.3 us` 基线。

## 证据位置

- PMU 总报告：`docs/moe-ep8-pmu-analysis.md`
- 内网复现指南：`docs/moe-ep8-470us-intranet-handoff.md`
- 原版泳道图：`artifacts/moe-ep8-baseline-490us-swimlane-20260729/`
- activation/quant 日志：`artifacts/moe-ep8-act-quant-fusion-20260729.log`
- INT4/NZ 探针：`artifacts/moe-int4-nz-probe-20260729/`
- FullLoadA 探针：`artifacts/moe-int4-nz-fullload-probe-20260729/`
- PTOAS 0.50 探针：`artifacts/moe-int4-nz-ptoas050-probe-20260729/`
