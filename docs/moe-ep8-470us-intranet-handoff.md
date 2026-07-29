# MoE EP8 470 us 内网交接与环境复现指南

更新时间：2026-07-29

## 1. 目标

在 Ascend A2/A3 环境中验证并持续优化 DeepSeek V4 Flash standalone MoE：

- EP（Expert Parallel）固定为 8。
- balanced routing。
- 每个 rank 16 个本地专家。
- 固定使用偶数卡 `0,2,4,6,8,10,12,14`。
- 每次性能测试共 10 轮：2 轮 warmup + 8 轮计时。
- golden correctness 必须通过。
- 目标是把可重复的 round-wise last-arriving 延迟从约 `501 us`
  降到 `470 us` 或更低。

这里的“达到 470 us”必须是实际内核优化结果，不能仅靠改变统计口径、
减少测试轮数或去掉必要同步获得。

## 2. 当前代码状态

目标仓库和分支：

| 项目 | 内容 |
| --- | --- |
| 仓库 | `https://github.com/sunkaixuan2018/pypto-lib.git` |
| 分支 | `skx/optimize-moe` |
| 当前提交 | `262d69448b0a87376e0896744c74fe36984eee86` |
| 上游基点 | `a46139a`，来自 `hw-native-sys/pypto-lib` main |
| 性能基线提交 | `2c7392af841531f43db69cded522631c7bfaaa1f` |
| 当前候选状态 | 恢复基线 tile，仅将 gate/up pipeline 从 `stage=2` 改为 `stage=1` |

2026-07-29 已通过 `git ls-remote` 确认远端
`skx/optimize-moe` 指向 `262d69448b0a87376e0896744c74fe36984eee86`。

主要代码变化：

1. `models/deepseek/v4-flash/moe.py`
   - Flash MoE 的 EP 缩放改为每 rank 16 个专家。
   - standalone MoE 的 dispatch 前加入一次跨 rank barrier。
2. `models/deepseek/v4-flash/expert_routed.py`
   - 当前提交恢复基线 `K_TILE=512`、`MM_INTER_TILE=256`。
   - 仅 gate/up 两个 K 循环使用 `stage=1`；W2 仍为 `stage=2`。
3. `docs/moe-ep8-pmu-analysis.md`
   - 保存 gate/up、activation、W2 的 PMU 分析。

关键实验提交：

| 提交 | 单变量/用途 | 结论 |
| --- | --- | --- |
| `14ebe2b` | standalone dispatch 前增加 barrier | 正确；降低 host union，但未消除各 rank 启动阶梯 |
| `76d077e` | `MM_GATE_INNER=2`，增加 gate/up fan-out | 与基线处于噪声范围 |
| `2c7392a` | 恢复 `MM_GATE_INNER=4` | 当前性能基线 |
| `7090d05` | 仅 `MM_INTER_TILE=128` | 编译器仍不能分配两个 Right buffer，未上 NPU |
| `e34de6b` | 仅 `K_TILE=256` | 仍被拆成两个 32 KiB Right 子块，未上 NPU |
| `014d923` | `N=128,K=256` 组合 | 真双缓冲成功，但性能退化到 `521.5 us` |
| `262d694` | 基线 tile，仅 gate/up `stage=1` | 当前待完成性能对照 |

## 3. 已获得的性能和 PMU 证据

### 3.1 可比较的基线

早期基线（`2c7392a`）8 个 round-wise last-arriving 样本：

```text
493.9, 492.5, 500.3, 505.0, 502.9, 493.9, 514.5, 501.7 us
median = 501.0 us
mean   = 500.6 us
```

2026-07-29 重新从隔离 checkout 复测的原版基线：

```text
503.3, 502.0, 492.2, 497.1, 495.4, 499.8, 491.5, 493.7 us
median = 496.3 us
mean   = 496.9 us
```

这次复测：

- 提交：`2c7392a`
- correctness：PASS
- 配置：EP8、balanced routing、16 experts/rank、偶数卡
- 轮数：2 warmup + 8 measured
- `host_union_mean_us=4291`

`host_union_mean_us` 受 host 启动偏斜影响很大，只保留为诊断信息，
不能作为优化主指标。

### 3.2 正确的性能统计方式

每一轮从 8 个 rank 的 `effective_us` 中取最小值，表示该轮最后进入计算、
因此未包含前序等待的 rank。再对 8 轮结果计算 median 和 mean。

不要使用以下值判断 MoE 是否达到 470 us：

- 日志最上方跨 rank 的 headline `effective_us`。
- 含明显启动偏斜的 `host_union_mean_us`。
- level-3 trace 中包含 barrier 等待的最长 rank makespan。

### 3.3 已否决的 tile 组合

`014d923` 首次让 gate/up 的两个软件 stage 映射到两个独立 Right 地址，
编译器 `PH-MR-001` 警告消失，但端到端结果为：

```text
531.4, 515.8, 521.5, 521.8, 521.5, 520.9, 520.6, 525.2 us
median = 521.5 us
mean   = 522.3 us
```

它比约 501 us 的基线退化约 20.5 us。原因是小 tile 同时把 gate/up 的
SPMD block 数从 2 增加到 4、K 逻辑迭代从 8 增加到 16；新增循环、
task 和 DMA 发射开销大于双缓冲收益。该 tile 家族已经关闭，不应继续盲扫。

### 3.4 PMU 结论

type-2 PMU 的主要结果：

- gate、up、W2 的 MTE2 busy 为 82%～84%。
- Cube busy 约为 10%。
- 每个 projection 每 rank 权重约 128 MiB，三个 projection 共约 384 MiB。
- 权重读取是 routed expert 的第一瓶颈。
- `exp_gate_up_act` 占 routed AIV 总工作量约 64%：
  - Vector 约 43%。
  - Scalar 约 34%。
  - MTE2 约 33%。

type-2 PMU 不能区分 L2 与 HBM，也没有 FIXPIPE 计数，因此不能把
“MTE2 高”直接解释为某一层缓存命中率问题。

完整数据见 `docs/moe-ep8-pmu-analysis.md`。

## 4. 原版泳道图

已经采集并拉回一份接近原版 490～501 us 性能的 level-3 泳道。

采集信息：

| 项目 | 内容 |
| --- | --- |
| 基线提交 | `2c7392a` |
| 远端构建目录 | `_jit_l3_moe_20260729_014328` |
| 选择的 rank | rank5 |
| trace makespan | `510.54 us` |
| correctness | PASS |

本地文件：

```text
artifacts/moe-ep8-baseline-490us-swimlane-20260729/rank5/
├── merged_swimlane_20260729_014346.json
├── l2_swimlane_records.json
├── name_map.json
└── deps.json
```

把 `merged_swimlane_20260729_014346.json` 拖入
`https://ui.perfetto.dev/` 即可查看。

注意：trace 会引入扰动，因此 `510.54 us` 不能替代无 profile 基线的
`496.3 us` median。

## 5. 下一步验证顺序

每完成一个完整候选，都应先保存 correctness、8 个原始样本、median/mean、
必要的 compiler/PMU 证据，然后由两个独立 reviewer 交叉复核下一步。

### 5.1 第一优先级：完成 stage1 单变量对照

当前提交 `262d694` 只改变 gate/up 的 pipeline depth：

- `N=256,K=512` 保持基线。
- gate/up：`stage=2 -> stage=1`。
- W2：不变。

第一轮任务因为脚本内的期望完整 SHA 写错而在运行前退出，没有性能含义。
已修正并重新排队为 `task_20260729_014541_371098729099`。内网环境应直接
按本文命令重新验证，不依赖该队列。

判断：

- 若明显优于相邻基线：保留 stage1，复测确认稳定性。
- 若处于噪声范围：说明当前被编译器串行化的 depth-2 没有可观收益，
  关闭这条杠杆。
- 若明显退化：说明 stage2 仍保留了部分有效重叠，恢复 stage2，
  再考虑保持 `N256/K512` 的显式 lifetime/pipeline 重构。

### 5.2 第二优先级：gate/up activation

仅在 stage1 对照完成并经过交叉复核后进入。

依据是 `exp_gate_up_act` 占 routed AIV 工作量约 64%，且 Vector、Scalar、
MTE2 都有明显占用。候选必须一次只改变一个因素，例如：

- 调整 activation 的 tile/inner，观察 task 数和 vector 利用率。
- 减少重复的 scale/activation 数据读取。
- 在数值完全等价的前提下减少 clip、sigmoid、SILU 路径中的中间操作。

每个候选必须先 golden PASS，再比较完整 EP8 数据；不要仅凭单 task
cycle 或 compile hint 宣布收益。

### 5.3 暂不优先

- 继续扫 `N/K/MM_GATE_INNER`：已有证据表明容易增加小 task 开销。
- 给已经淘汰的 `014d923` 补 type-2 PMU：不能改变下一步决策。
- 单独融合 gate/up：权重总读取量不变，PMU 不支持把它列为第一实验。
- 优先优化通信：当前 routed expert 的权重移动证据更强。

## 6. 固定依赖版本

为了在内网得到可比较结果，先使用下面的固定版本，不要直接取各仓库
最新 main：

| 仓库/组件 | URL | 固定提交/版本 | 用途 |
| --- | --- | --- | --- |
| pypto-lib 优化分支 | `https://github.com/sunkaixuan2018/pypto-lib.git` | `262d69448b0a87376e0896744c74fe36984eee86` | MoE 代码 |
| PyPTO | `https://github.com/hw-native-sys/pypto.git` | `1784c635b5ea750c5666dde0fe132aa2c1d10a34` | 编译前端 |
| Simpler | `https://github.com/hw-native-sys/simpler.git` | `9922afdb08cc6f203eaf39328661e2f2648d333d` | PyPTO runtime 子模块 |
| PTO-ISA | `https://github.com/hw-native-sys/pto-isa.git` | `83d01313d9bfc247c4b7c8bcf969d1019f0d106f` | ISA 定义 |
| libbacktrace | `https://github.com/ianlancetaylor/libbacktrace.git` | `6f8310e238fc3ce68f42f391cbe93fd156bb2c23` | PyPTO 子模块 |
| msgpack-c | `https://github.com/msgpack/msgpack-c.git` | `919908742b4fdbc575e77fe1a8657e70c9573c44` | PyPTO 子模块 |
| pypto-serving（可选） | `https://github.com/hw-native-sys/pypto-serving.git` | `e57e14fbf29ea7ef3c06f4edf666814eedad2c0b` | HTTP server 冒烟 |
| CANN | 预装 | `9.0.0` | NPU 软件栈 |
| PTOAS | 预装 | `0.48`，路径 `/usr/local/ptoas/0.48` | 汇编/编译工具 |
| Python | 预装 | `3.10`，原环境为 `3.10.9` | Python 运行时 |
| torch | 内网 CPU wheel 或系统包 | 原环境为 `2.8.0+cpu` | golden/reference |

`pypto-serving` 不是当前 MoE 性能证据的一部分。当前 checkout 的 HTTP
服务主要面向 Qwen3-14B；启动它只能证明 serving 环境正常，不能证明
DeepSeek V4 Flash MoE 达到 470 us。

## 7. 内网可访问 Git 镜像时：下载全部代码

下面命令在 Linux 内网机器执行。先把 `STACK_ROOT` 改成实际可写目录；
若不能访问 GitHub，把各 URL 改成对应的内网镜像 URL。

```bash
set -euo pipefail

export STACK_ROOT=/data/your_account/moe-470us
export REPOS_ROOT="$STACK_ROOT/repos"
mkdir -p "$REPOS_ROOT" "$STACK_ROOT/logs"

export PYPTO_LIB_URL=https://github.com/sunkaixuan2018/pypto-lib.git
export PYPTO_URL=https://github.com/hw-native-sys/pypto.git
export SIMPLER_URL=https://github.com/hw-native-sys/simpler.git
export PTO_ISA_URL=https://github.com/hw-native-sys/pto-isa.git
export SERVING_URL=https://github.com/hw-native-sys/pypto-serving.git

git clone --branch skx/optimize-moe --single-branch \
  "$PYPTO_LIB_URL" "$REPOS_ROOT/pypto-lib-opt"
git -C "$REPOS_ROOT/pypto-lib-opt" checkout \
  262d69448b0a87376e0896744c74fe36984eee86

git clone --recursive "$PYPTO_URL" "$REPOS_ROOT/pypto"
git -C "$REPOS_ROOT/pypto" checkout \
  1784c635b5ea750c5666dde0fe132aa2c1d10a34
git -C "$REPOS_ROOT/pypto" submodule sync --recursive
git -C "$REPOS_ROOT/pypto" submodule update --init --recursive

# 独立 Simpler checkout，便于单独验证；PyPTO/runtime 也会指向同一提交。
git clone "$SIMPLER_URL" "$REPOS_ROOT/simpler"
git -C "$REPOS_ROOT/simpler" checkout \
  9922afdb08cc6f203eaf39328661e2f2648d333d

git clone "$PTO_ISA_URL" "$REPOS_ROOT/pto-isa"
git -C "$REPOS_ROOT/pto-isa" checkout \
  83d01313d9bfc247c4b7c8bcf969d1019f0d106f

# 可选：只有需要启动 HTTP server 时才下载。
git clone --recursive "$SERVING_URL" "$REPOS_ROOT/pypto-serving"
git -C "$REPOS_ROOT/pypto-serving" checkout \
  e57e14fbf29ea7ef3c06f4edf666814eedad2c0b
git -C "$REPOS_ROOT/pypto-serving" submodule sync --recursive
git -C "$REPOS_ROOT/pypto-serving" submodule update --init --recursive

# 固定版本检查。
test "$(git -C "$REPOS_ROOT/pypto-lib-opt" rev-parse HEAD)" = \
  262d69448b0a87376e0896744c74fe36984eee86
test "$(git -C "$REPOS_ROOT/pypto" rev-parse HEAD)" = \
  1784c635b5ea750c5666dde0fe132aa2c1d10a34
test "$(git -C "$REPOS_ROOT/pypto/runtime" rev-parse HEAD)" = \
  9922afdb08cc6f203eaf39328661e2f2648d333d
test "$(git -C "$REPOS_ROOT/pto-isa" rev-parse HEAD)" = \
  83d01313d9bfc247c4b7c8bcf969d1019f0d106f
```

内网 agent 在操作每个仓库前应先阅读该仓库的 `AGENTS.md`。

## 8. 内网完全不能访问外网时：Git bundle 方案

在能访问 GitHub 的机器上准备 bundle：

```bash
set -euo pipefail

export BUNDLE_ROOT=/data/your_account/moe-470us-bundles
export MIRROR_ROOT="$BUNDLE_ROOT/mirrors"
mkdir -p "$BUNDLE_ROOT" "$MIRROR_ROOT"

git clone --mirror https://github.com/hw-native-sys/pypto-lib.git \
  "$MIRROR_ROOT/pypto-lib.git"
git -C "$MIRROR_ROOT/pypto-lib.git" fetch \
  https://github.com/sunkaixuan2018/pypto-lib.git \
  skx/optimize-moe:refs/heads/skx/optimize-moe
git -C "$MIRROR_ROOT/pypto-lib.git" bundle create \
  "$BUNDLE_ROOT/pypto-lib.bundle" --all

git clone --mirror https://github.com/hw-native-sys/pypto.git \
  "$MIRROR_ROOT/pypto.git"
git -C "$MIRROR_ROOT/pypto.git" bundle create \
  "$BUNDLE_ROOT/pypto.bundle" --all

git clone --mirror https://github.com/hw-native-sys/simpler.git \
  "$MIRROR_ROOT/simpler.git"
git -C "$MIRROR_ROOT/simpler.git" bundle create \
  "$BUNDLE_ROOT/simpler.bundle" --all

git clone --mirror https://github.com/hw-native-sys/pto-isa.git \
  "$MIRROR_ROOT/pto-isa.git"
git -C "$MIRROR_ROOT/pto-isa.git" bundle create \
  "$BUNDLE_ROOT/pto-isa.bundle" --all

git clone --mirror https://github.com/ianlancetaylor/libbacktrace.git \
  "$MIRROR_ROOT/libbacktrace.git"
git -C "$MIRROR_ROOT/libbacktrace.git" bundle create \
  "$BUNDLE_ROOT/libbacktrace.bundle" --all

git clone --mirror https://github.com/msgpack/msgpack-c.git \
  "$MIRROR_ROOT/msgpack-c.git"
git -C "$MIRROR_ROOT/msgpack-c.git" bundle create \
  "$BUNDLE_ROOT/msgpack-c.bundle" --all

git clone --mirror https://github.com/hw-native-sys/pypto-serving.git \
  "$MIRROR_ROOT/pypto-serving.git"
git -C "$MIRROR_ROOT/pypto-serving.git" bundle create \
  "$BUNDLE_ROOT/pypto-serving.bundle" --all
```

把整个 `moe-470us-bundles` 目录复制到内网共享盘后，内网 agent 可以从
bundle 克隆。PyPTO 的三个子模块需要放入对应目录：

```bash
set -euo pipefail

export STACK_ROOT=/data/your_account/moe-470us
export REPOS_ROOT="$STACK_ROOT/repos"
export BUNDLE_ROOT=/mnt/internal_share/moe-470us-bundles
mkdir -p "$REPOS_ROOT"

git clone "$BUNDLE_ROOT/pypto-lib.bundle" "$REPOS_ROOT/pypto-lib-opt"
git -C "$REPOS_ROOT/pypto-lib-opt" checkout \
  262d69448b0a87376e0896744c74fe36984eee86

git clone "$BUNDLE_ROOT/pypto.bundle" "$REPOS_ROOT/pypto"
git -C "$REPOS_ROOT/pypto" checkout \
  1784c635b5ea750c5666dde0fe132aa2c1d10a34

git clone "$BUNDLE_ROOT/libbacktrace.bundle" \
  "$REPOS_ROOT/pypto/3rdparty/libbacktrace"
git -C "$REPOS_ROOT/pypto/3rdparty/libbacktrace" checkout \
  6f8310e238fc3ce68f42f391cbe93fd156bb2c23

git clone "$BUNDLE_ROOT/msgpack-c.bundle" \
  "$REPOS_ROOT/pypto/3rdparty/msgpack-c"
git -C "$REPOS_ROOT/pypto/3rdparty/msgpack-c" checkout \
  919908742b4fdbc575e77fe1a8657e70c9573c44

git clone "$BUNDLE_ROOT/simpler.bundle" "$REPOS_ROOT/pypto/runtime"
git -C "$REPOS_ROOT/pypto/runtime" checkout \
  9922afdb08cc6f203eaf39328661e2f2648d333d

git clone "$BUNDLE_ROOT/simpler.bundle" "$REPOS_ROOT/simpler"
git -C "$REPOS_ROOT/simpler" checkout \
  9922afdb08cc6f203eaf39328661e2f2648d333d

git clone "$BUNDLE_ROOT/pto-isa.bundle" "$REPOS_ROOT/pto-isa"
git -C "$REPOS_ROOT/pto-isa" checkout \
  83d01313d9bfc247c4b7c8bcf969d1019f0d106f

# 可选 HTTP server。
git clone "$BUNDLE_ROOT/pypto-serving.bundle" "$REPOS_ROOT/pypto-serving"
git -C "$REPOS_ROOT/pypto-serving" checkout \
  e57e14fbf29ea7ef3c06f4edf666814eedad2c0b
SERVING_LIB_COMMIT=$(
  git -C "$REPOS_ROOT/pypto-serving" ls-tree HEAD pypto-lib | awk '{print $3}'
)
git clone "$BUNDLE_ROOT/pypto-lib.bundle" \
  "$REPOS_ROOT/pypto-serving/pypto-lib"
git -C "$REPOS_ROOT/pypto-serving/pypto-lib" checkout \
  "$SERVING_LIB_COMMIT"
```

除了 Git bundle，还要把以下非 Git 依赖放进内网：

- CANN 9.0.0 安装包。
- PTOAS 0.48。
- Python 3.10。
- AArch64 CPU torch 2.8.0 wheel，或等价的内网系统 torch。
- Qwen3-14B 模型权重（仅启动 HTTP server 时需要）。

## 9. 安装 MoE 验证环境

以下命令假设 CANN、PTOAS 和 CPU torch 已由内网环境提供。

```bash
set -euo pipefail

export STACK_ROOT=/data/your_account/moe-470us
export REPOS_ROOT="$STACK_ROOT/repos"
export PYPTO_LIB_ROOT="$REPOS_ROOT/pypto-lib-opt"
export PYPTO_ROOT="$REPOS_ROOT/pypto"
export PTO_ISA_ROOT="$REPOS_ROOT/pto-isa"
export PTOAS_ROOT=/usr/local/ptoas/0.48

cd "$PYPTO_LIB_ROOT"
python3 -m venv --system-site-packages .venv

PYTHONNOUSERSITE=1 .venv/bin/python -m pip install -U pip
PYTHONNOUSERSITE=1 .venv/bin/python -m pip install \
  pytest pytest-xdist==3.8.0 pytest-forked \
  cloudpickle numpy scikit-build-core nanobind cmake ninja

# 使用固定 PyPTO 和固定 Simpler runtime 构建，不从 PyPI 拉同名包。
PYTHONNOUSERSITE=1 .venv/bin/python -m pip install \
  --no-build-isolation --no-deps -e "$PYPTO_ROOT"
PYTHONNOUSERSITE=1 .venv/bin/python -m pip install \
  --no-build-isolation --no-deps -e "$PYPTO_ROOT/runtime"

# 检查实际导入路径，防止误用用户目录中的旧 editable 包。
PYTHONNOUSERSITE=1 .venv/bin/python - <<'PY'
import cloudpickle
import pypto
import simpler
import torch

print("pypto:", pypto.__file__)
print("simpler:", simpler.__file__)
print("torch:", torch.__version__, torch.__file__)
print("cloudpickle:", cloudpickle.__file__)
PY

test -f /usr/local/Ascend/cann-9.0.0/set_env.sh
test -x "$PTOAS_ROOT/bin/ptoas"
test "$(git -C "$PTO_ISA_ROOT" rev-parse HEAD)" = \
  83d01313d9bfc247c4b7c8bcf969d1019f0d106f
```

如果 `import torch` 失败，应安装内网提供的 AArch64 CPU torch wheel。
不要直接执行无版本、无源限制的 `pip install torch`，以免下载错误的 CUDA
wheel。

可选的无 NPU 基础检查：

```bash
cd "$PYPTO_ROOT"
PYTHONNOUSERSITE=1 "$PYPTO_LIB_ROOT/.venv/bin/python" \
  -m pytest tests/ut/core/test_error.py -n auto --maxprocesses 8 -v

cd "$REPOS_ROOT/simpler"
PYTHONNOUSERSITE=1 "$PYPTO_LIB_ROOT/.venv/bin/python" \
  -m pytest tests/ut/py -m 'not requires_hardware' -v

cd "$PYPTO_LIB_ROOT"
PYTHONNOUSERSITE=1 .venv/bin/python -m pytest tests/golden -v
```

## 10. 启动 EP8 balanced MoE 验证

先生成一个固定运行脚本：

```bash
cat > "$STACK_ROOT/run_moe_ep8.sh" <<'BASH'
#!/usr/bin/env bash
set -euo pipefail

: "${STACK_ROOT:?STACK_ROOT is required}"
: "${REPO:?REPO is required}"

EXPECTED_DEVICES=0,2,4,6,8,10,12,14
if [[ "${TASK_DEVICE:-}" != "$EXPECTED_DEVICES" ]]; then
    echo "Unexpected TASK_DEVICE=${TASK_DEVICE:-unset}; expected $EXPECTED_DEVICES" >&2
    exit 2
fi

VENV="$STACK_ROOT/repos/pypto-lib-opt/.venv"
PTO_ISA_ROOT="$STACK_ROOT/repos/pto-isa"
PTOAS_ROOT=/usr/local/ptoas/0.48

cd "$REPO"
set +u
source /usr/local/Ascend/cann-9.0.0/set_env.sh
set -u

VENV_SITE=$(
  PYTHONNOUSERSITE=1 "$VENV/bin/python" -c \
    'import site; print(site.getsitepackages()[0])'
)
export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO:$VENV_SITE:/usr/local/lib64/python3.10/site-packages:/usr/local/lib/python3.10/site-packages"
export PTOAS_ROOT
export PTO_ISA_ROOT
export SIMPLER_PTO_ISA_COMMIT=83d01313d9bfc247c4b7c8bcf969d1019f0d106f
export PATH="$PTOAS_ROOT/bin:$PATH"

export PYPTO_LOG_LEVEL=error
export PYPTO_WARNING_LEVEL=none
export PYPTO_RUNTIME_LOG=error
export PYPTO_BENCH=1
export PYPTO_BENCH_ROUNDS=8
export PYPTO_BENCH_WARMUP=2
export PYPTO_BENCH_RAW=1
export PTO2_RING_DEP_POOL=16384
export PTO2_RING_TASK_WINDOW=16384
export PTO2_RING_HEAP=1073741824

echo "commit=$(git rev-parse HEAD)"
echo "devices=$TASK_DEVICE"
exec "$VENV/bin/python" models/deepseek/v4-flash/moe.py \
  -p a2a3 \
  --ep 8 \
  -d "$TASK_DEVICE" \
  --balanced-routing
BASH

chmod +x "$STACK_ROOT/run_moe_ep8.sh"
```

如果内网有与 myserver 相同的 `task-submit`：

```bash
export STACK_ROOT=/data/your_account/moe-470us
export REPO="$STACK_ROOT/repos/pypto-lib-opt"

/usr/local/bin/task-submit \
  --device 0,2,4,6,8,10,12,14 \
  --max-time 0 \
  --env STACK_ROOT="$STACK_ROOT" \
  --env REPO="$REPO" \
  --run "bash $STACK_ROOT/run_moe_ep8.sh"
```

如果没有 `task-submit`，只能在已经通过内网调度系统独占
`0,2,4,6,8,10,12,14` 后直接运行：

```bash
export STACK_ROOT=/data/your_account/moe-470us
export REPO="$STACK_ROOT/repos/pypto-lib-opt"
export TASK_DEVICE=0,2,4,6,8,10,12,14
bash "$STACK_ROOT/run_moe_ep8.sh"
```

不要在卡未独占时直接运行，也不要为了更快出结果改用其他卡号。

### 同时保留基线 checkout

避免来回切分支，可以新建独立 worktree：

```bash
export STACK_ROOT=/data/your_account/moe-470us
git -C "$STACK_ROOT/repos/pypto-lib-opt" worktree add \
  "$STACK_ROOT/repos/pypto-lib-baseline" \
  2c7392af841531f43db69cded522631c7bfaaa1f
```

随后把运行命令中的 `REPO` 改成
`$STACK_ROOT/repos/pypto-lib-baseline` 即可复跑原版基线。

## 11. 安装并启动可选 HTTP server

此步骤用于验证 `pypto-serving` HTTP 环境，不是 DeepSeek MoE 470 us
验收。

```bash
set -euo pipefail

export STACK_ROOT=/data/your_account/moe-470us
export REPOS_ROOT="$STACK_ROOT/repos"
export SERVING_ROOT="$REPOS_ROOT/pypto-serving"
export PYPTO_ROOT="$REPOS_ROOT/pypto"

cd "$SERVING_ROOT"
python3 -m venv --system-site-packages .venv
PYTHONNOUSERSITE=1 .venv/bin/python -m pip install -U pip
PYTHONNOUSERSITE=1 .venv/bin/python -m pip install \
  pytest cloudpickle numpy scikit-build-core nanobind cmake ninja \
  safetensors transformers fastapi uvicorn pydantic aiohttp
PYTHONNOUSERSITE=1 .venv/bin/python -m pip install \
  --no-build-isolation --no-deps -e "$PYPTO_ROOT"
PYTHONNOUSERSITE=1 .venv/bin/python -m pip install \
  --no-build-isolation --no-deps -e "$PYPTO_ROOT/runtime"

PYTHONNOUSERSITE=1 .venv/bin/python -m pytest \
  tests/test_batching.py tests/test_parallel.py -q
```

准备好内网 Qwen3-14B 权重后启动：

```bash
export STACK_ROOT=/data/your_account/moe-470us
export SERVING_ROOT="$STACK_ROOT/repos/pypto-serving"
export MODEL_DIR=/data/models/Qwen3-14B
export PORT=8899

/usr/local/bin/task-submit --device auto --max-time 0 --run \
  "cd $SERVING_ROOT && \
   source /usr/local/Ascend/cann-9.0.0/set_env.sh && \
   export PYTHONNOUSERSITE=1 && \
   export PTOAS_ROOT=/usr/local/ptoas/0.48 && \
   export PTO_ISA_ROOT=$STACK_ROOT/repos/pto-isa && \
   export PATH=/usr/local/ptoas/0.48/bin:\$PATH && \
   export PTO2_RING_HEAP=4294967296 && \
   export PTO2_RING_TASK_WINDOW=1048576 && \
   export PTO2_RING_DEP_POOL=1048576 && \
   $SERVING_ROOT/.venv/bin/python -m python.cli.main \
     --model $MODEL_DIR \
     --backend npu \
     --platform a2a3 \
     --device {} \
     --host 0.0.0.0 \
     --port $PORT"
```

看到 `Application startup complete` 后检查：

```bash
curl --noproxy "*" "http://127.0.0.1:8899/health"

curl --noproxy "*" "http://127.0.0.1:8899/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Huawei is","max_tokens":32,"temperature":0.0}'
```

## 12. 给内网 agent 的交付要求

让内网 agent 最终返回：

1. 每个仓库的绝对路径、实际 commit 和 `git status --short`。
2. `pypto`、`simpler`、`torch` 的实际 import 路径。
3. CANN、PTOAS、PTO-ISA 的实际版本/commit。
4. golden correctness 结果。
5. 8 个 rank × 8 轮原始 `effective_us`。
6. 每轮 last-arriving 值及其 median、mean、min、max。
7. `host_union_mean_us`，但仅作为偏斜诊断。
8. 编译器 warning/hint 是否变化。
9. 如果候选有稳定收益，再补 PMU 或泳道，不要先对明显退化候选做重 profile。
10. 每个完整结果后，先做两位独立 reviewer 的交叉复核，再修改下一变量。

验收标准仍然是：相同 EP8 balanced、相同偶数卡、相同 2+8 轮口径下，
correctness PASS 且稳定 median `<=470 us`。
