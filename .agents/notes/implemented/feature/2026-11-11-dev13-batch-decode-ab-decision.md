# Agent Note: DEV-13 端到端 A/B 与批量门收窄 {4,8} 决策

Status: implemented

## Problem

DEV-12 交付的 `TLLM_BATCH_DECODE` opt-in 路径需要端到端判定：相对
**当前默认**（FFI CUDA Graph 逐序列重放，即 graph-scalar 强基线）是否有
可声明的收益，以及默认值该如何处置。单看 eager-scalar 对照会高估收益
——DEV-02 已测得 graph 本身是 decode 主要优化项（TPOT −35~45%）。

## Decision

四臂 A/B（paged-serving 闭环 c1/2/4/8 × work × 3 重复，RTX 5070 Ti，
Qwen2.5-0.5B W8A16）后做出两处决策：

1. **FFI 批量门由 {2,4,8} 收窄为 {4,8}**（`src/ffi.cpp` 预扫描）。实测
   graph+batch 在 c2 相对 graph-scalar 回退 ~11%（258–272 vs 296 tok/s）：
   B=2 eager 批量的权重摊薄收益 < graph 重放省下的逐 kernel launch 开销。
   kernel 模板实例 {2,4,8} 不变——M=2 仍是合法 kernel 形状，只是 FFI 不再
   在该行数下选用。v2 复测 c2 回到 296–298 tok/s，与 graph-scalar 持平。
2. **默认保持关闭（opt-in）**。收窄后 {4,8} 门在 c4/c8 对 graph-scalar
   全胜（吞吐 +10.6%/+24.8%，TPOT −12.7%/−26.2%），c1/c2 与基线路径
   逐位一致，TTFT p95 无 >10% 回退；但证据只覆盖单模型/单负载/单硬件，
   且 ttft_p95 存在 non_converged 单元格（c8），按既定门槛保守处置。

## Consequences

- `TLLM_BATCH_DECODE=1` 是测量支持的生产推荐配置：closed 并发 ≥4 的
  负载开它严格不劣于默认，c4/c8 显著更优；c1/c2 自动回落原路径。
- 回滚即删 env var——默认路径未改一行。
- 残留缺口：B=3/5/6/7 仍回落（模板未实例化）；批量路径 eager，
  非批量行才走 graph——按 B 分桶捕获（B∈{4,8} 各一张 graph）是
  DEV-14 候选，可同时覆盖 B=2 场景并进一步压缩 launch 开销。
- 结果包：paged-serving `benchmarks/serving/results/2026-11-11-RTX5070Ti-dev13/`
  （v1 四臂记录 {2,4,8} 门；graph-batch-v2 记录 {4,8} 门；dirty provenance
  如实披露：运行时 paged-serving 侧 dev13_ab.sh 未入库、v2 时 ffi.cpp 编辑
  未提交，本提交后 v2 源码可完整重建）。

## Verification

- 服务端到端：48+9 run 全部 `validate_results.py` 语义通过（12/12 ×4 臂 +
  9/9 v2），每臂 `audit_provenance.py` 通过。
- 差分：批量 on/off 经 C ABI 输出逐位一致（DEV-12 测试，本次无回归）。
- 全量 tiny-llm 套件 258 passed；memcheck 0 errors、racecheck 0 hazards。

## 备选方案

- **默认开启**：c4 吞吐提升 ~10.6% 贴近 10% 门槛下沿、单负载证据不足、
  ttft_p95 有 non_converged 单元格——按规格"门槛未满足保持 opt-in"执行，
  待 DEV-14 graph 分桶 + 更宽负载矩阵后再评。
- **B=2 保留批量**：eager-batch 臂内 B=2 相对 eager-scalar +29%，但该臂
  不是默认配置；相对真默认 graph-scalar 是回退——默认配置的对比口径为准。
- **{2,3,4,5,6,7,8} 全集合实例化**：编译期模板展开更多 M 值可行，但
  M=3/5/6/7 收益未经单测与端到端验证，寄存器压力未知；不扩大未验证面。
