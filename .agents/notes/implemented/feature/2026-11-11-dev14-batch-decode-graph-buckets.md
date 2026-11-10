# Agent Note: DEV-14 批量 decode graph 分桶 + 默认启用决策

Status: implemented

## Problem

DEV-13 实测发现批量 eager 与 graph 的收益来自两个不同开销轴：
eager 批量摊薄权重读取（c8 +87% vs eager-scalar），graph 消除逐 kernel
launch（c8 +50%）。B=2 时前者 < 后者（c2 回退 −11.5%），故门收窄
{4,8} 且批量保持 opt-in。剩余问题：批量前向能否进 graph 兼得两份
收益，并借此重估 B=2 与默认值。

## Decision

按 B 分桶捕获，**每 B 一张 graph**（B∈{2,4,8}），服务任意行集——与
scalar graph 同一设计：graph 只绑 handle 级固定缓冲地址，行差异全走
device 数据。补了两个缺口后 `runFfiPagedDecodeBatchDevice` 整体可捕获：

1. `views[r].visible_blocks` 原为行块数 host int，进
   `attention_decode_paged` 的 `table_len` 参数会被 graph 固化。
   kernel 内 `table_len` 仅是 `(b < table_len)` 越界保护（attention.cu
   L330），实际遍历由 device `visible_tokens` 界定；扁平表行 stride
   恒为 `max_num_blocks`，改传该常量即 graph 安全且语义不变
   （每行 nb 校验仍由 `tinyllm_step` 预扫描承担）。
2. 末层 hidden 映回原槽原按 `rows[r].s` 做 host 偏移 memcpy（槽位
   映射会被固化）。新增 `copy_rows_indexed` kernel（elementwise.cu）
   经 `d_batch_slots` device int[8] 索引映回，槽表随 metadata H2D。

捕获策略与 scalar 镜像：某 B 的首次批量步先 eager 执行本步、同步后
捕获（capture 不执行 kernel）并实例化；之后命中即 `cudaGraphLaunch`；
单 B 捕获/实例化失败仅标 `failed[B]` 保持 eager，不株连其它 B 与
scalar graph。metadata H2D、views 构造、`sequences[].position` 记账
全在 capture 之外。

**默认值翻正**：`TLLM_BATCH_DECODE` 由 opt-in（=1 开启）改为默认
启用（=0 显式关闭）。DEV-13 v3 四档复测全过收敛门：

| 并发 | graph-scalar | batch graph（默认） | Δ |
|---|---|---|---|
| c1 | 284.3 | 281–284（同路径 B=1） | 0% |
| c2 | 296.4 | 363.4 | +22.6% |
| c4 | 306.4 | 410.9 | +34.1% |
| c8 | 310.6 | 456.3 | +46.9% |

TPOT p50：c2 −24%、c4 −32.8%、c8 −41.6%；TTFT p95 无 >10% 回退；
重复极差 <1% 远小于收益。既定默认门槛（≥10% 收益、超噪声、c1
回退≤5%、无 TTFT 回退、无正确性错误、收敛门）全部满足。

## Consequences

- `TLLM_BATCH_DECODE=0` 是回滚开关：默认代码路径之外的逐序列
  graph/eager 行为未变。
- B=3/5/6/7 与非 decode/prefill/logprobs/策略 2 行仍整体回落（模板
  未实例化、行语义不满足）。
- 批量 graph 为 handle 级资源，每 handle 至多 3 张（B=2/4/8 惰性捕获）；
  析构按 exec→graph 顺序释放。
- 结果包：paged-serving `results/2026-11-11-RTX5070Ti-dev13/graph-batch-v3*`
  （v3 门 {2,4,8} + graph 分桶；v3-default 为无 env 默认行为抽查）。

## Verification

- `BatchDecodeGraphBucketsMatchScalar`：B=4×6 步 → seq1 退出、seq4
  进（同 B 不同行集/槽位/块表）→ 再 6 步 → 降 B=2×4 步，与逐序列
  路径逐位一致；debug 日志确认 B=4/B=2 各捕获一次、后续步重放。
- 既有 `BatchDecodeMatchesScalarThroughCAbi`（B=4）与
  `FallsBackForUnsupportedCounts`（B=3 回落）无回归；baseline 改用
  `TLLM_BATCH_DECODE=0` 显式关闭。
- 全套 259 passed / 0 failed；memcheck 0 errors；racecheck 0 hazards
  （BatchDecode 三用例，含 graph 路径实跑）。
- 端到端：paged-serving 重链后 v3 扫 9 run + 默认行为抽查 1 run，
  数值与 env=1 一致（c8 453.5 tok/s）。

## 备选方案

- **行集签名键控 graph**（按 (B, 槽位集) 缓存多张）：可保留 host 偏移
  memcpy，但签名组合爆炸、命中率依赖负载稳定性；`d_batch_slots` 索引
  一处改动即免键控，更简单且覆盖任意行集。
- **padding B=3/5/6/7 到 4/8**：需保留物理块供死行 scatter，复杂度
  高、白付算力；维持回落。
- **捕获 H2D memcpy 入 graph**：host 源指针（栈数组）不稳定，不可行；
  upload 保持 capture 之外。
- **维持 opt-in 不翻正**：DEV-13 门槛全部满足后再维持关闭反而背离
  既定决策框架；但证据仍限单模型/单负载/单硬件，已在 report 与
  career-prep 如实限定。
