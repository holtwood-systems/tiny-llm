# Agent Note: batch decode 投影批处理设计包（DEV-10）

Status: implemented（设计已交付待评审）

## Problem

serving 实测（2026-11-10 RTX 5070 Ti 正式包）：closed c1→c8 吞吐停在
~283–310 tok/s 平台、TPOT 随并发近线性——调度层 batching 但层主体逐
序列执行，权重读取/launch 不摊薄。需要一份可独立评审的行映射/
workspace/oracle 设计，作为 DEV-11/12 的前置。

## Decision

设计包 `docs/architecture/batch-decode-projections-design.md`：
可选路径（paged + logprobs_k==0 + 纯 decode 行 + B≥2），B 行压紧
`[B,hidden]` 批量 embed/RMSNorm/QKV/O/FFN/残差，KV scatter 与
attention 逐行保留；metadata 全固定地址 per-handle 缓冲；
`TLLM_BATCH_DECODE` 默认关；eager 先行、graph 分桶后议；scalar 路径
+ 独立算子参考双 oracle，argmax 逐位一致 + 预登记 logits 容差。

关键事实依据（写进设计）：LayerWorkspace 缓冲已是 [max_batch,*]
形状；`d_block_tables` 从单表扩为 batch 扁平表+行偏移；M∈[2,8] 的
w8a16 dispatch 按 M*N 在 reference/tiled 间路由（DEV-11 实测对象）。

## Alternatives

- ragged batched attention 一次到位：复杂度不同量级，明确排除（§1）。
- 整路径重写 forward：评审面爆炸，只新增不替换。

## Consequences

DEV-11 解锁（小 M W8A16 实测）；DEV-12 按 §10 清单小提交接入。
生产 dispatch 未变；若需 ABI 变更须先交兼容设计（本设计不需要）。
