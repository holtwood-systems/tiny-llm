# Agent Note: decode 行批量前向接线（DEV-12，TLLM_BATCH_DECODE opt-in）

Status: implemented

## Problem

DEV-10 设计包要求把策略 1 纯 decode 行的层主体 GEMM 从逐序列循环改为
`[B, hidden]` 单次调用以摊薄权重读取；DEV-11 实测并实现了小 M kernel
（`w8a16_matmul_smallm_transposed_kernel`，M∈{2,4,8} 时 B×M=1 提速
1.7–6.3×）。剩下的是把行映射、metadata 上传与逐行 attention 接到
`tinyllm_step`，且默认行为不能变。

## Decision

新增 `TransformerLayer::forwardPagedBatch`（transformer.cpp）与
`TLLM_BATCH_DECODE=1` 候选开关（ffi.cpp），eager、opt-in、不改 C ABI：

- **层主体批量**：rmsNorm/QKV/bias/O/FFN 一次 M=B 调用——这些 kernel 本就
  按 num_tokens 行批量，B∈{2,4,8} 时自动命中 DEV-11 的 small-M kernel。
- **逐行隔离保留**：`apply_rope_inplace_per_token_positions`（既有 kernel）
  一次覆盖全部行的绝对位置；scatter/attention 逐行循环，复用每条序列
  自己的块表切片与 `decode_len` device int——不引入新 attention kernel。
- **行压缩**：`tinyllm_step` 预扫描用与主循环同一套 ABI 校验收集行；
  行数恰为 {2,4,8} 才进入批量，否则全部回落——B=3/5/6/7 与
  prefill/logprobs/策略 2 行行为零变化。
- **metadata**：`d_batch_token_in/rope_pos/decode_len`（[8]）+ 扁平
  `d_batch_block_tables`（8×max_num_blocks，按行 stride）+ `d_batch_work`
  （[8×hidden]），每步 3 个定长 H2D + 逐行块表上传。
- **Graph**：批量路径 eager，不建不碰 graph；`TLLM_FFI_DEBUG` 下打印实际
  路径（batched N rows / fallback）。

## Consequences

- "decode 投影批处理可用"成立（opt-in）；不能声明完整 batch
  attention、ragged prefill 或生产吞吐提升——后者归 DEV-13 端到端 A/B。
- 失败语义：host 预校验可在层前按行归因；batch 内任一行非法或
  kernel/device 级错误 → 本批各行记失败（调用级），不株连其它序列。
- `LayerWorkspace` 缓冲已 [max_tokens,*]，零新增层缓冲；新 handle 缓冲
  约 8×max_num_blocks×4B + 8×hidden×2B。

## Verification

- `FfiPagedDispatchTest.BatchDecodeMatchesScalarThroughCAbi`：4 序列同
  step ×12 步、不同 prompt 长度（位置/块表/跨块尾部不同）、乱序物理块
  id，TLLM_BATCH_DECODE on/off 输出逐位一致（debug 日志确认批量路径
  实际执行 ok=1）。
- `BatchDecodeFallsBackForUnsupportedCounts`：B=3 回落逐序列，全行
  seq_status=OK。
- 全套 `*Paged*/*Ffi*/*W8A16*`：61 passed；memcheck 0 errors、
  racecheck 0 hazards（新路径实跑）。
- 待补：真实模型双 GQA 几何 golden、混合 prefill+decode step、
  KV 资源守恒专项——按 DEV-12 规格留差分清单下轮补齐。

## 备选方案

- **扁平块表 + bt_offsets kernel 参数（DEV-10 §6 原案）**：为 graph
  兼容预留；eager 路径逐行 view 直接用切片指针更简单，graph 化推迟到
  DEV-13 之后按需。
- **逐行 RoPE 循环**：可行但每行一次 launch；per-token-positions kernel
  已存在且一次调用覆盖全部行。
- **B=3/5/6/7 padding 到 4/8**：少一个回落分支，但死行要写隔离逻辑且
  白付算力；v1 直接回落，正确性边界最清楚。
