# Agent Note: tinyllm_step ABI v3——逐序列 seq_status、惰性 positions 与分页 decode CUDA Graph

Status: implemented

## Problem

ABI v2 下 `tinyllm_step` 的失败语义是调用级的：任何序列出错（分配失败、
decode `seq_len != 1`、kernel 异常）都让整个调用返回 `TLLM_ERR`，调用方
无法区分"整批没跑"和"某一条序列挂了"，更无法让健康序列继续推进。上层
（paged-serving）不得不把单序列故障放大为整批 `BackendError`。

同时存在两个耦合问题：

- `positions` 在 v2 末期被收紧为严格校验（见
  [2026-11-08-ffi-ignored-inputs](../bug-fix/2026-11-08-ffi-ignored-inputs.md)），
  失败即 `TLLM_ERR`——这与"逐序列归因、不株连 batch"的新失败模型直接冲突；
- 分页 decode 每步要为一组 kernel 付 launch 开销，batch=1~4 的服务场景
  launch overhead 占比可观，但 kernel 地址与参数结构每步都重建。

## Decision

`tinyllm_step` 升级 ABI v3：

```c
int tinyllm_step(TinyLlmHandle h, const int *seq_ids, const int *tokens,
                 const int *seq_lens, const int *positions,
                 int num_sequences, int *output_tokens, int *seq_status);
```

- 新增必填出参 `seq_status[num_sequences]`：每序列写 `TLLM_SEQ_OK`（0）
  或 `TLLM_SEQ_ERR`（负）。`TLLM_OK` 只表示"调用本身执行完"，逐序列
  失败不再株连 batch；非法调用级参数（nullptr、`num_sequences <= 0`）
  仍返回 `TLLM_ERR`。
- `positions` 恢复为**惰性参数**：后端 `SeqState` 内部跟踪位置，参数
  不参与校验也不参与计算。这一条取代
  [2026-11-08-ffi-ignored-inputs](../bug-fix/2026-11-08-ffi-ignored-inputs.md)
  中的 positions 严格校验决定；该笔记的几何字段校验决定不受影响。
- decode `seq_len != 1` 不再是调用级拒绝：该序列记 `TLLM_SEQ_ERR`、
  不消耗 token 切片、不推进 KV/position，其余序列正常执行（原校验的
  "防 `toks[0]` 误用"意图由跳过切片实现）。
- 失败序列不推进 `SeqState`：KV 槽位与 position 保持失败前状态，重试
  语义安全。

策略 1（分页）decode 路径接入 CUDA Graph 重放：

- kernel 地址固化进 graph，token/位置/可见长度经固定 device buffer
  每步 H2D 后 `cudaGraphLaunch`；`TLLM_FFI_CUDA_GRAPHS=0` 可整体关闭
  回退逐 kernel launch。
- graph 资源（exec/graph/g_hidden_row）挂进 `TinyLlmHandleImpl`，由
  `~TinyLlmHandleImpl()` 统一回收——与
  [2026-11-08-ffi-load-failure-leak](../bug-fix/2026-11-08-ffi-load-failure-leak.md)
  的析构清理路径共用，load 失败中途路径同样覆盖。

配套：paged-serving `src/tiny_llm_ffi.rs` extern 声明与同批 PR 同步
v3 签名；`cross-repo-contracts.md` §10.1 记录双仓契约。

## Alternatives considered

- **保留 v2 严格 positions 校验，只加 seq_status。** 校验失败仍要
  `TLLM_ERR` 整批拒绝，与逐序列归因模型冲突；要么把校验降级为逐序列
  失败——但后端根本不读 positions，为不消费的参数维持一套校验只会
  制造误报面，惰性化更诚实。
- **positions 惰性 + decode `len != 1` 仍整批 `TLLM_ERR`。** 混合模型：
  调用方必须预判哪些错误是调用级哪些是序列级，语义不一致。统一到
  逐序列后，`len != 1` 的防护意图（防误读 `toks[0]`）由"不消耗切片"
  完整保留。
- **per-seq CUDA Graph（每序列一张图）。** 图按 batch 形状固化，序列数
  变化就要重录；整批一张图 + 固定 device buffer 覆盖变长参数更省。
- **graph 资源在 `tinyllm_free` 单独释放。** master 析构统一回收刚落地
  （[ffi-load-failure-leak](../bug-fix/2026-11-08-ffi-load-failure-leak.md)），
  另开释放路径会让 load 中途失败路径重新漏掉 graph 资源。

## Consequences

Benefits:

- 单序列故障可隔离：健康序列继续推进，上层能把缺失输出归因到具体
  `seq_id`。
- 分页 decode 每步省一组 kernel launch 开销；batch 形状不变时零重建成本。
- 失败语义统一：调用级错误（参数）与序列级错误（执行）不再混用
  `TLLM_ERR`。

Costs and limits:

- `seq_status` 成为必填出参——所有调用方（含 paged-serving extern）必须
  同步升级，v2/v3 混用会在 ABI 边界直接崩溃，双仓 PR 必须同批评审。
- `positions` 惰性意味着调用方位置记账错误不会再被后端发现；防线从
  "后端校验"移回"调用方与 `SeqState` 天然一致"——这是契约文档承诺的
  设计，不是校验保证。
- CUDA Graph 依赖 kernel 参数可固化：后续若引入每步变化的 kernel 形状
  （如变长 split-KV），该路径需重录或回退。

## Verification

- `FfiPagedDispatchTest` 8/8：`PartialBatchFailureDoesNotCascade` 验证
  逐序列隔离；env 注入失败验证 `seq_status` 归因；
  `TinyLlmStepInertPositionsAndPerSeqLenCheck` 验证惰性 positions +
  `len != 1` 逐序列失败。
- `FFITest` 5/5 真实模型（Qwen2.5-0.5B GGUF，RTX 5070 Ti / sm_120）：
  含 `PagedKVStrategyMatchesContiguous` 策略 1/2 C ABI 差分。
- lib + tests + bench 全量 nvcc 编译通过。
