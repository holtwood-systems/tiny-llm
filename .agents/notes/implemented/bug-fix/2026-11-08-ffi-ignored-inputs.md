# Agent Note: C ABI 不再静默忽略 positions 与几何字段

Status: implemented

## Problem

`tinyllm_step` 的 `positions` 参数从未被读取：后端始终按内部
`SeqState::position` 计算 RoPE 位置与 KV 写入槽位。调用方（paged-serving）
传入的 `positions` 语义与后端一致（prefill = 0..len-1，decode =
`context_len - 1`），所以现有链路能跑；但一旦调用方记账出错（chunked
prefill、错误 context_len、decode 传 len>1），后端不会发现，token 会写到
错误的绝对位置并产出静默错误的数值。

`tinyllm_load` 的 `TinyLlmConfig` 有 6 个几何字段（hidden_dim、num_layers、
num_heads、num_kv_heads、head_dim、vocab_size）。paged-serving 统一传 0，
注释说明“维度字段由 GGUF 提取”；但后端既不校验也不记录，若调用方传了
非 0 值，会以为自己的配置生效，实际按 GGUF 的另一套几何运行。

## Decision

`tinyllm_step`：

- `positions` 加入必填参数检查（nullptr → `TLLM_ERR`）。
- 每序列执行前校验：decode 的 `seq_len` 必须为 1；prefill 段
  `positions[i] == i`；decode 段 `positions[0] == st.position`。
- 不一致直接返回 `TLLM_ERR`（无部分执行，因为校验在任何 kernel 之前）。

> **2026-11-09 更新**：以上 positions 校验决定已被 ABI v3 取代——
> `positions` 恢复为惰性参数，`seq_len != 1` 降级为逐序列失败，
> 见
> [2026-11-09-abi-v3-seq-status-graphs](../feature/2026-11-09-abi-v3-seq-status-graphs.md)。
> 下述 `tinyllm_load` 几何字段校验决定仍然成立。

`tinyllm_load`：

- 6 个几何字段传 0 视为“未指定，采用 GGUF”；非 0 且与
  `extractModelConfig()` 的结果不一致时返回 NULL，错误信息形如
  `config.hidden_dim=896 does not match GGUF value 1024`。
- 校验发生在权重加载之前，失败路径无 GPU 资源分配。

同步更新 `include/tiny_llm/ffi.h` 与 paged-serving 的
`src/tiny_llm_ffi.rs` 契约注释；paged-serving 实际传参不变（几何字段仍为
0，positions 本来就与后端记账一致）。

## Alternatives considered

- **只补文档、不改行为。** 审计要求“校验或注释契约”；仅注释无法阻止调用方
  记账分叉后的静默错误，采用校验 + 注释。
- **要求 `positions` 与内部位置严格相等但允许 decode len>1。** decode 分支
  只 embed 第一个 token，len>1 的其余 token 会被静默丢弃；必须显式拒绝。
- **几何字段非 0 时覆盖 GGUF 配置。** 权重形状来自 GGUF，覆盖配置只会让
  加载器按错误形状解释文件；拒绝更安全。
- **允许几何字段为 0 以外的“未指定”哨兵（如 -1）。** 0 已是现有调用方的
  约定值，引入新哨兵没有收益。

## Consequences

Benefits:

- 调用方与后端的位置记账一旦分叉，立即得到 `TLLM_ERR`，而不是错误的生成
  结果。
- 几何字段的“指定了却没用上”变成显式错误。

Costs and limits:

- 这是 C ABI 行为收紧：此前传 `positions = nullptr` 或 decode `len > 1`
  的调用方会被拒绝；仓库内调用方（paged-serving 与 GTest）已核对并适配。
- 本机无 nvcc/GPU，`src/ffi.cpp` 未编译；`tests/test_ffi*.cpp` 未运行。
  需在有 CUDA 的环境跑 `tiny_llm_tests`，尤其是
  `FfiPagedDispatchTest`（已把 `decodeStep` 从 `pos = -1` 改为传真实
  绝对位置）。

## Verification

- 逐项核对仓库内 `tinyllm_step` 调用点：
  - `tests/test_ffi.cpp`：prefill positions 为 0..5、decode 从 6 递增，
    `TinyLlmConfig` 几何字段全 0 → 通过新校验；
  - `tests/test_ffi_paged_dispatch.cpp`：`decodeStep` 原先固定传 -1
    （依赖“被忽略”），已改为传 `kPromptLen + i` / `kPromptLen`；
    prefill 使用 `promptPositions()` = 0..len-1；
  - paged-serving `build_execution_batch`：prefill 0..len-1、decode
    `context_len - 1`，与后端 `st.position` 逐步一致。
- 静态核对：`tinyllm_step` 的校验位于所有 kernel launch 之前，失败不会留下
  部分执行的 KV / position 状态。
- `git diff --check`：通过。
- clang 18 CUDA host-only 语法检查（Triton CUDA 13 头 + `nv/target` 与
  logger shim，`-fgpu-defer-diag`）：`src/ffi.cpp` 通过，新增的
  positions / 几何字段校验可编译。
- CUDA 12.2.2 devel 容器 + `-DCMAKE_CUDA_ARCHITECTURES=89` 完整编译
  `tiny_llm_tests`（nvcc 12.2.140 / GCC 11.4），发现 261 项用例；新增
  `TinyLlmLoadRejectsMismatchedGeometry` 与
  `TinyLlmStepRejectsMismatchedPositions`，无 GPU 时按 fixture 跳过。
- 未运行：真实 GPU 上的 nvcc 设备执行、`tiny_llm_tests` 全量、paged-serving
  真后端集成测试（本机无 GPU）。
