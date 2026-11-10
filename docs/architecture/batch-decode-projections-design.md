# Batch decode 投影批处理设计包（TLLM-BATCHDEC）

> **Status: PROPOSED（2026-11-10）—— 待 review，尚未批准。**
> 本文是 DEV-10 交付物：行映射、metadata、workspace、oracle 与失败/Graph 策略
> 的可独立评审设计。**设计未获批准前，不得修改 production dispatch，不引入新 C ABI。**
> 动机数据：paged-serving `results/2026-11-10-RTX5070Ti-graph-on` 显示 closed
> c1→c8 吞吐停在 ~283–310 tok/s 平台、TPOT 随并发近线性上升——调度层有
> batching 而层主体逐序列执行，权重读取与 launch 不摊薄。

## 1. Decision summary

- **Chosen design**：在 `tinyllm_step` 内新增一条**可选** decode 路径：把本步
  全部满足适用集合的 decode 行压紧为 B 行 `[B, hidden]`，一次 embed、逐层
  批量 RMSNorm/Q/K/V/O/FFN/残差，attention 与 KV scatter 仍按行逐序列执行，
  最后复用既有 `d_batch_hidden` 尾处理（`finalNormAndComputeLogitsBatch` +
  `greedy_argmax_batch`）映回原槽位。
- **适用集合（全部同时成立才走新路径）**：
  1. `h->paged_kv == true`（策略 1；策略 2 连续 KV 不纳入本设计）；
  2. `logprobs_k == 0` 的普通 greedy decode；
  3. `is_prefill[s] == false && seq_lens[s] == 1` 的行，且 `B ≥ 2`
     （B == 1 走既有单序列路径，不付批量开销）。
  prefill、混合步、logprobs、策略 2 一律走原路径——与原循环共存，
  同一 step 内两类行分开处理（见 §3 序）。
- **Why**：层主体每步对每个序列重复读同一套量化权重（Q4_K_M→W8A16），
  B 行共享权重读取理论上把投影的权重访存摊薄 ~B 倍；末端批处理已存在，
  本设计只补层主体缺口，改动面最小。
- **Rejected alternatives**：
  - *整层 attention 也批处理（ragged paged attention）*：收益最大但复杂度
    完全不同量级（变长可见窗口、跨块尾部、per-row block table）；属后续
  - *重写为通用 batched forward*：破坏 prefill/策略 2 现有路径的评审面，
    本设计只新增不替换。
  - *把 scalar 循环包成"假 batch"*（仍逐序列 kernel，只合并 H2D）：省 launch
    但权重读取不摊薄，测不到目标效应。

## 2. 行映射（row mapping）

host 侧预校验通过后，为每个入选行建立 `DecodeRow`：

| 字段 | 类型 | 来源 | 用途 |
|---|---|---|---|
| `slot` | int | 原输入下标 s | 结果映回 next_tokens[s]/d_batch_hidden[s] |
| `seq_id` | int | seq_ids[s] | SeqState 引用、状态推进 |
| `pos` | int | st.position | 绝对位置 → rope_pos[r]、decode_len[r]=pos+1 |
| `nb` | int | num_blocks[s] | 本行 block table 长度（校验 + 偏移） |
| `bt_off` | int | 扁平 block_tables 内偏移 | 本行表切片起点（device 侧 row→bt 映射） |

压紧规则：跳过即失败（`len<=0`、未分配、`pos>=max_seq_len`、块表非法）
的行不进 `DecodeRow`，`seq_status[s]` 维持 `TLLM_SEQ_ERR`，与原语义一致。
`pos` 在 scatter/gather 校验中即 `view.position`；`decode_len = pos + 1`。
结束/失败行不占行；step 后槽位复用由 SeqState 生命周期自然保证。

## 3. Metadata 布局、上传时机、地址稳定性、ownership

全部 **per-handle device 缓冲**，`tinyllm_load` 按 `max_batch_size` 一次分配
（沿用 `DeviceBuffer<T>` 惯例，RAII 释放；不因 B 变化反复 cudaMalloc）：

| 缓冲 | 形状 | owner | 上传时机 | 地址稳定性 |
|---|---|---|---|---|
| `d_batch_token_in` | int[max_batch] | handle | 每步一次 H2D（B 个 token） | 固定 |
| `d_batch_rope_pos` | int[max_batch] | handle | 同上（各行绝对位置） | 固定 |
| `d_batch_decode_len` | int[max_batch] | handle | 同上（pos+1） | 固定 |
| `d_batch_bt_offsets` | int[max_batch] | handle | 同上（各行表起点） | 固定 |
| `d_block_tables`（既有） | int[max_num_blocks] | handle | 每步一次 H2D（Σnb ≤ max_num_blocks 的连续扁平布局） | 固定 |
| `g_hidden_batch` | half[max_batch × hidden] | handle | kernel 内写 | 固定 |
| `g_hidden_row`（既有） | half[hidden] | handle | B=1/回退路径复用 | 固定 |

- `positions` C ABI 语义不变（惰性，不读取）；批量 RoPE 位置由后端
  `d_batch_rope_pos` 构建，**不**来自调用方。
- 现有 `d_block_tables` 语义从「逐序列单表」扩展为「全 batch 连续扁平
  表 + 每行偏移」；策略 1 单序列路径继续使用其前 nb 项，布局兼容。
- 上传合批：当前 decode 每序列 3 次 1-int H2D + 1 次块表 H2D；新路径每步
  共 4 次定长 H2D + 1 次块表 H2D，与 B 无关。

## 4. Workspace 尺寸与容量

`LayerWorkspace` 缓冲已是 `[max_batch, dim]` 形状（设计时预留），本设计
**不新增层缓冲**，只定义行维度 B 的实际用法；容量 `B_cap = max_batch_size`，
与 `max_seq_len`（位置上限）和 `max_visible_tokens`（KV 池容量）正交：

| 张量 | 尺寸公式 | 备注 |
|---|---|---|
| g_hidden_batch | B_cap × hidden_dim | decode 工作行，层间串行复用 |
| norm_output | B_cap × hidden_dim | 既有 ws_->norm_output |
| q_buf | B_cap × num_heads×head_dim | 既有 |
| k_buf/v_buf | B_cap × kv_dim | 既有 |
| attn_buf/attn_output | B_cap × hidden_dim | 既有 |
| ffn_gate/ffn_up | B_cap × intermediate_dim | 既有 |
| k_scratch/v_scratch | max_visible_tokens × kv_dim（**不按 B 放大**） | attention 逐行复用，见 §5 |

层间串行复用：每层的 RMSNorm→FFN 写读全部经同一组 ws_ 缓冲，与现状
一致；不按层复制。显存增量 = g_hidden_batch（B_cap×hidden×2B，Qwen2.5-
0.5B 时 8×896×2≈14KB）+ 4×B_cap×4B 标量——可忽略，**不误乘上下文长度**。

容量测试边界按实际公式：B ∈ {B_cap−1, B_cap, B_cap+1}；B_cap+1 时
第 B_cap+1 行回退原路径或整步回退（实现期定），不凭未定位常数设限。

## 5. 层内顺序与逐序列残留

每层顺序（attention 前批量，attention 逐行，attention 后批量）：

1. `rmsNorm(g_hidden_batch → norm_output, B)` — 既有 kernel 已支持 num_tokens。
2. `w8a16_matmul(norm_output → q/k/v_buf, M=B)` + `add_bias_inplace(B)`。
   **注意**：现有 dispatch 对 M∈[2,8] 按 `M*N` 选 reference/tiled（N=896 时
   B=2/4 走 reference、B=8 走 tiled）——形状相关路由是 DEV-11 的实测对象，
   本设计不预判其优劣，只要求 M>1 数值正确。
3. RoPE：**需要按行绝对位置的新变体** `apply_rope_inplace_rows`（每行独立
   `pos[r]`）；现有 `apply_rope_inplace` 只接受单一 device pos。
   若 DEV-12 实现期选择逐行调用现有 kernel，必须如实记录为「按行循环」
   而非批 kernel——默认设计目标是行索引变体。
4. KV scatter → pool：逐行 B 次调用现有 scatter kernel（各行块表切片
   `d_block_tables + d_batch_bt_offsets[r]`，pos/decode_len 用行标量）。
   逐序列保留属设计内，不宣称批量。
5. Attention：逐行 B 次 `attentionPaged` 等价调用，复用同一
   k_scratch/v_scratch（每次调用覆盖）。行乱序块表、跨块尾部沿用现有
   kernel 防护。
6. `w8a16_matmul(attn_buf → attn_output/wo, M=B)`、残差 `add_inplace(B)`。
7. `rmsNorm(B)` → `feedForward`（gate/up matmul M=B → swiglu → down M=B）
   → 残差 `add_inplace(B)`。

## 6. 失败语义

- **host 预校验**（进 B 行之前）：逐行可归因——非法 len/未分配/坏块表/
  pos 越界 ⇒ `seq_status[s]=TLLM_SEQ_ERR`，其余行照常（与原循环一致）。
- **批量段设备/kernel 失败**：不可归因到单行 ⇒ 按现有调用级失败处理：
  本步全部 B 行 `seq_status=TLLM_SEQ_ERR`（与单序列 kernel 失败的语义
  粒度一致），不扩大「单序列隔离」承诺，不引入 broad catch。
- 致命 CUDA 错误（context 毒化）维持既有行为：向上抛 TLLM_ERR。

## 7. Graph 策略

- **先 eager**：批量路径默认 eager 执行，实验开关 `TLLM_BATCH_DECODE`
  （0=off 默认 / 1=适用集合内启用）独立记录实际路径。
- 既有单序列 graph（绑定 g_token_in/g_decode_len/g_rope_pos/g_hidden_row
  固定地址）**不能套用到 B 行**：地址与形状都变。旧 handle graph 对
  B>1 步一律不重放。
- 稳定后再评估**按 B 分桶捕获**（B∈{2,4,8} 各一张，行数不足 bucket 时
  padding 空行、输出丢弃）：可行因为 g_hidden_batch/d_batch_* 均为固定
  地址，行内容经 H2D 注入。是否捕获属 DEV-12/13 决策，此处只预留
  缓冲设计兼容（已定：全固定地址）。
- 记录口径：实验开关值与实际使用路径（eager/replay）分别写入结果
  provenance，不把「配置了 Graph」当「使用了 Graph」。

## 8. Oracle 与容差

- **差分 oracle**：原 scalar 逐序列路径（同一 handle、同权重、同
  tokenizer）——同 prompt 逐位置 logits/token 对照。
- **独立算子参考**：测试侧 CPU dequant+矩乘参考（`tests/test_w8a16_matmul`
  已有 CpuReference 模式）验证投影层；**两路径不得共享同一实现**
  （scalar kernel 与 batched kernel 走不同 dispatch 分支天然满足）。
- **容差（看结果前登记）**：greedy golden 样本 **argmax token 逐位一致**；
  logits（fp16）报告 max/p99 |Δ|，门禁 bound `|Δ| ≤ 0.125`（fp16 在
  logits 幅值域的舍入量级）；发生边界 argmax 分歧时保留首个分歧、
  top-2 margin 与独立参考，提交审阅，不改 expected。

## 9. 测试设计

| 类别 | 用例 |
|---|---|
| 行数 | B=1（回退）/2/4/8；B_cap±1 边界 |
| 行位置 | seq_id 不连续（如 {0,3,7}）；context 不等长（pos 各异） |
| 失败行 | 首/中/尾行失败（坏块表、越界 pos、未分配 seq_id） |
| 块表 | 物理块乱序、跨块尾部（nb 非整除）、Σnb > max_num_blocks 拒绝 |
| 生命周期 | 行结束后释放、新请求复用同槽位 |
| 回退 | prefill+decode 混合 step、logprobs>0、策略 2 |
| 尺寸 | N−1/N/N+1 按 `B_cap=max_batch_size` 实际公式 |
| Oracle | 两种真实 GQA 几何（14→2 与 12→2 或 32→8）golden token 对照 |

## 10. 逐步实施清单（DEV-12 按此推进，均为小提交）

1. 行收集与 `DecodeRow` 预校验（纯 host，单测）。
2. metadata 缓冲分配与合批 H2D（单测计数上传次数/布局）。
3. `TLLM_BATCH_DECODE` 开关 + B 行 embed + `[B,hidden]` 批量 RMSNorm/
   QKV/bias/RoPE-rows（逐行 scatter/attention 暂用循环）+ 批量 O/残差/FFN；
   结果经 d_batch_hidden 映回。
4. 差分测试接入 oracle（scalar vs batch 同步 logits/token）。
5. eager 稳定后评估 bucket 化 graph（可选，单独提交）。
6. DEV-13 端到端 A/B 与默认值决策。

## 11.5 DEV-12 就绪结论（2026-11-11，DEV-11 落地后补充）

DEV-11 完成后复核 `forwardPaged`/`attentionPaged`/`feedForward`：层主体已按
`num_tokens` 行批量——rmsNorm、Q/K/V/O、FFN 七个 GEMM 与 bias/silu/add
全部行数无关，`M∈{2,4,8}` 时自动走新 small-M kernel。**因此 DEV-12 不需要
新 kernel**，需要的新代码只有：

- `TransformerLayer::forwardPagedBatch(x[B,H], views[B])`：批量 norm/QKV/
  bias → **逐行** `apply_rope_inplace`（`rope_pos=d_pos+r`，kernel 不变）→
  **逐行** `paged_scatter_blocks_dev` + `attention_decode_paged`
  （`block_table=该序列自有指针`、`decode_len=d_len+r`）→ 批量 O/残差/FFN。
- ffi.cpp 行压缩与开关：eager 路径无需 DEV-10 §6 的扁平块表——每行直接用
  序列自有 `d_block_table` 指针；每步 2 个定长 H2D（`d_batch_rope_pos`、
  `d_batch_decode_len`）。
- `LayerWorkspace` 缓冲已是 `[max_batch,*]`（transformer.h:22-31），零新增。

修正 DEV-10 §10 第 3 步的预期：`TLLM_BATCH_DECODE` 开关 + 单
`forwardPagedBatch` 即覆盖第 2/3/5 阶段，工作量集中在行映射与差分测试。

## 12. 范围外（不在本设计与后续实现中承诺）

chunked prefill、prefix cache/sharing、抢占、TP、ragged/batched
attention（§5 逐行保留）、新 C ABI、Rust 侧改动（除必要的对接测试）。
