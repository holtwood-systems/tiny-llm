# Agent Note: 小 M W8A16 批量转置 kernel（DEV-11 测量与实现）

Status: implemented

## Problem

batch decode 设计（DEV-10，
[2026-11-10-batch-decode-projections-design](../architecture/2026-11-10-batch-decode-projections-design.md)）
要求把 B∈{2,4,8} 行的投影/FFN GEMM 合并为单次 M=B 调用以摊薄权重读取；
但现有 `w8a16_matmul` dispatch 对 M>1 只有 reference（M*N<4096）与
tiled 两条路，从未在真实层形状上与小 M decode 场景对表。DEV-11 要求先
测量、只在瓶颈证实时实现。

## Decision

先建 `kernel_bench --smallm` 对照测量（B 次 M=1 转置快路径 vs 单次 M=B，
随机数据 + 独立 CPU float 参考校验），用 Qwen2.5-0.5B 真实层形状
（q/o 896→896、k/v 896→128、gate/up 896→4864、down 4864→896，gs=128）。

实测（RTX 5070 Ti，结果存
`docs/performance/results/data/2026-11-11-dev11-smallm-w8a16.csv`）：
旧路由在全部 12 组 (M,shape) 上劣于 B×M=1，最差 M=2×gate/up 慢 16.7×——
瓶颈证实，按 DEV-11 第 3 步实现新 kernel。

`w8a16_matmul_smallm_transposed_kernel<MVAL>`（MVAL∈{2,4,8}）：
一个 warp 负责一个输出列，寄存器持 MVAL 个行累加器；`weight_t[col,k]`
每 lane 只读一次摊薄到全部 M 行，input 行为 warp 内广播读。dispatch 在
`M∈{2,4,8} 且转置缓冲存在` 时进入；非模板 M 与无转置缓冲交回既有
reference/tiled 路由，调用签名不变。

## Consequences

- 相对 B×M=1 提速 1.7–6.3×，相对旧路由最高 ~27×（见结果文件与
  `2026-11-11-dev11-smallm-w8a16.md`）。
- 仅为 kernel 级收益：attention/KV scatter 仍逐序列，端到端效果待
  DEV-12（接入）/DEV-13（A/B）。
- 模板化原因：初版 runtime-M 循环 + acc[8] 在 M=2/down 形状慢 2.1×
  （寄存器压力），改为 MVAL 模板实例后全形状转正。
- 旧 scalar/逐序列路径保留为 oracle；新路径无独立开关——它只在
  `weight_t!=nullptr && M∈{2,4,8}` 时激活，上层批量 decode 未接入前
  生产 M=1 调用行为不变。

## Verification

- 新增 4 项测试（`test_w8a16_matmul.cu`）：真实形状 ×B、N/K 尾与组边界、
  零输入、M=3/5/16 回退路由——全部通过，逐行对 CPU float 参考。
- `tiny_llm_tests`：256 passed / 11 env-skipped / 0 failed。
- `compute-sanitizer memcheck`：0 errors；`racecheck`：0 hazards。
- bench verify 列全 ok（max_abs ≤0.062，fp16 累加预期内）。

## 备选方案

- **调整路由阈值、复用 tiled kernel**：tiled 以 block tile 为单位，M=2
  时大半 tile 浪费，实测 M=2×gate/up 0.344ms vs 新 kernel 0.012ms，
  调阈值救不回。
- **沿用 runtime-M 单 kernel（acc[8]）**：少一次实例化，但 M=2 时寄存器
  按 8 分配导致 occupancy 损失，down 形状慢于基线 2.1×；模板实例
  让 acc 尺寸精确，代价是多两个实例化。
- **cuBLAS/外部库**：仓内零外部 BLAS 依赖是既定约束（教学+可控性），
  且 M≤8 的 weight-stationary 访存模式手工 kernel 已足够。
