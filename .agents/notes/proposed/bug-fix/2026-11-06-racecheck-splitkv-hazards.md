# Agent Note: racecheck 在 split-KV 归约路径报 3 hazards 且插桩下数值失败

Status: proposed

## Problem

2026-11-06 RTX 4090（sm_89 构建）compute-sanitizer 首次运行：
memcheck 42/42 干净，但 racecheck 报 3 hazards（161 errors / 69
warnings），全部位于共享内存 online-softmax 归约：
`attention_decode_kernel`（读 +0x1e10 / 写 +0x2030）与
`attention_decode_splitkv_kernel`（读 +0x20f0 / 写 +0x2330）。
同时 `SplitKvTest.SplitMatchesIndependentOracle` 在 racecheck 插桩下
数值失败（num_splits=16 生产形状，max|diff|=0.083 vs 容差 0.002）。
同一用例在正常执行与 memcheck 下通过（252/252 gate、42/42 memcheck）。

## Proposal（待验证，非结论）

先分离两种解释：

1. **真实竞态**：racecheck 串行化改变时序后暴露归约读写竞争。
   已核对 `block_reduce_max_dyn`/`block_reduce_sum_dyn` 的
   write→`__syncthreads`→read 结构，未见明显缺失；下一步应对
   `scores`/`red`/`out_acc` 的每次跨 warp 读写逐一标注访问范围，
   用 `--racecheck-report all` 或 PTX 级定位 hazard 的具体共享变量。
2. **racecheck 误报 + 插桩归约顺序变化**：warp-synchronous 模式
   是 racecheck 已知误报源；插桩下 FP 求和顺序变化也可能超出
   oracle 容差。可用 `__syncwarp()` 显式化或把容差与顺序解耦
   （固定累加顺序的参考实现）来区分。

## Alternatives considered

- 直接把 oracle 容差放宽：掩盖两种解释的差异，否决。
- 直接加 `__syncthreads()` 试错：无定位的盲目修改可能引入死锁
  （barrier 在 warp 分支内），否决。

## Risks

- 若为真实竞态，生产流量下特定时序可能触发错误输出——当前
  gate 通过不能证明安全；
- 若为误报，过度加固会增加 split 路径的同步开销，需要 A/B 验证
  TPOT 影响。

## Acceptance criteria

- 明确 hazard 指向的共享变量与访问线程范围，给出"真实/误报"结论；
- 若真实：修复后 racecheck 0 hazards 且 SplitKv oracle 在插桩下通过，
  同时 gpu-gate 252/252 与 split-KV bitwise 一致性不回归；
- 若误报：在 evidence README 记录判据（如 `--racecheck-report all`
  无 hazard + 顺序固定后数值一致）。

## Verification

`compute-sanitizer --tool racecheck --print-limit 3 ./build/tiny_llm_tests
--gtest_filter='SplitKvTest.SplitMatchesIndependentOracle'` 复现；
证据见 `docs/performance/results/data/2026-11-06-sanitizer/`。
