# Agent Note: racecheck 在 split-KV 归约路径报 3 hazards 且插桩下数值失败

Status: implemented

## Problem

2026-11-06 RTX 4090（sm_89 构建）compute-sanitizer 首次运行：
memcheck 42/42 干净，但 racecheck 报 3 hazards（161 errors / 69
warnings），全部位于共享内存 online-softmax 归约：
`attention_decode_kernel`（读 +0x1e10 / 写 +0x2030）与
`attention_decode_splitkv_kernel`（读 +0x20f0 / 写 +0x2330）。
同时 `SplitKvTest.SplitMatchesIndependentOracle` 在 racecheck 插桩下
数值失败（num_splits=16 生产形状，max|diff|=0.083 vs 容差 0.002）。
同一用例在正常执行与 memcheck 下通过（252/252 gate、42/42 memcheck）。

## Decision

根因已由 SASS 定位，为**真实竞态**（非 racecheck 误报）：

- `/*1e10*/ LDS R0, [0x200]` 读共享内存 0x200 = `red[0]`
  （scores 占 ATTEND_TILE=128 float = 0x200 字节）；
- `/*2030*/ STS [R14+0x200], R3` 写 `red[wid]`（wid=0 即 red[0]）；
- 两指令之间**没有 BAR.SYNC**（屏障在写之后 0x2040）。

即 `block_reduce_max_dyn` 末尾的 `val = red[0]` 读与紧随其后的
`block_reduce_sum_dyn` 开头的 `red[wid] = val` 写之间缺屏障：快 warp
可在慢 warp 读取前覆写 red[0]。修复：两个归约函数在 `val = red[0]`
之后各加一个 `__syncthreads()` 收尾屏障。

原始分析路径（保留备查）：

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

## Consequences

- racecheck 42/42 通过、**0 hazards**；memcheck 42/42、0 errors；
  SplitKv oracle 在插桩下通过。
- gpu-gate **252/252** 全过（此前 44 项失败是另一陷阱：旧 build 缓存
  里 CMAKE_CUDA_ARCHITECTURES=52 静默生效，见同批 CMakeLists 加固）。
- 性能影响：decode TPOT 2.2694ms vs 修复前 2.267ms（+0.1%，噪声内），
  throughput 440.6 tok/s。
- 生产意义：修复前该竞态在特定 warp 时序下可能产生错误 attention
  输出；gate 通过不能证明安全，sanitizer 才是本结论的依据。

## Verification

`compute-sanitizer --tool racecheck --print-limit 3 ./build/tiny_llm_tests
--gtest_filter='SplitKvTest.SplitMatchesIndependentOracle'` 复现；
证据见 `docs/performance/results/data/2026-11-06-sanitizer/`。
