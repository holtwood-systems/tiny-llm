# 2026-11-06 compute-sanitizer 结果（RTX 4090，sm_89 构建）

ncu 在本机因 `ERR_NVGPUCTRPERM`（宿主 profiling 权限）不可用，但
compute-sanitizer 可用——本目录是仓库首份 sanitizer 证据。

## 环境

- GPU：RTX 4090，驱动 550.107.02；CUDA 12.8 toolkit 构建（`-arch=native`→sm_89）
- 二进制：`build/tiny_llm_tests`（commit 见本提交）
- 过滤：`AttentionTest.*:RMSNormTest.*:RoPETest.*:PagedKvTest.*:PagedDirectTest.*:SplitKvTest.*:ElementwiseTest.*`
  （7 套件 42 个用例，kernel 密集路径）

## memcheck（`sanitizer-memcheck.log`）

**42/42 通过，0 errors**——自研 kernel 在本机 sm_89 构建下无非法
访存（越界读写、未初始化读等）。

## racecheck（`sanitizer-racecheck.log`）

**3 hazards（161 errors / 69 warnings）**，全部位于共享内存
online-softmax 归约路径：

- `attention_decode_kernel`（读 +0x1e10 / 写 +0x2030，16 hazards）
- `attention_decode_splitkv_kernel`（读 +0x20f0 / 写 +0x2330，16 hazards）

并且 `SplitKvTest.SplitMatchesIndependentOracle` 在 racecheck 插桩下
**数值失败**：生产形状 `num_splits=16` 时 `max|diff|=0.083`（容差
0.002）。同一用例在正常执行与 memcheck 下均通过（gpu-gate 252/252、
本目录 memcheck 42/42）。

## 判读（诚实边界）

- 这是**潜在竞态信号**：racecheck 串行化改变了执行时序后 split-KV
  归约出现数值偏差，符合"正常调度侥幸正确、时序扰动暴露"的模式；
- 但也可能是 racecheck 对 warp-synchronous 归约模式的已知误报，
  数值偏差也可能来自插桩下的不同 FP 归约顺序。**两种解释尚未分离**；
- 已人工核对 `block_reduce_max_dyn`/`block_reduce_sum_dyn` 的
  barrier 结构（写→`__syncthreads`→读），未发现明显缺失；根因需
  专门排查（复现命令：`compute-sanitizer --tool racecheck --print-limit 3
  ./build/tiny_llm_tests --gtest_filter='SplitKvTest.SplitMatchesIndependentOracle'`）。

## 根因与修复（同日完成）

SASS 定位确认为**真实竞态**：读 `LDS R0, [0x200]`（red[0]）与写
`STS [R14+0x200]`（red[wid]）之间无 BAR.SYNC——`block_reduce_max_dyn`
末尾读取与下一次 `block_reduce_sum_dyn` 开头写入之间缺屏障。
修复：两个归约函数在 `val = red[0]` 后各加收尾 `__syncthreads()`。

修复后（`racecheck-after-fix.log` / `memcheck-after-fix.log`）：
**racecheck 42/42、0 hazards；memcheck 42/42、0 errors**；
gpu-gate 252/252；decode TPOT 2.2694ms（+0.1%，噪声内）。
细节见 `.agents/notes/implemented/bug-fix/2026-11-06-racecheck-splitkv-hazards.md`。

注：修复过程中的 44 项门禁失败是另一个陷阱——旧 build 缓存里
`CMAKE_CUDA_ARCHITECTURES=52` 在增量 reconfigure 时静默生效（见
`2026-11-05-cuda-arch-default-never-applies.md` 的后续节），与竞态
修复无关。
