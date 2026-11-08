# Agent Note: GGUF 张量读取先校验声明尺寸不超过文件剩余字节

Status: implemented

## Problem

`GGUFParser::readTensorData` 先按文件声明的 dimensions 计算 `size`，随后直接
`std::vector<uint8_t> data(size)`，再 `file.read(...)`：

```cpp
size_t size = tensor.calculateSize();
if (!tensor.dimensions.empty() && size == 0) { ...err... }
std::vector<uint8_t> data(size);
file.read(reinterpret_cast<char *>(data.data()), size);
```

`calculateSize()` 只防乘法回绕，不检查 `size` 与文件实际剩余字节的关系。一个
几百字节的伪造 GGUF 只要把某个 tensor 的 dimensions 声明成巨大值，就会在读取
之前真实分配数 GB（`std::vector` 值初始化会触碰页面），导致 OOM-kill 或长时间
卡死，而不是返回解析错误。`tinyllm_load` 的 `catch (std::exception)` 只能接住
`bad_alloc`，接不住 Linux OOM killer。

同文件的 `readArray()` 已经用了 `remainingBytes(file)` 做同型检查
（`src/gguf_parser.cpp:348`），`readTensorData` 是缺口。

## Decision

在分配之前比较 `size` 与 `remainingBytes(file)`：

```cpp
const size_t remaining = remainingBytes(file);
if (size > remaining) {
    return err("Tensor data exceeds remaining file bytes for: " + tensor.name);
}
std::vector<uint8_t> data(size);
```

新增 `GGUFParser.RejectsTensorDataExceedingFileRemainder`：构造 `n_dims=1`、
`dims=[1024]`（F32，4096 字节）、`offset=0`、文件在 tensor info 后结束的最小
GGUF；`parse()` 应成功，`readTensorData()` 必须返回包含 `remaining` 的错误，
而不是先分配再读失败。

## Alternatives considered

- **只依赖 `file.read` 失败。** 旧路径已经是这样：读失败会返回错误，但分配
  已经发生，DoS 面仍在；否决。
- **给 tensor 尺寸设全局上限（如 1 GiB）。** 对合法大模型权重不友好，且需要
  额外配置；按文件实际剩余字节校验是更直接的不变量；否决。
- **在 `parse()` 阶段校验所有 tensor 的 offset+size。** 更早失败，但需要
  `parse()` 做完整布局审计，且 `readTensorData` 仍应自防御；本轮先修读取
  路径，parse 阶段的整体布局审计留作后续。

## Consequences

Benefits:

- 伪造 GGUF 无法再用声明尺寸触发超大分配；读取路径在分配前返回可诊断错误。
- 与 `readArray()` 的防御口径一致。

Costs and limits:

- 本机无 nvcc/GPU，未运行 `tiny_llm_tests`；新增 GTest 待有 CUDA 的环境执行。
- 只覆盖单个 tensor 的声明尺寸与文件剩余字节；不检查 tensor 之间是否重叠、
  offset 是否对齐，也不替代完整的 GGUF 布局校验。
- `remainingBytes()` 通过 `seekg` 两次求剩余长度；对超大文件没有额外 I/O，
  但会改变流位置后复位，当前调用点随后立即 `read`，不依赖原位置。

## Verification

- 用最小 logger shim 编译真实 `src/gguf_parser.cpp` 的 host-only harness：
  构造上述 58 字节 GGUF，`parse()` 输出 `tensors=1 data_offset=64`，
  `readTensorData()` 返回 `Tensor data exceeds remaining file bytes for: t0`。
  命令与输出见本次会话；修复前该路径会先分配 4096 字节（声明更大时可达 GB 级）。
- 新增 GTest 用例的断言与 harness 相同；待有 CUDA 环境执行。
- `git diff --check`：通过。
