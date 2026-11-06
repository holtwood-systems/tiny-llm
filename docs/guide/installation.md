# 安装

本指南介绍如何在您的系统上安装 Tiny-LLM。

## 系统要求

### 硬件要求

- **GPU**: NVIDIA GPU，计算能力 7.0+（Volta 或更新）
- **内存**: 推理至少需要 8GB GPU 内存
- **存储**: 2GB 用于构建产物

### 软件要求

- **操作系统**: Linux（推荐 Ubuntu 20.04+）或 Windows 10+
- **CUDA 工具包**: 11.0 或更高版本
- **CMake**: 3.18 或更高版本
- **C++ 编译器**: C++17 兼容（GCC 9+、Clang 10+、MSVC 2019+）

## 安装步骤

### 1. 安装 CUDA 工具包

确保 CUDA 工具包 11.0+ 已安装，且 `nvcc` 在 PATH 中：

```bash
nvcc --version
```

### 2. 克隆仓库

```bash
git clone https://github.com/holtwood-systems/tiny-llm.git
cd tiny-llm
```

### 3. 配置构建

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
```

### 4. 构建

```bash
cmake --build build -j$(nproc)
```

### 5. 运行测试（可选）

```bash
ctest --test-dir build --output-on-failure
```

## 构建选项

| 选项 | 默认值 | 描述 |
|------|--------|------|
| `CMAKE_BUILD_TYPE` | `Release` | 构建类型（Release/Debug） |
| `BUILD_TESTS` | `OFF` | 构建测试套件 |
| `BUILD_EXAMPLES` | `OFF` | 构建示例程序 |
| `CUDA_ARCH` | 自动 | CUDA 架构（例如 `80;86`） |

## 故障排除

常见问题请参阅[故障排除指南](/guide/troubleshooting)。

## 下一步

- [快速入门教程](/guide/quickstart) - 构建您的第一个推理
- [架构概览](/architecture/inference-engine) - 了解内部原理
