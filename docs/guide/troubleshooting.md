---
layout: doc
title: "故障排除 — Tiny-LLM"
description: "Tiny-LLM 常见问题和解决方案"
---

# 故障排除

Tiny-LLM 常见问题和解决方案。

---

## 目录

- [编译问题](#编译问题)
- [运行时问题](#运行时问题)
- [性能问题](#性能问题)
- [模型加载问题](#模型加载问题)
- [获取帮助](#获取帮助)

---

## 编译问题

### CUDA 未找到

**错误**: `Could not find CUDA` 或 `nvcc not found`

**解决方案**:
```bash
# 检查 CUDA 安装
nvcc --version

# 显式设置 CUDA 路径
cmake .. -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.2

# 或添加到 PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### CMake 版本过低

**错误**: `CMake 3.18 or higher is required`

**解决方案**:
```bash
# 使用 pip
pip install --upgrade cmake

# 使用 snap (Ubuntu)
sudo snap install cmake --classic

# 从源码编译
curl -L https://cmake.org/files/v3.28/cmake-3.28.0.tar.gz | tar xz
cd cmake-3.28.0 && ./bootstrap && make && sudo make install
```

### 不支持 C++17

**错误**: `error: 'auto' in lambda parameter not supported`

**解决方案**:
```bash
# 检查编译器版本
gcc --version  # 应为 9+
clang --version  # 应为 10+

# 指定编译器
cmake .. -DCMAKE_CXX_COMPILER=g++-11

# 或使用环境变量
CC=gcc-11 CXX=g++-11 cmake ..
```

### CUDA 架构不匹配

**错误**: `No kernel image is available for execution on the device`

**解决方案**:
```bash
# 检查 GPU 计算能力
nvidia-smi --query-gpu=compute_cap --format=csv

# 为特定架构编译
cmake .. -DCUDA_ARCH="80"  # SM 8.0 (A100)
cmake .. -DCUDA_ARCH="86"  # SM 8.6 (RTX 3090)
cmake .. -DCUDA_ARCH="89"  # SM 8.9 (RTX 4090)

# 或使用原生检测
cmake .. -DCUDA_ARCH="native"
```

---

## 运行时问题

### CUDA 显存不足

**错误**: `CUDA out of memory` 或 `cudaErrorMemoryAllocation`

**解决方案**:

1. **减小 batch 大小**
   ```cpp
   cache_config.max_batch_size = 1;  // 从 4 减小
   ```

2. **减小序列长度**
   ```cpp
   config.max_seq_len = 1024;  // 从 2048 减小
   ```

3. **监控显存使用**
   ```cpp
   size_t free, total;
   cudaMemGetInfo(&free, &total);
   std::cout << "剩余: " << free / 1024 / 1024 << " MB" << std::endl;
   ```

### 非法内存访问

**错误**: `an illegal memory access was encountered`

**可能原因**:
- 模型文件格式错误
- 模型维度和配置不匹配
- 内存未初始化

**解决方案**:

1. **启用 Debug 模式**
   ```bash
   cmake .. -DCMAKE_BUILD_TYPE=Debug
   CUDA_LAUNCH_BLOCKING=1 ./tiny_llm_demo
   ```

2. **使用 cuda-memcheck 运行**
   ```bash
   cuda-memcheck ./tiny_llm_demo
   compute-sanitizer ./tiny_llm_demo
   ```

3. **验证模型维度**
   ```cpp
   std::cout << "Config: " << config.hidden_dim 
             << " x " << config.num_layers << std::endl;
   ```

### 生成速度慢

**可能原因**:
- Debug 构建
- 未使用 W8A16 量化
- CUDA 架构设置错误

**解决方案**:

1. **使用 Release 构建**
   ```bash
   cmake .. -DCMAKE_BUILD_TYPE=Release
   ```

2. **验证 GPU 利用率**
   ```bash
   watch -n 1 nvidia-smi
   ```

3. **性能分析**
   ```bash
   nsys profile -o profile ./tiny_llm_demo
   nsys-ui profile.qdrep
   ```

---

## 性能问题

### GPU 利用率低

**症状**: GPU 利用率 < 50%

**解决方案**:

1. **增加 batch 大小**
2. **检查显存带宽瓶颈操作**
3. **使用 Nsight Compute 分析 kernel**

### 显存带宽瓶颈

**症状**: Decode 阶段比预期慢

**原因**: Attention decode 受显存带宽限制

**解决方案**:
- 使用更高带宽的 GPU
- 减小 KV 缓存大小（更小 batch/序列长度）
- 启用 flash attention（如果可用）

---

## 模型加载问题

### 无效模型文件

**错误**: `Failed to load model: invalid format`

**检查清单**:
- [ ] 文件存在且可读
- [ ] 魔数匹配（前 4 字节）
- [ ] 版本受支持
- [ ] 维度与配置匹配

### 维度不匹配

**错误**: `Weight dimension mismatch`

**解决方案**:
```cpp
// 验证配置
std::cout << "vocab_size: " << config.vocab_size << std::endl;
std::cout << "hidden_dim: " << config.hidden_dim << std::endl;
std::cout << "intermediate_dim: " << config.intermediate_dim << std::endl;
```

---

## 获取帮助

### 报告问题时请包含的信息

报告问题时请提供：

1. **系统信息**
   ```bash
   nvidia-smi
   nvcc --version
   cmake --version
   ```

2. **构建输出**
   ```bash
   cmake .. 2>&1 | tee cmake.log
   make VERBOSE=1 2>&1 | tee build.log
   ```

3. **运行时错误**
   ```bash
   CUDA_LAUNCH_BLOCKING=1 ./tiny_llm_demo 2>&1 | tee runtime.log
   ```

### 资源

- [GitHub Issues](https://github.com/holtwood-systems/tiny-llm/issues)
- [文档](https://holtwood-systems.github.io/tiny-llm/)
- [API 参考](/api/inference-engine)
