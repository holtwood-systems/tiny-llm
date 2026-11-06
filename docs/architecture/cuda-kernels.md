# CUDA Kernels

Hand-tuned CUDA kernels for inference optimization.

## Overview

Tiny-LLM implements custom CUDA kernels optimized for Transformer inference workloads:

```mermaid
mindmap
  root((CUDA Kernels))
    Matrix Operations
      w8a16_matmul
      Tensor Core INT8
    Attention
      attention_decode
      attention_prefill
      Online softmax
    Normalization
      rmsnorm
      Warp reduction
    Position Encoding
      rope
      Cached trigonometry
    Elementwise
      silu
      gelu
      Vectorized ops
```

---

## W8A16 Matrix Multiplication

### Kernel Signature

```cpp
void w8a16_matmul(
    const half* input,      // [M, K] FP16
    const int8_t* weight,   // [K, N] INT8
    const half* scales,     // [K/128, N] FP16
    half* output,           // [M, N] FP16
    int M, int N, int K,
    int group_size = 128,
    cudaStream_t stream = 0
);
```

### Optimization Pipeline

```mermaid
flowchart LR
    subgraph Input["Input Processing"]
        A["Vectorized Load<br/>128-bit per access"]
    end

    subgraph Compute["Computation"]
        B["Shared Memory Tiling"]
        C["INT8 Tensor Core"]
        D["Warp Shuffle Reduction"]
    end

    subgraph Output["Output"]
        E["Dequantize with Scale"]
        F["Coalesced Write"]
    end

    Input --> Compute --> Output

    style Input fill:#76B900,stroke:#5a8f00,color:#fff
    style Compute fill:#8B5CF6,stroke:#7c3aed,color:#fff
    style Output fill:#00D4AA,stroke:#00C49A,color:#fff
```

### Tiling Strategy

```mermaid
flowchart TB
    subgraph Grid["Grid Configuration"]
        G1["Block: (128, 128)"]
        G2["Warp: (32, 4)"]
        G3["Thread: 8 elements"]
    end

    subgraph SharedMem["Shared Memory"]
        S1["Input Tile: 128×128 FP16"]
        S2["Weight Tile: 128×128 INT8"]
        S3["Scale Cache: 128/128 = 1"]
    end

    Grid --> SharedMem

    style Grid fill:#00D4AA,stroke:#00C49A,color:#fff
    style SharedMem fill:#F59E0B,stroke:#d97706,color:#fff
```

### Code Example

```cpp
template<int TILE_M, int TILE_N, int TILE_K>
__global__ void w8a16_matmul_kernel(
    const half* __restrict__ input,
    const int8_t* __restrict__ weight,
    const half* __restrict__ scales,
    half* __restrict__ output,
    int M, int N, int K, int group_size
) {
    __shared__ half input_tile[TILE_M][TILE_K];
    __shared__ int8_t weight_tile[TILE_K][TILE_N];
    
    int row = blockIdx.y * TILE_M + threadIdx.y;
    int col = blockIdx.x * TILE_N + threadIdx.x;
    
    half accumulator = 0.0f;
    
    for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {
        // Collaborative load
        load_input_tile(input_tile, input, row, k_tile);
        load_weight_tile(weight_tile, weight, k_tile, col);
        __syncthreads();
        
        // Compute with scale
        half scale = scales[(k_tile / group_size) * N + col];
        for (int k = 0; k < TILE_K; ++k) {
            accumulator += input_tile[threadIdx.y][k] * 
                           (half(weight_tile[k][threadIdx.x]) * scale);
        }
        __syncthreads();
    }
    
    if (row < M && col < N) {
        output[row * N + col] = accumulator;
    }
}
```

---

## Attention Kernels

### Decode Attention

Single token attention against cached KV.

```mermaid
flowchart TB
    subgraph Input["Input"]
        Q["Query: [1, heads, 1, dim]"]
        K["K Cache: [1, heads, seq, dim]"]
        V["V Cache: [1, heads, seq, dim]"]
    end

    subgraph Compute["Computation"]
        S["Score = Q @ K^T<br/>[1, heads, 1, seq]"]
        SM["Softmax"]
        O["Output = Score @ V<br/>[1, heads, 1, dim]"]
    end

    Q --> S
    K --> S
    S --> SM --> O
    V --> O

    style Input fill:#76B900,stroke:#5a8f00,color:#fff
    style Compute fill:#8B5CF6,stroke:#7c3aed,color:#fff
```

### Prefill Attention

Multi-token attention with causal masking.

```mermaid
flowchart TB
    subgraph Input["Input"]
        Q2["Query: [batch, heads, seq, dim]"]
        K2["Key: [batch, heads, seq, dim]"]
        V2["Value: [batch, heads, seq, dim]"]
    end

    subgraph Compute["Computation"]
        S2["Score = Q @ K^T / sqrt(d)"]
        M["Apply Causal Mask"]
        SM2["Online Softmax"]
        O2["Output = Score @ V"]
    end

    Q2 --> S2
    K2 --> S2
    S2 --> M --> SM2 --> O2
    V2 --> O2

    style Input fill:#76B900,stroke:#5a8f00,color:#fff
    style Compute fill:#8B5CF6,stroke:#7c3aed,color:#fff
```

### Online Softmax

Numerical stability for large sequences:

```cpp
// Online softmax: compute max and sum in single pass
__device__ void online_softmax(
    float* scores,      // [seq_len]
    int seq_len
) {
    float max_val = -INFINITY;
    float sum = 0.0f;
    
    // Single pass: max and exp sum
    for (int i = 0; i < seq_len; ++i) {
        float old_max = max_val;
        max_val = fmaxf(max_val, scores[i]);
        sum = sum * expf(old_max - max_val) + expf(scores[i] - max_val);
    }
    
    // Normalize
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < seq_len; ++i) {
        scores[i] = expf(scores[i] - max_val) * inv_sum;
    }
}
```

---

## Normalization Kernels

### RMSNorm

```mermaid
flowchart LR
    subgraph Input["Input"]
        X["X: [batch, seq, dim]"]
        W["Weight: [dim]"]
    end

    subgraph Compute["Computation"]
        S["Sum of squares"]
        R["RSQRT(sum / dim)"]
        N["X * rsqrt * W"]
    end

    X --> S --> R --> N
    W --> N

    style Input fill:#76B900,stroke:#5a8f00,color:#fff
    style Compute fill:#00D4AA,stroke:#00C49A,color:#fff
```

### Warp Reduction

```cpp
__device__ float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void rmsnorm_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    half* __restrict__ output,
    int hidden_dim,
    float eps
) {
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    
    float sum = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float x = __half2float(input[idx * hidden_dim + i]);
        sum += x * x;
    }
    
    sum = warp_reduce_sum(sum);
    // ... reduce across warps ...
    
    float rms_rcp = rsqrtf(sum / hidden_dim + eps);
    
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float x = __half2float(input[idx * hidden_dim + i]);
        float w = __half2float(weight[i]);
        output[idx * hidden_dim + i] = __float2half(x * rms_rcp * w);
    }
}
```

---

## RoPE (Rotary Position Embedding)

### Implementation

```mermaid
flowchart TB
    subgraph Compute["Position Encoding"]
        F["freq = 1 / theta^(2i/dim)"]
        A["angle = pos × freq"]
        C["cos(angle), sin(angle)"]
    end

    subgraph Apply["Apply Rotation"]
        R["rotated = rotate_half(x)"]
        O["out = x × cos + rotated × sin"]
    end

    Compute --> Apply

    style Compute fill:#76B900,stroke:#5a8f00,color:#fff
    style Apply fill:#00D4AA,stroke:#00C49A,color:#fff
```

### Cached Trigonometry

```cpp
// Precompute cos/sin tables
void rope_precompute(
    half* cos_table,     // [max_seq, head_dim/2]
    half* sin_table,     // [max_seq, head_dim/2]
    int max_seq_len,
    int head_dim,
    float theta = 10000.0f
);

// Apply RoPE using cached tables
__global__ void rope_kernel(
    half* query,         // [batch, heads, seq, dim]
    const half* cos,
    const half* sin,
    int position
);
```

---

## 性能基准

> 本页不引用实测性能数字。kernel 的数值正确性由 `tests/test_kernels.cu` 与
> `tests/test_w8a16_matmul.cu` 的差分测试保证；吞吐与带宽数据需要在真实硬件上
> 用基准脚本产出，计划见仓库根目录 [ROADMAP](https://github.com/holtwood-systems/tiny-llm/blob/master/ROADMAP.md)。

---

## Optimization Guidelines

### Memory Access Patterns

```mermaid
flowchart TB
    subgraph Good["Good Patterns"]
        G1["Coalesced global access"]
        G2["Shared memory bank conflict free"]
        G3["Vectorized loads (128-bit)"]
    end

    subgraph Bad["Bad Patterns"]
        B1["Strided access"]
        B2["Bank conflicts"]
        B3["Scalar loads"]
    end

    style Good fill:#00D4AA,stroke:#00C49A,color:#fff
    style Bad fill:#ef4444,stroke:#dc2626,color:#fff
```

### Occupancy Considerations

| Factor | Recommendation |
|--------|----------------|
| Threads per Block | 128-512 (multiples of 32) |
| Shared Memory | < 48KB per block |
| Registers | < 64 per thread |
| Active Blocks | ≥ 4 per SM |

---

## References

- [FlashAttention](https://arxiv.org/abs/2205.14135) - Dao et al., NeurIPS 2022
- [Cutlass Library](https://github.com/NVIDIA/cutlass) - NVIDIA
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
