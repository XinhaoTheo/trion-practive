# Triton Conv3d — Implicit GEMM 实现与优化记录

## 概述

从零用 Triton 实现 3D 卷积 (Conv3d)，核心思路是 **Implicit GEMM**：
- 不物化 im2col 矩阵，通过索引计算隐式展开
- M = C_out, N = batch * D_out * H_out * W_out, K = C_in * kD * kH * kW

对标 PyTorch `torch.nn.functional.conv3d`（底层 cuDNN）。

## Benchmark 配置

- N=2, C_in=32, C_out=64, kernel=3x3x3, stride=1, padding=1
- 变量：spatial_size (D=H=W)，取 8~48
- dtype: FP16
- 硬件：PSC Bridges-2 GPU 节点

## 优化历程

### V0: 基础 Implicit GEMM

最直接的实现：
- 1D launch grid，行主序 `pid_m = pid // num_pid_n` 映射
- K 循环内每次迭代做 4 个整数除法解码 `rk -> (c_in, kd, kh, kw)`
- 5 个 mask AND 在一起（K边界 + N边界 + d/h/w 范围检查）
- weight 用原始 5D stride 寻址（4 个 stride 乘加）
- autotune 搜索 BLOCK_M/N/K

结果：spatial=32 时 ~28 TFLOPS，cuDNN ~83 TFLOPS（**33%**）

### V1: L2 Swizzle

仿照 matmul tutorial 加入 grouped ordering：
- `GROUP_SIZE_M` 参数，把 program 按 group 打包成正方形，提升 L2 命中率
- autotune 搜索 GROUP_SIZE_M in {1, 4, 8}

结果：基本无提升（+0~3%）。原因：K 太小（864），L2 reuse 机会本来就少，瓶颈不在 L2 带宽而在 K 循环内的整数除法和 mask 开销。

### V2: 显式 Zero-Pad + Weight 2D Reshape + K 循环重构 (当前版本)

三个优化同时实施：

**A. 显式 Zero-Pad（消除 d/h/w mask）**
- wrapper 里用 `F.pad` 预先 pad 输入
- kernel 内删掉 `pad_d/h/w` 参数和 3 个空间范围检查
- input_mask 从 5 个 AND 降到 2 个

**B. K 循环重构（消除循环内整数除法）**
- `BLOCK_K = next_pow2(kD*kH*kW)` = 32（对 3x3x3）
- `rk -> (rk_d, rk_h, rk_w)` 解码在循环外只算一次
- 空间地址 `d_in/h_in/w_in` 和 base pointer 也只算一次
- 外层循环按 `c_in` 推进，循环体只剩 2 个 load + 1 个 dot，**0 次整数除法**

**C. Weight 2D Reshape（消除 4 个 weight stride）**
- `weight.view(C_out, -1)`，kernel 内 weight 地址 = `base + c * KDHW`
- 删掉 `weight_stride_ic/kd/kh/kw` 四个参数

结果：

| spatial | cuDNN | V0 | V2 | V0->V2 提升 | vs cuDNN |
|---------|-------|------|------|-------------|----------|
| 8 | 4.1 | 2.3 | 3.4 | 1.5x | 82% |
| 16 | 31.1 | 16.2 | 25.1 | 1.5x | 81% |
| 32 | 83.0 | 28.9 | 77.2 | **2.7x** | **93%** |
| 48 | 141.6 | 27.8 | 80.0 | **2.9x** | 56% |

spatial=32 时达到 cuDNN 的 **93%**。

## 已知问题

### Saw-tooth 性能波动

spatial 非 2 的幂时性能大幅下降（32: 77 → 40: 38 → 48: 80 TFLOPS）。
原因：
1. `total_out` 非 2 的幂时，整数除法开销增大，L2 cache 对齐变差
2. 最后一个 N tile 被大量 mask 掉，做了无效计算

### cuDNN 在大 spatial 上差距拉大

cuDNN 对 k=3 stride=1 使用 Winograd 变换，实际只做 ~45% 的理论 FLOPs，但 benchmark 按理论 FLOPs 计算 TFLOPS，导致 cuDNN 数字被"虚高" ~2.25x。

spatial=32 时 cuDNN 报 83 TFLOPS（实际 ~37 TFLOPS），Triton 77 TFLOPS → Triton 真实吞吐其实是 cuDNN 的 ~2x。

### 寄存器压力限制 tile 放大

当前 kernel 预计算了完整的 `input_base_ptrs` (BLOCK_K x BLOCK_N) 并跨 c_in 循环存活，占用大量寄存器。BLOCK_N > 128 时会 spill 到 local memory。这限制了 arithmetic intensity 的进一步提升。

## 下一步优化方向

1. **降低寄存器压力**：不预计算完整指针数组，循环内即时计算地址（只做标量加法），使 BLOCK_N 可以放大到 256
2. **提高 arithmetic intensity**：放大 tile 到 128x256 / 256x128
3. **用 kernel=5 或 7 做 benchmark**：排除 Winograd 干扰，做更公平的对比
