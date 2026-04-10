"""
Triton Conv3d 实现 — 从零开始学习 Triton 编写 3D 卷积
=======================================================

学习路径: Vector Add → MatMul → im2col + MatMul = Conv3d

核心思想:
    Conv3d 可以通过 im2col 变换转化为矩阵乘法问题:
    1. 将输入展开为 im2col 矩阵 (implicit GEMM, 不真正物化)
    2. 对展开的输入和权重做矩阵乘法
    3. 得到输出

PyTorch Conv3d 公式:
    out(N_i, C_out_j) = bias(C_out_j) +
        sum_{k=0}^{C_in-1} weight(C_out_j, k) ★ input(N_i, k)

    其中 ★ 是 3D 互相关操作 (cross-correlation)

输入形状: (N, C_in, D_in, H_in, W_in)
权重形状: (C_out, C_in, kD, kH, kW)
输出形状: (N, C_out, D_out, H_out, W_out)

其中:
    D_out = (D_in + 2*pad_d - kD) // stride_d + 1
    H_out = (H_in + 2*pad_h - kH) // stride_h + 1
    W_out = (W_in + 2*pad_w - kW) // stride_w + 1
"""

import torch
import triton
import triton.language as tl

# ============================================================================
# 第一步: 理解 Triton 基础 (来自 Vector Add 教程)
# ============================================================================
# Triton 的核心概念:
# 1. @triton.jit — 标记一个函数为 Triton kernel
# 2. tl.program_id(axis) — 获取当前 program 的 ID (类似 CUDA 的 blockIdx)
# 3. tl.arange(0, N) — 生成 [0, 1, ..., N-1] 的向量
# 4. tl.load / tl.store — 从 DRAM 加载/存储数据
# 5. tl.dot — 块级矩阵乘法 (Triton 的核心优势)
# 6. tl.constexpr — 编译时常量 (block size 等)

# ============================================================================
# 第二步: 理解 Conv3d → GEMM 的转换
# ============================================================================
#
# === Implicit GEMM 方法 ===
#
# 我们把 Conv3d 看成一个大矩阵乘法:
#
# 矩阵 A (权重): shape = (C_out, C_in * kD * kH * kW)
#   - 每一行是一个输出通道的所有权重展平
#
# 矩阵 B (im2col 展开的输入): shape = (C_in * kD * kH * kW, N * D_out * H_out * W_out)
#   - 每一列对应一个输出位置, 包含该位置卷积窗口内所有输入值
#
# 矩阵 C (输出): shape = (C_out, N * D_out * H_out * W_out)
#   - C = A @ B
#   - 然后 reshape 为 (N, C_out, D_out, H_out, W_out)
#
# 关键优化: 我们不真正创建 im2col 矩阵 (太大了!),
# 而是在 kernel 内部通过索引计算 "隐式" 地访问对应位置。


# ============================================================================
# 第三步: Triton Conv3d Kernel
# ============================================================================
#
# Autotune configs: Triton 会在这些配置中搜索最优的一个
# - BLOCK_M/N/K: tile 大小, 越大 Tensor Core 利用率越高, 但占用更多 shared memory
# - num_warps: 每个 program 用多少 warp (32 线程一组), 一般 4 或 8
# - num_stages: 软件流水线深度, 越大延迟隐藏越好但占更多寄存器
#
# key=["out_channels", "in_channels", "out_depth", "out_height", "out_width",
#      "kernel_d", "kernel_h", "kernel_w"]:
#   这些参数变化时, autotune 会重新搜索最优配置 (并缓存结果)

def get_autotune_configs():
    """BLOCK_K 不再 autotune, 由 wrapper 根据 kernel size 计算。"""
    configs = []
    for BM in [32, 64, 128]:
        for BN in [32, 64, 128]:
            for GM in [1, 4, 8]:
                for num_warps in [4, 8]:
                    for num_stages in [2, 3, 4]:
                        if BM * BN > 128 * 128:
                            continue
                        configs.append(
                            triton.Config(
                                {"BLOCK_M": BM, "BLOCK_N": BN,
                                 "GROUP_SIZE_M": GM},
                                num_warps=num_warps,
                                num_stages=num_stages,
                            )
                        )
    return configs


@triton.autotune(
    configs=get_autotune_configs(),
    key=[
        "out_channels", "in_channels",
        "out_depth", "out_height", "out_width",
        "kernel_d", "kernel_h", "kernel_w",
    ],
)
@triton.jit
def conv3d_kernel(
    # === 指针 ===
    input_ptr,      # 已预先 zero-pad 的输入, shape: (N, C_in, D_pad, H_pad, W_pad)
    weight_ptr,     # 2D 展平的权重, shape: (C_out, C_in * kD * kH * kW)
    bias_ptr,       # 偏置指针, shape: (C_out,), 可以为 None
    output_ptr,     # 输出张量指针, shape: (N, C_out, D_out, H_out, W_out)

    # === 维度 ===
    batch_size,     # N
    in_channels,    # C_in
    out_channels,   # C_out
    out_depth,      # D_out
    out_height,     # H_out
    out_width,      # W_out

    # === 卷积核尺寸 ===
    kernel_d,       # kD
    kernel_h,       # kH
    kernel_w,       # kW

    # === conv stride ===
    stride_d,
    stride_h,
    stride_w,

    # === 输入张量的 stride (已 pad 后的内存步长) ===
    input_stride_n,
    input_stride_c,
    input_stride_d,
    input_stride_h,
    input_stride_w,

    # === 权重张量的 stride (2D: 只需 oc 方向) ===
    weight_stride_oc,

    # === 输出张量的 stride ===
    output_stride_n,
    output_stride_c,
    output_stride_d,
    output_stride_h,
    output_stride_w,

    # === 常量 ===
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,      # 输出通道方向的 block
    BLOCK_N: tl.constexpr,      # 输出空间位置方向的 block
    BLOCK_K: tl.constexpr,      # = next_pow2(kD * kH * kW), 由 wrapper 设定
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Conv3d 的 Triton kernel, 使用 implicit GEMM 方法。

    GEMM 视角:
        M = C_out                           (输出通道数)
        N = batch_size * D_out * H_out * W_out   (所有输出空间位置)
        K = C_in * kD * kH * kW             (reduction 维度)

    每个 Triton program 计算输出矩阵 C 的一个 (BLOCK_M, BLOCK_N) 的块。
    """

    # ----- 1. 确定当前 program 负责的输出块 -----
    # 使用 1D launch grid, 需要手动映射到 2D (M, N)
    pid = tl.program_id(axis=0)

    # 总共有多少个 program 沿 M 和 N 方向
    num_pid_m = tl.cdiv(out_channels, BLOCK_M)
    num_pid_n = tl.cdiv(batch_size * out_depth * out_height * out_width, BLOCK_N)

    # 映射: pid -> (pid_m, pid_n), 使用 grouped ordering 优化 L2 cache
    # 把 program 按 GROUP_SIZE_M 行打包成一个 "super-group", 在 group 内
    # 先沿 M 方向走, 走完再换列, 这样同时活跃的 program 落在一个正方形里,
    # 共享更多的 input/weight tile, 显著提升 L2 命中率。
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    # 处理最后一个 group 不满 GROUP_SIZE_M 行的情况
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----- 2. 计算当前 block 的输出通道索引和空间位置索引 -----
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # (BLOCK_M,)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # (BLOCK_N,)

    out_spatial = out_depth * out_height * out_width
    total_out = batch_size * out_spatial

    # 从 rn 解码出 (n, d_out, h_out, w_out)
    n_idx = rn // out_spatial
    spatial_idx = rn % out_spatial
    d_out_idx = spatial_idx // (out_height * out_width)
    hw_idx = spatial_idx % (out_height * out_width)
    h_out_idx = hw_idx // out_width
    w_out_idx = hw_idx % out_width

    # 输入空间起点 (输入已被预先 zero-pad, 所以不再减 pad)
    d_base = d_out_idx * stride_d   # (BLOCK_N,)
    h_base = h_out_idx * stride_h
    w_base = w_out_idx * stride_w

    # ----- 3. 重构的 K 循环 -----
    # BLOCK_K = next_pow2(kD*kH*kW), 外层循环走 c_in
    # rk 解码只在循环外算一次, 彻底消除循环内的整数除法
    KDHW = kernel_d * kernel_h * kernel_w
    KHW = kernel_h * kernel_w
    rk = tl.arange(0, BLOCK_K)              # (BLOCK_K,)
    rk_valid = rk < KDHW                     # 尾部 padding mask
    rk_d = rk // KHW
    rk_h = (rk % KHW) // kernel_w
    rk_w = rk % kernel_w

    # 输入空间地址, 只在循环外算一次 (BLOCK_K, BLOCK_N)
    d_in = d_base[None, :] + rk_d[:, None]
    h_in = h_base[None, :] + rk_h[:, None]
    w_in = w_base[None, :] + rk_w[:, None]

    # boundary mask
    mask_m = rm < out_channels              # (BLOCK_M,)
    mask_n = rn < total_out                 # (BLOCK_N,)

    # base pointers (不含 c_in 偏移, 循环内只加一个标量)
    input_base_ptrs = (input_ptr
                      + n_idx[None, :] * input_stride_n
                      + d_in * input_stride_d
                      + h_in * input_stride_h
                      + w_in * input_stride_w)           # (BLOCK_K, BLOCK_N)
    weight_base_ptrs = (weight_ptr
                       + rm[:, None] * weight_stride_oc
                       + rk[None, :])                     # (BLOCK_M, BLOCK_K)

    input_mask = rk_valid[:, None] & mask_n[None, :]      # (BLOCK_K, BLOCK_N)
    weight_mask = mask_m[:, None] & rk_valid[None, :]      # (BLOCK_M, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # 主循环: 每次推进一个 c_in, 循环体内 0 次整数除法
    for c in range(0, in_channels):
        w = tl.load(weight_base_ptrs + c * KDHW,
                    mask=weight_mask, other=0.0)           # (BLOCK_M, BLOCK_K)
        x = tl.load(input_base_ptrs + c * input_stride_c,
                    mask=input_mask, other=0.0)            # (BLOCK_K, BLOCK_N)
        acc = tl.dot(w, x, acc)

    # ----- 4. bias -----
    if HAS_BIAS:
        bias = tl.load(bias_ptr + rm, mask=mask_m, other=0.0)
        acc += bias[:, None]

    # ----- 5. store -----
    c_out = acc.to(output_ptr.dtype.element_ty)

    output_ptrs = (output_ptr
                  + n_idx[None, :] * output_stride_n
                  + rm[:, None] * output_stride_c
                  + d_out_idx[None, :] * output_stride_d
                  + h_out_idx[None, :] * output_stride_h
                  + w_out_idx[None, :] * output_stride_w)
    output_mask = mask_m[:, None] & mask_n[None, :]

    tl.store(output_ptrs, c_out, mask=output_mask)


# ============================================================================
# 第四步: Python Wrapper 函数
# ============================================================================

def triton_conv3d(
    input: torch.Tensor,    # (N, C_in, D_in, H_in, W_in)
    weight: torch.Tensor,   # (C_out, C_in, kD, kH, kW)
    bias: torch.Tensor = None,
    stride: tuple = (1, 1, 1),
    padding: tuple = (0, 0, 0),
) -> torch.Tensor:
    """
    使用 Triton 实现的 Conv3d, 对标 torch.nn.Conv3d。

    参数:
        input:   (N, C_in, D_in, H_in, W_in) 输入张量
        weight:  (C_out, C_in, kD, kH, kW) 卷积核
        bias:    (C_out,) 偏置, 可选
        stride:  (stride_d, stride_h, stride_w)
        padding: (pad_d, pad_h, pad_w)

    返回:
        output:  (N, C_out, D_out, H_out, W_out)
    """
    # --- 检查输入 ---
    assert input.is_contiguous(), "输入必须是 contiguous 的"
    assert weight.is_contiguous(), "权重必须是 contiguous 的"
    assert input.ndim == 5, f"输入必须是 5D 张量, 但得到 {input.ndim}D"
    assert weight.ndim == 5, f"权重必须是 5D 张量, 但得到 {weight.ndim}D"

    N, C_in, D_in, H_in, W_in = input.shape
    C_out, C_in_w, kD, kH, kW = weight.shape
    assert C_in == C_in_w, f"输入通道 {C_in} != 权重输入通道 {C_in_w}"

    stride_d, stride_h, stride_w = stride
    pad_d, pad_h, pad_w = padding

    # --- 计算输出尺寸 ---
    D_out = (D_in + 2 * pad_d - kD) // stride_d + 1
    H_out = (H_in + 2 * pad_h - kH) // stride_h + 1
    W_out = (W_in + 2 * pad_w - kW) // stride_w + 1

    assert D_out > 0 and H_out > 0 and W_out > 0, \
        f"输出尺寸无效: ({D_out}, {H_out}, {W_out})"

    # === 优化 A: 显式 zero-pad 输入, kernel 内不再需要 d/h/w 的 mask ===
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        input = torch.nn.functional.pad(
            input, (pad_w, pad_w, pad_h, pad_h, pad_d, pad_d)
        )
    # input 现在是 (N, C_in, D_in+2*pad_d, H_in+2*pad_h, W_in+2*pad_w)

    # === 优化 C: weight reshape 成 2D, 消除 kernel 内的 4 个 stride 计算 ===
    weight_2d = weight.view(C_out, -1)   # (C_out, C_in*kD*kH*kW), 共享内存

    # --- 分配输出 ---
    output = torch.empty(
        (N, C_out, D_out, H_out, W_out),
        device=input.device, dtype=input.dtype
    )

    M = C_out
    total_N = N * D_out * H_out * W_out

    # === 优化 B: BLOCK_K 由 kernel size 决定, 不再 autotune ===
    KDHW = kD * kH * kW
    BLOCK_K = max(triton.next_power_of_2(KDHW), 16)

    # --- 准备 bias ---
    HAS_BIAS = bias is not None
    if not HAS_BIAS:
        bias = torch.empty(0, device=input.device, dtype=input.dtype)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(total_N, META["BLOCK_N"]),
    )

    conv3d_kernel[grid](
        input, weight_2d, bias, output,
        # 维度
        N, C_in,
        C_out, D_out, H_out, W_out,
        kD, kH, kW,
        stride_d, stride_h, stride_w,
        # 输入 stride (来自 padded 后的 input)
        input.stride(0), input.stride(1), input.stride(2),
        input.stride(3), input.stride(4),
        # 权重 stride (2D, 只需 oc 方向)
        weight_2d.stride(0),
        # 输出 stride
        output.stride(0), output.stride(1), output.stride(2),
        output.stride(3), output.stride(4),
        # 常量
        HAS_BIAS=HAS_BIAS,
        BLOCK_K=BLOCK_K,
    )

    return output


# ============================================================================
# 第五步: 测试 — 对比 PyTorch 结果
# ============================================================================

def test_conv3d():
    """测试 Triton Conv3d 是否和 PyTorch Conv3d 结果一致。"""

    torch.manual_seed(42)
    device = 'cuda'
    dtype = torch.float32

    # 打印 GPU 信息
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Capability: {torch.cuda.get_device_capability(0)}")
    print()

    # --- 测试用例 ---
    test_cases = [
        # (N, C_in, D, H, W, C_out, kD, kH, kW, stride, padding, use_bias, 描述)
        (1, 1, 4, 4, 4, 1, 3, 3, 3, (1,1,1), (0,0,0), False,
         "最简单: 单 batch, 单通道, 无 padding"),

        (1, 1, 4, 4, 4, 1, 3, 3, 3, (1,1,1), (1,1,1), False,
         "加 padding"),

        (1, 3, 8, 8, 8, 16, 3, 3, 3, (1,1,1), (1,1,1), True,
         "多通道 + bias"),

        (2, 3, 8, 8, 8, 16, 3, 3, 3, (2,2,2), (1,1,1), True,
         "stride=2 + bias"),

        (2, 16, 8, 8, 8, 32, 3, 3, 3, (1,1,1), (1,1,1), True,
         "更大的通道数"),
    ]

    print("=" * 70)
    print("Triton Conv3d 正确性测试")
    print("=" * 70)

    all_passed = True
    for (N, C_in, D, H, W, C_out, kD, kH, kW,
         stride, padding, use_bias, desc) in test_cases:

        # 创建输入
        x = torch.randn(N, C_in, D, H, W, device=device, dtype=dtype)
        w = torch.randn(C_out, C_in, kD, kH, kW, device=device, dtype=dtype)
        b = torch.randn(C_out, device=device, dtype=dtype) if use_bias else None

        # PyTorch 参考结果
        torch_out = torch.nn.functional.conv3d(x, w, b, stride=stride, padding=padding)

        # Triton 结果
        triton_out = triton_conv3d(x, w, b, stride=stride, padding=padding)

        # 比较 (TF32 精度容差: ~1e-2, 符合 Tensor Core 实际表现)
        max_diff = (torch_out - triton_out).abs().max().item()
        passed = torch.allclose(torch_out, triton_out, atol=1e-2, rtol=1e-2)

        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} | {desc}")
        print(f"       shape: ({N},{C_in},{D},{H},{W}) -> ({N},{C_out},"
              f"{torch_out.shape[2]},{torch_out.shape[3]},{torch_out.shape[4]})")
        print(f"       max diff: {max_diff:.6e}")

        if not passed:
            all_passed = False

    print("=" * 70)
    if all_passed:
        print("🎉 所有测试通过!")
    else:
        print("⚠️  部分测试失败, 请检查实现")
    print("=" * 70)


# ============================================================================
# 第六步: 学习笔记总结
# ============================================================================

LEARNING_NOTES = """
╔══════════════════════════════════════════════════════════════════════╗
║                    Triton Conv3d 学习路线图                          ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  1. 先学 Vector Add (01-vector-add)                                  ║
║     → 理解 program_id, arange, load/store, mask                      ║
║                                                                      ║
║  2. 再学 MatMul (03-matrix-multiplication)                           ║
║     → 理解 tl.dot, 多维指针算术, L2 cache 优化                       ║
║     → 理解 blocked accumulation (K方向循环)                          ║
║                                                                      ║
║  3. Conv3d = Implicit GEMM                                           ║
║     → 不物化 im2col 矩阵, 通过索引计算隐式展开                       ║
║     → M = C_out, N = batch*D_out*H_out*W_out, K = C_in*kD*kH*kW     ║
║     → kernel 内部: 从线性索引 → (n, d, h, w) → 计算输入地址           ║
║                                                                      ║
║  ──── 关键 Triton API ────                                           ║
║  tl.program_id(0)      : 获取 block ID                               ║
║  tl.arange(0, N)       : 创建 [0..N-1] 向量                          ║
║  tl.load(ptr, mask)    : 带 mask 的内存加载                           ║
║  tl.store(ptr, val)    : 写回内存                                     ║
║  tl.dot(a, b, acc)     : 块矩阵乘法, 累加到 acc                      ║
║  tl.zeros(shape, dtype): 初始化全零张量                               ║
║  tl.cdiv(a, b)         : 向上取整除法                                 ║
║                                                                      ║
║  ──── Conv3d 中的难点 ────                                           ║
║  1. 从 rk (reduction索引) 解码出 (c_in, kd, kh, kw)                  ║
║  2. 从 rn (空间索引) 解码出 (n, d_out, h_out, w_out)                  ║
║  3. 计算对应的输入位置: d_in = d_out*stride - pad + kd               ║
║  4. Padding mask: 判断输入位置是否在有效范围内                        ║
║                                                                      ║
║  ──── 优化方向 ────                                                  ║
║  • @triton.autotune 搜索最优 BLOCK_M/N/K                             ║
║  • L2 cache 优化 (grouped ordering)                                  ║
║  • 支持 dilation, groups                                             ║
║  • FP16 / BF16 精度                                                  ║
║  • 反向传播 kernel (backward pass)                                   ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""


if __name__ == "__main__":
    print(LEARNING_NOTES)
    test_conv3d()
