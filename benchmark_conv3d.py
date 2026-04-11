"""
Triton Conv3d vs PyTorch Conv3d — 正确性测试 + 性能对比
========================================================

这个文件做两件事:
    1. 正确性验证 (仿照 Triton 官方 matmul tutorial 的风格):
       - 使用 FP16 输入 (深度学习标准 dtype)
       - 容差: atol=1e-2, rtol=0
       - 打印输出 tensor 便于人眼检查
    2. 性能 benchmark: 用 triton.testing 测 TFLOPS, 画图对比

精度说明:
    Ampere+ GPU (A100/H100/L40S) 上, tl.dot 和 cuDNN 默认都用 Tensor Core,
    FP16 输入的乘加误差比 FP32 还小 (因为输入本身就只有 FP16 精度)。
    这是深度学习业界的标准做法, 不需要强制 FP32。
"""

import torch
import triton
from conv3d_triton import triton_conv3d


DEVICE = torch.device('cuda')


def is_cuda():
    return torch.cuda.is_available()


# ============================================================================
# 第一部分: 正确性测试 (仿照 matmul tutorial)
# ============================================================================

def correctness_test():
    """
    对比 Triton 和 PyTorch 的 conv3d 输出。
    使用 FP16 + atol=1e-2, rtol=0 (matmul tutorial 风格)。
    """
    print("=" * 70)
    print("Conv3d 正确性测试 (Triton vs PyTorch)")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
    print()

    # --- 一个有代表性的测试 case ---
    # N=2, C_in=16, D=H=W=16, C_out=32, kernel=3, padding=1
    # 用 rand-0.5 而不是 randn (matmul tutorial 风格), 输入范围 [-0.5, 0.5]
    # 这样 FP16 累积误差才能控制在 atol=1e-2 之内
    torch.manual_seed(0)
    x = torch.rand((2, 16, 16, 16, 16), device=DEVICE, dtype=torch.float16) - 0.5
    w = torch.rand((32, 16, 3, 3, 3),    device=DEVICE, dtype=torch.float16) - 0.5
    b = torch.rand((32,),                device=DEVICE, dtype=torch.float16) - 0.5

    triton_output = triton_conv3d(x, w, b, stride=(1,1,1), padding=(1,1,1))
    torch_output  = torch.nn.functional.conv3d(x, w, b, stride=1, padding=1)

    max_diff = (triton_output - torch_output).abs().max().item()
    rel_diff = ((triton_output - torch_output).abs() /
                (torch_output.abs() + 1e-6)).max().item()
    print(f"max abs diff: {max_diff:.4e}")
    print(f"max rel diff: {rel_diff:.4e}")

    if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0):
        print("✅ Triton and Torch match")
        passed = True
    else:
        print("❌ Triton and Torch differ")
        passed = False

    print("=" * 70)
    print()
    return passed


# ============================================================================
# 第二部分: 性能 Benchmark
# ============================================================================
#
# FLOPs 计算: Conv3d 每个输出元素做 2*C_in*kD*kH*kW 次浮点运算(乘加)
# 总 FLOPs = 2 * N * C_out * D_out * H_out * W_out * C_in * kD * kH * kW
#
# 我们固定一些参数, 变化 "特征图大小" 作为 x 轴

# 固定 benchmark 参数
BATCH_SIZE = 2
IN_CHANNELS = 32
OUT_CHANNELS = 64
KERNEL_SIZE = 3
STRIDE = 1
PADDING = 1   # same padding, 输出尺寸 = 输入尺寸


# ---- Benchmark 1: D=H=W 对称, 含大 spatial ----
configs_symmetric = [
    triton.testing.Benchmark(
        x_names=["spatial_size"],
        x_vals=[8, 16, 24, 32, 48, 64, 80, 96, 128],
        line_arg="provider",
        line_vals=["torch", "triton"],
        line_names=["PyTorch (cuDNN)", "Triton"],
        styles=[("green", "-"), ("blue", "-")],
        ylabel="TFLOPS",
        xlabel="Spatial Size (D=H=W)",
        plot_name=f"conv3d-symmetric-N{BATCH_SIZE}-Cin{IN_CHANNELS}-Cout{OUT_CHANNELS}-k{KERNEL_SIZE}",
        args={
            "N": BATCH_SIZE,
            "C_in": IN_CHANNELS,
            "C_out": OUT_CHANNELS,
            "K": KERNEL_SIZE,
        },
    )
]


@triton.testing.perf_report(configs_symmetric)
def benchmark_symmetric(spatial_size, N, C_in, C_out, K, provider):
    """Benchmark D=H=W 对称场景。"""
    D = H = W = spatial_size
    return _run_benchmark(N, C_in, C_out, K, D, H, W, provider)


# ---- Benchmark 2: 真实场景非对称 D/H/W ----
# 模拟视频 (T短, H/W大) 和医学影像 (D/H/W都大) 的典型 shape
REALISTIC_SHAPES = [
    # (label, D, H, W, 描述)
    ("8x56x56",     8,  56,  56),    # 视频模型早期层 (I3D/SlowFast pool 后)
    ("16x112x112", 16, 112, 112),    # 视频模型 C3D 输入
    ("32x64x64",   32,  64,  64),    # 医学影像 downsample 后
    ("8x224x224",   8, 224, 224),    # 视频模型 SlowFast 输入
    ("64x64x64",   64,  64,  64),    # 医学影像中间层
    ("128x64x64", 128,  64,  64),    # 医学影像 (D 长)
    ("32x128x128", 32, 128, 128),    # 医学影像常见
    ("64x128x128", 64, 128, 128),    # 医学影像大体数据
]

configs_realistic = [
    triton.testing.Benchmark(
        x_names=["shape_idx"],
        x_vals=list(range(len(REALISTIC_SHAPES))),
        line_arg="provider",
        line_vals=["torch", "triton"],
        line_names=["PyTorch (cuDNN)", "Triton"],
        styles=[("green", "-"), ("blue", "-")],
        ylabel="TFLOPS",
        xlabel="Shape",
        plot_name=f"conv3d-realistic-N{BATCH_SIZE}-Cin{IN_CHANNELS}-Cout{OUT_CHANNELS}-k{KERNEL_SIZE}",
        args={
            "N": BATCH_SIZE,
            "C_in": IN_CHANNELS,
            "C_out": OUT_CHANNELS,
            "K": KERNEL_SIZE,
        },
    )
]


@triton.testing.perf_report(configs_realistic)
def benchmark_realistic(shape_idx, N, C_in, C_out, K, provider):
    """Benchmark 真实场景非对称 D/H/W。"""
    label, D, H, W = REALISTIC_SHAPES[shape_idx]
    return _run_benchmark(N, C_in, C_out, K, D, H, W, provider)


def _run_benchmark(N, C_in, C_out, K, D, H, W, provider):
    """公共 benchmark 逻辑。"""
    x = torch.randn(N, C_in, D, H, W, device=DEVICE, dtype=torch.float16)
    w = torch.randn(C_out, C_in, K, K, K, device=DEVICE, dtype=torch.float16)
    b = torch.randn(C_out, device=DEVICE, dtype=torch.float16)

    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.nn.functional.conv3d(
                x, w, b, stride=STRIDE, padding=PADDING
            ),
            quantiles=quantiles,
        )
    elif provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: triton_conv3d(
                x, w, b, stride=(STRIDE,)*3, padding=(PADDING,)*3
            ),
            quantiles=quantiles,
        )
    else:
        raise ValueError(f"未知 provider: {provider}")

    D_out = (D + 2*PADDING - K) // STRIDE + 1
    H_out = (H + 2*PADDING - K) // STRIDE + 1
    W_out = (W + 2*PADDING - K) // STRIDE + 1

    flops = 2 * N * C_out * D_out * H_out * W_out * C_in * K * K * K

    perf = lambda ms: flops * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


# ============================================================================
# 入口
# ============================================================================

if __name__ == "__main__":
    # 1. 先跑正确性测试
    passed = correctness_test()

    if not passed:
        print("⚠️  正确性测试有失败, 仍然运行 benchmark 供参考")
        print()

    common_info = (f"固定参数: N={BATCH_SIZE}, C_in={IN_CHANNELS}, "
                   f"C_out={OUT_CHANNELS}, kernel={KERNEL_SIZE}, "
                   f"stride={STRIDE}, padding={PADDING}")

    # 2. Benchmark 1: 对称 D=H=W (含大 spatial)
    print("=" * 70)
    print("Benchmark 1: 对称 D=H=W (Triton vs PyTorch cuDNN)")
    print("=" * 70)
    print(common_info)
    print()

    benchmark_symmetric.run(show_plots=False, print_data=True, save_path=".")

    # 3. Benchmark 2: 真实场景非对称 D/H/W
    print()
    print("=" * 70)
    print("Benchmark 2: 真实场景 (视频 / 医学影像)")
    print("=" * 70)
    print(common_info)
    print("Shapes:")
    for i, (label, D, H, W) in enumerate(REALISTIC_SHAPES):
        print(f"  [{i}] {label:>16s}  (D={D}, H={H}, W={W})")
    print()

    benchmark_realistic.run(show_plots=False, print_data=True, save_path=".")

    print()
    print("📊 结果图已保存到当前目录 (conv3d-*.png)")
    print("💡 查看图片: 在 VSCode 或 scp 下载后打开")
