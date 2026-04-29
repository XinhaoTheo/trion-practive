"""
Triton Conv3d Implementation
=======================================================

Learning path: Vector Add -> MatMul -> im2col + MatMul = Conv3d

Core idea:
    Conv3d can be converted to matrix multiplication via im2col:
    1. Expand input into an im2col matrix (implicit GEMM, not actually materialized)
    2. Multiply the expanded input with the weights
    3. Produce the output

PyTorch Conv3d formula:
    out(N_i, C_out_j) = bias(C_out_j) +
        sum_{k=0}^{C_in-1} weight(C_out_j, k) * input(N_i, k)

    where * is the 3D cross-correlation operation

Input shape:  (N, C_in, D_in, H_in, W_in)
Weight shape: (C_out, C_in, kD, kH, kW)
Output shape: (N, C_out, D_out, H_out, W_out)

where:
    D_out = (D_in + 2*pad_d - kD) // stride_d + 1
    H_out = (H_in + 2*pad_h - kH) // stride_h + 1
    W_out = (W_in + 2*pad_w - kW) // stride_w + 1
"""

import torch
import triton
import triton.language as tl

# ============================================================================
# Step 1: Conv3d -> GEMM conversion
# ============================================================================
#
# === Implicit GEMM method ===
#
# We treat Conv3d as a big matrix multiplication:
#
# Matrix A (weights): shape = (C_out, C_in * kD * kH * kW)
#   - every row corresponds to one output channel, containing all weights for that channel
#
# Matrix B (im2col expanded input): shape = (C_in * kD * kH * kW, N * D_out * H_out * W_out)
#   - every column corresponds to one output position, containing all input values in that convolutional window
#
# Matrix C (output): shape = (C_out, N * D_out * H_out * W_out)
#   - C = A @ B
#   - then reshape to (N, C_out, D_out, H_out, W_out)
#
# Key optimization: we don't actually create the im2col matrix in memory
# (which would be huge and sparse),
# we access it through index calculation.


# ============================================================================
# Step 2: Triton Conv3d Kernel
# ============================================================================
#
# Autotune configs: Triton will choose the most optimized one based on input size and GPU architecture.
# - BLOCK_M/N/K: tile size, Tensor Core, more shared memory
# - num_warps: every program uses this many warps (usually 4 or 8) (every warp 32 threads)
# - num_stages: pipelining stages for better latency hiding, usually 2-4
#
# key=["out_channels", "in_channels", "out_depth", "out_height", "out_width",
#      "kernel_d", "kernel_h", "kernel_w"]:
#   when these parameters change, the optimal config may change, so we need to re-autotune.

def get_autotune_configs():
    """Autotune BLOCK_M/N/K + GROUP_SIZE_M."""
    configs = []
    for BM in [32, 64, 128]:
        for BN in [32, 64, 128]:
            for BK in [32, 64]:
                for GM in [1, 4, 8]:
                    for num_warps in [4, 8]:
                        for num_stages in [2, 3, 4]:
                            if BM * BN > 128 * 128:
                                continue
                            configs.append(
                                triton.Config(
                                    {"BLOCK_M": BM, "BLOCK_N": BN,
                                     "BLOCK_K": BK, "GROUP_SIZE_M": GM},
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
    # Pointers
    input_ptr,      # Pre-zero-padded input, shape: (N, C_in, D_pad, H_pad, W_pad)
    weight_ptr,     # 2D flattened weights, shape: (C_out, C_in * kD * kH * kW)
    bias_ptr,       # Bias pointer, shape: (C_out,), can be None
    output_ptr,     # Output tensor pointer, shape: (N, C_out, D_out, H_out, W_out)

    # Dimensions 
    batch_size,     # N
    in_channels,    # C_in
    out_channels,   # C_out
    out_depth,      # D_out
    out_height,     # H_out
    out_width,      # W_out

    # Kernel size
    kernel_d,       # kD
    kernel_h,       # kH
    kernel_w,       # kW

    # Conv stride
    stride_d,
    stride_h,
    stride_w,

    # Input tensor stride (post-pad memory stride)
    input_stride_n,
    input_stride_c,
    input_stride_d,
    input_stride_h,
    input_stride_w,

    # Weight tensor stride (2D: only oc direction needed)
    weight_stride_oc,

    # Output tensor stride
    output_stride_n,
    output_stride_c,
    output_stride_d,
    output_stride_h,
    output_stride_w,

    # Constants 
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,      # Block along output channel direction
    BLOCK_N: tl.constexpr,      # Block along output spatial position direction
    BLOCK_K: tl.constexpr,      # Block along reduction direction, searched by autotune
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Triton kernel for Conv3d using the implicit GEMM method.

    GEMM view:
        M = C_out                                (number of output channels)
        N = batch_size * D_out * H_out * W_out   (all output spatial positions)
        K = C_in * kD * kH * kW                  (reduction dimension)

    Each Triton program computes a (BLOCK_M, BLOCK_N) block of the output matrix C.
    """

    # 1. Determine which output block this program is responsible for
    # Using a 1D launch grid, need to manually map to 2D (M, N)
    pid = tl.program_id(axis=0)

    # Total number of programs along M and N directions
    num_pid_m = tl.cdiv(out_channels, BLOCK_M)
    num_pid_n = tl.cdiv(batch_size * out_depth * out_height * out_width, BLOCK_N)

    # Mapping: pid -> (pid_m, pid_n), using grouped ordering.
    # GROUP_SIZE_M rows. Within a group,
    # row along M first, then switch columns.
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    # Handle the last group not being full GROUP_SIZE_M rows
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # 2. Compute output channel indices and spatial position indices for this block 
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # (BLOCK_M,)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # (BLOCK_N,)

    out_spatial = out_depth * out_height * out_width
    total_out = batch_size * out_spatial

    # Decode (n, d_out, h_out, w_out) from rn
    n_idx = rn // out_spatial
    spatial_idx = rn % out_spatial
    d_out_idx = spatial_idx // (out_height * out_width)
    hw_idx = spatial_idx % (out_height * out_width)
    h_out_idx = hw_idx // out_width
    w_out_idx = hw_idx % out_width

    # 3. Reduction loop: accumulate along K = C_in * kD * kH * kW 
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    K = in_channels * kernel_d * kernel_h * kernel_w
    KHW = kernel_h * kernel_w
    KDHW = kernel_d * KHW

    for k_start in range(0, K, BLOCK_K):
        rk = k_start + tl.arange(0, BLOCK_K)   # (BLOCK_K,)

        # Decode (c_in, kd, kh, kw) from rk
        rk_c = rk // KDHW
        rk_rem = rk % KDHW
        rk_d = rk_rem // KHW
        rk_hk = rk_rem % KHW
        rk_h = rk_hk // kernel_w
        rk_w = rk_hk % kernel_w

        # Load weights (2D layout: weight_ptr + rm * stride_oc + rk)
        weight_ptrs = (weight_ptr
                      + rm[:, None] * weight_stride_oc
                      + rk[None, :])
        weight_mask = (rm[:, None] < out_channels) & (rk[None, :] < K)
        w = tl.load(weight_ptrs, mask=weight_mask, other=0.0)

        # Load input (pre-zero-padded, no d/h/w range check needed)
        d_in = d_out_idx[None, :] * stride_d + rk_d[:, None]
        h_in = h_out_idx[None, :] * stride_h + rk_h[:, None]
        w_in = w_out_idx[None, :] * stride_w + rk_w[:, None]

        input_ptrs = (input_ptr
                     + n_idx[None, :] * input_stride_n
                     + rk_c[:, None] * input_stride_c
                     + d_in * input_stride_d
                     + h_in * input_stride_h
                     + w_in * input_stride_w)
        input_mask = (rk[:, None] < K) & (rn[None, :] < total_out)
        x = tl.load(input_ptrs, mask=input_mask, other=0.0)

        acc = tl.dot(w, x, acc)

    # 4. Add bias if we have
    if HAS_BIAS:
        bias = tl.load(bias_ptr + rm, mask=rm < out_channels, other=0.0)
        acc += bias[:, None]

    # 5. Store back
    c_out = acc.to(output_ptr.dtype.element_ty)

    output_ptrs = (output_ptr
                  + n_idx[None, :] * output_stride_n
                  + rm[:, None] * output_stride_c
                  + d_out_idx[None, :] * output_stride_d
                  + h_out_idx[None, :] * output_stride_h
                  + w_out_idx[None, :] * output_stride_w)
    output_mask = (rm[:, None] < out_channels) & (rn[None, :] < total_out)

    tl.store(output_ptrs, c_out, mask=output_mask)


# ============================================================================
# Step 3: Python Wrapper
# ============================================================================

def triton_conv3d(
    input: torch.Tensor,    # (N, C_in, D_in, H_in, W_in)
    weight: torch.Tensor,   # (C_out, C_in, kD, kH, kW)
    bias: torch.Tensor = None,
    stride: tuple = (1, 1, 1),
    padding: tuple = (0, 0, 0),
) -> torch.Tensor:
    """
    Conv3d implemented in Triton, matching torch.nn.Conv3d.

    Args:
        input:   (N, C_in, D_in, H_in, W_in) input tensor
        weight:  (C_out, C_in, kD, kH, kW) conv kernel
        bias:    (C_out,) bias, optional
        stride:  (stride_d, stride_h, stride_w)
        padding: (pad_d, pad_h, pad_w)

    Returns:
        output:  (N, C_out, D_out, H_out, W_out)
    """
    # --- Validate input ---
    assert input.is_contiguous(), "input must be contiguous"
    assert weight.is_contiguous(), "weight must be contiguous"
    assert input.ndim == 5, f"input must be a 5D tensor, got {input.ndim}D"
    assert weight.ndim == 5, f"weight must be a 5D tensor, got {weight.ndim}D"

    N, C_in, D_in, H_in, W_in = input.shape
    C_out, C_in_w, kD, kH, kW = weight.shape
    assert C_in == C_in_w, f"input channels {C_in} != weight input channels {C_in_w}"

    stride_d, stride_h, stride_w = stride
    pad_d, pad_h, pad_w = padding

    # --- Compute output size ---
    D_out = (D_in + 2 * pad_d - kD) // stride_d + 1
    H_out = (H_in + 2 * pad_h - kH) // stride_h + 1
    W_out = (W_in + 2 * pad_w - kW) // stride_w + 1

    assert D_out > 0 and H_out > 0 and W_out > 0, \
        f"invalid output size: ({D_out}, {H_out}, {W_out})"

    # === Optimization A: explicitly zero-pad input, kernel no longer needs d/h/w mask ===
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        input = torch.nn.functional.pad(
            input, (pad_w, pad_w, pad_h, pad_h, pad_d, pad_d)
        )
    # input is now (N, C_in, D_in+2*pad_d, H_in+2*pad_h, W_in+2*pad_w)

    # === Optimization C: reshape weight to 2D, eliminating 4 stride computations in the kernel ===
    weight_2d = weight.view(C_out, -1)   # (C_out, C_in*kD*kH*kW), shared memory

    # --- Allocate output ---
    output = torch.empty(
        (N, C_out, D_out, H_out, W_out),
        device=input.device, dtype=input.dtype
    )

    M = C_out
    total_N = N * D_out * H_out * W_out

    # --- Prepare bias ---
    HAS_BIAS = bias is not None
    if not HAS_BIAS:
        bias = torch.empty(0, device=input.device, dtype=input.dtype)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(total_N, META["BLOCK_N"]),
    )

    conv3d_kernel[grid](
        input, weight_2d, bias, output,
        # Dimensions
        N, C_in,
        C_out, D_out, H_out, W_out,
        kD, kH, kW,
        stride_d, stride_h, stride_w,
        # Input stride (from padded input)
        input.stride(0), input.stride(1), input.stride(2),
        input.stride(3), input.stride(4),
        # Weight stride (2D, only oc direction)
        weight_2d.stride(0),
        # Output stride
        output.stride(0), output.stride(1), output.stride(2),
        output.stride(3), output.stride(4),
        # Constants
        HAS_BIAS=HAS_BIAS,
    )

    return output


# ============================================================================
# Step 4: Tests — compare against PyTorch
# ============================================================================

def test_conv3d():
    """Test that Triton Conv3d matches PyTorch Conv3d results."""

    torch.manual_seed(42)
    device = 'cuda'
    dtype = torch.float32

    # Print GPU info
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Capability: {torch.cuda.get_device_capability(0)}")
    print()

    # --- Test cases ---
    test_cases = [
        # (N, C_in, D, H, W, C_out, kD, kH, kW, stride, padding, use_bias, description)
        (1, 1, 4, 4, 4, 1, 3, 3, 3, (1,1,1), (0,0,0), False,
         "simplest: single batch, single channel, no padding"),

        (1, 1, 4, 4, 4, 1, 3, 3, 3, (1,1,1), (1,1,1), False,
         "with padding"),

        (1, 3, 8, 8, 8, 16, 3, 3, 3, (1,1,1), (1,1,1), True,
         "multi-channel + bias"),

        (2, 3, 8, 8, 8, 16, 3, 3, 3, (2,2,2), (1,1,1), True,
         "stride=2 + bias"),

        (2, 16, 8, 8, 8, 32, 3, 3, 3, (1,1,1), (1,1,1), True,
         "larger channel count"),
    ]

    print("=" * 70)
    print("Triton Conv3d correctness test")
    print("=" * 70)

    all_passed = True
    for (N, C_in, D, H, W, C_out, kD, kH, kW,
         stride, padding, use_bias, desc) in test_cases:

        # Create input
        x = torch.randn(N, C_in, D, H, W, device=device, dtype=dtype)
        w = torch.randn(C_out, C_in, kD, kH, kW, device=device, dtype=dtype)
        b = torch.randn(C_out, device=device, dtype=dtype) if use_bias else None

        # PyTorch reference result
        torch_out = torch.nn.functional.conv3d(x, w, b, stride=stride, padding=padding)

        # Triton result
        triton_out = triton_conv3d(x, w, b, stride=stride, padding=padding)

        # Compare (TF32 tolerance: ~1e-2, matches actual Tensor Core behavior)
        max_diff = (torch_out - triton_out).abs().max().item()
        passed = torch.allclose(torch_out, triton_out, atol=1e-2, rtol=1e-2)

        status = "PASS" if passed else "FAIL"
        print(f"{status} | {desc}")
        print(f"       shape: ({N},{C_in},{D},{H},{W}) -> ({N},{C_out},"
              f"{torch_out.shape[2]},{torch_out.shape[3]},{torch_out.shape[4]})")
        print(f"       max diff: {max_diff:.6e}")

        if not passed:
            all_passed = False

    print("=" * 70)
    if all_passed:
        print("All tests passed!")
    else:
        print("Some tests failed, please check the implementation")
    print("=" * 70)


if __name__ == "__main__":
    test_conv3d()
