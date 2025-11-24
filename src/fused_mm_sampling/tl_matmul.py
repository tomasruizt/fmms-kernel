import torch
import triton
import triton.language as tl


# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_N": bsz_v,
                "BLOCK_SIZE_K": bsz_d,
                "BLOCK_SIZE_M": bsz_h,
                "GROUP_SIZE_M": 4,
            },
            num_warps=num_warps,
            num_stages=num_stages,
            maxnreg=maxnreg,
        )
        for bsz_v in [32, 4 * 32, 8 * 32]
        for bsz_d in [32, 64]
        for bsz_h in [16, 64, 128, 256]
        for num_warps in [8]  # Default 4
        for maxnreg in [128]  # Previously 255
        for num_stages in [3]  # Higher values increase SRAM requirements, but 4 outperfomed 2.
    ],
    key=["M", "N", "K"],
    cache_results=True,
)
@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    # Matrix dimensions
    M,  # noqa: N803
    N,  # noqa: N803
    K,  # noqa: N803
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,  # noqa: N803
    BLOCK_SIZE_N: tl.constexpr,  # noqa: N803
    BLOCK_SIZE_K: tl.constexpr,  # noqa: N803
    GROUP_SIZE_M: tl.constexpr,  # noqa: N803
    ACTIVATION: tl.constexpr,  # noqa: N803
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # Process N dimension first, then M (matching fused kernel pattern).
    # This enables processing many N blocks for the same M block, allowing
    # A matrix (small M dimension) to be reused from L2 cache.
    pid_n = tl.program_id(axis=0)  # N dimension first (like vocab in fused kernel)
    pid_m = tl.program_id(axis=1)  # M dimension second (like hidden_states in fused kernel)

    # Swizzle for L2 cache optimization (N first to match fused kernel)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_n, pid_m = tl.swizzle2d(pid_n, pid_m, num_pid_n, num_pid_m, GROUP_SIZE_M)

    # -----------------------------------------------------------
    # Add some integer bound assumptions.
    # This helps to guide integer analysis in the backend to optimize
    # load/store offset address calculation
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.bfloat16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# We can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `matmul_kernel`.
@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)


def matmul(a, b, activation=""):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape  # noqa: N806
    K, N = b.shape  # noqa: N806
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)

    # 2D launch kernel with N first to match fused kernel pattern.
    # This enables processing many N blocks for the same M block,
    # allowing A matrix (small M dimension) to be reused from L2 cache.
    def grid(meta):
        return (
            triton.cdiv(N, meta["BLOCK_SIZE_N"]),  # N first (like vocab in fused kernel)
            triton.cdiv(M, meta["BLOCK_SIZE_M"]),  # M second (like hidden_states in fused kernel)
        )

    matmul_kernel[grid](
        a,
        b,
        c,
        M,  # noqa: N803
        N,  # noqa: N803
        K,  # noqa: N803
        a.stride(0),
        a.stride(1),  #
        b.stride(0),
        b.stride(1),  #
        c.stride(0),
        c.stride(1),  #
        ACTIVATION=activation,  #
    )
    return c
