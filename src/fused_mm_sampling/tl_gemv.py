import torch
import triton
import triton.language as tl


def tl_gemv(
    x: torch.Tensor,
    Y: torch.Tensor,  # noqa: N803
) -> torch.Tensor:
    """
    x: tensor of shape (M)
    Y: matrix of shape (M, B)
    return: matrix-vector product of Y and x, shaped (B)
    """
    assert x.shape[0] == Y.shape[0], (x.shape, Y.shape)
    M, B = Y.shape  # noqa: N806
    out = torch.empty(B, device=x.device, dtype=x.dtype)

    def grid(meta):
        return (triton.cdiv(B, meta["BLOCK_SIZE_B"]),)

    tl_gemv_kernel[grid](x, Y, out, M, B)
    return out


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_B": bsz_b,
                "x_tile_side": x_tile_side,
                "k_tile_side": k_tile_side,
            },
        )
        # for bsz_b in [8, 16, 32]
        # for x_tile_side in [8, 16, 32]
        # for k_tile_side in [16, 32, 64]
        for bsz_b in [16]
        for x_tile_side in [16]
        for k_tile_side in [16]
    ],
    key=["M", "B"],
    cache_results=True,
)
@triton.jit
def tl_gemv_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    M: tl.constexpr,  # noqa: N803
    B: tl.constexpr,  # noqa: N803
    BLOCK_SIZE_B: tl.constexpr,  # noqa: N803
    x_tile_side: tl.constexpr,
    k_tile_side: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    x_base_offs = tl.arange(0, x_tile_side * k_tile_side).reshape(x_tile_side, k_tile_side)
    y_base_offs = (
        tl.arange(0, BLOCK_SIZE_B)[None, :] + B * tl.arange(0, x_tile_side * k_tile_side)[:, None]
    )
    mult = tl.zeros((x_tile_side, x_tile_side * BLOCK_SIZE_B), dtype=tl.float32)
    for tile_offs in range(0, M, x_tile_side * k_tile_side):
        x_offs = x_base_offs + tile_offs
        x_tile = tl.load(x_ptr + x_offs, mask=x_offs < M, other=0.0)

        y_offs = y_base_offs + tile_offs * B + pid * BLOCK_SIZE_B
        row_ok = y_offs < B * M
        col_ok = (pid * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B))[None, :] < B
        y_tile = tl.load(y_ptr + y_offs, mask=row_ok & col_ok, other=0.0)
        y_tile = y_tile.T.reshape(BLOCK_SIZE_B * x_tile_side, k_tile_side).T

        mult = tl.dot(x_tile, y_tile, acc=mult)  # (M, B*M)
    indices = (
        tl.arange(0, x_tile_side)[:, None] + x_tile_side * tl.arange(0, BLOCK_SIZE_B)[None, :]
    )  # (M, B)
    diagonals = mult.gather(axis=1, index=indices)  # (M, B)
    dot_product = diagonals.sum(0)  # (B)
    tl.store(
        out_ptr + pid * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B),
        dot_product,
        mask=pid * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B) < B,
    )
