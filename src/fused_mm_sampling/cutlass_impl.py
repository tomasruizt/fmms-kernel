"""CUTLASS 3.x implementation of the FMMS kernel (SM90+).

Uses CUTLASS 3.x GemmUniversal with collective builders and TMA for the
matmul, followed by a custom Gumbel-max sampling kernel. Stage 2 reduction
is done in Python (identical to the Triton and CUDA wrappers).

Requirements:
  - SM90+ GPU (H100 or newer).
  - CUTLASS 3.x headers: pip install nvidia-cutlass
    OR set CUTLASS_PATH to the CUTLASS include directory.
  - CUDA toolkit with nvcc supporting compute_90a.
"""

import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_CSRC_DIR = Path(__file__).resolve().parent / "csrc"

_module = None


def _get_cutlass_include():
    """Find CUTLASS include directory."""
    # Check explicit env var first
    cutlass_path = os.environ.get("CUTLASS_PATH")
    if cutlass_path:
        return cutlass_path

    # Try nvidia-cutlass pip package (headers live in cutlass_library/source/include/)
    try:
        import cutlass_library

        include_dir = Path(cutlass_library.__file__).resolve().parent / "source" / "include"
        if include_dir.exists():
            return str(include_dir)
    except ImportError:
        pass

    # Try common system locations
    for candidate in [
        "/usr/local/cutlass/include",
        "/usr/include/cutlass",
        Path.home() / "cutlass" / "include",
    ]:
        if Path(candidate).exists():
            return str(candidate)

    raise RuntimeError(
        "CUTLASS headers not found. Install with: pip install nvidia-cutlass\n"
        "Or set CUTLASS_PATH to the CUTLASS include directory."
    )


def _sm_version() -> str:
    """Return the SM version string (e.g. '80' for A100, '90' for H100)."""
    major, minor = torch.cuda.get_device_capability()
    return f"{major}{minor}"


def _get_module():
    global _module
    if _module is not None:
        return _module

    cutlass_include = _get_cutlass_include()
    sm = _sm_version()
    sm_int = int(sm)

    if sm_int < 90:
        raise RuntimeError(f"CUTLASS 3.x FMMS kernel requires SM90+ (H100 or newer), got SM{sm}")

    # SM90+ needs the 'a' suffix for TMA/WGMMA instructions
    sm_gencode = f"{sm}a"

    # Include header content in the extension name hash so changes to .hpp
    # files invalidate the JIT cache (torch only tracks files in sources=).
    header_hash = _csrc_headers_hash()
    name = f"fmms_cutlass_{header_hash}"

    import time

    had_cache = _find_cached_so(name) is not None

    t0 = time.monotonic()
    _module = load(
        name=name,
        sources=[str(_CSRC_DIR / "fmms_cutlass_kernel.cu")],
        extra_include_paths=[cutlass_include],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            f"-gencode=arch=compute_{sm_gencode},code=sm_{sm_gencode}",
            "-std=c++17",
        ],
        verbose=os.environ.get("FMMS_CUDA_VERBOSE", "") == "1",
    )
    elapsed = time.monotonic() - t0
    print(
        f"CUTLASS FMMS kernel loaded ({'likely cached' if had_cache else 'likely compiled'}) in {elapsed:.1f}s"
    )
    return _module


def _find_cached_so(name: str) -> Path | None:
    """Return path to a cached .so for a JIT extension, or None if not cached."""
    from torch.utils.cpp_extension import _get_build_directory

    build_dir = Path(_get_build_directory(name, verbose=False))
    # torch appends _v{version} when sources change; glob for any version.
    matches = sorted(build_dir.parent.glob(f"{name}*/{name}*.so"))
    return matches[0] if matches else None


def _csrc_headers_hash() -> str:
    """Hash all .hpp files in csrc/ to detect header changes for JIT cache invalidation."""
    import hashlib

    h = hashlib.md5()
    for hpp in sorted(_CSRC_DIR.glob("*.hpp")):
        h.update(hpp.read_bytes())
    return h.hexdigest()[:8]


TILE_V = 128


def test_evt_add1(
    weights: torch.Tensor,  # [V, D] bfloat16
    hidden_states: torch.Tensor,  # [H, D] bfloat16
) -> torch.Tensor:
    """Test EVT epilogue: returns matmul(weights, hidden_states.T) + 1.0.

    Validates that the Epilogue Visitor Tree infrastructure works end-to-end
    by adding a scalar (1.0) to every accumulator element.
    """
    mod = _get_module()
    return mod.test_evt_add1(weights, hidden_states)


def test_evt_row_reduce(
    weights: torch.Tensor,  # [V, D] bfloat16
    hidden_states: torch.Tensor,  # [H, D] bfloat16
) -> tuple[torch.Tensor, torch.Tensor]:
    """Test EVT row reduction: returns (logits, row_max).

    logits = matmul(weights, hidden_states.T)  shape [V, H]
    row_max = max(logits, dim=0).values         shape [H]

    Validates that Sm90RowReduction with maximum works correctly.
    """
    mod = _get_module()
    return mod.test_evt_row_reduce(weights, hidden_states)


def test_row_argmax(
    weights: torch.Tensor,  # [V, D] bfloat16
    hidden_states: torch.Tensor,  # [H, D] bfloat16
    temperature: torch.Tensor,  # 0-d float32 GPU tensor
    seed: int = 0,
) -> torch.Tensor:
    """2-stage Gumbel-max sampling: GEMM + per-tile argmax + Python reduction.

    Stage 1: CUTLASS GEMM with EVT epilogue applies temperature scaling,
    adds Gumbel noise, and computes per-tile argmax. No intermediate [V, H]
    logits buffer.
    Stage 2: Python reduces across tiles to get the global argmax.
    """
    mod = _get_module()
    inv_temperature = 1.0 / temperature  # 0-d GPU tensor
    tile_max_vals, tile_max_idxs = mod.test_row_argmax(
        weights, hidden_states, inv_temperature, seed
    )
    # Stage 2: reduce across V-tiles
    best_tiles = tile_max_vals.argmax(dim=0)  # [H]
    argmax_idxs = tile_max_idxs.gather(0, best_tiles.unsqueeze(0).to(torch.int32)).squeeze(0)
    return argmax_idxs.to(torch.int64)


def fused_mm_sample_cutlass_evt(
    weights: torch.Tensor,  # [V, D] bfloat16
    hidden_states: torch.Tensor,  # [H, D] bfloat16
    num_samples: int,
    temperature: torch.Tensor,  # scalar (0-d)
    seed: int = 0,
) -> torch.Tensor:
    """Fused matrix-multiply & sampling using CUTLASS EVT (single kernel per sample).

    The EVT epilogue fuses temperature scaling, Gumbel noise, and per-tile argmax
    into the GEMM epilogue. No intermediate [V, H] logits buffer is allocated.
    For num_samples > 1, runs the GEMM once per sample with different seeds.
    """
    V, D = weights.shape  # noqa: N806
    H = hidden_states.shape[0]  # noqa: N806
    assert hidden_states.shape[1] == D

    # CUTLASS TMA requires D aligned to 8 for bf16 (16 bytes)
    D_aligned = ((D + 7) // 8) * 8  # noqa: N806
    if D_aligned != D:
        weights = torch.nn.functional.pad(weights, (0, D_aligned - D))
        hidden_states = torch.nn.functional.pad(hidden_states, (0, D_aligned - D))

    # Compute inv_temperature as a 0-d GPU tensor (no CPU-GPU sync)
    if temperature.dtype != torch.float32:
        temperature = temperature.float()
    inv_temperature = 1.0 / temperature

    mod = _get_module()

    samples = torch.empty((H, num_samples), dtype=torch.int64, device=weights.device)

    for s in range(num_samples):
        sample_seed = seed + s * 1000003  # different seed per sample
        tile_max_vals, tile_max_idxs = mod.test_row_argmax(
            weights, hidden_states, inv_temperature, sample_seed
        )
        # Stage 2: reduce across V-tiles
        best_tiles = tile_max_vals.argmax(dim=0)  # [H]
        argmax_idxs = tile_max_idxs.gather(0, best_tiles.unsqueeze(0).to(torch.int32)).squeeze(0)
        samples[:, s] = argmax_idxs.to(torch.int64)

    return samples


def fused_mm_sample_cutlass(
    weights: torch.Tensor,  # [V, D] bfloat16
    hidden_states: torch.Tensor,  # [H, D] bfloat16
    num_samples: int,
    temperature: torch.Tensor,  # scalar (0-d)
    seed: int = 0,
) -> torch.Tensor:
    """Fused matrix-multiply & sampling using CUTLASS GEMM + Gumbel argmax."""
    V, D = weights.shape  # noqa: N806
    H = hidden_states.shape[0]  # noqa: N806
    assert hidden_states.shape[1] == D

    # CUTLASS TMA requires D aligned to 8 for bf16 (16 bytes).
    # Zero-pad doesn't affect matmul results for original columns.
    D_aligned = ((D + 7) // 8) * 8  # noqa: N806
    if D_aligned != D:
        weights = torch.nn.functional.pad(weights, (0, D_aligned - D))
        hidden_states = torch.nn.functional.pad(hidden_states, (0, D_aligned - D))

    n_tiles_v = (V + TILE_V - 1) // TILE_V

    # Temperature must be float32 on GPU
    if temperature.dtype != torch.float32:
        temperature = temperature.float()

    maxs = torch.empty((n_tiles_v, H, num_samples), dtype=torch.float32, device=weights.device)
    maxs_idx = torch.empty((n_tiles_v, H, num_samples), dtype=torch.long, device=weights.device)

    mod = _get_module()
    mod.fmms_cutlass_stage1(weights, hidden_states, maxs, maxs_idx, temperature, seed)

    # Stage 2: reduce across V-tiles (identical to Triton/CUDA wrappers)
    idxs = maxs.max(dim=0).indices  # [H, num_samples]
    samples = maxs_idx.gather(dim=0, index=idxs.unsqueeze(0)).squeeze(0)
    return samples  # [H, num_samples]
