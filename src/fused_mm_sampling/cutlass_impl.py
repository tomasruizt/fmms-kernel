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

    _module = load(
        name="fmms_cutlass",
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
    return _module


TILE_V = 128


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
