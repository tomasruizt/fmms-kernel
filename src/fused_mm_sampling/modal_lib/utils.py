import modal


def make_app():
    return modal.App("fused-matmul-sample")


def make_image():
    img = modal.Image.from_registry("pytorch/pytorch:2.10.0-cuda13.0-cudnn9-devel")
    deps = [
        "flashinfer-python",
        "pandas",
        "pydantic-settings",
        "matplotlib",
        "nvtx",
        "llnl-hatchet",
        "nvidia-cutlass",
        "cuda-bench",
        "cupti-python",
    ]
    return img.uv_pip_install(deps)


volume_path = "/vol-fused-mm-sample"


def make_volumes():
    return {volume_path: modal.Volume.from_name("fused-mm-sample")}


def enable_cuda_jit_cache():
    """Persist JIT-compiled CUDA extensions on the Modal volume.

    Call this at the start of any Modal function that JIT-compiles CUDA code
    (e.g. CUTLASS kernels via torch.utils.cpp_extension.load()). Subsequent
    runs with unchanged source skip the expensive nvcc compilation.
    """
    import os

    os.environ["TORCH_EXTENSIONS_DIR"] = f"{volume_path}/cache/torch_extensions"
