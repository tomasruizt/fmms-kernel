"""Test EVT epilogues on Modal H100 (SM90+)."""

from ..testing import assert_sampling_distribution
from .utils import clear_cuda_jit_cache, enable_cuda_jit_cache, make_app, make_image, make_volumes

app = make_app()


@app.function(gpu="H100", image=make_image(), volumes=make_volumes(), timeout=20 * 60)
def test_evt(clear_cache: bool = False, test: str = "sampling-evt"):
    enable_cuda_jit_cache()
    if clear_cache:
        clear_cuda_jit_cache()
    if test == "sampling":
        run_test_cutlass_sampling()
    elif test == "sampling-evt":
        run_test_cutlass_evt_sampling()
    else:
        raise ValueError(f"Unknown test: {test}")


def run_test_cutlass_sampling():
    for vocab_size in [100, 200, 256]:
        for n_hidden_states in [1, 2]:
            assert_sampling_distribution("fused-cutlass", vocab_size, n_hidden_states)
            print(f"fused-cutlass sampling: V={vocab_size}, H={n_hidden_states}: PASS")

    print("\nAll fused-cutlass sampling tests passed!")


def run_test_cutlass_evt_sampling():
    for vocab_size in [100, 200, 256]:
        for n_hidden_states in [1, 2]:
            assert_sampling_distribution("fused-cutlass-evt", vocab_size, n_hidden_states)
            print(f"fused-cutlass-evt sampling: V={vocab_size}, H={n_hidden_states}: PASS")

    print("\nAll fused-cutlass-evt sampling tests passed!")


@app.local_entrypoint()
def main(clear_cache: bool = False, test: str = "sampling-evt"):
    test_evt.remote(clear_cache=clear_cache, test=test)
