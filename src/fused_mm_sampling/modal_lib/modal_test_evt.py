"""Test EVT epilogues on Modal H100 (SM90+)."""

from .utils import enable_cuda_jit_cache, make_app, make_image, make_volumes

app = make_app()


@app.function(gpu="H100", image=make_image(), volumes=make_volumes(), timeout=20 * 60)
def test_evt():
    enable_cuda_jit_cache()
    run_test_evt_add1()
    run_test_evt_row_reduce()


def run_test_evt_add1():
    import torch

    from ..cutlass_impl import test_evt_add1

    device = torch.device("cuda")
    torch.manual_seed(42)

    for V, D, H in [(256, 128, 1), (256, 128, 4), (1024, 256, 7), (151936, 4096, 4)]:  # noqa: N806
        weights = torch.randn(V, D, device=device, dtype=torch.bfloat16)
        hidden_states = torch.randn(H, D, device=device, dtype=torch.bfloat16)

        result = test_evt_add1(weights, hidden_states)

        # Reference: matmul + 1
        reference = (weights.float() @ hidden_states.float().T) + 1.0

        max_err = (result - reference).abs().max().item()
        print(f"EVT add1: V={V}, D={D}, H={H}: max_err={max_err:.4f} (shape={result.shape})")
        assert max_err < 1.0, f"EVT add1 failed: max_err={max_err}"

    print("\nAll EVT add1 tests passed!")


def run_test_evt_row_reduce():
    import torch

    from ..cutlass_impl import test_evt_row_reduce

    device = torch.device("cuda")
    torch.manual_seed(42)

    for V, D, H in [(256, 128, 1), (256, 128, 4), (1024, 256, 7), (151936, 4096, 4)]:  # noqa: N806
        weights = torch.randn(V, D, device=device, dtype=torch.bfloat16)
        hidden_states = torch.randn(H, D, device=device, dtype=torch.bfloat16)

        logits, row_max = test_evt_row_reduce(weights, hidden_states)

        # Reference: matmul and max across V (dim=0)
        ref_logits = weights.float() @ hidden_states.float().T
        ref_max = ref_logits.max(dim=0).values

        logits_err = (logits - ref_logits).abs().max().item()
        max_err = (row_max - ref_max).abs().max().item()
        print(
            f"EVT row_reduce: V={V}, D={D}, H={H}: "
            f"logits_err={logits_err:.4f}, max_err={max_err:.4f} "
            f"(logits={logits.shape}, row_max={row_max.shape})"
        )
        assert logits_err < 1.0, f"EVT row_reduce logits failed: err={logits_err}"
        assert max_err < 1.0, f"EVT row_reduce max failed: err={max_err}"

    print("\nAll EVT row_reduce tests passed!")


@app.local_entrypoint()
def main():
    test_evt.remote()
