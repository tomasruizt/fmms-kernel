"""Test EVT epilogue on Modal H100 (SM90+)."""

from .utils import make_app, make_image, make_volumes

app = make_app()


@app.function(gpu="H100", image=make_image(), volumes=make_volumes(), timeout=20 * 60)
def test_evt():
    run_test_evt()


def run_test_evt():
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
        print(f"V={V}, D={D}, H={H}: max_err={max_err:.4f} (shape={result.shape})")
        assert max_err < 1.0, f"EVT add1 failed: max_err={max_err}"

    print("\nAll EVT add1 tests passed!")


@app.local_entrypoint()
def main():
    test_evt.remote()
