"""Test EVT epilogues on Modal H100 (SM90+)."""

from dataclasses import dataclass

from ..testing import assert_sampling_distribution
from .utils import clear_cuda_jit_cache, enable_cuda_jit_cache, make_app, make_image, make_volumes


@dataclass
class CaseArgs:
    V: int
    D: int
    H: int
    temp: float = 1.0


ARGMAX_CASES = [
    CaseArgs(V=256, D=128, H=1),
    CaseArgs(V=256, D=128, H=4),
    CaseArgs(V=1024, D=256, H=7),
    CaseArgs(V=151936, D=4096, H=4),
    CaseArgs(V=151936, D=4096, H=4, temp=0.5),
]

app = make_app()


@app.function(gpu="H100", image=make_image(), volumes=make_volumes(), timeout=20 * 60)
def test_evt(clear_cache: bool = False, test: str = "argmax"):
    enable_cuda_jit_cache()
    if clear_cache:
        clear_cuda_jit_cache()
    if test == "argmax":
        run_test_evt_row_argmax()
    elif test == "sampling":
        run_test_cutlass_sampling()
    else:
        raise ValueError(f"Unknown test: {test}")


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


def run_test_evt_row_argmax():
    for case in ARGMAX_CASES:
        assert_row_argmax(case)
    print("\nAll EVT row_argmax tests passed!")


def assert_row_argmax(case: CaseArgs):
    import torch

    from ..cutlass_impl import test_row_argmax

    device = torch.device("cuda")
    V, D, H, temperature = case.V, case.D, case.H, case.temp  # noqa: N806
    weights = torch.randn(V, D, device=device, dtype=torch.bfloat16)
    hidden_states = torch.randn(H, D, device=device, dtype=torch.bfloat16)

    result = test_row_argmax(weights, hidden_states, temperature=temperature)

    # Reference: matmul, scale by 1/temperature, then argmax across V (dim=0)
    ref_logits = weights.float() @ hidden_states.float().T
    ref_scaled = ref_logits / temperature
    ref_argmax = ref_scaled.argmax(dim=0)

    match = (result.long() == ref_argmax).all().item()
    mismatches = (result.long() != ref_argmax).sum().item()
    print(
        f"EVT row_argmax: V={V}, D={D}, H={H}, T={temperature}: "
        f"match={match}, mismatches={mismatches}"
    )
    if not match:
        for h in range(min(H, 8)):
            if result[h].item() != ref_argmax[h].item():
                got_idx = result[h].item()
                want_idx = ref_argmax[h].item()
                got_val = ref_scaled[got_idx, h].item()
                want_val = ref_scaled[want_idx, h].item()
                print(
                    f"  h={h}: got idx={got_idx} (val={got_val:.4f}), "
                    f"want idx={want_idx} (val={want_val:.4f}), "
                    f"diff={want_val - got_val:.6f}"
                )
    assert match, f"EVT row_argmax failed: {mismatches}/{H} mismatches"


def run_test_cutlass_sampling():
    for vocab_size in [100, 200, 256]:
        for n_hidden_states in [1, 2]:
            assert_sampling_distribution("fused-cutlass", vocab_size, n_hidden_states)
            print(f"fused-cutlass sampling: V={vocab_size}, H={n_hidden_states}: PASS")

    print("\nAll fused-cutlass sampling tests passed!")


@app.local_entrypoint()
def main(clear_cache: bool = False, test: str = "argmax"):
    test_evt.remote(clear_cache=clear_cache, test=test)
