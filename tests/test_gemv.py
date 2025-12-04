import torch

from fused_mm_sampling.tl_gemv import tl_gemv


def test_gemv():
    device = torch.device("cuda")
    hidden_size = 4096
    vocab_size = 128_000
    hidden_states = torch.randn(hidden_size, device=device)
    weights = torch.randn(hidden_size, vocab_size, device=device)
    expected = hidden_states @ weights

    tl_result = tl_gemv(hidden_states, weights)
    assert torch.cosine_similarity(expected, tl_result, dim=0) > 0.99
