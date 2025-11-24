import torch

from fused_mm_sampling.tl_matmul import matmul


def test_matmul():
    device = torch.device("cuda")
    torch.manual_seed(0)
    a = torch.randn(100, 100, device=device, dtype=torch.bfloat16)
    b = torch.randn(100, 100, device=device, dtype=torch.bfloat16)
    c = matmul(a, b)
    assert torch.allclose(c, a @ b)
