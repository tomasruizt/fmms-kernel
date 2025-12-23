from .utils import make_app, make_image

app = make_app()
image = make_image()


def torch_matmul(inner_dim: int):
    import torch

    device = torch.device("cuda")
    a = torch.randn(1000, inner_dim, device=device, dtype=torch.bfloat16)
    b = torch.randn(2000, inner_dim, device=device, dtype=torch.bfloat16)
    c = a @ b.T
    print(c.shape)


@app.function(gpu="H100", image=image)
def modal_matmul(inner_dim: int):
    return torch_matmul(inner_dim)


@app.local_entrypoint()
def main():
    modal_matmul.remote(inner_dim=200)
