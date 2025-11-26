from .utils import make_app, make_image

app = make_app()
image = make_image()


def square(inner_dim: int):
    import torch

    device = torch.device("cuda")
    a = torch.randn(1000, inner_dim, device=device, dtype=torch.bfloat16)
    b = torch.randn(inner_dim, 1000, device=device, dtype=torch.bfloat16)
    c = a @ b
    print(c.shape)


@app.function(gpu="H100", image=image)
def modal_square(inner_dim: int):
    return square(inner_dim)


@app.local_entrypoint()
def main():
    modal_square.remote(inner_dim=200)
