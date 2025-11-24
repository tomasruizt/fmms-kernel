import modal
import torch

app = modal.App("example-get-started")
image = modal.Image.from_registry("pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel")


@app.function(gpu="H100", image=image)
def square(inner_dim: int):
    device = torch.device("cuda")
    a = torch.randn(1000, inner_dim, device=device, dtype=torch.bfloat16)
    b = torch.randn(inner_dim, 1000, device=device, dtype=torch.bfloat16)
    c = a @ b
    print(c.shape)


@app.local_entrypoint()
def main():
    square.remote(inner_dim=200)
