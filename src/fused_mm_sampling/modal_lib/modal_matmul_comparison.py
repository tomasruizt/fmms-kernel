import os

from ..bench.matmul_comparison import matmul_comparison_main
from .utils import make_app, make_image, make_volumes

app = make_app()

gpu = os.getenv("GPU")
print(f"Using GPU: {gpu}")


@app.function(gpu=gpu, image=make_image(), volumes=make_volumes(), timeout=10 * 60)
def my_func():
    matmul_comparison_main()


@app.local_entrypoint()
def main():
    my_func.remote()
