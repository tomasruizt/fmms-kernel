import os

from ..persistent_matmul import main as persistent_matmul_main
from .utils import make_app, make_image, make_volumes

app = make_app()

gpu = os.getenv("GPU")
print(f"Using GPU: {gpu}")


@app.function(gpu=gpu, image=make_image(), volumes=make_volumes(), timeout=10 * 60)
def speed_test():
    persistent_matmul_main()


@app.local_entrypoint()
def main():
    speed_test.remote()
