from ..persistent_matmul import main as persistent_matmul_main
from .utils import make_app, make_image, make_volumes

app = make_app()


@app.function(gpu="H100", image=make_image(), volumes=make_volumes())
def speed_test():
    persistent_matmul_main()


@app.local_entrypoint()
def main():
    speed_test.remote()
