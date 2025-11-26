from ..bench.speed_test import Args, run_speed_test
from .utils import make_app, make_image

args = Args(n_hidden_states=4)
app = make_app()


@app.function(gpu="H100", image=make_image())
def speed_test(args: Args):
    run_speed_test(args)


@app.local_entrypoint()
def main():
    speed_test.remote(args=args)
