from pathlib import Path

from ..bench.speed_test import Args, run_speed_test
from .utils import make_app, make_image, make_volumes, volume_path

args = Args(n_hidden_states=4, tgt_dir=Path(volume_path) / "speed-test")
app = make_app()


@app.function(gpu="H100", image=make_image(), volumes=make_volumes(), timeout=20 * 60)
def speed_test(args: Args, clear_cache: bool = False):
    from .utils import clear_cuda_jit_cache, enable_cuda_jit_cache

    enable_cuda_jit_cache()
    if clear_cache:
        clear_cuda_jit_cache()
    run_speed_test(args)


@app.local_entrypoint()
def main(name: str = "", clear_cache: bool = False):
    a = args if not name else args.model_copy(update={"name": name})
    speed_test.remote(args=a, clear_cache=clear_cache)
