from pathlib import Path

from ..bench.triton_benchmark import Args, run_triton_bechmark
from .utils import make_app, make_image, make_volumes, volume_path

args = Args(tgt_dir=Path(volume_path) / "triton-bench")
app = make_app()


@app.function(gpu="H100", image=make_image(), volumes=make_volumes(), timeout=10 * 60)
def function(args: Args):
    run_triton_bechmark(args)


@app.local_entrypoint()
def main():
    function.remote(args=args)
