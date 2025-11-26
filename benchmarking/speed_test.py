import os

os.environ["TRITON_PRINT_AUTOTUNING"] = "1"
# os.environ["TRITON_INTERPRET"] = "1"

from fused_mm_sampling.bench.speed_test import CliArgs, run_speed_test

if __name__ == "__main__":
    args = CliArgs()
    run_speed_test(args)
