#GPU := h100
#GPU := a100-80gb
GPU := b200
TRITON_BENCH_DIR = benchmarking/modal-results/triton-bench-$(GPU)

modal-speed-test:
	modal run -m src.fused_mm_sampling.modal_lib.modal_speed_test

modal-triton-benchmark: modal-create-results-triton-bench modal-get-results-triton-bench modal-plot-triton-bench

modal-create-results-triton-bench:
	mkdir -p $(TRITON_BENCH_DIR)
	GPU=$(GPU) TGT_DIR="/vol-fused-mm-sample/triton-bench-$(GPU)" \
	modal run \
		-m src.fused_mm_sampling.modal_lib.modal_triton_benchmark \
		> $(TRITON_BENCH_DIR)/logs.txt

modal-plot-triton-bench:
	python benchmarking/plot-triton-bench.py --tgt_dir $(TRITON_BENCH_DIR)

modal-example:
	modal run -m src.fused_mm_sampling.modal_lib.modal_example

modal-get-results-speed-test:
	mkdir -p benchmarking/modal-results/
	cd benchmarking/modal-results/ && modal volume get fused-mm-sample speed-test

modal-get-results-triton-bench:
	mkdir -p benchmarking/modal-results/
	cd benchmarking/modal-results/ && modal volume get fused-mm-sample triton-bench-$(GPU)

modal-persistent-matmul:
	modal run -m src.fused_mm_sampling.modal_lib.modal_persistent_matmul