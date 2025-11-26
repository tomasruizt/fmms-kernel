modal-speed-test:
	modal run -m src.fused_mm_sampling.modal_lib.modal_speed_test

modal-triton-benchmark: modal-create-results-triton-bench modal-get-results-triton-bench modal-plot-triton-bench

modal-create-results-triton-bench:
	mkdir -p benchmarking/modal-results/triton-bench/
	modal run \
		-m src.fused_mm_sampling.modal_lib.modal_triton_benchmark \
		> benchmarking/modal-results/triton-bench/logs.txt

modal-plot-triton-bench:
	python benchmarking/plot-triton-bench.py --tgt_dir benchmarking/modal-results/triton-bench/

modal-example:
	modal run -m src.fused_mm_sampling.modal_lib.modal_example

modal-get-results-speed-test:
	mkdir -p benchmarking/modal-results/
	cd benchmarking/modal-results/ && modal volume get fused-mm-sample speed-test

modal-get-results-triton-bench:
	mkdir -p benchmarking/modal-results/
	cd benchmarking/modal-results/ && modal volume get fused-mm-sample triton-bench