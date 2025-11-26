modal-speed-test:
	modal run -m src.fused_mm_sampling.modal_lib.modal_speed_test

modal-example:
	modal run -m src.fused_mm_sampling.modal_lib.modal_example

modal-get-results-speed-test:
	modal volume get fused-mm-sample speed-test
	mkdir -p benchmarking/modal-results/
	mv speed-test benchmarking/modal-results/