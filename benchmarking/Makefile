profile-mem:
	python profile-mem.py
	echo "Converting memory snapshot to HTML"
	python _memory_viz.py trace_plot benchmarking/memory/mem-snapshot.pickle -o benchmarking/memory/mem-snapshot.html

download-html-conversion-tool:
	wget https://raw.githubusercontent.com/pytorch/pytorch/refs/heads/main/torch/cuda/_memory_viz.py


speed-test:
	python speed_test.py

NCU=$(shell which ncu)
PYTHON=$(shell which python)
N_HIDDEN_STATES=4
N_RUNS_BENCHMARK=1

# NCU profiling targets:
# --import-source yes: Embeds source code into the .ncu-rep file, allowing it to be
#   viewed on a different machine without path remapping. The source is embedded
#   regardless of where you open the report.
# --source-folders: Comma-separated search paths for source files to import into report.
#   For Triton kernels, this helps NCU find the Python source files.
ncu-profile-fused-triton:
	mkdir -p profiles/ncu
	$(NCU) --set detailed \
		--nvtx \
		--nvtx-include "kernel/" \
		--kernel-name fused_mm_sample_triton_kernel \
		--source-folders $$(pwd)/../src/fused_mm_sampling \
		--import-source yes \
		-fo profiles/ncu/fused-triton $(PYTHON) ./speed_test.py --name fused-triton --n_hidden_states $(N_HIDDEN_STATES) --n_runs_benchmark $(N_RUNS_BENCHMARK)


ncu-profile-flashinfer:
	mkdir -p profiles/ncu
	ncu --set detailed \
		--nvtx \
		--nvtx-include "kernel/" \
		-fo profiles/ncu/flashinfer python ./speed_test.py --name flashinfer:sampling_from_logits --n_hidden_states $(N_HIDDEN_STATES) --n_runs_benchmark $(N_RUNS_BENCHMARK)


ncu-profile-naive-compiled:
	mkdir -p profiles/ncu
	ncu --set detailed \
		--nvtx \
		--nvtx-include "kernel/" \
		-fo profiles/ncu/naive-compiled python ./speed_test.py --name naive-compiled --n_hidden_states $(N_HIDDEN_STATES) --n_runs_benchmark $(N_RUNS_BENCHMARK)


nsight-profile-fused-triton:
	mkdir -p profiles/nsight
	nsys profile \
		--force-overwrite true \
		-o profiles/nsight/fused-triton python ./speed_test.py --name fused-triton --n_hidden_states $(N_HIDDEN_STATES) --n_runs_benchmark $(N_RUNS_BENCHMARK)

nsight-profile-naive-compiled:
	mkdir -p profiles/nsight
	nsys profile \
		--force-overwrite true \
		-o profiles/nsight/naive-compiled python ./speed_test.py --name naive-compiled --n_hidden_states $(N_HIDDEN_STATES) --n_runs_benchmark $(N_RUNS_BENCHMARK)

nsight-profile-flashinfer:
	mkdir -p profiles/nsight
	nsys profile \
		--force-overwrite true \
		-o profiles/nsight/flashinfer python ./speed_test.py --name flashinfer:sampling_from_logits --n_hidden_states $(N_HIDDEN_STATES) --n_runs_benchmark $(N_RUNS_BENCHMARK)

nsight-profile-naive-pt:
	mkdir -p profiles/nsight
	nsys profile \
		--force-overwrite true \
		-o profiles/nsight/naive-pt python ./speed_test.py --name naive-pt --n_hidden_states $(N_HIDDEN_STATES) --n_runs_benchmark $(N_RUNS_BENCHMARK)

ncu-export-fused-triton:
	mkdir -p profiles/ncu-txt
	$(NCU) --set full \
		--nvtx \
		--nvtx-include "kernel/" \
		--kernel-name fused_mm_sample_triton_kernel \
		$(PYTHON) ./speed_test.py --name fused-triton --n_hidden_states $(N_HIDDEN_STATES) --n_runs_benchmark $(N_RUNS_BENCHMARK) > profiles/ncu-txt/fused-triton.txt

ncu-export-naive-tl-matmul:
	mkdir -p profiles/ncu-txt
	$(NCU) --set full \
		--nvtx \
		--nvtx-include "kernel/" \
		--kernel-name matmul_kernel \
		$(PYTHON) ./speed_test.py --name naive-tl-matmul --n_hidden_states $(N_HIDDEN_STATES) --n_runs_benchmark $(N_RUNS_BENCHMARK) > profiles/ncu-txt/naive-tl-matmul.txt

ncu-export-naive-compiled:
	mkdir -p profiles/ncu-txt
	$(NCU) --set full \
		--nvtx \
		--nvtx-include "kernel/" \
		$(PYTHON) ./speed_test.py --name naive-compiled --n_hidden_states $(N_HIDDEN_STATES) --n_runs_benchmark $(N_RUNS_BENCHMARK) > profiles/ncu-txt/naive-compiled.txt

ncu-export-flashinfer:
	mkdir -p profiles/ncu-txt
	$(NCU) --set full \
		--nvtx \
		--nvtx-include "kernel/" \
		$(PYTHON) ./speed_test.py --name flashinfer:sampling_from_logits --n_hidden_states $(N_HIDDEN_STATES) --n_runs_benchmark $(N_RUNS_BENCHMARK) > profiles/ncu-txt/flashinfer.txt

ncu-export-all: ncu-export-fused-triton ncu-export-naive-compiled ncu-export-flashinfer ncu-export-naive-tl-matmul
	@echo "NCU reports exported to profiles/ncu-txt/"

triton-benchmark:
	@if [ -d profiles/triton-bench/ ]; then rm -rf profiles/triton-bench/; fi
	mkdir -p profiles/triton-bench
	python triton_benchmark.py > profiles/triton-bench/logs.txt
	python plot-triton-bench.py