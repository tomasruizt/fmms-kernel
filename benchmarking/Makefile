profile-mem:
	python profile-mem.py
	echo "Converting memory snapshot to HTML"
	python _memory_viz.py trace_plot benchmarking/memory/mem-snapshot.pickle -o benchmarking/memory/mem-snapshot.html

download-html-conversion-tool:
	wget https://raw.githubusercontent.com/pytorch/pytorch/refs/heads/main/torch/cuda/_memory_viz.py


speed-test:
	python speed_test.py

# NCU profiling targets:
# --import-source yes: Embeds source code into the .ncu-rep file, allowing it to be
#   viewed on a different machine without path remapping. The source is embedded
#   regardless of where you open the report.
# --source-folders: Comma-separated search paths for source files to import into report.
#   For Triton kernels, this helps NCU find the Python source files.
ncu-profile-fused-triton:
	mkdir -p profiles/ncu
	ncu --set detailed \
		--nvtx \
		--nvtx-include "kernel/" \
		--source-folders $$(pwd)/../src/fused_mm_sampling \
		--import-source yes \
		-fo profiles/ncu/fused-triton python ./speed_test.py --name fused-triton --n_hidden_states 1


ncu-profile-flashinfer:
	mkdir -p profiles/ncu
	ncu --set detailed \
		--nvtx \
		--nvtx-include "kernel/" \
		-fo profiles/ncu/flashinfer python ./speed_test.py --name flashinfer:sampling_from_logits --n_hidden_states 1


ncu-profile-naive-compiled:
	mkdir -p profiles/ncu
	ncu --set detailed \
		--nvtx \
		--nvtx-include "kernel/" \
		-fo profiles/ncu/naive-compiled python ./speed_test.py --name naive-compiled --n_hidden_states 1


nsight-profile-fused-triton:
	mkdir -p profiles/nsight
	nsys profile \
		--force-overwrite true \
		-o profiles/nsight/fused-triton python ./speed_test.py --name fused-triton

nsight-profile-naive-compiled:
	mkdir -p profiles/nsight
	nsys profile \
		--force-overwrite true \
		-o profiles/nsight/naive-compiled python ./speed_test.py --name naive-compiled

nsight-profile-naive-pt:
	mkdir -p profiles/nsight
	nsys profile \
		--force-overwrite true \
		-o profiles/nsight/naive-pt python ./speed_test.py --name naive-pt

ncu-export-fused-triton:
	mkdir -p profiles/ncu-txt
	ncu --set full \
		--launch-skip 10 \
		--kernel-name fused_mm_sample_triton_kernel \
		python ./speed_test.py --name fused-triton --n_hidden_states 1 > profiles/ncu-txt/fused-triton.txt

ncu-export-naive-compiled:
	mkdir -p profiles/ncu-txt
	ncu --nvtx \
		--nvtx-include "kernel/" \
		--launch-skip 10 \
		python ./speed_test.py --name naive-compiled > profiles/ncu-txt/naive-compiled.txt

ncu-export-all: ncu-export-fused-triton ncu-export-naive-compiled
	@echo "NCU reports exported to profiles/ncu-txt/"

triton-benchmark:
	@if [ -d profiles/triton-bench/ ]; then trash-put profiles/triton-bench/; fi
	mkdir -p profiles/triton-bench
	python triton_benchmark.py > profiles/triton-bench/logs.txt
	python plot-triton-bench.py