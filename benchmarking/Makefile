profile-mem:
	python profile-mem.py
	echo "Converting memory snapshot to HTML"
	python _memory_viz.py trace_plot benchmarking/memory/mem-snapshot.pickle -o benchmarking/memory/mem-snapshot.html

download-html-conversion-tool:
	wget https://raw.githubusercontent.com/pytorch/pytorch/refs/heads/main/torch/cuda/_memory_viz.py


speed-test:
	python speed_test.py

NCU=$(shell which ncu)
NSYS=$(shell which nsys)
PYTHON=$(shell which python)
# Get the directory containing the Python executable (venv's bin directory)
VENV_BIN=$(shell dirname $(PYTHON))
N_HIDDEN_STATES=1
N_RUNS_BENCHMARK=1
# Can be set to 'sudo' or to nothing
SUDO=sudo
NSIGHT_RESULTS_DIR=profiles/nsight
NCU_RESULTS_DIR=profiles/ncu
PROTON_RESULTS_DIR=profiles/proton

# NCU profiling targets:
# --import-source yes: Embeds source code into the .ncu-rep file, allowing it to be
#   viewed on a different machine without path remapping. The source is embedded
#   regardless of where you open the report.
# --source-folders: Comma-separated search paths for source files to import into report.
#   For Triton kernels, this helps NCU find the Python source files.
ncu-profile-fused-triton:
	mkdir -p $(NCU_RESULTS_DIR)
	$(SUDO) $(NCU) --set full \
		--nvtx \
		--nvtx-include "kernel/" \
		--kernel-name fused_mm_sample_triton_kernel \
		--source-folders $$(pwd)/../src/fused_mm_sampling \
		--import-source yes \
		-fo $(NCU_RESULTS_DIR)/fused-triton $(PYTHON) ./speed_test.py --name fused-triton --n_hidden_states $(N_HIDDEN_STATES) --n_runs_benchmark $(N_RUNS_BENCHMARK)


ncu-profile-flashinfer:
	mkdir -p $(NCU_RESULTS_DIR)
	$(SUDO) env "PATH=$$PATH:$(VENV_BIN)" $(NCU) --set full \
		--nvtx \
		--nvtx-include "kernel/" \
		-fo $(NCU_RESULTS_DIR)/flashinfer $(PYTHON) ./speed_test.py --name flashinfer:sampling_from_logits --n_hidden_states $(N_HIDDEN_STATES) --n_runs_benchmark $(N_RUNS_BENCHMARK)


ncu-profile-naive-compiled:
	mkdir -p $(NCU_RESULTS_DIR)
	$(SUDO) $(NCU) --set full \
		--nvtx \
		--nvtx-include "kernel/" \
		-fo $(NCU_RESULTS_DIR)/naive-compiled $(PYTHON) ./speed_test.py --name naive-compiled --n_hidden_states $(N_HIDDEN_STATES) --n_runs_benchmark $(N_RUNS_BENCHMARK)


nsight-profile-fused-triton:
	mkdir -p $(NSIGHT_RESULTS_DIR)
	$(SUDO) $(NSYS) profile \
		--force-overwrite true \
		-o $(NSIGHT_RESULTS_DIR)/fused-triton $(PYTHON) ./speed_test.py --name fused-triton --n_hidden_states $(N_HIDDEN_STATES) --n_runs_benchmark $(N_RUNS_BENCHMARK)

nsight-profile-naive-compiled:
	mkdir -p $(NSIGHT_RESULTS_DIR)
	$(SUDO) $(NSYS) profile \
		--force-overwrite true \
		-o $(NSIGHT_RESULTS_DIR)/naive-compiled $(PYTHON) ./speed_test.py --name naive-compiled --n_hidden_states $(N_HIDDEN_STATES) --n_runs_benchmark $(N_RUNS_BENCHMARK)

nsight-profile-flashinfer:
	mkdir -p $(NSIGHT_RESULTS_DIR)
	$(SUDO) $(NSYS) profile \
		--force-overwrite true \
		-o $(NSIGHT_RESULTS_DIR)/flashinfer $(PYTHON) ./speed_test.py --name flashinfer:sampling_from_logits --n_hidden_states $(N_HIDDEN_STATES) --n_runs_benchmark $(N_RUNS_BENCHMARK)

nsight-profile-naive-pt:
	mkdir -p $(NSIGHT_RESULTS_DIR)
	$(SUDO) $(NSYS) profile \
		--force-overwrite true \
		-o $(NSIGHT_RESULTS_DIR)/naive-pt $(PYTHON) ./speed_test.py --name naive-pt --n_hidden_states $(N_HIDDEN_STATES) --n_runs_benchmark $(N_RUNS_BENCHMARK)

ncu-export-fused-triton:
	mkdir -p profiles/ncu-txt
	sudo $(NCU) --set full \
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
	PYTHONUNBUFFERED=1 $(PYTHON) triton_benchmark.py --tgt_dir profiles/triton-bench/ > profiles/triton-bench/logs.txt
	$(PYTHON) plot-triton-bench.py

clean-triton-autotunes:
	rm -rf ~/.triton/cache

nsight-comparison:
	make nsight-profile-fused-triton N_HIDDEN_STATES=256 NSIGHT_RESULTS_DIR=profiles/nsight-bsz256
	make nsight-profile-flashinfer N_HIDDEN_STATES=256 NSIGHT_RESULTS_DIR=profiles/nsight-bsz256
	make nsight-profile-fused-triton N_HIDDEN_STATES=1 NSIGHT_RESULTS_DIR=profiles/nsight-bsz1
	make nsight-profile-flashinfer N_HIDDEN_STATES=1 NSIGHT_RESULTS_DIR=profiles/nsight-bsz1

ncu-comparison:
	make ncu-profile-fused-triton N_HIDDEN_STATES=256 NCU_RESULTS_DIR=profiles/ncu-bsz256
	make ncu-profile-flashinfer N_HIDDEN_STATES=256 NCU_RESULTS_DIR=profiles/ncu-bsz256
	make ncu-profile-fused-triton N_HIDDEN_STATES=1 NCU_RESULTS_DIR=profiles/ncu-bsz1
	make ncu-profile-flashinfer N_HIDDEN_STATES=1 NCU_RESULTS_DIR=profiles/ncu-bsz1

proton-profile-fused-triton:
	rm -rf $(PROTON_RESULTS_DIR)
	mkdir -p $(PROTON_RESULTS_DIR)
	$(PYTHON) ./speed_test.py --n_hidden_states=1 --use_proton=True --name=fused-triton --n_runs_benchmark=10 --n_runs_warmup=0
	proton-viewer -m num_samples/%,time/s kernel.hatchet -i timing > $(PROTON_RESULTS_DIR)/kernel-by-line.txt
	mv kernel.hatchet $(PROTON_RESULTS_DIR)/kernel.hatchet