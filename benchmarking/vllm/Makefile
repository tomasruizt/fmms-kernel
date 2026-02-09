VLLM_DIR := $(HOME)/code/vllm
RESULTS_DIR := $(HOME)/code/fused-mm-sample/benchmarking/vllm
VLLM := $(VLLM_DIR)/venv/bin/vllm
PYTHON := $(HOME)/code/fused-mm-sample/.venv/bin/python
BENCH_PARAMS := $(RESULTS_DIR)/bench-params.json

MODEL := Qwen/Qwen3-1.7B
MODEL_SLUG := $(lastword $(subst /, ,$(MODEL)))
MAX_MODEL_LEN := 1024
MODEL_DIR := $(RESULTS_DIR)/$(MODEL_SLUG)

SERVE_FLAGS := --max-model-len $(MAX_MODEL_LEN) --no-enable-prefix-caching --uvicorn-log-level warning
BENCH_FLAGS := --dataset-name hf --dataset-path AI-MO/aimo-validation-aime --hf-output-len 256 --temperature 0.6 --top-k -1 --top-p 1.0 --backend openai-chat --endpoint /v1/chat/completions

AFTER_BENCH := curl -s -X POST http://localhost:8000/reset_prefix_cache && curl -s -X POST http://localhost:8000/reset_mm_cache && curl -s -X POST http://localhost:8000/reset_encoder_cache

SWEEP_FLAGS := --bench-params "$(BENCH_PARAMS)" --after-bench-cmd "$(AFTER_BENCH)" --num-runs 3 --show-stdout
QUICK_BENCH_PARAMS := $(RESULTS_DIR)/quick-bench-params.json
QUICK_SWEEP_FLAGS := --bench-params "$(QUICK_BENCH_PARAMS)" --after-bench-cmd "$(AFTER_BENCH)" --num-runs 1 --show-stdout

# After each sweep, move the log into the latest timestamped results directory.
define move_log
	mv "$(1).tmp" "$$(ls -td $(2)/*/ | head -1)/sweep.log"
endef

define collect
	$(PYTHON) "$(RESULTS_DIR)/collect_results.py" "$(MODEL_DIR)" | tee "$(MODEL_DIR)/results.txt"
endef

.PHONY: all quick baseline fmms-triton fmms-flashinfer quick-baseline quick-fmms-triton quick-fmms-flashinfer

$(MODEL_DIR):
	mkdir -p "$@"

all: | $(MODEL_DIR)
	$(MAKE) baseline fmms-triton fmms-flashinfer
	$(call collect)

quick: | $(MODEL_DIR)
	$(MAKE) quick-baseline quick-fmms-triton quick-fmms-flashinfer
	$(call collect)

baseline:
	$(VLLM) bench sweep serve \
		--serve-cmd "$(VLLM) serve $(MODEL) $(SERVE_FLAGS)" \
		--bench-cmd "$(VLLM) bench serve $(BENCH_FLAGS)" \
		$(SWEEP_FLAGS) \
		-o "$(MODEL_DIR)/baseline" \
		2>&1 | tee "$(MODEL_DIR)/baseline.tmp"
	$(call move_log,$(MODEL_DIR)/baseline,$(MODEL_DIR)/baseline)

fmms-triton:
	$(VLLM) bench sweep serve \
		--serve-cmd "env VLLM_USE_FMMS_SAMPLER=1 VLLM_FMMS_PROVIDER=fused-triton $(VLLM) serve $(MODEL) $(SERVE_FLAGS)" \
		--bench-cmd "$(VLLM) bench serve $(BENCH_FLAGS)" \
		$(SWEEP_FLAGS) \
		-o "$(MODEL_DIR)/fmms-triton" \
		2>&1 | tee "$(MODEL_DIR)/fmms-triton.tmp"
	$(call move_log,$(MODEL_DIR)/fmms-triton,$(MODEL_DIR)/fmms-triton)

fmms-flashinfer:
	$(VLLM) bench sweep serve \
		--serve-cmd "env VLLM_USE_FMMS_SAMPLER=1 VLLM_FMMS_PROVIDER=flashinfer:top_k_top_p_sampling_from_logits $(VLLM) serve $(MODEL) $(SERVE_FLAGS)" \
		--bench-cmd "$(VLLM) bench serve $(BENCH_FLAGS)" \
		$(SWEEP_FLAGS) \
		-o "$(MODEL_DIR)/fmms-flashinfer" \
		2>&1 | tee "$(MODEL_DIR)/fmms-flashinfer.tmp"
	$(call move_log,$(MODEL_DIR)/fmms-flashinfer,$(MODEL_DIR)/fmms-flashinfer)

quick-baseline:
	$(VLLM) bench sweep serve \
		--serve-cmd "$(VLLM) serve $(MODEL) $(SERVE_FLAGS) --enforce-eager" \
		--bench-cmd "$(VLLM) bench serve $(BENCH_FLAGS)" \
		$(QUICK_SWEEP_FLAGS) \
		-o "$(MODEL_DIR)/baseline" \
		2>&1 | tee "$(MODEL_DIR)/baseline.tmp"
	$(call move_log,$(MODEL_DIR)/baseline,$(MODEL_DIR)/baseline)

quick-fmms-triton:
	$(VLLM) bench sweep serve \
		--serve-cmd "env VLLM_USE_FMMS_SAMPLER=1 VLLM_FMMS_PROVIDER=fused-triton $(VLLM) serve $(MODEL) $(SERVE_FLAGS) --enforce-eager" \
		--bench-cmd "$(VLLM) bench serve $(BENCH_FLAGS)" \
		$(QUICK_SWEEP_FLAGS) \
		-o "$(MODEL_DIR)/fmms-triton" \
		2>&1 | tee "$(MODEL_DIR)/fmms-triton.tmp"
	$(call move_log,$(MODEL_DIR)/fmms-triton,$(MODEL_DIR)/fmms-triton)

quick-fmms-flashinfer:
	$(VLLM) bench sweep serve \
		--serve-cmd "env VLLM_USE_FMMS_SAMPLER=1 VLLM_FMMS_PROVIDER=flashinfer:top_k_top_p_sampling_from_logits $(VLLM) serve $(MODEL) $(SERVE_FLAGS) --enforce-eager" \
		--bench-cmd "$(VLLM) bench serve $(BENCH_FLAGS)" \
		$(QUICK_SWEEP_FLAGS) \
		-o "$(MODEL_DIR)/fmms-flashinfer" \
		2>&1 | tee "$(MODEL_DIR)/fmms-flashinfer.tmp"
	$(call move_log,$(MODEL_DIR)/fmms-flashinfer,$(MODEL_DIR)/fmms-flashinfer)
