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

LM_EVAL := $(VLLM_DIR)/venv/bin/lm_eval
EVAL_TASK := gsm8k_cot_zeroshot
EVAL_MAX_MODEL_LEN := 32768
EVAL_DIR := $(RESULTS_DIR)/eval
JUDGE_DIRS := $(EVAL_DIR)/baseline $(EVAL_DIR)/baseline-fmms-params $(EVAL_DIR)/fmms-triton
# Qwen3-1.7B generation_config.json defaults + enough tokens for thinking chains:
EVAL_GEN_KWARGS := temperature=0.6 top_p=0.95 top_k=20 max_tokens=16384
# FMMS-compatible params (no top-k/top-p filtering):
EVAL_GEN_KWARGS_FMMS := temperature=0.6 top_p=1.0 top_k=-1 max_tokens=16384

# Usage: $(call run_eval,<env_prefix>,<output_subdir>,<gen_kwargs>)
# Starts vLLM in background, runs lm_eval, saves results, kills vLLM.
# If gen_kwargs (3) is omitted, defaults to EVAL_GEN_KWARGS.
define run_eval
	mkdir -p "$(EVAL_DIR)/$(2)"
	$(1) $(VLLM) serve $(MODEL) --max-model-len $(EVAL_MAX_MODEL_LEN) --no-enable-prefix-caching --uvicorn-log-level warning --reasoning-parser qwen3 \
		> "$(EVAL_DIR)/$(2)/vllm.log" 2>&1 & \
	VLLM_PID=$$!; \
	echo "Waiting for vLLM server (PID $$VLLM_PID)..."; \
	for i in $$(seq 1 120); do \
		curl -sf http://localhost:8000/health > /dev/null 2>&1 && break; \
		sleep 1; \
	done; \
	$(LM_EVAL) \
		--model local-chat-completions \
		--model_args model=$(MODEL),base_url=http://localhost:8000/v1/chat/completions,num_concurrent=32,max_retries=3,tokenized_requests=False \
		--apply_chat_template \
		--tasks $(EVAL_TASK) \
		--gen_kwargs $(if $(3),$(3),$(EVAL_GEN_KWARGS)) \
		--log_samples \
		--output_path "$(EVAL_DIR)/$(2)" \
		2>&1 | tee "$(EVAL_DIR)/$(2)/lm_eval.log"; \
	STATUS=$$?; \
	kill $$VLLM_PID 2>/dev/null; wait $$VLLM_PID 2>/dev/null; \
	exit $$STATUS
endef

.PHONY: all quick baseline fmms-triton fmms-flashinfer quick-baseline quick-fmms-triton quick-fmms-flashinfer eval-baseline eval-baseline-fmms-params eval-fmms-triton judge

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

eval-baseline:
	$(call run_eval,env,baseline)
	$(PYTHON) "$(RESULTS_DIR)/judge_eval.py" "$(EVAL_DIR)/baseline"

eval-baseline-fmms-params:
	$(call run_eval,env,baseline-fmms-params,$(EVAL_GEN_KWARGS_FMMS))
	$(PYTHON) "$(RESULTS_DIR)/judge_eval.py" "$(EVAL_DIR)/baseline-fmms-params"

eval-fmms-triton:
	$(call run_eval,env VLLM_USE_FMMS_SAMPLER=1 VLLM_FMMS_PROVIDER=fused-triton,fmms-triton,$(EVAL_GEN_KWARGS_FMMS))
	$(PYTHON) "$(RESULTS_DIR)/judge_eval.py" "$(EVAL_DIR)/fmms-triton"

judge:
	$(PYTHON) "$(RESULTS_DIR)/judge_eval.py" $(JUDGE_DIRS)
