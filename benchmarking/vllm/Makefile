RESULTS_DIR := $(CURDIR)
VLLM := $(shell which vllm)
PYTHON := $(shell which python)
BENCH_PARAMS := $(RESULTS_DIR)/bench-params.json

MODEL := openai/gpt-oss-120b
MODEL_SLUG := $(lastword $(subst /, ,$(MODEL)))
MAX_MODEL_LEN := 1024
MODEL_DIR := $(RESULTS_DIR)/$(MODEL_SLUG)

SERVE_FLAGS := --max-model-len $(MAX_MODEL_LEN) --no-enable-prefix-caching --uvicorn-log-level warning
BENCH_FLAGS := --dataset-name hf --dataset-path AI-MO/aimo-validation-aime --hf-output-len 256 --temperature 0.6 --top-k -1 --top-p 1.0 --backend openai-chat --endpoint /v1/chat/completions

AFTER_BENCH := curl -s -X POST http://localhost:8000/reset_prefix_cache && curl -s -X POST http://localhost:8000/reset_mm_cache && curl -s -X POST http://localhost:8000/reset_encoder_cache

SWEEP_FLAGS := --bench-params "$(BENCH_PARAMS)" --after-bench-cmd "$(AFTER_BENCH)" --num-runs 5 --show-stdout
QUICK_BENCH_PARAMS := $(RESULTS_DIR)/quick-bench-params.json
QUICK_SWEEP_FLAGS := --bench-params "$(QUICK_BENCH_PARAMS)" --after-bench-cmd "$(AFTER_BENCH)" --num-runs 1 --show-stdout

# Move the .tmp log into the latest timestamped results directory.
# Called with ; after the sweep so it runs even if the sweep exits non-zero.
define move_log
mv "$(1).tmp" "$$(ls -td $(2)/*/ | head -1)/sweep.log" 2>/dev/null || true
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

# --- nsys profiling ---
# Profiles vLLM decode steps under nsys. Uses cudaProfilerApi capture range so
# nsys only records between /start_profile and /stop_profile, skipping model
# load, warmup, and CUDA graph capture.
NSYS := $(or $(shell which nsys 2>/dev/null),$(wildcard /opt/nvidia/nsight-systems/*/bin/nsys))
GPU := H100
NSYS_DIR := $(RESULTS_DIR)/profiles/nsight/$(GPU)/$(MODEL_SLUG)
NSYS_MAX_TOKENS := 10
NSYS_PROMPT := What is 2+2?
NSYS_MODEL := openai/gpt-oss-20b

# Usage: $(call run_nsys,<env_prefix>,<output_name>)
# 1. Start vLLM under nsys with cudaProfilerApi capture.
# 2. Wait for /health.
# 3. Send a warmup request (fills CUDA graphs, JIT caches).
# 4. /start_profile  →  single request (NSYS_MAX_TOKENS tokens)  →  /stop_profile.
# 5. Kill the server. Profile saved to NSYS_DIR/<output_name>.nsys-rep.
define run_nsys
	mkdir -p "$(NSYS_DIR)"
	$(1) VLLM_NVTX_SCOPES_FOR_PROFILING=1 \
	    sudo -E setsid $(NSYS) profile \
	        --capture-range=cudaProfilerApi --capture-range-end=stop \
	        --force-overwrite true \
	        -o "$(NSYS_DIR)/$(2)" \
	        $(VLLM) serve $(NSYS_MODEL) $(SERVE_FLAGS) \
	            --gpu-memory-utilization 0.90 \
	            --profiler-config.profiler=cuda \
	        > "$(NSYS_DIR)/$(2)-server.log" 2>&1 & \
	NSYS_PID=$$!; \
	echo "Waiting for vLLM under nsys (PID $$NSYS_PID)..."; \
	for i in $$(seq 1 180); do \
	    curl -sf http://localhost:8000/health > /dev/null 2>&1 && break; \
	    sleep 1; \
	done; \
	echo "Warmup request..."; \
	curl -s http://localhost:8000/v1/chat/completions \
	    -H "Content-Type: application/json" \
	    -d '{"model":"$(NSYS_MODEL)","messages":[{"role":"user","content":"$(NSYS_PROMPT)"}],"max_tokens":$(NSYS_MAX_TOKENS),"temperature":0.6,"top_p":1.0,"top_k":-1}' \
	    > /dev/null; \
	echo "Starting profile..."; \
	curl -s -X POST http://localhost:8000/start_profile; \
	curl -s http://localhost:8000/v1/chat/completions \
	    -H "Content-Type: application/json" \
	    -d '{"model":"$(NSYS_MODEL)","messages":[{"role":"user","content":"$(NSYS_PROMPT)"}],"max_tokens":$(NSYS_MAX_TOKENS),"temperature":0.6,"top_p":1.0,"top_k":-1}' \
	    | tee "$(NSYS_DIR)/$(2)-response.json"; \
	echo; \
	curl -s -X POST http://localhost:8000/stop_profile; \
	echo "Stopping server..."; \
	sleep 2; \
	sudo kill $$NSYS_PID 2>/dev/null; sudo pkill -f "bin/vllm serve" 2>/dev/null; wait $$NSYS_PID 2>/dev/null; \
	echo "Waiting for GPU memory to free..."; \
	for i in $$(seq 1 30); do \
	    FREE=$$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1); \
	    TOTAL=$$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1); \
	    [ "$$FREE" -gt "$$((TOTAL - 1000))" ] 2>/dev/null && break; \
	    sleep 1; \
	done; \
	echo "Profile saved to $(NSYS_DIR)/$(2).nsys-rep"
endef

nsys-baseline:
	$(call run_nsys,env,baseline)

nsys-fmms-triton:
	$(call run_nsys,env VLLM_USE_FMMS_SAMPLER=1 VLLM_FMMS_PROVIDER=fused-triton,fmms-triton)

nsys-fmms-flashinfer:
	$(call run_nsys,env VLLM_USE_FMMS_SAMPLER=1 VLLM_FMMS_PROVIDER=flashinfer:top_k_top_p_sampling_from_logits,fmms-flashinfer)

nsys-all:
	$(MAKE) nsys-baseline nsys-fmms-triton

.PHONY: all quick baseline fmms-triton fmms-flashinfer quick-baseline quick-fmms-triton quick-fmms-flashinfer eval-baseline eval-baseline-fmms-params eval-fmms-triton judge nsys-baseline nsys-fmms-triton nsys-fmms-flashinfer nsys-all engine-stats

$(MODEL_DIR):
	mkdir -p "$@"

all: | $(MODEL_DIR)
	$(MAKE) baseline fmms-triton
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
		2>&1 | tee "$(MODEL_DIR)/baseline.tmp"; \
	$(call move_log,$(MODEL_DIR)/baseline,$(MODEL_DIR)/baseline)

fmms-triton:
	$(VLLM) bench sweep serve \
		--serve-cmd "env VLLM_USE_FMMS_SAMPLER=1 VLLM_FMMS_PROVIDER=fused-triton $(VLLM) serve $(MODEL) $(SERVE_FLAGS)" \
		--bench-cmd "$(VLLM) bench serve $(BENCH_FLAGS)" \
		$(SWEEP_FLAGS) \
		-o "$(MODEL_DIR)/fmms-triton" \
		2>&1 | tee "$(MODEL_DIR)/fmms-triton.tmp"; \
	$(call move_log,$(MODEL_DIR)/fmms-triton,$(MODEL_DIR)/fmms-triton)

fmms-flashinfer:
	$(VLLM) bench sweep serve \
		--serve-cmd "env VLLM_USE_FMMS_SAMPLER=1 VLLM_FMMS_PROVIDER=flashinfer:top_k_top_p_sampling_from_logits $(VLLM) serve $(MODEL) $(SERVE_FLAGS)" \
		--bench-cmd "$(VLLM) bench serve $(BENCH_FLAGS)" \
		$(SWEEP_FLAGS) \
		-o "$(MODEL_DIR)/fmms-flashinfer" \
		2>&1 | tee "$(MODEL_DIR)/fmms-flashinfer.tmp"; \
	$(call move_log,$(MODEL_DIR)/fmms-flashinfer,$(MODEL_DIR)/fmms-flashinfer)

quick-baseline:
	$(VLLM) bench sweep serve \
		--serve-cmd "$(VLLM) serve $(MODEL) $(SERVE_FLAGS) --enforce-eager" \
		--bench-cmd "$(VLLM) bench serve $(BENCH_FLAGS)" \
		$(QUICK_SWEEP_FLAGS) \
		-o "$(MODEL_DIR)/baseline" \
		2>&1 | tee "$(MODEL_DIR)/baseline.tmp"; \
	$(call move_log,$(MODEL_DIR)/baseline,$(MODEL_DIR)/baseline)

quick-fmms-triton:
	$(VLLM) bench sweep serve \
		--serve-cmd "env VLLM_USE_FMMS_SAMPLER=1 VLLM_FMMS_PROVIDER=fused-triton $(VLLM) serve $(MODEL) $(SERVE_FLAGS) --enforce-eager" \
		--bench-cmd "$(VLLM) bench serve $(BENCH_FLAGS)" \
		$(QUICK_SWEEP_FLAGS) \
		-o "$(MODEL_DIR)/fmms-triton" \
		2>&1 | tee "$(MODEL_DIR)/fmms-triton.tmp"; \
	$(call move_log,$(MODEL_DIR)/fmms-triton,$(MODEL_DIR)/fmms-triton)

quick-fmms-flashinfer:
	$(VLLM) bench sweep serve \
		--serve-cmd "env VLLM_USE_FMMS_SAMPLER=1 VLLM_FMMS_PROVIDER=flashinfer:top_k_top_p_sampling_from_logits $(VLLM) serve $(MODEL) $(SERVE_FLAGS) --enforce-eager" \
		--bench-cmd "$(VLLM) bench serve $(BENCH_FLAGS)" \
		$(QUICK_SWEEP_FLAGS) \
		-o "$(MODEL_DIR)/fmms-flashinfer" \
		2>&1 | tee "$(MODEL_DIR)/fmms-flashinfer.tmp"; \
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

engine-stats:
	@for log in $(MODEL_DIR)/*/; do \
		latest=$$(ls -td "$$log"*/ 2>/dev/null | head -1); \
		if [ -f "$$latest/sweep.log" ]; then \
			echo "=== $$(basename $$log) ==="; \
			$(PYTHON) "$(RESULTS_DIR)/parse_engine_stats.py" "$$latest/sweep.log" --by-concurrency; \
			echo; \
		fi; \
	done
