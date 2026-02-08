# LM Head Configurations Across LLMs

The fused matmul+sampling kernel operates on the LM head: a `[hidden_size, vocab_size]` weight matrix.
To make meaningful benchmark claims, we need configurations that represent real LLMs.

## Dense Models

| Model | vocab_size | hidden_size | LM Head (bf16) |
|-------|-----------|-------------|-----------------|
| Qwen2.5 0.5B | 151,936 | 896 | 260 MB |
| Qwen3 0.6B | 151,936 | 1,024 | 297 MB |
| Gemma 3 1B | 262,144 | 1,152 | 576 MB |
| Qwen2.5 1.5B | 151,936 | 1,536 | 445 MB |
| Llama 3.2 1B | 128,256 | 2,048 | 501 MB |
| Gemma 2 2B | 256,000 | 2,304 | 1.1 GB |
| Qwen3 4B | 151,936 | 2,560 | 742 MB |
| Gemma 3 4B | 262,208 | 2,560 | 1.3 GB |
| Llama 3.2 3B | 128,256 | 3,072 | 751 MB |
| Phi-3 Mini 3.8B | 32,064 | 3,072 | 188 MB |
| Phi-4 Mini 3.8B | 200,064 | 3,072 | 1.2 GB |
| Gemma 2 9B | 256,000 | 3,584 | 1.7 GB |
| Qwen2.5 7B | 152,064 | 3,584 | 1.0 GB |
| Mistral 7B | 32,000 | 4,096 | 250 MB |
| Llama 3 8B | 128,256 | 4,096 | 1.0 GB |
| Qwen3 8B | 151,936 | 4,096 | 1.2 GB |
| Gemma 3 12B | 262,208 | 3,840 | 1.9 GB |
| Gemma 2 27B | 256,000 | 4,608 | 2.2 GB |
| Qwen2.5 14B | 152,064 | 5,120 | 1.5 GB |
| Qwen2.5 32B | 152,064 | 5,120 | 1.5 GB |
| Qwen3 14B | 151,936 | 5,120 | 1.5 GB |
| Qwen3 32B | 151,936 | 5,120 | 1.5 GB |
| Phi-4 14B | 100,352 | 5,120 | 980 MB |
| Gemma 3 27B | 262,208 | 5,376 | 2.7 GB |
| Llama 3 70B | 128,256 | 8,192 | 2.0 GB |
| Qwen2.5 72B | 152,064 | 8,192 | 2.4 GB |
| Command R 35B | 256,000 | 8,192 | 4.0 GB |
| Command R+ 104B | 256,000 | 12,288 | 6.0 GB |
| Llama 3.1 405B | 128,256 | 16,384 | 4.0 GB |

## MoE Models

| Model | vocab_size | hidden_size | Experts | Active/Token | LM Head (bf16) |
|-------|-----------|-------------|---------|-------------|-----------------|
| Mixtral 8x7B | 32,000 | 4,096 | 8 | 2 | 250 MB |
| Mixtral 8x22B | 32,000 | 6,144 | 8 | 2 | 375 MB |
| Qwen3 30B-A3B | 151,936 | 2,048 | 128 | 8 | 594 MB |
| Qwen2 57B-A14B | 151,936 | 3,584 | 64 | 8 | 1.0 GB |
| Qwen3 235B-A22B | 151,936 | 4,096 | 128 | 8 | 1.2 GB |
| DeepSeek V2 | 102,400 | 5,120 | 160 | 6 | 1.0 GB |
| DeepSeek V3 | 129,280 | 7,168 | 256 | 8 | 1.8 GB |

## MoE and the LM Head

The LM head in MoE models is identical to dense models: a single `nn.Linear(hidden_size, vocab_size)`.
MoE only replaces the FFN/MLP blocks inside transformer layers with sparse expert routing.
Embeddings, attention, layer norms, and the LM head are all standard dense layers.

This means the fused matmul+sampling kernel works the same way regardless of whether the model
is dense or MoE. The key observation is that MoE models have smaller `hidden_size` relative to
their total parameter count (e.g., DeepSeek V3 at 671B total params has `hidden_size=7168`,
comparable to Llama 70B's 8192).

## Tensor Parallelism for the LM Head

When models are served with tensor parallelism (TP), the LM head weight is sharded along the
**vocab dimension**. Each GPU holds `[hidden_size, vocab_size / TP]` and computes partial logits.

Examples at TP=8:

| Model | Full vocab_size | Per-GPU vocab_size |
|-------|-----------------|--------------------|
| Llama 3 70B | 128,256 | 16,032 |
| Qwen2.5 72B | 152,064 | 19,008 |
| Gemma 3 27B | 262,208 | 32,776 |
| DeepSeek V3 | 129,280 | 16,160 |

This is relevant for benchmarking: in production multi-GPU serving, each GPU operates on a
sharded vocab size (16K-64K) rather than the full 128K-256K.

## Suggested Benchmark Sizes

Natural clustering of `(vocab_size, hidden_size)` across models:

| Size | vocab_size | hidden_size | Representative Models |
|------|-----------|-------------|-----------------------|
| **Small** | 128,256 | 4,096 | Llama 3 8B, Qwen3 8B, Mistral 7B, Qwen3-235B-A22B (MoE) |
| **Large** | 128,256 | 8,192 | Llama 3 70B, Qwen2.5 72B, DeepSeek V3 (~7K hidden) |

A third tier could capture large-vocabulary models, since vocab size significantly affects
memory access patterns:

| Size | vocab_size | hidden_size | Representative Models |
|------|-----------|-------------|-----------------------|
| **Large-vocab** | 256,000 | 4,608 | Gemma 2 27B, Gemma 3 27B, Command R |
