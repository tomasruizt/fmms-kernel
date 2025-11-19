import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationMixin


def to_chat_text(tokenizer: AutoTokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    return text


def main():
    model_name = "Qwen/Qwen3-0.6B"

    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model: GenerationMixin = AutoModelForCausalLM.from_pretrained(
        model_name, dtype="bfloat16", device_map="auto"
    )

    prompts = [
        "Give me a short introduction to large language model.",
        "What is the capital of France?",
    ]
    texts = [to_chat_text(tokenizer, prompt) for prompt in prompts]
    tokenizer.padding_side = "left"
    model_inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)

    with torch.inference_mode():
        outputs = model.model.forward(**model_inputs)

    os.makedirs("qwen3-0.6b", exist_ok=True)
    hidden_states = outputs.last_hidden_state[:, -1, :].T  # [D, n_hidden_states]
    torch.save(hidden_states, "qwen3-0.6b/hidden_states.pt")
    weights = model.lm_head.weight.data  # [V, D]
    torch.save(weights, "qwen3-0.6b/weights.pt")


if __name__ == "__main__":
    main()

# for response in tokenizer.batch_decode(generated_ids):
#     print(response)
#     print("-" * 100)

#     samples = sample(
#         weights=model.lm_head.weight,
#         hidden_states=hidden_states,
#         num_samples=1,
#         temperature=0.7,
#     )
#     model_inputs["input_ids"] = torch.cat([model_inputs["input_ids"], samples], dim=-1)
#     model_inputs = model._update_model_kwargs_for_generation(
#         outputs=outputs,
#         model_kwargs=model_inputs,
#         is_encoder_decoder=False
#     )
