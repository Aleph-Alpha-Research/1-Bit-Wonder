import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from llama3_qat.inference_kernel import QuantizedLinearTriton

logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)


# --------------------------------------------------------------------------- #
#  Model loading                                                              #
# --------------------------------------------------------------------------- #


def load_safetensors(model_path: Path) -> dict[str, Tensor]:
    """Load all safetensor files from directory."""
    from safetensors import safe_open

    state_dict = {}
    safetensor_files = sorted(model_path.glob("model*.safetensors"))

    if not safetensor_files:
        raise FileNotFoundError(f"No safetensor files found in {model_path}")

    for sf_file in safetensor_files:
        with safe_open(sf_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

    return state_dict


def load_model(
    model_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    """Load quantized model with Triton kernel support."""
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    model_path = Path(model_path)

    # Load configs
    with open(model_path / "config.json") as f:
        hf_config = json.load(f)

    is_quantized_model = (model_path / "quantization_config.json").is_file()
    if is_quantized_model:
        with open(model_path / "quantization_config.json") as f:
            quant_config = json.load(f)
    else:
        quant_config = None

    if is_quantized_model:
        bits = quant_config["bits"]
        block_size = quant_config["block_size"]

        print(
            f"Loading {hf_config['num_hidden_layers']}-layer model with {bits}-bit quantization..."
        )
    else:
        bits = None
        block_size = None
        print(f"Loading {hf_config['num_hidden_layers']}-layer model in Bfloat16...")

    # Create model architecture with meta tensors (no memory allocation)
    # This is MUCH faster than initializing random weights
    config = AutoConfig.from_pretrained(model_path)
    print("Creating model skeleton (meta tensors)...")
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)

    # Load state dict
    print("Loading weights from safetensors...")
    state_dict = load_safetensors(model_path)

    def replace_linear(module, prefix=""):
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name

            if isinstance(child, nn.Linear) and "lm_head" not in name:
                # Create quantized layer
                quant_layer = QuantizedLinearTriton(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    bits=bits,
                    block_size=block_size,
                )

                # Load quantized weights - use new naming convention
                weight_packed_key = f"{full_name}.weight_packed"
                scales_key = f"{full_name}.scales"
                centroids_key = f"{full_name}.centroids"

                quant_layer.weight_packed = state_dict[weight_packed_key]
                quant_layer.scales = state_dict[scales_key]
                quant_layer.centroids = state_dict[centroids_key]

                setattr(module, name, quant_layer)
            else:
                replace_linear(child, full_name)

    if is_quantized_model:
        # Replace Linear layers with QuantizedLinearTriton where we have quantized weights
        print("Replacing linear layers with quantized versions...")
        replace_linear(model)

        # Load remaining weights (embeddings, norms) - need to materialize meta tensors
        print("Loading embeddings and layer norms...")
    else:
        print("Loading model weights...")

    for name, param in list(model.named_parameters()):
        if param.device.type == "meta" and name in state_dict:
            # Materialize the meta tensor with actual data
            parts = name.split(".")
            module = model
            for part in parts[:-1]:
                module = getattr(module, part)
            setattr(module, parts[-1], nn.Parameter(state_dict[name].to(dtype)))

    # Move to device
    model = model.to(device)
    model.eval()

    if is_quantized_model:
        # Count layers
        num_quant = sum(
            1 for m in model.modules() if isinstance(m, QuantizedLinearTriton)
        )
        print(f"✅ Model loaded: {num_quant} {bits}-bit quantized layers on {device}")
    else:
        print(f"✅ Model loaded in {dtype} on {device}")

    return model


def load_from_hf(
    repo_id: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    cache_dir: str = None,
):
    """Load quantized model from HuggingFace Hub."""
    from huggingface_hub import snapshot_download

    print(f"Downloading model from {repo_id}...")
    local_path = snapshot_download(
        repo_id,
        cache_dir=cache_dir,
        allow_patterns=["*.safetensors", "*.json"],
    )

    return load_model(local_path, device=device, dtype=dtype)


# --------------------------------------------------------------------------- #
#  Generation                                                                 #
# --------------------------------------------------------------------------- #

# LLaMA 3 chat template (fallback if tokenizer doesn't have one)
# Matches Llama3Formatter with strip_content=True
LLAMA3_CHAT_TEMPLATE = """{%- set loop_messages = messages -%}
{%- for message in loop_messages -%}
{%- set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' -%}
{%- if loop.index0 == 0 -%}
{%- set content = bos_token + content -%}
{%- endif -%}
{{ content }}
{%- endfor -%}
{%- if add_generation_prompt -%}
{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif -%}"""


def format_chat_prompt(tokenizer, user_message: str, system_prompt: str = None) -> str:
    """Format a user message using the tokenizer's chat template.

    Matches Llama3Formatter behavior with strip_content=True.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})

    # Use chat template if available and set
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    elif hasattr(tokenizer, "apply_chat_template"):
        # Try with our fallback template
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                chat_template=LLAMA3_CHAT_TEMPLATE,
            )
        except Exception:
            pass

    # Final fallback - matches Llama3Formatter exactly (strip_content=True)
    # Format: begin_of_text + system_header + system.strip() + eot + user_header + user.strip() + eot + assistant_header
    if system_prompt:
        return (
            f"<|begin_of_text|>"
            f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt.strip()}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n{user_message.strip()}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    return (
        f"<|begin_of_text|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{user_message.strip()}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def interactive_chat(
    model,
    tokenizer,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    """Interactive chat loop with streaming output."""

    system_prompt = "You are a helpful AI assistant."

    eos_token_ids = (
        [tokenizer.eos_token_id]
        if isinstance(tokenizer.eos_token_id, int)
        else list(tokenizer.eos_token_id or [])
    )
    for stop_id in [128001, 128008, 128009]:
        if stop_id not in eos_token_ids and stop_id < tokenizer.vocab_size:
            eos_token_ids.append(stop_id)

    while True:
        try:
            user_input = input("\001\033[92m\002You:\001\033[0m\002 ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        if user_input.lower() == "clear":
            model.reset_cache()
            print("Context cleared.")
            continue

        # Format with chat template
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate with streaming
        print("\n")
        print("\033[94mAssistant:\033[0m ", end="", flush=True)

        # Build generation kwargs
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            "eos_token_id": eos_token_ids,
        }

        # Only add sampling parameters when do_sample=True
        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        else:
            gen_kwargs["do_sample"] = False

        outputs = model.generate(**inputs, **gen_kwargs)

        input_len = inputs["input_ids"].shape[1]

        # Decode only the generated part
        generated_ids = outputs[0, input_len:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        print(f"{response}")

        print("\n")


def main():
    parser = argparse.ArgumentParser(description="Inference with Triton LUT kernels")
    parser.add_argument("--model_path", help="Path to local HF model directory")
    parser.add_argument("--hf_repo", help="HuggingFace repo ID (e.g., user/model)")
    parser.add_argument(
        "--max_tokens", type=int, default=256, help="Max tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Sampling temperature"
    )
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--device", default="cuda", help="Device")

    args = parser.parse_args()

    if not args.model_path and not args.hf_repo:
        parser.error("Either --model_path or --hf_repo is required")

    # Load model
    if args.hf_repo:
        model = load_from_hf(args.hf_repo, device=args.device)
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.hf_repo)
    else:
        model = load_model(args.model_path, device=args.device)
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    print("\n" + "=" * 60)
    print("Chat with LLaMA")
    print("Type 'quit' or 'exit' to end, 'clear' to reset context")
    print("=" * 60 + "\n")

    interactive_chat(
        model,
        tokenizer,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )


if __name__ == "__main__":
    main()
