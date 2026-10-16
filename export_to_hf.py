from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import save_file
from torch import Tensor
from tqdm import tqdm

# Add repo root to path for imports
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from llama3_qat.experiment.config import Quantization
from llama3_qat.experiment import llama3_qat_configs
from torchtitan.models.llama3.model.args import TransformerModelArgs


LLAMA3_CHAT_TEMPLATE = """{%- set loop_messages = messages -%}
{%- for message in loop_messages -%}
{%- set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' -%}
{%- if loop.index0 == 0 -%}
{%- set content = bos_token + content -%}
{%- endif -%}
{{ content }}
{%- endfor -%}
{%- if add_generation_prompt -%}
{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}
{%- endif -%}"""


# --------------------------------------------------------------------------- #
#  HuggingFace Config Generation                                              #
# --------------------------------------------------------------------------- #


def create_hf_config(
    model_args: TransformerModelArgs,
    vocab_size: int = 128256,
) -> dict[str, Any]:
    """Create HuggingFace config.json from TransformerModelArgs."""
    # Calculate intermediate size (MLP ffn_dim)
    ffn_dim_multiplier = model_args.ffn_dim_multiplier or 1.0
    multiple_of = model_args.multiple_of
    hidden_dim = int(2 * (4 * model_args.dim) / 3)
    hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

    # Map RoPE scaling args to HuggingFace format (LLaMA 3.1 style)
    rope_args = model_args.rope_scaling_args
    rope_scaling = {
        "rope_type": "llama3",
        "factor": rope_args.scaling_factor,
        "low_freq_factor": rope_args.low_freq_factor,
        "high_freq_factor": rope_args.high_freq_factor,
        "original_max_position_embeddings": rope_args.original_max_position_embeddings,
    }

    return {
        "architectures": ["LlamaForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 128000,
        "eos_token_id": 128009,
        "hidden_act": "silu",
        "hidden_size": model_args.dim,
        "initializer_range": 0.02,
        "intermediate_size": hidden_dim,
        "max_position_embeddings": model_args.max_seq_len,
        "mlp_bias": False,
        "model_type": "llama",
        "num_attention_heads": model_args.n_heads,
        "num_hidden_layers": model_args.n_layers,
        "num_key_value_heads": model_args.n_kv_heads or model_args.n_heads,
        "pretraining_tp": 1,
        "rms_norm_eps": model_args.norm_eps,
        "rope_scaling": rope_scaling,
        "rope_theta": model_args.rope_theta,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "use_cache": True,
        "vocab_size": vocab_size,
    }


def create_generation_config() -> dict[str, Any]:
    """Create generation_config.json for LLaMA 3 model."""
    return {
        "bos_token_id": 128000,
        "eos_token_id": [128001, 128008, 128009],
        "pad_token_id": 128004,
        "max_length": 8192,
        "do_sample": True,
        "temperature": 0.0,
        "top_p": 0.9,
    }


# --------------------------------------------------------------------------- #
#  Name Mapping                                                               #
# --------------------------------------------------------------------------- #


def torchtitan_to_hf_name(tt_name: str) -> str | None:
    """Convert torchtitan checkpoint key to HuggingFace format.

    Args:
        tt_name: Torchtitan parameter name

    Returns:
        HuggingFace parameter name, or None if should be skipped
    """
    # Skip quantizer buffers and internal state
    if any(x in tt_name for x in [".quantizer.", "._qat_enabled", "._is_initialized"]):
        return None

    # Embedding layer
    if tt_name == "tok_embeddings.weight":
        return "model.embed_tokens.weight"

    # Output norm
    if tt_name == "norm.weight":
        return "model.norm.weight"

    # Output projection
    if tt_name == "output.weight":
        return "lm_head.weight"

    # Transformer layers
    if tt_name.startswith("layers."):
        parts = tt_name.split(".")
        layer_idx = parts[1]
        rest = ".".join(parts[2:])

        mapping = {
            "attention.wq.weight": f"model.layers.{layer_idx}.self_attn.q_proj.weight",
            "attention.wk.weight": f"model.layers.{layer_idx}.self_attn.k_proj.weight",
            "attention.wv.weight": f"model.layers.{layer_idx}.self_attn.v_proj.weight",
            "attention.wo.weight": f"model.layers.{layer_idx}.self_attn.o_proj.weight",
            "feed_forward.w1.weight": f"model.layers.{layer_idx}.mlp.gate_proj.weight",
            "feed_forward.w2.weight": f"model.layers.{layer_idx}.mlp.down_proj.weight",
            "feed_forward.w3.weight": f"model.layers.{layer_idx}.mlp.up_proj.weight",
            "attention_norm.weight": f"model.layers.{layer_idx}.input_layernorm.weight",
            "ffn_norm.weight": f"model.layers.{layer_idx}.post_attention_layernorm.weight",
        }

        return mapping.get(rest)

    return None


# --------------------------------------------------------------------------- #
#  Checkpoint Loading                                                         #
# --------------------------------------------------------------------------- #


def load_dcp_checkpoint(
    checkpoint_dir: str | Path,
    device: str = "cpu",
) -> dict[str, Tensor]:
    """Load DCP checkpoint into flat state dict.

    Supports:
    1. Consolidated .pt file (fastest, if available)
    2. DCP shards (slower, especially over NFS)

    Args:
        checkpoint_dir: Path to DCP checkpoint directory
        device: Device to load tensors to

    Returns:
        Flat state dict with all parameters
    """
    checkpoint_path = Path(checkpoint_dir)

    print(f"Loading DCP checkpoint from {checkpoint_dir}...")
    start_time = time.time()

    # Check for consolidated checkpoint first (much faster)
    consolidated_path = checkpoint_path / "_consolidated.pt"
    if consolidated_path.exists():
        print("  Found consolidated checkpoint, loading directly...")
        state_dict = torch.load(
            consolidated_path, map_location=device, weights_only=False
        )
        elapsed = time.time() - start_time
        print(f"  Loaded {len(state_dict)} tensors in {elapsed:.1f}s")
        return state_dict

    # Fall back to DCP loading
    print("  Loading from DCP shards (this may take 5-15 minutes for large models)...")
    reader = dcp.FileSystemReader(str(checkpoint_path))
    metadata = reader.read_metadata()

    # Build state dict with empty tensors
    state_dict = {}
    for key, meta in metadata.state_dict_metadata.items():
        if hasattr(meta, "size"):
            state_dict[key] = torch.empty(meta.size, device=device)

    print(f"  Created {len(state_dict)} empty tensors, loading weights...")

    # Load checkpoint
    dcp.load(state_dict, storage_reader=reader)

    elapsed = time.time() - start_time
    print(f"  Loaded {len(state_dict)} tensors in {elapsed:.1f}s")

    return state_dict


# --------------------------------------------------------------------------- #
#  Quantization Utilities                                                     #
# --------------------------------------------------------------------------- #


def pack_indices_to_uint8(indices: Tensor, bits: int) -> Tensor:
    """Pack quantization indices into uint8 bytes.

    Uses cartesian product approach for building the packed representation.

    Args:
        indices: Integer indices [0, 2**bits) with shape [out_features, in_features]
        bits: Bits per index (1, 2, 4, or 8)

    Returns:
        Packed uint8 tensor [out_features, in_features // elements_per_byte]
    """
    elements_per_byte = 8 // bits
    out_features, in_features = indices.shape

    indices = indices.reshape(out_features, -1, elements_per_byte)
    packed = torch.zeros(
        out_features,
        in_features // elements_per_byte,
        dtype=torch.uint8,
        device=indices.device,
    )

    for i in range(elements_per_byte):
        shift = (elements_per_byte - 1 - i) * bits
        packed |= indices[:, :, i].to(torch.uint8) << shift

    return packed


def quantize_weight(
    weight: Tensor,
    centroids: Tensor,
    bits: int,
    block_size: int = 64,
    relative_scale: str = "absmean",
) -> tuple[Tensor, Tensor, Tensor]:
    """Quantize weight using given centroids.

    Args:
        weight: Weight tensor [out_features, in_features]
        centroids: Learned centroids [num_centroids]
        bits: Quantization bits (1, 2, 4, 8)
        block_size: Block size for scaling (block_dim_2)
        relative_scale: Scale computation method

    Returns:
        Tuple of (packed_weights, scales, sorted_centroids)
    """
    out_features, in_features = weight.shape
    num_blocks = in_features // block_size

    # Reshape for block processing [out, num_blocks, block_size]
    w_blocked = weight.view(out_features, num_blocks, block_size)

    # Compute scales per block
    eps = 1e-5
    if relative_scale == "absmax":
        scales = w_blocked.abs().amax(dim=-1).clamp(min=eps)
    elif relative_scale == "absmean":
        scales = w_blocked.abs().mean(dim=-1).clamp(min=eps)
    elif relative_scale == "rms":
        scales = w_blocked.pow(2).mean(dim=-1).sqrt().clamp(min=eps)
    else:  # "none"
        scales = torch.ones(
            out_features, num_blocks, device=weight.device, dtype=weight.dtype
        )

    # Normalize weights
    w_norm = w_blocked / scales.unsqueeze(-1)
    w_flat = w_norm.reshape(out_features, -1)

    # Sort centroids and find nearest using bucketize
    centroids_sorted, _ = centroids.sort()
    boundaries = (centroids_sorted[:-1] + centroids_sorted[1:]) / 2
    indices = torch.bucketize(w_flat.float(), boundaries.float())

    # Pack to bytes
    packed = pack_indices_to_uint8(indices, bits)

    return packed, scales, centroids_sorted


def convert_interleaved_to_split(weight: torch.Tensor, n_heads: int) -> torch.Tensor:
    """
    Convert Q/K projection weights from interleaved to split-halves RoPE layout.

    Args:
        weight: Shape (n_heads * head_dim, hidden_dim) for Q/K projections
        n_heads: Number of attention heads (use n_kv_heads for K)

    Returns:
        Permuted weight matrix compatible with split-halves RoPE
    """
    out_features, in_features = weight.shape
    head_dim = out_features // n_heads

    # Reshape to (n_heads, head_dim, in_features)
    w = weight.view(n_heads, head_dim, in_features)

    # Current layout per head: [re0, im0, re1, im1, ...]
    # Split into pairs
    w = w.view(n_heads, head_dim // 2, 2, in_features)

    # Reorder: gather all real parts, then all imaginary parts
    # From (n_heads, head_dim//2, 2, in) -> (n_heads, 2, head_dim//2, in)
    w = w.permute(0, 2, 1, 3)

    # Flatten back: (n_heads, head_dim, in_features)
    w = w.reshape(n_heads, head_dim, in_features)

    # Final shape: (n_heads * head_dim, in_features)
    return w.view(out_features, in_features)


# --------------------------------------------------------------------------- #
#  Export Pipeline                                                            #
# --------------------------------------------------------------------------- #


def export_model(
    checkpoint_dir: str | Path,
    output_dir: str | Path,
    model_args: TransformerModelArgs,
    quant_config: Quantization | None,
    tokenizer_path: str | Path | None,
    max_shard_size_gb: float = 5.0,
) -> Path:
    """Export quantized model to HuggingFace-compatible format.

    Args:
        checkpoint_dir: Path to DCP checkpoint
        output_dir: Output directory
        model_args: Model configuration from TransformerModelArgs
        quant_config: Quantization configuration
        tokenizer_path: Path to tokenizer (local or HF model id)
        max_shard_size_gb: Maximum shard size in GB for splitting large models

    Returns:
        Path to output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Exporting model")
    print(f"  Model: dim={model_args.dim}, n_layers={model_args.n_layers}")
    if quant_config is not None:
        print(
            f"  Quantization: {quant_config.target_bit_width}-bit, block_size={quant_config.target_bit_width or 64}"
        )
        print(f"  Relative scale: {quant_config.relative_scale}")
    else:
        print(f"  Not applying Quantization")
    print(f"  Output: {output_dir}")
    print(f"{'=' * 60}\n")

    # Load checkpoint
    state_dict = load_dcp_checkpoint(checkpoint_dir)

    # Collect per-layer centroids
    centroids_map = {}

    for key, tensor in state_dict.items():
        if ".quantizer.centroids" in key:
            layer_path = key.replace(".quantizer.centroids", "")
            centroids_map[layer_path] = tensor.float()

    print(f"Found {len(centroids_map)} per-layer centroids")
    if len(centroids_map) > 0:
        sample_centroids = list(centroids_map.values())[0]
        print(f"  Sample centroids: {sample_centroids.sort()[0].tolist()[:5]}...")

    # Process weights - all go into single state dict
    state: dict[str, Tensor] = {}

    print("\nProcessing weights...")
    for tt_name, tensor in tqdm(state_dict.items(), desc="Converting"):
        hf_name = torchtitan_to_hf_name(tt_name)
        if hf_name is None:
            continue

        # Check if should skip quantization (embeddings, norms, output)
        if quant_config is None:
            # we skip quantization altogether if no quantization config is given
            should_skip = True
        else:
            skip_layers = quant_config.ignore_names or []
            should_skip = any(skip in tt_name for skip in skip_layers)

        is_embedding = "tok_embeddings" in tt_name or "output" in tt_name
        is_norm = "norm" in tt_name

        # HF uses a different rope implementation which we need to account for
        if "wq" in tt_name:
            tensor = convert_interleaved_to_split(tensor, model_args.n_heads)
        if "wk" in tt_name:
            tensor = convert_interleaved_to_split(tensor, model_args.n_kv_heads)

        if should_skip or is_embedding or is_norm:
            # Save as bf16 directly
            state[hf_name] = tensor.to(torch.bfloat16).cpu()
        else:
            # Quantize linear layers
            layer_path = tt_name.replace(".weight", "")
            centroids = centroids_map.get(layer_path)
            try:
                packed, scales, centroids_sorted = quantize_weight(
                    tensor.float(),
                    centroids.float(),
                    bits=quant_config.target_bit_width,
                    block_size=quant_config.block_dim_2 or 64,
                    relative_scale=quant_config.relative_scale,
                )

                base_name = hf_name.replace(".weight", "")
                state[f"{base_name}.weight_packed"] = packed.cpu()
                state[f"{base_name}.scales"] = scales.to(torch.bfloat16).cpu()
                state[f"{base_name}.centroids"] = centroids_sorted.to(
                    torch.float32
                ).cpu()

            except Exception as e:
                print(f"  Warning: Failed to quantize {tt_name}: {e}")
                state[hf_name] = tensor.to(torch.bfloat16).cpu()

    del state_dict
    gc.collect()

    # Save weights to root directory
    print("\nSaving weights...")
    _save_sharded(state, output_dir, max_shard_size_gb)

    # Save tokenizer
    actual_vocab_size = model_args.vocab_size
    if tokenizer_path:
        print("\nSaving tokenizer...")
        actual_vocab_size = _save_tokenizer(output_dir, tokenizer_path)

    # Save config files
    print("\nSaving config files...")

    hf_config = create_hf_config(model_args, vocab_size=actual_vocab_size)
    with open(output_dir / "config.json", "w") as f:
        json.dump(hf_config, f, indent=2)

    # quantization_config.json
    if quant_config is not None:
        quant_dict = {
            "quant_method": quant_config.quantizer,
            "bits": quant_config.target_bit_width,
            "block_size": quant_config.block_dim_2 or 64,
            "relative_scale": quant_config.relative_scale,
            "num_centroids": 2 ** (quant_config.target_bit_width),
        }
        if len(centroids_map) > 0:
            sample_c = list(centroids_map.values())[0]
            quant_dict["centroids"] = sample_c.sort()[0].tolist()
        with open(output_dir / "quantization_config.json", "w") as f:
            json.dump(quant_dict, f, indent=2)

    # generation_config.json
    gen_config = create_generation_config()
    with open(output_dir / "generation_config.json", "w") as f:
        json.dump(gen_config, f, indent=2)

    # README.md
    if quant_config is None:
        _save_readme_unquantized(output_dir, model_args)
    else:
        _save_readme_quantized(output_dir, model_args, quant_config)

    print(f"\n{'=' * 60}")
    print(f"Export complete!")
    print(f"Output directory: {output_dir}")
    print(f"{'=' * 60}")

    return output_dir


def _save_sharded(
    state_dict: dict[str, Tensor],
    output_dir: Path,
    max_shard_size_gb: float = 5.0,
) -> None:
    """Save state dict, splitting into shards if necessary."""
    max_shard_bytes = int(max_shard_size_gb * 1e9)
    total_size = sum(t.numel() * t.element_size() for t in state_dict.values())

    if total_size <= max_shard_bytes:
        # Single file
        save_file(state_dict, output_dir / "model.safetensors")
        return

    # Split into shards
    current_shard = {}
    current_size = 0
    shard_idx = 0
    weight_map = {}

    items = list(state_dict.items())
    num_shards = (total_size // max_shard_bytes) + 1

    for key, tensor in items:
        tensor_size = tensor.numel() * tensor.element_size()

        if current_size + tensor_size > max_shard_bytes and current_shard:
            # Save current shard
            shard_name = f"model-{shard_idx + 1:05d}-of-{num_shards:05d}.safetensors"
            save_file(current_shard, output_dir / shard_name)
            for k in current_shard:
                weight_map[k] = shard_name
            current_shard = {}
            current_size = 0
            shard_idx += 1

        current_shard[key] = tensor
        current_size += tensor_size

    # Save final shard
    if current_shard:
        shard_name = f"model-{shard_idx + 1:05d}-of-{num_shards:05d}.safetensors"
        save_file(current_shard, output_dir / shard_name)
        for k in current_shard:
            weight_map[k] = shard_name

    # Save index
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)


def _estimate_params(model_args: TransformerModelArgs) -> int:
    """Estimate total number of parameters from model config."""
    dim = model_args.dim
    n_layers = model_args.n_layers
    n_heads = model_args.n_heads
    n_kv_heads = model_args.n_kv_heads or n_heads
    vocab_size = model_args.vocab_size

    # FFN hidden dim
    ffn_hidden = int(8 * dim / 3)
    ffn_hidden = model_args.multiple_of * (
        (ffn_hidden + model_args.multiple_of - 1) // model_args.multiple_of
    )
    if model_args.ffn_dim_multiplier is not None:
        ffn_hidden = int(ffn_hidden * model_args.ffn_dim_multiplier)

    head_dim = dim // n_heads

    # Per-layer params
    attn_params = dim * (n_heads + 2 * n_kv_heads + n_heads) * head_dim
    ffn_params = dim * ffn_hidden * 3  # gate, up, down
    norm_params = dim * 2  # attention_norm, ffn_norm
    layer_params = attn_params + ffn_params + norm_params

    # Total
    embed_params = vocab_size * dim * 2  # tok_embeddings + output
    total = embed_params + n_layers * layer_params + dim  # + final norm

    return total


def _save_tokenizer(output_dir: Path, tokenizer_path: str | Path) -> int:
    """Copy tokenizer to output directory.

    Returns:
        Vocabulary size
    """
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True
        )

        # Add chat template if missing
        if tokenizer.chat_template is None:
            tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE

        tokenizer.save_pretrained(output_dir)
        print(f"  Tokenizer saved (vocab_size={tokenizer.vocab_size})")
        return tokenizer.vocab_size

    except Exception as e:
        print(f"  Warning: Could not save tokenizer: {e}")
        return 128256


def _save_readme_quantized(
    output_dir: Path,
    model_args: TransformerModelArgs,
    quant_config: Quantization,
) -> None:
    """Generate README.md for the model."""
    bits = quant_config.target_bit_width
    block_size = quant_config.block_dim_2
    relative_scale = quant_config.relative_scale

    # Compute model size string
    n_params = _estimate_params(model_args)
    model_size = f"{n_params / 1e9:.1f}B"

    readme = f"""# LLaMA {model_size} - {bits}-bit Quantized

## Model Details
- **Architecture**: LLaMA
- **Size**: {model_size} ({n_params:,} parameters)
- **Quantization**: {bits}-bit k-means with per-{block_size} block {relative_scale} scaling
- **Centroids**: Defined per Layer

## Directory Structure
```
.
├── config.json                  # HuggingFace model config
├── quantization_config.json     # Quantization parameters
├── generation_config.json       # Default generation settings
├── tokenizer.json              # Tokenizer files
└── model.safetensors           # Weights (quantized linear layers + bf16 embeddings/norms)
```

## Weight Format
Linear layers are quantized:
- `*.weight_packed`: Packed uint8 indices ({8 // bits} elements per byte)
- `*.scales`: Per-block scale factors (bfloat16)
- `*.centroids`: K-means centroids for this layer (float32)

Embeddings and norms are stored as bfloat16.

## License
See LICENSE file in the repository.
"""
    with open(output_dir / "README.md", "w") as f:
        f.write(readme)


def _save_readme_unquantized(
    output_dir: Path,
    model_args: TransformerModelArgs,
) -> None:
    """Generate README.md for the model."""

    # Compute model size string
    n_params = _estimate_params(model_args)
    model_size = f"{n_params / 1e9:.1f}B"

    readme = f"""# LLaMA {model_size} - Bfloat16

## Model Details
- **Architecture**: LLaMA
- **Size**: {model_size} ({n_params:,} parameters)

## Directory Structure
```
.
├── config.json                  # HuggingFace model config
├── generation_config.json       # Default generation settings
├── tokenizer.json              # Tokenizer files
└── model.safetensors           # Weights (in Bfloat16)
```

## License
See LICENSE file in the repository.
"""
    with open(output_dir / "README.md", "w") as f:
        f.write(readme)


def upload_to_hub(
    model_dir: str | Path,
    repo_id: str,
    token: str | None = None,
    private: bool = False,
) -> str:
    """Upload model to HuggingFace Hub.

    Args:
        model_dir: Local directory with model files
        repo_id: HuggingFace repository id (e.g., "username/model-name")
        token: HuggingFace token (uses cached token if None)
        private: Whether to create private repository

    Returns:
        URL of the uploaded model
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise ImportError(
            "huggingface_hub is required. Install with: pip install huggingface_hub"
        )

    print(f"\nUploading to HuggingFace Hub: {repo_id}")

    api = HfApi(token=token)
    api.create_repo(repo_id, private=private, exist_ok=True)

    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=repo_id,
        commit_message="Upload model",
    )

    url = f"https://huggingface.co/{repo_id}"
    print(f"Upload complete: {url}")
    return url


# --------------------------------------------------------------------------- #
#  CLI                                                                        #
# --------------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser(
        description="Export quantized models to HuggingFace format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic export
    python export_to_hf.py /path/to/checkpoint ./output --model_size 8B

    # With custom quantization settings
    python export_to_hf.py /path/to/checkpoint ./output \\
        --model_size 30B --bits 1 --block_size 64 --relative_scale absmean

    # Upload to HuggingFace Hub
    python export_to_hf.py /path/to/checkpoint ./output \\
        --model_size 8B --upload_to_hub username/my-quantized-llama
        """,
    )

    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Path to DCP checkpoint directory",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Output directory for HuggingFace model",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="8B",
        choices=list(llama3_qat_configs.keys()),
        help="Model size configuration",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=1,
        choices=[1, 2, 4, 8, 16],
        help="Quantization bit width. 16bit = Bfloat16",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=64,
        help="Block size for quantization scaling",
    )
    parser.add_argument(
        "--relative_scale",
        type=str,
        default="absmean",
        choices=["absmean", "absmax", "rms", "none"],
        help="Relative scale computation method",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="meta-llama/Llama-3.1-8B",
        help="Path to tokenizer (local path or HF model id)",
    )
    parser.add_argument(
        "--upload_to_hub",
        type=str,
        default=None,
        help="Upload to HuggingFace Hub (provide repo_id)",
    )
    parser.add_argument(
        "--hub_private",
        action="store_true",
        help="Make HuggingFace repo private",
    )

    args = parser.parse_args()

    # Get model config from llama3_lowp_configs
    model_args = llama3_qat_configs[args.model_size]

    # Create quantization config
    if args.bits == 16:
        quant_config = None
    else:
        quant_config = Quantization(
            target_bit_width=args.bits,
            block_dim_2=args.block_size,
            relative_scale=args.relative_scale,
        )

    output_dir = export_model(
        checkpoint_dir=args.checkpoint_path,
        output_dir=args.output_dir,
        model_args=model_args,
        quant_config=quant_config,
        tokenizer_path=args.tokenizer_path,
    )

    if args.upload_to_hub:
        upload_to_hub(
            model_dir=output_dir,
            repo_id=args.upload_to_hub,
            private=args.hub_private,
        )


if __name__ == "__main__":
    main()
