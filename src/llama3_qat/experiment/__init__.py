from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.components.validate import build_validator
from torchtitan.distributed.pipeline_parallel import pipeline_llm

from torchtitan.hf_datasets.text_datasets import build_text_dataloader

from torchtitan.models.llama3.infra.parallelize import parallelize_llama
from torchtitan.models.llama3.model.args import TransformerModelArgs
from torchtitan.models.llama3.model.model import Transformer
from torchtitan.protocols.train_spec import register_train_spec, TrainSpec
from torchtitan.protocols.model_converter import register_model_converter
from .quantized_linear import QuantizedLinearConverter

from torchtitan.hf_datasets.text_datasets import DatasetConfig, DATASETS, load_dataset, _process_c4_text

__all__ = [
    "parallelize_llama",
    "pipeline_llm",
    "TransformerModelArgs",
    "Transformer",
    "QATTransformer",
    "llama3_lowp_configs",
]


llama3_lowp_configs = {
    "debugmodel": TransformerModelArgs(
        dim=256, n_layers=6, n_heads=16, rope_theta=500000
    ),
    "500M": TransformerModelArgs(
        dim=1536,
        n_layers=12,
        n_heads=12,
        n_kv_heads=6,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
        attn_mask_type="block_causal",
        attn_type="flex",
    ),
    "1B": TransformerModelArgs(
        dim=2048,
        n_layers=16,
        n_heads=16,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
        attn_mask_type="block_causal",
        attn_type="flex",
    ),
    "3B": TransformerModelArgs(
        dim=3072,
        n_layers=24,
        n_heads=24,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
        attn_mask_type="block_causal",
        attn_type="flex",
    ),
    "8B": TransformerModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
        attn_mask_type="block_causal",
        attn_type="flex",
    ),
    "11B": TransformerModelArgs(
        dim=4096,
        n_layers=48,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
        attn_mask_type="block_causal",
        attn_type="flex",
    ),
    "30B": TransformerModelArgs(
        dim=6144,
        n_layers=60,
        n_heads=48,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
        attn_mask_type="block_causal",
        attn_type="flex",
    ),
}


register_train_spec(
    name="llama3_qat",
    train_spec=TrainSpec(
        model_cls=Transformer,
        model_args=llama3_lowp_configs,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llm,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_text_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
    ),
)

register_model_converter(QuantizedLinearConverter, "quantized_linear")

DATASETS["test_dataset"] = DatasetConfig(
        path="debug_assets/c4_test",
        loader=lambda path: load_dataset(path, split="train"),
        sample_processor=_process_c4_text,
    )
