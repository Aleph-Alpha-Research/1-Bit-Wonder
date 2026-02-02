from dataclasses import asdict

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from torchtitan.config import JobConfig
from .quantizer import (
    BaseQuantizer,
    QuantizerArgs,
    QUANTIZERS,
)
from torchtitan.distributed import ParallelDims
from torchtitan.protocols.model_converter import ModelConverter
from torchtitan.tools.logging import logger
from .config import Quantization
from .quantizer import NonlinearQuantizer

# --------------------------------------------------------------------------- #
#  Model-wide replacement helper                                             #
# --------------------------------------------------------------------------- #


def replace_linear_with_quantized_linear(
    module: nn.Module,
    *,
    config: Quantization | None = None,
) -> None:

    for name, child in list(module.named_children()):
        replace_linear_with_quantized_linear(
            child,
            config=config,
        )

        if type(child) is nn.Linear and name not in config.ignore_names:
            ql = QuantizedLinear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                config=config,
            )
            # Added for safety: copy weights/bias
            ql.to(device=child.weight.device, dtype=child.weight.dtype)
            ql.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                ql.bias.data.copy_(child.bias.data)

            if isinstance(module, (nn.ModuleDict, nn.ModuleList)):
                module[name] = ql
            else:
                setattr(module, name, ql)


# --------------------------------------------------------------------------- #
#  Low-Precision Linear                                                       #
# --------------------------------------------------------------------------- #


class QuantizedLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        config: Quantization | None = None,
    ):
        super().__init__(in_features, out_features, bias=bias)
        config = config or Quantization()

        self.config = config

        quantizer_args = QuantizerArgs(
            target_bit_width=config.target_bit_width,
            block_dim_1=config.block_dim_1,
            block_dim_2=config.block_dim_2,
            relative_scale=config.relative_scale,
        )
        self.quantizer: BaseQuantizer = QUANTIZERS[config.quantizer](quantizer_args)

        # per-layer deterministic id for RNG syncing across replicas (set by helper)
        self._ql_uid: int = 0  # assigned later

        # QAT gate: full precision until qat_start_step, then quantize.
        # Persistent so checkpoints restore the correct regime.
        enabled_from_init = self.config.qat_start_step <= 0
        self.register_buffer(
            "_qat_enabled",
            torch.tensor(enabled_from_init, dtype=torch.bool),
            persistent=True,
        )

    @torch.no_grad()
    def set_qat_enabled(self, enabled: bool) -> None:
        """Sets whether QAT is enabled (i.e., quantization is applied in forward)."""
        self._qat_enabled.fill_(enabled)

    @torch.no_grad()
    def update_quantization_buffer(self, fisher_diag: Tensor, step: int) -> None:
        """Updates buffers of quantizers if applicable"""

        if isinstance(self.quantizer, NonlinearQuantizer):
            # Deterministic RNG per layer+step; CPU generator to avoid device quirks
            gen = torch.Generator(device=self.weight.device)
            seed = (self._ql_uid * 1_000_003 + step) & 0x7FFF_FFFF
            gen.manual_seed(seed)
            # Call NonlinearQuantizer.update_centroids with `weight=precond`
            self.quantizer.update_centroids(
                w=self.weight,
                generator=gen,
                centroid_update_algorithm=self.config.centroid_update_algorithm,
                weight=fisher_diag,
            )
        else:
            raise ValueError(
                f"Quantization buffer update not implemented for {type(self.quantizer)}"
            )

    def forward(self, x: Tensor) -> Tensor:
        if not bool(self._qat_enabled.item()):
            w_q = self.weight
        else:
            w_q = self.quantizer(self.weight)

        # dtype harmonization
        if w_q.dtype != x.dtype:
            w_q = w_q.to(dtype=x.dtype)
        b = self.bias
        if b is not None and b.dtype != x.dtype:
            b = b.to(dtype=x.dtype)

        return F.linear(x, w_q, b)


# ------------------------------ converter ---------------------------------- #


def _assign_ql_uids(model: nn.Module) -> None:
    """Assign a deterministic per-layer uid to every QuantizedLinear.
    Must be called on all ranks with identical module traversal order.
    """
    uid = 0
    for m in model.modules():
        if isinstance(m, QuantizedLinear):
            m._ql_uid = uid
            uid += 1


class QuantizedLinearConverter(ModelConverter):
    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        self.config: Quantization = job_config.quantization

    def convert(self, model: nn.Module):
        replace_linear_with_quantized_linear(
            model,
            config=self.config,
        )
        # assign deterministic layer ids (same across replicas)
        _assign_ql_uids(model)

        logger.info(
            f"Swapped to quantized linear layers with config: {asdict(self.config)}"
        )
