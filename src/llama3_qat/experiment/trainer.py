from typing import Iterable
import torch
from torchtitan.train import Trainer
from .config import Quantization as QuantizationConfig
from .quantized_linear import QuantizedLinear
from torchtitan.tools.logging import logger


def get_fisher_diagonal_weight_per_param(
    trainer: Trainer,
) -> dict[torch.nn.Parameter, torch.Tensor]:
    weight_per_param = dict()
    beta2 = trainer.job_config.optimizer.beta2
    for opt in trainer.optimizers.optimizers:
        # Only use AdamW optimizers
        assert isinstance(opt, (torch.optim.AdamW, torch.optim.Adam)), (
            "Currently only Adam is supported for getting diagonal Fisher weights"
        )
        for group in opt.param_groups:
            for p in group["params"]:
                assert isinstance(p, torch.nn.Parameter)
                state = opt.state.get(p, {})
                v = state["exp_avg_sq"]  # ~ E[g^2]
                step_state = state["step"]
                # Bias-corrected second moment: v_hat ≈ E[g^2] ≈ diag(F)
                v_hat = v / (1.0 - beta2**step_state)

                # Use v_hat as Fisher-diagonal proxy; normalise so magnitudes are tame
                fisher_diag = v_hat / v_hat.mean().clamp_min(1e-12)
                weight_per_param[p] = fisher_diag

    return weight_per_param


class QATTrainer(Trainer):
    def _quantization_buffer_update_callback(self) -> None:
        qcfg: QuantizationConfig = self.job_config.quantization
        qat_start_step = qcfg.qat_start_step or 0

        # Flip QAT gate on all QuantizedLinear layers
        qat_enabled = self.step >= qat_start_step
        qat_activation_step = self.step == qat_start_step
        if qat_activation_step:
            logger.info(
                "Enabling QAT at step %d for all QuantizedLinear layers.",
                self.step,
            )
            for model in self.model_parts:
                for module in model.modules():
                    if isinstance(module, QuantizedLinear):
                        # only write if changed (avoids needless D2D syncs)
                        if bool(module._qat_enabled.item()) != qat_enabled:
                            module.set_qat_enabled(qat_enabled)

        # If we're still in FP phase, do not update quantization buffers.
        if not qat_enabled or (qcfg.quantizer != "nonlinear"):
            return

        # Buffer update logic begins.
        buffer_update_interval = qcfg.buffer_update_interval or 0
        buffer_update_start_step = qat_start_step or 0

        if (self.step == buffer_update_start_step) or (
            buffer_update_interval > 0 and (self.step % buffer_update_interval == 0)
        ):
            # --- Loop over QuantizedLinear + NonlinearQuantizer layers ---
            num_layers = 0
            num_updated = 0

            # 1) Build mapping from parameter to its approximative Fisher diagonal weight, used for Fisher k-means algorithm
            # if using nonlinear quantizer
            fisher_diag_weights_per_param = get_fisher_diagonal_weight_per_param(self)

            # 2) Loop over all QuantizedLinear layers and update buffer
            for model in self.model_parts:
                for module in model.modules():
                    if not isinstance(module, QuantizedLinear):
                        continue

                    num_layers += 1
                    param = module.weight
                    fisher_diag = fisher_diag_weights_per_param[param]

                    module.update_quantization_buffer(
                        fisher_diag=fisher_diag, step=self.step
                    )
                    num_updated += 1
        else:
            return

        if num_layers > 0:
            logger.info(
                "Quantization buffer update callback at step %d: updated %d/%d quantized linear layers.",
                self.step,
                num_updated,
                num_layers,
            )

    def train_step(
        self, data_iterator: Iterable[tuple[dict[str, torch.Tensor], torch.Tensor]]
    ):
        self._quantization_buffer_update_callback()
        return super().train_step(data_iterator)
