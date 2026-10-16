from dataclasses import dataclass
from typing import Literal

import torch
from einops import rearrange
from torch.distributed.tensor import DTensor

from .lloyd_max import lut_lloyd_max


@dataclass
class QuantizerArgs:
    target_bit_width: int = 8

    block_dim_1: int | None = 1

    block_dim_2: int | None = 64

    relative_scale: Literal["none", "absmax", "absmean", "rms"] = "absmax"

    eps: float = 1e-5


class BaseQuantizer(torch.nn.Module):
    """Quantizer base class. Forward pass takes an input weight and applies block-wise quantization + dequantization with
    straight-through estimator.

    """

    def __init__(self, args: QuantizerArgs):
        super().__init__()
        self.args = args

        # block versions of quant_dequant where the input of shape (out_features, in_features)
        # gets transformed to shape (out_features, num_blocks, block_size)
        self._block_quant_dequant = torch.vmap(self._quant_dequant)

    def round(self, w: torch.Tensor) -> torch.Tensor:
        return w.round()

    def range_transform(self, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Transformation that gets applied to w before being quantized"""
        s = self._get_tensor_scale(w)
        return w / s, s

    def inverse_range_transform(self, w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Inverse transformation applied for dequantization to recover original w"""
        return s * w

    def _get_tensor_scale(self, w: torch.Tensor) -> torch.Tensor:
        eps = self.args.eps
        match self.args.relative_scale:
            case "none":
                return torch.tensor([1.0], device=w.device, dtype=w.dtype)
            case "absmax":
                return w.abs().amax().clamp(min=eps)
            case "absmean":
                return w.abs().mean().clamp(min=eps)
            case "rms":
                return w.square().mean().sqrt().clamp(min=eps)
            case _:
                raise ValueError

    def _quant_dequant(self, w: torch.Tensor) -> torch.Tensor:
        """
        Quantize and immediately dequantize tensor w.

        This method applies to (de)quantization to a single block.
        Blockwise quantization is handled automatically based on this method.
        Therefore, subclasses who override this method do not have to concern themselves with a blockwise implementation.

        :param w: tensor to be quantized and dequantized
        :return: the quantized and dequantized tensor
        """
        w_transf, s = self.range_transform(w)
        w_quant = self.round(w_transf)
        return self.inverse_range_transform(w_quant, s)

    def _block_wise_reshape(self, w: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """Splits matrix into blocks of shape (block_dim_1, block_dim_2)

        w = 1 2 3 4
            5 6 7 8
        ex. with block_dim_1 = 1, block_dim_2 = 2 -->
        blocks = [[1,2], [3,4], [5,6], [7,8]]
        ex. with block_dim_1 = 2, block_dim_2 = 2 -->
        blocks = [block_1, block_2] = [[[1,2], [5,6]], [[3,4], [7,8]]]

        :param w: Input matrix of shape (M, N) to be split
        :returns: Blockwise split of the matrix (num_blocks, block_dim_1, block_dim_2)
                  and M / block_dim_1, N / block_dim_2 to restore original shape later
        """
        out_features, in_features = w.shape
        a = self.args.block_dim_1 or out_features
        b = self.args.block_dim_2 or in_features
        if out_features % a != 0:
            raise ValueError(
                f"out_features {out_features} not divisible by block_dim {self.args.block_dim_1}"
            )
        if in_features % b != 0:
            raise ValueError(
                f"in_features {in_features} not divisible by block_size {self.args.block_dim_2}"
            )

        n = out_features // a
        m = in_features // b

        return rearrange(w, "(n a) (m b) -> (n m) a b", a=a, b=b), n, m

    def _block_wise_quant_dequant(self, w: torch.Tensor) -> torch.Tensor:
        """
        Quantize and immediately dequantize tensor w in a block-wise fashion.

        To (de)quantize each block, `self._quant_dequant()` is used.
        This methods assumes that the input tensor `w` is of shape `(out_features, in_features)` and
        splits it into blocks along the last dimension (in_features).

        :param w: The tensor to be quantized and dequantized. Has to be a 2D tensor of shape `(out_features, in_features)`.
        :returns: The quantized and dequantized tensor
        """
        w_reshaped, num_blocks_1, num_blocks_2 = self._block_wise_reshape(w)
        wq_reshaped: torch.Tensor = self._block_quant_dequant(w_reshaped)
        # rearrange to original shape. NOTE: wq_reshaped.reshape_as(w) scrambles the dimensions and gives incorrect result!
        return rearrange(
            wq_reshaped, "(n m) a b -> (n a) (m b)", n=num_blocks_1, m=num_blocks_2
        )

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        return w + (self._block_wise_quant_dequant(w) - w).detach()


class SymmetricIntQuantizer(BaseQuantizer):
    """Applies uniform integer quantization with fixed bit width."""

    def __init__(self, args: QuantizerArgs):
        super().__init__(args)
        self.grid_upper_bound = 2 ** (self.args.target_bit_width - 1) - 1

    def range_transform(self, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Scales relative to the grid size, then clips to [-grid_boundary, grid_boundary]
        s = self._get_tensor_scale(w)
        s = torch.div(s, self.grid_upper_bound)
        w_scaled = (w / s).clamp(-self.grid_upper_bound, self.grid_upper_bound)
        return w_scaled, s


class TernaryQuantizer(BaseQuantizer):
    """
    Quantise weights to {-1,0,+1} with abs-mean scale (single scalar) following 1.58-bit LLMs.
    See: https://arxiv.org/pdf/2402.17764
    """

    def __init__(self, args: QuantizerArgs):
        super().__init__(args)
        assert self.args.relative_scale == "absmean", (
            f"Ternary quantizer has to use absmean relative scale, {self.args.relative_scale} selected."
        )

    def range_transform(self, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # scales, then clips values to [-1, 1]
        s = self._get_tensor_scale(w)
        w_scaled = (w / s).clamp(-1, 1)
        return w_scaled, s


class BinaryQuantizer(BaseQuantizer):
    """
    Quantise weights to {-1,+1} following BitLinear.
    See: https://github.com/kyegomez/BitNet/blob/main/bitnet/bitlinear.py
    and https://arxiv.org/pdf/2310.11453
    """

    def __init__(self, args: QuantizerArgs):
        super().__init__(args)
        assert self.args.relative_scale == "absmean", (
            f"Binary quantizer has to use absmean relative scale, {self.args.relative_scale} selected."
        )

    def range_transform(
        self, w: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        s = self._get_tensor_scale(w)
        gamma = w.mean()
        return (w - gamma), s

    def round(self, w: torch.Tensor) -> torch.Tensor:
        return w.sign()


class NonlinearQuantizer(BaseQuantizer):
    """Applies nonlinear quantization using rounding to nearest centroids."""

    def __init__(self, args: QuantizerArgs):
        super().__init__(args)
        self.num_centroids = 2**self.args.target_bit_width
        self.centroids: torch.Tensor
        self.init_weights()
        self._block_get_transformed_weight = torch.vmap(
            self._get_transformed_weight,
        )
        self.register_buffer("_is_initialized", torch.tensor(False), persistent=True)

    def init_weights(
        self,
        buffer_device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        buffer_device = buffer_device or torch.device("cpu")
        dtype = dtype or torch.float32

        # This methods gets called during __init__, but centroids will be set to zero since meta device is used.
        # It is important to still register the buffer here so that it is part of the state dict.
        # Otherwise the callback logic for updating centroids collides with checkpoint loading/saving.
        self.register_buffer(
            "centroids",
            torch.empty(self.num_centroids, dtype=dtype, device=buffer_device),
            persistent=True,
        )

    def set_centroids(self, centroids: torch.Tensor) -> None:
        centroids, _ = torch.sort(centroids)
        assert centroids.shape == (self.num_centroids,)
        self.centroids.copy_(centroids)
        if not self._is_initialized:
            self._is_initialized.copy_(True)

    @torch.no_grad()
    def update_centroids(
        self,
        w: torch.Tensor,
        generator: torch.Generator,
        centroid_update_algorithm: str = "kmeans",
        weight: torch.Tensor | None = None,
    ) -> None:
        wn = self.get_transformed_weight(w)

        # Handle Distributed Tensor
        if isinstance(wn, DTensor):
            # If we have a weight (importance), we need to gather it as well
            if weight is not None:
                if isinstance(weight, DTensor):
                    weight = weight.full_tensor()
                elif isinstance(weight, torch.Tensor):
                    # Assume weight is the local shard matching wn's local shard.
                    # We wrap it in a DTensor with the same spec as wn to gather it.
                    weight = DTensor.from_local(
                        weight, wn.device_mesh, wn.placements
                    ).full_tensor()

            # Gather the transformed weights
            wn = wn.full_tensor()

        if not torch.isfinite(wn).all():
            return  # keep previous centroids

        flat_data = wn.reshape(-1)
        flat_weight: torch.Tensor | None = None

        if weight is not None:
            if weight.shape != wn.shape:
                raise ValueError(
                    f"NonlinearQuantizer.update_centroids: weight.shape {weight.shape} "
                    f"must match wn.shape {wn.shape}"
                )

            flat_weight = weight.reshape(-1)

            # --- Change weight type if its not same type as w ---
            if type(flat_weight) is not type(flat_data):
                flat_weight = flat_weight.to(flat_data)

        match centroid_update_algorithm:
            case "kmeans":
                centroids = lut_lloyd_max(
                    flat_data,
                    bits=None,
                    codepoints=self.num_centroids,
                    init="kmeans++",
                    generator=generator,
                ).to(dtype=w.dtype, device=w.device)
            case "fischer_kmeans":
                centroids = lut_lloyd_max(
                    flat_data,
                    bits=None,
                    codepoints=self.num_centroids,
                    weight=flat_weight,
                    init="kmeans++",
                    generator=generator,
                ).to(dtype=w.dtype, device=w.device)
            case _:
                raise ValueError

        self.set_centroids(centroids)

    def round(self, w: torch.Tensor):
        # Optimization: Use bucketize for 1D nearest neighbor search.
        # This avoids expanding the tensor to (..., num_centroids) which consumes huge memory
        # and bandwidth, causing significant slowdowns (5-10x) when num_centroids > 16.
        boundaries = (self.centroids[:-1] + self.centroids[1:]) / 2
        idx = torch.bucketize(w, boundaries.to(dtype=w.dtype))

        return self.centroids[idx]

    def _get_transformed_weight(self, w: torch.Tensor) -> torch.Tensor:
        return self.range_transform(w)[0]

    @torch.no_grad()
    def get_transformed_weight(self, w: torch.Tensor) -> torch.Tensor:
        """Returns the block wise transformed weight for re-calculating centroids."""
        w_reshaped, num_blocks_1, num_blocks_2 = self._block_wise_reshape(w)
        w_transf: torch.Tensor = self._block_get_transformed_weight(w_reshaped)
        return rearrange(
            w_transf, "(n m) a b -> (n a) (m b)", n=num_blocks_1, m=num_blocks_2
        )

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        if not self._is_initialized:
            return w
        return super().forward(w)


QUANTIZERS = {
    "sym_int": SymmetricIntQuantizer,
    "ternary": TernaryQuantizer,
    "binary": BinaryQuantizer,
    "nonlinear": NonlinearQuantizer,
}
