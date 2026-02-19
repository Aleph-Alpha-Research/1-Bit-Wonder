import torch
from fdk.kernels import run_mm_lut, run_mv_lut, mm_lut_ref

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    print("Warning: Triton not available, using torch.compile fallback")


def build_lut8(centroids: torch.Tensor, bits: int) -> torch.Tensor:
    """Build 256-entry LUT from centroids for packed byte format."""
    elements_per_byte = 8 // bits
    return torch.cartesian_prod(*([centroids] * elements_per_byte)).view(
        256, elements_per_byte
    )


_LUT_CACHE: dict[tuple, torch.Tensor] = {}


def mm_lut(
    x: torch.Tensor,
    weight_packed: torch.Tensor,
    centroids: torch.Tensor,
    scales: torch.Tensor,
    bits: int = 4,
) -> torch.Tensor:
    """Matrix multiplication with LUT-based dequantization."""
    global _LUT_CACHE

    use_triton = TRITON_AVAILABLE and x.is_cuda

    batch_dims = x.shape[:-1]
    in_features = x.shape[-1]
    out_features = scales.shape[0]

    x_flat = x.reshape(-1, in_features).contiguous()
    m = x_flat.shape[0]

    # Build LUT8 (cached)
    cache_key = (id(centroids), centroids.device, bits)
    if cache_key not in _LUT_CACHE:
        _LUT_CACHE[cache_key] = build_lut8(centroids, bits).to(x.device, x.dtype)
    lut8 = _LUT_CACHE[cache_key]

    weight_packed = weight_packed.contiguous()
    scales = scales.to(x.dtype).contiguous()

    if use_triton:
        out = torch.empty((m, out_features), device=x.device, dtype=x.dtype)
        if m == 1:
            run_mv_lut(x_flat[0], weight_packed, lut8, scales, out[0])
        else:
            run_mm_lut(x_flat, weight_packed, lut8, scales, out)
    else:
        out = mm_lut_ref(x_flat, weight_packed, lut8, scales)

    return out.reshape(*batch_dims, out_features)


class QuantizedLinearTriton(torch.nn.Module):
    """Quantized linear layer using Triton LUT kernels for fast inference."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 4,
        block_size: int = 64,
        device: torch.device = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        assert block_size == 64, "Currently only a block size of 64 is supported"
        self.block_size = block_size
        self.num_centroids = 2**bits

        elements_per_byte = 8 // bits
        packed_in = in_features // elements_per_byte
        num_blocks = in_features // block_size

        # Buffers for quantized weights
        self.register_buffer(
            "weight_packed",
            torch.zeros(out_features, packed_in, dtype=torch.uint8, device=device),
        )
        self.register_buffer(
            "scales", torch.zeros(out_features, num_blocks, dtype=dtype, device=device)
        )
        self.register_buffer(
            "centroids",
            torch.zeros(self.num_centroids, dtype=torch.float32, device=device),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using Triton mm_lut kernel."""
        # x: [batch, seq, in_features] or [seq, in_features]
        orig_shape = x.shape
        if x.dim() == 3:
            batch, seq, _ = x.shape
            x = x.view(batch * seq, -1)
        elif x.dim() == 1:
            x = x.unsqueeze(0)

        # mm_lut expects centroids (not pre-built LUT), it builds LUT internally
        out = mm_lut(
            x.contiguous(),
            self.weight_packed,
            self.centroids,  # Pass centroids, mm_lut builds LUT
            self.scales.to(x.dtype),
            bits=self.bits,
        )

        # Reshape back
        if len(orig_shape) == 3:
            out = out.view(batch, seq, -1)
        elif len(orig_shape) == 1:
            out = out.squeeze(0)

        return out

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, bits={self.bits}, block={self.block_size}"
